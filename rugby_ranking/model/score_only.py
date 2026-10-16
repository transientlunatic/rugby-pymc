"""Score-only Bayesian team strengths.

The event-level model (``RugbyModel``) needs named scorers and lineups, which
the data only has from 2011-12. This model uses only the final score of each
match, so it can estimate team strengths for every season in the data.

Each match contributes two observations, the points one side scored against
the other. For a team ``e`` (a team within one competition) playing opponent
``o`` in season ``s``::

    points ~ NegativeBinomial(mu, alpha_comp)
    log mu = base[comp] + home[comp] * is_home + attack[e, s] - defence[o, s]

A team's attack and defence are random walks over seasons, starting from a
team-level level that is shared by all its seasons, so a side with few matches
in a season borrows from its neighbouring seasons rather than being estimated
alone. ``defence`` is larger for a *better* defence, as in ``RugbyModel``.

Summaries are expressed as the expected margin against a reference side, one
whose attack and defence are the average of the competition that season, at a
venue half way between home and away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class ScoreOnlyConfig:
    attack_sd: float = 0.35        # prior sd of a team's overall level (log scale)
    walk_sd: float = 0.15          # prior scale of the season-to-season drift
    vi_iterations: int = 30000
    seed: int = 42


def season_year(season: str) -> int:
    return int(str(season)[:4])


def prepare(matches: pd.DataFrame) -> dict:
    """Index matches (columns: competition, season, home_team, away_team,
    home_score, away_score) for the model."""
    m = matches.dropna(subset=["home_score", "away_score"]).copy()
    m["year"] = m["season"].map(season_year)
    years = np.arange(m["year"].min(), m["year"].max() + 1)
    comps = sorted(m["competition"].unique())
    ents = sorted(set(zip(m["competition"], m["home_team"])) | set(zip(m["competition"], m["away_team"])))
    ent_ix = {e: i for i, e in enumerate(ents)}
    comp_ix = {c: i for i, c in enumerate(comps)}
    first = np.full(len(ents), len(years), dtype=int)
    last = np.zeros(len(ents), dtype=int)
    for side in ("home_team", "away_team"):
        for (c, t), yr in zip(zip(m["competition"], m[side]), m["year"]):
            i, y = ent_ix[(c, t)], int(yr - years[0])
            first[i], last[i] = min(first[i], y), max(last[i], y)
    rows = []
    for r in m.itertuples(index=False):
        y = int(r.year - years[0])
        h, a, c = ent_ix[(r.competition, r.home_team)], ent_ix[(r.competition, r.away_team)], comp_ix[r.competition]
        rows.append((h, a, y, c, 1, r.home_score))
        rows.append((a, h, y, c, 0, r.away_score))
    d = np.array(rows, dtype=float)
    return dict(
        ents=ents, comps=comps, years=years, first=first, last=last,
        e_for=d[:, 0].astype(int), e_against=d[:, 1].astype(int), s=d[:, 2].astype(int),
        c=d[:, 3].astype(int), home=d[:, 4], pts=d[:, 5].astype(int),
    )


def build(data: dict, config: ScoreOnlyConfig | None = None) -> pm.Model:
    cfg = config or ScoreOnlyConfig()
    E, S, C = len(data["ents"]), len(data["years"]), len(data["comps"])
    # innovations only after a team's first season in the data
    live = (np.arange(S)[None, :] > data["first"][:, None]).astype(float)
    with pm.Model() as model:
        base = pm.Normal("base", 3.0, 0.5, shape=C)
        home = pm.Normal("home", 0.2, 0.15, shape=C)
        log_alpha = pm.Normal("log_alpha", 2.0, 0.7, shape=C)
        level_sd = pm.HalfNormal("level_sd", cfg.attack_sd, shape=2)
        walk_sd = pm.HalfNormal("walk_sd", cfg.walk_sd, shape=2)
        level = pm.Normal("level", 0.0, 1.0, shape=(2, E))
        step = pm.Normal("step", 0.0, 1.0, shape=(2, E, S))
        walk = pt_cumsum(step * live[None, :, :]) * walk_sd[:, None, None]
        strength = pm.Deterministic("strength", level[:, :, None] * level_sd[:, None, None] + walk)  # (2, E, S)
        attack, defence = strength[0], strength[1]
        eta = (base[data["c"]] + home[data["c"]] * data["home"]
               + attack[data["e_for"], data["s"]] - defence[data["e_against"], data["s"]])
        mu = pm.math.exp(eta)
        pm.NegativeBinomial("points", mu=mu, alpha=pm.math.exp(log_alpha)[data["c"]], observed=data["pts"])
    return model


def pt_cumsum(x):
    import pytensor.tensor as pt

    return pt.cumsum(x, axis=-1)


def fit(model: pm.Model, config: ScoreOnlyConfig | None = None, n_samples: int = 500, progressbar: bool = False):
    cfg = config or ScoreOnlyConfig()
    with model:
        approx = pm.fit(cfg.vi_iterations, method="advi", random_seed=cfg.seed, progressbar=progressbar)
    return approx, approx.sample(n_samples, random_seed=cfg.seed)


def summarise(data: dict, post, level: float = 0.9) -> pd.DataFrame:
    """One row per team-season with at least one match: attack, defence and
    expected margin against the reference side (posterior mean and interval)."""
    st = post.posterior["strength"].stack(s=("chain", "draw")).transpose("s", ...).values   # (draws, 2, E, S)
    base = post.posterior["base"].stack(s=("chain", "draw")).transpose("s", ...).values       # (draws, C)
    home = post.posterior["home"].stack(s=("chain", "draw")).transpose("s", ...).values
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    q = lambda x: (float(x.mean()), float(np.quantile(x, lo_q)), float(np.quantile(x, hi_q)))  # noqa: E731
    # which (team, season) pairs played, and how many matches
    played = {}
    for e, s in zip(data["e_for"], data["s"]):      # one 'for' row per match the team played
        played[(e, s)] = played.get((e, s), 0) + 1
    rows = []
    for ci, comp in enumerate(data["comps"]):
        ent_ids = [i for i, (c, _) in enumerate(data["ents"]) if c == comp]
        for s, year in enumerate(data["years"]):
            active = [i for i in ent_ids if (i, s) in played]
            if len(active) < 2:
                continue
            att = st[:, 0, active, s]
            dfn = st[:, 1, active, s]
            att = att - att.mean(axis=1, keepdims=True)    # centre on the competition average that season
            dfn = dfn - dfn.mean(axis=1, keepdims=True)
            mid = (base[:, ci] + home[:, ci] / 2)[:, None]
            margin = np.exp(mid + att) - np.exp(mid - dfn)
            for j, i in enumerate(active):
                a, d, m = q(att[:, j]), q(dfn[:, j]), q(margin[:, j])
                rows.append(dict(
                    team=data["ents"][i][1], competition=comp, season=f"{year}-{year + 1}",
                    attack_mean=a[0], attack_lower=a[1], attack_upper=a[2],
                    defense_mean=d[0], defense_lower=d[1], defense_upper=d[2],
                    ref_margin_mean=m[0], ref_margin_lower=m[1], ref_margin_upper=m[2],
                    n_matches=played[(i, s)]))
    return pd.DataFrame(rows)
