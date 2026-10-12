"""Bayesian analysis of a forecasting competition.

With many entrants, naive pairwise tables mislead: some pair always looks
significant by chance, matches in one weekend or league are not independent,
and "model X wins in league Y" is mostly noise. This fits a hierarchical model
of the entrants' per-match losses instead:

    loss[m, j] = mu[m] + a[j] + b[origin(m), j] + g[league(m), j] + eps

* ``mu[m]``  - how hard match ``m`` was, shared by every entrant (so entrants
  are compared on the same matches, exactly);
* ``a[j]``   - entrant ``j``'s overall effect (lower loss is better), the
  quantity the leaderboard ranks;
* ``b[o, j]`` - how entrant ``j`` did in time block ``o`` (a backtest origin).
  Because blocks vary, uncertainty reflects the number of independent *periods*
  and not just the number of matches;
* ``g[c, j]`` - entrant ``j``'s league-specific effect, shrunk toward its
  overall effect, which guards against reading noise as "good in the Top 14";
* ``eps``    - Gaussian noise.

The target is each entrant's *mean* loss, because that is what a proper scoring
rule rewards: the occasional confident miss is the point, not an outlier to
down-weight (a Student-t error would estimate the typical match instead and
can rank a model that is good on average games above one that is better on
average).

``mu[m]`` is integrated out exactly. The contrasts between entrants within a
match are independent with common variance, so the likelihood is that of the
centred losses with ``J - 1`` degrees of freedom per match. This also handles
two entrants correctly (one independent contrast per match, not two copies).
``a``, ``b`` and ``g`` are centred across entrants: only differences between
entrants are identified, and only those are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rugby_ranking.backtest import metrics

_LOSS_METRICS = metrics.LOSSES


@dataclass
class CompetitionAnalysis:
    entrants: list[str]
    metric: str
    n_matches: int
    n_blocks: int
    leaderboard: pd.DataFrame        # per entrant: effect, interval, P(best), expected rank
    pairwise: pd.DataFrame           # P(row beats column), overall
    by_league: pd.DataFrame          # per league x entrant: effect with interval
    block_sd: float                  # posterior mean sd of period-to-period variation
    league_sd: float                 # posterior mean sd of league-specific effects
    diagnostics: dict

    def __str__(self) -> str:
        pd.set_option("display.float_format", "{:.3f}".format, "display.width", 200)
        return (
            f"{len(self.entrants)} entrants, {self.n_matches} matches in {self.n_blocks} blocks; "
            f"metric: {self.metric} (lower is better)\n"
            f"period-to-period sd {self.block_sd:.3f}, league-specific sd {self.league_sd:.3f}\n\n"
            f"{self.leaderboard.to_string()}\n\nP(row beats column):\n{self.pairwise.to_string()}"
        )


def _centre(x, axis):
    return x - x.mean(axis=axis, keepdims=True)


def analyse_competition(frame: pd.DataFrame, metric: str = "logloss", **kwargs) -> CompetitionAnalysis:
    """Analyse a long backtest frame (one row per forecaster and match).

    Only matches every entrant forecast are used. ``metric`` is a per-match
    loss from ``rugby_ranking.backtest.metrics.LOSSES``; lower is better.
    Keyword arguments go to ``analyse_losses``.
    """
    if metric not in _LOSS_METRICS:
        raise ValueError(f"metric must be one of {_LOSS_METRICS}")
    d = metrics.add_losses(metrics.common_matches(frame))
    losses = d.rename(columns={"forecaster": "entrant", metric: "loss", "origin": "block", "competition": "league"})
    return analyse_losses(losses[["match_id", "entrant", "loss", "block", "league"]], metric=metric, **kwargs)


def analyse_losses(
    losses: pd.DataFrame,
    metric: str = "loss",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 0,
    interval: float = 0.94,
    progressbar: bool = False,
) -> CompetitionAnalysis:
    """Fit the competition model to per-match losses.

    ``losses`` has one row per (match, entrant) with columns ``match_id``,
    ``entrant``, ``loss`` (lower is better), ``block`` (time period) and
    ``league``; every entrant must appear for every match.
    """
    import pymc as pm
    import pytensor.tensor as pt

    wide = losses.pivot(index="match_id", columns="entrant", values="loss")
    if wide.isna().any().any():
        raise ValueError("every entrant must have a loss for every match")
    entrants = list(wide.columns)
    if len(entrants) < 2:
        raise ValueError("need at least two entrants with forecasts for the same matches")
    info = losses.drop_duplicates("match_id").set_index("match_id").loc[wide.index]
    J, M = len(entrants), len(wide)
    block = pd.Categorical(info["block"]).codes
    league_cat = pd.Categorical(info["league"])
    league, leagues = league_cat.codes, list(league_cat.categories)
    O, C = int(block.max()) + 1, len(leagues)
    y = wide.to_numpy()                                           # (M, J)

    dc = y - y.mean(axis=1, keepdims=True)                          # centred within match, (M, J)

    with pm.Model():
        a_raw = pm.Normal("a_raw", 0.0, 0.1, shape=J)
        tau_o = pm.HalfNormal("tau_block", 0.05)
        tau_c = pm.HalfNormal("tau_league", 0.05)
        b_raw = pm.Normal("b_raw", 0.0, 1.0, shape=(O, J))
        g_raw = pm.Normal("g_raw", 0.0, 1.0, shape=(C, J))
        a = pm.Deterministic("a", a_raw - a_raw.mean())
        b = pm.Deterministic("b", (b_raw - b_raw.mean(axis=1, keepdims=True)) * tau_o)
        g = pm.Deterministic("g", (g_raw - g_raw.mean(axis=1, keepdims=True)) * tau_c)
        sigma = pm.HalfNormal("sigma", 0.5)
        resid = dc - (a[None, :] + b[block] + g[league])
        pm.Potential("loglik", -0.5 * M * (J - 1) * pt.log(2 * np.pi * sigma ** 2)
                     - 0.5 * (resid ** 2).sum() / sigma ** 2)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          progressbar=progressbar, target_accept=0.9)

    post = idata.posterior
    a_s = post["a"].values.reshape(-1, J)                          # (S, J) overall effect
    g_s = post["g"].values.reshape(-1, C, J)                       # (S, C, J)
    lo_q, hi_q = (1 - interval) / 2 * 100, 100 - (1 - interval) / 2 * 100

    ranks = a_s.argsort(axis=1).argsort(axis=1) + 1                # 1 = lowest loss = best
    board = pd.DataFrame({
        "effect": a_s.mean(0),
        "lo": np.percentile(a_s, lo_q, axis=0),
        "hi": np.percentile(a_s, hi_q, axis=0),
        "p_best": (ranks == 1).mean(0),
        "expected_rank": ranks.mean(0),
        "raw_mean_loss": wide.mean().to_numpy(),
    }, index=entrants).sort_values("effect")

    pair = pd.DataFrame(
        [[float((a_s[:, i] < a_s[:, k]).mean()) if i != k else np.nan for k in range(J)] for i in range(J)],
        index=entrants, columns=entrants,
    ).loc[board.index, board.index]

    rows = []
    for ci, comp in enumerate(leagues):
        total = a_s + g_s[:, ci, :]
        for j, name in enumerate(entrants):
            rows.append(dict(competition=comp, entrant=name, effect=total[:, j].mean(),
                             lo=np.percentile(total[:, j], lo_q), hi=np.percentile(total[:, j], hi_q),
                             n_matches=int((league == ci).sum())))
    by_league = pd.DataFrame(rows)

    summ = {}
    try:
        import arviz as az
        s = az.summary(idata, var_names=["tau_block", "tau_league", "sigma", "a_raw"])
        summ = {"max_rhat": float(s["r_hat"].max()), "min_ess_bulk": float(s["ess_bulk"].min())}
    except Exception:  # diagnostics are a courtesy
        pass
    n_div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else None

    return CompetitionAnalysis(
        entrants=entrants, metric=metric, n_matches=M, n_blocks=O, leaderboard=board, pairwise=pair,
        by_league=by_league, block_sd=float(post["tau_block"].mean()), league_sd=float(post["tau_league"].mean()),
        diagnostics={**summ, "divergences": n_div},
    )
