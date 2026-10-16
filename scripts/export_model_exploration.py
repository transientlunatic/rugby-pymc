"""Export what the fitted Bayesian model has learned, for the blog's model-exploration plots.

Reads a checkpoint written by ``fit_strength_history.py`` (same data, same player/team indexing) and writes
three JSON files:

* ``rugby_model.json``      -- global parameters: baseline scoring rates by jersey number, the home
                              uplift per score type, the loadings and standard deviations of the random
                              effects, plus the empirical home advantage by competition and season.
* ``rugby_players.json``    -- per-player try-scoring and kicking effects (with 90% intervals) and what the
                              player actually did, for players with enough minutes.
* ``rugby_team_effects.json`` -- per team-season attack and defence effects, aligned to
                              ``league_placings.json`` so the clubs match the other charts.

    python scripts/export_model_exploration.py --data-dir ../Rugby-Data --checkpoint strength_history_v2 \\
        --team-effects team_strength_history_v2.json --out-dir ../blog/_data

Effects are on the log-rate scale of a Poisson model, so ``exp(effect) - 1`` is a proportional change in a
rate, relative to an average player in the same position (players) or an average side (teams).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fit_strength_history as fit  # noqa: E402
from export_strength_series import CANON  # noqa: E402

SCORE_TYPES = ("tries", "penalties", "conversions", "drop_goals")
POSITIONS = {
    1: "Loosehead prop", 2: "Hooker", 3: "Tighthead prop", 4: "Lock", 5: "Lock", 6: "Flanker", 7: "Flanker",
    8: "Number eight", 9: "Scrum-half", 10: "Fly-half", 11: "Left wing", 12: "Inside centre",
    13: "Outside centre", 14: "Right wing", 15: "Full-back", 16: "Replacement hooker",
    17: "Replacement prop", 18: "Replacement prop", 19: "Replacement lock", 20: "Replacement back row",
    21: "Replacement scrum-half", 22: "Replacement fly-half", 23: "Replacement back",
}
# a coarser grouping for filtering players
GROUPS = {1: "Front row", 2: "Front row", 3: "Front row", 4: "Second row", 5: "Second row", 6: "Back row",
          7: "Back row", 8: "Back row", 9: "Half-backs", 10: "Half-backs", 11: "Back three", 12: "Centres",
          13: "Centres", 14: "Back three", 15: "Back three"}
REPLACEMENT_AS = {16: 2, 17: 1, 18: 3, 19: 4, 20: 7, 21: 9, 22: 10, 23: 15}
EMPIRICAL_COMPETITIONS = ("celtic", "premiership", "top14", "pro-d2", "championship", "super-rugby",
                          "euro-champions", "euro-challenge")


def stat(a: np.ndarray, scale: float = 1.0, digits: int = 3):
    """[mean, 5th, 95th percentile] of draws along axis 0: floats for 1-D draws, arrays for 2-D."""
    lo, hi = np.percentile(a, [5, 95], axis=0)
    out = [np.round(np.asarray(x) * scale, digits) for x in (a.mean(axis=0), lo, hi)]
    return [float(x) for x in out] if a.ndim == 1 else out


def draws(post, name: str) -> np.ndarray:
    v = post[name].values
    return v.reshape(-1, *v.shape[2:])


def load_model(data_dir: Path, checkpoint: str, start_year: int, min_recorded: float):
    from rugby_ranking.model.core import ModelConfig, RugbyModel
    from rugby_ranking.model.inference import ModelFitter

    dataset, df = fit.load_observations(data_dir, start_year, min_recorded)
    model = RugbyModel(ModelConfig(include_defense=True, separate_kicking_effect=True, time_varying_effects=False))
    model.build_joint(df)
    fitter = ModelFitter.load(checkpoint, model)
    return model, fitter.trace.posterior, df


def model_params(post, model) -> dict:
    alpha, theta, eta = draws(post, "alpha"), draws(post, "theta_position"), draws(post, "eta_home")
    pos_ids = {int(k): v for k, v in model._position_ids.items()}      # jersey number -> index
    order = sorted(pos_ids)
    # expected scores per 80 minutes for an average player and side, playing away
    rate = {s: {str(p): stat(np.exp(alpha[:, i] + theta[:, i, pos_ids[p]])) for p in order}
            for i, s in enumerate(SCORE_TYPES)}

    sig_t, sig_d = draws(post, "sigma_team"), draws(post, "sigma_defense")
    sig_pt, sig_pk = draws(post, "sigma_player_try"), draws(post, "sigma_player_kick")
    lam_t, lam_d = draws(post, "lambda_team"), draws(post, "lambda_defense")
    lam_pt, lam_pk = draws(post, "lambda_player_try"), draws(post, "lambda_player_kick")
    return dict(
        score_types=list(SCORE_TYPES),
        positions=[dict(n=p, label=POSITIONS.get(p, str(p))) for p in order],
        rate_per_80=rate,
        home_uplift_pct={s: stat(np.exp(eta[:, i]) - 1, 100, 1) for i, s in enumerate(SCORE_TYPES)},
        # how strongly each random effect shows up in each score type (loadings), and how large it is
        loadings=dict(team=[stat(lam_t[:, i]) for i in range(4)],
                      player_try=[stat(lam_pt[:, i]) for i in range(4)],
                      player_kick=[stat(lam_pk[:, i]) for i in range(4)]),
        sigma=dict(team=stat(sig_t), defence=stat(sig_d), player_try=stat(sig_pt), player_kick=stat(sig_pk)),
        # the spread each effect actually produces, on the log-rate scale of the score it matters most for
        effective_sd=[
            dict(key="team", label="Team attack (tries)", sd=stat(sig_t * lam_t[:, 0])),
            dict(key="defence", label="Team defence (tries)", sd=stat(sig_d * lam_d)),
            dict(key="player_try", label="Player try-scoring", sd=stat(sig_pt * lam_pt[:, 0])),
            dict(key="player_kick", label="Player goal-kicking (penalties)", sd=stat(sig_pk * lam_pk[:, 1])),
        ],
    )


def empirical_home(data_dir: Path) -> list[dict]:
    from rugby_ranking.forecast.store import MatchStore

    m = MatchStore.from_data_dir(data_dir).played
    m = m[m["competition"].isin(EMPIRICAL_COMPETITIONS)].copy()
    m["margin"] = m["home_score"] - m["away_score"]
    out = []
    for (comp, season), g in m.groupby(["competition", "season"]):
        if len(g) < 40:
            continue
        decided = g[g["margin"] != 0]
        out.append(dict(competition=comp, season=season, n=int(len(g)),
                        home_win_pct=round(100 * float((g["margin"] > 0).mean() + 0.5 * (g["margin"] == 0).mean()), 1),
                        margin=round(float(g["margin"].mean()), 2),
                        margin_se=round(float(g["margin"].std(ddof=1) / np.sqrt(len(g))), 2),
                        home_pts=round(float(g["home_score"].mean()), 2), away_pts=round(float(g["away_score"].mean()), 2),
                        decided=int(len(decided))))
    return out


def players(post, model, df: pd.DataFrame, min_minutes: float) -> list[dict]:
    sig_pt, sig_pk = draws(post, "sigma_player_try"), draws(post, "sigma_player_kick")
    lam_pt, lam_pk = draws(post, "lambda_player_try"), draws(post, "lambda_player_kick")
    raw_t, raw_k = draws(post, "beta_player_try_raw"), draws(post, "beta_player_kick_raw")
    ids = model._player_ids

    df = df.sort_values("date")
    by = df.groupby("player_name")
    tot = by.agg(minutes=("minutes_played", "sum"), games=("match_id", "nunique"), tries=("tries", "sum"),
                 pens=("penalties", "sum"), cons=("conversions", "sum"), team=("team", "last"),
                 season=("season", "last"), first=("season", "first"))
    # main position: the jersey with the most minutes among 1-15 (replacements count as the starter they cover)
    pos = df.assign(p=df["position"].replace(REPLACEMENT_AS).astype(int))
    main = pos.groupby(["player_name", "p"])["minutes_played"].sum().reset_index().sort_values("minutes_played") \
        .groupby("player_name").tail(1).set_index("player_name")["p"]
    tot = tot[tot["minutes"] >= min_minutes]

    idx = np.array([ids[n] for n in tot.index])
    beta_t = sig_pt[:, None] * lam_pt[:, 0:1] * raw_t[:, idx]            # effect on the try rate
    beta_k = sig_pk[:, None] * lam_pk[:, 1:2] * raw_k[:, idx]            # effect on the penalty rate
    te, ke = stat(np.exp(beta_t) - 1, 100, 1), stat(np.exp(beta_k) - 1, 100, 1)
    out = []
    for j, (name, r) in enumerate(tot.iterrows()):
        p = int(main.get(name, 0))
        out.append(dict(n=name, t=r["team"], s=r["season"], f=r["first"], m=int(r["minutes"]), g=int(r["games"]),
                        p=p, grp=GROUPS.get(p, "Other"), tr=int(r["tries"]), pk=int(r["pens"]), cv=int(r["cons"]),
                        te=[float(te[0][j]), float(te[1][j]), float(te[2][j])], ke=[float(ke[0][j]), float(ke[1][j]), float(ke[2][j])]))
    return out


def team_effects(path: Path, data_dir: Path, min_matches: int = 8) -> list[dict]:
    placings = json.load(open(data_dir / "dashboard/data/league_placings.json"))
    rows: dict[tuple[str, str], list[dict]] = {}
    for r in json.load(open(path)):
        if r.get("n_matches", min_matches) >= min_matches:
            rows.setdefault((CANON.get(r["team"], r["team"]), r["season"]), []).append(r)
    mean = lambda rs, k: round(sum(x[k] for x in rs) / len(rs), 3)  # noqa: E731
    out = []
    for lg in placings:
        seasons = []
        for s in lg["seasons"]:
            teams = []
            for t in s["table"]:
                rs = rows.get((t["team"], s["season"]))
                if rs:
                    teams.append(dict(team=t["short"], pos=t["position"],
                                      a=[mean(rs, "attack_mean"), mean(rs, "attack_lower"), mean(rs, "attack_upper")],
                                      d=[mean(rs, "defense_mean"), mean(rs, "defense_lower"), mean(rs, "defense_upper")]))
            if teams:
                seasons.append(dict(season=s["season"], status=s["status"], teams=teams))
        out.append(dict(league=lg["league"], label=lg["label"], seasons=seasons))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True, help="Rugby-Data checkout")
    ap.add_argument("--checkpoint", required=True, help="checkpoint name from fit_strength_history.py --save-as")
    ap.add_argument("--team-effects", type=Path, required=True, help="the --out of fit_strength_history.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--min-recorded", type=float, default=0.9)
    ap.add_argument("--min-minutes", type=float, default=900, help="players below this are not exported")
    args = ap.parse_args()

    model, post, df = load_model(args.data_dir, args.checkpoint, args.start_year, args.min_recorded)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = model_params(post, model)
    params["home_empirical"] = empirical_home(args.data_dir)
    params["n_players"], params["n_team_seasons"] = len(model._player_ids), len(model._team_season_ids)
    params["n_player_matches"] = int(len(df))
    (args.out_dir / "rugby_model.json").write_text(json.dumps(params, separators=(",", ":")))
    pl = players(post, model, df, args.min_minutes)
    (args.out_dir / "rugby_players.json").write_text(json.dumps(pl, separators=(",", ":")))
    te = team_effects(args.team_effects, args.data_dir)
    (args.out_dir / "rugby_team_effects.json").write_text(json.dumps(te, separators=(",", ":")))
    print(f"wrote model parameters, {len(pl)} players, {sum(len(s['teams']) for lg in te for s in lg['seasons'])} team-seasons")


if __name__ == "__main__":
    main()
