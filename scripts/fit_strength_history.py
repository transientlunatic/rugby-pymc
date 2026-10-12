"""Fit the Bayesian model over a long history and export team strengths per season.

The production export (``rugby_ranking.tools.export_dashboard_data``) fits only
the last few seasons, so ``team_strength_rankings.json`` starts in 2023-24.
This script fits the same joint model (``ModelConfig`` defaults of the
production export: defense and separate kicking effects, static team-season
effects) by ADVI on every season from ``--start-year``, then writes team
strengths for each season in the same row format as ``team_strength_rankings.json``.

Matches only count where scoring events were recorded: the model attributes
scores to players, so a team-match with lineups but no recorded scorers would
otherwise look like a team that scored nothing. A team-match is kept if the
recorded player points reach ``--min-recorded`` of the team's score (default
0.9). Seasons before 2011-12 have no scoring events at all, so cannot be fitted.

    python scripts/fit_strength_history.py --data-dir ../Rugby-Data \\
        --start-year 2011 --out team_strength_history.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rugby_ranking.model.core import ModelConfig, RugbyModel  # noqa: E402
from rugby_ranking.model.data import MatchDataset  # noqa: E402
from rugby_ranking.model.inference import InferenceConfig, ModelFitter  # noqa: E402
from rugby_ranking.model.predictions import MatchPredictor, estimate_kicking_rates  # noqa: E402
from rugby_ranking.model.strength_rankings import compute_team_rankings  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_observations(data_dir: Path, start_year: int, min_recorded: float):
    dataset = MatchDataset(str(data_dir), fuzzy_match_names=False)
    dataset.load_json_files()
    df = dataset.to_dataframe(played_only=True)
    df = df[df["position"].between(1, 23)].copy()
    df = df[df["season"].str.extract(r"(\d{4})")[0].astype(int) >= start_year]

    rec = df.groupby(["match_id", "team"])["total_points"].transform("sum")
    complete = (df["team_score"] > 0) & (rec >= min_recorded * df["team_score"])
    log(f"{len(df):,} player-match rows from {start_year}; keeping {int(complete.sum()):,} "
        f"({complete.mean():.0%}) where scoring events are recorded")
    return dataset, df[complete].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True, help="Rugby-Data checkout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-year", type=int, default=2011, help="first season start year to fit")
    ap.add_argument("--min-recorded", type=float, default=0.9)
    ap.add_argument("--vi-iterations", type=int, default=30000)
    ap.add_argument("--n-samples", type=int, default=1000, help="posterior draws per season for the reference margin")
    ap.add_argument("--save-as", help="also save the fitted model as this checkpoint name")
    args = ap.parse_args()

    dataset, df = load_observations(args.data_dir, args.start_year, args.min_recorded)
    log(f"seasons: {sorted(df['season'].unique())[0]} to {sorted(df['season'].unique())[-1]}, "
        f"{df['player_name'].nunique():,} players, {df['team'].nunique()} teams")

    model = RugbyModel(ModelConfig(include_defense=True, separate_kicking_effect=True,
                                   time_varying_effects=False))
    model.build_joint(df)
    log(f"built model: {len(model._player_ids):,} players, {len(model._team_season_ids)} team-seasons")

    t = time.time()
    fitter = ModelFitter(model, config=InferenceConfig(vi_n_iterations=args.vi_iterations, vi_method="advi"))
    trace = fitter.fit_vi(progressbar=False, random_seed=42)
    log(f"VI fit took {time.time() - t:.0f}s for {args.vi_iterations:,} iterations")
    hist = getattr(getattr(fitter, "_vi_approx", None), "hist", None)
    if hist is not None and len(hist) >= 4000:
        # a crude convergence check: mean loss over the last 1000 iterations vs the 1000 before
        a, b = float(hist[-2000:-1000].mean()), float(hist[-1000:].mean())
        log(f"loss: {a:.4g} (iterations -2000..-1000) -> {b:.4g} (last 1000), change {(b - a) / abs(a):+.2%}")
    if args.save_as:
        log(f"saved checkpoint to {fitter.save(args.save_as)}")

    # how many matches each team-season's estimate rests on (thin ones are mostly prior)
    counts = df.groupby(["team", "season"])["match_id"].nunique()
    predictor = MatchPredictor(model, trace, kicking_rates=estimate_kicking_rates(dataset.matches))
    rows = []
    for season in sorted(x for x in df["season"].unique() if re.fullmatch(r"\d{4}-\d{4}", x)):
        ranks = compute_team_rankings(predictor, season=season, n_samples=args.n_samples, seed=42)
        ranks = ranks[ranks["season"] == season].copy()   # the fallback can return other seasons
        ranks["n_matches"] = [int(counts.get((t, season), 0)) for t in ranks["team"]]
        rows.extend(ranks.to_dict("records"))
        log(f"{season}: {len(ranks)} teams")
    args.out.write_text(json.dumps(rows))
    log(f"wrote {len(rows)} team-season rows to {args.out}")


if __name__ == "__main__":
    main()
