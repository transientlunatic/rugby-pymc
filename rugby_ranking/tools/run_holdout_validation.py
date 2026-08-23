"""
Automated out-of-sample holdout validation: fits the production model on all
but the most recent slice of matches, evaluates win accuracy / Brier score /
score RMSE on the held-out matches it never saw, and compares against the
cheapest possible skill-free baseline (computed from training data only).

This is the tool version of the one-off validation run recorded in
validation_reports/2026-08-23_teams-only-holdout.json -- see that file and
validation_reports/2026-08-23_run_script.py for the original run and the
methodology writeup in MODEL_EXPLAINED.md. Promoted here so it can run on a
schedule (see .github/workflows/deploy-dashboard.yml) instead of being a
one-off someone has to remember to re-run.

Design notes
------------
- The holdout split is relative to `--as-of` (defaults to today), so this
  produces a genuinely new out-of-sample check each time it runs, not a
  replay of the same fixed 2026-08-23 snapshot.
- This does NOT gate the build on metric thresholds. There is exactly one
  historical data point (the 2026-08-23 run) -- nowhere near enough to set a
  sane regression threshold without either blocking on noise (VI has run-to-
  run Monte Carlo variance) or picking an arbitrary number. It fails loudly
  only if the run itself is broken (e.g. every single prediction errors),
  and otherwise always writes the numbers so a real history can accumulate.
  Add a threshold-based gate once there's a few weeks of runs to calibrate
  against -- see ROADMAP.md.

Usage:
    python -m rugby_ranking.tools.run_holdout_validation \
        --data-dir Rugby-Data --out-dir dashboard/data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, InferenceConfig
from rugby_ranking.model.predictions import MatchPredictor
from rugby_ranking.model.validation import calibration_analysis


def _make_logger(out_dir: Path):
    log_path = out_dir / "holdout_validation_log.txt"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return log


def _match_level_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match: home/away team, score, season -- from home-team rows."""
    home_rows = df[df["is_home"] == 1].drop_duplicates("match_id")
    matches = home_rows[
        ["match_id", "team", "opponent", "season", "team_score", "opponent_score", "date"]
    ].copy()
    matches = matches.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            "team_score": "home_score",
            "opponent_score": "away_score",
        }
    )
    return matches.dropna(subset=["home_score", "away_score"])


def run_holdout_validation(
    data_dir: Path,
    out_dir: Path,
    as_of: str | None = None,
    n_recent_seasons: int = 3,
    test_fraction: float = 0.15,
    vi_iterations: int = 20000,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(out_dir)

    t_start = time.time()
    log(f"Loading Rugby-Data from {data_dir} (fuzzy_match_names=False)...")
    ds = MatchDataset(str(data_dir), fuzzy_match_names=False)
    ds.load_json_files()
    df = ds.to_dataframe(played_only=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    as_of_ts = pd.Timestamp(as_of, tz="UTC") if as_of else pd.Timestamp.now(tz="UTC")
    df = df[df["date"] <= as_of_ts]
    df["season_start_year"] = df["season"].str.extract(r"(\d{4})").astype(int)
    recent_years = sorted(df["season_start_year"].unique())[-n_recent_seasons:]
    df = df[df["season_start_year"].isin(recent_years)].copy()
    log(f"Recent-seasons slice: years={recent_years}, rows={len(df)}, "
        f"matches={df['match_id'].nunique()}, players={df['player_name'].nunique()}, "
        f"teams={df['team'].nunique()}")

    # ---- temporal train/test split (last N% of matches by date) -----------
    match_dates = df.groupby("match_id")["date"].first().sort_values()
    n_matches = len(match_dates)
    n_test = max(1, int(round(n_matches * test_fraction)))
    test_match_ids = set(match_dates.iloc[-n_test:].index)
    train_match_ids = set(match_dates.iloc[:-n_test].index)

    df_train = df[df["match_id"].isin(train_match_ids)].copy()
    df_test = df[df["match_id"].isin(test_match_ids)].copy()
    log(f"Split: {len(train_match_ids)} train matches / {len(test_match_ids)} test matches "
        f"(test date range {match_dates.iloc[-n_test:].min()} to {match_dates.iloc[-n_test:].max()})")

    # ---- build + fit (production config, VI) -------------------------------
    config = ModelConfig(include_defense=True, separate_kicking_effect=True)
    model = RugbyModel(config=config)
    log("Building joint model on training data...")
    model.build_joint(df_train)

    log(f"Fitting VI (ADVI, n={vi_iterations})...")
    t0 = time.time()
    fitter = ModelFitter(model, config=InferenceConfig(vi_n_iterations=vi_iterations, vi_method="advi"))
    trace = fitter.fit_vi(progressbar=False, random_seed=42)
    fit_time = time.time() - t0
    log(f"VI fit done in {fit_time:.1f}s ({fit_time/vi_iterations*1000:.2f} ms/iter)")

    report: dict = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of": as_of_ts.isoformat(),
        "data_slice": {
            "recent_season_start_years": [int(y) for y in recent_years],
            "n_rows_total": int(len(df)),
            "n_train_matches": len(train_match_ids),
            "n_test_matches": len(test_match_ids),
        },
        "fit": {"method": "vi/advi", "n_iterations": vi_iterations, "fit_seconds": fit_time},
    }

    # ---- out-of-sample posterior-predictive calibration (PIT/MACE) --------
    log("Sampling posterior predictive on held-out test rows...")
    known_test_rows = model._filter_to_known_entities(df_test)
    report["n_test_rows_known_to_model"] = int(len(known_test_rows))
    report["n_test_rows_total"] = int(len(df_test))
    if len(known_test_rows) > 0:
        ppc = model.sample_posterior_predictive(trace, df_test, random_seed=42)
        calibration = {}
        for score_type in config.score_types:
            var_name = f"y_{score_type}"
            if var_name not in ppc.posterior_predictive:
                continue
            obs = ppc.observed_data[var_name].values
            samples = ppc.posterior_predictive[var_name].values.reshape(-1, len(obs))
            cal = calibration_analysis(obs, samples, n_bins=10)
            calibration[score_type] = {
                "n_observations": cal["n_observations"],
                "mean_predicted": cal["mean_predicted"],
                "mean_observed": cal["mean_observed"],
                "mean_absolute_calibration_error": cal["mean_absolute_calibration_error"],
            }
            log(f"  OOS {score_type:12s} pred_mean={cal['mean_predicted']:.4f} "
                f"obs_mean={cal['mean_observed']:.4f} MACE={cal['mean_absolute_calibration_error']:.4f}")
        report["out_of_sample_calibration"] = calibration
    else:
        log("  No test rows overlap with training index -- skipping PPC calibration")
        report["out_of_sample_calibration"] = {"skipped": True}

    # ---- match-level accuracy / Brier score on the test set ---------------
    log("Building match-level test set and running teams-only predictions...")
    matches = _match_level_frame(df_test)
    log(f"  {len(matches)} test matches with recorded scores")

    predictor = MatchPredictor(model, trace=trace)
    actual_outcomes, briers, sq_errors_home, sq_errors_away, abs_errors = [], [], [], [], []
    n_ok, n_failed = 0, 0
    for _, row in matches.iterrows():
        try:
            pred = predictor.predict_teams_only(row["home_team"], row["away_team"], row["season"], n_samples=500)
        except Exception:  # noqa: BLE001 -- want to count failures, not crash the run
            n_failed += 1
            continue
        n_ok += 1
        home_score, away_score = row["home_score"], row["away_score"]
        if home_score > away_score:
            actual = "home"
        elif away_score > home_score:
            actual = "away"
        else:
            actual = "draw"
        probs = {"home": pred.home_win_prob, "away": pred.away_win_prob, "draw": pred.draw_prob}
        predicted_winner = max(probs, key=probs.get)
        actual_outcomes.append(predicted_winner == actual)

        onehot = {"home": [1, 0, 0], "away": [0, 1, 0], "draw": [0, 0, 1]}[actual]
        p = [probs["home"], probs["away"], probs["draw"]]
        briers.append(sum((pi - oi) ** 2 for pi, oi in zip(p, onehot)))

        sq_errors_home.append((pred.home.mean - home_score) ** 2)
        sq_errors_away.append((pred.away.mean - away_score) ** 2)
        abs_errors.append(abs(pred.home.mean - home_score))
        abs_errors.append(abs(pred.away.mean - away_score))

    log(f"  predictions ok={n_ok} failed={n_failed}")
    if n_ok > 0:
        model_accuracy = float(np.mean(actual_outcomes))
        model_brier = float(np.mean(briers))
        model_rmse = float(np.sqrt(np.mean(sq_errors_home + sq_errors_away)))
        model_mae = float(np.mean(abs_errors))
        log(f"  MODEL   win_accuracy={model_accuracy:.1%} brier={model_brier:.4f} "
            f"score_RMSE={model_rmse:.2f} score_MAE={model_mae:.2f}")
    else:
        model_accuracy = model_brier = model_rmse = model_mae = None

    # ---- trivial baseline: fixed rate/mean from TRAINING data only --------
    # Computed from df_train, never from `matches` (the test set) -- using
    # the test set here would leak test outcomes into the "skill-free"
    # baseline and unfairly inflate its apparent performance (a real bug
    # caught in review on the original one-off run -- see PR #1).
    train_matches = _match_level_frame(df_train)
    home_win_rate = float((train_matches["home_score"] > train_matches["away_score"]).mean())
    away_win_rate = float((train_matches["away_score"] > train_matches["home_score"]).mean())
    draw_rate = float((train_matches["home_score"] == train_matches["away_score"]).mean())
    baseline_winner = max(
        {"home": home_win_rate, "away": away_win_rate, "draw": draw_rate},
        key=lambda k: {"home": home_win_rate, "away": away_win_rate, "draw": draw_rate}[k],
    )
    mean_home_score = float(train_matches["home_score"].mean())
    mean_away_score = float(train_matches["away_score"].mean())

    baseline_correct = []
    baseline_briers = []
    baseline_sq_errors = []
    for _, row in matches.iterrows():
        if row["home_score"] > row["away_score"]:
            actual = "home"
        elif row["away_score"] > row["home_score"]:
            actual = "away"
        else:
            actual = "draw"
        baseline_correct.append(baseline_winner == actual)
        onehot = {"home": [1, 0, 0], "away": [0, 1, 0], "draw": [0, 0, 1]}[actual]
        p = [home_win_rate, away_win_rate, draw_rate]
        baseline_briers.append(sum((pi - oi) ** 2 for pi, oi in zip(p, onehot)))
        baseline_sq_errors.append((mean_home_score - row["home_score"]) ** 2)
        baseline_sq_errors.append((mean_away_score - row["away_score"]) ** 2)

    baseline_accuracy = float(np.mean(baseline_correct))
    baseline_brier = float(np.mean(baseline_briers))
    baseline_rmse = float(np.sqrt(np.mean(baseline_sq_errors)))
    log(f"  BASELINE (train-set rates; always predict '{baseline_winner}') "
        f"win_accuracy={baseline_accuracy:.1%} brier={baseline_brier:.4f} "
        f"score_RMSE={baseline_rmse:.2f} (predicting train-set mean score every time)")

    report["match_level_test"] = {
        "n_matches": len(matches),
        "n_predicted_ok": n_ok,
        "n_predicted_failed": n_failed,
        "model": {
            "win_accuracy": model_accuracy,
            "brier_score": model_brier,
            "score_rmse": model_rmse,
            "score_mae": model_mae,
        },
        "baseline_always_predict_mode_outcome": {
            "computed_from": "training set only (no test-set leakage)",
            "outcome_predicted": baseline_winner,
            "win_accuracy": baseline_accuracy,
            "brier_score": baseline_brier,
            "score_rmse_predicting_mean_score": baseline_rmse,
            "home_win_rate": home_win_rate,
            "away_win_rate": away_win_rate,
            "draw_rate": draw_rate,
        },
    }

    report["total_wall_seconds"] = time.time() - t_start
    out_path = out_dir / "holdout_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"DONE. Report written to {out_path}. Total wall time {report['total_wall_seconds']:.1f}s")

    if n_ok == 0:
        raise RuntimeError(
            "Every single teams-only prediction failed on the holdout set -- "
            "the validation run itself is broken (not just the model), see log."
        )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("Rugby-Data"),
        help="Path to the Rugby-Data checkout (default: ./Rugby-Data, matching how CI checks it out)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("dashboard/data"),
        help="Where to write the log and JSON report (default: ./dashboard/data)",
    )
    parser.add_argument("--as-of", default=None, help="Only use matches on/before this date (default: today)")
    parser.add_argument("--n-recent-seasons", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--vi-iterations", type=int, default=20000)
    args = parser.parse_args()

    try:
        run_holdout_validation(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            as_of=args.as_of,
            n_recent_seasons=args.n_recent_seasons,
            test_fraction=args.test_fraction,
            vi_iterations=args.vi_iterations,
        )
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
