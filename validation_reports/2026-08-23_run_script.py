"""
A real, out-of-sample validation run against actual Rugby-Data, replacing the
placeholder numbers in MODEL_EXPLAINED.md with measured ones.

Config:
- Data: last 3 seasons by season_start_year, matches on/before 2026-08-23
  (the CI/production default in export_dashboard_data.py, but with a proper
  date-based season filter instead of the buggy `iloc[-3:]` on max-date-sorted
  season groups, which picks up out-of-order seasons on this dataset).
- Model: production config (include_defense=True, separate_kicking_effect=True),
  fit via VI/ADVI (the "fast weekly" path), fuzzy_match_names=False (matches
  run_diagnostics.py's documented perf tradeoff -- fuzzy matching alone did not
  finish in 90s on the full dataset).
- Split: temporal holdout -- last 15% of matches by date are test, never seen
  during fitting. This is what MODEL_EXPLAINED.md's accuracy table implicitly
  claims to report and never actually computed.
- Baseline: "always predict home win" using the empirical home-win rate and
  mean score computed from the TRAINING set only, applied as a fixed
  prediction to every test match -- the cheapest possible skill-free
  reference point that still respects the train/test boundary. (An earlier
  version of this script computed the baseline's rates from the test set
  itself -- data leakage caught in review -- which inflated the baseline's
  apparent performance. Fixed here.)

Usage:
    python 2026-08-23_run_script.py --data-dir /path/to/Rugby-Data --out-dir /path/to/output
Both default to sensible relative locations (see argparse below) so this is
rerunnable by others, not just reproducible on the machine that produced it.
"""
from __future__ import annotations

import argparse
import json
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
    log_path = out_dir / "validation_log.txt"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("Rugby-Data"),
        help="Path to the Rugby-Data checkout (default: ./Rugby-Data, matching how CI checks it out)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).parent,
        help="Where to write the log and JSON report (default: next to this script)",
    )
    parser.add_argument("--as-of", default="2026-08-23", help="Only use matches on/before this date")
    parser.add_argument("--n-recent-seasons", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--vi-iterations", type=int, default=20000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(args.out_dir)

    t_start = time.time()
    log(f"Loading Rugby-Data from {args.data_dir} (fuzzy_match_names=False)...")
    ds = MatchDataset(str(args.data_dir), fuzzy_match_names=False)
    ds.load_json_files()
    df = ds.to_dataframe(played_only=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    as_of = pd.Timestamp(args.as_of, tz="UTC")
    df = df[df["date"] <= as_of]
    df["season_start_year"] = df["season"].str.extract(r"(\d{4})").astype(int)
    recent_years = sorted(df["season_start_year"].unique())[-args.n_recent_seasons:]
    df = df[df["season_start_year"].isin(recent_years)].copy()
    log(f"Recent-seasons slice: years={recent_years}, rows={len(df)}, "
        f"matches={df['match_id'].nunique()}, players={df['player_name'].nunique()}, "
        f"teams={df['team'].nunique()}")

    # ---- temporal train/test split (last N% of matches by date) -----------
    match_dates = df.groupby("match_id")["date"].first().sort_values()
    n_matches = len(match_dates)
    n_test = max(1, int(round(n_matches * args.test_fraction)))
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

    n_iter = args.vi_iterations
    log(f"Fitting VI (ADVI, n={n_iter})...")
    t0 = time.time()
    fitter = ModelFitter(model, config=InferenceConfig(vi_n_iterations=n_iter, vi_method="advi"))
    trace = fitter.fit_vi(progressbar=False, random_seed=42)
    fit_time = time.time() - t0
    log(f"VI fit done in {fit_time:.1f}s ({fit_time/n_iter*1000:.2f} ms/iter)")

    checkpoint_path = fitter.save("real_validation_run")
    log(f"Checkpoint saved to {checkpoint_path}")

    report: dict = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_slice": {
            "recent_season_start_years": [int(y) for y in recent_years],
            "n_rows_total": int(len(df)),
            "n_train_matches": len(train_match_ids),
            "n_test_matches": len(test_match_ids),
        },
        "fit": {"method": "vi/advi", "n_iterations": n_iter, "fit_seconds": fit_time},
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
    # baseline and unfairly inflate its apparent performance.
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
    out_path = args.out_dir / "real_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"DONE. Report written to {out_path}. Total wall time {report['total_wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
