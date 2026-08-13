"""
Model health diagnostics against a trained checkpoint.

Runs the checks from notebooks/09_validation_and_diagnostics.py as a
non-interactive script so they can be scheduled in CI:

1. Trace convergence (Rhat, ESS, divergences) on the checkpoint's scalar
   hyperparameters.
2. Posterior predictive calibration (PIT-based) against real match data —
   does the model's generative distribution match what's actually observed?
3. Match-outcome calibration from the prediction archive (Brier score,
   outcome accuracy, 90% CI coverage) — how has the model actually done on
   real upcoming-match forecasts, if any have been archived and scored yet?

This does NOT run simulation-based calibration (the P-P tests in
tests/test_statistical_calibration.py) — those validate the inference
*procedure* using synthetic data and are expensive (~30-40 min with VI).
Run them separately: pytest -m "statistical and not slow".
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import arviz as az

from rugby_ranking.model.core import ModelConfig, RugbyModel
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.inference import ModelFitter
from rugby_ranking.model.validation import calibration_analysis

_SCALAR_PARAMS = [
    "alpha", "eta_home", "sigma_team", "sigma_defense",
    "sigma_player_try", "sigma_player_kick", "lambda_defense",
]


def run_diagnostics(
    checkpoint_name: str,
    data_dir: Path,
    archive_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """
    Run trace + posterior-predictive + match-outcome diagnostics for a
    checkpoint, using real match data (not simulated).

    Returns a JSON-serializable report dict; also writes it to output_path
    if given.
    """
    data_dir = Path(data_dir)
    cache_dir = Path("~/.cache/rugby_ranking").expanduser()

    print(f"Loading checkpoint: {checkpoint_name}")
    temp_trace = az.from_netcdf(cache_dir / checkpoint_name / "trace.nc")
    has_time_varying = "gamma_team_base_raw" in temp_trace.posterior.data_vars

    config = ModelConfig(
        include_defense=True,
        separate_kicking_effect=True,
        time_varying_effects=has_time_varying,
    )
    model = RugbyModel(config=config)
    fitter = ModelFitter.load(checkpoint_name, model)
    trace = fitter.trace

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_name,
    }

    # --- 1. Trace diagnostics -----------------------------------------------
    # Rhat/ESS/divergences are MCMC-specific — they need multiple independent
    # chains to be meaningful. VI checkpoints (a single "chain" of draws from
    # the variational approximation, no sample_stats) skip straight to
    # reporting posterior summary stats; SBC (tests/test_statistical_calibration.py)
    # is the applicable calibration check for those.
    print("Checking trace diagnostics...")
    scalar_params = [p for p in _SCALAR_PARAMS if p in trace.posterior.data_vars]
    is_mcmc = trace.posterior.sizes.get("chain", 1) > 1 and "sample_stats" in trace.groups()

    if is_mcmc:
        summary = az.summary(trace, var_names=scalar_params, kind="stats")
        n_draws = trace.posterior.sizes["chain"] * trace.posterior.sizes["draw"]
        rhat_max = float(summary["r_hat"].max())
        ess_bulk_min_ratio = float((summary["ess_bulk"] / n_draws).min())
        ess_tail_min_ratio = float((summary["ess_tail"] / n_draws).min())
        n_divergences = (
            int(trace.sample_stats["diverging"].sum())
            if "diverging" in trace.sample_stats.data_vars
            else None
        )
        report["trace_diagnostics"] = {
            "inference_method": "mcmc",
            "scalar_params_checked": scalar_params,
            "rhat_max": rhat_max,
            "rhat_ok": rhat_max < 1.01,
            "ess_bulk_min_ratio": ess_bulk_min_ratio,
            "ess_tail_min_ratio": ess_tail_min_ratio,
            "ess_ok": ess_bulk_min_ratio > 0.1 and ess_tail_min_ratio > 0.1,
            "n_divergences": n_divergences,
        }
        print(
            f"  Rhat max={rhat_max:.4f}  "
            f"ESS bulk/tail min ratio={ess_bulk_min_ratio:.3f}/{ess_tail_min_ratio:.3f}  "
            f"divergences={n_divergences}"
        )
    else:
        report["trace_diagnostics"] = {
            "inference_method": "vi",
            "scalar_params_checked": scalar_params,
            "note": (
                "Rhat/ESS/divergences don't apply to a single VI approximation. "
                "See tests/test_statistical_calibration.py (SBC / P-P tests) for "
                "the applicable calibration check on VI posteriors."
            ),
        }
        print("  VI checkpoint — Rhat/ESS not applicable, skipping (see SBC tests instead)")

    # --- 2. Posterior predictive calibration on real data ------------------
    print("Loading match data...")
    dataset = MatchDataset(data_dir, fuzzy_match_names=False)
    dataset.load_json_files()
    df = dataset.to_dataframe(played_only=True)
    report["n_observations_available"] = int(len(df))

    known_rows = len(model._filter_to_known_entities(df))
    if known_rows == 0:
        print(
            "  Skipping: none of the players/team-seasons in --data-dir match "
            "this checkpoint's index mappings (checkpoint is likely trained on "
            "a different data slice than what's on disk now)."
        )
        report["posterior_predictive_calibration"] = {
            "skipped": True,
            "reason": "no overlapping players/team-seasons between checkpoint and data_dir",
        }
    else:
        print(f"Sampling posterior predictive on {known_rows:,} matching rows "
              "(this may take a few minutes)...")
        ppc = model.sample_posterior_predictive(trace, df, random_seed=42)

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
            print(
                f"  {score_type:12s} pred_mean={cal['mean_predicted']:.4f}  "
                f"obs_mean={cal['mean_observed']:.4f}  "
                f"MACE={cal['mean_absolute_calibration_error']:.4f}"
            )
        report["posterior_predictive_calibration"] = calibration

    # --- 3. Match-outcome calibration from the prediction archive ----------
    if archive_dir is not None:
        from rugby_ranking.model.prediction_archive import PredictionArchiver

        print("Checking match-outcome calibration from prediction archive...")
        archiver = PredictionArchiver(archive_dir=Path(archive_dir))
        match_calibration = archiver.calibration_report()
        report["match_outcome_calibration"] = match_calibration
        if match_calibration.get("n", 0) > 0:
            print(
                f"  n={match_calibration['n']}  "
                f"accuracy={match_calibration['outcome_accuracy']:.1%}  "
                f"brier={match_calibration['brier_score']:.4f}  "
                f"home CI coverage={match_calibration['home_ci_coverage']:.1%}  "
                f"away CI coverage={match_calibration['away_ci_coverage']:.1%}"
            )
        else:
            print("  No archived predictions with results yet.")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model health diagnostics against a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint name (e.g. weekly-model)")
    parser.add_argument("--data-dir", required=True, type=Path, help="Path to Rugby-Data json/ directory")
    parser.add_argument(
        "--archive-dir", type=Path, default=None,
        help="Path to prediction archive (e.g. Rugby-Data/dashboard/data/prediction_archive)"
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON report")
    args = parser.parse_args()

    run_diagnostics(
        checkpoint_name=args.checkpoint,
        data_dir=args.data_dir,
        archive_dir=args.archive_dir,
        output_path=args.output,
    )
