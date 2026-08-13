#!/usr/bin/env python
"""
# Model Diagnostics & Validation

This notebook demonstrates model quality assessment and inference diagnostics.

**Topics**:
1. Posterior trace diagnostics (Rhat, ESS, divergences)
2. Posterior predictive checks
3. Prediction calibration
4. Model comparison (static vs time-varying, with/without defense)
5. Train/test validation
"""

from rugby_ranking.notebook_utils import setup_notebook_environment, load_model_and_trace, print_summary
from rugby_ranking.model.validation import (
    temporal_split,
    compute_validation_metrics,
    calibration_analysis,
)
import arviz as az
import pandas as pd
import numpy as np

# Setup: load data and configure plots
dataset, df, model_dir = setup_notebook_environment()

# %%
# ## 1. Load Model
#
# Load a trained model checkpoint and examine its inference configuration.

model, trace = load_model_and_trace("latest")

print(f"Model Type: {'Time-varying' if model.config.time_varying_effects else 'Static'}")
print(f"Separate Kicking Effects: {model.config.separate_kicking_effect}")
print(f"Include Defense: {model.config.include_defense}")
print(f"\nPosterior dimensions:")
print(f"  Chains: {trace.posterior.dims['chain']}")
print(f"  Draws: {trace.posterior.dims['draw']}")
print(f"  Warmup: {trace.posterior.dims.get('warmup', 'N/A')}")

# %%
# ## 2. Trace Diagnostics
#
# Assess whether the MCMC chain has converged using Rhat (should be < 1.01) and ESS (effective sample size).

# Summary of key parameters
summary = az.summary(
    trace,
    var_names=['alpha', 'sigma_player_try', 'sigma_team'],
    kind='stats',
)

print("Key Parameter Summary:")
print(summary)

# Check for convergence issues
print("\nConvergence Check:")
rhat = summary['r_hat']
problems = (rhat > 1.01).sum()
if problems == 0:
    print("✓ All Rhat < 1.01 (good convergence)")
else:
    print(f"✗ {problems} parameters have Rhat > 1.01 (check convergence)")

# %%
# ESS Ratio (effective sample size / total samples)
print("ESS Ratios (should be > 0.1):")
ess_bulk = summary['ess_bulk'] / (trace.posterior.dims['chain'] * trace.posterior.dims['draw'])
ess_tail = summary['ess_tail'] / (trace.posterior.dims['chain'] * trace.posterior.dims['draw'])

print(f"  Bulk: {ess_bulk.min():.3f} - {ess_bulk.max():.3f}")
print(f"  Tail: {ess_tail.min():.3f} - {ess_tail.max():.3f}")

low_ess = (ess_bulk < 0.1).sum() + (ess_tail < 0.1).sum()
if low_ess == 0:
    print("✓ All ESS ratios acceptable")
else:
    print(f"⚠️  {low_ess} parameters have low ESS")

# %%
# ## 3. Divergences
#
# Check for post-warmup divergences (indicates sampling difficulties).

# Check for divergences
if 'diverging' in trace.sample_stats.data_vars:
    divergences = trace.sample_stats.diverging.sum()
    total = trace.posterior.dims['chain'] * trace.posterior.dims['draw']
    div_pct = (divergences / total) * 100

    if divergences == 0:
        print(f"✓ No divergences (good)")
    elif div_pct < 1:
        print(f"⚠️  {divergences} divergences ({div_pct:.1f}%, acceptable)")
    else:
        print(f"✗ {divergences} divergences ({div_pct:.1f}%, consider re-tuning)")
else:
    print("(Divergence information not available in trace)")

# %%
# ## 4. Posterior Predictive Checks
#
# Compare observed data distributions to replicated datasets drawn from the
# posterior predictive. For a well-specified model the two should agree.
#
# model.sample_posterior_predictive() rebuilds the PyMC model context using
# the checkpoint's index mappings, filters df to known entities, and calls
# pm.sample_posterior_predictive() — so this works even after load_checkpoint().

print("Running posterior predictive checks (this may take a few minutes)...")
ppc = model.sample_posterior_predictive(trace, df, random_seed=42)
print("✓ PPC sampling complete.")

# %%
# ### 4a. Observed vs predicted distributions (az.plot_ppc)
#
# For each score type, the thin lines are draws from the posterior predictive;
# the thick line is the observed distribution. Close agreement means the model
# captures the marginal count distributions.

import matplotlib.pyplot as plt

score_types = list(model.config.score_types)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Posterior Predictive Checks — observed vs predicted distributions")

for ax, score_type in zip(axes.flat, score_types):
    var_name = f"y_{score_type}"
    if var_name not in ppc.posterior_predictive:
        ax.set_title(f"{score_type} (not available)")
        continue

    # Observed counts
    obs = ppc.observed_data[var_name].values
    # PPC samples (shape: chain × draw × obs) → flatten to (n_samples, n_obs)
    ppc_samples = ppc.posterior_predictive[var_name].values.reshape(-1, len(obs))

    # Plot: distribution of mean predicted counts vs observed mean
    ppc_means = ppc_samples.mean(axis=1)  # mean over observations for each draw
    obs_mean = obs.mean()

    ax.hist(ppc_means, bins=50, alpha=0.6, color="steelblue", label="Predicted means")
    ax.axvline(obs_mean, color="red", linewidth=2, label=f"Observed mean ({obs_mean:.3f})")
    ax.set_title(score_type)
    ax.set_xlabel("Mean count per player-match")
    ax.set_ylabel("Posterior predictive draws")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ppc_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: ppc_distributions.png")

# %%
# ### 4b. PIT histogram and calibration curves
#
# The Probability Integral Transform (PIT) should be uniform if the model is
# well-calibrated. Systematic deviations indicate over/under-dispersion or
# bias in specific count ranges.

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("PIT Histograms and Calibration Curves")

for col, score_type in enumerate(score_types):
    var_name = f"y_{score_type}"
    if var_name not in ppc.posterior_predictive:
        continue

    obs = ppc.observed_data[var_name].values
    ppc_samples = ppc.posterior_predictive[var_name].values.reshape(-1, len(obs))

    cal = calibration_analysis(obs, ppc_samples, n_bins=10)

    # PIT histogram
    ax_pit = axes[0, col]
    bin_edges = np.array(cal["pit_bin_edges"])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax_pit.bar(bin_centers, cal["pit_histogram"], width=0.09, alpha=0.7, color="steelblue")
    ax_pit.axhline(cal["pit_expected_per_bin"], color="red", linestyle="--", label="Expected (uniform)")
    ax_pit.set_title(f"{score_type}\nPIT histogram")
    ax_pit.set_xlabel("PIT value")
    ax_pit.set_ylabel("Count")
    ax_pit.legend(fontsize=7)

    # Calibration curve
    ax_cal = axes[1, col]
    if cal["calibration_curve"]:
        pred_m = [b["pred_mean"] for b in cal["calibration_curve"]]
        obs_m = [b["obs_mean"] for b in cal["calibration_curve"]]
        ax_cal.scatter(pred_m, obs_m, s=40, color="steelblue", zorder=3)
        lim = max(max(pred_m), max(obs_m)) * 1.05
        ax_cal.plot([0, lim], [0, lim], "r--", label="Perfect calibration")
        ax_cal.set_title(f"Calibration curve\nMACE={cal['mean_absolute_calibration_error']:.4f}")
        ax_cal.set_xlabel("Mean predicted count")
        ax_cal.set_ylabel("Mean observed count")
        ax_cal.legend(fontsize=7)

    print(
        f"{score_type:15s}  pred_mean={cal['mean_predicted']:.4f}  "
        f"obs_mean={cal['mean_observed']:.4f}  "
        f"MACE={cal['mean_absolute_calibration_error']:.4f}"
    )

plt.tight_layout()
plt.savefig("ppc_calibration.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: ppc_calibration.png")

# %%
# ## 5. Match-level Prediction Calibration
#
# Player-level PPC checks the score-count model. For match outcome calibration
# (home/draw/away win probabilities and CI coverage) we use the prediction
# archive, which stores historical predictions and compares them to actual results.

try:
    from rugby_ranking.model.prediction_archive import PredictionArchiver
    archiver = PredictionArchiver()
    report = archiver.calibration_report()

    if report.get("n", 0) == 0:
        print("No archived predictions with results yet.")
        print("Run `rugby analysis calibration` after results are ingested.")
    else:
        print(f"\n=== Match Prediction Calibration (n={report['n']}) ===")
        print(f"Outcome accuracy  : {report['outcome_accuracy']:.1%}")
        print(f"Brier score       : {report['brier_score']:.4f}  (0=perfect, 1=worst)")
        print(f"Mean home error   : {report['mean_home_error']:+.1f} pts")
        print(f"Mean away error   : {report['mean_away_error']:+.1f} pts")
        print(f"MAE margin        : {report['mae_margin']:.1f} pts")
        print(f"Home 90% CI cov.  : {report['home_ci_coverage']:.1%}  (target: 90%)")
        print(f"Away 90% CI cov.  : {report['away_ci_coverage']:.1%}  (target: 90%)")

        # Brier skill score vs. reference (uniform 1/3 each)
        brier_ref = 2 / 3  # Uniform 1/3 over three outcomes: (1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2 = 2/3
        bss = 1.0 - report['brier_score'] / brier_ref
        print(f"Brier skill score : {bss:.3f}  (>0 beats reference)")
except FileNotFoundError:
    print("Prediction archive not found. Run predictions with `rugby predict` and ingest results first.")

# %%
# ## 6. Model Comparison
#
# Compare different model variants (static vs time-varying, with/without defense).

# TODO: Load multiple models and compare
# models = {
#     'static': load_model_and_trace('static_model'),
#     'time_varying': load_model_and_trace('timevarying_model'),
#     'defense': load_model_and_trace('defense_model'),
# }

print("Model comparison not yet implemented.")
print("This would compare LOO-CV scores and other metrics across models.")
