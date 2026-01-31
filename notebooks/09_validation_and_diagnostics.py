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
    temporal_train_test_split,
    validate_predictions,
    log_likelihood_scores,
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
# Compare observed data to predictions from the posterior to check model fit.

# TODO: Implement posterior predictive checks
# This requires computing predictions for each posterior sample
# and comparing to observed scoring events

print("Posterior predictive checks not yet implemented.")
print("This would compare observed vs predicted score distributions.")

# %%
# ## 5. Prediction Calibration
# 
# Assess whether predicted probabilities match observed frequencies on held-out data.

# Temporal train/test split
all_dates = df['date'].unique()
split_date = pd.Timestamp(np.percentile(all_dates, 80))

train_df = df[df['date'] < split_date].copy()
test_df = df[df['date'] >= split_date].copy()

print(f"Train set: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} records)")
print(f"Test set:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} records)")
print(f"\nTest set contains {test_df['player_name'].nunique()} players, {test_df.groupby(['date', 'team']).ngroups} matches")

# %%
# TODO: Compute predictions on test set
# This would predict matches in the test set and compare to actual outcomes

print("Calibration analysis not yet implemented.")
print("This would compute prediction accuracy and probability calibration.")

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
