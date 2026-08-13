# Model Validation

This guide covers the three complementary validation tools available for assessing model quality. Together they answer the key questions raised in [ROADMAP.md](../../ROADMAP.md):

- Are the predicted score distributions well-calibrated?
- Does the model systematically fail in specific scenarios?
- Are VI posterior uncertainty estimates reliable compared to MCMC?

---

## Overview

| Tool | What it tests | Speed | When to run |
|------|--------------|-------|-------------|
| **Posterior predictive checks** | Does the model structure fit the data? | Minutes | After any model change |
| **PIT calibration** | Are predicted count distributions accurate? | Minutes (uses PPC output) | After any model change |
| **Simulation-Based Calibration** | Is the inference engine itself calibrated? | Hours | Before/after major structural changes |

---

## Posterior Predictive Checks

A posterior predictive check (PPC) asks: *if we simulated new data from the fitted model, would it look like the data we actually observed?* Systematic discrepancies indicate that the model structure is wrong, not just imprecise.

### Running PPCs

After loading a checkpoint, call `sample_posterior_predictive()` on the model. This rebuilds the PyMC model context using the checkpoint's index mappings, filters the dataset to known entities, and runs `pm.sample_posterior_predictive()`.

```python
from rugby_ranking.notebook_utils import setup_notebook_environment, load_model_and_trace

dataset, df, _ = setup_notebook_environment()
model, trace = load_model_and_trace("latest")

# Generates posterior predictive samples for all score types.
# Rows with players/teams not in the checkpoint are silently filtered.
ppc = model.sample_posterior_predictive(trace, df, random_seed=42)
```

The returned `InferenceData` object has `posterior_predictive` and `observed_data` groups, directly usable with ArviZ:

```python
import arviz as az

az.plot_ppc(ppc, var_names=["y_tries"], observed=True)
```

### What to Look For

**Good fit:** The posterior predictive distribution (thin lines) closely envelopes the observed distribution (thick line).

**Over-dispersion:** Predicted distributions are wider than observed — model variance is too high.

**Under-dispersion:** Predicted distributions are narrower than observed — model variance is too low (common when using VI).

**Systematic bias:** The centres of the distributions don't align — the model over- or under-predicts at the mean.

The notebook `notebooks/09_validation_and_diagnostics.py` runs PPCs for all four score types and saves plots.

---

## PIT Calibration

The Probability Integral Transform (PIT) provides a formal statistical test for calibration. For each observation $y_i$, compute the fraction of posterior predictive samples that are less than $y_i$:

$$u_i = P(\tilde{y} < y_i)$$

Under a correctly specified model, $u_i \sim \text{Uniform}(0, 1)$.

For discrete count data (Poisson), the randomised PIT is used to handle probability mass at the observed value:

$$u_i = P(\tilde{y} < y_i) + V_i \cdot P(\tilde{y} = y_i), \quad V_i \sim \text{Uniform}(0, 1)$$

### Running PIT Calibration

```python
import numpy as np
from rugby_ranking.model.validation import calibration_analysis

# Extract observed counts and PPC samples for one score type
obs = ppc.observed_data["y_tries"].values
ppc_samples = ppc.posterior_predictive["y_tries"].values.reshape(-1, len(obs))

result = calibration_analysis(obs, ppc_samples, n_bins=10)
```

The returned dictionary contains:

| Key | Description |
|-----|-------------|
| `pit_values` | PIT value per observation (should be uniform) |
| `pit_histogram` | Bin counts (should be approximately flat) |
| `pit_expected_per_bin` | Expected count if uniform |
| `calibration_curve` | Binned predicted mean vs. observed mean |
| `mean_absolute_calibration_error` | Mean \|pred − obs\| per bin (lower is better) |
| `mean_predicted` / `mean_observed` | Overall mean comparison |

### Interpreting the PIT Histogram

| Shape | Interpretation |
|-------|---------------|
| Flat | Well-calibrated |
| U-shaped (high at 0 and 1) | Posterior too narrow (underestimates uncertainty) |
| Hump-shaped (high in centre) | Posterior too wide (overestimates uncertainty) |
| Skewed | Systematic bias in predicted mean |

A U-shaped PIT histogram is the characteristic signature of mean-field variational inference (ADVI), which tends to underestimate posterior variance.

---

## Simulation-Based Calibration

Simulation-Based Calibration (SBC) is the same test used in gravitational wave astronomy to validate Bayesian inference pipelines (Abbott et al. 2016, PRX 6 041015; Talts et al. 2018, arXiv:1804.06788). It directly answers: *does the inference engine recover the correct posterior?*

### The Method

1. Draw a parameter value and synthetic dataset jointly from the prior:
   $\theta^\star \sim p(\theta)$, $y^\star \sim p(y \mid \theta^\star)$
2. Run inference on $y^\star$ to obtain the posterior $p(\theta \mid y^\star)$
3. Compute the **rank** of $\theta^\star$ within the posterior samples:
   $r = \#\{\theta_s < \theta^\star\} / S$
4. Repeat $N$ times. The ranks $r_1, \ldots, r_N$ should be $\text{Uniform}(0, 1)$.

A P-P plot of the empirical CDF of ranks against the theoretical uniform diagonal visualises the calibration. Deviations indicate miscalibration.

### What SBC Reveals

| P-P plot deviation | Interpretation |
|--------------------|---------------|
| On the diagonal | Inference is well-calibrated for this parameter |
| Below diagonal at edges | Posterior too narrow (VI underestimation) |
| Above diagonal at centre | Posterior too wide |
| Shifted left or right | Posterior mean is biased |

### Running SBC Tests

SBC tests are in `tests/test_statistical_calibration.py` under the `statistical` marker. They are excluded from the normal test run and should be run when making major model changes.

```bash
# Run VI-based SBC (recommended starting point, ~15 min for n=30)
pytest tests/test_statistical_calibration.py -m statistical -v -s

# Run with fewer simulations for a quick check
pytest tests/test_statistical_calibration.py -m statistical -v -s --sbc-sims 10

# Run MCMC reference SBC (gold standard, several hours)
pytest tests/test_statistical_calibration.py -m "statistical and slow" -v -s
```

Each run saves a P-P plot to `tests/statistical_outputs/` for visual inspection.

### Parameters Tested

The tests currently cover scalar parameters only, as these are unambiguous across simulations:

| Parameter | Description |
|-----------|-------------|
| `alpha` | Per-score-type baseline rate |
| `eta_home` | Home advantage |
| `sigma_player` | Player effect scale (hyperparameter) |
| `sigma_team` | Team effect scale (hyperparameter) |

Vector parameters (individual player and team effects) are not tested because SBC for hierarchical random effects requires consistent label assignment across simulations, which is non-trivial in this setting.

### Using MCMC to Diagnose VI

The test suite provides both a VI test class (`TestSBC_VI`) and an MCMC reference class (`TestSBC_MCMC`):

- **VI passes, MCMC passes**: VI is adequately calibrated for these parameters.
- **VI fails, MCMC passes**: VI is underestimating uncertainty — a known limitation of mean-field ADVI. Consider whether this matters for the intended use case.
- **MCMC fails**: The model is mis-specified for these parameters regardless of the inference method.

---

## Recommended Workflow

### After any model change

```bash
# 1. Fit model on training data as usual
# 2. Run PPC (in notebook or directly):
python notebooks/09_validation_and_diagnostics.py

# 3. Check output plots and printed MACE values
```

### Before or after a structural model change

```bash
# Run SBC to verify inference calibration
pytest tests/test_statistical_calibration.py -m statistical -v -s
```

### Interpreting a failed SBC

A KS test failure (`p < 0.01`) at `n_sims=30` is strong evidence of miscalibration. Inspect the P-P plot in `tests/statistical_outputs/`:

1. **U-shaped PIT or below-diagonal P-P plot** → VI posterior too narrow. Check whether match predictions are also overconfident (see `PredictionArchiver.calibration_report()`).
2. **Skewed P-P plot** → bias in posterior mean. Check prior specification and whether the minimal dataset is representative.
3. **One parameter fails, others pass** → localised problem. Check that parameter's prior and likelihood contribution.

---

## Further Reading

- [Model Fundamentals](model_fundamentals) — model structure and equations
- [Weekly Workflow](weekly_workflow) — training and update process
- `notebooks/09_validation_and_diagnostics.py` — interactive PPC and calibration notebook
- `tests/test_statistical_calibration.py` — SBC test implementation with full documentation
- Talts et al. (2018), "Validating Bayesian Inference Algorithms with Simulation-Based Calibration", [arXiv:1804.06788](https://arxiv.org/abs/1804.06788)
