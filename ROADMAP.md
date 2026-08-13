# Rugby Ranking — Roadmap

This is the living project roadmap. For the full development history and implementation details see [PLAN.md](PLAN.md).

---

## What's Working in Production (as of 2026-03-21)

- **Weekly VI training** via GitHub Actions, warm-started from MCMC Release checkpoints
- **Dashboard** deployed to GitHub Pages with team strength, player rankings, upcoming predictions
- **Match predictions**: teams-only and full-lineup, with uncertainty quantification
- **Squad analysis**: strength, depth, lineup prediction, injury impact
- **Paths to victory**: Six Nations CLI (`rugby analysis paths-to-victory`)
- **Knockout bracket forecasting**: bracket structure, conditional predictions, tournament simulation
- **Prediction archive**: auto-archives on `upcoming`/`predict`, `ingest-results` CLI for result ingestion, `calibration` CLI for accuracy reporting, `prediction_history.json` exported for blog widget

---

## Near-Term Priorities

### 1. Posterior predictive checks and model validation

**Infrastructure complete.** The validation toolchain is now in place:

- `model.sample_posterior_predictive(trace, df)` — rebuilds the PyMC model context from a checkpoint and runs `pm.sample_posterior_predictive()`; returns ArviZ `InferenceData` for use with `az.plot_ppc()`
- `validation.calibration_analysis(obs, ppc_samples)` — randomised PIT histogram and calibration curve for Poisson count predictions
- `notebooks/09_validation_and_diagnostics.py` — filled-in notebook covering PPC plots, PIT histograms, calibration curves for all score types, and match-level Brier scores from the prediction archive
- `tests/test_statistical_calibration.py` — SBC / P-P tests (the same methodology used in GW astronomy) for VI and MCMC, run with `pytest -m statistical`

**What still needs doing:**

- Run the notebook on the current MCMC checkpoint and inspect results
- Check whether the PIT histograms show VI underestimation (expected U-shaped pattern)
- Stratify calibration by competition/team/season to identify systematic biases
- Run full SBC (`pytest -m statistical`) to formally quantify VI vs MCMC calibration gap

**Why first**: Identifies where complexity is actually needed vs. where the current model is already good enough. This should inform all structural changes.

---

### 2. Review defensive model structure

**Code change complete.** `delta_defense` now only applies to `tries`. The `ModelConfig.defense_score_types` field (default `("tries",)`) controls which score types get the opponent defensive term. `lambda_defense` is now a scalar rather than a per-score-type vector.

**What still needs doing:**

- Re-run MCMC to get a new checkpoint with the corrected model structure (existing checkpoints are incompatible — `lambda_defense` shape changed from `n_score_types` to scalar)
- Compare predictive performance (Brier scores, PIT histograms) before/after to confirm improvement
- Decide whether penalty conceding rate deserves a separate attacking-pressure term

---

### 3. URC playoff / knockout predictions

URC playoffs begin ~May. The knockout forecasting infrastructure (Phase 4f) is implemented but two items remain:

- Integrate knockout paths with paths-to-victory analysis (pool + knockout combined)
- Knockout-specific visualisations (bracket with probabilities, stage survival curves)

**Effort**: Medium

---

### 4. Historical validation: Six Nations 2026 paths-to-victory

The Six Nations just ended. This is the best window to validate whether the paths-to-victory predictions matched actual outcomes:

- Did the model correctly identify critical games?
- Were the "must-win" scenarios accurate?
- How well-calibrated were the tournament finish probabilities?

**Effort**: Small (run PathsAnalyzer on historical Six Nations data, compare to outcomes)

---

## Deferred / Someday

These are real ideas but should wait until the model quality work (items 3–4) is done:

- **Survival component** for substitution/exposure modelling (currently uses simple exposure offset)
- **Time-varying effects** within season (player form trends) — implementation exists behind a config flag but not validated
- **Player-team interaction effects** (`δ_player×team`) for transfers
- **Career trajectories** (random walk player effects across seasons)
- **Non-linear within-season trends** (splines, GP)
- **Age-based effects** (requires DOB data, or we could use the period of time the player has been active in the data as a proxy, so career age rather than biological age)
- **Game state effects** (score differential, red card periods)
- **Phase 4b combinatorial enumeration** for late-tournament paths (MCMC simulation approach is sufficient for now)

---

## Known Issues

| Issue | File | Severity |
|-------|------|----------|
| ~~`recent_seasons_only=50` date filtering workaround~~ | `tools/export_dashboard_data.py` | Fixed |
| ~~`delta_defense` applied to conversions/penalties~~ | `model/core.py` | Fixed |
| ~~Prediction archiver not wired into pipeline~~ | `model/prediction_archive.py` | Fixed |
| `MatchPredictor.predict_teams_only` season arg required, not optional | Various docs fixed | Fixed |
