# Rugby Ranking — Roadmap

This is the living project roadmap. For the full development history and implementation details see [PLAN.md](PLAN.md).

---

## What's Working in Production (corrected 2026-08-23 — see below)

**This section previously described the intended pipeline, not the actual
one.** Checked against real evidence (CI run history, repo contents) on
2026-08-23:

- **Dashboard deployment**: ❌ **not actually working.** Every scheduled/push
  run of `deploy-dashboard.yml` had failed (33/33 runs since Feb 2026),
  always within seconds, on `ModuleNotFoundError: No module named 'sklearn'`
  — an undeclared dependency. No `gh-pages` branch has ever existed for this
  repo. Fixed by adding scikit-learn and scipy to `pyproject.toml` and
  installing via `pip install -e .[viz]` in CI instead of a hand-maintained
  package list — first real deploy attempt is the next scheduled/manual run.
- **Weekly VI training warm-started from MCMC checkpoints**: ❌ **doesn't
  exist as automation.** No `train-model-weekly.yml` workflow exists (it's
  referenced by name in `tests/test_statistical_calibration.py` docstrings
  but was never created). The only workflow cold-starts a fresh VI fit from
  scratch each time it runs. Separately, warm-start *persistence* is itself
  broken: `ModelFitter.save()` cannot pickle the VI approximation object on
  the currently pinned PyMC version (`'functools.partial' object has no
  attribute '__name__'`, caught and silently discarded) — so even a wired-up
  weekly job couldn't warm-start across runs today.
- **`prediction_history.json` / prediction archive "in production"**: ❌ not
  evidenced. `PredictionArchiver` and the `calibration` CLI are fully
  implemented, but no archived prediction file exists anywhere in the repo
  or its `gh-pages`-equivalent output, and there's no sign
  `calibration_report()` has ever been run against real results.
- **Match predictions, squad analysis, paths-to-victory, knockout
  forecasting**: ✅ implemented and unit-tested (module-level), but not
  validated against real outcomes until the holdout run below — treat
  "implemented" and "known to work well" as separate claims going forward.

**First real, out-of-sample validation run** (2026-08-23, teams-only,
228-match holdout — see `validation_reports/2026-08-23_teams-only-holdout.json`
and the real numbers now in [MODEL_EXPLAINED.md](MODEL_EXPLAINED.md#model-validation-how-well-does-it-work)):
model beats a trivial "always predict home win" baseline (computed from
training data only, no test leakage) on win accuracy (70.2% vs 68.0%, not
statistically distinguishable at n=228) and Brier score (0.420 vs 0.451),
but is marginally *worse* than "always predict the mean score" on score
RMSE (13.04 vs 12.92). Per-player try/conversion rate calibration is good
out-of-sample; penalties are over-predicted by ~60%.

**This validation now runs automatically.** `deploy-dashboard.yml` calls
`rugby_ranking.tools.run_holdout_validation` after each dashboard export —
fresh temporal holdout split relative to the run date, fresh VI fit, results
written to `dashboard/data/holdout_validation_report.json` (published with
the rest of the dashboard) and summarized in the workflow's step summary via
`rugby_ranking.tools.print_holdout_summary`. It does **not** gate the build
on the metrics — see "Add a regression gate" below for why not yet.

---

## Near-Term Priorities

### 1. Posterior predictive checks and model validation

**Done, for the first time, 2026-08-23, and now automated weekly.** See the
validation run above. Infrastructure (PPC, PIT/MACE, SBC) was already in
place but had never been run against real data and recorded before this.

**What still needs doing:**

- Add a regression gate once there's a few weeks of automated runs to
  calibrate a threshold against — right now there is exactly one manual data
  point plus whatever accumulates from the new weekly runs, nowhere near
  enough to set a bound without either blocking on VI's normal run-to-run
  Monte Carlo noise or picking an arbitrary number
- Run this same split under MCMC to check whether VI's known miscalibration
  on `alpha_tries`/`sigma_player_kick` (SBC tests) moves these numbers
- ~~Investigate the ~60% penalty over-prediction~~ — **Root-caused,
  2026-08-24.** It's real season-over-season drift, not a model or data
  bug: penalty rate per 80 minutes falls from 0.101 (2024-25) to 0.073
  (2025-26) to 0.043 (2026-27, partial) in the training slice, a
  train/test ratio (1.63x) that accounts for essentially all of the
  measured 1.68x over-prediction. `alpha[penalties]` is a single constant
  fit across the whole training window, so it reflects the blended
  (higher) historical rate. Checked and ruled out: scorer-name
  surname-fallback mis-attribution (only 1.43% of penalty-scoring events
  are ambiguous by surname — nowhere near enough to explain a 68% bias).
  See `MODEL_EXPLAINED.md`'s per-player calibration section for the full
  numbers. **Not yet fixed** — needs a season-level trend/random-walk
  term on score-type intercepts, or recency-weighted/truncated training;
  `time_varying_effects` already in `ModelConfig` doesn't cover this, it
  only models within-season form, not across-season drift. This is a
  structural change touching every score type's intercept, not a one-line
  fix — own task, below.
- Add a season-level trend to score-type intercepts (or recency-weight /
  truncate the training window) so the model tracks a drifting rate
  instead of fitting a single constant across seasons — see the penalty
  finding above; likely affects all four score types to some degree, not
  just penalties
- Stratify calibration by competition/team/season to identify systematic biases
- ~~Build the Elo/simple-baseline comparison~~ / ~~Replicate on other
  splits~~ — **Done, 2026-08-24.** See `rugby_ranking/model/elo.py` and the
  "Replication across three holdout splits" section in
  `MODEL_EXPLAINED.md`. The initial single-split result ("Elo beats the
  model on every metric") **did not replicate** across two more holdout
  windows (Jan-May 2025, Feb-May 2024): win accuracy is a wash (mean 71.3%
  model vs 71.8% Elo, direction flips between splits), RMSE is a wash
  overall (Elo's edge was concentrated in the first split; model is
  marginally ahead in the other two), and only Brier score shows a
  consistent, repeatable Elo advantage across all three splits. Revised
  conclusion: model and Elo are roughly comparable on match
  outcome/scoreline prediction; Elo calibrates win probabilities a bit
  better. Still an uncomfortable result for the added complexity given
  Elo's near-zero cost vs. ~5 min of VI per fit — but a materially
  different, better-supported claim than the original one-split finding.
  What the model still owns uniquely is per-player rates and
  lineup-conditional predictions, which Elo cannot produce at all.
- The automated run only checks the `include_defense=True,
  separate_kicking_effect=True` production config via VI — it fits its own
  model independently of the dashboard's own checkpoint, so it's a second
  ~5-minute VI fit per CI run. Fine for weekly cadence; would need
  minibatching or a shared fit if this ever needs to run more often.

**Why first**: Identifies where complexity is actually needed vs. where the
current model is already good enough. This should inform all structural
changes — and the results so far (model ≈ trivial baseline on score RMSE;
model ≈ Elo on outcome/score prediction across three replicated splits,
with Elo modestly better calibrated) are exactly the kind of signal this
was supposed to surface.

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

### 5. Try-scoring attribution: is per-scorer credit fair?

**Investigated and largely settled, 2026-08-26.** A try is substantially a
team output (build-up, phases won, a break made by someone else) with one
player finishing it — unlike a kick, which is a clean individual act.
Tested whether crediting presence on the pitch, rather than personal
scoring, would capture real signal the current per-scorer model misses.
See `MODEL_EXPLAINED.md`'s "Is per-scorer try credit fair?" section for
full numbers.

**Result: no.** Two independent methods (binary on/off with team-season
demeaning, n=7,869; continuous minutes regression with fixed effects,
n=308,121 panel rows) agree: a player's presence doesn't move their
teammates' try output, once personal scoring is excluded from the
comparison. Position-stratified, the effect tracks who actually scores
tries personally (backs, plus hooker/flanker/no.8), not general
contribution — **props specifically show a tight null** (p=0.45–0.67,
effect bounded to roughly ±0.03-0.06 tries/match), the cleanest possible
test case since they're a pure enabling role that almost never scores.
What does show up for high personal scorers is a redistribution effect
(their presence takes tries away from teammates roughly in proportion to
what they add themselves) — not a pie-growing team effect.

**What this means**: no evidence supports moving try attribution from
per-scorer credit toward a team/lineup-presence model. Doesn't mean props
(or other enablers) don't matter to tries in reality — only that whatever
they contribute isn't recoverable from box-score presence data (this
dataset has no scrum-penalty, metres-carried, or tackle-broken fields that
might actually capture it). Not pursuing a team-attribution rebuild of the
try-scoring likelihood based on this evidence.

**Effort spent**: Small (data analysis only, no model changes)

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
