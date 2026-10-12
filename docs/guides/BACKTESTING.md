# Backtesting and the Forecaster protocol

Every model that predicts matches implements `rugby_ranking.forecast.Forecaster`,
so any two can be compared on the same matches with the same scoring rules.

```bash
python -m rugby_ranking.backtest --data-dir ../Rugby-Data \
    --start 2022-09-01 --step-days 28 \
    --forecasters constant,elo \
    --competitions premiership,top14,celtic
```

`bayes` (plain joint model) and `bayes-prod` (the weekly job's configuration) are also available but refit the hierarchical model at every origin
(minutes each with `--vi-iterations 8000`), so use a coarse `--step-days` for it.

## How it works

At each origin (every `step_days`) the harness

1. gives each forecaster a `History`: only matches **before** the origin;
2. asks it for forecasts of the matches in the **next** `step_days`, passing
   fixtures without scores;
3. records the forecasts alongside the results.

A fit that raises is recorded in `result.failures` and skipped, not fatal.
Comparisons use only matches **every** forecaster predicted (`None` from
`predict` means "can't forecast this", e.g. an unseen team), and report a
paired bootstrap interval on the loss difference.

## Writing a forecaster

```python
from rugby_ranking.forecast import Forecast

class MyModel:
    name = "mine"                      # unique within a run

    def fit(self, history):            # history.matches, .player_rows(), .match_objects()
        ...

    def predict(self, fixtures):       # DataFrame: match_id, date, season, competition, home_team, away_team
        return [Forecast(home_mean=.., away_mean=.., home_win_prob=.., draw_prob=..) for _ in fixtures.itertuples()]
```

`Forecast` needs score means and a home-win probability. `margin_sd` and 90%
intervals are optional and only scored when present.

```python
from rugby_ranking.backtest import BacktestConfig, run_backtest
from rugby_ranking.forecast import MatchStore, ConstantBaseline, EloForecaster

store = MatchStore.from_data_dir("../Rugby-Data")
res = run_backtest(store, [ConstantBaseline(), EloForecaster(), MyModel()],
                   BacktestConfig(start="2022-09-01", step_days=28))
res.summary(by=["competition"])      # accuracy, Brier, log loss, margin/total error, coverage
res.compare("mine", "elo")           # paired differences with 95% intervals
res.calibration("mine")              # predicted vs observed home-win rate
```

## Metrics

- **brier / logloss**: three-way (home / draw / away); lower is better.
- **accuracy**: winner picked right, excluding draws.
- **margin_mae, total_mae**: absolute errors in points.
- **margin_bias, total_bias**: actual minus predicted, so a negative
  `total_bias` means the forecaster over-predicts scoring.
- **home_ci90 / away_ci90**: share of scores inside the 90% interval (want ~0.9).
- **margin_z_sd**: sd of margin error / predicted margin sd (want ~1; below 1
  means the forecaster's uncertainty is too wide).

## References

- `ConstantBaseline`: recent per-competition average match, no team information.
  A model that cannot beat it has learned nothing.
- `EloForecaster`: global Elo with a linear rating-difference-to-margin map.

## Data quality

`MatchStore.from_data_dir` cleans known source problems and records what it
touched in `store.quality` (summary: `store.quality_summary()`):

| issue | action |
|---|---|
| `blank_team`, `same_team` | excluded (international fixtures whose team names failed to parse) |
| `duplicate` | excluded (same teams and kickoff) |
| `scoreless_draw` | kept as **unplayed** (0-0 placeholders for cancelled or Covid-era fixtures) |
| `odd_season_label` | reported only |
| `missing_season` | reported only (e.g. Top 14 2012-13, 2016-17, 2019-21; Championship 2023-25) |

Pass `clean=False` to see the raw data. Season labels written with reversed
years in a filename (`top14-2011-2010.json`) are put in order by the loader.
Reconciliation of scoring events against final scores is a separate check:
`python -m rugby_ranking.sport audit` (see `SPORT_SPEC.md`).

## Calibration layer

A forecaster's scoring level, home advantage or spread can be wrong for one
league without its team strengths being wrong. `rugby_ranking.backtest.calibration`
learns those corrections from the forecaster's own *earlier* out-of-sample
errors and applies them to later forecasts, so an idea can be tried in seconds
on forecasts that are already stored (no refit):

```bash
python -m rugby_ranking.backtest --data-dir ../Rugby-Data --start 2022-10-01 \
    --forecasters elo,bayes-prod --step-days 56 --out forecasts.csv      # slow, once
python -m rugby_ranking.backtest.replay forecasts.csv --calibrate --reference elo
```

```python
res = BacktestResult.from_csv("forecasts.csv").with_calibration(CalibrationConfig(margin=False))
res.compare("bayes-prod+cal", "bayes-prod")
```

Per competition, with shrinkage toward pooled values:

- **level**: shift the predicted total by the mean total error;
- **margin** (off by default): shift the predicted margin by the mean margin error (a home-advantage correction). Shifts learned from a few dozen matches per league overfit and made margin error worse, both on a year and on three years of `bayes-prod` forecasts;
- **spread**: re-derive the win probability from Normal(margin, s) with `s` learned from the margin errors, and rescale the score intervals. The base draw probability is kept.

Forecasts at origin *k* are corrected using only results from origins before *k*;
a test poisons later results and checks nothing changes. Origins with fewer than
`min_history` past forecasts are returned unchanged. `half_life_origins` lets
corrections follow drift. Calibrated forecasters are named `<name>+cal`.

## Comparing many models

With several entrants, a table of pairwise paired differences misleads: some
pair always looks significant by chance, and matches within a period or league
are not independent. `rugby_ranking.competition` fits a hierarchical model to
the per-match losses instead:

```bash
python -m rugby_ranking.backtest.replay forecasts.csv --calibrate --analyse --metric logloss
```

It reports each entrant's effect (mean loss relative to the average entrant, with
a 94% interval), P(best), expected rank and P(row beats column). Match
difficulty is shared across entrants; period (backtest origin) and league
effects are random, so uncertainty reflects the number of independent *periods*,
not just the number of games; and league-specific results are shrunk toward the
overall effect so a lucky league isn't read as skill.

The target is each entrant's **mean** loss: errors are Gaussian, not
heavy-tailed. A proper scoring rule rewards the average, and the occasional
confident miss is the point, not an outlier to down-weight. (A Student-t error
estimates the typical match and ranked a worse model first in an early version;
a test now guards against that.)

On a year of forecasts the plain bootstrap called Elo's lead over the Bayesian
model significant; with only five independent periods the hierarchical analysis
puts P(Elo best) at about 0.8 and cannot separate the two Bayesian variants.

### Result on three years of forecasts (2022-10 to 2026-10)

2,835 matches in 25 time blocks, every entrant calibrated the same way
(level + spread), mean three-way log loss, lower is better:

| entrant | raw mean loss | effect (94% interval) | P(best) |
|---|---|---|---|
| bayes-prod+cal | 0.670 | -0.016 (-0.026, -0.005) | 0.77 |
| elo+cal | 0.683 | -0.010 (-0.020, 0.001) | 0.22 |
| elo | 0.696 | 0.003 | 0.005 |
| bayes-prod | 0.695 | 0.005 | 0.002 |
| constant | 0.692 | 0.007 | 0.001 |

Uncalibrated, none of the models beat the constant baseline on win probability:
their probabilities are overconfident. The Bayesian model's *margins* are the
best of the three (error 12.7 against 12.8 for Elo and 13.2 for the constant),
so its team-strength signal is real; the spread layer fixes the scale that turns
margins into probabilities. Calibration also helps Elo, but not the constant
baseline, which has nothing to correct. Elo is better than the Bayesian model
in the Championship.
