# Backtesting and the Forecaster protocol

Every model that predicts matches implements `rugby_ranking.forecast.Forecaster`,
so any two can be compared on the same matches with the same scoring rules.

```bash
python -m rugby_ranking.backtest --data-dir ../Rugby-Data \
    --start 2022-09-01 --step-days 28 \
    --forecasters constant,elo \
    --competitions premiership,top14,celtic
```

`bayes` is also available but refits the hierarchical model at every origin
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
