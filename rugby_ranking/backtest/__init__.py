"""Rolling-origin backtesting for any ``Forecaster``."""

from rugby_ranking.backtest.harness import BacktestConfig, BacktestResult, run_backtest
from rugby_ranking.backtest.metrics import calibration_table, paired_diff, summarise

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "summarise",
    "paired_diff",
    "calibration_table",
]
