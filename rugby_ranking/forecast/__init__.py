"""Common forecasting interface so different models can be compared fairly.

``Forecaster`` is the protocol every model implements (fit on history before a
date, predict fixtures after it); ``MatchStore`` is the single tidy match table
the backtest harness (``rugby_ranking.backtest``) draws from.
"""

from rugby_ranking.forecast.base import Forecast, Forecaster, History
from rugby_ranking.forecast.baselines import ConstantBaseline, EloForecaster
from rugby_ranking.forecast.store import MatchStore

__all__ = [
    "Forecast",
    "Forecaster",
    "History",
    "MatchStore",
    "ConstantBaseline",
    "EloForecaster",
]
