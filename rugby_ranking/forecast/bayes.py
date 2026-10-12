"""Adapter putting the hierarchical Bayesian model behind the Forecaster protocol."""

from __future__ import annotations

import re

import pandas as pd

from rugby_ranking.forecast.base import Forecast, History


class BayesianForecaster:
    """Refits the joint model (variational inference by default) on the most
    recent seasons before each origin and predicts with team-only forecasts,
    which is what the production pipeline archives before lineups are known.

    Fitting dominates a backtest's cost (minutes per origin), so use a coarse
    ``step_days`` for this one.
    """

    def __init__(
        self,
        config=None,
        n_recent_seasons: int = 3,
        vi_iterations: int = 20000,
        n_samples: int = 500,
        seed: int = 42,
        name: str = "bayes",
    ):
        self.name = name
        self.config = config
        self.n_recent_seasons = n_recent_seasons
        self.vi_iterations = vi_iterations
        self.n_samples = n_samples
        self.seed = seed
        self._predictor = None

    def fit(self, history: History) -> None:
        from rugby_ranking.model.core import ModelConfig, RugbyModel
        from rugby_ranking.model.inference import InferenceConfig, ModelFitter
        from rugby_ranking.model.predictions import MatchPredictor, estimate_kicking_rates

        df = history.player_rows()
        years = df["season"].str.extract(r"(\d{4})")[0].astype(int)
        keep = sorted(years.unique())[-self.n_recent_seasons:]
        df = df[years.isin(keep)].copy()

        model = RugbyModel(config=self.config or ModelConfig(include_defense=True, separate_kicking_effect=True))
        model.build_joint(df)
        fitter = ModelFitter(
            model, config=InferenceConfig(vi_n_iterations=self.vi_iterations, vi_method="advi")
        )
        trace = fitter.fit_vi(progressbar=False, random_seed=self.seed)
        rates = estimate_kicking_rates(history.match_objects())
        self._predictor = MatchPredictor(model, trace, kicking_rates=rates)

    def predict(self, fixtures: pd.DataFrame) -> list[Forecast | None]:
        out: list[Forecast | None] = []
        for r in fixtures.itertuples(index=False):
            try:
                pred = self._predictor.predict_teams_only(
                    r.home_team, r.away_team, season=r.season,
                    n_samples=self.n_samples, competition=r.competition,
                )
                out.append(Forecast.from_match_prediction(pred))
            except ValueError:  # team never seen in the training window
                out.append(None)
        return out
