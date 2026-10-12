"""The Forecaster protocol and the value types it exchanges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from rugby_ranking.forecast.store import MatchStore
    from rugby_ranking.model.data import MatchData
    from rugby_ranking.model.predictions import MatchPrediction

FIXTURE_COLUMNS = ("match_id", "date", "season", "competition", "home_team", "away_team")


@dataclass(frozen=True)
class Forecast:
    """A pre-match forecast, deliberately the lowest common denominator.

    Every model can supply score means and a home-win probability; the rest is
    optional and only scored when present.
    """

    home_mean: float
    away_mean: float
    home_win_prob: float
    draw_prob: float = 0.0
    margin_sd: float | None = None
    home_interval: tuple[float, float] | None = None  # central 90% interval
    away_interval: tuple[float, float] | None = None

    @property
    def away_win_prob(self) -> float:
        return max(0.0, 1.0 - self.home_win_prob - self.draw_prob)

    @property
    def margin(self) -> float:
        return self.home_mean - self.away_mean

    @property
    def total(self) -> float:
        return self.home_mean + self.away_mean

    @classmethod
    def from_match_prediction(cls, pred: "MatchPrediction") -> "Forecast":
        return cls(
            home_mean=float(pred.home.mean),
            away_mean=float(pred.away.mean),
            home_win_prob=float(pred.home_win_prob),
            draw_prob=float(pred.draw_prob),
            margin_sd=float(pred.margin_std),
            home_interval=(float(pred.home.ci_lower), float(pred.home.ci_upper)),
            away_interval=(float(pred.away.ci_lower), float(pred.away.ci_upper)),
        )


@dataclass
class History:
    """Everything a forecaster may learn from at an origin: matches strictly
    before ``as_of``. The harness builds it, so a forecaster cannot peek."""

    as_of: pd.Timestamp
    matches: pd.DataFrame  # played matches with date < as_of (see MatchStore)
    store: "MatchStore"

    def player_rows(self) -> pd.DataFrame:
        """Player-match rows (``MatchDataset.to_dataframe`` layout) before ``as_of``."""
        return self.store.player_rows(before=self.as_of)

    def match_objects(self) -> "list[MatchData]":
        """Parsed ``MatchData`` (lineups, scoring events) before ``as_of``."""
        return self.store.match_objects(before=self.as_of)


@runtime_checkable
class Forecaster(Protocol):
    """A model that can be refit at any origin and asked about fixtures.

    ``fit`` is called once per backtest origin with only the past, and must
    leave the forecaster ready to ``predict`` (it may reuse earlier state as
    long as it ends up equivalent to fitting on ``history`` alone).
    ``predict`` takes a frame with ``FIXTURE_COLUMNS`` and returns one entry
    per row, in order; ``None`` means "cannot forecast this fixture" (e.g. an
    unseen team), and such fixtures are excluded from paired comparisons.
    """

    name: str

    def fit(self, history: History) -> None: ...

    def predict(self, fixtures: pd.DataFrame) -> "list[Forecast | None]": ...
