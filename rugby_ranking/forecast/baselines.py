"""Reference forecasters. Any new model should be benchmarked against these."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rugby_ranking.forecast.base import Forecast, History
from rugby_ranking.model.elo import EloRatingSystem, calibrate_elo

POOLED = "__all__"
_MIN_COMP_MATCHES = 30


def _recent(matches: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    return matches[matches["date"] >= as_of - pd.Timedelta(days=days)]


class ConstantBaseline:
    """Predicts the recent per-competition average match every time: the
    skill-free reference ("home advantage and league scoring level, nothing
    about the teams"). Anything that can't beat this has learned nothing."""

    def __init__(self, window_days: int = 3 * 365, name: str = "constant"):
        self.name = name
        self.window_days = window_days
        self._by_comp: dict[str, Forecast] = {}

    @staticmethod
    def _summarise(m: pd.DataFrame) -> Forecast:
        hs, as_ = m["home_score"].to_numpy(float), m["away_score"].to_numpy(float)
        draw = float(np.mean(hs == as_))
        home_win = float(np.mean(hs > as_))
        return Forecast(
            home_mean=float(hs.mean()),
            away_mean=float(as_.mean()),
            home_win_prob=home_win,
            draw_prob=max(draw, 0.005),
            margin_sd=float((hs - as_).std(ddof=1)) if len(hs) > 1 else None,
            home_interval=tuple(np.percentile(hs, [5, 95])),
            away_interval=tuple(np.percentile(as_, [5, 95])),
        )

    def fit(self, history: History) -> None:
        m = _recent(history.matches, history.as_of, self.window_days)
        if m.empty:
            m = history.matches
        self._by_comp = {POOLED: self._summarise(m)}
        for comp, g in m.groupby("competition"):
            if len(g) >= _MIN_COMP_MATCHES:
                self._by_comp[comp] = self._summarise(g)

    def predict(self, fixtures: pd.DataFrame) -> list[Forecast | None]:
        return [self._by_comp.get(c, self._by_comp[POOLED]) for c in fixtures["competition"]]


class EloForecaster:
    """Elo ratings (global pool, season regression) with a linear
    rating-difference -> margin map and a per-competition recent scoring level.

    Incremental: ratings are carried across origins and only matches newer than
    the last fit are processed, so a rolling backtest costs one pass over the
    data. Going back in time resets the state. ``k`` / ``home_advantage`` are
    picked once, on the first fit, by the existing ``calibrate_elo`` grid
    search over that history only (never later data).
    """

    def __init__(
        self,
        k: float | None = None,
        home_advantage: float | None = None,
        calibrate_on: int = 3000,
        margin_window_days: int = 3 * 365,
        totals_window_days: int = 2 * 365,
        name: str = "elo",
    ):
        self.name = name
        self._fixed = (k, home_advantage)
        self.calibrate_on = calibrate_on
        self.margin_window_days = margin_window_days
        self.totals_window_days = totals_window_days
        self._reset()

    def _reset(self) -> None:
        k, ha = self._fixed
        self.elo = EloRatingSystem(k=k if k is not None else 20.0,
                                   home_advantage=ha if ha is not None else 60.0)
        self._calibrated = k is not None and ha is not None
        self._last_date: pd.Timestamp | None = None
        self._as_of: pd.Timestamp | None = None
        # one record per processed match: date, comp, elo diff, margin, total
        self._rec: list[tuple] = []
        self._slope, self._intercept, self._margin_sd = 0.1, 0.0, None
        self._totals: dict[str, float] = {}
        self._draw = 0.02

    def fit(self, history: History) -> None:
        if self._as_of is not None and history.as_of < self._as_of:
            self._reset()
        m = history.matches
        if not self._calibrated and len(m) >= 500:
            tuned = calibrate_elo(m.tail(self.calibrate_on))
            # Parameters changed, so ratings built under the old ones are stale:
            # start over and reprocess all of ``history``.
            self.elo = EloRatingSystem(k=tuned.k, home_advantage=tuned.home_advantage)
            self._calibrated = True
            self._rec, self._last_date = [], None
        new = m if self._last_date is None else m[m["date"] > self._last_date]
        for r in new.itertuples(index=False):
            diff = self.elo.get_rating(r.home_team) + self.elo.home_advantage - self.elo.get_rating(r.away_team)
            self._rec.append((r.date, r.competition, diff, r.home_score - r.away_score, r.home_score + r.away_score))
            self.elo.update(r.home_team, r.away_team, r.home_score, r.away_score, r.season)
        if len(new):
            self._last_date = new["date"].max()
        self._as_of = history.as_of
        self._refit_maps(history.as_of)

    def _refit_maps(self, as_of: pd.Timestamp) -> None:
        if not self._rec:
            return
        rec = pd.DataFrame(self._rec, columns=["date", "comp", "diff", "margin", "total"])
        w = rec[rec["date"] >= as_of - pd.Timedelta(days=self.margin_window_days)]
        if len(w) >= 50 and w["diff"].std() > 0:
            self._slope, self._intercept = np.polyfit(w["diff"], w["margin"], 1)
            resid = w["margin"] - (self._intercept + self._slope * w["diff"])
            self._margin_sd = float(resid.std(ddof=2))
        t = rec[rec["date"] >= as_of - pd.Timedelta(days=self.totals_window_days)]
        if t.empty:
            t = rec
        self._totals = {POOLED: float(t["total"].mean())}
        for comp, g in t.groupby("comp"):
            if len(g) >= _MIN_COMP_MATCHES:
                self._totals[comp] = float(g["total"].mean())
        self._draw = max(float((t["margin"] == 0).mean()), 0.005)

    def predict(self, fixtures: pd.DataFrame) -> list[Forecast | None]:
        out = []
        for r in fixtures.itertuples(index=False):
            diff = self.elo.get_rating(r.home_team) + self.elo.home_advantage - self.elo.get_rating(r.away_team)
            margin = self._intercept + self._slope * diff
            total = self._totals.get(r.competition, self._totals.get(POOLED, 50.0))
            p = self.elo.win_probabilities(r.home_team, r.away_team)  # expected score, draws = 1/2
            out.append(Forecast(
                home_mean=(total + margin) / 2,
                away_mean=(total - margin) / 2,
                home_win_prob=float(np.clip(p - self._draw / 2, 0.0, 1.0)),
                draw_prob=self._draw,
                margin_sd=self._margin_sd,
            ))
        return out
