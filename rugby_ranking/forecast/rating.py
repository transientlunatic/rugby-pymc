"""A Bayesian rating model: Gaussian team ratings updated by an exact Kalman
filter, with per-league home advantage and scoring level learned online.

Each team has a rating ``r ~ N(m, v)`` on the scale of expected margin against
an average team (points). A match margin is

    margin = r_home - r_away + h_c + e,     e ~ N(0, s_c^2)

for league ``c``. This is linear-Gaussian, so the update after each match is
exact (a Kalman step on the two ratings). Between matches the variance grows
with elapsed days (form drifts); when a team starts a new season its rating
regresses toward average and gains a turnover variance (squad changes).

Predictions are fully probabilistic: margin ~ N(mu, v_home + v_away + s_c^2),
so a promoted or long-absent team gets wider, honest predictions. Win, draw and
away probabilities come from the integer margin (draw = |margin| < 0.5).

It is fast and incremental like Elo, but carries uncertainty, learns each
league's home advantage and noise, and lets cup fixtures link the leagues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

from rugby_ranking.forecast.base import Forecast, History

POOLED = "__all__"
_MIN_COMP_FOR_TOTALS = 30


@dataclass(frozen=True)
class RatingConfig:
    """Defaults were tuned by coordinate descent on mean log loss over the
    2019-09 to 2022-09 backtest window only (the landscape is flat: tuning
    moved log loss from 0.6601 to 0.6559). Later periods are out of sample."""

    drift_per_day: float = 0.005     # rating variance added per day (points^2)
    season_sd: float = 3.0           # sd of the squad-turnover jump at a new season (points)
    season_shrink: float = 0.9       # share of the rating kept across a season boundary
    prior_sd: float = 14.0           # sd of a team seen for the first time (points)
    home_advantage0: float = 4.0     # starting home advantage for a new league (points)
    margin_sd0: float = 14.0         # starting margin noise for a new league (points)
    league_lr: float = 0.01          # learning rate of per-league home advantage and noise
    min_margin_sd: float = 6.0       # floor on the league noise (points)
    totals_lr: float = 0.01          # learning rate of the per-league scoring level
    max_gap_days: float = 400.0      # cap on drift accumulated over a long absence


class RatingForecaster:
    """``Forecaster`` using Gaussian ratings and an exact Kalman update."""

    def __init__(self, config: RatingConfig | None = None, name: str = "rating"):
        self.name = name
        self.cfg = config or RatingConfig()
        self._reset()

    # -- state ---------------------------------------------------------------

    def _reset(self) -> None:
        self.mean: dict[str, float] = {}
        self.var: dict[str, float] = {}
        self.last_date: dict[str, pd.Timestamp] = {}
        self.last_season: dict[str, str] = {}
        self.home_adv: dict[str, float] = {}
        self.noise2: dict[str, float] = {}
        self.total: dict[str, float] = {}
        self.n_comp: dict[str, int] = {}
        self._last_processed: pd.Timestamp | None = None
        self._as_of: pd.Timestamp | None = None

    def _state_at(self, team: str, date: pd.Timestamp, season: str) -> tuple[float, float]:
        """(mean, variance) of ``team``'s rating at ``date``, without changing state:
        drift since the last match, plus the season turnover if ``season`` is new."""
        c = self.cfg
        if team not in self.mean:
            return 0.0, c.prior_sd ** 2
        m, v = self.mean[team], self.var[team]
        gap = min(max((date - self.last_date[team]).total_seconds() / 86400.0, 0.0), c.max_gap_days)
        v += c.drift_per_day * gap
        if self.last_season[team] != season:
            m *= c.season_shrink
            v = v * c.season_shrink ** 2 + c.season_sd ** 2
        return m, v

    def _league(self, comp: str) -> tuple[float, float]:
        c = self.cfg
        return self.home_adv.get(comp, c.home_advantage0), self.noise2.get(comp, c.margin_sd0 ** 2)

    def _update(self, row) -> None:
        c = self.cfg
        mh, vh = self._state_at(row.home_team, row.date, row.season)
        ma, va = self._state_at(row.away_team, row.date, row.season)
        h, s2 = self._league(row.competition)
        margin = float(row.home_score - row.away_score)

        innovation = margin - (mh - ma + h)
        S = vh + va + s2                                   # innovation variance
        self.mean[row.home_team], self.var[row.home_team] = mh + vh / S * innovation, vh - vh * vh / S
        self.mean[row.away_team], self.var[row.away_team] = ma - va / S * innovation, va - va * va / S
        for team in (row.home_team, row.away_team):
            self.last_date[team], self.last_season[team] = row.date, row.season

        # Per-league home advantage and noise, by stochastic approximation. The
        # innovation variance beyond the rating uncertainty is the league noise.
        self.home_adv[row.competition] = h + c.league_lr * innovation
        self.noise2[row.competition] = max(
            (1 - c.league_lr) * s2 + c.league_lr * max(innovation ** 2 - (vh + va), 0.0), c.min_margin_sd ** 2
        )
        total = float(row.home_score + row.away_score)
        prev = self.total.get(row.competition, total)
        self.total[row.competition] = prev + c.totals_lr * (total - prev) if row.competition in self.total else total
        self.n_comp[row.competition] = self.n_comp.get(row.competition, 0) + 1

    # -- Forecaster protocol ----------------------------------------------------

    def fit(self, history: History) -> None:
        if self._as_of is not None and history.as_of < self._as_of:
            self._reset()
        m = history.matches
        new = m if self._last_processed is None else m[m["date"] > self._last_processed]
        for row in new.sort_values("date", kind="stable").itertuples(index=False):
            self._update(row)
        if len(new):
            self._last_processed = new["date"].max()
        self._as_of = history.as_of

    def predict(self, fixtures: pd.DataFrame) -> list[Forecast | None]:
        pooled_total = (np.mean([v for k, v in self.total.items() if self.n_comp[k] >= _MIN_COMP_FOR_TOTALS])
                        if any(self.n_comp[k] >= _MIN_COMP_FOR_TOTALS for k in self.total) else 50.0)
        out = []
        for r in fixtures.itertuples(index=False):
            mh, vh = self._state_at(r.home_team, r.date, r.season)
            ma, va = self._state_at(r.away_team, r.date, r.season)
            h, s2 = self._league(r.competition)
            mu, sd = mh - ma + h, float(np.sqrt(vh + va + s2))
            p_home = 1.0 - ndtr((0.5 - mu) / sd)           # integer margin >= 1
            p_away = ndtr((-0.5 - mu) / sd)                # integer margin <= -1
            total = self.total.get(r.competition) if self.n_comp.get(r.competition, 0) >= _MIN_COMP_FOR_TOTALS else None
            total = pooled_total if total is None else total
            out.append(Forecast(
                home_mean=(total + mu) / 2, away_mean=(total - mu) / 2,
                home_win_prob=float(p_home), draw_prob=float(max(1.0 - p_home - p_away, 0.0)),
                margin_sd=sd,
            ))
        return out

    # -- inspection ------------------------------------------------------------

    def ratings(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Current rating mean and sd per team (at its last match, or ``as_of``)."""
        rows = []
        for t in self.mean:
            m, v = (self.mean[t], self.var[t]) if as_of is None else self._state_at(t, as_of, self.last_season[t])
            rows.append(dict(team=t, rating=m, sd=float(np.sqrt(v)), last_match=self.last_date[t]))
        return pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)
