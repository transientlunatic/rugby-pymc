"""
Standard Elo rating system for rugby teams -- a serious, well-established
simple baseline to compare the Bayesian hierarchical model against.

Context: the holdout validation (see run_holdout_validation.py,
MODEL_EXPLAINED.md) showed the full model barely beats -- and on score RMSE,
loses to -- the cheapest possible skill-free baseline ("always predict the
training-set mean/home win"). That's not a fair fight: a real ranking system
should be compared against a real, if simple, ranking system. Elo is the
standard choice: two free parameters (K-factor, home advantage), one rating
per team, updated after each match. If the Bayesian model can't beat Elo,
that's a much more meaningful signal about whether its added complexity
(player effects, defense terms, kicking split, joint likelihood) earns its
cost than "it beat a coin flip" ever was.

Design choices, and why:
- Global rating pool (not per-competition): teams cross competitions
  (Champions Cup, internationals) often enough that splitting would lose
  information, and the Bayesian model doesn't condition on competition
  either -- keeps the comparison apples-to-apples.
- Season regression to the mean: ratings blend 1/3 of the way back to the
  pool average at each season boundary, a standard Elo enhancement (used by
  e.g. FiveThirtyEight's sports Elo systems) that prevents ratings from
  drifting on stale information across close-season squad turnover. Fixed
  at 1/3 rather than tuned, to keep the calibration grid small -- this is a
  baseline, not the object of study.
- K-factor and home advantage ARE calibrated, via a small grid search on an
  inner chronological split of the training data only (never the held-out
  test set -- that would repeat the exact leakage bug fixed in PR #1).
- Score prediction (for RMSE/MAE comparability with the Bayesian model's
  score predictions): a simple two-parameter linear regression of match
  margin (home_score - away_score) on Elo rating difference, fit on
  training data; combined with the training-set mean total score to split
  into predicted home/away scores. This is the standard technique sports
  analytics sites use to turn a rating system into a point-spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _expected_score(rating_diff: float) -> float:
    """Standard Elo expected-score (win probability) function."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


@dataclass
class EloRatingSystem:
    k: float = 20.0
    home_advantage: float = 60.0
    initial_rating: float = 1500.0
    season_regression: float = 1.0 / 3.0

    ratings: dict = field(default_factory=dict)
    _last_season: str | None = field(default=None, repr=False)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _regress_to_mean_if_new_season(self, season: str) -> None:
        if self._last_season is not None and season != self._last_season and self.ratings:
            pool_mean = float(np.mean(list(self.ratings.values())))
            for team in self.ratings:
                self.ratings[team] += self.season_regression * (pool_mean - self.ratings[team])
        self._last_season = season

    def win_probabilities(self, home_team: str, away_team: str) -> float:
        """P(home win), from current ratings + home advantage. Ignores draws
        (see `predict_match_probs` for the draw-aware version used for
        Brier-score comparison)."""
        diff = (self.get_rating(home_team) + self.home_advantage) - self.get_rating(away_team)
        return _expected_score(diff)

    def update(self, home_team: str, away_team: str, home_score: float, away_score: float, season: str) -> None:
        self._regress_to_mean_if_new_season(season)

        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)
        expected_home = _expected_score((r_home + self.home_advantage) - r_away)

        if home_score > away_score:
            actual_home = 1.0
        elif away_score > home_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        self.ratings[home_team] = r_home + self.k * (actual_home - expected_home)
        self.ratings[away_team] = r_away + self.k * ((1 - actual_home) - (1 - expected_home))

    def fit(self, matches: pd.DataFrame) -> "EloRatingSystem":
        """Process matches chronologically. `matches` needs columns
        home_team, away_team, home_score, away_score, season, date -- one
        row per match, already sorted or not (this sorts by date)."""
        for _, row in matches.sort_values("date").iterrows():
            self.update(row["home_team"], row["away_team"], row["home_score"], row["away_score"], row["season"])
        return self


@dataclass
class MarginModel:
    """
    Linear map from Elo rating difference (including home advantage) to
    predicted score margin, plus the training-set mean total score -- turns
    Elo ratings into predicted home/away scores, the way sports-analytics
    sites turn a rating system into a point-spread.
    """
    intercept: float = 0.0
    slope: float = 0.0
    mean_total_score: float = 0.0

    def fit(self, elo_diffs: np.ndarray, margins: np.ndarray, totals: np.ndarray) -> "MarginModel":
        # Closed-form simple OLS: margin = intercept + slope * elo_diff
        x = np.asarray(elo_diffs, dtype=float)
        y = np.asarray(margins, dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        denom = np.sum((x - x_mean) ** 2)
        self.slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom > 0 else 0.0
        self.intercept = float(y_mean - self.slope * x_mean)
        self.mean_total_score = float(np.mean(totals))
        return self

    def predict_scores(self, elo_diff: float) -> tuple[float, float]:
        margin = self.intercept + self.slope * elo_diff
        home = (self.mean_total_score + margin) / 2
        away = (self.mean_total_score - margin) / 2
        return home, away


def calibrate_elo(
    train_matches: pd.DataFrame,
    k_grid: list[float] = (10.0, 20.0, 32.0),
    home_advantage_grid: list[float] = (0.0, 40.0, 80.0, 120.0),
    inner_val_fraction: float = 0.15,
) -> EloRatingSystem:
    """
    Pick (k, home_advantage) by grid search on an INNER chronological split
    of train_matches (never the real held-out test set -- see module
    docstring). Scores each candidate by Brier score on the inner
    validation slice, refits the winning config on the FULL train_matches,
    and returns that.
    """
    matches = train_matches.sort_values("date").reset_index(drop=True)
    n_val = max(1, int(round(len(matches) * inner_val_fraction)))
    inner_train, inner_val = matches.iloc[:-n_val], matches.iloc[-n_val:]

    best_brier = None
    best_params = (k_grid[0], home_advantage_grid[0])
    for k in k_grid:
        for home_adv in home_advantage_grid:
            elo = EloRatingSystem(k=k, home_advantage=home_adv).fit(inner_train)
            briers = []
            for _, row in inner_val.iterrows():
                p_home = elo.win_probabilities(row["home_team"], row["away_team"])
                actual = 1.0 if row["home_score"] > row["away_score"] else 0.0
                briers.append((p_home - actual) ** 2)
                elo.update(row["home_team"], row["away_team"], row["home_score"], row["away_score"], row["season"])
            brier = float(np.mean(briers)) if briers else float("inf")
            if best_brier is None or brier < best_brier:
                best_brier = brier
                best_params = (k, home_adv)

    k, home_adv = best_params
    return EloRatingSystem(k=k, home_advantage=home_adv).fit(matches)
