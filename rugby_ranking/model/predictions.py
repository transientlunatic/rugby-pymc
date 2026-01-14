"""
Match prediction functionality for rugby ranking models.

Supports:
- Team-only predictions (when lineups unknown)
- Full lineup predictions (when team sheets available)
- Proper uncertainty quantification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import arviz as az

from rugby_ranking.model.core import RugbyModel


@dataclass
class ScorePrediction:
    """Predicted score distribution for a single team."""

    team: str
    mean: float
    std: float
    median: float
    ci_lower: float  # 5th percentile
    ci_upper: float  # 95th percentile
    samples: np.ndarray | None = None  # Full posterior samples if requested


@dataclass
class MatchPrediction:
    """Full match prediction with both teams."""

    home: ScorePrediction
    away: ScorePrediction
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    predicted_margin: float  # home - away
    margin_std: float

    def summary(self) -> str:
        """Human-readable prediction summary."""
        return (
            f"{self.home.team} vs {self.away.team}\n"
            f"  Predicted: {self.home.mean:.1f} - {self.away.mean:.1f}\n"
            f"  Home win: {self.home_win_prob:.1%}, "
            f"Away win: {self.away_win_prob:.1%}, "
            f"Draw: {self.draw_prob:.1%}\n"
            f"  90% CI margin: [{self.predicted_margin - 1.645*self.margin_std:.0f}, "
            f"{self.predicted_margin + 1.645*self.margin_std:.0f}]"
        )


class MatchPredictor:
    """
    Generate match predictions from fitted models.

    Handles two prediction modes:
    1. Teams-only: Uses team-season effects + average player/position effects
    2. Full lineup: Uses specific player effects for announced squads
    """

    def __init__(self, model: RugbyModel, trace: az.InferenceData | None = None):
        self.model = model
        self.trace = trace or model.trace

        if self.trace is None:
            raise ValueError("No trace available. Fit the model first.")

    def predict_teams_only(
        self,
        home_team: str,
        away_team: str,
        season: str,
        n_samples: int = 1000,
    ) -> MatchPrediction:
        """
        Predict match outcome using only team identities.

        Uses team-season effects and marginalizes over typical player/position
        contributions. Higher uncertainty than full-lineup predictions.

        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season identifier (e.g., "2025-2026")
            n_samples: Number of posterior samples to use

        Returns:
            MatchPrediction with score distributions
        """
        # Get team-season indices
        home_ts = (home_team, season)
        away_ts = (away_team, season)

        if home_ts not in self.model._team_season_ids:
            raise ValueError(f"Unknown team-season: {home_team} in {season}")
        if away_ts not in self.model._team_season_ids:
            raise ValueError(f"Unknown team-season: {away_team} in {season}")

        home_idx = self.model._team_season_ids[home_ts]
        away_idx = self.model._team_season_ids[away_ts]

        # Extract posterior samples
        posterior = self.trace.posterior

        # Flatten chains
        alpha = posterior["alpha"].values.flatten()[:n_samples]
        gamma = posterior["gamma_team_season"].values.reshape(-1, posterior["gamma_team_season"].shape[-1])[:n_samples]
        eta_home = posterior["eta_home"].values.flatten()[:n_samples]

        # Team effects
        home_team_effect = gamma[:, home_idx]
        away_team_effect = gamma[:, away_idx]

        # For teams-only, we use average player contribution
        # This adds extra uncertainty
        avg_player_effect = 0  # Centered at 0 by construction
        player_uncertainty = posterior["sigma_player"].values.flatten()[:n_samples]

        # Simulate scores
        # Using typical 15 scoring opportunities per match as baseline
        # (this is calibrated from observed match totals)
        baseline_scoring_rate = 15

        home_log_rate = (
            alpha +
            home_team_effect +
            avg_player_effect +
            eta_home +  # Home advantage
            np.random.normal(0, player_uncertainty)  # Player uncertainty
        )

        away_log_rate = (
            alpha +
            away_team_effect +
            avg_player_effect +
            np.random.normal(0, player_uncertainty)
        )

        # Convert to expected points (rough scaling)
        # Average points per match ~ 25, so scale accordingly
        points_scale = 25.0 / np.exp(alpha.mean())

        home_expected = np.exp(home_log_rate) * points_scale
        away_expected = np.exp(away_log_rate) * points_scale

        # Sample actual scores (Poisson would give too low variance for rugby)
        # Use negative binomial for overdispersion
        home_scores = np.random.negative_binomial(
            n=5,  # Dispersion parameter
            p=5 / (5 + home_expected),
        )
        away_scores = np.random.negative_binomial(
            n=5,
            p=5 / (5 + away_expected),
        )

        return self._build_prediction(
            home_team, away_team,
            home_scores, away_scores,
        )

    def predict_full_lineup(
        self,
        home_team: str,
        away_team: str,
        home_lineup: dict[int, str],  # position -> player name
        away_lineup: dict[int, str],
        season: str,
        n_samples: int = 1000,
    ) -> MatchPrediction:
        """
        Predict match outcome using full announced lineups.

        Uses specific player effects for each position. Lower uncertainty
        than teams-only predictions.

        Args:
            home_team: Home team name
            away_team: Away team name
            home_lineup: Dict mapping position (1-23) to player name
            away_lineup: Dict mapping position (1-23) to player name
            season: Season identifier
            n_samples: Number of posterior samples to use

        Returns:
            MatchPrediction with score distributions
        """
        # Get team-season indices
        home_ts = (home_team, season)
        away_ts = (away_team, season)

        home_ts_idx = self.model._team_season_ids.get(home_ts)
        away_ts_idx = self.model._team_season_ids.get(away_ts)

        if home_ts_idx is None:
            raise ValueError(f"Unknown team-season: {home_team} in {season}")
        if away_ts_idx is None:
            raise ValueError(f"Unknown team-season: {away_team} in {season}")

        # Extract posterior samples
        posterior = self.trace.posterior

        alpha = posterior["alpha"].values.flatten()[:n_samples]
        gamma = posterior["gamma_team_season"].values.reshape(
            -1, posterior["gamma_team_season"].shape[-1]
        )[:n_samples]
        beta = posterior["beta_player"].values.reshape(
            -1, posterior["beta_player"].shape[-1]
        )[:n_samples]
        theta = posterior["theta_position"].values.reshape(
            -1, posterior["theta_position"].shape[-1]
        )[:n_samples]
        eta_home = posterior["eta_home"].values.flatten()[:n_samples]

        # Compute team scoring rates from lineup
        home_rate = self._compute_lineup_rate(
            home_lineup, home_ts_idx,
            alpha, beta, gamma, theta, eta_home,
            is_home=True, n_samples=n_samples
        )

        away_rate = self._compute_lineup_rate(
            away_lineup, away_ts_idx,
            alpha, beta, gamma, theta, eta_home,
            is_home=False, n_samples=n_samples
        )

        # Scale to expected points
        points_scale = 25.0 / np.exp(alpha.mean())

        home_expected = home_rate * points_scale
        away_expected = away_rate * points_scale

        # Sample scores
        home_scores = np.random.negative_binomial(
            n=6,  # Slightly less dispersed with known lineup
            p=6 / (6 + home_expected),
        )
        away_scores = np.random.negative_binomial(
            n=6,
            p=6 / (6 + away_expected),
        )

        return self._build_prediction(
            home_team, away_team,
            home_scores, away_scores,
        )

    def _compute_lineup_rate(
        self,
        lineup: dict[int, str],
        team_season_idx: int,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        theta: np.ndarray,
        eta_home: np.ndarray,
        is_home: bool,
        n_samples: int,
    ) -> np.ndarray:
        """Compute expected scoring rate for a lineup."""
        # Start with baseline
        log_rate = alpha.copy()

        # Add team effect
        log_rate += gamma[:, team_season_idx]

        # Add home advantage if applicable
        if is_home:
            log_rate += eta_home

        # Sum player and position contributions
        player_contribution = np.zeros(n_samples)
        for position, player_name in lineup.items():
            if player_name in self.model._player_ids:
                player_idx = self.model._player_ids[player_name]
                player_contribution += beta[:, player_idx]

            # Position effect (0-indexed)
            if 1 <= position <= 23:
                player_contribution += theta[:, position - 1]

        # Average over 15 players (starting XV)
        log_rate += player_contribution / 15

        return np.exp(log_rate)

    def _build_prediction(
        self,
        home_team: str,
        away_team: str,
        home_scores: np.ndarray,
        away_scores: np.ndarray,
    ) -> MatchPrediction:
        """Build MatchPrediction from score samples."""
        home_pred = ScorePrediction(
            team=home_team,
            mean=home_scores.mean(),
            std=home_scores.std(),
            median=np.median(home_scores),
            ci_lower=np.percentile(home_scores, 5),
            ci_upper=np.percentile(home_scores, 95),
            samples=home_scores,
        )

        away_pred = ScorePrediction(
            team=away_team,
            mean=away_scores.mean(),
            std=away_scores.std(),
            median=np.median(away_scores),
            ci_lower=np.percentile(away_scores, 5),
            ci_upper=np.percentile(away_scores, 95),
            samples=away_scores,
        )

        # Win probabilities
        home_wins = (home_scores > away_scores).mean()
        away_wins = (away_scores > home_scores).mean()
        draws = (home_scores == away_scores).mean()

        margin = home_scores - away_scores

        return MatchPrediction(
            home=home_pred,
            away=away_pred,
            home_win_prob=home_wins,
            away_win_prob=away_wins,
            draw_prob=draws,
            predicted_margin=margin.mean(),
            margin_std=margin.std(),
        )

    def predict_upcoming(
        self,
        unplayed_matches: list,  # List of MatchData
        season: str,
    ) -> pd.DataFrame:
        """
        Generate predictions for all upcoming matches.

        Args:
            unplayed_matches: List of MatchData with no scores
            season: Current season

        Returns:
            DataFrame with predictions for each match
        """
        predictions = []

        for match in unplayed_matches:
            try:
                pred = self.predict_teams_only(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    season=season,
                )

                predictions.append({
                    "match_id": match.match_id,
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_score_pred": pred.home.mean,
                    "away_score_pred": pred.away.mean,
                    "home_win_prob": pred.home_win_prob,
                    "away_win_prob": pred.away_win_prob,
                    "draw_prob": pred.draw_prob,
                    "predicted_margin": pred.predicted_margin,
                })
            except ValueError as e:
                print(f"Skipping {match.match_id}: {e}")
                continue

        return pd.DataFrame(predictions)
