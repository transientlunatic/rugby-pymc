"""
Match prediction functionality for rugby ranking models.

Supports:
- Team-only predictions (when lineups unknown)
- Full lineup predictions (when team sheets available)
- Proper uncertainty quantification
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import arviz as az

from rugby_ranking.model.core import RugbyModel
from rugby_ranking.model.data import normalize_team_name


# Average points per try-equivalent scoring opportunity
# Try=5, Conversion~0.7*2=1.4, Penalty~0.3*3=0.9 per match
# Roughly 5 points per successful try-scoring attack
POINTS_PER_TRY = 5.0

# Typical number of try-scoring opportunities per team per match
# Based on observed data: ~3.5 tries per team per match average
TRIES_PER_TEAM_PER_MATCH = 3.5

# Average conversion rate
CONVERSION_RATE = 0.70

# Average penalties per match per team
PENALTIES_PER_MATCH = 2.5


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
            f"  Predicted: {self.home.mean:.0f} - {self.away.mean:.0f}\n"
            f"  Home win: {self.home_win_prob:.1%}, "
            f"Away win: {self.away_win_prob:.1%}, "
            f"Draw: {self.draw_prob:.1%}\n"
            f"  90% CI: [{self.home.ci_lower:.0f}-{self.home.ci_upper:.0f}] vs "
            f"[{self.away.ci_lower:.0f}-{self.away.ci_upper:.0f}]"
        )


class MatchPredictor:
    """
    Generate match predictions from fitted models.

    The model estimates per-player try-scoring rates. To predict match scores:
    1. Aggregate player/team effects into a team-level try rate multiplier
    2. Apply to baseline tries per team (~3.5)
    3. Convert tries to points (try=5 + conversion~1.4 = 6.4 avg)
    4. Add penalty contribution (~7.5 points per team)
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
        """
        # Normalize team names
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

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
        n_total = posterior["alpha"].values.size
        sample_idx = np.random.choice(n_total, size=n_samples, replace=n_total < n_samples)

        # Flatten and sample
        gamma_flat = posterior["gamma_team_season"].values.reshape(-1, posterior["gamma_team_season"].shape[-1])
        eta_flat = posterior["eta_home"].values.flatten()
        sigma_player_flat = posterior["sigma_player"].values.flatten()

        gamma = gamma_flat[sample_idx]
        eta_home = eta_flat[sample_idx]
        sigma_player = sigma_player_flat[sample_idx]

        # Team effects (relative to average team)
        home_team_effect = gamma[:, home_idx]
        away_team_effect = gamma[:, away_idx]

        # For teams-only prediction, add uncertainty for unknown player composition
        # This represents variation in which players actually play
        player_uncertainty = np.random.normal(0, sigma_player * 0.5, size=n_samples)

        # Compute team strength multipliers (exponentiated random effects)
        # These multiply the baseline try rate
        home_multiplier = np.exp(home_team_effect + eta_home + player_uncertainty)
        away_multiplier = np.exp(away_team_effect + np.random.normal(0, sigma_player * 0.5, size=n_samples))

        # Expected tries for each team
        home_tries_expected = TRIES_PER_TEAM_PER_MATCH * home_multiplier
        away_tries_expected = TRIES_PER_TEAM_PER_MATCH * away_multiplier

        # Sample actual tries (Poisson)
        home_tries = np.random.poisson(home_tries_expected)
        away_tries = np.random.poisson(away_tries_expected)

        # Convert to points
        # Tries: 5 points each
        # Conversions: ~70% success rate, 2 points each
        # Penalties: add baseline amount with some variance
        home_conversions = np.random.binomial(home_tries, CONVERSION_RATE)
        away_conversions = np.random.binomial(away_tries, CONVERSION_RATE)

        home_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)
        away_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)

        home_scores = (
            home_tries * 5 +
            home_conversions * 2 +
            home_penalties * 3
        )
        away_scores = (
            away_tries * 5 +
            away_conversions * 2 +
            away_penalties * 3
        )

        return self._build_prediction(home_team, away_team, home_scores, away_scores)

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
        """
        # Normalize team names
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

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
        n_total = posterior["alpha"].values.size
        sample_idx = np.random.choice(n_total, size=n_samples, replace=n_total < n_samples)

        gamma_flat = posterior["gamma_team_season"].values.reshape(-1, posterior["gamma_team_season"].shape[-1])
        beta_flat = posterior["beta_player"].values.reshape(-1, posterior["beta_player"].shape[-1])
        theta_flat = posterior["theta_position"].values.reshape(-1, posterior["theta_position"].shape[-1])
        eta_flat = posterior["eta_home"].values.flatten()

        gamma = gamma_flat[sample_idx]
        beta = beta_flat[sample_idx]
        theta = theta_flat[sample_idx]
        eta_home = eta_flat[sample_idx]

        # Compute lineup strength for each team
        home_effect = self._compute_lineup_effect(
            home_lineup, home_ts_idx, gamma, beta, theta, eta_home, is_home=True
        )
        away_effect = self._compute_lineup_effect(
            away_lineup, away_ts_idx, gamma, beta, theta, eta_home, is_home=False
        )

        # Convert to try multipliers
        home_multiplier = np.exp(home_effect)
        away_multiplier = np.exp(away_effect)

        # Expected tries
        home_tries_expected = TRIES_PER_TEAM_PER_MATCH * home_multiplier
        away_tries_expected = TRIES_PER_TEAM_PER_MATCH * away_multiplier

        # Sample tries and convert to points
        home_tries = np.random.poisson(home_tries_expected)
        away_tries = np.random.poisson(away_tries_expected)

        home_conversions = np.random.binomial(home_tries, CONVERSION_RATE)
        away_conversions = np.random.binomial(away_tries, CONVERSION_RATE)

        home_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)
        away_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)

        home_scores = home_tries * 5 + home_conversions * 2 + home_penalties * 3
        away_scores = away_tries * 5 + away_conversions * 2 + away_penalties * 3

        return self._build_prediction(home_team, away_team, home_scores, away_scores)

    def _compute_lineup_effect(
        self,
        lineup: dict[int, str],
        team_season_idx: int,
        gamma: np.ndarray,
        beta: np.ndarray,
        theta: np.ndarray,
        eta_home: np.ndarray,
        is_home: bool,
    ) -> np.ndarray:
        """Compute aggregate team effect from lineup."""
        n_samples = len(gamma)

        # Team-season effect
        effect = gamma[:, team_season_idx].copy()

        # Home advantage
        if is_home:
            effect += eta_home

        # Sum player contributions (for starting XV, positions 1-15)
        n_players_counted = 0
        for position, player_name in lineup.items():
            if position > 15:  # Only count starters for team strength
                continue

            if player_name in self.model._player_ids:
                player_idx = self.model._player_ids[player_name]
                effect += beta[:, player_idx]
                n_players_counted += 1

            # Position effect
            if 1 <= position <= 23:
                effect += theta[:, position - 1]

        # Average over counted players (if any)
        if n_players_counted > 0:
            # Don't divide - we want the sum of player effects
            # But we should normalize by typical lineup size for comparability
            pass  # Keep raw sum

        return effect

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
            mean=float(home_scores.mean()),
            std=float(home_scores.std()),
            median=float(np.median(home_scores)),
            ci_lower=float(np.percentile(home_scores, 5)),
            ci_upper=float(np.percentile(home_scores, 95)),
            samples=home_scores,
        )

        away_pred = ScorePrediction(
            team=away_team,
            mean=float(away_scores.mean()),
            std=float(np.percentile(away_scores, 95)),
            median=float(np.median(away_scores)),
            ci_lower=float(np.percentile(away_scores, 5)),
            ci_upper=float(np.percentile(away_scores, 95)),
            samples=away_scores,
        )

        # Win probabilities
        home_wins = float((home_scores > away_scores).mean())
        away_wins = float((away_scores > home_scores).mean())
        draws = float((home_scores == away_scores).mean())

        margin = home_scores - away_scores

        return MatchPrediction(
            home=home_pred,
            away=away_pred,
            home_win_prob=home_wins,
            away_win_prob=away_wins,
            draw_prob=draws,
            predicted_margin=float(margin.mean()),
            margin_std=float(margin.std()),
        )

    def predict_upcoming(
        self,
        unplayed_matches: list,  # List of MatchData
        season: str,
    ) -> pd.DataFrame:
        """
        Generate predictions for all upcoming matches.
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
