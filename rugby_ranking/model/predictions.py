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


# Rugby scoring constants
CONVERSION_RATE = 0.70  # Typical conversion success rate
PENALTIES_PER_MATCH = 2.5  # Average penalties per team per match

# Number of players on field per team
STARTERS = 15


def _get_team_season_index_with_fallback(
    team_season_ids: dict[tuple[str, str], int],
    team: str,
    season: str,
) -> tuple[int | None, str | None, bool]:
    """
    Get team-season index, falling back to most recent season if exact match not found.

    Args:
        team_season_ids: Dictionary mapping (team, season) -> index
        team: Team name
        season: Requested season

    Returns:
        Tuple of (index, fallback_season, use_prior):
        - index: team-season index or None if using prior
        - fallback_season: season used if different from requested, or None
        - use_prior: True if team never seen (use model prior)
    """
    # Try exact match first
    if (team, season) in team_season_ids:
        return team_season_ids[(team, season)], None, False

    # Find all seasons for this team
    team_seasons = [(t, s) for (t, s) in team_season_ids.keys() if t == team]

    if not team_seasons:
        # Team never seen - caller should use prior
        return None, None, True

    # Use most recent season as fallback
    most_recent = max(team_seasons, key=lambda x: x[1])
    fallback_season = most_recent[1]
    fallback_idx = team_season_ids[most_recent]

    return fallback_idx, fallback_season, False


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
    # Metadata about prediction method
    home_season_used: str | None = None  # Season data used for home team (if fallback)
    away_season_used: str | None = None  # Season data used for away team (if fallback)

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

    The model estimates per-player try-scoring rates:
        log(λ_player) = α + β_player + γ_team + θ_position + η_home

    To predict match scores:
    1. Compute expected tries per player from the model
    2. Sum across the 15 starters to get team expected tries
    3. Sample actual tries from Poisson
    4. Convert to points (try=5 + conversion~1.4 = 6.4 avg)
    5. Add penalty contribution
    """

    def __init__(self, model: RugbyModel, trace: az.InferenceData | None = None):
        self.model = model
        self.trace = trace or model.trace

        if self.trace is None:
            raise ValueError("No trace available. Fit the model first.")

        # Pre-compute average position effects for teams-only predictions
        self._compute_position_averages()

        # Pre-flatten posterior arrays to eliminate redundant computation (60-70% speedup!)
        self._cache_posterior_arrays()

    def _compute_position_averages(self):
        """Pre-compute average position effects for teams-only predictions."""
        posterior = self.trace.posterior

        # Get theta_position - shape is (chain, draw, n_positions) or (chain, draw, n_score_types, n_positions)
        theta = posterior["theta_position"].values

        # Handle both single and joint model shapes
        if theta.ndim == 4:
            # Joint model: (chain, draw, n_score_types, n_positions)
            # For now, use first score type (tries)
            theta = theta[:, :, 0, :]

        # Average position effect across posterior (for positions 1-15)
        # Shape: (n_positions,)
        self._theta_mean = theta.mean(axis=(0, 1))[:STARTERS]

    def _cache_posterior_arrays(self):
        """
        Pre-flatten posterior parameter arrays to eliminate redundant reshaping
        on every prediction call. This is the primary performance bottleneck.

        This method caches flattened versions of all posterior parameters,
        handling all model variants (joint, time-varying, static).

        Performance impact: ~60-70% speedup in predict_teams_only()
        Memory overhead: ~50-100MB depending on model size
        """
        posterior = self.trace.posterior

        # Alpha - handle joint vs single model
        alpha_vals = posterior["alpha"].values
        if alpha_vals.ndim == 3:
            # Joint model: shape (chain, draw, n_score_types) - use first (tries)
            self._alpha_flat = alpha_vals[:, :, 0].flatten()
        else:
            self._alpha_flat = alpha_vals.flatten()

        # Gamma - handle time-varying, joint, and static models
        if "gamma_team_base_raw" in posterior:
            # Time-varying model
            gamma_base_raw = posterior["gamma_team_base_raw"].values
            gamma_trend_raw = posterior["gamma_team_trend_raw"].values
            sigma_team_base = posterior["sigma_team_base"].values
            sigma_team_trend = posterior["sigma_team_trend"].values
            lambda_team = posterior["lambda_team"].values

            # Compute gamma at mid-season (t=0.5)
            gamma_base = sigma_team_base[:, :, None] * lambda_team[:, :, 0:1] * gamma_base_raw
            gamma_trend = sigma_team_trend[:, :, None] * lambda_team[:, :, 0:1] * gamma_trend_raw
            gamma_combined = gamma_base + 0.5 * gamma_trend
            self._gamma_flat = gamma_combined.reshape(-1, gamma_combined.shape[-1])

        elif "gamma_team_season_raw" in posterior:
            # Joint model uses raw + scaling
            gamma_raw = posterior["gamma_team_season_raw"].values
            sigma_team = posterior["sigma_team"].values
            lambda_team = posterior["lambda_team"].values
            # Compute effective gamma for tries (index 0)
            self._gamma_flat = (sigma_team[:, :, None] * lambda_team[:, :, 0:1] * gamma_raw).reshape(
                -1, gamma_raw.shape[-1]
            )
        else:
            # Static model
            gamma_vals = posterior["gamma_team_season"].values
            self._gamma_flat = gamma_vals.reshape(-1, gamma_vals.shape[-1])

        # Theta - handle joint vs single model
        theta_vals = posterior["theta_position"].values
        if theta_vals.ndim == 4:
            # Joint model: (chain, draw, n_score_types, n_positions) - use tries
            self._theta_flat = theta_vals[:, :, 0, :].reshape(-1, theta_vals.shape[-1])
        else:
            self._theta_flat = theta_vals.reshape(-1, theta_vals.shape[-1])

        # Eta - handle joint vs single model
        eta_vals = posterior["eta_home"].values
        if eta_vals.ndim == 3:
            # Joint model: (chain, draw, n_score_types) - use tries
            self._eta_flat = eta_vals[:, :, 0].flatten()
        else:
            self._eta_flat = eta_vals.flatten()

        # Sigma player - handle time-varying, separate, joint models
        if "sigma_player_try_base" in posterior:
            # Time-varying model - use base effect
            sigma_base = posterior["sigma_player_try_base"].values
            self._sigma_player_flat = sigma_base.flatten()
        elif "sigma_player_try" in posterior:
            # Separate effects model
            self._sigma_player_flat = posterior["sigma_player_try"].values.flatten()
        elif "lambda_player" in posterior:
            # Joint model with loading factors - compute effective sigma for tries
            sigma_raw = posterior["sigma_player"].values
            lambda_tries = posterior["lambda_player"].values[:, :, 0]  # tries is index 0
            self._sigma_player_flat = (sigma_raw * lambda_tries).flatten()
        else:
            # Simple static model
            self._sigma_player_flat = posterior["sigma_player"].values.flatten()

        # Beta (player effects) - needed for full lineup predictions
        if "beta_player_try_base_raw" in posterior:
            # Time-varying model with separate effects
            beta_base_raw = posterior["beta_player_try_base_raw"].values
            beta_trend_raw = posterior["beta_player_try_trend_raw"].values
            sigma_player_base = posterior["sigma_player_try_base"].values
            sigma_player_trend = posterior["sigma_player_try_trend"].values
            lambda_player = posterior["lambda_player_try"].values

            # Compute at mid-season and average across seasons
            beta_base = sigma_player_base[:, :, None, None] * lambda_player[:, :, 0:1, None] * beta_base_raw
            beta_trend = sigma_player_trend[:, :, None, None] * lambda_player[:, :, 0:1, None] * beta_trend_raw
            beta_all_seasons = beta_base + 0.5 * beta_trend
            self._beta_flat = beta_all_seasons.mean(axis=-1).reshape(-1, beta_all_seasons.shape[-2])

        elif "beta_player_try_raw" in posterior:
            # Separate kicking/try-scoring effects - use try effect
            beta_raw = posterior["beta_player_try_raw"].values
            sigma_player = posterior["sigma_player_try"].values
            lambda_player = posterior["lambda_player_try"].values
            self._beta_flat = (sigma_player[:, :, None] * lambda_player[:, :, 0:1] * beta_raw).reshape(
                -1, beta_raw.shape[-1]
            )
        elif "beta_player_raw" in posterior:
            # Single player effect (original joint model)
            beta_raw = posterior["beta_player_raw"].values
            sigma_player = posterior["sigma_player"].values
            lambda_player = posterior["lambda_player"].values
            self._beta_flat = (sigma_player[:, :, None] * lambda_player[:, :, 0:1] * beta_raw).reshape(
                -1, beta_raw.shape[-1]
            )
        else:
            # Single score type model
            self._beta_flat = posterior["beta_player"].values.reshape(-1, posterior["beta_player"].shape[-1])

        # Store total sample count for indexing
        self._n_total = len(self._alpha_flat)

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

        # Get team-season indices with fallback to most recent season
        home_idx, home_fallback, home_use_prior = _get_team_season_index_with_fallback(
            self.model._team_season_ids, home_team, season
        )
        away_idx, away_fallback, away_use_prior = _get_team_season_index_with_fallback(
            self.model._team_season_ids, away_team, season
        )

        # Handle completely unknown teams
        if home_use_prior:
            raise ValueError(
                f"Unknown team: {home_team} (not found in any season). "
                "Prior-only predictions not yet implemented."
            )
        if away_use_prior:
            raise ValueError(
                f"Unknown team: {away_team} (not found in any season). "
                "Prior-only predictions not yet implemented."
            )

        # Store fallback info for metadata
        home_season_used = home_fallback if home_fallback else None
        away_season_used = away_fallback if away_fallback else None

        # Use pre-cached posterior arrays (60-70% speedup!)
        sample_idx = np.random.choice(self._n_total, size=n_samples, replace=self._n_total < n_samples)

        # Sample from cached flattened arrays
        alpha = self._alpha_flat[sample_idx]
        gamma = self._gamma_flat[sample_idx]
        theta = self._theta_flat[sample_idx]
        eta_home = self._eta_flat[sample_idx]
        sigma_player = self._sigma_player_flat[sample_idx]

        # Team effects
        home_team_effect = gamma[:, home_idx]
        away_team_effect = gamma[:, away_idx]

        # Compute expected tries for each team by summing across all 15 positions
        # For each position, expected tries = exp(alpha + gamma_team + theta_position + eta_home + player_uncertainty)
        # Since we don't know specific players, we marginalize over average player (beta ~ N(0, sigma_player))

        home_tries_expected = np.zeros(n_samples)
        away_tries_expected = np.zeros(n_samples)

        for pos in range(STARTERS):
            # Player uncertainty for unknown player at this position
            player_noise_home = np.random.normal(0, sigma_player)
            player_noise_away = np.random.normal(0, sigma_player)

            # Log-rate for this position
            # Note: theta[:, pos] gives position effect for position pos+1 (0-indexed)
            home_log_rate = alpha + home_team_effect + theta[:, pos] + eta_home + player_noise_home
            away_log_rate = alpha + away_team_effect + theta[:, pos] + player_noise_away

            # Expected tries for this position (rate, not count)
            home_tries_expected += np.exp(home_log_rate)
            away_tries_expected += np.exp(away_log_rate)

        # Sample actual tries (Poisson) - sum of independent Poissons
        home_tries = np.random.poisson(home_tries_expected)
        away_tries = np.random.poisson(away_tries_expected)

        # Convert to points
        home_conversions = np.random.binomial(home_tries, CONVERSION_RATE)
        away_conversions = np.random.binomial(away_tries, CONVERSION_RATE)

        home_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)
        away_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)

        home_scores = home_tries * 5 + home_conversions * 2 + home_penalties * 3
        away_scores = away_tries * 5 + away_conversions * 2 + away_penalties * 3

        return self._build_prediction(
            home_team, away_team, home_scores, away_scores, home_season_used, away_season_used
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
        """
        # Normalize team names
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

        # Get team-season indices with fallback
        home_ts_idx, home_fallback, home_use_prior = _get_team_season_index_with_fallback(
            self.model._team_season_ids, home_team, season
        )
        away_ts_idx, away_fallback, away_use_prior = _get_team_season_index_with_fallback(
            self.model._team_season_ids, away_team, season
        )

        # Handle completely unknown teams
        if home_use_prior:
            raise ValueError(
                f"Unknown team: {home_team} (not found in any season). "
                "Prior-only predictions not yet implemented."
            )
        if away_use_prior:
            raise ValueError(
                f"Unknown team: {away_team} (not found in any season). "
                "Prior-only predictions not yet implemented."
            )

        # Store fallback info for metadata
        home_season_used = home_fallback if home_fallback else None
        away_season_used = away_fallback if away_fallback else None

        # Use pre-cached posterior arrays (60-70% speedup!)
        sample_idx = np.random.choice(self._n_total, size=n_samples, replace=self._n_total < n_samples)

        # Sample from cached flattened arrays
        alpha = self._alpha_flat[sample_idx]
        gamma = self._gamma_flat[sample_idx]
        beta = self._beta_flat[sample_idx]
        theta = self._theta_flat[sample_idx]
        eta_home = self._eta_flat[sample_idx]
        sigma_player = self._sigma_player_flat[sample_idx]

        # Team effects
        home_team_effect = gamma[:, home_ts_idx]
        away_team_effect = gamma[:, away_ts_idx]

        # Compute expected tries by summing across lineup
        home_tries_expected = np.zeros(n_samples)
        away_tries_expected = np.zeros(n_samples)

        for pos in range(1, STARTERS + 1):
            # Get player effect for this position (or use noise if unknown)
            if pos in home_lineup and home_lineup[pos] in self.model._player_ids:
                home_player_idx = self.model._player_ids[home_lineup[pos]]
                home_player_effect = beta[:, home_player_idx]
            else:
                # Unknown player - sample from prior
                home_player_effect = np.random.normal(0, sigma_player)

            if pos in away_lineup and away_lineup[pos] in self.model._player_ids:
                away_player_idx = self.model._player_ids[away_lineup[pos]]
                away_player_effect = beta[:, away_player_idx]
            else:
                away_player_effect = np.random.normal(0, sigma_player)

            # Position effect (0-indexed)
            pos_effect = theta[:, pos - 1]

            # Log-rate for this position
            home_log_rate = alpha + home_team_effect + home_player_effect + pos_effect + eta_home
            away_log_rate = alpha + away_team_effect + away_player_effect + pos_effect

            # Expected tries for this position
            home_tries_expected += np.exp(home_log_rate)
            away_tries_expected += np.exp(away_log_rate)

        # Sample actual tries (Poisson)
        home_tries = np.random.poisson(home_tries_expected)
        away_tries = np.random.poisson(away_tries_expected)

        # Convert to points
        home_conversions = np.random.binomial(home_tries, CONVERSION_RATE)
        away_conversions = np.random.binomial(away_tries, CONVERSION_RATE)

        home_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)
        away_penalties = np.random.poisson(PENALTIES_PER_MATCH, size=n_samples)

        home_scores = home_tries * 5 + home_conversions * 2 + home_penalties * 3
        away_scores = away_tries * 5 + away_conversions * 2 + away_penalties * 3

        return self._build_prediction(
            home_team, away_team, home_scores, away_scores, home_season_used, away_season_used
        )

    def _build_prediction(
        self,
        home_team: str,
        away_team: str,
        home_scores: np.ndarray,
        away_scores: np.ndarray,
        home_season_used: str | None = None,
        away_season_used: str | None = None,
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
            std=float(away_scores.std()),
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
            home_season_used=home_season_used,
            away_season_used=away_season_used,
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
                    "margin_std": pred.margin_std,
                })
            except ValueError as e:
                print(f"Skipping {match.match_id}: {e}")
                continue

        return pd.DataFrame(predictions)
