"""
Core PyMC model definition for rugby player and team ranking.

This module implements a hierarchical Bayesian model with:
- Player random effects (intrinsic ability, follows player across teams)
- Team-season random effects (system/coaching quality)
- Player-team interaction effects (fit between player and system)
- Position effects (base rates by jersey number)
- Exposure adjustment (minutes played as offset)
- Separate processes for different scoring types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


@dataclass
class ModelConfig:
    """Configuration for the rugby ranking model."""

    # Which scoring types to model
    score_types: tuple[str, ...] = ("tries", "penalties", "conversions", "drop_goals")

    # Prior scales
    player_effect_sd: float = 0.5
    team_effect_sd: float = 0.3
    player_team_effect_sd: float = 0.2
    position_effect_sd: float = 0.5

    # Separate player effects for different scoring types
    separate_kicking_effect: bool = True
    player_kicking_effect_sd: float = 0.5
    player_try_effect_sd: float = 0.5

    # Defensive effects
    include_defense: bool = True
    defense_effect_sd: float = 0.3

    # Home advantage
    home_advantage_prior_mean: float = 0.1
    home_advantage_prior_sd: float = 0.1

    # Exposure
    reference_minutes: float = 80.0  # Full match

    # Position groupings for scoring types
    # These positions are more likely to score tries
    try_scoring_positions: tuple[int, ...] = (11, 12, 13, 14, 15)  # Back line
    # These positions take kicks
    kicking_positions: tuple[int, ...] = (10, 15)  # Fly-half, fullback

    # Time-varying effects (within-season trends)
    time_varying_effects: bool = False  # Enable within-season form trends
    player_trend_sd: float = 0.1  # Prior SD for player trend slopes
    team_trend_sd: float = 0.1  # Prior SD for team trend slopes
    season_evolution_sd: float = 0.2  # Prior SD for season-to-season base changes


class RugbyModel:
    """
    Hierarchical Bayesian model for rugby player and team ranking.

    The model structure:
        log(λ_score[i,m]) = α
                          + β_player[player_id]
                          + γ_team[team_id, season_id]
                          + δ_player_team[player_id, team_id]
                          + θ_position[position]
                          + η_home × is_home
                          + log(minutes / 80)  # exposure offset

        N_score[i,m] ~ Poisson(λ_score[i,m])

    Where separate models are fit for tries, penalties, conversions, drop_goals.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model: pm.Model | None = None
        self.trace = None

        # Index mappings (set during build)
        self._player_ids: dict[str, int] = {}
        self._team_ids: dict[str, int] = {}
        self._season_ids: dict[str, int] = {}
        self._team_season_ids: dict[tuple[str, str], int] = {}
        self._position_ids: dict[int, int] = {}  # Map raw position to 0-indexed

    def build(self, df: pd.DataFrame, score_type: str = "tries") -> pm.Model:
        """
        Build the PyMC model for a specific scoring type.

        Args:
            df: DataFrame from MatchDataset.to_dataframe()
            score_type: One of 'tries', 'penalties', 'conversions', 'drop_goals'

        Returns:
            PyMC model ready for sampling
        """
        if score_type not in self.config.score_types:
            raise ValueError(f"Unknown score type: {score_type}")

        # Build indices
        self._build_indices(df)

        # Prepare data arrays
        data = self._prepare_data(df, score_type)

        with pm.Model() as model:
            # === Data ===
            player_idx = pm.Data("player_idx", data["player_idx"])
            team_idx = pm.Data("team_idx", data["team_idx"])
            team_season_idx = pm.Data("team_season_idx", data["team_season_idx"])
            opponent_team_season_idx = pm.Data(
                "opponent_team_season_idx", data["opponent_team_season_idx"]
            )
            position_idx = pm.Data("position_idx", data["position_idx"])
            is_home = pm.Data("is_home", data["is_home"])
            exposure = pm.Data("exposure", data["exposure"])
            observed = pm.Data("observed", data["observed"])

            n_players = len(self._player_ids)
            n_teams = len(self._team_ids)
            n_team_seasons = len(self._team_season_ids)
            n_positions = len(self._position_ids)

            # === Hyperpriors ===
            # Player effect variance
            sigma_player = pm.HalfNormal("sigma_player", sigma=self.config.player_effect_sd)

            # Team-season effect variance
            sigma_team = pm.HalfNormal("sigma_team", sigma=self.config.team_effect_sd)

            # === Random Effects ===
            # Player intrinsic ability (follows player across teams)
            beta_player = pm.Normal(
                "beta_player",
                mu=0,
                sigma=sigma_player,
                shape=n_players,
            )

            # Team-season offensive effect (coaching, tactics, squad quality)
            gamma_team_season = pm.Normal(
                "gamma_team_season",
                mu=0,
                sigma=sigma_team,
                shape=n_team_seasons,
            )

            # Defensive effects (reduce opponent scoring)
            if self.config.include_defense:
                sigma_defense = pm.HalfNormal("sigma_defense", sigma=self.config.defense_effect_sd)
                delta_defense = pm.Normal(
                    "delta_defense",
                    mu=0,
                    sigma=sigma_defense,
                    shape=n_team_seasons,
                )

            # Position effects (some positions score more)
            theta_position = pm.Normal(
                "theta_position",
                mu=0,
                sigma=self.config.position_effect_sd,
                shape=n_positions,
            )

            # === Fixed Effects ===
            # Intercept (baseline rate)
            alpha = pm.Normal("alpha", mu=-2, sigma=1)

            # Home advantage
            eta_home = pm.Normal(
                "eta_home",
                mu=self.config.home_advantage_prior_mean,
                sigma=self.config.home_advantage_prior_sd,
            )

            # === Linear Predictor ===
            log_lambda = (
                alpha
                + beta_player[player_idx]
                + gamma_team_season[team_season_idx]
                + theta_position[position_idx]  # Already 0-indexed
                + eta_home * is_home
                + pt.log(exposure)  # Exposure offset
            )

            # Add defensive effect if enabled
            if self.config.include_defense:
                log_lambda = log_lambda - delta_defense[opponent_team_season_idx]

            lambda_ = pt.exp(log_lambda)

            # === Likelihood ===
            y = pm.Poisson("y", mu=lambda_, observed=observed)

        self.model = model
        return model

    def build_joint(self, df: pd.DataFrame) -> pm.Model:
        """
        Build a joint model for all scoring types.

        This shares player and team effects across scoring types while allowing
        different baseline rates and position effects for each type.

        If config.separate_kicking_effect is True, uses separate player effects
        for try-scoring vs kicking ability.
        """
        self._build_indices(df)

        with pm.Model() as model:
            n_players = len(self._player_ids)
            n_team_seasons = len(self._team_season_ids)
            n_positions = len(self._position_ids)
            n_score_types = len(self.config.score_types)

            # === Shared Hyperpriors ===
            sigma_team = pm.HalfNormal("sigma_team", sigma=self.config.team_effect_sd)

            # Defensive effects hyperprior
            if self.config.include_defense:
                sigma_defense = pm.HalfNormal(
                    "sigma_defense", sigma=self.config.defense_effect_sd
                )

            # === Shared Random Effects ===
            if self.config.separate_kicking_effect:
                # Separate player effects for try-scoring vs kicking
                sigma_player_try = pm.HalfNormal(
                    "sigma_player_try", sigma=self.config.player_try_effect_sd
                )
                sigma_player_kick = pm.HalfNormal(
                    "sigma_player_kick", sigma=self.config.player_kicking_effect_sd
                )

                beta_player_try_raw = pm.Normal(
                    "beta_player_try_raw", mu=0, sigma=1, shape=n_players
                )
                beta_player_kick_raw = pm.Normal(
                    "beta_player_kick_raw", mu=0, sigma=1, shape=n_players
                )

                # Type-specific loading factors (how much each score type loads on try vs kick)
                lambda_player_try = pm.HalfNormal(
                    "lambda_player_try", sigma=0.5, shape=n_score_types
                )
                lambda_player_kick = pm.HalfNormal(
                    "lambda_player_kick", sigma=0.5, shape=n_score_types
                )
            else:
                # Original single player effect
                sigma_player = pm.HalfNormal("sigma_player", sigma=self.config.player_effect_sd)
                beta_player_raw = pm.Normal("beta_player_raw", mu=0, sigma=1, shape=n_players)
                lambda_player = pm.HalfNormal("lambda_player", sigma=0.5, shape=n_score_types)

            # Team-season offensive effect (shared)
            gamma_team_season_raw = pm.Normal(
                "gamma_team_season_raw", mu=0, sigma=1, shape=n_team_seasons
            )

            # Team-season defensive effect (shared)
            if self.config.include_defense:
                delta_defense_raw = pm.Normal(
                    "delta_defense_raw", mu=0, sigma=1, shape=n_team_seasons
                )

            # === Score-Type Specific Parameters ===
            # Intercepts for each score type
            alpha = pm.Normal("alpha", mu=-2, sigma=1, shape=n_score_types)

            # Team effect loadings per score type
            lambda_team = pm.HalfNormal("lambda_team", sigma=0.5, shape=n_score_types)

            # Defensive effect loadings per score type
            if self.config.include_defense:
                lambda_defense = pm.HalfNormal(
                    "lambda_defense", sigma=0.5, shape=n_score_types
                )

            # Position effects per score type
            theta_position = pm.Normal(
                "theta_position",
                mu=0,
                sigma=self.config.position_effect_sd,
                shape=(n_score_types, n_positions),
            )

            # Home advantage per score type
            eta_home = pm.Normal(
                "eta_home",
                mu=self.config.home_advantage_prior_mean,
                sigma=self.config.home_advantage_prior_sd,
                shape=n_score_types,
            )

            # === Build likelihood for each score type ===
            for s, score_type in enumerate(self.config.score_types):
                data = self._prepare_data(df, score_type)

                player_idx = pm.Data(f"player_idx_{score_type}", data["player_idx"])
                team_season_idx = pm.Data(
                    f"team_season_idx_{score_type}", data["team_season_idx"]
                )
                opponent_team_season_idx_data = pm.Data(
                    f"opponent_team_season_idx_{score_type}",
                    data["opponent_team_season_idx"],
                )
                position_idx = pm.Data(f"position_idx_{score_type}", data["position_idx"])
                is_home_data = pm.Data(f"is_home_{score_type}", data["is_home"])
                exposure = pm.Data(f"exposure_{score_type}", data["exposure"])
                observed = pm.Data(f"observed_{score_type}", data["observed"])

                # Compute player effects based on score type
                if self.config.separate_kicking_effect:
                    # Tries use try-scoring effect, kicks use kicking effect
                    if score_type == "tries":
                        beta_player = (
                            sigma_player_try * lambda_player_try[s] * beta_player_try_raw
                        )
                    else:  # conversions, penalties, drop_goals
                        beta_player = (
                            sigma_player_kick * lambda_player_kick[s] * beta_player_kick_raw
                        )
                else:
                    # Original single player effect
                    beta_player = sigma_player * lambda_player[s] * beta_player_raw

                # Scaled team offensive effect
                gamma_team_season = sigma_team * lambda_team[s] * gamma_team_season_raw

                # Scaled defensive effect
                if self.config.include_defense:
                    delta_defense = (
                        sigma_defense * lambda_defense[s] * delta_defense_raw
                    )

                log_lambda = (
                    alpha[s]
                    + beta_player[player_idx]
                    + gamma_team_season[team_season_idx]
                    + theta_position[s, position_idx]  # Already 0-indexed
                    + eta_home[s] * is_home_data
                    + pt.log(exposure)
                )

                # Subtract opponent defensive effect
                if self.config.include_defense:
                    log_lambda = log_lambda - delta_defense[opponent_team_season_idx_data]

                lambda_ = pt.exp(log_lambda)

                pm.Poisson(f"y_{score_type}", mu=lambda_, observed=observed)

        self.model = model
        return model

    def build_joint_minibatch(
        self, df: pd.DataFrame, minibatch_size: int = 1024
    ) -> pm.Model:
        """
        Build joint model with minibatch support for faster VI.

        Uses data subsampling with total_size for proper likelihood scaling.
        For true minibatch SGD, use an external training loop that updates
        pm.MutableData between iterations.

        Args:
            df: DataFrame from MatchDataset.to_dataframe()
            minibatch_size: Approximate size of data subset per score type

        Returns:
            PyMC model with scaled likelihood for minibatch training
        """
        self._build_indices(df)

        with pm.Model() as model:
            n_players = len(self._player_ids)
            n_team_seasons = len(self._team_season_ids)
            n_positions = len(self._position_ids)
            n_score_types = len(self.config.score_types)

            # === Shared Hyperpriors ===
            sigma_team = pm.HalfNormal("sigma_team", sigma=self.config.team_effect_sd)

            # Defensive effects hyperprior
            if self.config.include_defense:
                sigma_defense = pm.HalfNormal(
                    "sigma_defense", sigma=self.config.defense_effect_sd
                )

            # === Shared Random Effects ===
            if self.config.separate_kicking_effect:
                sigma_player_try = pm.HalfNormal(
                    "sigma_player_try", sigma=self.config.player_try_effect_sd
                )
                sigma_player_kick = pm.HalfNormal(
                    "sigma_player_kick", sigma=self.config.player_kicking_effect_sd
                )

                beta_player_try_raw = pm.Normal(
                    "beta_player_try_raw", mu=0, sigma=1, shape=n_players
                )
                beta_player_kick_raw = pm.Normal(
                    "beta_player_kick_raw", mu=0, sigma=1, shape=n_players
                )

                lambda_player_try = pm.HalfNormal(
                    "lambda_player_try", sigma=0.5, shape=n_score_types
                )
                lambda_player_kick = pm.HalfNormal(
                    "lambda_player_kick", sigma=0.5, shape=n_score_types
                )
            else:
                sigma_player = pm.HalfNormal(
                    "sigma_player", sigma=self.config.player_effect_sd
                )
                beta_player_raw = pm.Normal(
                    "beta_player_raw", mu=0, sigma=1, shape=n_players
                )
                lambda_player = pm.HalfNormal(
                    "lambda_player", sigma=0.5, shape=n_score_types
                )

            # Team-season offensive effect (shared)
            gamma_team_season_raw = pm.Normal(
                "gamma_team_season_raw", mu=0, sigma=1, shape=n_team_seasons
            )

            # Team-season defensive effect (shared)
            if self.config.include_defense:
                delta_defense_raw = pm.Normal(
                    "delta_defense_raw", mu=0, sigma=1, shape=n_team_seasons
                )

            # === Score-Type Specific Parameters ===
            alpha = pm.Normal("alpha", mu=-2, sigma=1, shape=n_score_types)
            lambda_team = pm.HalfNormal("lambda_team", sigma=0.5, shape=n_score_types)

            if self.config.include_defense:
                lambda_defense = pm.HalfNormal(
                    "lambda_defense", sigma=0.5, shape=n_score_types
                )

            theta_position = pm.Normal(
                "theta_position",
                mu=0,
                sigma=self.config.position_effect_sd,
                shape=(n_score_types, n_positions),
            )

            eta_home = pm.Normal(
                "eta_home",
                mu=self.config.home_advantage_prior_mean,
                sigma=self.config.home_advantage_prior_sd,
                shape=n_score_types,
            )

            # === Build likelihood for each score type with minibatches ===
            for s, score_type in enumerate(self.config.score_types):
                data = self._prepare_data(df, score_type)
                n_total = len(data["observed"])

                # Subsample data for faster VI (with proper scaling via total_size)
                # For true minibatch SGD, wrap this in a training loop
                if n_total > minibatch_size:
                    indices = np.random.choice(n_total, size=minibatch_size, replace=False)
                    data_subset = {k: v[indices] for k, v in data.items()}
                else:
                    data_subset = data

                # Use pm.Data for PyMC 4.x/5.x compatibility
                player_idx = pm.Data(
                    f"player_idx_{score_type}_mb", data_subset["player_idx"]
                )
                team_season_idx = pm.Data(
                    f"team_season_idx_{score_type}_mb", data_subset["team_season_idx"]
                )
                opponent_team_season_idx_data = pm.Data(
                    f"opponent_team_season_idx_{score_type}_mb",
                    data_subset["opponent_team_season_idx"],
                )
                position_idx = pm.Data(
                    f"position_idx_{score_type}_mb", data_subset["position_idx"]
                )
                is_home_data = pm.Data(
                    f"is_home_{score_type}_mb", data_subset["is_home"]
                )
                exposure = pm.Data(
                    f"exposure_{score_type}_mb", data_subset["exposure"]
                )
                observed = pm.Data(
                    f"observed_{score_type}_mb", data_subset["observed"]
                )

                # Compute player effects
                if self.config.separate_kicking_effect:
                    if score_type == "tries":
                        beta_player = (
                            sigma_player_try
                            * lambda_player_try[s]
                            * beta_player_try_raw
                        )
                    else:
                        beta_player = (
                            sigma_player_kick
                            * lambda_player_kick[s]
                            * beta_player_kick_raw
                        )
                else:
                    beta_player = sigma_player * lambda_player[s] * beta_player_raw

                gamma_team_season = sigma_team * lambda_team[s] * gamma_team_season_raw

                if self.config.include_defense:
                    delta_defense = (
                        sigma_defense * lambda_defense[s] * delta_defense_raw
                    )

                log_lambda = (
                    alpha[s]
                    + beta_player[player_idx]
                    + gamma_team_season[team_season_idx]
                    + theta_position[s, position_idx]  # Already 0-indexed
                    + eta_home[s] * is_home_data
                    + pt.log(exposure)
                )

                if self.config.include_defense:
                    log_lambda = log_lambda - delta_defense[
                        opponent_team_season_idx_data
                    ]

                lambda_ = pt.exp(log_lambda)

                # Likelihood with total_size for proper scaling
                pm.Poisson(
                    f"y_{score_type}",
                    mu=lambda_,
                    observed=observed,
                    total_size=n_total,  # Scale likelihood by full dataset size
                )

        self.model = model
        return model

    def build_joint_time_varying(self, df: pd.DataFrame) -> pm.Model:
        """
        Build a joint model with time-varying effects within seasons.

        This model includes:
        - Base player/team effects that evolve between seasons (random walk)
        - Trend slopes that capture within-season form changes
        - Separate effects for try-scoring vs kicking ability

        Model structure for each observation:
            log(λ[i,m]) = α[score_type]
                        + (beta_player_base[i, season] + beta_player_trend[i, season] * t_m)
                        + (gamma_team_base[j, season] + gamma_team_trend[j, season] * t_m)
                        + theta_position[score_type, position]
                        + eta_home[score_type] * is_home
                        + log(exposure)

        Where t_m is the normalized time within season (0 = start, 1 = end).

        Args:
            df: DataFrame from MatchDataset.to_dataframe() with 'date' column

        Returns:
            PyMC model with time-varying effects
        """
        if not self.config.time_varying_effects:
            raise ValueError(
                "time_varying_effects must be True in config to use this method. "
                "Use build_joint() for static effects."
            )

        self._build_indices(df)

        with pm.Model() as model:
            n_players = len(self._player_ids)
            n_team_seasons = len(self._team_season_ids)
            n_positions = len(self._position_ids)
            n_score_types = len(self.config.score_types)
            n_seasons = len(self._season_ids)

            # === Hyperpriors ===
            sigma_team_base = pm.HalfNormal("sigma_team_base", sigma=self.config.team_effect_sd)
            sigma_team_trend = pm.HalfNormal("sigma_team_trend", sigma=self.config.team_trend_sd)

            if self.config.include_defense:
                sigma_defense = pm.HalfNormal(
                    "sigma_defense", sigma=self.config.defense_effect_sd
                )

            # === Player Effects (Separate Try-Scoring vs Kicking) ===
            if self.config.separate_kicking_effect:
                # Try-scoring ability
                sigma_player_try_base = pm.HalfNormal(
                    "sigma_player_try_base", sigma=self.config.player_try_effect_sd
                )
                sigma_player_try_trend = pm.HalfNormal(
                    "sigma_player_try_trend", sigma=self.config.player_trend_sd
                )

                # Kicking ability
                sigma_player_kick_base = pm.HalfNormal(
                    "sigma_player_kick_base", sigma=self.config.player_kicking_effect_sd
                )
                sigma_player_kick_trend = pm.HalfNormal(
                    "sigma_player_kick_trend", sigma=self.config.player_trend_sd
                )

                # Player base effects (raw, per season)
                # Shape: (n_players, n_seasons)
                beta_player_try_base_raw = pm.Normal(
                    "beta_player_try_base_raw",
                    mu=0,
                    sigma=1,
                    shape=(n_players, n_seasons)
                )
                beta_player_kick_base_raw = pm.Normal(
                    "beta_player_kick_base_raw",
                    mu=0,
                    sigma=1,
                    shape=(n_players, n_seasons)
                )

                # Player trend slopes (raw, per season)
                beta_player_try_trend_raw = pm.Normal(
                    "beta_player_try_trend_raw",
                    mu=0,
                    sigma=1,
                    shape=(n_players, n_seasons)
                )
                beta_player_kick_trend_raw = pm.Normal(
                    "beta_player_kick_trend_raw",
                    mu=0,
                    sigma=1,
                    shape=(n_players, n_seasons)
                )

                # Loading factors per score type
                lambda_player_try = pm.HalfNormal(
                    "lambda_player_try", sigma=0.5, shape=n_score_types
                )
                lambda_player_kick = pm.HalfNormal(
                    "lambda_player_kick", sigma=0.5, shape=n_score_types
                )
            else:
                # Single player effect (not recommended with time-varying)
                raise NotImplementedError(
                    "Time-varying effects with unified player effects not yet implemented"
                )

            # === Team Effects (Base + Trend per team-season) ===
            gamma_team_base_raw = pm.Normal(
                "gamma_team_base_raw", mu=0, sigma=1, shape=n_team_seasons
            )
            gamma_team_trend_raw = pm.Normal(
                "gamma_team_trend_raw", mu=0, sigma=1, shape=n_team_seasons
            )

            # === Defensive Effects ===
            if self.config.include_defense:
                delta_defense_raw = pm.Normal(
                    "delta_defense_raw", mu=0, sigma=1, shape=n_team_seasons
                )

            # === Score-Type Specific Parameters ===
            alpha = pm.Normal("alpha", mu=-2, sigma=1, shape=n_score_types)
            lambda_team = pm.HalfNormal("lambda_team", sigma=0.5, shape=n_score_types)

            if self.config.include_defense:
                lambda_defense = pm.HalfNormal(
                    "lambda_defense", sigma=0.5, shape=n_score_types
                )

            theta_position = pm.Normal(
                "theta_position",
                mu=0,
                sigma=self.config.position_effect_sd,
                shape=(n_score_types, n_positions),
            )

            eta_home = pm.Normal(
                "eta_home",
                mu=self.config.home_advantage_prior_mean,
                sigma=self.config.home_advantage_prior_sd,
                shape=n_score_types,
            )

            # === Build likelihood for each score type ===
            for s, score_type in enumerate(self.config.score_types):
                data = self._prepare_data(df, score_type)

                player_idx = pm.Data(f"player_idx_{score_type}", data["player_idx"])
                team_season_idx = pm.Data(
                    f"team_season_idx_{score_type}", data["team_season_idx"]
                )
                opponent_team_season_idx_data = pm.Data(
                    f"opponent_team_season_idx_{score_type}",
                    data["opponent_team_season_idx"],
                )
                position_idx = pm.Data(f"position_idx_{score_type}", data["position_idx"])
                is_home_data = pm.Data(f"is_home_{score_type}", data["is_home"])
                exposure = pm.Data(f"exposure_{score_type}", data["exposure"])
                observed = pm.Data(f"observed_{score_type}", data["observed"])
                season_progress = pm.Data(
                    f"season_progress_{score_type}", data["season_progress"]
                )

                # Get season indices for each observation
                # We need to map team_season to season
                team_season_to_season = {
                    ts_id: self._season_ids[season]
                    for (team, season), ts_id in self._team_season_ids.items()
                }
                season_idx = np.array([
                    team_season_to_season[ts] for ts in data["team_season_idx"]
                ])
                season_idx_data = pm.Data(f"season_idx_{score_type}", season_idx)

                # Compute player effects (base + trend * progress)
                if score_type == "tries":
                    # Try-scoring effect
                    beta_player_base = (
                        sigma_player_try_base
                        * lambda_player_try[s]
                        * beta_player_try_base_raw[player_idx, season_idx_data]
                    )
                    beta_player_trend = (
                        sigma_player_try_trend
                        * lambda_player_try[s]
                        * beta_player_try_trend_raw[player_idx, season_idx_data]
                    )
                else:
                    # Kicking effect
                    beta_player_base = (
                        sigma_player_kick_base
                        * lambda_player_kick[s]
                        * beta_player_kick_base_raw[player_idx, season_idx_data]
                    )
                    beta_player_trend = (
                        sigma_player_kick_trend
                        * lambda_player_kick[s]
                        * beta_player_kick_trend_raw[player_idx, season_idx_data]
                    )

                # Player effect with trend
                beta_player = beta_player_base + beta_player_trend * season_progress

                # Team effects (base + trend * progress)
                gamma_team_base = sigma_team_base * lambda_team[s] * gamma_team_base_raw
                gamma_team_trend = sigma_team_trend * lambda_team[s] * gamma_team_trend_raw

                gamma_team = (
                    gamma_team_base[team_season_idx]
                    + gamma_team_trend[team_season_idx] * season_progress
                )

                # Defensive effect
                if self.config.include_defense:
                    delta_defense = sigma_defense * lambda_defense[s] * delta_defense_raw

                # Linear predictor
                log_lambda = (
                    alpha[s]
                    + beta_player
                    + gamma_team
                    + theta_position[s, position_idx]
                    + eta_home[s] * is_home_data
                    + pt.log(exposure)
                )

                # Subtract opponent defensive effect
                if self.config.include_defense:
                    log_lambda = log_lambda - delta_defense[opponent_team_season_idx_data]

                lambda_ = pt.exp(log_lambda)

                # Likelihood
                pm.Poisson(f"y_{score_type}", mu=lambda_, observed=observed)

        self.model = model
        return model

    def _build_indices(self, df: pd.DataFrame) -> None:
        """Build mappings from names to integer indices."""
        # Players
        unique_players = df["player_name"].unique()
        self._player_ids = {name: i for i, name in enumerate(unique_players)}

        # Teams
        unique_teams = df["team"].unique()
        self._team_ids = {name: i for i, name in enumerate(unique_teams)}

        # Seasons
        unique_seasons = df["season"].unique()
        self._season_ids = {name: i for i, name in enumerate(unique_seasons)}

        # Team-season combinations
        team_seasons = df[["team", "season"]].drop_duplicates()
        self._team_season_ids = {
            (row["team"], row["season"]): i
            for i, (_, row) in enumerate(team_seasons.iterrows())
        }

        # Positions - map raw position values to 0-indexed
        unique_positions = sorted(df["position"].unique())
        self._position_ids = {pos: i for i, pos in enumerate(unique_positions)}

    def _prepare_data(self, df: pd.DataFrame, score_type: str) -> dict[str, np.ndarray]:
        """Prepare data arrays for PyMC model."""
        # Filter to observations with positive exposure
        mask = df["minutes_played"] > 0
        df_filtered = df[mask].copy()

        # Compute season progress if time-varying effects enabled
        if self.config.time_varying_effects:
            # Get season boundaries
            season_dates = df.groupby("season")["date"].agg(["min", "max"])
            df_filtered = df_filtered.merge(
                season_dates.rename(columns={"min": "season_start", "max": "season_end"}),
                left_on="season",
                right_index=True,
                how="left"
            )

            # Compute progress (0 = season start, 1 = season end)
            df_filtered["days_into_season"] = (
                df_filtered["date"] - df_filtered["season_start"]
            ).dt.days
            df_filtered["season_duration"] = (
                df_filtered["season_end"] - df_filtered["season_start"]
            ).dt.days
            df_filtered["season_progress"] = (
                df_filtered["days_into_season"] / df_filtered["season_duration"]
            ).clip(0, 1)  # Ensure in [0, 1]

            season_progress = df_filtered["season_progress"].values
        else:
            season_progress = None

        player_idx = np.array([self._player_ids[p] for p in df_filtered["player_name"]])
        team_idx = np.array([self._team_ids[t] for t in df_filtered["team"]])
        team_season_idx = np.array([
            self._team_season_ids[(t, s)]
            for t, s in zip(df_filtered["team"], df_filtered["season"])
        ])

        # Opponent team-season indices for defensive effects
        opponent_team_season_idx = np.array([
            self._team_season_ids.get((opp, s), 0)  # Default to 0 if not found
            for opp, s in zip(df_filtered["opponent"], df_filtered["season"])
        ])

        # Map positions to 0-indexed values
        position_idx = np.array([self._position_ids[p] for p in df_filtered["position"]])
        is_home = df_filtered["is_home"].astype(float).values
        exposure = df_filtered["minutes_played"].values / self.config.reference_minutes
        observed = df_filtered[score_type].values

        result = {
            "player_idx": player_idx,
            "team_idx": team_idx,
            "team_season_idx": team_season_idx,
            "opponent_team_season_idx": opponent_team_season_idx,
            "position_idx": position_idx,
            "is_home": is_home,
            "exposure": exposure,
            "observed": observed,
        }

        if season_progress is not None:
            result["season_progress"] = season_progress

        return result

    def get_player_rankings(
        self,
        trace=None,
        score_type: str | None = None,
        top_n: int = 20,
        df: pd.DataFrame | None = None,
        min_scores: int | None = None,
    ) -> pd.DataFrame:
        """
        Extract player rankings from posterior.

        Args:
            trace: ArviZ InferenceData (uses self.trace if None)
            score_type: For joint models, which score type's rankings
            top_n: Number of top players to return
            df: DataFrame from MatchDataset.to_dataframe() for computing score counts
            min_scores: Minimum number of scores required to be included in rankings

        Returns:
            DataFrame with player rankings (includes total_scores column if df provided)
        """
        trace = trace or self.trace
        if trace is None:
            raise ValueError("No trace available. Run inference first.")

        # Detect model type by checking posterior variables
        has_separate_effects = "beta_player_try_raw" in trace.posterior
        is_joint_model = "beta_player_raw" in trace.posterior or has_separate_effects

        # Extract player effects
        if has_separate_effects:
            # Joint model with separate kicking/try-scoring effects
            if score_type and score_type in self.config.score_types:
                score_idx = self.config.score_types.index(score_type)

                if score_type == "tries":
                    # Use try-scoring effect
                    beta_raw = trace.posterior["beta_player_try_raw"].values
                    sigma = trace.posterior["sigma_player_try"].values
                    lambda_p = trace.posterior["lambda_player_try"].values[..., score_idx]
                else:
                    # Use kicking effect for penalties, conversions, drop_goals
                    beta_raw = trace.posterior["beta_player_kick_raw"].values
                    sigma = trace.posterior["sigma_player_kick"].values
                    lambda_p = trace.posterior["lambda_player_kick"].values[..., score_idx]

                beta = sigma[..., None] * lambda_p[..., None] * beta_raw
            else:
                # Default to tries (index 0) if no score_type specified
                beta_raw = trace.posterior["beta_player_try_raw"].values
                sigma = trace.posterior["sigma_player_try"].values
                lambda_p = trace.posterior["lambda_player_try"].values[..., 0]
                beta = sigma[..., None] * lambda_p[..., None] * beta_raw

        elif is_joint_model:
            # Joint model - original single player effect
            beta_raw = trace.posterior["beta_player_raw"].values
            sigma = trace.posterior["sigma_player"].values

            if score_type and score_type in self.config.score_types:
                score_idx = self.config.score_types.index(score_type)
                lambda_p = trace.posterior["lambda_player"].values[..., score_idx]
                beta = sigma[..., None] * lambda_p[..., None] * beta_raw
            else:
                # Default to tries (index 0) if no score_type specified
                lambda_p = trace.posterior["lambda_player"].values[..., 0]
                beta = sigma[..., None] * lambda_p[..., None] * beta_raw
        else:
            # Single score type model
            beta = trace.posterior["beta_player"].values

        # Compute summary statistics
        beta_mean = beta.mean(axis=(0, 1))
        beta_std = beta.std(axis=(0, 1))
        beta_lower = np.percentile(beta, 2.5, axis=(0, 1))
        beta_upper = np.percentile(beta, 97.5, axis=(0, 1))

        # Build DataFrame
        inv_player_ids = {v: k for k, v in self._player_ids.items()}
        rankings = pd.DataFrame({
            "player": [inv_player_ids[i] for i in range(len(beta_mean))],
            "effect_mean": beta_mean,
            "effect_std": beta_std,
            "effect_lower": beta_lower,
            "effect_upper": beta_upper,
        })

        # Add score counts if dataframe is provided
        if df is not None and score_type:
            score_counts = df.groupby("player_name")[score_type].sum().reset_index()
            score_counts.columns = ["player", "total_scores"]
            rankings = rankings.merge(score_counts, on="player", how="left")
            rankings["total_scores"] = rankings["total_scores"].fillna(0).astype(int)

            # Apply minimum score threshold if specified
            if min_scores is not None:
                rankings = rankings[rankings["total_scores"] >= min_scores]

        return (
            rankings
            .sort_values("effect_mean", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def get_team_rankings(
        self,
        trace=None,
        season: str | None = None,
        score_type: str | None = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Extract team rankings from posterior.

        Args:
            trace: ArviZ InferenceData
            season: Filter to specific season (returns most recent if None)
            score_type: For joint models, which score type's rankings
            top_n: Number of teams to return

        Returns:
            DataFrame with team rankings
        """
        trace = trace or self.trace
        if trace is None:
            raise ValueError("No trace available. Run inference first.")

        # Detect if this is a joint model by checking for gamma_team_season_raw
        is_joint_model = "gamma_team_season_raw" in trace.posterior

        if is_joint_model:
            # Joint model - need to compute effective team effect
            gamma_raw = trace.posterior["gamma_team_season_raw"].values
            sigma = trace.posterior["sigma_team"].values

            if score_type and score_type in self.config.score_types:
                score_idx = self.config.score_types.index(score_type)
                lambda_t = trace.posterior["lambda_team"].values[..., score_idx]
                gamma = sigma[..., None] * lambda_t[..., None] * gamma_raw
            else:
                # Default to tries (index 0) if no score_type specified
                lambda_t = trace.posterior["lambda_team"].values[..., 0]
                gamma = sigma[..., None] * lambda_t[..., None] * gamma_raw
        else:
            # Single score type model
            gamma = trace.posterior["gamma_team_season"].values

        # Compute summary
        gamma_mean = gamma.mean(axis=(0, 1))
        gamma_std = gamma.std(axis=(0, 1))
        gamma_lower = np.percentile(gamma, 2.5, axis=(0, 1))
        gamma_upper = np.percentile(gamma, 97.5, axis=(0, 1))

        # Build DataFrame
        inv_team_season_ids = {v: k for k, v in self._team_season_ids.items()}
        rankings = pd.DataFrame({
            "team": [inv_team_season_ids[i][0] for i in range(len(gamma_mean))],
            "season": [inv_team_season_ids[i][1] for i in range(len(gamma_mean))],
            "effect_mean": gamma_mean,
            "effect_std": gamma_std,
            "effect_lower": gamma_lower,
            "effect_upper": gamma_upper,
        })

        if season:
            rankings = rankings[rankings["season"] == season]

        return (
            rankings
            .sort_values("effect_mean", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def get_defensive_rankings(
        self,
        trace=None,
        season: str | None = None,
        score_type: str | None = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Extract defensive rankings from posterior.

        Higher defensive effect = better defense (reduces opponent scoring more).

        Args:
            trace: ArviZ InferenceData
            season: Filter to specific season (returns most recent if None)
            score_type: For joint models, which score type's rankings
            top_n: Number of teams to return

        Returns:
            DataFrame with defensive rankings
        """
        trace = trace or self.trace
        if trace is None:
            raise ValueError("No trace available. Run inference first.")

        if not self.config.include_defense:
            raise ValueError("Model was not built with defensive effects enabled")

        # Check if this is a joint model
        is_joint_model = "delta_defense_raw" in trace.posterior

        if is_joint_model:
            # Joint model - need to compute effective defensive effect
            delta_raw = trace.posterior["delta_defense_raw"].values
            sigma = trace.posterior["sigma_defense"].values

            if score_type and score_type in self.config.score_types:
                score_idx = self.config.score_types.index(score_type)
                lambda_d = trace.posterior["lambda_defense"].values[..., score_idx]
                delta = sigma[..., None] * lambda_d[..., None] * delta_raw
            else:
                # Default to tries (index 0) if no score_type specified
                lambda_d = trace.posterior["lambda_defense"].values[..., 0]
                delta = sigma[..., None] * lambda_d[..., None] * delta_raw
        else:
            # Single score type model
            delta = trace.posterior["delta_defense"].values

        # Compute summary
        delta_mean = delta.mean(axis=(0, 1))
        delta_std = delta.std(axis=(0, 1))
        delta_lower = np.percentile(delta, 2.5, axis=(0, 1))
        delta_upper = np.percentile(delta, 97.5, axis=(0, 1))

        # Build DataFrame
        inv_team_season_ids = {v: k for k, v in self._team_season_ids.items()}
        rankings = pd.DataFrame({
            "team": [inv_team_season_ids[i][0] for i in range(len(delta_mean))],
            "season": [inv_team_season_ids[i][1] for i in range(len(delta_mean))],
            "defense_mean": delta_mean,
            "defense_std": delta_std,
            "defense_lower": delta_lower,
            "defense_upper": delta_upper,
        })

        if season:
            rankings = rankings[rankings["season"] == season]

        return (
            rankings
            .sort_values("defense_mean", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    @property
    def player_ids(self) -> dict[str, int]:
        return self._player_ids.copy()

    @property
    def team_ids(self) -> dict[str, int]:
        return self._team_ids.copy()
