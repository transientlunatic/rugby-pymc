"""
Season prediction functionality for rugby competitions.

Supports:
- Predicting all remaining matches in a season
- Monte Carlo simulation of final standings
- Playoff qualification probabilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rugby_ranking.model.predictions import MatchPredictor, MatchPrediction
from rugby_ranking.model.league_table import LeagueTable, BonusPointRules


@dataclass
class SeasonPrediction:
    """Predicted season outcome."""

    current_standings: pd.DataFrame
    predicted_standings: pd.DataFrame
    position_probabilities: pd.DataFrame | None = None  # P(team finishes in position k)
    playoff_probabilities: pd.DataFrame | None = None  # P(team makes playoffs)
    remaining_fixtures: pd.DataFrame | None = None
    simulation_samples: 'SeasonSimulationSamples | None' = None


@dataclass
class SeasonSimulationSamples:
    """Detailed samples from season simulations."""

    teams: list[str]
    fixtures: list[dict]
    game_outcomes: np.ndarray  # shape: (n_simulations, n_games), values: 0=home_win,1=draw,2=away_win
    final_positions: np.ndarray  # shape: (n_simulations, n_teams), 1-based positions


class SeasonPredictor:
    """
    Predict season outcomes using match predictions and league table simulation.

    Usage:
        >>> predictor = SeasonPredictor(model, trace, competition="urc")
        >>> season_pred = predictor.predict_season(
        ...     played_matches=played_df,
        ...     remaining_fixtures=fixtures_df,
        ...     season="2024-2025",
        ...     n_simulations=1000
        ... )
        >>> print(season_pred.predicted_standings)
        >>> print(season_pred.playoff_probabilities)
    """

    def __init__(
        self,
        match_predictor: MatchPredictor,
        competition: BonusPointRules | str = BonusPointRules.URC,
        playoff_spots: int = 8,
    ):
        """
        Initialize season predictor.

        Args:
            match_predictor: Fitted MatchPredictor for match outcome prediction
            competition: Competition rules for bonus points
            playoff_spots: Number of teams that qualify for playoffs
        """
        self.match_predictor = match_predictor
        self.league_table = LeagueTable(bonus_rules=competition)
        self.playoff_spots = playoff_spots

    def predict_season(
        self,
        played_matches: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        season: str,
        n_simulations: int = 1000,
        return_samples: bool = False,
    ) -> SeasonPrediction:
        """
        Predict final season standings using Monte Carlo simulation.

        Args:
            played_matches: Completed matches (one row per team per match) with columns:
                - team, opponent, score, opponent_score, tries, is_home
            remaining_fixtures: Upcoming matches with columns:
                - home_team, away_team, date (optional)
            season: Season string (e.g., "2024-2025")
            n_simulations: Number of Monte Carlo iterations
            return_samples: If True, return full simulation samples

        Returns:
            SeasonPrediction with current and predicted standings, probabilities
        """
        # Compute current standings
        current_standings = self.league_table.compute_standings(played_matches)

        # Ensure all teams from remaining fixtures are in standings
        # (even if they haven't played yet)
        if len(remaining_fixtures) > 0:
            import pandas as pd
            # Get all teams from fixtures
            fixture_teams = set()
            if 'home_team' in remaining_fixtures.columns:
                fixture_teams.update(remaining_fixtures['home_team'].unique())
            if 'away_team' in remaining_fixtures.columns:
                fixture_teams.update(remaining_fixtures['away_team'].unique())

            # Find teams not yet in standings
            existing_teams = set(current_standings['team'].values) if len(current_standings) > 0 else set()
            missing_teams = fixture_teams - existing_teams

            # Add missing teams with 0 points
            if missing_teams:
                missing_rows = pd.DataFrame([
                    {
                        'team': team,
                        'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
                        'points_for': 0, 'points_against': 0, 'points_diff': 0,
                        'tries_for': 0, 'tries_against': 0,
                        'try_bonus': 0, 'losing_bonus': 0, 'bonus_points': 0,
                        'match_points': 0, 'total_points': 0, 'position': 0
                    }
                    for team in sorted(missing_teams)
                ])
                current_standings = pd.concat([current_standings, missing_rows], ignore_index=True)

        # Predict remaining matches (get expected scores)
        remaining_predictions = self._predict_remaining_matches(
            remaining_fixtures, season
        )

        # Run Monte Carlo simulation
        final_standings, position_probs, playoff_probs, simulation_samples = self._simulate_season(
            current_standings=current_standings,
            remaining_predictions=remaining_predictions,
            n_simulations=n_simulations,
            return_samples=return_samples,
        )

        return SeasonPrediction(
            current_standings=current_standings,
            predicted_standings=final_standings,
            position_probabilities=position_probs,
            playoff_probabilities=playoff_probs,
            remaining_fixtures=remaining_predictions,
            simulation_samples=simulation_samples,
        )

    def _predict_remaining_matches(
        self,
        fixtures: pd.DataFrame,
        season: str,
    ) -> pd.DataFrame:
        """
        Predict outcomes for all remaining fixtures.

        Args:
            fixtures: DataFrame with home_team, away_team columns
            season: Season string

        Returns:
            DataFrame with predicted scores and probabilities
        """
        predictions = []

        for _, fixture in fixtures.iterrows():
            try:
                pred = self.match_predictor.predict_teams_only(
                    home_team=fixture['home_team'],
                    away_team=fixture['away_team'],
                    season=season,
                    n_samples=1000,
                )

                # Estimate tries from score samples
                # Use the samples to get tries distribution
                home_tries_mean = pred.home.samples.mean() / 7 * 1.4  # Rough estimate
                away_tries_mean = pred.away.samples.mean() / 7 * 1.4

                predictions.append({
                    'home_team': fixture['home_team'],
                    'away_team': fixture['away_team'],
                    'date': fixture.get('date', None),
                    'home_score_pred': pred.home.mean,
                    'away_score_pred': pred.away.mean,
                    'home_tries_pred': home_tries_mean,
                    'away_tries_pred': away_tries_mean,
                    'home_win_prob': pred.home_win_prob,
                    'away_win_prob': pred.away_win_prob,
                    'draw_prob': pred.draw_prob,
                    'home_score_samples': pred.home.samples,
                    'away_score_samples': pred.away.samples,
                })
            except ValueError as e:
                print(f"Skipping fixture {fixture['home_team']} vs {fixture['away_team']}: {e}")
                continue

        return pd.DataFrame(predictions)

    def _simulate_season(
        self,
        current_standings: pd.DataFrame,
        remaining_predictions: pd.DataFrame,
        n_simulations: int,
        return_samples: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SeasonSimulationSamples | None]:
        """
        Monte Carlo simulation of remaining season.

        Args:
            current_standings: Current league table
            remaining_predictions: Predicted match outcomes with samples
            n_simulations: Number of simulations

        Returns:
            (final_standings, position_probabilities, playoff_probabilities)
        """
        teams = current_standings['team'].values
        n_teams = len(teams)

        # Initialize simulation results
        # Store final position for each team in each simulation
        position_counts = np.zeros((n_teams, n_teams), dtype=int)
        playoff_counts = np.zeros(n_teams, dtype=int)

        # Store cumulative stats for expected final standings
        total_points_sum = np.zeros(n_teams)
        total_won_sum = np.zeros(n_teams)
        total_diff_sum = np.zeros(n_teams)

        # Create team index mapping
        team_to_idx = {team: i for i, team in enumerate(teams)}

        # Optional detailed samples for downstream analysis
        fixtures = []
        game_outcomes = None
        final_positions = None
        if return_samples:
            fixtures = [
                {
                    'home_team': row['home_team'],
                    'away_team': row['away_team'],
                    'date': row.get('date', None),
                }
                for _, row in remaining_predictions.iterrows()
            ]
            game_outcomes = np.zeros((n_simulations, len(remaining_predictions)), dtype=np.int8)
            final_positions = np.zeros((n_simulations, n_teams), dtype=np.int16)

        # Run simulations
        for sim in range(n_simulations):
            # Start with current standings
            sim_standings = current_standings.copy()

            # Simulate each remaining match
            for match_idx, (_, match) in enumerate(remaining_predictions.iterrows()):
                # Sample from predicted score distributions
                home_score = np.random.choice(match['home_score_samples'])
                away_score = np.random.choice(match['away_score_samples'])

                if return_samples:
                    if home_score > away_score:
                        game_outcomes[sim, match_idx] = 0
                    elif home_score < away_score:
                        game_outcomes[sim, match_idx] = 2
                    else:
                        game_outcomes[sim, match_idx] = 1

                # Estimate tries (rough approximation from score)
                home_tries = int(np.round(home_score / 7 * 1.4))
                away_tries = int(np.round(away_score / 7 * 1.4))

                # Create match results
                simulated_matches = pd.DataFrame([
                    {
                        'team': match['home_team'],
                        'opponent': match['away_team'],
                        'score': home_score,
                        'opponent_score': away_score,
                        'tries': home_tries,
                        'opponent_tries': away_tries,
                        'is_home': True,
                    },
                    {
                        'team': match['away_team'],
                        'opponent': match['home_team'],
                        'score': away_score,
                        'opponent_score': home_score,
                        'tries': away_tries,
                        'opponent_tries': home_tries,
                        'is_home': False,
                    }
                ])

                # Update standings with this match
                match_standings = self.league_table.compute_standings(
                    simulated_matches,
                    opponent_tries_col='opponent_tries'
                )

                # Merge with existing standings
                for _, row in match_standings.iterrows():
                    team = row['team']
                    if team in sim_standings['team'].values:
                        idx = sim_standings[sim_standings['team'] == team].index[0]
                        sim_standings.loc[idx, 'played'] += row['played']
                        sim_standings.loc[idx, 'won'] += row['won']
                        sim_standings.loc[idx, 'drawn'] += row['drawn']
                        sim_standings.loc[idx, 'lost'] += row['lost']
                        sim_standings.loc[idx, 'points_for'] += row['points_for']
                        sim_standings.loc[idx, 'points_against'] += row['points_against']
                        sim_standings.loc[idx, 'tries_for'] += row['tries_for']
                        sim_standings.loc[idx, 'tries_against'] += row['tries_against']
                        sim_standings.loc[idx, 'try_bonus'] += row['try_bonus']
                        sim_standings.loc[idx, 'losing_bonus'] += row['losing_bonus']

            # Recalculate final standings
            sim_standings['bonus_points'] = (
                sim_standings['try_bonus'] + sim_standings['losing_bonus']
            )
            sim_standings['match_points'] = (
                sim_standings['won'] * 4 + sim_standings['drawn'] * 2
            )
            sim_standings['total_points'] = (
                sim_standings['match_points'] + sim_standings['bonus_points']
            )
            sim_standings['points_diff'] = (
                sim_standings['points_for'] - sim_standings['points_against']
            )

            # Sort to get final positions
            sim_standings = sim_standings.sort_values(
                by=['total_points', 'points_diff', 'tries_for'],
                ascending=[False, False, False]
            ).reset_index(drop=True)

            sim_standings['position'] = range(1, len(sim_standings) + 1)

            # Record results for this simulation
            for _, row in sim_standings.iterrows():
                team = row['team']
                position = row['position'] - 1  # 0-indexed
                team_idx = team_to_idx[team]

                position_counts[team_idx, position] += 1
                total_points_sum[team_idx] += row['total_points']
                total_won_sum[team_idx] += row['won']
                total_diff_sum[team_idx] += row['points_diff']

                # Check if team makes playoffs
                if row['position'] <= self.playoff_spots:
                    playoff_counts[team_idx] += 1

                if return_samples:
                    final_positions[sim, team_idx] = row['position']

        # Compute expected final standings (mean across simulations)
        # Round to integers for better readability
        expected_standings = pd.DataFrame({
            'team': teams,
            'expected_points': (total_points_sum / n_simulations).round(0).astype(int),
            'expected_wins': (total_won_sum / n_simulations).round(0).astype(int),
            'expected_diff': (total_diff_sum / n_simulations).round(0).astype(int),
        })

        # Sort by expected points
        expected_standings = expected_standings.sort_values(
            by=['expected_points', 'expected_diff'],
            ascending=[False, False]
        ).reset_index(drop=True)
        expected_standings['predicted_position'] = range(1, len(expected_standings) + 1)

        # Compute position probabilities
        position_probs = pd.DataFrame(
            position_counts / n_simulations,
            index=teams,
            columns=[f'P(pos {i+1})' for i in range(n_teams)]
        )

        # Add most likely position
        position_probs['most_likely_position'] = position_probs.values.argmax(axis=1) + 1

        # Compute playoff probabilities
        playoff_probs = pd.DataFrame({
            'team': teams,
            'playoff_probability': playoff_counts / n_simulations,
        }).sort_values('playoff_probability', ascending=False).reset_index(drop=True)

        simulation_samples = None
        if return_samples:
            simulation_samples = SeasonSimulationSamples(
                teams=list(teams),
                fixtures=fixtures,
                game_outcomes=game_outcomes,
                final_positions=final_positions,
            )

        return expected_standings, position_probs, playoff_probs, simulation_samples

    def format_predictions(self, season_pred: SeasonPrediction) -> str:
        """
        Format season predictions as human-readable text.

        Args:
            season_pred: Season prediction results

        Returns:
            Formatted string
        """
        lines = []

        lines.append("=" * 70)
        lines.append("SEASON PREDICTION")
        lines.append("=" * 70)

        # Current standings
        lines.append("\nCURRENT STANDINGS:")
        lines.append("-" * 70)
        for _, row in season_pred.current_standings.head(10).iterrows():
            lines.append(
                f"{row['position']:2d}. {row['team']:<20} "
                f"P:{row['played']:2d} W:{row['won']:2d} "
                f"Pts:{row['total_points']:3d}"
            )

        # Predicted final standings
        lines.append("\n\nPREDICTED FINAL STANDINGS:")
        lines.append("-" * 70)
        for _, row in season_pred.predicted_standings.head(10).iterrows():
            lines.append(
                f"{row['predicted_position']:2d}. {row['team']:<20} "
                f"Pts:{row['expected_points']:.1f} "
                f"Diff:{row['expected_diff']:+.1f}"
            )

        # Playoff probabilities
        if season_pred.playoff_probabilities is not None:
            lines.append(f"\n\nPLAYOFF PROBABILITIES (Top {self.playoff_spots}):")
            lines.append("-" * 70)
            for _, row in season_pred.playoff_probabilities.head(self.playoff_spots + 2).iterrows():
                lines.append(
                    f"{row['team']:<20} {row['playoff_probability']:.1%}"
                )

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)
