"""
League table computation for rugby competitions.

Supports:
- Computing standings from match results
- Bonus points calculation (URC/Premiership/Top14 rules)
- Current and projected standings
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd
import numpy as np


class BonusPointRules(Enum):
    """Different bonus point systems used by major competitions."""

    URC = "urc"  # United Rugby Championship
    CELTIC = "celtic"  # Celtic League (now URC) - same rules as URC
    PREMIERSHIP = "premiership"  # English Premiership
    TOP14 = "top14"  # French Top 14
    SIX_NATIONS = "six-nations"  # Six Nations Championship (bonus points from 2017)
    EURO_CHAMPIONS = "euro-champions"  # European Rugby Champions Cup
    EURO_CHALLENGE = "euro-challenge"  # European Rugby Challenge Cup


@dataclass
class BonusPointConfig:
    """Configuration for bonus point calculation."""

    try_bonus_threshold: int = 4  # Number of tries needed for bonus point
    try_bonus_points: int = 1  # Points awarded for try bonus
    try_bonus_relative: bool = False  # If True, threshold is relative to opponent tries
    losing_bonus_margin: int = 7  # Maximum margin to get losing bonus
    losing_bonus_points: int = 1  # Points awarded for losing bonus
    win_points: int = 4  # Points for a win
    draw_points: int = 2  # Points for a draw
    loss_points: int = 0  # Points for a loss (excluding bonuses)

    @classmethod
    def from_competition(cls, competition: BonusPointRules | str) -> BonusPointConfig:
        """Get bonus point configuration for a competition."""
        if isinstance(competition, str):
            competition = BonusPointRules(competition.lower())

        if competition == BonusPointRules.URC:
            return cls(
                try_bonus_threshold=4,
                try_bonus_points=1,
                try_bonus_relative=False,
                losing_bonus_margin=7,
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        elif competition == BonusPointRules.PREMIERSHIP:
            return cls(
                try_bonus_threshold=4,
                try_bonus_points=1,
                try_bonus_relative=False,
                losing_bonus_margin=7,
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        elif competition == BonusPointRules.TOP14:
            # Top14 has different rules
            return cls(
                try_bonus_threshold=3,  # 3 tries more than opponent
                try_bonus_points=1,
                try_bonus_relative=True,  # Relative to opponent
                losing_bonus_margin=5,  # Lose by 5 or less
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        elif competition == BonusPointRules.SIX_NATIONS:
            # Six Nations (introduced bonus points in 2017)
            # Same rules as URC/Premiership
            return cls(
                try_bonus_threshold=4,
                try_bonus_points=1,
                try_bonus_relative=False,
                losing_bonus_margin=7,
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        elif competition == BonusPointRules.CELTIC:
            # Celtic League (now URC) - same rules as URC
            return cls(
                try_bonus_threshold=4,
                try_bonus_points=1,
                try_bonus_relative=False,
                losing_bonus_margin=7,
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        elif competition == BonusPointRules.EURO_CHAMPIONS:
            # European Champions Cup - same rules as URC/Premiership
            return cls(
                try_bonus_threshold=4,
                try_bonus_points=1,
                try_bonus_relative=False,
                losing_bonus_margin=7,
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        elif competition == BonusPointRules.EURO_CHALLENGE:
            # European Challenge Cup - same rules as URC/Premiership
            return cls(
                try_bonus_threshold=4,
                try_bonus_points=1,
                try_bonus_relative=False,
                losing_bonus_margin=7,
                losing_bonus_points=1,
                win_points=4,
                draw_points=2,
                loss_points=0,
            )
        else:
            raise ValueError(f"Unknown competition: {competition}")


class LeagueTable:
    """
    Compute league standings from match results.

    Usage:
        >>> table = LeagueTable(bonus_rules=BonusPointRules.URC)
        >>> standings = table.compute_standings(matches_df)
        >>> print(standings[['team', 'played', 'won', 'total_points']])
    """

    def __init__(
        self,
        bonus_rules: BonusPointRules | str = BonusPointRules.URC,
        config: BonusPointConfig | None = None,
    ):
        """
        Initialize league table calculator.

        Args:
            bonus_rules: Competition to use for bonus point rules
            config: Custom bonus point configuration (overrides bonus_rules)
        """
        if config is not None:
            self.config = config
        else:
            self.config = BonusPointConfig.from_competition(bonus_rules)

    def compute_standings(
        self,
        matches: pd.DataFrame,
        team_col: str = "team",
        opponent_col: str = "opponent",
        score_col: str = "score",
        opponent_score_col: str = "opponent_score",
        tries_col: str = "tries",
        opponent_tries_col: str | None = None,
        home_col: str = "is_home",
    ) -> pd.DataFrame:
        """
        Compute league standings from match results.

        Args:
            matches: DataFrame with match results (one row per team per match)
            team_col: Column name for team
            opponent_col: Column name for opponent
            score_col: Column name for team score
            opponent_score_col: Column name for opponent score
            tries_col: Column name for tries scored
            opponent_tries_col: Column name for opponent tries (optional)
            home_col: Column name for home/away indicator

        Returns:
            DataFrame with columns:
                - team: Team name
                - played: Matches played
                - won: Matches won
                - drawn: Matches drawn
                - lost: Matches lost
                - points_for: Total points scored
                - points_against: Total points conceded
                - points_diff: Points difference
                - tries_for: Total tries scored
                - tries_against: Total tries conceded (if opponent_tries_col provided)
                - try_bonus: Try bonus points earned
                - losing_bonus: Losing bonus points earned
                - bonus_points: Total bonus points
                - match_points: Points from wins/draws/losses
                - total_points: Total league points
                - position: League position (1 = first)
        """
        # Handle empty matches - return empty standings table with proper columns
        if len(matches) == 0:
            return pd.DataFrame(columns=[
                'team', 'played', 'won', 'drawn', 'lost',
                'points_for', 'points_against', 'points_diff',
                'tries_for', 'tries_against',
                'try_bonus', 'losing_bonus', 'bonus_points',
                'match_points', 'total_points', 'position'
            ])

        # Validate required columns
        required = [team_col, opponent_col, score_col, opponent_score_col, tries_col]
        missing = [col for col in required if col not in matches.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Initialize standings dict
        teams = matches[team_col].unique()
        standings = {team: {
            'team': team,
            'played': 0,
            'won': 0,
            'drawn': 0,
            'lost': 0,
            'points_for': 0,
            'points_against': 0,
            'tries_for': 0,
            'tries_against': 0,
            'try_bonus': 0,
            'losing_bonus': 0,
        } for team in teams}

        # Process each match
        for _, match in matches.iterrows():
            team = match[team_col]
            score = match[score_col]
            opponent_score = match[opponent_score_col]
            tries = match[tries_col]

            # Update basic stats
            standings[team]['played'] += 1
            standings[team]['points_for'] += score
            standings[team]['points_against'] += opponent_score
            standings[team]['tries_for'] += tries

            opponent_tries = None
            if opponent_tries_col and opponent_tries_col in matches.columns:
                opponent_tries = match[opponent_tries_col]
                standings[team]['tries_against'] += opponent_tries

            # Determine result
            if score > opponent_score:
                # Win
                standings[team]['won'] += 1
            elif score == opponent_score:
                # Draw
                standings[team]['drawn'] += 1
            else:
                # Loss
                standings[team]['lost'] += 1

                # Check for losing bonus
                margin = opponent_score - score
                if margin <= self.config.losing_bonus_margin:
                    standings[team]['losing_bonus'] += self.config.losing_bonus_points

            # Check for try bonus
            if self.config.try_bonus_relative:
                # Relative to opponent (e.g., Top14: score 3+ more tries than opponent)
                if opponent_tries is not None:
                    try_diff = tries - opponent_tries
                    if try_diff >= self.config.try_bonus_threshold:
                        standings[team]['try_bonus'] += self.config.try_bonus_points
            else:
                # Absolute threshold (e.g., URC/Premiership: score 4+ tries)
                if tries >= self.config.try_bonus_threshold:
                    standings[team]['try_bonus'] += self.config.try_bonus_points

        # Convert to DataFrame and compute final columns
        df = pd.DataFrame(list(standings.values()))

        # Compute points
        df['bonus_points'] = df['try_bonus'] + df['losing_bonus']
        df['match_points'] = (
            df['won'] * self.config.win_points +
            df['drawn'] * self.config.draw_points +
            df['lost'] * self.config.loss_points
        )
        df['total_points'] = df['match_points'] + df['bonus_points']
        df['points_diff'] = df['points_for'] - df['points_against']

        # Sort by total points (desc), then points difference (desc), then tries scored (desc)
        df = df.sort_values(
            by=['total_points', 'points_diff', 'tries_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        # Add position
        df['position'] = range(1, len(df) + 1)

        # Reorder columns for readability
        columns = [
            'position', 'team', 'played', 'won', 'drawn', 'lost',
            'points_for', 'points_against', 'points_diff',
            'tries_for', 'tries_against',
            'try_bonus', 'losing_bonus', 'bonus_points',
            'match_points', 'total_points'
        ]

        return df[columns]

    def compute_standings_from_predictions(
        self,
        current_standings: pd.DataFrame,
        predicted_matches: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Project standings by adding predicted match results to current standings.

        Args:
            current_standings: Current league table (from compute_standings)
            predicted_matches: DataFrame with predicted results containing:
                - home_team, away_team
                - home_score_pred, away_score_pred
                - home_tries_pred, away_tries_pred (optional)

        Returns:
            Projected standings DataFrame
        """
        # Convert predicted matches to the format expected by compute_standings
        # Create two rows per match (one for each team)
        home_matches = pd.DataFrame({
            'team': predicted_matches['home_team'],
            'opponent': predicted_matches['away_team'],
            'score': predicted_matches['home_score_pred'].round().astype(int),
            'opponent_score': predicted_matches['away_score_pred'].round().astype(int),
            'tries': predicted_matches.get('home_tries_pred',
                predicted_matches['home_score_pred'] / 7 * 1.4).round().astype(int),
            'is_home': True,
        })

        away_matches = pd.DataFrame({
            'team': predicted_matches['away_team'],
            'opponent': predicted_matches['home_team'],
            'score': predicted_matches['away_score_pred'].round().astype(int),
            'opponent_score': predicted_matches['home_score_pred'].round().astype(int),
            'tries': predicted_matches.get('away_tries_pred',
                predicted_matches['away_score_pred'] / 7 * 1.4).round().astype(int),
            'is_home': False,
        })

        predicted_results = pd.concat([home_matches, away_matches], ignore_index=True)

        # Compute standings from predicted matches
        predicted_standings = self.compute_standings(predicted_results)

        # Merge with current standings
        # For simplicity, we'll add the new stats to existing stats
        # In reality, you'd want to combine them more carefully

        merged = current_standings.set_index('team').add(
            predicted_standings.set_index('team'),
            fill_value=0
        ).reset_index()

        # Recalculate total points and position
        merged['total_points'] = merged['match_points'] + merged['bonus_points']
        merged['points_diff'] = merged['points_for'] - merged['points_against']

        merged = merged.sort_values(
            by=['total_points', 'points_diff', 'tries_for'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        merged['position'] = range(1, len(merged) + 1)

        return merged

    def simulate_season(
        self,
        current_standings: pd.DataFrame,
        predictions: pd.DataFrame,
        n_simulations: int = 1000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Monte Carlo simulation of remaining season.

        Args:
            current_standings: Current league table
            predictions: Match predictions with samples (from MatchPredictor)
            n_simulations: Number of Monte Carlo iterations

        Returns:
            (mean_standings, position_probabilities)
            - mean_standings: Expected final standings
            - position_probabilities: P(team finishes in position k) for each team
        """
        # This is a placeholder for full Monte Carlo implementation
        # For now, return expected standings
        raise NotImplementedError("Monte Carlo simulation not yet implemented")


def format_table(
    standings: pd.DataFrame,
    max_teams: int | None = None,
    highlight_positions: list[int] | None = None,
) -> str:
    """
    Format standings as a human-readable table.

    Args:
        standings: Standings DataFrame
        max_teams: Maximum number of teams to display
        highlight_positions: Positions to highlight (e.g., playoff spots)

    Returns:
        Formatted table string
    """
    df = standings.copy()

    if max_teams:
        df = df.head(max_teams)

    # Format column names
    col_names = {
        'position': 'Pos',
        'team': 'Team',
        'played': 'P',
        'won': 'W',
        'drawn': 'D',
        'lost': 'L',
        'points_for': 'F',
        'points_against': 'A',
        'points_diff': '+/-',
        'tries_for': 'TF',
        'bonus_points': 'BP',
        'total_points': 'Pts',
    }

    display_cols = ['position', 'team', 'played', 'won', 'drawn', 'lost',
                   'points_for', 'points_against', 'points_diff',
                   'tries_for', 'bonus_points', 'total_points']

    df_display = df[display_cols].rename(columns=col_names)

    # Format as string table
    lines = []

    # Header
    header = " | ".join([
        f"{col_names[col]:>4}" if col != 'team' else f"{col_names[col]:<20}"
        for col in display_cols
    ])
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for _, row in df_display.iterrows():
        pos = int(row['Pos'])
        highlight = highlight_positions and pos in highlight_positions
        prefix = "* " if highlight else "  "

        line = prefix + " | ".join([
            f"{row[col_names[col]]:>4}" if col != 'team' else f"{row[col_names[col]]:<20}"
            for col in display_cols
        ])
        lines.append(line)

    return "\n".join(lines)
