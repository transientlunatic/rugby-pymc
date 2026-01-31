"""
Paths to Victory analysis for knockout tournaments.

Extends PathsAnalyzer to show paths to reaching specific rounds
or winning tournaments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal
from collections import defaultdict

import numpy as np
import pandas as pd

from rugby_ranking.model.paths_to_victory import PathsAnalyzer, PathsOutput, Condition, ScenarioCluster
from rugby_ranking.model.bracket_predictor import BracketPrediction, BracketPredictor
from rugby_ranking.model.bracket import BracketStructure
from rugby_ranking.model.predictions import MatchPredictor


class TournamentPathsAnalyzer:
    """
    Analyze paths for teams to reach tournament goals (semifinals, finals, championship).

    Shows:
    - Direct path: Team's own matches to win
    - Indirect influences: Other matches affecting bracket matchups
    - Favorable draws: Probability of easier matchup paths

    Usage:
        >>> analyzer = TournamentPathsAnalyzer(bracket_prediction, match_predictor)
        >>> paths = analyzer.analyze_tournament_paths(
        ...     team="Leinster",
        ...     target="champion"
        ... )
        >>> print(paths.narrative)
    """

    def __init__(
        self,
        bracket_prediction: BracketPrediction,
        match_predictor: MatchPredictor,
    ):
        """
        Initialize tournament paths analyzer.

        Args:
            bracket_prediction: BracketPrediction with advancement probabilities
            match_predictor: MatchPredictor for calculating win probabilities
        """
        self.prediction = bracket_prediction
        self.bracket = bracket_prediction.bracket
        self.match_predictor = match_predictor

    def analyze_tournament_paths(
        self,
        team: str,
        target: Literal["quarterfinal", "semifinal", "final", "champion"] = "champion",
    ) -> PathsOutput:
        """
        Analyze paths for team to reach target goal.

        Args:
            team: Team to analyze
            target: Goal to reach (quarterfinal, semifinal, final, or champion)

        Returns:
            PathsOutput with conditions, critical matches, and narrative
        """
        # Get team's probability of reaching target
        probability = self._get_target_probability(team, target)

        # Identify critical matches (team's own matches)
        critical_matches = self._identify_critical_matches(team, target)

        # Identify likely opponents in each round
        likely_opponents = self._identify_likely_opponents(team, target)

        # Analyze favorable bracket draws
        draw_favorability = self._analyze_draw_favorability(team, target)

        # Extract conditions
        conditions = self._extract_conditions(team, target, critical_matches, likely_opponents)

        # Create narrative
        narrative = self._generate_narrative(
            team=team,
            target=target,
            probability=probability,
            critical_matches=critical_matches,
            likely_opponents=likely_opponents,
            draw_favorability=draw_favorability,
        )

        return PathsOutput(
            team=team,
            target_position=self._target_to_position(target),
            probability=probability,
            method="mcmc",  # Bracket simulation is Monte Carlo based
            conditions=conditions,
            critical_games=critical_matches,
            scenario_clusters=[],  # TODO: Implement scenario clustering
            narrative=narrative,
        )

    def _get_target_probability(self, team: str, target: str) -> float:
        """Get probability of team reaching target."""
        adv_probs = self.prediction.advancement_probs

        team_row = adv_probs[adv_probs["team"] == team]
        if len(team_row) == 0:
            return 0.0

        if target == "champion":
            return team_row.iloc[0].get("champion_prob", 0.0)
        else:
            col_name = f"{target}_prob"
            return team_row.iloc[0].get(col_name, 0.0)

    def _identify_critical_matches(
        self,
        team: str,
        target: str,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identify critical matches team must win to reach target.

        Returns:
            List of ((home, away), importance) tuples
        """
        critical = []

        # Find team's matches in each round
        target_rounds = self._get_rounds_to_target(target)

        for round_name in target_rounds:
            # Find team's match in this round (if deterministic)
            team_match = self._find_team_match_in_round(team, round_name)

            if team_match:
                # Calculate importance (must win to advance)
                importance = 1.0  # Binary: must win
                critical.append(((team_match["opponent"], team), importance))

        return critical

    def _get_rounds_to_target(self, target: str) -> List[str]:
        """Get list of rounds team must pass through to reach target."""
        all_rounds = self.bracket.rounds

        if target == "champion":
            return all_rounds
        else:
            # Return rounds up to and including target
            if target in all_rounds:
                target_idx = all_rounds.index(target)
                return all_rounds[: target_idx + 1]
            return []

    def _find_team_match_in_round(
        self,
        team: str,
        round_name: str,
    ) -> Dict | None:
        """
        Find team's match in a specific round.

        Returns:
            Dict with opponent and match details, or None if not deterministic
        """
        round_matches = self.bracket.get_round_matches(round_name)

        for match in round_matches:
            if match.is_determined():
                if match.home_team == team:
                    return {
                        "opponent": match.away_team,
                        "is_home": True,
                        "match_id": match.id,
                    }
                elif match.away_team == team:
                    return {
                        "opponent": match.home_team,
                        "is_home": False,
                        "match_id": match.id,
                    }

        return None

    def _identify_likely_opponents(
        self,
        team: str,
        target: str,
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Identify likely opponents in each round.

        Returns:
            Dict mapping round_name to [(opponent, matchup_prob, win_prob), ...]
        """
        opponents = {}
        target_rounds = self._get_rounds_to_target(target)

        # This would require analyzing simulation results to see
        # which teams team played against in each round
        # For now, return placeholder

        # TODO: Implement by analyzing prediction._simulation_results
        return opponents

    def _analyze_draw_favorability(
        self,
        team: str,
        target: str,
    ) -> Dict[str, float]:
        """
        Analyze favorability of potential bracket draws.

        Returns:
            Dict with draw favorability metrics
        """
        # Calculate expected win probability across potential matchups
        # weighted by matchup probability

        # TODO: Implement
        return {
            "expected_difficulty": 0.5,  # Placeholder
            "favorable_draw_prob": 0.3,  # P(getting easier than average matchups)
        }

    def _extract_conditions(
        self,
        team: str,
        target: str,
        critical_matches: List,
        likely_opponents: Dict,
    ) -> List[Condition]:
        """Extract conditions affecting team's path to target."""
        conditions = []

        # Add conditions for team's own matches
        for (home, away), importance in critical_matches:
            opponent = home if away == team else away
            is_home = (home == team)

            # Get win probability
            prediction = self.match_predictor.predict_match(
                home_team=home,
                away_team=away,
                is_home=is_home,
            )

            win_prob = prediction.home_win_prob if is_home else (1 - prediction.home_win_prob)

            conditions.append(
                Condition(
                    game=(home, away),
                    outcome="home_win" if is_home else "away_win",
                    frequency=1.0,  # Must happen
                    conditional_prob=win_prob,
                    importance=importance,
                    team_controls=True,
                )
            )

        return conditions

    def _generate_narrative(
        self,
        team: str,
        target: str,
        probability: float,
        critical_matches: List,
        likely_opponents: Dict,
        draw_favorability: Dict,
    ) -> str:
        """Generate human-readable narrative for tournament path."""
        lines = []

        # Header
        target_text = {
            "quarterfinal": "reach the quarterfinals",
            "semifinal": "reach the semifinals",
            "final": "reach the final",
            "champion": "win the tournament",
        }

        lines.append(
            f"{team} can {target_text[target]} with {probability:.1%} probability.\n"
        )

        # Path through rounds
        if critical_matches:
            lines.append("Path to victory:")

            for (home, away), importance in critical_matches:
                opponent = home if away == team else away
                is_home = (home == team)

                # Get win probability
                prediction = self.match_predictor.predict_match(
                    home_team=home,
                    away_team=away,
                    is_home=is_home,
                )

                win_prob = prediction.home_win_prob if is_home else (1 - prediction.home_win_prob)

                location = "at home" if is_home else "away"
                lines.append(f"  Must beat {opponent} {location} ({win_prob:.1%} likely)")

        # Likely opponents (if available)
        if likely_opponents:
            lines.append("\nLikely opponents:")
            for round_name, opponents in likely_opponents.items():
                lines.append(f"  {round_name.capitalize()}:")
                for opponent, matchup_prob, win_prob in opponents[:3]:  # Top 3
                    lines.append(
                        f"    - {opponent} ({matchup_prob:.0%} chance): "
                        f"Win probability {win_prob:.1%}"
                    )

        # Draw favorability
        if draw_favorability:
            expected_diff = draw_favorability.get("expected_difficulty", 0.5)
            if expected_diff < 0.45:
                lines.append(
                    f"\n{team} has a relatively favorable bracket draw "
                    f"(expected difficulty: {expected_diff:.1%})"
                )
            elif expected_diff > 0.55:
                lines.append(
                    f"\n{team} faces a challenging bracket draw "
                    f"(expected difficulty: {expected_diff:.1%})"
                )

        return "\n".join(lines)

    def _target_to_position(self, target: str) -> int:
        """Convert target round to numeric position."""
        mapping = {
            "champion": 1,
            "final": 2,
            "semifinal": 4,
            "quarterfinal": 8,
        }
        return mapping.get(target, 1)

    def compare_tournament_impacts(
        self,
        team: str,
        matches_to_analyze: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Analyze impact of other matches on team's tournament chances.

        Args:
            team: Team to analyze
            matches_to_analyze: List of (home, away) matches to analyze

        Returns:
            DataFrame with columns: match, outcome, impact_on_champion_prob
        """
        # For each match, calculate:
        # - P(team wins tournament | home wins match)
        # - P(team wins tournament | away wins match)
        # - Impact = difference

        # This requires conditional probability analysis on simulation results

        # TODO: Implement
        return pd.DataFrame()

    def get_optimal_matchup_preferences(
        self,
        team: str,
    ) -> pd.DataFrame:
        """
        Get team's preferred opponents for each potential matchup.

        Returns:
            DataFrame ranking potential opponents by win probability
        """
        # Get all potential opponents
        adv_probs = self.prediction.advancement_probs

        # For each opponent, calculate win probability
        matchups = []

        for _, row in adv_probs.iterrows():
            opponent = row["team"]

            if opponent == team:
                continue

            # Calculate win probability (neutral venue)
            prediction = self.match_predictor.predict_match(
                home_team=team,
                away_team=opponent,
                is_home=False,  # Neutral
            )

            matchups.append(
                {
                    "opponent": opponent,
                    "win_probability": prediction.home_win_prob,
                    "opponent_strength": row.get("champion_prob", 0),
                }
            )

        df = pd.DataFrame(matchups)
        df = df.sort_values("win_probability", ascending=False)

        return df
