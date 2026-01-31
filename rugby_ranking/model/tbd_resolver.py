"""
TBD (To Be Determined) resolution for knockout tournaments.

Resolves placeholder teams based on pool standings, qualification rules,
and probabilistic predictions.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import re

import pandas as pd
import numpy as np

from rugby_ranking.model.bracket import TBD
from rugby_ranking.model.season_predictor import SeasonPredictor, SeasonPrediction


class TBDResolver:
    """
    Resolves TBD placeholders to actual teams with probabilities.

    Handles various qualification scenarios:
    - Pool winners/runners-up
    - Best runners-up across pools
    - Seeded positions based on points
    - Probabilistic qualification when pools incomplete
    """

    def __init__(self, season_predictor: SeasonPredictor | None = None):
        """
        Initialize TBD resolver.

        Args:
            season_predictor: Optional SeasonPredictor for probabilistic resolution
                when pool stage is incomplete
        """
        self.season_predictor = season_predictor

    def resolve(
        self,
        tbd: TBD,
        pool_standings: pd.DataFrame | None = None,
        season_prediction: SeasonPrediction | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Resolve a TBD placeholder to likely teams with probabilities.

        Args:
            tbd: TBD placeholder to resolve
            pool_standings: Current or final pool standings
            season_prediction: Predicted final standings (if pools incomplete)

        Returns:
            List of (team_name, probability) tuples, sorted by probability
        """
        # Use predicted standings if provided and pools incomplete
        standings = season_prediction.predicted_standings if season_prediction else pool_standings

        if standings is None:
            return []

        # Parse TBD source to determine resolution method
        source_lower = tbd.source.lower()

        # Pool-based qualification
        if "pool" in source_lower:
            return self._resolve_pool_qualification(tbd, standings)

        # Overall position (e.g., "1st place", "8th place")
        if "place" in source_lower:
            return self._resolve_overall_position(tbd, standings)

        # Match progression (e.g., "Winner QF1")
        if "winner" in source_lower or "loser" in source_lower:
            # These are resolved during bracket simulation
            return []

        return []

    def _resolve_pool_qualification(
        self,
        tbd: TBD,
        standings: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """
        Resolve pool-based TBD (winners, runners-up, etc.).

        Handles:
        - "Pool A winner" -> 1st in Pool A
        - "Pool B runner-up" -> 2nd in Pool B
        - "Best runner-up" -> Best 2nd place across pools
        - "Pool 1st #3" -> 3rd ranked pool winner
        """
        source = tbd.source
        source_lower = source.lower()

        # Pattern: "Pool X winner"
        if "winner" in source_lower:
            pool = self._extract_pool_name(source)
            if pool:
                return self._get_pool_position_teams(standings, pool, 1)

        # Pattern: "Pool X runner-up"
        if "runner-up" in source_lower or "runner up" in source_lower:
            pool = self._extract_pool_name(source)
            if pool:
                return self._get_pool_position_teams(standings, pool, 2)

        # Pattern: "Best runner-up"
        if "best runner-up" in source_lower or "best runner up" in source_lower:
            return self._get_best_runner_up(standings)

        # Pattern: "Pool 1st #3" (3rd ranked pool winner)
        if re.search(r'pool\s+\d+(?:st|nd|rd|th)\s*#\d+', source_lower):
            position = self._extract_pool_position(source)
            rank = self._extract_rank_number(source)
            if position and rank:
                return self._get_ranked_pool_finisher(standings, position, rank)

        return []

    def _resolve_overall_position(
        self,
        tbd: TBD,
        standings: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """
        Resolve overall position (e.g., "1st place", "8th place").

        Returns single team if standings are final, or multiple teams with
        probabilities if using predicted standings.
        """
        position = self._extract_position_number(tbd.source)
        if position is None:
            return []

        # Check if we have playoff_probabilities (predicted standings)
        if "position_prob" in standings.columns:
            # Return teams with probability of finishing in this position
            teams_at_position = standings[
                standings[f"position_{position}_prob"] > 0.01
            ]
            return [
                (row["team"], row[f"position_{position}_prob"])
                for _, row in teams_at_position.iterrows()
            ]
        else:
            # Final standings - return single team
            if len(standings) >= position:
                team = standings.iloc[position - 1]["team"]
                return [(team, 1.0)]

        return []

    def _get_pool_position_teams(
        self,
        standings: pd.DataFrame,
        pool: str,
        position: int,
    ) -> List[Tuple[str, float]]:
        """Get team(s) at specific position in a pool."""
        if "pool" not in standings.columns:
            return []

        pool_data = standings[standings["pool"] == pool].copy()

        # If pools complete, return single team
        if "points" in pool_data.columns:
            pool_data = pool_data.sort_values("points", ascending=False)
            if len(pool_data) >= position:
                team = pool_data.iloc[position - 1]["team"]
                return [(team, 1.0)]

        # If using predictions, return multiple teams with probabilities
        # (would need position probabilities per pool)
        return []

    def _get_best_runner_up(
        self,
        standings: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """
        Get best runner-up across all pools.

        Compares 2nd place teams from each pool by points.
        """
        if "pool" not in standings.columns:
            return []

        runners_up = []
        for pool in standings["pool"].unique():
            pool_data = standings[standings["pool"] == pool].sort_values(
                "points", ascending=False
            )
            if len(pool_data) >= 2:
                runner_up = pool_data.iloc[1]
                runners_up.append((
                    runner_up["team"],
                    runner_up.get("points", 0),
                ))

        if not runners_up:
            return []

        # Sort by points and return best
        runners_up.sort(key=lambda x: x[1], reverse=True)
        best_team = runners_up[0][0]

        return [(best_team, 1.0)]

    def _get_ranked_pool_finisher(
        self,
        standings: pd.DataFrame,
        pool_position: int,
        rank: int,
    ) -> List[Tuple[str, float]]:
        """
        Get nth-ranked team that finished at pool_position in their pool.

        E.g., 3rd best pool winner (pool_position=1, rank=3)
        """
        if "pool" not in standings.columns:
            return []

        # Get all teams at this pool position
        finishers = []
        for pool in standings["pool"].unique():
            pool_data = standings[standings["pool"] == pool].sort_values(
                "points", ascending=False
            )
            if len(pool_data) >= pool_position:
                team_data = pool_data.iloc[pool_position - 1]
                finishers.append((
                    team_data["team"],
                    team_data.get("points", 0),
                ))

        # Sort by points and get rank-th team
        finishers.sort(key=lambda x: x[1], reverse=True)
        if len(finishers) >= rank:
            team = finishers[rank - 1][0]
            return [(team, 1.0)]

        return []

    def _extract_pool_name(self, source: str) -> str | None:
        """Extract pool name from source string (e.g., 'A', 'B')."""
        match = re.search(r'Pool\s+([A-Za-z])', source, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _extract_position_number(self, source: str) -> int | None:
        """Extract position number from source like '3rd place'."""
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s+place', source, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_pool_position(self, source: str) -> int | None:
        """Extract pool position from source like 'Pool 1st #3'."""
        match = re.search(r'pool\s+(\d+)(?:st|nd|rd|th)', source, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_rank_number(self, source: str) -> int | None:
        """Extract rank number from source like 'Pool 1st #3'."""
        match = re.search(r'#(\d+)', source)
        return int(match.group(1)) if match else None


def resolve_match_progression_tbd(
    tbd: TBD,
    bracket_state: Dict[str, str],
) -> str | None:
    """
    Resolve TBD based on earlier match results.

    Args:
        tbd: TBD placeholder referencing match result
        bracket_state: Dict mapping match_id to winner

    Returns:
        Team name if resolvable, None otherwise

    Examples:
        >>> state = {"QF1": "Leinster", "QF2": "Toulouse"}
        >>> tbd = TBD("Winner QF1")
        >>> resolve_match_progression_tbd(tbd, state)
        "Leinster"
    """
    source = tbd.source
    source_lower = source.lower()

    # Extract match ID
    match_id = None

    if "winner" in source_lower:
        # Pattern: "Winner QF1", "Winner of QF1"
        match = re.search(r'winner\s+(?:of\s+)?(\w+\d*)', source_lower)
        if match:
            match_id = match.group(1).upper()

    elif "loser" in source_lower:
        # Pattern: "Loser SF2"
        match = re.search(r'loser\s+(?:of\s+)?(\w+\d*)', source_lower)
        if match:
            match_id = match.group(1).upper()
            # Would need loser tracking in bracket_state

    if match_id and match_id in bracket_state:
        return bracket_state[match_id]

    return None


def create_tbd_from_string(source: str, criteria: Dict = None) -> TBD:
    """
    Convenience function to create TBD from string description.

    Args:
        source: Description of how team will be determined
        criteria: Optional additional criteria

    Returns:
        TBD object

    Examples:
        >>> create_tbd_from_string("Pool A winner")
        TBD('Pool A winner')
        >>> create_tbd_from_string("Best runner-up", {"method": "points"})
        TBD('Best runner-up', {'method': 'points'})
    """
    return TBD(source=source, criteria=criteria or {})
