"""
Knockout tournament bracket prediction using Monte Carlo simulation.

Predicts bracket progression, advancement probabilities, and likely matchups
for tournaments with TBD participants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import copy

import numpy as np
import pandas as pd

from rugby_ranking.model.bracket import BracketStructure, BracketMatch, TBD
from rugby_ranking.model.predictions import MatchPredictor


@dataclass
class BracketPrediction:
    """
    Predicted bracket outcomes from Monte Carlo simulation.

    Attributes:
        bracket: Original bracket structure
        advancement_probs: P(team reaches each round) DataFrame
        match_probabilities: Predicted outcomes for each match
        modal_bracket: Most likely complete bracket
        n_simulations: Number of simulations run
    """

    bracket: BracketStructure
    advancement_probs: pd.DataFrame  # Columns: team, qf_prob, sf_prob, final_prob, champion_prob
    match_probabilities: Dict[str, pd.DataFrame]  # match_id -> DataFrame with matchup probabilities
    modal_bracket: BracketStructure  # Most likely bracket realization
    n_simulations: int
    _simulation_results: Optional[List[Dict]] = None  # Raw simulation data


class BracketPredictor:
    """
    Predict knockout tournament bracket progression using Monte Carlo simulation.

    Simulates bracket multiple times to estimate:
    - Probability each team reaches each round
    - Likely matchups in TBD matches
    - Overall tournament winner probabilities

    Usage:
        >>> predictor = BracketPredictor(match_predictor, bracket_structure)
        >>> prediction = predictor.predict_bracket(
        ...     pool_standings=standings_df,
        ...     n_simulations=10000
        ... )
        >>> print(prediction.advancement_probs)
    """

    def __init__(
        self,
        match_predictor: MatchPredictor,
        bracket_structure: BracketStructure,
        seed: int | None = None,
    ):
        """
        Initialize bracket predictor.

        Args:
            match_predictor: Fitted MatchPredictor for match outcome prediction
            bracket_structure: BracketStructure defining tournament format
            seed: Random seed for reproducibility
        """
        self.match_predictor = match_predictor
        self.bracket = bracket_structure
        self.rng = np.random.default_rng(seed)

    def predict_bracket(
        self,
        pool_standings: pd.DataFrame | None = None,
        completed_knockout_matches: pd.DataFrame | None = None,
        n_simulations: int = 10000,
        return_simulation_details: bool = False,
    ) -> BracketPrediction:
        """
        Simulate bracket progression to predict outcomes.

        Args:
            pool_standings: Final pool standings for resolving initial TBD
                Required columns: team, position, pool (if multi-pool)
            completed_knockout_matches: Already-played knockout matches
                Required columns: match_id, winner
            n_simulations: Number of bracket simulations to run
            return_simulation_details: If True, store raw simulation results

        Returns:
            BracketPrediction with advancement probabilities and matchup likelihoods
        """
        # Resolve initial TBD from pool standings
        tbd_resolution = self._resolve_pool_tbd(pool_standings)

        # Apply completed matches to bracket
        bracket_state = self.bracket.clone()
        if completed_knockout_matches is not None:
            self._apply_completed_matches(bracket_state, completed_knockout_matches)

        # Run simulations
        simulation_results = []
        for _ in range(n_simulations):
            result = self._simulate_single_bracket(bracket_state, tbd_resolution)
            simulation_results.append(result)

        # Aggregate results
        advancement_probs = self._compute_advancement_probabilities(simulation_results)
        match_probs = self._compute_match_probabilities(simulation_results)
        modal_bracket = self._compute_modal_bracket(simulation_results, bracket_state)

        return BracketPrediction(
            bracket=self.bracket,
            advancement_probs=advancement_probs,
            match_probabilities=match_probs,
            modal_bracket=modal_bracket,
            n_simulations=n_simulations,
            _simulation_results=simulation_results if return_simulation_details else None,
        )

    def _resolve_pool_tbd(
        self,
        pool_standings: pd.DataFrame | None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Resolve TBD placeholders from pool standings.

        Args:
            pool_standings: Final pool standings

        Returns:
            Dict mapping TBD source string to [(team, probability), ...]
        """
        if pool_standings is None:
            return {}

        resolution = {}

        # Get all TBD placeholders
        for match_id, tbd in self.bracket.get_all_tbd_placeholders():
            source = tbd.source

            # Skip if already resolved
            if source in resolution:
                continue

            # Parse source to determine team
            teams = self._parse_pool_source(source, pool_standings)
            if teams:
                resolution[source] = teams

        return resolution

    def _parse_pool_source(
        self,
        source: str,
        pool_standings: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """
        Parse a pool-based TBD source to extract teams.

        Handles patterns like:
        - "Pool A winner" -> 1st place in Pool A
        - "Pool B runner-up" -> 2nd place in Pool B
        - "Pool 1st #3" -> 3rd ranked pool winner
        - "1st place" -> Overall 1st place
        """
        source_lower = source.lower()

        # Pattern: "Pool X winner/runner-up"
        if "pool" in source_lower and "winner" in source_lower:
            pool_name = self._extract_pool_name(source)
            if pool_name:
                team = self._get_pool_position(pool_standings, pool_name, 1)
                return [(team, 1.0)] if team else []

        if "pool" in source_lower and "runner-up" in source_lower:
            pool_name = self._extract_pool_name(source)
            if pool_name:
                team = self._get_pool_position(pool_standings, pool_name, 2)
                return [(team, 1.0)] if team else []

        # Pattern: "1st place", "2nd place", etc.
        if "place" in source_lower:
            position = self._extract_position_number(source)
            if position:
                team = self._get_overall_position(pool_standings, position)
                return [(team, 1.0)] if team else []

        # Pattern: "Pool 1st #3" (3rd ranked pool winner)
        if "pool 1st" in source_lower or "pool 2nd" in source_lower:
            pool_position = 1 if "1st" in source_lower else 2
            rank = self._extract_rank_number(source)
            if rank:
                team = self._get_ranked_pool_finisher(pool_standings, pool_position, rank)
                return [(team, 1.0)] if team else []

        return []

    def _extract_pool_name(self, source: str) -> str | None:
        """Extract pool name from source string."""
        import re
        match = re.search(r'Pool\s+([A-Z])', source, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def _extract_position_number(self, source: str) -> int | None:
        """Extract position number from source like '3rd place'."""
        import re
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s+place', source, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_rank_number(self, source: str) -> int | None:
        """Extract rank number from source like 'Pool 1st #3'."""
        import re
        match = re.search(r'#(\d+)', source)
        return int(match.group(1)) if match else None

    def _get_pool_position(
        self,
        pool_standings: pd.DataFrame,
        pool: str,
        position: int,
    ) -> str | None:
        """Get team at specific position in a pool."""
        if "pool" not in pool_standings.columns:
            return None

        pool_data = pool_standings[pool_standings["pool"] == pool]
        if len(pool_data) >= position:
            return pool_data.iloc[position - 1]["team"]
        return None

    def _get_overall_position(
        self,
        pool_standings: pd.DataFrame,
        position: int,
    ) -> str | None:
        """Get team at specific overall position."""
        if len(pool_standings) >= position:
            return pool_standings.iloc[position - 1]["team"]
        return None

    def _get_ranked_pool_finisher(
        self,
        pool_standings: pd.DataFrame,
        pool_position: int,
        rank: int,
    ) -> str | None:
        """
        Get nth-ranked team that finished at pool_position in their pool.

        E.g., 3rd best pool winner (pool_position=1, rank=3)
        """
        if "pool" not in pool_standings.columns:
            return None

        # Get all teams at this pool position
        finishers = []
        for pool in pool_standings["pool"].unique():
            team = self._get_pool_position(pool_standings, pool, pool_position)
            if team:
                team_data = pool_standings[pool_standings["team"] == team].iloc[0]
                finishers.append((team, team_data.get("points", 0)))

        # Sort by points and get rank-th team
        finishers.sort(key=lambda x: x[1], reverse=True)
        if len(finishers) >= rank:
            return finishers[rank - 1][0]
        return None

    def _apply_completed_matches(
        self,
        bracket: BracketStructure,
        completed_matches: pd.DataFrame,
    ) -> None:
        """
        Update bracket with completed match results.

        Args:
            bracket: Bracket to update (modified in place)
            completed_matches: DataFrame with match_id and winner columns
        """
        for _, row in completed_matches.iterrows():
            match_id = row["match_id"]
            winner = row["winner"]

            if match_id not in bracket.matches:
                continue

            # Update dependent matches with winner
            for dep_match_id in bracket.get_dependent_matches(match_id):
                dep_match = bracket.matches[dep_match_id]

                # Replace TBD with winner
                if isinstance(dep_match.home_team, TBD):
                    if match_id in dep_match.depends_on:
                        dep_match.home_team = winner

                if isinstance(dep_match.away_team, TBD):
                    if match_id in dep_match.depends_on:
                        dep_match.away_team = winner

    def _simulate_single_bracket(
        self,
        bracket: BracketStructure,
        tbd_resolution: Dict[str, List[Tuple[str, float]]],
    ) -> Dict:
        """
        Simulate one complete bracket progression.

        Args:
            bracket: Bracket structure (possibly with TBD)
            tbd_resolution: Mapping of TBD sources to team probabilities

        Returns:
            Dict with simulation results:
                - round_results: {match_id: winner}
                - team_progress: {team: furthest_round}
                - champion: tournament winner
        """
        # Clone bracket for this simulation
        sim_bracket = bracket.clone()

        # Resolve initial TBD by sampling
        self._sample_tbd_participants(sim_bracket, tbd_resolution)

        # Simulate rounds in order
        round_results = {}
        for round_name in sim_bracket.rounds:
            round_matches = sim_bracket.get_round_matches(round_name)

            for match in round_matches:
                if not match.is_determined():
                    # Resolve from previous round
                    self._resolve_match_dependencies(sim_bracket, match, round_results)

                # Simulate match
                winner = self._simulate_match(match)
                round_results[match.id] = winner

                # Propagate winner to dependent matches
                for dep_id in sim_bracket.get_dependent_matches(match.id):
                    self._update_dependent_match(sim_bracket, dep_id, match.id, winner)

        # Extract team progress
        team_progress = self._extract_team_progress(sim_bracket, round_results)

        # Get champion (winner of final)
        final_matches = sim_bracket.get_round_matches("final")
        champion = round_results.get(final_matches[0].id) if final_matches else None

        return {
            "round_results": round_results,
            "team_progress": team_progress,
            "champion": champion,
        }

    def _sample_tbd_participants(
        self,
        bracket: BracketStructure,
        tbd_resolution: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """Sample teams for TBD placeholders based on probabilities."""
        for match in bracket.matches.values():
            # Resolve home team if TBD
            if isinstance(match.home_team, TBD):
                if match.home_team.source in tbd_resolution:
                    teams_probs = tbd_resolution[match.home_team.source]
                    teams = [t for t, _ in teams_probs]
                    probs = [p for _, p in teams_probs]
                    match.home_team = self.rng.choice(teams, p=probs)

            # Resolve away team if TBD
            if isinstance(match.away_team, TBD):
                if match.away_team.source in tbd_resolution:
                    teams_probs = tbd_resolution[match.away_team.source]
                    teams = [t for t, _ in teams_probs]
                    probs = [p for _, p in teams_probs]
                    match.away_team = self.rng.choice(teams, p=probs)

    def _resolve_match_dependencies(
        self,
        bracket: BracketStructure,
        match: BracketMatch,
        round_results: Dict[str, str],
    ) -> None:
        """Resolve match participants from previous match results."""
        # This is handled in _update_dependent_match during simulation
        pass

    def _update_dependent_match(
        self,
        bracket: BracketStructure,
        match_id: str,
        completed_match_id: str,
        winner: str,
    ) -> None:
        """Update dependent match with winner of completed match."""
        match = bracket.matches[match_id]

        # Check if home team references this completed match
        if isinstance(match.home_team, TBD):
            if f"Winner {completed_match_id}" in match.home_team.source:
                match.home_team = winner

        # Check if away team references this completed match
        if isinstance(match.away_team, TBD):
            if f"Winner {completed_match_id}" in match.away_team.source:
                match.away_team = winner

    def _simulate_match(self, match: BracketMatch) -> str:
        """
        Simulate a single match and return winner.

        Args:
            match: Match to simulate (must have determined participants)

        Returns:
            Winning team name
        """
        if not match.is_determined():
            raise ValueError(f"Cannot simulate match {match.id} with TBD participants")

        home_team = match.home_team
        away_team = match.away_team

        # Get match prediction
        prediction = self.match_predictor.predict_match(
            home_team=home_team,
            away_team=away_team,
            is_home=match.home_advantage,
        )

        # Sample winner based on probabilities
        # prediction.home_win_prob is P(home wins)
        if self.rng.random() < prediction.home_win_prob:
            return home_team
        else:
            return away_team

    def _extract_team_progress(
        self,
        bracket: BracketStructure,
        round_results: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Extract how far each team progressed in the tournament.

        Returns:
            Dict mapping team name to furthest round reached
        """
        progress = {}

        for round_name in bracket.rounds:
            for match in bracket.get_round_matches(round_name):
                # Both teams reached this round
                if match.is_determined():
                    progress[match.home_team] = round_name
                    progress[match.away_team] = round_name

                # Winner advances
                if match.id in round_results:
                    winner = round_results[match.id]
                    # Update winner's progress (will be overwritten in next round)
                    progress[winner] = round_name

        return progress

    def _compute_advancement_probabilities(
        self,
        simulation_results: List[Dict],
    ) -> pd.DataFrame:
        """
        Compute probability each team reaches each round.

        Args:
            simulation_results: List of simulation result dicts

        Returns:
            DataFrame with columns: team, qf_prob, sf_prob, final_prob, champion_prob
        """
        # Count how many times each team reached each round
        round_counts = defaultdict(lambda: defaultdict(int))

        for sim in simulation_results:
            team_progress = sim["team_progress"]
            champion = sim["champion"]

            for team, furthest_round in team_progress.items():
                # Increment count for this round and all earlier rounds
                for round_name in self.bracket.rounds:
                    round_idx = self.bracket.rounds.index(furthest_round)
                    current_idx = self.bracket.rounds.index(round_name)

                    if current_idx <= round_idx:
                        round_counts[team][round_name] += 1

            # Mark champion
            if champion:
                round_counts[champion]["champion"] = round_counts[champion].get("champion", 0) + 1

        # Convert to DataFrame
        data = []
        n_sims = len(simulation_results)

        for team, counts in round_counts.items():
            row = {"team": team}

            for round_name in self.bracket.rounds:
                col_name = f"{round_name}_prob"
                row[col_name] = counts.get(round_name, 0) / n_sims

            row["champion_prob"] = counts.get("champion", 0) / n_sims
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by championship probability
        df = df.sort_values("champion_prob", ascending=False).reset_index(drop=True)

        return df

    def _compute_match_probabilities(
        self,
        simulation_results: List[Dict],
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute matchup probabilities for each match.

        Returns:
            Dict mapping match_id to DataFrame with columns:
                home_team, away_team, probability, home_win_prob
        """
        match_probs = {}

        # Collect matchups for each match
        matchups = defaultdict(Counter)

        for sim in simulation_results:
            # For each match, track the matchup
            for match_id in sim["round_results"].keys():
                # Need to track what the matchup was in this simulation
                # This requires storing matchup info in simulation results
                # For now, we'll compute from available data
                pass

        # TODO: Need to modify simulation to track matchups
        # For now, return empty dict
        return match_probs

    def _compute_modal_bracket(
        self,
        simulation_results: List[Dict],
        initial_bracket: BracketStructure,
    ) -> BracketStructure:
        """
        Compute most likely bracket realization.

        Returns:
            BracketStructure with most common winner in each match
        """
        # Find modal winner for each match
        match_winners = defaultdict(Counter)

        for sim in simulation_results:
            for match_id, winner in sim["round_results"].items():
                match_winners[match_id][winner] += 1

        # Create modal bracket
        modal = initial_bracket.clone()

        for match_id, winner_counts in match_winners.items():
            most_common_winner = winner_counts.most_common(1)[0][0]
            # Store in bracket somehow (would need additional structure)

        return modal

    def get_likely_matchup(
        self,
        match_id: str,
        threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get most likely participants for a specific match.

        Args:
            match_id: Match to query
            threshold: Only return matchups with P > threshold

        Returns:
            DataFrame with columns: home_team, away_team, probability, home_win_prob
        """
        # TODO: Implement based on match_probabilities
        return pd.DataFrame()
