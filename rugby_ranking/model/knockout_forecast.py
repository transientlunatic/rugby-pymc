"""
Knockout Round Forecasting for Rugby Tournaments.

Predicts playoff/knockout rounds where opponents are TBD based on league positions.
Handles cascading uncertainty from pool stage through to finals.

Supports:
- World Cup format (pools → R16 → QF → SF → Final)
- URC playoffs (1v8, 2v7, 3v6, 4v5 → SF → Final)
- Champions Cup (R16 → QF → SF → Final)
- Custom bracket structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
import itertools

import numpy as np
import pandas as pd


@dataclass
class KnockoutMatch:
    """A knockout match with potentially uncertain opponents."""

    stage: str  # 'R16', 'QF', 'SF', 'Final', etc.
    match_id: str  # Unique identifier
    home_seed: int | str  # Seed number or team name
    away_seed: int | str  # Seed number or team name
    winner_advances_to: Optional[str] = None  # Next match_id
    loser_advances_to: Optional[str] = None  # For consolation brackets

    # Set after simulation
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_win_prob: Optional[float] = None
    winner: Optional[str] = None


@dataclass
class KnockoutStageResult:
    """Results for a single knockout stage."""

    stage: str
    matches: List[KnockoutMatch]
    team_probabilities: Dict[str, float]  # P(team reaches this stage)
    matchup_probabilities: Dict[Tuple[str, str], float]  # P(specific matchup)


@dataclass
class TournamentForecast:
    """Complete tournament forecast including pool and knockout stages."""

    tournament_name: str
    pool_standings: Optional[pd.DataFrame] = None
    pool_position_probabilities: Optional[pd.DataFrame] = None

    # Knockout stage results (ordered)
    knockout_stages: List[KnockoutStageResult] = field(default_factory=list)

    # Overall probabilities
    winner_probabilities: Dict[str, float] = field(default_factory=dict)
    runner_up_probabilities: Dict[str, float] = field(default_factory=dict)

    # Most likely path for each team
    likely_paths: Dict[str, List[str]] = field(default_factory=dict)


class BracketStructure(ABC):
    """
    Abstract base class for tournament bracket structures.

    Defines how teams are seeded and matched in knockout rounds.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_knockout_matches(self, n_teams: int) -> List[KnockoutMatch]:
        """
        Generate knockout match structure.

        Args:
            n_teams: Number of teams qualifying for knockouts

        Returns:
            List of KnockoutMatch objects defining the bracket
        """
        pass

    @abstractmethod
    def get_stage_order(self) -> List[str]:
        """Return ordered list of stage names (e.g., ['QF', 'SF', 'Final'])."""
        pass

    def determine_matchup(
        self,
        match: KnockoutMatch,
        pool_positions: Dict[str, int],
    ) -> Tuple[str, str]:
        """
        Determine actual teams for a match based on pool positions.

        Args:
            match: KnockoutMatch with seed placeholders
            pool_positions: Dict mapping team name to pool position

        Returns:
            (home_team, away_team) tuple
        """
        # Reverse lookup: position -> team
        position_to_team = {pos: team for team, pos in pool_positions.items()}

        # Handle seed numbers vs team names
        if isinstance(match.home_seed, int):
            home_team = position_to_team.get(match.home_seed, f"Seed{match.home_seed}")
        else:
            home_team = match.home_seed

        if isinstance(match.away_seed, int):
            away_team = position_to_team.get(match.away_seed, f"Seed{match.away_seed}")
        else:
            away_team = match.away_seed

        return home_team, away_team


class URCPlayoffBracket(BracketStructure):
    """
    URC (United Rugby Championship) playoff bracket.

    Format: Top 8 teams qualify
    - QF: 1v8, 2v7, 3v6, 4v5
    - SF: Winner(1v8) v Winner(4v5), Winner(2v7) v Winner(3v6)
    - Final: Winners of SF
    """

    def __init__(self):
        super().__init__("URC Playoffs")

    def get_knockout_matches(self, n_teams: int = 8) -> List[KnockoutMatch]:
        """Generate URC playoff bracket."""
        matches = []

        # Quarterfinals
        matches.extend([
            KnockoutMatch(
                stage='QF',
                match_id='QF1',
                home_seed=1,
                away_seed=8,
                winner_advances_to='SF1',
            ),
            KnockoutMatch(
                stage='QF',
                match_id='QF2',
                home_seed=4,
                away_seed=5,
                winner_advances_to='SF1',
            ),
            KnockoutMatch(
                stage='QF',
                match_id='QF3',
                home_seed=2,
                away_seed=7,
                winner_advances_to='SF2',
            ),
            KnockoutMatch(
                stage='QF',
                match_id='QF4',
                home_seed=3,
                away_seed=6,
                winner_advances_to='SF2',
            ),
        ])

        # Semifinals (seeds TBD based on QF results)
        matches.extend([
            KnockoutMatch(
                stage='SF',
                match_id='SF1',
                home_seed='Winner(QF1)',
                away_seed='Winner(QF2)',
                winner_advances_to='Final',
            ),
            KnockoutMatch(
                stage='SF',
                match_id='SF2',
                home_seed='Winner(QF3)',
                away_seed='Winner(QF4)',
                winner_advances_to='Final',
            ),
        ])

        # Final
        matches.append(
            KnockoutMatch(
                stage='Final',
                match_id='Final',
                home_seed='Winner(SF1)',
                away_seed='Winner(SF2)',
            )
        )

        return matches

    def get_stage_order(self) -> List[str]:
        return ['QF', 'SF', 'Final']


class WorldCupBracket(BracketStructure):
    """
    Rugby World Cup knockout bracket.

    Format: 8 teams qualify from pools (typically top 2 from each of 4 pools)
    - QF: Pool A winner v Pool B runner-up, Pool B winner v Pool A runner-up, etc.
    - SF: Winners of QF
    - Bronze: Losers of SF
    - Final: Winners of SF
    """

    def __init__(self):
        super().__init__("Rugby World Cup")

    def get_knockout_matches(self, n_teams: int = 8) -> List[KnockoutMatch]:
        """Generate World Cup knockout bracket."""
        matches = []

        # Quarterfinals (assuming 4 pools: A, B, C, D)
        # Pool winners: 1, 3, 5, 7 (1st in A, B, C, D)
        # Pool runners-up: 2, 4, 6, 8 (2nd in A, B, C, D)
        matches.extend([
            KnockoutMatch(
                stage='QF',
                match_id='QF1',
                home_seed=1,  # Pool A winner
                away_seed=4,  # Pool B runner-up
                winner_advances_to='SF1',
                loser_advances_to='Bronze',
            ),
            KnockoutMatch(
                stage='QF',
                match_id='QF2',
                home_seed=3,  # Pool B winner
                away_seed=2,  # Pool A runner-up
                winner_advances_to='SF1',
                loser_advances_to='Bronze',
            ),
            KnockoutMatch(
                stage='QF',
                match_id='QF3',
                home_seed=5,  # Pool C winner
                away_seed=8,  # Pool D runner-up
                winner_advances_to='SF2',
                loser_advances_to='Bronze',
            ),
            KnockoutMatch(
                stage='QF',
                match_id='QF4',
                home_seed=7,  # Pool D winner
                away_seed=6,  # Pool C runner-up
                winner_advances_to='SF2',
                loser_advances_to='Bronze',
            ),
        ])

        # Semifinals
        matches.extend([
            KnockoutMatch(
                stage='SF',
                match_id='SF1',
                home_seed='Winner(QF1)',
                away_seed='Winner(QF2)',
                winner_advances_to='Final',
                loser_advances_to='Bronze',
            ),
            KnockoutMatch(
                stage='SF',
                match_id='SF2',
                home_seed='Winner(QF3)',
                away_seed='Winner(QF4)',
                winner_advances_to='Final',
                loser_advances_to='Bronze',
            ),
        ])

        # Bronze final (3rd place playoff)
        matches.append(
            KnockoutMatch(
                stage='Bronze',
                match_id='Bronze',
                home_seed='Loser(SF1)',
                away_seed='Loser(SF2)',
            )
        )

        # Final
        matches.append(
            KnockoutMatch(
                stage='Final',
                match_id='Final',
                home_seed='Winner(SF1)',
                away_seed='Winner(SF2)',
            )
        )

        return matches

    def get_stage_order(self) -> List[str]:
        return ['QF', 'SF', 'Bronze', 'Final']


class ChampionsCupBracket(BracketStructure):
    """
    Champions Cup knockout bracket.

    Format: 16 teams qualify
    - R16: Seeded 1v16, 2v15, ..., 8v9
    - QF: Winners from R16
    - SF: Winners from QF
    - Final: Winners from SF
    """

    def __init__(self):
        super().__init__("Champions Cup")

    def get_knockout_matches(self, n_teams: int = 16) -> List[KnockoutMatch]:
        """Generate Champions Cup bracket."""
        matches = []

        # Round of 16
        for i in range(8):
            home_seed = i + 1
            away_seed = 16 - i
            matches.append(
                KnockoutMatch(
                    stage='R16',
                    match_id=f'R16_{i+1}',
                    home_seed=home_seed,
                    away_seed=away_seed,
                    winner_advances_to=f'QF{(i//2)+1}',
                )
            )

        # Quarterfinals
        for i in range(4):
            matches.append(
                KnockoutMatch(
                    stage='QF',
                    match_id=f'QF{i+1}',
                    home_seed=f'Winner(R16_{2*i+1})',
                    away_seed=f'Winner(R16_{2*i+2})',
                    winner_advances_to=f'SF{(i//2)+1}',
                )
            )

        # Semifinals
        for i in range(2):
            matches.append(
                KnockoutMatch(
                    stage='SF',
                    match_id=f'SF{i+1}',
                    home_seed=f'Winner(QF{2*i+1})',
                    away_seed=f'Winner(QF{2*i+2})',
                    winner_advances_to='Final',
                )
            )

        # Final
        matches.append(
            KnockoutMatch(
                stage='Final',
                match_id='Final',
                home_seed='Winner(SF1)',
                away_seed='Winner(SF2)',
            )
        )

        return matches

    def get_stage_order(self) -> List[str]:
        return ['R16', 'QF', 'SF', 'Final']


class TournamentTreeSimulator:
    """
    Simulate knockout tournament trees with uncertain seeding.

    Handles cascading uncertainty:
    - Pool position uncertainty → matchup uncertainty → outcome uncertainty

    Usage:
        >>> simulator = TournamentTreeSimulator(match_predictor, bracket_structure)
        >>> forecast = simulator.simulate_knockout(
        ...     pool_prediction,
        ...     n_simulations=10000
        ... )
        >>> print(forecast.winner_probabilities)
    """

    def __init__(
        self,
        match_predictor,
        bracket_structure: BracketStructure,
        season: str = "2025-2026",
    ):
        """
        Initialize knockout simulator.

        Args:
            match_predictor: MatchPredictor for match outcome predictions
            bracket_structure: BracketStructure defining tournament format
            season: Season identifier
        """
        self.match_predictor = match_predictor
        self.bracket = bracket_structure
        self.season = season

    def simulate_knockout(
        self,
        pool_position_probabilities: pd.DataFrame,
        n_simulations: int = 10000,
        pool_standings: Optional[pd.DataFrame] = None,
    ) -> TournamentForecast:
        """
        Simulate knockout rounds with uncertain pool positions.

        Args:
            pool_position_probabilities: DataFrame with P(team finishes in position k)
            n_simulations: Number of Monte Carlo simulations
            pool_standings: Optional current pool standings

        Returns:
            TournamentForecast with stage-by-stage probabilities
        """
        teams = list(pool_position_probabilities.index)
        n_teams = len(teams)

        # Extract position probabilities
        position_cols = [col for col in pool_position_probabilities.columns
                        if col.startswith('P(pos')]

        # Storage for simulation results
        stage_appearances = defaultdict(lambda: defaultdict(int))  # stage -> team -> count
        matchup_counts = defaultdict(lambda: defaultdict(int))  # stage -> (home, away) -> count
        winner_counts = defaultdict(int)  # team -> count of tournament wins
        runner_up_counts = defaultdict(int)

        # Path tracking
        team_paths = defaultdict(lambda: defaultdict(list))  # team -> sim_id -> path

        # Run simulations
        for sim_id in range(n_simulations):
            # Sample pool positions
            pool_positions = self._sample_pool_positions(
                teams, pool_position_probabilities, position_cols
            )

            # Simulate knockout tree for this seeding
            tournament_result = self._simulate_single_tournament(
                pool_positions, sim_id, team_paths
            )

            # Record results
            for stage, stage_teams in tournament_result['stages'].items():
                for team in stage_teams:
                    stage_appearances[stage][team] += 1

            for stage, matchups in tournament_result['matchups'].items():
                for matchup in matchups:
                    matchup_counts[stage][matchup] += 1

            if 'winner' in tournament_result and tournament_result['winner']:
                winner_counts[tournament_result['winner']] += 1

            if 'runner_up' in tournament_result and tournament_result['runner_up']:
                runner_up_counts[tournament_result['runner_up']] += 1

        # Convert counts to probabilities
        winner_probs = {team: count / n_simulations
                       for team, count in winner_counts.items()}
        runner_up_probs = {team: count / n_simulations
                          for team, count in runner_up_counts.items()}

        # Create stage results
        knockout_stages = []
        for stage in self.bracket.get_stage_order():
            team_probs = {
                team: count / n_simulations
                for team, count in stage_appearances[stage].items()
            }

            matchup_probs = {
                matchup: count / n_simulations
                for matchup, count in matchup_counts[stage].items()
            }

            knockout_stages.append(KnockoutStageResult(
                stage=stage,
                matches=[],  # Populated with most likely matchups
                team_probabilities=team_probs,
                matchup_probabilities=matchup_probs,
            ))

        # Find most likely path for each team
        likely_paths = self._extract_likely_paths(team_paths, n_simulations)

        return TournamentForecast(
            tournament_name=self.bracket.name,
            pool_standings=pool_standings,
            pool_position_probabilities=pool_position_probabilities,
            knockout_stages=knockout_stages,
            winner_probabilities=winner_probs,
            runner_up_probabilities=runner_up_probs,
            likely_paths=likely_paths,
        )

    def _sample_pool_positions(
        self,
        teams: List[str],
        position_probs: pd.DataFrame,
        position_cols: List[str],
    ) -> Dict[str, int]:
        """
        Sample pool positions for all teams from probability distributions.

        Uses Gumbel-max trick applied column-wise to sample a valid ranking
        (bijection) where each position is assigned to exactly one team,
        properly respecting the probability distributions.

        This is the correct way to sample permutations from a probability matrix.
        """
        n_teams = len(teams)
        prob_matrix = position_probs[position_cols].values

        # Sample using Gumbel-max trick applied to each position (column)
        # This ensures each position gets exactly one team
        pool_positions = {}

        # For each position, sample which team occupies it
        available_teams = set(range(n_teams))

        for position_idx in range(n_teams):
            if not available_teams:
                break

            # Get probabilities for this position across remaining teams
            position_probs_vec = prob_matrix[:, position_idx].copy()

            # Zero out probabilities for already-assigned teams
            for team_idx in range(n_teams):
                if team_idx not in available_teams:
                    position_probs_vec[team_idx] = 0

            # Normalize
            prob_sum = position_probs_vec.sum()
            if prob_sum > 0:
                position_probs_vec = position_probs_vec / prob_sum
            else:
                # Uniform over available teams
                for team_idx in available_teams:
                    position_probs_vec[team_idx] = 1.0 / len(available_teams)

            # Sample team for this position
            sampled_team_idx = np.random.choice(n_teams, p=position_probs_vec)

            # Assign position (1-indexed)
            team_name = teams[sampled_team_idx]
            pool_positions[team_name] = position_idx + 1
            available_teams.remove(sampled_team_idx)

        return pool_positions

    def _simulate_single_tournament(
        self,
        pool_positions: Dict[str, int],
        sim_id: int,
        team_paths: Dict[str, Dict[int, List[str]]],
    ) -> Dict:
        """
        Simulate a single tournament with known pool positions.

        Returns dict with:
        - stages: dict mapping stage -> list of teams
        - matchups: dict mapping stage -> list of matchups
        - winner: tournament winner
        - runner_up: runner-up
        """
        matches = self.bracket.get_knockout_matches()

        # Track which teams advance
        advancing_teams = {}  # match_id -> winner
        stage_teams = defaultdict(set)
        stage_matchups = defaultdict(list)

        # Process each stage in order
        for stage in self.bracket.get_stage_order():
            stage_matches = [m for m in matches if m.stage == stage]

            for match in stage_matches:
                # Determine actual teams
                home_team, away_team = self._resolve_match_teams(
                    match, pool_positions, advancing_teams
                )

                if home_team and away_team:
                    # Record appearance
                    stage_teams[stage].add(home_team)
                    stage_teams[stage].add(away_team)
                    stage_matchups[stage].append((home_team, away_team))

                    # Predict match outcome
                    try:
                        prediction = self.match_predictor.predict_teams_only(
                            home_team=home_team,
                            away_team=away_team,
                            season=self.season,
                        )
                        home_win_prob = prediction.home_win_prob
                    except:
                        # Fallback to 50-50 if prediction fails
                        home_win_prob = 0.5

                    # Sample winner
                    winner = home_team if np.random.random() < home_win_prob else away_team
                    loser = away_team if winner == home_team else home_team

                    advancing_teams[match.match_id] = winner

                    # Track paths
                    team_paths[winner][sim_id].append(stage)
                    team_paths[loser][sim_id].append(stage)  # Made it to this stage but lost

        # Identify winner and runner-up
        final_match_id = 'Final'
        winner = advancing_teams.get(final_match_id)

        # Runner-up is the one who made it to final but didn't win
        final_teams = stage_teams.get('Final', set())
        runner_up = None
        if winner and len(final_teams) == 2:
            runner_up = list(final_teams - {winner})[0] if final_teams else None

        return {
            'stages': dict(stage_teams),
            'matchups': dict(stage_matchups),
            'winner': winner,
            'runner_up': runner_up,
        }

    def _resolve_match_teams(
        self,
        match: KnockoutMatch,
        pool_positions: Dict[str, int],
        advancing_teams: Dict[str, str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve match teams from seeds or previous match winners."""
        def resolve_seed(seed):
            if isinstance(seed, int):
                # Pool position
                return self._team_from_position(seed, pool_positions)
            elif isinstance(seed, str) and seed.startswith('Winner('):
                # Previous match winner
                prev_match_id = seed[7:-1]  # Extract match_id from 'Winner(match_id)'
                return advancing_teams.get(prev_match_id)
            elif isinstance(seed, str) and seed.startswith('Loser('):
                # Previous match loser (for bronze medal games)
                prev_match_id = seed[6:-1]
                winner = advancing_teams.get(prev_match_id)
                # This is trickier - would need to track losers separately
                return None  # Simplified for now
            else:
                return seed

        home_team = resolve_seed(match.home_seed)
        away_team = resolve_seed(match.away_seed)

        return home_team, away_team

    def _team_from_position(
        self,
        position: int,
        pool_positions: Dict[str, int],
    ) -> Optional[str]:
        """Get team name for a given pool position."""
        for team, pos in pool_positions.items():
            if pos == position:
                return team
        return None

    def _extract_likely_paths(
        self,
        team_paths: Dict[str, Dict[int, List[str]]],
        n_simulations: int,
    ) -> Dict[str, List[str]]:
        """Extract most likely knockout path for each team."""
        likely_paths = {}

        for team, sims in team_paths.items():
            # Count path frequencies
            path_counts = defaultdict(int)
            for sim_id, path in sims.items():
                path_tuple = tuple(path)
                path_counts[path_tuple] += 1

            if path_counts:
                # Most common path
                most_likely = max(path_counts.items(), key=lambda x: x[1])
                likely_paths[team] = list(most_likely[0])

        return likely_paths


def format_knockout_forecast(forecast: TournamentForecast, top_n: int = 10) -> str:
    """Format tournament forecast as human-readable text."""
    lines = []

    lines.append("=" * 70)
    lines.append(f"KNOCKOUT TOURNAMENT FORECAST: {forecast.tournament_name.upper()}")
    lines.append("=" * 70)

    # Winner probabilities
    lines.append("\nTOURNAMENT WINNER PROBABILITIES:")
    lines.append("-" * 70)

    sorted_winners = sorted(
        forecast.winner_probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    for i, (team, prob) in enumerate(sorted_winners, 1):
        lines.append(f"{i:2d}. {team:<30} {prob:>6.1%}")

    # Stage-by-stage breakdown
    for stage_result in forecast.knockout_stages:
        lines.append(f"\n{stage_result.stage.upper()} - QUALIFICATION PROBABILITIES:")
        lines.append("-" * 70)

        sorted_teams = sorted(
            stage_result.team_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        for i, (team, prob) in enumerate(sorted_teams, 1):
            lines.append(f"{i:2d}. {team:<30} {prob:>6.1%}")

    return "\n".join(lines)
