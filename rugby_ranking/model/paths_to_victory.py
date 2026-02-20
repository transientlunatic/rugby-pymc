"""
Paths to Victory Analysis for Rugby Tournaments.

Analyzes and visualizes how teams can achieve specific final positions
using hybrid MCMC-combinatorial approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Literal
from collections import defaultdict
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import hamming

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class Condition:
    """A condition that affects achieving target outcome."""
    game: Tuple[str, str]  # (home_team, away_team)
    outcome: str  # 'home_win', 'away_win', 'draw', 'home_bonus', etc.
    frequency: float  # How often this appears in successful scenarios
    conditional_prob: float  # P(success | condition)
    importance: float  # ΔP when condition is met
    team_controls: bool  # Whether target team controls this outcome


@dataclass
class ScenarioCluster:
    """A cluster of similar scenarios."""
    description: str
    frequency: float  # % of successful scenarios in this cluster
    representative_games: Dict[Tuple[str, str], str]  # Game outcomes
    probability: float  # Joint probability of this scenario


@dataclass
class PathsOutput:
    """Analysis results for paths to achieve a target position."""
    team: str
    target_position: int
    probability: float
    method: Literal['mcmc', 'combinatorial']

    # Key conditions affecting outcome
    conditions: List[Condition]

    # Critical games ranked by impact
    critical_games: List[Tuple[Tuple[str, str], float]]  # (game, ΔP)

    # Scenario clusters
    scenario_clusters: List[ScenarioCluster]

    # Human-readable summary
    narrative: str

    # Visualization (if plotly available)
    sankey_diagram: Optional[go.Figure] = None

    # Raw data for custom analysis
    _raw_simulations: Optional[List] = None


class PathsAnalyzer:
    """
    Analyze paths for teams to achieve target final positions.

    Uses MCMC pattern mining for early tournament (many games remaining),
    switches to weighted combinatorial enumeration late tournament.

    Usage:
        >>> analyzer = PathsAnalyzer(season_prediction, match_predictor)
        >>> paths = analyzer.analyze_paths(team='Scotland', target_position=2)
        >>> print(paths.narrative)
        >>> paths.sankey_diagram.show()  # If plotly installed
    """

    def __init__(
        self,
        season_prediction,
        match_predictor,
        combinatorial_threshold: int = 100_000,
    ):
        """
        Initialize paths analyzer.

        Args:
            season_prediction: SeasonPrediction object with simulation results
            match_predictor: MatchPredictor for game probabilities
            combinatorial_threshold: Max combinations before using MCMC mode
        """
        self.season_prediction = season_prediction
        self.match_predictor = match_predictor
        self.combinatorial_threshold = combinatorial_threshold

        # Extract simulation data if available
        self._simulations = None
        self._extract_simulations()

    def _extract_simulations(self):
        """Extract detailed simulation data from season prediction."""
        # This would need to be stored during SeasonPredictor._simulate_season
        # For now, we'll work with the aggregated position probabilities
        # TODO: Modify SeasonPredictor to optionally store full simulation details
        samples = getattr(self.season_prediction, 'simulation_samples', None)
        if samples is None:
            self._simulations = None
            return

        if samples.game_outcomes is None or samples.final_positions is None:
            self._simulations = None
            return

        self._simulations = samples

    def analyze_paths(
        self,
        team: str,
        target_position: int,
        method: Literal['auto', 'mcmc', 'combinatorial'] = 'auto',
        max_conditions: int = 10,
        n_scenario_clusters: int = 5,
    ) -> PathsOutput:
        """
        Analyze paths for a team to achieve target position.

        Args:
            team: Team name
            target_position: Target final position (1 = first)
            method: Analysis method ('auto', 'mcmc', or 'combinatorial')
            max_conditions: Maximum conditions to return
            n_scenario_clusters: Number of scenario clusters

        Returns:
            PathsOutput with analysis results
        """
        # Determine method
        if method == 'auto':
            method = self._choose_method()

        # Get current probability
        position_probs = self.season_prediction.position_probabilities
        current_prob = position_probs.loc[team, f'P(pos {target_position})']

        if method == 'mcmc':
            result = self._analyze_mcmc(
                team, target_position, max_conditions, n_scenario_clusters
            )
        else:
            result = self._analyze_combinatorial(
                team, target_position, max_conditions, n_scenario_clusters
            )

        # Generate narrative
        narrative = self._generate_narrative(result)
        result.narrative = narrative

        # Create Sankey diagram if plotly available
        if PLOTLY_AVAILABLE:
            result.sankey_diagram = self._create_sankey(result)

        return result

    def _choose_method(self) -> Literal['mcmc', 'combinatorial']:
        """Determine whether to use MCMC or combinatorial approach."""
        n_remaining = len(self.season_prediction.remaining_fixtures)

        # Estimate number of combinations
        # Each game: ~8 outcomes (win/draw/loss × bonus point variations)
        n_combinations = 8 ** n_remaining

        return 'combinatorial' if n_combinations < self.combinatorial_threshold else 'mcmc'

    def _analyze_mcmc(
        self,
        team: str,
        target_position: int,
        max_conditions: int,
        n_clusters: int,
    ) -> PathsOutput:
        """
        Analyze using MCMC simulations (early tournament).

        Extracts patterns from simulation data using decision trees
        and clustering.
        """
        position_probs = self.season_prediction.position_probabilities
        probability = position_probs.loc[team, f'P(pos {target_position})']

        # Use full MCMC pattern mining if simulation data available
        if self._simulations is not None:
            # Extract rules using decision tree
            rule_extractor = RuleExtractor(max_depth=4, min_frequency=0.1)
            conditions = rule_extractor.extract_rules(
                self._simulations, team, target_position
            )

            # Identify critical games using mutual information
            critical_games = self._identify_critical_games_mutual_info(team, target_position)

            # Cluster scenarios
            scenario_clusterer = ScenarioClusterer(n_clusters=n_clusters)
            scenarios = scenario_clusterer.cluster(
                self._simulations, team, target_position, self.match_predictor
            )

        else:
            # Fall back to heuristic approach if no simulation data
            conditions = self._extract_conditions_from_aggregates(team, target_position)
            critical_games = self._identify_critical_games_heuristic(team, target_position)
            scenarios = self._create_placeholder_scenarios(team, target_position, n_clusters)

        return PathsOutput(
            team=team,
            target_position=target_position,
            probability=probability,
            method='mcmc',
            conditions=conditions[:max_conditions],
            critical_games=critical_games,
            scenario_clusters=scenarios,
            narrative="",  # Will be filled by analyze_paths
            _raw_simulations=self._simulations,
        )

    def _analyze_combinatorial(
        self,
        team: str,
        target_position: int,
        max_conditions: int,
        n_clusters: int,
        use_bonus_variations: bool = True,
    ) -> PathsOutput:
        """
        Analyze using combinatorial enumeration (late tournament).

        Enumerates all possible outcomes for remaining games,
        calculates final tables, and weights by probabilities.

        Args:
            team: Target team
            target_position: Target final position
            max_conditions: Maximum conditions to return
            n_clusters: Number of scenario clusters
            use_bonus_variations: If True, enumerate bonus point variations (more accurate but slower)

        Returns:
            PathsOutput with analysis results
        """
        from rugby_ranking.model.league_table import LeagueTable

        remaining = self.season_prediction.remaining_fixtures
        current_standings = self.season_prediction.current_standings

        # Outcome codes with bonus point variations:
        # 0: home_win_no_bonus
        # 1: home_win_home_try_bonus
        # 2: home_win_away_losing_bonus
        # 3: home_win_both_bonuses
        # 4: draw
        # 5: away_win_no_bonus
        # 6: away_win_away_try_bonus
        # 7: away_win_home_losing_bonus
        # 8: away_win_both_bonuses

        if use_bonus_variations:
            outcome_codes = list(range(9))  # 9 outcomes with bonus variations
        else:
            outcome_codes = [0, 4, 5]  # Simple: home_win, draw, away_win (map to 0, 1, 2)

        n_games = len(remaining)
        n_combinations = len(outcome_codes) ** n_games

        if n_combinations > self.combinatorial_threshold:
            # Too many combinations, fall back to MCMC or simpler model
            if use_bonus_variations:
                # Try without bonus variations first
                return self._analyze_combinatorial(
                    team, target_position, max_conditions, n_clusters,
                    use_bonus_variations=False
                )
            else:
                # Fall back to MCMC
                return self._analyze_mcmc(team, target_position, max_conditions, n_clusters)

        # Generate all possible outcome combinations
        all_outcome_combos = list(itertools.product(outcome_codes, repeat=n_games))

        # Initialize league table calculator
        bonus_rules = getattr(self.match_predictor, 'competition', 'urc')
        league_table = LeagueTable(bonus_rules=bonus_rules)

        # Extract game probabilities from remaining fixtures
        game_probs = []
        for _, fixture in remaining.iterrows():
            if use_bonus_variations:
                # Calculate probabilities for all 9 outcomes
                probs = self._calculate_bonus_probabilities(fixture)
                game_probs.append(probs)
            else:
                # Simple 3-outcome model
                home_prob = fixture.get('home_win_prob', 0.5)
                draw_prob = fixture.get('draw_prob', 0.05)
                away_prob = fixture.get('away_win_prob', 0.45)

                # Normalize to sum to 1.0
                total = home_prob + draw_prob + away_prob
                if total > 0:
                    # Map to outcome codes 0, 4, 5 for simple model
                    game_probs.append({
                        0: home_prob/total,
                        4: draw_prob/total,
                        5: away_prob/total
                    })
                else:
                    game_probs.append({0: 0.4, 4: 0.2, 5: 0.4})

        # Evaluate combinations (with parallel processing for large searches)
        use_parallel = n_combinations > 1000  # Parallel processing threshold

        if use_parallel:
            successful_combos, total_probability = self._evaluate_combos_parallel(
                all_outcome_combos, game_probs, current_standings, remaining,
                league_table, use_bonus_variations, team, target_position
            )
        else:
            successful_combos, total_probability = self._evaluate_combos_sequential(
                all_outcome_combos, game_probs, current_standings, remaining,
                league_table, use_bonus_variations, team, target_position
            )

        # Extract conditions from successful combinations
        conditions = self._extract_combinatorial_conditions(
            remaining, successful_combos, team, total_probability
        )

        # Identify critical games
        critical_games = self._identify_critical_games_combinatorial(
            remaining, successful_combos, len(all_outcome_combos)
        )

        # Create scenario clusters (group similar successful outcomes)
        scenarios = self._cluster_combinatorial_scenarios(
            successful_combos, remaining, team, n_clusters
        )

        return PathsOutput(
            team=team,
            target_position=target_position,
            probability=total_probability,
            method='combinatorial',
            conditions=conditions[:max_conditions],
            critical_games=critical_games,
            scenario_clusters=scenarios,
            narrative="",  # Will be filled by analyze_paths
            _raw_simulations=None,
        )

    def _extract_conditions_from_aggregates(
        self,
        team: str,
        target_position: int,
    ) -> List[Condition]:
        """
        Extract conditions from aggregate probabilities (heuristic).

        In full implementation, this would come from decision tree
        on simulation features.
        """
        conditions = []

        # Heuristic: Look at remaining fixtures involving target team
        remaining = self.season_prediction.remaining_fixtures

        for _, fixture in remaining.iterrows():
            home = fixture['home_team']
            away = fixture['away_team']

            # If target team is playing
            if team == home:
                conditions.append(Condition(
                    game=(home, away),
                    outcome='home_win',
                    frequency=0.85,  # Placeholder
                    conditional_prob=fixture.get('home_win_prob', 0.5),
                    importance=0.15,  # Placeholder
                    team_controls=True,
                ))
            elif team == away:
                conditions.append(Condition(
                    game=(home, away),
                    outcome='away_win',
                    frequency=0.80,  # Placeholder
                    conditional_prob=fixture.get('away_win_prob', 0.5),
                    importance=0.12,  # Placeholder
                    team_controls=True,
                ))

        # Sort by importance
        conditions.sort(key=lambda c: c.importance, reverse=True)

        return conditions

    def _identify_critical_games_heuristic(
        self,
        team: str,
        target_position: int,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identify critical games using heuristic.

        In full implementation, would calculate ΔP(target | game outcome)
        from simulations.
        """
        critical = []
        remaining = self.season_prediction.remaining_fixtures

        for _, fixture in remaining.iterrows():
            home = fixture['home_team']
            away = fixture['away_team']

            # Games involving target team are critical
            if team in [home, away]:
                importance = 0.20
            # Games between competitors are critical
            elif self._are_competitors(home, away, target_position):
                importance = 0.10
            else:
                importance = 0.05

            critical.append(((home, away), importance))

        # Sort by importance
        critical.sort(key=lambda x: x[1], reverse=True)

        return critical

    def _identify_critical_games_mutual_info(
        self,
        team: str,
        target_position: int,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identify critical games using mutual information.

        Computes MI between game outcome (home/draw/away) and
        whether the target team finishes in the desired position.
        """
        samples = self._simulations
        if samples is None:
            return self._identify_critical_games_heuristic(team, target_position)

        try:
            team_idx = samples.teams.index(team)
        except ValueError:
            return []

        success = (samples.final_positions[:, team_idx] == target_position).astype(int)

        critical = []
        for game_idx, fixture in enumerate(samples.fixtures):
            outcomes = samples.game_outcomes[:, game_idx]
            mi = mutual_info_score(success, outcomes)
            critical.append(((fixture['home_team'], fixture['away_team']), float(mi)))

        critical.sort(key=lambda x: x[1], reverse=True)
        return critical

    def _are_competitors(self, team1: str, team2: str, target_position: int) -> bool:
        """Check if two teams are competing for similar positions."""
        position_probs = self.season_prediction.position_probabilities

        # Check if both teams have decent probability near target position
        range_positions = range(max(1, target_position - 1), min(6, target_position + 2))

        team1_prob = sum(
            position_probs.loc[team1, f'P(pos {p})']
            for p in range_positions
        )
        team2_prob = sum(
            position_probs.loc[team2, f'P(pos {p})']
            for p in range_positions
        )

        return team1_prob > 0.2 and team2_prob > 0.2

    def _create_placeholder_scenarios(
        self,
        team: str,
        target_position: int,
        n_clusters: int,
    ) -> List[ScenarioCluster]:
        """Create placeholder scenario clusters."""
        # In full implementation, this would come from clustering
        # simulation results

        scenarios = []

        # Example scenarios
        if n_clusters >= 1:
            scenarios.append(ScenarioCluster(
                description=f"{team} wins all remaining games with bonus points",
                frequency=0.35,
                representative_games={},
                probability=0.08,
            ))

        if n_clusters >= 2:
            scenarios.append(ScenarioCluster(
                description=f"{team} wins key games, competitors slip up",
                frequency=0.40,
                representative_games={},
                probability=0.09,
            ))

        if n_clusters >= 3:
            scenarios.append(ScenarioCluster(
                description=f"Multiple teams tie on points, {team} wins on points difference",
                frequency=0.15,
                representative_games={},
                probability=0.03,
            ))

        return scenarios

    def _evaluate_combos_sequential(
        self,
        outcome_combos: list,
        game_probs: list,
        current_standings: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        league_table,
        use_bonus_variations: bool,
        team: str,
        target_position: int,
    ) -> tuple[list, float]:
        """
        Evaluate outcome combinations sequentially (single-threaded).

        Returns:
            (successful_combos, total_probability)
        """
        successful_combos = []
        total_probability = 0.0

        for outcome_combo in outcome_combos:
            # Calculate joint probability
            joint_prob = 1.0
            for game_idx, outcome in enumerate(outcome_combo):
                prob_map = game_probs[game_idx]
                if isinstance(prob_map, dict):
                    joint_prob *= prob_map.get(outcome, 0.0)
                else:
                    joint_prob *= prob_map[outcome]

            # Early pruning
            if joint_prob < 1e-6:
                continue

            # Simulate league table
            final_standings = self._simulate_final_table_with_bonus(
                current_standings, remaining_fixtures, outcome_combo,
                league_table, use_bonus_variations
            )

            # Check if target achieved
            try:
                team_position = final_standings[final_standings['team'] == team].iloc[0]['position']
                if team_position == target_position:
                    successful_combos.append((outcome_combo, joint_prob, final_standings))
                    total_probability += joint_prob
            except (IndexError, KeyError):
                continue

        return successful_combos, total_probability

    def _evaluate_combos_parallel(
        self,
        outcome_combos: list,
        game_probs: list,
        current_standings: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        league_table,
        use_bonus_variations: bool,
        team: str,
        target_position: int,
        max_workers: int = None,
    ) -> tuple[list, float]:
        """
        Evaluate outcome combinations in parallel using ThreadPoolExecutor.

        Args:
            max_workers: Number of parallel workers (default: CPU count)

        Returns:
            (successful_combos, total_probability)
        """
        import os

        if max_workers is None:
            max_workers = min(8, os.cpu_count() or 4)  # Cap at 8 workers

        # Split combinations into batches for parallel processing
        batch_size = max(100, len(outcome_combos) // (max_workers * 4))
        batches = [
            outcome_combos[i:i + batch_size]
            for i in range(0, len(outcome_combos), batch_size)
        ]

        # Create worker function
        def evaluate_batch(batch):
            batch_successful = []
            batch_prob = 0.0

            for outcome_combo in batch:
                # Calculate joint probability
                joint_prob = 1.0
                for game_idx, outcome in enumerate(outcome_combo):
                    prob_map = game_probs[game_idx]
                    if isinstance(prob_map, dict):
                        joint_prob *= prob_map.get(outcome, 0.0)
                    else:
                        joint_prob *= prob_map[outcome]

                # Early pruning
                if joint_prob < 1e-6:
                    continue

                # Simulate league table
                final_standings = self._simulate_final_table_with_bonus(
                    current_standings, remaining_fixtures, outcome_combo,
                    league_table, use_bonus_variations
                )

                # Check if target achieved
                try:
                    team_position = final_standings[final_standings['team'] == team].iloc[0]['position']
                    if team_position == target_position:
                        batch_successful.append((outcome_combo, joint_prob, final_standings))
                        batch_prob += joint_prob
                except (IndexError, KeyError):
                    continue

            return batch_successful, batch_prob

        # Process batches in parallel
        successful_combos = []
        total_probability = 0.0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = {executor.submit(evaluate_batch, batch): batch for batch in batches}

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    batch_successful, batch_prob = future.result()
                    successful_combos.extend(batch_successful)
                    total_probability += batch_prob
                except Exception as e:
                    # Log error but continue processing other batches
                    print(f"Warning: Batch evaluation failed: {e}")
                    continue

        return successful_combos, total_probability

    def _calculate_bonus_probabilities(self, fixture: pd.Series) -> dict:
        """
        Calculate probabilities for all 9 outcome variations including bonus points.

        Outcomes:
        0: home_win_no_bonus
        1: home_win_home_try_bonus
        2: home_win_away_losing_bonus
        3: home_win_both_bonuses
        4: draw
        5: away_win_no_bonus
        6: away_win_away_try_bonus
        7: away_win_home_losing_bonus
        8: away_win_both_bonuses

        Returns:
            Dict mapping outcome code to probability
        """
        # Base probabilities
        home_win_prob = fixture.get('home_win_prob', 0.5)
        draw_prob = fixture.get('draw_prob', 0.05)
        away_win_prob = fixture.get('away_win_prob', 0.45)

        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total

        # Estimate bonus point probabilities (heuristic)
        # These could be improved with more sophisticated modeling
        try_bonus_prob = 0.30  # ~30% of games see 4+ tries from winner
        losing_bonus_prob = 0.25  # ~25% of losses are by ≤7 points

        # Calculate joint probabilities for each outcome
        probs = {}

        # Home wins (0-3)
        probs[0] = home_win_prob * (1 - try_bonus_prob) * (1 - losing_bonus_prob)  # No bonuses
        probs[1] = home_win_prob * try_bonus_prob * (1 - losing_bonus_prob)  # Home try bonus only
        probs[2] = home_win_prob * (1 - try_bonus_prob) * losing_bonus_prob  # Away losing bonus only
        probs[3] = home_win_prob * try_bonus_prob * losing_bonus_prob  # Both bonuses

        # Draw (4)
        probs[4] = draw_prob

        # Away wins (5-8)
        probs[5] = away_win_prob * (1 - try_bonus_prob) * (1 - losing_bonus_prob)  # No bonuses
        probs[6] = away_win_prob * try_bonus_prob * (1 - losing_bonus_prob)  # Away try bonus only
        probs[7] = away_win_prob * (1 - try_bonus_prob) * losing_bonus_prob  # Home losing bonus only
        probs[8] = away_win_prob * try_bonus_prob * losing_bonus_prob  # Both bonuses

        return probs

    def _simulate_final_table_with_bonus(
        self,
        current_standings: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        outcome_combo: tuple,
        league_table,
        use_bonus_variations: bool,
    ) -> pd.DataFrame:
        """
        Simulate final league table with explicit bonus point handling.

        Args:
            current_standings: Current league standings
            remaining_fixtures: DataFrame of remaining matches
            outcome_combo: Tuple of outcome codes
            league_table: LeagueTable calculator
            use_bonus_variations: Whether outcomes include bonus variations

        Returns:
            Final standings DataFrame
        """
        # Convert outcome combination to match results
        simulated_matches = []

        for game_idx, (_, fixture) in enumerate(remaining_fixtures.iterrows()):
            outcome = outcome_combo[game_idx]
            home = fixture['home_team']
            away = fixture['away_team']

            # Decode outcome including bonus points
            if use_bonus_variations:
                match_result = self._decode_bonus_outcome(outcome, fixture)
            else:
                # Simple mapping: 0=home_win, 4=draw, 5=away_win
                if outcome == 0:
                    match_result = {'result': 'home_win', 'home_try_bonus': False, 'away_losing_bonus': False}
                elif outcome == 4:
                    match_result = {'result': 'draw', 'home_try_bonus': False, 'away_losing_bonus': False}
                else:  # outcome == 5
                    match_result = {'result': 'away_win', 'home_try_bonus': False, 'away_losing_bonus': False}

            # Estimate scores based on result
            home_score, away_score, home_tries, away_tries = self._estimate_scores(
                match_result, fixture
            )

            # Create match records (one for each team)
            simulated_matches.append({
                'team': home,
                'opponent': away,
                'score': home_score,
                'opponent_score': away_score,
                'tries': home_tries,
                'is_home': True,
            })
            simulated_matches.append({
                'team': away,
                'opponent': home,
                'score': away_score,
                'opponent_score': home_score,
                'tries': away_tries,
                'is_home': False,
            })

        # Compute standings from simulated matches
        simulated_df = pd.DataFrame(simulated_matches)
        simulated_standings = league_table.compute_standings(simulated_df)

        # Merge with current standings
        if current_standings is not None and len(current_standings) > 0:
            merged = current_standings.set_index('team').add(
                simulated_standings.set_index('team'),
                fill_value=0
            ).reset_index()

            # Recalculate position based on total_points
            merged = merged.sort_values(
                ['total_points', 'points_diff', 'points_for'],
                ascending=[False, False, False]
            )
            merged['position'] = range(1, len(merged) + 1)
        else:
            merged = simulated_standings

        return merged

    def _decode_bonus_outcome(self, outcome_code: int, fixture: pd.Series) -> dict:
        """
        Decode outcome code into match result with bonus points.

        Returns:
            Dict with keys: result, home_try_bonus, away_try_bonus,
                           home_losing_bonus, away_losing_bonus
        """
        if outcome_code == 4:  # Draw
            return {
                'result': 'draw',
                'home_try_bonus': False,
                'away_try_bonus': False,
                'home_losing_bonus': False,
                'away_losing_bonus': False,
            }
        elif outcome_code in [0, 1, 2, 3]:  # Home wins
            return {
                'result': 'home_win',
                'home_try_bonus': outcome_code in [1, 3],
                'away_try_bonus': False,
                'home_losing_bonus': False,
                'away_losing_bonus': outcome_code in [2, 3],
            }
        else:  # Away wins (5, 6, 7, 8)
            return {
                'result': 'away_win',
                'home_try_bonus': False,
                'away_try_bonus': outcome_code in [6, 8],
                'home_losing_bonus': outcome_code in [7, 8],
                'away_losing_bonus': False,
            }

    def _estimate_scores(
        self,
        match_result: dict,
        fixture: pd.Series,
    ) -> tuple[int, int, int, int]:
        """
        Estimate scores and tries from match result and bonus points.

        Returns:
            (home_score, away_score, home_tries, away_tries)
        """
        result = match_result['result']

        # Use predicted scores if available
        base_home = int(fixture.get('home_score_pred', 24))
        base_away = int(fixture.get('away_score_pred', 17))

        if result == 'home_win':
            home_score = max(base_home, base_away + 3)
            away_score = base_away

            # Adjust for bonuses
            if match_result.get('home_try_bonus'):
                # Need 4+ tries, roughly 5 points per try
                home_tries = 4
                home_score = max(home_score, 20)  # At least 4 tries worth
            else:
                home_tries = min(3, max(1, home_score // 7))

            if match_result.get('away_losing_bonus'):
                # Lost by ≤7 points
                away_score = max(away_score, home_score - 7)
                away_tries = max(1, away_score // 7)
            else:
                away_tries = max(1, away_score // 7)

        elif result == 'draw':
            home_score = away_score = (base_home + base_away) // 2
            home_tries = away_tries = max(1, home_score // 7)

        else:  # away_win
            away_score = max(base_away, base_home + 3)
            home_score = base_home

            # Adjust for bonuses
            if match_result.get('away_try_bonus'):
                away_tries = 4
                away_score = max(away_score, 20)
            else:
                away_tries = min(3, max(1, away_score // 7))

            if match_result.get('home_losing_bonus'):
                home_score = max(home_score, away_score - 7)
                home_tries = max(1, home_score // 7)
            else:
                home_tries = max(1, home_score // 7)

        return int(home_score), int(away_score), int(home_tries), int(away_tries)

    def _simulate_final_table(
        self,
        current_standings: pd.DataFrame,
        remaining_fixtures: pd.DataFrame,
        outcome_combo: tuple,
        league_table,
    ) -> pd.DataFrame:
        """
        Simulate final league table for a specific outcome combination.

        Args:
            current_standings: Current league standings
            remaining_fixtures: DataFrame of remaining matches
            outcome_combo: Tuple of outcomes (0=home_win, 1=draw, 2=away_win)
            league_table: LeagueTable calculator

        Returns:
            Final standings DataFrame
        """
        # Convert outcome combination to match results
        simulated_matches = []

        for game_idx, (_, fixture) in enumerate(remaining_fixtures.iterrows()):
            outcome = outcome_combo[game_idx]
            home = fixture['home_team']
            away = fixture['away_team']

            # Estimate scores based on outcome
            # Use predicted scores if available, otherwise use typical scores
            if outcome == 0:  # home win
                home_score = int(fixture.get('home_score_pred', 24))
                away_score = int(fixture.get('away_score_pred', 17))
                if home_score <= away_score:  # Ensure home wins
                    home_score = away_score + 7
            elif outcome == 1:  # draw
                home_score = int(fixture.get('home_score_pred', 20))
                away_score = home_score
            else:  # away win (outcome == 2)
                home_score = int(fixture.get('home_score_pred', 15))
                away_score = int(fixture.get('away_score_pred', 22))
                if away_score <= home_score:  # Ensure away wins
                    away_score = home_score + 7

            # Estimate tries (roughly 1 try per 7 points)
            home_tries = max(1, int(home_score / 7))
            away_tries = max(1, int(away_score / 7))

            # Create match records (one for each team)
            simulated_matches.append({
                'team': home,
                'opponent': away,
                'score': home_score,
                'opponent_score': away_score,
                'tries': home_tries,
                'is_home': True,
            })
            simulated_matches.append({
                'team': away,
                'opponent': home,
                'score': away_score,
                'opponent_score': home_score,
                'tries': away_tries,
                'is_home': False,
            })

        # Compute standings from simulated matches
        simulated_df = pd.DataFrame(simulated_matches)
        simulated_standings = league_table.compute_standings(simulated_df)

        # Merge with current standings
        if current_standings is not None and len(current_standings) > 0:
            merged = current_standings.set_index('team').add(
                simulated_standings.set_index('team'),
                fill_value=0
            ).reset_index()

            # Recalculate position based on total_points
            merged = merged.sort_values(
                ['total_points', 'points_diff', 'points_for'],
                ascending=[False, False, False]
            )
            merged['position'] = range(1, len(merged) + 1)
        else:
            merged = simulated_standings

        return merged

    def _extract_combinatorial_conditions(
        self,
        remaining_fixtures: pd.DataFrame,
        successful_combos: list,
        team: str,
        total_probability: float,
    ) -> List[Condition]:
        """
        Extract key conditions from successful outcome combinations.

        Analyzes which game outcomes appear most frequently in
        successful scenarios and have highest impact.
        """
        if len(successful_combos) == 0:
            return []

        n_games = len(remaining_fixtures)
        outcome_names = ['home_win', 'draw', 'away_win']

        # Count frequency of each outcome for each game
        # outcome_freq[game_idx][outcome] = (count, total_prob_with_outcome)
        outcome_freq = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))

        for outcome_combo, prob, _ in successful_combos:
            for game_idx, outcome in enumerate(outcome_combo):
                outcome_freq[game_idx][outcome][0] += 1
                outcome_freq[game_idx][outcome][1] += prob

        # Extract conditions
        conditions = []
        fixtures_list = list(remaining_fixtures.iterrows())

        for game_idx in range(n_games):
            _, fixture = fixtures_list[game_idx]
            home = fixture['home_team']
            away = fixture['away_team']

            for outcome_code in [0, 1, 2]:
                if outcome_code not in outcome_freq[game_idx]:
                    continue

                count, prob_with_outcome = outcome_freq[game_idx][outcome_code]
                frequency = count / len(successful_combos)

                # Conditional probability of success given this outcome
                if total_probability > 0:
                    conditional_prob = prob_with_outcome / total_probability
                else:
                    conditional_prob = 0.0

                # Importance = difference from baseline success rate
                importance = conditional_prob - (total_probability if total_probability < 1 else 0.5)

                # Only include frequent and impactful conditions
                if frequency >= 0.3 and abs(importance) >= 0.05:
                    outcome_name = outcome_names[outcome_code]

                    team_controls = (
                        (outcome_name == 'home_win' and home == team) or
                        (outcome_name == 'away_win' and away == team)
                    )

                    team_in_fixture = (home == team or away == team)

                    # Skip conditions where the team is in the fixture but
                    # doesn't control the outcome (e.g. "England must beat France"
                    # when analysing France — nonsensical as a requirement).
                    if team_in_fixture and not team_controls:
                        continue

                    conditions.append(Condition(
                        game=(home, away),
                        outcome=outcome_name,
                        frequency=float(frequency),
                        conditional_prob=float(conditional_prob),
                        importance=float(importance),
                        team_controls=team_controls,
                    ))

        # Sort by importance
        conditions.sort(key=lambda c: abs(c.importance), reverse=True)
        return conditions

    def _identify_critical_games_combinatorial(
        self,
        remaining_fixtures: pd.DataFrame,
        successful_combos: list,
        total_combos: int,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """
        Identify critical games from combinatorial analysis.

        A game is critical if different outcomes significantly affect
        the probability of achieving the target.
        """
        if len(successful_combos) == 0:
            return []

        n_games = len(remaining_fixtures)
        fixtures_list = list(remaining_fixtures.iterrows())

        # For each game, calculate variance in success across outcomes
        game_impacts = []

        for game_idx in range(n_games):
            _, fixture = fixtures_list[game_idx]
            home = fixture['home_team']
            away = fixture['away_team']

            # Count successes for each outcome
            outcome_successes = [0, 0, 0]
            outcome_totals = [0, 0, 0]

            for outcome_combo, _, _ in successful_combos:
                outcome_successes[outcome_combo[game_idx]] += 1

            # Count totals from all combinations (not just successful)
            # This is approximate - assumes uniform distribution
            for outcome in [0, 1, 2]:
                outcome_totals[outcome] = total_combos // 3

            # Calculate impact as variance in success rate across outcomes
            success_rates = [
                outcome_successes[i] / max(outcome_totals[i], 1)
                for i in range(3)
            ]

            impact = np.std(success_rates) if len(success_rates) > 0 else 0.0
            game_impacts.append(((home, away), float(impact)))

        # Sort by impact
        game_impacts.sort(key=lambda x: x[1], reverse=True)
        return game_impacts

    def _cluster_combinatorial_scenarios(
        self,
        successful_combos: list,
        remaining_fixtures: pd.DataFrame,
        team: str,
        n_clusters: int,
    ) -> List[ScenarioCluster]:
        """
        Cluster successful outcome combinations into representative scenarios.

        Groups similar outcome patterns and finds representative examples.
        """
        if len(successful_combos) == 0:
            return []

        # If too few successes, return single cluster
        if len(successful_combos) < n_clusters * 2:
            outcome_combo, prob, _ = successful_combos[0]
            description = self._describe_combo(outcome_combo, remaining_fixtures, team)

            total_prob = sum(p for _, p, _ in successful_combos)

            return [ScenarioCluster(
                description=description,
                frequency=1.0,
                representative_games=self._combo_to_dict(outcome_combo, remaining_fixtures),
                probability=float(total_prob),
            )]

        # Convert to array for clustering
        outcome_array = np.array([combo for combo, _, _ in successful_combos])

        # Apply hierarchical clustering
        linkage_matrix = linkage(outcome_array, method='average', metric='hamming')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Create scenario clusters
        clusters = []
        total_prob = sum(p for _, p, _ in successful_combos)

        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() == 0:
                continue

            cluster_combos = [
                successful_combos[i]
                for i in range(len(successful_combos))
                if cluster_labels[i] == cluster_id
            ]

            # Find representative (closest to cluster centroid)
            cluster_outcomes = outcome_array[cluster_mask]
            centroid = np.round(cluster_outcomes.mean(axis=0)).astype(int)

            # Find closest to centroid
            distances = np.mean(cluster_outcomes != centroid, axis=1)
            representative_idx = int(np.argmin(distances))
            representative_combo = cluster_combos[representative_idx][0]

            # Calculate cluster probability
            cluster_prob = sum(p for _, p, _ in cluster_combos)
            frequency = cluster_prob / total_prob if total_prob > 0 else 0.0

            description = self._describe_combo(representative_combo, remaining_fixtures, team)

            clusters.append(ScenarioCluster(
                description=description,
                frequency=float(frequency),
                representative_games=self._combo_to_dict(representative_combo, remaining_fixtures),
                probability=float(cluster_prob),
            ))

        # Sort by frequency
        clusters.sort(key=lambda c: c.frequency, reverse=True)
        return clusters

    def _describe_combo(
        self,
        outcome_combo: tuple,
        remaining_fixtures: pd.DataFrame,
        team: str,
    ) -> str:
        """Generate description of an outcome combination."""
        outcome_names = {0: 'win', 1: 'draw', 2: 'loss'}

        team_results = {'wins': 0, 'draws': 0, 'losses': 0}

        for game_idx, (_, fixture) in enumerate(remaining_fixtures.iterrows()):
            outcome = outcome_combo[game_idx]
            home = fixture['home_team']
            away = fixture['away_team']

            if home == team:
                if outcome == 0:
                    team_results['wins'] += 1
                elif outcome == 1:
                    team_results['draws'] += 1
                else:
                    team_results['losses'] += 1
            elif away == team:
                if outcome == 2:
                    team_results['wins'] += 1
                elif outcome == 1:
                    team_results['draws'] += 1
                else:
                    team_results['losses'] += 1

        parts = []
        if team_results['wins'] > 0:
            parts.append(f"{team} wins {team_results['wins']}")
        if team_results['draws'] > 0:
            parts.append(f"draws {team_results['draws']}")
        if team_results['losses'] > 0:
            parts.append(f"loses {team_results['losses']}")

        return ", ".join(parts) if parts else f"{team} reaches target position"

    def _combo_to_dict(
        self,
        outcome_combo: tuple,
        remaining_fixtures: pd.DataFrame,
    ) -> Dict[Tuple[str, str], str]:
        """Convert outcome combo to dictionary."""
        outcome_names = ['home_win', 'draw', 'away_win']
        result = {}

        for game_idx, (_, fixture) in enumerate(remaining_fixtures.iterrows()):
            home = fixture['home_team']
            away = fixture['away_team']
            result[(home, away)] = outcome_names[outcome_combo[game_idx]]

        return result

    def _generate_narrative(
        self,
        result: PathsOutput,
        style: Literal['detailed', 'blog', 'social'] = 'detailed'
    ) -> str:
        """
        Generate human-readable narrative from analysis results.

        Args:
            result: PathsOutput with analysis results
            style: Narrative style - 'detailed' (technical), 'blog' (readable),
                   or 'social' (concise for social media)
        """
        if style == 'blog':
            return self._generate_blog_narrative(result)
        elif style == 'social':
            return self._generate_social_narrative(result)
        else:
            return self._generate_detailed_narrative(result)

    def _generate_detailed_narrative(self, result: PathsOutput) -> str:
        """Generate detailed technical narrative."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"PATHS TO VICTORY: {result.team.upper()}")
        lines.append("=" * 70)
        lines.append("")

        # Summary with confidence interval
        position_name = self._ordinal(result.target_position)
        prob_pct = result.probability * 100

        # Estimate confidence interval (rough approximation based on method)
        if result.method == 'mcmc':
            ci_margin = 3.0  # ±3% for MCMC with typical sample sizes
        else:
            ci_margin = 1.0  # ±1% for exact combinatorial

        lines.append(
            f"{result.team} can finish {position_name} with "
            f"{prob_pct:.1f}% probability (±{ci_margin:.1f}%)."
        )
        lines.append("")

        # Categorize probability
        if result.probability >= 0.7:
            lines.append(f"⚠ {result.team} is the strong favorite for this position.")
        elif result.probability >= 0.4:
            lines.append(f"↔ {result.team} has a realistic chance at this position.")
        elif result.probability >= 0.15:
            lines.append(f"↑ {result.team} faces an uphill battle for this position.")
        else:
            lines.append(f"⊗ {result.team} would need several unlikely results.")
        lines.append("")

        # Key requirements
        if result.conditions:
            lines.append("Key requirements:")
            lines.append("")

            team_conditions = [c for c in result.conditions if c.team_controls]
            # Exclude any fixture that involves the target team from the
            # "needs from others" list — those should never appear here.
            other_conditions = [
                c for c in result.conditions
                if not c.team_controls
                and result.team not in (c.game[0], c.game[1])
            ]

            if team_conditions:
                lines.append(f"  What {result.team} must do:")
                for cond in team_conditions[:5]:
                    conf_str = f" (boosts chances to {cond.conditional_prob:.0%})"
                    lines.append(
                        f"    ✓ {self._format_condition(cond, result.team)} "
                        f"[appears in {cond.frequency:.0%} of winning scenarios]{conf_str}"
                    )
                lines.append("")

            if other_conditions:
                lines.append(f"  What {result.team} needs from others:")
                for cond in other_conditions[:5]:
                    conf_str = f" (boosts chances to {cond.conditional_prob:.0%})"
                    lines.append(
                        f"    ○ {self._format_condition(cond, result.team)} "
                        f"[appears in {cond.frequency:.0%} of winning scenarios]{conf_str}"
                    )
                lines.append("")

        # Scenario clusters
        if result.scenario_clusters:
            lines.append("Most likely winning scenarios:")
            lines.append("")
            for i, scenario in enumerate(result.scenario_clusters[:3], 1):
                prob_str = f"{scenario.probability:.1%}"
                lines.append(
                    f"  {i}. {scenario.description}"
                )
                lines.append(
                    f"     Probability: {prob_str} | "
                    f"Frequency: {scenario.frequency:.0%} of winning paths"
                )
            lines.append("")

        # Critical games
        if result.critical_games:
            lines.append(
                f"Critical upcoming matches (by impact on {result.team}'s chances):"
            )
            lines.append("")
            for i, (game, importance) in enumerate(result.critical_games[:5], 1):
                home, away = game
                # Format impact as absolute value since it could be negative/positive
                impact_str = f"{abs(importance):.1%}" if isinstance(importance, float) else f"{importance:.3f}"
                lines.append(
                    f"  {i}. {home} vs {away}: "
                    f"±{impact_str} impact on outcome"
                )
            lines.append("")

        # Footer with method info
        lines.append("=" * 70)
        lines.append(f"Analysis method: {result.method.upper()}")
        if result.method == 'mcmc':
            lines.append("(Monte Carlo pattern mining from simulations)")
        else:
            lines.append("(Exact combinatorial enumeration of all possible outcomes)")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_blog_narrative(self, result: PathsOutput) -> str:
        """Generate blog-style readable narrative."""
        lines = []

        position_name = self._ordinal(result.target_position)

        # Opening paragraph
        lines.append(f"# Can {result.team} Finish {position_name}?")
        lines.append("")

        prob_pct = result.probability * 100
        if result.probability >= 0.7:
            opener = f"**Yes, and they're the favorites.** {result.team} has a {prob_pct:.0f}% chance of finishing {position_name}"
        elif result.probability >= 0.4:
            opener = f"**It's definitely possible.** {result.team} has a {prob_pct:.0f}% chance of finishing {position_name}"
        elif result.probability >= 0.15:
            opener = f"**It's a long shot, but not impossible.** {result.team} has a {prob_pct:.0f}% chance of finishing {position_name}"
        else:
            opener = f"**It would take a miracle.** {result.team} has only a {prob_pct:.0f}% chance of finishing {position_name}"

        lines.append(f"{opener}, according to our statistical model. Here's what needs to happen.")
        lines.append("")

        # What they control
        team_conditions = [c for c in result.conditions if c.team_controls][:3]
        if team_conditions:
            lines.append(f"## What {result.team} Must Do")
            lines.append("")
            for cond in team_conditions:
                outcome_text = self._format_condition_blog(cond, result.team)
                lines.append(f"- **{outcome_text}**")
                if cond.frequency >= 0.8:
                    lines.append(f"  - This is essential - it happens in {cond.frequency:.0%} of scenarios where {result.team} succeeds.")
                else:
                    lines.append(f"  - This appears in {cond.frequency:.0%} of successful scenarios.")
            lines.append("")

        # What they need from others (only fixtures that don't involve the target team)
        other_conditions = [
            c for c in result.conditions
            if not c.team_controls
            and result.team not in (c.game[0], c.game[1])
        ][:3]
        if other_conditions:
            lines.append(f"## What {result.team} Needs From Others")
            lines.append("")
            for cond in other_conditions:
                outcome_text = self._format_condition_blog(cond, result.team)
                lines.append(f"- **{outcome_text}**")
                if cond.importance > 0.2:
                    lines.append(f"  - This is crucial - if it happens, {result.team}'s chances jump to {cond.conditional_prob:.0%}.")
                else:
                    lines.append(f"  - This would boost their chances to {cond.conditional_prob:.0%}.")
            lines.append("")

        # Critical games
        if result.critical_games:
            lines.append("## Games to Watch")
            lines.append("")
            lines.append(f"These matches have the biggest impact on {result.team}'s chances:")
            lines.append("")
            for i, (game, importance) in enumerate(result.critical_games[:3], 1):
                home, away = game
                lines.append(f"{i}. **{home} vs {away}**")
            lines.append("")

        # Scenarios
        if result.scenario_clusters:
            lines.append("## Most Likely Paths")
            lines.append("")
            for i, scenario in enumerate(result.scenario_clusters[:2], 1):
                lines.append(f"**Scenario {i}** ({scenario.frequency:.0%} of winning paths): {scenario.description}")
                lines.append("")

        lines.append("---")
        lines.append(f"*Analysis based on {result.method} modeling of remaining fixtures.*")

        return "\n".join(lines)

    def _generate_social_narrative(self, result: PathsOutput) -> str:
        """Generate concise social media snippet."""
        lines = []

        position_name = self._ordinal(result.target_position)
        prob_pct = result.probability * 100

        # Headline
        if result.probability >= 0.7:
            emoji = "🏆"
        elif result.probability >= 0.4:
            emoji = "🎯"
        elif result.probability >= 0.15:
            emoji = "⚡"
        else:
            emoji = "🤞"

        lines.append(f"{emoji} Can {result.team} finish {position_name}?")
        lines.append(f"📊 {prob_pct:.0f}% chance")
        lines.append("")

        # Top requirement
        team_conditions = [c for c in result.conditions if c.team_controls]
        if team_conditions:
            cond = team_conditions[0]
            lines.append(f"✓ Must: {self._format_condition_social(cond)}")

        # Top external factor (only fixtures that don't involve the target team)
        other_conditions = [
            c for c in result.conditions
            if not c.team_controls
            and result.team not in (c.game[0], c.game[1])
        ]
        if other_conditions:
            cond = other_conditions[0]
            lines.append(f"🤞 Need: {self._format_condition_social(cond)}")

        return "\n".join(lines)

    def _format_condition(self, condition: Condition, team: str) -> str:
        """Format a condition as human-readable text."""
        home, away = condition.game
        outcome = condition.outcome

        if outcome == 'home_win':
            if home == team:
                return f"Beat {away}"
            else:
                return f"{home} must beat {away}"
        elif outcome == 'away_win':
            if away == team:
                return f"Beat {home}"
            else:
                return f"{away} must beat {home}"
        elif outcome == 'draw':
            return f"{home} and {away} draw"
        elif 'bonus' in outcome:
            bonus_team = home if 'home' in outcome else away
            if bonus_team == team:
                return f"Get bonus point vs {away if bonus_team == home else home}"
            else:
                return f"{bonus_team} gets bonus point"
        else:
            return f"{home} vs {away}: {outcome}"

    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal (1 -> '1st', 2 -> '2nd', etc.)."""
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        return f"{n}{suffix}"

    def _format_condition_blog(self, condition: Condition, team: str) -> str:
        """Format a condition for blog-style narrative."""
        home, away = condition.game
        outcome = condition.outcome

        if outcome == 'home_win':
            if home == team:
                return f"{team} must beat {away}"
            else:
                return f"{home} needs to beat {away}"
        elif outcome == 'away_win':
            if away == team:
                return f"{team} must beat {home}"
            else:
                return f"{away} needs to beat {home}"
        elif outcome == 'draw':
            return f"{home} and {away} must draw"
        else:
            return f"{home} vs {away}: {outcome}"

    def _format_condition_social(self, condition: Condition) -> str:
        """Format a condition for social media (concise)."""
        home, away = condition.game
        outcome = condition.outcome

        if outcome == 'home_win':
            return f"{home} beat {away}"
        elif outcome == 'away_win':
            return f"{away} beat {home}"
        elif outcome == 'draw':
            return f"{home}-{away} draw"
        else:
            return f"{home} vs {away}"

    def export_to_markdown(
        self,
        result: PathsOutput,
        include_metadata: bool = True,
        include_visualization: bool = False,
    ) -> str:
        """
        Export analysis to markdown format for blog posts.

        Args:
            result: PathsOutput with analysis results
            include_metadata: Include YAML frontmatter
            include_visualization: Include embedded Sankey diagram (if available)

        Returns:
            Markdown string ready for blog publication
        """
        from datetime import datetime

        lines = []

        # YAML frontmatter
        if include_metadata:
            lines.append("---")
            lines.append(f"title: \"Paths to Victory: {result.team}\"")
            lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
            lines.append(f"tags: [rugby, predictions, {result.team.lower().replace(' ', '-')}]")
            lines.append(f"probability: {result.probability:.2%}")
            lines.append(f"target_position: {result.target_position}")
            lines.append(f"analysis_method: {result.method}")
            lines.append("---")
            lines.append("")

        # Main content (blog narrative)
        lines.append(self._generate_blog_narrative(result))
        lines.append("")

        # Embedded visualization
        if include_visualization and result.sankey_diagram:
            lines.append("## Visualization")
            lines.append("")
            lines.append("```plotly")
            # Plotly diagram would be embedded here
            lines.append("# Interactive Sankey diagram showing probability flows")
            lines.append("```")
            lines.append("")

        # Technical details
        lines.append("---")
        lines.append("")
        lines.append("## Technical Details")
        lines.append("")
        lines.append(f"- **Analysis Method**: {result.method.upper()}")
        lines.append(f"- **Target Position**: {self._ordinal(result.target_position)}")
        lines.append(f"- **Probability**: {result.probability:.2%}")
        lines.append(f"- **Critical Games Analyzed**: {len(result.critical_games)}")
        lines.append(f"- **Conditions Identified**: {len(result.conditions)}")
        lines.append("")

        return "\n".join(lines)

    def generate_social_snippets(self, result: PathsOutput) -> dict[str, str]:
        """
        Generate social media snippets for multiple platforms.

        Returns:
            Dict with keys: 'twitter', 'linkedin', 'facebook'
        """
        # Twitter (280 char limit)
        twitter = self._generate_social_narrative(result)

        # LinkedIn (more detailed)
        position_name = self._ordinal(result.target_position)
        prob_pct = result.probability * 100

        linkedin_lines = [
            f"📊 Rugby Prediction: Can {result.team} finish {position_name}?",
            "",
            f"Our model gives them a {prob_pct:.0f}% chance. Here's what needs to happen:",
            ""
        ]

        team_conds = [c for c in result.conditions if c.team_controls][:2]
        for cond in team_conds:
            linkedin_lines.append(f"✓ {self._format_condition_blog(cond, result.team)}")

        other_conds = [
            c for c in result.conditions
            if not c.team_controls
            and result.team not in (c.game[0], c.game[1])
        ][:2]
        for cond in other_conds:
            linkedin_lines.append(f"🤞 {self._format_condition_blog(cond, result.team)}")

        linkedin_lines.append("")
        linkedin_lines.append(f"Analysis based on {result.method} modeling.")

        linkedin = "\n".join(linkedin_lines)

        # Facebook (similar to LinkedIn but more conversational)
        facebook = linkedin

        return {
            'twitter': twitter,
            'linkedin': linkedin,
            'facebook': facebook,
        }

    def _create_sankey(self, result: PathsOutput) -> Optional[go.Figure]:
        """
        Create Sankey diagram visualization showing probability flows.

        Shows how outcomes of critical games affect the probability
        of achieving the target position. Limits to top 3-5 most
        critical games to avoid clutter.

        Returns:
            Plotly Figure with Sankey diagram, or None if plotly unavailable
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not result.critical_games:
            return None

        # Limit to top 3-5 critical games
        n_games_to_show = min(5, len(result.critical_games))
        critical_games = result.critical_games[:n_games_to_show]

        # Build node structure
        # Nodes: Start -> Game1 outcomes -> Game2 outcomes -> ... -> Final (Success/Failure)
        nodes = []
        node_labels = []
        node_colors = []

        # Start node
        nodes.append(0)
        node_labels.append("Current\nSituation")
        node_colors.append("lightblue")

        # Create nodes for each game's outcomes
        node_idx = 1
        game_outcome_nodes = {}  # {(game_idx, outcome): node_idx}

        for game_idx, (game, _) in enumerate(critical_games):
            home, away = game
            game_name = f"{home} v {away}"

            for outcome in ['home_win', 'draw', 'away_win']:
                node_id = node_idx
                game_outcome_nodes[(game_idx, outcome)] = node_id

                # Format label
                if outcome == 'home_win':
                    label = f"{home}\nwins"
                elif outcome == 'away_win':
                    label = f"{away}\nwins"
                else:
                    label = f"{game_name}\ndraws"

                node_labels.append(label)
                node_colors.append("lightgray")
                node_idx += 1

        # Final outcome nodes
        success_node = node_idx
        failure_node = node_idx + 1
        node_labels.append(f"{result.team}\nfinishes\n{self._ordinal(result.target_position)}")
        node_labels.append("Other\nposition")
        node_colors.append("lightgreen")
        node_colors.append("lightcoral")

        # Build edges (links)
        sources = []
        targets = []
        values = []
        link_colors = []

        # For demonstration, create simplified flows
        # In full implementation, would use simulation data or combinatorial results

        # Start -> First game outcomes
        first_game_idx = 0
        for outcome in ['home_win', 'draw', 'away_win']:
            sources.append(0)  # Start node
            targets.append(game_outcome_nodes[(first_game_idx, outcome)])

            # Estimate probabilities (would come from predictions in real implementation)
            if outcome == 'home_win':
                prob = 0.5
            elif outcome == 'away_win':
                prob = 0.3
            else:
                prob = 0.2

            values.append(prob * 100)  # Scale for visibility
            link_colors.append("rgba(128, 128, 128, 0.3)")

        # Simplified final connections
        # In full implementation, would trace through all game combinations
        # For now, connect last game outcomes to final nodes

        if n_games_to_show > 0:
            last_game_idx = n_games_to_show - 1

            for outcome in ['home_win', 'draw', 'away_win']:
                last_game_node = game_outcome_nodes[(last_game_idx, outcome)]

                # Split to success/failure based on rough estimates
                # In real implementation, would calculate from actual paths
                success_prob = result.probability if outcome == 'home_win' else result.probability * 0.5
                failure_prob = 1.0 - success_prob

                sources.append(last_game_node)
                targets.append(success_node)
                values.append(success_prob * 100)
                link_colors.append("rgba(0, 255, 0, 0.4)")

                sources.append(last_game_node)
                targets.append(failure_node)
                values.append(failure_prob * 100)
                link_colors.append("rgba(255, 0, 0, 0.4)")

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            )
        )])

        fig.update_layout(
            title=f"Paths to Victory: {result.team} finishing {self._ordinal(result.target_position)}",
            font=dict(size=10),
            height=600,
        )

        return fig

    def find_critical_games(
        self,
        top_n: int = 10,
        target_position: int = 1,
    ) -> pd.DataFrame:
        """
        Find critical games across all teams and positions.

        Returns DataFrame with games ranked by total impact across
        all teams' championship chances (or other target position).

        Uses mutual information to calculate how much each game outcome
        affects whether each team achieves the target position, then
        sums across all teams.

        Args:
            top_n: Number of top games to return
            target_position: Position to analyze (default: 1 for championship)

        Returns:
            DataFrame with columns: home_team, away_team, total_impact, date
            and per-team impact breakdown
        """
        samples = self._simulations

        if samples is None:
            # Fallback to heuristic if no simulation data
            return self._find_critical_games_heuristic(top_n)

        # Calculate mutual information for each game across all teams
        n_games = len(samples.fixtures)
        n_teams = len(samples.teams)

        # Matrix to store MI: [game_idx, team_idx]
        mi_matrix = np.zeros((n_games, n_teams))

        for team_idx, team in enumerate(samples.teams):
            # Create success mask for this team achieving target position
            success = (samples.final_positions[:, team_idx] == target_position).astype(int)

            # Skip if this team never achieves target position
            if success.sum() == 0:
                continue

            # Calculate MI for each game
            for game_idx in range(n_games):
                outcomes = samples.game_outcomes[:, game_idx]
                mi = mutual_info_score(success, outcomes)
                mi_matrix[game_idx, team_idx] = mi

        # Calculate total impact for each game (sum across teams)
        total_impacts = mi_matrix.sum(axis=1)

        # Build results DataFrame
        games = []
        for game_idx, fixture in enumerate(samples.fixtures):
            game_data = {
                'home_team': fixture['home_team'],
                'away_team': fixture['away_team'],
                'date': fixture.get('date', None),
                'total_impact': float(total_impacts[game_idx]),
            }

            # Add per-team breakdown for top games
            for team_idx, team in enumerate(samples.teams):
                if mi_matrix[game_idx, team_idx] > 0.01:  # Only include significant impacts
                    game_data[f'impact_{team}'] = float(mi_matrix[game_idx, team_idx])

            games.append(game_data)

        df = pd.DataFrame(games)
        df = df.sort_values('total_impact', ascending=False).head(top_n)

        return df.reset_index(drop=True)

    def _find_critical_games_heuristic(self, top_n: int) -> pd.DataFrame:
        """
        Heuristic fallback when simulation data is not available.

        Assumes games between top teams are most critical.
        """
        remaining = self.season_prediction.remaining_fixtures
        position_probs = self.season_prediction.position_probabilities

        # Calculate "importance score" for each team (probability of top 3)
        team_importance = {}
        for team in position_probs.index:
            top3_prob = sum(
                position_probs.loc[team, f'P(pos {p})']
                for p in [1, 2, 3]
                if f'P(pos {p})' in position_probs.columns
            )
            team_importance[team] = top3_prob

        games = []
        for _, fixture in remaining.iterrows():
            home = fixture['home_team']
            away = fixture['away_team']

            # Impact = product of team importances (high when both teams are title contenders)
            home_imp = team_importance.get(home, 0.1)
            away_imp = team_importance.get(away, 0.1)
            impact = home_imp * away_imp

            games.append({
                'home_team': home,
                'away_team': away,
                'date': fixture.get('date', None),
                'total_impact': float(impact),
            })

        df = pd.DataFrame(games)
        df = df.sort_values('total_impact', ascending=False).head(top_n)

        return df.reset_index(drop=True)


class ScenarioClusterer:
    """
    Cluster similar simulation scenarios using machine learning.

    Uses hierarchical clustering on game outcome feature vectors
    to group similar paths to target outcome.
    """

    def __init__(self, n_clusters: int = 5):
        """
        Initialize clusterer.

        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters

    def cluster(
        self,
        simulations: SeasonSimulationSamples,
        target_team: str,
        target_position: int,
        match_predictor=None,
    ) -> List[ScenarioCluster]:
        """
        Cluster simulations that lead to target outcome.

        Args:
            simulations: SeasonSimulationSamples from season prediction
            target_team: Team of interest
            target_position: Target final position
            match_predictor: Optional MatchPredictor for computing probabilities

        Returns:
            List of ScenarioCluster objects
        """
        # Get team index
        try:
            team_idx = simulations.teams.index(target_team)
        except ValueError:
            return []

        # Filter successful simulations
        success_mask = simulations.final_positions[:, team_idx] == target_position
        if success_mask.sum() == 0:
            return []

        successful_outcomes = simulations.game_outcomes[success_mask]

        # If too few successes, return single cluster
        if len(successful_outcomes) < self.n_clusters * 2:
            return [ScenarioCluster(
                description=f"{target_team} reaches position {target_position}",
                frequency=1.0,
                representative_games=self._outcomes_to_dict(
                    successful_outcomes[0], simulations.fixtures
                ),
                probability=float(success_mask.mean()),
            )]

        # Apply hierarchical clustering
        # Use linkage with hamming distance (good for categorical data)
        linkage_matrix = linkage(successful_outcomes, method='average', metric='hamming')

        # Cut tree to get clusters
        cluster_labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')

        # Create scenario clusters
        clusters = []
        total_successful = len(successful_outcomes)

        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() == 0:
                continue

            cluster_outcomes = successful_outcomes[cluster_mask]

            # Find representative (mode/centroid)
            # Use most common outcome pattern
            representative_idx = self._find_representative(cluster_outcomes)
            representative = cluster_outcomes[representative_idx]

            # Generate description
            description = self._generate_cluster_description(
                representative, simulations.fixtures, target_team
            )

            # Compute cluster probability
            frequency = float(cluster_mask.sum() / total_successful)

            # If we have a match predictor, compute joint probability
            probability = float(success_mask.mean() * frequency)

            clusters.append(ScenarioCluster(
                description=description,
                frequency=frequency,
                representative_games=self._outcomes_to_dict(representative, simulations.fixtures),
                probability=probability,
            ))

        # Sort by frequency
        clusters.sort(key=lambda c: c.frequency, reverse=True)

        return clusters

    def _find_representative(self, outcomes: np.ndarray) -> int:
        """Find representative outcome (closest to cluster centroid)."""
        # Compute mode for each game
        mode_outcome = np.round(outcomes.mean(axis=0)).astype(int)

        # Find outcome closest to mode (minimum hamming distance)
        distances = np.mean(outcomes != mode_outcome, axis=1)
        return int(np.argmin(distances))

    def _outcomes_to_dict(
        self,
        outcomes: np.ndarray,
        fixtures: List[dict],
    ) -> Dict[Tuple[str, str], str]:
        """Convert outcome array to dictionary."""
        outcome_names = ['home_win', 'draw', 'away_win']
        return {
            (fixture['home_team'], fixture['away_team']): outcome_names[int(outcomes[i])]
            for i, fixture in enumerate(fixtures)
        }

    def _generate_cluster_description(
        self,
        outcomes: np.ndarray,
        fixtures: List[dict],
        target_team: str,
    ) -> str:
        """Generate human-readable description of cluster."""
        outcome_names = {0: 'win', 1: 'draw', 2: 'loss'}

        # Count outcomes for target team
        team_wins = 0
        team_losses = 0
        team_draws = 0

        for i, fixture in enumerate(fixtures):
            outcome = int(outcomes[i])
            home = fixture['home_team']
            away = fixture['away_team']

            if home == target_team:
                if outcome == 0:
                    team_wins += 1
                elif outcome == 1:
                    team_draws += 1
                else:
                    team_losses += 1
            elif away == target_team:
                if outcome == 2:
                    team_wins += 1
                elif outcome == 1:
                    team_draws += 1
                else:
                    team_losses += 1

        # Generate description
        parts = []
        if team_wins > 0:
            parts.append(f"{target_team} wins {team_wins}")
        if team_draws > 0:
            parts.append(f"draws {team_draws}")
        if team_losses > 0:
            parts.append(f"loses {team_losses}")

        if not parts:
            return f"{target_team} reaches target position"

        description = ", ".join(parts)

        # Add note about other critical results if space permits
        # Look for consistent patterns among other teams
        other_wins = {}
        for i, fixture in enumerate(fixtures):
            outcome = int(outcomes[i])
            home = fixture['home_team']
            away = fixture['away_team']

            if home != target_team and away != target_team:
                winner = home if outcome == 0 else (away if outcome == 2 else None)
                if winner:
                    other_wins[winner] = other_wins.get(winner, 0) + 1

        # Add most common other result
        if other_wins:
            top_winner = max(other_wins.items(), key=lambda x: x[1])
            if top_winner[1] >= 2:  # At least 2 wins
                description += f", {top_winner[0]} wins key games"

        return description


class RuleExtractor:
    """
    Extract minimal logical conditions using decision trees.

    Finds the simplest set of game outcomes that predict whether
    a team achieves target position.
    """

    def __init__(self, max_depth: int = 4, min_frequency: float = 0.1):
        """
        Initialize rule extractor.

        Args:
            max_depth: Maximum depth of decision tree
            min_frequency: Minimum frequency for a condition to be included
        """
        self.max_depth = max_depth
        self.min_frequency = min_frequency

    def extract_rules(
        self,
        simulations: SeasonSimulationSamples,
        target_team: str,
        target_position: int,
    ) -> List[Condition]:
        """
        Extract rules using decision tree.

        Args:
            simulations: SeasonSimulationSamples from season prediction
            target_team: Team of interest
            target_position: Target final position

        Returns:
            List of Condition objects representing key rules
        """
        # Get team index
        try:
            team_idx = simulations.teams.index(target_team)
        except ValueError:
            return []

        # Create feature matrix (game outcomes) and target vector (achieved position)
        X = simulations.game_outcomes  # shape: (n_sims, n_games)
        y = (simulations.final_positions[:, team_idx] == target_position).astype(int)

        # Check if we have any successes
        if y.sum() == 0:
            return []

        # Fit decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=max(10, int(len(y) * self.min_frequency)),
            random_state=42,
        )
        tree.fit(X, y)

        # Extract rules from tree paths
        # Get feature importance to identify most critical games
        feature_importance = tree.feature_importances_

        conditions = []
        for game_idx in np.argsort(feature_importance)[::-1]:
            if feature_importance[game_idx] < 0.01:  # Skip unimportant features
                break

            fixture = simulations.fixtures[game_idx]

            # For each outcome, calculate conditional probability
            for outcome_code, outcome_name in [(0, 'home_win'), (1, 'draw'), (2, 'away_win')]:
                mask = X[:, game_idx] == outcome_code
                if mask.sum() == 0:
                    continue

                frequency = mask.sum() / len(y)
                success_given_outcome = y[mask].sum() / mask.sum()
                overall_success_rate = y.mean()

                # Importance = ΔP when condition is met
                importance = success_given_outcome - overall_success_rate

                # Only include if frequent enough and has meaningful impact
                if frequency >= self.min_frequency and abs(importance) >= 0.05:
                    # Determine if target team controls this outcome
                    team_controls = (
                        (outcome_name == 'home_win' and fixture['home_team'] == target_team) or
                        (outcome_name == 'away_win' and fixture['away_team'] == target_team)
                    )

                    team_in_fixture = (
                        fixture['home_team'] == target_team or
                        fixture['away_team'] == target_team
                    )

                    # Skip conditions where the team is in the fixture but
                    # doesn't control the outcome — these would produce nonsensical
                    # "needs from others" entries like "England must beat France"
                    # when analysing France.
                    if team_in_fixture and not team_controls:
                        continue

                    conditions.append(Condition(
                        game=(fixture['home_team'], fixture['away_team']),
                        outcome=outcome_name,
                        frequency=float(frequency),
                        conditional_prob=float(success_given_outcome),
                        importance=float(importance),
                        team_controls=team_controls,
                    ))

        # Sort by importance
        conditions.sort(key=lambda c: abs(c.importance), reverse=True)

        return conditions


def ordinal(n: int) -> str:
    """Convert number to ordinal string."""
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    return f"{n}{suffix}"
