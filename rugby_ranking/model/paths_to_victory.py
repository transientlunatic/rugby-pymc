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

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score

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
        # This requires access to raw simulation data
        # For now, return placeholder based on aggregate probabilities

        position_probs = self.season_prediction.position_probabilities
        probability = position_probs.loc[team, f'P(pos {target_position})']

        # TODO: Implement full MCMC pattern mining
        # Would need:
        # 1. Access to individual simulation results
        # 2. Feature extraction (game outcomes per sim)
        # 3. Decision tree on features
        # 4. Scenario clustering

        conditions = self._extract_conditions_from_aggregates(team, target_position)
        if self._simulations is not None:
            critical_games = self._identify_critical_games_mutual_info(team, target_position)
        else:
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
        )

    def _analyze_combinatorial(
        self,
        team: str,
        target_position: int,
        max_conditions: int,
        n_clusters: int,
    ) -> PathsOutput:
        """
        Analyze using combinatorial enumeration (late tournament).

        Enumerates all possible outcomes for remaining games,
        calculates final tables, and weights by probabilities.
        """
        # Generate all possible outcomes
        remaining = self.season_prediction.remaining_fixtures

        # For demonstration, use simplified outcomes (win/draw/loss only)
        # Real implementation would include bonus points
        all_outcomes = list(itertools.product(
            ['home_win', 'draw', 'away_win'],
            repeat=len(remaining)
        ))

        # Calculate final table for each outcome
        # Weight by match prediction probabilities
        # Find outcomes where team achieves target position
        # Extract and simplify conditions

        # TODO: Full implementation
        # This is complex - need to:
        # 1. Simulate each outcome combination
        # 2. Calculate final league table
        # 3. Weight by joint probability from match predictions
        # 4. Boolean minimization of successful outcomes

        # For now, use MCMC approach
        return self._analyze_mcmc(team, target_position, max_conditions, n_clusters)

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

    def _generate_narrative(self, result: PathsOutput) -> str:
        """Generate human-readable narrative from analysis results."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"PATHS TO VICTORY: {result.team.upper()}")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        position_name = self._ordinal(result.target_position)
        lines.append(
            f"{result.team} can finish {position_name} with "
            f"{result.probability:.1%} probability."
        )
        lines.append("")

        # Key requirements
        if result.conditions:
            lines.append("Key requirements:")
            lines.append("")

            team_conditions = [c for c in result.conditions if c.team_controls]
            other_conditions = [c for c in result.conditions if not c.team_controls]

            if team_conditions:
                lines.append(f"  What {result.team} must do:")
                for cond in team_conditions[:5]:
                    lines.append(
                        f"    ✓ {self._format_condition(cond, result.team)} "
                        f"[{cond.frequency:.0%}]"
                    )
                lines.append("")

            if other_conditions:
                lines.append(f"  What {result.team} needs from others:")
                for cond in other_conditions[:5]:
                    lines.append(
                        f"    ○ {self._format_condition(cond, result.team)} "
                        f"[{cond.frequency:.0%}]"
                    )
                lines.append("")

        # Scenario clusters
        if result.scenario_clusters:
            lines.append("Example scenarios:")
            lines.append("")
            for i, scenario in enumerate(result.scenario_clusters[:3], 1):
                lines.append(
                    f"  {i}. {scenario.description} "
                    f"({scenario.frequency:.0%} of successful paths)"
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
                lines.append(
                    f"  {i}. {home} vs {away}: "
                    f"{importance:+.0%} impact"
                )
            lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"Analysis method: {result.method.upper()}")
        lines.append("=" * 70)

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

    def _create_sankey(self, result: PathsOutput) -> Optional[go.Figure]:
        """Create Sankey diagram visualization."""
        if not PLOTLY_AVAILABLE:
            return None

        # TODO: Implement Sankey diagram
        # This requires detailed path information from simulations
        # For now, return None

        return None

    def find_critical_games(
        self,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Find critical games across all teams and positions.

        Returns DataFrame with games ranked by total impact across
        all teams' championship chances.

        Args:
            top_n: Number of top games to return

        Returns:
            DataFrame with columns: home_team, away_team, total_impact,
            and impact breakdown by team
        """
        # TODO: Implement by analyzing impact of each game on
        # position probabilities across all teams

        remaining = self.season_prediction.remaining_fixtures

        games = []
        for _, fixture in remaining.iterrows():
            games.append({
                'home_team': fixture['home_team'],
                'away_team': fixture['away_team'],
                'date': fixture.get('date', None),
                'total_impact': 0.5,  # Placeholder
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

    def cluster(self, simulations: List, target_team: str, target_position: int):
        """
        Cluster simulations that lead to target outcome.

        Args:
            simulations: List of simulation results
            target_team: Team of interest
            target_position: Target final position

        Returns:
            List of ScenarioCluster objects
        """
        # TODO: Implement clustering
        # 1. Filter successful simulations
        # 2. Extract feature vectors (game outcomes)
        # 3. Apply hierarchical clustering
        # 4. Find representative from each cluster
        # 5. Generate descriptions

        raise NotImplementedError


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
        simulations: List,
        target_team: str,
        target_position: int,
    ) -> List[Condition]:
        """
        Extract rules using decision tree.

        Args:
            simulations: List of simulation results
            target_team: Team of interest
            target_position: Target final position

        Returns:
            List of Condition objects representing key rules
        """
        # TODO: Implement decision tree rule extraction
        # 1. Create feature matrix from simulations
        # 2. Create target vector (achieved position or not)
        # 3. Fit decision tree
        # 4. Extract rules from tree paths
        # 5. Calculate frequencies and conditional probabilities

        raise NotImplementedError


def ordinal(n: int) -> str:
    """Convert number to ordinal string."""
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    return f"{n}{suffix}"
