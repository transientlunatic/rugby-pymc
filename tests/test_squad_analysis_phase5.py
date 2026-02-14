"""
Tests for Phase 5 Squad Analysis functionality.

Tests lineup prediction, injury impact analysis, and squad comparison.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from rugby_ranking.model.squad_analysis import (
    SquadAnalysis,
    LineupPredictor,
    InjuryImpactAnalyzer,
    SquadBasedPredictor,
    SquadComparator,
    SquadAnalyzer,
    export_squad_analysis_to_markdown,
    export_tournament_comparison_to_markdown,
)


@pytest.fixture
def sample_squad():
    """Create a sample squad for testing."""
    players = []

    # Front row
    for i in range(4):
        players.append({'player': f'Prop {i}', 'primary_position': 'Prop',
                       'club': 'Test Club', 'secondary_positions': []})
    for i in range(3):
        players.append({'player': f'Hooker {i}', 'primary_position': 'Hooker',
                       'club': 'Test Club', 'secondary_positions': []})

    # Second row
    for i in range(3):
        players.append({'player': f'Lock {i}', 'primary_position': 'Lock',
                       'club': 'Test Club', 'secondary_positions': []})

    # Back row
    for i in range(4):
        players.append({'player': f'Flanker {i}', 'primary_position': 'Flanker',
                       'club': 'Test Club', 'secondary_positions': []})
    for i in range(2):
        players.append({'player': f'Number Eight {i}', 'primary_position': 'Number 8',
                       'club': 'Test Club', 'secondary_positions': []})

    # Half-backs
    for i in range(3):
        players.append({'player': f'Scrum-half {i}', 'primary_position': 'Scrum-half',
                       'club': 'Test Club', 'secondary_positions': []})
    for i in range(2):
        players.append({'player': f'Fly-half {i}', 'primary_position': 'Fly-half',
                       'club': 'Test Club', 'secondary_positions': []})

    # Centres
    for i in range(4):
        players.append({'player': f'Centre {i}', 'primary_position': 'Centre',
                       'club': 'Test Club', 'secondary_positions': []})

    # Back three
    for i in range(4):
        players.append({'player': f'Wing {i}', 'primary_position': 'Wing',
                       'club': 'Test Club', 'secondary_positions': []})
    for i in range(2):
        players.append({'player': f'Fullback {i}', 'primary_position': 'Fullback',
                       'club': 'Test Club', 'secondary_positions': []})

    return pd.DataFrame(players)


@pytest.fixture
def sample_ratings(sample_squad):
    """Create sample player ratings."""
    ratings = []

    for player in sample_squad['player']:
        # Give varying ratings
        base_rating = np.random.uniform(-0.5, 0.5)
        ratings.append({
            'player': player,
            'score_type': 'tries',
            'rating_mean': base_rating,
            'rating_std': 0.1,
            'percentile': 0.5,
            'in_model': True,
            'matched_name': player,
        })

    return pd.DataFrame(ratings)


@pytest.fixture
def sample_analysis(sample_squad, sample_ratings):
    """Create a sample SquadAnalysis."""
    # Create depth chart
    depth_chart = {}
    for position in sample_squad['primary_position'].unique():
        players = sample_squad[sample_squad['primary_position'] == position]['player'].tolist()
        player_ratings = []
        for player in players:
            rating = sample_ratings[sample_ratings['player'] == player]['rating_mean'].iloc[0]
            player_ratings.append((player, rating))
        player_ratings.sort(key=lambda x: x[1], reverse=True)
        depth_chart[position] = player_ratings

    # Create position strength
    position_strength = []
    for position, players in depth_chart.items():
        if len(players) > 0:
            first = players[0][1]
            second = players[1][1] if len(players) > 1 else 0.0
            position_strength.append({
                'position': position,
                'first_choice_rating': first,
                'second_choice_rating': second,
                'depth_score': second / first if first != 0 else 0.5,
                'expected_strength': 0.75 * first + 0.25 * second,
                'n_players': len(players),
                'first_choice_player': players[0][0],
                'second_choice_player': players[1][0] if len(players) > 1 else None,
            })

    position_strength_df = pd.DataFrame(position_strength)

    return SquadAnalysis(
        team='Test Team',
        season='2024-2025',
        squad=sample_squad,
        player_ratings=sample_ratings,
        depth_chart=depth_chart,
        position_strength=position_strength_df,
        overall_strength=0.65,
        depth_score=0.70,
    )


class TestLineupPredictor:
    """Test LineupPredictor functionality."""

    def test_predict_lineup(self, sample_analysis):
        """Test basic lineup prediction."""
        predictor = LineupPredictor()

        lineup = predictor.predict_lineup(sample_analysis)

        assert 'starting_xv' in lineup
        assert 'bench' in lineup
        assert 'total_rating' in lineup
        assert 'coverage_valid' in lineup

        # Check we have 15 starters
        assert len(lineup['starting_xv']) == 15

        # Check we have up to 8 bench players
        assert len(lineup['bench']) <= 8

        # Check no duplicates
        all_players = set(lineup['starting_xv'].values()) | set(lineup['bench'])
        assert len(all_players) == len(lineup['starting_xv']) + len(lineup['bench'])

    def test_predict_lineup_with_unavailable(self, sample_analysis):
        """Test lineup prediction with unavailable players."""
        predictor = LineupPredictor()

        # Make top prop unavailable
        unavailable = ['Prop 0']

        lineup = predictor.predict_lineup(sample_analysis, unavailable=unavailable)

        # Check unavailable player not selected
        all_selected = set(lineup['starting_xv'].values()) | set(lineup['bench'])
        assert 'Prop 0' not in all_selected

    def test_predict_lineup_distribution(self, sample_analysis):
        """Test lineup distribution sampling."""
        predictor = LineupPredictor()

        distribution = predictor.predict_lineup_distribution(
            sample_analysis,
            n_samples=20,
            uncertainty_factor=0.1
        )

        assert isinstance(distribution, pd.DataFrame)
        assert 'player' in distribution.columns
        assert 'selection_probability' in distribution.columns
        assert 'likely_role' in distribution.columns

        # Check probabilities are valid
        assert (distribution['selection_probability'] >= 0).all()
        assert (distribution['selection_probability'] <= 1).all()

    def test_position_coverage_validation(self, sample_analysis):
        """Test position coverage validation."""
        predictor = LineupPredictor()

        lineup = predictor.predict_lineup(sample_analysis)

        # Coverage should be valid for a full squad
        assert lineup['coverage_valid']


class TestInjuryImpactAnalyzer:
    """Test InjuryImpactAnalyzer functionality."""

    def test_analyze_player_impact(self, sample_analysis):
        """Test single player impact analysis."""
        predictor = LineupPredictor()
        analyzer = InjuryImpactAnalyzer(predictor)

        impact = analyzer.analyze_player_impact('Prop 0', sample_analysis)

        assert 'player' in impact
        assert 'position' in impact
        assert 'replacement' in impact
        assert 'rating_drop' in impact
        assert 'criticality_score' in impact

        # Criticality should be between 0 and 1
        if 'criticality_score' in impact:
            assert 0 <= impact['criticality_score'] <= 1

    def test_identify_critical_players(self, sample_analysis):
        """Test critical players identification."""
        predictor = LineupPredictor()
        analyzer = InjuryImpactAnalyzer(predictor)

        critical = analyzer.identify_critical_players(sample_analysis, top_n=5)

        assert isinstance(critical, pd.DataFrame)
        assert len(critical) <= 5

        if len(critical) > 0:
            assert 'player' in critical.columns
            assert 'criticality_score' in critical.columns

            # Check sorted by criticality (descending)
            assert (critical['criticality_score'].diff().dropna() <= 0).all()

    def test_analyze_squad_robustness(self, sample_analysis):
        """Test squad robustness analysis."""
        predictor = LineupPredictor()
        analyzer = InjuryImpactAnalyzer(predictor)

        robustness = analyzer.analyze_squad_robustness(
            sample_analysis,
            n_simulations=10,
            injury_prob=0.1
        )

        assert 'mean_impact' in robustness
        assert 'std_impact' in robustness
        assert 'worst_case' in robustness
        assert 'best_case' in robustness
        assert 'robustness_score' in robustness
        assert 'vulnerable_positions' in robustness

        # Robustness score should be between 0 and 1
        assert 0 <= robustness['robustness_score'] <= 1


class TestSquadBasedPredictor:
    """Test SquadBasedPredictor functionality."""

    def test_predict_with_squads(self, sample_analysis):
        """Test squad-based match prediction."""
        predictor = LineupPredictor()
        match_predictor = Mock()

        squad_predictor = SquadBasedPredictor(match_predictor, predictor)

        # Create two squad analyses
        home_analysis = sample_analysis
        away_analysis = sample_analysis

        prediction = squad_predictor.predict_with_squads(
            home_analysis,
            away_analysis,
            season='2024-2025',
            n_lineup_samples=5
        )

        assert 'home_win_prob' in prediction
        assert 'away_win_prob' in prediction
        assert 'draw_prob' in prediction
        assert 'expected_home_score' in prediction
        assert 'expected_away_score' in prediction

        # Probabilities should sum to ~1
        prob_sum = prediction['home_win_prob'] + prediction['away_win_prob'] + prediction['draw_prob']
        assert 0.9 <= prob_sum <= 1.1


class TestSquadComparator:
    """Test SquadComparator functionality."""

    def test_compare_squads(self, sample_squad, sample_ratings):
        """Test multi-squad comparison."""
        # Create mock analyzer
        analyzer = Mock(spec=SquadAnalyzer)

        def mock_analyze(squad, team, season):
            return SquadAnalysis(
                team=team,
                season=season,
                squad=squad,
                player_ratings=sample_ratings,
                overall_strength=np.random.uniform(0.5, 0.8),
                depth_score=np.random.uniform(0.6, 0.9),
            )

        analyzer.analyze_squad = mock_analyze

        comparator = SquadComparator(analyzer)

        squads = {
            'Team A': sample_squad,
            'Team B': sample_squad,
            'Team C': sample_squad,
        }

        comparison = comparator.compare_squads(squads, '2024-2025')

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert 'team' in comparison.columns
        assert 'overall_strength' in comparison.columns
        assert 'depth_score' in comparison.columns


class TestExportFunctions:
    """Test export and visualization functions."""

    def test_export_squad_analysis_to_markdown(self, sample_analysis):
        """Test markdown export."""
        markdown = export_squad_analysis_to_markdown(sample_analysis)

        assert isinstance(markdown, str)
        assert 'Squad Analysis' in markdown
        assert sample_analysis.team in markdown
        assert '|' in markdown  # Should have tables

    def test_export_tournament_comparison_to_markdown(self):
        """Test tournament comparison export."""
        comparison_df = pd.DataFrame({
            'team': ['Team A', 'Team B'],
            'overall_strength': [0.75, 0.65],
            'depth_score': [0.80, 0.70],
            'squad_size': [35, 33],
        })

        analyses = {}  # Can be empty for basic test

        markdown = export_tournament_comparison_to_markdown(
            comparison_df,
            analyses,
            'Test Tournament'
        )

        assert isinstance(markdown, str)
        assert 'Test Tournament' in markdown
        assert 'Team A' in markdown
        assert 'Team B' in markdown


def test_lineup_predictor_positions():
    """Test that STARTING_XV_POSITIONS is correctly defined."""
    assert len(LineupPredictor.STARTING_XV_POSITIONS) == 15
    assert 1 in LineupPredictor.STARTING_XV_POSITIONS
    assert 15 in LineupPredictor.STARTING_XV_POSITIONS

    # Check key positions
    assert LineupPredictor.STARTING_XV_POSITIONS[2] == 'Hooker'
    assert LineupPredictor.STARTING_XV_POSITIONS[9] == 'Scrum-half'
    assert LineupPredictor.STARTING_XV_POSITIONS[10] == 'Fly-half'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
