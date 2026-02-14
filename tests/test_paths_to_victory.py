"""
Integration tests for paths to victory analysis.

Tests the complete paths analysis pipeline including:
- MCMC pattern mining
- Combinatorial enumeration
- Bonus point variations
- Parallel processing
- Critical games identification
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock

from rugby_ranking.model.paths_to_victory import (
    PathsAnalyzer,
    PathsOutput,
    Condition,
    ScenarioCluster,
)
from rugby_ranking.model.season_predictor import SeasonPrediction, SeasonSimulationSamples


@pytest.fixture
def mock_season_prediction():
    """Create a mock season prediction with simulation samples."""
    # Create mock current standings
    current_standings = pd.DataFrame({
        'team': ['Team A', 'Team B', 'Team C', 'Team D'],
        'played': [10, 10, 10, 10],
        'won': [7, 6, 5, 3],
        'total_points': [28, 24, 20, 12],
        'position': [1, 2, 3, 4],
        'points_for': [200, 180, 160, 120],
        'points_against': [150, 160, 170, 200],
        'points_diff': [50, 20, -10, -80],
    })

    # Create mock remaining fixtures (3 games)
    remaining_fixtures = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team D', 'Team C', 'Team D'],
        'home_win_prob': [0.7, 0.6, 0.5],
        'draw_prob': [0.1, 0.1, 0.1],
        'away_win_prob': [0.2, 0.3, 0.4],
        'home_score_pred': [24, 22, 20],
        'away_score_pred': [15, 18, 19],
    })

    # Create mock simulation samples (100 simulations, 3 games, 4 teams)
    n_sims = 100
    n_games = 3
    n_teams = 4

    # Generate random game outcomes (0=home_win, 1=draw, 2=away_win)
    np.random.seed(42)
    game_outcomes = np.random.choice([0, 1, 2], size=(n_sims, n_games), p=[0.6, 0.1, 0.3])

    # Generate random final positions
    final_positions = np.zeros((n_sims, n_teams), dtype=int)
    for i in range(n_sims):
        final_positions[i] = np.random.permutation([1, 2, 3, 4])

    simulation_samples = SeasonSimulationSamples(
        teams=['Team A', 'Team B', 'Team C', 'Team D'],
        fixtures=[
            {'home_team': 'Team A', 'away_team': 'Team D'},
            {'home_team': 'Team B', 'away_team': 'Team C'},
            {'home_team': 'Team C', 'away_team': 'Team D'},
        ],
        game_outcomes=game_outcomes,
        final_positions=final_positions,
    )

    # Create position probabilities
    position_probs = pd.DataFrame({
        'team': ['Team A', 'Team B', 'Team C', 'Team D'],
        'P(pos 1)': [0.50, 0.30, 0.15, 0.05],
        'P(pos 2)': [0.30, 0.40, 0.20, 0.10],
        'P(pos 3)': [0.15, 0.20, 0.40, 0.25],
        'P(pos 4)': [0.05, 0.10, 0.25, 0.60],
    }).set_index('team')

    # Create season prediction
    prediction = SeasonPrediction(
        current_standings=current_standings,
        predicted_standings=current_standings.copy(),  # Simplified
        remaining_fixtures=remaining_fixtures,
        position_probabilities=position_probs,
        playoff_probabilities=None,
        simulation_samples=simulation_samples,
    )

    return prediction


@pytest.fixture
def mock_match_predictor():
    """Create a mock match predictor."""
    predictor = Mock()
    predictor.competition = 'urc'
    return predictor


class TestPathsAnalyzer:
    """Test suite for PathsAnalyzer class."""

    def test_initialization(self, mock_season_prediction, mock_match_predictor):
        """Test PathsAnalyzer initialization."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        assert analyzer.season_prediction == mock_season_prediction
        assert analyzer.match_predictor == mock_match_predictor
        assert analyzer.combinatorial_threshold == 100_000
        assert analyzer._simulations is not None

    def test_choose_method_early_tournament(self, mock_season_prediction, mock_match_predictor):
        """Test method selection for early tournament (many games remaining)."""
        # Create prediction with many remaining games
        many_games = pd.concat([mock_season_prediction.remaining_fixtures] * 4)
        mock_season_prediction.remaining_fixtures = many_games

        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)
        method = analyzer._choose_method()

        assert method == 'mcmc'  # Too many combinations for combinatorial

    def test_choose_method_late_tournament(self, mock_season_prediction, mock_match_predictor):
        """Test method selection for late tournament (few games remaining)."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)
        method = analyzer._choose_method()

        # With 3 games, 3^3 = 27 combinations < 100k threshold
        assert method == 'combinatorial'

    def test_analyze_paths_mcmc(self, mock_season_prediction, mock_match_predictor):
        """Test paths analysis using MCMC method."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team A',
            target_position=1,
            method='mcmc',
            max_conditions=5,
        )

        assert isinstance(result, PathsOutput)
        assert result.team == 'Team A'
        assert result.target_position == 1
        assert result.method == 'mcmc'
        assert 0 <= result.probability <= 1
        assert isinstance(result.conditions, list)
        assert isinstance(result.critical_games, list)
        assert isinstance(result.narrative, str)

    def test_analyze_paths_combinatorial(self, mock_season_prediction, mock_match_predictor):
        """Test paths analysis using combinatorial method."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team B',
            target_position=2,
            method='combinatorial',
            max_conditions=5,
        )

        assert isinstance(result, PathsOutput)
        assert result.team == 'Team B'
        assert result.target_position == 2
        assert result.method == 'combinatorial'
        assert 0 <= result.probability <= 1
        assert len(result.conditions) <= 5

    def test_analyze_paths_auto_selection(self, mock_season_prediction, mock_match_predictor):
        """Test automatic method selection."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team C',
            target_position=3,
            method='auto',
        )

        assert isinstance(result, PathsOutput)
        # Should choose combinatorial for 3 games
        assert result.method == 'combinatorial'

    def test_find_critical_games(self, mock_season_prediction, mock_match_predictor):
        """Test critical games identification."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        critical_games = analyzer.find_critical_games(top_n=3)

        assert isinstance(critical_games, pd.DataFrame)
        assert len(critical_games) <= 3
        assert 'home_team' in critical_games.columns
        assert 'away_team' in critical_games.columns
        assert 'total_impact' in critical_games.columns

        # Impact should be non-negative
        assert (critical_games['total_impact'] >= 0).all()

    # TODO: Bonus probabilities test - method not in current implementation
    # def test_calculate_bonus_probabilities(self, mock_season_prediction, mock_match_predictor):
    #     """Test bonus point probability calculation."""
    #     # This would test explicit bonus point enumeration (future optimization)
    #     pass

    def test_decode_bonus_outcome(self, mock_season_prediction, mock_match_predictor):
        """Test bonus outcome decoding."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        # Test home win with both bonuses
        result = analyzer._decode_bonus_outcome(3, pd.Series())
        assert result['result'] == 'home_win'
        assert result['home_try_bonus'] is True
        assert result['away_losing_bonus'] is True

        # Test draw
        result = analyzer._decode_bonus_outcome(4, pd.Series())
        assert result['result'] == 'draw'

        # Test away win with try bonus only
        result = analyzer._decode_bonus_outcome(6, pd.Series())
        assert result['result'] == 'away_win'
        assert result['away_try_bonus'] is True
        assert result['home_losing_bonus'] is False

    def test_estimate_scores(self, mock_season_prediction, mock_match_predictor):
        """Test score estimation from match result."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        fixture = mock_season_prediction.remaining_fixtures.iloc[0]

        # Test home win with home try bonus
        match_result = {
            'result': 'home_win',
            'home_try_bonus': True,
            'away_losing_bonus': False,
        }

        home_score, away_score, home_tries, away_tries = analyzer._estimate_scores(
            match_result, fixture
        )

        assert home_score > away_score  # Home wins
        assert home_tries >= 4  # Try bonus requires 4+ tries
        assert isinstance(home_score, int)
        assert isinstance(away_score, int)

    def test_early_pruning(self, mock_season_prediction, mock_match_predictor):
        """Test that early pruning skips low-probability outcomes."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        # Modify probabilities to create very low probability outcomes
        fixtures = mock_season_prediction.remaining_fixtures.copy()
        fixtures['home_win_prob'] = 0.99
        fixtures['away_win_prob'] = 0.005
        fixtures['draw_prob'] = 0.005
        mock_season_prediction.remaining_fixtures = fixtures

        result = analyzer.analyze_paths(
            team='Team A',
            target_position=1,
            method='combinatorial',
        )

        # Should complete without evaluating all 3^3 = 27 combinations
        assert result.probability > 0

    # TODO: Parallel/sequential evaluation tests - not yet implemented
    # def test_parallel_vs_sequential_consistency(self, mock_season_prediction, mock_match_predictor):
    #     """Test that parallel and sequential evaluation give same results."""
    #     # This would test parallel processing optimization (Phase 4b optional)
    #     pass


class TestConditionExtraction:
    """Test condition extraction from combinatorial analysis."""

    def test_extract_combinatorial_conditions(self, mock_season_prediction, mock_match_predictor):
        """Test extracting conditions from successful combinations."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        # Create mock successful combinations
        successful_combos = [
            ((0, 0, 0), 0.4, None),  # All home wins
            ((0, 0, 1), 0.3, None),  # Two home wins
            ((0, 1, 0), 0.2, None),  # Two home wins (different pattern)
        ]

        conditions = analyzer._extract_combinatorial_conditions(
            mock_season_prediction.remaining_fixtures,
            successful_combos,
            'Team A',
            0.9  # total probability
        )

        assert isinstance(conditions, list)
        assert all(isinstance(c, Condition) for c in conditions)

        # First game should appear as home_win frequently
        first_game_conditions = [c for c in conditions if c.game[0] == 'Team A']
        if first_game_conditions:
            assert first_game_conditions[0].outcome == 'home_win'
            assert first_game_conditions[0].frequency >= 0.9  # Appears in all scenarios


# TODO: Bonus variations test - parameter not yet implemented
# Bonus point variations are currently estimated from scores rather than explicit enumeration
# @pytest.mark.parametrize("use_bonus_variations", [True, False])
# def test_bonus_variations_parameter(mock_season_prediction, mock_match_predictor, use_bonus_variations):
#     """Test that bonus variations parameter works correctly."""
#     pass


class TestNarrativeGeneration:
    """Test narrative generation features (Phase 4c)."""

    def test_generate_detailed_narrative(self, mock_season_prediction, mock_match_predictor):
        """Test detailed narrative generation."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team A',
            target_position=1,
            method='mcmc',
        )

        narrative = analyzer._generate_detailed_narrative(result)

        assert isinstance(narrative, str)
        assert 'Team A' in narrative.upper()
        assert '1st' in narrative or 'first' in narrative.lower()
        assert result.method.upper() in narrative
        assert len(narrative) > 100  # Should be detailed

    def test_generate_blog_narrative(self, mock_season_prediction, mock_match_predictor):
        """Test blog-style narrative generation."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team B',
            target_position=2,
            method='combinatorial',
        )

        blog_narrative = analyzer._generate_blog_narrative(result)

        assert isinstance(blog_narrative, str)
        assert blog_narrative.startswith('# ')  # Markdown heading
        assert 'Team B' in blog_narrative
        assert '##' in blog_narrative  # Should have subheadings
        assert '**' in blog_narrative  # Should have bold text

    def test_generate_social_narrative(self, mock_season_prediction, mock_match_predictor):
        """Test social media narrative generation."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team C',
            target_position=3,
            method='mcmc',
        )

        social = analyzer._generate_social_narrative(result)

        assert isinstance(social, str)
        assert 'Team C' in social
        assert len(social) < 500  # Should be concise for social media
        # Should contain emoji indicators
        assert any(emoji in social for emoji in ['🏆', '🎯', '⚡', '🤞', '📊', '✓'])

    def test_narrative_style_parameter(self, mock_season_prediction, mock_match_predictor):
        """Test that narrative style parameter works."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team A',
            target_position=1,
            method='mcmc',
        )

        # Test all three styles
        detailed = analyzer._generate_narrative(result, style='detailed')
        blog = analyzer._generate_narrative(result, style='blog')
        social = analyzer._generate_narrative(result, style='social')

        assert len(detailed) > len(blog) > len(social)
        assert '=' * 70 in detailed  # Technical formatting
        assert '# ' in blog  # Markdown formatting
        assert len(social) < 500  # Concise


class TestBlogExport:
    """Test blog post export functionality (Phase 4c)."""

    def test_export_to_markdown(self, mock_season_prediction, mock_match_predictor):
        """Test markdown export with metadata."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team A',
            target_position=1,
            method='combinatorial',
        )

        markdown = analyzer.export_to_markdown(
            result,
            include_metadata=True,
            include_visualization=False,
        )

        assert isinstance(markdown, str)
        assert '---' in markdown  # YAML frontmatter
        assert 'title:' in markdown
        assert 'date:' in markdown
        assert 'Team A' in markdown
        assert '## Technical Details' in markdown

    def test_export_without_metadata(self, mock_season_prediction, mock_match_predictor):
        """Test markdown export without frontmatter."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team B',
            target_position=2,
            method='mcmc',
        )

        markdown = analyzer.export_to_markdown(
            result,
            include_metadata=False,
        )

        assert isinstance(markdown, str)
        assert not markdown.startswith('---')  # No frontmatter
        assert 'Team B' in markdown

    def test_generate_social_snippets(self, mock_season_prediction, mock_match_predictor):
        """Test social media snippet generation."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team C',
            target_position=3,
            method='combinatorial',
        )

        snippets = analyzer.generate_social_snippets(result)

        assert isinstance(snippets, dict)
        assert 'twitter' in snippets
        assert 'linkedin' in snippets
        assert 'facebook' in snippets

        # Twitter should be concise
        assert len(snippets['twitter']) < 500

        # All should mention Team C
        for platform, snippet in snippets.items():
            assert 'Team C' in snippet
            assert isinstance(snippet, str)


class TestSankeyDiagram:
    """Test Sankey diagram visualization (Phase 4c)."""

    def test_create_sankey(self, mock_season_prediction, mock_match_predictor):
        """Test Sankey diagram creation."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team A',
            target_position=1,
            method='mcmc',
        )

        sankey = analyzer._create_sankey(result)

        # Should return None if plotly not available, or a Figure if it is
        if sankey is not None:
            # If plotly is available, check the figure
            assert hasattr(sankey, 'data')
            assert len(sankey.data) > 0
            assert 'Team A' in str(sankey.layout.title.text)

    def test_sankey_attached_to_result(self, mock_season_prediction, mock_match_predictor):
        """Test that Sankey diagram is attached to PathsOutput."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        result = analyzer.analyze_paths(
            team='Team B',
            target_position=2,
            method='combinatorial',
        )

        # Check if sankey_diagram attribute exists
        assert hasattr(result, 'sankey_diagram')

        # If plotly is available and diagram was created, verify structure
        if result.sankey_diagram is not None:
            assert hasattr(result.sankey_diagram, 'show')


class TestConditionFormatting:
    """Test condition formatting helpers."""

    def test_format_condition(self, mock_season_prediction, mock_match_predictor):
        """Test basic condition formatting."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        condition = Condition(
            game=('Team A', 'Team D'),
            outcome='home_win',
            frequency=0.8,
            conditional_prob=0.9,
            importance=0.2,
            team_controls=True,
        )

        formatted = analyzer._format_condition(condition, 'Team A')
        assert 'Beat Team D' in formatted or 'beat Team D' in formatted

    def test_format_condition_blog(self, mock_season_prediction, mock_match_predictor):
        """Test blog-style condition formatting."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        condition = Condition(
            game=('Team B', 'Team C'),
            outcome='away_win',
            frequency=0.7,
            conditional_prob=0.85,
            importance=0.15,
            team_controls=False,
        )

        formatted = analyzer._format_condition_blog(condition, 'Team A')
        assert 'Team C' in formatted
        assert 'beat' in formatted.lower()

    def test_format_condition_social(self, mock_season_prediction, mock_match_predictor):
        """Test social media condition formatting."""
        analyzer = PathsAnalyzer(mock_season_prediction, mock_match_predictor)

        condition = Condition(
            game=('Team A', 'Team B'),
            outcome='draw',
            frequency=0.5,
            conditional_prob=0.6,
            importance=0.1,
            team_controls=False,
        )

        formatted = analyzer._format_condition_social(condition)
        assert 'Team A' in formatted
        assert 'Team B' in formatted
        assert len(formatted) < 50  # Should be concise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
