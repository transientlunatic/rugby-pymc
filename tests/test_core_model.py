"""Unit tests for rugby_ranking.model.core module."""

import pytest
import numpy as np
import pandas as pd
import pymc as pm

from rugby_ranking.model.core import RugbyModel, ModelConfig


@pytest.fixture
def sample_data():
    """Create sample rugby data for testing."""
    np.random.seed(42)
    
    data = {
        'date': pd.date_range('2023-01-01', periods=100),
        'team': np.random.choice(['Leinster', 'Munster', 'Ulster'], 100),
        'opponent': np.random.choice(['Leinster', 'Munster', 'Ulster'], 100),
        'player_name': np.random.choice(['Player A', 'Player B', 'Player C', 'Player D'], 100),
        'position': np.random.randint(1, 16, 100),  # Valid rugby positions
        'minutes': np.random.uniform(60, 80, 100),
        'season': '2023-2024',
        'tries': np.random.poisson(0.2, 100),
        'penalties': np.random.poisson(0.5, 100),
        'conversions': np.random.poisson(0.1, 100),
        'drop_goals': np.random.poisson(0.05, 100),
    }
    return pd.DataFrame(data)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.player_effect_sd == 0.5
        assert config.separate_kicking_effect is True
        assert config.time_varying_effects is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(
            player_effect_sd=0.8,
            separate_kicking_effect=False,
            time_varying_effects=True,
        )
        assert config.player_effect_sd == 0.8
        assert config.separate_kicking_effect is False
        assert config.time_varying_effects is True

    def test_score_types_tuple(self):
        """Test score types are immutable."""
        config = ModelConfig()
        assert isinstance(config.score_types, tuple)
        assert 'tries' in config.score_types


class TestRugbyModel:
    """Test core RugbyModel functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = RugbyModel()
        assert model.config is not None
        assert model.model is None  # Not built yet
        assert len(model._player_ids) == 0

    def test_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = ModelConfig(player_effect_sd=0.8)
        model = RugbyModel(config=config)
        assert model.config.player_effect_sd == 0.8

    def test_build_single_score_type(self, sample_data):
        """Test building model for single score type."""
        model = RugbyModel()
        pm_model = model.build(sample_data, score_type='tries')
        
        assert pm_model is not None
        assert isinstance(pm_model, pm.Model)
        assert len(model._player_ids) > 0
        assert len(model._team_ids) > 0

    def test_build_creates_indices(self, sample_data):
        """Test that building creates proper index mappings."""
        model = RugbyModel()
        model.build(sample_data, score_type='tries')
        
        # Check indices are created
        assert len(model._player_ids) == sample_data['player_name'].nunique()
        assert len(model._team_ids) == sample_data['team'].nunique()
        assert len(model._season_ids) == sample_data['season'].nunique()
        
        # Check indices are zero-indexed
        assert min(model._player_ids.values()) == 0
        assert min(model._team_ids.values()) == 0

    def test_build_joint_model(self, sample_data):
        """Test building joint model with multiple score types."""
        model = RugbyModel()
        pm_model = model.build_joint(sample_data)
        
        assert pm_model is not None
        assert len(model._player_ids) > 0

    def test_invalid_score_type(self, sample_data):
        """Test that invalid score type raises error."""
        model = RugbyModel()
        with pytest.raises((ValueError, KeyError)):
            model.build(sample_data, score_type='invalid_type')

    def test_separate_kicking_effects(self, sample_data):
        """Test model with separate kicking effects."""
        config = ModelConfig(separate_kicking_effect=True)
        model = RugbyModel(config=config)
        pm_model = model.build_joint(sample_data)
        
        assert pm_model is not None
        # Should have separate effects in the trace later

    def test_time_varying_effects(self, sample_data):
        """Test model with time-varying effects."""
        config = ModelConfig(time_varying_effects=True)
        model = RugbyModel(config=config)
        pm_model = model.build_joint(sample_data)
        
        assert pm_model is not None


class TestModelIndexing:
    """Test internal indexing mechanisms."""

    def test_team_season_indexing(self, sample_data):
        """Test team-season pair indexing."""
        model = RugbyModel()
        model.build(sample_data, score_type='tries')
        
        # Create team-season pairs
        team_seasons = sample_data[['team', 'season']].drop_duplicates()
        
        assert len(model._team_season_ids) >= len(team_seasons)

    def test_position_indexing(self, sample_data):
        """Test position indexing."""
        model = RugbyModel()
        model.build(sample_data, score_type='tries')
        
        # Positions should be re-indexed starting from 0
        positions = sample_data['position'].unique()
        assert len(model._position_ids) == len(positions)
        assert min(model._position_ids.values()) == 0
