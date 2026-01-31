"""Unit tests for rugby_ranking.model.data_validation module."""

import pytest
import pandas as pd
import numpy as np

from rugby_ranking.model.data_validation import (
    detect_kicking_anomalies,
    clean_kicking_data,
    validate_position_scores,
)
from rugby_ranking.utils.constants import KICKING_POSITIONS, NON_KICKING_POSITIONS


@pytest.fixture
def sample_validation_data():
    """Create sample data with known anomalies."""
    data = {
        'player_name': ['Fly-half A', 'Fly-half A', 'Prop B', 'Prop B', 'Scrum-half C'] * 5,
        'position': [10, 10, 1, 1, 9] * 5,
        'team': ['Leinster'] * 25,
        'date': pd.date_range('2023-01-01', periods=25),
        'season': '2023-2024',
        'tries': [0, 0, 0, 0, 0] * 5,
        'penalties': [2, 2, 5, 5, 3] * 5,  # Prop taking penalties is anomalous
        'conversions': [1, 1, 2, 2, 0] * 5,  # Prop taking conversions is anomalous
        'drop_goals': [0, 0, 0, 0, 0] * 5,
        'minutes': 80,
    }
    return pd.DataFrame(data)


class TestDetectKickingAnomalies:
    """Test detection of kicking score anomalies."""

    def test_detects_anomalies(self, sample_validation_data):
        """Test that anomalies are detected."""
        df = sample_validation_data
        anomalies = detect_kicking_anomalies(df, verbose=False)
        
        # Should detect that Prop B took penalties/conversions
        assert len(anomalies) > 0

    def test_prop_anomalies(self, sample_validation_data):
        """Test detection of prop (position 1) anomalies."""
        df = sample_validation_data
        anomalies = detect_kicking_anomalies(df, verbose=False)
        
        # Prop B should be flagged
        prop_anomalies = anomalies[anomalies['player_name'] == 'Prop B']
        assert len(prop_anomalies) > 0

    def test_no_false_positives(self):
        """Test that clean data doesn't flag anomalies."""
        clean_data = {
            'player_name': ['Fly-half A', 'Scrum-half B', 'Prop C'],
            'position': [10, 9, 1],  # 10, 9 are kickers; 1 is not
            'team': ['Leinster', 'Leinster', 'Leinster'],
            'date': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'season': '2023-2024',
            'tries': [0, 0, 0],
            'penalties': [2, 1, 0],  # Only kickers take penalties
            'conversions': [1, 0, 0],  # Only kickers take conversions
            'drop_goals': [0, 0, 0],
            'minutes': 80,
        }
        df = pd.DataFrame(clean_data)
        anomalies = detect_kicking_anomalies(df, verbose=False)
        
        # Should detect no anomalies (or very few)
        assert len(anomalies) == 0

    def test_anomaly_columns(self, sample_validation_data):
        """Test that anomalies have required columns."""
        df = sample_validation_data
        anomalies = detect_kicking_anomalies(df, verbose=False)
        
        if len(anomalies) > 0:
            assert 'player_name' in anomalies.columns
            assert 'position' in anomalies.columns
            assert 'likely_reason' in anomalies.columns


class TestCleanKickingData:
    """Test kicking data cleaning."""

    def test_remove_strategy(self, sample_validation_data):
        """Test 'remove' cleaning strategy."""
        df = sample_validation_data.copy()
        df_clean = clean_kicking_data(df, strategy='remove', verbose=False)
        
        # Should be same or smaller
        assert len(df_clean) <= len(df)

    def test_redistribute_strategy(self, sample_validation_data):
        """Test 'redistribute' cleaning strategy."""
        df = sample_validation_data.copy()
        df_clean = clean_kicking_data(df, strategy='redistribute', verbose=False)
        
        # Should be same size (redistributing, not removing)
        assert len(df_clean) == len(df)

    def test_anomalies_reduced(self, sample_validation_data):
        """Test that cleaning reduces anomalies."""
        df = sample_validation_data.copy()
        
        anomalies_before = detect_kicking_anomalies(df, verbose=False)
        df_clean = clean_kicking_data(df, strategy='remove', verbose=False)
        anomalies_after = detect_kicking_anomalies(df_clean, verbose=False)
        
        assert len(anomalies_after) <= len(anomalies_before)

    def test_data_integrity(self, sample_validation_data):
        """Test that cleaning preserves data integrity."""
        df = sample_validation_data.copy()
        original_cols = set(df.columns)
        
        df_clean = clean_kicking_data(df, strategy='remove', verbose=False)
        
        assert set(df_clean.columns) == original_cols


class TestValidatePositionScores:
    """Test position-based score validation."""

    def test_validation_returns_dict(self, sample_validation_data):
        """Test that validation returns a dict."""
        df = sample_validation_data
        result = validate_position_scores(df)
        
        assert isinstance(result, dict)

    def test_position_grouping(self):
        """Test position grouping in validation."""
        from rugby_ranking.utils.constants import KICKING_POSITIONS
        
        # Verify constants are correct
        assert 10 in KICKING_POSITIONS  # Fly-half
        assert 9 in KICKING_POSITIONS  # Scrum-half
        assert 1 not in KICKING_POSITIONS  # Prop

    def test_validates_kickers(self, sample_validation_data):
        """Test that validation correctly identifies kickers."""
        df = sample_validation_data
        result = validate_position_scores(df)
        
        # Should have analyzed kicking positions
        assert result is not None
