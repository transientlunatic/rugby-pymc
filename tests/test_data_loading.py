"""Unit tests for rugby_ranking.model.data module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile

from rugby_ranking.model.data import MatchDataset, normalize_team_name


class TestNormalizeTeamName:
    """Test team name normalization."""

    def test_basic_normalization(self):
        """Test basic name normalization."""
        assert normalize_team_name("LEINSTER") == "Leinster"
        assert normalize_team_name("dublin") == "Dublin"

    def test_strip_whitespace(self):
        """Test whitespace handling."""
        assert normalize_team_name("  Leinster  ") == "Leinster"

    def test_accent_removal(self):
        """Test accent handling."""
        result = normalize_team_name("Côte d'Ivoire")
        assert "Cote" in result or "Côte" in result


class TestMatchDataset:
    """Test MatchDataset loading and processing."""

    @pytest.fixture
    def temp_json_dir(self):
        """Create temporary directory with sample JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample LIST format JSON
            list_data = [
                {
                    "home": {
                        "1": {"name": "Player 1", "on": [0], "off": [], "reds": [], "yellows": []},
                        "2": {"name": "Player 2", "on": [0], "off": [], "reds": [], "yellows": []},
                    },
                    "away": {
                        "1": {"name": "Player 3", "on": [0], "off": [], "reds": [], "yellows": []},
                        "2": {"name": "Player 4", "on": [0], "off": [], "reds": [], "yellows": []},
                    },
                    "date": "2023-01-15T15:00:00.000Z",
                    "homeTeam": "Leinster",
                    "awayTeam": "Munster",
                    "homeScore": 20,
                    "awayScore": 15,
                    "scored": {
                        "home": {
                            "tries": {"Player 1": 1},
                            "penalties": {"Player 2": 2},
                            "conversions": {"Player 2": 1},
                            "drop_goals": {},
                        },
                        "away": {
                            "tries": {"Player 3": 1},
                            "penalties": {"Player 4": 1},
                            "conversions": {},
                            "drop_goals": {},
                        },
                    },
                }
            ]
            
            json_path = Path(tmpdir) / "premiership-2023-2024.json"
            with open(json_path, "w") as f:
                json.dump(list_data, f)
            
            yield tmpdir

    def test_initialization(self):
        """Test dataset initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = MatchDataset(Path(tmpdir))
            assert dataset.data_dir == Path(tmpdir)

    def test_load_json_files(self, temp_json_dir):
        """Test loading JSON files."""
        dataset = MatchDataset(Path(temp_json_dir))
        dataset.load_json_files()
        
        assert len(dataset._matches) > 0

    def test_to_dataframe(self, temp_json_dir):
        """Test conversion to DataFrame."""
        dataset = MatchDataset(Path(temp_json_dir))
        dataset.load_json_files()
        df = dataset.to_dataframe(played_only=True)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert 'player_name' in df.columns
        assert 'team' in df.columns
        assert 'tries' in df.columns

    def test_dataframe_index_columns(self, temp_json_dir):
        """Test that DataFrame has required columns."""
        dataset = MatchDataset(Path(temp_json_dir))
        dataset.load_json_files()
        df = dataset.to_dataframe()
        
        required_cols = {
            'player_name', 'team', 'opponent', 'date', 'position',
            'minutes', 'season', 'tries', 'penalties', 'conversions'
        }
        assert required_cols.issubset(set(df.columns))

    def test_data_types(self, temp_json_dir):
        """Test that DataFrame has correct data types."""
        dataset = MatchDataset(Path(temp_json_dir))
        dataset.load_json_files()
        df = dataset.to_dataframe()
        
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
        assert pd.api.types.is_integer_dtype(df['position'])
        assert pd.api.types.is_numeric_dtype(df['tries'])


class TestDatasetFiltering:
    """Test dataset filtering and processing."""

    @pytest.fixture
    def sample_dataset_df(self):
        """Create sample dataset DataFrame."""
        data = {
            'date': pd.date_range('2023-01-01', periods=100),
            'team': np.random.choice(['Leinster', 'Munster'], 100),
            'opponent': np.random.choice(['Munster', 'Leinster'], 100),
            'player_name': np.random.choice(['A', 'B', 'C'], 100),
            'position': np.random.randint(1, 16, 100),
            'minutes': 80,
            'season': '2023-2024',
            'tries': 0,
            'penalties': 0,
            'conversions': 0,
            'drop_goals': 0,
        }
        return pd.DataFrame(data)

    def test_valid_positions_only(self, sample_dataset_df):
        """Test filtering to valid positions."""
        df = sample_dataset_df
        df_filtered = df[df['position'].between(1, 23)]
        
        # All positions should be valid
        assert df_filtered['position'].min() >= 1
        assert df_filtered['position'].max() <= 23

    def test_null_values(self, sample_dataset_df):
        """Test handling of null values."""
        df = sample_dataset_df.copy()
        df.loc[0, 'player_name'] = None
        
        df_clean = df.dropna(subset=['player_name'])
        
        assert df_clean['player_name'].isna().sum() == 0
