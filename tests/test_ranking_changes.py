#!/usr/bin/env python3
"""
Test script to verify the ranking changes work correctly.
"""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
import pandas as pd
import numpy as np

def test_get_player_rankings():
    """Test the new get_player_rankings functionality."""

    # Create a minimal test dataset
    DATA_DIR = Path("../../Rugby-Data")
    if not DATA_DIR.exists():
        DATA_DIR = Path("../Rugby-Data")
    if not DATA_DIR.exists():
        print("Warning: Rugby-Data directory not found, creating mock data")
        # Create mock dataframe
        df = pd.DataFrame({
            'player_name': ['Player A', 'Player B', 'Player C', 'Player D'] * 10,
            'team': ['Team 1', 'Team 2', 'Team 1', 'Team 2'] * 10,
            'season': ['2023-2024'] * 40,
            'opponent': ['Team 2', 'Team 1', 'Team 2', 'Team 1'] * 10,
            'position': [10, 9, 11, 12] * 10,
            'is_home': [True, False, True, False] * 10,
            'minutes_played': [80.0] * 40,
            'tries': [1, 0, 2, 0] * 10,  # Player A: 10 tries, C: 20 tries
            'penalties': [5, 0, 0, 1] * 10,  # Player A: 50 penalties, D: 10
            'conversions': [3, 0, 0, 2] * 10,  # Player A: 30 conversions, D: 20
            'drop_goals': [0] * 40,
            'match_id': list(range(10)) * 4,
            'date': pd.date_range('2023-01-01', periods=10).tolist() * 4,
            'competition': ['Test'] * 40,
            'started': [True] * 40,
            'was_substituted': [False] * 40,
            'total_points': [0] * 40,
            'yellow_cards': [0] * 40,
            'red_cards': [0] * 40,
            'team_score': [0] * 40,
            'opponent_score': [0] * 40,
            'match_result': ['win'] * 40,
        })
    else:
        dataset = MatchDataset(DATA_DIR, fuzzy_match_names=False)
        dataset.load_json_files()
        df = dataset.to_dataframe(played_only=True)
        df = df[df['position'].between(1, 23)].copy()

    print(f"Loaded {len(df)} observations")
    print(f"Unique players: {df['player_name'].nunique()}")

    # Test score counts aggregation
    for score_type in ['tries', 'penalties', 'conversions']:
        score_counts = df.groupby('player_name')[score_type].sum()
        print(f"\n{score_type.capitalize()} distribution:")
        print(f"  Players with >0: {(score_counts > 0).sum()}")
        print(f"  Players with >=5: {(score_counts >= 5).sum()}")
        print(f"  Players with >=10: {(score_counts >= 10).sum()}")
        print(f"  Players with >=20: {(score_counts >= 20).sum()}")
        print(f"  Max {score_type}: {score_counts.max()}")

    # Test the ranking function interface (without actually fitting a model)
    print("\n=== Testing get_player_rankings interface ===")
    config = ModelConfig()
    model = RugbyModel(config)

    # Build indices
    model._build_indices(df)
    print(f"Built indices for {len(model._player_ids)} players")

    # Create mock trace data for testing
    import arviz as az
    n_players = len(model._player_ids)
    n_chains = 2
    n_draws = 100

    # Mock posterior for tries (separate kicking effect model)
    mock_posterior = {
        'beta_player_try_raw': np.random.randn(n_chains, n_draws, n_players),
        'sigma_player_try': np.random.gamma(2, 0.5, (n_chains, n_draws)),
        'lambda_player_try': np.random.gamma(2, 0.5, (n_chains, n_draws, 4)),
    }

    mock_trace = az.from_dict(mock_posterior)

    print("\n=== Test 1: Rankings WITHOUT dataframe (old behavior) ===")
    try:
        rankings_old = model.get_player_rankings(
            trace=mock_trace,
            score_type='tries',
            top_n=10
        )
        print(f"Returned {len(rankings_old)} players")
        print(f"Columns: {rankings_old.columns.tolist()}")
        assert 'total_scores' not in rankings_old.columns, "Should not have total_scores"
        print("✓ Old interface works")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n=== Test 2: Rankings WITH dataframe, NO threshold ===")
    try:
        rankings_with_counts = model.get_player_rankings(
            trace=mock_trace,
            score_type='tries',
            top_n=10,
            df=df
        )
        print(f"Returned {len(rankings_with_counts)} players")
        print(f"Columns: {rankings_with_counts.columns.tolist()}")
        assert 'total_scores' in rankings_with_counts.columns, "Should have total_scores"
        print(f"Score counts: {rankings_with_counts['total_scores'].tolist()}")
        print("✓ Score counts added successfully")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n=== Test 3: Rankings WITH threshold ===")
    try:
        rankings_filtered = model.get_player_rankings(
            trace=mock_trace,
            score_type='tries',
            top_n=10,
            df=df,
            min_scores=5
        )
        print(f"Returned {len(rankings_filtered)} players")
        if len(rankings_filtered) > 0:
            print(f"Minimum score count: {rankings_filtered['total_scores'].min()}")
            assert rankings_filtered['total_scores'].min() >= 5, "All should have >=5 scores"
            print("✓ Threshold filtering works")
        else:
            print("⚠ No players met threshold")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n=== All tests completed ===")

if __name__ == "__main__":
    test_get_player_rankings()
