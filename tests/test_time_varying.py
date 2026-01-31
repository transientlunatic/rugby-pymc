#!/usr/bin/env python3
"""
Test the time-varying effects model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
import pandas as pd

def test_time_varying_model():
    """Test that the time-varying model builds correctly."""

    print("=" * 70)
    print("TESTING TIME-VARYING EFFECTS MODEL")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    DATA_DIR = Path("../Rugby-Data")
    dataset = MatchDataset(DATA_DIR, fuzzy_match_names=False)
    dataset.load_json_files()
    df = dataset.to_dataframe(played_only=True)
    df = df[df['position'].between(1, 23)].copy()

    # Filter to smaller subset for faster testing
    print(f"   Total observations: {len(df)}")
    recent_seasons = df['season'].unique()[-3:]  # Last 3 seasons
    df_subset = df[df['season'].isin(recent_seasons)].copy()
    print(f"   Test subset (last 3 seasons): {len(df_subset)} observations")
    print(f"   Seasons: {list(recent_seasons)}")

    # Test 1: Build static model (baseline)
    print("\n2. Building static model (baseline)...")
    config_static = ModelConfig(
        score_types=("tries", "penalties"),  # Just 2 types for speed
        separate_kicking_effect=True,
        time_varying_effects=False,  # Static
    )
    model_static = RugbyModel(config_static)
    try:
        pymc_model_static = model_static.build_joint(df_subset)
        print(f"   ✓ Static model built successfully")
        print(f"   Variables: {list(pymc_model_static.named_vars.keys())[:10]}...")
    except Exception as e:
        print(f"   ✗ Static model failed: {e}")
        return False

    # Test 2: Build time-varying model
    print("\n3. Building time-varying model...")
    config_tv = ModelConfig(
        score_types=("tries", "penalties"),
        separate_kicking_effect=True,
        time_varying_effects=True,  # Time-varying
        player_trend_sd=0.1,
        team_trend_sd=0.1,
    )
    model_tv = RugbyModel(config_tv)
    try:
        pymc_model_tv = model_tv.build_joint_time_varying(df_subset)
        print(f"   ✓ Time-varying model built successfully")
        print(f"   Variables: {list(pymc_model_tv.named_vars.keys())[:15]}...")
    except Exception as e:
        print(f"   ✗ Time-varying model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Check model structure
    print("\n4. Checking model structure...")

    # Check for time-varying parameters
    expected_vars = [
        'beta_player_try_base_raw',
        'beta_player_try_trend_raw',
        'beta_player_kick_base_raw',
        'beta_player_kick_trend_raw',
        'gamma_team_base_raw',
        'gamma_team_trend_raw',
    ]

    for var_name in expected_vars:
        if var_name in pymc_model_tv.named_vars:
            var = pymc_model_tv.named_vars[var_name]
            print(f"   ✓ {var_name}: shape {var.eval().shape}")
        else:
            print(f"   ✗ {var_name}: NOT FOUND")
            return False

    # Test 4: Verify season_progress data
    print("\n5. Checking season_progress data...")
    for score_type in config_tv.score_types:
        data_var_name = f"season_progress_{score_type}"
        if data_var_name in pymc_model_tv.named_vars:
            progress_data = pymc_model_tv.named_vars[data_var_name].eval()
            print(f"   ✓ {data_var_name}: min={progress_data.min():.3f}, "
                  f"max={progress_data.max():.3f}, mean={progress_data.mean():.3f}")
            if progress_data.min() < 0 or progress_data.max() > 1:
                print(f"   ⚠ Warning: season_progress outside [0, 1]!")
        else:
            print(f"   ✗ {data_var_name}: NOT FOUND")
            return False

    # Test 5: Compare model complexities
    print("\n6. Model comparison...")
    n_vars_static = len([v for v in pymc_model_static.named_vars
                         if not v.startswith('player_idx')
                         and not v.startswith('y_')])
    n_vars_tv = len([v for v in pymc_model_tv.named_vars
                     if not v.startswith('player_idx')
                     and not v.startswith('y_')])

    print(f"   Static model: {n_vars_static} random variables")
    print(f"   Time-varying model: {n_vars_tv} random variables")
    print(f"   Additional complexity: +{n_vars_tv - n_vars_static} variables")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nTime-varying model structure:")
    print("  - Player base effects per season")
    print("  - Player trend slopes per season")
    print("  - Team base effects per team-season")
    print("  - Team trend slopes per team-season")
    print("  - Separate try-scoring vs kicking abilities")
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = test_time_varying_model()
    exit(0 if success else 1)
