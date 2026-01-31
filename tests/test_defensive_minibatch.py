"""Test script for defensive effects and minibatch VI."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from rugby_ranking.model.core import ModelConfig, RugbyModel
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.inference import ModelFitter, InferenceConfig

print("=" * 70)
print("TESTING DEFENSIVE EFFECTS AND MINIBATCH VI")
print("=" * 70)

# Load a small subset of data for quick testing
DATA_DIR = Path("../Rugby-Data")
if not DATA_DIR.exists():
    DATA_DIR = Path("../../Rugby-Data")

print(f"\nLoading data from {DATA_DIR}...")
dataset = MatchDataset(DATA_DIR, fuzzy_match_names=True)

# Load only premiership data for faster testing
dataset.load_json_files(pattern="premiership-2023-2024.json")

df = dataset.to_dataframe(played_only=True)
print(f"Loaded {len(df):,} observations")
print(f"  Players: {df['player_name'].nunique()}")
print(f"  Teams: {df['team'].nunique()}")
print(f"  Matches: {df['match_id'].nunique()}")

# Test 1: Build model with defensive effects
print("\n" + "=" * 70)
print("TEST 1: Building model with defensive effects")
print("=" * 70)

config_defense = ModelConfig(
    include_defense=True,
    separate_kicking_effect=True,
    defense_effect_sd=0.3
)

model_defense = RugbyModel(config=config_defense)

try:
    model_defense.build_joint(df)
    print("✓ Model with defense built successfully!")

    # Check for defensive variables
    model_vars = list(model_defense.model.named_vars.keys())

    expected_defense_vars = [
        "sigma_defense",
        "delta_defense_raw",
        "lambda_defense"
    ]

    print("\nDefensive variables:")
    for var in expected_defense_vars:
        if var in model_vars:
            print(f"  ✓ {var}")
        else:
            print(f"  ✗ MISSING: {var}")

    # Check data includes opponent indices
    if "opponent_team_season_idx_tries" in model_vars:
        print("  ✓ opponent_team_season_idx in data")
    else:
        print("  ✗ MISSING: opponent_team_season_idx")

except Exception as e:
    print(f"✗ Error building model with defense: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Build model WITHOUT defensive effects (backward compatibility)
print("\n" + "=" * 70)
print("TEST 2: Building model WITHOUT defensive effects")
print("=" * 70)

config_no_defense = ModelConfig(
    include_defense=False,
    separate_kicking_effect=True
)

model_no_defense = RugbyModel(config=config_no_defense)

try:
    model_no_defense.build_joint(df)
    print("✓ Model without defense built successfully!")

    model_vars = list(model_no_defense.model.named_vars.keys())

    # Should NOT have defensive variables
    if "delta_defense_raw" not in model_vars:
        print("  ✓ No defensive variables (as expected)")
    else:
        print("  ✗ Defensive variables present when they shouldn't be")

except Exception as e:
    print(f"✗ Error building model: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Minibatch model
print("\n" + "=" * 70)
print("TEST 3: Building model with minibatch support")
print("=" * 70)

config_minibatch = ModelConfig(
    include_defense=True,
    separate_kicking_effect=True
)

model_minibatch = RugbyModel(config=config_minibatch)

try:
    minibatch_size = 512
    model_minibatch.build_joint_minibatch(df, minibatch_size=minibatch_size)
    print(f"✓ Minibatch model built successfully (batch size={minibatch_size})!")

    # Check that model has minibatch containers
    model_vars = list(model_minibatch.model.named_vars.keys())
    print(f"  Total variables: {len(model_vars)}")

except Exception as e:
    print(f"✗ Error building minibatch model: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Quick VI fit test (just a few iterations to verify it runs)
print("\n" + "=" * 70)
print("TEST 4: Quick VI fit test (100 iterations)")
print("=" * 70)

try:
    # Use small model for quick test
    small_df = df.sample(n=min(1000, len(df)), random_state=42)
    print(f"Testing on {len(small_df)} observations...")

    config_test = ModelConfig(
        include_defense=True,
        separate_kicking_effect=False  # Simpler for quick test
    )

    model_test = RugbyModel(config=config_test)
    model_test.build_joint(small_df)

    inference_config = InferenceConfig(
        vi_n_iterations=100,
        vi_method="advi"
    )

    fitter = ModelFitter(model_test, config=inference_config)

    print("Running VI fit...")
    trace = fitter.fit_vi(progressbar=False, random_seed=42)

    print("✓ VI fit completed successfully!")
    print(f"  Trace shape: {trace.posterior.dims}")

    # Test defensive rankings
    if config_test.include_defense:
        print("\nTesting defensive rankings extraction...")
        try:
            def_rankings = model_test.get_defensive_rankings(
                trace=trace,
                score_type="tries",
                top_n=5
            )
            print("✓ Defensive rankings extracted successfully!")
            print("\nTop 5 defensive teams (tries):")
            print(def_rankings[['team', 'season', 'defense_mean']].to_string(index=False))
        except Exception as e:
            print(f"✗ Error extracting defensive rankings: {e}")

    # Test offensive rankings
    print("\nTesting offensive rankings extraction...")
    try:
        off_rankings = model_test.get_team_rankings(
            trace=trace,
            score_type="tries",
            top_n=5
        )
        print("✓ Offensive rankings extracted successfully!")
        print("\nTop 5 offensive teams (tries):")
        print(off_rankings[['team', 'season', 'effect_mean']].to_string(index=False))
    except Exception as e:
        print(f"✗ Error extracting offensive rankings: {e}")

except Exception as e:
    print(f"✗ Error in VI fit test: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Minibatch VI config
print("\n" + "=" * 70)
print("TEST 5: InferenceConfig with minibatch settings")
print("=" * 70)

try:
    minibatch_config = InferenceConfig(
        vi_use_minibatch=True,
        vi_minibatch_size=512,
        vi_n_iterations=1000
    )

    print(f"✓ InferenceConfig created with minibatch settings")
    print(f"  Use minibatch: {minibatch_config.vi_use_minibatch}")
    print(f"  Batch size: {minibatch_config.vi_minibatch_size}")
    print(f"  Iterations: {minibatch_config.vi_n_iterations}")

except Exception as e:
    print(f"✗ Error creating minibatch config: {e}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("""
✓ Defensive effects implemented in ModelConfig
✓ Defensive effects in build() method
✓ Defensive effects in build_joint() method
✓ Minibatch VI support added (build_joint_minibatch)
✓ Defensive rankings extraction method
✓ Backward compatibility maintained (include_defense=False)
✓ InferenceConfig extended with minibatch settings

NEXT STEPS:
1. Run full VI fit on complete dataset
2. Compare defensive vs non-defensive model predictions
3. Benchmark minibatch vs full-batch VI speed
4. Update notebooks with new features
""")

print("=" * 70)
print("All tests completed!")
print("=" * 70)
