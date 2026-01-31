"""Test script for separate kicking/try-scoring player effects."""

import numpy as np
import pandas as pd
from rugby_ranking.model.core import ModelConfig, RugbyModel
from rugby_ranking.model.data import MatchDataset

# Create a small test dataset
data = {
    "player_name": ["Player A", "Player B", "Player A", "Player B"] * 5,
    "team": ["Team X", "Team X", "Team Y", "Team Y"] * 5,
    "season": ["2023-2024"] * 20,
    "position": [10, 15, 10, 15] * 5,
    "minutes_played": [80] * 20,
    "is_home": [True, True, False, False] * 5,
    "tries": [1, 2, 0, 1, 1, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 1, 2, 0, 1],
    "penalties": [2, 0, 3, 1, 2, 0, 1, 0, 2, 0, 3, 1, 2, 0, 1, 0, 2, 0, 3, 1],
    "conversions": [1, 2, 0, 1, 1, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 1, 2, 0, 1],
    "drop_goals": [0] * 20,
}

df = pd.DataFrame(data)

print("Testing model with separate kicking/try-scoring effects...")
print("=" * 60)

# Test 1: Build model with separate effects (default)
print("\n1. Building model with separate_kicking_effect=True...")
config1 = ModelConfig(separate_kicking_effect=True)
model1 = RugbyModel(config=config1)

try:
    model1.build_joint(df)
    print("   ✓ Model built successfully!")
    print(f"   Model variables: {list(model1.model.named_vars.keys())[:10]}...")

    # Check that the expected variables exist
    expected_vars = ["beta_player_try_raw", "beta_player_kick_raw",
                     "sigma_player_try", "sigma_player_kick",
                     "lambda_player_try", "lambda_player_kick"]

    model_vars = list(model1.model.named_vars.keys())
    for var in expected_vars:
        if var in model_vars:
            print(f"   ✓ Found variable: {var}")
        else:
            print(f"   ✗ Missing variable: {var}")

except Exception as e:
    print(f"   ✗ Error building model: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Build model without separate effects (backward compatibility)
print("\n2. Building model with separate_kicking_effect=False...")
config2 = ModelConfig(separate_kicking_effect=False)
model2 = RugbyModel(config=config2)

try:
    model2.build_joint(df)
    print("   ✓ Model built successfully!")
    print(f"   Model variables: {list(model2.model.named_vars.keys())[:10]}...")

    # Check that the expected variables exist
    expected_vars = ["beta_player_raw", "sigma_player", "lambda_player"]

    model_vars = list(model2.model.named_vars.keys())
    for var in expected_vars:
        if var in model_vars:
            print(f"   ✓ Found variable: {var}")
        else:
            print(f"   ✗ Missing variable: {var}")

except Exception as e:
    print(f"   ✗ Error building model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing complete!")
