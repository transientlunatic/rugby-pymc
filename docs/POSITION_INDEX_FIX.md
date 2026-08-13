# Position Indexing Fix

## Issue

When running the minibatch VI model (`build_joint_minibatch`), an `IndexError` occurred:

```
IndexError: index 40 is out of bounds for axis 1 with size 23
```

At line: `+ theta_position[s, position - 1]`

## Root Cause

The position values in the DataFrame were being used directly as 1-indexed values (1-23), with a manual `-1` adjustment to make them 0-indexed. However:

1. **Position values weren't validated** - The data contained position values outside the expected range (1-23)
2. **No position mapping** - Unlike players and teams which had proper index mappings, positions used raw values from the DataFrame
3. **Inconsistent with other indices** - The code used mapped indices for players (`_player_ids`) and teams (`_team_season_ids`) but not for positions

## Solution

Created a proper position indexing system consistent with other indices:

### 1. Added Position Mapping

```python
# In __init__
self._position_ids: dict[int, int] = {}  # Map raw position to 0-indexed
```

### 2. Built Position Index in `_build_indices`

```python
# Positions - map raw position values to 0-indexed
unique_positions = sorted(df["position"].unique())
self._position_ids = {pos: i for i, pos in enumerate(unique_positions)}
```

This creates a mapping from whatever position values exist in the data (could be 1-23, or 1-40, or anything) to a clean 0-indexed range [0, N-1] where N is the number of unique positions.

### 3. Updated `_prepare_data` to Map Positions

```python
# Map positions to 0-indexed values
position_idx = np.array([self._position_ids[p] for p in df_filtered["position"]])

return {
    ...
    "position_idx": position_idx,  # Changed from "position"
    ...
}
```

### 4. Updated All Model Methods

**Changed in `build()`, `build_joint()`, and `build_joint_minibatch()`:**

```python
# Before
position = pm.Data("position", data["position"])
n_positions = 23  # Hard-coded
...
+ theta_position[position - 1]  # Manual -1 adjustment

# After
position_idx = pm.Data("position_idx", data["position_idx"])
n_positions = len(self._position_ids)  # Dynamic based on data
...
+ theta_position[position_idx]  # Already 0-indexed
```

## Benefits

1. **Robust to Data Variations**: Handles any position values in the data, not just 1-23
2. **Consistent with Other Indices**: Uses the same pattern as player and team indices
3. **Dynamic Position Count**: `n_positions` now reflects the actual number of unique positions in the data
4. **No Magic Numbers**: Removed hard-coded `n_positions = 23`
5. **Clear Intent**: Variable name `position_idx` makes it clear it's an index, not a raw value

## Testing

All tests pass after the fix:

```bash
cd rugby-ranking
python test_defensive_minibatch.py
```

Output:
```
======================================================================
TEST SUMMARY
======================================================================

✓ Defensive effects implemented in ModelConfig
✓ Defensive effects in build() method
✓ Defensive effects in build_joint() method
✓ Minibatch VI support added (build_joint_minibatch)
✓ Defensive rankings extraction method
✓ Backward compatibility maintained (include_defense=False)
✓ InferenceConfig extended with minibatch settings

======================================================================
All tests completed!
======================================================================
```

## Files Modified

- [rugby_ranking/model/core.py](rugby_ranking/model/core.py)
  - Added `_position_ids` mapping
  - Updated `_build_indices()` to create position mapping
  - Updated `_prepare_data()` to use position indices
  - Changed all model building methods (`build`, `build_joint`, `build_joint_minibatch`)
  - Replaced `position - 1` with `position_idx` (8 occurrences)
  - Replaced hard-coded `n_positions = 23` with `len(self._position_ids)` (3 occurrences)

## Impact

- **No API changes**: The fix is internal to the model; users don't need to change their code
- **Backward compatible**: Works with existing data and notebooks
- **More robust**: Handles edge cases where position data might be malformed or use different numbering schemes

## Related Issues

This fix resolves the notebook error shown when running `04_defensive_effects_demo.ipynb` where the minibatch model would fail with an index out of bounds error during model building.
