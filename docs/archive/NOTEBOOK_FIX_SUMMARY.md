# Notebook Fix Summary

## Problem
The notebook had reversions and couldn't handle the separate kicking model correctly. Specifically:
1. Cell 15 had a syntax error (using bare identifiers instead of strings as dict keys)
2. ModelConfig didn't explicitly show `separate_kicking_effect=True`
3. Some cells may have been missing the new `df` and `min_scores` parameters

## What is the "Separate Kicking Model"?

The `ModelConfig` has a setting `separate_kicking_effect=True` (default) which changes how player abilities are modeled:

**Without separate effects** (old):
- All scoring types use the same `beta_player` coefficient
- A player good at tries would also rank high for kicks

**With separate effects** (new, default):
- **Tries** use `beta_player_try` (try-scoring ability)
- **Penalties/Conversions/Drop Goals** use `beta_player_kick` (kicking ability)
- These are independent abilities - being a good try-scorer doesn't make you a good kicker

This is crucial because forwards (props, locks) score tries but rarely kick, while fly-halves kick but may score fewer tries.

## Fixes Applied

### Cell 5 - ModelConfig
**Issue**: Didn't explicitly show `separate_kicking_effect` setting

**Fix**: Added explicit parameters with comments
```python
config = ModelConfig(
    score_types=("tries", "penalties", "conversions", "drop_goals"),
    player_effect_sd=0.5,
    team_effect_sd=0.3,
    position_effect_sd=0.5,
    separate_kicking_effect=True,  # Explicitly set for clarity
    include_defense=True,
)
```

### Cell 15 - Visualization
**Issue**: Syntax error - used bare identifiers instead of strings
```python
# ✗ WRONG - causes NameError
min_threshold = {tries: 5, penalties: 15, ...}.get(score_type, 5)
```

**Fix**: Use string keys
```python
# ✓ CORRECT
min_threshold = {'tries': 10, 'penalties': 20, 'conversions': 20, 'drop_goals': 2}.get(score_type, 5)
```

Also ensured:
- Passes `df=df` and `min_scores=min_threshold` to `get_player_rankings()`
- Shows score counts in labels: `"Player Name (123)"`
- Updates title to show minimum threshold

### Cells 14 & 18 - Already Correct
These cells were already properly updated from the previous fix and didn't need changes.

## Validation Results

All checks passed ✅:

```
[Cell 5] ModelConfig:
  ✓ separate_kicking_effect explicitly set
  ✓ include_defense explicitly set

[Cell 14] show_rankings function:
  ✓ Function signature has min_scores parameter
  ✓ Function passes df and min_scores to get_player_rankings
  ✓ Tries minimum set to 10
  ✓ Kicks minimum set to 20

[Cell 15] Visualization:
  ✓ Using string keys in threshold dictionary
  ✓ Passing df and min_scores to get_player_rankings
  ✓ Checking for total_scores column

[Cell 18] Best Kickers:
  ✓ Both penalty and conversion rankings use df parameter
  ✓ Both penalty and conversion rankings use min_scores=20
```

## Impact

With these fixes, the notebook will:
1. ✅ Run without syntax errors
2. ✅ Correctly use separate kicking effects (tries vs kicks are independent)
3. ✅ Show score counts alongside rankings
4. ✅ Filter players by minimum thresholds
5. ✅ Display clear configuration so users understand the model

## Why This Matters for the Separate Kicking Model

The minimum thresholds are **especially important** with separate kicking effects because:

1. **Small sample problem**: With few kicking attempts, the posterior for `beta_player_kick` is very uncertain
2. **False positives**: A prop who successfully kicked 1-2 penalties might get a high posterior mean due to noise
3. **Independent effects**: Since try-scoring and kicking are separate, you can't rely on overall performance to filter out bad estimates

**Example**:
- Ed Byrne (prop): 2 successful penalties → Very uncertain `beta_player_kick` → Could rank high
- Owen Farrell (fly-half): 787 successful penalties → Precise `beta_player_kick` → Reliable ranking

With `min_scores=20`, only players with sufficient kicking attempts (like Farrell) appear in rankings.

## Files Modified
- `notebooks/02_model_fitting.ipynb` - Cells 5 and 15 fixed

## Validation Tool
Created `validate_notebook_fixes.py` which checks:
- ModelConfig parameters are explicit
- All ranking calls use `df` and `min_scores`
- No syntax errors in dictionary keys
- Proper thresholds applied
