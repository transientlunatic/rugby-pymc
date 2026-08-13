# Ranking Improvements Summary

## Problem
Player rankings showed unexpected results, such as props like Ed Byrne appearing as top kickers. This occurred because:

1. **No minimum thresholds** - Players with just 1-2 kicks ranked alongside those with 50+ kicks
2. **No visibility into attempt counts** - Rankings only showed model effects, not actual scoring statistics
3. **Combined effect inflation** - The "Best Kickers" section added penalty and conversion effects together, even though they share the same underlying coefficient

## Solution
Added score counts and minimum thresholds to the ranking system.

### Changes Made

#### 1. Modified `get_player_rankings()` Method
**File:** `rugby_ranking/model/core.py:611-714`

Added two new optional parameters:
- `df: pd.DataFrame | None = None` - The match dataset for computing score counts
- `min_scores: int | None = None` - Minimum number of scores required for inclusion

**New functionality:**
- Computes total scores per player from the dataframe
- Adds `total_scores` column to rankings output
- Filters players below the minimum threshold before ranking
- Maintains backward compatibility (both parameters are optional)

**Example usage:**
```python
# Old way (still works)
rankings = model.get_player_rankings(trace=trace, score_type='tries', top_n=20)

# New way with score counts
rankings = model.get_player_rankings(
    trace=trace,
    score_type='tries',
    top_n=20,
    df=df,
    min_scores=10  # Only players with 10+ tries
)
```

#### 2. Updated Notebook Cells
**File:** `notebooks/02_model_fitting.ipynb`

**Cell 14 - `show_rankings()` function:**
- Added `min_scores` parameter with default of 5
- Passes `df` and `min_scores` to `get_player_rankings()`
- Updated display message to show threshold
- Different thresholds for each score type:
  - Tries: minimum 10
  - Penalties: minimum 20
  - Conversions: minimum 20

**Cell 15 - Visualization:**
- Added minimum thresholds per score type
- Shows score counts in parentheses next to player names
- Updated title to indicate minimum threshold

**Cell 18 - Best Kickers:**
- Requires minimum 20 penalties AND 20 conversions
- Shows `total_scores_pen`, `total_scores_con`, and `total_kicks` columns
- Updated visualization title to show threshold requirements

### Thresholds Applied

Based on the data distribution:

| Score Type | Minimum Threshold | Players Affected |
|------------|-------------------|------------------|
| Tries | 10 | 1,326 players (from 6,247) |
| Penalties | 20 | 303 players (from 1,029) |
| Conversions | 20 | 315 players (from 2,620) |
| Drop Goals | 2 | N/A |

### Testing

Created `test_ranking_changes.py` which verifies:
- ✅ Backward compatibility (old interface still works)
- ✅ Score counts added correctly when `df` provided
- ✅ Threshold filtering works as expected
- ✅ All players in filtered rankings meet minimum requirement

**Test results:**
```
=== Test 1: Rankings WITHOUT dataframe (old behavior) ===
✓ Old interface works

=== Test 2: Rankings WITH dataframe, NO threshold ===
✓ Score counts added successfully

=== Test 3: Rankings WITH threshold ===
✓ Threshold filtering works
```

### Benefits

1. **Eliminates outliers** - Props with 1-2 kicks no longer appear in top kicker rankings
2. **Transparency** - Users can see exactly how many scores each player has
3. **Statistical validity** - Rankings based on sufficient sample sizes
4. **Flexibility** - Thresholds can be adjusted per analysis needs
5. **Backward compatible** - Existing code continues to work

### Example Output

**Before (no thresholds):**
```
Top 20 Penalty Kickers:
   player           effect_mean  effect_std  ...
0  Ed Byrne         0.85         0.42        ...
1  Jonny Wilkinson  0.82         0.15        ...
```

**After (with thresholds):**
```
Top 20 Penalty Kickers (minimum 20 penalties):
   player           effect_mean  effect_std  ...  total_scores
0  Jonny Wilkinson  0.82         0.15        ...  787
1  Dan Carter       0.78         0.16        ...  542
2  Ronan O'Gara     0.75         0.17        ...  456
...
```

### Files Modified
- `rugby_ranking/model/core.py` - Added parameters and logic
- `notebooks/02_model_fitting.ipynb` - Updated 3 cells (14, 15, 18)

### Files Created
- `update_notebook_rankings.py` - Script to update notebook cells
- `update_viz_cell.py` - Script to update visualization cell
- `test_ranking_changes.py` - Test suite for changes
- `RANKING_IMPROVEMENTS.md` - This document

### Backup
Original notebook backed up to: `notebooks/02_model_fitting.ipynb.backup`
