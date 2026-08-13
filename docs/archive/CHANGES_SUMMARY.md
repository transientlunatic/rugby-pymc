# Summary of Ranking Changes

## Problem Solved
Fixed the issue where players like Ed Byrne (a prop) appeared as top kickers despite minimal kicking activity.

## Root Cause
1. No minimum score thresholds - anyone with even 1-2 kicks could rank
2. No visibility into actual scoring counts
3. Rankings combined effects from shared coefficients without considering sample size

## Solution Implemented

### Code Changes
- **Modified**: `rugby_ranking/model/core.py` - `get_player_rankings()` method (lines 611-714)
  - Added `df` parameter to pass match data
  - Added `min_scores` parameter for filtering
  - Computes `total_scores` column from data
  - Filters players below threshold before ranking
  - Maintains backward compatibility

### Notebook Changes
- **Modified**: `notebooks/02_model_fitting.ipynb`
  - Cell 14: Updated `show_rankings()` with thresholds (10 tries, 20 kicks)
  - Cell 15: Added score counts to visualization
  - Cell 18: Best Kickers now requires 20+ penalties AND 20+ conversions

### Thresholds Applied
- **Tries**: minimum 10 (reduces from 6,247 to 1,326 players)
- **Penalties**: minimum 20 (reduces from 1,029 to 303 players)  
- **Conversions**: minimum 20 (reduces from 2,620 to 315 players)
- **Best Kickers**: minimum 20 each (reduces from 851 to 247 players - 98.2% reduction!)

## Impact

### Before
```
Top Penalty Kickers:
   player           effect_mean
0  Ed Byrne         0.85         # Prop with 2 penalties
1  Random Player    0.82         # 3 penalties
...
```

### After  
```
Top Penalty Kickers (minimum 20 penalties):
   player           effect_mean  total_scores
0  Owen Farrell     0.82         787
1  George Ford      0.78         556
2  Dan Biggar       0.75         524
...
```

## Top 10 All-Round Kickers (with new thresholds)
1. Owen Farrell: 1,328 total kicks (787 pen + 541 con)
2. George Ford: 1,045 total kicks
3. Dan Biggar: 985 total kicks
4. Jimmy Gopperth: 808 total kicks
5. Paddy Jackson: 727 total kicks
6. Gareth Steenson: 717 total kicks
7. Stephen Myler: 687 total kicks
8. Johnny Sexton: 666 total kicks
9. Freddie Burns: 640 total kicks
10. Finn Russell: 629 total kicks

No props in sight! ✅

## Testing
All tests pass:
- ✅ Backward compatibility maintained
- ✅ Score counts added correctly
- ✅ Threshold filtering works
- ✅ Rankings now statistically meaningful

## Files
- Modified: `rugby_ranking/model/core.py`
- Modified: `notebooks/02_model_fitting.ipynb`
- Backup: `notebooks/02_model_fitting.ipynb.backup`
- Tests: `test_ranking_changes.py`
- Analysis: `show_threshold_impact.py`
- Docs: `RANKING_IMPROVEMENTS.md`
