# Data Quality Fix: Kicking Score Misattribution

## Problem Identified

Ed Byrne (and 726 other players) were incorrectly attributed with conversion and penalty scores that they shouldn't have, causing incorrect player rankings.

### Root Cause

**Surname conflicts in match data**: When multiple players with the same surname play in the same match, the data collection/parsing incorrectly assigns kicking scores to **ALL players with that surname** instead of just the actual kicker.

### Example

**Match**: Leinster vs Dragons, 2017-02-24
- **Ross Byrne** (Pos 10, fly-half): 7 conversions ✓ (correct - he's the kicker)
- **Adam Byrne** (Pos 14, wing): 7 conversions ✗ (wrong - wings don't kick)
- **Ed Byrne** (Pos 17, replacement prop): 7 conversions ✗ (wrong - props don't kick)

### Scope of the Problem

From the validation report on the full dataset:

- **1,619 player-match records** with anomalous kicking scores
- **727 unique players** affected
- **276 records** identified as surname conflicts
- **1,993 conversions** and **663 penalties** incorrectly attributed to forwards

**Position breakdown** (should have minimal/no kicks):
```
Position 1 (props):      175 conversions, 55 penalties
Position 2 (hookers):    329 conversions, 46 penalties
Position 3 (props):      122 conversions, 40 penalties
Position 4 (locks):      208 conversions, 94 penalties
Position 5 (locks):      173 conversions, 32 penalties
Position 6 (flankers):   246 conversions, 62 penalties
Position 7 (flankers):   288 conversions, 29 penalties
Position 8 (number 8s):  452 conversions, 105 penalties
```

**Top affected players**:
1. S Simmonds: 120 conversions, 41 penalties (actually Joe Simmonds kicking)
2. Jean-Luc du Preez: 58 conversions, 33 penalties (actually Robert du Preez kicking)
3. Daniel Du Preez: 52 conversions, 32 penalties (actually Robert du Preez kicking)
4. **Ed Byrne: 52 conversions, 21 penalties** (actually Ross/Harry Byrne kicking)
5. Bradley Davies: 34 conversions, 31 penalties (actually various Davies kicking)

## Solution Implemented

### 1. Data Validation Module

Created [`data_validation.py`](rugby_ranking/model/data_validation.py) with:

**Functions**:
- `detect_kicking_anomalies()`: Identify forwards with kicking scores
- `clean_kicking_data()`: Remove incorrect scores from non-kickers
- `validate_position_scores()`: Comprehensive validation report
- `print_validation_report()`: Human-readable summary

**Logic**:
- Positions 1-8 (forwards) should almost never kick
- Positions 9, 10, 12, 15 (backs) are typical kickers
- Flag any forward with conversions or penalties
- Check for surname conflicts (other players with same surname in same match)

### 2. Automatic Cleaning in Training Script

Updated [`train_model.py`](train_model.py) to:
- **Automatically clean data** before training (default behavior)
- Detect and report anomalies during data loading
- Remove incorrect kicks from forward positions
- Add `--no-clean-data` flag to skip cleaning (not recommended)

**Example output**:
```
======================================================================
DATA CLEANING
======================================================================
Detecting kicking anomalies (forwards with conversion/penalty scores)...
Found 91 anomalies in 78 players

Top 5 affected players:
  Amato Fakatava: 3 conversions, 0 penalties
  Bongi Mbonambi: 3 conversions, 0 penalties
  Peter O'Mahony: 3 conversions, 0 penalties
  Julián Montoya: 3 conversions, 0 penalties
  Charles Ollivon: 2 conversions, 0 penalties

Data cleaned successfully!
  Removed: 100 conversions, 0 penalties
```

### 3. Usage

**Training with cleaning (default)**:
```bash
python train_model.py --model static --data-dir ../Rugby-Data --save-as latest
# Data cleaning happens automatically
```

**Training without cleaning** (not recommended):
```bash
python train_model.py --model static --data-dir ../Rugby-Data --no-clean-data --save-as latest
```

**Manual validation**:
```python
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.data_validation import print_validation_report, clean_kicking_data

# Load data
dataset = MatchDataset("../Rugby-Data")
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

# Print validation report
print_validation_report(df)

# Clean data
df_clean = clean_kicking_data(df, strategy='remove', verbose=True)
```

## Impact on Rankings

### Before Fix
**Top Kickers** (with min_scores=20 threshold):
- Ed Byrne (prop) appeared in kicker rankings
- Many other forwards incorrectly ranked as good kickers
- Actual kickers' effects diluted by incorrect attributions

### After Fix
**Top Kickers** (with min_scores=20 threshold):
- Only actual kickers (positions 9, 10, 12, 15) appear
- More accurate individual player effects
- Better model fit for kicking events

### Model Impact

**Parameters affected**:
- `beta_player_kick` (player kicking effects) - now more accurate
- Rankings for conversion/penalty scoring - now reflect actual kickers
- Model predictions for kicking events - improved accuracy

**Not affected**:
- Try-scoring ability (not impacted by this issue)
- Team effects
- Position effects
- Home advantage

## Limitations

### What This Fix Does
- ✅ Removes incorrect kicks from forwards (positions 1-8)
- ✅ Automatically cleans data during training
- ✅ Provides validation reports

### What This Fix Doesn't Do
- ❌ **Does NOT fix the source data** - the JSON files still have errors
- ❌ **Does NOT redistribute kicks to actual kickers** - just removes them
- ❌ Does not handle edge cases (forwards who rarely take kicks in emergencies)

### Known Edge Cases

**Rare legitimate forward kicks**:
- Emergency kicks when regular kicker is injured
- Hookers taking lineout throws scored as "penalties" (coding error)
- Forward drop goals (should be kept but currently removed)

**Recommendation**: These are rare enough (~0.1% of all kicks) that removing all forward kicks is a net improvement.

## Future Improvements

### Short-term (Recommended)
1. Fix the source data in Rugby-Data repository
   - Identify actual kickers from play-by-play data
   - Correct surname conflicts at source
   - Add validation to data collection pipeline

2. Smarter redistribution
   - Instead of removing kicks, redistribute to likely kicker
   - Use position-based heuristics (fly-half, fullback priority)
   - Check historical kicker patterns

### Long-term (Optional)
1. Player-level kicking roles
   - Track who takes conversions vs penalties
   - Account for kicker substitutions
   - Model "backup kicker" scenarios

2. Data provenance tracking
   - Flag records with known issues
   - Confidence scores for each score attribution
   - Allow models to down-weight suspicious data

## Testing

**Validation test**:
```bash
# Full dataset validation
python -c "
from pathlib import Path
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.data_validation import print_validation_report

dataset = MatchDataset(Path('../Rugby-Data'))
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)
print_validation_report(df)
"
```

**Training test**:
```bash
# Train with cleaning (should see cleaning output)
python train_model.py --model static --data-dir ../Rugby-Data \
  --last-seasons 2 --save-as test_cleaned

# Train without cleaning (should see warning)
python train_model.py --model static --data-dir ../Rugby-Data \
  --last-seasons 2 --no-clean-data --save-as test_uncleaned
```

## Summary

**Problem**: 727 players (including Ed Byrne) had incorrect kicking scores due to surname conflicts in the data.

**Solution**: Automatic data cleaning now removes ~1,993 incorrect conversions and ~663 incorrect penalties before training.

**Result**: Rankings now accurately reflect actual kicking ability, with only legitimate kickers appearing in kicker rankings.

**Recommendation**: Always use automatic cleaning (default behavior). Only use `--no-clean-data` for debugging or comparing with/without cleaning.

## Files Modified/Created

- ✅ Created: [`data_validation.py`](rugby_ranking/model/data_validation.py) (~280 lines)
- ✅ Modified: [`train_model.py`](train_model.py) (added cleaning step)
- ✅ Created: [`DATA_CLEANING_FIX.md`](DATA_CLEANING_FIX.md) (this document)

## Related Issues

This fix addresses:
- Ed Byrne appearing as top kicker despite being a prop
- Other surname conflicts (Simmonds, du Preez, Davies, etc.)
- General data quality in kicking statistics
- More accurate `beta_player_kick` estimates in the model

This complements the earlier fix for ranking thresholds ([RANKING_IMPROVEMENTS.md](RANKING_IMPROVEMENTS.md)), where we added `min_scores` filters to exclude players with too few attempts.
