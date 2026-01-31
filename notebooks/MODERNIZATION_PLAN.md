# Notebook Modernization Plan

## Current State Analysis

### Notebook Inventory

| # | Notebook | Status | Lines | Priority |
|---|----------|--------|-------|----------|
| 1 | 01_data_exploration.ipynb | Active | 1,059 | HIGH - Foundation |
| 2 | 02_model_fitting.ipynb | Active | 2,973 | HIGH - Core analysis |
| 3 | 03_predictions.ipynb | **EMPTY** | 16 | SKIP |
| 4 | 04_defensive_effects_demo.ipynb | Active | 3,476 | HIGH - Large |
| 5 | 05_time_varying_effects.ipynb | Active | 585 | MEDIUM |
| 6 | 06_league_table_and_season_prediction.ipynb | Active | 537 | MEDIUM |
| 6b | 06_...streamlined.ipynb | Backup | 558 | SKIP (backup) |
| 7 | 07_paths_to_victory_demo.ipynb | Active | 729 | MEDIUM |
| 8 | 08_squad_analysis_demo.ipynb | Active | 711 | MEDIUM |

**Total**: 9 notebooks (1 empty, 1 backup)
**Workload**: 8 active notebooks to modernize

---

## Boilerplate Patterns Found

### Pattern 1: Import Setup (Found in: 01, 02, 04, 05, 06, 07, 08)

**Current** (15+ lines per notebook):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.predictions import MatchPredictor
from rugby_ranking.cli import load_checkpoint

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
%matplotlib inline
```

**Replace with** (1 line):
```python
from rugby_ranking.notebook_utils import setup_notebook_environment
dataset, df, model_dir = setup_notebook_environment()
```

### Pattern 2: Data Loading (Found in: 01, 02, 04, 05, 06, 07, 08)

**Current** (10+ lines per notebook):
```python
DATA_DIR = Path("../../Rugby-Data")
dataset = MatchDataset(DATA_DIR, fuzzy_match_names=False)
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)
df = df[df['position'].between(1, 23)].copy()

print(f"Data loaded: {len(df):,} observations")
print(f"Seasons: {df['season'].nunique()}")
print(f"Players: {df['player_name'].nunique()}")
```

**Replace with** (Already done by `setup_notebook_environment()`)

### Pattern 3: Checkpoint Loading (Found in: 02, 04, 05, 07, 08)

**Current**:
```python
from rugby_ranking.cli import load_checkpoint
model, trace = load_checkpoint("latest", verbose=True)
```

**Improvement available**:
```python
from rugby_ranking.notebook_utils import load_model_and_trace
model, trace = load_model_and_trace("latest")
```

### Pattern 4: Rankings Display (Found in: 02, 04, 05, 07)

**Current** (Varies, ~10-20 lines):
```python
rankings = model.get_player_rankings(trace, score_type='tries')
rankings = rankings.sort_values('median_effect', ascending=False)
top_20 = rankings.head(20)
display(top_20[['player', 'mean_effect', 'median_effect', 'std_effect']])
```

**Can use helper**:
```python
from rugby_ranking.notebook_utils import get_top_players
get_top_players(trace, model, score_type="tries", top=20)
```

### Pattern 5: Summary Statistics (Found in: 01, 02, 04, 05, 06, 07)

**Current** (Varies):
```python
print(f"Players: {df['player_name'].nunique():,}")
print(f"Teams: {df['team'].nunique()}")
print(f"Matches: {df.groupby(['date', 'team']).ngroups:,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
```

**Can use helper**:
```python
from rugby_ranking.notebook_utils import print_summary
print_summary(df, "Dataset Overview")
```

---

## Feature Coverage Analysis

### Fully Demonstrated ✅

- ✅ Data loading and exploration (01)
- ✅ Model building (02)
- ✅ Player rankings (02)
- ✅ Team rankings (02)
- ✅ Match predictions (02)
- ✅ Defensive effects (04)
- ✅ Time-varying effects (05)
- ✅ League tables (06)
- ✅ Season prediction (06)
- ✅ Paths to victory (07, heuristic-based)
- ✅ Squad analysis (08)

### Partially Demonstrated ⚠️

- ⚠️ **Fuzzy name matching**: Used in data loading but not analyzed/demonstrated
- ⚠️ **Separate kicking effects**: Model config shown but not detailed analysis
- ⚠️ **Position effects**: Visualized but not detailed breakdown
- ⚠️ **Model validation**: No train/test split or cross-validation demo
- ⚠️ **Prediction uncertainty**: Shown but not formally analyzed
- ⚠️ **Data validation**: No demonstration of cleaning/anomaly detection

### Not Demonstrated ❌

- ❌ **Validation module**: `model.validation` has train/test splits but no notebook
- ❌ **Name analysis**: `model.name_analysis` tools not shown
- ❌ **Position groupings**: Can aggregate positions but not shown
- ❌ **Bracket prediction**: `model.bracket` module exists but not demonstrated
- ❌ **Error handling**: No notebook showing failure modes
- ❌ **Hyperparameter tuning**: No systematic parameter exploration
- ❌ **Posterior predictive checks**: Model fit assessment
- ❌ **Inference diagnostics**: Rhat, ESS, divergences

---

## Recommended Updates

### Priority 1: Reduce Boilerplate (All notebooks)
- Replace imports with `setup_notebook_environment()`
- Use `notebook_utils` helpers throughout
- Estimated savings: **20-30 lines per notebook** × 8 = 160-240 lines total

### Priority 2: Add Missing Demonstrations (New notebooks or additions)
1. **09_validation_and_diagnostics.ipynb**
   - Train/test split strategies
   - Model diagnostics (Rhat, ESS, divergences)
   - Posterior predictive checks
   - Prediction calibration

2. **10_data_validation_and_cleaning.ipynb**
   - Kicking anomaly detection
   - Data cleaning strategies
   - Player name matching analysis
   - Validation report

3. **11_hyperparameter_exploration.ipynb** (Optional)
   - Prior sensitivity
   - Model configuration comparison
   - Impact of separate kicking effects

### Priority 3: Enhance Existing Notebooks
- **02**: Add position effect breakdown, separate kicking comparison
- **04**: Use new visualization utilities
- **05**: Compare time-varying vs static model
- **06**: Show multiple bonus point systems side-by-side
- **07**: Add uncertainty quantification
- **08**: Add injury impact analysis

---

## Implementation Strategy

### Phase 1: Quick Wins (30 min)
1. Create compact version of 01 (data exploration)
2. Reduce 02 boilerplate
3. Test setup works across all notebooks

### Phase 2: Major Cleanup (60 min)
1. Update all 8 active notebooks with new imports
2. Replace ranking/summary boilerplate with helpers
3. Verify all execute without errors

### Phase 3: Feature Additions (Optional)
1. Create validation/diagnostics notebook
2. Create data quality notebook
3. Enhance existing notebooks with new features

---

## Estimated Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total lines | ~10,000 | ~9,200 | -8% |
| Average per notebook | ~1,250 | ~1,150 | -100 lines |
| Import boilerplate | 8×15 = 120 | 8×2 = 16 | -104 lines |
| Data loading duplication | 8×10 = 80 | 0 | -80 lines |
| Features demonstrated | 10 | 13-15 | +3-5 new |
| Test coverage | Partial | Comprehensive | ✅ |

---

## File Organization

After updates:
```
notebooks/
├── 01_data_exploration.ipynb          # Compact, uses utils
├── 02_model_fitting.ipynb             # Streamlined, many features
├── 03_predictions.ipynb               # [Keep empty, replace with real demo]
├── 04_defensive_effects_demo.ipynb    # Updated with new features
├── 05_time_varying_effects.ipynb      # Enhanced comparison
├── 06_league_table_and_season_prediction.ipynb  # Clean version
├── 07_paths_to_victory_demo.ipynb     # With uncertainty
├── 08_squad_analysis_demo.ipynb       # With injury analysis
├── 09_validation_and_diagnostics.ipynb  # NEW: Model diagnostics
├── 10_data_validation_and_cleaning.ipynb # NEW: Data quality
└── README.md  # Added: Notebook guide
```
