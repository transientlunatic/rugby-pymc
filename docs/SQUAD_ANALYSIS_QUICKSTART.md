# Squad Analysis - Quick Start Guide

## Overview

Phase 5a (Squad Parser & Data Entry) and Phase 5b (Squad Strength Analysis) are **now complete**!

You can now:
- ✅ Parse squad lists from Wikipedia/text
- ✅ Analyze squad strength and depth using your trained model
- ✅ Compare squads across tournaments
- ✅ Identify vulnerable positions and critical players

## Installation

No additional dependencies required beyond the existing rugby-ranking setup.

## Quick Start

### 1. Input a Squad (from Wikipedia)

Copy a squad list from Wikipedia (e.g., [Scotland 2025 Six Nations squad](https://en.wikipedia.org/wiki/2025_Six_Nations_Championship_squads)).

**Interactive mode:**
```bash
rugby-ranking squad input --team "Scotland" --season "2024-2025"

# Then paste the squad text and press Ctrl+D (Unix) or Ctrl+Z (Windows)
```

**From file:**
```bash
rugby-ranking squad input --team "Scotland" --season "2024-2025" --file squad.txt
```

**Supported formats:**
- Wikipedia format (recommended) - auto-detected
- Simple comma-separated: `Player Name, Club, Position`
- CSV format with headers

Squad will be saved to: `squads/scotland_2024-2025.csv`

### 2. Analyze Squad Strength

**Requires a trained model checkpoint.** See main README for training instructions.

```bash
rugby-ranking squad analyze --team "Scotland" --season "2024-2025" --checkpoint latest
```

**Output:**
```
======================================================================
SQUAD ANALYSIS: SCOTLAND (2024-2025)
======================================================================

Overall Strength: 84/100
Squad Depth Score: 76/100

POSITION STRENGTHS
----------------------------------------------------------------------
Position             1st Choice   Depth      Strength
----------------------------------------------------------------------
Back Row               0.91      95%        94%
Centres                0.87      94%        92%
Hookers                0.88      97%        91%
...

VULNERABLE POSITIONS
----------------------------------------------------------------------
Fly-half: Large drop-off from Finn Russell to Adam Hastings (-21%)
Scrum-half: Moderate depth concerns

STRONGEST POSITIONS
----------------------------------------------------------------------
Back Row: Excellent depth across all positions
Centres: Multiple high-quality options

MOST LIKELY STARTING XV
----------------------------------------------------------------------
Prop                 Pierre Schoeman
Hooker               George Turner
...

LIKELY BENCH
----------------------------------------------------------------------
1. Johnny Matthews
2. Rory Sutherland
...
======================================================================
```

Report saved to: `reports/scotland_2024-2025_analysis.txt`

### 3. Compare Tournament Squads

After inputting all Six Nations squads:

```bash
rugby-ranking squad compare --tournament six-nations --season "2024-2025"
```

**Output:**
```
======================================================================
SIX NATIONS SQUAD COMPARISON (2024-2025)
======================================================================

OVERALL RANKINGS
----------------------------------------------------------------------
Rank   Team                  Strength     Depth
----------------------------------------------------------------------
1      Ireland               91/100       85/100
2      France                89/100       82/100
3      Scotland              84/100       76/100
4      England               82/100       79/100
5      Wales                 74/100       68/100
6      Italy                 68/100       62/100

======================================================================
```

Full report saved to: `reports/six-nations_2024-2025_comparison.txt`

## Python API

### Parse Squad

```python
from rugby_ranking.model.squad_analysis import SquadParser

parser = SquadParser()

# From Wikipedia text
squad = parser.parse_text(
    wikipedia_text,
    team="Scotland",
    season="2024-2025",
    format='auto'  # or 'wikipedia', 'simple', 'csv'
)

# Save
squad.to_csv('squads/scotland_2024-2025.csv', index=False)
```

### Analyze Squad

```python
from rugby_ranking.model.squad_analysis import SquadAnalyzer, format_squad_analysis
from rugby_ranking.model.core import RugbyModel
from rugby_ranking.model.inference import ModelFitter
import pandas as pd

# Load model
model = RugbyModel()
fitter = ModelFitter.load('latest', model)

# Load squad
squad = pd.read_csv('squads/scotland_2024-2025.csv')

# Analyze
analyzer = SquadAnalyzer(model, fitter.trace)
analysis = analyzer.analyze_squad(squad, "Scotland", "2024-2025")

# Display
report = format_squad_analysis(analysis, detailed=True)
print(report)

# Access results
print(f"Overall strength: {analysis.overall_strength:.2f}")
print(f"Depth score: {analysis.depth_score:.2f}")
print(f"Likely XV: {analysis.likely_xv}")
```

## Features Implemented

### Phase 5a: Squad Parser & Data Entry ✅

- **SquadParser.parse_text()**: Parse Wikipedia, simple, and CSV formats
- **Position inference**: Maps position text to standard positions
- **CLI commands**: Interactive and file-based squad input
- **Storage system**: Squads saved to `squads/{team}_{season}.csv`

### Phase 5b: Squad Strength Analysis ✅

- **SquadAnalyzer.get_player_ratings()**: Extract player ratings from model
  - Handles players not in model (new caps)
  - Fuzzy name matching for player lookup
  - Rating uncertainty (credible intervals)

- **SquadAnalyzer.create_depth_chart()**: Build depth charts by position
  - Ranks players by model ratings
  - Accounts for positional versatility
  - Identifies 1st, 2nd, 3rd choice

- **SquadAnalyzer.calculate_position_strength()**: Position-level metrics
  - Expected strength per position
  - Depth quality (drop-off measurement)
  - Vulnerable position identification

- **SquadAnalyzer.calculate_squad_depth_score()**: Overall depth metric
  - Average depth across positions
  - Strongest/weakest position groups

- **Reporting**: Formatted analysis reports
  - Overall strength and depth scores
  - Position-by-position breakdown
  - Likely starting XV and bench
  - Vulnerable positions
  - Save to text files

## What's Next?

### Phase 5c: Lineup Prediction (For match predictions)

- Predict likely starting XV for specific matchups
- Monte Carlo sampling of possible lineups
- Selection probabilities per player

### Phase 5d: Squad-Based Predictions

- Predict matches with lineup uncertainty
- "Scotland vs England with likely lineups: 62% (±5% lineup uncertainty)"
- Scenario comparison

### Phase 5e: Injury Impact Analysis

- Calculate Δ(P(win)) if specific player unavailable
- Identify most critical players
- Squad robustness scores

### Phase 5f: Squad Comparison & Tournament Analysis

- Head-to-head positional comparisons
- Matchup advantages
- Tactical implications

## Examples

### Example 1: Input Scotland Squad

```bash
# Copy squad from Wikipedia
rugby-ranking squad input --team "Scotland" --season "2024-2025"

# Paste squad text, press Ctrl+D
```

### Example 2: Analyze All Six Nations Squads

```bash
# Input all squads
for team in "Scotland" "England" "Ireland" "France" "Wales" "Italy"; do
  rugby-ranking squad input --team "$team" --season "2024-2025"
  # (paste squad for each team)
done

# Compare
rugby-ranking squad compare --tournament six-nations --season "2024-2025"
```

### Example 3: Python Workflow

```python
import pandas as pd
from rugby_ranking.model.squad_analysis import (
    SquadParser,
    SquadAnalyzer,
    format_squad_analysis
)
from rugby_ranking.model.core import RugbyModel
from rugby_ranking.model.inference import ModelFitter

# 1. Parse squad
parser = SquadParser()
squad = parser.parse_text(wikipedia_text, "Scotland", "2024-2025")
squad.to_csv('squads/scotland_2024-2025.csv')

# 2. Load model
model = RugbyModel()
fitter = ModelFitter.load('latest', model)

# 3. Analyze
analyzer = SquadAnalyzer(model, fitter.trace)
analysis = analyzer.analyze_squad(squad, "Scotland", "2024-2025")

# 4. Generate report
print(format_squad_analysis(analysis))

# 5. Export for blog
with open('blog/scotland_squad_analysis.md', 'w') as f:
    f.write(format_squad_analysis(analysis, detailed=True))
```

## Testing

Test the parser without a trained model:

```bash
python test_parser_only.py
```

This will:
1. Parse a sample Scotland squad
2. Save to `squads/test_scotland.csv`
3. Display parsed players

## Troubleshooting

### "Squad file not found"

Make sure you've run `rugby-ranking squad input` first to create the squad CSV.

### "Checkpoint not found"

Train a model first: `rugby-ranking update --data-dir /path/to/Rugby-Data`

### "Player not in model"

New caps or returning players will show with default ratings (0.0 mean, high uncertainty). This is expected.

### Parser issues

- Ensure Wikipedia format has clear section headers (Forwards/Backs)
- Position headers should be on their own lines
- Player lines should have format: `Number. Name (Club)`

## Timeline

- **Now**: ✅ Phase 5a and 5b complete
- **Before squads announced (~Jan 20)**: Input all Six Nations squads
- **Week before tournament**: Generate pre-tournament analysis reports
- **Weekly during tournament**: Update injury impact as needed

## Files Created

- `rugby_ranking/model/squad_analysis.py` - Full implementation
- `rugby_ranking/cli.py` - CLI commands added
- `notebooks/08_squad_analysis_demo.ipynb` - Demo notebook
- `docs/SQUAD_ANALYSIS_DESIGN.md` - Full technical design
- `docs/SQUAD_ANALYSIS_QUICKSTART.md` - This guide
- `test_parser_only.py` - Standalone parser test

## Next Steps

1. **Train or load your model**: `rugby-ranking update --data-dir /path/to/Rugby-Data`
2. **Input Six Nations squads** as they're announced (around Jan 20)
3. **Analyze squads** before the tournament starts
4. **Generate pre-tournament blog content** with squad comparisons
5. **Weekly updates** on injury impacts during the tournament
