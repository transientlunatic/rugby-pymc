# Squad Analysis

## Overview

The `SquadAnalyzer` assesses a team's pre-match strength from a named squad, using player ratings from the fitted model.

## Basic Usage

```python
import pandas as pd
from rugby_ranking.model.squad_analysis import SquadAnalyzer

analyzer = SquadAnalyzer(model, trace)

# Load squad (columns: player, position_text)
squad = pd.read_csv("squads/scotland_2025-2026.csv")

# Analyze
analysis = analyzer.analyze_squad(
    squad,
    team="Scotland",
    season="2025-2026",
)
```

## Squad CSV Format

Squad files should have at minimum:
- `player` — player name (matched against model's player index)
- `position_text` — position name (e.g. "Loosehead Prop", "Fly-half")

Pre-scraped squad files for Six Nations teams are in the `squads/` directory of the Rugby-Data repository.

## Squad Strength Comparison

```python
# Compare multiple squads — SquadComparator wraps a SquadAnalyzer
from rugby_ranking.model.squad_analysis import SquadComparator

scotland_squad = pd.read_csv("squads/scotland_2025-2026.csv")
ireland_squad = pd.read_csv("squads/ireland_2025-2026.csv")

analyzer = SquadAnalyzer(model, trace)
comparator = SquadComparator(analyzer)
comparison = comparator.compare_squads(
    {"Scotland": scotland_squad, "Ireland": ireland_squad},
    season="2025-2026",
)
print(comparison)
```

## See Also

- [Predictions Guide](predictions)
- [SQUAD_ANALYSIS_QUICKSTART.md](../SQUAD_ANALYSIS_QUICKSTART.md)
- [SQUAD_ANALYSIS_DESIGN.md](../SQUAD_ANALYSIS_DESIGN.md)
