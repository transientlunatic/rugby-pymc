# Data Utilities Guide

Streamlined utilities for working with rugby match data, league tables, and season predictions.

## Overview

The `rugby_ranking.model.data_utils` module provides convenience functions that eliminate boilerplate code when working with match data. These utilities are designed for:
- Interactive notebooks
- Quick analysis scripts
- Production data pipelines

## Quick Start

```python
from rugby_ranking.model.data_utils import quick_load, quick_standings

# Load all data
dataset = quick_load("../../Rugby-Data")

# Get current standings
standings = quick_standings(dataset, season="2024-2025", competition="celtic")
print(standings[['team', 'position', 'total_points']])
```

## API Reference

### Loading Data

#### `quick_load(data_dir="../../Rugby-Data")`

Load all match data in one line.

**Returns:** `MatchDataset` with all matches loaded

**Example:**
```python
dataset = quick_load()
print(f"Loaded {len(dataset.matches)} matches")
```

---

### Quick Analysis

#### `quick_standings(dataset, season, competition, bonus_rules="URC", top_n=20)`

Get current league standings in one line.

**Parameters:**
- `dataset`: MatchDataset with loaded matches
- `season`: Season string (e.g., "2024-2025")
- `competition`: Competition name (e.g., "celtic", "premiership")
- `bonus_rules`: Bonus point system ("URC", "PREMIERSHIP", "TOP14")
- `top_n`: Number of teams to return (default: 20)

**Returns:** DataFrame with standings

**Example:**
```python
standings = quick_standings(dataset, "2024-2025", "celtic")
print(standings[['team', 'position', 'total_points', 'wins', 'losses']])
```

#### `get_competition_summary(dataset)`

Get summary of available competitions and seasons.

**Returns:** DataFrame with columns: competition, season, total_matches, played_matches, remaining_matches

**Example:**
```python
summary = get_competition_summary(dataset)
print(summary)
#        competition      season  total_matches  played_matches  remaining_matches
# 0           celtic  2024-2025            153              89                 64
# 1      premiership  2024-2025            132              76                 56
```

---

### Season Prediction Data Prep

#### `prepare_season_data(dataset, season, competition, cutoff_date=None, include_tries=True)`

Prepare both played matches and remaining fixtures for season prediction in one call.

**Parameters:**
- `dataset`: MatchDataset containing all matches
- `season`: Season string (e.g., "2024-2025")
- `competition`: Competition name
- `cutoff_date`: Date to split played vs remaining (default: now)
- `include_tries`: Whether to include try counts for bonus points

**Returns:** Tuple of `(played_matches_df, remaining_fixtures_df)`
- `played_matches_df`: League table format (2 rows per match)
- `remaining_fixtures_df`: Fixtures format (1 row per match)

**Example:**
```python
played, fixtures = prepare_season_data(
    dataset,
    season="2024-2025",
    competition="celtic"
)

# Ready for season prediction!
prediction = season_predictor.predict_season(
    played_matches=played,
    remaining_fixtures=fixtures,
    season="2024-2025"
)
```

---

### Data Conversion

#### `matches_to_league_table_format(matches, include_tries=True)`

Convert list of matches to league table format (one row per team per match).

**Parameters:**
- `matches`: List of MatchData objects
- `include_tries`: Include try counts for bonus point calculation

**Returns:** DataFrame with columns: team, opponent, score, opponent_score, tries, opponent_tries, is_home, date

**Example:**
```python
from rugby_ranking.model.data_utils import filter_matches

# Get matches for a season
matches = filter_matches(dataset, season="2024-2025", competition="celtic", played_only=True)

# Convert to league table format
df = matches_to_league_table_format(matches)

# Now compute standings
from rugby_ranking.model.league_table import LeagueTable, BonusPointRules
table = LeagueTable(bonus_rules=BonusPointRules.URC)
standings = table.compute_standings(df, opponent_tries_col='opponent_tries')
```

#### `matches_to_fixtures_format(matches, future_only=True)`

Convert matches to fixtures format (one row per match, not per team).

**Parameters:**
- `matches`: List of MatchData objects
- `future_only`: If True, only include unplayed matches

**Returns:** DataFrame with columns: home_team, away_team, date

**Example:**
```python
# Get remaining fixtures
matches = filter_matches(dataset, season="2024-2025", competition="celtic")
fixtures = matches_to_fixtures_format(matches, future_only=True)
print(f"Found {len(fixtures)} remaining matches")
```

---

### Filtering

#### `filter_matches(dataset, season=None, competition=None, played_only=False, date_from=None, date_to=None)`

Filter matches by various criteria.

**Parameters:**
- `dataset`: MatchDataset to filter
- `season`: Season string, or None for all seasons
- `competition`: Competition name (case-insensitive substring match), or None for all
- `played_only`: If True, only return matches with scores
- `date_from`: Only include matches on or after this date
- `date_to`: Only include matches on or before this date

**Returns:** Filtered list of MatchData objects

**Examples:**
```python
# Get all URC matches from 2024-2025 that have been played
matches = filter_matches(
    dataset,
    season="2024-2025",
    competition="celtic",
    played_only=True
)

# Get matches from last 30 days
from datetime import datetime, timedelta
recent = filter_matches(
    dataset,
    date_from=datetime.now() - timedelta(days=30)
)

# Get all Premiership matches across all seasons
prem_matches = filter_matches(dataset, competition="premiership")
```

---

### Utilities

#### `count_tries(scores)`

Count tries from scoring events list.

**Parameters:**
- `scores`: List of scoring event dicts with 'type' key

**Returns:** Integer count of tries

**Example:**
```python
match = dataset.matches[0]
home_tries = count_tries(match.home_scores)
away_tries = count_tries(match.away_scores)
```

---

## Code Comparison

### Before (Manual Data Prep)

```python
# Load data
DATA_DIR = Path("../../Rugby-Data")
dataset = MatchDataset(DATA_DIR)
dataset.load_json_files()

# Filter matches
SEASON = "2024-2025"
COMPETITION = "celtic"
season_matches = [
    m for m in dataset.matches
    if m.season == SEASON and COMPETITION.lower() in m.competition.lower() and m.is_played
]

# Count tries manually
def count_tries(scores):
    if not scores:
        return 0
    return sum(1 for s in scores if s.get('type', '').lower() in ['try', 't'])

# Build DataFrame
match_rows = []
for match in season_matches:
    home_tries = count_tries(match.home_scores)
    away_tries = count_tries(match.away_scores)

    match_rows.append({
        'team': match.home_team,
        'opponent': match.away_team,
        'score': match.home_score,
        'opponent_score': match.away_score,
        'tries': home_tries,
        'opponent_tries': away_tries,
        'is_home': True,
        'date': match.date,
    })

    match_rows.append({
        'team': match.away_team,
        'opponent': match.home_team,
        'score': match.away_score,
        'opponent_score': match.home_score,
        'tries': away_tries,
        'opponent_tries': home_tries,
        'is_home': False,
        'date': match.date,
    })

matches_df = pd.DataFrame(match_rows)

# Compute standings
table = LeagueTable(bonus_rules=BonusPointRules.URC)
standings = table.compute_standings(matches_df, opponent_tries_col='opponent_tries')
```

**Lines of code: ~40**

### After (With Data Utils)

```python
from rugby_ranking.model.data_utils import quick_load, quick_standings

dataset = quick_load("../../Rugby-Data")
standings = quick_standings(dataset, "2024-2025", "celtic")
```

**Lines of code: 2 (95% reduction!)**

---

## Common Workflows

### Workflow 1: Quick Analysis

```python
from rugby_ranking.model.data_utils import quick_load, quick_standings

# Load and analyze
dataset = quick_load()
standings = quick_standings(dataset, "2024-2025", "celtic")
print(standings.head(10))
```

### Workflow 2: Season Prediction

```python
from rugby_ranking.model.data_utils import quick_load, prepare_season_data
from rugby_ranking.model.season_predictor import SeasonPredictor
from rugby_ranking.model.predictions import MatchPredictor

# Load data
dataset = quick_load()

# Prepare data
played, fixtures = prepare_season_data(dataset, "2024-2025", "celtic")

# Load model and predict
model, trace = load_checkpoint("latest")
match_predictor = MatchPredictor(model, trace)
season_predictor = SeasonPredictor(match_predictor, competition="URC")

prediction = season_predictor.predict_season(
    played_matches=played,
    remaining_fixtures=fixtures,
    season="2024-2025",
    n_simulations=1000
)
```

### Workflow 3: Custom Analysis

```python
from rugby_ranking.model.data_utils import (
    quick_load,
    filter_matches,
    matches_to_league_table_format
)

# Load data
dataset = quick_load()

# Get specific subset
matches = filter_matches(
    dataset,
    season="2024-2025",
    competition="celtic",
    played_only=True
)

# Convert to analysis format
df = matches_to_league_table_format(matches)

# Now do custom analysis
home_advantage = df.groupby('is_home')['score'].mean()
print(f"Average home score: {home_advantage[True]:.1f}")
print(f"Average away score: {home_advantage[False]:.1f}")
```

### Workflow 4: Multi-Competition Comparison

```python
from rugby_ranking.model.data_utils import quick_load, quick_standings

dataset = quick_load()

competitions = ["celtic", "premiership", "top14"]
for comp in competitions:
    standings = quick_standings(dataset, "2024-2025", comp, top_n=1)
    leader = standings.iloc[0]
    print(f"{comp.upper()}: {leader['team']} ({leader['total_points']} pts)")
```

---

## Performance Tips

1. **Load once, use many times**: Call `quick_load()` once and reuse the dataset
2. **Filter before convert**: Use `filter_matches()` before conversion functions
3. **Cache standings**: Store standings results if computing multiple times
4. **Use played_only**: When you only need completed matches, set `played_only=True` to skip unplayed

---

## See Also

- [League Table API](../rugby_ranking/model/league_table.py)
- [Season Predictor API](../rugby_ranking/model/season_predictor.py)
- [Streamlined Notebook](../notebooks/06_league_table_and_season_prediction_streamlined.ipynb)
- [Original Notebook](../notebooks/06_league_table_and_season_prediction.ipynb) (for comparison)
