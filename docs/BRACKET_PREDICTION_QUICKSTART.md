# Bracket Prediction - Quick Start Guide

## Problem Solved

Previously, matches with "TBC" (To Be Confirmed) teams were simply filtered out:
```python
# OLD approach
matches = [m for m in tournament.matches
           if m.home.team.name != "TBC" and m.away.team.name != "TBC"]
```

**Now you can predict these matches!** The system predicts:
- Who will advance to knockout stages
- Likely bracket matchups
- Probability each team reaches each round
- Paths to winning the tournament

## 30-Second Example

```python
from rugby_ranking.model.bracket_predictor import BracketPredictor
from rugby_ranking.model.bracket import create_world_cup_bracket
import pandas as pd

# 1. Create bracket
bracket = create_world_cup_bracket()

# 2. Pool standings
standings = pd.DataFrame([
    {"pool": "A", "team": "France", "points": 19, "position": 1},
    {"pool": "A", "team": "New Zealand", "points": 17, "position": 2},
    # ...
])

# 3. Predict
predictor = BracketPredictor(match_predictor, bracket)
prediction = predictor.predict_bracket(
    pool_standings=standings,
    n_simulations=10000
)

# 4. Results
print(prediction.advancement_probs)
```

## Key Features

### 1. TBD Resolution
Automatically resolves placeholders like:
- "Pool A winner" → France
- "Best runner-up" → Calculates from standings
- "Winner QF1" → Simulated from earlier rounds

### 2. Bracket Simulation
- Runs 10,000+ Monte Carlo simulations
- Uses your trained `MatchPredictor` for realism
- Tracks advancement through bracket
- Aggregates into probabilities

### 3. Paths to Victory
```python
from rugby_ranking.model.tournament_paths import TournamentPathsAnalyzer

analyzer = TournamentPathsAnalyzer(prediction, match_predictor)
paths = analyzer.analyze_tournament_paths(team="France", target="champion")
print(paths.narrative)
```

Output:
```
France can win the tournament with 23% probability.

Path to victory:
  Must beat New Zealand in QF (72% likely)
  Must beat Ireland in SF (51% likely)
  Must beat South Africa in Final (45% likely)
```

### 4. Live Updates
```python
# After quarterfinals complete
completed = pd.DataFrame([
    {"match_id": "QF1", "winner": "France"},
    {"match_id": "QF2", "winner": "Ireland"},
])

updated = predictor.predict_bracket(
    completed_knockout_matches=completed,
    n_simulations=10000
)
```

## Pre-Built Brackets

```python
from rugby_ranking.model.bracket import (
    create_world_cup_bracket,      # QF → SF → Final + 3rd place
    create_champions_cup_bracket,  # R16 → QF → SF → Final
    create_urc_playoffs_bracket,   # QF → SF → Final
)
```

## Custom Brackets

```python
from rugby_ranking.model.bracket import BracketStructure

bracket = BracketStructure("My Tournament", {
    "rounds": ["quarterfinal", "semifinal", "final"],
    "matches": [
        {
            "id": "QF1",
            "round_type": "quarterfinal",
            "round_number": 1,
            "home": {"team": "TBD", "source": "1st place"},
            "away": {"team": "TBD", "source": "8th place"},
            "winner_advances_to": "SF1",
        },
        # ...
    ]
})
```

## Integration with Existing Code

### Option 1: Predict TBC Matches Directly
```python
# Instead of filtering TBC matches, predict them
knockout_prediction = bracket_predictor.predict_bracket(
    pool_standings=league_table,
    n_simulations=5000
)

# Get predictions for specific match
final_matchups = knockout_prediction.get_likely_matchup("Final")
```

### Option 2: Add to predictions.py Workflow
```python
# In scripts/predictions.py

# Existing league predictions
season_pred = season_predictor.predict_season(...)

# NEW: Add knockout predictions
if has_knockout_stage:
    bracket = create_urc_playoffs_bracket()
    bracket_pred = bracket_predictor.predict_bracket(
        pool_standings=season_pred.predicted_standings,
        n_simulations=5000
    )

    # Display championship odds
    print(bracket_pred.advancement_probs[["team", "champion_prob"]])
```

## What's in the Box

**Core Modules:**
- `bracket.py`: Bracket structures, TBD handling
- `bracket_predictor.py`: Monte Carlo simulation
- `tbd_resolver.py`: Smart TBD → team resolution
- `tournament_paths.py`: Paths to victory analysis

**Examples:**
- `bracket_prediction_example.py`: 5 complete examples

**Tests:**
- `test_bracket.py`: Unit tests for all functionality

**Documentation:**
- `BRACKET_PREDICTION_DESIGN.md`: Full technical design
- `BRACKET_PREDICTION_SUMMARY.md`: Complete implementation guide
- `BRACKET_PREDICTION_QUICKSTART.md`: This file

## Common Use Cases

### Use Case 1: Tournament Preview
"Who's most likely to win the World Cup?"
```python
prediction = predictor.predict_bracket(pool_standings, n_simulations=10000)
print(prediction.advancement_probs.sort_values("champion_prob", ascending=False))
```

### Use Case 2: Team Analysis
"What's France's path to victory?"
```python
analyzer = TournamentPathsAnalyzer(prediction, match_predictor)
paths = analyzer.analyze_tournament_paths(team="France", target="champion")
print(paths.narrative)
```

### Use Case 3: Bracket Scenarios
"What if Ireland beats France in the QF?"
```python
# Run prediction with Ireland winning QF1
completed = pd.DataFrame([{"match_id": "QF1", "winner": "Ireland"}])
alt_prediction = predictor.predict_bracket(
    completed_knockout_matches=completed,
    n_simulations=5000
)
```

### Use Case 4: Weekly Updates
"Update predictions as the tournament progresses"
```python
# Week 1: Predict from pools
week1 = predictor.predict_bracket(pool_standings, n_simulations=10000)

# Week 2: QFs complete
completed_qf = get_completed_matches()
week2 = predictor.predict_bracket(
    completed_knockout_matches=completed_qf,
    n_simulations=10000
)
```

## Performance

- **10,000 simulations**: ~5-30 seconds
- **Caching**: Match predictions cached for speed
- **Parallelization**: Can run simulations in parallel (future enhancement)

## Next Steps

1. **Try the examples**: `python examples/bracket_prediction_example.py`
2. **Read the design doc**: `docs/BRACKET_PREDICTION_DESIGN.md`
3. **Integrate with your workflow**: Update `scripts/predictions.py`
4. **Create visualizations**: Add interactive bracket diagrams

## Help

- Full documentation: `docs/BRACKET_PREDICTION_SUMMARY.md`
- Technical design: `docs/BRACKET_PREDICTION_DESIGN.md`
- Examples: `examples/bracket_prediction_example.py`
- Tests: `tests/test_bracket.py`
