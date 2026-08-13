# League Table and Season Prediction

Implemented comprehensive league table computation and season prediction functionality to support:
- Computing standings from match results
- Bonus points calculation for different competitions
- Predicting final season standings with Monte Carlo simulation
- Playoff qualification probabilities

## Overview

The implementation consists of three main components:

1. **LeagueTable**: Computes standings from match results with configurable bonus point rules
2. **SeasonPredictor**: Predicts season outcomes using match predictions and Monte Carlo simulation
3. **BonusPointConfig**: Flexible configuration for different competition rules

## LeagueTable

### Features

- Compute current standings from match results
- Support for multiple bonus point systems (URC/Premiership/Top14)
- Automatic sorting by points, then goal difference, then tries scored
- Try bonus calculation (absolute or relative thresholds)
- Losing bonus calculation
- Formatted table output

### Usage

```python
from rugby_ranking.model.league_table import LeagueTable, BonusPointRules

# Create table with URC rules
table = LeagueTable(bonus_rules=BonusPointRules.URC)

# Compute standings from match results
standings = table.compute_standings(
    matches=matches_df,
    opponent_tries_col='opponent_tries'  # Optional for relative try bonus
)

# Display formatted table
from rugby_ranking.model.league_table import format_table
print(format_table(standings, max_teams=10))
```

### Bonus Point Systems

#### URC/Premiership Rules
- **Win**: 4 points
- **Draw**: 2 points
- **Loss**: 0 points
- **Try bonus**: 1 point for scoring 4+ tries (absolute threshold)
- **Losing bonus**: 1 point for losing by ≤7 points

```python
table = LeagueTable(bonus_rules=BonusPointRules.URC)
# or
table = LeagueTable(bonus_rules=BonusPointRules.PREMIERSHIP)
```

#### Top14 Rules
- **Win**: 4 points
- **Draw**: 2 points
- **Loss**: 0 points
- **Try bonus**: 1 point for scoring 3+ more tries than opponent (relative threshold)
- **Losing bonus**: 1 point for losing by ≤5 points

```python
table = LeagueTable(bonus_rules=BonusPointRules.TOP14)
```

#### Custom Rules

```python
from rugby_ranking.model.league_table import BonusPointConfig

custom_config = BonusPointConfig(
    try_bonus_threshold=5,       # Need 5 tries for bonus
    try_bonus_relative=False,     # Absolute threshold
    losing_bonus_margin=10,       # Lose by ≤10 for bonus
    win_points=4,
    draw_points=2,
    loss_points=0,
)

table = LeagueTable(config=custom_config)
```

### Data Format

The `compute_standings()` method expects a DataFrame with one row per team per match:

```python
matches = pd.DataFrame({
    'team': ['Leinster', 'Munster', ...],
    'opponent': ['Munster', 'Leinster', ...],
    'score': [28, 14, ...],
    'opponent_score': [14, 28, ...],
    'tries': [4, 2, ...],
    'opponent_tries': [2, 4, ...],  # Optional but needed for Top14 rules
    'is_home': [True, False, ...],
})

standings = table.compute_standings(
    matches,
    opponent_tries_col='opponent_tries'
)
```

### Output Format

The standings DataFrame contains:

| Column | Description |
|--------|-------------|
| `position` | League position (1 = first) |
| `team` | Team name |
| `played` | Matches played |
| `won` | Matches won |
| `drawn` | Matches drawn |
| `lost` | Matches lost |
| `points_for` | Total points scored |
| `points_against` | Total points conceded |
| `points_diff` | Points difference |
| `tries_for` | Total tries scored |
| `tries_against` | Total tries conceded |
| `try_bonus` | Try bonus points earned |
| `losing_bonus` | Losing bonus points earned |
| `bonus_points` | Total bonus points |
| `match_points` | Points from wins/draws/losses |
| `total_points` | Total league points |

## SeasonPredictor

### Features

- Predict all remaining matches using the fitted model
- Monte Carlo simulation of final standings
- Position probabilities: P(team finishes in position k)
- Playoff qualification probabilities
- Expected final points and goal difference

### Usage

```python
from rugby_ranking.model.season_predictor import SeasonPredictor
from rugby_ranking.model.predictions import MatchPredictor
from rugby_ranking.model.league_table import BonusPointRules
from rugby_ranking.cli import load_checkpoint

# Load model
model, trace = load_checkpoint("latest")

# Create predictors
match_predictor = MatchPredictor(model, trace)
season_predictor = SeasonPredictor(
    match_predictor=match_predictor,
    competition=BonusPointRules.URC,
    playoff_spots=8,  # URC has 8 playoff spots
)

# Predict season
season_pred = season_predictor.predict_season(
    played_matches=played_df,         # Completed matches
    remaining_fixtures=fixtures_df,    # Upcoming matches
    season="2024-2025",
    n_simulations=1000,                # Monte Carlo iterations
)

# Display results
print(season_predictor.format_predictions(season_pred))

# Access specific components
print(season_pred.current_standings)
print(season_pred.predicted_standings)
print(season_pred.playoff_probabilities)
print(season_pred.position_probabilities)
```

### Data Formats

#### Played Matches
Same format as LeagueTable input (one row per team per match):

```python
played_matches = pd.DataFrame({
    'team': ['Leinster', 'Munster', ...],
    'opponent': ['Munster', 'Leinster', ...],
    'score': [28, 14, ...],
    'opponent_score': [14, 28, ...],
    'tries': [4, 2, ...],
    'opponent_tries': [2, 4, ...],
    'is_home': [True, False, ...],
})
```

#### Remaining Fixtures
One row per match (not per team):

```python
remaining_fixtures = pd.DataFrame({
    'home_team': ['Leinster', 'Ulster', ...],
    'away_team': ['Munster', 'Connacht', ...],
    'date': ['2025-02-01', '2025-02-02', ...],  # Optional
})
```

### Output Components

#### Current Standings
Current league table from played matches (same format as LeagueTable output).

#### Predicted Standings
Expected final standings from Monte Carlo simulation:

| Column | Description |
|--------|-------------|
| `predicted_position` | Expected final position |
| `team` | Team name |
| `expected_points` | Mean total points across simulations |
| `expected_wins` | Mean wins across simulations |
| `expected_diff` | Mean points difference across simulations |

#### Position Probabilities
Probability distribution over final positions for each team:

| Column | Description |
|--------|-------------|
| `P(pos 1)` | Probability of finishing 1st |
| `P(pos 2)` | Probability of finishing 2nd |
| ... | ... |
| `most_likely_position` | Position with highest probability |

#### Playoff Probabilities
Probability of qualifying for playoffs:

| Column | Description |
|--------|-------------|
| `team` | Team name |
| `playoff_probability` | P(finish in top N positions) |

### Example Output

```
======================================================================
SEASON PREDICTION
======================================================================

CURRENT STANDINGS:
----------------------------------------------------------------------
 1. Leinster             P: 2 W: 2 Pts: 10
 2. Ulster               P: 2 W: 1 Pts:  6
 3. Munster              P: 2 W: 0 Pts:  3
 4. Connacht             P: 2 W: 0 Pts:  1


PREDICTED FINAL STANDINGS:
----------------------------------------------------------------------
 1. Leinster             Pts:15.6 Diff:+41.4
 2. Ulster               Pts:12.0 Diff:+1.6
 3. Munster              Pts:8.8 Diff:-14.6
 4. Connacht             Pts:6.9 Diff:-28.4


PLAYOFF PROBABILITIES (Top 2):
----------------------------------------------------------------------
Leinster             96.0%
Ulster               71.0%
Munster              25.0%
Connacht             8.0%

======================================================================
```

## Monte Carlo Simulation Details

The season predictor uses Monte Carlo simulation to account for uncertainty:

1. **For each simulation**:
   - Start with current standings
   - For each remaining fixture:
     - Sample scores from the predicted distributions
     - Estimate tries from scores
     - Update standings with match result
   - Compute final standings
   - Record final position for each team

2. **After N simulations**:
   - Compute expected final points (mean across simulations)
   - Compute position probabilities (frequency of each position)
   - Compute playoff probabilities (frequency of top N finishes)

This approach captures:
- Uncertainty in individual match outcomes
- Correlations between team performances
- Impact of bonus points on final standings
- Tail risks (unlikely but possible scenarios)

## Integration with Model

The season predictor integrates seamlessly with the rugby ranking model:

```python
# 1. Train model
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter

dataset = MatchDataset(data_dir)
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

model = RugbyModel(ModelConfig())
model.build(df, score_type="tries")

fitter = ModelFitter(model)
trace = fitter.fit_vi()
fitter.save("my_model")

# 2. Load model and predict season
from rugby_ranking.cli import load_checkpoint
model, trace = load_checkpoint("my_model")

match_predictor = MatchPredictor(model, trace)
season_predictor = SeasonPredictor(match_predictor)

# 3. Prepare data
played_matches = ...  # From your data source
remaining_fixtures = ...  # From fixture list

# 4. Run prediction
season_pred = season_predictor.predict_season(
    played_matches=played_matches,
    remaining_fixtures=remaining_fixtures,
    season="2024-2025",
    n_simulations=1000,
)
```

## Command-Line Usage

You can use the league table and season prediction functionality from the command line:

```bash
# Train model
python train_model.py --model static --data-dir ../Rugby-Data --save-as latest

# Python script to predict season
python -c "
from rugby_ranking.cli import load_checkpoint
from rugby_ranking.model.predictions import MatchPredictor
from rugby_ranking.model.season_predictor import SeasonPredictor
import pandas as pd

# Load model
model, trace = load_checkpoint('latest')

# Load data
played_matches = pd.read_csv('played_matches.csv')
remaining_fixtures = pd.read_csv('remaining_fixtures.csv')

# Predict
match_predictor = MatchPredictor(model, trace)
season_predictor = SeasonPredictor(match_predictor, competition='urc')
season_pred = season_predictor.predict_season(
    played_matches, remaining_fixtures, '2024-2025', n_simulations=1000
)

# Display
print(season_predictor.format_predictions(season_pred))
"
```

## Performance Considerations

### Computational Cost

- **League table computation**: O(n_matches)
  - Very fast, even for large datasets
  - No model inference required

- **Season prediction**: O(n_simulations × n_remaining_matches × n_samples)
  - Each simulation requires:
    - Predicting all remaining matches (~1000 samples each)
    - Updating standings (~100-500 ms per simulation)
  - Typical runtime for 1000 simulations: 1-5 minutes

### Recommendations

- Use 100-500 simulations for quick estimates
- Use 1000-5000 simulations for publishable results
- Use 10000+ simulations for high-stakes predictions
- Consider parallelization for large-scale predictions

## Testing

Test suites verify correctness:

```bash
# Test league table
python test_league_table.py

# Test season predictor
python test_season_predictor.py
```

**Test Coverage**:
- ✅ Basic standings computation
- ✅ Bonus point calculation (URC/Premiership/Top14)
- ✅ Position ordering (points → diff → tries)
- ✅ Season prediction with Monte Carlo
- ✅ Probability distributions
- ✅ Formatted output

## Files

### Implementation
- `rugby_ranking/model/league_table.py`: League table computation (~380 lines)
- `rugby_ranking/model/season_predictor.py`: Season prediction (~350 lines)

### Tests
- `test_league_table.py`: League table tests (~200 lines)
- `test_season_predictor.py`: Season prediction tests (~250 lines)

### Documentation
- `LEAGUE_TABLE_AND_SEASON_PREDICTION.md`: This file

## Future Enhancements

### Planned
- [ ] Playoff bracket prediction
- [ ] Home/away fixtures balance analysis
- [ ] Strength of schedule adjustments
- [ ] Interactive dashboard integration

### Potential
- [ ] Multiple competition support in single season
- [ ] Relegation probabilities (for leagues with relegation)
- [ ] Historical accuracy analysis (backtest predictions)
- [ ] Confidence intervals on position probabilities
- [ ] Expected value of remaining fixtures (for playoff chase)

## Related Documentation

- [Model Core Documentation](rugby_ranking/model/core.py)
- [Predictions Documentation](rugby_ranking/model/predictions.py)
- [CLI Documentation](rugby_ranking/cli.py)
- [Training Script](train_model.py)
- [Time-Varying Effects](TIME_VARYING_IMPLEMENTATION.md)
- [Ranking Improvements](RANKING_IMPROVEMENTS.md)
