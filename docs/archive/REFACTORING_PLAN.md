# Refactoring and Enhancement Plan

## Goals

1. **Separate concerns**: Training scripts vs visualization notebooks
2. **Model compatibility**: Ensure all variants work together
3. **Season predictions**: Predict all unplayed matches, compute tables, predict playoffs
4. **Dashboard integration**: Show current and predicted league standings

## Current State

### Model Variants
- ✅ Static joint model (`build_joint`)
- ✅ Static joint with minibatch (`build_joint_minibatch`)
- ✅ Time-varying model (`build_joint_time_varying`)
- ✅ Separate kicking/try-scoring effects
- ✅ Defensive effects

### Issues
- Training happens in notebooks
- No unified training script
- No playoff/finals prediction
- No league table computation
- Dashboard doesn't show predictions

## Phase 1: Model Compatibility Audit

### Tasks
- [ ] Test that all model variants can:
  - Build successfully
  - Fit with VI
  - Save/load checkpoints
  - Extract rankings
  - Make predictions
- [ ] Document which features work with which variants
- [ ] Fix any incompatibilities

### Test Matrix
```
                    | Static | Time-Varying | Minibatch |
--------------------|--------|--------------|-----------|
Build               |   ✓    |      ?       |     ✓     |
Fit VI              |   ✓    |      ?       |     ✓     |
Fit MCMC            |   ✓    |      ?       |     ✓     |
Save/Load           |   ✓    |      ?       |     ✓     |
get_player_rankings |   ✓    |      ?       |     ✓     |
get_team_rankings   |   ✓    |      ?       |     ✓     |
Predictions         |   ✓    |      ?       |     ?     |
```

## Phase 2: Unified Training Script

### Goals
- Single script to train any model variant
- CLI interface with clear options
- Automatic checkpoint management
- Progress tracking and diagnostics

### Design

```bash
# Train static model
python train_model.py --model static --data-dir ../Rugby-Data

# Train time-varying model
python train_model.py --model time-varying --data-dir ../Rugby-Data

# Resume from checkpoint
python train_model.py --model static --resume joint_model_v2

# Custom config
python train_model.py --model static --config config.yaml
```

### Features
- Model selection (static, time-varying, minibatch)
- Data filtering (seasons, competitions)
- Inference method (VI, MCMC)
- Checkpoint auto-save
- Diagnostics output
- Validation metrics

### File: `train_model.py`

```python
import argparse
from pathlib import Path
from rugby_ranking.model.data import MatchDataset
from rugby_ranking.model.core import RugbyModel, ModelConfig
from rugby_ranking.model.inference import ModelFitter, InferenceConfig

def main():
    parser = argparse.ArgumentParser(description='Train rugby ranking model')
    parser.add_argument('--model', choices=['static', 'time-varying', 'minibatch'],
                       default='static')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--resume', type=str, help='Checkpoint name to resume from')
    parser.add_argument('--method', choices=['vi', 'mcmc'], default='vi')
    parser.add_argument('--save-as', type=str, help='Checkpoint name to save')
    parser.add_argument('--seasons', type=int, help='Only use last N seasons')
    args = parser.parse_args()

    # Load data
    # Build model
    # Fit
    # Save
    # Report diagnostics
```

## Phase 3: Visualization-Only Notebooks

### Convert Notebooks
- **02_model_fitting.ipynb** → Visualization only, load checkpoint
- **03_predictions.ipynb** → Visualization only
- **04_defensive_effects_demo.ipynb** → Visualization only
- **05_time_varying_effects.ipynb** → Visualization only

### Pattern
```python
# OLD (training in notebook)
model = RugbyModel(config)
model.build_joint(df)
fitter = ModelFitter(model, inference_config)
trace = fitter.fit_vi()

# NEW (load checkpoint)
from rugby_ranking.cli import load_checkpoint
model, trace = load_checkpoint("joint_model_v2")
```

## Phase 4: League Table Computation

### Design

```python
class LeagueTable:
    """Compute league standings from match results."""

    def __init__(self, matches: pd.DataFrame):
        self.matches = matches

    def compute_standings(self, competition: str, season: str) -> pd.DataFrame:
        """
        Compute current standings.

        Returns:
            DataFrame with columns:
            - team: Team name
            - played: Matches played
            - won: Matches won
            - drawn: Matches drawn
            - lost: Matches lost
            - points_for: Total points scored
            - points_against: Total points conceded
            - points_diff: Points difference
            - bonus_points: Bonus points earned
            - total_points: League points
            - position: Current position
        """
        pass

    def compute_bonus_points(self, row) -> int:
        """
        Compute bonus points for a match.

        Rugby bonus point rules:
        - 4+ tries: 1 bonus point (try bonus)
        - Lose by ≤7: 1 bonus point (losing bonus)

        Returns bonus points earned (0-2)
        """
        pass
```

### File: `rugby_ranking/predictions/league_table.py`

## Phase 5: Season Prediction

### Unplayed Matches

```python
class SeasonPredictor:
    """Predict all remaining matches in a season."""

    def __init__(self, model: RugbyModel, trace, schedule: pd.DataFrame):
        self.model = model
        self.trace = trace
        self.schedule = schedule

    def predict_remaining_matches(self, competition: str, season: str) -> pd.DataFrame:
        """
        Predict all unplayed matches.

        Returns:
            DataFrame with columns:
            - date: Match date
            - home_team: Home team
            - away_team: Away team
            - home_win_prob: Probability of home win
            - away_win_prob: Probability of away win
            - draw_prob: Probability of draw
            - home_score_mean: Expected home score
            - away_score_mean: Expected away score
            - home_tries_mean: Expected home tries
            - away_tries_mean: Expected away tries
        """
        pass

    def simulate_season(self, n_simulations: int = 1000) -> pd.DataFrame:
        """
        Simulate remaining season N times.

        Returns final league tables from each simulation.
        """
        pass

    def compute_playoff_probabilities(self) -> pd.DataFrame:
        """
        From simulations, compute:
        - P(finish 1st)
        - P(finish 2nd)
        - ...
        - P(make playoffs)
        - P(win championship)
        """
        pass
```

### File: `rugby_ranking/predictions/season.py`

## Phase 6: Playoff/Finals Prediction

### Design

```python
class PlayoffPredictor:
    """Predict playoff structure and finals."""

    def __init__(self, model: RugbyModel, trace):
        self.model = model
        self.trace = trace

    def predict_playoff_bracket(self, standings: pd.DataFrame,
                               playoff_format: str) -> dict:
        """
        Given current/predicted standings, determine playoff matchups.

        playoff_format options:
        - "urc": Top 8, 1v8, 2v7, 3v6, 4v5
        - "premiership": Top 4, 1v4, 2v3
        - "top14": Top 6, complex format
        """
        pass

    def predict_knockout_round(self, matchups: list) -> pd.DataFrame:
        """Predict results of playoff round."""
        pass

    def predict_championship(self, top_4: list) -> dict:
        """
        Predict semi-finals and final.

        Returns:
        - P(each team wins championship)
        - Most likely final matchup
        - Expected final score
        """
        pass
```

### File: `rugby_ranking/predictions/playoffs.py`

## Phase 7: Dashboard Integration

### New Dashboard Features

1. **Current League Tables**
   - Live standings for each competition
   - Update after each match week

2. **Predicted League Tables**
   - Expected final standings
   - Uncertainty intervals

3. **Playoff Probabilities**
   - Bar charts showing P(make playoffs) for each team
   - P(win championship)

4. **Upcoming Matches**
   - Next round predictions
   - Win probabilities
   - Expected scores

5. **Season Scenarios**
   - "If Team X wins next 3 games, they have Y% chance of playoffs"

### Dashboard Structure

```
Dashboard/
├── League Tables/
│   ├── Current Standings (live data)
│   └── Predicted Final Standings (from simulations)
├── Playoff Race/
│   ├── Playoff Probabilities (bar chart)
│   └── Championship Odds (bar chart)
├── Upcoming Matches/
│   ├── This Week's Predictions
│   └── Rest of Season Schedule
└── Team Pages/
    ├── Current Form (time-varying effects)
    ├── Remaining Fixtures
    └── Playoff Scenarios
```

### Files to Update
- `dashboard/app.py` - Add new pages
- `dashboard/data_loader.py` - Load predictions
- `export_dashboard_data.py` - Export predictions

## Phase 8: Testing

### End-to-End Test

```bash
# 1. Train model
python train_model.py --model time-varying --data-dir ../Rugby-Data --save-as test_model

# 2. Predict season
python predict_season.py --checkpoint test_model --competition "URC" --season "2024-2025"

# 3. Export for dashboard
python export_dashboard_data.py --checkpoint test_model

# 4. Run dashboard
streamlit run dashboard/app.py
```

### Test Cases
- [ ] Train all model variants
- [ ] Load checkpoints
- [ ] Predict matches
- [ ] Compute league tables
- [ ] Simulate seasons
- [ ] Predict playoffs
- [ ] Export data
- [ ] Dashboard displays correctly

## Implementation Order

1. **Week 1**: Model compatibility audit and fixes
2. **Week 1**: Create `train_model.py` script
3. **Week 1**: Create `LeagueTable` class
4. **Week 2**: Create `SeasonPredictor` class
5. **Week 2**: Create `PlayoffPredictor` class
6. **Week 2**: Update notebooks to load checkpoints
7. **Week 3**: Dashboard integration
8. **Week 3**: End-to-end testing

## File Structure (New)

```
rugby-ranking/
├── train_model.py              # NEW: Unified training script
├── predict_season.py           # NEW: Season prediction script
├── rugby_ranking/
│   ├── predictions/
│   │   ├── __init__.py
│   │   ├── league_table.py     # NEW: League standings
│   │   ├── season.py           # NEW: Season simulation
│   │   └── playoffs.py         # NEW: Playoff prediction
│   └── cli.py                  # UPDATE: Add checkpoint loading
└── notebooks/
    ├── 02_model_fitting.ipynb  # UPDATE: Load checkpoint only
    ├── 03_predictions.ipynb    # UPDATE: Load checkpoint only
    ├── 04_defensive_effects.ipynb
    └── 05_time_varying_effects.ipynb
```

## Questions to Resolve

1. **Bonus point rules**: Do all competitions use same rules?
2. **Playoff formats**: URC vs Premiership vs Top14 differences?
3. **Schedule data**: Where do we get fixture lists?
4. **Time-varying predictions**: Use current form or season average?
5. **Checkpoint versioning**: How to manage multiple checkpoints?

## Success Criteria

- ✅ All models trainable via CLI
- ✅ Notebooks never fit models (only visualize)
- ✅ Can predict all unplayed matches
- ✅ Can compute league tables with bonus points
- ✅ Can predict playoff brackets
- ✅ Dashboard shows current and predicted standings
- ✅ Full pipeline works end-to-end
