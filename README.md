# Rugby Ranking

Bayesian hierarchical models for ranking rugby union players and teams, with match score predictions.

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Load Data and Fit Model

```python
from pathlib import Path
from rugby_ranking.model import MatchDataset, RugbyModel, ModelFitter

# Load match data
dataset = MatchDataset(Path("/path/to/Rugby-Data"))
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

# Build and fit model
model = RugbyModel()
# For joint model with separate kicking/try-scoring effects (default):
model.build_joint(df)

# Or for single score type:
# model.build(df, score_type="tries")

fitter = ModelFitter(model)
trace = fitter.fit_vi()  # Fast (~5 min)
# trace = fitter.fit_mcmc()  # Thorough (~30-60 min)

# Save for later
fitter.save("my_checkpoint")
```

### 2. Extract Rankings

```python
# Top try-scorers
player_rankings = model.get_player_rankings(top_n=20)
print(player_rankings)

# Best teams this season
team_rankings = model.get_team_rankings(season="2025-2026", top_n=10)
print(team_rankings)
```

### 3. Predict Matches

```python
from rugby_ranking.model import MatchPredictor

predictor = MatchPredictor(model)

# Teams only (higher uncertainty)
pred = predictor.predict_teams_only("Leinster", "Munster", season="2025-2026")
print(pred.summary())

# With full lineups (lower uncertainty)
home_lineup = {1: "Andrew Porter", 2: "Dan Sheehan", ...}
away_lineup = {1: "Jeremy Loughman", 2: "Niall Scannell", ...}
pred = predictor.predict_full_lineup(
    "Leinster", "Munster",
    home_lineup, away_lineup,
    season="2025-2026"
)
```

## Command Line Interface

```bash
# Weekly model update
rugby-ranking update --data-dir /path/to/Rugby-Data --method vi

# View rankings
rugby-ranking rankings --type players --top 20
rugby-ranking rankings --type teams --season 2025-2026

# Predict a match
rugby-ranking predict --home "Leinster" --away "Munster"
```

## Model Architecture

### Hierarchical Structure

The model estimates scoring rates using a hierarchical Poisson regression:

```
log(λ) = α + β_player + γ_team_season + θ_position + η_home + log(exposure)
```

Where:
- **β_player**: Intrinsic player ability (follows player across teams)
  - When `separate_kicking_effect=True` (default), uses separate effects:
    - **β_player_try**: Try-scoring ability (used for tries)
    - **β_player_kick**: Kicking ability (used for conversions, penalties, drop goals)
- **γ_team_season**: Team system effect (coaching, tactics, squad quality)
- **θ_position**: Positional base rate (wings score more tries than props)
- **η_home**: Home advantage
- **exposure**: Minutes played / 80 (adjusts for substitutions)

### Scoring Types

Separate models for each scoring type:
- **Tries** (5 pts): All positions, backs score more
- **Conversions** (2 pts): Primarily fly-halves (#10) and fullbacks (#15)
- **Penalties** (3 pts): Primarily fly-halves (#10) and fullbacks (#15)
- **Drop goals** (3 pts): Rare, primarily fly-halves

The model recognizes that try-scoring and kicking are distinct skills. By default, it uses:
- **β_player_try** for tries (running ability, positioning, finishing)
- **β_player_kick** for conversions, penalties, and drop goals (kicking accuracy, technique)

This allows players to have different rankings for try-scoring vs kicking ability.

### Configuration Options

You can customize the model behavior using `ModelConfig`:

```python
from rugby_ranking.model import ModelConfig, RugbyModel

# Use separate kicking and try-scoring effects (default)
config = ModelConfig(separate_kicking_effect=True)
model = RugbyModel(config=config)

# Or use a single player effect for all scoring types (legacy behavior)
config = ModelConfig(separate_kicking_effect=False)
model = RugbyModel(config=config)

# Adjust prior scales
config = ModelConfig(
    player_try_effect_sd=0.5,
    player_kicking_effect_sd=0.5,
    team_effect_sd=0.3,
    position_effect_sd=0.5,
)
```

### Player Mobility

Players changing teams are handled naturally:
- `β_player` persists across teams (intrinsic ability)
- `γ_team_season` captures the system they're playing in
- Model can estimate "how much of a player's output is them vs their team"

## Inference Options

| Method | Speed | Use Case |
|--------|-------|----------|
| **VI (ADVI)** | ~5 min | Weekly updates, quick iteration |
| **MCMC (NUTS)** | ~30-60 min | Monthly validation, final results |

### Weekly Update Workflow

```python
# Load previous checkpoint
fitter = ModelFitter.load("latest", model)

# Update with new data
dataset.load_json_files()
df = dataset.to_dataframe()
model.build(df, score_type="tries")

# Fast refit using warm start
trace = fitter.fit_vi(warm_start=True)
fitter.save("latest")
```

### Diagnostics

```python
diag = fitter.diagnostics()
print(f"R-hat max: {diag['r_hat_max']:.3f}")  # Should be < 1.01
print(f"ESS min: {diag['ess_bulk_min']:.0f}")  # Should be > 400
```

## Project Structure

```
rugby_ranking/
├── model/
│   ├── data.py         # Data loading and preprocessing
│   ├── core.py         # PyMC model definition
│   ├── inference.py    # MCMC/VI fitting, checkpoints
│   └── predictions.py  # Match predictions
├── cli.py              # Command-line interface
notebooks/
├── 01_data_exploration.ipynb
└── 02_model_fitting.ipynb
```

## Data Requirements

Expects JSON files from the Rugby-Data repository with:
- Match lineups (positions 1-23)
- Scoring events (type, player, minute)
- Substitution times (on/off arrays)
- Card events (yellows, reds)

Supports both LIST format (recent files) and DICT format (older files).

## Training on HTCondor

For long-running training jobs on HTCondor clusters, see [docs/HTCONDOR_TRAINING.md](docs/HTCONDOR_TRAINING.md) for:
- Automatic checkpointing and resume
- HTCondor submission script
- Best practices for resilient training

## Documentation

- [PLAN.md](PLAN.md) - Project roadmap and implementation status
- [docs/INDEX.md](docs/INDEX.md) - Comprehensive documentation index
- [docs/HTCONDOR_TRAINING.md](docs/HTCONDOR_TRAINING.md) - HTCondor training guide
- [docs/DASHBOARD.md](docs/DASHBOARD.md) - Web dashboard setup

See [docs/](docs/) for detailed documentation on implementation, features, and fixes.
