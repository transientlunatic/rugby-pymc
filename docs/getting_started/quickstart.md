# Quick Start

## 1. Load and Fit Model

```python
from pathlib import Path
from rugby_ranking.model import MatchDataset, RugbyModel, ModelFitter

# Load match data from Rugby-Data repository
dataset = MatchDataset(Path("/path/to/Rugby-Data"))
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

# Build model (joint model with separate try-scoring / kicking effects)
model = RugbyModel()
model.build_joint(df)

# Fit with variational inference (fast, ~5 minutes)
fitter = ModelFitter(model)
trace = fitter.fit_vi()

# Or use MCMC for thorough fitting (slow, ~4-8 hours)
# trace = fitter.fit_mcmc()

# Save checkpoint for later use
fitter.save("my_model")
```

## 2. Generate Rankings

After fitting, rankings are extracted directly from the model:

```python
# Top try-scorers (uses model.trace set automatically by fitter)
player_rankings = model.get_player_rankings(score_type='tries', top_n=20)
print(player_rankings)

# Best kickers
kicker_rankings = model.get_player_rankings(score_type='penalties', top_n=20)
print(kicker_rankings)

# Best teams this season
team_rankings = model.get_team_rankings(season="2025-2026", top_n=10)
print(team_rankings)
```

## 3. Predict Match Outcomes

```python
from rugby_ranking.model import MatchPredictor

# Create predictor (trace is set on model.trace after fitting)
predictor = MatchPredictor(model, trace)

# Teams only (no lineup needed — higher uncertainty)
pred = predictor.predict_teams_only("Leinster", "Munster", season="2025-2026")
print(pred.summary())

# With full lineups (lower uncertainty)
home_lineup = {1: "Andrew Porter", 2: "Dan Sheehan", 3: "Tadhg Furlong",
               4: "Joe McCarthy", 5: "James Ryan", 6: "Caelan Doris",
               7: "Josh van der Flier", 8: "Jack Conan", 9: "Jamison Gibson-Park",
               10: "Ross Byrne", 11: "James Lowe", 12: "Robbie Henshaw",
               13: "Garry Ringrose", 14: "Jordan Larmour", 15: "Hugo Keenan"}
away_lineup = {1: "Jeremy Loughman", 2: "Niall Scannell", 3: "Stephen Archer",
               4: "Jean Kleyn", 5: "Fineen Wycherley", 6: "Peter O'Mahony",
               7: "John Hodnett", 8: "Gavin Coombes", 9: "Craig Casey",
               10: "Jack Crowley", 11: "Shane Daly", 12: "Malakai Fekitoa",
               13: "Rory Scannell", 14: "Simon Zebo", 15: "Mike Haley"}
pred = predictor.predict_full_lineup(
    "Leinster", "Munster",
    home_lineup, away_lineup,
    season="2025-2026"
)
print(pred.summary())
```

## 4. Analyze Squad Strength

```python
import pandas as pd
from rugby_ranking.model.squad_analysis import SquadAnalyzer

analyzer = SquadAnalyzer(model, trace)

# Load squad from CSV (columns: player, position_text)
squad = pd.read_csv("scotland_2025.csv")

# Get comprehensive squad analysis
analysis = analyzer.analyze_squad(squad, team="Scotland", season="2025-2026")
print(analysis)
```

## 5. Load a Saved Checkpoint

```python
from rugby_ranking.model import RugbyModel, ModelFitter

# Reload a previously saved model
model = RugbyModel()
fitter = ModelFitter(model)
fitter.load("my_model")  # restores model + trace

# Now use as normal
predictor = MatchPredictor(model, fitter.trace)
```

## Next Steps

- Read about [Model Architecture](../guides/model_fundamentals)
- Check [Weekly Training Workflow](../guides/weekly_workflow)
- Explore [Predictions Guide](../guides/predictions)
