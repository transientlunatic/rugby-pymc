# Rugby Ranking Model - Project Plan

## Overview

A Bayesian hierarchical model for ranking rugby union players and teams, with support for match score predictions. The model is designed for weekly incremental updates as new matches are played.

## Data Source

- **Repository**: `Rugby-Data` (separate repository)
- **Coverage**: ~20 years of URC/Celtic, Premiership, Top14, European competitions
- **Format**: JSON files with match lineups, scoring events, substitutions, cards
- **Scale**: ~288k player-match observations, ~8.3k players, ~6.5k matches

## Model Architecture

### Joint Survival-Poisson Structure

The model combines:
1. **Poisson processes** for discrete scoring events (tries, conversions, penalties, drop goals)
2. **Survival component** (future) for time-to-event modelling (substitutions, cards)

### Hierarchical Random Effects

```
log(λ_score[i,m]) = α                           # baseline
                  + β_player[i]                 # intrinsic player ability
                  + γ_team[j,s]                 # team-season system effect
                  + δ_player×team[i,j]          # player-team fit (future)
                  + θ_position[k]               # positional base rate
                  + η_home × is_home            # home advantage
                  + log(minutes / 80)           # exposure offset
```

### Player Mobility Handling

- `β_player[i]` follows player across teams
- `γ_team[j,s]` is team-season specific (captures coaching, squad changes)
- `δ_player×team[i,j]` (future) captures player-system fit

## Implementation Status

### Completed (Phase 1)

- [x] Repository structure and pyproject.toml
- [x] Data pipeline (`model/data.py`)
  - [x] LIST format loader (recent files)
  - [x] DICT format loader (older files)
  - [x] Player-match observation extraction
  - [x] Exposure time calculation from on/off arrays
  - [x] Scoring event counting (case-insensitive)
  - [x] Player mobility tracking
- [x] Core model definition (`model/core.py`)
  - [x] Single score type model
  - [x] Joint model for all score types
  - [x] Player/team ranking extraction
- [x] Inference machinery (`model/inference.py`)
  - [x] MCMC fitting
  - [x] Variational inference
  - [x] Checkpoint save/load
  - [x] Warm-start support
- [x] Prediction module (`model/predictions.py`)
  - [x] Teams-only predictions
  - [x] Full-lineup predictions
- [x] CLI (`cli.py`)
- [x] Initial exploration notebook

### Next Steps (Phase 2)

- [ ] Install package and run exploration notebook to validate end-to-end
- [ ] Fit simple model (tries only) to validate infrastructure
- [ ] Tune priors based on posterior predictive checks
- [ ] Add proper position groupings (backs vs forwards, kickers vs non-kickers)

### Future Work (Phase 3)

- [ ] Survival component for substitution/exposure modelling
- [ ] Player-team interaction effects for transfers
- [ ] Game state effects (score differential, red card periods)
- [ ] Time-varying player effects (form, aging)

### Future Work (Phase 4)

- [ ] Calibration validation (predicted probabilities vs outcomes)
- [ ] Backtesting framework
- [ ] Automated weekly update pipeline
- [ ] Web dashboard for rankings/predictions

## Key Design Decisions

### Inference Strategy

| Frequency | Method | Use Case |
|-----------|--------|----------|
| Weekly | VI (ADVI) | Fast updates, ~2-5 min |
| Monthly | Full MCMC | Validation, ~30-60 min |

VI warm-starts from previous posterior for efficiency.

### Prediction Modes

1. **Teams-only** (1 week before): Higher uncertainty, marginalizes over likely lineups
2. **Full-lineup** (1-2 days before): Lower uncertainty, uses announced team sheets

### Scoring Types

Modelled as separate processes with shared player/team effects:
- Tries (5 points)
- Conversions (2 points) - conditional on team tries
- Penalties (3 points) - primarily fly-halves/fullbacks
- Drop goals (3 points) - rare

## File Structure

```
rugby-ranking/
├── pyproject.toml
├── PLAN.md                    # This file
├── rugby_ranking/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   └── model/
│       ├── __init__.py
│       ├── data.py            # Data pipeline
│       ├── core.py            # PyMC model definition
│       ├── inference.py       # Fitting machinery
│       └── predictions.py     # Match predictions
└── notebooks/
    └── 01_data_exploration.ipynb
```

## Dependencies

- PyMC >= 5.10 (Bayesian modelling)
- ArviZ >= 0.17 (diagnostics, visualization)
- pandas, numpy, xarray (data handling)
- matplotlib, seaborn (plotting)

## Usage

```bash
# Install
pip install -e /path/to/rugby-ranking

# Weekly update
rugby-ranking update --data-dir /path/to/Rugby-Data --method vi

# View rankings
rugby-ranking rankings --type players --top 20
rugby-ranking rankings --type teams --season 2025-2026

# Predict match
rugby-ranking predict --home "Leinster" --away "Munster"
```

## Notes

- Data quality: Some position numbers > 23 exist in source data (filter to 1-23)
- Player disambiguation: Some common surnames may conflate different players
- `euro-challenge-2021-2022.json` has unknown format (skipped)
