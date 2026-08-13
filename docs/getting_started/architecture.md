# Architecture Overview

## System Design

The rugby-ranking system combines several components:

```
Data Pipeline
    ↓
MatchDataset (loads JSON)
    ↓
Data Validation & Cleaning
    ↓
Core Model (PyMC)
    ├─ Hierarchical random effects
    ├─ Position effects
    ├─ Home advantage
    └─ Exposure adjustment
    ↓
Inference Engine
    ├─ Variational Inference (fast, weekly)
    └─ MCMC (thorough, monthly validation)
    ↓
Predictions & Analysis
    ├─ Match predictions
    ├─ Rankings
    ├─ Squad analysis
    └─ Paths to victory
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `model.data` | Load and parse JSON match data |
| `model.core` | PyMC model definition |
| `model.inference` | Fitting algorithms (VI, MCMC) |
| `model.predictions` | Match predictions and rankings |
| `model.squad_analysis` | Squad strength and depth |
| `model.league_table` | Tournament standings |
| `cli` | Command-line interface |
| `utils` | Shared utilities and constants |

## Model Structure

The hierarchical model with joint scoring:

$$\log(\lambda_{i,j,k}) = \alpha_k + \beta_{i} + \gamma_{j} + \theta_p + \eta_h + \log(m/80)$$

Where:
- $\alpha_k$ = baseline for score type $k$
- $\beta_i$ = player ability effect
- $\gamma_j$ = team-season quality effect
- $\theta_p$ = position effect
- $\eta_h$ = home advantage
- $m$ = minutes played (exposure)

The model is estimated for each score type (tries, conversions, penalties, drop goals) separately but with shared random effects.

See [Model Fundamentals](model_fundamentals) for detailed explanation.
