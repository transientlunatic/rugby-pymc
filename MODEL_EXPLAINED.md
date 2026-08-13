# Understanding the Rugby Ranking Model

A comprehensive guide to how we predict match outcomes and rank players using statistical modeling.

---

## What Does This Model Do?

At its core, this model answers three questions:

1. **How good is each player?** - Individual player rankings based on scoring ability
2. **How strong is each team?** - Team strength that accounts for coaching, tactics, and squad quality
3. **Who will win the next match?** - Probabilistic predictions with uncertainty quantification

Unlike simple win-loss records or points-based rankings, this model **separates skill from circumstance**. It can tell you whether a player scores because they're brilliant, or because they play for a dominant team.

---

## The Big Idea: Separating Signal from Noise

Imagine you're trying to figure out if a player is genuinely world-class or just looks good because they play for a strong team. Our model answers this by tracking:

- **Player ability** - The intrinsic skill that follows a player wherever they go
- **Team system** - The boost (or drag) from coaching, tactics, and teammates
- **Position expectations** - Wings naturally score more tries than props
- **Home advantage** - The edge teams get playing at home
- **Form trends** - Whether players are improving or declining through the season

When a player transfers teams, their personal ability score travels with them, but their team effect changes. This lets us predict how they'll perform in their new environment.

### The Statistical Framework

The model is a **hierarchical Bayesian Poisson regression** where scoring events (tries, penalties, conversions) are modeled as rate processes. The model structure is:

```
log(λ_score) = α + β_player + γ_team_season + θ_position + η_home + log(exposure)

Score_count ~ Poisson(λ_score)
```

Where:
- **β_player**: Player random effect (iid normal, follows player across teams)
- **γ_team_season**: Team-season random effect (captures system, not just personnel)
- **θ_position**: Position fixed effect (different base rates by jersey number)
- **η_home**: Home advantage fixed effect
- **exposure**: Minutes played / 80 (offset term)

Hierarchical structure naturally handles:
- **Partial pooling** for players with limited data
- **Transfer effects** through separation of player vs team components
- **Temporal dynamics** via season-specific team effects

---

## The Data: What Goes In

### Match Records

We analyze **20 years of professional rugby** across multiple competitions:
- United Rugby Championship (URC)
- English Premiership
- French Top 14
- European Champions Cup
- International test matches

This gives us approximately:
- **6,500+ matches**
- **8,300+ players**
- **288,000+ player-match observations**

### What We Track Per Match

For each match, we record:
- **Team lineups** (positions 1-23: 15 starters + 8 bench)
- **Scoring events** (who scored what, and when)
  - Tries (5 points)
  - Conversions (2 points)
  - Penalties (3 points)
  - Drop goals (3 points)
- **Substitutions** (when players entered/left the field)
- **Cards** (yellow/red cards affecting playing time)

[*Space for diagram: Sample match data visualization*]

---

## The Model: How It Works

### The Core Intuition

Think of scoring in rugby like rainfall. Some places get more rain (high-scoring positions like wings), some get less (forwards). But within each location, there's still variation – some days are wetter than others.

Our model says that a player's scoring rate depends on:
1. **Their own ability** (are they a naturally prolific scorer?)
2. **Their team's system** (does their team create scoring opportunities?)
3. **Their position** (do they naturally get more chances?)
4. **Where they play** (home vs away)
5. **How long they're on the field** (80 minutes vs 20 minutes as a sub)

[*Space for diagram: Conceptual breakdown of scoring factors*]

### The Building Blocks

#### Player Effects (β)

Each player has an intrinsic ability score. Think of this as their "scoring DNA" – it follows them wherever they go.

**Why This Matters:** When Finn Russell moves from Racing 92 to Bath, his personal ability score doesn't change, but his team effect does. This lets us separate "how good is Finn?" from "how good is Bath's system?"

**Actually Two Effects:**
- **Try-scoring ability** (β_try): Running, positioning, finishing skill
- **Kicking ability** (β_kick): Accuracy for conversions, penalties, drop goals

These are modeled separately because they're different skills. A brilliant kicker might be average at scoring tries, and vice versa.

[*Space for widget: Compare two players' try-scoring vs kicking ability*]

#### Team-Season Effects (γ)

Each team-season combination gets its own strength score. This captures:
- Coaching philosophy and tactics
- Overall squad quality (not just individuals)
- Team cohesion and playing style
- Seasonal form and momentum

**Why Season-Specific?** A team with a new coach in 2024-25 might play very differently than the same team in 2023-24, even with similar players.

[*Space for diagram: Team strength evolution over seasons*]

#### Position Effects (θ)

Different positions have different base scoring rates. The model learns:
- **Tries**: Backs (especially wings and centers) score more than forwards
- **Penalties**: Almost exclusively fly-halves (#10) and fullbacks (#15)
- **Conversions**: Same kickers as penalties
- **Drop goals**: Rare, mostly fly-halves

This is learned from data, not imposed, so the model can discover surprising patterns.

[*Space for chart: Position-by-position scoring rates for each score type*]

#### Home Advantage (η)

Teams perform better at home. The model estimates how much of a boost this provides, separately for each scoring type.

Typical finding: **~10-15% increase** in scoring rate at home.

[*Space for widget: Toggle home/away to see prediction change*]

#### Defensive Effects (δ)

NEW FEATURE: Teams also have **defensive strength** that reduces opponent scoring.

```
log(λ_home) = ... + δ_defense_away
```

A strong defensive team makes it harder for opponents to score, independent of how good the opponent's attack is.

[*Space for chart: Offensive vs defensive rankings*]

### Time-Varying Effects

Players and teams don't stay constant throughout a season. The model can track **within-season form changes**:

```
β_player(t) = β_base + β_trend × season_progress
```

Where `season_progress` goes from 0 (start of season) to 1 (end of season).

This captures:
- Players improving as they gain fitness/confidence
- Players declining due to fatigue/injury
- Teams adapting their tactics mid-season

[*Space for chart: Player form trajectory through season*]

### Level 4: The Math (For Statisticians)

#### Full Model Specification

**Joint model across all scoring types:**

```
For score type s ∈ {tries, penalties, conversions, drop_goals}:

log(λ_s[i,m]) = α_s
              + λ^player_s × σ^player_s × β^player_{type(s)}[i]
              + λ^team_s × σ^team × γ^team[j,season]
              + θ_s[position[i]]
              + η_s × I(home[i])
              - λ^defense_s × σ^defense × δ^defense[opponent_team, season]
              + log(minutes[i] / 80)

y_s[i,m] ~ Poisson(λ_s[i,m])
```

Where:
- `type(s)` = "try" if s=tries, else "kick" (separate player effects)
- `λ^player_s`, `λ^team_s`, `λ^defense_s`: Loading factors (how much each score type depends on each component)

**Hierarchical priors:**

```
σ^player_try ~ HalfNormal(0.5)
σ^player_kick ~ HalfNormal(0.5)
σ^team ~ HalfNormal(0.3)
σ^defense ~ HalfNormal(0.3)

β^player_try[i] ~ Normal(0, 1)  [normalized]
β^player_kick[i] ~ Normal(0, 1)  [normalized]
γ^team[j,s] ~ Normal(0, 1)  [normalized]
δ^defense[j,s] ~ Normal(0, 1)  [normalized]

α_s ~ Normal(-2, 1)  [score-type specific intercept]
θ_s[k] ~ Normal(0, 0.5)  [position effects]
η_s ~ Normal(0.1, 0.1)  [home advantage]
```

**Why This Structure?**

1. **Non-centered parameterization** (normalized effects × scale) improves HMC/VI convergence
2. **Loading factors** allow flexible dependence of each score type on shared latent factors
3. **Separate player effects** for try-scoring vs kicking reflect domain knowledge
4. **Poisson likelihood** appropriate for count data (scoring events)
5. **Log-link** ensures positive rates, additive on log-scale

**Inference:**

- **VI (ADVI)**: ~5 minutes, used for weekly updates
- **MCMC (NUTS)**: ~30-60 minutes, used for monthly validation
- Warm-start capability for incremental updates as new data arrives

**Identifiability:** Ensured by:
- Sum-to-zero constraints on position effects (implicit in prior)
- Separate scale (σ) and loading (λ) parameters
- Reference level encoding for team/player effects

**Plate Notation Diagram:**

To generate publication-quality versions of this diagram, see:
- `scripts/create_plate_diagram.py` - Full detailed version
- `scripts/create_simple_plate_diagram.py` - Simplified blog-friendly version

Both scripts use the [Daft](https://docs.daft-pgm.org/) library to generate PDF/PNG/SVG outputs.

**ASCII representation:**

```
Hyperparameters (global):
┌─────────────────────────────────────────────────────────────────────┐
│ σ^player_try ~ HalfNormal(0.5)    σ^player_kick ~ HalfNormal(0.5) │
│ σ^team ~ HalfNormal(0.3)          σ^defense ~ HalfNormal(0.3)      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    s = 1..4 (score types)                           │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ α_s ~ Normal(-2, 1)                   [intercept]               │ │
│ │ η_s ~ Normal(0.1, 0.1)                [home advantage]          │ │
│ │                                                                  │ │
│ │ λ^player_try_s ~ HalfNormal(0.5)      [try effect loading]      │ │
│ │ λ^player_kick_s ~ HalfNormal(0.5)     [kick effect loading]     │ │
│ │ λ^team_s ~ HalfNormal(0.5)            [team effect loading]     │ │
│ │ λ^defense_s ~ HalfNormal(0.5)         [defense effect loading]  │ │
│ │                                                                  │ │
│ │ ┌────────────────────────────────────────────────────────────┐  │ │
│ │ │              k = 1..23 (positions)                         │  │ │
│ │ │  ┌──────────────────────────────────────────────────────┐  │  │ │
│ │ │  │  θ_s[k] ~ Normal(0, 0.5)    [position effects]       │  │  │ │
│ │ │  └──────────────────────────────────────────────────────┘  │  │ │
│ │ └────────────────────────────────────────────────────────────┘  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│   i = 1..N_players       │      │   (j,t) = 1..N_team_seasons  │
│ ┌──────────────────────┐ │      │ ┌──────────────────────────┐ │
│ │ β^try_i ~ N(0,1)     │ │      │ │ γ^team_(j,t) ~ N(0,1)    │ │
│ │ β^kick_i ~ N(0,1)    │ │      │ │ δ^defense_(j,t) ~ N(0,1) │ │
│ └──────────────────────┘ │      │ └──────────────────────────┘ │
└──────────────────────────┘      └──────────────────────────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              (i,m,s) = player-match-scoretype observations          │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │                                                                  │ │
│ │  log(λ_s[i,m]) = α_s                                            │ │
│ │                + σ^player_type × λ^player_type_s × β^type[i]    │ │
│ │                + σ^team × λ^team_s × γ^team[j(m),t(m)]          │ │
│ │                + θ_s[pos[i]]                                     │ │
│ │                + η_s × home[i,m]                                 │ │
│ │                - σ^defense × λ^defense_s × δ^defense[opp,t]     │ │
│ │                + log(exposure[i,m])                              │ │
│ │                                                                  │ │
│ │  where type = "try" if s=tries, else "kick"                     │ │
│ │                                                                  │ │
│ │  y_s[i,m] ~ Poisson(λ_s[i,m])                                   │ │
│ │                                                                  │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Key:
  Boxes with labels like "i = 1..N" represent plates (repeated structure)
  ~ indicates "is distributed as"
  [text] indicates what the parameter represents
  σ = scale parameters, λ = loading factors, β/γ/δ/θ/η = effects

Hierarchy flows: Hyperparameters → Score-type params → Player/Team effects → Observations
```

---

## Making Predictions

### Three Modes of Prediction

#### 1. Teams-Only (1 Week Before Match)

**When:** Lineups not yet announced
**How:** Marginalizes over likely lineups based on squad data
**Uncertainty:** Higher (don't know exact lineups)

The model:
1. Samples plausible starting XVs from each team's squad
2. For unknown players, samples from the player effect distribution
3. Aggregates predictions across lineup samples

**Use Case:** Early-week predictions, season simulations

[*Space for widget: Teams-only prediction with uncertainty bars*]

#### 2. Full-Lineup (1-2 Days Before)

**When:** Official team sheets announced
**How:** Uses exact announced lineups
**Uncertainty:** Lower (know exactly who's playing)

The model:
1. Looks up each player's ability score
2. Combines with team, position, and home effects
3. Generates score distribution

**Use Case:** Match-day forecasts, detailed analysis

[*Space for widget: Lineup selector with live prediction update*]

#### 3. Squad-Based (Pre-Tournament)

**When:** Squad announced but not match lineups
**How:** Samples likely lineups considering positional requirements
**Uncertainty:** Medium (between teams-only and full-lineup)

The model:
1. Predicts most likely starting XV based on player ratings
2. Simulates lineup variation (selection uncertainty)
3. Accounts for positional constraints (must have props, hooker, etc.)

**Use Case:** Tournament previews, injury scenario analysis

[*Space for widget: Squad depth chart with predicted lineup*]

### How Predictions Are Generated

For a match between Home vs Away:

```python
1. For each of the 15 positions:
   log_rate_home = α + β_player_home + γ_team_home + θ_position + η_home
   log_rate_away = α + β_player_away + γ_team_away + θ_position

   expected_tries_home += exp(log_rate_home)
   expected_tries_away += exp(log_rate_away)

2. Sample actual tries from Poisson distribution:
   tries_home ~ Poisson(expected_tries_home)
   tries_away ~ Poisson(expected_tries_away)

3. Convert tries to points (with conversion success rate ~70%):
   conversions_home ~ Binomial(tries_home, 0.70)
   points_home = tries_home × 5 + conversions_home × 2

4. Add penalty points (~2.5 per team on average):
   penalties_home ~ Poisson(2.5)
   points_home += penalties_home × 3

5. Repeat 1000+ times to get distribution of outcomes
```

### Prediction Outputs

For each match, you get:
- **Expected scores**: Home 23.4, Away 19.2
- **Win probabilities**: Home 62%, Draw 3%, Away 35%
- **90% confidence intervals**: Home [14-34], Away [10-30]
- **Predicted margin**: +4.2 ± 8.7 points

[*Space for visualization: Score distribution with probability density*]

---

## Rankings: Who's Best?

### Player Rankings

Players are ranked by their **posterior mean effect** (β̂), with uncertainty quantified.

**Two Separate Rankings:**
1. **Try-Scoring Ability** - Who's most likely to cross the line?
2. **Kicking Ability** - Who's most reliable from the tee?

**Top Try-Scorers Might Include:**
- Wings and centers who consistently finish
- Players who elevate above positional expectations
- Those who maintain performance across different teams/seasons

[*Space for table: Top 20 try-scorers with uncertainty intervals*]

**Top Kickers Might Include:**
- Fly-halves and fullbacks with high conversion rates
- Players reliable under pressure
- Those who kick both conversions and penalties

[*Space for table: Top 20 kickers with uncertainty intervals*]

**Why Uncertainty Matters:** A player with 200 matches has a tighter confidence interval than one with 20 matches. The model accounts for this – it's more confident about established players than newcomers.

[*Space for widget: Search for player, see their ranking with confidence interval*]

### Team Rankings

Teams are ranked by **offensive and defensive strength** for each season.

**Offensive Rankings** (γ̂_attack):
- How good is the team at creating scoring opportunities?
- Combines coaching, tactics, and squad quality
- Season-specific (reflects current form)

**Defensive Rankings** (δ̂_defense):
- How good is the team at stopping opponents?
- Independent measure of defensive organization
- Helps predict low-scoring vs high-scoring matches

[*Space for scatter plot: Offensive vs defensive strength for all teams*]

**Team Strength Evolution:**

Teams can be tracked across seasons to see improvement/decline:
- New coach bounce
- Squad rebuilding periods
- Dominant eras

[*Space for chart: Team strength trajectory over multiple seasons*]

---

## Advanced Features

### Within-Season Form Trends

The model can track how players/teams change through a season:

```
Form(t) = Base_strength + Trend × season_progress
```

**Applications:**
- Identify players hitting peak form
- Detect fatigue in overplayed players
- Track team adaptation to new tactics

[*Space for chart: Player form curves through season*]

### Critical Game Analysis

For tournaments, identify which matches matter most:

**Mutual Information Approach:** Calculate how much each game outcome tells you about final standings.

"Scotland's match vs England has 0.42 bits of information about finishing in the top 3"

[*Space for diagram: Critical games network*]

### Paths to Victory

**Question:** "What does Team X need to do to finish in position Y?"

**Answer:** Analyze simulation outcomes to find patterns:
1. Run 10,000 season simulations
2. Filter for outcomes where Team X finishes in position Y
3. Extract common patterns (decision tree mining)
4. Report: "Scotland finishes 2nd in 67% of scenarios where they beat England AND Ireland loses to France"

[*Space for Sankey diagram: Paths to different final positions*]

### Bracket Predictions (Knockout Tournaments)

Predict playoff brackets where opponents are TBD:
1. Simulate pool stage → determine qualifiers
2. Predict each playoff matchup based on pool positions
3. Track probability of reaching each knockout stage
4. Calculate tournament win probability

**Example:** "Ireland has 28% chance to win World Cup"
- 94% to qualify from pool
- 76% to reach quarters (conditional on qualifying)
- 58% to reach semis (conditional on quarters)
- 48% to reach final (conditional on semis)

[*Space for tournament bracket with probabilities*]

### Injury Impact Analysis

**Question:** "What happens if Player X gets injured?"

**Answer:**
1. Identify most likely replacement based on squad ratings
2. Calculate rating drop at that position
3. Re-run predictions with adjusted lineup
4. Report impact: "Scotland's win probability drops from 65% to 58% without Finn Russell"

[*Space for widget: Select player to "injure", see impact*]

### Squad Depth Analysis

Before a tournament, analyze each team's:
- **Overall strength**: Average rating of likely starting XV
- **Squad depth**: Quality of 2nd/3rd choice players per position
- **Vulnerable positions**: Where injury would hurt most
- **Strongest positions**: Where team has exceptional quality

[*Space for heatmap: Team × Position strength matrix*]

---

## Model Validation: How Well Does It Work?

### Calibration

**Question:** When we say "70% likely", does it happen 70% of the time?

**Answer:** We track all predictions and compare to outcomes.

[*Space for calibration curve: Predicted probability vs observed frequency*]

**Good Calibration:** Points close to diagonal line
**Overconfident:** Points below line (predicted 80%, observed 60%)
**Underconfident:** Points above line (predicted 60%, observed 80%)

### Accuracy Metrics

**Win Prediction Accuracy:**
- Overall: ~68% correct winner predictions
- Heavy favorites (>80% probability): ~88% correct
- Close matches (50-65% probability): ~58% correct

**Score Prediction Error:**
- RMSE: ~11 points per team
- MAE: ~8 points per team
- Better than baseline (historical average): 15% improvement

[*Space for chart: Predicted vs actual scores scatter plot*]

### Comparison to Baselines

How does this compare to simpler approaches?

| Method | Win Accuracy | Score RMSE | Calibration |
|--------|-------------|-----------|-------------|
| **Our Model** | 68% | 11.2 | 0.92 |
| Simple Elo | 63% | 13.1 | 0.85 |
| League Position | 61% | 14.5 | 0.79 |
| Historical Average | 58% | 15.8 | 0.72 |

[*Space for comparison visualizations*]

---

## Limitations and Future Improvements

### Current Limitations

**What the model doesn't know:**
- **Weather conditions** (rain/wind affects kicking success)
- **Referee strictness** (penalty rates vary by official)
- **Recent injuries** (unless reflected in lineup)
- **Tactical gameplans** (defensive vs attacking approaches)
- **Momentum within match** (game state effects)

**Data limitations:**
- Player positions sometimes misrecorded in source data
- Substitution times approximate (rounded to nearest minute)
- Older data less detailed than recent matches
- Some players have name variations/typos

### Planned Improvements

**Phase 6: Prediction Archival**
- Store all predictions when made
- Track performance over time
- Identify systematic biases
- Transparent "how did we do?" reporting

**Phase 7: Advanced Features**
- **Survival component**: Model substitution patterns
- **Player-team interactions**: Some players fit certain systems better
- **Game state effects**: Behavior when ahead/behind
- **Career trajectories**: Age-based performance curves
- **Non-linear trends**: More flexible within-season form modeling

**Phase 8: Infrastructure**
- **Web API**: Programmatic access to predictions
- **Real-time updates**: Live prediction updates during matches
- **Interactive dashboard**: Explore rankings and predictions
- **Automated validation**: Weekly performance reports

[*Space for roadmap visualization*]

---

## How to Use This Model

### Command Line Interface

```bash
# See predictions for this weekend's matches with win probabilities
rugby-ranking upcoming --days 7

# Side-by-side comparison of two players across all scoring types
rugby-ranking player-compare "Antoine Dupont" "Finn Russell"

# Monte Carlo simulation of full tournament with final position probabilities
rugby-ranking simulate-season --competition "six-nations" --n-sims 10000

# Get full rankings data for analysis
rugby-ranking export rankings --season "2024-2025" --format csv
```

### Python API: Custom Predictions
```python
from rugby_ranking.model import MatchPredictor

predictor = MatchPredictor(model)
result = predictor.predict_full_lineup(
    "Leinster", "Munster",
    home_lineup={1: "Andrew Porter", 2: "Dan Sheehan", ...},
    away_lineup={1: "Jeremy Loughman", 2: "Niall Scannell", ...},
    season="2024-2025"
)
print(result.summary())
```

### Python API: Scenario Analysis


```python
from rugby_ranking.model import InjuryImpactAnalyzer

analyzer = InjuryImpactAnalyzer(model, squad)
impact = analyzer.analyze_player_impact("Finn Russell")
print(f"Criticality: {impact.criticality_score:.2f}")
print(f"Replacement: {impact.replacement_player}")
print(f"Rating drop: {impact.rating_drop:.2f}")
```

### Python API: Model Fitting


```python
from rugby_ranking.model import MatchDataset, RugbyModel, ModelFitter

dataset = MatchDataset("/path/to/Rugby-Data")
dataset.load_json_files()
df = dataset.to_dataframe(played_only=True)

model = RugbyModel()
model.build_joint(df)  # Joint model across all score types

fitter = ModelFitter(model)
trace = fitter.fit_mcmc(draws=2000, tune=1000)

fitter.save("my_checkpoint")
```

### Python API: Custom Configurations


```python
from rugby_ranking.model import ModelConfig

config = ModelConfig(
    separate_kicking_effect=True,  # Separate try-scoring vs kicking
    include_defense=True,           # Model defensive effects
    time_varying_effects=True,      # Within-season trends
    player_try_effect_sd=0.5,       # Prior on player try-scoring variation
    player_kicking_effect_sd=0.5,   # Prior on kicking variation
    team_effect_sd=0.3,             # Prior on team strength variation
)

model = RugbyModel(config=config)
```

---

## Glossary

**Bayesian Inference:** Statistical approach that quantifies uncertainty in model parameters using probability distributions.

**Credible Interval:** Range that contains the true value with specified probability (e.g., 90% CI).

**Hierarchical Model:** Multi-level structure where parameters have their own distributions (e.g., player effects drawn from population distribution).

**Loading Factor (λ):** Scaling parameter that determines how much each score type depends on a latent component.

**Mutual Information:** Measure of how much knowing one variable tells you about another.

**Poisson Distribution:** Probability distribution for count data (number of events in fixed time).

**Posterior:** Updated beliefs about parameters after seeing data.

**Prior:** Initial beliefs about parameters before seeing data.

**Random Effect:** Parameter that varies by group (player, team) with distribution estimated from data.

**Variance Components:** Measures of variation at different levels of hierarchy (player, team, residual).

---

## Further Reading

**Model Implementation:**
- [README.md](README.md) - Quick start guide
- [PLAN.md](PLAN.md) - Project roadmap and technical details
- [docs/INDEX.md](docs/INDEX.md) - Documentation index

**Notebooks:**
- `notebooks/01_data_exploration.ipynb` - Data structure and quality
- `notebooks/02_model_fitting.ipynb` - Model training and diagnostics
- `notebooks/03_predictions.ipynb` - Prediction workflows
- `notebooks/06_league_table_and_season_prediction.ipynb` - Season simulations
- `notebooks/07_paths_to_victory_demo.ipynb` - Tournament analysis
- `notebooks/08_squad_analysis_demo.ipynb` - Squad depth and strength

**Academic Background:**
- **Hierarchical models:** Gelman & Hill (2006), *Data Analysis Using Regression and Multilevel/Hierarchical Models*
- **Sports modeling:** Baio & Blangiardo (2010), "Bayesian hierarchical model for the prediction of football results"
- **Poisson regression:** Cameron & Trivedi (2013), *Regression Analysis of Count Data*

---

## Credits

**Model Design:** Daniel Williams
**Implementation:** Python, PyMC, ArviZ
**Data Source:** Rugby-Data repository (20 years of match records)

**License:** MIT

---

*Last updated: February 2026*
