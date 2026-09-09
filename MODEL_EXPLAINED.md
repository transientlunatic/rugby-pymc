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

> **Update (2026-08-23):** The numbers below were, until this update, illustrative
> placeholders — nobody had actually run the validation tooling against real
> data and recorded the result. They read as measured results, so they were
> replaced with a real one. See `validation_reports/2026-08-23_teams-only-holdout.json`
> for the full report and `validation_reports/2026-08-23_run_script.py` for
> the script that produced it; re-run and replace as the model changes.
> A first version of this validation computed the baseline's home-win rate
> and mean score from the held-out test set itself (data leakage, caught in
> PR review) — fixed by computing the baseline from the training set only,
> as the methodology below now describes. Re-running with the fix barely
> moved the numbers (this dataset's train/test splits happen to have similar
> home-win rates), but the corrected numbers are what's reported here.
>
> **Update (2026-08-24):** Added a real Elo rating baseline (see
> `rugby_ranking/model/elo.py`) — the trivial baseline above is the floor,
> not a fair fight for a ranking system. On the single split available at
> the time, Elo beat the Bayesian model on every metric measured.
>
> **Update (2026-08-24, later): replicated on two more holdout splits, and
> it doesn't hold up as a clean win.** See "Replication across three
> holdout splits" below — Elo's edge was specific to that one split, not a
> general result.

### Methodology

- **Data**: last 3 seasons of real match data (2024–2026 start years),
  69,897 player-match rows across 1,523 matches.
- **Split**: temporal holdout — trained on the first 1,295 matches
  chronologically, evaluated on the following 228 matches the model never
  saw during fitting (Dec 2025–Mar 2026).
- **Model**: production config (`include_defense=True,
  separate_kicking_effect=True`), fit via VI/ADVI (the fast weekly path),
  20,000 iterations. Fit took ~420s.
- **Predictions**: `MatchPredictor.predict_teams_only()` (higher-uncertainty
  mode; no lineup information used) for each of the 228 held-out matches.
- **Trivial baseline**: the cheapest possible skill-free reference, computed
  entirely from the training set (never the test set) — predict the
  training set's empirical home-win rate (67.6%, so "always predict home")
  as a fixed probability for every match, and the training set's mean score
  for every scoreline.
- **Elo baseline**: a standard Elo rating system, one rating per team across
  a global pool (not split by competition, to stay comparable with the
  Bayesian model, which also doesn't condition on competition). K-factor and
  home-advantage are calibrated by grid search on an *inner* chronological
  split of the training data only (same leakage discipline as the trivial
  baseline — the real test set is never touched during calibration), with
  season regression-to-the-mean and a linear margin model turning rating
  differences into predicted scores (the standard sports-analytics
  point-spread technique). See `rugby_ranking/model/elo.py` for the full
  implementation and rationale.

### Results (out-of-sample, 228 held-out matches)

| Metric | Model | Elo baseline (k=32, home_adv=80) | Trivial baseline (train-set rate/mean) |
|--------|-------|------|------|
| Win accuracy | 70.6% | **73.2%** | 68.0% |
| Brier score (lower is better) | 0.4215 | **0.4174** | 0.4509 |
| Score RMSE | 13.04 | **12.29** | 12.92 |
| Score MAE | 10.59 | — | — |

**Honest read (superseded — see replication below):** on this one split,
Elo beat the full Bayesian hierarchical model on every metric measured.
That looked like a consistent, single-direction result. It wasn't — see
below.

### Replication across three holdout splits

The single-split result above was flagged in this document itself as
needing replication before treating "Elo wins" as settled. Re-ran the same
methodology (production model config, VI/20000 iterations, Elo calibrated
fresh on each split's training data) at three different `--as-of` cutoffs,
giving three non-overlapping test windows:

| Test window (n matches) | Model win acc | Elo win acc | Model Brier | Elo Brier | Model RMSE | Elo RMSE |
|---|---|---|---|---|---|---|
| Dec 2025 – Mar 2026 (228) | 70.6% | **73.2%** | 0.4215 | **0.4174** | 13.04 | **12.29** |
| Jan – May 2025 (244) | 69.7% | 69.7% | 0.4338 | **0.4233** | **12.20** | 12.24 |
| Feb – May 2024 (246) | **73.6%** | 72.4% | 0.4058 | **0.4045** | **11.62** | 11.67 |
| **Mean across splits** | 71.3% | 71.8% | 0.4204 | 0.4151 | 12.29 | 12.07 |

**Honest read:** the original "Elo wins everything" result does not
replicate. It was specific to the one split first tested (which also has
the largest test set and the most recent data). Across all three splits:

- **Win accuracy**: a wash. Elo wins split 1, ties split 2, loses split 3.
  Mean difference (+0.5 points to Elo) is well inside noise at n≈240 per
  split.
- **Brier score**: Elo wins all three splits, consistently, though narrowly
  in two of them. This is the one place a real, repeatable Elo edge shows
  up — its win-probability calibration is a bit better than the Bayesian
  model's across different time periods.
- **Score RMSE**: a wash overall. Elo's big win-margin-averaged headline
  number was carried almost entirely by split 1; in the other two splits
  the model is marginally *better* on RMSE, by less than the difference
  between two coin flips.

**Revised conclusion**: the Bayesian model and Elo are roughly comparable
on match outcome and score prediction — neither clearly beats the other
once you look past a single split. Elo does appear to have a small, real
edge in probability calibration (Brier). Given that Elo is a two-parameter
system that fits in milliseconds and the Bayesian model takes ~5 minutes of
VI per fit, "roughly comparable, Elo calibrates its win probabilities a bit
better" is still not a flattering result for the added complexity — but
it's a materially different, more measured claim than "Elo wins
everything," and it's the one actually supported by three splits rather
than one. What the Bayesian model has that Elo fundamentally cannot
produce — per-player scoring rates, lineup-conditional predictions,
uncertainty intervals — is unaffected by any of this; whether that extra
output justifies the complexity for this project's actual use case remains
a real, open question this validation doesn't answer by itself.

### Per-player scoring-rate calibration (out-of-sample PIT)

This is a different, more encouraging result: how well do the underlying
per-player Poisson rates match reality, independent of how they aggregate
into a match-level score.

| Score type | Predicted mean | Observed mean | MACE |
|---|---|---|---|
| Tries | 0.1542 | 0.1581 | 0.007 |
| Conversions | 0.1204 | 0.1189 | 0.033 |
| Penalties | 0.0662 | 0.0394 | 0.027 (over-predicts) |
| Drop goals | 0.0021 | 0.0003 | 0.003 |

Tries and conversions are well-calibrated out-of-sample. Penalties are
over-predicted by about 68% relative to the observed rate.

**Update (2026-08-24): root-caused, and it isn't a model bug.** Penalty
rate is falling sharply, season over season, in the recent-seasons slice
this validation trains on:

| Season | Penalties / 80 min |
|---|---|
| 2024-2025 | 0.1006 |
| 2025-2026 | 0.0731 |
| 2026-2027 (partial) | 0.0428 |

Train period (bulk of 2024-2025 + most of 2025-2026) pools to 0.093
penalties/80min; the test period (the most recent ~228 matches) pools to
0.057 — a 1.63x ratio, which alone accounts for essentially all of the
1.68x over-prediction ratio above. The model's `alpha[penalties]`
intercept is a single constant fit across the whole training window, so
it reflects the training period's blended (higher) rate and doesn't know
the rate kept falling into the test period. This tracks a real,
identifiable trend, likely from law/officiating changes aimed at reducing
kickable penalties (rugby has seen several such trials across this
window) rather than a genuine change in team discipline — but this
dataset can't distinguish those causes, only measure the trend.

Ruled out as a contributing cause: attribution errors in scorer-name
matching. `MatchDataset._count_scoring_events` falls back to surname-only
matching when a scorer's full name doesn't match a lineup entry exactly,
which could in principle mis-attribute or double-count a score when two
players share a surname. Checked directly against the raw match data:
47.8% of matches have *some* duplicate surname across both 23-player
squads (unsurprising), but only 1.43% of actual penalty-scoring events
(193 of 13,520) hit that ambiguous fallback path at all — nowhere near
enough to explain a 68% systematic bias, and it would bias observed
counts (inflating them), not predicted ones.

**What this means for the model**: `alpha[s]` (and every other
score-type intercept) assumes a rate that's constant over the training
window. `time_varying_effects` (see `ModelConfig`) already exists but
only models *within-season* form trends (0→1 progress through a single
season) — it doesn't reach across season boundaries, so it wouldn't have
caught this. A real fix needs either a season-level trend/random-walk
term on the intercepts, or recency-weighting/truncating the training
window so stale seasons stop pulling the fitted rate up. Not implemented
here — this is a structural model change that touches every score type's
intercept, not a one-line fix, and deserves its own validation pass
rather than being rushed alongside this investigation. Tracked in
ROADMAP.md.

### Is per-scorer try credit fair? Testing the team-attribution hypothesis

The model credits tries to whoever's name is against the score in the data
— the same treatment as conversions and penalties. But a try is
substantially a team output (phases won, go-forward ball, a break made by
someone else) with one player finishing it, unlike a kick, which is a
clean individual act independent of teammates. That raises a real
question: is crediting only the scorer the wrong unit of attribution, and
would crediting presence on the pitch (or lineup contribution more
broadly) capture more signal?

Tested directly (2026-08-26) with two methods on the 2023+ seasons slice,
both using **teammates' tries only** (the player's own tries excluded, so
personal scoring isn't mistaken for a broader team effect) and both
demeaning by team-season (or using a within-player fixed-effects
regression) to remove team-quality confounds — comparing matches where a
player was in the lineup against matches their team played without them:

1. **Binary on/off** (7,869 player-team-season comparisons, ≥3 matches
   each side): teammates score **0.090 fewer tries/match** when the player
   is present than when absent (p<0.0001, 95% CI [−0.112, −0.068]).
2. **Continuous minutes regression** (308,121 player-match panel rows, far
   more statistical power): confirms the same direction with much tighter
   estimates.

**Position-stratified, this splits cleanly by who actually scores, not by
forward/back:**

| Position group | n (panel rows) | Effect of 80 vs 0 min | p-value |
|---|---|---|---|
| **Props (1, 3)** | 51,419 | **−0.027 tries/match** | **0.45 (null)** |
| Front row (1,2,3) | 76,287 | −0.086 | 0.003 |
| All forwards (1-8) | 174,540 | −0.078 | <0.0001 |
| All backs (9-15) | 133,581 | −0.144 | <0.0001 |
| Back three + centres | 89,728 | −0.166 | <0.0001 |

Every individual back position is significant and negative except
fly-half (p=0.35 — a distributor/kicker, not primarily a finisher). Among
forwards, locks and props show nothing (p=0.21–0.67), but hooker
(p<0.0001), openside flanker (p=0.008), and number eight (p=0.03) do —
and those three, plus the significant backs, are exactly `positions.py`'s
own `HIGH_TRY_SCORERS` list (wings, fullback, flankers, number eight) plus
hooker (a real occasional try-scorer via driving mauls, just not on that
list). The effect tracks **personal scoring involvement**, not general
team contribution.

**Conclusion: this doesn't support reframing try credit toward team
presence, and props are the cleanest possible negative case.** A prop is
about as pure an "enabling, rarely-scoring" role as exists in rugby — if
box-score presence were going to reveal a broad team-attribution signal
anywhere, it should be there. It doesn't: the effect is a tight null
(±0.03–0.06 tries/match) at both binary and continuous resolution. What
*does* show up is a redistribution pattern among players who personally
score — a strong finisher's presence takes tries away from teammates
roughly in proportion to what he adds himself, netting out close to the
model's existing small positive total-team-tries effect from an earlier,
less careful cut of this same test (+0.042 tries/match, before excluding
personal scoring) — not evidence of the team scoring more overall.

Honest limitation: this measures *presence*, not contribution quality or
role (a struggling prop counts the same as a dominant one), and it
compares "this specific player" against "whoever replaced him," not
"a player at this position" against "nobody" — for a specialist position
like prop, clubs generally carry adequate like-for-like depth, which is a
plausible reason a real per-match contribution could exist without
showing up in aggregate replacement comparisons. So this rules out
*recovering* a team-attribution signal from match-level try counts via
presence — it doesn't rule out that props matter to tries in ways this
dataset can't see (scrum penalties won, metres in the tackle, none of
which are recorded here).

### What this doesn't cover yet

- No MCMC comparison run on this same split (would show whether VI's known
  miscalibration on `alpha_tries`/`sigma_player_kick` — see the SBC tests —
  actually moves these numbers).
- Three fixed holdout splits (see replication above), not a true rolling
  backtest with many overlapping windows — the direction (Elo/model roughly
  comparable, Elo a bit better calibrated) is more trustworthy now than the
  single-split result was, but three points is still a small sample of
  possible splits.
- Teams-only predictions only; full-lineup mode not evaluated here.

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

*Last updated: August 2026 (validation section: real measured results, see above)*
