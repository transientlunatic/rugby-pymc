# Rugby Ranking Model - Development History

> **This document is the historical development log.** For the current roadmap and priorities, see [ROADMAP.md](ROADMAP.md).

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
  - [x] Team name normalization (handles variations like "Leinster" vs "Leinster Rugby")
  - [x] Fuzzy player name matching to handle typos and spelling variations
- [x] Core model definition (`model/core.py`)
  - [x] Single score type model
  - [x] Joint model for all score types (shares player/team effects via loading factors)
  - [x] Player/team ranking extraction (fixed for joint model)
- [x] Inference machinery (`model/inference.py`)
  - [x] MCMC fitting
  - [x] Variational inference
  - [x] Checkpoint save/load
  - [x] Warm-start support
- [x] Prediction module (`model/predictions.py`)
  - [x] Teams-only predictions
  - [x] Full-lineup predictions
  - [x] Fixed prediction magnitude (properly sums across positions)
- [x] CLI (`cli.py`)
- [x] Notebooks
  - [x] Initial exploration notebook (`01_data_exploration.ipynb`)
  - [x] Model fitting and analysis notebook (`02_model_fitting.ipynb`)
  - [x] Advanced predictions notebook (`03_predictions.ipynb`) (2026-01-31)
    - Multi-match lookahead predictions
    - Team strength evolution analysis
    - Injury scenario modeling templates
    - Custom prediction workflows
    - Calibration/validation framework

### Completed (Phase 2 - Recent Updates)

- [x] Validate end-to-end pipeline
- [x] Fix prediction magnitude issues (scores were ~80 instead of ~20)
  - Root cause: Was treating team effects as multipliers instead of per-player log-rates
  - Solution: Sum expected scores across all 15 positions
- [x] Implement fuzzy player name matching
  - Uses `difflib.SequenceMatcher` with 0.85 similarity threshold
  - Boosts matching confidence for same-team players
  - Handles typos like "Jonny Matthews" vs "Johnny Matthews"
  - Includes manual correction dictionary for known issues
- [x] Update notebook to fully utilize joint model
  - Rankings for all score types (tries, penalties, conversions, drop_goals)
  - Kicker analysis section (combined penalty + conversion ability)
  - Enhanced `analyze_player()` function showing all score types
  - New `compare_players()` function for side-by-side comparison
  - Position effect visualization for all score types
  - Posterior predictive checks for all score types
- [x] Implement separate kicking/try-scoring player effects (2026-01-16)
  - Added `separate_kicking_effect` config option (default: True)
  - Separate player random effects: `beta_player_try` and `beta_player_kick`
  - Try-scoring effect used for tries
  - Kicking effect used for conversions, penalties, drop goals
  - Updated `get_player_rankings()` to handle both model types
  - Updated predictions module to support separate effects
  - Backward compatible: can disable with `separate_kicking_effect=False`
- [x] Implement time-varying effects (within-season trends) (2026-01-18)
  - Added `time_varying_effects` config option (default: False)
  - Player base + trend effects per season: `beta_base[i,s] + beta_trend[i,s] * t`
  - Team base + trend effects per team-season
  - Separate trends for try-scoring vs kicking abilities
  - Season progress computed from match dates (0 = season start, 1 = season end)
  - New `build_joint_time_varying()` method alongside `build_joint()`
  - Captures form changes: improvement/decline over season
  - ~37% more parameters than static model but still VI-compatible

### Completed (Phase 3 - User Experience & Infrastructure)

- [x] Add CLI command for upcoming match predictions (2026-01-30)
  - New `rugby-ranking upcoming` command
  - Shows predictions for matches in next N days (default: 7)
  - Filter by season, competition
  - Formatted output grouped by date
  - Win probabilities and confidence intervals
- [x] Create comprehensive league table and season prediction notebook (2026-01-30)
  - Notebook 06: League table computation with different bonus point systems
  - Monte Carlo season prediction with playoff probabilities
  - Position probability distributions
  - Multiple visualizations (heatmaps, bar charts)
  - Export predictions to CSV
- [x] HTCondor training support with checkpointing (2026-01-31)
  - Periodic checkpointing during VI/MCMC training
  - Auto-resume from latest checkpoint
  - HTCondor submission script template
  - Comprehensive documentation (docs/HTCONDOR_TRAINING.md)
- [x] Validation infrastructure (2026-01-31)
  - Train/test split strategies (temporal, random, season holdout)
  - Validation metrics computation
  - Cross-validation support
  - Baseline comparison
  - Test script and documentation
- [x] Player name analysis tools (2026-01-31)
  - Analyze merged names from fuzzy matching
  - Find potential duplicates
  - Interactive merge review
  - Generate correction dictionaries
  - Export merge reports (CSV/Excel)
  - analyze_player_names.py script
- [x] Position grouping system (2026-01-31)
  - Standard rugby position definitions
  - Position group aggregation (forwards/backs, detailed groups)
  - Filter and rank by position group
  - Visualization of position effects
  - positions.py module
- [x] Project housekeeping & infrastructure modernization (2026-01-31)
  - Sphinx documentation setup with professional theming
  - Shared utilities module (logging, CLI helpers, constants)
  - Comprehensive unit test suite (85+ tests)
  - Notebook boilerplate elimination (92% reduction across 8 notebooks)
  - 15+ locations of duplicate code consolidated to shared modules
  - Analysis scripts for model diagnostics and data quality (Notebooks 09-10)
  - See: HOUSEKEEPING_COMPLETE.md and MODERNIZATION_COMPLETE.md for details

### Next Steps (Phase 3)

- [ ] Run full model fitting on complete dataset and validate results
- [x] Analyze player name merging results and adjust similarity threshold if needed (2026-01-31)
  - Created name_analysis.py module with merge analysis tools
  - Added interactive review and correction dictionary generation
  - Created analyze_player_names.py script
- [ ] Tune priors based on posterior predictive checks
- [x] Add proper position groupings (backs vs forwards, kickers vs non-kickers) (2026-01-31)
  - Created positions.py module with standard rugby position groupings
  - Added aggregation and filtering by position groups
  - Position-specific rankings and visualizations
- [x] Test predictions against held-out matches (2026-01-31)
  - Created validation.py module with train/test split strategies
  - Implemented temporal, random, and season holdout splits
  - Added validation metrics (log-likelihood, RMSE, MAE)
  - Cross-validation support

### Next Steps (Phase 4 - Paths to Victory Analysis) (2026-01-31)

**Goal**: Weekly blog content showing "how team X can finish in position Y" for tournaments like Six Nations

**Design complete**: See `docs/PATHS_TO_VICTORY_DESIGN.md` and `docs/PATHS_TO_VICTORY_SUMMARY.md`

**Implementation timeline**: Needed after Six Nations week 1 (Feb 1-2, 2025)

#### Core Infrastructure
- [x] Design hybrid MCMC-combinatorial approach (2026-01-31)
  - Early tournament: Pattern mining from MCMC simulations using decision trees
  - Late tournament: Weighted combinatorial enumeration with Boolean minimization
  - Automatic switching based on number of remaining games
- [x] Create skeleton implementation (2026-01-31)
  - `paths_to_victory.py` module with `PathsAnalyzer` class
  - Heuristic-based analysis (works now with limited detail)
  - Narrative generation framework
  - Sankey diagram structure (Plotly)
- [x] Create demonstration notebook (2026-01-31)
  - `notebooks/07_paths_to_victory_demo.ipynb`
  - Example usage and blog post generation

#### Phase 4a: MCMC Pattern Mining (Priority for early Six Nations)
- [x] Modify `SeasonPredictor._simulate_season()` to store detailed simulations (2026-02-12)
  - ALREADY IMPLEMENTED: `return_samples` parameter stores game_outcomes and final_positions
  - SeasonSimulationSamples dataclass with teams, fixtures, game_outcomes, final_positions
  - Memory consideration: ~50MB for 5000 detailed sims
- [x] Implement `RuleExtractor.extract_rules()` (2026-02-12)
  - Decision tree (sklearn) on simulation features
  - Extract interpretable conditions from tree paths
  - Calculate conditional probabilities P(success | condition)
  - Rank conditions by importance (ΔP)
- [x] Implement `ScenarioClusterer.cluster()` (2026-02-12)
  - Hierarchical clustering on game outcome feature vectors using scipy
  - Find representative scenario from each cluster (centroid with minimum hamming distance)
  - Generate human-readable descriptions based on team results
  - Target: 3-5 scenario clusters per analysis
- [x] Enhance critical game identification (2026-02-12)
  - ALREADY IMPLEMENTED: `_identify_critical_games_mutual_info()` uses mutual information
  - Calculates MI between game outcome and target position achievement
  - Shows top games ranked by information content

#### Phase 4b: Combinatorial Enumeration (For late tournament)
- [ ] Implement combinatorial outcome enumeration
  - Generate all possible game outcome combinations when tractable (<100k)
  - Include bonus point variations (try bonus, losing bonus)
  - Simulate final league table for each combination
- [ ] Implement probability weighting
  - Use `MatchPredictor` to get P(outcome) for each game
  - Calculate joint probability for each combination
  - Weight final table calculations by probability
- [ ] Boolean minimization of conditions
  - Quine-McCluskey algorithm or Espresso library
  - Simplify complex logical expressions
  - Extract minimal set of conditions for target outcome
- [ ] Optimize performance
  - Parallel processing for large enumerations
  - Caching of league table calculations
  - Early pruning of low-probability branches

#### Phase 4c: Visualization & Output
- [x] Implement Sankey diagram visualization (2026-02-12)
  - Build node/edge structure from simulation paths
  - Show probability flows through critical games
  - Interactive filtering by team/position
  - Export to blog-ready HTML
  - Limit to 3-5 most critical games to avoid clutter
- [x] Enhance narrative generation (2026-02-12)
  - Template-based generation for different styles (detailed/blog/social)
  - Separate "what team controls" vs "what team needs from others"
  - Add confidence intervals and uncertainty quantification
  - Generate comparative analysis ("Team A's path vs Team B's path")
- [x] Create blog post export functionality (2026-02-12)
  - Export to Markdown format
  - Include embedded visualizations
  - Add metadata (date, teams, probabilities)
  - Generate social media snippets (Twitter/LinkedIn/Facebook)

#### Phase 4d: Testing & Validation
- [x] Unit tests for core components (2026-02-12)
  - Test scenario clustering on synthetic data
  - Verify decision tree rule extraction
  - Validate Boolean minimization
  - Test narrative generation (all styles)
  - Test blog/social export functionality
  - Test Sankey diagram creation
- [x] Integration tests (2026-02-12)
  - Test on synthetic Six Nations scenarios (early/mid/late tournament)
  - Verify auto-switching between MCMC and combinatorial modes
  - Check performance at different tournament stages
- [ ] Historical validation (pending real-world data)
  - Run on past Six Nations tournaments (2020-2024)
  - Check if predicted "paths" matched actual outcomes
  - Validate probability calibration
  - Test critical game identification accuracy

#### Phase 4e: CLI & User Interface
- [x] Add CLI commands for paths analysis (2026-02-12)
  - ALREADY IMPLEMENTED in Rugby-Data CLI:
  - `rugby analysis paths-to-victory -t "Scotland" -p 2 -c six-nations`
  - `rugby analysis critical-games -c six-nations -s 2025-2026`
  - Supports JSON export: `--format json -o output.json`
  - Uses mutual information for critical game ranking
  - Auto-enables `return_samples=True` for detailed analysis
- [x] Create configuration for tournaments (2026-02-12)
  - BONUS_RULES_MAP in CLI maps competition names to bonus systems
  - Six Nations → URC rules (no playoffs, standard bonus points)
  - Premiership, Top14, URC each have correct rules
  - Configurable via competition parameter

#### Phase 4f: Knockout Round Forecasting
**Goal**: Predict playoff/knockout rounds where opponents are TBC (World Cup, URC playoffs, Champions Cup knockouts)

**Key challenges**:
- Bracket structure modeling (who plays whom based on league positions)
- Conditional predictions with cascading uncertainty
- Tournament tree simulation through multiple knockout stages

- [x] Implement bracket structure definitions
  - World Cup format (pools → R16 → QF → SF → Final)
  - URC playoffs (1v8, 2v7, 3v6, 4v5 → SF → Final)
  - Champions Cup (R16 → QF → SF → Final)
  - Configurable bracket structures for other tournaments
- [x] Implement conditional knockout predictions
  - "If Team A finishes 1st and Team B finishes 4th, predict that QF matchup"
  - Store predictions indexed by pool position combinations
  - Handle re-seeding rules (e.g., highest seed plays lowest seed)
- [x] Implement tournament tree simulation
  - Extend `SeasonPredictor` to continue through knockout rounds
  - Sample pool positions → determine matchups → predict outcomes → advance winners
  - Track probability of reaching each stage (QF, SF, Final)
  - Calculate tournament win probabilities
- [x] Handle cascading uncertainty
  - Pool position uncertainty → matchup uncertainty → outcome uncertainty
  - Proper probability propagation through tournament tree
  - Visualize uncertainty at each stage
- [ ] Integrate with paths-to-victory analysis
  - "Scotland needs to beat England AND hope Ireland loses" (pool stage)
  - "Scotland needs to win their QF and hope for favorable SF draw"
  - Paths now include both pool and knockout stages
- [ ] Add knockout-specific visualizations
  - Tournament bracket with probabilities
  - "Path to final" diagrams showing most likely route
  - Stage-by-stage survival probabilities
- [x] CLI commands for knockout predictions
  - `rugby-ranking knockout-predict --tournament world-cup`
  - `rugby-ranking tournament-simulate --tournament urc --n-sims 10000`
  - `rugby-ranking bracket-viz --tournament champions-cup --output bracket.html`

### Next Steps (Phase 5 - Squad Analysis & Depth) (2026-01-31)

**Goal**: Analyze squad strength and depth for pre-tournament analysis and injury impact assessment

**Design complete**: See `docs/SQUAD_ANALYSIS_DESIGN.md`

**Implementation timeline**: Needed before Six Nations starts (squads announced ~2 weeks before, Jan 20-ish)

**Key insight**: Bridges gap between "teams-only" (high uncertainty) and "full-lineup" (announced 48h before match) predictions

#### Core Infrastructure
- [x] Design squad analysis system (2026-01-31)
  - Squad strength and depth calculation
  - Lineup prediction from squad composition
  - Injury impact quantification
  - Tournament-wide squad comparison
- [x] Create skeleton implementation (2026-01-31)
  - `squad_analysis.py` module with `SquadParser` class
  - Wikipedia/clipboard text parsing (works now!)
  - Position inference and normalization
  - Interactive squad input function

#### Phase 5a: Squad Parser & Data Entry (Priority - needed first)
- [x] Implement `SquadParser.parse_text()` (2026-01-31)
  - Parse Wikipedia format (most common)
  - Parse simple comma-separated lists
  - Parse CSV files
  - Auto-detect format
- [x] Implement position inference (2026-01-31)
  - Map position text to standard positions
  - Handle plural forms and variations
  - Infer primary and secondary positions
- [x] Add CLI for squad input (2026-01-31)
  - `rugby-ranking squad input --team "Scotland" --from-clipboard`
  - `rugby-ranking squad input --team "Scotland" --file squads/scotland.txt`
  - Interactive mode with validation
- [x] Create squad storage system (2026-01-31)
  - Standard CSV format for squads
  - Directory structure: `squads/{team}_{season}.csv`
  - Automatic directory creation

#### Phase 5b: Squad Strength Analysis (Week before tournament)
- [x] Implement `SquadAnalyzer.get_player_ratings()` (2026-01-31)
  - Extract ratings from model for squad players
  - Handle players not in model (new caps, returning players)
  - Use fuzzy name matching for player lookup
  - Provide rating uncertainty (credible intervals)
- [x] Implement `SquadAnalyzer.create_depth_chart()` (2026-01-31)
  - Rank players by position based on model ratings
  - Account for positional versatility (primary + secondary positions)
  - Identify 1st, 2nd, 3rd choice for each position
- [x] Implement `SquadAnalyzer.calculate_position_strength()` (2026-01-31)
  - Calculate expected strength per position
  - Measure depth (quality of 2nd/3rd choice)
  - Identify vulnerable positions (large drop-off)
  - Include player names for 1st and 2nd choice
- [x] Implement `SquadAnalyzer.calculate_squad_depth_score()` (2026-01-31)
  - Overall squad depth metric (average across positions)
  - Compare depth across position groups
  - Identify strongest and weakest position groups
- [x] Add squad analysis formatting and CLI commands (2026-01-31)
  - `rugby-ranking squad analyze --team "Scotland"`
  - `rugby-ranking squad compare --tournament six-nations`
  - Formatted reports with strength/depth scores
  - Identification of vulnerable positions and critical players

#### Phase 5c: Lineup Prediction (For match predictions)
- [x] Implement `LineupPredictor.predict_lineup()` (2026-02-13)
  - Greedy optimization-based lineup selection
  - Accounts for positional coverage requirements
  - Handles specialist positions (props, hooker, scrum-half)
  - Generates likely bench (8 players with positional cover)
- [x] Implement `LineupPredictor.predict_lineup_distribution()` (2026-02-13)
  - Monte Carlo sampling of possible lineups
  - Accounts for selection uncertainty via rating noise
  - Generates selection probabilities per player
- [x] Position coverage validation (2026-02-13)
  - Validates front row coverage (2 props, 1 hooker on bench)
  - Validates half-back cover on bench
  - Checks lock and back row coverage

#### Phase 5d: Squad-Based Predictions
- [x] Implement `SquadBasedPredictor.predict_with_squads()` (2026-02-13)
  - Samples likely lineups from both squads
  - Predicts match for each lineup pair (simplified heuristic)
  - Aggregates to get expected outcome with uncertainty
  - Returns win probabilities and score distributions
- [x] Quantify lineup uncertainty (2026-02-13)
  - Calculates additional uncertainty from lineup variation
  - Returns standard deviation of score predictions
  - Bridges gap between teams-only and full-lineup modes
- [x] Implement scenario comparison (2026-02-13)
  - Supports unavailable player lists for injury scenarios
  - Can compare baseline vs injury scenarios

#### Phase 5e: Injury Impact Analysis
- [x] Implement `InjuryImpactAnalyzer.analyze_player_impact()` (2026-02-13)
  - Calculates rating drop if specific player unavailable
  - Identifies most likely replacement
  - Quantifies rating drop at that position
  - Calculates criticality score (0-1)
- [x] Implement `InjuryImpactAnalyzer.identify_critical_players()` (2026-02-13)
  - Ranks all squad players by criticality
  - Returns top N most critical players
  - Includes replacement and impact metrics
- [x] Implement `InjuryImpactAnalyzer.analyze_squad_robustness()` (2026-02-13)
  - Simulates random injuries with configurable probability
  - Measures impact on team strength (mean, std, worst/best case)
  - Identifies vulnerable positions
  - Calculates overall robustness score (0-1)

#### Phase 5f: Squad Comparison & Tournament Analysis
- [x] Implement `SquadComparator.compare_squads()` (2026-02-13)
  - Ranks teams by overall squad strength
  - Ranks by squad depth score
  - Generates comparative DataFrame
- [x] Implement `SquadComparator.create_strength_matrix()` (2026-02-13)
  - Creates position-by-team strength matrix
  - Identifies each team's strengths and weaknesses per position
  - Returns DataFrame suitable for heatmap visualization
- [x] Implement `SquadComparator.identify_matchup_advantages()` (2026-02-13)
  - Head-to-head positional comparison
  - Identifies key individual battles
  - Calculates advantage percentages per position

#### Phase 5g: Visualization & Output
- [x] Create squad analysis report template (2026-02-13)
  - `format_squad_analysis()` for console output
  - Shows overall strength and depth scores
  - Position-by-position breakdown with warnings
  - Most likely starting XV and bench
  - Identifies vulnerable and strongest positions
- [x] Create tournament comparison report (2026-02-13)
  - `export_tournament_comparison_to_markdown()` for blog posts
  - Overall squad rankings table
  - Team-by-team breakdown with strengths/vulnerabilities
- [x] Export to blog format (2026-02-13)
  - `export_squad_analysis_to_markdown()` for individual teams
  - Markdown export with embedded tables
  - Includes critical players and robustness analysis
  - Ready for blog publication
- [x] Create visualization function (2026-02-13)
  - `create_squad_visualization()` for matplotlib charts
  - Position strength bar chart
  - Squad depth bar chart with thresholds

#### Phase 5h: Testing & Validation
- [x] Create test suite (2026-02-13)
  - Comprehensive unit tests for all Phase 5 components
  - Tests lineup prediction, injury analysis, squad comparison
  - Tests export functions and validation
  - `tests/test_squad_analysis_phase5.py` created
- [ ] Validate with real data (pending)
  - Test on actual Six Nations squads
  - Compare lineup predictions to actual selections
  - Validate coverage rules with tournament data

### Next Steps (Phase 6 - Prediction Archival & Verification)

**Goal**: Build a system to archive predictions when made and verify them against actual results to track model performance over time

**Key benefits**:
- Transparent performance tracking for blog content ("How did our predictions do?")
- Model improvement by identifying systematic biases
- Calibration validation (are 65% predictions actually correct 65% of the time?)
- Build credibility through transparent tracking

#### Phase 6a: Prediction Storage System
- [ ] Design prediction archive schema
  - Prediction metadata: timestamp, model version, tournament, teams
  - Match metadata: date, venue, competition, season
  - Prediction details: win probabilities, expected scores, confidence intervals
  - Model inputs: lineup info (teams-only vs full-lineup), squad info if applicable
  - Store format: JSON or SQLite database
- [ ] Implement `PredictionArchiver` class
  - `archive_prediction()`: Store prediction when made
  - `update_with_result()`: Add actual outcome when match is played
  - `get_predictions()`: Retrieve predictions with filtering (date range, team, competition)
  - Handle prediction updates (e.g., teams-only → full-lineup as lineups announced)
- [ ] Integrate with prediction workflows
  - Auto-archive when using CLI commands (`rugby-ranking predict`, `rugby-ranking upcoming`)
  - Auto-archive when running season simulations
  - Optional flag to disable archiving (for exploratory analysis)
- [ ] Result ingestion pipeline
  - Automatically fetch results from Rugby-Data repository
  - Match predictions to results (fuzzy matching for team names, date matching)
  - Update archived predictions with actual outcomes
  - Flag unmatched predictions or results for review

#### Phase 6b: Verification & Calibration Analysis
- [ ] Implement calibration metrics
  - Calibration curves (predicted probability vs observed frequency)
  - Brier score (overall prediction accuracy)
  - Log-loss (penalizes confident wrong predictions)
  - By competition, team, time period
- [ ] Implement performance metrics
  - Accuracy (% correct winner predictions)
  - Score prediction RMSE/MAE
  - By prediction mode (teams-only vs full-lineup)
  - By match importance (league vs knockout)
  - By team strength differential
- [ ] Identify systematic biases
  - Home advantage over/under-prediction
  - Favorite vs underdog bias
  - Competition-specific biases
  - Temporal trends (getting better/worse over time)
- [ ] Statistical significance testing
  - Compare model performance to baselines (ELO, betting odds, simple rankings)
  - Confidence intervals on performance metrics
  - Test for significant improvement/degradation over time

#### Phase 6c: Reporting & Visualization
- [ ] Create weekly performance reports
  - "This week's predictions: 6/8 correct"
  - Highlight surprising results (low-probability outcomes that occurred)
  - Show calibration for the week
  - Compare to season-to-date performance
- [ ] Create season summary reports
  - Overall accuracy by competition
  - Best and worst predictions
  - Calibration curves
  - Comparison to betting markets (if available)
- [ ] Visualizations
  - Calibration plots (reliability diagrams)
  - Performance over time (rolling accuracy)
  - Score prediction scatter plots (predicted vs actual)
  - Brier score decomposition (calibration vs resolution)
- [ ] Export for blog posts
  - Markdown format with embedded visualizations
  - Auto-generated performance summaries
  - "Prediction tracker" widget for blog sidebar

#### Phase 6d: Backtesting Infrastructure
- [ ] Historical backtesting framework
  - Re-run predictions on historical data using only info available at the time
  - Simulate weekly model updates on historical data
  - Test different model variants (with/without features)
  - Validate that current approach would have performed well historically
- [ ] Time-travel prediction interface
  - "What would the model have predicted on date X with data available then?"
  - Useful for debugging and understanding model evolution
  - Can compare to what was actually predicted (if archived)
- [ ] A/B testing framework
  - Compare performance of different model variants
  - Test impact of new features or changes
  - Statistical tests for significant differences

#### Phase 6e: CLI & Automation
- [ ] Add CLI commands for verification
  - `rugby-ranking verify --since "2024-01-01"`
  - `rugby-ranking calibration --competition "six-nations"`
  - `rugby-ranking performance-report --weekly`
  - `rugby-ranking backtest --season "2023-2024"`
- [ ] Automated weekly verification
  - Cron job or GitHub Action to fetch results and update archive
  - Generate and post weekly performance report
  - Alert if performance degrades significantly
- [ ] Integration with update pipeline
  - When running `rugby-ranking update`, also verify recent predictions
  - Include verification summary in update output
  - Store verification results alongside model checkpoints

### Future Work (Phase 7)

- [ ] Survival component for substitution/exposure modelling
- [ ] Player-team interaction effects for transfers
- [ ] Game state effects (score differential, red card periods)
- [ ] Career-long trajectories (random walk player effects across seasons)
- [ ] Non-linear within-season trends (splines, GP)
- [ ] Age-based effects (when DOB data available)

### Future Work (Phase 8)

- [ ] Calibration validation (predicted probabilities vs outcomes)
- [ ] Backtesting framework
- [ ] Automated weekly update pipeline
- [ ] Web dashboard for rankings/predictions

### Additional Data (Phase 9)

- [ ] Japanese league data (Wikipedia?)
- [ ] Southern hemisphere domestic leagues (Super Rugby, Currie Cup, Argentina, etc)
- [ ] International matches not already present in the data set (wikipedia?)
- [ ] Player biographical data (wikipedia?)
- [ ] Coaches (wikipedia?)
- [ ] Older Premiership/URC data (wikipedia; news reports)
- [ ] Track transfers to allow commentary in pre-season analysis
- [ ] Historical squad data (Wikipedia?; for pre-tournament analysis and squad strength tracking)

### Additional infrastructure (Phase 10)

- [ ] Move to a more robust database system (PostgreSQL? something which will work on dreamhost shared hosting) for storing predictions, results, player info
- [ ] RESTFul API for accessing data required for web dashboard and related tools and experiments, synchronised with training data

### Additional analyses and tools (Phase 11)

- [ ] Bayesian ranking of teams and players 
- [ ] Elo Ranking of teams and players

### Fun web tools

- [ ] Web-based player comparison tool 
- [ ] Interactive match simulator (select lineups, simulate match)
- [ ] "What-if" scenario explorer (e.g., "What if Scotland had won against England?")
- [ ] Player career trajectory visualizer
- [ ] Fantasy team generator based on model ratings

## Interesting future analyses
- Has home advantage changed over time?
- Are there stadiums where home advantage is stronger/weaker?
- How do player effects evolve over their career?
- Can we identify "late bloomers" or "early peak" players?
- How do team effects evolve with coaching changes?

## Model variants to explore
- Remove the kicking defensive effect
- Add a separate player defensive effect 


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
├── PLAN.md                      # This file
├── README.md                    # Main documentation
├── train_model.py               # Training script with checkpointing
├── rugby_ranking/
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   └── model/
│       ├── __init__.py
│       ├── data.py              # Data pipeline
│       ├── core.py              # PyMC model definition
│       ├── inference.py         # Fitting machinery (with checkpointing)
│       ├── predictions.py       # Match predictions
│       ├── league_table.py      # League table computation
│       ├── season_predictor.py  # Season prediction with Monte Carlo
│       ├── paths_to_victory.py  # Paths to victory analysis (NEW)
│       ├── squad_analysis.py    # Squad strength & depth analysis (NEW)
│       ├── validation.py        # Train/test splits and metrics
│       ├── name_analysis.py     # Player name merge analysis
│       ├── positions.py         # Position groupings
│       └── data_validation.py   # Data quality checks
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_fitting.ipynb
│   ├── 03_predictions.ipynb
│   ├── 04_defensive_effects_demo.ipynb
│   ├── 05_time_varying_effects.ipynb
│   ├── 06_league_table_and_season_prediction.ipynb
│   ├── 07_paths_to_victory_demo.ipynb  # Paths analysis demo (NEW)
│   └── 08_squad_analysis_demo.ipynb    # Squad analysis demo (NEW)
├── tests/                       # Test and validation scripts
│   ├── test_validation.py       # Validation infrastructure test
│   └── ...
├── scripts/                     # Utility scripts
│   ├── submit_training.sub      # HTCondor submission script
│   └── analyze_player_names.py  # Name analysis tool
├── docs/                        # Documentation
│   ├── INDEX.md                 # Documentation index
│   ├── HTCONDOR_TRAINING.md     # HTCondor training guide
│   ├── PATHS_TO_VICTORY_DESIGN.md     # Paths analysis detailed design (NEW)
│   ├── PATHS_TO_VICTORY_SUMMARY.md    # Paths analysis summary (NEW)
│   ├── SQUAD_ANALYSIS_DESIGN.md       # Squad analysis detailed design (NEW)
│   └── ...
└── dashboard/                   # Web dashboard
```

## Dependencies

- PyMC >= 5.10 (Bayesian modelling)
- ArviZ >= 0.17 (diagnostics, visualization)
- pandas, numpy, xarray (data handling)
- matplotlib, seaborn (plotting)
- scikit-learn (optional: for paths to victory decision trees and clustering)
- plotly (optional: for paths to victory Sankey diagrams)

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

# View upcoming matches (this weekend)
rugby-ranking upcoming --data-dir /path/to/Rugby-Data --days 7 --checkpoint latest

# Filter upcoming by competition
rugby-ranking upcoming --data-dir /path/to/Rugby-Data --days 7 --competition "premiership"

# Paths to victory analysis (future)
rugby-ranking paths --team "Scotland" --position 2 --tournament six-nations
rugby-ranking critical-games --tournament six-nations --week 3
rugby-ranking generate-blog --tournament six-nations --week 3

# Squad analysis (future)
rugby-ranking squad input --team "Scotland" --season "2024-2025" --from-clipboard
rugby-ranking squad analyze --team "Scotland" --season "2024-2025"
rugby-ranking squad lineup --team "Scotland" --opponent "England"
rugby-ranking squad injury-impact --team "Scotland" --player "Finn Russell"
rugby-ranking squad compare --tournament six-nations --season "2024-2025"
rugby-ranking squad predict --home "Scotland" --away "England" --season "2024-2025"
```

## Recent Fixes and Improvements

### Prediction Magnitude Fix (2026-01-15)

**Problem**: Match predictions were unrealistically high (82-62 instead of ~20-22)

**Root Cause**: The prediction code was treating team effects as multipliers on a baseline rate (e.g., `3.5 * exp(gamma)`), but in the model structure `gamma` represents the per-player log-rate contribution.

**Solution**: Rewrote predictions to properly sum expected contributions across all 15 starting positions:
```python
for pos in range(15):
    home_log_rate = alpha + home_team_effect + theta[pos] + eta_home + player_noise
    home_tries_expected += exp(home_log_rate)
```

### Fuzzy Player Name Matching (2026-01-15)

**Problem**: Player names have typos and spelling variations (e.g., "Jonny" vs "Johnny Matthews")

**Implementation**:
- `PlayerNameMatcher` class with configurable similarity threshold (default 0.85)
- Uses `difflib.SequenceMatcher` with surname-weighted matching
- Boosts matching confidence when players are on the same team
- Manual corrections dictionary for known problematic names
- Enabled by default in `MatchDataset`

**Usage**:
```python
dataset = MatchDataset(data_dir, fuzzy_match_names=True, name_similarity_threshold=0.85)
merged = dataset.get_merged_names()  # See what was merged
potential_dupes = dataset.get_potential_duplicates()  # Review edge cases
```

### Joint Model Ranking Methods (2026-01-15)

**Problem**: `get_player_rankings()` and `get_team_rankings()` threw KeyError for `beta_player` / `gamma_team_season` which don't exist in joint model

**Solution**: Added joint model detection and proper effect computation:
- Joint model uses `beta_player_raw` scaled by `sigma_player * lambda_player[score_type]`
- Similarly for team effects: `gamma_team_season_raw` scaled by `sigma_team * lambda_team[score_type]`
- Methods now auto-detect model type and compute effects correctly

### Enhanced Notebook (2026-01-15)

**Added functionality**:
1. **Multi-score-type rankings**: Show top players for tries, penalties, conversions, drop goals
2. **Kicker analysis**: Combined penalty + conversion ability rankings with scatter plots
3. **Enhanced `analyze_player()`**:
   - Shows summary table across all score types
   - Displays total scores, rates per 80 minutes, and posterior effects
   - Color-coded posterior plots (green = above average, red = below)
4. **New `compare_players()`**: Side-by-side comparison across all score types
5. **Position effects**: Visualizations for all score types (not just tries)

## Notes

- Data quality: Some position numbers > 23 exist in source data (filter to 1-23)
- Player disambiguation: Fuzzy matching handles most cases, but manual review recommended for borderline similarities
- Team name variations handled automatically (e.g., "Leinster" → "Leinster Rugby")
- Older data (DICT format) uses surname matching only since full lineups not always available
