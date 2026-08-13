# Paths to Victory - Design Document

## Overview

A system for analyzing and visualizing how teams can achieve specific final positions in a tournament (e.g., Six Nations). Uses hybrid MCMC-combinatorial approach that adapts based on how many games remain.

## Use Cases

1. **Weekly blog content**: "How Scotland can finish 1st/2nd/3rd this week"
2. **Interactive tool**: Users select team + target position, get narrative + visualization
3. **Critical game identification**: Which upcoming matches most affect outcomes

## Architecture

### Core Components

```
paths_to_victory.py
├── PathsAnalyzer           # Main analysis class
├── ScenarioClusterer       # Groups similar simulation outcomes
├── RuleExtractor           # Generates minimal logical conditions
├── NarrativeGenerator      # Creates human-readable descriptions
└── SankeyVisualizer        # Interactive flow diagrams
```

### Data Flow

```
SeasonPrediction (MCMC sims)
    ↓
PathsAnalyzer.analyze_paths(team="Scotland", position=2)
    ↓
    ├→ ScenarioClusterer: Group similar paths
    ├→ RuleExtractor: Find minimal conditions
    └→ NarrativeGenerator: Create readable summary
    ↓
PathsOutput:
  - probability: 0.23
  - critical_games: [(England vs Italy, ΔP=0.15), ...]
  - key_conditions: ["England must lose to Italy (89%)", ...]
  - scenario_clusters: [...]
```

## Phase 1: MCMC Pattern Mining (Many Games Remaining)

### Algorithm

```python
def analyze_paths_mcmc(target_team, target_position, simulations):
    """Early tournament: Extract patterns from MCMC simulations."""

    # 1. Filter successful scenarios
    successful_sims = [
        sim for sim in simulations
        if sim.final_position[target_team] == target_position
    ]

    # 2. Extract game outcomes from each simulation
    features = extract_game_outcomes(successful_sims)
    # features[i,j] = 1 if team_a beat team_b in simulation i

    # 3. Find critical games using decision tree
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(all_features, is_successful)
    critical_conditions = extract_rules_from_tree(tree)

    # 4. Calculate conditional probabilities
    for condition in critical_conditions:
        p_success_given_condition = count_sims(condition & success) / count_sims(condition)

    # 5. Generate narrative
    return format_narrative(critical_conditions, probabilities)
```

### Key Features to Extract

For each simulation, create binary feature vector:
- Game results: `england_beat_italy`, `france_beat_wales`, ...
- Bonus points: `scotland_bonus_vs_ireland`, ...
- Point thresholds: `scotland_total_points_gte_15`, ...
- Head-to-heads: `scotland_beat_france`, ...

### Rule Extraction Methods

**Option 1: Decision Trees** (sklearn)
- Pros: Interpretable, automatic feature selection
- Cons: Greedy splitting may miss interactions

**Option 2: Association Rules** (mlxtend.frequent_patterns)
- Pros: Finds co-occurring conditions
- Cons: Generates many rules, needs filtering
- Example: `{england_lose_italy, france_no_bonus} → {scotland_2nd}` (confidence: 0.85)

**Option 3: Random Forest Feature Importance**
- Pros: Robust, ranks game importance
- Cons: Less interpretable than decision trees

**Recommendation**: Use decision tree for primary rules, feature importance for ranking games

## Phase 2: Weighted Combinatorics (Few Games Remaining)

### Switchover Threshold

```python
def should_use_combinatorics(remaining_fixtures):
    """Determine if combinatorial enumeration is tractable."""
    n_games = len(remaining_fixtures)

    # Each game: Win/Draw/Loss × Bonus point combinations
    # Approximate: ~8 outcomes per game (win±bonus, draw, loss±bonus)
    n_combinations = 8 ** n_games

    # Tractable if < 100k combinations (runs in seconds)
    return n_combinations < 100_000
```

For Six Nations:
- Round 5 (5 games): ~33k combinations ✓
- Round 4 (10 games): ~1B combinations ✗

### Combinatorial Enumeration Algorithm

```python
def analyze_paths_combinatorial(target_team, target_position,
                                 current_table, remaining_fixtures,
                                 match_predictor):
    """Late tournament: Enumerate and weight all outcomes."""

    # 1. Generate all possible game outcomes
    all_outcomes = enumerate_game_outcomes(remaining_fixtures)
    # Each outcome: {('ENG', 'ITA'): ('win', True, False), ...}
    #                                  result  try_bp  losing_bp

    # 2. For each outcome, calculate final table
    final_tables = []
    for outcome in all_outcomes:
        table = simulate_outcome(current_table, outcome)
        final_tables.append(table)

    # 3. Weight by MCMC probabilities
    weights = []
    for outcome in all_outcomes:
        p = 1.0
        for game, result in outcome.items():
            pred = match_predictor.predict(game)
            p *= pred.probability(result)
        weights.append(p)

    # 4. Find outcomes where target achieved
    successful_outcomes = [
        (outcome, weight, table)
        for outcome, weight, table in zip(all_outcomes, weights, final_tables)
        if table.position[target_team] == target_position
    ]

    # 5. Simplify using Boolean minimization
    conditions = extract_conditions(successful_outcomes)
    simplified = minimize_boolean_expression(conditions)

    return {
        'probability': sum(w for _, w, _ in successful_outcomes),
        'conditions': simplified,
        'top_paths': sorted(successful_outcomes, key=lambda x: x[1])[:10]
    }
```

### Boolean Minimization

Use Quine-McCluskey or Espresso algorithm to simplify logical expressions:

```
Input:
  (ENG_lose_ITA ∧ FRA_lose_WAL) ∨
  (ENG_lose_ITA ∧ FRA_draw_WAL) ∨
  (ENG_lose_FRA ∧ IRE_lose_SCO)

Output:
  ENG_lose_ITA ∨ (ENG_lose_FRA ∧ IRE_lose_SCO)
```

## Scenario Clustering

Group similar simulation outcomes to avoid overwhelming the user.

### Clustering Features

For each simulation, extract:
- **Game results vector**: [1, 0, 1, ...] for each game (win/loss/draw)
- **Points achieved**: Final points for each team
- **Critical game outcomes**: Did high-impact games go favorably?

### Clustering Method

```python
from sklearn.cluster import AgglomerativeClustering

def cluster_scenarios(successful_sims, n_clusters=5):
    """Group similar simulation outcomes."""

    # Extract feature vectors
    X = np.array([
        encode_simulation(sim) for sim in successful_sims
    ])

    # Hierarchical clustering (interpretable dendrograms)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)

    # Find representative scenario from each cluster
    representatives = []
    for i in range(n_clusters):
        cluster_sims = [s for s, l in zip(successful_sims, labels) if l == i]
        # Pick centroid or most probable
        rep = find_representative(cluster_sims)
        representatives.append({
            'scenario': rep,
            'frequency': len(cluster_sims) / len(successful_sims),
            'description': generate_description(rep)
        })

    return representatives
```

### Representative Selection

Within each cluster, choose representative by:
1. **Centroid**: Closest to cluster mean
2. **Most probable**: Highest joint probability
3. **Simplest**: Fewest surprising results

## Narrative Generation

Convert technical conditions into readable text.

### Templates

```python
NARRATIVE_TEMPLATES = {
    'must_win': "{team} must beat {opponent}",
    'must_not_lose': "{team} cannot lose to {opponent}",
    'needs_help': "{team} needs {helper} to beat {opponent}",
    'needs_bonus': "{team} needs bonus point vs {opponent}",
    'point_threshold': "{team} needs ≥{points} total points",
}

def generate_narrative(paths_output):
    """Create human-readable summary."""

    team = paths_output.team
    position = paths_output.target_position
    prob = paths_output.probability

    narrative = [
        f"{team} can finish {ordinal(position)} with {prob:.0%} probability.",
        "",
        "Key requirements:"
    ]

    # Sort conditions by importance (conditional probability increase)
    for condition in sorted(paths_output.conditions, key=lambda c: c.importance, reverse=True):
        # Group by control
        if condition.team == team:
            narrative.append(f"  ✓ {format_condition(condition)} [{condition.frequency:.0%}]")
        else:
            narrative.append(f"  ○ {format_condition(condition)} [{condition.frequency:.0%}]")

    # Add scenario examples
    narrative.extend([
        "",
        "Example scenarios:",
    ])

    for i, scenario in enumerate(paths_output.scenario_clusters[:3], 1):
        narrative.append(f"{i}. {scenario.description} ({scenario.frequency:.0%} of successful paths)")

    return "\n".join(narrative)
```

### Example Output

```
Scotland can finish 2nd with 23% probability.

Key requirements:
  ✓ Scotland must beat Ireland [89%]
  ✓ Scotland needs bonus point vs England [67%]
  ○ France must not beat Wales with bonus [82%]
  ○ England must lose to Italy [45%]

Example scenarios:
1. Scotland wins remaining games, France loses to Wales (43% of successful paths)
2. Scotland beats Ireland, England loses to Italy, even if France wins (31%)
3. Four-way tie on points, Scotland wins on points difference (18%)

Critical upcoming matches (by impact on Scotland's chances):
1. England vs Italy (Feb 10): +15% if Italy wins
2. France vs Wales (Feb 10): +12% if Wales wins or draws
3. Scotland vs Ireland (Feb 17): +45% if Scotland wins
```

## Sankey Diagram Design

### Structure

```
Layer 1: Current position probabilities
   ↓
Layer 2-N: Critical game outcomes (only show high-impact games)
   ↓
Layer N+1: Final position probabilities
```

### Example for "Scotland finishes 2nd"

```
Current       Game 1           Game 2          Final
[Scotland:    [ENG-ITA]        [FRA-WAL]       [Position]
 3rd: 45%]
              ├→ ITA wins      ├→ WAL wins      → 2nd: 35%
              │  (0.30)        │  (0.40)
              │                └→ FRA wins      → 3rd: 15%
              │                   (0.60)
              │
              └→ ENG wins      ├→ WAL wins      → 3rd: 25%
                 (0.70)        │  (0.40)
                               └→ FRA wins      → 4th: 25%
                                  (0.60)
```

### Implementation

Use **plotly.graph_objects.Sankey** for interactive diagrams:

```python
import plotly.graph_objects as go

def create_sankey_diagram(paths_output, max_games=3):
    """Create interactive Sankey diagram."""

    # Select top N most impactful games
    critical_games = paths_output.critical_games[:max_games]

    # Build nodes
    nodes = ['Current']
    labels = ['Current state']
    colors = ['lightgray']

    # Add game outcome nodes
    for game in critical_games:
        for outcome in ['Win', 'Draw', 'Loss']:
            node_id = f"{game.home_team}-{game.away_team}-{outcome}"
            nodes.append(node_id)
            labels.append(f"{game.home_team} {outcome}")
            colors.append(get_team_color(game.home_team))

    # Add final position nodes
    for pos in range(1, 7):  # Six Nations positions
        nodes.append(f"Final-{pos}")
        labels.append(f"{ordinal(pos)} place")
        colors.append('lightgreen' if pos == paths_output.target_position else 'lightgray')

    # Build flows (source, target, value)
    flows = calculate_flows(paths_output, nodes, critical_games)

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=colors,
        ),
        link=dict(
            source=flows['source'],
            target=flows['target'],
            value=flows['value'],
            color=flows['color'],
        )
    ))

    fig.update_layout(
        title=f"Paths to {paths_output.team} finishing {ordinal(paths_output.target_position)}",
        font_size=12,
    )

    return fig
```

## API Design

### Main Interface

```python
from rugby_ranking.model.paths_to_victory import PathsAnalyzer

# Initialize
analyzer = PathsAnalyzer(
    season_predictor=predictor,
    simulations=season_pred  # From SeasonPredictor
)

# Analyze specific outcome
paths = analyzer.analyze_paths(
    team='Scotland',
    target_position=2,
    method='auto'  # 'mcmc', 'combinatorial', or 'auto'
)

# Generate outputs
print(paths.narrative)
paths.sankey_diagram.show()
paths.export_to_csv('scotland_paths_to_2nd.csv')

# Find critical games for all teams
critical_games = analyzer.find_critical_games()
# Returns: [(game, {team: ΔP}), ...] sorted by total impact
```

### Output Format

```python
@dataclass
class PathsOutput:
    team: str
    target_position: int
    probability: float

    # Conditions (sorted by importance)
    conditions: List[Condition]
    # Condition = {game, outcome, frequency, conditional_probability}

    # Game importance
    critical_games: List[Tuple[Game, float]]  # (game, ΔP)

    # Scenario clusters
    scenario_clusters: List[ScenarioCluster]
    # ScenarioCluster = {representative_sim, frequency, description}

    # Visualizations
    narrative: str
    sankey_diagram: go.Figure

    # Raw data for custom analysis
    successful_simulations: List[Simulation] | None
    all_combinations: List[Tuple[Outcome, float]] | None  # If combinatorial
```

## Performance Considerations

### MCMC Mode
- **Input**: 1000-10000 simulations
- **Processing**: Decision tree + rule extraction
- **Time**: <1 second

### Combinatorial Mode
- **Input**: All outcomes for N remaining games
- **Constraint**: Only use if <100k combinations
- **Processing**: Enumerate + weight + minimize
- **Time**: ~10 seconds for 50k combinations

### Optimization Strategies

1. **Lazy evaluation**: Only analyze requested team+position pairs
2. **Caching**: Store critical games analysis (same for all teams)
3. **Parallel processing**: Analyze multiple teams simultaneously
4. **Progressive disclosure**: Show top-level summary first, details on demand

## Testing Strategy

### Unit Tests
- `test_scenario_clustering`: Verify similar sims grouped correctly
- `test_rule_extraction`: Check decision tree rules are logical
- `test_boolean_minimization`: Validate simplified expressions
- `test_narrative_generation`: Ensure readable output

### Integration Tests
- `test_six_nations_week_1`: Early tournament (MCMC mode)
- `test_six_nations_week_5`: Late tournament (combinatorial mode)
- `test_auto_mode_switch`: Verify correct method selection

### Validation
- **Historical validation**: Run on past Six Nations tournaments
  - Did predicted "paths" match actual outcomes?
  - Were critical games correctly identified?
- **Probability calibration**: Do 23% predictions occur 23% of the time?

## Future Enhancements

1. **Time-based paths**: Show how probabilities evolve week-by-week
2. **Interactive dashboard**: User adjusts game results, sees updated paths
3. **Comparative analysis**: "Scotland's easiest path vs hardest path to 2nd"
4. **What-if scenarios**: "If Scotland beat England by 20+, what changes?"
5. **Multi-team narratives**: "The race for 1st place between Ireland and France"
6. **Expected value analysis**: Not just position, but expected total points
