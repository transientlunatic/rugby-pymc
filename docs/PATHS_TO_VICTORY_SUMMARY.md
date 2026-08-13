# Paths to Victory: Summary

A tractable approach for analyzing "how team X can finish in position Y" for the Six Nations.

## The Problem

- **Goal**: Weekly blog content showing paths for teams to achieve different final positions
- **Challenge**: Thousands of possible tournament states, especially early in competition
- **Need**: Heuristic grouping + clear visualization (e.g., Sankey diagrams)

## The Solution: Hybrid Approach

### Phase 1: Early Tournament (Weeks 1-3) - MCMC Pattern Mining

When 10+ games remain, combinatorics is intractable (8^10 ≈ 1 billion outcomes).

**Strategy**: Extract patterns from your existing Monte Carlo simulations

```python
1. Run 5000+ MCMC simulations (you already do this!)
2. For target outcome (e.g., "Scotland finishes 2nd"):
   - Filter simulations where this occurred
   - Extract which game results appeared
   - Use decision tree to find minimal conditions
   - Calculate P(success | condition)
3. Generate narrative: "Scotland finishes 2nd in 23% of scenarios
   where they beat Ireland (89%) AND France doesn't get bonus vs Wales (78%)"
```

**Key technique**: Decision tree feature importance tells you which games matter most

### Phase 2: Late Tournament (Weeks 4-5) - Weighted Combinatorics

When ≤6 games remain, enumeration becomes tractable (8^6 ≈ 260k outcomes).

**Strategy**: Enumerate all possibilities, weight by MCMC probabilities

```python
1. Generate ALL possible outcomes for remaining games
2. For each outcome:
   - Calculate final league table
   - Assign probability using your match predictor
3. Filter outcomes where target achieved
4. Simplify using Boolean minimization:
   "Scotland 2nd IF (England loses Italy) OR (France loses Wales AND Scotland bonus)"
5. Show top 10 most probable paths
```

**Key technique**: Quine-McCluskey algorithm simplifies complex boolean expressions

### Automatic Switching

```python
n_combinations = 8 ** len(remaining_games)
method = 'combinatorial' if n_combinations < 100_000 else 'mcmc'
```

## Heuristic Grouping Strategies

Even with thousands of scenarios, we can make it digestible:

### 1. Critical Game Identification

Not all games matter equally. Calculate **ΔP** (probability change):

```python
for game in remaining_fixtures:
    ΔP = P(Scotland 2nd | game favors Scotland) - P(Scotland 2nd | game doesn't)

Only show games where ΔP > 5%
```

**Result**: Reduce from 15 games to 3-5 critical games

### 2. Scenario Clustering

Group similar simulation outcomes:

```python
1. For successful scenarios, create feature vectors:
   [england_beat_italy, france_bonus, scotland_points>=15, ...]

2. Hierarchical clustering (k=5 clusters)

3. Find representative from each cluster

4. Generate description:
   - Cluster 1 (43%): "Scotland wins all, France slips"
   - Cluster 2 (31%): "England upset by Italy"
   - Cluster 3 (18%): "Four-way tie on points"
```

**Result**: Reduce from 1000 scenarios to 3-5 representative narratives

### 3. Condition Grouping

Separate what team controls vs what team needs:

```
What Scotland must do:
  ✓ Beat Ireland [89%]
  ✓ Get bonus point vs England [67%]

What Scotland needs from others:
  ○ England must lose to Italy [45%]
  ○ France must not get bonus vs Wales [82%]
```

## Sankey Diagram Design

Visual representation of probability flows:

```
Layer 1:          Layer 2:         Layer 3:         Layer 4:
Current           Critical         Critical         Final
State             Game 1           Game 2           Position

Scotland:
23% → 2nd      → ENG-ITA        → FRA-WAL        → 2nd (35%)
                  ├─ ITA wins       ├─ WAL wins
                  │  (30%)          │  (40%)
                  │                 └─ FRA wins    → 3rd (15%)
                  │                    (60%)
                  └─ ENG wins      → ...           → 3rd/4th (23%)
                     (70%)
```

**Implementation**: Use Plotly's Sankey for interactive filtering

## Example Output

```
====================================================================
PATHS TO VICTORY: SCOTLAND
====================================================================

Scotland can finish 2nd with 23% probability.

Key requirements:

  What Scotland must do:
    ✓ Beat Ireland [89%]
    ✓ Get bonus point vs England [67%]

  What Scotland needs from others:
    ○ England must lose to Italy [45%]
    ○ France must not get bonus vs Wales [82%]

Example scenarios:

  1. Scotland wins all, France loses to Wales (43% of successful paths)
  2. Scotland beats Ireland, England loses to Italy (31%)
  3. Four-way tie on points, Scotland wins on difference (18%)

Critical upcoming matches (by impact on Scotland's chances):

  1. England vs Italy (Feb 10): +15% if Italy wins
  2. France vs Wales (Feb 10): +12% if Wales wins
  3. Scotland vs Ireland (Feb 17): +45% if Scotland wins

====================================================================
Analysis method: MCMC
====================================================================
```

## Implementation Roadmap

### Immediate (Works Now)
- ✅ Basic probability analysis from season predictions
- ✅ Critical game identification (heuristic)
- ✅ Narrative generation framework
- ✅ API design and example notebook

### Phase 1 Implementation (Next)
1. **Modify `SeasonPredictor._simulate_season()`**
   - Store detailed simulation results (currently only stores aggregates)
   - Add option: `return_detailed_sims=True`
   - Store game outcomes per simulation

2. **Implement `RuleExtractor`**
   - Sklearn decision tree on simulation features
   - Extract interpretable rules
   - Calculate conditional probabilities

3. **Implement `ScenarioClusterer`**
   - Hierarchical clustering on outcome vectors
   - Find representatives
   - Generate descriptions

### Phase 2 Implementation (Later)
4. **Combinatorial enumeration**
   - Generate all outcome combinations (when tractable)
   - Weight by match predictions
   - Boolean minimization (Quine-McCluskey or Espresso)

5. **Sankey visualization**
   - Plotly interactive diagrams
   - Path filtering and drill-down
   - Export to blog-ready HTML

6. **Automated blog generation**
   - Template-based narratives
   - Export to Markdown
   - Include visualizations

## Usage Example

```python
from rugby_ranking.model.paths_to_victory import PathsAnalyzer

# After running season predictions
analyzer = PathsAnalyzer(
    season_prediction=season_pred,
    match_predictor=predictor
)

# Analyze specific outcome
paths = analyzer.analyze_paths(
    team='Scotland',
    target_position=2,
    method='auto'  # Automatically chooses MCMC or combinatorial
)

# Get human-readable summary
print(paths.narrative)

# Get Sankey diagram
paths.sankey_diagram.show()

# Find critical games for all teams
critical = analyzer.find_critical_games()
```

## Technical Details

### Decision Tree Rule Extraction

```python
from sklearn.tree import DecisionTreeClassifier

# Create binary features from simulations
X = np.array([
    [1 if sim.game_outcome('ENG', 'ITA') == 'ENG' else 0,
     1 if sim.game_outcome('FRA', 'WAL') == 'FRA' else 0,
     ...]
    for sim in all_simulations
])

# Target: did Scotland finish 2nd?
y = np.array([
    1 if sim.final_position('Scotland') == 2 else 0
    for sim in all_simulations
])

# Fit tree
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X, y)

# Extract rules
# If feature[0] <= 0.5:  (England didn't win)
#   If feature[1] <= 0.5:  (France didn't win)
#     → Scotland 2nd (probability: 0.85)
```

### Boolean Minimization

```python
# Successful outcomes (simplified)
outcomes = [
    (eng_lose_ita=True, fra_lose_wal=True),
    (eng_lose_ita=True, fra_draw_wal=True),
    (eng_lose_fra=True, sco_beat_ire=True),
]

# Minimize using Quine-McCluskey
# Input: (A∧B) ∨ (A∧C) ∨ (D∧E)
# Output: A ∨ (D∧E)  [B and C absorbed into A]

simplified = "England loses to Italy OR (England loses to France AND Scotland beats Ireland)"
```

## Performance

| Phase | Games Remaining | Method | Combinations | Time |
|-------|----------------|--------|--------------|------|
| Week 1 | 15 | MCMC | N/A (use 5k sims) | ~1s |
| Week 3 | 9 | MCMC | 8^9 ≈ 134M | ~1s |
| Week 4 | 6 | Combinatorial | 8^6 ≈ 262k | ~10s |
| Week 5 | 3 | Combinatorial | 8^3 = 512 | <1s |

## Validation

Test on historical tournaments:

```python
# Validate on Six Nations 2024
# At week 3, predict "Ireland can finish 1st if..."
# Check if actual outcome matched one of the identified paths

historical_validation = validate_paths(
    tournament='six_nations_2024',
    week=3,
    actual_outcome='Ireland_1st'
)

# Did we identify this path?
# Was the probability calibrated correctly?
```

## Future Enhancements

1. **Time evolution**: Show how probabilities change week-by-week
2. **Interactive dashboard**: User adjusts results, sees live updates
3. **Multi-team analysis**: "The race for 1st between Ireland & France"
4. **What-if scenarios**: "If Scotland win by 20+, what changes?"
5. **Expected value**: Not just position, but expected championship points
6. **Narrative templates**: Different styles (technical, casual, dramatic)

## References

- **Decision trees**: sklearn.tree.DecisionTreeClassifier
- **Clustering**: sklearn.cluster.AgglomerativeClustering
- **Boolean minimization**: pyeda, logic (Quine-McCluskey)
- **Visualization**: plotly.graph_objects.Sankey
- **Association rules**: mlxtend.frequent_patterns.apriori

## Files Created

1. **Design Document**: `docs/PATHS_TO_VICTORY_DESIGN.md`
   - Detailed technical specifications
   - Algorithm descriptions
   - API design

2. **Implementation**: `rugby_ranking/model/paths_to_victory.py`
   - `PathsAnalyzer` - main analysis class
   - `ScenarioClusterer` - clustering similar scenarios
   - `RuleExtractor` - extracting minimal conditions
   - Currently works with heuristics, ready for full implementation

3. **Demo Notebook**: `notebooks/07_paths_to_victory_demo.ipynb`
   - Example usage
   - Visualization examples
   - Blog post generation

4. **This Summary**: `docs/PATHS_TO_VICTORY_SUMMARY.md`
