# Knockout Tournament Bracket Prediction Design

## Overview

Extends the rugby ranking model to predict knockout tournament bracket progression, including handling "TBC" (To Be Confirmed) matches where participants haven't been determined yet.

## Problem Statement

Current limitations:
- Matches with "TBC" teams are filtered out entirely (Rugby-Data/scripts/predictions.py:81)
- No way to predict knockout bracket progression
- No "paths to victory" for tournament championships
- Can't answer questions like "What's the probability Team X wins the tournament?"

## Solution Architecture

### 1. Bracket Structure Module (`bracket.py`)

Represents knockout tournament bracket structure and dependencies.

```python
@dataclass
class BracketMatch:
    """A single match in a knockout bracket."""
    round: str  # "QF1", "SF1", "Final", etc.
    round_type: Literal["quarterfinal", "semifinal", "final", "third_place"]
    home_team: str | TBD  # Could be team name or placeholder
    away_team: str | TBD
    depends_on: List[str] = []  # Match IDs this depends on
    winner_advances_to: str | None = None  # Match ID winner advances to
    date: datetime | None = None
    venue: str | None = None

@dataclass
class TBD:
    """Placeholder for to-be-determined team."""
    source: str  # "Winner of QF1", "Pool A winner", "Best runner-up", etc.
    criteria: dict  # Qualification criteria

class BracketStructure:
    """
    Models a knockout tournament bracket.

    Attributes:
        matches: Dict[str, BracketMatch] - All matches indexed by ID
        rounds: List[str] - Round names in order
        advancement_rules: Dict - How winners advance
    """

    def __init__(self, structure: dict):
        """Initialize from structure definition."""

    def get_round_matches(self, round_name: str) -> List[BracketMatch]:
        """Get all matches in a specific round."""

    def get_dependencies(self, match_id: str) -> List[str]:
        """Get all matches that must complete before this one."""

    def is_match_determined(self, match_id: str) -> bool:
        """Check if both participants are known."""

    def resolve_tbd(self, tbd: TBD, pool_standings: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Resolve a TBD placeholder to likely teams with probabilities.

        Returns:
            List of (team_name, probability) tuples
        """
```

**Standard Bracket Templates:**

```python
# European Champions Cup knockout (16 teams)
CHAMPIONS_CUP_R16_BRACKET = {
    "rounds": ["round_of_16", "quarterfinal", "semifinal", "final"],
    "matches": {
        "R16_1": {"home": TBD("Pool 1st #1"), "away": TBD("Pool 2nd #8")},
        "R16_2": {"home": TBD("Pool 1st #2"), "away": TBD("Pool 2nd #7")},
        # ... 8 R16 matches
        "QF1": {"home": TBD("Winner R16_1"), "away": TBD("Winner R16_2")},
        # ... 4 QF matches
        "SF1": {"home": TBD("Winner QF1"), "away": TBD("Winner QF2")},
        "SF2": {"home": TBD("Winner QF3"), "away": TBD("Winner QF4")},
        "Final": {"home": TBD("Winner SF1"), "away": TBD("Winner SF2")}
    }
}

# Rugby World Cup knockout (8 teams)
WORLD_CUP_BRACKET = {
    "rounds": ["quarterfinal", "semifinal", "final", "third_place"],
    "matches": {
        "QF1": {"home": TBD("Pool A winner"), "away": TBD("Pool B runner-up")},
        "QF2": {"home": TBD("Pool C winner"), "away": TBD("Pool D runner-up")},
        "QF3": {"home": TBD("Pool B winner"), "away": TBD("Pool A runner-up")},
        "QF4": {"home": TBD("Pool D winner"), "away": TBD("Pool C runner-up")},
        "SF1": {"home": TBD("Winner QF1"), "away": TBD("Winner QF2")},
        "SF2": {"home": TBD("Winner QF3"), "away": TBD("Winner QF4")},
        "Final": {"home": TBD("Winner SF1"), "away": TBD("Winner SF2")},
        "Third": {"home": TBD("Loser SF1"), "away": TBD("Loser SF2")}
    }
}
```

### 2. Bracket Prediction Module (`bracket_predictor.py`)

Uses Monte Carlo simulation to predict bracket progression.

```python
@dataclass
class BracketPrediction:
    """Predicted bracket outcomes."""
    bracket: BracketStructure

    # Team advancement probabilities
    advancement_probs: pd.DataFrame  # P(team reaches each round)
    # Columns: team, quarterfinal_prob, semifinal_prob, final_prob, champion_prob

    # Match outcome probabilities for TBC matches
    match_probabilities: Dict[str, pd.DataFrame]
    # Key: match_id, Value: DataFrame with (home_team, away_team, home_win_prob)

    # Likely bracket paths
    modal_bracket: BracketStructure  # Most likely complete bracket

    # Simulation details
    n_simulations: int
    _simulation_results: List[Dict] | None = None

class BracketPredictor:
    """
    Predict knockout tournament bracket progression.

    Usage:
        >>> predictor = BracketPredictor(match_predictor, bracket_structure)
        >>> # After pool stage completes
        >>> prediction = predictor.predict_bracket(
        ...     pool_standings=pool_df,
        ...     completed_matches=completed_df,
        ...     n_simulations=10000
        ... )
        >>> print(prediction.advancement_probs)
        >>> print(prediction.match_probabilities['Final'])
    """

    def __init__(
        self,
        match_predictor: MatchPredictor,
        bracket_structure: BracketStructure,
    ):
        self.match_predictor = match_predictor
        self.bracket = bracket_structure

    def predict_bracket(
        self,
        pool_standings: pd.DataFrame | None = None,
        completed_knockout_matches: pd.DataFrame | None = None,
        n_simulations: int = 10000,
    ) -> BracketPrediction:
        """
        Simulate bracket progression.

        Args:
            pool_standings: Current or final pool standings (for resolving TBD)
            completed_knockout_matches: Already-played knockout matches
            n_simulations: Number of bracket simulations to run

        Returns:
            BracketPrediction with advancement probabilities and likely matchups
        """

    def _simulate_single_bracket(
        self,
        resolved_bracket: BracketStructure,
    ) -> Dict[str, str]:
        """
        Simulate one complete bracket progression.

        Returns:
            Dict mapping match_id to winner team name
        """

    def _resolve_tbd_participants(
        self,
        pool_standings: pd.DataFrame,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        For each TBD placeholder, get likely teams and probabilities.

        Returns:
            Dict mapping TBD source to [(team, prob), ...]
        """

    def get_likely_matchup(
        self,
        match_id: str,
        threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Get most likely participants for a specific match.

        Args:
            match_id: Match to query
            threshold: Only return teams with P > threshold

        Returns:
            DataFrame with columns: home_team, away_team, probability, home_win_prob
        """
```

**Simulation Algorithm:**

```python
def _simulate_single_bracket(self):
    """
    1. For each TBD participant:
       - Sample team based on advancement probabilities

    2. Progress through rounds in order:
       - For each match with known participants:
         a. Use match_predictor to get win probabilities
         b. Sample winner from distribution
         c. Update bracket with winner
         d. Resolve downstream TBD references

    3. Track:
       - Which teams reached each round
       - Match outcomes
       - Final champion

    4. Return simulation result dict
    """
```

### 3. Integration with Paths to Victory

Extend `PathsAnalyzer` to handle tournament brackets.

```python
class TournamentPathsAnalyzer(PathsAnalyzer):
    """
    Extended paths analyzer for knockout tournaments.

    Shows paths to:
    - Reaching specific rounds (semifinal, final)
    - Winning the tournament
    - Favorable bracket matchups
    """

    def analyze_tournament_paths(
        self,
        team: str,
        target: Literal["semifinal", "final", "champion"],
        bracket_prediction: BracketPrediction,
    ) -> PathsOutput:
        """
        Analyze paths for team to reach target round/outcome.

        Returns narrative like:
        "France can win the tournament with 18% probability.

        Path to victory:
          Quarterfinal: Must beat Ireland (72% likely)
          Semifinal: Most likely opponent is South Africa (45%)
                     Win probability vs SA: 38%
                     Alternative: New Zealand (35%), win prob 41%
          Final: Most likely opponents:
                 - Ireland (28%): Win probability 35%
                 - England (22%): Win probability 52%

        Critical matches:
          1. QF vs Ireland: +72% to reach SF (must win)
          2. Pool match vs Italy: Winning increases chance of
             favorable QF matchup from 35% to 68%
        "
        """
```

**Path Analysis for Tournaments:**

1. **Direct paths** - Team's own matches:
   - Must win QF to advance
   - Must win SF to reach final
   - Must win final to be champion

2. **Indirect paths** - Other matches affecting bracket:
   - Pool results affecting QF seeding/matchups
   - Parallel bracket results affecting likely SF/Final opponents
   - "Draw favorability" - probability of getting easier matchups

3. **Conditional probabilities**:
   - P(win tournament | beat QF opponent A) vs P(win | beat opponent B)
   - Impact of pool results on tournament win probability

### 4. TBC Resolution Module

Helper functions for resolving TBC placeholders.

```python
class TBDResolver:
    """Resolves TBD placeholders to likely teams."""

    def __init__(self, season_predictor: SeasonPredictor):
        self.predictor = season_predictor

    def resolve_pool_qualification(
        self,
        tbd: TBD,
        pool_standings: pd.DataFrame | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Resolve TBD based on pool qualification rules.

        Examples:
            "Pool A winner" → Use current/predicted standings
            "Best runner-up" → Compare runner-up points across pools
            "Pool 1st #3" → 3rd ranked pool winner by points
        """

    def resolve_match_progression(
        self,
        tbd: TBD,
        bracket_simulation: Dict,
    ) -> str:
        """
        Resolve TBD based on earlier bracket results.

        Examples:
            "Winner QF1" → Look up QF1 result in simulation
            "Loser SF2" → Look up SF2 loser in simulation
        """
```

### 5. Data Format Updates

Update Rugby-Data JSON format to include bracket structure:

```json
{
  "competition": "European Champions Cup",
  "season": "2024-2025",
  "stages": [
    {
      "name": "pool",
      "type": "league",
      "matches": [...]
    },
    {
      "name": "knockout",
      "type": "bracket",
      "structure": {
        "rounds": ["round_of_16", "quarterfinal", "semifinal", "final"],
        "matches": [
          {
            "id": "QF1",
            "round": "quarterfinal",
            "home": {"team": "TBC", "source": "Pool winner #1"},
            "away": {"team": "TBC", "source": "Pool runner-up #8"},
            "date": "2025-04-05",
            "venue": "TBC"
          }
        ]
      },
      "matches": [...]
    }
  ]
}
```

## Implementation Plan

### Phase 1: Core Bracket Structure ✓ (Design complete)
- [ ] Implement `BracketMatch` and `TBD` dataclasses
- [ ] Implement `BracketStructure` class
- [ ] Create standard bracket templates (Champions Cup, World Cup, URC playoffs)
- [ ] Add tests for bracket structure

### Phase 2: Bracket Prediction
- [ ] Implement `BracketPredictor` class
- [ ] Implement `_simulate_single_bracket()` method
- [ ] Implement TBD resolution from pool standings
- [ ] Calculate advancement probabilities
- [ ] Add tests for bracket simulation

### Phase 3: TBC Match Handling
- [ ] Implement `TBDResolver` class
- [ ] Add logic to match pool qualification criteria
- [ ] Handle different qualification scenarios (best runners-up, etc.)
- [ ] Update data loading to parse TBC information

### Phase 4: Paths to Victory Integration
- [ ] Extend `PathsAnalyzer` for tournament paths
- [ ] Implement `analyze_tournament_paths()` method
- [ ] Generate narratives for tournament paths
- [ ] Add visualization for bracket progression

### Phase 5: Examples & Documentation
- [ ] Create example notebook for Champions Cup prediction
- [ ] Create example notebook for World Cup prediction
- [ ] Document bracket structure format
- [ ] Add usage guide

## Usage Examples

### Example 1: Predict Champions Cup Knockout

```python
from rugby_ranking.model import MatchPredictor, BracketPredictor
from rugby_ranking.model.bracket import BracketStructure, CHAMPIONS_CUP_R16_BRACKET

# Fit model on historical data
predictor = MatchPredictor(...)
predictor.fit(historical_matches)

# Load pool standings after pool stage completes
pool_standings = pd.read_csv("champions_cup_2025_pool_standings.csv")

# Create bracket predictor
bracket = BracketStructure(CHAMPIONS_CUP_R16_BRACKET)
bracket_predictor = BracketPredictor(predictor, bracket)

# Predict bracket
prediction = bracket_predictor.predict_bracket(
    pool_standings=pool_standings,
    n_simulations=10000
)

# View advancement probabilities
print(prediction.advancement_probs)
#          team  quarterfinal  semifinal     final  champion
# 0     Leinster         0.95       0.72      0.45      0.23
# 1   Toulouse          0.98       0.68      0.38      0.18
# 2   Saracens         0.88       0.51      0.22      0.09

# View likely final matchups
print(prediction.get_likely_matchup("Final"))
#    home_team  away_team  probability  home_win_prob
# 0   Leinster  Toulouse         0.18           0.48
# 1   Leinster  Saracens         0.12           0.62
# 2  Toulouse   Saracens         0.09           0.54
```

### Example 2: Paths to Tournament Victory

```python
from rugby_ranking.model import TournamentPathsAnalyzer

analyzer = TournamentPathsAnalyzer(prediction, predictor)
paths = analyzer.analyze_tournament_paths(
    team="Leinster",
    target="champion",
    bracket_prediction=prediction
)

print(paths.narrative)
# Leinster can win the tournament with 23% probability.
#
# Quarterfinal: Must beat La Rochelle (82% likely)
# Semifinal: Most likely opponent Toulouse (45%)
#            Win probability vs Toulouse: 48%
# Final: Most likely opponents:
#        - Saracens (31%): Win probability 62%
#        - Toulouse (28%): Win probability 48%
#
# Critical factors:
#   ✓ Home advantage in QF increases win prob by 18%
#   ○ If Toulouse loses SF, easier final matchup (+12% win prob)

# View critical games
for game, impact in paths.critical_games[:5]:
    print(f"{game[0]} vs {game[1]}: ΔP = {impact:.1%}")
```

### Example 3: Update Predictions as Tournament Progresses

```python
# After quarterfinals complete
completed_qf = pd.DataFrame([
    {"match_id": "QF1", "winner": "Leinster"},
    {"match_id": "QF2", "winner": "Toulouse"},
    {"match_id": "QF3", "winner": "Saracens"},
    {"match_id": "QF4", "winner": "Northampton"},
])

# Re-predict with updated bracket
updated_prediction = bracket_predictor.predict_bracket(
    completed_knockout_matches=completed_qf,
    n_simulations=10000
)

# Now semifinals are determined, get predictions
print(updated_prediction.match_probabilities["SF1"])
#  home_team away_team  home_win_prob
# 0  Leinster  Toulouse          0.522

print(updated_prediction.match_probabilities["SF2"])
#    home_team     away_team  home_win_prob
# 0  Saracens  Northampton          0.681
```

## Technical Considerations

### Monte Carlo Sample Size
- Pool qualification (many games): 1,000-5,000 simulations sufficient
- Bracket progression (fewer games): 10,000-50,000 simulations for stable probabilities
- Trade-off between accuracy and compute time

### Handling Home Advantage in Knockout
- European competitions: Higher seed gets home advantage
- World Cup: Neutral venues
- Need to incorporate home/away into bracket structure

### Seeding Rules
- Champions Cup: Pool position determines R16 seeding
- URC Playoffs: 1st plays 8th, 2nd plays 7th, etc.
- Need flexible seeding system in bracket structure

### Performance Optimization
- Cache match predictions (same matchup predicted multiple times)
- Parallelize simulations across cores
- Early stopping if probabilities converge

### Edge Cases
- Ties in pool standings (tiebreaker rules)
- Teams withdrawing from competition
- Matches postponed/rescheduled
- Home venue changes

## Future Enhancements

1. **Real-time Updates**: Automatically update predictions as matches complete
2. **Bracket Visualization**: Interactive bracket diagrams with probabilities
3. **Historical Validation**: Backtest predictions on past tournaments
4. **Multi-Stage Tournaments**: Handle complex formats (pools → playoffs → knockout)
5. **Betting Integration**: Compare model odds to bookmaker odds
6. **Impact Analysis**: "If Team X wins this match, how does it affect Team Y's chances?"

## References

- Current season predictor: `rugby_ranking/model/season_predictor.py`
- Paths to victory framework: `rugby_ranking/model/paths_to_victory.py`
- TBC filtering example: `Rugby-Data/scripts/predictions.py:81`
- Tournament structure: `Rugby-Data/rugby/tournament.py`
