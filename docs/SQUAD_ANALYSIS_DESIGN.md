# Squad Analysis & Depth - Design Document

## Overview

Analyze squad strength, depth, and likely lineups for tournaments where squads are announced but match-day teams are not yet selected (e.g., Six Nations squad announcements ~2 weeks before tournament).

## Use Cases

1. **Pre-tournament analysis**: "Ireland has the strongest back row depth in the tournament"
2. **Lineup prediction**: "Most likely Scotland XV based on squad and player ratings"
3. **Squad-based predictions**: "Scotland vs England with likely lineups: 62% Scotland win"
4. **Injury impact analysis**: "If Finn Russell is injured, Scotland's win probability drops by 15%"
5. **Depth comparison**: "France has better depth at 9/10 than any other team"
6. **Weekly content**: "Key injury concerns for Round 3" with quantified impact

## Problem Statement

Current prediction modes:
- **Teams-only**: High uncertainty, doesn't account for squad composition
- **Full-lineup**: Low uncertainty, but lineups not announced until 48h before match

**Gap**: When squads are announced but lineups aren't (weeks before tournament)

**Solution**: Analyze squad composition to:
1. Predict most likely starting XVs
2. Make informed predictions with squad-based uncertainty
3. Quantify impact of player availability
4. Compare squad depth across teams

## Architecture

### Core Components

```
squad_analysis.py
├── SquadParser           # Parse squad lists from text/Wikipedia
├── SquadAnalyzer         # Analyze squad strength and depth
├── LineupPredictor       # Predict likely starting XVs
├── InjuryImpactAnalyzer  # Quantify impact of missing players
└── SquadComparator       # Compare squads across teams
```

### Data Flow

```
Text Squad List (Wikipedia/copy-paste)
    ↓
SquadParser.parse(text, team, season)
    ↓
Squad DataFrame: [player, position, caps, age, ...]
    ↓
SquadAnalyzer.analyze(squad, model, trace)
    ↓
    ├→ Player ratings from model
    ├→ Position depth charts
    ├→ Strength scores per position
    └→ Overall squad strength
    ↓
LineupPredictor.predict_lineup(squad, opponent, ratings)
    ↓
Likely XV + Alternatives
    ↓
MatchPredictor.predict_with_squad(home_squad, away_squad)
    ↓
Prediction with lineup uncertainty
```

## Component Details

### 1. SquadParser

Parse squad lists from various text formats.

#### Input Formats

**Wikipedia format** (most common):
```
Forwards

Props
1. Andrew Porter (Leinster)
3. Tadhg Furlong (Leinster)
17. Jeremy Loughman (Munster)

Hookers
2. Dan Sheehan (Leinster)
16. Ronan Kelleher (Leinster)

...
```

**Simple list format**:
```
Andrew Porter, Leinster, Prop
Tadhg Furlong, Leinster, Prop
Dan Sheehan, Leinster, Hooker
...
```

**CSV format**:
```csv
Player,Team,Position,Caps,Age
Andrew Porter,Leinster,Prop,58,28
Tadhg Furlong,Leinster,Prop,68,32
```

#### Implementation

```python
class SquadParser:
    """Parse squad lists from various text formats."""

    def parse_text(
        self,
        text: str,
        team: str,
        season: str,
        format: Literal['auto', 'wikipedia', 'simple', 'csv'] = 'auto'
    ) -> pd.DataFrame:
        """
        Parse squad text into structured DataFrame.

        Returns:
            DataFrame with columns: player, position, club, caps (optional),
                                   age (optional), primary_position,
                                   secondary_positions
        """

    def parse_wikipedia(self, text: str) -> pd.DataFrame:
        """Parse Wikipedia-style squad list."""
        # Detect sections: "Forwards", "Backs"
        # Detect position headers: "Props", "Hookers", etc.
        # Extract player names (remove squad numbers)
        # Extract clubs (text in parentheses)

    def parse_simple(self, text: str) -> pd.DataFrame:
        """Parse simple comma/tab separated list."""

    def parse_csv(self, text: str) -> pd.DataFrame:
        """Parse CSV format."""

    def infer_positions(self, position_text: str) -> Tuple[str, List[str]]:
        """
        Infer primary and secondary positions from text.

        Examples:
            "Prop" → ("Prop", [])
            "Loosehead Prop" → ("Prop", [])
            "Flanker / Number 8" → ("Flanker", ["Number 8"])
        """

    def normalize_player_name(self, name: str) -> str:
        """
        Normalize player name for matching with model data.

        - Remove squad numbers
        - Handle nicknames
        - Standardize capitalization
        """
```

#### Interactive Input

```python
def input_squad_interactive(team: str, season: str) -> pd.DataFrame:
    """
    Interactive squad input from command line.

    Usage:
        >>> squad = input_squad_interactive("Scotland", "2024-2025")
        Paste squad list (Wikipedia format recommended).
        Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:

        [User pastes text]

        Parsed 35 players:
        - 18 forwards
        - 17 backs

        Review positions:
        1. Finn Russell → Fly-half ✓
        2. Johnny Matthews → Hooker ✓
        ...

        Accept? (y/n): y
        Squad saved to: squads/scotland_2024-2025.csv
    """
```

### 2. SquadAnalyzer

Analyze squad strength using model player ratings.

```python
class SquadAnalyzer:
    """Analyze squad strength and depth."""

    def __init__(self, model, trace, dataset):
        self.model = model
        self.trace = trace
        self.dataset = dataset

    def analyze_squad(
        self,
        squad: pd.DataFrame,
        team: str,
        season: str,
    ) -> SquadAnalysis:
        """
        Comprehensive squad analysis.

        Returns:
            SquadAnalysis with:
            - player_ratings: DataFrame of player ratings by score type
            - position_depth: Depth chart for each position
            - strength_by_position: Expected strength per position
            - overall_strength: Overall squad strength score
            - likely_xv: Most likely starting XV
            - depth_score: Squad depth score (1st vs 2nd choice drop-off)
        """

    def get_player_ratings(
        self,
        players: List[str],
        season: str,
        score_types: List[str] = ['tries', 'penalties', 'conversions']
    ) -> pd.DataFrame:
        """
        Get model ratings for list of players.

        Returns:
            DataFrame with columns: player, score_type, rating_mean,
                                   rating_std, percentile
        """
        # Use model.get_player_rankings() for players in squad
        # Handle players not in model (new caps, returning players)

    def create_depth_chart(
        self,
        squad: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Create depth chart for each position.

        Returns:
            {
                'Loosehead Prop': [('Andrew Porter', 0.85), ('Jeremy Loughman', 0.62), ...],
                'Hooker': [('Dan Sheehan', 0.91), ('Ronan Kelleher', 0.87), ...],
                ...
            }
        """

    def calculate_position_strength(
        self,
        depth_chart: Dict[str, List[Tuple[str, float]]],
    ) -> pd.DataFrame:
        """
        Calculate expected strength for each position.

        Accounts for:
        - Strongest available player
        - Depth (quality of 2nd, 3rd choice)
        - Positional coverage

        Returns:
            DataFrame with: position, first_choice_rating, second_choice_rating,
                          depth_score, expected_strength
        """

    def calculate_squad_depth_score(
        self,
        position_strength: pd.DataFrame,
    ) -> float:
        """
        Overall squad depth score.

        Measures drop-off from 1st to 2nd choice across all positions.

        Returns:
            Score 0-1 where:
            - 1.0 = perfect depth (no drop-off)
            - 0.5 = moderate depth (some drop-off)
            - 0.0 = poor depth (large drop-off)
        """
```

#### Position Groupings

Standard rugby positions with coverage rules:

```python
POSITION_GROUPS = {
    'Front Row': {
        'Loosehead Prop': {'primary': [1], 'cover': []},
        'Hooker': {'primary': [2], 'cover': []},
        'Tighthead Prop': {'primary': [3], 'cover': []},
    },
    'Second Row': {
        'Lock': {'primary': [4, 5], 'cover': []},
    },
    'Back Row': {
        'Blindside Flanker': {'primary': [6], 'cover': [7, 8]},
        'Openside Flanker': {'primary': [7], 'cover': [6]},
        'Number 8': {'primary': [8], 'cover': [6]},
    },
    'Half-backs': {
        'Scrum-half': {'primary': [9], 'cover': []},
        'Fly-half': {'primary': [10], 'cover': [12, 15]},
    },
    'Centres': {
        'Inside Centre': {'primary': [12], 'cover': [10, 13]},
        'Outside Centre': {'primary': [13], 'cover': [12, 11, 14]},
    },
    'Back Three': {
        'Wing': {'primary': [11, 14], 'cover': [15, 13]},
        'Fullback': {'primary': [15], 'cover': [10, 11, 14]},
    }
}

# Minimum squad requirements for Six Nations (typical 35-player squad)
SQUAD_REQUIREMENTS = {
    'Props': 5,  # At least 2 loosehead, 2 tighthead, 1 can play both
    'Hookers': 3,
    'Locks': 4,
    'Back Row': 5,
    'Scrum-halves': 3,
    'Fly-halves': 2,
    'Centres': 4,
    'Back Three': 5,
}
```

### 3. LineupPredictor

Predict most likely starting XVs based on squad analysis.

```python
class LineupPredictor:
    """Predict likely starting lineups from squad."""

    def predict_lineup(
        self,
        squad_analysis: SquadAnalysis,
        opponent: str | None = None,
        constraints: Dict | None = None,
    ) -> LineupPrediction:
        """
        Predict most likely starting XV.

        Args:
            squad_analysis: SquadAnalysis from SquadAnalyzer
            opponent: Opponent team (optional, for tactical considerations)
            constraints: Known lineup constraints (e.g., {'Fly-half': 'Finn Russell'})

        Returns:
            LineupPrediction with:
            - starting_xv: Most likely starting 15
            - bench: Most likely bench (8 players)
            - alternatives: Alternative lineups with probabilities
            - confidence: Confidence in prediction (0-1)
        """

    def predict_lineup_distribution(
        self,
        squad_analysis: SquadAnalysis,
        n_samples: int = 1000,
    ) -> pd.DataFrame:
        """
        Generate distribution of possible lineups.

        Uses sampling to account for:
        - Uncertain selections (close calls)
        - Positional coverage constraints
        - Form/fitness uncertainty

        Returns:
            DataFrame with: player, position, selection_probability
        """

    def select_optimal_xv(
        self,
        depth_chart: Dict,
        constraints: Dict | None = None,
    ) -> Dict[str, str]:
        """
        Select optimal XV using constraint optimization.

        Objective: Maximize total team strength
        Constraints:
        - One player per position
        - Players can only play positions they're listed for
        - Bench must provide positional cover
        - Respect user constraints

        Returns:
            {position: player} for starting XV
        """
```

### 4. SquadBasedPredictor

Make predictions using squad analysis with lineup uncertainty.

```python
class SquadBasedPredictor:
    """Make predictions accounting for lineup uncertainty."""

    def __init__(self, match_predictor: MatchPredictor):
        self.match_predictor = match_predictor

    def predict_with_squads(
        self,
        home_squad_analysis: SquadAnalysis,
        away_squad_analysis: SquadAnalysis,
        season: str,
        n_lineup_samples: int = 100,
        n_score_samples: int = 1000,
    ) -> SquadBasedPrediction:
        """
        Predict match with squad-based lineup uncertainty.

        Algorithm:
        1. Sample possible lineups from each squad (n_lineup_samples)
        2. For each lineup pair:
           - Predict match using MatchPredictor.predict_full_lineup()
           - Weight by lineup probabilities
        3. Aggregate predictions

        Returns:
            SquadBasedPrediction with:
            - expected_score: Expected score for each team
            - win_probabilities: Win/draw/loss probabilities
            - uncertainty: Additional uncertainty from lineup variation
            - key_players: Players with highest impact on outcome
        """

    def compare_lineup_scenarios(
        self,
        home_squad: SquadAnalysis,
        away_squad: SquadAnalysis,
        scenarios: List[Dict],
    ) -> pd.DataFrame:
        """
        Compare predictions under different lineup scenarios.

        Example scenarios:
        [
            {'name': 'Best available', 'home': {}, 'away': {}},
            {'name': 'Russell injured', 'home': {'Fly-half': 'Adam Hastings'}, 'away': {}},
            {'name': 'Both fly-halves injured',
             'home': {'Fly-half': 'Blair Kinghorn'},
             'away': {'Fly-half': 'George Ford'}},
        ]

        Returns:
            DataFrame comparing predictions across scenarios
        """
```

### 5. InjuryImpactAnalyzer

Quantify impact of player unavailability.

```python
class InjuryImpactAnalyzer:
    """Analyze impact of player injuries/unavailability."""

    def analyze_player_impact(
        self,
        player: str,
        squad_analysis: SquadAnalysis,
        opponent_squad: SquadAnalysis,
        season: str,
    ) -> PlayerImpact:
        """
        Quantify impact of losing a specific player.

        Returns:
            PlayerImpact with:
            - player: Player name
            - position: Primary position
            - replacement: Most likely replacement
            - rating_drop: Drop in position strength
            - win_prob_change: Δ(P(win)) if player is out
            - criticality_score: How critical player is (0-1)
        """

    def identify_critical_players(
        self,
        squad_analysis: SquadAnalysis,
        opponent_squad: SquadAnalysis,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Identify most critical players in squad.

        Returns:
            DataFrame with: player, position, win_prob_change, criticality_score
            Sorted by criticality (impact if unavailable)
        """

    def analyze_squad_robustness(
        self,
        squad_analysis: SquadAnalysis,
    ) -> SquadRobustness:
        """
        Analyze squad robustness to injuries.

        Simulates random injuries and measures impact.

        Returns:
            SquadRobustness with:
            - vulnerable_positions: Positions with worst depth
            - critical_players: Players whose loss would hurt most
            - robustness_score: Overall robustness (0-1)
            - injury_scenarios: Impact of 1, 2, 3+ injuries
        """
```

### 6. SquadComparator

Compare squads across teams for tournament analysis.

```python
class SquadComparator:
    """Compare squads across multiple teams."""

    def compare_squads(
        self,
        squad_analyses: Dict[str, SquadAnalysis],
    ) -> SquadComparison:
        """
        Compare squads across tournament.

        Returns:
            SquadComparison with:
            - overall_rankings: Teams ranked by overall squad strength
            - position_rankings: Teams ranked by position group
            - depth_rankings: Teams ranked by squad depth
            - comparative_analysis: Strengths/weaknesses per team
        """

    def create_strength_matrix(
        self,
        squad_analyses: Dict[str, SquadAnalysis],
    ) -> pd.DataFrame:
        """
        Create matrix of squad strengths by position.

        Returns:
            DataFrame with teams as rows, position groups as columns
        """

    def identify_matchup_advantages(
        self,
        team_a: str,
        team_b: str,
        squad_a: SquadAnalysis,
        squad_b: SquadAnalysis,
    ) -> MatchupAnalysis:
        """
        Identify positional advantages in head-to-head matchup.

        Returns:
            MatchupAnalysis with:
            - team_a_advantages: Positions where team A is stronger
            - team_b_advantages: Positions where team B is stronger
            - key_battles: Most impactful individual matchups
        """
```

## Data Structures

### SquadAnalysis

```python
@dataclass
class SquadAnalysis:
    """Complete squad analysis."""
    team: str
    season: str
    squad: pd.DataFrame  # Raw squad data

    # Player ratings
    player_ratings: pd.DataFrame  # player, score_type, rating_mean, rating_std

    # Depth charts
    depth_chart: Dict[str, List[Tuple[str, float]]]  # position → [(player, rating), ...]

    # Strength metrics
    position_strength: pd.DataFrame  # position, 1st_choice, 2nd_choice, depth_score
    overall_strength: float  # 0-1 score
    depth_score: float  # 0-1 score

    # Likely XV
    likely_xv: Dict[str, str]  # position → player
    likely_bench: List[str]

    # Alternatives
    selection_uncertainty: pd.DataFrame  # player, position, selection_probability
```

### LineupPrediction

```python
@dataclass
class LineupPrediction:
    """Predicted lineup."""
    starting_xv: Dict[str, str]  # position → player
    bench: List[str]

    # Alternatives for uncertain positions
    alternatives: Dict[str, List[Tuple[str, float]]]  # position → [(player, prob), ...]

    # Confidence
    confidence: float  # 0-1, how certain is this lineup
    uncertain_positions: List[str]  # Positions with <70% confidence
```

### PlayerImpact

```python
@dataclass
class PlayerImpact:
    """Impact of losing a player."""
    player: str
    position: str
    replacement: str

    # Rating impact
    rating_drop: float  # Drop in position strength

    # Match impact
    win_prob_change: float  # Δ(P(win)) if player is out
    score_change: float  # Expected Δ(score)

    # Overall criticality
    criticality_score: float  # 0-1, how critical is this player
    rank: int  # Rank among squad players
```

## User Interface

### CLI Commands

```bash
# Input squad
rugby-ranking squad input --team "Scotland" --season "2024-2025" --from-clipboard

# Or from file
rugby-ranking squad input --team "Scotland" --season "2024-2025" --file squads/scotland.txt

# Analyze squad
rugby-ranking squad analyze --team "Scotland" --season "2024-2025"

# Predict lineup
rugby-ranking squad lineup --team "Scotland" --opponent "England"

# Injury impact
rugby-ranking squad injury-impact --team "Scotland" --player "Finn Russell"

# Compare squads
rugby-ranking squad compare --tournament six-nations --season "2024-2025"

# Squad-based prediction
rugby-ranking squad predict --home "Scotland" --away "England" --season "2024-2025"
```

### Interactive Workflow

```python
# 1. Input squads for all Six Nations teams
from rugby_ranking.model.squad_analysis import SquadParser, SquadAnalyzer

parser = SquadParser()

# Paste from Wikipedia
scotland_text = """
Forwards
Props
1. Pierre Schoeman (Edinburgh)
3. Zander Fagerson (Glasgow Warriors)
...
"""

scotland_squad = parser.parse_text(scotland_text, team="Scotland", season="2024-2025")
scotland_squad.to_csv('squads/scotland_2024-2025.csv')

# 2. Analyze all squads
analyzer = SquadAnalyzer(model, trace, dataset)

squads = {}
for team in ['Scotland', 'England', 'Ireland', 'France', 'Wales', 'Italy']:
    squad_df = pd.read_csv(f'squads/{team}_2024-2025.csv')
    squads[team] = analyzer.analyze_squad(squad_df, team, "2024-2025")

# 3. Compare squads
from rugby_ranking.model.squad_analysis import SquadComparator

comparator = SquadComparator()
comparison = comparator.compare_squads(squads)

print(comparison.overall_rankings)
#      Team  Overall Strength  Depth Score  Total Rating
# 1  Ireland            0.91         0.85         145.2
# 2   France            0.89         0.82         142.8
# 3 Scotland            0.84         0.76         138.1
# ...

# 4. Injury impact analysis
from rugby_ranking.model.squad_analysis import InjuryImpactAnalyzer

injury_analyzer = InjuryImpactAnalyzer()
impact = injury_analyzer.identify_critical_players(
    squads['Scotland'],
    squads['England'],
    top_n=10
)

print(impact)
#            Player      Position  Win Prob Change  Criticality
# 1   Finn Russell      Fly-half           -0.15         0.92
# 2  Sione Tuipulotu  Inside Centre        -0.08         0.78
# 3  Duhan van der Merwe  Wing             -0.07         0.74
# ...

# 5. Squad-based prediction
from rugby_ranking.model.squad_analysis import SquadBasedPredictor

squad_predictor = SquadBasedPredictor(match_predictor)
prediction = squad_predictor.predict_with_squads(
    squads['Scotland'],
    squads['England'],
    season="2024-2025"
)

print(prediction)
# Scotland vs England (likely lineups)
# Scotland: 24.3 (±2.8)
# England:  21.7 (±2.5)
#
# Win probabilities:
#   Scotland: 62%
#   Draw: 3%
#   England: 35%
#
# Uncertainty from lineup variation: ±5% win probability
```

## Output Examples

### Squad Analysis Report

```
====================================================================
SQUAD ANALYSIS: SCOTLAND (2024-2025 Six Nations)
====================================================================

Overall Strength: 84/100 (3rd in tournament)
Squad Depth Score: 76/100 (3rd in tournament)

POSITION STRENGTHS
------------------------------------------------------------------
Position Group          1st Choice    2nd Choice    Depth Score
------------------------------------------------------------------
Props                      0.82          0.71          0.86
Hookers                    0.88          0.85          0.97
Locks                      0.79          0.74          0.94
Back Row                   0.91          0.86          0.95
Half-backs                 0.89          0.68          0.76  ⚠
Centres                    0.87          0.82          0.94
Back Three                 0.85          0.80          0.94

VULNERABLE POSITIONS
------------------------------------------------------------------
1. Fly-half: Large drop-off from Russell to Hastings (-0.21)
2. Scrum-half: Moderate depth concerns
3. Loosehead Prop: Only 2 specialist looseheads

STRONGEST POSITIONS
------------------------------------------------------------------
1. Back Row: Excellent depth across all positions
2. Centres: Multiple high-quality options
3. Hookers: Minimal drop-off to 2nd choice

MOST LIKELY STARTING XV
------------------------------------------------------------------
 1. Pierre Schoeman (LHP)
 2. George Turner (HK)
 3. Zander Fagerson (THP)
 4. Grant Gilchrist (LK)
 5. Scott Cummings (LK)
 6. Matt Fagerson (FL)
 7. Rory Darge (FL)
 8. Jack Dempsey (N8)
 9. Ben White (SH)
10. Finn Russell (FH) ★
11. Duhan van der Merwe (W) ★
12. Sione Tuipulotu (C) ★
13. Huw Jones (C)
14. Darcy Graham (W)
15. Blair Kinghorn (FB)

★ = Critical player (high impact if unavailable)

BENCH (LIKELY)
------------------------------------------------------------------
16. Johnny Matthews (HK)
17. Rory Sutherland (PR)
18. Elliot Millar-Mills (PR)
19. Richie Gray (LK)
20. Andy Christie (BR)
21. George Horne (SH)
22. Adam Hastings (FH/C)
23. Kyle Rowe (FB/W)

CRITICAL PLAYERS (Top 5)
------------------------------------------------------------------
1. Finn Russell (FH): -15% win probability if out
2. Sione Tuipulotu (C): -8% win probability if out
3. Duhan van der Merwe (W): -7% win probability if out
4. Rory Darge (FL): -6% win probability if out
5. Zander Fagerson (THP): -5% win probability if out
====================================================================
```

### Tournament Squad Comparison

```
====================================================================
SIX NATIONS 2025 - SQUAD COMPARISON
====================================================================

OVERALL RANKINGS
------------------------------------------------------------------
Rank  Team      Strength  Depth  Top XV Rating  Weakest Position
------------------------------------------------------------------
 1   Ireland     91/100   85/100    147.2        Props
 2   France      89/100   82/100    144.6        Scrum-half
 3   Scotland    84/100   76/100    138.1        Fly-half
 4   England     82/100   79/100    136.9        Centres
 5   Wales       74/100   68/100    128.4        Front Row
 6   Italy       68/100   62/100    121.7        Half-backs

POSITION GROUP RANKINGS
------------------------------------------------------------------
Front Row:  1. Ireland  2. France  3. Scotland  4. England  5. Wales  6. Italy
Second Row: 1. Ireland  2. France  3. England   4. Scotland 5. Wales  6. Italy
Back Row:   1. France   2. Ireland 3. Scotland  4. England  5. Wales  6. Italy
Half-backs: 1. Ireland  2. Scotland 3. France   4. England  5. Wales  6. Italy
Centres:    1. Ireland  2. Scotland 3. France   4. Wales    5. England 6. Italy
Back Three: 1. France   2. Ireland 3. Scotland  4. England  5. Italy   6. Wales

KEY FINDINGS
------------------------------------------------------------------
• Ireland has the most balanced squad with no major weaknesses
• France has the best back row depth in the tournament
• Scotland heavily reliant on Finn Russell at fly-half
• England has surprising depth issues in the centres
• Wales vulnerable in the front row
• Italy competitive in back three but weak in pack

UPSET POTENTIAL
------------------------------------------------------------------
Teams most vulnerable to injuries:
1. Scotland (fly-half dependency)
2. Wales (front row depth)
3. England (centre depth)
====================================================================
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Before Six Nations)
1. **SquadParser** - Parse text squads from Wikipedia/clipboard
2. **SquadAnalyzer** - Basic strength analysis using existing player ratings
3. **CLI for squad input** - Easy way to input squads
4. **Squad storage** - CSV/JSON format for squad data

### Phase 2: Analysis & Prediction (Week 1)
5. **LineupPredictor** - Predict likely starting XVs
6. **SquadBasedPredictor** - Predictions with lineup uncertainty
7. **Position depth charts** - Visualize depth at each position

### Phase 3: Impact Analysis (Week 2-3)
8. **InjuryImpactAnalyzer** - Quantify player criticality
9. **SquadComparator** - Tournament-wide comparisons
10. **Matchup analysis** - Positional advantage identification

### Phase 4: Polish & Visualization (Ongoing)
11. **Notebook examples** - Demo all functionality
12. **Export to blog format** - Markdown/HTML export
13. **Interactive visualizations** - Depth charts, strength matrices

## Testing Strategy

### Unit Tests
- Test parser on various Wikipedia formats
- Test position inference logic
- Test depth chart generation
- Test lineup optimization

### Integration Tests
- Parse actual Six Nations squads
- Verify player ratings match model
- Test end-to-end prediction workflow

### Validation
- Compare predicted lineups to actual selections
- Validate injury impact against historical data
- Check squad rankings against expert consensus

## Future Enhancements

1. **Automated squad scraping** - Auto-fetch from Wikipedia/official sites
2. **Form tracking** - Adjust ratings based on recent performances
3. **Tactical analysis** - Account for playing style in lineup prediction
4. **Fitness tracking** - Incorporate injury reports and fitness data
5. **Historical squad analysis** - Analyze past squads and selection patterns
6. **Bench impact** - Model impact of substitutions during game
