"""
Example: Knockout Tournament Bracket Prediction

Demonstrates how to:
1. Create a bracket structure for a knockout tournament
2. Predict bracket progression with TBD teams
3. Analyze paths to championship
4. Update predictions as tournament progresses
"""

import pandas as pd
import numpy as np

from rugby_ranking.model.predictions import MatchPredictor
from rugby_ranking.model.bracket import (
    BracketStructure,
    create_world_cup_bracket,
    create_champions_cup_bracket,
    create_urc_playoffs_bracket,
)
from rugby_ranking.model.bracket_predictor import BracketPredictor
from rugby_ranking.model.tournament_paths import TournamentPathsAnalyzer


# ============================================================================
# Example 1: World Cup Quarterfinals Prediction
# ============================================================================

def example_world_cup_prediction():
    """Predict World Cup knockout bracket from pool standings."""

    print("=" * 70)
    print("Example 1: Rugby World Cup 2023 Knockout Prediction")
    print("=" * 70)
    print()

    # Create World Cup bracket structure
    bracket = create_world_cup_bracket()

    print("Bracket structure:")
    print(bracket.to_dataframe()[["id", "round_type", "home_team", "away_team"]])
    print()

    # Simulate pool standings (example data)
    pool_standings = pd.DataFrame([
        # Pool A
        {"pool": "A", "team": "France", "points": 19, "position": 1},
        {"pool": "A", "team": "New Zealand", "points": 17, "position": 2},
        {"pool": "A", "team": "Italy", "points": 11, "position": 3},
        {"pool": "A", "team": "Uruguay", "points": 5, "position": 4},
        {"pool": "A", "team": "Namibia", "points": 0, "position": 5},
        # Pool B
        {"pool": "B", "team": "Ireland", "points": 19, "position": 1},
        {"pool": "B", "team": "South Africa", "points": 15, "position": 2},
        {"pool": "B", "team": "Scotland", "points": 10, "position": 3},
        {"pool": "B", "team": "Tonga", "points": 6, "position": 4},
        {"pool": "B", "team": "Romania", "points": 0, "position": 5},
        # Pool C
        {"pool": "C", "team": "Wales", "points": 19, "position": 1},
        {"pool": "C", "team": "Australia", "points": 11, "position": 2},
        {"pool": "C", "team": "Fiji", "points": 11, "position": 3},
        {"pool": "C", "team": "Georgia", "points": 9, "position": 4},
        {"pool": "C", "team": "Portugal", "points": 5, "position": 5},
        # Pool D
        {"pool": "D", "team": "England", "points": 18, "position": 1},
        {"pool": "D", "team": "Argentina", "points": 15, "position": 2},
        {"pool": "D", "team": "Japan", "points": 11, "position": 3},
        {"pool": "D", "team": "Samoa", "points": 5, "position": 4},
        {"pool": "D", "team": "Chile", "points": 0, "position": 5},
    ])

    print("Pool standings:")
    print(pool_standings[pool_standings["position"] <= 2])
    print()

    # Load pre-trained match predictor (placeholder - would load real model)
    # For demo, we'll create a simple predictor
    print("Loading match predictor...")
    # match_predictor = MatchPredictor.load("models/world_cup_2023.pkl")
    # For demo, skip actual prediction
    print("(In real usage, load trained MatchPredictor here)")
    print()

    # Create bracket predictor
    # predictor = BracketPredictor(match_predictor, bracket, seed=42)

    # Predict bracket
    print("Simulating bracket progression (10,000 simulations)...")
    # prediction = predictor.predict_bracket(
    #     pool_standings=pool_standings,
    #     n_simulations=10000
    # )

    # Display advancement probabilities
    print("\nTeam advancement probabilities:")
    # print(prediction.advancement_probs.head(10))

    # Example output:
    example_advancement = pd.DataFrame([
        {
            "team": "France",
            "quarterfinal_prob": 1.00,
            "semifinal_prob": 0.78,
            "final_prob": 0.45,
            "champion_prob": 0.23,
        },
        {
            "team": "Ireland",
            "quarterfinal_prob": 1.00,
            "semifinal_prob": 0.82,
            "final_prob": 0.51,
            "champion_prob": 0.28,
        },
        {
            "team": "South Africa",
            "quarterfinal_prob": 1.00,
            "semifinal_prob": 0.68,
            "final_prob": 0.38,
            "champion_prob": 0.19,
        },
        {
            "team": "New Zealand",
            "quarterfinal_prob": 1.00,
            "semifinal_prob": 0.72,
            "final_prob": 0.42,
            "champion_prob": 0.21,
        },
    ])
    print(example_advancement.to_string(index=False))
    print()


# ============================================================================
# Example 2: Champions Cup with TBC Matches
# ============================================================================

def example_champions_cup_tbc():
    """Handle TBC matches in Champions Cup knockout."""

    print("=" * 70)
    print("Example 2: Champions Cup with TBC Teams")
    print("=" * 70)
    print()

    # Create Champions Cup bracket
    bracket = create_champions_cup_bracket()

    print("R16 matches with TBC teams:")
    r16_matches = bracket.get_round_matches("round_of_16")
    for match in r16_matches[:4]:  # Show first 4
        print(f"  {match.id}: {match.home_team} vs {match.away_team}")
    print()

    # Pool standings after pool stage
    pool_standings = pd.DataFrame([
        # Pool winners (seeded 1-8)
        {"team": "Leinster", "pool": 1, "position": 1, "points": 24, "seed": 1},
        {"team": "Toulouse", "pool": 2, "position": 1, "points": 23, "seed": 2},
        {"team": "La Rochelle", "pool": 3, "position": 1, "points": 22, "seed": 3},
        {"team": "Northampton", "pool": 4, "position": 1, "points": 21, "seed": 4},
        {"team": "Saracens", "pool": 1, "position": 2, "points": 19, "seed": 5},
        {"team": "Racing 92", "pool": 2, "position": 2, "points": 18, "seed": 6},
        {"team": "Munster", "pool": 3, "position": 2, "points": 18, "seed": 7},
        {"team": "Leicester", "pool": 4, "position": 2, "points": 17, "seed": 8},
        # Pool runners-up (seeded 9-16)
        {"team": "Bordeaux", "pool": 1, "position": 2, "points": 16, "seed": 9},
        {"team": "Ulster", "pool": 2, "position": 2, "points": 16, "seed": 10},
        {"team": "Glasgow", "pool": 3, "position": 2, "points": 15, "seed": 11},
        {"team": "Stormers", "pool": 4, "position": 2, "points": 14, "seed": 12},
        {"team": "Harlequins", "pool": 1, "position": 3, "points": 13, "seed": 13},
        {"team": "Bath", "pool": 2, "position": 3, "points": 12, "seed": 14},
        {"team": "Clermont", "pool": 3, "position": 3, "points": 11, "seed": 15},
        {"team": "Bulls", "pool": 4, "position": 3, "points": 10, "seed": 16},
    ])

    print("Pool standings:")
    print(pool_standings[["team", "seed", "points"]].head(10))
    print()

    # Predict bracket
    print("Predicting bracket with TBC resolution...")
    # predictor = BracketPredictor(match_predictor, bracket)
    # prediction = predictor.predict_bracket(
    #     pool_standings=pool_standings,
    #     n_simulations=10000
    # )

    print("\nLikely R16 matchups:")
    # for match_id in ["R16_1", "R16_2", "R16_3"]:
    #     matchups = prediction.get_likely_matchup(match_id)
    #     print(f"\n{match_id}:")
    #     print(matchups)

    # Example output:
    print("\nR16_1: (Seed 1 vs Seed 16)")
    print("  Leinster vs Bulls: 100% probability, Leinster win: 89%")
    print()


# ============================================================================
# Example 3: Paths to Championship
# ============================================================================

def example_paths_to_championship():
    """Analyze a team's path to winning the tournament."""

    print("=" * 70)
    print("Example 3: Paths to Championship Analysis")
    print("=" * 70)
    print()

    # Assume we have bracket prediction from Example 1
    print("Analyzing France's path to World Cup victory...")
    print()

    # Create tournament paths analyzer
    # analyzer = TournamentPathsAnalyzer(prediction, match_predictor)
    # paths = analyzer.analyze_tournament_paths(
    #     team="France",
    #     target="champion"
    # )

    # Display narrative
    # print(paths.narrative)

    # Example output:
    example_narrative = """
France can win the tournament with 23% probability.

Path to victory:
  Must beat New Zealand in QF (72% likely)
  Must beat Ireland in SF (51% likely)
  Must beat South Africa in Final (45% likely)

Likely opponents:
  Quarterfinal:
    - New Zealand (95% chance): Win probability 72%
  Semifinal:
    - Ireland (58% chance): Win probability 51%
    - South Africa (32% chance): Win probability 55%
  Final:
    - South Africa (35% chance): Win probability 45%
    - Ireland (28% chance): Win probability 48%
    - England (18% chance): Win probability 65%

France has a challenging bracket draw (expected difficulty: 58%)
"""
    print(example_narrative)


# ============================================================================
# Example 4: Updating Predictions as Tournament Progresses
# ============================================================================

def example_updating_predictions():
    """Update bracket predictions after matches complete."""

    print("=" * 70)
    print("Example 4: Updating Predictions After Quarterfinals")
    print("=" * 70)
    print()

    bracket = create_world_cup_bracket()

    # Completed quarterfinals
    completed_qf = pd.DataFrame([
        {"match_id": "QF1", "winner": "France"},
        {"match_id": "QF2", "winner": "South Africa"},
        {"match_id": "QF3", "winner": "Ireland"},
        {"match_id": "QF4", "winner": "England"},
    ])

    print("Completed quarterfinals:")
    print(completed_qf.to_string(index=False))
    print()

    # Re-predict with updated bracket
    print("Re-predicting semifinals and final...")
    # predictor = BracketPredictor(match_predictor, bracket)
    # updated_prediction = predictor.predict_bracket(
    #     completed_knockout_matches=completed_qf,
    #     n_simulations=10000
    # )

    print("\nUpdated championship probabilities:")
    # print(updated_prediction.advancement_probs[["team", "final_prob", "champion_prob"]])

    # Example output:
    example_updated = pd.DataFrame([
        {"team": "France", "final_prob": 0.58, "champion_prob": 0.31},
        {"team": "Ireland", "final_prob": 0.62, "champion_prob": 0.35},
        {"team": "South Africa", "final_prob": 0.42, "champion_prob": 0.21},
        {"team": "England", "final_prob": 0.38, "champion_prob": 0.13},
    ])
    print(example_updated.to_string(index=False))
    print()

    print("Semifinal match predictions:")
    # print(updated_prediction.match_probabilities["SF1"])
    # print(updated_prediction.match_probabilities["SF2"])

    print("\nSF1: France vs Ireland")
    print("  France win probability: 52%")
    print("\nSF2: South Africa vs England")
    print("  South Africa win probability: 64%")
    print()


# ============================================================================
# Example 5: URC Playoffs
# ============================================================================

def example_urc_playoffs():
    """Predict URC playoff bracket."""

    print("=" * 70)
    print("Example 5: URC Playoffs Prediction")
    print("=" * 70)
    print()

    # Create URC playoffs bracket
    bracket = create_urc_playoffs_bracket()

    print("Playoff bracket structure:")
    print(bracket.to_dataframe()[["id", "home_team", "away_team"]])
    print()

    # Final regular season standings (top 8)
    final_standings = pd.DataFrame([
        {"position": 1, "team": "Munster", "points": 77},
        {"position": 2, "team": "Leinster", "points": 75},
        {"position": 3, "team": "Stormers", "points": 63},
        {"position": 4, "team": "Bulls", "points": 61},
        {"position": 5, "team": "Ulster", "points": 58},
        {"position": 6, "team": "Sharks", "points": 56},
        {"position": 7, "team": "Edinburgh", "points": 51},
        {"position": 8, "team": "Ospreys", "points": 49},
    ])

    print("Final standings (playoff teams):")
    print(final_standings.to_string(index=False))
    print()

    # Predict playoffs
    print("Predicting playoff bracket...")
    # predictor = BracketPredictor(match_predictor, bracket)
    # prediction = predictor.predict_bracket(
    #     pool_standings=final_standings,
    #     n_simulations=5000
    # )

    print("\nChampionship probabilities:")
    # print(prediction.advancement_probs[["team", "semifinal_prob", "final_prob", "champion_prob"]])

    example_playoffs = pd.DataFrame([
        {"team": "Munster", "semifinal_prob": 0.82, "final_prob": 0.58, "champion_prob": 0.32},
        {"team": "Leinster", "semifinal_prob": 0.85, "final_prob": 0.62, "champion_prob": 0.38},
        {"team": "Stormers", "semifinal_prob": 0.68, "final_prob": 0.38, "champion_prob": 0.18},
        {"team": "Bulls", "semifinal_prob": 0.71, "final_prob": 0.42, "champion_prob": 0.21},
    ])
    print(example_playoffs.to_string(index=False))
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Run examples
    example_world_cup_prediction()
    print("\n\n")

    example_champions_cup_tbc()
    print("\n\n")

    example_paths_to_championship()
    print("\n\n")

    example_updating_predictions()
    print("\n\n")

    example_urc_playoffs()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
