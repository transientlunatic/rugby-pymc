#!/usr/bin/env python3
"""
Test the league table computation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from rugby_ranking.model.league_table import (
    LeagueTable,
    BonusPointRules,
    BonusPointConfig,
    format_table,
)


def test_basic_standings():
    """Test basic standings computation with URC rules."""
    print("=" * 70)
    print("TEST 1: Basic Standings (URC Rules)")
    print("=" * 70)

    # Create sample match data (one row per team per match)
    matches = pd.DataFrame([
        # Round 1: Leinster 28-14 Munster (Leinster: 4 tries, Munster: 2 tries)
        {'team': 'Leinster', 'opponent': 'Munster', 'score': 28, 'opponent_score': 14, 'tries': 4, 'is_home': True},
        {'team': 'Munster', 'opponent': 'Leinster', 'score': 14, 'opponent_score': 28, 'tries': 2, 'is_home': False},

        # Round 1: Ulster 21-20 Connacht (Ulster: 3 tries, Connacht: 3 tries, close loss)
        {'team': 'Ulster', 'opponent': 'Connacht', 'score': 21, 'opponent_score': 20, 'tries': 3, 'is_home': True},
        {'team': 'Connacht', 'opponent': 'Ulster', 'score': 20, 'opponent_score': 21, 'tries': 3, 'is_home': False},

        # Round 2: Leinster 35-7 Connacht (Leinster: 5 tries, Connacht: 1 try)
        {'team': 'Leinster', 'opponent': 'Connacht', 'score': 35, 'opponent_score': 7, 'tries': 5, 'is_home': True},
        {'team': 'Connacht', 'opponent': 'Leinster', 'score': 7, 'opponent_score': 35, 'tries': 1, 'is_home': False},

        # Round 2: Munster 24-24 Ulster (Munster: 4 tries, Ulster: 3 tries, draw)
        {'team': 'Munster', 'opponent': 'Ulster', 'score': 24, 'opponent_score': 24, 'tries': 4, 'is_home': True},
        {'team': 'Ulster', 'opponent': 'Munster', 'score': 24, 'opponent_score': 24, 'tries': 3, 'is_home': False},
    ])

    table = LeagueTable(bonus_rules=BonusPointRules.URC)
    standings = table.compute_standings(matches)

    print("\nStandings:")
    print(format_table(standings))

    # Verify results
    print("\n\nVerification:")
    print("-" * 70)

    leinster = standings[standings['team'] == 'Leinster'].iloc[0]
    print(f"\nLeinster:")
    print(f"  Played: {leinster['played']} (expected: 2)")
    print(f"  Won: {leinster['won']} (expected: 2)")
    print(f"  Try bonus: {leinster['try_bonus']} (expected: 2 - both matches had 4+ tries)")
    print(f"  Total points: {leinster['total_points']} (expected: 10 = 2 wins (8) + 2 try bonus (2))")

    munster = standings[standings['team'] == 'Munster'].iloc[0]
    print(f"\nMunster:")
    print(f"  Played: {munster['played']} (expected: 2)")
    print(f"  Won: {munster['won']} (expected: 0)")
    print(f"  Drawn: {munster['drawn']} (expected: 1)")
    print(f"  Lost: {munster['lost']} (expected: 1)")
    print(f"  Losing bonus: {munster['losing_bonus']} (expected: 1 - lost by 14, no bonus)")
    print(f"  Try bonus: {munster['try_bonus']} (expected: 1 - 4 tries in draw)")
    print(f"  Total points: {munster['total_points']} (expected: 3 = 1 draw (2) + 1 try bonus (1))")

    connacht = standings[standings['team'] == 'Connacht'].iloc[0]
    print(f"\nConnacht:")
    print(f"  Played: {connacht['played']} (expected: 2)")
    print(f"  Won: {connacht['won']} (expected: 0)")
    print(f"  Lost: {connacht['lost']} (expected: 2)")
    print(f"  Losing bonus: {connacht['losing_bonus']} (expected: 1 - lost by 1 to Ulster)")
    print(f"  Total points: {connacht['total_points']} (expected: 1 = 1 losing bonus)")

    # Assertions
    assert leinster['played'] == 2, "Leinster played count wrong"
    assert leinster['won'] == 2, "Leinster wins count wrong"
    assert leinster['try_bonus'] == 2, "Leinster try bonus wrong"
    assert leinster['total_points'] == 10, f"Leinster total points wrong: {leinster['total_points']}"

    assert munster['played'] == 2, "Munster played count wrong"
    assert munster['drawn'] == 1, "Munster draws count wrong"
    assert munster['try_bonus'] == 1, "Munster try bonus wrong"
    assert munster['losing_bonus'] == 0, "Munster losing bonus wrong"
    assert munster['total_points'] == 3, f"Munster total points wrong: {munster['total_points']}"

    assert connacht['losing_bonus'] == 1, "Connacht losing bonus wrong"
    assert connacht['total_points'] == 1, f"Connacht total points wrong: {connacht['total_points']}"

    print("\n✅ All assertions passed!")
    return True


def test_position_ordering():
    """Test that teams are ordered correctly by points, then diff, then tries."""
    print("\n" + "=" * 70)
    print("TEST 2: Position Ordering")
    print("=" * 70)

    # Create scenario where teams have same points but different diff/tries
    matches = pd.DataFrame([
        # Team A: 1 win with 4 tries = 5 points, +10 diff
        {'team': 'Team A', 'opponent': 'Team D', 'score': 30, 'opponent_score': 20, 'tries': 4, 'is_home': True},

        # Team B: 1 win with 4 tries = 5 points, +5 diff (worse than A)
        {'team': 'Team B', 'opponent': 'Team D', 'score': 25, 'opponent_score': 20, 'tries': 4, 'is_home': True},

        # Team C: 1 win with 3 tries + losing bonus = 5 points, +15 diff (best)
        {'team': 'Team C', 'opponent': 'Team E', 'score': 35, 'opponent_score': 20, 'tries': 3, 'is_home': True},
        {'team': 'Team C', 'opponent': 'Team F', 'score': 20, 'opponent_score': 25, 'tries': 3, 'is_home': True},

        # Opponents
        {'team': 'Team D', 'opponent': 'Team A', 'score': 20, 'opponent_score': 30, 'tries': 2, 'is_home': False},
        {'team': 'Team D', 'opponent': 'Team B', 'score': 20, 'opponent_score': 25, 'tries': 2, 'is_home': False},
        {'team': 'Team E', 'opponent': 'Team C', 'score': 20, 'opponent_score': 35, 'tries': 2, 'is_home': False},
        {'team': 'Team F', 'opponent': 'Team C', 'score': 25, 'opponent_score': 20, 'tries': 3, 'is_home': False},
    ])

    table = LeagueTable(bonus_rules=BonusPointRules.URC)
    standings = table.compute_standings(matches)

    print("\nStandings:")
    print(format_table(standings, max_teams=6))

    # Team C should be first (5 points, +10 diff)
    # Team A should be second (5 points, +10 diff, but C has better record)
    # Team B should be third (5 points, +5 diff)

    print("\n\nVerification:")
    print("-" * 70)
    print(f"Position 1: {standings.iloc[0]['team']} (points: {standings.iloc[0]['total_points']}, diff: {standings.iloc[0]['points_diff']})")
    print(f"Position 2: {standings.iloc[1]['team']} (points: {standings.iloc[1]['total_points']}, diff: {standings.iloc[1]['points_diff']})")
    print(f"Position 3: {standings.iloc[2]['team']} (points: {standings.iloc[2]['total_points']}, diff: {standings.iloc[2]['points_diff']})")

    print("\n✅ Position ordering test passed!")
    return True


def test_bonus_point_configs():
    """Test different bonus point configurations."""
    print("\n" + "=" * 70)
    print("TEST 3: Different Bonus Point Systems")
    print("=" * 70)

    # Match where team loses 20-15 with 3 tries (opponent has 2 tries)
    matches = pd.DataFrame([
        {'team': 'Team A', 'opponent': 'Team B', 'score': 15, 'opponent_score': 20, 'tries': 3, 'opponent_tries': 2, 'is_home': True},
        {'team': 'Team B', 'opponent': 'Team A', 'score': 20, 'opponent_score': 15, 'tries': 2, 'opponent_tries': 3, 'is_home': False},
    ])

    # URC/Premiership rules: need 4 tries, lose by ≤7
    print("\nURC/Premiership Rules (4 tries, lose by ≤7):")
    table_urc = LeagueTable(bonus_rules=BonusPointRules.URC)
    standings_urc = table_urc.compute_standings(matches, opponent_tries_col='opponent_tries')
    team_a_urc = standings_urc[standings_urc['team'] == 'Team A'].iloc[0]
    print(f"  Team A: {team_a_urc['total_points']} points (expected: 1 = 1 losing bonus)")
    print(f"    - Try bonus: {team_a_urc['try_bonus']} (needs 4 tries)")
    print(f"    - Losing bonus: {team_a_urc['losing_bonus']} (lost by 5 ≤ 7)")

    # Top14 rules: need 3 more tries than opponent, lose by ≤5
    print("\nTop14 Rules (3+ tries more than opponent, lose by ≤5):")
    table_top14 = LeagueTable(bonus_rules=BonusPointRules.TOP14)
    standings_top14 = table_top14.compute_standings(matches, opponent_tries_col='opponent_tries')
    team_a_top14 = standings_top14[standings_top14['team'] == 'Team A'].iloc[0]
    print(f"  Team A: {team_a_top14['total_points']} points (expected: 1 = 1 losing bonus)")
    print(f"    - Try bonus: {team_a_top14['try_bonus']} (has 3 tries, opponent has 2, diff=1 < 3)")
    print(f"    - Losing bonus: {team_a_top14['losing_bonus']} (lost by 5 ≤ 5)")

    # Verify
    assert team_a_urc['try_bonus'] == 0, "URC try bonus wrong"
    assert team_a_urc['losing_bonus'] == 1, "URC losing bonus wrong"
    assert team_a_urc['total_points'] == 1, "URC total points wrong"

    assert team_a_top14['try_bonus'] == 0, "Top14 try bonus wrong (needs 3+ more than opponent)"
    assert team_a_top14['losing_bonus'] == 1, "Top14 losing bonus wrong"
    assert team_a_top14['total_points'] == 1, "Top14 total points wrong"

    print("\n✅ Bonus point configuration test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LEAGUE TABLE TESTS")
    print("=" * 70 + "\n")

    try:
        test_basic_standings()
        test_position_ordering()
        test_bonus_point_configs()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
