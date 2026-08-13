#!/usr/bin/env python
"""
Test squad analysis functionality.

This script demonstrates:
1. Parsing a squad from Wikipedia-style text
2. Analyzing squad strength (requires trained model)
3. Generating analysis report
"""

from rugby_ranking.model.squad_analysis import SquadParser, SquadAnalyzer, format_squad_analysis

# Example Scotland squad (abbreviated)
SCOTLAND_SQUAD = """
Forwards

Props
1. Pierre Schoeman (Edinburgh)
3. Zander Fagerson (Glasgow Warriors)
17. Rory Sutherland (Glasgow Warriors)

Hookers
2. George Turner (Glasgow Warriors)
16. Johnny Matthews (Glasgow Warriors)

Locks
4. Grant Gilchrist (Edinburgh)
5. Scott Cummings (Glasgow Warriors)

Back row
6. Matt Fagerson (Glasgow Warriors)
7. Rory Darge (Glasgow Warriors)
8. Jack Dempsey (Edinburgh)

Backs

Scrum-halves
9. Ben White (Toulon)
21. George Horne (Glasgow Warriors)

Fly-halves
10. Finn Russell (Bath)
22. Adam Hastings (Glasgow Warriors)

Centres
12. Sione Tuipulotu (Glasgow Warriors)
13. Huw Jones (Glasgow Warriors)

Wings
11. Duhan van der Merwe (Edinburgh)
14. Darcy Graham (Edinburgh)

Fullbacks
15. Blair Kinghorn (Toulouse)
23. Kyle Rowe (Glasgow Warriors)
"""


def test_parser():
    """Test squad parser."""
    print("=" * 70)
    print("TEST 1: Squad Parser")
    print("=" * 70)
    print()

    parser = SquadParser()
    squad = parser.parse_text(
        SCOTLAND_SQUAD,
        team="Scotland",
        season="2024-2025"
    )

    print(f"✓ Parsed {len(squad)} players")
    print()

    # Display results
    print("Parsed squad:")
    print(squad[['player', 'primary_position', 'club']].to_string(index=False))
    print()

    # Save
    squad.to_csv('squads/test_scotland_2024-2025.csv', index=False)
    print("✓ Saved to: squads/test_scotland_2024-2025.csv")
    print()

    return squad


def test_analyzer(squad):
    """Test squad analyzer (requires trained model)."""
    print("=" * 70)
    print("TEST 2: Squad Analyzer (with mock data)")
    print("=" * 70)
    print()

    # For testing without a trained model, we'll create a mock analyzer
    # In production, you would load a real model:
    # from rugby_ranking.model.core import RugbyModel
    # from rugby_ranking.model.inference import ModelFitter
    # model = RugbyModel()
    # fitter = ModelFitter.load('latest', model)
    # analyzer = SquadAnalyzer(model, fitter.trace)

    print("Note: This test requires a trained model checkpoint.")
    print("To run full analysis:")
    print("  1. Train model: rugby-ranking update --data-dir /path/to/Rugby-Data")
    print("  2. Analyze squad: rugby-ranking squad analyze --team Scotland")
    print()

    # For now, just show what the structure looks like
    print("Expected output structure:")
    print("-" * 70)
    print("Overall Strength: XX/100")
    print("Squad Depth Score: XX/100")
    print()
    print("POSITION STRENGTHS")
    print("-" * 70)
    print("Position             1st Choice   Depth      Strength")
    print("-" * 70)
    print("Back Row               0.91      95%        94%")
    print("Centres                0.87      94%        92%")
    print("...")
    print()


def main():
    """Run tests."""
    import os

    # Create output directory
    os.makedirs('squads', exist_ok=True)

    # Test 1: Parser
    squad = test_parser()

    # Test 2: Analyzer (mock)
    test_analyzer(squad)

    print("=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Input actual squads:")
    print("   rugby-ranking squad input --team 'Scotland' --season '2024-2025'")
    print()
    print("2. Analyze squads (requires trained model):")
    print("   rugby-ranking squad analyze --team 'Scotland'")
    print()
    print("3. Compare tournament squads:")
    print("   rugby-ranking squad compare --tournament six-nations")
    print()


if __name__ == '__main__':
    main()
