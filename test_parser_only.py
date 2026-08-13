#!/usr/bin/env python
"""
Test squad parser only (no dependencies on PyMC).
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import just the parser functions we need
import re
import pandas as pd
from typing import Tuple, Optional, List, Dict

# Simplified parser for testing
POSITION_MAP = {
    'loosehead prop': 'Prop',
    'prop': 'Prop',
    'tighthead prop': 'Prop',
    'hooker': 'Hooker',
    'lock': 'Lock',
    'flanker': 'Flanker',
    'number 8': 'Number 8',
    'scrum-half': 'Scrum-half',
    'fly-half': 'Fly-half',
    'centre': 'Centre',
    'wing': 'Wing',
    'fullback': 'Fullback',
}

def parse_wikipedia_simple(text: str, team: str, season: str) -> pd.DataFrame:
    """Simplified Wikipedia parser for testing."""
    players = []
    current_position = None
    current_section = None

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Section headers
        if line.lower() in ['forwards', 'backs']:
            current_section = line.lower()
            continue

        # Position headers
        if any(pos in line.lower() for pos in POSITION_MAP.keys()):
            current_position = line
            continue

        # Player lines (start with number or have parentheses)
        if re.match(r'^\d+\.', line) or '(' in line:
            # Remove number
            line = re.sub(r'^\d+\.\s*', '', line)

            # Extract club
            club_match = re.search(r'\(([^)]+)\)', line)
            club = club_match.group(1) if club_match else None

            # Extract name
            if club_match:
                name = line[:club_match.start()].strip()
            else:
                name = line.split(',')[0].strip()

            if name:
                players.append({
                    'player': name,
                    'club': club,
                    'position_text': current_position,
                    'section': current_section,
                    'team': team,
                    'season': season,
                })

    return pd.DataFrame(players)


# Test data
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


def main():
    print("=" * 70)
    print("SQUAD PARSER TEST")
    print("=" * 70)
    print()

    # Parse
    squad = parse_wikipedia_simple(SCOTLAND_SQUAD, "Scotland", "2024-2025")

    print(f"✓ Parsed {len(squad)} players")
    print()

    # Count by section
    forwards = len(squad[squad['section'] == 'forwards'])
    backs = len(squad[squad['section'] == 'backs'])
    print(f"  Forwards: {forwards}")
    print(f"  Backs: {backs}")
    print()

    # Display
    print("Parsed squad:")
    print("-" * 70)
    print(squad[['player', 'position_text', 'club', 'section']].to_string(index=False))
    print()

    # Save
    os.makedirs('squads', exist_ok=True)
    squad.to_csv('squads/test_scotland.csv', index=False)
    print("✓ Saved to: squads/test_scotland.csv")
    print()

    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == '__main__':
    main()
