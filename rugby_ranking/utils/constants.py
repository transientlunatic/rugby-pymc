"""Shared constants for rugby-ranking models."""

# Positions that can take kicks
KICKING_POSITIONS = frozenset({9, 10, 12, 15})  # Scrum-half, fly-half, inside center, fullback

# Positions that rarely take kicks
NON_KICKING_POSITIONS = frozenset({1, 2, 3, 4, 5, 6, 7, 8})  # Front 5 and back row

# Positions that score tries
TRY_SCORING_POSITIONS = frozenset({11, 12, 13, 14, 15})  # Back line

# Scoring adjustments (points per score type)
SCORING_ADJUSTMENT = {
    "tries": 5,
    "conversions": 2,
    "penalties": 3,
    "drop_goals": 3,
}

# Typical conversion success rate
CONVERSION_RATE = 0.70

# Average penalties per team per match
PENALTIES_PER_MATCH = 2.5

# Players on field per team
STARTERS = 15
