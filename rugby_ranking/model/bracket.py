"""
Knockout tournament bracket structures for rugby competitions.

Provides data structures and templates for modeling knockout tournaments
with support for TBD (To Be Determined) participants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Dict, List, Tuple
from datetime import datetime

import pandas as pd


@dataclass
class TBD:
    """
    Placeholder for a to-be-determined team in a knockout bracket.

    Examples:
        TBD("Winner QF1") - Winner of quarterfinal 1
        TBD("Pool A winner") - Team finishing first in Pool A
        TBD("Best runner-up", criteria={"method": "points"}) - Best 2nd place team
    """

    source: str  # Description of how team will be determined
    criteria: Dict = field(default_factory=dict)  # Additional qualification criteria

    def __str__(self) -> str:
        return f"TBD: {self.source}"

    def __repr__(self) -> str:
        if self.criteria:
            return f"TBD('{self.source}', {self.criteria})"
        return f"TBD('{self.source}')"


@dataclass
class BracketMatch:
    """
    A single match in a knockout tournament bracket.

    Attributes:
        id: Unique match identifier (e.g., "QF1", "SF2", "Final")
        round_type: Type of round
        round_number: Which instance of this round type (e.g., 1st QF, 2nd QF)
        home_team: Home team name or TBD placeholder
        away_team: Away team name or TBD placeholder
        depends_on: Match IDs that must complete before this match
        winner_advances_to: Match ID the winner progresses to (None for final)
        loser_advances_to: Match ID the loser progresses to (for 3rd place playoff)
        date: Scheduled date
        venue: Match venue
        home_advantage: Whether home team has advantage (False for neutral venue)
    """

    id: str
    round_type: Literal["round_of_16", "quarterfinal", "semifinal", "final", "third_place"]
    round_number: int
    home_team: str | TBD
    away_team: str | TBD
    depends_on: List[str] = field(default_factory=list)
    winner_advances_to: str | None = None
    loser_advances_to: str | None = None
    date: datetime | None = None
    venue: str | None = None
    home_advantage: bool = True

    def is_determined(self) -> bool:
        """Check if both participants are known (not TBD)."""
        return (
            not isinstance(self.home_team, TBD)
            and not isinstance(self.away_team, TBD)
        )

    def has_dependencies(self) -> bool:
        """Check if this match depends on other matches."""
        return len(self.depends_on) > 0

    def get_tbd_placeholders(self) -> List[TBD]:
        """Get list of TBD placeholders in this match."""
        placeholders = []
        if isinstance(self.home_team, TBD):
            placeholders.append(self.home_team)
        if isinstance(self.away_team, TBD):
            placeholders.append(self.away_team)
        return placeholders


class BracketStructure:
    """
    Models a knockout tournament bracket with dependencies and advancement rules.

    Attributes:
        name: Tournament name
        matches: Dictionary mapping match ID to BracketMatch
        rounds: List of round names in order
    """

    def __init__(self, name: str, structure: Dict):
        """
        Initialize bracket from structure definition.

        Args:
            name: Tournament name
            structure: Dictionary defining bracket structure with keys:
                - 'rounds': List of round names
                - 'matches': List of match definitions
        """
        self.name = name
        self.rounds: List[str] = structure.get("rounds", [])
        self.matches: Dict[str, BracketMatch] = {}

        # Build matches
        for match_def in structure.get("matches", []):
            match = self._build_match(match_def)
            self.matches[match.id] = match

    def _build_match(self, match_def: Dict) -> BracketMatch:
        """Build BracketMatch from definition dictionary."""
        # Parse home/away teams (could be team name or TBD)
        home = self._parse_team(match_def.get("home", {}))
        away = self._parse_team(match_def.get("away", {}))

        # Parse date if provided
        date = None
        if "date" in match_def:
            if isinstance(match_def["date"], str):
                date = datetime.fromisoformat(match_def["date"])
            else:
                date = match_def["date"]

        return BracketMatch(
            id=match_def["id"],
            round_type=match_def["round_type"],
            round_number=match_def.get("round_number", 1),
            home_team=home,
            away_team=away,
            depends_on=match_def.get("depends_on", []),
            winner_advances_to=match_def.get("winner_advances_to"),
            loser_advances_to=match_def.get("loser_advances_to"),
            date=date,
            venue=match_def.get("venue"),
            home_advantage=match_def.get("home_advantage", True),
        )

    def _parse_team(self, team_def: Dict | str) -> str | TBD:
        """Parse team from definition (could be name or TBD)."""
        if isinstance(team_def, str):
            return team_def
        elif isinstance(team_def, dict):
            if team_def.get("team") == "TBC" or team_def.get("team") == "TBD":
                return TBD(
                    source=team_def.get("source", "Unknown"),
                    criteria=team_def.get("criteria", {})
                )
            else:
                return team_def.get("team", "Unknown")
        else:
            return "Unknown"

    def get_round_matches(self, round_type: str) -> List[BracketMatch]:
        """
        Get all matches in a specific round.

        Args:
            round_type: Round type to filter by

        Returns:
            List of BracketMatch objects in that round
        """
        return [
            match for match in self.matches.values()
            if match.round_type == round_type
        ]

    def get_dependencies(self, match_id: str) -> List[str]:
        """
        Get all matches that must complete before the specified match.

        Args:
            match_id: Match to query

        Returns:
            List of match IDs this match depends on
        """
        if match_id not in self.matches:
            return []
        return self.matches[match_id].depends_on

    def get_dependent_matches(self, match_id: str) -> List[str]:
        """
        Get all matches that depend on the specified match.

        Args:
            match_id: Match to query

        Returns:
            List of match IDs that depend on this match
        """
        return [
            mid for mid, match in self.matches.items()
            if match_id in match.depends_on
        ]

    def is_match_determined(self, match_id: str) -> bool:
        """
        Check if both participants in a match are known.

        Args:
            match_id: Match to check

        Returns:
            True if both teams are known, False if any are TBD
        """
        if match_id not in self.matches:
            return False
        return self.matches[match_id].is_determined()

    def get_all_tbd_placeholders(self) -> List[Tuple[str, TBD]]:
        """
        Get all TBD placeholders across the entire bracket.

        Returns:
            List of (match_id, TBD) tuples
        """
        placeholders = []
        for match_id, match in self.matches.items():
            for tbd in match.get_tbd_placeholders():
                placeholders.append((match_id, tbd))
        return placeholders

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert bracket to DataFrame representation.

        Returns:
            DataFrame with columns: id, round_type, home_team, away_team, etc.
        """
        data = []
        for match in self.matches.values():
            data.append({
                "id": match.id,
                "round_type": match.round_type,
                "round_number": match.round_number,
                "home_team": str(match.home_team),
                "away_team": str(match.away_team),
                "determined": match.is_determined(),
                "depends_on": ", ".join(match.depends_on),
                "winner_advances_to": match.winner_advances_to,
                "date": match.date,
                "venue": match.venue,
                "home_advantage": match.home_advantage,
            })
        return pd.DataFrame(data)

    def clone(self) -> BracketStructure:
        """Create a deep copy of this bracket."""
        import copy
        return copy.deepcopy(self)


# Standard bracket templates
def create_champions_cup_bracket() -> BracketStructure:
    """
    Create European Champions Cup knockout bracket (Round of 16 onwards).

    Format: 16 teams → R16 → QF → SF → Final
    Seeding: Pool winners seeded 1-8, runners-up seeded 9-16
    Home advantage: Higher seed hosts
    """
    structure = {
        "rounds": ["round_of_16", "quarterfinal", "semifinal", "final"],
        "matches": [
            # Round of 16
            {
                "id": "R16_1",
                "round_type": "round_of_16",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Pool 1st #1"},
                "away": {"team": "TBD", "source": "Pool 2nd #8"},
                "winner_advances_to": "QF1",
            },
            {
                "id": "R16_2",
                "round_type": "round_of_16",
                "round_number": 2,
                "home": {"team": "TBD", "source": "Pool 1st #2"},
                "away": {"team": "TBD", "source": "Pool 2nd #7"},
                "winner_advances_to": "QF1",
            },
            {
                "id": "R16_3",
                "round_type": "round_of_16",
                "round_number": 3,
                "home": {"team": "TBD", "source": "Pool 1st #3"},
                "away": {"team": "TBD", "source": "Pool 2nd #6"},
                "winner_advances_to": "QF2",
            },
            {
                "id": "R16_4",
                "round_type": "round_of_16",
                "round_number": 4,
                "home": {"team": "TBD", "source": "Pool 1st #4"},
                "away": {"team": "TBD", "source": "Pool 2nd #5"},
                "winner_advances_to": "QF2",
            },
            {
                "id": "R16_5",
                "round_type": "round_of_16",
                "round_number": 5,
                "home": {"team": "TBD", "source": "Pool 1st #5"},
                "away": {"team": "TBD", "source": "Pool 2nd #4"},
                "winner_advances_to": "QF3",
            },
            {
                "id": "R16_6",
                "round_type": "round_of_16",
                "round_number": 6,
                "home": {"team": "TBD", "source": "Pool 1st #6"},
                "away": {"team": "TBD", "source": "Pool 2nd #3"},
                "winner_advances_to": "QF3",
            },
            {
                "id": "R16_7",
                "round_type": "round_of_16",
                "round_number": 7,
                "home": {"team": "TBD", "source": "Pool 1st #7"},
                "away": {"team": "TBD", "source": "Pool 2nd #2"},
                "winner_advances_to": "QF4",
            },
            {
                "id": "R16_8",
                "round_type": "round_of_16",
                "round_number": 8,
                "home": {"team": "TBD", "source": "Pool 1st #8"},
                "away": {"team": "TBD", "source": "Pool 2nd #1"},
                "winner_advances_to": "QF4",
            },
            # Quarterfinals
            {
                "id": "QF1",
                "round_type": "quarterfinal",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Winner R16_1"},
                "away": {"team": "TBD", "source": "Winner R16_2"},
                "depends_on": ["R16_1", "R16_2"],
                "winner_advances_to": "SF1",
            },
            {
                "id": "QF2",
                "round_type": "quarterfinal",
                "round_number": 2,
                "home": {"team": "TBD", "source": "Winner R16_3"},
                "away": {"team": "TBD", "source": "Winner R16_4"},
                "depends_on": ["R16_3", "R16_4"],
                "winner_advances_to": "SF1",
            },
            {
                "id": "QF3",
                "round_type": "quarterfinal",
                "round_number": 3,
                "home": {"team": "TBD", "source": "Winner R16_5"},
                "away": {"team": "TBD", "source": "Winner R16_6"},
                "depends_on": ["R16_5", "R16_6"],
                "winner_advances_to": "SF2",
            },
            {
                "id": "QF4",
                "round_type": "quarterfinal",
                "round_number": 4,
                "home": {"team": "TBD", "source": "Winner R16_7"},
                "away": {"team": "TBD", "source": "Winner R16_8"},
                "depends_on": ["R16_7", "R16_8"],
                "winner_advances_to": "SF2",
            },
            # Semifinals
            {
                "id": "SF1",
                "round_type": "semifinal",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Winner QF1"},
                "away": {"team": "TBD", "source": "Winner QF2"},
                "depends_on": ["QF1", "QF2"],
                "winner_advances_to": "Final",
            },
            {
                "id": "SF2",
                "round_type": "semifinal",
                "round_number": 2,
                "home": {"team": "TBD", "source": "Winner QF3"},
                "away": {"team": "TBD", "source": "Winner QF4"},
                "depends_on": ["QF3", "QF4"],
                "winner_advances_to": "Final",
            },
            # Final
            {
                "id": "Final",
                "round_type": "final",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Winner SF1"},
                "away": {"team": "TBD", "source": "Winner SF2"},
                "depends_on": ["SF1", "SF2"],
                "home_advantage": False,  # Neutral venue
            },
        ],
    }

    return BracketStructure("Champions Cup", structure)


def create_world_cup_bracket() -> BracketStructure:
    """
    Create Rugby World Cup knockout bracket (Quarterfinals onwards).

    Format: 8 teams → QF → SF → Final + 3rd place
    Qualification: Pool winners and runners-up
    Venue: All neutral venues
    """
    structure = {
        "rounds": ["quarterfinal", "semifinal", "final", "third_place"],
        "matches": [
            # Quarterfinals
            {
                "id": "QF1",
                "round_type": "quarterfinal",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Pool A winner"},
                "away": {"team": "TBD", "source": "Pool B runner-up"},
                "winner_advances_to": "SF1",
                "loser_advances_to": None,
                "home_advantage": False,
            },
            {
                "id": "QF2",
                "round_type": "quarterfinal",
                "round_number": 2,
                "home": {"team": "TBD", "source": "Pool C winner"},
                "away": {"team": "TBD", "source": "Pool D runner-up"},
                "winner_advances_to": "SF2",
                "loser_advances_to": None,
                "home_advantage": False,
            },
            {
                "id": "QF3",
                "round_type": "quarterfinal",
                "round_number": 3,
                "home": {"team": "TBD", "source": "Pool B winner"},
                "away": {"team": "TBD", "source": "Pool A runner-up"},
                "winner_advances_to": "SF1",
                "loser_advances_to": None,
                "home_advantage": False,
            },
            {
                "id": "QF4",
                "round_type": "quarterfinal",
                "round_number": 4,
                "home": {"team": "TBD", "source": "Pool D winner"},
                "away": {"team": "TBD", "source": "Pool C runner-up"},
                "winner_advances_to": "SF2",
                "loser_advances_to": None,
                "home_advantage": False,
            },
            # Semifinals
            {
                "id": "SF1",
                "round_type": "semifinal",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Winner QF1"},
                "away": {"team": "TBD", "source": "Winner QF3"},
                "depends_on": ["QF1", "QF3"],
                "winner_advances_to": "Final",
                "loser_advances_to": "Third",
                "home_advantage": False,
            },
            {
                "id": "SF2",
                "round_type": "semifinal",
                "round_number": 2,
                "home": {"team": "TBD", "source": "Winner QF2"},
                "away": {"team": "TBD", "source": "Winner QF4"},
                "depends_on": ["QF2", "QF4"],
                "winner_advances_to": "Final",
                "loser_advances_to": "Third",
                "home_advantage": False,
            },
            # Final
            {
                "id": "Final",
                "round_type": "final",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Winner SF1"},
                "away": {"team": "TBD", "source": "Winner SF2"},
                "depends_on": ["SF1", "SF2"],
                "home_advantage": False,
            },
            # Third place playoff
            {
                "id": "Third",
                "round_type": "third_place",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Loser SF1"},
                "away": {"team": "TBD", "source": "Loser SF2"},
                "depends_on": ["SF1", "SF2"],
                "home_advantage": False,
            },
        ],
    }

    return BracketStructure("Rugby World Cup", structure)


def create_urc_playoffs_bracket() -> BracketStructure:
    """
    Create URC (United Rugby Championship) playoffs bracket.

    Format: 8 teams → QF → SF → Final
    Seeding: Standard 1v8, 2v7, 3v6, 4v5
    Home advantage: Higher seed hosts until final
    """
    structure = {
        "rounds": ["quarterfinal", "semifinal", "final"],
        "matches": [
            # Quarterfinals
            {
                "id": "QF1",
                "round_type": "quarterfinal",
                "round_number": 1,
                "home": {"team": "TBD", "source": "1st place"},
                "away": {"team": "TBD", "source": "8th place"},
                "winner_advances_to": "SF1",
                "home_advantage": True,
            },
            {
                "id": "QF2",
                "round_type": "quarterfinal",
                "round_number": 2,
                "home": {"team": "TBD", "source": "4th place"},
                "away": {"team": "TBD", "source": "5th place"},
                "winner_advances_to": "SF1",
                "home_advantage": True,
            },
            {
                "id": "QF3",
                "round_type": "quarterfinal",
                "round_number": 3,
                "home": {"team": "TBD", "source": "2nd place"},
                "away": {"team": "TBD", "source": "7th place"},
                "winner_advances_to": "SF2",
                "home_advantage": True,
            },
            {
                "id": "QF4",
                "round_type": "quarterfinal",
                "round_number": 4,
                "home": {"team": "TBD", "source": "3rd place"},
                "away": {"team": "TBD", "source": "6th place"},
                "winner_advances_to": "SF2",
                "home_advantage": True,
            },
            # Semifinals
            {
                "id": "SF1",
                "round_type": "semifinal",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Higher seed from QF1/QF2"},
                "away": {"team": "TBD", "source": "Lower seed from QF1/QF2"},
                "depends_on": ["QF1", "QF2"],
                "winner_advances_to": "Final",
                "home_advantage": True,
            },
            {
                "id": "SF2",
                "round_type": "semifinal",
                "round_number": 2,
                "home": {"team": "TBD", "source": "Higher seed from QF3/QF4"},
                "away": {"team": "TBD", "source": "Lower seed from QF3/QF4"},
                "depends_on": ["QF3", "QF4"],
                "winner_advances_to": "Final",
                "home_advantage": True,
            },
            # Final
            {
                "id": "Final",
                "round_type": "final",
                "round_number": 1,
                "home": {"team": "TBD", "source": "Higher seed from SF"},
                "away": {"team": "TBD", "source": "Lower seed from SF"},
                "depends_on": ["SF1", "SF2"],
                "home_advantage": True,
            },
        ],
    }

    return BracketStructure("URC Playoffs", structure)
