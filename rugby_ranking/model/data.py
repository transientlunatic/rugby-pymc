"""
Data pipeline for extracting player-match observations from rugby JSON files.

This module transforms raw match data into a format suitable for Bayesian modelling,
with proper handling of:
- Player exposure times (substitutions)
- Scoring events by type
- Player mobility across teams/seasons
- Disciplinary events (cards)
- Player name normalization and fuzzy matching
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd


# Team name normalization mapping
# Maps variant names to canonical names
TEAM_NAME_ALIASES = {
    # URC/Celtic teams with variants
    "Glasgow ": "Glasgow Warriors",
    "Glasgow": "Glasgow Warriors",
    "Edinburgh": "Edinburgh Rugby",
    "Leinster": "Leinster Rugby",
    "Munster": "Munster Rugby",
    "Ulster": "Ulster Rugby",
    "Connacht": "Connacht Rugby",
    "Cardiff": "Cardiff Rugby",
    "Cardiff Blues": "Cardiff Rugby",
    "Dragons": "Dragons RFC",
    "Benetton": "Benetton Rugby",
    "Zebre": "Zebre Parma",
    # South African teams
    "Sharks": "Hollywoodbets Sharks",
    "Lions": "Emirates Lions",
    "Stormers": "DHL Stormers",
    "Blue Bulls": "Vodacom Bulls",
    "Bulls": "Vodacom Bulls",
    # Premiership teams (common variants)
    "Northampton": "Northampton Saints",
    "Newcastle Falcons": "Newcastle",
    "Newcastle Red Bulls": "Newcastle",
    "Leicester": "Leicester Tigers",
    "Bath": "Bath Rugby",
    "Sale": "Sale Sharks",
    "Exeter": "Exeter Chiefs",
    "Harlequins": "Harlequins",
    "Wasps": "Wasps",
    "Saracens": "Saracens",
    "Newcastle": "Newcastle Falcons",
    "Worcester": "Worcester Warriors",
    "Bristol": "Bristol Bears",
    "Gloucester": "Gloucester Rugby",
    "London Irish": "London Irish",
    # French Teams
    "Lyon O.U.": "Lyon",
    "Racing": "Racing 92",
    "Stade Francais Paris": "Stade Francais",
    "Clermont Auvergne": "Clermont",
    "Bordeaux-Beg": "Bordeaux-Begles",
    "Castres Olympique": "Castres",
    "Biarritz Olympique": "Biarritz",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to canonical form."""
    name = name.strip()
    return TEAM_NAME_ALIASES.get(name, name)


# ============================================================================
# Player Name Normalization
# ============================================================================

# Common name variations to normalize
# Maps variant spellings to canonical form
PLAYER_NAME_CORRECTIONS = {
    # Common first name variations
    "Jonny": "Johnny",
    "Johnnie": "Johnny",
    "Jon ": "John ",
    "Stu ": "Stuart ",
    "Steve ": "Steven ",
    "Mike ": "Michael ",
    "Mick ": "Michael ",
    "Rob ": "Robert ",
    "Bob ": "Robert ",
    "Bobby ": "Robert ",
    "Bill ": "William ",
    "Billy ": "William ",
    "Will ": "William ",
    "Willy ": "William ",
    "Willie ": "William ",
    "Tom ": "Thomas ",
    "Tommy ": "Thomas ",
    "Sam ": "Samuel ",
    "Sammy ": "Samuel ",
    "Dan ": "Daniel ",
    "Danny ": "Daniel ",
    "Andy ": "Andrew ",
    "Drew ": "Andrew ",
    "Nick ": "Nicholas ",
    "Nicky ": "Nicholas ",
    "Chris ": "Christopher ",
    "Matt ": "Matthew ",
    "Matty ": "Matthew ",
    "Alex ": "Alexander ",
    "Alec ": "Alexander ",
    "Jim ": "James ",
    "Jimmy ": "James ",
    "Jamie ": "James ",
    "Dave ": "David ",
    "Davey ": "David ",
    "Ben ": "Benjamin ",
    "Benny ": "Benjamin ",
    "Joe ": "Joseph ",
    "Joey ": "Joseph ",
    "Ed ": "Edward ",
    "Eddie ": "Edward ",
    "Ted ": "Edward ",
    "Teddy ": "Edward ",
    "Tony ": "Anthony ",
    "Pete ": "Peter ",
    "Paddy ": "Patrick ",
    "Pat ": "Patrick ",
    "Rory ": "Ruairi ",
    "Tadhg ": "Tadgh ",
}


def normalize_player_name(name: str) -> str:
    """
    Normalize a player name to canonical form.

    Handles:
    - Whitespace normalization
    - Common nickname/full name variations
    - Accented characters
    - Case normalization for matching
    """
    if not name:
        return name

    # Strip and normalize whitespace
    name = " ".join(name.split())

    # We don't apply nickname normalization by default as it could
    # create false matches. The fuzzy matcher will handle these.
    # This function just does basic cleanup.

    return name


def _name_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two player names.

    Uses a combination of:
    - Sequence matching for overall similarity
    - Special handling for first/last name swaps
    - Tolerance for common typos
    """
    if not name1 or not name2:
        return 0.0

    # Normalize for comparison
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return 1.0

    # Basic sequence matching
    ratio = SequenceMatcher(None, n1, n2).ratio()

    # Check if surnames match (more important in rugby context)
    parts1 = n1.split()
    parts2 = n2.split()

    if len(parts1) >= 2 and len(parts2) >= 2:
        # Compare last names
        if parts1[-1] == parts2[-1]:
            # Same surname - boost similarity
            ratio = max(ratio, 0.7 + 0.3 * SequenceMatcher(None, parts1[0], parts2[0]).ratio())
        # Check for first name initial match (e.g., "J. Smith" vs "John Smith")
        elif len(parts1[0]) == 2 and parts1[0].endswith('.'):
            if parts2[0].startswith(parts1[0][0]) and parts1[-1] == parts2[-1]:
                ratio = max(ratio, 0.85)
        elif len(parts2[0]) == 2 and parts2[0].endswith('.'):
            if parts1[0].startswith(parts2[0][0]) and parts1[-1] == parts2[-1]:
                ratio = max(ratio, 0.85)

    return ratio


class PlayerNameMatcher:
    """
    Fuzzy matcher for player names that builds a canonical mapping.

    Usage:
        matcher = PlayerNameMatcher(threshold=0.85)

        # Add names as they're encountered
        canonical1 = matcher.add_name("Johnny Matthews", team="Glasgow Warriors")
        canonical2 = matcher.add_name("Jonny Matthews", team="Glasgow Warriors")

        # Both should return the same canonical name
        assert canonical1 == canonical2
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the matcher.

        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider
                                  two names as the same person. Default 0.85
                                  is conservative to avoid false matches.
        """
        self.threshold = similarity_threshold
        # Maps canonical name -> list of variant names seen
        self._canonical_to_variants: dict[str, set[str]] = defaultdict(set)
        # Maps any name -> canonical name
        self._name_to_canonical: dict[str, str] = {}
        # Track which teams each player has played for (helps disambiguation)
        self._player_teams: dict[str, set[str]] = defaultdict(set)

    def add_name(self, name: str, team: str | None = None) -> str:
        """
        Add a player name and return its canonical form.

        If the name matches an existing player (same team context, similar name),
        returns the canonical form. Otherwise, creates a new canonical entry.

        Args:
            name: Player name to add
            team: Team the player is associated with (helps disambiguation)

        Returns:
            Canonical form of the name
        """
        name = normalize_player_name(name)
        if not name:
            return name

        # Check if we've seen this exact name before
        if name in self._name_to_canonical:
            canonical = self._name_to_canonical[name]
            if team:
                self._player_teams[canonical].add(team)
            return canonical

        # Look for similar names
        best_match = None
        best_score = 0.0

        for canonical, variants in self._canonical_to_variants.items():
            # Check similarity with canonical and all variants
            for variant in [canonical] + list(variants):
                score = _name_similarity(name, variant)

                # If team context matches, boost confidence
                if team and team in self._player_teams.get(canonical, set()):
                    score = min(1.0, score + 0.1)

                if score > best_score:
                    best_score = score
                    best_match = canonical

        # If we found a good match above threshold, use that canonical form
        if best_match and best_score >= self.threshold:
            self._name_to_canonical[name] = best_match
            self._canonical_to_variants[best_match].add(name)
            if team:
                self._player_teams[best_match].add(team)
            return best_match

        # No match found - this is a new player
        self._canonical_to_variants[name].add(name)
        self._name_to_canonical[name] = name
        if team:
            self._player_teams[name].add(team)
        return name

    def get_canonical(self, name: str) -> str:
        """Get the canonical form of a name (or the name itself if not seen)."""
        name = normalize_player_name(name)
        return self._name_to_canonical.get(name, name)

    def get_variants(self, canonical_name: str) -> set[str]:
        """Get all variant spellings seen for a canonical name."""
        return self._canonical_to_variants.get(canonical_name, set())

    def get_all_mappings(self) -> dict[str, str]:
        """Get the full mapping of all names to their canonical forms."""
        return self._name_to_canonical.copy()

    def get_potential_duplicates(self, threshold: float | None = None) -> list[tuple[str, str, float]]:
        """
        Find potential duplicate players that might need manual review.

        Returns pairs of canonical names that are similar but weren't
        automatically merged (e.g., same surname, different first name).

        Args:
            threshold: Similarity threshold for flagging (default: self.threshold - 0.1)

        Returns:
            List of (name1, name2, similarity_score) tuples
        """
        threshold = threshold or (self.threshold - 0.1)
        canonicals = list(self._canonical_to_variants.keys())
        potential_dupes = []

        for i, name1 in enumerate(canonicals):
            for name2 in canonicals[i + 1:]:
                score = _name_similarity(name1, name2)
                if threshold <= score < self.threshold:
                    potential_dupes.append((name1, name2, score))

        return sorted(potential_dupes, key=lambda x: -x[2])

    def merge_players(self, name1: str, name2: str, keep: str | None = None) -> str:
        """
        Manually merge two players that should be the same person.

        Args:
            name1: First player name
            name2: Second player name
            keep: Which name to use as canonical (default: name1)

        Returns:
            The canonical name after merging
        """
        canonical1 = self.get_canonical(name1)
        canonical2 = self.get_canonical(name2)

        if canonical1 == canonical2:
            return canonical1  # Already merged

        keep = keep or canonical1
        remove = canonical2 if keep == canonical1 else canonical1

        # Move all variants from 'remove' to 'keep'
        for variant in self._canonical_to_variants[remove]:
            self._name_to_canonical[variant] = keep
            self._canonical_to_variants[keep].add(variant)

        # Merge team associations
        self._player_teams[keep].update(self._player_teams.get(remove, set()))

        # Clean up
        del self._canonical_to_variants[remove]
        if remove in self._player_teams:
            del self._player_teams[remove]

        return keep


@dataclass
class PlayerMatchObservation:
    """A single player's participation in a single match."""

    # Identifiers
    player_name: str
    team: str
    opponent: str
    match_id: str
    season: str
    competition: str
    date: datetime

    # Location
    is_home: bool
    position: int  # Jersey number 1-23

    # Exposure
    minutes_played: float
    started: bool
    was_substituted: bool

    # Scoring events
    tries: int = 0
    conversions: int = 0
    penalties: int = 0
    drop_goals: int = 0

    # Disciplinary
    yellow_cards: int = 0
    red_cards: int = 0
    yellow_card_minutes: list[int] = field(default_factory=list)
    red_card_minutes: list[int] = field(default_factory=list)

    # Match context
    team_score: int = 0
    opponent_score: int = 0

    @property
    def total_points(self) -> int:
        return (self.tries * 5 + self.conversions * 2 +
                self.penalties * 3 + self.drop_goals * 3)

    @property
    def match_result(self) -> str:
        if self.team_score > self.opponent_score:
            return "win"
        elif self.team_score < self.opponent_score:
            return "loss"
        return "draw"


@dataclass
class MatchData:
    """Processed data for a single match."""

    match_id: str
    date: datetime
    season: str
    competition: str
    stadium: str

    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None

    home_lineup: dict  # position -> player data
    away_lineup: dict
    home_scores: list[dict]  # scoring events
    away_scores: list[dict]

    attendance: int | None = None
    round: int | None = None
    round_type: str | None = None

    @property
    def is_played(self) -> bool:
        return self.home_score is not None and self.away_score is not None



class MatchDataset:
    """
    Container for rugby match data with methods for extracting model-ready observations.

    Handles:
    - Loading from JSON files (both legacy and modern formats)
    - Player identification and tracking across teams/seasons
    - Extraction of player-match observations with exposure times
    - Fuzzy matching of player names to handle typos
    """

    def __init__(
        self,
        data_dir: str | Path,
        fuzzy_match_names: bool = True,
        name_similarity_threshold: float = 0.85,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to directory containing JSON match files
            fuzzy_match_names: If True, use fuzzy matching to normalize player names
            name_similarity_threshold: Threshold for fuzzy name matching (0-1)
        """
        self.data_dir = Path(data_dir)
        self.matches: list[MatchData] = []
        self._player_index: dict[str, int] = {}  # player_name -> unique id
        self._team_index: dict[str, int] = {}    # team_name -> unique id

        # Player name matching
        self._fuzzy_match = fuzzy_match_names
        self._name_matcher = PlayerNameMatcher(similarity_threshold=name_similarity_threshold)

    def load_json_files(self, pattern: str = "*.json") -> None:
        """Load all JSON files matching pattern from data directory."""
        json_dir = self.data_dir / "json"
        if not json_dir.exists():
            json_dir = self.data_dir

        for json_file in sorted(json_dir.glob(pattern)):
            self._load_single_file(json_file)

        print(f"Loaded {len(self.matches)} matches from {json_dir}")
        print(f"Found {len(self._player_index)} unique players")
        print(f"Found {len(self._team_index)} unique teams")

        # Report on name merging if fuzzy matching is enabled
        if self._fuzzy_match:
            merged_count = sum(
                1 for variants in self._name_matcher._canonical_to_variants.values()
                if len(variants) > 1
            )
            if merged_count > 0:
                print(f"Merged {merged_count} player name variants via fuzzy matching")

    def _load_single_file(self, json_path: Path) -> None:
        """Load a single JSON file, detecting format automatically."""
        # Extract competition and season from filename
        # e.g., "premiership-2021-2022.json" -> ("premiership", "2021-2022")
        stem = json_path.stem
        parts = stem.rsplit("-", 2)
        if len(parts) >= 3:
            competition = "-".join(parts[:-2])
            season = f"{parts[-2]}-{parts[-1]}"
        else:
            competition = stem
            season = "unknown"

        with open(json_path, "r") as f:
            data = json.load(f)

        # Detect format:
        # 1. LIST format: data = [match1, match2, ...] where each match has home/away dicts
        # 2. DICT format: data = {home: {0: {...}, 1: {...}}, away: {...}, date: {...}, ...}
        if isinstance(data, list):
            self._load_list_format(data, competition, season)
        elif isinstance(data, dict) and "home" in data and "away" in data:
            self._load_dict_format(data, competition, season)
        else:
            print(f"Unknown format in {json_path}, skipping")

    def _load_list_format(self, data: list, competition: str, season: str) -> None:
        """
        Load LIST format: data = [match1, match2, ...].

        Each match has structure:
        {
            "home": {"team": ..., "score": ..., "lineup": {...}, "scores": [...]},
            "away": {"team": ..., "score": ..., "lineup": {...}, "scores": [...]},
            "date": "2024-09-21T19:35:00",
            "stadium": "...",
            ...
        }
        """
        for i, match in enumerate(data):
            try:
                home = match.get("home", {})
                away = match.get("away", {})

                # Parse date
                date_str = match.get("date", "")
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    # Ensure timezone-aware (assume UTC if naive)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError, TypeError):
                    date = datetime.now(timezone.utc)

                # Handle team as dict or string, then normalize
                home_team = home.get("team", "Unknown") if isinstance(home, dict) else "Unknown"
                away_team = away.get("team", "Unknown") if isinstance(away, dict) else "Unknown"

                if isinstance(home_team, dict):
                    home_team = home_team.get("name", "Unknown")
                if isinstance(away_team, dict):
                    away_team = away_team.get("name", "Unknown")

                # Normalize team names
                home_team = normalize_team_name(home_team)
                away_team = normalize_team_name(away_team)

                # Get lineups
                home_lineup = home.get("lineup", {}) if isinstance(home, dict) else {}
                away_lineup = away.get("lineup", {}) if isinstance(away, dict) else {}
                
                # Allow matches without player-level data (for prediction purposes)
                # Note: Matches without lineups can still be used for team-only predictions
                # if not home_lineup and not away_lineup:
                #     continue

                match_data = MatchData(
                    match_id=f"{competition}_{season}_{i}",
                    date=date,
                    season=season,
                    competition=competition,
                    stadium=match.get("stadium", ""),
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home.get("score") if isinstance(home, dict) else None,
                    away_score=away.get("score") if isinstance(away, dict) else None,
                    home_lineup=home_lineup,
                    away_lineup=away_lineup,
                    home_scores=home.get("scores", []) if isinstance(home, dict) else [],
                    away_scores=away.get("scores", []) if isinstance(away, dict) else [],
                    attendance=match.get("attendance"),
                    round=match.get("round"),
                    round_type=match.get("round_type"),
                )

                self.matches.append(match_data)
                self._index_team(match_data.home_team)
                self._index_team(match_data.away_team)
                self._index_players_from_lineup(match_data.home_lineup, team=match_data.home_team)
                self._index_players_from_lineup(match_data.away_lineup, team=match_data.away_team)

            except (KeyError, TypeError) as e:
                print(f"Error loading list match {i} from {competition} {season}: {e}")
                continue

    def _load_dict_format(self, data: dict, competition: str, season: str) -> None:
        """
        Load DICT format: data = {home: {0: {...}, 1: {...}}, away: {...}, date: {...}, ...}.

        Top-level keys are 'home', 'away', 'date', 'stadium', etc.
        Each is indexed by match number as string keys ("0", "1", ...).
        """
        home_data = data["home"]
        away_data = data["away"]
        dates = data.get("date", {})
        stadiums = data.get("stadium", {})

        # Iterate over matches (keyed by index)
        for match_idx in home_data.keys():
            try:
                home = home_data[match_idx]
                away = away_data[match_idx]

                # Parse date
                date_str = dates.get(match_idx, "") if isinstance(dates, dict) else ""
                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    # Ensure timezone-aware (assume UTC if naive)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError, TypeError):
                    date = datetime.now(timezone.utc)

                match_id = f"{competition}_{season}_{match_idx}"

                # Handle potentially unplayed matches
                home_score = home.get("score") if isinstance(home, dict) else None
                away_score = away.get("score") if isinstance(away, dict) else None

                stadium = stadiums.get(match_idx, "") if isinstance(stadiums, dict) else ""

                # Extract and normalize team names
                home_team = home.get("team", "Unknown") if isinstance(home, dict) else "Unknown"
                away_team = away.get("team", "Unknown") if isinstance(away, dict) else "Unknown"
                home_team = normalize_team_name(home_team)
                away_team = normalize_team_name(away_team)

                # Get lineups
                home_lineup = home.get("lineup", {}) if isinstance(home, dict) else {}
                away_lineup = away.get("lineup", {}) if isinstance(away, dict) else {}
                
                # Allow matches without player-level data (for prediction purposes)
                # Note: Matches without lineups can still be used for team-only predictions
                # if not home_lineup and not away_lineup:
                #     continue

                match = MatchData(
                    match_id=match_id,
                    date=date,
                    season=season,
                    competition=competition,
                    stadium=stadium,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    home_lineup=home_lineup,
                    away_lineup=away_lineup,
                    home_scores=home.get("scores", []) if isinstance(home, dict) else [],
                    away_scores=away.get("scores", []) if isinstance(away, dict) else [],
                )

                self.matches.append(match)

                # Index teams and players
                self._index_team(match.home_team)
                self._index_team(match.away_team)
                self._index_players_from_lineup(match.home_lineup, team=match.home_team)
                self._index_players_from_lineup(match.away_lineup, team=match.away_team)

            except (KeyError, TypeError) as e:
                print(f"Error loading match {match_idx} from {competition} {season}: {e}")
                continue

    def _index_team(self, team_name: str) -> int:
        """Get or create a unique index for a team (normalized)."""
        normalized = normalize_team_name(team_name)
        if normalized not in self._team_index:
            self._team_index[normalized] = len(self._team_index)
        return self._team_index[normalized]

    def _index_players_from_lineup(self, lineup: dict, team: str | None = None) -> None:
        """Index all players from a lineup, applying fuzzy name matching."""
        for position, player_data in lineup.items():
            if isinstance(player_data, dict):
                name = player_data.get("name", "")
                if not name:
                    continue

                # Apply fuzzy matching to get canonical name
                if self._fuzzy_match:
                    canonical_name = self._name_matcher.add_name(name, team=team)
                else:
                    canonical_name = name

                if canonical_name not in self._player_index:
                    self._player_index[canonical_name] = len(self._player_index)

    def _calculate_minutes(self, player_data: dict, match_duration: int = 80) -> float:
        """Calculate minutes played from on/off arrays."""
        on_times = player_data.get("on", [])
        off_times = player_data.get("off", [])

        if not on_times:
            return 0.0

        total_minutes = 0.0

        # Pair up on/off times
        for i, on_time in enumerate(on_times):
            if i < len(off_times):
                off_time = off_times[i]
            else:
                off_time = match_duration
            total_minutes += max(0, off_time - on_time)

        return min(total_minutes, match_duration)

    def _count_scoring_events(
        self, player_name: str, scores: list[dict]
    ) -> dict[str, int]:
        """
        Count scoring events for a player.

        Handles both full name matching (recent data) and surname-only matching
        (older data where scores use surnames but lineups have full names).
        """
        counts = {"tries": 0, "conversions": 0, "penalties": 0, "drop_goals": 0}

        type_mapping = {
            "try": "tries",
            "conversion": "conversions",
            "penalty": "penalties",
            "drop goal": "drop_goals",
            "dropgoal": "drop_goals",
        }

        # Extract surname for fallback matching
        # Handle compound surnames like "van der Merwe"
        name_parts = player_name.split()
        if len(name_parts) >= 2:
            # Check for compound surname prefixes
            compound_prefixes = {"van", "de", "du", "le", "o'", "mc", "mac"}
            if name_parts[-2].lower() in compound_prefixes:
                surname = " ".join(name_parts[-2:])
            else:
                surname = name_parts[-1]
        else:
            surname = player_name

        for score in scores:
            scorer = score.get("player", "")
            score_type = score.get("type", "").lower()

            if score_type not in type_mapping:
                continue

            # Try exact match first (recent data format)
            if scorer == player_name:
                counts[type_mapping[score_type]] += 1
            # Fallback: surname match (older data format)
            elif scorer.lower() == surname.lower():
                counts[type_mapping[score_type]] += 1

        return counts

    def iter_player_observations(
        self,
        played_only: bool = True,
        min_minutes: float = 0.0,
    ) -> Iterator[PlayerMatchObservation]:
        """
        Iterate over all player-match observations.

        Args:
            played_only: If True, only include matches that have been played
            min_minutes: Minimum minutes played to include observation

        Yields:
            PlayerMatchObservation for each player in each match
        """
        for match in self.matches:
            if played_only and not match.is_played:
                continue

            # Process home team
            yield from self._extract_team_observations(
                match=match,
                lineup=match.home_lineup,
                scores=match.home_scores,
                team=match.home_team,
                opponent=match.away_team,
                team_score=match.home_score or 0,
                opponent_score=match.away_score or 0,
                is_home=True,
                min_minutes=min_minutes,
            )

            # Process away team
            yield from self._extract_team_observations(
                match=match,
                lineup=match.away_lineup,
                scores=match.away_scores,
                team=match.away_team,
                opponent=match.home_team,
                team_score=match.away_score or 0,
                opponent_score=match.home_score or 0,
                is_home=False,
                min_minutes=min_minutes,
            )

    def _extract_team_observations(
        self,
        match: MatchData,
        lineup: dict,
        scores: list[dict],
        team: str,
        opponent: str,
        team_score: int,
        opponent_score: int,
        is_home: bool,
        min_minutes: float,
    ) -> Iterator[PlayerMatchObservation]:
        """Extract observations for all players in a team's lineup."""
        for position_str, player_data in lineup.items():
            if not isinstance(player_data, dict):
                continue

            raw_name = player_data.get("name", "")
            if not raw_name:
                continue

            # Get canonical player name
            if self._fuzzy_match:
                player_name = self._name_matcher.get_canonical(raw_name)
            else:
                player_name = raw_name

            try:
                position = int(position_str)
            except ValueError:
                continue

            minutes = self._calculate_minutes(player_data)
            if minutes < min_minutes:
                continue

            on_times = player_data.get("on", [])
            off_times = player_data.get("off", [])
            started = 0 in on_times
            was_substituted = len(off_times) > 0

            scoring = self._count_scoring_events(player_name, scores)

            yellow_minutes = player_data.get("yellows", [])
            red_minutes = player_data.get("reds", [])

            yield PlayerMatchObservation(
                player_name=player_name,
                team=team,
                opponent=opponent,
                match_id=match.match_id,
                season=match.season,
                competition=match.competition,
                date=match.date,
                is_home=is_home,
                position=position,
                minutes_played=minutes,
                started=started,
                was_substituted=was_substituted,
                tries=scoring["tries"],
                conversions=scoring["conversions"],
                penalties=scoring["penalties"],
                drop_goals=scoring["drop_goals"],
                yellow_cards=len(yellow_minutes),
                red_cards=len(red_minutes),
                yellow_card_minutes=yellow_minutes,
                red_card_minutes=red_minutes,
                team_score=team_score,
                opponent_score=opponent_score,
            )

    def to_dataframe(self, played_only: bool = True) -> pd.DataFrame:
        """Convert all player observations to a pandas DataFrame."""
        observations = list(self.iter_player_observations(played_only=played_only))

        if not observations:
            return pd.DataFrame()

        records = []
        for obs in observations:
            records.append({
                "player_name": obs.player_name,
                "player_id": self._player_index.get(obs.player_name, -1),
                "team": obs.team,
                "team_id": self._team_index.get(obs.team, -1),
                "opponent": obs.opponent,
                "opponent_id": self._team_index.get(obs.opponent, -1),
                "match_id": obs.match_id,
                "season": obs.season,
                "competition": obs.competition,
                "date": obs.date,
                "is_home": obs.is_home,
                "position": obs.position,
                "minutes_played": obs.minutes_played,
                "started": obs.started,
                "was_substituted": obs.was_substituted,
                "tries": obs.tries,
                "conversions": obs.conversions,
                "penalties": obs.penalties,
                "drop_goals": obs.drop_goals,
                "total_points": obs.total_points,
                "yellow_cards": obs.yellow_cards,
                "red_cards": obs.red_cards,
                "team_score": obs.team_score,
                "opponent_score": obs.opponent_score,
                "match_result": obs.match_result,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df.sort_values("date").reset_index(drop=True)

    def get_player_history(self, player_name: str) -> pd.DataFrame:
        """Get all observations for a specific player."""
        df = self.to_dataframe()
        return df[df["player_name"] == player_name].copy()

    def get_player_teams(self) -> pd.DataFrame:
        """Get player-team-season mapping for tracking mobility."""
        df = self.to_dataframe()
        return (
            df.groupby(["player_name", "team", "season"])
            .agg({
                "match_id": "count",
                "minutes_played": "sum",
                "tries": "sum",
                "total_points": "sum",
            })
            .rename(columns={"match_id": "matches"})
            .reset_index()
        )

    def get_unplayed_matches(self) -> list[MatchData]:
        """Get all matches that haven't been played yet (for predictions)."""
        return [m for m in self.matches if not m.is_played]

    @property
    def player_ids(self) -> dict[str, int]:
        """Mapping of player names to unique integer IDs."""
        return self._player_index.copy()

    @property
    def team_ids(self) -> dict[str, int]:
        """Mapping of team names to unique integer IDs."""
        return self._team_index.copy()

    @property
    def name_matcher(self) -> PlayerNameMatcher:
        """Access the player name matcher for inspection or manual merging."""
        return self._name_matcher

    def get_merged_names(self) -> list[tuple[str, set[str]]]:
        """
        Get all players whose names were merged via fuzzy matching.

        Returns:
            List of (canonical_name, {variant1, variant2, ...}) tuples
        """
        return [
            (canonical, variants)
            for canonical, variants in self._name_matcher._canonical_to_variants.items()
            if len(variants) > 1
        ]

    def get_potential_duplicates(self) -> list[tuple[str, str, float]]:
        """
        Get potential duplicate players that weren't automatically merged.

        These are players with similar names that might be the same person
        but didn't meet the similarity threshold.

        Returns:
            List of (name1, name2, similarity_score) tuples
        """
        return self._name_matcher.get_potential_duplicates()

    def merge_players(self, name1: str, name2: str, keep: str | None = None) -> str:
        """
        Manually merge two players that should be the same person.

        Call this after loading if you identify duplicates that weren't
        automatically merged.

        Args:
            name1: First player name
            name2: Second player name
            keep: Which name to use as canonical (default: name1)

        Returns:
            The canonical name after merging
        """
        return self._name_matcher.merge_players(name1, name2, keep)
