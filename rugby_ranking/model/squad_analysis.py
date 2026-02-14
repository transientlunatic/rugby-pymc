"""
Squad Analysis for Rugby Tournaments.

Analyze squad strength, depth, and predict likely lineups when squads
are announced but match-day teams are not yet selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Literal
import re

import pandas as pd
import numpy as np


# Standard rugby positions with numerical mappings
POSITION_MAP = {
    # Props
    'loosehead prop': 'Prop',
    'loosehead': 'Prop',
    'lhp': 'Prop',
    'prop': 'Prop',
    'tighthead prop': 'Prop',
    'tighthead': 'Prop',
    'thp': 'Prop',

    # Hooker
    'hooker': 'Hooker',
    'hk': 'Hooker',

    # Locks
    'lock': 'Lock',
    'second row': 'Lock',

    # Back row
    'flanker': 'Flanker',
    'blindside flanker': 'Flanker',
    'openside flanker': 'Flanker',
    'number 8': 'Number 8',
    'no. 8': 'Number 8',
    'no 8': 'Number 8',
    '8': 'Number 8',

    # Half-backs
    'scrum-half': 'Scrum-half',
    'scrumhalf': 'Scrum-half',
    'scrum half': 'Scrum-half',
    'fly-half': 'Fly-half',
    'flyhalf': 'Fly-half',
    'fly half': 'Fly-half',
    'out-half': 'Fly-half',
    'stand-off': 'Fly-half',

    # Centres
    'centre': 'Centre',
    'center': 'Centre',
    'inside centre': 'Centre',
    'outside centre': 'Centre',

    # Back three
    'wing': 'Wing',
    'winger': 'Wing',
    'fullback': 'Fullback',
    'full-back': 'Fullback',
    'full back': 'Fullback',
}

# Position groupings
POSITION_GROUPS = {
    'Front Row': ['Prop', 'Hooker'],
    'Second Row': ['Lock'],
    'Back Row': ['Flanker', 'Number 8'],
    'Half-backs': ['Scrum-half', 'Fly-half'],
    'Centres': ['Centre'],
    'Back Three': ['Wing', 'Fullback'],
}


@dataclass
class SquadAnalysis:
    """Complete squad analysis results."""
    team: str
    season: str
    squad: pd.DataFrame

    # Player ratings from model
    player_ratings: Optional[pd.DataFrame] = None

    # Depth analysis
    depth_chart: Optional[Dict[str, List[Tuple[str, float]]]] = None
    position_strength: Optional[pd.DataFrame] = None
    overall_strength: Optional[float] = None
    depth_score: Optional[float] = None

    # Lineup prediction
    likely_xv: Optional[Dict[str, str]] = None
    likely_bench: Optional[List[str]] = None
    selection_uncertainty: Optional[pd.DataFrame] = None


class SquadParser:
    """
    Parse squad lists from various text formats.

    Handles:
    - Wikipedia squad lists (most common format)
    - Simple comma/tab-separated lists
    - CSV files

    Usage:
        >>> parser = SquadParser()
        >>> squad = parser.parse_text(wikipedia_text, team="Scotland", season="2024-2025")
        >>> squad.to_csv('squads/scotland_2024-2025.csv')
    """

    def __init__(self):
        self.position_map = POSITION_MAP

    def parse_text(
        self,
        text: str,
        team: str,
        season: str,
        format: Literal['auto', 'wikipedia', 'wikipedia_sixnations', 'simple', 'csv'] = 'auto',
    ) -> pd.DataFrame:
        """
        Parse squad text into structured DataFrame.

        Args:
            text: Squad list text
            team: Team name
            season: Season (e.g., "2024-2025")
            format: Format hint ('auto', 'wikipedia', 'wikipedia_sixnations', 'simple', 'csv')

        Returns:
            DataFrame with columns: player, position, club, primary_position,
                                   secondary_positions
        """
        # Auto-detect format
        if format == 'auto':
            format = self._detect_format(text)

        if format == 'wikipedia':
            return self._parse_wikipedia(text, team, season)
        elif format == 'wikipedia_sixnations':
            return self._parse_wikipedia_sixnations(text, team, season)
        elif format == 'simple':
            return self._parse_simple(text, team, season)
        elif format == 'csv':
            return self._parse_csv(text, team, season)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _detect_format(self, text: str) -> str:
        """Auto-detect squad list format."""
        # Check for tab-separated Wikipedia Six Nations format
        # (name, position, DOB, caps, country, club)
        if '\t' in text:
            lines = [l for l in text.strip().split('\n') if l.strip()]
            if lines:
                # Check first few lines for tab-separated format with position as 2nd column
                first_line = lines[0].split('\t')
                if len(first_line) >= 2:
                    # Check if 2nd column looks like a position
                    possible_pos = first_line[1].strip().lower()
                    if any(pos_keyword in possible_pos for pos_keyword in 
                           ['prop', 'hooker', 'lock', 'flanker', 'number', 'scrum', 'fly', 'centre', 'wing', 'fullback']):
                        return 'wikipedia_sixnations'
        
        # Check for Wikipedia-style headers (Forwards, Props, etc.)
        if re.search(r'(Forwards|Backs|Props|Hookers)', text, re.IGNORECASE):
            return 'wikipedia'

        # Check for CSV headers
        if re.search(r'(Player|Name|Position).*,', text):
            return 'csv'

        # Default to simple format
        return 'simple'

    def _parse_wikipedia(
        self,
        text: str,
        team: str,
        season: str,
    ) -> pd.DataFrame:
        """
        Parse Wikipedia-style squad list.

        Example format:
            Forwards

            Props
            1. Andrew Porter (Leinster)
            3. Tadhg Furlong (Leinster)

            Hookers
            2. Dan Sheehan (Leinster)
        """
        players = []
        current_position = None
        current_section = None

        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers (Forwards/Backs)
            if line.lower() in ['forwards', 'backs']:
                current_section = line.lower()
                continue

            # Check for position headers
            position = self._extract_position_header(line)
            if position:
                current_position = position
                continue

            # Parse player line
            player_data = self._parse_player_line(line, current_position)
            if player_data:
                player_data['team'] = team
                player_data['season'] = season
                player_data['section'] = current_section
                players.append(player_data)

        if not players:
            raise ValueError("No players found in squad text. Please check format.")

        df = pd.DataFrame(players)

        # Infer primary/secondary positions
        df = self._infer_positions(df)

        return df

    def _parse_wikipedia_sixnations(
        self,
        text: str,
        team: str,
        season: str,
    ) -> pd.DataFrame:
        """
        Parse tab-separated Wikipedia Six Nations format.

        Example format:
            Ewan Ashman 	Hooker 	3 April 2000 (aged 25) 	32 	Scotland Edinburgh
            Dave Cherry 	Hooker 	3 January 1991 (aged 35) 	16 	France Vannes
            George Turner 	Hooker 	10 August 1992 (aged 33) 	50 	England Harlequins

        Columns: name, position, DOB, caps, country, club
        """
        players = []
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split on tabs
            parts = [p.strip() for p in line.split('\t')]

            # Need at least: name, position
            if len(parts) < 2:
                continue

            # Extract fields
            name = self._normalize_player_name(parts[0])
            position_text = parts[1] if parts[1] else None

            # Club is in the last column (after country)
            # Format is usually "Country ClubName" so we split on space and take from 2nd word onward
            club = None
            if len(parts) >= 5:
                # Last field is "Country Club" format
                country_club = parts[4]
                country_club_parts = country_club.split(None, 1)  # Split on first whitespace
                if len(country_club_parts) > 1:
                    club = country_club_parts[1]
                    
            # DOB is in parts[2] if available
            # Caps are in parts[3] if available

            if not name:
                continue

            player_data = {
                'player': name,
                'position_text': position_text,
                'club': club,
                'team': team,
                'season': season,
                'section': 'mixed',  # Will be determined by position
            }
            players.append(player_data)

        if not players:
            raise ValueError("No players found in squad text. Please check format.")

        df = pd.DataFrame(players)

        # Infer primary/secondary positions
        df = self._infer_positions(df)

        # Infer section (Forwards/Backs) from position
        def get_section(pos):
            if pos in ['Prop', 'Hooker', 'Lock', 'Flanker', 'Number 8']:
                return 'forwards'
            elif pos in ['Scrum-half', 'Fly-half', 'Centre', 'Wing', 'Fullback']:
                return 'backs'
            return 'mixed'

        df['section'] = df['primary_position'].apply(get_section)

        return df

    def _extract_position_header(self, line: str) -> Optional[str]:
        """Extract position from header line."""
        # Common position headers
        headers = [
            'props', 'loosehead props', 'tighthead props',
            'hookers', 'locks', 'second row',
            'flankers', 'number 8', 'back row',
            'scrum-halves', 'fly-halves', 'half-backs',
            'centres', 'wings', 'fullbacks', 'back three'
        ]

        line_lower = line.lower().strip()

        # Exact match
        if line_lower in headers:
            return line

        # Fuzzy match (e.g., "Prop" matches "props")
        for header in headers:
            if header in line_lower or line_lower in header:
                return line

        return None

    def _parse_player_line(
        self,
        line: str,
        current_position: Optional[str],
    ) -> Optional[Dict]:
        """
        Parse individual player line.

        Handles formats:
        - "1. Andrew Porter (Leinster)"
        - "Andrew Porter (Leinster)"
        - "Andrew Porter, Leinster"
        """
        # Remove squad number at start (e.g., "1. ")
        line = re.sub(r'^\d+\.\s*', '', line)

        # Skip if line is too short
        if len(line) < 5:
            return None

        # Extract club from parentheses
        club_match = re.search(r'\(([^)]+)\)', line)
        club = club_match.group(1).strip() if club_match else None

        # Extract player name (before club)
        if club_match:
            name = line[:club_match.start()].strip()
        else:
            # Try comma separator
            parts = line.split(',')
            name = parts[0].strip()
            club = parts[1].strip() if len(parts) > 1 else None

        # Clean up name
        name = self._normalize_player_name(name)

        if not name:
            return None

        return {
            'player': name,
            'club': club,
            'position_text': current_position,
        }

    def _normalize_player_name(self, name: str) -> str:
        """
        Normalize player name.

        - Remove squad numbers
        - Remove extra whitespace
        - Handle special characters
        """
        # Remove leading/trailing whitespace
        name = name.strip()

        # Remove squad numbers that might be embedded
        name = re.sub(r'\b\d{1,2}\b\.?\s*', '', name)

        # Standardize multiple spaces
        name = re.sub(r'\s+', ' ', name)

        # Remove parenthetical notes (e.g., "(c)" for captain)
        name = re.sub(r'\s*\([^)]*\)\s*', '', name)

        return name.strip()

    def _parse_simple(
        self,
        text: str,
        team: str,
        season: str,
    ) -> pd.DataFrame:
        """
        Parse simple comma/tab-separated format.

        Example:
            Andrew Porter, Leinster, Prop
            Dan Sheehan, Leinster, Hooker
        """
        players = []
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try comma separator
            parts = [p.strip() for p in line.split(',')]

            if len(parts) >= 2:
                player_data = {
                    'player': self._normalize_player_name(parts[0]),
                    'club': parts[1] if len(parts) > 1 else None,
                    'position_text': parts[2] if len(parts) > 2 else None,
                    'team': team,
                    'season': season,
                }
                players.append(player_data)

        df = pd.DataFrame(players)
        df = self._infer_positions(df)
        return df

    def _parse_csv(
        self,
        text: str,
        team: str,
        season: str,
    ) -> pd.DataFrame:
        """Parse CSV format."""
        from io import StringIO

        # Try to parse as CSV
        try:
            df = pd.read_csv(StringIO(text))

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()

            # Map columns
            col_map = {
                'name': 'player',
                'team': 'club',
                'club': 'club',
            }

            df = df.rename(columns=col_map)

            # Add team and season if not present
            if 'team' not in df.columns:
                df['team'] = team
            if 'season' not in df.columns:
                df['season'] = season

            df = self._infer_positions(df)
            return df

        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

    def _infer_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer primary and secondary positions from position text.

        Args:
            df: DataFrame with 'position_text' column

        Returns:
            DataFrame with added columns: 'primary_position', 'secondary_positions'
        """
        if 'position_text' not in df.columns:
            df['primary_position'] = None
            df['secondary_positions'] = None
            return df

        primary_positions = []
        secondary_positions = []

        for pos_text in df['position_text']:
            primary, secondary = self._parse_position_text(pos_text)
            primary_positions.append(primary)
            secondary_positions.append(secondary)

        df['primary_position'] = primary_positions
        df['secondary_positions'] = secondary_positions

        return df

    def _parse_position_text(
        self,
        pos_text: Optional[str],
    ) -> Tuple[Optional[str], List[str]]:
        """
        Parse position text into primary and secondary positions.

        Examples:
            "Prop" → ("Prop", [])
            "Loosehead Prop" → ("Prop", [])
            "Flanker / Number 8" → ("Flanker", ["Number 8"])
            "Props" → ("Prop", [])
        """
        if not pos_text:
            return None, []

        pos_text = pos_text.lower().strip()

        # Handle plural forms
        pos_text = pos_text.rstrip('s')

        # Split on / or "and" for multiple positions
        parts = re.split(r'[/,]|\band\b', pos_text)
        parts = [p.strip() for p in parts if p.strip()]

        positions = []
        for part in parts:
            # Map to standard position
            standard_pos = self.position_map.get(part)
            if standard_pos:
                positions.append(standard_pos)

        if not positions:
            # Couldn't parse, return original text
            return pos_text, []

        return positions[0], positions[1:] if len(positions) > 1 else []

    def save_squad(self, squad: pd.DataFrame, filepath: str):
        """Save squad to CSV file."""
        squad.to_csv(filepath, index=False)
        print(f"Squad saved to: {filepath}")

    def load_squad(self, filepath: str) -> pd.DataFrame:
        """Load squad from CSV file."""
        return pd.read_csv(filepath)


class SquadAnalyzer:
    """
    Analyze squad strength using model player ratings.

    Uses trained model to extract player ratings and analyze squad
    depth, strength, and likely lineups.
    """

    def __init__(self, model, trace, dataset=None):
        self.model = model
        self.trace = trace
        self.dataset = dataset

    def _normalize_squad_format(self, squad: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize squad DataFrame to standard format.

        Handles both old format (player, position_text) and new Wikipedia
        scraper format (name, position).

        Args:
            squad: Input DataFrame in any supported format

        Returns:
            Normalized DataFrame with 'player', 'position_text', and 'primary_position' columns
        """
        squad = squad.copy()

        # Map 'name' -> 'player' if needed
        if 'name' in squad.columns and 'player' not in squad.columns:
            squad['player'] = squad['name']

        # Map 'position' -> 'position_text' if needed
        if 'position' in squad.columns and 'position_text' not in squad.columns:
            squad['position_text'] = squad['position']

        # Map 'position' -> 'primary_position' if needed
        if 'position' in squad.columns and 'primary_position' not in squad.columns:
            squad['primary_position'] = squad['position']
        elif 'position_text' in squad.columns and 'primary_position' not in squad.columns:
            squad['primary_position'] = squad['position_text']

        # Add 'secondary_positions' if missing (default to empty list)
        if 'secondary_positions' not in squad.columns:
            squad['secondary_positions'] = '[]'

        # Ensure required columns exist
        if 'player' not in squad.columns:
            raise ValueError("Squad DataFrame must have either 'player' or 'name' column")
        if 'position_text' not in squad.columns:
            raise ValueError("Squad DataFrame must have either 'position_text' or 'position' column")

        return squad

    def analyze_squad(
        self,
        squad: pd.DataFrame,
        team: str,
        season: str,
    ) -> SquadAnalysis:
        """
        Comprehensive squad analysis.

        Args:
            squad: Squad DataFrame from SquadParser or Wikipedia scraper
                   Accepts both formats:
                   - Old format: 'player', 'position_text', 'club'
                   - New format: 'name', 'position', 'club'
            team: Team name
            season: Season (e.g., "2024-2025")

        Returns:
            SquadAnalysis with ratings, depth charts, strength scores
        """
        # Normalize column names to handle both formats
        squad = self._normalize_squad_format(squad)

        print(f"\nAnalyzing squad for {team} ({season})...")
        print("=" * 60)

        # Get player ratings from model
        print("Extracting player ratings from model...")
        player_ratings = self.get_player_ratings(
            squad['player'].tolist(),
            season=season
        )

        # Create depth chart
        print("Creating depth chart...")
        depth_chart = self.create_depth_chart(squad, player_ratings)

        # Calculate position strength
        print("Calculating position strength...")
        position_strength = self.calculate_position_strength(depth_chart)

        # Calculate overall metrics
        overall_strength = position_strength['expected_strength'].mean()
        depth_score = self.calculate_squad_depth_score(position_strength)

        # Predict likely XV (simple heuristic for now)
        likely_xv, likely_bench = self._predict_simple_lineup(depth_chart)

        # Selection uncertainty
        selection_uncertainty = self._calculate_selection_uncertainty(depth_chart)

        print(f"✓ Analysis complete")
        print(f"  Overall Strength: {overall_strength:.2f}/1.00")
        print(f"  Squad Depth Score: {depth_score:.2f}/1.00")
        print()

        return SquadAnalysis(
            team=team,
            season=season,
            squad=squad,
            player_ratings=player_ratings,
            depth_chart=depth_chart,
            position_strength=position_strength,
            overall_strength=overall_strength,
            depth_score=depth_score,
            likely_xv=likely_xv,
            likely_bench=likely_bench,
            selection_uncertainty=selection_uncertainty,
        )

    def get_player_ratings(
        self,
        players: List[str],
        season: str,
        score_types: List[str] = None,
    ) -> pd.DataFrame:
        """
        Get model ratings for list of players.

        Args:
            players: List of player names
            season: Season to get ratings for
            score_types: Score types to include (default: ['tries'])

        Returns:
            DataFrame with columns: player, score_type, rating_mean,
                                   rating_std, percentile, in_model
        """
        if score_types is None:
            score_types = ['tries']

        results = []

        # Get full player rankings from model
        for score_type in score_types:
            try:
                all_rankings = self.model.get_player_rankings(
                    self.trace,
                    score_type=score_type,
                    top_n=None  # Get all players
                )

                # Match squad players to model rankings
                for player in players:
                    # Try exact match first
                    matches = all_rankings[all_rankings['player'] == player]

                    # If no exact match, try fuzzy matching
                    if len(matches) == 0:
                        # Simple fuzzy: check if player surname is in model
                        player_parts = player.split()
                        if len(player_parts) >= 2:
                            surname = player_parts[-1]
                            matches = all_rankings[
                                all_rankings['player'].str.contains(surname, case=False)
                            ]

                    if len(matches) > 0:
                        # Take first (best) match
                        match = matches.iloc[0]
                        results.append({
                            'player': player,
                            'score_type': score_type,
                            'rating_mean': match['effect_mean'],
                            'rating_std': match.get('effect_std', 0.1),
                            'percentile': match.get('percentile', 0.5),
                            'in_model': True,
                            'matched_name': match['player'],
                        })
                    else:
                        # Player not in model (new cap or returning player)
                        results.append({
                            'player': player,
                            'score_type': score_type,
                            'rating_mean': 0.0,  # League average
                            'rating_std': 0.2,  # High uncertainty
                            'percentile': 0.5,  # Unknown
                            'in_model': False,
                            'matched_name': None,
                        })

            except Exception as e:
                print(f"Warning: Could not get rankings for {score_type}: {e}")
                # Add default ratings
                for player in players:
                    results.append({
                        'player': player,
                        'score_type': score_type,
                        'rating_mean': 0.0,
                        'rating_std': 0.2,
                        'percentile': 0.5,
                        'in_model': False,
                        'matched_name': None,
                    })

        df = pd.DataFrame(results)

        # Report matching stats
        if len(df) > 0:
            matched = df[df['in_model']]['player'].nunique()
            total = len(players)
            print(f"  Matched {matched}/{total} players to model")
            if matched < total:
                unmatched = set(players) - set(df[df['in_model']]['player'])
                print(f"  Unmatched players: {', '.join(sorted(unmatched)[:5])}...")

        return df

    def create_depth_chart(
        self,
        squad: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Create depth chart for each position.

        Args:
            squad: Squad DataFrame with position information
            ratings: Player ratings DataFrame

        Returns:
            Dictionary mapping position -> [(player, rating), ...]
            sorted by rating (descending)
        """
        depth_chart = {}

        # Get unique positions
        positions = squad['primary_position'].dropna().unique()

        for position in positions:
            # Get players for this position
            position_players = squad[squad['primary_position'] == position]['player'].tolist()

            # Also check secondary positions
            secondary_players = squad[
                squad['secondary_positions'].apply(
                    lambda x: position in x if isinstance(x, list) else False
                )
            ]['player'].tolist()

            all_position_players = list(set(position_players + secondary_players))

            # Get ratings for these players (use 'tries' as proxy for overall ability)
            player_ratings_list = []
            for player in all_position_players:
                player_rating = ratings[
                    (ratings['player'] == player) &
                    (ratings['score_type'] == 'tries')
                ]

                if len(player_rating) > 0:
                    rating = player_rating.iloc[0]['rating_mean']
                else:
                    rating = 0.0

                player_ratings_list.append((player, rating))

            # Sort by rating (descending)
            player_ratings_list.sort(key=lambda x: x[1], reverse=True)

            depth_chart[position] = player_ratings_list

        return depth_chart

    def calculate_position_strength(
        self,
        depth_chart: Dict[str, List[Tuple[str, float]]],
    ) -> pd.DataFrame:
        """
        Calculate expected strength for each position.

        Args:
            depth_chart: Depth chart from create_depth_chart()

        Returns:
            DataFrame with columns: position, first_choice_rating,
                                   second_choice_rating, depth_score,
                                   expected_strength
        """
        results = []

        for position, players_ratings in depth_chart.items():
            if len(players_ratings) == 0:
                # No players for this position
                results.append({
                    'position': position,
                    'first_choice_rating': 0.0,
                    'second_choice_rating': 0.0,
                    'depth_score': 0.0,
                    'expected_strength': 0.0,
                    'n_players': 0,
                })
                continue

            # Get top ratings
            first_choice = players_ratings[0][1] if len(players_ratings) >= 1 else 0.0
            second_choice = players_ratings[1][1] if len(players_ratings) >= 2 else 0.0
            third_choice = players_ratings[2][1] if len(players_ratings) >= 3 else 0.0

            # Depth score: how much drop-off from 1st to 2nd choice
            if first_choice != 0:
                depth_score = max(0, min(1, (second_choice / first_choice)))
            else:
                depth_score = 0.5

            # Expected strength: weighted average (75% first choice, 25% depth)
            expected_strength = 0.75 * self._normalize_rating(first_choice) + \
                               0.25 * depth_score

            results.append({
                'position': position,
                'first_choice_rating': first_choice,
                'second_choice_rating': second_choice,
                'depth_score': depth_score,
                'expected_strength': expected_strength,
                'n_players': len(players_ratings),
                'first_choice_player': players_ratings[0][0],
                'second_choice_player': players_ratings[1][0] if len(players_ratings) >= 2 else None,
            })

        df = pd.DataFrame(results)
        df = df.sort_values('expected_strength', ascending=False)
        return df.reset_index(drop=True)

    def calculate_squad_depth_score(
        self,
        position_strength: pd.DataFrame,
    ) -> float:
        """
        Overall squad depth score.

        Measures average depth across all positions.

        Args:
            position_strength: DataFrame from calculate_position_strength()

        Returns:
            Score 0-1 where:
            - 1.0 = perfect depth (no drop-off)
            - 0.5 = moderate depth
            - 0.0 = poor depth
        """
        if len(position_strength) == 0:
            return 0.0

        # Average depth score across positions
        return position_strength['depth_score'].mean()

    def _normalize_rating(self, rating: float) -> float:
        """
        Normalize rating to 0-1 scale.

        Assumes ratings are centered around 0 with typical range [-2, +2]
        """
        # Sigmoid-like normalization
        return 1 / (1 + np.exp(-rating))

    def _predict_simple_lineup(
        self,
        depth_chart: Dict[str, List[Tuple[str, float]]],
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Simple lineup prediction: select top player from each position.

        Args:
            depth_chart: Depth chart from create_depth_chart()

        Returns:
            (likely_xv, likely_bench) where:
            - likely_xv: Dict[position, player] for starting XV
            - likely_bench: List of bench players
        """
        likely_xv = {}
        bench_candidates = []

        # Standard positions for starting XV
        starting_positions = [
            'Prop', 'Hooker', 'Lock', 'Flanker', 'Number 8',
            'Scrum-half', 'Fly-half', 'Centre', 'Wing', 'Fullback'
        ]

        for position, players in depth_chart.items():
            if len(players) > 0:
                # First choice for starting
                likely_xv[position] = players[0][0]

                # Rest are bench candidates
                bench_candidates.extend([p[0] for p in players[1:]])

        # Simple bench selection: next best players not in starting XV
        likely_bench = bench_candidates[:8] if len(bench_candidates) >= 8 else bench_candidates

        return likely_xv, likely_bench

    def _calculate_selection_uncertainty(
        self,
        depth_chart: Dict[str, List[Tuple[str, float]]],
    ) -> pd.DataFrame:
        """
        Calculate selection uncertainty for each position.

        Args:
            depth_chart: Depth chart from create_depth_chart()

        Returns:
            DataFrame with player selection probabilities
        """
        results = []

        for position, players in depth_chart.items():
            if len(players) == 0:
                continue

            # Simple heuristic: probability based on rating gap
            ratings = [r for _, r in players]
            total_rating = sum(max(0, r) for r in ratings)

            if total_rating > 0:
                for player, rating in players:
                    prob = max(0, rating) / total_rating
                    results.append({
                        'player': player,
                        'position': position,
                        'selection_probability': prob,
                    })
            else:
                # Equal probability if all ratings are 0 or negative
                prob = 1.0 / len(players)
                for player, _ in players:
                    results.append({
                        'player': player,
                        'position': position,
                        'selection_probability': prob,
                    })

        return pd.DataFrame(results)


class LineupPredictor:
    """
    Predict likely starting lineups from squad.

    Uses optimization to select best XV while respecting positional
    coverage requirements (specialist props, hooker, scrum-half).
    """

    # Standard starting XV positions (numbered 1-15)
    STARTING_XV_POSITIONS = {
        1: 'Prop',  # Loosehead
        2: 'Hooker',
        3: 'Prop',  # Tighthead
        4: 'Lock',
        5: 'Lock',
        6: 'Flanker',
        7: 'Flanker',
        8: 'Number 8',
        9: 'Scrum-half',
        10: 'Fly-half',
        11: 'Wing',
        12: 'Centre',
        13: 'Centre',
        14: 'Wing',
        15: 'Fullback',
    }

    # Required bench positions (typically 16-23)
    BENCH_REQUIREMENTS = {
        'Prop': 2,  # Need front row cover
        'Hooker': 1,
        'Lock': 1,  # Second row cover
        'Back row': 1,  # Flanker or Number 8
        'Scrum-half': 1,  # Half-back cover
        'Backs': 2,  # Utility backs
    }

    def __init__(self, model=None, trace=None):
        """
        Initialize LineupPredictor.

        Args:
            model: Fitted rugby model (optional, for rating-based predictions)
            trace: Model trace (optional)
        """
        self.model = model
        self.trace = trace

    def predict_lineup(
        self,
        squad_analysis: SquadAnalysis,
        unavailable: List[str] = None,
    ) -> Dict[str, any]:
        """
        Predict most likely starting XV.

        Uses optimization to select best possible lineup while respecting
        positional requirements.

        Args:
            squad_analysis: SquadAnalysis with player ratings
            unavailable: List of unavailable players (injuries, etc.)

        Returns:
            Dict with:
                - starting_xv: Dict[position_number, player_name]
                - bench: List[player_name] (8 players)
                - total_rating: Overall lineup quality score
                - coverage_valid: Whether lineup meets positional requirements
        """
        unavailable = unavailable or []

        # Get available players
        squad = squad_analysis.squad
        available = squad[~squad['player'].isin(unavailable)].copy()

        if len(available) < 23:
            raise ValueError(
                f"Insufficient players: {len(available)} available, need at least 23"
            )

        # Get player ratings
        ratings = squad_analysis.player_ratings
        if ratings is None:
            raise ValueError("Squad analysis must include player ratings")

        # Create rating lookup (use try-scoring rating as proxy for overall ability)
        player_rating_map = {}
        for _, row in ratings[ratings['score_type'] == 'tries'].iterrows():
            player_rating_map[row['player']] = row['rating_mean']

        # Build lineup using greedy selection with positional constraints
        starting_xv = {}
        selected_players = set()

        # Fill starting positions in order of specificity
        # 1. Front row (most specialist)
        for pos_num in [1, 2, 3]:
            position = self.STARTING_XV_POSITIONS[pos_num]
            candidates = self._get_candidates_for_position(
                available, position, selected_players, player_rating_map
            )
            if not candidates:
                raise ValueError(f"No available {position} for position {pos_num}")

            # Select best available
            best_player = max(candidates, key=lambda p: player_rating_map.get(p, 0.0))
            starting_xv[pos_num] = best_player
            selected_players.add(best_player)

        # 2. Scrum-half (specialist position)
        candidates = self._get_candidates_for_position(
            available, 'Scrum-half', selected_players, player_rating_map
        )
        if candidates:
            starting_xv[9] = max(candidates, key=lambda p: player_rating_map.get(p, 0.0))
            selected_players.add(starting_xv[9])

        # 3. Fill remaining positions by rating
        for pos_num, position in self.STARTING_XV_POSITIONS.items():
            if pos_num in starting_xv:
                continue  # Already filled

            candidates = self._get_candidates_for_position(
                available, position, selected_players, player_rating_map
            )
            if not candidates:
                # No primary candidates, try versatile players
                candidates = self._get_secondary_candidates(
                    available, position, selected_players, player_rating_map
                )

            if not candidates:
                raise ValueError(f"No available {position} for position {pos_num}")

            best_player = max(candidates, key=lambda p: player_rating_map.get(p, 0.0))
            starting_xv[pos_num] = best_player
            selected_players.add(best_player)

        # Select bench (8 players with positional cover)
        bench = self._select_bench(
            available, selected_players, player_rating_map
        )

        # Calculate total rating
        total_rating = sum(
            player_rating_map.get(p, 0.0) for p in starting_xv.values()
        ) / 15.0

        # Validate coverage
        coverage_valid = self._validate_coverage(starting_xv, bench, available)

        return {
            'starting_xv': starting_xv,
            'bench': bench,
            'total_rating': total_rating,
            'coverage_valid': coverage_valid,
        }

    def predict_lineup_distribution(
        self,
        squad_analysis: SquadAnalysis,
        n_samples: int = 100,
        uncertainty_factor: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate distribution of likely lineups via Monte Carlo sampling.

        Accounts for selection uncertainty when player ratings are close.

        Args:
            squad_analysis: SquadAnalysis with player ratings
            n_samples: Number of lineup samples
            uncertainty_factor: Amount of noise to add to ratings (0-1)

        Returns:
            DataFrame with player selection probabilities
        """
        # Get base ratings
        ratings = squad_analysis.player_ratings
        if ratings is None:
            raise ValueError("Squad analysis must include player ratings")

        player_ratings = ratings[ratings['score_type'] == 'tries'].copy()

        selection_counts = {}

        for _ in range(n_samples):
            # Add noise to ratings
            noisy_ratings = player_ratings.copy()
            noisy_ratings['rating_mean'] = (
                noisy_ratings['rating_mean'] +
                np.random.normal(0, uncertainty_factor, len(noisy_ratings))
            )

            # Create temporary squad analysis with noisy ratings
            noisy_analysis = SquadAnalysis(
                team=squad_analysis.team,
                season=squad_analysis.season,
                squad=squad_analysis.squad,
                player_ratings=noisy_ratings,
                depth_chart=squad_analysis.depth_chart,
            )

            # Predict lineup
            try:
                lineup = self.predict_lineup(noisy_analysis)

                # Count selections
                for player in lineup['starting_xv'].values():
                    selection_counts[player] = selection_counts.get(player, 0) + 1

                for player in lineup['bench']:
                    selection_counts[player] = selection_counts.get(player, 0) + 0.5

            except ValueError:
                continue  # Skip if lineup invalid

        # Convert to probabilities
        results = []
        for player, count in selection_counts.items():
            results.append({
                'player': player,
                'selection_probability': count / n_samples,
                'likely_role': 'starter' if count > n_samples * 0.5 else 'bench',
            })

        df = pd.DataFrame(results)
        return df.sort_values('selection_probability', ascending=False).reset_index(drop=True)

    def _get_candidates_for_position(
        self,
        squad: pd.DataFrame,
        position: str,
        excluded: set,
        ratings: Dict[str, float],
    ) -> List[str]:
        """Get primary candidates for a position."""
        candidates = squad[
            (squad['primary_position'] == position) &
            (~squad['player'].isin(excluded))
        ]['player'].tolist()

        return candidates

    def _get_secondary_candidates(
        self,
        squad: pd.DataFrame,
        position: str,
        excluded: set,
        ratings: Dict[str, float],
    ) -> List[str]:
        """Get secondary candidates (versatile players)."""
        candidates = squad[
            (squad['secondary_positions'].apply(
                lambda x: position in str(x) if pd.notna(x) else False
            )) &
            (~squad['player'].isin(excluded))
        ]['player'].tolist()

        return candidates

    def _select_bench(
        self,
        squad: pd.DataFrame,
        selected_starters: set,
        ratings: Dict[str, float],
    ) -> List[str]:
        """
        Select 8-player bench with positional coverage.

        Standard bench: 2 props, 1 hooker, 1 lock, 1 back row, 1 scrum-half, 2 backs
        """
        bench = []
        available = squad[~squad['player'].isin(selected_starters)].copy()

        # Add rating column for sorting
        available['rating'] = available['player'].map(lambda p: ratings.get(p, 0.0))

        # Priority positions for bench
        bench_positions = [
            ('Prop', 2),
            ('Hooker', 1),
            ('Lock', 1),
            ('Flanker', 1),  # Back row cover
            ('Scrum-half', 1),
        ]

        for position, count in bench_positions:
            candidates = available[
                available['primary_position'] == position
            ].nlargest(count, 'rating')

            for _, player_row in candidates.iterrows():
                bench.append(player_row['player'])
                available = available[available['player'] != player_row['player']]

        # Fill remaining bench spots with best available backs
        remaining = 8 - len(bench)
        if remaining > 0:
            back_positions = ['Fly-half', 'Centre', 'Wing', 'Fullback']
            backs = available[
                available['primary_position'].isin(back_positions)
            ].nlargest(remaining, 'rating')

            for _, player_row in backs.iterrows():
                bench.append(player_row['player'])

        return bench[:8]

    def _validate_coverage(
        self,
        starting_xv: Dict[int, str],
        bench: List[str],
        squad: pd.DataFrame,
    ) -> bool:
        """
        Validate that lineup meets positional coverage requirements.

        Checks:
        - Front row cover on bench (props + hooker)
        - Scrum-half on field or bench
        - Adequate back row, lock cover
        """
        # Get positions for bench players
        bench_positions = []
        for player in bench:
            player_row = squad[squad['player'] == player]
            if not player_row.empty:
                bench_positions.append(player_row.iloc[0]['primary_position'])

        # Check front row cover
        has_prop_cover = bench_positions.count('Prop') >= 2
        has_hooker_cover = bench_positions.count('Hooker') >= 1

        # Check half-back cover
        has_scrumhalf = 'Scrum-half' in bench_positions

        # Check lock/back row cover
        has_lock_cover = bench_positions.count('Lock') >= 1

        return has_prop_cover and has_hooker_cover and has_scrumhalf and has_lock_cover


class InjuryImpactAnalyzer:
    """
    Analyze impact of player injuries/unavailability on team strength.
    """

    def __init__(self, lineup_predictor: LineupPredictor):
        """
        Initialize InjuryImpactAnalyzer.

        Args:
            lineup_predictor: LineupPredictor instance for lineup generation
        """
        self.lineup_predictor = lineup_predictor

    def analyze_player_impact(
        self,
        player: str,
        squad_analysis: SquadAnalysis,
    ) -> Dict:
        """
        Quantify impact of losing a specific player.

        Args:
            player: Player name
            squad_analysis: SquadAnalysis with ratings

        Returns:
            Dict with:
                - player: Player name
                - position: Primary position
                - baseline_rating: Player's rating
                - replacement: Most likely replacement
                - replacement_rating: Replacement's rating
                - rating_drop: Difference in rating
                - relative_impact: Rating drop as % of team strength
                - criticality_score: Overall criticality (0-1)
        """
        # Get baseline lineup
        try:
            baseline = self.lineup_predictor.predict_lineup(squad_analysis)
            baseline_rating = baseline['total_rating']
        except ValueError as e:
            return {
                'player': player,
                'error': f"Could not generate baseline lineup: {e}",
            }

        # Get lineup without this player
        try:
            without_player = self.lineup_predictor.predict_lineup(
                squad_analysis,
                unavailable=[player]
            )
            without_rating = without_player['total_rating']
        except ValueError as e:
            # Player is critical - no valid replacement
            return {
                'player': player,
                'position': self._get_player_position(player, squad_analysis.squad),
                'baseline_rating': self._get_player_rating(player, squad_analysis.player_ratings),
                'replacement': None,
                'replacement_rating': 0.0,
                'rating_drop': baseline_rating,
                'relative_impact': 1.0,
                'criticality_score': 1.0,
                'error': f"No valid replacement: {e}",
            }

        # Identify replacement
        player_rating = self._get_player_rating(player, squad_analysis.player_ratings)
        player_position = self._get_player_position(player, squad_analysis.squad)

        # Find who replaced this player
        replacement = None
        for pos, starter in without_player['starting_xv'].items():
            if starter not in baseline['starting_xv'].values():
                replacement = starter
                break

        replacement_rating = (
            self._get_player_rating(replacement, squad_analysis.player_ratings)
            if replacement else 0.0
        )

        # Calculate impact metrics
        rating_drop = baseline_rating - without_rating
        relative_impact = rating_drop / baseline_rating if baseline_rating > 0 else 0.0

        # Criticality score combines rating drop and replaceability
        criticality_score = min(1.0, relative_impact * 2.0)  # Scale to 0-1

        return {
            'player': player,
            'position': player_position,
            'baseline_rating': player_rating,
            'replacement': replacement,
            'replacement_rating': replacement_rating,
            'rating_drop': rating_drop,
            'relative_impact': relative_impact,
            'criticality_score': criticality_score,
        }

    def identify_critical_players(
        self,
        squad_analysis: SquadAnalysis,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Identify most critical players in squad.

        Args:
            squad_analysis: SquadAnalysis with ratings
            top_n: Number of critical players to return

        Returns:
            DataFrame with player criticality rankings
        """
        results = []

        # Analyze impact of each player
        for player in squad_analysis.squad['player']:
            impact = self.analyze_player_impact(player, squad_analysis)
            if 'error' not in impact or impact.get('criticality_score', 0) > 0:
                results.append(impact)

        df = pd.DataFrame(results)

        if len(df) == 0:
            return df

        # Sort by criticality
        df = df.sort_values('criticality_score', ascending=False)

        return df.head(top_n).reset_index(drop=True)

    def analyze_squad_robustness(
        self,
        squad_analysis: SquadAnalysis,
        n_simulations: int = 100,
        injury_prob: float = 0.15,
    ) -> Dict:
        """
        Analyze squad robustness to random injuries.

        Simulates random injuries and measures impact on team strength.

        Args:
            squad_analysis: SquadAnalysis with ratings
            n_simulations: Number of injury scenarios to simulate
            injury_prob: Probability each player is injured

        Returns:
            Dict with:
                - mean_impact: Average strength loss from injuries
                - std_impact: Standard deviation of impact
                - worst_case: Worst simulated scenario
                - best_case: Best simulated scenario
                - robustness_score: Overall robustness (0-1, higher is better)
                - vulnerable_positions: Positions most affected
        """
        baseline = self.lineup_predictor.predict_lineup(squad_analysis)
        baseline_rating = baseline['total_rating']

        impacts = []
        position_impacts = {}

        for _ in range(n_simulations):
            # Randomly select injured players
            squad = squad_analysis.squad
            injured = squad[
                np.random.random(len(squad)) < injury_prob
            ]['player'].tolist()

            if not injured:
                # No injuries in this simulation
                impacts.append(0.0)
                continue

            # Get lineup with injuries
            try:
                with_injuries = self.lineup_predictor.predict_lineup(
                    squad_analysis,
                    unavailable=injured
                )
                impact = baseline_rating - with_injuries['total_rating']
                impacts.append(impact)

                # Track position-specific impacts
                for player in injured:
                    pos = self._get_player_position(player, squad)
                    if pos:
                        if pos not in position_impacts:
                            position_impacts[pos] = []
                        position_impacts[pos].append(impact)

            except ValueError:
                # Could not field valid lineup
                impacts.append(baseline_rating)  # Total collapse

        # Compute statistics
        impacts = np.array(impacts)
        mean_impact = impacts.mean()
        std_impact = impacts.std()
        worst_case = impacts.max()
        best_case = impacts.min()

        # Robustness score: inverse of normalized mean impact
        # High score = team maintains strength despite injuries
        robustness_score = max(0.0, 1.0 - (mean_impact / baseline_rating))

        # Identify vulnerable positions
        vulnerable_positions = []
        for pos, impacts_list in position_impacts.items():
            avg_impact = np.mean(impacts_list)
            vulnerable_positions.append({
                'position': pos,
                'average_impact': avg_impact,
            })

        vulnerable_positions = sorted(
            vulnerable_positions,
            key=lambda x: x['average_impact'],
            reverse=True
        )[:5]

        return {
            'mean_impact': mean_impact,
            'std_impact': std_impact,
            'worst_case': worst_case,
            'best_case': best_case,
            'robustness_score': robustness_score,
            'vulnerable_positions': vulnerable_positions,
        }

    def _get_player_rating(
        self,
        player: str,
        ratings: pd.DataFrame,
    ) -> float:
        """Get player's try-scoring rating."""
        if ratings is None or player is None:
            return 0.0

        player_ratings = ratings[
            (ratings['player'] == player) &
            (ratings['score_type'] == 'tries')
        ]

        if len(player_ratings) > 0:
            return player_ratings.iloc[0]['rating_mean']
        return 0.0

    def _get_player_position(
        self,
        player: str,
        squad: pd.DataFrame,
    ) -> Optional[str]:
        """Get player's primary position."""
        player_row = squad[squad['player'] == player]
        if not player_row.empty:
            return player_row.iloc[0]['primary_position']
        return None


class SquadBasedPredictor:
    """
    Generate match predictions using squad data (pre-team announcement).

    Bridges gap between teams-only predictions (high uncertainty) and
    full-lineup predictions (announced 48h before match).
    """

    def __init__(self, match_predictor, lineup_predictor: LineupPredictor):
        """
        Initialize SquadBasedPredictor.

        Args:
            match_predictor: MatchPredictor for match outcome prediction
            lineup_predictor: LineupPredictor for lineup generation
        """
        self.match_predictor = match_predictor
        self.lineup_predictor = lineup_predictor

    def predict_with_squads(
        self,
        home_squad_analysis: SquadAnalysis,
        away_squad_analysis: SquadAnalysis,
        season: str,
        n_lineup_samples: int = 50,
    ) -> Dict:
        """
        Predict match outcome using squad data.

        Samples likely lineups from both squads and aggregates predictions.

        Args:
            home_squad_analysis: Home team squad analysis
            away_squad_analysis: Away team squad analysis
            season: Season for prediction
            n_lineup_samples: Number of lineup samples per team

        Returns:
            Dict with:
                - home_win_prob: Probability home wins
                - away_win_prob: Probability away wins
                - draw_prob: Probability of draw
                - expected_home_score: Mean home score
                - expected_away_score: Mean away score
                - score_uncertainty: Additional uncertainty from lineup variation
                - lineup_samples: Sample lineups used (optional)
        """
        # Sample lineups from both teams
        home_lineups = self._sample_lineups(home_squad_analysis, n_lineup_samples)
        away_lineups = self._sample_lineups(away_squad_analysis, n_lineup_samples)

        # Predict match for each lineup combination (sample subset for efficiency)
        n_match_samples = min(100, n_lineup_samples)
        home_scores = []
        away_scores = []
        home_wins = 0
        away_wins = 0
        draws = 0

        for _ in range(n_match_samples):
            # Randomly select lineup samples
            home_lineup = home_lineups[np.random.randint(len(home_lineups))]
            away_lineup = away_lineups[np.random.randint(len(away_lineups))]

            # Predict match (this would need full integration with MatchPredictor)
            # For now, use a simplified rating-based prediction
            home_rating = home_lineup['total_rating']
            away_rating = away_lineup['total_rating']

            # Simple logistic model: P(home win) ~ sigmoid(home_advantage + rating_diff)
            rating_diff = home_rating - away_rating
            home_advantage = 0.1  # Typical home advantage

            # Predicted scores based on ratings (rough heuristic)
            base_score = 20
            home_score = base_score + rating_diff * 10 + home_advantage * 10
            away_score = base_score - rating_diff * 10

            # Add noise
            home_score += np.random.normal(0, 5)
            away_score += np.random.normal(0, 5)

            home_scores.append(home_score)
            away_scores.append(away_score)

            if home_score > away_score:
                home_wins += 1
            elif away_score > home_score:
                away_wins += 1
            else:
                draws += 1

        # Compute probabilities
        total = n_match_samples
        home_win_prob = home_wins / total
        away_win_prob = away_wins / total
        draw_prob = draws / total

        # Expected scores
        expected_home_score = np.mean(home_scores)
        expected_away_score = np.mean(away_scores)

        # Uncertainty from lineup variation
        score_uncertainty = {
            'home_std': np.std(home_scores),
            'away_std': np.std(away_scores),
        }

        return {
            'home_team': home_squad_analysis.team,
            'away_team': away_squad_analysis.team,
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'draw_prob': draw_prob,
            'expected_home_score': expected_home_score,
            'expected_away_score': expected_away_score,
            'score_uncertainty': score_uncertainty,
            'prediction_mode': 'squad-based',
        }

    def _sample_lineups(
        self,
        squad_analysis: SquadAnalysis,
        n_samples: int,
    ) -> List[Dict]:
        """Generate sample lineups from squad."""
        lineups = []

        for _ in range(n_samples):
            try:
                lineup = self.lineup_predictor.predict_lineup(squad_analysis)
                lineups.append(lineup)
            except ValueError:
                continue

        if not lineups:
            # Fallback: use simple lineup if predictions fail
            lineup = self.lineup_predictor.predict_lineup(squad_analysis)
            lineups = [lineup] * n_samples

        return lineups


class SquadComparator:
    """
    Compare multiple squads for tournament analysis.

    Generates pre-tournament squad rankings and identifies
    strengths/weaknesses across teams.
    """

    def __init__(self, squad_analyzer: SquadAnalyzer):
        """
        Initialize SquadComparator.

        Args:
            squad_analyzer: SquadAnalyzer instance
        """
        self.squad_analyzer = squad_analyzer

    def compare_squads(
        self,
        squads: Dict[str, pd.DataFrame],
        season: str,
    ) -> pd.DataFrame:
        """
        Compare multiple team squads.

        Args:
            squads: Dict mapping team name -> squad DataFrame
            season: Season

        Returns:
            DataFrame with team rankings by:
                - overall_strength
                - depth_score
                - position group strengths
        """
        results = []

        for team, squad_df in squads.items():
            try:
                analysis = self.squad_analyzer.analyze_squad(squad_df, team, season)

                results.append({
                    'team': team,
                    'overall_strength': analysis.overall_strength,
                    'depth_score': analysis.depth_score,
                    'squad_size': len(squad_df),
                })

            except Exception as e:
                print(f"Warning: Could not analyze {team}: {e}")
                continue

        df = pd.DataFrame(results)

        if len(df) > 0:
            df = df.sort_values('overall_strength', ascending=False)

        return df.reset_index(drop=True)

    def create_strength_matrix(
        self,
        squad_analyses: Dict[str, SquadAnalysis],
    ) -> pd.DataFrame:
        """
        Create position strength matrix across teams.

        Args:
            squad_analyses: Dict mapping team -> SquadAnalysis

        Returns:
            DataFrame with teams as rows, positions as columns
        """
        position_data = {}

        for team, analysis in squad_analyses.items():
            if analysis.position_strength is None:
                continue

            position_data[team] = {}
            for _, row in analysis.position_strength.iterrows():
                position_data[team][row['position']] = row['expected_strength']

        df = pd.DataFrame(position_data).T

        return df

    def identify_matchup_advantages(
        self,
        home_analysis: SquadAnalysis,
        away_analysis: SquadAnalysis,
    ) -> List[Dict]:
        """
        Identify key individual battles in a matchup.

        Args:
            home_analysis: Home team squad analysis
            away_analysis: Away team squad analysis

        Returns:
            List of position matchups with advantage indicators
        """
        matchups = []

        if home_analysis.position_strength is None or away_analysis.position_strength is None:
            return matchups

        # Get position strengths
        home_positions = home_analysis.position_strength.set_index('position')
        away_positions = away_analysis.position_strength.set_index('position')

        # Compare each position
        for position in home_positions.index:
            if position not in away_positions.index:
                continue

            home_strength = home_positions.loc[position, 'expected_strength']
            away_strength = away_positions.loc[position, 'expected_strength']

            advantage = home_strength - away_strength
            advantage_pct = (advantage / max(home_strength, away_strength, 0.01)) * 100

            matchups.append({
                'position': position,
                'home_strength': home_strength,
                'away_strength': away_strength,
                'advantage': advantage,
                'advantage_pct': advantage_pct,
                'favored_team': home_analysis.team if advantage > 0 else away_analysis.team,
            })

        # Sort by magnitude of advantage
        matchups = sorted(matchups, key=lambda x: abs(x['advantage']), reverse=True)

        return matchups


def format_squad_analysis(analysis: SquadAnalysis, detailed: bool = True) -> str:
    """
    Format squad analysis as human-readable report.

    Args:
        analysis: SquadAnalysis from SquadAnalyzer
        detailed: Include detailed breakdown

    Returns:
        Formatted string report
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(f"SQUAD ANALYSIS: {analysis.team.upper()} ({analysis.season})")
    lines.append("=" * 70)
    lines.append("")

    # Overall scores
    if analysis.overall_strength is not None:
        lines.append(f"Overall Strength: {analysis.overall_strength*100:.0f}/100")
    if analysis.depth_score is not None:
        lines.append(f"Squad Depth Score: {analysis.depth_score*100:.0f}/100")
    lines.append("")

    # Position strengths
    if detailed and analysis.position_strength is not None:
        lines.append("POSITION STRENGTHS")
        lines.append("-" * 70)
        lines.append(f"{'Position':<20} {'1st Choice':<12} {'Depth':<10} {'Strength':<10}")
        lines.append("-" * 70)

        for _, row in analysis.position_strength.iterrows():
            pos = row['position']
            first_choice = row['first_choice_rating']
            depth = row['depth_score']
            strength = row['expected_strength']

            # Add warning for low depth
            warning = " ⚠" if depth < 0.6 else ""

            lines.append(
                f"{pos:<20} {first_choice:>6.2f}       "
                f"{depth:>6.2%}     {strength:>6.2%}{warning}"
            )

        lines.append("")

        # Vulnerable positions
        vulnerable = analysis.position_strength[
            analysis.position_strength['depth_score'] < 0.7
        ].head(3)

        if len(vulnerable) > 0:
            lines.append("VULNERABLE POSITIONS")
            lines.append("-" * 70)
            for _, row in vulnerable.iterrows():
                first = row.get('first_choice_player', '?')
                second = row.get('second_choice_player', '?')
                drop = (1 - row['depth_score']) * 100
                lines.append(
                    f"{row['position']}: Large drop-off from {first} to "
                    f"{second} (-{drop:.0f}%)"
                )
            lines.append("")

        # Strongest positions
        strongest = analysis.position_strength.nlargest(3, 'expected_strength')
        if len(strongest) > 0:
            lines.append("STRONGEST POSITIONS")
            lines.append("-" * 70)
            for _, row in strongest.iterrows():
                lines.append(f"{row['position']}: Excellent depth and quality")
            lines.append("")

    # Likely XV
    if analysis.likely_xv:
        lines.append("MOST LIKELY STARTING XV")
        lines.append("-" * 70)

        # Group by position
        for position, player in sorted(analysis.likely_xv.items()):
            lines.append(f"{position:<20} {player}")

        lines.append("")

    # Bench
    if analysis.likely_bench:
        lines.append("LIKELY BENCH")
        lines.append("-" * 70)
        for i, player in enumerate(analysis.likely_bench[:8], 1):
            lines.append(f"{i}. {player}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def input_squad_interactive(team: str, season: str) -> pd.DataFrame:
    """
    Interactive squad input from command line or clipboard.

    Usage:
        >>> squad = input_squad_interactive("Scotland", "2024-2025")
        Paste squad list (Wikipedia format recommended).
        Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:

        [User pastes text]

        Parsed 35 players:
        - 18 forwards
        - 17 backs

        Accept? (y/n): y
    """
    import sys

    print(f"\nInput squad for {team} ({season})")
    print("=" * 60)
    print("Paste squad list (Wikipedia format recommended).")
    print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:")
    print()

    # Read from stdin
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    text = '\n'.join(lines)

    # Parse squad
    parser = SquadParser()
    try:
        squad = parser.parse_text(text, team=team, season=season)

        print(f"\nParsed {len(squad)} players:")

        # Count forwards vs backs
        if 'section' in squad.columns:
            forwards = len(squad[squad['section'] == 'forwards'])
            backs = len(squad[squad['section'] == 'backs'])
            print(f"  - {forwards} forwards")
            print(f"  - {backs} backs")

        # Show sample
        print("\nSample players:")
        print(squad[['player', 'primary_position', 'club']].head(5))

        # Confirm
        print("\nAccept this squad? (y/n): ", end='')
        response = input().strip().lower()

        if response == 'y':
            return squad
        else:
            print("Squad input cancelled.")
            return None

    except Exception as e:
        print(f"\nError parsing squad: {e}")
        print("Please check the format and try again.")
        return None


def export_squad_analysis_to_markdown(
    analysis: SquadAnalysis,
    critical_players: pd.DataFrame = None,
    robustness: Dict = None,
    output_path: str = None,
) -> str:
    """
    Export squad analysis to blog-ready Markdown format.

    Args:
        analysis: SquadAnalysis
        critical_players: Critical players DataFrame (optional)
        robustness: Squad robustness analysis (optional)
        output_path: Path to save markdown file (optional)

    Returns:
        Markdown formatted string
    """
    from datetime import datetime

    lines = []

    # Header
    lines.append(f"# Squad Analysis: {analysis.team}")
    lines.append(f"*Season: {analysis.season}*\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("---\n")

    # Summary
    lines.append("## Summary\n")
    if analysis.overall_strength is not None:
        lines.append(f"- **Overall Squad Strength:** {analysis.overall_strength*100:.0f}/100")
    if analysis.depth_score is not None:
        lines.append(f"- **Squad Depth:** {analysis.depth_score*100:.0f}/100")
    lines.append(f"- **Squad Size:** {len(analysis.squad)} players\n")

    # Position Analysis
    if analysis.position_strength is not None:
        lines.append("## Position-by-Position Breakdown\n")
        lines.append("| Position | 1st Choice | 2nd Choice | Depth | Strength |")
        lines.append("|----------|-----------|------------|-------|----------|")

        for _, row in analysis.position_strength.iterrows():
            first = row.get('first_choice_player', '')
            second = row.get('second_choice_player', '')
            depth = row['depth_score']
            strength = row['expected_strength']

            # Warning emoji for vulnerable positions
            warning = " ⚠️" if depth < 0.6 else ""

            lines.append(
                f"| {row['position']} | {first} | {second} | "
                f"{depth:.1%} | {strength:.1%}{warning} |"
            )

        lines.append("")

        # Strongest positions
        strongest = analysis.position_strength.nlargest(3, 'expected_strength')
        lines.append("### Strengths\n")
        for _, row in strongest.iterrows():
            lines.append(f"- **{row['position']}**: Excellent depth and quality")
        lines.append("")

        # Vulnerable positions
        vulnerable = analysis.position_strength[
            analysis.position_strength['depth_score'] < 0.7
        ].head(3)

        if len(vulnerable) > 0:
            lines.append("### Vulnerabilities\n")
            for _, row in vulnerable.iterrows():
                first = row.get('first_choice_player', '?')
                second = row.get('second_choice_player', '?')
                drop = (1 - row['depth_score']) * 100
                lines.append(
                    f"- **{row['position']}**: Significant drop-off from {first} to "
                    f"{second} (-{drop:.0f}%)"
                )
            lines.append("")

    # Critical Players
    if critical_players is not None and len(critical_players) > 0:
        lines.append("## Most Critical Players\n")
        lines.append("Players whose absence would most impact team performance:\n")
        lines.append("| Rank | Player | Position | Criticality | Replacement |")
        lines.append("|------|--------|----------|-------------|-------------|")

        for i, row in critical_players.head(10).iterrows():
            replacement = row.get('replacement', 'None')
            criticality = row.get('criticality_score', 0)
            lines.append(
                f"| {i+1} | {row['player']} | {row.get('position', '?')} | "
                f"{criticality:.0%} | {replacement} |"
            )

        lines.append("")

    # Squad Robustness
    if robustness is not None:
        lines.append("## Squad Robustness Analysis\n")
        lines.append("Resilience to injuries:\n")
        lines.append(f"- **Robustness Score:** {robustness['robustness_score']:.0%}")
        lines.append(f"- **Average Impact:** {robustness['mean_impact']:.2f} rating points")
        lines.append(f"- **Worst Case:** {robustness['worst_case']:.2f} rating points\n")

        if robustness.get('vulnerable_positions'):
            lines.append("### Most Vulnerable to Injuries:\n")
            for pos_impact in robustness['vulnerable_positions'][:3]:
                lines.append(
                    f"- **{pos_impact['position']}**: "
                    f"{pos_impact['average_impact']:.2f} avg impact"
                )
            lines.append("")

    # Predicted Starting XV
    if analysis.likely_xv:
        lines.append("## Predicted Starting XV\n")
        lines.append("| No. | Position | Player |")
        lines.append("|-----|----------|--------|")

        for pos_num in sorted(analysis.likely_xv.keys()):
            player = analysis.likely_xv[pos_num]
            lines.append(f"| {pos_num} | {LineupPredictor.STARTING_XV_POSITIONS[pos_num]} | {player} |")

        lines.append("")

        # Bench
        if analysis.likely_bench:
            lines.append("### Bench\n")
            for i, player in enumerate(analysis.likely_bench[:8], 16):
                lines.append(f"{i}. {player}")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by Rugby Ranking Model*")

    markdown = "\n".join(lines)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(markdown)
        print(f"Markdown exported to: {output_path}")

    return markdown


def export_tournament_comparison_to_markdown(
    squad_comparisons: pd.DataFrame,
    squad_analyses: Dict[str, SquadAnalysis],
    tournament_name: str,
    output_path: str = None,
) -> str:
    """
    Export tournament squad comparison to blog-ready Markdown.

    Args:
        squad_comparisons: DataFrame from SquadComparator.compare_squads()
        squad_analyses: Dict mapping team -> SquadAnalysis
        tournament_name: Tournament name (e.g., "Six Nations 2025")
        output_path: Path to save markdown file (optional)

    Returns:
        Markdown formatted string
    """
    from datetime import datetime

    lines = []

    # Header
    lines.append(f"# {tournament_name} Squad Analysis")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    lines.append("---\n")

    # Overall Rankings
    lines.append("## Overall Squad Rankings\n")
    lines.append("| Rank | Team | Strength | Depth | Squad Size |")
    lines.append("|------|------|----------|-------|------------|")

    for i, row in squad_comparisons.iterrows():
        strength = row['overall_strength']
        depth = row['depth_score']
        size = row['squad_size']
        lines.append(
            f"| {i+1} | {row['team']} | {strength:.0%} | {depth:.0%} | {size} |"
        )

    lines.append("")

    # Team-by-team breakdown
    lines.append("## Team-by-Team Analysis\n")

    for team in squad_comparisons['team']:
        if team not in squad_analyses:
            continue

        analysis = squad_analyses[team]
        lines.append(f"### {team}\n")

        # Summary
        if analysis.overall_strength:
            lines.append(f"**Strength:** {analysis.overall_strength:.0%} | ")
        if analysis.depth_score:
            lines.append(f"**Depth:** {analysis.depth_score:.0%}\n")

        # Top strengths
        if analysis.position_strength is not None:
            strongest = analysis.position_strength.nlargest(3, 'expected_strength')
            lines.append("\n**Strengths:**")
            for _, row in strongest.iterrows():
                lines.append(f"- {row['position']}")

            # Vulnerabilities
            vulnerable = analysis.position_strength[
                analysis.position_strength['depth_score'] < 0.7
            ].head(2)

            if len(vulnerable) > 0:
                lines.append("\n**Vulnerabilities:**")
                for _, row in vulnerable.iterrows():
                    lines.append(f"- {row['position']}")

        lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by Rugby Ranking Model*")

    markdown = "\n".join(lines)

    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(markdown)
        print(f"Tournament comparison exported to: {output_path}")

    return markdown


def create_squad_visualization(
    analysis: SquadAnalysis,
    output_path: str = None,
):
    """
    Create visualization of squad strength and depth.

    Args:
        analysis: SquadAnalysis
        output_path: Path to save figure (optional)

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available for visualization")
        return None

    if analysis.position_strength is None:
        print("Warning: No position strength data available")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Position strength
    pos_strength = analysis.position_strength.sort_values('expected_strength', ascending=True)

    colors = ['green' if x > 0.65 else 'orange' if x > 0.5 else 'red'
              for x in pos_strength['expected_strength']]

    ax1.barh(pos_strength['position'], pos_strength['expected_strength'], color=colors, alpha=0.7)
    ax1.set_xlabel('Expected Strength')
    ax1.set_title(f'{analysis.team} - Position Strength')
    ax1.set_xlim(0, 1)
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Depth scores
    pos_depth = analysis.position_strength.sort_values('depth_score', ascending=True)

    colors2 = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red'
               for x in pos_depth['depth_score']]

    ax2.barh(pos_depth['position'], pos_depth['depth_score'], color=colors2, alpha=0.7)
    ax2.set_xlabel('Depth Score')
    ax2.set_title(f'{analysis.team} - Squad Depth')
    ax2.set_xlim(0, 1)
    ax2.axvline(0.7, color='gray', linestyle='--', alpha=0.5, label='Good depth threshold')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

    return fig
