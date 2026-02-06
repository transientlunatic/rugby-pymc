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

    TODO: Implement lineup prediction functionality
    """

    def predict_lineup(self, squad_analysis: SquadAnalysis) -> Dict[str, str]:
        """
        Predict most likely starting XV.

        TODO: Implement optimization-based lineup selection
        """
        raise NotImplementedError


class InjuryImpactAnalyzer:
    """
    Analyze impact of player injuries/unavailability.

    TODO: Implement injury impact analysis
    """

    def analyze_player_impact(
        self,
        player: str,
        squad_analysis: SquadAnalysis,
    ) -> Dict:
        """
        Quantify impact of losing a specific player.

        TODO: Implement impact calculation
        """
        raise NotImplementedError


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
