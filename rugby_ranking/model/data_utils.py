"""
Utility functions for converting match data to league table and prediction formats.

These utilities eliminate boilerplate code when working with league tables
and season predictions in notebooks and scripts.
"""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timezone

import pandas as pd

from rugby_ranking.model.data import MatchData, MatchDataset


def count_tries(scores: list[dict]) -> int:
    """
    Count tries from scoring events.

    Args:
        scores: List of scoring event dicts with 'type' key

    Returns:
        Number of tries scored

    Examples:
        >>> scores = [{'type': 'try', 'player': 'Smith'}, {'type': 'penalty'}]
        >>> count_tries(scores)
        1
    """
    if not scores:
        return 0
    return sum(1 for s in scores if s.get('type', '').lower() in ['try', 't'])


def matches_to_league_table_format(
    matches: List[MatchData],
    include_tries: bool = True,
) -> pd.DataFrame:
    """
    Convert matches to league table format (one row per team per match).

    Args:
        matches: List of MatchData objects
        include_tries: If True, include try counts for bonus point calculation

    Returns:
        DataFrame with columns:
            - team: Team name
            - opponent: Opponent name
            - score: Team's score
            - opponent_score: Opponent's score
            - tries: Team's try count (if include_tries=True)
            - opponent_tries: Opponent's try count (if include_tries=True)
            - is_home: Whether team was home (True/False)
            - date: Match date

    Examples:
        >>> dataset = MatchDataset("data/")
        >>> dataset.load_json_files()
        >>> matches = [m for m in dataset.matches if m.season == "2024-2025"]
        >>> df = matches_to_league_table_format(matches)
        >>> # Now ready for LeagueTable.compute_standings(df)
    """
    rows = []

    for match in matches:
        # Skip unplayed matches
        if not match.is_played:
            continue

        # Count tries if requested
        if include_tries:
            home_tries = count_tries(match.home_scores)
            away_tries = count_tries(match.away_scores)
        else:
            home_tries = 0
            away_tries = 0

        # Home team row
        rows.append({
            'team': match.home_team,
            'opponent': match.away_team,
            'score': match.home_score,
            'opponent_score': match.away_score,
            'tries': home_tries,
            'opponent_tries': away_tries,
            'is_home': True,
            'date': match.date,
        })

        # Away team row
        rows.append({
            'team': match.away_team,
            'opponent': match.home_team,
            'score': match.away_score,
            'opponent_score': match.home_score,
            'tries': away_tries,
            'opponent_tries': home_tries,
            'is_home': False,
            'date': match.date,
        })

    return pd.DataFrame(rows)


def matches_to_fixtures_format(
    matches: List[MatchData],
    future_only: bool = True,
) -> pd.DataFrame:
    """
    Convert matches to fixtures format (one row per match, not per team).

    Args:
        matches: List of MatchData objects
        future_only: If True, only include unplayed matches

    Returns:
        DataFrame with columns:
            - home_team: Home team name
            - away_team: Away team name
            - date: Match date

    Examples:
        >>> dataset = MatchDataset("data/")
        >>> dataset.load_json_files()
        >>> matches = [m for m in dataset.matches if m.season == "2024-2025"]
        >>> fixtures = matches_to_fixtures_format(matches, future_only=True)
        >>> # Now ready for SeasonPredictor.predict_season(remaining_fixtures=fixtures)
    """
    rows = []

    for match in matches:
        # If future_only, skip played matches
        if future_only and match.is_played:
            continue

        rows.append({
            'home_team': match.home_team,
            'away_team': match.away_team,
            'date': match.date,
        })

    return pd.DataFrame(rows)


def filter_matches(
    dataset: MatchDataset,
    season: Optional[str] = None,
    competition: Optional[str] = None,
    played_only: bool = False,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[MatchData]:
    """
    Filter matches by various criteria.

    Args:
        dataset: MatchDataset to filter
        season: Season string (e.g., "2024-2025"), or None for all seasons
        competition: Competition name (case-insensitive substring match), or None for all
        played_only: If True, only return matches with scores
        date_from: Only include matches on or after this date
        date_to: Only include matches on or before this date

    Returns:
        Filtered list of MatchData objects

    Examples:
        >>> dataset = MatchDataset("data/")
        >>> dataset.load_json_files()
        >>> # Get all URC matches from 2024-2025 season that have been played
        >>> matches = filter_matches(
        ...     dataset,
        ...     season="2024-2025",
        ...     competition="celtic",
        ...     played_only=True
        ... )
    """
    matches = dataset.matches

    # Filter by season
    if season is not None:
        matches = [m for m in matches if m.season == season]

    # Filter by competition (substring match, case-insensitive)
    if competition is not None:
        comp_lower = competition.lower()
        matches = [m for m in matches if comp_lower in m.competition.lower()]

    # Filter by played status
    if played_only:
        matches = [m for m in matches if m.is_played]

    # Filter by date range
    if date_from is not None:
        matches = [m for m in matches if m.date >= date_from]

    if date_to is not None:
        matches = [m for m in matches if m.date <= date_to]

    return matches


def prepare_season_data(
    dataset: MatchDataset,
    season: str,
    competition: str,
    cutoff_date: Optional[datetime] = None,
    include_tries: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare both played matches and remaining fixtures for season prediction.

    This is a convenience function that combines filtering, conversion to league table
    format, and fixtures extraction in one call.

    Args:
        dataset: MatchDataset containing all matches
        season: Season string (e.g., "2024-2025")
        competition: Competition name (e.g., "celtic", "premiership")
        cutoff_date: Date to split played vs remaining (default: now)
        include_tries: Whether to include try counts for bonus points

    Returns:
        Tuple of (played_matches_df, remaining_fixtures_df):
            - played_matches_df: League table format (2 rows per match)
            - remaining_fixtures_df: Fixtures format (1 row per match)

    Examples:
        >>> dataset = MatchDataset("data/")
        >>> dataset.load_json_files()
        >>>
        >>> # Get data for season prediction
        >>> played, fixtures = prepare_season_data(
        ...     dataset,
        ...     season="2024-2025",
        ...     competition="celtic"
        ... )
        >>>
        >>> # Now ready for SeasonPredictor.predict_season()
        >>> prediction = season_predictor.predict_season(
        ...     played_matches=played,
        ...     remaining_fixtures=fixtures,
        ...     season="2024-2025"
        ... )
    """
    if cutoff_date is None:
        cutoff_date = datetime.now(timezone.utc)

    # Get all matches for this season/competition
    all_matches = filter_matches(
        dataset,
        season=season,
        competition=competition
    )

    # Split into played and remaining
    played_matches = [m for m in all_matches if m.is_played and m.date < cutoff_date]
    remaining_matches = [m for m in all_matches if not m.is_played or m.date >= cutoff_date]

    # Convert to appropriate formats
    played_df = matches_to_league_table_format(played_matches, include_tries=include_tries)
    fixtures_df = matches_to_fixtures_format(remaining_matches, future_only=False)

    return played_df, fixtures_df


def get_competition_summary(dataset: MatchDataset) -> pd.DataFrame:
    """
    Get summary of available competitions and seasons.

    Args:
        dataset: MatchDataset to summarize

    Returns:
        DataFrame with columns: competition, season, total_matches, played_matches

    Examples:
        >>> dataset = MatchDataset("data/")
        >>> dataset.load_json_files()
        >>> summary = get_competition_summary(dataset)
        >>> print(summary)
    """
    summaries = []

    # Group matches by competition and season
    matches_by_comp_season = {}
    for match in dataset.matches:
        key = (match.competition, match.season)
        if key not in matches_by_comp_season:
            matches_by_comp_season[key] = []
        matches_by_comp_season[key].append(match)

    # Compute summaries
    for (comp, season), matches in matches_by_comp_season.items():
        played = sum(1 for m in matches if m.is_played)
        summaries.append({
            'competition': comp,
            'season': season,
            'total_matches': len(matches),
            'played_matches': played,
            'remaining_matches': len(matches) - played,
        })

    df = pd.DataFrame(summaries)
    df = df.sort_values(['competition', 'season']).reset_index(drop=True)

    return df


def quick_standings(
    dataset: MatchDataset,
    season: str,
    competition: str,
    bonus_rules: str = "URC",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Quick one-liner to get current standings.

    Args:
        dataset: MatchDataset with loaded matches
        season: Season string
        competition: Competition name
        bonus_rules: Bonus point system ("URC", "PREMIERSHIP", "TOP14")
        top_n: Number of teams to return

    Returns:
        Standings DataFrame (top N teams)

    Examples:
        >>> dataset = MatchDataset("data/")
        >>> dataset.load_json_files()
        >>> standings = quick_standings(dataset, "2024-2025", "celtic")
        >>> print(standings[['team', 'position', 'total_points']])
    """
    from rugby_ranking.model.league_table import LeagueTable, BonusPointRules

    # Parse bonus rules
    rules_map = {
        "URC": BonusPointRules.URC,
        "PREMIERSHIP": BonusPointRules.PREMIERSHIP,
        "TOP14": BonusPointRules.TOP14,
    }
    rules = rules_map.get(bonus_rules.upper(), BonusPointRules.URC)

    # Get played matches
    matches = filter_matches(dataset, season=season, competition=competition, played_only=True)

    # Convert to league table format
    df = matches_to_league_table_format(matches, include_tries=True)

    # Compute standings
    table = LeagueTable(bonus_rules=rules)
    standings = table.compute_standings(df, opponent_tries_col='opponent_tries')

    return standings.head(top_n)


# Convenience function for notebook imports
def quick_load(data_dir: str = "../../Rugby-Data") -> MatchDataset:
    """
    Quick one-liner to load all data.

    Args:
        data_dir: Path to Rugby-Data directory

    Returns:
        Loaded MatchDataset

    Examples:
        >>> dataset = quick_load()
        >>> print(f"Loaded {len(dataset.matches)} matches")
    """
    from pathlib import Path

    dataset = MatchDataset(Path(data_dir), fuzzy_match_names=False)
    dataset.load_json_files()
    return dataset
