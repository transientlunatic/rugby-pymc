"""One tidy match table, built from the existing ``MatchDataset`` parser so
file-format quirks (column-oriented JSON, mixed timezones, name aliases) are
handled in exactly one place."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rugby_ranking.forecast.base import History

MATCH_COLUMNS = (
    "match_id", "date", "season", "competition",
    "home_team", "away_team", "home_score", "away_score",
)

def utc(ts) -> pd.Timestamp:
    """Any timestamp-like as a UTC-aware Timestamp."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# Competitions that aren't a club/season league and break the season logic.
DEFAULT_EXCLUDE = frozenset({"rugby-world"})


QUALITY_COLUMNS = ("match_id", "competition", "season", "issue", "action", "detail")


def clean_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply conservative fixes for known problems in the source data and
    return (clean matches, report of everything touched).

    Excluded: blank team names (international fixtures whose names failed to
    parse), a team playing itself, and exact duplicates (same teams and
    kickoff). Scoreless 0-0 results are kept as *unplayed*: in senior
    rugby union they essentially never happen, and in the data they are
    cancelled or placeholder fixtures (e.g. Covid-era). Season labels whose
    years aren't consecutive are reported but left alone.
    """
    m = matches.copy()
    rows: list[dict] = []

    def note(mask: pd.Series, issue: str, action: str, detail=""):
        for r in m[mask].itertuples(index=False):
            rows.append(dict(match_id=r.match_id, competition=r.competition, season=r.season,
                             issue=issue, action=action, detail=detail))

    blank = m["home_team"].str.strip().eq("") | m["away_team"].str.strip().eq("")
    note(blank, "blank_team", "excluded")
    m = m[~blank]

    same = m["home_team"] == m["away_team"]
    note(same, "same_team", "excluded")
    m = m[~same]

    dup = m.duplicated(["home_team", "away_team", "date"], keep="first")
    note(dup, "duplicate", "excluded", "same teams and kickoff as an earlier row")
    m = m[~dup]

    scoreless = (m["home_score"] == 0) & (m["away_score"] == 0)
    note(scoreless, "scoreless_draw", "treated as unplayed")
    m = m.copy()
    m.loc[scoreless, ["home_score", "away_score"]] = np.nan

    years = m["season"].str.extract(r"(\d{4})-(\d{4})").astype(float)
    odd = years[1].notna() & (years[1] != years[0] + 1)
    note(odd, "odd_season_label", "reported only", "season years are not consecutive")

    # Seasons absent between a competition's first and last (informational).
    start = m["season"].str.extract(r"(\d{4})")[0].astype(float)
    for comp, g in m.assign(start=start).groupby("competition"):
        have = set(g["start"].dropna().astype(int))
        for year in range(min(have, default=0), max(have, default=-1) + 1):
            if year not in have:
                rows.append(dict(match_id=None, competition=comp, season=f"{year}-{year + 1}",
                                 issue="missing_season", action="reported only",
                                 detail="no matches between the competition's first and last season"))

    report = pd.DataFrame(rows, columns=list(QUALITY_COLUMNS))
    return m.reset_index(drop=True), report


class MatchStore:
    """All matches (played and fixtures) as a DataFrame, plus lazy access to
    the player-level rows and parsed matches the Bayesian model needs."""

    def __init__(self, matches: pd.DataFrame, dataset=None, quality: pd.DataFrame | None = None):
        missing = set(MATCH_COLUMNS) - set(matches.columns)
        if missing:
            raise ValueError(f"matches is missing columns: {sorted(missing)}")
        m = matches.loc[:, list(MATCH_COLUMNS)].copy()
        m["date"] = pd.to_datetime(m["date"], utc=True)
        if m["match_id"].duplicated().any():
            raise ValueError("match_id must be unique")
        self.matches = m.sort_values("date", kind="stable").reset_index(drop=True)
        self._dataset = dataset
        self.quality = quality if quality is not None else pd.DataFrame(columns=list(QUALITY_COLUMNS))
        self._player_frame: pd.DataFrame | None = None

    @classmethod
    def from_data_dir(
        cls, data_dir: str | Path, exclude_competitions=DEFAULT_EXCLUDE, clean: bool = True
    ) -> "MatchStore":
        """Build from a Rugby-Data checkout. With ``clean`` (default) known data
        problems are excluded or flagged; see ``clean_matches`` and ``quality``."""
        from rugby_ranking.model.data import MatchDataset, normalize_team_name

        dataset = MatchDataset(str(data_dir), fuzzy_match_names=False)
        dataset.load_json_files()
        dataset.matches = [m for m in dataset.matches if m.competition not in exclude_competitions]
        frame = pd.DataFrame(
            {
                "match_id": m.match_id,
                "date": m.date,
                "season": m.season,
                "competition": m.competition,
                "home_team": normalize_team_name(m.home_team),
                "away_team": normalize_team_name(m.away_team),
                "home_score": m.home_score,
                "away_score": m.away_score,
            }
            for m in dataset.matches
        )
        quality = None
        if clean:
            frame, quality = clean_matches(frame)
        return cls(frame, dataset=dataset, quality=quality)

    def quality_summary(self) -> pd.DataFrame:
        """Counts of data-quality issues by kind and competition."""
        if self.quality.empty:
            return pd.DataFrame(columns=["issue", "action", "competition", "n"])
        return (self.quality.groupby(["issue", "action", "competition"]).size()
                .rename("n").reset_index())

    @property
    def played(self) -> pd.DataFrame:
        return self.matches[self.matches["home_score"].notna() & self.matches["away_score"].notna()]

    def history(self, as_of) -> History:
        as_of = utc(as_of)
        past = self.played[self.played["date"] < as_of]
        return History(as_of=as_of, matches=past, store=self)

    def fixtures(self, start, end) -> pd.DataFrame:
        """Played matches with start <= date < end (the ones a backtest scores)."""
        p = self.played
        return p[(p["date"] >= utc(start)) & (p["date"] < utc(end))]

    # -- player-level data, only needed by the Bayesian adapter ---------------

    def _require_dataset(self):
        if self._dataset is None:
            raise RuntimeError("this MatchStore was built from a frame, without lineup data")
        return self._dataset

    def player_rows(self, before) -> pd.DataFrame:
        if self._player_frame is None:
            df = self._require_dataset().to_dataframe(played_only=True)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            self._player_frame = df
        f = self._player_frame
        # only matches that survived cleaning and count as played
        f = f[f["match_id"].isin(set(self.played["match_id"])) & (f["date"] < utc(before))]
        return f.copy()

    def match_objects(self, before) -> list:
        before = utc(before)
        keep = set(self.matches.loc[self.matches["date"] < before, "match_id"])
        return [m for m in self._require_dataset().matches if m.match_id in keep]
