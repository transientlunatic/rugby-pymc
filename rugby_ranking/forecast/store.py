"""One tidy match table, built from the existing ``MatchDataset`` parser so
file-format quirks (column-oriented JSON, mixed timezones, name aliases) are
handled in exactly one place."""

from __future__ import annotations

from pathlib import Path

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


class MatchStore:
    """All matches (played and fixtures) as a DataFrame, plus lazy access to
    the player-level rows and parsed matches the Bayesian model needs."""

    def __init__(self, matches: pd.DataFrame, dataset=None):
        missing = set(MATCH_COLUMNS) - set(matches.columns)
        if missing:
            raise ValueError(f"matches is missing columns: {sorted(missing)}")
        m = matches.loc[:, list(MATCH_COLUMNS)].copy()
        m["date"] = pd.to_datetime(m["date"], utc=True)
        if m["match_id"].duplicated().any():
            raise ValueError("match_id must be unique")
        self.matches = m.sort_values("date", kind="stable").reset_index(drop=True)
        self._dataset = dataset
        self._player_frame: pd.DataFrame | None = None

    @classmethod
    def from_data_dir(
        cls, data_dir: str | Path, exclude_competitions=DEFAULT_EXCLUDE
    ) -> "MatchStore":
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
        return cls(frame, dataset=dataset)

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
        return self._player_frame[self._player_frame["date"] < utc(before)].copy()

    def match_objects(self, before) -> list:
        before = utc(before)
        keep = set(self.matches.loc[self.matches["date"] < before, "match_id"])
        return [m for m in self._require_dataset().matches if m.match_id in keep]
