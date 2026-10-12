"""Rolling-origin backtest: at each origin refit every forecaster on matches
before it, then score forecasts for the matches in the following window."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from rugby_ranking.backtest import metrics
from rugby_ranking.forecast.base import FIXTURE_COLUMNS, Forecaster
from rugby_ranking.forecast.store import MatchStore, utc


@dataclass
class BacktestConfig:
    start: str                       # first origin
    end: str | None = None           # last window ends here (default: last played match)
    step_days: int = 28              # origin spacing = scoring window length
    eval_competitions: tuple[str, ...] | None = None  # None = every competition
    min_history: int = 500           # skip origins with less history than this


@dataclass
class BacktestResult:
    frame: pd.DataFrame              # one row per (forecaster, match) forecast
    n_fixtures: int                  # matches scored in total (before forecasters opted out)
    failures: list[tuple] = field(default_factory=list)  # (origin, forecaster, message)

    @property
    def forecasters(self) -> list[str]:
        return sorted(self.frame["forecaster"].unique())

    def coverage(self) -> pd.Series:
        """Share of fixtures each forecaster produced a forecast for."""
        return self.frame.groupby("forecaster")["match_id"].nunique() / max(self.n_fixtures, 1)

    def summary(self, by: list[str] | None = None, paired: bool = True) -> pd.DataFrame:
        frame = metrics.common_matches(self.frame) if paired else self.frame
        return metrics.summarise(frame, by=by)

    def compare(self, a: str, b: str, metric_names: Sequence[str] = ("brier", "logloss", "abs_margin_err", "abs_total_err")) -> pd.DataFrame:
        return pd.DataFrame([metrics.paired_diff(self.frame, a, b, m) for m in metric_names]).set_index("metric")

    def calibration(self, forecaster: str) -> pd.DataFrame:
        return metrics.calibration_table(self.frame, forecaster)


def _rows(origin, fixtures: pd.DataFrame, scores: pd.DataFrame, name: str, forecasts) -> list[dict]:
    rows = []
    for fx, sc, f in zip(fixtures.itertuples(index=False), scores.itertuples(index=False), forecasts):
        if f is None:
            continue
        rows.append({
            "origin": origin, "forecaster": name,
            "match_id": fx.match_id, "date": fx.date, "season": fx.season, "competition": fx.competition,
            "home_team": fx.home_team, "away_team": fx.away_team,
            "home_score": sc.home_score, "away_score": sc.away_score,
            "home_mean": f.home_mean, "away_mean": f.away_mean,
            "home_win_prob": f.home_win_prob, "draw_prob": f.draw_prob,
            "margin_sd": f.margin_sd if f.margin_sd is not None else np.nan,
            "h_lo": f.home_interval[0] if f.home_interval else np.nan,
            "h_hi": f.home_interval[1] if f.home_interval else np.nan,
            "a_lo": f.away_interval[0] if f.away_interval else np.nan,
            "a_hi": f.away_interval[1] if f.away_interval else np.nan,
        })
    return rows


def run_backtest(
    store: MatchStore,
    forecasters: Sequence[Forecaster],
    config: BacktestConfig,
    log: Callable[[str], None] = print,
) -> BacktestResult:
    names = [f.name for f in forecasters]
    if len(set(names)) != len(names):
        raise ValueError(f"forecaster names must be unique, got {names}")

    played = store.played
    end = utc(config.end) if config.end else played["date"].max() + pd.Timedelta(seconds=1)
    step = pd.Timedelta(days=config.step_days)
    origins = pd.date_range(utc(config.start), end, freq=step, inclusive="left")

    rows: list[dict] = []
    failures: list[tuple] = []
    n_fixtures = 0
    for origin in origins:
        window_end = min(origin + step, end)
        fixtures = store.fixtures(origin, window_end)
        if config.eval_competitions is not None:
            fixtures = fixtures[fixtures["competition"].isin(config.eval_competitions)]
        history = store.history(origin)
        if fixtures.empty or len(history.matches) < config.min_history:
            continue
        # No look-ahead: history strictly precedes the origin, fixtures start at it,
        # and forecasters never see the scores.
        assert history.matches["date"].max() < origin <= fixtures["date"].min()
        n_fixtures += len(fixtures)
        asked = fixtures.loc[:, list(FIXTURE_COLUMNS)].reset_index(drop=True)
        truth = fixtures.loc[:, ["home_score", "away_score"]].reset_index(drop=True)

        for f in forecasters:
            t0 = time.time()
            try:
                f.fit(history)
                forecasts = f.predict(asked)
                if len(forecasts) != len(asked):
                    raise ValueError(f"predict returned {len(forecasts)} forecasts for {len(asked)} fixtures")
            except Exception as exc:  # one bad origin must not sink a long run
                failures.append((origin, f.name, repr(exc)))
                log(f"[{origin.date()}] {f.name}: FAILED {exc!r}")
                continue
            rows.extend(_rows(origin, asked, truth, f.name, forecasts))
            log(f"[{origin.date()}] {f.name}: {len(asked)} fixtures in {time.time() - t0:.1f}s")

    frame = pd.DataFrame(rows)
    return BacktestResult(frame=frame, n_fixtures=n_fixtures, failures=failures)
