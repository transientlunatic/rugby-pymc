"""Tests for the Forecaster protocol and rolling-origin backtest harness.

Synthetic data only: a small league where "Strong" reliably beats "Weak", so
we know what a competent forecaster must learn, and where look-ahead leaks
would be detectable.
"""

import numpy as np
import pandas as pd
import pytest

from rugby_ranking.backtest import BacktestConfig, paired_diff, run_backtest, summarise
from rugby_ranking.backtest import metrics
from rugby_ranking.forecast import ConstantBaseline, EloForecaster, Forecast, Forecaster, MatchStore

TEAMS = ["Strong", "Mid1", "Mid2", "Weak"]
STRENGTH = {"Strong": 12.0, "Mid1": 2.0, "Mid2": -2.0, "Weak": -12.0}


def make_store(n_rounds=140, seed=0) -> MatchStore:
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    start = pd.Timestamp("2020-09-01", tz="UTC")
    for rnd in range(n_rounds):
        for h in TEAMS:
            for a in TEAMS:
                if h == a or rng.random() > 0.35:
                    continue
                margin = STRENGTH[h] - STRENGTH[a] + 3 + rng.normal(0, 8)
                total = 50 + rng.normal(0, 6)
                year = (start + pd.Timedelta(days=7 * rnd)).year
                rows.append(dict(
                    match_id=f"m{i}", date=start + pd.Timedelta(days=7 * rnd, hours=i % 5),
                    season=f"{year}-{year + 1}", competition="league", home_team=h, away_team=a,
                    home_score=round((total + margin) / 2), away_score=round((total - margin) / 2),
                ))
                i += 1
    return MatchStore(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def store():
    return make_store()


CFG = dict(start="2022-09-01", step_days=28, min_history=100)


class SpyForecaster:
    """Records what the harness shows it, to prove there is no look-ahead."""
    name = "spy"

    def __init__(self):
        self.calls = []

    def fit(self, history):
        self.as_of = history.as_of
        self.max_train = history.matches["date"].max()

    def predict(self, fixtures):
        self.calls.append((self.as_of, self.max_train, fixtures["date"].min(), list(fixtures.columns)))
        return [Forecast(25, 25, 0.5, 0.0) for _ in range(len(fixtures))]


class TestHarness:
    def test_no_lookahead_and_scores_hidden(self, store):
        spy = SpyForecaster()
        run_backtest(store, [spy], BacktestConfig(**CFG), log=lambda m: None)
        assert spy.calls
        for as_of, max_train, first_fixture, cols in spy.calls:
            assert max_train < as_of <= first_fixture
            assert "home_score" not in cols and "away_score" not in cols

    def test_every_fixture_scored_once(self, store):
        res = run_backtest(store, [ConstantBaseline(), EloForecaster()], BacktestConfig(**CFG), log=lambda m: None)
        counts = res.frame.groupby("forecaster")["match_id"].nunique()
        assert (counts == res.n_fixtures).all()
        assert not res.frame.duplicated(["forecaster", "match_id"]).any()
        assert res.frame["date"].min() >= pd.Timestamp("2022-09-01", tz="UTC")

    def test_failures_are_recorded_not_fatal(self, store):
        class Boom(SpyForecaster):
            name = "boom"
            def fit(self, history):
                raise RuntimeError("nope")
        res = run_backtest(store, [Boom(), ConstantBaseline()], BacktestConfig(**CFG), log=lambda m: None)
        assert res.failures and res.failures[0][1] == "boom"
        assert res.forecasters == ["constant"]

    def test_none_forecasts_excluded_from_paired_summary(self, store):
        class Picky(SpyForecaster):
            name = "picky"
            def predict(self, fixtures):
                return [None if i % 2 else Forecast(25, 25, 0.5) for i in range(len(fixtures))]
        res = run_backtest(store, [Picky(), ConstantBaseline()], BacktestConfig(**CFG), log=lambda m: None)
        assert res.coverage()["picky"] < 0.6
        assert res.summary().groupby("forecaster")["n"].first().nunique() == 1  # same matches

    def test_duplicate_names_rejected(self, store):
        with pytest.raises(ValueError):
            run_backtest(store, [ConstantBaseline(), ConstantBaseline()], BacktestConfig(**CFG))

    def test_protocol(self):
        assert isinstance(ConstantBaseline(), Forecaster)
        assert isinstance(EloForecaster(), Forecaster)


class TestForecasters:
    def test_elo_learns_the_strong_team_and_beats_constant(self, store):
        res = run_backtest(store, [ConstantBaseline(), EloForecaster()], BacktestConfig(**CFG), log=lambda m: None)
        d = res.compare("elo", "constant", ("brier", "abs_margin_err"))
        assert d.loc["brier", "hi"] < 0          # significantly better
        assert d.loc["abs_margin_err", "hi"] < 0
        s = res.frame[res.frame["forecaster"] == "elo"]
        strong_home = s[(s["home_team"] == "Strong") & (s["away_team"] == "Weak")]
        assert strong_home["home_win_prob"].mean() > 0.8

    def test_elo_is_incremental_but_equivalent_to_a_fresh_fit(self, store):
        h1, h2 = store.history("2022-06-01"), store.history("2023-06-01")
        inc = EloForecaster()
        inc.fit(h1); inc.fit(h2)
        fresh = EloForecaster()
        fresh.fit(h2)
        # calibration differs by design (first-fit history), so compare with fixed params
        k, ha = inc.elo.k, inc.elo.home_advantage
        fresh = EloForecaster(k=k, home_advantage=ha)
        fresh.fit(h2)
        assert inc.elo.ratings == pytest.approx(fresh.elo.ratings)

    def test_elo_resets_when_time_goes_backwards(self, store):
        e = EloForecaster(k=20, home_advantage=60)
        e.fit(store.history("2023-06-01"))
        later = dict(e.elo.ratings)
        e.fit(store.history("2022-06-01"))
        assert e.elo.ratings != later

    def test_constant_baseline_matches_hand_computation(self):
        m = pd.DataFrame({
            "match_id": ["a", "b", "c", "d"], "date": pd.date_range("2024-01-01", periods=4, tz="UTC"),
            "season": "2023-2024", "competition": "x", "home_team": "A", "away_team": "B",
            "home_score": [30, 20, 10, 20], "away_score": [10, 20, 20, 10],
        })
        c = ConstantBaseline()
        c.fit(MatchStore(m).history("2025-01-01"))
        f = c.predict(m.iloc[:1])[0]
        assert f.home_mean == 20 and f.away_mean == 15
        assert f.home_win_prob == 0.5 and f.draw_prob == 0.25


class TestMetrics:
    @staticmethod
    def frame(rows):
        base = dict(origin=0, date=pd.Timestamp("2024-01-01", tz="UTC"), season="s", competition="c",
                    home_team="A", away_team="B", margin_sd=np.nan, h_lo=np.nan, h_hi=np.nan, a_lo=np.nan, a_hi=np.nan,
                    draw_prob=0.0)
        return pd.DataFrame([{**base, **r} for r in rows])

    def test_hand_computed_losses(self):
        df = self.frame([
            dict(forecaster="f", match_id="1", home_score=30, away_score=10, home_mean=28, away_mean=14, home_win_prob=0.8),
            dict(forecaster="f", match_id="2", home_score=10, away_score=20, home_mean=20, away_mean=18, home_win_prob=0.6),
        ])
        d = metrics.add_losses(df)
        # match 1: p = (0.8, 0, 0.2), outcome home -> 0.04 + 0 + 0.04
        assert d["brier"].iloc[0] == pytest.approx(0.08)
        assert d["logloss"].iloc[0] == pytest.approx(-np.log(0.8))
        assert d["margin_err"].tolist() == [6.0, -12.0]   # actual - predicted
        assert d["total_err"].tolist() == [-2.0, -8.0]    # over-predicted scoring is negative
        assert d["correct"].tolist() == [1.0, 0.0]

    def test_draw_is_excluded_from_accuracy_but_scored_by_brier(self):
        df = self.frame([dict(forecaster="f", match_id="1", home_score=10, away_score=10,
                              home_mean=20, away_mean=15, home_win_prob=0.6, draw_prob=0.1)])
        d = metrics.add_losses(df)
        assert np.isnan(d["correct"].iloc[0])
        assert d["brier"].iloc[0] == pytest.approx(0.6**2 + 0.9**2 + 0.3**2)

    def test_paired_diff_sign_and_interval(self):
        rng = np.random.default_rng(1)
        n = 200
        outcome_home = rng.random(n) < 0.7
        rows = []
        for i in range(n):
            hs, as_ = (20, 10) if outcome_home[i] else (10, 20)
            for name, p in (("sharp", 0.9 if outcome_home[i] else 0.1), ("flat", 0.5)):
                rows.append(dict(forecaster=name, match_id=str(i), home_score=hs, away_score=as_,
                                 home_mean=15, away_mean=15, home_win_prob=p))
        d = paired_diff(self.frame(rows), "sharp", "flat", "brier")
        assert d["n"] == n and d["mean"] < 0 and d["hi"] < 0
        assert paired_diff(self.frame(rows), "flat", "sharp", "brier")["lo"] > 0
        with pytest.raises(ValueError):
            paired_diff(self.frame(rows), "sharp", "flat", "nonsense")

    def test_summarise_runs_by_group(self, store):
        res = run_backtest(store, [ConstantBaseline()], BacktestConfig(**CFG), log=lambda m: None)
        s = summarise(res.frame, by=["competition"])
        assert {"accuracy", "brier", "margin_mae", "total_bias"} <= set(s.columns)
        assert s["n"].sum() == len(res.frame)


def test_store_validates_input():
    with pytest.raises(ValueError):
        MatchStore(pd.DataFrame({"match_id": ["a"]}))
    store = make_store(n_rounds=5)
    with pytest.raises(RuntimeError):
        store.player_rows(before="2030-01-01")


class TestDataQuality:
    @staticmethod
    def raw(rows):
        base = dict(season="2023-2024", competition="c", home_score=20.0, away_score=10.0)
        out = []
        for i, r in enumerate(rows):
            d = {**base, "match_id": f"m{i}", "date": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i)}
            out.append({**d, **r})
        return pd.DataFrame(out)

    def test_each_rule(self):
        from rugby_ranking.forecast.store import clean_matches
        df = self.raw([
            dict(home_team="A", away_team="B"),                                   # fine
            dict(home_team="", away_team=""),                                     # blank names
            dict(home_team="  ", away_team="B"),                                  # blank (whitespace)
            dict(home_team="C", away_team="C"),                                   # plays itself
            dict(home_team="A", away_team="D", home_score=0.0, away_score=0.0),   # scoreless placeholder
            dict(home_team="E", away_team="F", season="2011-2010"),               # odd label
        ])
        dup = df.iloc[[0]].assign(match_id="dup")                                 # same teams and kickoff
        clean, report = clean_matches(pd.concat([df, dup], ignore_index=True))
        report = report[report["issue"] != "missing_season"]   # covered separately
        issues = report.groupby("issue")["match_id"].apply(sorted).to_dict()
        assert issues == {
            "blank_team": ["m1", "m2"], "same_team": ["m3"], "duplicate": ["dup"],
            "scoreless_draw": ["m4"], "odd_season_label": ["m5"],
        }
        assert set(clean["match_id"]) == {"m0", "m4", "m5"}
        row = clean[clean["match_id"] == "m4"].iloc[0]
        assert np.isnan(row["home_score"]) and np.isnan(row["away_score"])      # unplayed, not deleted
        assert report.set_index("match_id").loc["m5", "action"] == "reported only"

    def test_clean_data_reports_nothing_and_changes_nothing(self):
        from rugby_ranking.forecast.store import clean_matches
        df = self.raw([dict(home_team="A", away_team="B"), dict(home_team="B", away_team="A")])
        clean, report = clean_matches(df)
        assert report.empty and clean.equals(df.reset_index(drop=True))

    def test_scoreless_matches_do_not_reach_forecasters(self):
        from rugby_ranking.forecast.store import clean_matches
        df = self.raw([dict(home_team="A", away_team="B", home_score=0.0, away_score=0.0)] * 1
                      + [dict(home_team="A", away_team="B")])
        clean, report = clean_matches(df)
        store = MatchStore(clean, quality=report)
        assert len(store.played) == 1
        assert store.quality_summary()["n"].sum() == 1

    def test_missing_seasons_are_reported(self):
        from rugby_ranking.forecast.store import clean_matches
        df = self.raw([
            dict(home_team="A", away_team="B", season="2019-2020"),
            dict(home_team="A", away_team="B", season="2022-2023"),
        ])
        _, report = clean_matches(df)
        gaps = report[report["issue"] == "missing_season"]["season"].tolist()
        assert gaps == ["2020-2021", "2021-2022"]


def test_reversed_season_labels_are_put_in_order():
    from rugby_ranking.model.data import normalize_season_label
    assert normalize_season_label("2011-2010") == "2010-2011"
    assert normalize_season_label("2012-2011") == "2011-2012"
    assert normalize_season_label("2023-2024") == "2023-2024"
    assert normalize_season_label("cup-2023") == "cup-2023"
