"""The Bayesian rating model: exact Kalman updates, uncertainty, leagues."""

import numpy as np
import pandas as pd
import pytest

from rugby_ranking.backtest import BacktestConfig, run_backtest
from rugby_ranking.forecast import ConstantBaseline, EloForecaster, Forecaster, MatchStore, RatingConfig, RatingForecaster
from rugby_ranking.forecast.base import History

from test_backtest import make_store


def hist(rows, as_of="2030-01-01"):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    store = MatchStore(df.assign(match_id=[f"m{i}" for i in range(len(df))]))
    return store.history(as_of)


def match(date, home, away, hs, as_, comp="L", season="2024-2025"):
    return dict(date=date, season=season, competition=comp, home_team=home, away_team=away,
                home_score=hs, away_score=as_)


FLAT = RatingConfig(drift_per_day=0.0, season_sd=0.0, season_shrink=1.0, league_lr=0.0)


class TestKalman:
    def test_one_match_matches_the_hand_computed_update(self):
        cfg = RatingConfig(drift_per_day=0.0, season_sd=0.0, season_shrink=1.0, league_lr=0.0,
                           prior_sd=10.0, home_advantage0=3.0, margin_sd0=12.0)
        f = RatingForecaster(cfg)
        f.fit(hist([match("2024-01-01", "A", "B", 30, 10)]))
        vh = va = 100.0
        S = vh + va + 144.0
        innovation = 20 - 3.0                                   # margin minus home advantage
        assert f.mean["A"] == pytest.approx(vh / S * innovation)
        assert f.mean["B"] == pytest.approx(-va / S * innovation)
        assert f.var["A"] == pytest.approx(vh - vh ** 2 / S)
        assert f.var["B"] == pytest.approx(va - va ** 2 / S)

    def test_evidence_shrinks_uncertainty_and_unseen_teams_are_uncertain(self):
        f = RatingForecaster(FLAT)
        f.fit(hist([match(f"2024-0{1 + i // 28}-{1 + i % 28:02d}", "A", "B", 20, 15) for i in range(60)][:50]))
        fx = pd.DataFrame([dict(match_id="x", date=pd.Timestamp("2030-01-02", tz="UTC"), season="2024-2025",
                                competition="L", home_team="A", away_team="B"),
                           dict(match_id="y", date=pd.Timestamp("2030-01-02", tz="UTC"), season="2024-2025",
                                competition="L", home_team="A", away_team="NEW")])
        known, unseen = f.predict(fx)
        assert unseen.margin_sd > known.margin_sd

    def test_probabilities_are_valid_and_symmetric_for_equal_teams(self):
        f = RatingForecaster(RatingConfig(home_advantage0=0.0))
        f.fit(hist([match("2024-01-01", "A", "B", 10, 10)]))
        fx = pd.DataFrame([dict(match_id="x", date=pd.Timestamp("2030-01-02", tz="UTC"), season="2024-2025",
                                competition="L", home_team="C", away_team="D")])
        p, = f.predict(fx)
        away = 1 - p.home_win_prob - p.draw_prob
        assert p.home_win_prob == pytest.approx(away, abs=1e-9)
        assert 0 < p.draw_prob < 0.1 and p.home_win_prob + p.draw_prob + away == pytest.approx(1.0)


class TestDynamics:
    def test_variance_grows_with_time_since_the_last_match(self):
        f = RatingForecaster(RatingConfig(drift_per_day=0.5, season_shrink=1.0, season_sd=0.0))
        f.fit(hist([match("2024-01-01", "A", "B", 20, 10)]))
        v0 = f.var["A"]
        _, v_soon = f._state_at("A", pd.Timestamp("2024-01-11", tz="UTC"), "2024-2025")
        _, v_later = f._state_at("A", pd.Timestamp("2024-03-01", tz="UTC"), "2024-2025")
        assert v0 < v_soon < v_later

    def test_a_new_season_regresses_the_rating_and_adds_turnover_variance(self):
        f = RatingForecaster(RatingConfig(drift_per_day=0.0, season_shrink=0.5, season_sd=5.0))
        f.fit(hist([match("2024-01-01", "A", "B", 40, 10)]))
        m, v = f.mean["A"], f.var["A"]
        same = f._state_at("A", pd.Timestamp("2024-02-01", tz="UTC"), "2024-2025")
        new = f._state_at("A", pd.Timestamp("2024-02-01", tz="UTC"), "2025-2026")
        assert same == (m, v)
        assert new[0] == pytest.approx(0.5 * m)
        assert new[1] == pytest.approx(0.25 * v + 25.0)

    def test_each_league_learns_its_own_home_advantage(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(400):
            comp, ha = ("big", 10.0) if i % 2 else ("small", 1.0)
            rows.append(match(pd.Timestamp("2022-01-01") + pd.Timedelta(days=i), "A" if i % 3 else "C", "B",
                              0, 0, comp=comp, season="2022-2023"))
            margin = ha + rng.normal(0, 10)
            rows[-1]["home_score"], rows[-1]["away_score"] = 30 + max(margin, 0), 30 + max(-margin, 0)
        f = RatingForecaster(RatingConfig(league_lr=0.03))
        f.fit(hist(rows))
        assert f.home_adv["big"] > f.home_adv["small"] + 4


class TestForecaster:
    def test_protocol_and_incremental_equals_fresh(self):
        store = make_store(n_rounds=60)
        assert isinstance(RatingForecaster(), Forecaster)
        inc = RatingForecaster()
        inc.fit(store.history("2021-06-01")); inc.fit(store.history("2022-03-01"))
        fresh = RatingForecaster()
        fresh.fit(store.history("2022-03-01"))
        assert inc.mean == pytest.approx(fresh.mean) and inc.var == pytest.approx(fresh.var)
        assert inc.home_adv == pytest.approx(fresh.home_adv)

    def test_resets_when_time_goes_backwards(self):
        store = make_store(n_rounds=60)
        f = RatingForecaster()
        f.fit(store.history("2022-03-01"))
        late = dict(f.mean)
        f.fit(store.history("2021-06-01"))
        assert f.mean != late

    def test_learns_the_strong_team_and_beats_baselines_on_the_harness(self):
        store = make_store()
        res = run_backtest(store, [ConstantBaseline(), EloForecaster(), RatingForecaster()],
                           BacktestConfig(start="2022-09-01", step_days=28, min_history=100), log=lambda m: None)
        assert res.compare("rating", "constant", ("brier",)).loc["brier", "hi"] < 0
        assert res.compare("rating", "elo", ("brier",)).loc["brier", "mean"] < 0.002   # at least as good as Elo
        ratings = RatingForecaster()
        ratings.fit(store.history("2030-01-01"))
        order = list(ratings.ratings()["team"])
        assert order[0] == "Strong" and order[-1] == "Weak"

    def test_no_look_ahead_predictions_ignore_results_from_the_origin_onward(self):
        store = make_store(n_rounds=80)
        origin = pd.Timestamp("2022-06-01", tz="UTC")
        fixtures = store.fixtures(origin, origin + pd.Timedelta(days=28)).loc[:, ["match_id", "date", "season", "competition", "home_team", "away_team"]]

        def forecasts(matches):
            poisoned = MatchStore(matches)
            f = RatingForecaster()
            f.fit(poisoned.history(origin))
            return [(p.home_mean, p.home_win_prob, p.margin_sd) for p in f.predict(fixtures)]

        clean = store.matches.copy()
        dirty = clean.copy()
        late = dirty["date"] >= origin
        dirty.loc[late, ["home_score", "away_score"]] = dirty.loc[late, ["home_score", "away_score"]] + 80
        assert forecasts(clean) == forecasts(dirty)
