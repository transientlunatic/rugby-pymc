"""Calibration layer: corrections learned from earlier origins only."""

import numpy as np
import pandas as pd
import pytest

from rugby_ranking.backtest import BacktestResult, CalibrationConfig, add_calibrated, metrics
from rugby_ranking.backtest.calibration import SUFFIX, calibrate_forecaster


def make_frame(n_origins=12, per_origin=40, seed=0, total_bias=None, margin_sd_pred=12.0, noise=12.0,
               drift=0.0, comps=("A", "B")):
    """Forecasts from a 'base' whose margins are unbiased, whose totals may be
    biased per competition (pred - actual = total_bias[c]), and whose stated
    margin sd may be wrong relative to the true noise."""
    total_bias = total_bias or {}
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_origins):
        for i in range(per_origin):
            c = comps[i % len(comps)]
            pred_margin = rng.normal(0, 8)
            true_total = 55 + rng.normal(0, 6)
            pred_total = true_total + total_bias.get(c, 0.0) + drift * k
            margin = pred_margin + rng.normal(0, noise)
            hs, as_ = (true_total + margin) / 2, (true_total - margin) / 2
            ph = float(1 / (1 + np.exp(-pred_margin / (margin_sd_pred / 1.7))))
            rows.append(dict(
                origin=pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=28 * k),
                forecaster="base", match_id=f"{k}-{i}", date=pd.Timestamp("2024-01-10", tz="UTC") + pd.Timedelta(days=28 * k),
                season="s", competition=c, home_team="H", away_team="A",
                home_score=hs, away_score=as_,
                home_mean=(pred_total + pred_margin) / 2, away_mean=(pred_total - pred_margin) / 2,
                home_win_prob=ph * 0.98, draw_prob=0.02, margin_sd=margin_sd_pred,
                h_lo=(pred_total + pred_margin) / 2 - 15, h_hi=(pred_total + pred_margin) / 2 + 15,
                a_lo=(pred_total - pred_margin) / 2 - 15, a_hi=(pred_total - pred_margin) / 2 + 15,
            ))
    return pd.DataFrame(rows)


def late(df, n=4):
    origins = sorted(df["origin"].unique())[-n:]
    return df[df["origin"].isin(origins)]


class TestLeakFree:
    def test_corrections_at_an_origin_ignore_that_origin_and_later_results(self):
        df = make_frame(total_bias={"A": 8.0})
        base = calibrate_forecaster(df)
        origins = sorted(df["origin"].unique())
        k = origins[6]
        poisoned = df.copy()
        future = poisoned["origin"] >= k          # this origin's own results and everything after
        poisoned.loc[future, ["home_score", "away_score"]] += 500
        again = calibrate_forecaster(poisoned)
        cols = ["home_mean", "away_mean", "home_win_prob", "margin_sd", "h_lo", "h_hi"]
        before = base[base["origin"] <= k].sort_values("match_id").reset_index(drop=True)[cols]
        after = again[again["origin"] <= k].sort_values("match_id").reset_index(drop=True)[cols]
        pd.testing.assert_frame_equal(before, after)

    def test_later_origins_do_use_earlier_results(self):
        df = make_frame(total_bias={"A": 8.0})
        poisoned = df.copy()
        poisoned.loc[poisoned["origin"] == sorted(df["origin"].unique())[0], "home_score"] += 100
        a = calibrate_forecaster(df).sort_values("match_id")["home_mean"].to_numpy()
        b = calibrate_forecaster(poisoned).sort_values("match_id")["home_mean"].to_numpy()
        assert not np.allclose(a, b)


class TestCorrections:
    def test_too_little_history_changes_nothing_but_the_name(self):
        df = make_frame(n_origins=1)
        out = calibrate_forecaster(df)
        assert (out["forecaster"] == "base" + SUFFIX).all()
        pd.testing.assert_frame_equal(
            out.drop(columns="forecaster").reset_index(drop=True),
            df.drop(columns="forecaster").sort_values(["origin", "date"]).reset_index(drop=True),
        )

    def test_level_removes_a_league_specific_scoring_bias(self):
        df = make_frame(n_origins=16, per_origin=80, total_bias={"A": 8.0})     # predicts A's totals 8 too high
        before = metrics.add_losses(late(df))
        after = metrics.add_losses(late(calibrate_forecaster(df, CalibrationConfig(margin=False, spread=False))))
        err = lambda d, c: d[d["competition"] == c]["total_err"].mean()
        assert err(before, "A") == pytest.approx(-8.0, abs=1.5)
        assert abs(err(after, "A")) < 2.5
        assert abs(err(after, "B")) < 2.0            # the well-calibrated league is not disturbed

    def test_small_leagues_are_shrunk_harder_than_large_ones(self):
        df = make_frame(n_origins=6, per_origin=60, comps=("big", "big", "big", "small"))
        df.loc[df["competition"].isin(["big", "small"]), ["home_score", "away_score"]] += 0   # no-op, keeps types
        df["home_mean"] -= 5
        df["away_mean"] -= 5                                       # 10 points too low everywhere
        out = calibrate_forecaster(df, CalibrationConfig(margin=False, spread=False, shrinkage=80.0))
        last = out[out["origin"] == out["origin"].max()]
        gain = lambda c: (last[last["competition"] == c]["home_mean"] + last[last["competition"] == c]["away_mean"]).mean() \
            - (df[(df["origin"] == df["origin"].max()) & (df["competition"] == c)][["home_mean", "away_mean"]].sum(axis=1)).mean()
        # both pull up toward the pooled correction; neither overshoots the 10 points
        assert 0 < gain("small") <= 10.5 and 0 < gain("big") <= 10.5

    def test_half_life_tracks_a_drifting_bias(self):
        df = make_frame(n_origins=16, per_origin=60, drift=1.5, seed=3)    # bias grows each origin
        err = lambda cfg: metrics.add_losses(late(calibrate_forecaster(df, cfg)))["total_err"].abs().mean()
        flat = err(CalibrationConfig(margin=False, spread=False))
        tracking = err(CalibrationConfig(margin=False, spread=False, half_life_origins=2.0))
        assert tracking < flat


class TestSpread:
    def test_overconfident_win_probabilities_are_recalibrated(self):
        df = make_frame(n_origins=14, per_origin=100, margin_sd_pred=4.0, noise=14.0, seed=2)   # claims sd 4, truth 14
        cal = calibrate_forecaster(df, CalibrationConfig(level=False, margin=False))
        raw = metrics.add_losses(late(df, 5))
        fixed = metrics.add_losses(late(cal, 5))
        assert fixed["brier"].mean() < raw["brier"].mean() - 0.01
        assert fixed["margin_z"].std() == pytest.approx(1.0, abs=0.15)
        assert raw["margin_z"].std() > 2.5

    def test_probabilities_stay_valid_and_the_draw_probability_is_kept(self):
        out = calibrate_forecaster(make_frame())
        late_rows = out[out["origin"] == out["origin"].max()]
        away = 1 - late_rows["home_win_prob"] - late_rows["draw_prob"]
        assert late_rows["home_win_prob"].between(0, 1).all() and (away >= -1e-9).all()
        assert (late_rows["draw_prob"] == 0.02).all()

    def test_intervals_widen_with_the_spread_and_missing_ones_stay_missing(self):
        df = make_frame(n_origins=10, margin_sd_pred=4.0, noise=14.0)
        out = calibrate_forecaster(df, CalibrationConfig(level=False, margin=False))
        last = out["origin"].max()
        old_w = (df[df["origin"] == last]["h_hi"] - df[df["origin"] == last]["h_lo"]).mean()
        new_w = (out[out["origin"] == last]["h_hi"] - out[out["origin"] == last]["h_lo"]).mean()
        assert new_w > 2 * old_w
        nan = df.assign(h_lo=np.nan, h_hi=np.nan, a_lo=np.nan, a_hi=np.nan)
        res = calibrate_forecaster(nan)
        assert res[["h_lo", "h_hi", "a_lo", "a_hi"]].isna().all().all()


class TestIntegration:
    def test_add_calibrated_keeps_the_originals_and_suffixes_the_new_ones(self):
        df = make_frame()
        both = add_calibrated(df)
        assert set(both["forecaster"]) == {"base", "base" + SUFFIX}
        assert len(both) == 2 * len(df)

    def test_with_calibration_and_csv_round_trip(self, tmp_path):
        res = BacktestResult.from_frame(make_frame())
        path = tmp_path / "f.csv"
        res.frame.to_csv(path, index=False)
        back = BacktestResult.from_csv(path)
        assert back.n_fixtures == res.n_fixtures
        cal = back.with_calibration()
        assert cal.forecasters == ["base", "base" + SUFFIX]
        assert cal.n_fixtures == res.n_fixtures
        assert not cal.summary().empty
