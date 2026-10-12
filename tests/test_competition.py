"""Bayesian analysis of a forecasting competition, on synthetic losses with
known structure."""

import numpy as np
import pandas as pd
import pytest

from rugby_ranking.competition import analyse_competition, analyse_losses

FAST = dict(draws=300, tune=300, chains=2, seed=1)


def synth(effects, n_blocks=8, per_block=60, sigma=0.4, block_sd=0.0, league_effects=None,
          noise=None, seed=0, leagues=("A", "B", "C")):
    """Per-match losses: match difficulty + entrant effect + block/league effects + noise."""
    rng = np.random.default_rng(seed)
    names = list(effects)
    rows, mid = [], 0
    b_eff = rng.normal(0, block_sd, (n_blocks, len(names))) if block_sd else np.zeros((n_blocks, len(names)))
    for b in range(n_blocks):
        for i in range(per_block):
            league = leagues[i % len(leagues)]
            difficulty = rng.normal(0.7, 0.3)
            for j, n in enumerate(names):
                eps = noise(rng) if noise else rng.normal(0, sigma)
                rows.append(dict(match_id=mid, entrant=n, block=b, league=league,
                                 loss=difficulty + effects[n] + b_eff[b, j]
                                 + (league_effects or {}).get((league, n), 0.0) + eps))
            mid += 1
    return pd.DataFrame(rows)


class TestRanking:
    def test_recovers_the_true_order_and_is_confident_about_the_best(self):
        a = analyse_losses(synth({"best": -0.05, "mid": 0.0, "mid2": 0.02, "worst": 0.06}, per_block=120, sigma=0.12), **FAST)
        assert list(a.leaderboard.index) == ["best", "mid", "mid2", "worst"]
        assert a.leaderboard.loc["best", "p_best"] > 0.9
        assert a.pairwise.loc["best", "worst"] > 0.99
        # effects are differences from the average entrant, so they sum to ~0
        assert a.leaderboard["effect"].sum() == pytest.approx(0.0, abs=1e-6)
        assert a.leaderboard.loc["best", "effect"] == pytest.approx(-0.05 - 0.0075, abs=0.03)

    def test_identical_entrants_are_not_separated(self):
        a = analyse_losses(synth({"x": 0.0, "y": 0.0, "z": 0.0}, seed=4), **FAST)
        assert a.leaderboard["p_best"].max() < 0.7
        lb = a.leaderboard
        assert ((lb["lo"] < 0) & (lb["hi"] > 0)).all()

    def test_two_entrants_are_not_double_counted(self):
        df = synth({"x": 0.0, "y": 0.03}, n_blocks=6, per_block=150, seed=2)
        a = analyse_losses(df, **FAST)
        wide = df.pivot(index="match_id", columns="entrant", values="loss")
        d = wide["y"] - wide["x"]
        iid_se = d.std() / np.sqrt(len(d))                       # standard error of the mean difference
        # posterior sd of (y - x) must be close to the iid standard error, not 1/sqrt(2) of it
        lb = a.leaderboard
        eff_diff_sd = (lb.loc["y", "hi"] - lb.loc["y", "lo"]) / 3.76 * 2   # 94% interval of a half-difference
        # Counting each match twice would give about iid_se / sqrt(2). The model also allows for
        # period and league variation it cannot rule out, so it may be somewhat wider, never narrower.
        assert iid_se * 0.85 < eff_diff_sd < iid_se * 2.2


class TestDiagnostics:
    def test_split_rhat_flags_chains_that_disagree(self):
        from rugby_ranking.competition.analysis import _split_rhat

        rng = np.random.default_rng(0)
        agree = rng.normal(0, 1, (4, 500))
        disagree = agree + np.array([[0.0], [0.0], [3.0], [3.0]])        # two chains stuck elsewhere
        assert _split_rhat(agree) == pytest.approx(1.0, abs=0.02)
        assert _split_rhat(disagree) > 1.5

    def test_analysis_reports_convergence(self):
        a = analyse_losses(synth({"x": 0.0, "y": 0.02}, per_block=40), **FAST)
        assert a.diagnostics["max_rhat"] < 1.1
        assert a.diagnostics["divergences"] == 0


class TestUncertainty:
    def test_period_to_period_swings_widen_the_intervals(self):
        flat = analyse_losses(synth({"x": 0.0, "y": 0.02}, block_sd=0.0, seed=3), **FAST)
        swingy = analyse_losses(synth({"x": 0.0, "y": 0.02}, block_sd=0.08, seed=3), **FAST)
        w = lambda a: a.leaderboard.loc["y", "hi"] - a.leaderboard.loc["y", "lo"]
        assert w(swingy) > 1.25 * w(flat)
        assert swingy.block_sd > flat.block_sd

    def test_the_target_is_the_mean_loss_not_the_typical_match(self):
        # Same typical performance, but "risky" occasionally takes a huge loss
        # (a confident miss). Its mean loss is worse, and a heavy-tailed error
        # model would not see it.
        rng = np.random.default_rng(5)
        df = synth({"safe": 0.0, "risky": 0.0}, n_blocks=8, per_block=150, sigma=0.15, seed=6)
        mask = (df["entrant"] == "risky") & (rng.random(len(df)) < 0.06)
        df.loc[mask, "loss"] += 1.8
        a = analyse_losses(df, **FAST)
        assert a.leaderboard.loc["risky", "effect"] > a.leaderboard.loc["safe", "effect"]
        assert a.leaderboard.index[0] == "safe"
        assert a.leaderboard.loc["safe", "raw_mean_loss"] < a.leaderboard.loc["risky", "raw_mean_loss"]
        # the posterior effect tracks the raw mean difference
        raw = a.leaderboard["raw_mean_loss"]
        assert (a.leaderboard.loc["risky", "effect"] - a.leaderboard.loc["safe", "effect"]) == pytest.approx(
            raw["risky"] - raw["safe"], abs=0.02)

    def test_league_specific_results_are_shrunk_toward_the_overall_effect(self):
        # y looks much better than x in one league by luck of a tiny sample.
        df = synth({"x": 0.0, "y": 0.0}, n_blocks=8, per_block=90, leagues=("A", "B", "tiny"), seed=7)
        keep = ~((df["league"] == "tiny") & (df["match_id"] % 9 != 2))     # keep a third of 'tiny' (30 matches)
        df = df[keep]
        tiny_y = (df["league"] == "tiny") & (df["entrant"] == "y")
        df.loc[tiny_y, "loss"] -= 0.15
        a = analyse_losses(df, **FAST)
        raw_gap = (df[(df.league == "tiny") & (df.entrant == "y")]["loss"].mean()
                   - df[(df.league == "tiny") & (df.entrant == "x")]["loss"].mean())
        bl = a.by_league.set_index(["competition", "entrant"])
        shrunk_gap = bl.loc[("tiny", "y"), "effect"] - bl.loc[("tiny", "x"), "effect"]
        assert raw_gap < -0.1
        assert shrunk_gap > raw_gap * 0.9        # closer to zero than the raw gap (-0.15)
        assert shrunk_gap < 0


class TestInputs:
    def test_needs_complete_data_and_two_entrants(self):
        df = synth({"x": 0.0, "y": 0.0}, n_blocks=2, per_block=10)
        with pytest.raises(ValueError):
            analyse_losses(df.iloc[:-1], **FAST)
        with pytest.raises(ValueError):
            analyse_losses(df[df["entrant"] == "x"], **FAST)

    def test_frame_wrapper_uses_only_matches_every_entrant_forecast(self):
        from test_calibration import make_frame

        a, b = make_frame(n_origins=6, per_origin=30, seed=1), make_frame(n_origins=6, per_origin=30, seed=1)
        b["forecaster"] = "other"
        b = b[~b["match_id"].str.endswith("-3")]                 # 'other' misses some matches
        res = analyse_competition(pd.concat([a, b]), metric="logloss", **FAST)
        assert res.n_matches == len(b["match_id"].unique())
        assert set(res.entrants) == {"base", "other"}
        with pytest.raises(ValueError):
            analyse_competition(pd.concat([a, b]), metric="nonsense")
