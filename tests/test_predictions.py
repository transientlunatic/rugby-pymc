"""Unit tests for rugby_ranking.model.predictions.MatchPredictor.

Uses hand-crafted (not fitted) posterior arrays with known values, wired
through arviz.from_dict, so these tests are fast and fully deterministic --
no VI/MCMC sampling involved.
"""

import numpy as np
import pandas as pd
import arviz as az
import pytest

from rugby_ranking.model.core import ModelConfig, RugbyModel
from datetime import datetime, timezone

from rugby_ranking.model.data import MatchData
from rugby_ranking.model.predictions import (
    ALL_COMPETITIONS,
    MatchPredictor,
    estimate_kicking_rates,
)


def _tiny_two_team_df():
    """One player per position (1-15) per team, one match, one score type
    (tries) -- just enough for RugbyModel.build_joint() to build its index
    mappings, with all STARTERS positions represented (predict_teams_only
    always loops over positions 1-15 regardless of how many the model
    actually saw). Actual values feeding predictions come from the
    hand-crafted trace, not from fitting on this data.
    """
    positions = list(range(1, 16))
    n = len(positions)
    return pd.DataFrame(
        {
            "player_name": [f"A{p}" for p in positions] + [f"B{p}" for p in positions],
            "team": ["TeamA"] * n + ["TeamB"] * n,
            "opponent": ["TeamB"] * n + ["TeamA"] * n,
            "season": ["2024"] * (2 * n),
            "position": positions + positions,
            "minutes_played": [80] * (2 * n),
            "is_home": [True] * n + [False] * n,
            "tries": [1] * (2 * n),
        }
    )


def _build_model_and_trace(delta_defense_by_team: dict[str, float]):
    """Build a RugbyModel (for its index mappings only) plus a
    hand-crafted single-sample trace with gamma/eta_home/theta all zeroed
    out, so any difference in predicted tries between the two teams is
    attributable *only* to delta_defense.
    """
    config = ModelConfig(
        score_types=("tries",),
        include_defense=True,
        include_player_effect=False,
    )
    model = RugbyModel(config=config)
    model.build_joint(_tiny_two_team_df())

    ts_ids = model._team_season_ids
    n_team_seasons = len(ts_ids)
    n_positions = len(model._position_ids)

    delta_defense_raw = np.zeros(n_team_seasons)
    for team, value in delta_defense_by_team.items():
        delta_defense_raw[ts_ids[(team, "2024")]] = value

    # chain=1, draw=1 -- a single deterministic posterior "sample".
    posterior = {
        "alpha": np.array([[[-2.0]]]),  # (chain, draw, n_score_types)
        "gamma_team_season_raw": np.zeros((1, 1, n_team_seasons)),
        "sigma_team": np.array([[1.0]]),
        "lambda_team": np.array([[[1.0]]]),
        "theta_position": np.zeros((1, 1, 1, n_positions)),  # (chain, draw, n_score_types, n_positions)
        "eta_home": np.array([[[0.0]]]),
        "delta_defense_raw": delta_defense_raw.reshape(1, 1, n_team_seasons),
        "sigma_defense": np.array([[1.0]]),
        "lambda_defense": np.array([[1.0]]),  # scalar RV -- no trailing shape dim
    }
    trace = az.from_dict(posterior=posterior)
    return model, trace


class TestOpponentDefenseInPredictions:
    """MatchPredictor must actually apply the model's fitted opponent
    defense term (delta_defense) -- it was previously computed by
    core.py's build_joint() but silently never read back out in
    predict_teams_only()/predict_full_lineup(), so a team's defensive
    quality had zero effect on its opponents' predicted scores.
    """

    def test_strong_defense_suppresses_opponent_tries(self):
        # TeamB's defense (5) is far stronger than TeamA's (0); attack,
        # position and home-advantage effects are all zeroed out, so any
        # score difference must come from delta_defense.
        model, trace = _build_model_and_trace({"TeamA": 0.0, "TeamB": 5.0})
        predictor = MatchPredictor(model, trace)
        assert predictor._has_defense_effect is True

        pred = predictor.predict_teams_only(
            home_team="TeamA", away_team="TeamB", season="2024", n_samples=3000
        )
        # TeamA's tries are suppressed by TeamB's strong defense;
        # TeamB's tries are barely suppressed by TeamA's weak defense.
        # Points include a fixed penalty-points baseline unrelated to
        # tries (PENALTIES_PER_MATCH), so compare with a wide margin
        # rather than near-zero absolute thresholds.
        assert pred.away.mean > 2 * pred.home.mean
        assert pred.away_win_prob > 0.7

    def test_symmetric_defense_gives_symmetric_scores(self):
        # Equal defense on both sides, with attack/position/home effects
        # already zeroed -- home and away must come out statistically
        # indistinguishable.
        model, trace = _build_model_and_trace({"TeamA": 3.0, "TeamB": 3.0})
        predictor = MatchPredictor(model, trace)

        pred = predictor.predict_teams_only(
            home_team="TeamA", away_team="TeamB", season="2024", n_samples=5000
        )
        assert pred.home.mean == pytest.approx(pred.away.mean, abs=1.0)

    def test_include_defense_false_skips_defense_entirely(self):
        # With include_defense=False, delta_defense_raw doesn't exist in
        # the posterior at all -- _has_defense_effect must be False and
        # predictions must not crash.
        config = ModelConfig(
            score_types=("tries",), include_defense=False, include_player_effect=False
        )
        model = RugbyModel(config=config)
        model.build_joint(_tiny_two_team_df())

        n_team_seasons = len(model._team_season_ids)
        n_positions = len(model._position_ids)
        posterior = {
            "alpha": np.array([[[-2.0]]]),
            "gamma_team_season_raw": np.zeros((1, 1, n_team_seasons)),
            "sigma_team": np.array([[1.0]]),
            "lambda_team": np.array([[[1.0]]]),
            "theta_position": np.zeros((1, 1, 1, n_positions)),
            "eta_home": np.array([[[0.0]]]),
        }
        trace = az.from_dict(posterior=posterior)
        predictor = MatchPredictor(model, trace)
        assert predictor._has_defense_effect is False

        pred = predictor.predict_teams_only(
            home_team="TeamA", away_team="TeamB", season="2024", n_samples=1000
        )
        assert pred.home.mean == pytest.approx(pred.away.mean, abs=1.0)


def _match(competition, season, home_events, away_events, i=0, home_score=None, away_score=None):
    """A played match. Final scores default to what the events add up to, i.e.
    a source that records everything; pass scores to model one that doesn't."""
    from rugby_ranking.sport import RUGBY_UNION

    when = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return MatchData(
        match_id=f"{competition}_{season}_{i}",
        date=when,
        season=season,
        competition=competition,
        stadium="",
        home_team="A",
        away_team="B",
        home_score=RUGBY_UNION.event_points(home_events, when)[0] if home_score is None else home_score,
        away_score=RUGBY_UNION.event_points(away_events, when)[0] if away_score is None else away_score,
        home_lineup={},
        away_lineup={},
        home_scores=home_events,
        away_scores=away_events,
    )


def _events(tries=0, convs=0, pens=0, penalty_tries=0):
    return (
        [{"type": "Try"}] * tries
        + [{"type": "Conversion"}] * convs
        + [{"type": "Penalty"}] * pens
        + [{"type": "Penalty Try"}] * penalty_tries
        + [{"type": "Missed conversion"}] * 5  # must not count as a conversion
    )


class TestEstimateKickingRates:
    def test_rates_by_competition_and_pooled(self):
        prem = [_match("prem", "2025-2026", _events(tries=2, convs=1, pens=1),
                       _events(tries=2, convs=2, pens=1, penalty_tries=1), i=i) for i in range(1)]
        prem += [_match("prem", "2025-2026", _events(), _events(), i=1)]
        t14 = [_match("t14", "2025-2026", _events(tries=1, convs=1, pens=3),
                      _events(tries=1, convs=0, pens=3), i=0)]
        rates = estimate_kicking_rates(prem + t14, min_team_games=2)
        # prem: 4 team-games, 4 kickable tries (the penalty try is excluded), 3 convs, 2 pens
        assert rates["prem"] == pytest.approx((2 / 4, 3 / 4))
        assert rates["t14"] == pytest.approx((3.0, 0.5))
        # pooled: 6 team-games, 6 kickable tries, 4 convs, 8 pens
        assert rates[ALL_COMPETITIONS] == pytest.approx((8 / 6, 4 / 6))

    def test_uses_only_recent_seasons_and_skips_thin_competitions(self):
        old = _match("prem", "2020-2021", _events(tries=1, pens=9), _events(tries=1, pens=9))
        new = _match("prem", "2025-2026", _events(tries=1, convs=1, pens=1), _events(tries=1, convs=1, pens=1))
        thin = _match("rare", "2025-2026", _events(tries=1, pens=4), _events(tries=1, pens=4))
        rates = estimate_kicking_rates([old, new, thin], n_recent_seasons=1, min_team_games=2)
        assert rates["prem"] == pytest.approx((1.0, 1.0))
        assert "rare" in rates  # exactly 2 team-games meets the minimum
        rates = estimate_kicking_rates([old, new, thin], n_recent_seasons=1, min_team_games=3)
        assert "rare" not in rates and "prem" not in rates

    def test_team_games_whose_events_miss_the_final_score_are_ignored(self):
        # A source that records tries but not kicks: 3 tries + 2 conversions
        # + a penalty scored (22) but only the tries (15) are in the events.
        tries_only = _match("super", "2025-2026", _events(tries=3), _events(tries=3), home_score=22, away_score=22)
        complete = _match("prem", "2025-2026", _events(tries=2, convs=2, pens=1), _events(tries=2, convs=1, pens=3))
        rates = estimate_kicking_rates([tries_only, complete], min_team_games=1)
        assert "super" not in rates
        # pooled uses only the complete match: 4 tries, 3 conversions, 4 penalties, 2 team-games
        assert rates[ALL_COMPETITIONS] == pytest.approx((4 / 2, 3 / 4))

    def test_penalty_try_with_its_implied_conversion_still_reconciles(self):
        m = _match("prem", "2025-2026", _events(tries=1, convs=1, penalty_tries=1), _events())
        assert m.home_score == 14                     # 5 + 2 + (5 + 2 implied)
        rates = estimate_kicking_rates([m], min_team_games=1)
        assert rates["prem"] == pytest.approx((0.0, 1.0))     # the kicked try was converted; the penalty try is excluded

    def test_no_data_gives_empty(self):
        assert estimate_kicking_rates([]) == {}
        assert estimate_kicking_rates([_match("prem", "2025-2026", [], [])]) == {}


class TestKickingRatesInPredictions:
    def test_penalty_rate_shifts_totals_per_competition(self):
        model, trace = _build_model_and_trace({"TeamA": 0.0, "TeamB": 0.0})
        low = MatchPredictor(model, trace, kicking_rates={"prem": (0.5, 0.7)})
        high = MatchPredictor(model, trace, kicking_rates={"prem": (4.5, 0.7)})
        kw = dict(home_team="TeamA", away_team="TeamB", season="2024", n_samples=20000)
        np.random.seed(0)
        p_low = low.predict_teams_only(competition="prem", **kw)
        np.random.seed(0)
        p_high = high.predict_teams_only(competition="prem", **kw)
        # +4 penalties * 3 points, per team
        assert p_high.home.mean - p_low.home.mean == pytest.approx(12.0, abs=0.5)
        assert p_high.away.mean - p_low.away.mean == pytest.approx(12.0, abs=0.5)

    def test_unknown_competition_falls_back_to_pooled_then_constants(self):
        model, trace = _build_model_and_trace({"TeamA": 0.0, "TeamB": 0.0})
        assert MatchPredictor(model, trace)._kicking_rates_for("prem") == (2.5, 0.70)
        rates = {"prem": (0.8, 0.73), ALL_COMPETITIONS: (1.2, 0.72)}
        predictor = MatchPredictor(model, trace, kicking_rates=rates)
        assert predictor._kicking_rates_for("prem") == (0.8, 0.73)
        assert predictor._kicking_rates_for("other") == (1.2, 0.72)
        assert predictor._kicking_rates_for(None) == (1.2, 0.72)


class TestStrengthRankingsKicking:
    """Reference-margin rankings use the predictor's empirical kicking rates
    (the same for both sides), not a fixed 2.5 penalties per team."""

    @staticmethod
    def rank(rates, competition=None):
        from rugby_ranking.model.strength_rankings import compute_team_rankings

        model, trace = _build_model_and_trace({"TeamA": 0.0, "TeamB": 0.0})
        predictor = MatchPredictor(model, trace, kicking_rates=rates)
        df = compute_team_rankings(predictor, season="2024", n_samples=20000, seed=0, competition=competition)
        return df.set_index("team").loc["TeamA"]

    def test_penalty_rate_sets_the_spread_not_the_mean(self):
        quiet = self.rank({ALL_COMPETITIONS: (0.2, 0.7)})
        noisy = self.rank({ALL_COMPETITIONS: (8.0, 0.7)})
        assert noisy.ref_margin_mean == pytest.approx(quiet.ref_margin_mean, abs=0.6)
        width = lambda r: r.ref_margin_upper - r.ref_margin_lower
        assert width(noisy) > width(quiet) + 4

    def test_competition_argument_selects_that_competitions_rates(self):
        rates = {ALL_COMPETITIONS: (0.2, 0.7), "kicky": (8.0, 0.7)}
        width = lambda r: r.ref_margin_upper - r.ref_margin_lower
        assert width(self.rank(rates, competition="kicky")) > width(self.rank(rates)) + 4
        # unknown competition falls back to the pooled rates
        assert width(self.rank(rates, competition="other")) == pytest.approx(width(self.rank(rates)))

    def test_without_rates_the_old_constants_apply(self):
        default = self.rank(None)
        explicit = self.rank({ALL_COMPETITIONS: (2.5, 0.70)})
        assert default.ref_margin_lower == explicit.ref_margin_lower
        assert default.ref_margin_upper == explicit.ref_margin_upper
