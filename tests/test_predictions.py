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
from rugby_ranking.model.predictions import MatchPredictor


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
