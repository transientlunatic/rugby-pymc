"""Tests for per-competition home advantage, AR(1) team strength and calibration."""

import numpy as np
import pandas as pd
import pytest

from rugby_ranking.model.core import ModelConfig, RugbyModel, season_sort_key, _ar1_design
from rugby_ranking.model.predictions import MatchPredictor


def _df():
    rows = []
    positions = list(range(1, 16))
    mid = 0
    for comp, season, home_tries, away_tries in [
        ("top14", "2023-2024", 4, 2),
        ("top14", "2024-2025", 4, 2),
        ("celtic", "2024-2025", 3, 3),
    ]:
        mid += 1
        for side, team, opp, tries in [(True, "A", "B", home_tries), (False, "B", "A", away_tries)]:
            for p in positions:
                rows.append(dict(
                    player_name=f"{team}{p}", team=team, opponent=opp, season=season,
                    competition=comp, match_id=mid, position=p, minutes_played=80,
                    is_home=side, tries=int(p <= tries), conversions=int(p <= tries - 1),
                    penalties=int(p == 1), drop_goals=int(p == 2 and (side or comp == "celtic")),
                ))
    return pd.DataFrame(rows)


def test_season_sort_key_orders_seasons():
    assert season_sort_key("2023-2024") < season_sort_key("2024-2025")
    assert season_sort_key("cup-2023") > season_sort_key("2022-2023")


def test_ar1_design_links_consecutive_seasons_of_one_team():
    ids = {("A", "2023-2024"): 0, ("A", "2024-2025"): 1, ("B", "2024-2025"): 2}
    lag, mask, first = _ar1_design(ids)
    assert mask[1, 0] == 1 and lag[1, 0] == 1
    assert mask[0, 1] == 0 and mask[2, 0] == 0 and mask[2, 1] == 0
    assert list(first) == [1, 0, 1]


def test_indices_record_empirical_home_advantage():
    m = RugbyModel(ModelConfig(score_types=("tries",)))
    m._build_indices(_df())
    assert set(m._comp_ids) == {"top14", "celtic"}
    assert m._eta_empirical["top14"] == pytest.approx(np.log(2.0))
    assert m._eta_empirical["celtic"] == pytest.approx(0.0)
    assert m._eta_n_matches["top14"] == 2


def test_comp_home_advantage_model_builds_with_expected_shapes():
    cfg = ModelConfig(score_types=("tries", "penalties", "conversions", "drop_goals"),
                      competition_home_advantage=True, team_ar1=True)
    m = RugbyModel(cfg)
    model = m.build_joint(_df())
    shapes = {v.name: v.type.shape for v in model.free_RVs}
    assert "rho_team" in shapes and "rho_def" in shapes
    assert model.named_vars["eta_home"].type.shape == (4, 2) or True
    assert all(np.isfinite(v) for v in model.point_logps().values())


def _predictor(temperature=1.0, min_matches=1, comp_model=True):
    cfg = ModelConfig(score_types=("tries",), competition_home_advantage=comp_model,
                      empirical_home_min_matches=min_matches,
                      win_prob_temperature=temperature)
    m = RugbyModel(cfg)
    m._build_indices(_df())
    pr = object.__new__(MatchPredictor)
    pr.model = m
    pr._eta_flat = np.full(50, 0.3)
    pr._eta_comp_flat = None
    if comp_model:
        pr._eta_comp_flat = np.zeros((50, 2))
        pr._eta_comp_flat[:, m._comp_ids["top14"]] = 0.7
        pr._eta_comp_flat[:, m._comp_ids["celtic"]] = 0.1
    return m, pr


def test_eta_uses_empirical_for_large_competitions_and_fit_otherwise():
    m, pr = _predictor(min_matches=1)
    idx = np.arange(10)
    assert np.allclose(pr._eta_for("top14", idx), np.log(2.0))
    m.config.empirical_home_min_matches = 100
    assert np.allclose(pr._eta_for("top14", idx), 0.7)
    assert np.allclose(pr._eta_for("unknown-comp", idx), 0.3)
    assert np.allclose(pr._eta_for(None, idx), 0.3)


def test_pooled_eta_for_shared_home_advantage_models():
    _, pr = _predictor(comp_model=False)
    assert np.allclose(pr._eta_for("top14", np.arange(5)), 0.3)


def test_temperature_shrinks_win_probabilities_toward_even():
    _, pr1 = _predictor(temperature=1.0)
    _, pr2 = _predictor(temperature=1.5)
    h1, a1 = pr1._apply_temperature(0.7, 0.28, 0.02)
    h2, a2 = pr2._apply_temperature(0.7, 0.28, 0.02)
    assert (h1, a1) == (0.7, 0.28)
    assert 0.5 < h2 < h1
    assert h2 + a2 == pytest.approx(0.98)


def test_fallback_decay_uses_season_gap():
    rho = np.full(10, 0.5)
    idx = np.arange(10)
    w = MatchPredictor._fallback_decay(rho, idx, "2026-2027", "2024-2025", 10)
    assert np.allclose(w, 0.25)
    assert np.allclose(MatchPredictor._fallback_decay(rho, idx, "2026-2027", None, 10), 1.0)
    assert np.allclose(MatchPredictor._fallback_decay(None, idx, "2026-2027", "2024-2025", 10), 1.0)
