"""The score-only model recovers known team strengths from simulated scores."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from rugby_ranking.model import score_only as so  # noqa: E402


def simulate(seed=0, n_teams=8, seasons=3, base=3.1, home=0.25):
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.25, n_teams)
    dfn = rng.normal(0, 0.25, n_teams)
    rows = []
    for y in range(seasons):
        for rep in range(2):                      # home and away fixture for every pair
            for h in range(n_teams):
                for a in range(n_teams):
                    if h == a:
                        continue
                    mu_h = np.exp(base + home + att[h] - dfn[a])
                    mu_a = np.exp(base + att[a] - dfn[h])
                    rows.append(dict(competition="L", season=f"{2020 + y}-{2021 + y}", home_team=f"T{h}", away_team=f"T{a}",
                                     home_score=rng.poisson(mu_h), away_score=rng.poisson(mu_a)))
    return pd.DataFrame(rows), att, dfn


def test_prepare_indexes_two_rows_per_match():
    df, _, _ = simulate(n_teams=4, seasons=2)
    d = so.prepare(df)
    assert len(d["pts"]) == 2 * len(df)
    assert d["home"].sum() == len(df)
    assert list(d["years"]) == [2020, 2021]


def test_recovers_strength_ordering():
    df, att, dfn = simulate()
    d = so.prepare(df)
    cfg = so.ScoreOnlyConfig(vi_iterations=8000)
    _, post = so.fit(so.build(d, cfg), cfg, n_samples=200)
    out = so.summarise(d, post)
    last = out[out.season == "2022-2023"].set_index("team")
    est = np.array([last.loc[f"T{i}", "attack_mean"] for i in range(len(att))])
    assert np.corrcoef(est, att - att.mean())[0, 1] > 0.8
    # margin against the reference side follows attack minus defence
    m = np.array([last.loc[f"T{i}", "ref_margin_mean"] for i in range(len(att))])
    assert np.corrcoef(m, (att + dfn))[0, 1] > 0.85
    assert (last["ref_margin_lower"] < last["ref_margin_mean"]).all() and (last["ref_margin_mean"] < last["ref_margin_upper"]).all()
    assert (last["n_matches"] == 14).all()      # 7 opponents, home and away
