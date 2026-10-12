"""Scoring rules for backtest results (a long frame, one row per forecast)."""

from __future__ import annotations

import numpy as np
import pandas as pd

LOSSES = ("brier", "logloss", "abs_margin_err", "sq_margin_err", "abs_total_err")
_EPS = 1e-3


def add_losses(df: pd.DataFrame) -> pd.DataFrame:
    """Per-forecast losses and diagnostics. Errors are actual minus predicted,
    so a negative ``total_err`` means the forecaster over-predicted scoring."""
    d = df.copy()
    margin = d["home_score"] - d["away_score"]
    d["outcome"] = np.select([margin > 0, margin < 0], [0, 2], default=1)  # 0 home win, 1 draw, 2 away win
    ph = d["home_win_prob"].to_numpy(float)
    pd_ = d["draw_prob"].to_numpy(float)
    pa = np.clip(1.0 - ph - pd_, 0.0, 1.0)
    probs = np.column_stack([ph, pd_, pa])
    onehot = np.eye(3)[d["outcome"].to_numpy(int)]
    d["brier"] = ((probs - onehot) ** 2).sum(axis=1)
    d["logloss"] = -np.log(np.clip((probs * onehot).sum(axis=1), _EPS, 1.0))
    pred_margin = d["home_mean"] - d["away_mean"]
    d["margin_err"] = margin - pred_margin
    d["abs_margin_err"] = d["margin_err"].abs()
    d["sq_margin_err"] = d["margin_err"] ** 2
    d["total_err"] = (d["home_score"] + d["away_score"]) - (d["home_mean"] + d["away_mean"])
    d["abs_total_err"] = d["total_err"].abs()
    d["correct"] = np.where(margin != 0, (ph > pa) == (margin > 0), np.nan)
    d["home_in_ci"] = ((d["home_score"] >= d["h_lo"]) & (d["home_score"] <= d["h_hi"])).where(d["h_lo"].notna())
    d["away_in_ci"] = ((d["away_score"] >= d["a_lo"]) & (d["away_score"] <= d["a_hi"])).where(d["a_lo"].notna())
    d["margin_z"] = d["margin_err"] / d["margin_sd"]
    return d


def common_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches every forecaster in ``df`` produced a forecast for, so
    differences aren't driven by which fixtures each one could handle."""
    n = df["forecaster"].nunique()
    counts = df.groupby("match_id")["forecaster"].nunique()
    return df[df["match_id"].isin(counts[counts == n].index)]


def summarise(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean losses per forecaster (and optional grouping columns)."""
    d = add_losses(df)
    keys = ["forecaster"] + list(by or [])

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "brier": g["brier"].mean(),
            "logloss": g["logloss"].mean(),
            "margin_mae": g["abs_margin_err"].mean(),
            "margin_bias": g["margin_err"].mean(),
            "total_bias": g["total_err"].mean(),
            "total_mae": g["abs_total_err"].mean(),
            "home_ci90": g["home_in_ci"].mean(),
            "away_ci90": g["away_in_ci"].mean(),
            "margin_z_sd": g["margin_z"].std(),
        })

    return d.groupby(keys, sort=True).apply(agg, include_groups=False)


def paired_diff(
    df: pd.DataFrame, a: str, b: str, metric: str = "brier",
    n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Mean of (loss_a - loss_b) over matches both forecast, with a bootstrap
    95% interval. Negative means ``a`` is better (lower loss)."""
    if metric not in LOSSES:
        raise ValueError(f"metric must be one of {LOSSES}")
    d = add_losses(df[df["forecaster"].isin([a, b])])
    wide = d.pivot(index="match_id", columns="forecaster", values=metric).dropna()
    diff = (wide[a] - wide[b]).to_numpy()
    if len(diff) == 0:
        return {"metric": metric, "n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metric": metric, "n": int(len(diff)), "mean": float(diff.mean()), "lo": float(lo), "hi": float(hi)}


def calibration_table(df: pd.DataFrame, forecaster: str, bins=(0, .3, .4, .5, .6, .7, .8, 1.0001)) -> pd.DataFrame:
    """Predicted vs observed home-win frequency by predicted-probability bin."""
    d = add_losses(df[df["forecaster"] == forecaster])
    d = d.assign(bin=pd.cut(d["home_win_prob"], list(bins), right=False), home_won=(d["outcome"] == 0).astype(float))
    return d.groupby("bin", observed=True).agg(
        n=("home_won", "size"), predicted=("home_win_prob", "mean"), observed=("home_won", "mean")
    )
