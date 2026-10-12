"""Post-hoc calibration of stored backtest forecasts, with no refitting.

A forecaster's scoring level, home advantage and spread can be wrong for a
particular league without its team strengths being wrong. This layer learns
those corrections from the forecaster's own *earlier* out-of-sample errors and
applies them to later forecasts, so a correction can be tested in seconds on
forecasts that are already stored.

Leak-free by construction: forecasts made at origin ``k`` are corrected using
only results from origins strictly before ``k`` (those matches had been played
by then). Origins with too little history come back unchanged.

Per competition ``c``, from past errors (actual - predicted):

* ``level``  - shift the predicted total by the (shrunk) mean total error;
* ``margin`` - shift the predicted margin by the (shrunk) mean margin error
  (a home-advantage correction);
* ``spread`` - re-derive the win probability from Normal(margin, s_c) with
  s_c learned from the centred margin errors, and rescale the score intervals
  to match. The draw probability is kept from the base forecast.

Estimates are shrunk toward pooled values (``shrinkage`` pseudo-matches), and
the pooled values toward zero (``pooled_shrinkage``), so a league with few
past matches borrows strength instead of overfitting. ``half_life_origins``
optionally down-weights old origins so corrections can follow drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

SUFFIX = "+cal"
_POOLED = "__pooled__"


@dataclass(frozen=True)
class CalibrationConfig:
    level: bool = True
    margin: bool = True
    spread: bool = True
    shrinkage: float = 40.0          # pseudo-matches pulling a league toward pooled
    pooled_shrinkage: float = 20.0   # pseudo-matches pulling pooled corrections toward zero
    half_life_origins: float | None = None
    min_history: int = 60            # past forecasts needed before correcting at all


def _errors(f: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "competition": f["competition"].to_numpy(),
        "tot": ((f["home_score"] + f["away_score"]) - (f["home_mean"] + f["away_mean"])).to_numpy(),
        "mar": ((f["home_score"] - f["away_score"]) - (f["home_mean"] - f["away_mean"])).to_numpy(),
        "sd": f["margin_sd"].to_numpy(),
    })


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w)) if np.sum(w) > 0 else 0.0


def fit_corrections(past: pd.DataFrame, now_index: int, origin_index: dict, cfg: CalibrationConfig) -> dict:
    """Per-competition corrections from earlier forecasts of one forecaster.

    Returns {competition or _POOLED: dict(tot, mar, s, rho)}.
    """
    e = _errors(past)
    age = now_index - past["origin"].map(origin_index).to_numpy()
    w = np.ones(len(e)) if cfg.half_life_origins is None else 0.5 ** ((age - 1) / cfg.half_life_origins)
    w = pd.Series(w, index=e.index)

    n_p = w.sum()
    lam_p = n_p / (n_p + cfg.pooled_shrinkage)
    mu_tot = lam_p * _wmean(e["tot"].to_numpy(), w.to_numpy())
    mu_mar = lam_p * _wmean(e["mar"].to_numpy(), w.to_numpy())
    centred = e["mar"] - _wmean(e["mar"].to_numpy(), w.to_numpy())
    s_pool = float(np.sqrt(_wmean((centred ** 2).to_numpy(), w.to_numpy())))
    sd_ok = e["sd"].notna()
    base_sd = _wmean(e.loc[sd_ok, "sd"].to_numpy(), w[sd_ok].to_numpy()) if sd_ok.any() else np.nan

    out = {_POOLED: dict(tot=mu_tot, mar=mu_mar, s=s_pool, base_sd=base_sd)}
    for comp, g in e.groupby("competition"):
        wc = w.loc[g.index]
        n = wc.sum()
        lam = n / (n + cfg.shrinkage)
        mean_tot, mean_mar = _wmean(g["tot"].to_numpy(), wc.to_numpy()), _wmean(g["mar"].to_numpy(), wc.to_numpy())
        var_c = _wmean(((g["mar"] - mean_mar) ** 2).to_numpy(), wc.to_numpy())
        out[comp] = dict(
            tot=mu_tot + lam * (mean_tot - mu_tot),
            mar=mu_mar + lam * (mean_mar - mu_mar),
            s=float(np.sqrt((1 - lam) * s_pool ** 2 + lam * var_c)),
            base_sd=base_sd,
        )
    return out


def apply_corrections(cur: pd.DataFrame, params: dict, cfg: CalibrationConfig) -> pd.DataFrame:
    """Corrected copy of ``cur`` (one forecaster, one origin)."""
    out = cur.copy()
    pooled = params[_POOLED]
    p = [params.get(c, pooled) for c in cur["competition"]]
    d_tot = np.array([x["tot"] for x in p]) if cfg.level else np.zeros(len(cur))
    d_mar = np.array([x["mar"] for x in p]) if cfg.margin else np.zeros(len(cur))
    s = np.array([x["s"] for x in p])

    hm, am = cur["home_mean"].to_numpy(float), cur["away_mean"].to_numpy(float)
    total, margin = hm + am + d_tot, hm - am + d_mar
    out["home_mean"], out["away_mean"] = (total + margin) / 2, (total - margin) / 2

    if cfg.spread:
        draw = np.clip(cur["draw_prob"].to_numpy(float), 0.0, 0.5)
        home_given_decisive = ndtr(margin / s)
        out["home_win_prob"] = (1 - draw) * home_given_decisive
        base_sd = cur["margin_sd"].to_numpy(float)
        out["margin_sd"] = s
        scale = np.where(np.isfinite(base_sd) & (base_sd > 0), s / base_sd, np.nan)
        for mean_col, lo, hi, old, new in (("home_mean", "h_lo", "h_hi", hm, out["home_mean"].to_numpy()),
                                            ("away_mean", "a_lo", "a_hi", am, out["away_mean"].to_numpy())):
            if cur[lo].notna().any():
                lo_w, hi_w = old - cur[lo].to_numpy(float), cur[hi].to_numpy(float) - old
                out[lo] = np.maximum(new - scale * lo_w, 0.0)
                out[hi] = new + scale * hi_w
    return out


def calibrate_forecaster(frame: pd.DataFrame, cfg: CalibrationConfig | None = None) -> pd.DataFrame:
    """Calibrated forecasts for one forecaster's rows (``forecaster`` column
    suffixed with ``+cal``). Origins with too little history are unchanged."""
    cfg = cfg or CalibrationConfig()
    f = frame.sort_values(["origin", "date"]).reset_index(drop=True)
    origins = sorted(f["origin"].unique())
    index = {o: i for i, o in enumerate(origins)}
    parts = []
    for i, o in enumerate(origins):
        cur = f[f["origin"] == o]
        past = f[f["origin"] < o]
        if len(past) < cfg.min_history:
            parts.append(cur)
        else:
            parts.append(apply_corrections(cur, fit_corrections(past, i, index, cfg), cfg))
    out = pd.concat(parts, ignore_index=True)
    out["forecaster"] = out["forecaster"] + SUFFIX
    return out


def add_calibrated(frame: pd.DataFrame, cfg: CalibrationConfig | None = None, forecasters=None) -> pd.DataFrame:
    """``frame`` plus calibrated versions of the chosen forecasters (default all)."""
    names = list(forecasters) if forecasters is not None else sorted(frame["forecaster"].unique())
    cal = [calibrate_forecaster(frame[frame["forecaster"] == n], cfg) for n in names]
    return pd.concat([frame, *cal], ignore_index=True)
