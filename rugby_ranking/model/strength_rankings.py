"""Strength-based team and player rankings derived from the Bayesian model.

Three complementary team ranking approaches:
  1. Attack strength  – posterior mean offensive team effect (log scale)
  2. Defensive strength – posterior mean defensive team effect (log scale)
  3. Reference margin  – posterior predictive points margin vs an average team
                         at a neutral ground (interpretable units: points)

Player rankings:
  - VORP (Value Over Replacement Player): expected scoring contribution per
    80 minutes above a positional-average replacement player, in points.
    "Finn Russell contributes +4.2 points per 80 minutes above an average
    international fly-half."

  - Squad criticality: expected reduction in match margin when a starter is
    replaced by a positional-average player (β=0). Ranks players within a
    squad by how much the team depends on them.

The reference team is the model prior mean: γ_attack = 0, δ_defense = 0.
This represents the average team across *all* seasons and competitions in the
training data.  To create an "average international" reference, pass
``reference_gamma`` / ``reference_delta`` computed from international teams'
posterior means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rugby_ranking.model.core import RugbyModel
from rugby_ranking.model.predictions import (
    CONVERSION_RATE,
    PENALTIES_PER_MATCH,
    STARTERS,
    MatchPredictor,
    _get_team_season_index_with_fallback,
)

# Points per scoring event
_SCORE_POINTS = {
    "tries": 5,
    "penalties": 3,
    "conversions": 2,
    "drop_goals": 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: extract full posterior arrays from a fitted MatchPredictor
# ─────────────────────────────────────────────────────────────────────────────


def _get_alpha_all(predictor: MatchPredictor) -> np.ndarray:
    """Return alpha as (N, n_score_types) where N = total posterior draws."""
    alpha_raw = predictor.trace.posterior["alpha"].values
    if alpha_raw.ndim == 3:
        # Joint model: (chain, draw, n_score_types)
        return alpha_raw.reshape(-1, alpha_raw.shape[-1])
    # Single score type: broadcast to (N, 1)
    return alpha_raw.flatten()[:, np.newaxis]


def _get_theta_all(predictor: MatchPredictor) -> np.ndarray:
    """Return theta_position as (N, n_score_types, n_positions)."""
    theta_raw = predictor.trace.posterior["theta_position"].values
    if theta_raw.ndim == 4:
        # Joint: (chain, draw, n_score_types, n_positions)
        return theta_raw.reshape(-1, theta_raw.shape[2], theta_raw.shape[3])
    # Single score type: (chain, draw, n_positions) → (N, 1, n_positions)
    return theta_raw.reshape(-1, 1, theta_raw.shape[-1])


def _get_delta_flat(predictor: MatchPredictor) -> np.ndarray | None:
    """Return defensive effect as (N, n_team_seasons) for the tries dimension.

    Returns None if the model has no defensive component.
    """
    posterior = predictor.trace.posterior
    if "delta_defense_raw" in posterior:
        delta_raw = posterior["delta_defense_raw"].values       # (ch, dr, n_ts)
        sigma_def = posterior["sigma_defense"].values           # (ch, dr)
        lambda_def = posterior["lambda_defense"].values
        # lambda_defense may be (ch, dr) or (ch, dr, n_score_types)
        if lambda_def.ndim == 2:
            # Single loading — same for all score types
            delta_eff = sigma_def * lambda_def           # (ch, dr)
            delta_eff = delta_eff[:, :, None] * delta_raw
        else:
            # Per-score-type loading: use tries (index 0)
            delta_eff = sigma_def[:, :, None] * lambda_def[:, :, 0:1] * delta_raw
        return delta_eff.reshape(-1, delta_raw.shape[-1])       # (N, n_ts)
    if "delta_defense" in posterior:
        d = posterior["delta_defense"].values
        return d.reshape(-1, d.shape[-1])
    return None


def _get_beta_try_kick(
    model: RugbyModel, trace
) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta_try, beta_kick) each shaped (N, n_players).

    beta_try  = effective try-scoring player effect (scaled, tries dim)
    beta_kick = effective kicking player effect (scaled, penalties dim)
    """
    posterior = trace.posterior
    n_players = len(model._player_ids)

    if "beta_player_try_raw" in posterior:
        beta_try_raw = posterior["beta_player_try_raw"].values
        sigma_try = posterior["sigma_player_try"].values
        lambda_try = posterior["lambda_player_try"].values   # (ch, dr, n_st)
        beta_try = (
            sigma_try[:, :, None] * lambda_try[:, :, 0:1] * beta_try_raw
        ).reshape(-1, n_players)

        beta_kick_raw = posterior["beta_player_kick_raw"].values
        sigma_kick = posterior["sigma_player_kick"].values
        lambda_kick = posterior["lambda_player_kick"].values  # (ch, dr, n_st)
        # Use penalties as the representative kicking dimension.
        # Look up the actual index to be robust to score-type reordering.
        _score_types = list(model.config.score_types)
        _pen_idx = _score_types.index("penalties") if "penalties" in _score_types else 1
        beta_kick = (
            sigma_kick[:, :, None] * lambda_kick[:, :, _pen_idx:_pen_idx + 1] * beta_kick_raw
        ).reshape(-1, n_players)

    elif "beta_player_raw" in posterior:
        beta_raw = posterior["beta_player_raw"].values
        sigma = posterior["sigma_player"].values
        lambda_p = posterior["lambda_player"].values          # (ch, dr, n_st)
        beta_try = (
            sigma[:, :, None] * lambda_p[:, :, 0:1] * beta_raw
        ).reshape(-1, n_players)
        beta_kick = (
            sigma[:, :, None] * lambda_p[:, :, 1:2] * beta_raw
        ).reshape(-1, n_players)

    else:
        beta = posterior["beta_player"].values.reshape(-1, n_players)
        beta_try = beta_kick = beta

    return beta_try, beta_kick


# ─────────────────────────────────────────────────────────────────────────────
# Team reference-margin ranking
# ─────────────────────────────────────────────────────────────────────────────


def _simulate_reference_margin(
    predictor: MatchPredictor,
    team_idx: int,
    delta_flat: np.ndarray | None,
    n_samples: int,
    rng: np.random.Generator,
    reference_gamma: float = 0.0,
    reference_delta: float = 0.0,
) -> np.ndarray:
    """Simulate (n_samples,) points-margin distribution for one team vs reference.

    Team T attacks against the reference's defense (``reference_delta``).
    The reference attacks against team T's defense (``delta_flat[:, team_idx]``).
    No home advantage for either side (neutral ground).
    Both sides' player lineups are marginalised (sampled from the prior).
    """
    sample_idx = rng.choice(
        predictor._n_total, size=n_samples, replace=predictor._n_total < n_samples
    )

    alpha = predictor._alpha_flat[sample_idx]           # (n,)
    gamma = predictor._gamma_flat[sample_idx]           # (n, n_ts)
    theta = predictor._theta_flat[sample_idx]           # (n, n_pos)
    sigma_player = predictor._sigma_player_flat[sample_idx]  # (n,)

    team_gamma = gamma[:, team_idx]                     # (n,)

    # Defensive effect the focal team exerts on the reference's scoring
    team_delta = np.zeros(n_samples)
    if delta_flat is not None:
        team_delta = delta_flat[sample_idx, team_idx]

    # Marginalise over unknown player lineups
    noise_team = rng.normal(0.0, sigma_player[:, None], size=(n_samples, STARTERS))
    noise_ref = rng.normal(0.0, sigma_player[:, None], size=(n_samples, STARTERS))

    team_try_rate = np.zeros(n_samples)
    ref_try_rate = np.zeros(n_samples)

    for pos in range(STARTERS):
        team_try_rate += np.exp(
            alpha + team_gamma + theta[:, pos] + noise_team[:, pos] - reference_delta
        )
        ref_try_rate += np.exp(
            alpha + reference_gamma + theta[:, pos] + noise_ref[:, pos] - team_delta
        )

    team_tries = rng.poisson(team_try_rate)
    ref_tries = rng.poisson(ref_try_rate)

    team_conv = rng.binomial(team_tries, CONVERSION_RATE)
    ref_conv = rng.binomial(ref_tries, CONVERSION_RATE)

    team_pen = rng.poisson(PENALTIES_PER_MATCH, size=n_samples)
    ref_pen = rng.poisson(PENALTIES_PER_MATCH, size=n_samples)

    team_score = team_tries * 5 + team_conv * 2 + team_pen * 3
    ref_score = ref_tries * 5 + ref_conv * 2 + ref_pen * 3

    return (team_score - ref_score).astype(float)


def compute_team_rankings(
    predictor: MatchPredictor,
    season: str,
    teams: list[str] | None = None,
    n_samples: int = 2000,
    ci: float = 0.90,
    seed: int | None = None,
    reference_gamma: float = 0.0,
    reference_delta: float = 0.0,
) -> pd.DataFrame:
    """Compute combined team rankings for a given season.

    Returns one row per team, sorted by ``ref_margin_mean`` (highest first).

    Columns
    -------
    team, season
    attack_mean / attack_lower / attack_upper
        Posterior offensive team effect on the log-rate scale.
        Larger = team creates more scoring opportunities.
    defense_mean / defense_lower / defense_upper
        Posterior defensive team effect on the log-rate scale.
        Larger = team suppresses opponent scoring more.
    ref_margin_mean / ref_margin_lower / ref_margin_upper
        Posterior predictive points margin vs the reference team at a neutral
        ground.  Positive = team beats the reference on average.
    win_prob_vs_ref
        Probability that the team beats the reference team.

    Parameters
    ----------
    predictor:
        Fitted ``MatchPredictor`` (provides cached posterior arrays).
    season:
        Season string e.g. ``"2024-2025"``.  Falls back to each team's most
        recent season if the exact season is not in the trace.
    teams:
        If provided, restrict output to these teams (matched after
        normalisation).
    n_samples:
        Posterior draws used in the reference-margin simulation.
    ci:
        Credible interval width (default 90 %).
    seed:
        Random seed for reproducibility.
    reference_gamma, reference_delta:
        Attack and defensive effect of the reference team (log scale).
        Default 0.0 = prior mean = "average team across all data".
        Pass the posterior mean over international teams to use an
        "average international" reference instead.
    """
    model = predictor.model
    rng = np.random.default_rng(seed)
    lo = (1.0 - ci) / 2 * 100
    hi = 100.0 - lo

    delta_flat = _get_delta_flat(predictor)
    gamma_flat = predictor._gamma_flat   # (N, n_team_seasons)

    # Collect all team-season pairs for the requested season
    all_pairs = list(model._team_season_ids.items())   # ((team, season), idx)
    target = [(t, s, idx) for (t, s), idx in all_pairs if s == season]

    if not target:
        # Fallback: each team's most recent season
        by_team: dict[str, list] = {}
        for (t, s), idx in all_pairs:
            by_team.setdefault(t, []).append((t, s, idx))
        target = [max(v, key=lambda x: x[1]) for v in by_team.values()]

    if teams is not None:
        from rugby_ranking.model.data import normalize_team_name
        norm = {normalize_team_name(t) for t in teams}
        target = [x for x in target if x[0] in norm]

    rows = []
    for team, s, ts_idx in target:
        # ── Attack (log scale) ───────────────────────────────────────────────
        att = gamma_flat[:, ts_idx]
        att_mean = float(att.mean())
        att_lo = float(np.percentile(att, lo))
        att_hi = float(np.percentile(att, hi))

        # ── Defense (log scale) ─────────────────────────────────────────────
        if delta_flat is not None:
            dfl = delta_flat[:, ts_idx]
            def_mean = float(dfl.mean())
            def_lo = float(np.percentile(dfl, lo))
            def_hi = float(np.percentile(dfl, hi))
        else:
            def_mean = def_lo = def_hi = float("nan")

        # ── Reference margin (points) ────────────────────────────────────────
        margin = _simulate_reference_margin(
            predictor, ts_idx, delta_flat, n_samples, rng,
            reference_gamma=reference_gamma,
            reference_delta=reference_delta,
        )
        ref_mean = float(margin.mean())
        ref_lo = float(np.percentile(margin, lo))
        ref_hi = float(np.percentile(margin, hi))
        win_prob = float((margin > 0).mean())

        rows.append({
            "team": team,
            "season": s,
            "attack_mean": att_mean,
            "attack_lower": att_lo,
            "attack_upper": att_hi,
            "defense_mean": def_mean,
            "defense_lower": def_lo,
            "defense_upper": def_hi,
            "ref_margin_mean": ref_mean,
            "ref_margin_lower": ref_lo,
            "ref_margin_upper": ref_hi,
            "win_prob_vs_ref": win_prob,
        })

    result = pd.DataFrame(rows)
    return result.sort_values("ref_margin_mean", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Player VORP
# ─────────────────────────────────────────────────────────────────────────────


def _get_primary_positions(df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    """Return a DataFrame with the primary position and match count per player."""
    pos_df = df[df["position"].notna() & (df["position"] > 0)].copy()
    pos_df["position"] = pos_df["position"].astype(int)

    counts = (
        pos_df.groupby(["player_name", "position"])
        .size()
        .reset_index(name="count")
    )
    # Primary = highest count; ties broken by smallest jersey number
    primary = (
        counts
        .sort_values(["player_name", "count", "position"], ascending=[True, False, True])
        .groupby("player_name")
        .first()
        .reset_index()[["player_name", "position"]]
        .rename(columns={"position": "primary_position"})
    )
    n_matches = df.groupby("player_name").size().reset_index(name="n_matches")
    result = primary.merge(n_matches, on="player_name", how="left")
    return result[result["n_matches"] >= min_matches]


def compute_player_vorp(
    model: RugbyModel,
    trace,
    df: pd.DataFrame,
    n_samples: int = 2000,
    min_matches: int = 5,
    ci: float = 0.90,
    seed: int | None = None,
    top_n: int | None = None,
    team_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Compute VORP (Value Over Replacement Player) for all eligible players.

    VORP measures expected additional points scored per 80 minutes above a
    positional-average replacement player, assuming an average team system
    (γ_team = 0) and a full 80-minute exposure.

    Formula (per score type *s*, at average team, for player *i* at position *p*):

        rate_player(s)      = exp(α_s + β_type(s)[i] + θ_s[p])
        rate_replacement(s) = exp(α_s          + θ_s[p])        # β = 0
        VORP_s              = (rate_player(s) − rate_replacement(s)) × pts_s

        VORP = Σ_s VORP_s

    where ``type(s) = "try"`` for tries, ``"kick"`` for all other score types.

    Parameters
    ----------
    model:
        Fitted ``RugbyModel`` with ``_player_ids`` and ``_position_ids`` populated.
    trace:
        ArviZ ``InferenceData`` from MCMC or VI.
    df:
        DataFrame from ``MatchDataset.to_dataframe()``, used to determine
        each player's primary position and match count.
    n_samples:
        Number of posterior draws to use.
    min_matches:
        Minimum number of player-match appearances to be included.
    ci:
        Credible interval width (default 90 %).
    seed:
        Random seed.
    top_n:
        If provided, return only the top-N players by mean VORP.
    team_filter:
        If provided, restrict to players who appeared for any of these teams.

    Returns
    -------
    DataFrame sorted by ``vorp_mean`` descending, with columns:
    player, primary_position, n_matches,
    vorp_mean, vorp_lower, vorp_upper,
    vorp_tries_mean, vorp_kicking_mean.
    """
    rng = np.random.default_rng(seed)
    posterior = trace.posterior
    lo = (1.0 - ci) / 2 * 100
    hi = 100.0 - lo

    # ── Player meta ──────────────────────────────────────────────────────────
    player_meta = _get_primary_positions(df, min_matches)

    if team_filter is not None:
        from rugby_ranking.model.data import normalize_team_name
        norm_teams = {normalize_team_name(t) for t in team_filter}
        players_for_teams = set(df[df["team"].isin(norm_teams)]["player_name"])
        player_meta = player_meta[player_meta["player_name"].isin(players_for_teams)]

    # ── Posterior arrays ─────────────────────────────────────────────────────
    # Build a temporary predictor-like object to reuse _get_alpha_all / _get_theta_all
    # but we actually need to create a temporary predictor or just use the trace directly.
    # We work directly from the trace to keep this function standalone.

    alpha_raw = posterior["alpha"].values
    if alpha_raw.ndim == 3:
        alpha_all = alpha_raw.reshape(-1, alpha_raw.shape[-1])   # (N, n_st)
    else:
        alpha_all = alpha_raw.flatten()[:, np.newaxis]            # (N, 1)
    N_total = alpha_all.shape[0]
    n_score_types = alpha_all.shape[1]

    theta_raw = posterior["theta_position"].values
    if theta_raw.ndim == 4:
        # (chain, draw, n_score_types, n_positions)
        theta_all = theta_raw.reshape(-1, theta_raw.shape[2], theta_raw.shape[3])
    else:
        theta_all = theta_raw.reshape(-1, 1, theta_raw.shape[-1])

    beta_try, beta_kick = _get_beta_try_kick(model, trace)

    score_types = model.config.score_types   # e.g. ("tries","penalties","conversions","drop_goals")

    # Draw sample indices once
    sample_idx = rng.choice(N_total, size=n_samples, replace=N_total < n_samples)
    alpha_s = alpha_all[sample_idx]      # (n_samples, n_st)
    theta_s = theta_all[sample_idx]      # (n_samples, n_st, n_pos)
    beta_try_s = beta_try[sample_idx]    # (n_samples, n_players)
    beta_kick_s = beta_kick[sample_idx]  # (n_samples, n_players)

    pos_id_map = model._position_ids     # jersey number → 0-indexed model position

    rows = []
    for _, row in player_meta.iterrows():
        pname = row["player_name"]
        primary_pos = int(row["primary_position"])
        n_matches = int(row["n_matches"])

        if pname not in model._player_ids:
            continue
        if primary_pos not in pos_id_map:
            continue

        pidx = model._player_ids[pname]
        pos_model_idx = pos_id_map[primary_pos]

        vorp_samples = np.zeros(n_samples)
        vorp_tries = np.zeros(n_samples)
        vorp_kicking = np.zeros(n_samples)

        for s_idx, s_name in enumerate(score_types):
            if s_idx >= n_score_types:
                break
            pts = _SCORE_POINTS.get(s_name, 3)
            a = alpha_s[:, s_idx]                                # (n_samples,)
            th = theta_s[:, s_idx, pos_model_idx]               # (n_samples,)
            b = beta_try_s[:, pidx] if s_name == "tries" else beta_kick_s[:, pidx]

            rate_player = np.exp(a + b + th)
            rate_repl = np.exp(a + th)
            vorp_s = (rate_player - rate_repl) * pts

            vorp_samples += vorp_s
            if s_name == "tries":
                vorp_tries += vorp_s
            else:
                vorp_kicking += vorp_s

        rows.append({
            "player": pname,
            "primary_position": primary_pos,
            "n_matches": n_matches,
            "vorp_mean": float(vorp_samples.mean()),
            "vorp_lower": float(np.percentile(vorp_samples, lo)),
            "vorp_upper": float(np.percentile(vorp_samples, hi)),
            "vorp_tries_mean": float(vorp_tries.mean()),
            "vorp_kicking_mean": float(vorp_kicking.mean()),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values("vorp_mean", ascending=False).reset_index(drop=True)
    if top_n is not None:
        result = result.head(top_n)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Squad criticality
# ─────────────────────────────────────────────────────────────────────────────


def compute_squad_criticality(
    predictor: MatchPredictor,
    squad: dict[int, str],
    team: str,
    season: str,
    n_samples: int = 2000,
    ci: float = 0.90,
    seed: int | None = None,
) -> pd.DataFrame:
    """Compute criticality for each starter in a squad.

    Criticality = expected reduction in match margin (vs reference team at
    neutral ground) when the player is replaced by a positional-average
    replacement player (β = 0).  Positive = team is worse without this player.

    This is VORP conditioned on the team's actual offensive system (γ_team)
    rather than the prior mean γ = 0, so it answers "how much does *this*
    team specifically lose when *this* player is unavailable?"

    The computation is analytical (no Poisson re-sampling) so uncertainty
    bounds reflect posterior uncertainty about player and team effects only,
    not match-to-match variance.

    Parameters
    ----------
    predictor:
        Fitted ``MatchPredictor``.
    squad:
        Mapping from jersey number (1–15) to player name.
    team:
        Team name.
    season:
        Season string.
    n_samples, ci, seed:
        As for ``compute_player_vorp``.

    Returns
    -------
    DataFrame sorted by ``criticality_mean`` descending with columns:
    player, position, criticality_mean, criticality_lower, criticality_upper.
    """
    from rugby_ranking.model.data import normalize_team_name

    model = predictor.model
    rng = np.random.default_rng(seed)
    lo = (1.0 - ci) / 2 * 100
    hi = 100.0 - lo

    team = normalize_team_name(team)
    team_idx, fallback, use_prior = _get_team_season_index_with_fallback(
        model._team_season_ids, team, season
    )
    if use_prior:
        raise ValueError(f"Unknown team: {team!r} — not found in any season.")

    sample_idx = rng.choice(
        predictor._n_total, size=n_samples, replace=predictor._n_total < n_samples
    )

    # Cached arrays (tries dimension) from MatchPredictor
    alpha = predictor._alpha_flat[sample_idx]            # (n,)
    gamma = predictor._gamma_flat[sample_idx]            # (n, n_ts)
    theta = predictor._theta_flat[sample_idx]            # (n, n_pos)
    beta = predictor._beta_flat[sample_idx]              # (n, n_players)

    team_gamma = gamma[:, team_idx]                      # (n,)

    # For multi-score-type VORP (penalties, conversions, drop goals), pull
    # the full alpha / theta / beta arrays from the trace.
    alpha_all = _get_alpha_all(predictor)                # (N, n_st)
    theta_all = _get_theta_all(predictor)                # (N, n_st, n_pos)
    beta_try_full, beta_kick_full = _get_beta_try_kick(model, predictor.trace)

    # Extract team's gamma for all score types via the gamma_flat from trace
    # (gamma_flat is already the tries-dim effective gamma from predictor)
    # For other score types we'd need separate lambdas — but for criticality
    # the relative ordering is dominated by tries, so we use the tries gamma
    # for simplicity and document this.
    alpha_s = alpha_all[sample_idx]                      # (n, n_st)
    theta_s = theta_all[sample_idx]                      # (n, n_st, n_pos)
    beta_try_s = beta_try_full[sample_idx]               # (n, n_players)
    beta_kick_s = beta_kick_full[sample_idx]             # (n, n_players)

    pos_id_map = model._position_ids
    score_types = model.config.score_types
    n_score_types = alpha_s.shape[1]

    # Hoist sigma_player outside the position loop — constant across positions.
    sigma_player = predictor._sigma_player_flat[sample_idx]       # (n,)

    rows = []
    for pos in range(1, STARTERS + 1):
        pname = squad.get(pos)
        if not pname:
            continue

        pos_model_idx = pos_id_map.get(pos, pos - 1)

        criticality_samples = np.zeros(n_samples)

        for s_idx, s_name in enumerate(score_types):
            if s_idx >= n_score_types:
                break
            pts = _SCORE_POINTS.get(s_name, 3)
            a = alpha_s[:, s_idx]                                 # (n,)
            th = theta_s[:, s_idx, pos_model_idx]                # (n,)
            # Use the team's γ effect (tries dimension is representative;
            # other dimensions would require per-score-type γ extraction)
            g = team_gamma

            if pname in model._player_ids:
                pidx = model._player_ids[pname]
                b = beta_try_s[:, pidx] if s_name == "tries" else beta_kick_s[:, pidx]
            else:
                # Unknown player — sample from prior once per draw.
                # sigma_player is (n_samples,); ensure b is always (n_samples,).
                b = rng.normal(0.0, np.broadcast_to(sigma_player, (n_samples,)))

            # Expected scoring rate: player vs positional average
            rate_player = np.exp(a + g + b + th)
            rate_repl = np.exp(a + g + th)          # β = 0
            criticality_samples += (rate_player - rate_repl) * pts

        rows.append({
            "player": pname,
            "position": pos,
            "criticality_mean": float(criticality_samples.mean()),
            "criticality_lower": float(np.percentile(criticality_samples, lo)),
            "criticality_upper": float(np.percentile(criticality_samples, hi)),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("criticality_mean", ascending=False).reset_index(drop=True)
