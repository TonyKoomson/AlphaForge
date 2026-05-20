"""
features/change_point.py
=========================
Change Point Detection for financial time series.

Detects structural breaks in the mean and/or variance of a return series —
regime shifts, vol clusters, and trend changes.

Methods implemented
-------------------
1. PELT (Pruned Exact Linear Time)   — Killick et al. (2012).
   Exact O(n log n) method, quadratic cost function (mean + variance change).
   Best for detecting multiple change points efficiently.

2. CUSUM (Cumulative Sum)            — Page (1954).
   Online-friendly; detects mean shifts with a cumulative sum statistic.
   Fast O(n), suitable for real-time streaming detection.

3. Bayesian Change Point (BCP)       — simple conjugate Gaussian model.
   Returns posterior probability of a change point at each bar.
   Smooths noisy signals; no threshold needed — use probabilities directly.

Usage
-----
    from features.change_point import detect_change_points, cusum_change_points

    # PELT: returns indices of detected change points
    cps = detect_change_points(returns, penalty=10.0)

    # CUSUM: returns binary series (1 = change detected at bar t)
    cusum = cusum_change_points(returns, threshold=4.0)

    # Bayesian: returns pd.Series of P(change at t)
    probs = bayesian_change_point_probs(returns, hazard_rate=1/50)

    # Add all three as features to a DataFrame
    df = add_change_point_features(df, col="close")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PELT — Pruned Exact Linear Time
# ---------------------------------------------------------------------------

def _pelt_cost(data: np.ndarray, start: int, end: int) -> float:
    """
    Quadratic (Gaussian) cost: negative log-likelihood of segment [start, end).
    Cost = n·log(σ²) for the segment, where σ² is the ML variance estimate.
    """
    seg = data[start:end]
    n   = len(seg)
    if n < 2:
        return 0.0
    mu  = seg.mean()
    var = max(np.mean((seg - mu) ** 2), 1e-12)
    return n * np.log(var)


def detect_change_points(
    data:    pd.Series,
    penalty: float = 10.0,
    min_seg: int   = 5,
) -> list[int]:
    """
    PELT algorithm for multiple change point detection.

    Minimises Σ_k C(segment_k) + penalty × K where K = number of segments.
    A higher penalty reduces the number of detected change points.

    Parameters
    ----------
    data    : pd.Series of returns (or any stationary signal).
    penalty : BIC-like penalty per added change point.  A rule of thumb is
              penalty = 2 × log(n) for the BIC criterion.
    min_seg : Minimum segment length (prevents over-segmentation).

    Returns
    -------
    List of integer index positions of change points (not including endpoints).
    """
    arr = np.asarray(data.dropna(), dtype=float)
    n   = len(arr)
    if n < 2 * min_seg:
        return []

    # Dynamic programming with pruning
    F    = np.full(n + 1, np.inf)
    F[0] = -penalty
    cp   = [-1] * (n + 1)
    admissible = [0]

    for t in range(min_seg, n + 1):
        candidates = [s for s in admissible if t - s >= min_seg]
        if not candidates:
            F[t]  = F[0] + _pelt_cost(arr, 0, t)
            cp[t] = 0
        else:
            costs = [F[s] + _pelt_cost(arr, s, t) + penalty for s in candidates]
            best  = int(np.argmin(costs))
            F[t]  = costs[best]
            cp[t] = candidates[best]

        # Prune: remove indices where F[s] + cost_lower_bound > F[t]
        admissible = [s for s in admissible if F[s] + 0.0 <= F[t]]
        admissible.append(t)

    # Backtrack
    change_points: list[int] = []
    idx = n
    while cp[idx] > 0:
        change_points.append(cp[idx])
        idx = cp[idx]
    change_points = sorted(change_points)

    # Map back to original series index (account for dropna offset)
    valid_idx = np.where(~np.isnan(data.values))[0]
    result = [int(valid_idx[i]) for i in change_points if i < len(valid_idx)]
    logger.debug("PELT: detected %d change points (penalty=%.1f)", len(result), penalty)
    return result


# ---------------------------------------------------------------------------
# CUSUM — Cumulative Sum
# ---------------------------------------------------------------------------

def cusum_change_points(
    data:       pd.Series,
    threshold:  float = 4.0,
    drift:      float = 0.0,
) -> pd.Series:
    """
    Two-sided CUSUM for mean shift detection.

    Parameters
    ----------
    data      : Return or signal series.
    threshold : Detection threshold (in units of std). Common: 4–5.
    drift     : Allowable drift before signalling (often 0.5 × shift_size).

    Returns
    -------
    pd.Series of booleans; True at bars where a change is signalled.
    """
    arr = np.asarray(data.fillna(0), dtype=float)
    mu  = arr.mean()
    std = max(arr.std(), 1e-9)
    z   = (arr - mu) / std

    s_pos = np.zeros(len(z))
    s_neg = np.zeros(len(z))
    alarm = np.zeros(len(z), dtype=bool)

    for t in range(1, len(z)):
        s_pos[t] = max(0.0, s_pos[t-1] + z[t] - drift)
        s_neg[t] = max(0.0, s_neg[t-1] - z[t] - drift)
        if s_pos[t] > threshold or s_neg[t] > threshold:
            alarm[t] = True
            s_pos[t] = 0.0    # reset after alarm
            s_neg[t] = 0.0

    return pd.Series(alarm, index=data.index, name="cusum_alarm")


# ---------------------------------------------------------------------------
# Bayesian Change Point (conjugate Gaussian)
# ---------------------------------------------------------------------------

def bayesian_change_point_probs(
    data:         pd.Series,
    hazard_rate:  float = 0.02,
) -> pd.Series:
    """
    Bayesian online change point detection (Adams & MacKay, 2007).

    Models each segment as iid Gaussian with unknown mean and variance.
    Returns P(change at bar t) — posterior probability at each bar.

    Parameters
    ----------
    data         : Return series.
    hazard_rate  : Prior probability of a change point at each bar (1/expected_run_length).
                   Default 0.02 implies ~50-bar average segment length.

    Returns
    -------
    pd.Series of posterior change-point probabilities in [0, 1].
    """
    arr = np.asarray(data.fillna(0), dtype=float)
    n   = len(arr)

    # Sufficient statistics for Gaussian predictive
    # Conjugate prior: Normal-Gamma (μ₀=0, κ₀=1, α₀=1, β₀=1)
    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0

    # Run-length distribution: R[t] = P(run length = r at time t)
    R  = np.zeros((n + 1, n + 2))   # +2 to accommodate run-length growth
    R[0, 0] = 1.0

    # Sufficient stats for each possible run-length start
    mu    = np.zeros(n + 1)
    kappa = np.full(n + 1, kappa0)
    alpha = np.full(n + 1, alpha0)
    beta  = np.full(n + 1, beta0)

    cp_probs = np.zeros(n)

    for t in range(n):
        x = arr[t]

        # Predictive probability for each run length (Student-t)
        pred = np.zeros(t + 1)
        for r in range(t + 1):
            nu = 2 * alpha[r]
            scale = np.sqrt(beta[r] * (kappa[r] + 1) / (alpha[r] * kappa[r]))
            z = (x - mu[r]) / max(scale, 1e-9)
            # Log t-pdf: lgamma(ν/2+1/2) - lgamma(ν/2) - 0.5·log(νπσ²) - (ν+1)/2·log(1+z²/ν)
            from math import lgamma, log, pi
            log_p = (lgamma((nu + 1) / 2) - lgamma(nu / 2)
                     - 0.5 * log(nu * pi * scale ** 2)
                     - (nu + 1) / 2 * log(1 + z ** 2 / max(nu, 1e-9)))
            pred[r] = np.exp(np.clip(log_p, -100, 0))

        # Update run-length distribution
        R_next = np.zeros(n + 2)
        for r in range(t + 1):
            R_next[r + 1] += R[t, r] * pred[r] * (1 - hazard_rate)  # run continues
            R_next[0]     += R[t, r] * pred[r] * hazard_rate          # change point

        norm = R_next.sum()
        if norm > 1e-12:
            R_next /= norm
        R[t + 1, :n + 2] = R_next

        # P(change at t) = mass at run-length = 0 after update
        cp_probs[t] = R_next[0]

        # Update sufficient statistics (vectorised, approx.)
        kappa_new = kappa[:t+1] + 1
        mu_new    = (kappa[:t+1] * mu[:t+1] + x) / kappa_new
        alpha_new = alpha[:t+1] + 0.5
        beta_new  = (beta[:t+1]
                     + 0.5 * kappa[:t+1] / kappa_new * (x - mu[:t+1]) ** 2)
        mu[:t+2]    = np.append(mu_new, mu0)
        kappa[:t+2] = np.append(kappa_new, kappa0)
        alpha[:t+2] = np.append(alpha_new, alpha0)
        beta[:t+2]  = np.append(beta_new, beta0)

    return pd.Series(cp_probs, index=data.index, name="cp_prob")


# ---------------------------------------------------------------------------
# Convenience: add change-point features to a DataFrame
# ---------------------------------------------------------------------------

def add_change_point_features(
    df:           pd.DataFrame,
    col:          str   = "close",
    pelt_penalty: float = 10.0,
    cusum_thresh: float = 4.0,
    hazard_rate:  float = 0.02,
) -> pd.DataFrame:
    """
    Add three change-point feature columns to df:
      cp_pelt   : 1 at PELT-detected change points, 0 elsewhere
      cp_cusum  : 1 when CUSUM alarm fires
      cp_prob   : Bayesian posterior probability of change at each bar

    Input column `col` should be a price series; returns are computed internally.
    """
    if col not in df.columns:
        logger.warning("add_change_point_features: column '%s' not found", col)
        return df

    rets = df[col].pct_change().fillna(0)
    out  = df.copy()

    # 1. PELT
    try:
        cp_idx          = detect_change_points(rets, penalty=pelt_penalty)
        pelt_series     = pd.Series(0, index=df.index, dtype=int)
        if cp_idx:
            pelt_series.iloc[cp_idx] = 1
        out["cp_pelt"]  = pelt_series
    except Exception as e:
        logger.debug("PELT failed: %s", e)
        out["cp_pelt"] = pd.Series(0, index=out.index, dtype=int)

    # 2. CUSUM
    try:
        out["cp_cusum"] = cusum_change_points(rets, threshold=cusum_thresh).astype(int)
    except Exception as e:
        logger.debug("CUSUM failed: %s", e)
        out["cp_cusum"] = pd.Series(0, index=out.index, dtype=int)

    # 3. Bayesian (smoothed probability — useful as a feature directly)
    try:
        out["cp_prob"] = bayesian_change_point_probs(rets, hazard_rate=hazard_rate)
    except Exception as e:
        logger.debug("Bayesian CP failed: %s", e)
        out["cp_prob"] = pd.Series(0.0, index=out.index, dtype=float)

    logger.debug("add_change_point_features: added cp_pelt, cp_cusum, cp_prob")
    return out
