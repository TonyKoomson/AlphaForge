"""
features/fracdiff.py
====================
Fractionally Differentiated Features (López de Prado, AFML Ch.5).

Problem
-------
Raw price series are non-stationary (I(1) or higher) — standard ML models
require stationarity. Integer differencing (d=1, returns) achieves stationarity
but destroys all memory. Fractional differentiation with 0 < d < 1 achieves
*near-stationarity* while preserving the maximum amount of historical memory.

Method
------
The fractional differencing operator weights:
    w_k = -(d - k + 1) / k * w_{k-1},  w_0 = 1

The infinite series is truncated at a minimum weight threshold τ.  Each
output value at time t is:
    x̃[t] = Σ_{k=0}^{T-1} w_k · x[t-k]

Optimal d selection
-------------------
The smallest d that makes the series stationary (ADF p-value < threshold),
found by binary search over [0, 1].  This maximises memory while achieving
the required stationarity.

Usage
-----
    from features.fracdiff import fracdiff, find_optimal_d, fracdiff_df

    # Single series
    fd = fracdiff(close, d=0.4, threshold=1e-4)

    # Auto-select optimal d
    d_opt = find_optimal_d(close)
    fd    = fracdiff(close, d=d_opt)

    # Full DataFrame (apply to all numeric columns)
    fd_df = fracdiff_df(prices_df, d=0.35)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core fractional differencing
# ---------------------------------------------------------------------------

def _frac_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """
    Compute FFD (Fixed-width window Fractional Differencing) weights.

    Truncates the infinite series when |w_k| < threshold.
    Returns weight vector w of shape (T,) where w[0] = 1.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])   # oldest weight first


def fracdiff(
    series:    pd.Series,
    d:         float,
    threshold: float = 1e-3,
) -> pd.Series:
    """
    Apply Fixed-width window Fractional Differencing to a price series.

    Parameters
    ----------
    series    : pd.Series of prices (or log-prices).
    d         : Fractional differencing order in (0, 1].  d=1 → returns.
    threshold : Drop weights below this absolute value (controls window width).

    Returns
    -------
    pd.Series with same index; leading NaN for the warmup window.
    """
    if not (0.0 <= d <= 2.0):
        raise ValueError(f"d must be in [0, 2], got {d}")
    if d == 0.0:
        return series.copy()

    w      = _frac_weights(d, threshold)
    width  = len(w)
    values = series.values.astype(float)
    n      = len(values)

    out = np.full(n, np.nan)
    for t in range(width - 1, n):
        out[t] = float(np.dot(w, values[t - width + 1: t + 1]))

    result = pd.Series(out, index=series.index, name=f"{series.name}_fd{d:.2f}")
    return result


def fracdiff_df(
    df:        pd.DataFrame,
    d:         float,
    threshold: float = 1e-4,
    columns:   Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply fracdiff to selected (or all numeric) columns.

    Returns a DataFrame with columns renamed to '{col}_fd{d:.2f}'.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    out  = {}
    for col in cols:
        out[f"{col}_fd{d:.2f}"] = fracdiff(df[col], d=d, threshold=threshold)
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Optimal-d selection via ADF stationarity test
# ---------------------------------------------------------------------------

def find_optimal_d(
    series:        pd.Series,
    d_range:       tuple[float, float] = (0.0, 1.0),
    adf_threshold: float = 0.05,
    tolerance:     float = 0.01,
    threshold:     float = 1e-4,
    max_iter:      int   = 20,
) -> float:
    """
    Binary search for the minimum d in d_range such that the fractionally
    differentiated series passes the ADF test at adf_threshold significance.

    If the series is already stationary at d=d_range[0], returns d_range[0].
    If not stationary even at d=d_range[1], returns d_range[1].

    Parameters
    ----------
    series        : Raw price (or log-price) series.
    d_range       : (d_min, d_max) search bracket.
    adf_threshold : ADF p-value threshold for stationarity (default 0.05).
    tolerance     : Stop when d_max - d_min < tolerance.
    threshold     : Weight threshold for fracdiff window.
    max_iter      : Maximum binary search iterations.

    Returns
    -------
    Optimal d as float.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed — returning d=0.4 default")
        return 0.4

    def _is_stationary(d: float) -> bool:
        fd = fracdiff(series.dropna(), d, threshold).dropna()
        if len(fd) < 20:
            return False
        pval = adfuller(fd, maxlag=1, regression="c", autolag=None)[1]
        return pval < adf_threshold

    d_lo, d_hi = d_range

    # Quick check boundaries
    if _is_stationary(d_lo):
        logger.debug("fracdiff: series already stationary at d=%.2f", d_lo)
        return d_lo
    if not _is_stationary(d_hi):
        logger.warning("fracdiff: series not stationary even at d=%.2f; using d_hi", d_hi)
        return d_hi

    for _ in range(max_iter):
        if d_hi - d_lo < tolerance:
            break
        d_mid = (d_lo + d_hi) / 2.0
        if _is_stationary(d_mid):
            d_hi = d_mid
        else:
            d_lo = d_mid

    optimal = round(d_hi, 3)
    logger.info("fracdiff: optimal d=%.3f (ADF threshold=%.2f)", optimal, adf_threshold)
    return optimal


# ---------------------------------------------------------------------------
# Convenience: add fracdiff features for OHLCV columns
# ---------------------------------------------------------------------------

def add_fracdiff_features(
    df:       pd.DataFrame,
    d:        Optional[float] = None,
    columns:  Optional[list[str]] = None,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """
    Add fractionally differentiated versions of price columns to a feature df.

    If d is None, auto-selects optimal d from the 'close' column.
    Columns default to ['close'] if present, else first numeric column.

    New columns named '{col}_fd' are appended; existing columns unchanged.
    """
    if df.empty:
        return df

    if columns is None:
        columns = [c for c in ["close", "open", "high", "low"] if c in df.columns]
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns[:1].tolist()

    if not columns:
        return df

    if d is None:
        ref_col = "close" if "close" in columns else columns[0]
        d = find_optimal_d(df[ref_col], d_range=(0.1, 1.0), threshold=threshold)

    out = df.copy()
    for col in columns:
        fd_col = fracdiff(df[col], d=d, threshold=threshold)
        out[f"{col}_fd"] = fd_col

    logger.debug("add_fracdiff_features: d=%.3f, cols=%s", d, columns)
    return out
