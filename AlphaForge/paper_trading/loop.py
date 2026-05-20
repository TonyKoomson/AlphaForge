"""
paper_trading/loop.py — Signal filtering utilities shared by backtest, paper-trade, and validate.

Provides two functions used by main.py's _load_signals() helper:
  _regime_conditional_signals  — apply per-regime long/short thresholds to raw probabilities
  _apply_signal_filters        — enforce minimum holding period and signal confirmation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


_REGIME_COL = "regime"   # column name in features DataFrame (0=unknown,1=bull,-1=bear,2=sideways,3=high_vol)

_REGIME_MAP = {
    1:   "bull",
    -1:  "bear",
    0:   "sideways",
    2:   "sideways",
    3:   "high_vol",
    -99: "unknown",
}

_DEFAULT_THRESHOLDS = {
    "bull":     {"long": 0.55, "short": 0.82},
    "sideways": {"long": 0.67, "short": 0.40},  # raised: model base-rate is ~0.64, require genuine signal
    "bear":     {"long": 0.75, "short": 0.52},
    "high_vol": {"long": 0.67, "short": 0.50},  # raised to match model output range
    "unknown":  {"long": 0.67, "short": 0.45},  # raised from 0.63
}


def _regime_conditional_signals(
    probas: np.ndarray,
    index: pd.DatetimeIndex,
    features: pd.DataFrame,
    regime_thresholds: Optional[dict] = None,
    fallback_threshold: float = 0.65,
    trend_filter_200ma: bool = False,
) -> pd.Series:
    """
    Convert raw model probabilities to long(+1)/short(-1)/flat(0) signals
    using per-regime thresholds.

    Parameters
    ----------
    probas              : 1-D array of P(up) for each bar.
    index               : DatetimeIndex aligned with probas.
    features            : Feature DataFrame — must contain 'regime' column.
    regime_thresholds   : dict mapping regime name to {long: float, short: float}.
    fallback_threshold  : Used when regime column is missing.
    trend_filter_200ma  : If True, suppress short signals when price > SMA-200.
    """
    thresholds = regime_thresholds or _DEFAULT_THRESHOLDS
    probas = np.asarray(probas)
    signals = pd.Series(0, index=index, dtype=int)

    if _REGIME_COL in features.columns:
        regime_raw = features[_REGIME_COL].reindex(index).fillna(0).astype(int)
        for i, (ts, p) in enumerate(zip(index, probas)):
            regime_name = _REGIME_MAP.get(int(regime_raw.iloc[i] if i < len(regime_raw) else 0), "unknown")
            rt = thresholds.get(regime_name, thresholds.get("unknown", {}))
            long_thr  = rt.get("long",  fallback_threshold)
            short_thr = rt.get("short", fallback_threshold)
            if p >= long_thr:
                signals.iloc[i] = 1
            elif p <= (1.0 - short_thr):
                signals.iloc[i] = -1
    else:
        # No regime column — use symmetric fallback threshold
        signals[probas >= fallback_threshold] = 1
        signals[probas <= (1.0 - fallback_threshold)] = -1

    # Optional: suppress short signals in bull trend (price > SMA-200)
    if trend_filter_200ma and "close" in features.columns:
        close = features["close"].reindex(index)
        sma200 = close.rolling(200, min_periods=100).mean()
        bull_mask = close > sma200
        signals[bull_mask & (signals == -1)] = 0

    return signals


def _apply_signal_filters(
    signals: pd.Series,
    min_holding_bars: int = 5,
    confirm_bars: int = 2,
) -> pd.Series:
    """
    Apply minimum holding period and signal confirmation filters.

    - confirm_bars     : require N consecutive bars of the same signal before acting
    - min_holding_bars : once in a position, hold for at least N bars

    Returns filtered signal series (same index, values in {-1, 0, 1}).
    """
    if len(signals) == 0:
        return signals.copy()

    arr = signals.values.copy()
    n = len(arr)

    # Step 1: confirmation filter — only enter when the same direction persists
    confirmed = np.zeros(n, dtype=int)
    if confirm_bars > 1:
        for i in range(confirm_bars - 1, n):
            window = arr[i - confirm_bars + 1 : i + 1]
            if np.all(window == 1):
                confirmed[i] = 1
            elif np.all(window == -1):
                confirmed[i] = -1
            # else: flat (0) — confirmation not met
    else:
        confirmed = arr.copy()

    # Step 2: minimum holding period — once position opens, hold it
    result = np.zeros(n, dtype=int)
    current_pos = 0
    bars_held   = 0

    for i in range(n):
        desired = confirmed[i]
        if current_pos != 0 and bars_held < min_holding_bars:
            # Still in holding period — maintain current position
            result[i] = current_pos
            bars_held += 1
        else:
            # Free to change position
            if desired != 0:
                result[i]   = desired
                current_pos = desired
                bars_held   = 1
            else:
                result[i]   = 0
                current_pos = 0
                bars_held   = 0

    return pd.Series(result, index=signals.index, dtype=int)
