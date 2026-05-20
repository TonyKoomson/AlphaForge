"""
Momentum baseline strategy: 20/50-day MA crossover filtered by RSI(14).

Signal rules:
  +1  (long)  when fast_ma > slow_ma AND rsi > rsi_threshold
  -1  (short) when fast_ma < slow_ma AND rsi < (100 - rsi_threshold)
   0  (flat)  otherwise

Designed to work both as a standalone callable and as a walk-forward
signal_fn with history warmup.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Core indicator helpers
# ---------------------------------------------------------------------------

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_signals(
    prices: pd.DataFrame | pd.Series,
    fast_period: int = 20,
    slow_period: int = 50,
    rsi_period: int = 14,
    rsi_threshold: float = 50.0,
    allow_short: bool = True,
) -> pd.Series:
    """
    Generate momentum signals from OHLCV data or a close price Series.

    Parameters
    ----------
    prices : DataFrame with a 'close' column, or a plain Series of close prices.
    fast_period : Short moving-average window (default 20).
    slow_period : Long moving-average window (default 50).
    rsi_period : RSI lookback (default 14).
    rsi_threshold : RSI mid-line; values above = bullish filter (default 50).
    allow_short : If False, short signals are replaced with 0 (long-only mode).

    Returns
    -------
    pd.Series of {-1, 0, 1} aligned to prices.index.
    NaN rows (warmup) are filled with 0.
    """
    close = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    close = close.astype(float)

    fast_ma = _sma(close, fast_period)
    slow_ma = _sma(close, slow_period)
    rsi = _rsi(close, rsi_period)

    long_cond = (fast_ma > slow_ma) & (rsi > rsi_threshold)
    short_cond = (fast_ma < slow_ma) & (rsi < (100.0 - rsi_threshold))

    signal = pd.Series(0, index=close.index, dtype=int)
    signal[long_cond] = 1
    if allow_short:
        signal[short_cond] = -1

    logger.debug(
        "compute_signals: %d longs, %d shorts, %d flat out of %d bars",
        (signal == 1).sum(),
        (signal == -1).sum(),
        (signal == 0).sum(),
        len(signal),
    )
    return signal


class MomentumStrategy:
    """
    Stateless momentum strategy callable compatible with the walk-forward API.

    Usage as a walk-forward signal_fn::

        strategy = MomentumStrategy(fast_period=20, slow_period=50)
        signals = strategy(history_prices, target_prices)

    ``history`` provides the warmup bars needed for MA/RSI initialisation.
    ``target`` is the out-of-sample window for which signals are returned.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        rsi_period: int = 14,
        rsi_threshold: float = 50.0,
        allow_short: bool = True,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.allow_short = allow_short

    # Minimum bars needed before signals are reliable
    @property
    def warmup_bars(self) -> int:
        return self.slow_period + self.rsi_period

    def generate(self, prices: pd.DataFrame | pd.Series) -> pd.Series:
        """Run on an arbitrary price series; NaN-warmup rows return 0."""
        return compute_signals(
            prices,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            rsi_period=self.rsi_period,
            rsi_threshold=self.rsi_threshold,
            allow_short=self.allow_short,
        )

    def __call__(
        self,
        history: pd.DataFrame | pd.Series,
        target: pd.DataFrame | pd.Series,
    ) -> pd.Series:
        """
        Walk-forward callable.

        Concatenates history (warmup) + target, generates signals over the
        full window, then slices to just the target index so indicators are
        properly initialised.
        """
        history_close = history["close"] if isinstance(history, pd.DataFrame) else history
        target_close = target["close"] if isinstance(target, pd.DataFrame) else target

        combined = pd.concat([history_close, target_close])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        all_signals = compute_signals(
            combined,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            rsi_period=self.rsi_period,
            rsi_threshold=self.rsi_threshold,
            allow_short=self.allow_short,
        )
        return all_signals.reindex(target_close.index).fillna(0).astype(int)

    def __repr__(self) -> str:
        return (
            f"MomentumStrategy(fast={self.fast_period}, slow={self.slow_period}, "
            f"rsi_period={self.rsi_period}, rsi_threshold={self.rsi_threshold}, "
            f"allow_short={self.allow_short})"
        )
