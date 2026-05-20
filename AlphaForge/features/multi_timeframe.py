"""
features/multi_timeframe.py
============================
Multi-Timeframe Signal Confirmation for AlphaForge v2.0.

Only enter a trade when signals agree across multiple timeframes.
Reduces false positives significantly — the same directional bet must
look attractive on daily, weekly, and monthly timeframes before entering.

Method
------
For each timeframe (e.g. 5, 21, 63 bars):
  1. Compute aggregate momentum score: EMA crossover + RSI direction.
  2. Convert to directional signal: +1 (bullish), -1 (bearish), 0 (neutral).
  3. Confirmation = all non-neutral timeframes agree on direction.
  4. Confidence = fraction of timeframes that agree.

Usage
-----
    from features.multi_timeframe import MultiTimeframeConfirmation

    mtf = MultiTimeframeConfirmation(timeframes=[5, 21, 63])
    confirmed_signal, confidence = mtf.confirm(close, base_signal)

    # Add MTF features to DataFrame
    df = mtf.add_features(df, close_col='close')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class MTFResult:
    signal:       int          # confirmed: +1, -1, or 0 (no agreement)
    confidence:   float        # fraction of timeframes agreeing
    tf_signals:   dict[int, int]
    agree:        bool


class MultiTimeframeConfirmation:
    """
    Confirm a base signal against multiple timeframes.

    Parameters
    ----------
    timeframes      : List of lookback windows in bars (e.g., [5, 21, 63]).
    min_agreement   : Fraction of timeframes that must agree (default: 0.67).
    rsi_period      : RSI period for directional momentum.
    require_all     : If True, ALL timeframes must agree (strict mode).
    """

    def __init__(
        self,
        timeframes:   tuple[int, ...] = (5, 21, 63),
        min_agreement: float          = 0.67,
        rsi_period:   int             = 14,
        require_all:  bool            = False,
    ) -> None:
        self.timeframes    = sorted(timeframes)
        self.min_agreement = min_agreement
        self.rsi_period    = rsi_period
        self.require_all   = require_all

    def _tf_signal(self, close: pd.Series, window: int) -> pd.Series:
        """Compute directional signal for a single timeframe."""
        fast = close.ewm(span=max(window // 3, 2), adjust=False).mean()
        slow = close.ewm(span=window, adjust=False).mean()
        cross = (fast > slow).astype(int) * 2 - 1   # +1 bullish, -1 bearish

        # RSI direction
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.rsi_period, min_periods=5).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.rsi_period, min_periods=5).mean()
        rs    = gain / (loss + 1e-9)
        rsi   = 100 - 100 / (1 + rs)
        rsi_dir = pd.Series(0, index=close.index, dtype=int)
        rsi_dir[rsi > 55] = 1
        rsi_dir[rsi < 45] = -1

        # Combine: only +1 if both agree on bullish, etc.
        combined = pd.Series(0, index=close.index, dtype=int)
        combined[(cross == 1) & (rsi_dir >= 0)] = 1
        combined[(cross == -1) & (rsi_dir <= 0)] = -1
        return combined

    def compute_all(self, close: pd.Series) -> pd.DataFrame:
        """Return DataFrame with a signal column per timeframe."""
        return pd.DataFrame(
            {f"mtf_{w}": self._tf_signal(close, w) for w in self.timeframes},
            index=close.index,
        )

    def confirm(self, close: pd.Series, base_signal: int, bar_idx: int = -1) -> MTFResult:
        """
        Check if base_signal is confirmed by all timeframes at bar bar_idx.

        Parameters
        ----------
        close       : Price series up to and including current bar.
        base_signal : Primary signal to confirm (+1 or -1).
        bar_idx     : Index within close to use (default -1 = latest bar).
        """
        tf_sigs: dict[int, int] = {}
        for w in self.timeframes:
            s = self._tf_signal(close, w)
            tf_sigs[w] = int(s.iloc[bar_idx]) if len(s) > abs(bar_idx) else 0

        non_neutral = {w: s for w, s in tf_sigs.items() if s != 0}
        if not non_neutral:
            return MTFResult(0, 0.0, tf_sigs, False)

        agree_count = sum(1 for s in non_neutral.values() if s == base_signal)
        frac = agree_count / len(non_neutral)

        if self.require_all:
            confirmed = frac == 1.0
        else:
            confirmed = frac >= self.min_agreement

        confirmed_signal = base_signal if confirmed else 0
        return MTFResult(confirmed_signal, round(frac, 3), tf_sigs, confirmed)

    def add_features(
        self,
        df:        pd.DataFrame,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Add MTF columns to df:
          mtf_{w}         : directional signal per timeframe
          mtf_agree       : fraction of timeframes agreeing (0–1)
          mtf_confirmed   : 1 if ≥ min_agreement timeframes agree, else 0
        """
        if close_col not in df.columns:
            return df
        out   = df.copy()
        close = df[close_col]
        mtf   = self.compute_all(close)

        for col in mtf.columns:
            out[col] = mtf[col]

        # Agreement: row-wise fraction of non-zero signals that agree
        def _row_agree(row: pd.Series) -> float:
            vals = [v for v in row.values if v != 0]
            if not vals:
                return 0.0
            mode = 1 if sum(1 for v in vals if v > 0) >= len(vals) / 2 else -1
            return sum(1 for v in vals if v == mode) / len(vals)

        out["mtf_agree"]     = mtf.apply(_row_agree, axis=1)
        out["mtf_confirmed"] = (out["mtf_agree"] >= self.min_agreement).astype(int)
        return out
