"""Regime detection and regime-aware signal adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger, max_drawdown, sharpe_ratio

logger = get_logger(__name__)


REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"


def _sma_slope(close: pd.Series, window: int = 50, slope_lookback: int = 5) -> pd.Series:
    sma = close.rolling(window, min_periods=window).mean()
    return (sma - sma.shift(slope_lookback)) / (sma.shift(slope_lookback).abs() + 1e-9)


def _historical_vol(close: pd.Series, window: int = 20) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window, min_periods=window).std() * np.sqrt(252)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean()
    tr_s = tr.rolling(period, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-9)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(period, min_periods=period).mean()


def detect_regime(
    df: pd.DataFrame,
    ma_window: int = 50,
    slope_lookback: int = 5,
    vol_window: int = 20,
    adx_period: int = 14,
    bull_slope_threshold: float = 0.002,
    bear_slope_threshold: float = -0.002,
    low_vol_threshold: float = 0.18,
    high_vol_threshold: float = 0.28,
    adx_trend_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Label each day as bull / bear / sideways using MA slope, volatility, and ADX.
    """
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_regime requires columns: {missing}")

    out = df.copy()
    close = out["close"].astype(float)
    slope = _sma_slope(close, window=ma_window, slope_lookback=slope_lookback)
    vol = _historical_vol(close, window=vol_window)
    adx = _adx(out, period=adx_period)

    bull_cond = (slope > bull_slope_threshold) & (vol <= low_vol_threshold) & (adx >= adx_trend_threshold)
    bear_cond = (slope < bear_slope_threshold) & (vol >= high_vol_threshold) & (adx >= adx_trend_threshold)
    sideways_cond = ~(bull_cond | bear_cond)

    regime = pd.Series(REGIME_SIDEWAYS, index=out.index, dtype="object")
    regime.loc[bull_cond] = REGIME_BULL
    regime.loc[bear_cond] = REGIME_BEAR
    regime.loc[sideways_cond] = REGIME_SIDEWAYS

    out["ma50_slope"] = slope
    out["hist_vol_20d"] = vol
    out["adx_14"] = adx
    out["regime"] = regime
    return out


@dataclass
class RegimeMetaLearner:
    """
    Simple regime-aware meta-learner that adjusts base signal strength.
    """

    bull_multiplier: float = 1.15
    bear_multiplier: float = 0.65
    sideways_multiplier: float = 0.80
    min_confidence: float = 0.60

    def adjust(self, base_signal: float, current_regime: str, confidence: float) -> float:
        if confidence < self.min_confidence:
            return 0.0

        if current_regime == REGIME_BULL:
            mult = self.bull_multiplier
        elif current_regime == REGIME_BEAR:
            mult = self.bear_multiplier
        else:
            mult = self.sideways_multiplier
        return float(np.clip(base_signal * mult * confidence, -1.0, 1.0))


def get_regime_adjusted_signal(base_signal: float, current_regime: str, confidence: float) -> float:
    """
    Adjust signal based on regime and confidence.
    """
    return RegimeMetaLearner().adjust(base_signal, current_regime, confidence)


def regime_performance_report(
    returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Compute Sharpe, max drawdown, and win rate broken down by regime.
    """
    aligned_ret = returns.reindex(regime_labels.index).fillna(0.0)
    aligned_reg = regime_labels.reindex(aligned_ret.index).fillna(REGIME_SIDEWAYS)

    rows: list[dict[str, float | str]] = []
    for reg in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
        r = aligned_ret[aligned_reg == reg]
        if r.empty:
            rows.append({"regime": reg, "n_bars": 0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0})
            continue
        equity = (1.0 + r).cumprod()
        rows.append(
            {
                "regime": reg,
                "n_bars": int(len(r)),
                "sharpe": float(sharpe_ratio(r)),
                "max_drawdown": float(max_drawdown(equity)),
                "win_rate": float((r > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_regime_overlay(
    equity_curve: pd.Series,
    regime_labels: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """
    Optional visualization: shade equity curve by detected regime periods.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping regime overlay plot")
        return

    eq = equity_curve.dropna()
    reg = regime_labels.reindex(eq.index).fillna(REGIME_SIDEWAYS)

    colors = {
        REGIME_BULL: "#C8E6C9",
        REGIME_BEAR: "#FFCDD2",
        REGIME_SIDEWAYS: "#FFF9C4",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq.values, color="#1565C0", linewidth=1.6, label="Equity")
    ax.set_title("Equity Curve with Regime Overlay")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)

    start = eq.index[0]
    current = reg.iloc[0]
    for i in range(1, len(eq)):
        if reg.iloc[i] != current:
            ax.axvspan(start, eq.index[i - 1], color=colors.get(current, "#E0E0E0"), alpha=0.25)
            start = eq.index[i]
            current = reg.iloc[i]
    ax.axvspan(start, eq.index[-1], color=colors.get(current, "#E0E0E0"), alpha=0.25)

    handles = [plt.Line2D([0], [0], color="#1565C0", lw=2, label="Equity")]
    for r in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[r], alpha=0.5, label=r.title()))
    ax.legend(handles=handles, loc="best")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved regime overlay chart to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
