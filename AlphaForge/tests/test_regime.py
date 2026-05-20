from __future__ import annotations

import numpy as np
import pandas as pd

from features.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    detect_regime,
    get_regime_adjusted_signal,
    regime_performance_report,
)


def _make_ohlc_from_close(close: np.ndarray, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    high = close * (1.002 + rng.uniform(0, 0.006, len(close)))
    low = close * (0.998 - rng.uniform(0, 0.006, len(close)))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, len(close)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def test_detect_regime_outputs_valid_labels():
    n = 320
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0002, 0.01, n)))
    df = _make_ohlc_from_close(close)
    out = detect_regime(df)
    assert "regime" in out.columns
    assert set(out["regime"].dropna().unique()).issubset({REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS})


def test_detect_regime_bull_on_strong_trend_low_vol():
    n = 320
    close = 100.0 * np.exp(np.linspace(0, 0.8, n))
    df = _make_ohlc_from_close(close)
    out = detect_regime(df, low_vol_threshold=0.35, adx_trend_threshold=10.0)
    tail = out["regime"].iloc[-60:]
    assert (tail == REGIME_BULL).mean() > 0.6


def test_detect_regime_sideways_on_flat_series():
    n = 320
    close = np.ones(n) * 100.0
    df = _make_ohlc_from_close(close)
    out = detect_regime(df, low_vol_threshold=0.05, high_vol_threshold=0.10, adx_trend_threshold=25.0)
    tail = out["regime"].iloc[-60:]
    assert (tail == REGIME_SIDEWAYS).mean() > 0.6


def test_adjusted_signal_is_zero_below_confidence_threshold():
    sig = get_regime_adjusted_signal(base_signal=1.0, current_regime=REGIME_BULL, confidence=0.2)
    assert sig == 0.0


def test_adjusted_signal_regime_scaling_logic():
    bull_sig = get_regime_adjusted_signal(base_signal=1.0, current_regime=REGIME_BULL, confidence=0.9)
    bear_sig = get_regime_adjusted_signal(base_signal=1.0, current_regime=REGIME_BEAR, confidence=0.9)
    assert bull_sig > bear_sig


def test_regime_performance_report_schema():
    idx = pd.bdate_range("2021-01-01", periods=90)
    returns = pd.Series(np.random.default_rng(4).normal(0.0002, 0.01, len(idx)), index=idx)
    labels = pd.Series([REGIME_BULL] * 30 + [REGIME_BEAR] * 30 + [REGIME_SIDEWAYS] * 30, index=idx)
    rep = regime_performance_report(returns, labels)
    assert set(rep.columns) == {"regime", "n_bars", "sharpe", "max_drawdown", "win_rate"}
    assert set(rep["regime"]) == {REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS}
