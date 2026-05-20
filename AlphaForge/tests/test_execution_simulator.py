from __future__ import annotations

import numpy as np
import pandas as pd

from execution.simulator import compare_execution_modes, simulate_execution


def _make_prices(n: int = 60, high_low_span: float = 0.01) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100.0 * np.exp(np.linspace(0, 0.12, n))
    high = close * (1.0 + high_low_span)
    low = close * (1.0 - high_low_span)
    volume = np.full(n, 2_000_000.0)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_partial_fills_when_volume_is_thin():
    prices = _make_prices(n=40, high_low_span=0.04)
    # Low volume to force partial fills on large rebalances
    prices["volume"] = 2_000.0
    target = pd.Series(np.where(np.arange(len(prices)) % 2 == 0, 1.0, -1.0), index=prices.index)
    out = simulate_execution(target, prices, volume=prices["volume"], mode="conservative")
    trade_log = out["trade_log"]
    assert len(trade_log) > 0
    assert (trade_log["filled_quantity"] <= trade_log["requested_quantity"]).all()
    assert (trade_log["partial_fill_reason"] == "volume_capped_partial_fill").any() or (
        trade_log["partial_fill_reason"] == "liquidity_too_thin"
    ).any()


def test_slippage_increases_with_higher_volatility():
    target = pd.Series(1.0, index=pd.bdate_range("2022-01-03", periods=45))
    low_vol_px = _make_prices(n=45, high_low_span=0.005)
    high_vol_px = _make_prices(n=45, high_low_span=0.06)
    out_low = simulate_execution(target, low_vol_px, mode="realistic")
    out_high = simulate_execution(target, high_vol_px, mode="realistic")
    low_slip = out_low["trade_log"]["slippage"].mean() if len(out_low["trade_log"]) else 0.0
    high_slip = out_high["trade_log"]["slippage"].mean() if len(out_high["trade_log"]) else 0.0
    assert high_slip >= low_slip


def test_execution_comparison_reports_equity_and_sharpe_deltas():
    prices = _make_prices(n=70, high_low_span=0.02)
    target = pd.Series(np.sin(np.linspace(0, 5, len(prices))), index=prices.index).clip(-1, 1)
    cmp_res = compare_execution_modes(target, prices, volume=prices["volume"])
    assert "difference" in cmp_res
    assert "final_equity_delta" in cmp_res["difference"]
    assert "sharpe_delta" in cmp_res["difference"]
