from __future__ import annotations

import numpy as np
import pandas as pd

from risk.position_sizing import PositionSizer, calculate_position_size


def _cfg() -> dict:
    return {
        "risk": {
            "target_annual_volatility": 0.15,
            "kelly_fraction": 0.25,
            "fixed_risk_pct": 0.01,
            "max_leverage": 2.0,
            "daily_loss_limit_pct": 0.03,
            "min_confidence_threshold": 0.60,
            "max_position_size_per_asset": 1.0,
            "max_portfolio_drawdown": 0.25,
        }
    }


def test_calculate_position_size_output_range():
    size = calculate_position_size(
        predicted_return=0.02,
        confidence=0.95,
        current_volatility=0.15,
        account_equity=100_000.0,
        config=_cfg(),
    )
    assert -1.0 <= size <= 1.0


def test_confidence_below_threshold_goes_flat():
    sizer = PositionSizer(config=_cfg())
    size = sizer.calculate_position_size(
        predicted_return=0.03,
        confidence=0.40,
        current_volatility=0.20,
        account_equity=100_000.0,
    )
    assert size == 0.0


def test_daily_loss_limit_goes_flat():
    sizer = PositionSizer(config=_cfg())
    size = sizer.calculate_position_size(
        predicted_return=0.04,
        confidence=0.90,
        current_volatility=0.15,
        account_equity=100_000.0,
        daily_return=-0.05,
    )
    assert size == 0.0


def test_hard_caps_never_exceed_asset_limit():
    sizer = PositionSizer(config=_cfg())
    sizes = []
    for edge in [0.01, 0.05, 0.20]:
        for vol in [0.05, 0.10, 0.20]:
            sizes.append(
                sizer.calculate_position_size(
                    predicted_return=edge,
                    confidence=0.99,
                    current_volatility=vol,
                    account_equity=200_000.0,
                )
            )
    assert all(abs(v) <= 1.0 for v in sizes)


def test_compute_position_series_respects_limits():
    sizer = PositionSizer(config=_cfg())
    idx = pd.bdate_range("2023-01-01", periods=40)
    prices = pd.Series(100.0 * np.exp(np.linspace(0, 0.1, len(idx))), index=idx)
    signals = pd.Series(1.0, index=idx)
    pred = pd.Series(0.02, index=idx)
    conf = pd.Series(0.95, index=idx)
    vol = pd.Series(0.12, index=idx)

    pos = sizer.compute_position_series(
        signals=signals,
        prices=prices,
        rolling_vol=vol,
        predicted_returns=pred,
        confidence_scores=conf,
        initial_equity=100_000.0,
    )
    assert (pos.abs() <= 1.0 + 1e-12).all()
