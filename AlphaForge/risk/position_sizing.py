"""Risk management and position sizing for simulation/backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger, load_config, max_drawdown

logger = get_logger(__name__)


@dataclass
class RiskParams:
    target_annual_volatility: float = 0.15
    kelly_fraction: float = 0.25
    fixed_risk_pct: float = 0.01
    max_leverage: float = 2.0
    daily_loss_limit_pct: float = 0.03
    min_confidence_threshold: float = 0.60
    max_position_size_per_asset: float = 1.0


class PositionSizer:
    """Combines volatility targeting, fractional Kelly, and fixed-risk sizing."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg = config or load_config()
        risk_cfg = self.cfg.get("risk", {})
        self.params = RiskParams(
            target_annual_volatility=float(risk_cfg.get("target_annual_volatility", 0.15)),
            kelly_fraction=float(risk_cfg.get("kelly_fraction", 0.25)),
            fixed_risk_pct=float(risk_cfg.get("fixed_risk_pct", 0.01)),
            max_leverage=float(risk_cfg.get("max_leverage", 2.0)),
            daily_loss_limit_pct=float(risk_cfg.get("daily_loss_limit_pct", 0.03)),
            min_confidence_threshold=float(risk_cfg.get("min_confidence_threshold", 0.60)),
            max_position_size_per_asset=float(risk_cfg.get("max_position_size_per_asset", 1.0)),
        )
        self.max_dd = float(risk_cfg.get("max_portfolio_drawdown", 0.25))

    def size_volatility_targeting(self, current_volatility: float) -> float:
        """Return exposure scalar based on target annual volatility."""
        if current_volatility <= 0 or np.isnan(current_volatility):
            return 0.0
        return self.params.target_annual_volatility / (current_volatility + 1e-9)

    # Backward-compatible helper used by paper_trading/loop.py
    def size_volatility_target(
        self,
        signal: float,
        predicted_vol: float,
        nav: float,
        price: float,
        periods_per_year: int = 252,
    ) -> int:
        _ = periods_per_year
        if predicted_vol <= 0 or price <= 0 or nav <= 0:
            return 0
        exposure = self.size_volatility_targeting(predicted_vol)
        exposure = float(np.clip(exposure, -self.params.max_position_size_per_asset, self.params.max_position_size_per_asset))
        target_qty = int(np.floor((abs(exposure) * nav) / price))
        return int(np.sign(signal) * target_qty)

    def size_fractional_kelly(self, predicted_edge: float, historical_variance: float) -> float:
        """Fractional Kelly sizing from edge and return variance."""
        if historical_variance <= 0 or np.isnan(historical_variance):
            return 0.0
        full_kelly = predicted_edge / (historical_variance + 1e-12)
        return self.params.kelly_fraction * full_kelly

    def size_fixed_risk(self, current_volatility: float) -> float:
        """
        Fixed-risk sizing: risk a fixed equity % per trade.
        Approximates stop distance as current volatility.
        """
        if current_volatility <= 0 or np.isnan(current_volatility):
            return 0.0
        return self.params.fixed_risk_pct / (current_volatility + 1e-9)

    def enforce_hard_limits(
        self,
        raw_size: float,
        confidence: float,
        daily_return: float = 0.0,
    ) -> float:
        if confidence < self.params.min_confidence_threshold:
            return 0.0
        if daily_return <= -abs(self.params.daily_loss_limit_pct):
            return 0.0
        size = float(np.clip(raw_size, -self.params.max_leverage, self.params.max_leverage))
        size = float(np.clip(size, -self.params.max_position_size_per_asset, self.params.max_position_size_per_asset))
        # Function contract is normalized recommendation in [-1, 1].
        return float(np.clip(size, -1.0, 1.0))

    def calculate_position_size(
        self,
        predicted_return: float,
        confidence: float,
        current_volatility: float,
        account_equity: float,
        historical_variance: Optional[float] = None,
        daily_return: float = 0.0,
    ) -> float:
        """
        Core sizing function returning a normalized position in [-1.0, +1.0].
        """
        _ = account_equity  # kept for API completeness and future extensions.
        edge = float(predicted_return)
        variance = float(historical_variance) if historical_variance is not None else float(current_volatility**2)

        vol_size = self.size_volatility_targeting(current_volatility)
        kelly_size = self.size_fractional_kelly(edge, variance)
        fixed_size = self.size_fixed_risk(current_volatility)

        base_magnitude = np.median([abs(vol_size), abs(kelly_size), abs(fixed_size)])
        signed = np.sign(edge) * base_magnitude
        confidence_scale = np.clip(
            (confidence - self.params.min_confidence_threshold) / (1 - self.params.min_confidence_threshold + 1e-9),
            0.0,
            1.0,
        )
        raw = signed * confidence_scale
        return self.enforce_hard_limits(raw, confidence=confidence, daily_return=daily_return)

    def compute_position_series(
        self,
        signals: pd.Series,
        prices: pd.Series,
        rolling_vol: Optional[pd.Series] = None,
        predicted_returns: Optional[pd.Series] = None,
        confidence_scores: Optional[pd.Series] = None,
        initial_equity: float = 100_000.0,
    ) -> pd.Series:
        """
        Convert raw directional signals into risk-managed normalized exposures.
        """
        idx = signals.index.intersection(prices.index)
        sig = signals.reindex(idx).fillna(0.0)
        close = prices.reindex(idx).astype(float)
        vol = rolling_vol.reindex(idx) if rolling_vol is not None else close.pct_change().rolling(21).std().mul(np.sqrt(252))
        pred = predicted_returns.reindex(idx) if predicted_returns is not None else sig * vol.fillna(0.0)
        conf = confidence_scores.reindex(idx) if confidence_scores is not None else pd.Series(1.0, index=idx)

        positions = pd.Series(0.0, index=idx)
        equity = float(initial_equity)
        prev_position = 0.0
        prev_close = close.iloc[0]

        for i, ts in enumerate(idx):
            daily_ret = 0.0
            if i > 0 and prev_close > 0:
                daily_ret = prev_position * ((close.iloc[i] / prev_close) - 1.0)
                equity *= (1.0 + daily_ret)
            prev_close = close.iloc[i]

            if sig.loc[ts] == 0:
                positions.loc[ts] = 0.0
                prev_position = 0.0
                continue

            sized = self.calculate_position_size(
                predicted_return=float(pred.loc[ts]),
                confidence=float(conf.loc[ts]),
                current_volatility=float(vol.loc[ts]) if not np.isnan(vol.loc[ts]) else 0.0,
                account_equity=equity,
                historical_variance=float((vol.loc[ts] ** 2)) if not np.isnan(vol.loc[ts]) else None,
                daily_return=float(daily_ret),
            )
            # Respect raw direction from caller.
            positions.loc[ts] = float(np.sign(sig.loc[ts]) * abs(sized))
            prev_position = positions.loc[ts]

        return positions

    def check_drawdown_halt(self, equity_curve: pd.Series) -> bool:
        if len(equity_curve) < 2:
            return False
        current_dd = abs(max_drawdown(equity_curve))
        return current_dd >= self.max_dd


def calculate_position_size(
    predicted_return: float,
    confidence: float,
    current_volatility: float,
    account_equity: float,
    config: Optional[dict] = None,
) -> float:
    """
    Public function required by the risk module API.
    Returns normalized size in [-1.0, +1.0].
    """
    return PositionSizer(config=config).calculate_position_size(
        predicted_return=predicted_return,
        confidence=confidence,
        current_volatility=current_volatility,
        account_equity=account_equity,
    )


def high_volatility_stress_test_periods() -> dict[str, tuple[str, str]]:
    """High-volatility focused stress periods."""
    return {
        "march_2020_crash": ("2020-02-19", "2020-03-31"),
        "2022_bear_market": ("2022-01-01", "2022-12-31"),
    }
