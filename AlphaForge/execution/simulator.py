"""
Simulated execution engine.
This module models order fills with realistic transaction costs.
NO real broker connections. NO live orders. Simulation only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger, load_config, sharpe_ratio

logger = get_logger(__name__)

OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["pending", "filled", "rejected", "cancelled"]
ExecutionMode = Literal["realistic", "conservative"]


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    ticker: str = ""
    side: OrderSide = "buy"
    quantity: int = 0
    limit_price: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: OrderStatus = "pending"
    fill_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    filled_quantity: int = 0
    requested_quantity: int = 0
    partial_fill_reason: str = ""


@dataclass
class Position:
    ticker: str
    quantity: int = 0
    avg_entry_price: float = 0.0
    realised_pnl: float = 0.0
    total_entry_commission: float = 0.0  # cumulative commission paid on all buy fills

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_entry_price

    def unrealised_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_entry_price)


class ExecutionSimulator:
    """
    Simulates order execution with slippage, commission, and market impact.
    All positions and P&L are tracked in memory — nothing is sent anywhere.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        initial_capital: float = 100_000.0,
        mode: ExecutionMode = "realistic",
    ) -> None:
        self.cfg = config or load_config()
        bt_cfg = self.cfg.get("backtest", {})
        exe_cfg = self.cfg.get("execution", {})

        self.mode = mode
        self.commission_pct = bt_cfg.get("commission_pct", 0.001)
        self.base_slippage_pct = bt_cfg.get("slippage_pct", 0.0005)
        self.spread_pct = bt_cfg.get("spread_pct", 0.0002)
        self.limit_buffer_bps = float(exe_cfg.get("limit_buffer_bps", 4))
        self.vol_window = int(exe_cfg.get("vol_window", 20))
        self.min_fill_ratio = float(exe_cfg.get("min_fill_ratio", 0.2))
        self.max_participation = float(exe_cfg.get("max_participation", 0.10))

        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: dict[str, Position] = {}
        self.order_log: list[Order] = []
        self.equity_history: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def submit_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        market_price: float,
        timestamp: datetime,
        day_ohlc: Optional[pd.Series] = None,
        day_volume: Optional[float] = None,
        day_volatility: Optional[float] = None,
    ) -> Order:
        order = Order(ticker=ticker, side=side, quantity=quantity, requested_quantity=quantity)

        if quantity <= 0:
            order.status = "rejected"
            logger.warning("Rejected order: quantity=%d", quantity)
            self.order_log.append(order)
            return order

        limit_price = self._get_limit_price(market_price, side, day_volatility)
        order.limit_price = limit_price

        if day_ohlc is not None and not self._limit_was_reached(day_ohlc, side, limit_price):
            order.status = "cancelled"
            order.partial_fill_reason = "limit_not_reached"
            self.order_log.append(order)
            return order

        fill_qty, partial_reason = self._compute_fill_quantity(quantity, day_volume)
        if fill_qty <= 0:
            order.status = "cancelled"
            order.partial_fill_reason = partial_reason or "no_liquidity"
            self.order_log.append(order)
            return order

        fill_price = self._simulate_fill(limit_price, side, day_volatility)
        commission = self._compute_commission(fill_price, fill_qty)
        cost = fill_price * fill_qty + commission

        if side == "buy" and cost > self.cash:
            max_affordable = int(np.floor(self.cash / max(fill_price, 1e-9)))
            fill_qty = min(fill_qty, max_affordable)
            commission = self._compute_commission(fill_price, fill_qty)
            cost = fill_price * fill_qty + commission
            partial_reason = "cash_limited" if fill_qty > 0 else "insufficient_cash"
        if side == "buy" and fill_qty <= 0:
            order.status = "rejected"
            logger.warning(
                "Rejected BUY %s x%d: insufficient cash (need %.2f, have %.2f)",
                ticker, quantity, cost, self.cash
            )
            self.order_log.append(order)
            return order

        self._apply_fill(
            order,
            ticker,
            side,
            fill_qty,
            fill_price,
            commission,
            timestamp,
            partial_reason=partial_reason,
        )
        return order

    def mark_to_market(self, prices: dict[str, float], timestamp: datetime) -> float:
        pos_value = sum(
            pos.quantity * prices[pos.ticker]
            for pos in self.positions.values()
            if pos.ticker in prices
        )
        nav = self.cash + pos_value
        self.equity_history.append((timestamp, nav))
        return nav

    @property
    def nav(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def equity_curve(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float)
        times, values = zip(*self.equity_history)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")

    def position_summary(self) -> pd.DataFrame:
        rows = [
            {
                "ticker": p.ticker,
                "quantity": p.quantity,
                "avg_entry": p.avg_entry_price,
                "market_value": p.market_value,
                "realised_pnl": p.realised_pnl,
            }
            for p in self.positions.values()
        ]
        return pd.DataFrame(rows)

    def trade_log(self) -> pd.DataFrame:
        rows = [
            {
                "id": o.id,
                "ticker": o.ticker,
                "side": o.side,
                "quantity": o.quantity,
                "fill_price": o.fill_price,
                "requested_quantity": o.requested_quantity,
                "filled_quantity": o.filled_quantity,
                "commission": o.commission,
                "slippage": o.slippage,
                "partial_fill_reason": o.partial_fill_reason,
                "status": o.status,
                "filled_at": o.filled_at,
            }
            for o in self.order_log
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        market_price: float,
        side: OrderSide,
        day_volatility: Optional[float],
    ) -> float:
        slip_mult = self._volatility_slippage_multiplier(day_volatility)
        spread = market_price * self.spread_pct / 2
        slip = market_price * self.base_slippage_pct * slip_mult
        if side == "buy":
            return market_price + spread + slip
        else:
            return market_price - spread - slip

    def _compute_commission(self, fill_price: float, quantity: int) -> float:
        return fill_price * quantity * self.commission_pct

    def _volatility_slippage_multiplier(self, day_volatility: Optional[float]) -> float:
        vol = 0.20 if day_volatility is None or np.isnan(day_volatility) else float(day_volatility)
        if self.mode == "conservative":
            base = 1.35
            sensitivity = 2.0
        else:
            base = 1.0
            sensitivity = 1.2
        return base * (1.0 + sensitivity * max(vol - 0.15, 0.0))

    def _get_limit_price(self, reference_price: float, side: OrderSide, day_volatility: Optional[float]) -> float:
        vol = 0.20 if day_volatility is None or np.isnan(day_volatility) else float(day_volatility)
        mode_buffer = 1.5 if self.mode == "conservative" else 1.0
        buffer = reference_price * (self.limit_buffer_bps / 10_000.0) * mode_buffer * (1 + max(vol - 0.15, 0.0))
        if side == "buy":
            return reference_price - buffer
        return reference_price + buffer

    def _limit_was_reached(self, day_ohlc: pd.Series, side: OrderSide, limit_price: float) -> bool:
        day_low = float(day_ohlc.get("low", np.nan))
        day_high = float(day_ohlc.get("high", np.nan))
        if np.isnan(day_low) or np.isnan(day_high):
            # Fallback: assume reachable when intraday range unavailable
            return True
        if side == "buy":
            return day_low <= limit_price
        return day_high >= limit_price

    def _compute_fill_quantity(self, requested_qty: int, day_volume: Optional[float]) -> tuple[int, str]:
        if day_volume is None or np.isnan(day_volume):
            return requested_qty, ""
        participation = self.max_participation * (0.6 if self.mode == "conservative" else 1.0)
        cap = int(max(np.floor(day_volume * participation), 0))
        if cap <= 0:
            return 0, "no_day_volume"
        if requested_qty <= cap:
            return requested_qty, ""
        partial_qty = cap
        if partial_qty < int(np.ceil(requested_qty * self.min_fill_ratio)):
            return 0, "liquidity_too_thin"
        return partial_qty, "volume_capped_partial_fill"

    def _apply_fill(
        self,
        order: Order,
        ticker: str,
        side: OrderSide,
        quantity: int,
        fill_price: float,
        commission: float,
        timestamp: datetime,
        partial_reason: str = "",
    ) -> None:
        order.fill_price = fill_price
        order.commission = commission
        order.slippage = abs(fill_price - (order.limit_price if order.limit_price is not None else fill_price))
        order.status = "filled"
        order.filled_at = timestamp
        order.filled_quantity = quantity
        order.partial_fill_reason = partial_reason
        self.order_log.append(order)

        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
        pos = self.positions[ticker]

        if side == "buy":
            if pos.quantity >= 0:
                # Opening or adding to a long position
                total_cost = pos.avg_entry_price * pos.quantity + fill_price * quantity
                pos.quantity += quantity
                pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0.0
                pos.total_entry_commission += commission
            else:
                # Covering a short position: profit = entry_price - cover_price
                cover_qty = min(quantity, abs(pos.quantity))
                gross = (pos.avg_entry_price - fill_price) * cover_qty
                entry_comm_share = pos.total_entry_commission * (cover_qty / max(abs(pos.quantity), 1))
                realised = gross - entry_comm_share - commission
                pos.realised_pnl += realised
                pos.total_entry_commission -= entry_comm_share
                pos.quantity += quantity
                if pos.quantity >= 0:
                    if ticker in self.positions:
                        del self.positions[ticker]
            self.cash -= fill_price * quantity + commission
        else:
            if pos.quantity > 0:
                # Closing or reducing a long position: profit = fill_price - entry_price
                close_qty = min(quantity, pos.quantity)
                gross = (fill_price - pos.avg_entry_price) * close_qty
                entry_comm_share = pos.total_entry_commission * (close_qty / max(pos.quantity, 1))
                realised = gross - entry_comm_share - commission
                pos.realised_pnl += realised
                pos.total_entry_commission -= entry_comm_share
                pos.quantity -= quantity
                if pos.quantity <= 0:
                    if ticker in self.positions:
                        del self.positions[ticker]
            else:
                # Opening or adding to a short position: record short entry price
                existing_short = abs(pos.quantity)
                total_short_notional = pos.avg_entry_price * existing_short + fill_price * quantity
                pos.quantity -= quantity  # becomes more negative
                new_short = abs(pos.quantity)
                pos.avg_entry_price = total_short_notional / new_short if new_short > 0 else fill_price
                pos.total_entry_commission += commission
            self.cash += fill_price * quantity - commission

        logger.debug(
            "FILL %s %s x%d @ %.4f | commission=%.2f | cash=%.2f",
            side.upper(), ticker, quantity, fill_price, commission, self.cash
        )


def simulate_execution(
    target_positions: pd.Series,
    prices: pd.DataFrame | pd.Series,
    volume: Optional[pd.Series] = None,
    mode: ExecutionMode = "realistic",
    initial_capital: float = 100_000.0,
    config: Optional[dict] = None,
    latency_model: Optional[LatencyModel] = None,
) -> dict[str, object]:
    """
    Simulate realistic execution from target positions.

    target_positions: normalized target exposure in [-1, 1].
    prices: DataFrame with close/high/low preferred (Series close also supported).
    """
    cfg = config or load_config()
    sim = ExecutionSimulator(config=cfg, initial_capital=initial_capital, mode=mode)

    if isinstance(prices, pd.Series):
        px = pd.DataFrame({"close": prices})
    else:
        px = prices.copy()
    if "close" not in px.columns:
        raise ValueError("prices must contain a 'close' column")
    if "high" not in px.columns:
        px["high"] = px["close"]
    if "low" not in px.columns:
        px["low"] = px["close"]
    if volume is None:
        vol_ser = px["volume"] if "volume" in px.columns else pd.Series(np.nan, index=px.index)
    else:
        vol_ser = volume.reindex(px.index)

    idx = target_positions.index.intersection(px.index)
    target = target_positions.reindex(idx).fillna(0.0).clip(-1.0, 1.0)
    close = px["close"].reindex(idx).astype(float)
    roll_vol = close.pct_change().rolling(20, min_periods=5).std().mul(np.sqrt(252))

    for ts in idx:
        mkt_price = float(close.loc[ts])
        sim.mark_to_market({"ASSET": mkt_price}, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)

        current_qty = sim.positions.get("ASSET").quantity if "ASSET" in sim.positions else 0
        nav = sim.nav
        target_qty = int(np.floor((target.loc[ts] * nav) / max(mkt_price, 1e-9)))
        delta = target_qty - current_qty
        if delta == 0:
            continue
        side: OrderSide = "buy" if delta > 0 else "sell"
        day_vol_val = float(vol_ser.loc[ts]) if ts in vol_ser.index and not pd.isna(vol_ser.loc[ts]) else None
        adjusted_price = mkt_price
        if latency_model is not None and day_vol_val is not None:
            adjusted_price = mkt_price * latency_model.price_adjustment_factor(
                abs(delta), day_vol_val, side
            )
        sim.submit_order(
            ticker="ASSET",
            side=side,
            quantity=abs(delta),
            market_price=adjusted_price,
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            day_ohlc=px.loc[ts, ["high", "low", "close"]],
            day_volume=day_vol_val,
            day_volatility=float(roll_vol.loc[ts]) if not pd.isna(roll_vol.loc[ts]) else None,
        )

    equity = sim.equity_curve()
    returns = equity.pct_change().fillna(0.0) if not equity.empty else pd.Series(dtype=float)
    out = {
        "equity_curve": equity,
        "returns": returns,
        "trade_log": sim.trade_log(),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else float(initial_capital),
        "sharpe_ratio": float(sharpe_ratio(returns)) if not returns.empty else 0.0,
        "mode": mode,
    }
    return out


@dataclass
class LatencyModel:
    """
    Models market-impact latency for the execution simulator.

    price_adjustment_factor() returns a multiplier applied to the fill price
    to represent the adverse price movement caused by order size relative to ADV.
    """
    base_latency_ms:              float = 5.0    # baseline inter-day latency (informational)
    jitter_std_ms:                float = 2.0    # random latency jitter (informational)
    market_impact_bps_per_pct:    float = 1.0    # basis-pts of impact per 1% of ADV

    def price_adjustment_factor(self, quantity: float, adv: float, side: OrderSide) -> float:
        """
        Return a fill-price multiplier accounting for market impact.

        Parameters
        ----------
        quantity : float   Shares / units being traded.
        adv : float        Average daily volume for the asset.
        side : OrderSide   "buy" or "sell".

        Returns
        -------
        float — multiply fill_price by this; >1 for buys (adverse), <1 for sells.
        """
        participation = abs(quantity) / max(adv, 1.0)       # fraction of ADV
        impact_bps    = self.market_impact_bps_per_pct * participation * 100
        direction     = 1 if side == "buy" else -1
        return 1.0 + direction * impact_bps / 10_000


def compare_execution_modes(
    target_positions: pd.Series,
    prices: pd.DataFrame | pd.Series,
    volume: Optional[pd.Series] = None,
    initial_capital: float = 100_000.0,
    config: Optional[dict] = None,
) -> dict[str, object]:
    """
    Compare idealized execution vs realistic simulator impact.
    """
    idx = target_positions.index
    if isinstance(prices, pd.DataFrame):
        close = prices["close"].reindex(idx).astype(float)
    else:
        close = prices.reindex(idx).astype(float)

    # Idealized: immediate frictionless rebalance to target.
    tgt = target_positions.reindex(idx).fillna(0.0).clip(-1.0, 1.0)
    ideal_ret = tgt.shift(1).fillna(0.0) * close.pct_change().fillna(0.0)
    ideal_eq = initial_capital * (1.0 + ideal_ret).cumprod()
    ideal = {
        "equity_curve": ideal_eq,
        "final_equity": float(ideal_eq.iloc[-1]) if not ideal_eq.empty else initial_capital,
        "sharpe_ratio": float(sharpe_ratio(ideal_ret)) if not ideal_ret.empty else 0.0,
    }

    realistic = simulate_execution(
        target_positions=target_positions,
        prices=prices,
        volume=volume,
        mode="realistic",
        initial_capital=initial_capital,
        config=config,
    )

    comparison = {
        "idealized": ideal,
        "realistic_execution": realistic,
        "difference": {
            "final_equity_delta": float(realistic["final_equity"] - ideal["final_equity"]),
            "sharpe_delta": float(realistic["sharpe_ratio"] - ideal["sharpe_ratio"]),
        },
    }
    return comparison
