"""
paper_trading/simulator.py — Bar-by-bar paper trading simulator.

Replays the trained model signal against historical (or live) market data,
applying the full risk management stack and logging every bar to CSV.

Outputs:
  logs/paper_trading/{ticker}_paper_trades.csv   — one row per bar
  logs/paper_trading/{ticker}_equity_curve.csv   — NAV time series

This is simulation only — no broker connections, no real money.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs/paper_trading")

# ── Column schema for the trades CSV ────────────────────────────────────────
_TRADE_COLS = [
    "timestamp", "ticker", "fill_status", "fill_price",
    "position_qty", "position_side", "proba", "regime",
    "daily_pnl_pct", "unrealised_pnl", "commission_paid",
    "stop_loss_triggered", "nav", "close",
]

_REGIME_MAP = {
    1: "bull", -1: "bear", 0: "sideways", 2: "sideways",
    3: "high_vol", -99: "unknown",
}

_DEFAULT_THRESHOLDS = {
    "bull":     {"long": 0.53, "short": 0.80},
    "sideways": {"long": 0.62, "short": 0.62},
    "bear":     {"long": 0.78, "short": 0.53},
    "high_vol": {"long": 0.68, "short": 0.68},
    "unknown":  {"long": 0.65, "short": 0.65},
}

# Regime-adaptive trailing stops: tighter in bear/high-vol to lock gains faster
_TRAIL_BY_REGIME = {
    "bull":     0.10,   # wide — ride multi-week uptrends
    "sideways": 0.06,   # moderate
    "bear":     0.04,   # tight — lock profits fast in down-trends
    "high_vol": 0.05,   # tight — volatility spikes eat gains quickly
    "unknown":  0.07,
}


@dataclass
class _Position:
    side: str = "FLAT"       # FLAT | LONG | SHORT
    qty: float = 0.0
    entry_price: float = 0.0
    trail_high: float = 0.0
    bars_held: int = 0
    stop_triggered: bool = False


@dataclass
class PaperTradingResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    final_nav: float
    total_return_pct: float
    n_trades: int
    win_rate: float


def run_paper_trade(
    features_df: pd.DataFrame,
    ticker: str,
    model,
    config: dict,
    initial_capital: float = 100_000.0,
    replay_start: Optional[str] = None,
) -> PaperTradingResult:
    """
    Simulate paper trading bar-by-bar.

    Parameters
    ----------
    features_df  : Feature matrix with 'close', 'regime' columns.
    ticker       : Ticker symbol (for logging).
    model        : Trained model with .predict_proba() method.
    config       : System config dict.
    initial_capital : Starting NAV.
    replay_start : ISO date string; earlier bars are skipped.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    bt_cfg   = config.get("backtest", {})
    risk_cfg = config.get("risk", {})
    commission_pct   = float(bt_cfg.get("commission_pct", 0.001))
    slippage_pct     = float(bt_cfg.get("slippage_pct", 0.0005))
    spread_pct       = float(bt_cfg.get("spread_pct", 0.0002))
    stop_loss_pct    = float(bt_cfg.get("stop_loss_pct",
                             risk_cfg.get("stop_loss_pct", 0.06)))
    trail_pct_default = float(bt_cfg.get("trailing_stop_pct",
                               risk_cfg.get("trailing_stop_trail_pct", 0.07)))
    take_profit_pct  = float(bt_cfg.get("take_profit_pct",
                              risk_cfg.get("take_profit_pct", 0.18)))
    min_holding      = int(bt_cfg.get("min_holding_bars", 3))
    confirm_bars     = int(bt_cfg.get("signal_confirmation_bars", 2))
    position_pct     = float(bt_cfg.get("position_size_pct", 0.95))
    daily_loss_limit = float(bt_cfg.get("daily_loss_limit_pct", 0.03))
    fast_entry_thr   = float(bt_cfg.get("fast_entry_threshold", 0.82))
    regime_thr       = bt_cfg.get("regime_thresholds", _DEFAULT_THRESHOLDS)

    # Pre-compute ATR for volatility-scaled position sizing
    _atr_col = next((c for c in ("atr", "ATR", "atr_14") if c in features_df.columns), None)

    # ── Filter to replay window ───────────────────────────────────────────────
    df = features_df.copy()
    if replay_start:
        df = df[df.index >= pd.Timestamp(replay_start)]
    if df.empty:
        raise ValueError(f"No data after replay_start={replay_start}")

    feat_cols = [c for c in df.columns
                 if c not in ("close", "open", "high", "low", "volume",
                              "label", "returns", "fwd_return")]

    # ── Generate model probabilities for all bars at once ────────────────────
    try:
        probas = model.predict_proba(df[feat_cols])
        if not isinstance(probas, np.ndarray):
            probas = np.array(probas, dtype=float)
    except Exception as exc:
        logger.warning("predict_proba failed: %s — using 0.5", exc)
        probas = np.full(len(df), 0.5)

    # ── Bar-by-bar simulation ─────────────────────────────────────────────────
    pos = _Position()
    nav = initial_capital
    peak_nav = initial_capital
    day_open_nav = initial_capital
    rows: list[dict] = []
    signal_buffer: list[int] = []   # for confirmation

    for i, (ts, row) in enumerate(df.iterrows()):
        close = float(row.get("close", row.get("Close", np.nan)))
        if np.isnan(close) or close <= 0:
            continue

        proba = float(probas[i]) if i < len(probas) else 0.5
        regime_code = int(row.get("regime", 0))
        regime = _REGIME_MAP.get(regime_code, "unknown")
        rt = regime_thr.get(regime, regime_thr.get("unknown", {}))
        long_thr  = float(rt.get("long",  0.65))
        short_thr = float(rt.get("short", 0.65))

        # Per-bar ATR-scaled position size: reduce when volatility is elevated
        _vol_scale = 1.0
        if _atr_col is not None and _atr_col in df.columns:
            _atr_val = float(df[_atr_col].iloc[i])
            _atr_pct = _atr_val / close if close > 0 else 0.0
            # Rolling 63-bar median ATR% for reference level
            _atr_median = float(df[_atr_col].iloc[max(0, i-63):i+1].median()) / close if close > 0 else _atr_val
            if _atr_median > 0 and _atr_pct > _atr_median:
                _vol_scale = min(1.0, _atr_median / _atr_pct)
                _vol_scale = max(0.60, _vol_scale)   # never below 60% of normal size

        # Regime-adaptive trailing stop for this bar
        trail_pct = _TRAIL_BY_REGIME.get(regime, trail_pct_default)

        # Raw signal
        if proba >= long_thr:
            raw_sig = 1
        elif proba <= (1.0 - short_thr):
            raw_sig = -1
        else:
            raw_sig = 0

        # High-conviction bypass: skip confirmation when model is very confident
        if abs(proba - 0.5) >= (fast_entry_thr - 0.5):
            confirmed_sig = raw_sig
        else:
            # Signal confirmation buffer
            signal_buffer.append(raw_sig)
            if len(signal_buffer) > confirm_bars:
                signal_buffer.pop(0)
            if len(signal_buffer) == confirm_bars and len(set(signal_buffer)) == 1:
                confirmed_sig = signal_buffer[-1]
            else:
                confirmed_sig = 0  # wait for confirmation

        # Daily loss limit
        day_pnl_pct = (nav - day_open_nav) / day_open_nav if day_open_nav > 0 else 0
        if day_pnl_pct <= -daily_loss_limit:
            confirmed_sig = 0  # flat for rest of day

        # ── Risk checks on existing position ─────────────────────────────────
        fill_status = "no_action"
        fill_price = 0.0
        commission = 0.0
        stop_triggered = False
        unrealised_pnl = 0.0

        if pos.side == "LONG":
            unrealised_pnl = (close - pos.entry_price) * pos.qty
            # Update trail high
            if close > pos.trail_high:
                pos.trail_high = close
            # Take-profit: lock gains when price rises enough
            if (close - pos.entry_price) / pos.entry_price >= take_profit_pct and pos.bars_held >= min_holding:
                fill_price = close * (1 - slippage_pct)
                commission = fill_price * pos.qty * commission_pct
                nav += (fill_price - pos.entry_price) * pos.qty - commission
                fill_status = "take_profit"
                pos = _Position()
            # Hard stop loss
            elif close <= pos.entry_price * (1 - stop_loss_pct):
                fill_price = close * (1 - slippage_pct)
                commission = fill_price * pos.qty * commission_pct
                nav += (fill_price - pos.entry_price) * pos.qty - commission
                fill_status = "stop_loss"
                stop_triggered = True
                pos = _Position()
            # Trailing stop
            elif close <= pos.trail_high * (1 - trail_pct) and pos.bars_held >= min_holding:
                fill_price = close * (1 - slippage_pct)
                commission = fill_price * pos.qty * commission_pct
                nav += (fill_price - pos.entry_price) * pos.qty - commission
                fill_status = "close_long"
                pos = _Position()
            else:
                pos.bars_held += 1

        elif pos.side == "SHORT":
            unrealised_pnl = (pos.entry_price - close) * pos.qty
            # Take-profit for short
            if (pos.entry_price - close) / pos.entry_price >= take_profit_pct and pos.bars_held >= min_holding:
                fill_price = close * (1 + slippage_pct)
                commission = fill_price * pos.qty * commission_pct
                nav += (pos.entry_price - fill_price) * pos.qty - commission
                fill_status = "take_profit"
                pos = _Position()
            elif close <= pos.trail_high * (1 - trail_pct) and pos.bars_held >= min_holding:
                fill_price = close * (1 + slippage_pct)
                commission = fill_price * pos.qty * commission_pct
                nav += (pos.entry_price - fill_price) * pos.qty - commission
                fill_status = "close_short"
                pos = _Position()
            else:
                pos.trail_high = min(pos.trail_high, close)
                pos.bars_held += 1

        # ── Open new position if flat and signal confirmed ────────────────────
        if pos.side == "FLAT" and fill_status == "no_action":
            _eff_pct = position_pct * _vol_scale
            if confirmed_sig == 1:
                qty = (nav * _eff_pct) / (close * (1 + spread_pct + slippage_pct))
                entry = close * (1 + slippage_pct + spread_pct)
                commission = entry * qty * commission_pct
                nav -= commission
                pos = _Position(side="LONG", qty=qty, entry_price=entry,
                                trail_high=entry)
                fill_price = entry
                fill_status = "long_entry"
                unrealised_pnl = 0.0
            elif confirmed_sig == -1:
                qty = (nav * _eff_pct) / (close * (1 + spread_pct + slippage_pct))
                entry = close * (1 - slippage_pct - spread_pct)
                commission = entry * qty * commission_pct
                nav -= commission
                pos = _Position(side="SHORT", qty=qty, entry_price=entry,
                                trail_high=entry)
                fill_price = entry
                fill_status = "short_entry"
                unrealised_pnl = 0.0

        # Track peak NAV and daily open
        if nav > peak_nav:
            peak_nav = nav
        if i == 0 or ts.date() != pd.Timestamp(rows[-1]["timestamp"]).date() if rows else True:
            day_open_nav = nav

        total_nav = nav + unrealised_pnl
        daily_pnl_pct = (total_nav - day_open_nav) / day_open_nav if day_open_nav > 0 else 0.0

        rows.append({
            "timestamp":        ts,
            "ticker":           ticker,
            "fill_status":      fill_status,
            "fill_price":       round(fill_price, 4) if fill_price else np.nan,
            "position_qty":     round(pos.qty, 4),
            "position_side":    pos.side,
            "proba":            round(proba, 4),
            "regime":           regime_code,
            "daily_pnl_pct":   round(daily_pnl_pct, 6),
            "unrealised_pnl":  round(unrealised_pnl, 2),
            "commission_paid": round(commission, 2),
            "stop_loss_triggered": stop_triggered,
            "nav":             round(total_nav, 2),
            "close":           round(close, 4),
        })

    trades_df = pd.DataFrame(rows, columns=_TRADE_COLS)
    equity_df = trades_df[["timestamp", "nav"]].copy()
    equity_df.columns = ["date", "nav"]
    equity_df["daily_return"] = equity_df["nav"].pct_change().fillna(0)

    # ── Save CSV logs ─────────────────────────────────────────────────────────
    trades_path = LOG_DIR / f"{ticker.lower()}_paper_trades.csv"
    equity_path = LOG_DIR / f"{ticker.lower()}_equity_curve.csv"
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    logger.info("Saved %d bars → %s", len(trades_df), trades_path)

    # ── Summary stats ─────────────────────────────────────────────────────────
    final_nav = float(equity_df["nav"].iloc[-1]) if len(equity_df) else initial_capital
    total_return = (final_nav - initial_capital) / initial_capital * 100
    _all_exits = ["stop_loss", "close_long", "close_short", "take_profit"]
    entry_rows = trades_df[trades_df["fill_status"].isin(
        ["long_entry", "short_entry"] + _all_exits
    )]
    n_trades = len(entry_rows[entry_rows["fill_status"].isin(["long_entry", "short_entry"])])
    exits = entry_rows[entry_rows["fill_status"].isin(_all_exits)]
    win_rate = 0.0
    if len(exits):
        wins = sum(1 for _, r in exits.iterrows()
                   if (r["daily_pnl_pct"] > 0))
        win_rate = wins / len(exits)

    return PaperTradingResult(
        trades=trades_df,
        equity=equity_df,
        final_nav=final_nav,
        total_return_pct=total_return,
        n_trades=n_trades,
        win_rate=win_rate,
    )
