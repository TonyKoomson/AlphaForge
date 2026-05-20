"""
risk/pretrade_risk.py
=====================
Pre-trade risk orchestration with 14 checks for AlphaForge v2.0.

All checks run before any simulated order is placed. Hard-block checks
reject the trade entirely; soft-block checks scale down the position.

Hard blocks (trade rejected):
    daily_loss_limit, drawdown_gate, vix_filter, max_leverage

Soft blocks (position scaled down):
    all other failed checks; scale = 0.5^n_failures (floor 0.10)

Usage
-----
    orch = PreTradeRiskOrchestrator(**cfg["pretrade_risk"])
    result = orch.run_checks(order, portfolio_state)
    if result.approved:
        execute(result.final_size)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

_HARD_BLOCKS = frozenset({"daily_loss_limit", "drawdown_gate", "vix_filter", "max_leverage"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PreTradeCheck:
    name: str
    passed: bool
    value: float
    limit: float
    message: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": round(float(self.value), 6),
            "limit": round(float(self.limit), 6),
            "message": self.message,
        }


@dataclass
class PreTradeResult:
    approved: bool
    checks: list = field(default_factory=list)           # list[PreTradeCheck]
    final_size: float = 0.0
    original_size: float = 0.0
    rejection_reason: Optional[str] = None
    adjustments_made: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "final_size": round(self.final_size, 6),
            "original_size": round(self.original_size, 6),
            "rejection_reason": self.rejection_reason,
            "adjustments_made": self.adjustments_made,
            "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
            "n_passed": sum(1 for c in self.checks if c.passed),
            "n_failed": sum(1 for c in self.checks if not c.passed),
        }

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        failed = [c.name for c in self.checks if not c.passed]
        return f"[{status}] size={self.final_size:.4f} failed={failed}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PreTradeRiskOrchestrator:
    """
    Runs 14 pre-trade risk checks before any simulated order is placed.

    Config keys  (config["pretrade_risk"])
    --------------------------------------
    max_position_size    float = 0.25   Fraction of portfolio per position
    max_sector_exposure  float = 0.40   Max sector concentration
    cvar_limit           float = 0.03   5%-CVaR * size must not exceed this
    min_cash_reserve     float = 0.05   Cash floor after trade
    max_leverage         float = 2.0    Absolute leverage cap (HARD BLOCK)
    max_portfolio_risk   float = 0.02   vol * |size| risk contribution cap
    max_daily_loss_pct   float = 0.03   Daily P&L floor (HARD BLOCK)
    max_drawdown_pct     float = 0.20   Drawdown gate (HARD BLOCK)
    vix_halt_threshold   float = 40.0   VIX level that halts trading (HARD BLOCK)
    correlation_gate     float = 0.85   Max correlation to any existing holding
    atr_multiplier       float = 2.0    ATR × multiplier = stop distance
    min_stop_loss_pct    float = 0.02   Minimum required stop-loss distance
    track_caps           dict  = {}     Track-specific exposure overrides
    """

    def __init__(
        self,
        max_position_size: float = 0.25,
        max_sector_exposure: float = 0.40,
        cvar_limit: float = 0.03,
        min_cash_reserve: float = 0.05,
        max_leverage: float = 2.0,
        max_portfolio_risk: float = 0.02,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.20,
        vix_halt_threshold: float = 40.0,
        correlation_gate: float = 0.85,
        atr_multiplier: float = 2.0,
        min_stop_loss_pct: float = 0.02,
        track_caps: Optional[dict] = None,
        **_kwargs,
    ) -> None:
        self.max_position_size   = float(max_position_size)
        self.max_sector_exposure = float(max_sector_exposure)
        self.cvar_limit          = float(cvar_limit)
        self.min_cash_reserve    = float(min_cash_reserve)
        self.max_leverage        = float(max_leverage)
        self.max_portfolio_risk  = float(max_portfolio_risk)
        self.max_daily_loss_pct  = float(max_daily_loss_pct)
        self.max_drawdown_pct    = float(max_drawdown_pct)
        self.vix_halt_threshold  = float(vix_halt_threshold)
        self.correlation_gate    = float(correlation_gate)
        self.atr_multiplier      = float(atr_multiplier)
        self.min_stop_loss_pct   = float(min_stop_loss_pct)
        self.track_caps          = track_caps or {}

    # ── Main entry ────────────────────────────────────────────────────────────

    def run_checks(
        self,
        order: dict,
        portfolio_state: dict,
        market_data: Optional[pd.DataFrame] = None,
    ) -> PreTradeResult:
        """
        Run all 14 pre-trade checks and return a PreTradeResult.

        Parameters
        ----------
        order : dict
            ticker          Asset symbol
            size            Proposed position (fraction of portfolio, signed)
            sector          Asset sector label (default "unknown")
            track           Research track ("defensive"|"aggressive"|...)
            confidence      Model confidence 0–1
            atr             ATR value (optional)
            current_price   Asset price (optional, for ATR check)
            stop_loss_pct   Stop distance (optional, for stop-loss check)
        portfolio_state : dict
            nav                 Current portfolio NAV
            positions           dict {ticker: size}
            sector_exposures    dict {sector: total_weight}
            current_drawdown    float 0–1
            daily_pnl_pct       Fractional daily P&L (negative = loss)
            cash_pct            Cash as fraction of NAV
            current_leverage    Current portfolio leverage
            returns_series      pd.Series of recent portfolio returns
            vix                 Current VIX level (optional)
            existing_returns    dict {ticker: pd.Series} for correlation
        """
        original_size = float(order.get("size", 0.0))
        final_size    = original_size
        adjustments: list[str] = []

        # --- Run all 14 checks ---
        checks: list[PreTradeCheck] = []

        checks.append(self._chk_max_position(original_size))            # 1
        checks.append(self._chk_sector_concentration(order, portfolio_state))  # 2
        checks.append(self._chk_cvar(portfolio_state, original_size))   # 3

        atr_chk, atr_size = self._chk_atr_sizing(order, original_size)  # 4
        checks.append(atr_chk)
        if not atr_chk.passed and abs(atr_size) < abs(final_size):
            final_size = atr_size
            adjustments.append(f"ATR cap → size={atr_size:.4f}")

        checks.append(self._chk_cash_reserve(portfolio_state, original_size))  # 5
        checks.append(self._chk_stop_loss(order))                        # 6
        checks.append(self._chk_track_exposure(order, portfolio_state))  # 7
        checks.append(self._chk_vol_targeting(order, portfolio_state))   # 8
        checks.append(self._chk_correlation(order, portfolio_state))     # 9
        checks.append(self._chk_leverage(portfolio_state, original_size))# 10
        checks.append(self._chk_daily_loss(portfolio_state))             # 11
        checks.append(self._chk_drawdown(portfolio_state))               # 12
        checks.append(self._chk_vix(portfolio_state))                    # 13
        checks.append(self._chk_risk_budget(order, portfolio_state, original_size))  # 14

        # Hard failures → outright rejection
        hard_failures = [c for c in checks if not c.passed and c.name in _HARD_BLOCKS]
        approved = not hard_failures
        rejection_reason = hard_failures[0].message if hard_failures else None

        # Soft failures → scale down
        soft_failures = [c for c in checks if not c.passed and c.name not in _HARD_BLOCKS]
        if soft_failures and approved:
            scale = max(0.5 ** len(soft_failures), 0.10)
            final_size *= scale
            adjustments.append(f"soft-block scale {scale:.2f} ({len(soft_failures)} failures)")

        if not approved:
            final_size = 0.0

        result = PreTradeResult(
            approved=approved,
            checks=checks,
            final_size=float(np.clip(final_size, -self.max_position_size, self.max_position_size)),
            original_size=original_size,
            rejection_reason=rejection_reason,
            adjustments_made=adjustments,
        )
        logger.debug("PreTrade %s", result.summary())
        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _chk_max_position(self, size: float) -> PreTradeCheck:
        v = abs(size)
        ok = v <= self.max_position_size
        return PreTradeCheck(
            "max_position_size", ok, v, self.max_position_size,
            f"|size|={v:.4f} {'ok' if ok else 'EXCEEDS'} max={self.max_position_size:.4f}",
        )

    def _chk_sector_concentration(self, order: dict, ps: dict) -> PreTradeCheck:
        sector = order.get("sector", "unknown")
        current = float(ps.get("sector_exposures", {}).get(sector, 0.0))
        projected = current + abs(float(order.get("size", 0.0)))
        ok = projected <= self.max_sector_exposure
        return PreTradeCheck(
            "sector_concentration", ok, projected, self.max_sector_exposure,
            f"sector '{sector}' projected={projected:.4f} limit={self.max_sector_exposure:.4f}",
        )

    def _chk_cvar(self, ps: dict, size: float) -> PreTradeCheck:
        ret = ps.get("returns_series")
        if ret is None or (isinstance(ret, (pd.Series, list)) and len(ret) < 20):
            return PreTradeCheck("cvar", True, 0.0, self.cvar_limit, "CVaR: no data (pass)")
        arr = ret.tail(252).dropna().values if isinstance(ret, pd.Series) else np.array(ret[-252:])
        var5 = float(np.percentile(arr, 5))
        tail = arr[arr <= var5]
        cvar = float(-tail.mean()) if len(tail) > 0 else 0.0
        projected = cvar * abs(size)
        ok = projected <= self.cvar_limit
        return PreTradeCheck(
            "cvar", ok, projected, self.cvar_limit,
            f"CVaR_contribution={projected:.5f} limit={self.cvar_limit:.4f}",
        )

    def _chk_atr_sizing(self, order: dict, original_size: float) -> tuple:
        atr   = float(order.get("atr", 0.0))
        price = float(order.get("current_price", 1.0))
        if atr <= 0 or price <= 0:
            return PreTradeCheck("atr_sizing", True, 0.0, 0.0, "ATR: no data (pass)"), original_size
        stop_pct = (atr * self.atr_multiplier) / price
        max_size = min(self.max_portfolio_risk / max(stop_pct, 0.001), self.max_position_size)
        abs_req  = abs(original_size)
        ok       = abs_req <= max_size
        adjusted = float(np.sign(original_size) * min(abs_req, max_size))
        return (
            PreTradeCheck(
                "atr_sizing", ok, abs_req, max_size,
                f"ATR-stop={stop_pct:.4f} max_size={max_size:.4f} requested={abs_req:.4f}",
            ),
            adjusted,
        )

    def _chk_cash_reserve(self, ps: dict, size: float) -> PreTradeCheck:
        cash     = float(ps.get("cash_pct", 1.0))
        projected = cash - abs(size)
        ok = projected >= self.min_cash_reserve
        return PreTradeCheck(
            "cash_reserve", ok, projected, self.min_cash_reserve,
            f"projected_cash={projected:.4f} min={self.min_cash_reserve:.4f}",
        )

    def _chk_stop_loss(self, order: dict) -> PreTradeCheck:
        stop = float(order.get("stop_loss_pct", self.min_stop_loss_pct))
        ok   = stop >= self.min_stop_loss_pct
        return PreTradeCheck(
            "stop_loss_required", ok, stop, self.min_stop_loss_pct,
            f"stop_loss={stop:.4f} min={self.min_stop_loss_pct:.4f}",
        )

    def _chk_track_exposure(self, order: dict, ps: dict) -> PreTradeCheck:
        track      = str(order.get("track", "default"))
        cap        = float(self.track_caps.get(track, self.max_position_size))
        positions  = ps.get("positions", {})
        total      = sum(abs(float(v)) for v in positions.values())
        projected  = total + abs(float(order.get("size", 0.0)))
        ok = projected <= cap
        return PreTradeCheck(
            "track_exposure", ok, projected, cap,
            f"track '{track}' total={projected:.4f} cap={cap:.4f}",
        )

    def _chk_vol_targeting(self, order: dict, ps: dict) -> PreTradeCheck:
        ret = ps.get("returns_series")
        if ret is None or (isinstance(ret, (pd.Series, list)) and len(ret) < 21):
            return PreTradeCheck("vol_targeting", True, 0.0, 0.0, "vol: no data (pass)")
        arr = ret.tail(21).values if isinstance(ret, pd.Series) else np.array(ret[-21:])
        vol = float(np.std(arr) * np.sqrt(252))
        conf = float(order.get("confidence", 1.0))
        size = abs(float(order.get("size", 0.0)))
        suggested_max = (0.15 / max(vol, 0.01)) * conf * 1.5  # 50% tolerance
        ok = size <= suggested_max
        return PreTradeCheck(
            "vol_targeting", ok, size, round(suggested_max, 6),
            f"vol={vol:.4f} suggested_max={suggested_max:.4f} size={size:.4f}",
        )

    def _chk_correlation(self, order: dict, ps: dict) -> PreTradeCheck:
        ticker   = str(order.get("ticker", ""))
        ext_rets = ps.get("existing_returns", {})
        new_rets = ext_rets.get(ticker) if ext_rets else None
        if new_rets is None or not ext_rets:
            return PreTradeCheck("correlation_gate", True, 0.0, self.correlation_gate, "no existing positions")
        max_corr = 0.0
        for t, er in ext_rets.items():
            if t == ticker:
                continue
            try:
                aligned = pd.concat([new_rets, er], axis=1).dropna()
                if len(aligned) >= 20:
                    c = float(aligned.corr().iloc[0, 1])
                    max_corr = max(max_corr, abs(c))
            except Exception:
                pass
        ok = max_corr <= self.correlation_gate
        return PreTradeCheck(
            "correlation_gate", ok, max_corr, self.correlation_gate,
            f"max_corr={max_corr:.4f} gate={self.correlation_gate:.4f}",
        )

    def _chk_leverage(self, ps: dict, size: float) -> PreTradeCheck:
        current   = float(ps.get("current_leverage", 0.0))
        projected = current + abs(size)
        ok = projected <= self.max_leverage
        return PreTradeCheck(
            "max_leverage", ok, projected, self.max_leverage,
            f"leverage={projected:.3f} {'ok' if ok else 'HARD BLOCK'} max={self.max_leverage:.3f}",
        )

    def _chk_daily_loss(self, ps: dict) -> PreTradeCheck:
        pnl = float(ps.get("daily_pnl_pct", 0.0))
        ok  = pnl > -self.max_daily_loss_pct
        return PreTradeCheck(
            "daily_loss_limit", ok, abs(min(pnl, 0.0)), self.max_daily_loss_pct,
            f"daily_pnl={pnl:.4f} {'ok' if ok else 'HARD BLOCK — limit breached'}",
        )

    def _chk_drawdown(self, ps: dict) -> PreTradeCheck:
        dd = float(ps.get("current_drawdown", 0.0))
        ok = dd <= self.max_drawdown_pct
        return PreTradeCheck(
            "drawdown_gate", ok, dd, self.max_drawdown_pct,
            f"drawdown={dd:.4f} {'ok' if ok else 'HARD BLOCK'}",
        )

    def _chk_vix(self, ps: dict) -> PreTradeCheck:
        vix = float(ps.get("vix", 0.0))
        if vix <= 0:
            return PreTradeCheck("vix_filter", True, 0.0, self.vix_halt_threshold, "VIX not provided (pass)")
        ok = vix <= self.vix_halt_threshold
        return PreTradeCheck(
            "vix_filter", ok, vix, self.vix_halt_threshold,
            f"VIX={vix:.1f} {'ok' if ok else 'HARD BLOCK — regime halt'}",
        )

    def _chk_risk_budget(self, order: dict, ps: dict, size: float) -> PreTradeCheck:
        ret = ps.get("returns_series")
        if ret is None or (isinstance(ret, (pd.Series, list)) and len(ret) < 21):
            return PreTradeCheck("portfolio_risk_budget", True, 0.0, self.max_portfolio_risk, "no data (pass)")
        arr = ret.tail(21).values if isinstance(ret, pd.Series) else np.array(ret[-21:])
        vol  = float(np.std(arr) * np.sqrt(252))
        contrib = abs(size) * vol
        ok = contrib <= self.max_portfolio_risk
        return PreTradeCheck(
            "portfolio_risk_budget", ok, contrib, self.max_portfolio_risk,
            f"risk_contribution={contrib:.5f} budget={self.max_portfolio_risk:.4f}",
        )
