"""
risk/dynamic_position_sizing.py
================================
Pluggable position-sizing engine for AlphaForge v2.0.

Wraps the existing ``PositionSizer`` from ``risk/position_sizing.py`` and adds
five extra methods with a unified ``SizingDecision`` output type.  The RL
policy can learn the sizing hyper-parameters via ``get_learnable_params()`` /
``set_learnable_params()``.

Methods
-------
  VOLATILITY_TARGET  — scale position so annualised volatility matches target
  FRACTIONAL_KELLY   — Kelly criterion with configurable fraction
  CONFIDENCE_EDGE    — linear scale from confidence and signal magnitude
  REGIME_AWARE       — pick method per market regime then apply multiplier
  DRAWDOWN_DERISKED  — multiplicative penalty when portfolio is in drawdown
  RISK_BUDGET        — cap size so marginal portfolio risk stays within budget

Drawdown de-risking tiers (applied after any method as a multiplier):
  drawdown < caution  (default 5%)  →  ×1.00
  drawdown < deriski  (default 10%) →  ×0.75
  drawdown < halt     (default 20%) →  ×0.50
  drawdown ≥ halt                   →  ×0.00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger
from risk.position_sizing import PositionSizer, RiskParams

logger = get_logger(__name__)

_SQRT252 = np.sqrt(252)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class SizingDecision:
    recommended_size:   float         # normalised exposure in [-1, 1]
    method_used:        str
    confidence:         float         # 0–1 edge estimate passed in by caller
    volatility_estimate: float        # annualised vol used in sizing
    risk_params_applied: dict = field(default_factory=dict)
    reasoning:          str   = ""    # one-line human-readable explanation


# ---------------------------------------------------------------------------
# Sizing method enum
# ---------------------------------------------------------------------------

class SizingMethod(str, Enum):
    VOLATILITY_TARGET  = "vol_target"
    FRACTIONAL_KELLY   = "kelly"
    CONFIDENCE_EDGE    = "confidence"
    REGIME_AWARE       = "regime"
    DRAWDOWN_DERISKED  = "drawdown"
    RISK_BUDGET        = "risk_budget"


# ---------------------------------------------------------------------------
# DynamicPositionSizer
# ---------------------------------------------------------------------------

class DynamicPositionSizer:
    """
    Unified, pluggable position sizer for AlphaForge v2.0.

    Usage
    -----
        dps = DynamicPositionSizer()
        decision = dps.size(
            signal=0.7, confidence=0.8,
            returns_series=ret_series,
            current_drawdown=0.06,
            regime="bull",
        )
        size = decision.recommended_size        # [-1, 1]
        params = dps.get_learnable_params()     # expose to RL policy
    """

    def __init__(
        self,
        method:             str   = SizingMethod.REGIME_AWARE,
        target_vol:         float = 0.15,
        kelly_fraction:     float = 0.25,
        max_position:       float = 0.25,
        max_portfolio_risk: float = 0.02,
        drawdown_derisking: bool  = True,
        # Drawdown tier thresholds
        dd_caution:         float = 0.05,
        dd_derisked:        float = 0.10,
        dd_halt:            float = 0.20,
        # Regime multipliers
        regime_multipliers: Optional[dict] = None,
        # Confidence gate
        min_confidence:     float = 0.55,
        config:             Optional[dict] = None,
    ) -> None:
        self.method             = SizingMethod(method) if isinstance(method, str) else method
        self.target_vol         = target_vol
        self.kelly_fraction     = kelly_fraction
        self.max_position       = max_position
        self.max_portfolio_risk = max_portfolio_risk
        self.drawdown_derisking = drawdown_derisking
        self.dd_caution         = dd_caution
        self.dd_derisked        = dd_derisked
        self.dd_halt            = dd_halt
        self.min_confidence     = min_confidence
        self.regime_multipliers = regime_multipliers or {
            "bull": 1.0,
            "bear": 0.5,
            "sideways": 0.75,
            "high_vol": 0.60,
            "unknown": 0.80,
        }
        self._sizer = PositionSizer(config=config)

    # ── Main API ──────────────────────────────────────────────────────────────

    def size(
        self,
        signal:           float,
        confidence:       float,
        returns_series:   Optional[pd.Series],
        current_drawdown: float = 0.0,
        regime:           str   = "unknown",
        portfolio_vol:    float = 0.15,
    ) -> SizingDecision:
        """
        Compute optimal position size.

        Parameters
        ----------
        signal : float
            Raw directional signal (positive = long, negative = short).
        confidence : float
            Model confidence in [0, 1].
        returns_series : pd.Series or None
            Recent daily returns used for vol estimation and Kelly.
        current_drawdown : float
            Current portfolio drawdown fraction (0–1); used for de-risking.
        regime : str
            Market regime string e.g. 'bull', 'bear', 'sideways'.
        portfolio_vol : float
            Current portfolio-level volatility (fallback if returns_series unavailable).
        """
        if confidence < self.min_confidence:
            return SizingDecision(
                recommended_size=0.0,
                method_used=self.method.value,
                confidence=confidence,
                volatility_estimate=0.0,
                reasoning=f"confidence {confidence:.2f} < threshold {self.min_confidence:.2f}",
            )

        vol = self._estimate_vol(returns_series, portfolio_vol)
        sign = float(np.sign(signal)) if signal != 0 else 1.0

        if self.method == SizingMethod.VOLATILITY_TARGET:
            base = self._vol_target(signal, vol)
            reasoning = f"vol_target: target={self.target_vol:.2f} vol={vol:.3f}"
        elif self.method == SizingMethod.FRACTIONAL_KELLY:
            base = self._kelly(signal, confidence, returns_series, vol)
            reasoning = f"kelly: frac={self.kelly_fraction:.2f} conf={confidence:.2f}"
        elif self.method == SizingMethod.CONFIDENCE_EDGE:
            base = self._confidence_edge(signal, confidence)
            reasoning = f"confidence_edge: signal={signal:.2f} conf={confidence:.2f}"
        elif self.method == SizingMethod.RISK_BUDGET:
            base = self._risk_budget(signal, vol, self.max_portfolio_risk)
            reasoning = f"risk_budget: max_risk={self.max_portfolio_risk:.3f} vol={vol:.3f}"
        else:  # REGIME_AWARE or DRAWDOWN_DERISKED
            base = self._regime_aware(signal, confidence, returns_series, regime, vol)
            reasoning = f"regime_aware: regime={regime} multiplier={self.regime_multipliers.get(regime.lower(), 0.8):.2f}"

        # Apply drawdown de-risking multiplier
        if self.drawdown_derisking:
            dd_mult = self._drawdown_multiplier(current_drawdown)
            if dd_mult < 1.0:
                reasoning += f" | dd_derisked(dd={current_drawdown:.2f}→×{dd_mult:.2f})"
            base *= dd_mult

        # Cap at max_position; preserve sign
        final = float(np.clip(base, -self.max_position, self.max_position))

        decision = SizingDecision(
            recommended_size=final,
            method_used=self.method.value,
            confidence=confidence,
            volatility_estimate=vol,
            risk_params_applied={
                "target_vol":         self.target_vol,
                "kelly_fraction":     self.kelly_fraction,
                "max_position":       self.max_position,
                "max_portfolio_risk": self.max_portfolio_risk,
                "dd_caution":         self.dd_caution,
                "dd_derisked":        self.dd_derisked,
                "dd_halt":            self.dd_halt,
            },
            reasoning=reasoning,
        )
        logger.debug("SizingDecision: %s size=%.4f", decision.reasoning, final)
        return decision

    # ── Sizing method implementations ─────────────────────────────────────────

    def _vol_target(self, signal: float, vol: float) -> float:
        if vol <= 0:
            return 0.0
        raw = (self.target_vol / (vol + 1e-9)) * np.sign(signal)
        return float(np.clip(raw, -self.max_position, self.max_position))

    def _kelly(
        self,
        signal: float,
        confidence: float,
        returns_series: Optional[pd.Series],
        vol: float,
    ) -> float:
        variance = vol ** 2 if vol > 0 else 0.04
        edge = abs(signal) * confidence
        full_kelly = edge / (variance + 1e-12)
        raw = self.kelly_fraction * full_kelly * np.sign(signal)
        return float(np.clip(raw, -self.max_position, self.max_position))

    def _confidence_edge(self, signal: float, confidence: float) -> float:
        edge_scale = max(confidence - self.min_confidence, 0.0) / max(1.0 - self.min_confidence, 1e-9)
        raw = signal * edge_scale
        return float(np.clip(raw, -self.max_position, self.max_position))

    def _regime_aware(
        self,
        signal: float,
        confidence: float,
        returns_series: Optional[pd.Series],
        regime: str,
        vol: float,
    ) -> float:
        reg = regime.lower()
        if reg in ("bull",):
            base = self._vol_target(signal, vol)
            # vol_target ignores confidence — apply confidence scaling so that
            # marginal signals (proba barely above threshold) don't get full 30x.
            # Maps [min_confidence, 1.0] → [0.50, 1.0]; capped to avoid exceeding max.
            _cf = max(0.0, confidence - self.min_confidence) / max(1.0 - self.min_confidence, 1e-8)
            base *= max(0.50, min(1.0, 0.50 + 0.50 * _cf))
        elif reg in ("bear", "high_vol"):
            base = self._kelly(signal, confidence, returns_series, vol)
        else:
            base = self._confidence_edge(signal, confidence)
        mult = self.regime_multipliers.get(reg, 0.80)
        return float(np.clip(base * mult, -self.max_position, self.max_position))

    def _risk_budget(self, signal: float, portfolio_vol: float, budget_pct: float) -> float:
        if portfolio_vol <= 0:
            return 0.0
        # marginal contribution: budget_pct = |size| * asset_vol
        size_mag = budget_pct / (portfolio_vol + 1e-9)
        return float(np.clip(np.sign(signal) * size_mag, -self.max_position, self.max_position))

    # ── Drawdown de-risking ───────────────────────────────────────────────────

    def _drawdown_multiplier(self, current_drawdown: float) -> float:
        dd = abs(current_drawdown)
        if dd >= self.dd_halt:
            return 0.0
        if dd >= self.dd_derisked:
            return 0.50
        if dd >= self.dd_caution:
            return 0.75
        return 1.0

    # ── Vol estimation helper ─────────────────────────────────────────────────

    def _estimate_vol(
        self,
        returns_series: Optional[pd.Series],
        fallback: float = 0.20,
    ) -> float:
        if returns_series is not None and len(returns_series) >= 5:
            ann_vol = float(returns_series.dropna().tail(21).std() * _SQRT252)
            return max(ann_vol, 1e-4)
        return max(fallback, 1e-4)

    # ── RL-learnable parameter interface ─────────────────────────────────────

    def get_learnable_params(self) -> dict:
        """Return sizing hyper-parameters that the RL policy may optimise."""
        return {
            "kelly_fraction":      self.kelly_fraction,
            "target_vol":          self.target_vol,
            "max_position":        self.max_position,
            "dd_caution":          self.dd_caution,
            "dd_derisked":         self.dd_derisked,
            "dd_halt":             self.dd_halt,
            "regime_bull_mult":    self.regime_multipliers.get("bull", 1.0),
            "regime_bear_mult":    self.regime_multipliers.get("bear", 0.5),
            "regime_sideways_mult":self.regime_multipliers.get("sideways", 0.75),
        }

    def set_learnable_params(self, params: dict) -> None:
        """Apply RL-learned parameter values with safety clipping."""
        if "kelly_fraction" in params:
            self.kelly_fraction = float(np.clip(params["kelly_fraction"], 0.05, 1.0))
        if "target_vol" in params:
            self.target_vol = float(np.clip(params["target_vol"], 0.05, 0.50))
        if "max_position" in params:
            self.max_position = float(np.clip(params["max_position"], 0.05, 1.0))
        if "dd_caution" in params:
            self.dd_caution = float(np.clip(params["dd_caution"], 0.01, 0.15))
        if "dd_derisked" in params:
            self.dd_derisked = float(np.clip(params["dd_derisked"], 0.05, 0.30))
        if "dd_halt" in params:
            self.dd_halt = float(np.clip(params["dd_halt"], 0.10, 0.50))
        for key in ("regime_bull_mult", "regime_bear_mult", "regime_sideways_mult"):
            if key in params:
                regime = key.split("_")[1]
                self.regime_multipliers[regime] = float(np.clip(params[key], 0.0, 2.0))
        logger.debug("DynamicPositionSizer: params updated via RL")
