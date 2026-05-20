"""
risk/drawdown_control.py
=========================
Real-time drawdown monitor with tiered de-risking for AlphaForge v2.0.

Tier transitions use hysteresis: a tier only *downgrades* (less restrictive)
after ``recovery_bars`` consecutive bars where the nav is above the tier's
entry threshold.  This prevents thrashing on choppy recoveries.

Tiers
-----
  normal    — drawdown < caution_threshold    → scale_factor = 1.00
  caution   — drawdown ≥ caution_threshold    → scale_factor = 0.75
  derisked  — drawdown ≥ derisked_threshold   → scale_factor = 0.50
  halted    — drawdown ≥ halt_threshold       → scale_factor = 0.00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.helpers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class DrawdownState:
    peak_nav:               float
    current_nav:            float
    current_drawdown_pct:   float           # 0.0 – 1.0
    tier:                   str             # "normal" | "caution" | "derisked" | "halted"
    scale_factor:           float           # 1.0 | 0.75 | 0.50 | 0.00
    recovery_bars_count:    int = 0         # consecutive bars above entry threshold
    events:                 list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "peak_nav":             self.peak_nav,
            "current_nav":          self.current_nav,
            "current_drawdown_pct": round(self.current_drawdown_pct, 5),
            "tier":                 self.tier,
            "scale_factor":         self.scale_factor,
            "recovery_bars_count":  self.recovery_bars_count,
            "n_events":             len(self.events),
        }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

_TIER_ORDER = ["normal", "caution", "derisked", "halted"]

_TIER_SCALE = {
    "normal":   1.00,
    "caution":  0.75,
    "derisked": 0.50,
    "halted":   0.00,
}


class DrawdownController:
    """
    Track live portfolio drawdown and return a position scale factor.

    Usage
    -----
        dc = DrawdownController()
        state = dc.update(current_nav=98_000.0, timestamp="2024-06-01")
        multiplier = dc.get_scale_factor()   # multiply every position size by this
    """

    def __init__(
        self,
        caution_threshold:   float = 0.05,
        derisked_threshold:  float = 0.10,
        halt_threshold:      float = 0.20,
        recovery_bars:       int   = 3,
    ) -> None:
        self.caution_threshold  = caution_threshold
        self.derisked_threshold = derisked_threshold
        self.halt_threshold     = halt_threshold
        self.recovery_bars      = recovery_bars

        self._peak_nav: Optional[float] = None
        self._state: DrawdownState = DrawdownState(
            peak_nav=0.0,
            current_nav=0.0,
            current_drawdown_pct=0.0,
            tier="normal",
            scale_factor=1.0,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, current_nav: float, timestamp: str = "") -> DrawdownState:
        """
        Record the latest NAV and return the updated DrawdownState.

        Parameters
        ----------
        current_nav : float
            Latest portfolio value.
        timestamp : str
            Optional ISO timestamp for event logging.
        """
        if self._peak_nav is None or current_nav > self._peak_nav:
            self._peak_nav = float(current_nav)

        dd = (self._peak_nav - current_nav) / max(self._peak_nav, 1e-9)
        new_tier = self._classify_tier(dd)
        old_tier = self._state.tier

        # Hysteresis: only downgrade (loosen) after recovery_bars consecutive bars
        if _TIER_ORDER.index(new_tier) < _TIER_ORDER.index(old_tier):
            self._state.recovery_bars_count += 1
            if self._state.recovery_bars_count >= self.recovery_bars:
                self._apply_tier_change(old_tier, new_tier, current_nav, dd, timestamp)
        elif new_tier != old_tier:
            # Upgrade (tighten) immediately
            self._state.recovery_bars_count = 0
            self._apply_tier_change(old_tier, new_tier, current_nav, dd, timestamp)
        else:
            self._state.recovery_bars_count = 0

        self._state.peak_nav = self._peak_nav
        self._state.current_nav = current_nav
        self._state.current_drawdown_pct = dd
        return self._state

    def get_scale_factor(self) -> float:
        """Return the current position scale factor (0.0 – 1.0)."""
        return self._state.scale_factor

    def is_halted(self) -> bool:
        return self._state.tier == "halted"

    def reset_peak(self) -> None:
        """Manually reset the peak NAV (use after confirmed full recovery)."""
        self._peak_nav = self._state.current_nav
        logger.info("DrawdownController: peak reset to %.2f", self._peak_nav)

    def recovery_required_pct(self) -> float:
        """Percentage gain needed for nav to recover above current peak."""
        if self._peak_nav is None or self._state.current_nav <= 0:
            return 0.0
        return max(0.0, self._peak_nav / max(self._state.current_nav, 1e-9) - 1.0)

    def summary(self) -> dict:
        return {
            **self._state.to_dict(),
            "recovery_required_pct": round(self.recovery_required_pct(), 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _classify_tier(self, drawdown_pct: float) -> str:
        if drawdown_pct >= self.halt_threshold:
            return "halted"
        if drawdown_pct >= self.derisked_threshold:
            return "derisked"
        if drawdown_pct >= self.caution_threshold:
            return "caution"
        return "normal"

    def _apply_tier_change(
        self,
        old_tier:    str,
        new_tier:    str,
        current_nav: float,
        dd:          float,
        timestamp:   str,
    ) -> None:
        self._state.tier = new_tier
        self._state.scale_factor = _TIER_SCALE[new_tier]
        self._state.recovery_bars_count = 0

        # When fully recovering to normal after a drawdown episode, reset the
        # peak baseline to current NAV.  Without this, one historical all-time-high
        # keeps the controller in caution indefinitely even after the strategy
        # has meaningfully recovered.
        if new_tier == "normal" and old_tier not in ("normal",):
            self._peak_nav = current_nav
            logger.info(
                "DrawdownController: recovered to normal — peak baseline reset to %.2f",
                current_nav,
            )

        event = {
            "timestamp":   timestamp,
            "from_tier":   old_tier,
            "to_tier":     new_tier,
            "drawdown":    round(dd, 5),
            "nav":         current_nav,
            "scale_factor":_TIER_SCALE[new_tier],
        }
        self._state.events.append(event)
        level = "warning" if new_tier in ("halted", "derisked") else "info"
        getattr(logger, level)(
            "DrawdownController: %s → %s | drawdown=%.2f%% scale=%.2f",
            old_tier, new_tier, dd * 100, _TIER_SCALE[new_tier],
        )
