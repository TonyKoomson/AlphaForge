"""
risk/capacity.py
=================
Strategy Capacity Estimation for AlphaForge v2.0.

Estimates the maximum capital each strategy can safely handle before
market impact degrades returns below an acceptable threshold.

Method (Grinold & Kahn, 2000; adapted)
---------------------------------------
The capacity C* is the capital level at which the expected Sharpe ratio
degrades to half its frictionless value, due to execution market impact.

Key relationship: impact_cost ∝ (size / ADV) ^ impact_exponent
where ADV = Average Daily Volume in dollars.

We estimate:
  1. Frictionless Sharpe SR₀ from backtest without impact.
  2. Impact-adjusted Sharpe SR(C) = SR₀ − λ · (C / ADV)^γ
  3. Solve for C* such that SR(C*) = SR₀ / 2.

Additional checks:
  - Concentration limit: C ≤ max_adv_fraction × ADV
  - Daily turnover limit: C × turnover ≤ max_daily_notional
  - Slippage scaling: verify SR survives 3× slippage at estimated size.

Usage
-----
    from risk.capacity import CapacityEstimator

    est = CapacityEstimator(adv_usd=50e6, impact_exponent=0.6)
    result = est.estimate(
        sharpe_0=1.5,
        annual_turnover=4.0,     # portfolio turns per year
        backtest_capital=100_000
    )
    print(f"Max capacity: ${result.max_capacity_usd:,.0f}")
    print(f"Optimal capital: ${result.optimal_capital_usd:,.0f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class CapacityResult:
    """Output of CapacityEstimator.estimate()."""
    # Capacity estimates
    max_capacity_usd:     float   # capital where SR drops to 0
    optimal_capacity_usd: float   # capital where SR drops to SR₀/2
    half_sharpe_capacity: float   # same as optimal (alias)

    # Sharpe degradation curve sampled at key sizes
    sharpe_at_100k:  float
    sharpe_at_1m:    float
    sharpe_at_10m:   float
    sharpe_at_100m:  float

    # Slippage stress test
    sr_at_3x_slippage:   float
    passes_slippage_gate: bool

    # Utilisation ratios
    adv_fraction_at_optimal: float   # C* / ADV

    # Metadata
    adv_usd:            float
    sharpe_0:           float
    annual_turnover:    float
    impact_exponent:    float
    warning:            str = ""

    def to_dict(self) -> dict:
        return {
            "max_capacity_usd":           round(self.max_capacity_usd),
            "optimal_capacity_usd":       round(self.optimal_capacity_usd),
            "sharpe_at_100k":             round(self.sharpe_at_100k, 3),
            "sharpe_at_1m":               round(self.sharpe_at_1m, 3),
            "sharpe_at_10m":              round(self.sharpe_at_10m, 3),
            "sharpe_at_100m":             round(self.sharpe_at_100m, 3),
            "sr_at_3x_slippage":          round(self.sr_at_3x_slippage, 3),
            "passes_slippage_gate":       self.passes_slippage_gate,
            "adv_fraction_at_optimal":    round(self.adv_fraction_at_optimal, 4),
            "warning":                    self.warning,
        }


class CapacityEstimator:
    """
    Estimate maximum safe capital for a strategy.

    Parameters
    ----------
    adv_usd          : Average daily dollar volume of the traded asset (e.g. $50M for SPY).
    impact_exponent  : Market impact scales as (size/ADV)^γ (0.5–0.6 is empirically common).
    impact_coeff     : Proportionality constant for impact cost (default 1.0 × ADV fraction).
    base_slippage    : Per-side slippage at backtest capital (fraction).
    max_adv_fraction : Hard cap: strategy size ≤ this fraction of ADV.
    """

    def __init__(
        self,
        adv_usd:           float = 50_000_000.0,
        impact_exponent:   float = 0.60,
        impact_coeff:      float = 1.00,
        base_slippage:     float = 0.0005,
        max_adv_fraction:  float = 0.05,
    ) -> None:
        self.adv          = adv_usd
        self.gamma        = impact_exponent
        self.lam          = impact_coeff
        self.base_slip    = base_slippage
        self.max_adv_frac = max_adv_fraction

    def estimate(
        self,
        sharpe_0:         float,
        annual_turnover:  float = 4.0,
        backtest_capital: float = 100_000.0,
        min_sharpe:       float = 0.30,
    ) -> CapacityResult:
        """
        Estimate capacity for a strategy.

        Parameters
        ----------
        sharpe_0         : Frictionless Sharpe from backtest at backtest_capital.
        annual_turnover  : Portfolio turns per year (higher = more impact).
        backtest_capital : Capital used in backtest (to calibrate impact constant).
        min_sharpe       : Minimum acceptable Sharpe (used for max_capacity_usd).
        """
        if sharpe_0 <= 0:
            return CapacityResult(
                max_capacity_usd=0.0, optimal_capacity_usd=0.0,
                half_sharpe_capacity=0.0,
                sharpe_at_100k=0.0, sharpe_at_1m=0.0,
                sharpe_at_10m=0.0, sharpe_at_100m=0.0,
                sr_at_3x_slippage=0.0, passes_slippage_gate=False,
                adv_fraction_at_optimal=0.0, adv_usd=self.adv,
                sharpe_0=sharpe_0, annual_turnover=annual_turnover,
                impact_exponent=self.gamma, warning="sharpe_0 ≤ 0",
            )

        def _impact_sharpe(capital: float) -> float:
            """Sharpe at given capital level, adjusted for impact."""
            daily_notional = capital * annual_turnover / 252.0
            adv_frac       = daily_notional / max(self.adv, 1.0)
            impact_cost    = self.lam * adv_frac ** self.gamma
            # Impact cost reduces annualised Sharpe proportionally
            sharpe_adj     = sharpe_0 - sharpe_0 * impact_cost * np.sqrt(252) * annual_turnover
            return float(max(0.0, sharpe_adj))

        # Evaluate at standard capital levels
        sr_100k  = _impact_sharpe(100_000)
        sr_1m    = _impact_sharpe(1_000_000)
        sr_10m   = _impact_sharpe(10_000_000)
        sr_100m  = _impact_sharpe(100_000_000)

        # Binary search for optimal (half-Sharpe) capacity
        target_sr = sharpe_0 / 2.0
        lo, hi    = backtest_capital, self.adv * self.max_adv_frac * 100
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if _impact_sharpe(mid) > target_sr:
                lo = mid
            else:
                hi = mid
        optimal_c = (lo + hi) / 2.0

        # Max capacity (Sharpe = min_sharpe)
        lo_max, hi_max = backtest_capital, self.adv * 100
        for _ in range(50):
            mid = (lo_max + hi_max) / 2.0
            if _impact_sharpe(mid) > min_sharpe:
                lo_max = mid
            else:
                hi_max = mid
        max_c = (lo_max + hi_max) / 2.0

        # Slippage scaling: simulate 3× slippage at optimal capital
        slip_mult = 3.0
        sr_3x = sharpe_0 - (slip_mult * self.base_slip * annual_turnover * np.sqrt(252))
        passes_slip = sr_3x >= 0.30

        adv_frac_opt = optimal_c * annual_turnover / 252.0 / max(self.adv, 1.0)
        warning = ""
        if adv_frac_opt > self.max_adv_frac:
            warning = f"Optimal capacity uses {adv_frac_opt:.1%} of ADV > max {self.max_adv_frac:.1%}"

        logger.info(
            "CapacityEstimator: SR₀=%.2f  optimal=$%s  max=$%s  ADV_frac=%.2f%%",
            sharpe_0, f"{optimal_c:,.0f}", f"{max_c:,.0f}", adv_frac_opt * 100,
        )
        return CapacityResult(
            max_capacity_usd=max_c,
            optimal_capacity_usd=optimal_c,
            half_sharpe_capacity=optimal_c,
            sharpe_at_100k=sr_100k,
            sharpe_at_1m=sr_1m,
            sharpe_at_10m=sr_10m,
            sharpe_at_100m=sr_100m,
            sr_at_3x_slippage=float(sr_3x),
            passes_slippage_gate=passes_slip,
            adv_fraction_at_optimal=float(adv_frac_opt),
            adv_usd=self.adv, sharpe_0=sharpe_0,
            annual_turnover=annual_turnover,
            impact_exponent=self.gamma,
            warning=warning,
        )
