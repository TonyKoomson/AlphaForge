"""
execution/almgren_chriss.py
============================
Almgren-Chriss Optimal Execution Model (Almgren & Chriss, 2001).

Problem
-------
When liquidating a large position, a trader faces a trade-off between:
  - Market impact (linear + nonlinear): trading fast causes price impact.
  - Timing risk (volatility): trading slowly exposes to adverse price moves.

The A-C model provides the analytically optimal execution trajectory that
minimises expected cost + λ × variance of total execution cost.

Key equations
-------------
Trajectory:   x_j = X · sinh(κ(T−t_j)) / sinh(κT)
where κ = √(λσ²/η), X = initial shares, T = horizon, τ = interval length.

Parameters
----------
  X       : Total shares to execute (positive = sell, negative = buy).
  T       : Total execution horizon in periods (e.g., 10 trading bars).
  N       : Number of execution intervals (sub-slices).
  σ       : Per-period price volatility (e.g., daily std of returns × price).
  η       : Temporary market impact parameter (cost per unit trading rate).
  γ       : Permanent market impact parameter (permanent price shift per share).
  λ       : Risk aversion parameter (higher = trade faster, less timing risk).

Output
------
  AlmgrenChrissResult with:
    trajectory    : np.ndarray (N+1,) — shares remaining at each period
    trade_list    : np.ndarray (N,)   — shares traded each period
    expected_cost : float             — expected total execution cost ($)
    variance_cost : float             — variance of execution cost ($²)
    efficient_frontier: list of (λ, expected_cost, variance_cost)

Usage
-----
    from execution.almgren_chriss import almgren_chriss_trajectory, AlmgrenChrissResult

    result = almgren_chriss_trajectory(
        X=10_000,    # sell 10,000 shares
        T=10,        # over 10 periods
        N=10,
        sigma=2.50,  # $2.50 per-period price vol
        eta=0.01,    # temporary impact
        gamma=0.001, # permanent impact
        lam=1e-6,    # risk aversion
    )
    print(result.trade_list)
    print(f"Expected cost: ${result.expected_cost:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class AlmgrenChrissResult:
    """Output of the Almgren-Chriss optimal execution solver."""
    X:              float               # Total shares to execute
    T:              int                 # Total periods
    N:              int                 # Number of intervals
    trajectory:     np.ndarray          # Shares remaining: shape (N+1,)
    trade_list:     np.ndarray          # Shares traded per interval: shape (N,)
    expected_cost:  float               # E[total cost] in $
    variance_cost:  float               # Var[total cost] in $²
    utility:        float               # E[cost] + λ·Var[cost]
    lam:            float               # Risk aversion used
    sigma:          float
    eta:            float
    gamma:          float
    efficient_frontier: list[dict] = field(default_factory=list)

    def vwap_deviation(self, prices: np.ndarray) -> float:
        """
        Compute the VWAP deviation of the execution trajectory vs.
        a flat (VWAP) execution, given a price path of length N.
        """
        if len(prices) < self.N:
            return 0.0
        flat_qty  = self.X / self.N
        flat_cost = float(np.dot(np.ones(self.N) * flat_qty, prices[: self.N]))
        ac_cost   = float(np.dot(self.trade_list, prices[: self.N]))
        return float(ac_cost - flat_cost) if flat_cost != 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "X": self.X,
            "T": self.T,
            "N": self.N,
            "expected_cost":  round(self.expected_cost, 4),
            "variance_cost":  round(self.variance_cost, 4),
            "utility":        round(self.utility, 4),
            "lam":            self.lam,
            "trade_list":     self.trade_list.tolist(),
        }


def almgren_chriss_trajectory(
    X:     float,
    T:     int   = 10,
    N:     int   = 10,
    sigma: float = 1.0,
    eta:   float = 0.01,
    gamma: float = 0.001,
    lam:   float = 1e-6,
) -> AlmgrenChrissResult:
    """
    Compute the optimal Almgren-Chriss execution trajectory.

    Parameters
    ----------
    X     : Total shares to trade (>0 sell, <0 buy).
    T     : Execution horizon (number of periods, same unit as sigma).
    N     : Number of sub-intervals (must equal T for simplicity).
    sigma : Per-period price volatility in price units (σ · S).
    eta   : Temporary market impact coefficient ($/share per share/period).
    gamma : Permanent market impact coefficient ($/share).
    lam   : Risk aversion (λ ≥ 0; higher → faster execution).

    Returns
    -------
    AlmgrenChrissResult
    """
    tau = T / N     # interval length

    # κ = √(λσ²/η)  — determines how "urgently" to trade
    kappa_sq = lam * sigma ** 2 / (eta + 1e-12)
    kappa    = float(np.sqrt(max(kappa_sq, 0.0)))

    # Optimal trajectory: x_j = X · sinh(κ(T − t_j)) / sinh(κT)
    times = np.linspace(0, T, N + 1)   # t_0 … t_N

    if kappa < 1e-10 or abs(np.sinh(kappa * T)) < 1e-12:
        # κ → 0: uniform liquidation (VWAP)
        trajectory = X * (1.0 - times / T)
    else:
        trajectory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

    trade_list = -np.diff(trajectory)   # positive = shares sold each period

    # Expected cost (Almgren-Chriss eq. 14)
    sum_traj_sq = float(np.sum(trajectory[:-1] ** 2))
    expected_cost = (
        0.5 * gamma * X ** 2
        + eta / tau * np.sum(trade_list ** 2)
        + 0.5 * gamma * np.sum(trade_list)   # permanent impact adjustment
    )

    # Variance of execution cost (eq. 17)
    variance_cost = sigma ** 2 * tau * float(np.sum(trajectory[:-1] ** 2))

    utility = expected_cost + lam * variance_cost

    logger.debug(
        "AlmgrenChriss: X=%.0f T=%d kappa=%.4f E[cost]=%.2f Var[cost]=%.2f",
        X, T, kappa, expected_cost, variance_cost,
    )
    return AlmgrenChrissResult(
        X=X, T=T, N=N,
        trajectory=trajectory,
        trade_list=trade_list,
        expected_cost=float(expected_cost),
        variance_cost=float(variance_cost),
        utility=float(utility),
        lam=lam, sigma=sigma, eta=eta, gamma=gamma,
    )


def efficient_frontier(
    X:     float,
    T:     int   = 10,
    N:     int   = 10,
    sigma: float = 1.0,
    eta:   float = 0.01,
    gamma: float = 0.001,
    n_points: int = 20,
) -> list[dict]:
    """
    Compute the Almgren-Chriss efficient frontier (expected cost vs. variance)
    over a range of risk-aversion parameters.

    Returns list of dicts with keys: lam, expected_cost, variance_cost, utility.
    """
    lambdas = np.logspace(-8, -2, n_points)
    frontier = []
    for lam in lambdas:
        r = almgren_chriss_trajectory(X, T, N, sigma, eta, gamma, float(lam))
        frontier.append({
            "lam":           float(lam),
            "expected_cost": round(r.expected_cost, 4),
            "variance_cost": round(r.variance_cost, 4),
            "utility":       round(r.utility, 4),
        })
    return frontier


def adaptive_ac_execution(
    X:             float,
    T:             int,
    N:             int,
    price_series:  np.ndarray,
    eta:           float = 0.01,
    gamma:         float = 0.001,
    lam:           float = 1e-6,
    vol_window:    int   = 5,
) -> AlmgrenChrissResult:
    """
    Adaptive variant: re-estimate sigma from realised volatility each period
    and recompute the remaining trajectory dynamically.

    This is a simplified receding-horizon implementation — each period, the
    remaining trajectory is re-optimised with updated volatility.

    price_series : observed prices of length ≥ N.
    """
    if len(price_series) < N:
        sigma_init = float(np.std(np.diff(price_series) / price_series[:-1]) * price_series[0])
        return almgren_chriss_trajectory(X, T, N, sigma_init, eta, gamma, lam)

    trade_schedule = []
    remaining      = float(X)
    rets_buffer: list[float] = []

    for i in range(N):
        px   = float(price_series[i])
        ret  = float(price_series[i] / price_series[max(i-1, 0)] - 1.0)
        rets_buffer.append(ret)
        tail = rets_buffer[-vol_window:]
        sigma_est = float(np.std(tail) * px) if len(tail) >= 2 else float(np.std(np.diff(price_series)) / px * px)

        periods_left = N - i
        if periods_left <= 0:
            trade_schedule.append(remaining)
            remaining = 0.0
            break

        sub = almgren_chriss_trajectory(remaining, periods_left, periods_left, sigma_est, eta, gamma, lam)
        trade_qty = float(sub.trade_list[0])
        trade_schedule.append(trade_qty)
        remaining -= trade_qty

    if remaining != 0.0:
        trade_schedule.append(remaining)

    n_actual    = len(trade_schedule)
    trajectory  = np.concatenate([[X], X - np.cumsum(trade_schedule)])
    trade_arr   = np.array(trade_schedule, dtype=float)
    sigma_final = float(np.std(np.diff(price_series) / price_series[:-1]) * price_series[0])

    return AlmgrenChrissResult(
        X=X, T=T, N=n_actual,
        trajectory=trajectory,
        trade_list=trade_arr,
        expected_cost=float(np.sum(trade_arr ** 2) * eta),
        variance_cost=float(sigma_final ** 2 * np.sum(trajectory[:-1] ** 2)),
        utility=0.0, lam=lam, sigma=sigma_final, eta=eta, gamma=gamma,
    )
