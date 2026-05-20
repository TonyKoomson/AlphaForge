"""
validation/robustness.py
========================
Robustness gates (anti-overfitting) for AlphaForge v2.0.

Implements five gates from Bailey & López de Prado (2014) + standard
quant finance practice:

1. DSR  — Deflated Sharpe Ratio  (accounts for multiple-testing bias)
2. CPCV — Combinatorially Purged Cross-Validation Sharpe
3. Param Stability — coefficient-of-variation of Sharpe across windows
4. Slippage Scaling — Sharpe survives 1× / 2× / 3× slippage multipliers
5. t-stat — raw significance test on OOS Sharpe
6. MinTRL — Minimum Track Record Length in years

All pass thresholds are configurable via config["robustness_gates"].

Usage
-----
    gates = RobustnessGates(**cfg["robustness_gates"])
    report = gates.evaluate(oos_returns, is_sharpe=1.8, oos_sharpe_list=[...])
    if report.overall_pass:
        promote_strategy()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger, sharpe_ratio

logger = get_logger(__name__)

_SQRT252 = math.sqrt(252)
_LOG2PI  = math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# Normal CDF / inv-CDF (pure stdlib — no scipy required)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Φ(x) — standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Φ⁻¹(p) — inverse normal CDF via rational approximation (Abramowitz & Stegun)."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RobustnessReport:
    # 1. DSR
    dsr: float = 0.0
    dsr_pass: bool = False

    # 2. CPCV
    cpcv_sharpe: float = 0.0
    cpcv_pass: bool = False

    # 3. Parameter stability
    param_stability_cv: float = 0.0          # CV of Sharpe across param windows
    param_stability_pass: bool = False

    # 4. Slippage scaling
    slippage_sharpes: dict = field(default_factory=dict)   # {1: sr, 2: sr, 3: sr}
    slippage_pass: bool = False

    # 5. t-stat
    t_stat: float = 0.0
    t_stat_pass: bool = False

    # 6. MinTRL
    min_trl_years: float = 0.0
    min_trl_pass: bool = False

    # 7. Equity curve R²
    equity_r2: float = 0.0
    equity_r2_pass: bool = False

    # 8. Positive folds fraction
    positive_folds_frac: float = 0.0
    positive_folds_pass: bool = False

    # Summary
    n_passed: int = 0
    overall_pass: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dsr": round(self.dsr, 4),
            "dsr_pass": self.dsr_pass,
            "cpcv_sharpe": round(self.cpcv_sharpe, 4),
            "cpcv_pass": self.cpcv_pass,
            "param_stability_cv": round(self.param_stability_cv, 4),
            "param_stability_pass": self.param_stability_pass,
            "slippage_sharpes": {str(k): round(v, 4) for k, v in self.slippage_sharpes.items()},
            "slippage_pass": self.slippage_pass,
            "t_stat": round(self.t_stat, 4),
            "t_stat_pass": self.t_stat_pass,
            "min_trl_years": round(self.min_trl_years, 2),
            "min_trl_pass": self.min_trl_pass,
            "n_passed": self.n_passed,
            "overall_pass": self.overall_pass,
        }


# ---------------------------------------------------------------------------
# RobustnessGates
# ---------------------------------------------------------------------------

class RobustnessGates:
    """
    Five robustness gates applied to a candidate strategy before promotion.

    Parameters (config["robustness_gates"])
    ----------------------------------------
    min_dsr                 float = 0.95
    min_t_stat              float = 3.0
    max_param_cv            float = 0.50   CV threshold for param stability
    min_sharpe_3x_slippage  float = 0.30   Sharpe after 3× slippage multiplier
    min_trl_confidence      float = 0.95   Probability confidence for MinTRL
    min_trials              int   = 10     Number of trials (for DSR correction)
    require_all             bool  = False  If False, need ≥ 4/6 gates to pass
    """

    def __init__(
        self,
        min_dsr: float = 0.95,
        min_t_stat: float = 3.0,
        max_param_cv: float = 0.50,
        min_sharpe_3x_slippage: float = 0.30,
        min_trl_confidence: float = 0.95,
        min_trials: int = 10,
        require_all: bool = False,
        min_equity_r2: float = 0.30,
        min_positive_folds_frac: float = 0.60,
        **_kwargs,
    ) -> None:
        self.min_dsr                 = float(min_dsr)
        self.min_t_stat              = float(min_t_stat)
        self.max_param_cv            = float(max_param_cv)
        self.min_sharpe_3x_slippage  = float(min_sharpe_3x_slippage)
        self.min_trl_confidence      = float(min_trl_confidence)
        self.min_trials              = int(min_trials)
        self.require_all             = bool(require_all)
        self.min_equity_r2           = float(min_equity_r2)
        self.min_positive_folds_frac = float(min_positive_folds_frac)

    # ── Main API ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        oos_returns: pd.Series,
        is_sharpe: float = 0.0,
        oos_sharpe_list: Optional[list] = None,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        param_sharpes: Optional[list] = None,
    ) -> RobustnessReport:
        """
        Evaluate all gates for a strategy.

        Parameters
        ----------
        oos_returns       Daily OOS return series (arithmetic).
        is_sharpe         In-sample Sharpe (for DSR benchmark).
        oos_sharpe_list   List of OOS Sharpe values from walk-forward folds
                          (for CPCV mean and stability).
        commission_pct    Per-side commission (for slippage scaling).
        slippage_pct      Base slippage (for slippage scaling).
        param_sharpes     Sharpe values across parameter window sweep
                          (for stability check). Falls back to oos_sharpe_list.
        """
        report = RobustnessReport()
        ret = oos_returns.dropna() if isinstance(oos_returns, pd.Series) else pd.Series(oos_returns).dropna()
        T   = len(ret)

        # 1. DSR
        sr_hat = float(sharpe_ratio(ret))
        report.dsr      = self._dsr(sr_hat, T, is_sharpe)
        report.dsr_pass = report.dsr >= self.min_dsr

        # 2. CPCV
        folds = oos_sharpe_list or []
        report.cpcv_sharpe = self._cpcv_sharpe(ret, folds)
        report.cpcv_pass   = report.cpcv_sharpe > 0.0

        # 3. Parameter stability
        stability_series = param_sharpes if param_sharpes else folds
        report.param_stability_cv   = self._param_stability_cv(stability_series)
        report.param_stability_pass = report.param_stability_cv < self.max_param_cv

        # 4. Slippage scaling
        base_cost = commission_pct + slippage_pct
        sr_by_mult: dict[int, float] = {}
        for mult in (1, 2, 3):
            adjusted = self._apply_cost_multiplier(ret, base_cost, mult)
            sr_by_mult[mult] = float(sharpe_ratio(adjusted))
        report.slippage_sharpes = sr_by_mult
        report.slippage_pass    = sr_by_mult.get(3, 0.0) >= self.min_sharpe_3x_slippage

        # 5. t-stat
        report.t_stat      = self._t_stat(ret)
        report.t_stat_pass = report.t_stat >= self.min_t_stat

        # 6. MinTRL
        report.min_trl_years = self._min_trl(ret, self.min_trl_confidence)
        observed_years       = T / 252.0
        report.min_trl_pass  = observed_years >= report.min_trl_years

        # 7. Equity curve R² (straight-line fit quality)
        equity = (1.0 + ret).cumprod() * 100.0
        try:
            from validation.overfitting_detector import equity_curve_r2
            report.equity_r2      = equity_curve_r2(equity)
        except Exception:
            y   = equity.values
            x   = np.arange(len(y), dtype=float)
            p   = np.polyfit(x, y, 1)
            res = y - np.polyval(p, x)
            ss_res = float(np.sum(res ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            report.equity_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        report.equity_r2_pass = report.equity_r2 >= self.min_equity_r2

        # 8. Positive folds fraction
        if oos_sharpe_list:
            n_pos = sum(1 for s in oos_sharpe_list if s > 0.0)
            report.positive_folds_frac = n_pos / len(oos_sharpe_list)
        else:
            report.positive_folds_frac = 1.0 if sr_hat > 0 else 0.0
        report.positive_folds_pass = report.positive_folds_frac >= self.min_positive_folds_frac

        # Summary (6 required gates + 2 quality gates; need ≥ 5/8 or all)
        passes = [
            report.dsr_pass,
            report.cpcv_pass,
            report.param_stability_pass,
            report.slippage_pass,
            report.t_stat_pass,
            report.min_trl_pass,
            report.equity_r2_pass,
            report.positive_folds_pass,
        ]
        report.n_passed     = sum(passes)
        report.overall_pass = all(passes) if self.require_all else report.n_passed >= 5

        logger.debug(
            "RobustnessGates: %d/8 passed | DSR=%.3f t=%.2f CPCV=%.3f CV=%.3f slippage3x=%.3f MinTRL=%.1fy R²=%.2f posfolds=%.2f",
            report.n_passed, report.dsr, report.t_stat, report.cpcv_sharpe,
            report.param_stability_cv, report.slippage_sharpes.get(3, 0.0),
            report.min_trl_years, report.equity_r2, report.positive_folds_frac,
        )
        return report

    # ── Gate implementations ──────────────────────────────────────────────────

    def _dsr(self, sr_hat: float, T: int, sr_star: float = 0.0) -> float:
        """
        Deflated Sharpe Ratio — Φ((SR_hat - SR*) × √T / σ_SR).

        where σ_SR = √((1 − γ3·SR + (γ4−1)/4·SR²) × (1/(T−1)))
        γ3 = skewness correction (≈ 0 for normally distributed returns).
        We use the simplified form assuming Gaussian returns.
        """
        if T < 2:
            return 0.0
        denom = math.sqrt(1.0 / (T - 1))
        if denom < 1e-12:
            return 0.0
        z = (sr_hat - sr_star) * math.sqrt(T) / max(math.sqrt(T) * denom, 1e-9)
        # Correct for multiple testing (Bonferroni-like): SR* raised by √(2 ln N)
        sr_benchmark = sr_star + math.sqrt(2.0 * math.log(max(self.min_trials, 1))) / math.sqrt(max(T, 1))
        z_adj = (sr_hat - sr_benchmark) * math.sqrt(T - 1)
        return float(np.clip(_norm_cdf(z_adj), 0.0, 1.0))

    def _cpcv_sharpe(self, ret: pd.Series, fold_sharpes: list) -> float:
        """
        Combinatorially Purged CV: mean OOS Sharpe across all folds.
        Falls back to sliding-window cross-validation if fold_sharpes empty.
        """
        if fold_sharpes:
            vals = [float(v) for v in fold_sharpes if np.isfinite(v)]
            return float(np.mean(vals)) if vals else 0.0

        # Sliding 5-fold approximation
        T = len(ret)
        if T < 50:
            return float(sharpe_ratio(ret))
        fold_size = T // 5
        sharpes = []
        for i in range(5):
            start = i * fold_size
            end   = start + fold_size
            test  = ret.iloc[start:end]
            if len(test) > 10:
                sharpes.append(float(sharpe_ratio(test)))
        return float(np.mean(sharpes)) if sharpes else 0.0

    def _param_stability_cv(self, sharpe_values: list) -> float:
        """Coefficient of variation of Sharpe across parameter windows."""
        vals = [float(v) for v in sharpe_values if np.isfinite(v)]
        if len(vals) < 2:
            return 0.0
        mean = float(np.mean(vals))
        if abs(mean) < 1e-9:
            return 1.0
        return float(np.std(vals, ddof=1) / abs(mean))

    def _apply_cost_multiplier(self, ret: pd.Series, base_cost: float, mult: int) -> pd.Series:
        """
        Approximate cost-multiplied returns.
        Every non-zero return is penalised by mult × base_cost on both sides.
        """
        cost_per_bar = base_cost * mult * 2  # round-trip
        sign_changes = (ret.shift(1) * ret < 0).astype(float)
        adjusted = ret - sign_changes * cost_per_bar
        return adjusted

    def _t_stat(self, ret: pd.Series) -> float:
        """t-statistic: mean / (std / sqrt(T)) × sqrt(252) annualisation."""
        if len(ret) < 2 or ret.std() < 1e-9:
            return 0.0
        T   = len(ret)
        ann = float(ret.mean() * _SQRT252 / (ret.std() + 1e-9))
        return float(ann * math.sqrt(T / 252.0))

    def _min_trl(self, ret: pd.Series, confidence: float = 0.95) -> float:
        """
        Minimum Track Record Length (years) needed to assert SR > 0 at given
        confidence level, accounting for non-normality of returns.

        MinTRL = (1/SR²) × (zα + zβ)² × (1 - γ3·SR + (γ4−1)/4·SR²)
        """
        sr = float(sharpe_ratio(ret))
        if abs(sr) < 1e-6:
            return 999.0
        gamma3 = float(ret.skew())    if len(ret) > 3 else 0.0
        gamma4 = float(ret.kurt() + 3) if len(ret) > 3 else 3.0  # excess → total

        z_alpha = _norm_ppf(confidence)
        z_beta  = _norm_ppf(0.50)       # β=0.50 → z_β=0

        correction = 1.0 - gamma3 * sr + ((gamma4 - 1.0) / 4.0) * (sr ** 2)
        correction = float(np.clip(correction, 0.5, 5.0))

        min_obs = ((z_alpha + z_beta) ** 2) * correction / (sr ** 2)
        return float(max(min_obs / 252.0, 0.0))
