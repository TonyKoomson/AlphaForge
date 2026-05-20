"""
risk/tail_risk.py
==================
Black-swan detection and tail-risk exposure scaling for AlphaForge v2.0.

Scenario classification
-----------------------
  normal      — vol_zscore < caution AND tail_prob > caution_prob
  caution     — vol_zscore ≥ caution OR tail_prob ≤ caution_prob
  extreme     — vol_zscore ≥ extreme threshold
  black_swan  — vol_zscore ≥ extreme AND tail_prob ≤ extreme_prob

Exposure multipliers
--------------------
  normal:               1.00
  caution:              0.70
  extreme / black_swan: 0.25

Stress tests reuse ``high_volatility_stress_test_periods()`` from
``risk/position_sizing.py`` (covid_crash, 2022_bear_market).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

_SQRT252 = np.sqrt(252)

# Scenario → position scale factor
_SCENARIO_EXPOSURE: dict[str, float] = {
    "normal":      1.00,
    "caution":     0.70,
    "extreme":     0.25,
    "black_swan":  0.25,
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class TailRiskAssessment:
    current_scenario:    str              # "normal" | "caution" | "extreme" | "black_swan"
    vol_zscore:          float
    tail_prob:           float            # empirical fraction of returns < -2σ in tail window
    recommended_exposure:float            # position scale factor 0–1
    stress_test_results: dict = field(default_factory=dict)   # scenario → simulated_return_pct

    def to_dict(self) -> dict:
        return {
            "current_scenario":     self.current_scenario,
            "vol_zscore":           round(self.vol_zscore, 4),
            "tail_prob":            round(self.tail_prob, 4),
            "recommended_exposure": round(self.recommended_exposure, 4),
            "stress_test_results":  self.stress_test_results,
        }


# ---------------------------------------------------------------------------
# TailRiskManager
# ---------------------------------------------------------------------------

class TailRiskManager:
    """
    Assess tail-risk conditions and recommend a position exposure multiplier.

    Usage
    -----
        trm = TailRiskManager()
        assessment = trm.assess(returns_series=ret_ser)
        multiplier = assessment.recommended_exposure   # multiply all position sizes by this
    """

    def __init__(
        self,
        vol_zscore_caution:  float = 2.0,
        vol_zscore_extreme:  float = 3.5,
        tail_prob_caution:   float = 0.05,   # <5% tail prob → caution
        tail_prob_extreme:   float = 0.01,   # <1% tail prob → extreme
        lookback_vol:        int   = 21,
        lookback_tail:       int   = 252,
    ) -> None:
        self.vol_zscore_caution  = vol_zscore_caution
        self.vol_zscore_extreme  = vol_zscore_extreme
        self.tail_prob_caution   = tail_prob_caution
        self.tail_prob_extreme   = tail_prob_extreme
        self.lookback_vol        = lookback_vol
        self.lookback_tail       = lookback_tail

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        returns_series: pd.Series,
        current_date: Optional[str] = None,
    ) -> TailRiskAssessment:
        """
        Compute tail-risk metrics and return an assessment.

        Parameters
        ----------
        returns_series : pd.Series
            Daily return series (index = dates).
        current_date : str, optional
            If supplied, filter returns to on-or-before this date.
        """
        if current_date is not None:
            cutoff = pd.Timestamp(current_date)
            returns_series = returns_series[returns_series.index <= cutoff]

        returns = returns_series.dropna()

        if len(returns) < self.lookback_vol + 5:
            logger.debug("TailRiskManager: insufficient data (%d bars)", len(returns))
            return TailRiskAssessment(
                current_scenario="normal",
                vol_zscore=0.0,
                tail_prob=0.10,
                recommended_exposure=1.0,
            )

        vol_z    = self._vol_zscore(returns)
        tail_p   = self._empirical_tail_prob(returns)
        scenario = self._detect_scenario(vol_z, tail_p)
        exposure = _SCENARIO_EXPOSURE[scenario]

        logger.debug(
            "TailRiskAssessment: scenario=%s vol_z=%.2f tail_prob=%.3f exposure=%.2f",
            scenario, vol_z, tail_p, exposure,
        )
        return TailRiskAssessment(
            current_scenario=scenario,
            vol_zscore=vol_z,
            tail_prob=tail_p,
            recommended_exposure=exposure,
        )

    def run_stress_tests(
        self,
        feat_df: pd.DataFrame,
        position_sizes: dict[str, float],
        stress_periods: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Simulate portfolio return during known stress windows.

        Parameters
        ----------
        feat_df : DataFrame with DatetimeIndex and a 'close' column.
        position_sizes : {asset_id → normalised size in [-1, 1]}.
        stress_periods : list of period names; defaults to all known periods.

        Returns
        -------
        dict : {period_name → simulated_portfolio_return_pct}
        """
        from risk.position_sizing import high_volatility_stress_test_periods
        all_periods = high_volatility_stress_test_periods()

        if stress_periods:
            periods = {k: v for k, v in all_periods.items() if k in stress_periods}
        else:
            periods = all_periods

        results: dict[str, float] = {}

        if "close" not in feat_df.columns:
            return results

        for name, (start_s, end_s) in periods.items():
            try:
                mask = (feat_df.index >= pd.Timestamp(start_s)) & (feat_df.index <= pd.Timestamp(end_s))
                period_df = feat_df[mask]
                if len(period_df) < 5:
                    results[name] = 0.0
                    continue
                ret = period_df["close"].pct_change().dropna()
                # Portfolio return: weighted sum of position × asset return
                # (single-asset fallback: use total_weight)
                total_weight = sum(abs(v) for v in position_sizes.values())
                if total_weight <= 0:
                    results[name] = 0.0
                    continue
                # Directional P&L (assume average position direction = sign of net weight)
                net_dir = np.sign(sum(position_sizes.values()))
                period_ret = float((1 + ret * net_dir * (total_weight / len(position_sizes))).prod() - 1)
                results[name] = round(period_ret * 100, 2)   # in percent
            except Exception as exc:
                logger.warning("TailRiskManager stress test '%s': %s", name, exc)
                results[name] = 0.0

        return results

    # ── Statistical helpers ───────────────────────────────────────────────────

    def _vol_zscore(self, returns: pd.Series) -> float:
        """How many σ is current vol above its rolling mean."""
        ret = returns.dropna()
        if len(ret) < self.lookback_vol + self.lookback_vol:
            return 0.0
        recent_vol = float(ret.tail(self.lookback_vol).std() * _SQRT252)
        hist_vol   = ret.tail(self.lookback_tail).rolling(self.lookback_vol).std() * _SQRT252
        mean_vol   = float(hist_vol.mean())
        std_vol    = float(hist_vol.std())
        if std_vol < 1e-9:
            return 0.0
        return (recent_vol - mean_vol) / std_vol

    def _empirical_tail_prob(self, returns: pd.Series) -> float:
        """Fraction of returns below -2σ in the tail lookback window."""
        ret = returns.dropna().tail(self.lookback_tail)
        if len(ret) < 20:
            return 0.10
        sigma   = float(ret.std())
        threshold = -2.0 * sigma
        return float((ret < threshold).mean())

    def _detect_scenario(self, vol_zscore: float, tail_prob: float) -> str:
        if vol_zscore >= self.vol_zscore_extreme and tail_prob <= self.tail_prob_extreme:
            return "black_swan"
        if vol_zscore >= self.vol_zscore_extreme:
            return "extreme"
        if vol_zscore >= self.vol_zscore_caution or tail_prob <= self.tail_prob_caution:
            return "caution"
        return "normal"

    def _scale_exposure(self, scenario: str) -> float:
        return _SCENARIO_EXPOSURE.get(scenario, 1.0)
