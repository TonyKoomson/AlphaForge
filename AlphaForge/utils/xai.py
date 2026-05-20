"""
utils/xai.py
============
Explainable AI (XAI) layer for AlphaForge v2.0.

Provides human-readable trade and portfolio explanations using:
  1. SHAP values (if `shap` is installed)
  2. Permutation importance fallback (pure numpy, no extra deps)

Key outputs
-----------
  TradeExplanation  — per-trade feature attribution + reasoning
  PortfolioExplanation — per-agent weight rationale

Usage
-----
    xai = XAIExplainer()
    xai.fit_baseline(feature_df, model)
    exp = xai.explain_trade(row, model, signal=1, confidence=0.72, regime="bull")
    print(exp.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

# Optional SHAP
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _shap = None       # type: ignore[assignment]
    _HAS_SHAP = False

_SENTINEL = "_feature_version"   # column to always exclude


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureContrib:
    feature: str
    value: float          # raw feature value
    importance: float     # SHAP or permutation importance (signed)
    direction: str        # "bullish" | "bearish" | "neutral"

    def to_dict(self) -> dict:
        return {
            "feature":    self.feature,
            "value":      round(float(self.value), 6),
            "importance": round(float(self.importance), 6),
            "direction":  self.direction,
        }


@dataclass
class TradeExplanation:
    timestamp: str
    ticker: str
    signal: float
    confidence: float
    regime: str
    sizing_reasoning: str
    risk_factors: list
    top_features: list = field(default_factory=list)   # list[FeatureContrib]
    raw_importances: dict = field(default_factory=dict)
    method: str = "permutation"

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "ticker":           self.ticker,
            "signal":           round(self.signal, 4),
            "confidence":       round(self.confidence, 4),
            "regime":           self.regime,
            "sizing_reasoning": self.sizing_reasoning,
            "risk_factors":     self.risk_factors,
            "top_features":     [f.to_dict() for f in self.top_features],
            "method":           self.method,
        }

    def summary(self) -> str:
        top = self.top_features[:3]
        feat_str = ", ".join(f"{f.feature}={f.value:.3f}({f.direction})" for f in top)
        return (
            f"[{self.ticker}] signal={self.signal:+.3f} conf={self.confidence:.2f} "
            f"regime={self.regime} | {feat_str}"
        )


@dataclass
class PortfolioExplanation:
    weights: dict
    method: str
    effective_n: float
    diversification_notes: list
    top_contributors: list   # list[dict]
    regime: str

    def to_dict(self) -> dict:
        return {
            "weights":                {k: round(v, 4) for k, v in self.weights.items()},
            "method":                 self.method,
            "effective_n":            round(self.effective_n, 3),
            "diversification_notes":  self.diversification_notes,
            "top_contributors":       self.top_contributors,
            "regime":                 self.regime,
        }


# ---------------------------------------------------------------------------
# XAI Explainer
# ---------------------------------------------------------------------------

class XAIExplainer:
    """
    Generates per-trade and per-portfolio explanations.

    Usage
    -----
        xai = XAIExplainer(top_k=10, use_shap=True)
        xai.fit_baseline(feature_df, model)
        exp = xai.explain_trade(row, model, signal, confidence, regime, ticker)
    """

    def __init__(self, top_k: int = 10, use_shap: bool = True) -> None:
        self.top_k      = int(top_k)
        self.use_shap   = use_shap and _HAS_SHAP
        self._baseline: Optional[np.ndarray] = None   # mean feature vector
        self._feat_cols: list[str]            = []
        self._shap_exp: Any = None
        self._history: list[dict] = []                # rolling explanation store

    # ── Setup ────────────────────────────────────────────────────────────────

    def fit_baseline(self, feature_df: pd.DataFrame, model: Any = None) -> None:
        """
        Set baseline (mean) features and optionally build SHAP explainer.

        Parameters
        ----------
        feature_df : DataFrame of features (rows = samples).
        model      : Fitted ML model with a predict_proba method (optional).
        """
        cols = [c for c in feature_df.columns if c != _SENTINEL and not c.startswith("_")]
        self._feat_cols = cols
        data = feature_df[cols].select_dtypes(include=[np.number]).dropna()
        self._baseline = data.mean(axis=0).values if not data.empty else None

        if self.use_shap and model is not None and self._baseline is not None:
            try:
                background = data.values[:min(100, len(data))]
                self._shap_exp = _shap.KernelExplainer(
                    model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                    background,
                )
                logger.info("XAI: SHAP KernelExplainer initialised on %d background samples", len(background))
            except Exception as exc:
                logger.warning("XAI: SHAP init failed (%s) — using permutation fallback", exc)
                self._shap_exp = None

    # ── Trade explanation ─────────────────────────────────────────────────────

    def explain_trade(
        self,
        feature_row: pd.Series,
        model: Any = None,
        signal: float = 0.0,
        confidence: float = 0.0,
        regime: str = "unknown",
        ticker: str = "",
        position_size: float = 0.0,
        pretrade_result: Optional[Any] = None,
    ) -> TradeExplanation:
        """
        Generate a trade explanation.

        Parameters
        ----------
        feature_row     Row of features for the current bar.
        model           Fitted model (for SHAP / permutation).
        signal          Raw signal value.
        confidence      Model confidence 0–1.
        regime          Current market regime label.
        ticker          Asset symbol.
        position_size   Final position size (post risk checks).
        pretrade_result PreTradeResult (optional, for risk factors).
        """
        ts = datetime.now(timezone.utc).isoformat()

        # Filter columns
        row = feature_row[[c for c in self._feat_cols if c in feature_row.index]]
        contribs = self._compute_importances(row, model, signal)

        top = sorted(contribs, key=lambda c: abs(c.importance), reverse=True)[: self.top_k]

        # Sizing reasoning
        direction = "long" if signal > 0 else "short" if signal < 0 else "flat"
        sizing_reason = (
            f"{direction} | conf={confidence:.2f} | regime={regime} | "
            f"size={position_size:.4f} | top_driver={top[0].feature if top else 'n/a'}"
        )

        # Risk factors from pretrade result
        risk_factors: list[str] = []
        if pretrade_result is not None:
            for chk in getattr(pretrade_result, "checks", []):
                if not chk.passed:
                    risk_factors.append(f"{chk.name}: {chk.message}")
        if not risk_factors:
            risk_factors = ["No risk flags"]

        exp = TradeExplanation(
            timestamp=ts,
            ticker=ticker,
            signal=float(signal),
            confidence=float(confidence),
            regime=regime,
            sizing_reasoning=sizing_reason,
            risk_factors=risk_factors,
            top_features=top,
            method="shap" if (self._shap_exp is not None) else "permutation",
        )

        self._history.append(exp.to_dict())
        if len(self._history) > 5000:
            self._history = self._history[-5000:]

        return exp

    def explain_portfolio(
        self,
        weights: dict,
        agents: list,
        regime: str = "unknown",
        method: str = "risk_parity",
    ) -> PortfolioExplanation:
        """
        Explain portfolio allocation across agents/strategies.

        Parameters
        ----------
        weights   {agent_id: weight} from PortfolioConstructor.
        agents    List of agent dicts with 'id', 'returns_series', 'sector'.
        regime    Current regime label.
        method    Construction method used.
        """
        if not weights:
            return PortfolioExplanation(
                weights={}, method=method, effective_n=0.0,
                diversification_notes=["No active strategies"],
                top_contributors=[], regime=regime,
            )

        w_arr = np.array(list(weights.values()))
        eff_n = float(1.0 / (w_arr ** 2).sum()) if (w_arr ** 2).sum() > 0 else 0.0

        notes = []
        if eff_n < 2.0:
            notes.append(f"Low diversification (ENB={eff_n:.1f}): concentrated in {max(weights, key=weights.get)}")
        elif eff_n >= len(weights) * 0.7:
            notes.append(f"Well diversified (ENB={eff_n:.1f} out of {len(weights)})")
        else:
            notes.append(f"Moderate diversification (ENB={eff_n:.1f})")

        # Sector concentration
        sectors: dict[str, float] = {}
        for a in agents:
            s = a.get("sector", "unknown")
            sectors[s] = sectors.get(s, 0.0) + weights.get(a["id"], 0.0)
        for sec, w in sectors.items():
            if w > 0.5:
                notes.append(f"High sector concentration: '{sec}' = {w:.1%}")

        top_contributors = sorted(
            [{"id": k, "weight": round(v, 4)} for k, v in weights.items()],
            key=lambda x: x["weight"], reverse=True,
        )[: self.top_k]

        return PortfolioExplanation(
            weights=weights,
            method=method,
            effective_n=eff_n,
            diversification_notes=notes,
            top_contributors=top_contributors,
            regime=regime,
        )

    # ── Feature importance summary ────────────────────────────────────────────

    def feature_importance_summary(self, lookback_n: int = 30) -> pd.DataFrame:
        """Return mean absolute importance per feature over recent explanations."""
        recent = self._history[-lookback_n:]
        if not recent:
            return pd.DataFrame(columns=["feature", "mean_abs_importance", "n_appearances"])
        counts: dict[str, list] = {}
        for rec in recent:
            for feat in rec.get("top_features", []):
                name = feat["feature"]
                if name not in counts:
                    counts[name] = []
                counts[name].append(abs(feat["importance"]))
        rows = [
            {"feature": k, "mean_abs_importance": float(np.mean(v)), "n_appearances": len(v)}
            for k, v in counts.items()
        ]
        df = pd.DataFrame(rows).sort_values("mean_abs_importance", ascending=False)
        return df.reset_index(drop=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_importances(
        self, row: pd.Series, model: Any, signal: float
    ) -> list:
        """Compute per-feature importances via SHAP or permutation."""
        if self._shap_exp is not None and model is not None:
            return self._shap_importances(row, signal)
        if model is not None and hasattr(model, "feature_importances_"):
            return self._tree_importances(row, model, signal)
        return self._gradient_importances(row, signal)

    def _shap_importances(self, row: pd.Series, signal: float) -> list:
        try:
            arr = row.values.reshape(1, -1)
            shap_vals = self._shap_exp.shap_values(arr, nsamples=50)
            if isinstance(shap_vals, list):
                # Multi-class: use class 1 (positive signal)
                vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            else:
                vals = shap_vals[0]
            return self._build_contribs(row, vals)
        except Exception as exc:
            logger.debug("SHAP failed (%s) — fallback to gradient", exc)
            return self._gradient_importances(row, signal)

    def _tree_importances(self, row: pd.Series, model: Any, signal: float) -> list:
        """Use model.feature_importances_ scaled by feature deviation from baseline."""
        fi = np.array(model.feature_importances_)
        if self._baseline is not None and len(fi) == len(self._baseline):
            deviation = row.values - self._baseline
            signed = fi * np.sign(deviation) * np.sign(signal)
        else:
            signed = fi * np.sign(signal)
        return self._build_contribs(row, signed)

    def _gradient_importances(self, row: pd.Series, signal: float) -> list:
        """
        Permutation-style: importance = |feature_value - baseline| × sign(deviation × signal).
        No model required.
        """
        if self._baseline is not None and len(self._baseline) == len(row):
            baseline = self._baseline
        else:
            baseline = np.zeros(len(row))
        deviation = row.values - baseline
        importance = np.abs(deviation) * np.sign(deviation * signal + 1e-9)
        return self._build_contribs(row, importance)

    def _build_contribs(self, row: pd.Series, importances: np.ndarray) -> list:
        feat_names = list(row.index)
        contribs = []
        for i, (name, imp) in enumerate(zip(feat_names, importances)):
            if abs(imp) < 1e-10:
                continue
            direction = "bullish" if imp > 0 else "bearish"
            contribs.append(FeatureContrib(
                feature=name,
                value=float(row.iloc[i]),
                importance=float(imp),
                direction=direction,
            ))
        return contribs
