"""
features/scaler.py
==================
Robust feature scaling pipeline for AlphaForge v2.0.

Solves the training/inference distribution mismatch that causes PSI > 9.0:
- Saves scaler fitted on training data; loads it at inference time.
- RobustScaler (median + IQR) is outlier-resistant vs StandardScaler.
- Per-feature winsorization before scaling clips extreme values.
- Optional regime-aware scaling: different center/scale per regime.

Usage
-----
    # Training time:
    scaler = FeatureScaler(regime_aware=True)
    X_scaled = scaler.fit_transform(X_train, regimes=regime_series)
    scaler.save("models/artifacts/spy_scaler.joblib")

    # Inference time:
    scaler = FeatureScaler.load("models/artifacts/spy_scaler.joblib")
    X_scaled = scaler.transform(X_live, regime=current_regime)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

_KNOWN_REGIMES = ("bull", "bear", "sideways", "high_vol", "unknown")


@dataclass
class _RegimeStats:
    """Per-regime median and IQR for one feature."""
    median: float = 0.0
    iqr:    float = 1.0
    low:    float = -3.0    # winsorization lower bound
    high:   float = 3.0     # winsorization upper bound


class FeatureScaler:
    """
    Robust per-feature scaler with optional regime-awareness.

    Algorithm
    ---------
    1. Winsorise each feature at (winsor_low, winsor_high) percentiles.
    2. Scale by median and IQR: z = (x - median) / max(IQR, epsilon).
    3. If regime_aware=True, compute separate median/IQR per regime;
       at inference time, use the current regime's stats.
    4. Falls back to global stats when a regime has < min_regime_samples.

    Parameters
    ----------
    regime_aware       : If True, compute per-regime statistics.
    winsor_low         : Lower winsorization percentile (default 1%).
    winsor_high        : Upper winsorization percentile (default 99%).
    eps                : Minimum IQR to avoid division by zero.
    min_regime_samples : Min samples to use regime-specific stats.
    """

    def __init__(
        self,
        regime_aware:       bool  = True,
        winsor_low:         float = 1.0,
        winsor_high:        float = 99.0,
        eps:                float = 1e-8,
        min_regime_samples: int   = 50,
    ) -> None:
        self.regime_aware       = regime_aware
        self.winsor_low         = winsor_low
        self.winsor_high        = winsor_high
        self.eps                = eps
        self.min_regime_samples = min_regime_samples

        # Global stats: {feature: (median, iqr, low_clip, high_clip)}
        self._global: dict[str, tuple[float, float, float, float]] = {}
        # Regime stats: {regime: {feature: _RegimeStats}}
        self._regime: dict[str, dict[str, _RegimeStats]] = {}
        self._feature_order: list[str] = []
        self._fitted = False

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X:       pd.DataFrame,
        regimes: Optional[pd.Series] = None,
    ) -> "FeatureScaler":
        """Fit scaler on training data."""
        self._feature_order = list(X.columns)

        for col in X.columns:
            vals = X[col].dropna().values.astype(float)
            if len(vals) == 0:
                self._global[col] = (0.0, 1.0, -1e9, 1e9)
                continue
            low_c  = float(np.percentile(vals, self.winsor_low))
            high_c = float(np.percentile(vals, self.winsor_high))
            clipped = np.clip(vals, low_c, high_c)
            median = float(np.median(clipped))
            q75, q25 = float(np.percentile(clipped, 75)), float(np.percentile(clipped, 25))
            iqr    = max(q75 - q25, self.eps)
            self._global[col] = (median, iqr, low_c, high_c)

        if self.regime_aware and regimes is not None:
            self._fit_regime_stats(X, regimes)

        self._fitted = True
        logger.info("FeatureScaler fitted: %d features, regime_aware=%s", len(X.columns), self.regime_aware)
        return self

    def _fit_regime_stats(self, X: pd.DataFrame, regimes: pd.Series) -> None:
        aligned = regimes.reindex(X.index).fillna("unknown").astype(str)
        for regime in aligned.unique():
            mask = aligned == regime
            sub  = X[mask]
            if len(sub) < self.min_regime_samples:
                continue
            self._regime[regime] = {}
            for col in X.columns:
                vals = sub[col].dropna().values.astype(float)
                if len(vals) < 10:
                    continue
                low_c  = float(np.percentile(vals, self.winsor_low))
                high_c = float(np.percentile(vals, self.winsor_high))
                clipped = np.clip(vals, low_c, high_c)
                median = float(np.median(clipped))
                q75, q25 = float(np.percentile(clipped, 75)), float(np.percentile(clipped, 25))
                iqr    = max(q75 - q25, self.eps)
                self._regime[regime][col] = _RegimeStats(
                    median=median, iqr=iqr, low=low_c, high=high_c
                )

    # ── Transformation ─────────────────────────────────────────────────────────

    def transform(
        self,
        X:      pd.DataFrame,
        regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply saved scaler to new data.

        Parameters
        ----------
        X      : Feature DataFrame (columns may be a subset of fitted columns).
        regime : Current market regime string (used if regime_aware=True).
        """
        if not self._fitted:
            logger.warning("FeatureScaler: not fitted — returning X unchanged")
            return X

        out = X.copy()
        regime_stats = self._regime.get(regime or "", {}) if self.regime_aware else {}

        for col in out.columns:
            if col not in self._global:
                continue
            glob_median, glob_iqr, low_c, high_c = self._global[col]

            rs = regime_stats.get(col)
            if rs is not None:
                median, iqr = rs.median, rs.iqr
                low_c, high_c = rs.low, rs.high
            else:
                median, iqr = glob_median, glob_iqr

            vals = out[col].values.astype(float)
            vals = np.clip(vals, low_c, high_c)          # winsorize
            out[col] = (vals - median) / iqr             # robust scale

        return out

    def fit_transform(
        self,
        X:       pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        regime:  Optional[str]       = None,
    ) -> pd.DataFrame:
        return self.fit(X, regimes=regimes).transform(X, regime=regime)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FeatureScaler saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        import joblib
        scaler = joblib.load(path)
        logger.info("FeatureScaler loaded ← %s (%d features)", path, len(scaler._feature_order))
        return scaler

    def to_dict(self) -> dict:
        return {
            "n_features":    len(self._feature_order),
            "regime_aware":  self.regime_aware,
            "n_regimes":     len(self._regime),
            "fitted":        self._fitted,
        }

    # ── Convenience path helpers ───────────────────────────────────────────────

    @staticmethod
    def artifact_path(artifacts_dir: str, ticker: str) -> str:
        return str(Path(artifacts_dir) / f"{ticker.lower()}_scaler.joblib")
