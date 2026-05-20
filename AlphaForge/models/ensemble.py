"""
models/ensemble.py
==================
Multi-Model Ensemble for AlphaForge v2.0.

Combines XGBoost + RandomForest + LogisticRegression into a stacked
or averaged ensemble. Each model contributes differently:
  - XGBoost:  captures non-linear feature interactions.
  - RandomForest: robust to noise; diverse via bagging.
  - LogReg:   linear baseline; calibrated probabilities; hard to overfit.

Diversity improves OOS Sharpe and reduces overfitting vs. any single model.

Usage
-----
    from models.ensemble import EnsembleModel

    ens = EnsembleModel(method='stack')     # or 'average' / 'vote'
    ens.fit(X_train, y_train)
    probas = ens.predict_proba(X_test)      # shape (n,)
    signals = ens.predict_signal(X_test, threshold=0.55)

Config (config["ensemble"])
-----------------------------
  method     str   = "average"   # average | stack | vote
  use_xgb    bool  = True
  use_rf     bool  = True
  use_logreg bool  = True
  n_xgb      int   = 3           # number of XGBoost estimators in sub-ensemble
  n_rf       int   = 1
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleFitResult:
    method:           str
    n_models:         int
    train_accuracy:   float
    duration_s:       float
    component_weights: dict[str, float]


class EnsembleModel:
    """
    Weighted ensemble of XGBoost, RandomForest, and LogisticRegression.

    Parameters
    ----------
    method     : 'average' | 'stack' | 'vote'
                 average = weighted mean of probabilities
                 stack   = meta-learner (LR) on base model outputs
                 vote    = majority vote on signals
    use_xgb, use_rf, use_logreg : which base models to include.
    n_xgb      : size of XGBoost sub-ensemble (averaged internally).
    """

    def __init__(
        self,
        method:     str  = "average",
        use_xgb:    bool = True,
        use_rf:     bool = True,
        use_logreg: bool = True,
        n_xgb:      int  = 3,
        n_rf:       int  = 1,
        **_kwargs,
    ) -> None:
        self.method     = method
        self.use_xgb    = use_xgb
        self.use_rf     = use_rf
        self.use_logreg = use_logreg
        self.n_xgb      = n_xgb
        self.n_rf       = n_rf

        self._models:  dict[str, Any] = {}
        self._weights: dict[str, float] = {}
        self._meta:    Any = None
        self._feature_cols: list[str] = []
        self._fitted = False

    # ── Building base models ──────────────────────────────────────────────────

    def _make_xgb(self):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                use_label_encoder=False, verbosity=0,
            )
        except ImportError:
            return self._make_rf()

    def _make_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1,
            min_samples_leaf=5, random_state=42,
        )

    def _make_logreg(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X:        pd.DataFrame,
        y:        pd.Series,
        val_frac: float = 0.20,
    ) -> EnsembleFitResult:
        """Fit all base models. val_frac held out for weight calibration."""
        t0 = time.time()
        self._feature_cols = list(X.columns)
        X_arr = X.values
        y_arr = y.reindex(X.index).fillna(0).values.astype(int)

        n_val = max(int(len(X) * val_frac), 20)
        X_tr, X_val = X_arr[:-n_val], X_arr[-n_val:]
        y_tr, y_val = y_arr[:-n_val], y_arr[-n_val:]

        if len(np.unique(y_tr)) < 2:
            # All same class — use trivial models
            self._fitted = False
            return EnsembleFitResult(self.method, 0, 0.5, 0.0, {})

        # Fit base models
        if self.use_xgb:
            xgb_models = [self._make_xgb() for _ in range(self.n_xgb)]
            for i, m in enumerate(xgb_models):
                try:
                    m.fit(X_tr, y_tr)
                except Exception as e:
                    logger.debug("XGBoost model %d: %s", i, e)
            self._models["xgb"] = xgb_models

        if self.use_rf:
            rf = self._make_rf()
            try:
                rf.fit(X_tr, y_tr)
                self._models["rf"] = [rf]
            except Exception as e:
                logger.debug("RF: %s", e)

        if self.use_logreg:
            lr = self._make_logreg()
            try:
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                lr.fit(sc.fit_transform(X_tr), y_tr)
                self._models["logreg"] = [lr]
                self._models["logreg_scaler"] = sc
            except Exception as e:
                logger.debug("LogReg: %s", e)

        # Calibrate weights on validation set
        self._calibrate_weights(X_val, y_val)

        # Stack meta-learner if requested
        if self.method == "stack" and len(self._models) > 1:
            self._fit_meta(X_val, y_val)

        self._fitted = True
        acc = float(np.mean((self.predict_proba(pd.DataFrame(X_val, columns=self._feature_cols)) > 0.5) == y_val))
        return EnsembleFitResult(
            self.method, len(self._models), acc, time.time() - t0, self._weights
        )

    def _calibrate_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Set component weights proportional to OOS accuracy."""
        for key, mlist in self._models.items():
            if key == "logreg_scaler":
                continue
            try:
                p = self._predict_component(key, X_val)
                acc = float(np.mean((p > 0.5) == y_val))
                self._weights[key] = max(0.0, acc - 0.5)
            except Exception:
                self._weights[key] = 0.0
        total = sum(self._weights.values())
        if total > 1e-9:
            self._weights = {k: v / total for k, v in self._weights.items()}
        else:
            # Equal weights fallback
            keys = [k for k in self._models if k != "logreg_scaler"]
            self._weights = {k: 1.0 / len(keys) for k in keys}

    def _fit_meta(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        meta_X = self._base_probas(pd.DataFrame(X_val, columns=self._feature_cols))
        if meta_X.shape[1] < 1:
            return
        self._meta = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        try:
            self._meta.fit(meta_X, y_val)
        except Exception:
            self._meta = None

    def _predict_component(self, key: str, X_arr: np.ndarray) -> np.ndarray:
        mlist = self._models[key]
        if key == "logreg":
            sc    = self._models.get("logreg_scaler")
            X_use = sc.transform(X_arr) if sc else X_arr
            return np.mean([m.predict_proba(X_use)[:, 1] for m in mlist], axis=0)
        return np.mean([m.predict_proba(X_arr)[:, 1] for m in mlist], axis=0)

    def _base_probas(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self._feature_cols].values
        cols = []
        for key in ("xgb", "rf", "logreg"):
            if key in self._models:
                try:
                    cols.append(self._predict_component(key, X_arr))
                except Exception:
                    pass
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class, shape (n,)."""
        if not self._fitted or not self._models:
            return np.full(len(X), 0.5)

        if self.method == "stack" and self._meta is not None:
            meta_X = self._base_probas(X)
            if meta_X.shape[1] > 0:
                return self._meta.predict_proba(meta_X)[:, 1]

        if self.method == "vote":
            sigs = []
            X_arr = X[self._feature_cols].values
            for key in ("xgb", "rf", "logreg"):
                if key in self._models:
                    try:
                        p = self._predict_component(key, X_arr)
                        sigs.append((p > 0.5).astype(float))
                    except Exception:
                        pass
            return np.mean(sigs, axis=0) if sigs else np.full(len(X), 0.5)

        # Weighted average (default)
        X_arr = X[self._feature_cols].values
        total_w = 0.0
        weighted_p = np.zeros(len(X))
        for key, w in self._weights.items():
            if key == "logreg_scaler" or key not in self._models:
                continue
            try:
                p = self._predict_component(key, X_arr)
                weighted_p += w * p
                total_w += w
            except Exception:
                pass
        return weighted_p / max(total_w, 1e-9)

    def predict_signal(self, X: pd.DataFrame, threshold: float = 0.55) -> np.ndarray:
        p = self.predict_proba(X)
        return np.where(p > threshold, 1, np.where(p < 1 - threshold, -1, 0))
