"""
features/regime_hmm.py
=======================
HMM-based regime detection for AlphaForge v2.0.

Implements a pure-numpy 3-state Gaussian HMM fitted via EM (Baum-Welch).
State decoding uses Viterbi. No scipy, no hmmlearn required.

States are automatically labelled after fitting by comparing each state's
mean return:
  highest mean  → "bull"
  lowest mean   → "bear"
  middle        → "sideways"

Features used:  log_return, realised_vol_21d  (z-scored before fitting)

Inflation Regime Matrix
-----------------------
`get_inflation_regime(cpi_yoy, gdp_growth)` classifies macro environment:
  goldilocks   — high growth, low inflation
  stagflation  — low growth, high inflation
  reflation    — high growth, high inflation
  deflation    — low growth, low inflation

Usage
-----
    det = HMMRegimeDetector()
    det.fit(price_df)                  # DataFrame with 'close' column
    labels = det.predict(price_df)     # pd.Series of regime strings
    proba  = det.predict_proba(price_df)   # DataFrame, columns = state names
    current = det.current_regime(price_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

_EPS  = 1e-300
_SQRT252 = np.sqrt(252)


# ---------------------------------------------------------------------------
# Pure-numpy Gaussian HMM
# ---------------------------------------------------------------------------

class GaussianHMM:
    """
    K-state Gaussian HMM with diagonal covariance, fitted via Baum-Welch EM.

    Parameters
    ----------
    n_states  : int   Number of hidden states.
    max_iter  : int   Maximum EM iterations.
    tol       : float Convergence threshold on log-likelihood improvement.
    """

    def __init__(self, n_states: int = 3, max_iter: int = 100, tol: float = 1e-4) -> None:
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tol      = float(tol)

        # Parameters (initialised on fit)
        self.pi:     Optional[np.ndarray] = None   # (K,) initial state probs
        self.A:      Optional[np.ndarray] = None   # (K,K) transition matrix
        self.mu:     Optional[np.ndarray] = None   # (K, D) emission means
        self.sigma2: Optional[np.ndarray] = None   # (K, D) emission variances
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit HMM via Baum-Welch.

        Parameters
        ----------
        X : (T, D) array of observations.
        """
        T, D = X.shape
        K    = self.n_states

        # K-means initialisation (3 passes)
        self.pi, self.A, self.mu, self.sigma2 = self._kmeans_init(X)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E step
            log_emission = self._log_emission(X)      # (T, K)
            alpha, log_scale = self._forward(log_emission)
            beta             = self._backward(log_emission, log_scale)
            gamma, xi        = self._e_step(alpha, beta, log_emission)

            ll = float(np.sum(log_scale))

            # M step
            self.pi     = np.clip(gamma[0], _EPS, None)
            self.pi    /= self.pi.sum()
            self.A      = xi.sum(axis=0)
            self.A     /= self.A.sum(axis=1, keepdims=True).clip(_EPS)

            self.mu     = (gamma[:, :, None] * X[:, None, :]).sum(axis=0) / gamma.sum(axis=0)[:, None].clip(_EPS)
            diff        = X[:, None, :] - self.mu[None, :, :]  # (T, K, D)
            self.sigma2 = (gamma[:, :, None] * diff ** 2).sum(axis=0) / gamma.sum(axis=0)[:, None].clip(_EPS)
            self.sigma2 = np.clip(self.sigma2, 1e-6, None)

            if abs(ll - prev_ll) < self.tol:
                logger.debug("HMM converged at iteration %d (ll=%.4f)", iteration, ll)
                break
            prev_ll = ll

        self._fitted = True
        return self

    # ── Decoding ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — returns (T,) array of state indices."""
        self._check_fitted()
        T = len(X)
        K = self.n_states
        log_em = self._log_emission(X)  # (T, K)
        log_A  = np.log(self.A + _EPS)
        log_pi = np.log(self.pi + _EPS)

        delta = np.full((T, K), -np.inf)
        psi   = np.zeros((T, K), dtype=int)

        delta[0] = log_pi + log_em[0]
        for t in range(1, T):
            trans = delta[t - 1, :, None] + log_A          # (K, K)
            psi[t]   = trans.argmax(axis=0)
            delta[t] = trans.max(axis=0) + log_em[t]

        # Backtrack
        path    = np.zeros(T, dtype=int)
        path[T - 1] = delta[T - 1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Filtered state probabilities — (T, K) via forward pass only.

        Uses only observations 1..t to compute P(state | obs_{1:t}), so there
        is no look-ahead. Previously used forward-backward (smoothed), which
        incorporated future observations and introduced look-ahead bias when
        these probabilities were used as model features.
        """
        self._check_fitted()
        log_em = self._log_emission(X)
        alpha, _ = self._forward(log_em)
        return alpha

    # ── Internal maths ────────────────────────────────────────────────────────

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """(T, K) log p(x_t | state k) under Gaussian with diagonal cov."""
        T, D = X.shape
        K    = self.n_states
        log_p = np.zeros((T, K))
        for k in range(K):
            diff    = X - self.mu[k]                         # (T, D)
            log_p[:, k] = (
                -0.5 * D * np.log(2 * np.pi)
                - 0.5 * np.sum(np.log(self.sigma2[k] + _EPS))
                - 0.5 * np.sum(diff ** 2 / (self.sigma2[k] + _EPS), axis=1)
            )
        return log_p

    def _forward(self, log_em: np.ndarray) -> tuple:
        """Scaled forward pass. Returns (alpha, log_scales) where alpha is normalised."""
        T, K = log_em.shape
        alpha      = np.zeros((T, K))
        log_scales = np.zeros(T)

        alpha[0] = np.exp(np.log(self.pi + _EPS) + log_em[0])
        s = alpha[0].sum()
        log_scales[0] = np.log(s + _EPS)
        alpha[0] /= (s + _EPS)

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * np.exp(log_em[t])
            s = alpha[t].sum()
            log_scales[t] = np.log(s + _EPS)
            alpha[t] /= (s + _EPS)
        return alpha, log_scales

    def _backward(self, log_em: np.ndarray, log_scales: np.ndarray) -> np.ndarray:
        """Scaled backward pass."""
        T, K = log_em.shape
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * np.exp(log_em[t + 1]) * beta[t + 1]).sum(axis=1)
            beta[t] /= (np.exp(log_scales[t + 1]) + _EPS)
        return beta

    def _e_step(
        self, alpha: np.ndarray, beta: np.ndarray, log_em: np.ndarray
    ) -> tuple:
        """Compute gamma and xi from alpha / beta."""
        T, K = alpha.shape
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True).clip(_EPS)

        em = np.exp(log_em)                                         # (T, K)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = (
                alpha[t, :, None]
                * self.A
                * em[t + 1, None, :]
                * beta[t + 1, None, :]
            )
            xi[t] /= xi[t].sum().clip(_EPS)
        return gamma, xi

    def _kmeans_init(self, X: np.ndarray) -> tuple:
        """K-means++ initialisation for HMM parameters."""
        T, D = X.shape
        K    = self.n_states

        # Simple K-means++ centres
        centres = [X[np.random.randint(T)]]
        for _ in range(1, K):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centres) for x in X])
            probs = dists / (dists.sum() + _EPS)
            idx   = np.searchsorted(np.cumsum(probs), np.random.rand())
            centres.append(X[min(idx, T - 1)])
        centres_arr = np.array(centres)

        # Assign labels
        labels = np.array([np.argmin(np.sum((X - c) ** 2, axis=1)) for c in centres_arr]).T
        # Actually compute nearest centre for each point
        dists_matrix = np.stack([np.sum((X - c) ** 2, axis=1) for c in centres_arr], axis=1)
        labels = dists_matrix.argmin(axis=1)  # (T,)

        mu     = np.stack([X[labels == k].mean(axis=0) if (labels == k).any() else X.mean(axis=0) for k in range(K)])
        sigma2 = np.stack([X[labels == k].var(axis=0).clip(1e-6) if (labels == k).any() else np.ones(D) for k in range(K)])

        # Uniform initialisation for transition / initial
        pi = np.ones(K) / K
        A  = np.ones((K, K)) / K

        return pi, A, mu, sigma2

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GaussianHMM is not fitted yet — call fit() first.")


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """
    High-level wrapper around GaussianHMM for market regime detection.

    States are auto-labelled by mean log-return rank:
      max mean  → "bull"
      min mean  → "bear"
      middle    → "sideways"

    Config keys (config["hmm_regime"])
    ------------------------------------
    n_states         int   = 3
    max_iter         int   = 100
    tol              float = 1e-4
    min_history      int   = 60    (minimum bars needed to fit)
    refit_interval_bars int = 63   (refit every N bars in rolling mode)
    """

    def __init__(
        self,
        n_states: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        min_history: int = 60,
        refit_interval_bars: int = 63,
        **_kwargs,
    ) -> None:
        self.n_states            = int(n_states)
        self.max_iter            = int(max_iter)
        self.tol                 = float(tol)
        self.min_history         = int(min_history)
        self.refit_interval_bars = int(refit_interval_bars)

        self._hmm:        Optional[GaussianHMM] = None
        self._label_map:  dict[int, str]        = {}
        self._fit_scaler: Optional[tuple]       = None  # (mean, std) for z-score

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit HMM on historical price DataFrame.

        Parameters
        ----------
        df : DataFrame with at least a 'close' column and DatetimeIndex.
        """
        X, _ = self._build_features(df)
        if len(X) < self.min_history:
            logger.warning("HMMRegimeDetector: insufficient data (%d < %d)", len(X), self.min_history)
            return self

        self._fit_scaler = (X.mean(axis=0), X.std(axis=0).clip(1e-9))
        Xn = (X - self._fit_scaler[0]) / self._fit_scaler[1]

        self._hmm = GaussianHMM(self.n_states, self.max_iter, self.tol)
        self._hmm.fit(Xn)
        self._build_label_map(Xn)
        logger.info("HMMRegimeDetector fitted (%d states, %d bars)", self.n_states, len(Xn))
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return pd.Series of regime string labels aligned to df.index."""
        if self._hmm is None:
            return pd.Series("unknown", index=df.index)

        Xn, valid_idx = self._build_features_normalised(df)
        if len(Xn) == 0:
            return pd.Series("unknown", index=df.index)

        states = self._hmm.predict(Xn)
        labels = pd.Series([self._label_map.get(int(s), "unknown") for s in states], index=valid_idx)
        return labels.reindex(df.index).fillna("unknown")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return (T, n_states) DataFrame of smoothed regime probabilities."""
        if self._hmm is None:
            cols = ["bull", "bear", "sideways"][: self.n_states]
            return pd.DataFrame(1.0 / self.n_states, index=df.index, columns=cols)

        Xn, valid_idx = self._build_features_normalised(df)
        if len(Xn) == 0:
            cols = list(self._label_map.values()) or ["unknown"]
            return pd.DataFrame(1.0 / len(cols), index=df.index, columns=cols)

        proba = self._hmm.predict_proba(Xn)
        cols  = [self._label_map.get(k, f"state_{k}") for k in range(self.n_states)]
        return pd.DataFrame(proba, index=valid_idx, columns=cols).reindex(df.index).fillna(1.0 / self.n_states)

    def current_regime(self, df: pd.DataFrame) -> str:
        """Return the regime label for the most recent bar."""
        labels = self.predict(df)
        if labels.empty:
            return "unknown"
        last = labels.dropna()
        return str(last.iloc[-1]) if not last.empty else "unknown"

    def is_fitted(self) -> bool:
        return self._hmm is not None and self._hmm._fitted

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """Extract log_return + realised_vol_21d as 2-column feature array."""
        close = df["close"].astype(float) if "close" in df.columns else df.iloc[:, 0].astype(float)
        log_ret = np.log(close / close.shift(1)).dropna()
        vol21   = log_ret.rolling(21, min_periods=10).std() * _SQRT252
        combined = pd.DataFrame({"log_ret": log_ret, "vol21": vol21}).dropna()
        return combined.values, combined.index

    def _build_features_normalised(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
        X, idx = self._build_features(df)
        if self._fit_scaler is None or len(X) == 0:
            return X, idx
        mean, std = self._fit_scaler
        Xn = (X - mean) / std
        return Xn, idx

    def _build_label_map(self, Xn: np.ndarray) -> None:
        """Assign regime names to state indices by mean log-return rank."""
        if self._hmm is None:
            return
        # mu[:, 0] = z-scored mean log return per state
        mean_rets = self._hmm.mu[:, 0]
        rank      = np.argsort(mean_rets)  # rank[0] = lowest (bear), rank[-1] = bull
        K = self.n_states
        labels: dict[int, str] = {}
        if K >= 3:
            labels[int(rank[0])]  = "bear"
            labels[int(rank[-1])] = "bull"
            for i, idx in enumerate(rank[1:-1]):
                labels[int(idx)] = "sideways" if i == 0 else f"neutral_{i}"
        elif K == 2:
            labels[int(rank[0])]  = "bear"
            labels[int(rank[-1])] = "bull"
        else:
            labels[int(rank[0])] = "bull"
        self._label_map = labels
        logger.debug("HMM label map: %s", labels)


# ---------------------------------------------------------------------------
# Inflation / Macro regime matrix
# ---------------------------------------------------------------------------

def get_inflation_regime(cpi_yoy: float, gdp_growth: float) -> str:
    """
    Classify macro environment into one of four regimes.

    Parameters
    ----------
    cpi_yoy    Annual CPI inflation (e.g. 0.04 = 4%).
    gdp_growth Annual GDP growth rate (e.g. 0.02 = 2%).

    Returns
    -------
    "goldilocks"  — high growth, low inflation  (equity-positive)
    "stagflation" — low growth, high inflation  (defensive assets)
    "reflation"   — high growth, high inflation (commodities, TIPS)
    "deflation"   — low growth, low inflation   (bonds, cash)
    """
    high_inflation = cpi_yoy > 0.035
    high_growth    = gdp_growth > 0.025

    if high_growth and not high_inflation:
        return "goldilocks"
    if not high_growth and high_inflation:
        return "stagflation"
    if high_growth and high_inflation:
        return "reflation"
    return "deflation"


MACRO_REGIME_PLAYBOOK: dict[str, dict] = {
    "goldilocks":  {"equity_bias": +1.0, "bond_bias": +0.5, "commodity_bias": +0.3},
    "reflation":   {"equity_bias": +0.5, "bond_bias": -0.5, "commodity_bias": +1.0},
    "stagflation": {"equity_bias": -0.5, "bond_bias": -0.3, "commodity_bias": +0.8},
    "deflation":   {"equity_bias": -0.3, "bond_bias": +1.0, "commodity_bias": -0.5},
}
