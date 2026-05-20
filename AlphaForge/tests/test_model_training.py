from __future__ import annotations

import numpy as np
import pandas as pd

from models.train import (
    _time_based_purged_folds,
    generate_signals,
    train_model,
)


def _make_feature_df(n: int = 700, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    ret_1d = pd.Series(close, index=idx).pct_change().fillna(0.0)
    feat = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
            "ret_1d": ret_1d.values,
            "ret_5d": ret_1d.rolling(5).sum().fillna(0.0).values,
            "vol_21d": ret_1d.rolling(21).std().fillna(0.0).values,
            "rsi_14": 50 + 10 * np.tanh(ret_1d.values * 20),
        },
        index=idx,
    )
    feat["fwd_return"] = feat["close"].pct_change(5).shift(-5)
    feat["label"] = (feat["fwd_return"] > 0).astype(int)
    return feat.dropna()


def test_purged_folds_have_temporal_separation_and_embargo():
    idx = pd.bdate_range("2020-01-01", periods=700)
    folds = _time_based_purged_folds(idx, train_months=6, test_months=1, embargo_months=1)
    assert len(folds) > 0

    for train_start, train_end, test_start, test_end in folds:
        assert train_start <= train_end
        assert test_start <= test_end
        assert train_end < test_start
        # Roughly one month embargo gap (calendar-day based).
        assert (test_start - train_end).days >= 28


def test_train_model_respects_purged_walk_forward_dates(tmp_path):
    features = _make_feature_df()
    cfg = {
        "model": {
            "artifacts_dir": str(tmp_path),
            "random_state": 42,
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "ensemble_size": 2,
            "confidence_threshold": 0.65,
        }
    }
    result = train_model(features, target_horizon=5, config=cfg, confidence_threshold=0.65)
    fold_results = result["report"]["fold_results"]
    assert len(fold_results) > 0

    for fold in fold_results:
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        assert train_end < test_start
        assert (test_start - train_end).days >= 28


def test_generate_signals_applies_confidence_threshold():
    class DummyEnsemble:
        confidence_threshold = 0.65
        feature_columns = ["f1", "f2"]

        def predict(self, X):
            pred = np.array([0.03, 0.005])
            # std=0.01 → conf≈0.92 (above 0.65), std=0.25 → conf≈0.50 (below 0.65)
            std = np.array([0.01, 0.25])
            lower = pred - 1.0
            upper = pred + 1.0
            return pred, std, lower, upper

    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.1, 0.2]})
    out = generate_signals(DummyEnsemble(), X, confidence_threshold=0.65)
    assert isinstance(out, pd.DataFrame)
    assert list(out["signal"]) == [1, 0]
