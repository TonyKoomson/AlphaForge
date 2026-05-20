"""ML training module: purged walk-forward XGBoost regression with uncertainty."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features.engine import _get_feature_cols, meta_label as _meta_label_fn
from features.momentum import compute_signals as baseline_momentum_signals
from utils.helpers import ensure_dir, get_logger, load_config

logger = get_logger(__name__)


def _clean_to_float32(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame slice to float32 and replace inf/NaN with 0.

    Uses numpy isfinite() instead of pandas replace() to avoid the
    (n_cols × n_rows) bool mask allocation that causes OOM on large datasets.
    Returns a new DataFrame with the same index/columns.
    """
    arr = df_slice.values.astype(np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return pd.DataFrame(arr, index=df_slice.index, columns=df_slice.columns)


def _finite_row_mask(df_slice: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True for rows where all values are finite."""
    return pd.Series(np.isfinite(df_slice.values).all(axis=1), index=df_slice.index)


@dataclass
class TrainedEnsembleModel:
    """Serializable wrapper for ensemble models and metadata."""

    models: list[Any]
    feature_columns: list[str]
    target_horizon: int
    confidence_threshold: float
    train_params: dict[str, Any]
    trained_at: str
    calibrator: Any = None       # optional IsotonicRegression fitted on OOS fold preds
    meta_model: Any = None       # secondary classifier: P(primary_signal_is_correct)
    model_weights: Any = None    # Sharpe-weighted blend weights (len == len(models))

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # reindex so optional feature columns absent at inference are zero-filled
        # rather than raising KeyError (fracdiff / change_point / MTF may be missing)
        X_use = X.reindex(columns=self.feature_columns, fill_value=0.0)
        preds_list = []
        for m in self.models:
            # Classifiers expose predict_proba — use P(class=1) for ensemble blending.
            # Regressors (XGBoost trained on binary targets) use predict directly.
            if hasattr(m, "predict_proba"):
                preds_list.append(m.predict_proba(X_use)[:, 1])
            else:
                preds_list.append(m.predict(X_use))
        if not preds_list:
            raise RuntimeError(
                "TrainedEnsembleModel.predict() called with zero trained members — "
                "all fold training tasks failed (OOM or training error). "
                "Check logs for member-level exceptions."
            )
        preds = np.column_stack(preds_list)
        # Sharpe-weighted ensemble blending: if weights are stored, use them;
        # otherwise fall back to equal weights. Weights are softmax-normalised
        # OOS Sharpe scores computed during walk-forward training.
        if self.model_weights is not None and len(self.model_weights) == preds.shape[1]:
            w = np.array(self.model_weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            w_sum = w.sum()
            if w_sum > 1e-9:
                w = w / w_sum
                mean_pred = (preds * w[np.newaxis, :]).sum(axis=1)
            else:
                mean_pred = preds.mean(axis=1)
        else:
            mean_pred = preds.mean(axis=1)
        std_pred = preds.std(axis=1)
        lower = np.quantile(preds, 0.10, axis=1)
        upper = np.quantile(preds, 0.90, axis=1)
        # Apply isotonic calibration when available — maps raw scores to true P(up)
        if self.calibrator is not None:
            try:
                mean_pred = np.clip(self.calibrator.predict(mean_pred), 0.0, 1.0)
            except Exception:
                pass  # calibration failure is non-fatal
        return mean_pred, std_pred, lower, upper

    def predict_raw(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Same as predict() but skips isotonic calibration.

        Isotonic calibrators can have wide flat steps that collapse all scores
        in a range to the same value, destroying cross-sectional rank ordering.
        Use this for ranking-based strategies where ordinal signal matters more
        than calibrated probability accuracy.
        """
        X_use = X.reindex(columns=self.feature_columns, fill_value=0.0)
        preds_list = []
        for m in self.models:
            if hasattr(m, "predict_proba"):
                preds_list.append(m.predict_proba(X_use)[:, 1])
            else:
                preds_list.append(m.predict(X_use))
        if not preds_list:
            raise RuntimeError(
                "TrainedEnsembleModel.predict_raw() called with zero trained members"
            )
        preds = np.column_stack(preds_list)
        if self.model_weights is not None and len(self.model_weights) == preds.shape[1]:
            w = np.array(self.model_weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            w_sum = w.sum()
            if w_sum > 1e-9:
                w = w / w_sum
                mean_pred = (preds * w[np.newaxis, :]).sum(axis=1)
            else:
                mean_pred = preds.mean(axis=1)
        else:
            mean_pred = preds.mean(axis=1)
        std_pred = preds.std(axis=1)
        lower = np.quantile(preds, 0.10, axis=1)
        upper = np.quantile(preds, 0.90, axis=1)
        return mean_pred, std_pred, lower, upper

    def predict_meta(self, X: pd.DataFrame, primary_probas: np.ndarray) -> np.ndarray:
        """Return P(primary_signal_is_correct) for each bar — used as a bet-size scaler.

        When the meta-model is absent (not yet trained or failed) returns 1.0 for all
        bars so the caller's position sizing logic is unaffected.
        """
        if self.meta_model is None:
            return np.ones(len(X))
        try:
            X_use = X.reindex(columns=self.feature_columns, fill_value=0.0).copy()
            # Append primary signal direction as additional context feature
            X_use["_prim_sig"] = np.sign(primary_probas - 0.5).astype(np.float32)
            return np.clip(self.meta_model.predict_proba(X_use)[:, 1], 0.0, 1.0)
        except Exception:
            return np.ones(len(X))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def _confidence_from_distribution(mean_pred: np.ndarray, std_pred: np.ndarray) -> np.ndarray:
    # For a binary classifier: confidence = distance from the 0.5 decision boundary,
    # discounted by ensemble disagreement (std_pred).
    # Returns values in [0, 1].
    distance = np.abs(mean_pred - 0.5)           # 0 = uncertain, 0.5 = fully confident
    uncertainty_penalty = np.clip(std_pred / 0.5, 0, 1)
    return np.clip(2.0 * distance * (1.0 - uncertainty_penalty), 0.0, 1.0)


def _time_based_purged_folds(
    index: pd.DatetimeIndex,
    train_months: int = 6,
    test_months: int = 1,
    embargo_months: int = 1,
    stride_months: int = 0,          # 0 = default to test_months (original behaviour)
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("features_df index must be a DatetimeIndex")

    idx = index.sort_values().unique()
    if len(idx) == 0:
        return []

    _stride = stride_months if stride_months > 0 else test_months

    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = idx.min()
    hard_end = idx.max()

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.DateOffset(months=embargo_months) + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_start > hard_end or test_end > hard_end:
            break

        folds.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=_stride)

    return folds


def _recency_weights(n: int, halflife: int = 63) -> np.ndarray:
    """Exponential decay giving more weight to recent bars.
    Caps the max/min weight ratio at 100:1 to keep weights numerically stable
    and avoid zero-count nodes in tree learners (especially LightGBM GPU)."""
    idx = np.arange(n, dtype=np.float64)
    # Compute raw exp weights
    exp_arg = idx / max(halflife, 1)
    # Clamp so the oldest bar gets at least 1% of the newest bar's weight
    # This caps the ratio at e^{4.6} ≈ 100:1
    exp_arg = np.clip(exp_arg, exp_arg[-1] - 4.6, None)
    w = np.exp(exp_arg - exp_arg[-1])   # shift so newest = 1.0
    w_mean = w.mean()
    if not np.isfinite(w_mean) or w_mean < 1e-30:
        return np.ones(n, dtype=np.float32)
    return (w / w_mean).astype(np.float32)


def _should_use_gpu(n_rows: int, params: dict[str, Any]) -> bool:
    """Return True when GPU acceleration is expected to give a net speedup."""
    gpu_setting = params.get("use_gpu", "auto")
    if gpu_setting is True or str(gpu_setting).lower() == "true":
        return True
    if gpu_setting is False or str(gpu_setting).lower() == "false":
        return False
    # auto: enable only when dataset is large enough to overcome transfer overhead
    gpu_min = int(params.get("gpu_min_rows", 20_000))
    return n_rows >= gpu_min


def _train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    seed: int,
    regime_weights: Optional[np.ndarray] = None,
):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    n_estimators = params.get("n_estimators", 400)
    early_stop   = params.get("early_stopping_rounds", 30)

    # XGBoost: use CPU hist (CUDA requires NVIDIA; AMD GPU uses LightGBM OpenCL instead)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.03),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",          # histogram bins — 2-3x faster than exact
        device="cpu",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=early_stop,
    )
    sw = _recency_weights(len(X_train), halflife=params.get("recency_halflife", 63))
    if regime_weights is not None and len(regime_weights) == len(sw):
        sw = sw * regime_weights
        sw = sw / sw.mean()

    # Use last 15% of training data as internal validation for early stopping
    _n_val = max(20, int(len(X_train) * 0.15))
    _X_es, _y_es = X_train.iloc[-_n_val:], y_train.iloc[-_n_val:]
    _X_fit, _y_fit = X_train.iloc[:-_n_val], y_train.iloc[:-_n_val]
    _sw_fit = sw[:-_n_val] if len(sw) > _n_val else sw

    model.fit(
        _X_fit, _y_fit,
        sample_weight=_sw_fit,
        eval_set=[(_X_es, _y_es)],
        verbose=False,
    )
    return model


def _train_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    seed: int,
):
    """LightGBM member — leaf-wise growth captures weak signals XGBoost misses."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None   # graceful degradation: skip if not installed

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / max(n_pos, 1)
    use_gpu = _should_use_gpu(len(X_train), params)
    early_stop = int(params.get("early_stopping_rounds", 30))

    model = lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=max(8, min(127, 2 ** params.get("max_depth", 4) - 1)),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_child_samples=20,
        scale_pos_weight=spw,
        objective="binary",
        device="gpu" if use_gpu else "cpu",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    sw = _recency_weights(len(X_train), halflife=params.get("recency_halflife", 63))
    # Use last 15% as internal validation set for early stopping
    _n_val = max(20, int(len(X_train) * 0.15))
    _X_es, _y_es = X_train.iloc[-_n_val:], y_train.iloc[-_n_val:]
    _X_fit, _y_fit = X_train.iloc[:-_n_val], y_train.iloc[:-_n_val]
    _sw_fit = sw[:-_n_val] if len(sw) > _n_val else sw
    callbacks = [lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                 lgb.log_evaluation(period=-1)]
    model.fit(
        _X_fit, _y_fit,
        sample_weight=_sw_fit,
        eval_set=[(_X_es, _y_es)],
        callbacks=callbacks,
    )
    return model


def _train_catboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
):
    """CatBoost member — symmetric tree growth, native cat support, strong regularisation."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None   # graceful degradation

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    model = CatBoostClassifier(
        iterations=params.get("n_estimators", 400),
        depth=min(params.get("max_depth", 4), 6),
        learning_rate=params.get("learning_rate", 0.03),
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=0.5,
        border_count=128,
        scale_pos_weight=spw,
        eval_metric="AUC",
        random_seed=seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )
    sw = sample_weight if sample_weight is not None else _recency_weights(len(X_train), halflife=params.get("recency_halflife", 63))
    model.fit(X_train, y_train, sample_weight=sw)
    return model


def _train_extratrees_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
    seed: int,
):
    """ExtraTreesClassifier — maximum randomisation for decorrelated ensemble diversity."""
    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError:
        return None
    sw = _recency_weights(len(X_train), halflife=params.get("recency_halflife", 63))
    model = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=params.get("max_depth", 5),
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sw)
    return model


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    directional_acc = float((np.sign(y_true.values) == np.sign(y_pred)).mean())
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_acc,
    }


def _signal_returns(y_true: pd.Series, signal: pd.Series) -> pd.Series:
    aligned = y_true.reindex(signal.index).fillna(0.0)
    return aligned * signal.astype(float)


def _strategy_metrics(signal_returns: pd.Series) -> dict[str, float]:
    if signal_returns.empty:
        return {"mean_return": 0.0, "sharpe_like": 0.0, "hit_rate": 0.0}
    vol = signal_returns.std()
    sharpe_like = float((signal_returns.mean() / (vol + 1e-12)) * np.sqrt(252))
    hit_rate = float((signal_returns > 0).mean())
    return {
        "mean_return": float(signal_returns.mean()),
        "sharpe_like": sharpe_like,
        "hit_rate": hit_rate,
    }


def _compute_shap_importance(model: Any, X: pd.DataFrame) -> dict[str, float]:
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed; logging model feature_importances_ as fallback")
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_)
            return {c: float(v) for c, v in zip(X.columns, fi)}
        return {}

    sample_n = min(len(X), 1000)
    X_sample = X.iloc[-sample_n:].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return {
        col: float(val)
        for col, val in sorted(
            zip(X_sample.columns, mean_abs),
            key=lambda kv: kv[1],
            reverse=True,
        )
    }


def train_model(
    features_df: pd.DataFrame,
    target_horizon: int = 5,
    config: Optional[dict] = None,
    confidence_threshold: float = 0.65,
    ticker: str = "",
) -> dict[str, Any]:
    """
    Train XGBoost ensemble with purged walk-forward CV and uncertainty gating.
    """
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    artifacts_dir = ensure_dir(model_cfg.get("artifacts_dir", "models/artifacts"))

    df = features_df.sort_index().copy()
    if "fwd_return" not in df.columns:
        if "close" not in df.columns:
            raise ValueError("features_df must contain either 'fwd_return' or 'close'")
        df["fwd_return"] = df["close"].pct_change(target_horizon).shift(-target_horizon)

    # Prefer triple-barrier label when available (richer signal — accounts for
    # stop-loss and take-profit hits rather than raw direction only).
    # Fall back to simple binary fwd_return label when tb_label is absent.
    if "tb_label" in df.columns and df["tb_label"].notna().mean() > 0.5:
        _target_col = "tb_label"
        logger.info("Training target: tb_label (triple-barrier) — richer signal quality")
    else:
        _target_col = None   # will derive from fwd_return below
        logger.info("Training target: fwd_return > 0 (simple binary)")

    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns available for training")

    # --- Walk-forward feature selection: rank by temporal stability across 3 blocks ---
    # The single-window approach (first 70%) produces bull-biased feature sets because
    # 2018-2021 is bull-dominated. Features selected only from that window can fail
    # catastrophically in 2022 bear market. Using median MI across 3 temporal blocks
    # ensures selected features work in both bull AND bear regimes.
    _max_features = int(model_cfg.get("max_train_features", 80))
    if len(feature_cols) > _max_features:
        try:
            from features.engine import FeatureSelector
            from sklearn.feature_selection import mutual_info_classif
            import warnings

            _n_blocks = 3
            _block_size = len(df) // _n_blocks
            _mi_blocks: list[np.ndarray] = []
            _target_series = df["fwd_return"] if _target_col is None else df[_target_col]

            for _bi in range(_n_blocks):
                _bs = _bi * _block_size
                _be = _bs + _block_size if _bi < _n_blocks - 1 else len(df)
                _Xb = df[feature_cols].iloc[_bs:_be].replace([np.inf, -np.inf], np.nan).fillna(0)
                _yb_raw = _target_series.iloc[_bs:_be]
                _yb = (_yb_raw > 0).astype(int).fillna(0)
                if _yb.sum() < 10 or len(_yb) < 60:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _mi_blocks.append(
                        mutual_info_classif(_Xb, _yb, random_state=42)
                    )

            if len(_mi_blocks) >= 2:
                _mi_arr = np.array(_mi_blocks)
                _mi_median = np.median(_mi_arr, axis=0)
                _mi_std = _mi_arr.std(axis=0)
                # Stability weight: penalise high variance across blocks
                _mi_cv = _mi_std / (_mi_median + 1e-8)
                _stability_weight = 1.0 / (1.0 + _mi_cv)
                _mi_stable = _mi_median * _stability_weight
                _mi_series = pd.Series(_mi_stable, index=feature_cols)
                logger.info(
                    "Walk-forward MI: %d blocks, feature stability scores computed "
                    "(top-3 stable: %s)",
                    len(_mi_blocks),
                    ", ".join(_mi_series.nlargest(3).index.tolist()),
                )
                # Build a proxy target from all blocks for FeatureSelector.
                # Subsample to 150k rows (float32) to avoid OOM on large pooled datasets.
                _yfs_all = (_target_series > 0).astype(int).fillna(0)
                _fs_max_rows = 150_000
                if len(df) > _fs_max_rows:
                    _fs_idx = np.random.default_rng(42).choice(len(df), _fs_max_rows, replace=False)
                    _fs_idx.sort()
                    _Xfs_all = _clean_to_float32(df[feature_cols].iloc[_fs_idx])
                    _yfs_all_fit = _yfs_all.iloc[_fs_idx]
                else:
                    _Xfs_all = _clean_to_float32(df[feature_cols])
                    _yfs_all_fit = _yfs_all
                # Inject pre-computed stable MI scores into FeatureSelector via fit override
                _fs = FeatureSelector(n_features=_max_features, max_correlation=0.80, use_shap=False)
                _fs.fit(_Xfs_all, _yfs_all_fit)
                # Re-rank by stable MI (overrides FeatureSelector's single-window MI)
                _avail = [c for c in _mi_series.index if c in _fs.importance_.index]
                _stable_rank = _mi_series.loc[_avail].rank(ascending=True) / len(_avail)
                _fs_rank = _fs.importance_.loc[_avail].rank(ascending=True) / len(_avail)
                _combined = 0.5 * _stable_rank + 0.5 * _fs_rank
                _sorted_feats = _combined.sort_values(ascending=False).index.tolist()
                # Apply correlation filter manually from FeatureSelector's corr matrix
                _corr = _Xfs_all[_sorted_feats].corr().abs()
                _selected: list[str] = []
                for _f in _sorted_feats:
                    if len(_selected) >= _max_features:
                        break
                    if not _selected:
                        _selected.append(_f)
                        continue
                    _max_corr = _corr.loc[_f, _selected].max()
                    if _max_corr < 0.80:
                        _selected.append(_f)
                feature_cols = _selected[:_max_features]
            else:
                # Fallback: single-window selection (fewer than 2 valid blocks).
                # Cap to 150k rows (float32) to avoid OOM on large pooled datasets.
                _fs_split = min(int(len(df) * 0.70), 150_000)
                _Xfs = _clean_to_float32(df[feature_cols].iloc[:_fs_split])
                _yfs_raw = _target_series.iloc[:_fs_split]
                _yfs = (_yfs_raw > 0).astype(int).fillna(0)
                _fs = FeatureSelector(n_features=_max_features, max_correlation=0.80, use_shap=False)
                _fs.fit(_Xfs, _yfs)
                feature_cols = _fs.selected_features_[:_max_features]

            logger.info("FeatureSelector: reduced %d → %d features (max_train_features=%d)",
                        len(_get_feature_cols(df)), len(feature_cols), _max_features)
        except Exception as _fse:
            logger.warning("Feature selection failed, using all features: %s", _fse)

    # Build mask: require all features AND the target to be non-NaN and finite.
    # Use numpy isfinite() to avoid the (n_cols × n_rows) bool mask that OOMs on large datasets.
    _label_mask = df["fwd_return"].notna() if _target_col is None else df[_target_col].notna()
    _feat_finite = _finite_row_mask(df[feature_cols])
    mask = _feat_finite & _label_mask
    df = df.loc[mask].copy()
    # Replace any residual inf/NaN in features with 0 (defensive)
    df[feature_cols] = _clean_to_float32(df[feature_cols])
    if len(df) < 252:
        raise ValueError("Not enough rows after filtering; need at least ~252 bars")

    # Fast mode: fewer folds, fewer ensemble members, smaller trees
    # Activated via config fast_mode: true OR env var ALPHAFORGE_FAST_MODE=1
    import os as _os
    _fast = bool(model_cfg.get("fast_mode", False)) or _os.environ.get("ALPHAFORGE_FAST_MODE", "0") == "1"

    # CV fold config: larger windows = fewer folds = faster training
    # Default (standard): 18-month train / 3-month test → ~14 folds on 6yr data
    # Fast mode: 24-month train / 6-month test → ~6 folds (overrides config values)
    # Legacy (old default 6/1): ~64 folds — DO NOT USE for bulk training
    if _fast:
        _cv_train = 24
        _cv_test  = 6
    else:
        _cv_train  = int(model_cfg.get("cv_train_months",  18))
        _cv_test   = int(model_cfg.get("cv_test_months",    3))
    _cv_stride = int(model_cfg.get("cv_stride_months", 0))   # 0 = default to test_months
    folds = _time_based_purged_folds(df.index, train_months=_cv_train,
                                     test_months=_cv_test, embargo_months=1,
                                     stride_months=_cv_stride)
    if not folds:
        raise ValueError(f"Unable to build walk-forward folds ({_cv_train}m train/{_cv_test}m test)")

    # Fast mode overrides config: smaller ensemble + fewer trees = 10-15x faster
    if _fast:
        ensemble_size = 2
        _n_est = 200
    else:
        ensemble_size = int(model_cfg.get("ensemble_size", 5))
        _n_est = int(model_cfg.get("n_estimators", 500))
    base_seed = int(model_cfg.get("random_state", 42))
    train_params = {
        "n_estimators": _n_est,
        "max_depth": model_cfg.get("max_depth", 4),
        "learning_rate": model_cfg.get("learning_rate", 0.03),
        "subsample": model_cfg.get("subsample", 0.8),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "reg_alpha": model_cfg.get("reg_alpha", 0.1),
        "reg_lambda": model_cfg.get("reg_lambda", 1.0),
        "early_stopping_rounds": int(model_cfg.get("early_stopping_rounds", 30)),
        "ensemble_size": ensemble_size,
        # GPU settings — auto-enable on large datasets (≥ gpu_min_rows rows)
        "use_gpu":      model_cfg.get("use_gpu", "auto"),
        "gpu_min_rows": int(model_cfg.get("gpu_min_rows", 20_000)),
    }
    logger.info(
        "Training config: fast_mode=%s | folds=%d | ensemble=%d | n_est=%d | cv=%dm/%dm",
        _fast, len(folds), ensemble_size, _n_est, _cv_train, _cv_test,
    )

    fold_results: list[dict[str, Any]] = []
    baseline_all_returns: list[pd.Series] = []
    ml_all_returns: list[pd.Series] = []
    # Collect OOS predictions and true labels for post-training calibration
    _calib_preds:  list[float] = []
    _calib_labels: list[int]   = []
    # Collect OOS data for meta-labeling (secondary bet-sizing classifier)
    _meta_X_list: list[pd.DataFrame] = []
    _meta_y_list: list[pd.Series]    = []
    # Per-member OOS Sharpe accumulator for Sharpe-weighted blending
    # Keys: member index (XGB 0..N-1, then RF, LightGBM, ExtraTrees)
    _member_sharpe_acc: list[list[float]] = []   # _member_sharpe_acc[fold] = [s0, s1, ...]

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        if train_mask.sum() < 60 or test_mask.sum() < 10:
            continue

        X_train = _clean_to_float32(df.loc[train_mask, feature_cols])
        X_test  = _clean_to_float32(df.loc[test_mask,  feature_cols])
        y_test  = df.loc[test_mask,  "fwd_return"]   # OOS evaluation always on raw returns

        # Build classification target for fitting: triple-barrier when available,
        # else binarise raw return (positive → 1, non-positive → 0).
        # For tb_label, exclude timeout rows (label==0) — they are ambiguous
        # and dilute the model's ability to distinguish long vs short signals.
        if _target_col is not None:
            _tb_conf = df.loc[train_mask, _target_col] != 0  # confirmed signals only
            X_train = X_train.loc[_tb_conf]
            y_train_fit = df.loc[train_mask, _target_col][_tb_conf].map({1: 1, -1: 0}).fillna(0)
        else:
            y_train_fit = (df.loc[train_mask, "fwd_return"] > 0).astype(int)

        # Sample weighting: regime-based only.
        # bear(0)=2.0, sideways(1)=1.0, bull(2)=2.0, high_vol(3)=1.5
        _reg_weights = None
        if len(X_train) > 0:
            if "regime" in df.columns:
                _rv = df.loc[X_train.index, "regime"].values
                _reg_weights = np.select(
                    [_rv == 2, _rv == 0, _rv == 3],
                    [2.0,       2.0,     1.5],
                    default=1.0,
                ).astype(np.float32)

        # Train all ensemble members concurrently:
        # - XGB members use CPU (n_jobs throttled so N parallel instances share cores)
        # - LightGBM uses GPU → runs simultaneously with XGB at no CPU cost
        # - RF uses CPU (fast enough to overlap with GPU work)
        import concurrent.futures as _cf
        import os as _os

        _n_cpu = max(1, (_os.cpu_count() or 4) // max(ensemble_size, 1))
        _xgb_params = {**train_params, "n_jobs": _n_cpu}

        def _xgb_task(i: int):
            return _train_xgb_classifier(
                X_train, y_train_fit, _xgb_params,
                seed=base_seed + i + fold_idx * 17,
                regime_weights=_reg_weights,
            )

        def _rf_task():
            try:
                from sklearn.ensemble import RandomForestClassifier
                _sw = _recency_weights(len(X_train))
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=6, min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=base_seed + fold_idx, n_jobs=_n_cpu,
                )
                rf.fit(X_train, y_train_fit, sample_weight=_sw)
                return rf
            except Exception as _e:
                logger.debug("RF member skipped: %s", _e)
                return None

        def _lgbm_task():
            return _train_lgbm_classifier(
                X_train, y_train_fit, train_params,
                seed=base_seed + fold_idx * 31,
            )

        # Submit all tasks; threads share address space — XGBoost/LightGBM/sklearn
        # release the GIL during their native C training routines.
        # For large folds (>100k rows), cap concurrency at 2 to avoid OOM — each
        # concurrent job holds its own X_train copy in memory (float32 = ~200MB/100k×86).
        def _et_task():
            return _train_extratrees_classifier(
                X_train, y_train_fit, train_params, seed=base_seed + fold_idx * 7,
            )

        _tasks: list[tuple[str, Any]] = [(f"xgb_{i}", _xgb_task, (i,)) for i in range(ensemble_size)]
        if not _fast:
            _tasks.append(("rf", _rf_task, ()))
            _tasks.append(("et", _et_task, ()))
        _tasks.append(("lgbm", _lgbm_task, ()))

        _large_fold = len(X_train) > 100_000
        _n_workers = 2 if _large_fold else len(_tasks)
        fold_members: list[Any] = []
        with _cf.ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _futures = {_pool.submit(fn, *args): name for name, fn, args in _tasks}
            for _fut in _cf.as_completed(_futures):
                try:
                    _m = _fut.result()
                    if _m is not None:
                        fold_members.append(_m)
                except Exception as _fe:
                    logger.warning("Fold %d member training failed (%s): %s", fold_idx + 1, _futures[_fut], _fe)

        if not fold_members:
            logger.warning(
                "Fold %d: all %d training tasks failed — skipping fold. "
                "If this repeats, reduce ensemble_size or set fast_mode: true in config.",
                fold_idx + 1, len(_tasks),
            )
            continue

        fold_model = TrainedEnsembleModel(
            models=fold_members,
            feature_columns=feature_cols,
            target_horizon=target_horizon,
            confidence_threshold=confidence_threshold,
            train_params=train_params,
            trained_at=datetime.now(timezone.utc).isoformat(),
        )
        pred, pred_std, _, _ = fold_model.predict(X_test)

        # Track per-member OOS Sharpe for Sharpe-weighted blending
        _fold_member_sharpes: list[float] = []
        X_test_use = X_test.reindex(columns=feature_cols, fill_value=0.0)
        for _m in fold_members:
            try:
                if hasattr(_m, "predict_proba"):
                    _mp = _m.predict_proba(X_test_use)[:, 1]
                else:
                    _mp = _m.predict(X_test_use)
                _msig = pd.Series(np.sign(_mp - 0.5).astype(int), index=X_test.index)
                _mret = _signal_returns(y_test, _msig)
                _mv   = _mret.std()
                _ms   = float((_mret.mean() / (_mv + 1e-12)) * np.sqrt(252)) if len(_mret) > 0 else 0.0
            except Exception:
                _ms = 0.0
            _fold_member_sharpes.append(_ms)
        _member_sharpe_acc.append(_fold_member_sharpes)
        conf = _confidence_from_distribution(pred, pred_std)
        ml_signal = pd.Series(
            np.where(conf >= confidence_threshold, np.sign(pred - 0.5), 0).astype(int),
            index=X_test.index,
        )

        baseline_signal = baseline_momentum_signals(df.loc[:test_end, "close"]).reindex(X_test.index).fillna(0).astype(int)
        ml_ret = _signal_returns(y_test, ml_signal)
        baseline_ret = _signal_returns(y_test, baseline_signal)
        ml_all_returns.append(ml_ret)
        baseline_all_returns.append(baseline_ret)

        # Collect OOS raw ensemble scores for isotonic calibration
        _true_bin = (y_test > 0).astype(int)
        _calib_preds.extend(pred.tolist())
        _calib_labels.extend(_true_bin.tolist())

        # Collect OOS data for meta-labeling — build secondary training set
        _prim_sig_series = pd.Series(np.sign(pred - 0.5).astype(int), index=X_test.index)
        if _target_col is not None and _target_col in df.columns:
            _tb_for_meta = df.loc[X_test.index, _target_col]
            _meta_lbl = _meta_label_fn(_prim_sig_series, _tb_for_meta)
        else:
            # Fall back to raw return direction when tb_label is absent
            _fwd_dir = pd.Series(np.sign(y_test.values), index=y_test.index)
            _meta_lbl = ((_prim_sig_series > 0) & (_fwd_dir > 0)) | ((_prim_sig_series < 0) & (_fwd_dir < 0))
            _meta_lbl = _meta_lbl.astype(int).rename("meta_label")
        # Only include bars with non-trivial forward moves (filter near-zero returns)
        _nontrivial = y_test.abs() > y_test.abs().median() * 0.25
        _X_meta_fold = X_test.loc[_nontrivial].copy()
        _X_meta_fold["_prim_sig"] = _prim_sig_series[_nontrivial]
        _meta_X_list.append(_X_meta_fold)
        _meta_y_list.append(_meta_lbl[_nontrivial])

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "model_metrics": _regression_metrics(y_test, pred),
                "ml_strategy_metrics": _strategy_metrics(ml_ret),
                "baseline_strategy_metrics": _strategy_metrics(baseline_ret),
                "avg_confidence": float(np.mean(conf)),
                "trade_rate": float((ml_signal != 0).mean()),
            }
        )

    if not fold_results:
        raise ValueError("No valid folds were generated with enough train/test samples")

    # ── Meta-labeling: secondary classifier for bet-size scaling ──
    # Trained on OOS predictions so it learns when the primary model is reliable.
    _meta_model = None
    _n_meta = sum(len(x) for x in _meta_X_list)
    if _meta_X_list and _n_meta >= 80:
        try:
            _X_meta_all = pd.concat(_meta_X_list)
            _y_meta_all = pd.concat(_meta_y_list).astype(int).reindex(_X_meta_all.index).fillna(0)
            _meta_params = dict(train_params)
            _meta_params.update({"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05})
            _meta_model = _train_xgb_classifier(_X_meta_all, _y_meta_all, _meta_params, seed=base_seed + 77777)
            logger.info(
                "Meta-classifier trained on %d OOS samples (%d folds); "
                "positive-rate=%.1f%%",
                _n_meta, len(_meta_X_list), float(_y_meta_all.mean()) * 100,
            )
        except Exception as _me:
            logger.warning("Meta-classifier training failed (non-fatal): %s", _me)

    # Fit final ensemble on full data available.
    # Final ensemble: same filtering as per-fold — exclude timeout rows for tb_label
    if _target_col is not None:
        _full_conf = df[_target_col] != 0
        X_full = _clean_to_float32(df.loc[_full_conf, feature_cols])
        y_full_fit = df.loc[_full_conf, _target_col].map({1: 1, -1: 0}).fillna(0)
        y_full = df.loc[_full_conf, "fwd_return"]
    else:
        X_full = _clean_to_float32(df[feature_cols])
        y_full_fit = (df["fwd_return"] > 0).astype(int)
        y_full = df["fwd_return"]

    # Regime weights for final ensemble (bear+bull boosted equally)
    _reg_weights_full = None
    if "regime" in df.columns and len(X_full) > 0:
        _rvf = df.loc[X_full.index, "regime"].values
        _rw_f = np.select(
            [_rvf == 2, _rvf == 0, _rvf == 3],
            [2.0,       2.0,       1.5],
            default=1.0,
        ).astype(np.float32)
        if _rw_f.sum() > 0:
            _reg_weights_full = _rw_f

    final_models: list[Any] = [
        _train_xgb_classifier(X_full, y_full_fit, train_params, seed=base_seed + 10_000 + i,
                               regime_weights=_reg_weights_full)
        for i in range(ensemble_size)
    ]
    # Add RF + LightGBM diversity members only in standard mode (slow; skip in fast mode)
    if not _fast:
        try:
            from sklearn.ensemble import RandomForestClassifier
            _sw_rf_final = _recency_weights(len(X_full))
            _rf_final = RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_leaf=5,
                class_weight="balanced",
                random_state=base_seed, n_jobs=-1,
            )
            _rf_final.fit(X_full, y_full_fit, sample_weight=_sw_rf_final)
            final_models.append(_rf_final)
            logger.info("RandomForest member added to final ensemble (%d total models)", len(final_models))
        except Exception as _rfe:
            logger.debug("RF final member skipped: %s", _rfe)

    # Add LightGBM to final ensemble (conservative: matches XGB depth/lr)
    _lgb_final = _train_lgbm_classifier(X_full, y_full_fit, train_params, seed=base_seed + 20_000)
    if _lgb_final is not None:
        final_models.append(_lgb_final)
        logger.info("LightGBM member added to final ensemble (%d total models)", len(final_models))

    # Add second LightGBM with deeper leaves for diversity (standard mode only)
    if not _fast:
        try:
            import lightgbm as _lgb2_mod
            _n_neg2 = int((y_full_fit == 0).sum())
            _n_pos2 = int((y_full_fit == 1).sum())
            _lgb2 = _lgb2_mod.LGBMClassifier(
                n_estimators=int(train_params.get("n_estimators", 500) * 0.6),
                max_depth=6,
                learning_rate=train_params.get("learning_rate", 0.03) * 2,
                num_leaves=31,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_samples=30,
                scale_pos_weight=_n_neg2 / max(_n_pos2, 1),
                objective="binary",
                random_state=base_seed + 25_000,
                n_jobs=-1,
                verbose=-1,
            )
            _sw_lgb2 = _recency_weights(len(X_full), halflife=42)
            _lgb2.fit(X_full, y_full_fit, sample_weight=_sw_lgb2)
            final_models.append(_lgb2)
            logger.info("LightGBM-v2 (deeper/wider) member added (%d total models)", len(final_models))
        except Exception as _lgb2e:
            logger.debug("LightGBM-v2 skipped: %s", _lgb2e)

    # --- Sharpe-weighted ensemble blending ---
    # Average per-member OOS Sharpe across folds. Use softmax to turn Sharpe scores
    # into positive normalised weights so every member contributes but better
    # performers get proportionally more influence.
    _model_weights: list[float] | None = None
    if _member_sharpe_acc:
        # Pad shorter fold rows to max width (some folds may have fewer members)
        _max_members = max(len(r) for r in _member_sharpe_acc)
        _sharpe_matrix = np.zeros((_max_members,), dtype=np.float64)
        _sharpe_counts = np.zeros((_max_members,), dtype=np.float64)
        for _row in _member_sharpe_acc:
            for _mi, _sv in enumerate(_row):
                _sharpe_matrix[_mi] += _sv
                _sharpe_counts[_mi] += 1
        _avg_sharpes = np.where(_sharpe_counts > 0, _sharpe_matrix / _sharpe_counts, 0.0)
        # Trim/pad to match final_models count
        _n_final = len(final_models)
        if len(_avg_sharpes) >= _n_final:
            _avg_sharpes = _avg_sharpes[:_n_final]
        else:
            _avg_sharpes = np.pad(_avg_sharpes, (0, _n_final - len(_avg_sharpes)))
        # Softmax with temperature=1 on Sharpe values → bounded positive weights
        _shifted = _avg_sharpes - _avg_sharpes.max()
        _exp_s = np.exp(np.clip(_shifted, -10, 0))
        _model_weights = (_exp_s / _exp_s.sum()).tolist()
        logger.info(
            "Sharpe-weighted blending: avg fold Sharpes=%s  weights=%s",
            np.round(_avg_sharpes, 3).tolist(),
            np.round(_model_weights, 3).tolist(),
        )

    # --- OOS isotonic calibration ---
    # Fit an IsotonicRegression mapping raw ensemble mean scores → true probabilities,
    # trained on all held-out fold predictions.  This removes systematic bias in the
    # raw ensemble output (e.g., XGBoost tends to be overconfident near the tails).
    # NOTE: When calibration maps too many bars to ~0.50, disable by setting _calibrator=None.
    _calibrator = None
    _fit_calibrator = True   # re-enabled — calibration improves Sharpe
    if _fit_calibrator and len(_calib_preds) >= 50:
        try:
            from sklearn.isotonic import IsotonicRegression
            _cal_X = np.array(_calib_preds, dtype=np.float64)
            _cal_y = np.array(_calib_labels, dtype=np.float64)
            _calibrator = IsotonicRegression(out_of_bounds="clip")
            _calibrator.fit(_cal_X, _cal_y)
            logger.info(
                "Isotonic calibrator fitted on %d OOS samples across %d folds",
                len(_cal_X), len(fold_results),
            )
        except Exception as _ce:
            logger.debug("Isotonic calibration skipped: %s", _ce)

    trained = TrainedEnsembleModel(
        models=final_models,
        feature_columns=feature_cols,
        target_horizon=target_horizon,
        confidence_threshold=confidence_threshold,
        train_params=train_params,
        trained_at=datetime.now(timezone.utc).isoformat(),
        calibrator=_calibrator,
        meta_model=_meta_model,
        model_weights=_model_weights,   # Sharpe-weighted blend; None falls back to equal weights
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = artifacts_dir / f"xgb_return_model_{ts}.joblib"
    latest_path = artifacts_dir / "latest_model.joblib"
    joblib.dump(trained, model_path)
    joblib.dump(trained, latest_path)
    # Also overwrite the ticker-specific path so ModelTrainer.load() picks it up
    if ticker:
        joblib.dump(trained, artifacts_dir / f"{ticker.lower()}_model.joblib")

    # Fit and save a FeatureScaler on the full training data so inference
    # uses the exact same distribution normalization as training.
    try:
        from features.scaler import FeatureScaler
        regime_col = "regime" if "regime" in df.columns else None
        regimes_s  = df[regime_col].map({1: "bull", -1: "bear", 0: "sideways"}) if regime_col else None
        _scaler = FeatureScaler(regime_aware=True)
        _scaler.fit(df[feature_cols], regimes=regimes_s)
        _scaler.save(str(artifacts_dir / "latest_scaler.joblib"))
        joblib.dump(_scaler, artifacts_dir / f"scaler_{ts}.joblib")
        # Save per-ticker scaler so universe_portfolio.py can load it alongside the model
        if ticker:
            joblib.dump(_scaler, artifacts_dir / f"{ticker.lower()}_scaler.joblib")
        logger.info("FeatureScaler saved alongside model")
    except Exception as _se:
        logger.warning("FeatureScaler save failed (non-fatal): %s", _se)

    shap_importance = _compute_shap_importance(final_models[0], X_full)
    ml_concat   = pd.concat(ml_all_returns)   if ml_all_returns   else pd.Series(dtype=float)
    base_concat = pd.concat(baseline_all_returns) if baseline_all_returns else pd.Series(dtype=float)

    # ── Robustness gates (anti-overfitting, anti-promotion of lucky strategies) ──
    robustness_report: dict = {}
    try:
        from validation.robustness import RobustnessGates
        rg_cfg     = cfg.get("robustness_gates", {})
        gates      = RobustnessGates(**rg_cfg)
        oos_sharpes = [
            f["ml_strategy_metrics"].get("sharpe_like", 0.0)
            for f in fold_results if "ml_strategy_metrics" in f
        ]
        rg_result = gates.evaluate(
            ml_concat,
            is_sharpe=float(np.mean([f.get("model_metrics", {}).get("directional_accuracy", 0.5)
                                     for f in fold_results])),
            oos_sharpe_list=oos_sharpes,
        )
        robustness_report = rg_result.to_dict()
        logger.info(
            "RobustnessGates: %d/8 passed (overall=%s)  R²=%.2f  pos_folds=%.0f%%",
            rg_result.n_passed, rg_result.overall_pass,
            rg_result.equity_r2, rg_result.positive_folds_frac * 100,
        )
        if not rg_result.overall_pass:
            logger.warning(
                "RobustnessGates FAILED — model may be overfitting. "
                "Consider running research-v2 to find a more robust strategy."
            )
    except Exception as _rge:
        logger.debug("RobustnessGates skipped (non-fatal): %s", _rge)
    comparison = {
        "ml": _strategy_metrics(ml_concat),
        "baseline_momentum": _strategy_metrics(base_concat),
        "improvement_mean_return": float(ml_concat.mean() - base_concat.mean()) if len(ml_concat) else 0.0,
    }

    report = {
        "trained_at":          datetime.now(timezone.utc).isoformat(),
        "target_horizon":      target_horizon,
        "target_label":        _target_col or "fwd_return>0",
        "confidence_threshold": confidence_threshold,
        "feature_count":       len(feature_cols),
        "n_ensemble_models":   len(final_models),
        "hyperparameters":     train_params,
        "fold_results":        fold_results,
        "comparison":          comparison,
        "robustness_gates":    robustness_report,
        "feature_importance_shap": shap_importance,
        "model_artifact":      str(model_path),
    }
    metrics_path = artifacts_dir / f"training_metrics_{ts}.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved model artifact to %s", model_path)
    logger.info("Saved training report to %s", metrics_path)

    return {
        "model": trained,
        "model_path": str(model_path),
        "latest_model_path": str(latest_path),
        "metrics_path": str(metrics_path),
        "report": report,
    }


def generate_signals(
    model: TrainedEnsembleModel,
    latest_features: pd.DataFrame | pd.Series,
    confidence_threshold: Optional[float] = None,
) -> dict[str, float] | pd.DataFrame:
    """
    Generate prediction + confidence + trading signal (-1, 0, +1).
    """
    threshold = model.confidence_threshold if confidence_threshold is None else confidence_threshold
    if isinstance(latest_features, pd.Series):
        X = latest_features.to_frame().T
        single_row = True
    else:
        X = latest_features.copy()
        single_row = len(X) == 1

    pred, pred_std, _, _ = model.predict(X)
    conf = _confidence_from_distribution(pred, pred_std)
    signal = np.where(conf >= threshold, np.sign(pred), 0).astype(int)

    out = pd.DataFrame(
        {
            "predicted_return": pred,
            "confidence_score": conf,
            "signal": signal,
        },
        index=X.index,
    )
    if single_row:
        row = out.iloc[0]
        return {
            "predicted_return": float(row["predicted_return"]),
            "confidence_score": float(row["confidence_score"]),
            "signal": int(row["signal"]),
        }
    return out


class ModelTrainer:
    """
    Backward-compatible wrapper around the new train_model/generate_signals API.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg = config or load_config()
        self.model_cfg = self.cfg.get("model", {})
        self.artifacts_dir = ensure_dir(self.model_cfg.get("artifacts_dir", "models/artifacts"))
        self.model: Optional[TrainedEnsembleModel] = None

    def train(self, features: pd.DataFrame, ticker: str) -> dict[str, float]:
        result = train_model(
            features_df=features,
            target_horizon=int(self.model_cfg.get("target_horizon", 5)),
            config=self.cfg,
            confidence_threshold=float(self.model_cfg.get("confidence_threshold", 0.65)),
            ticker=ticker,
        )
        self.model = result["model"]
        return result["report"]["comparison"]["ml"]

    def load(self, ticker: str) -> None:
        ticker_path = self.artifacts_dir / f"{ticker.lower()}_model.joblib"
        latest_path = self.artifacts_dir / "latest_model.joblib"
        load_path   = ticker_path if ticker_path.exists() else latest_path
        self.model  = joblib.load(load_path)
        logger.info("Loaded model from %s", load_path)
        # Load companion scaler if present
        scaler_path = self.artifacts_dir / "latest_scaler.joblib"
        try:
            from features.scaler import FeatureScaler
            if scaler_path.exists():
                self._scaler = joblib.load(str(scaler_path))
                logger.info("FeatureScaler loaded from %s", scaler_path)
            else:
                self._scaler = None
        except Exception:
            self._scaler = None

    def load_from_path(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """Load a specific model file by absolute path (used by ensemble trading)."""
        self.model = joblib.load(model_path)
        logger.info("Loaded model from explicit path: %s", model_path)
        self._scaler = None
        if scaler_path:
            try:
                from features.scaler import FeatureScaler
                sp = Path(scaler_path)
                if sp.exists():
                    self._scaler = joblib.load(str(sp))
                    logger.debug("FeatureScaler loaded from %s", sp)
            except Exception as exc:
                logger.debug("Ensemble scaler load skipped: %s", exc)

    def predict_proba(self, X: pd.DataFrame, regime: Optional[str] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        X_use = X
        if getattr(self, "_scaler", None) is not None:
            try:
                X_use = self._scaler.transform(X, regime=regime)
            except Exception as _se:
                logger.debug("Scaler transform skipped: %s", _se)
        pred, std, _, upper = self.model.predict(X_use)
        # Use the ensemble's 90th-percentile prediction (upper) as the signal.
        # The isotonic calibrator collapses ~86% of mean predictions to the same
        # value (0.638), destroying cross-sectional rank ordering. The upper
        # percentile preserves dispersion across bars and represents consensus
        # strength: a high upper value means even the most conservative ensemble
        # members agree on the direction.
        return np.clip(upper, 0.0, 1.0)

    def predict_meta_proba(
        self,
        X: pd.DataFrame,
        primary_probas: np.ndarray,
        regime: Optional[str] = None,
    ) -> np.ndarray:
        """Return P(primary_signal_is_correct) for bet-size scaling.

        Returns 1.0 for all bars when no meta-model is available (no-op).
        """
        if self.model is None or self.model.meta_model is None:
            return np.ones(len(X))
        X_use = X
        if getattr(self, "_scaler", None) is not None:
            try:
                X_use = self._scaler.transform(X, regime=regime)
            except Exception:
                pass
        return self.model.predict_meta(X_use, primary_probas)

    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        out = generate_signals(self.model, X)
        if isinstance(out, dict):
            return pd.DataFrame([out], index=X.index[:1])
        return out
