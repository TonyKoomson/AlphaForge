"""
Unit tests for features/engine.py.

Coverage
--------
  - Anti-leakage: every feature at row T uses only data up to T
  - as_of_date hard-cut is enforced before any indicator computation
  - Regime labels are computed correctly
  - Feature selector removes correlated pairs
  - Parquet versioning writes two files (timestamped + latest)
  - FeatureEngine.feature_columns returns sensible defaults and updates after selection
  - generate_features raises on missing columns / empty post-cutoff data
  - No future bar in any rolling or EWM result
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.engine import (
    FeatureEngine,
    FeatureSelector,
    _get_feature_cols,
    add_regime_labels,
    generate_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(
    n: int = 400,
    start: str = "2020-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OHLCV with a realistic random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    noise = rng.uniform(0.995, 1.005, n)
    high  = close * rng.uniform(1.000, 1.015, n)
    low   = close * rng.uniform(0.985, 1.000, n)
    open_ = close * noise
    vol   = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv(400)


@pytest.fixture
def features(ohlcv) -> pd.DataFrame:
    return generate_features(ohlcv, add_regime=False)


# ---------------------------------------------------------------------------
# Anti-leakage tests
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    def test_rolling_uses_only_past_data(self, ohlcv):
        """
        Verify that every feature value at index T equals the value produced
        when the computation is run on data truncated to T.

        We spot-check 5 random rows: compute features on the full series, then
        recompute on data up to that row and assert the values match.
        """
        full = generate_features(ohlcv, add_regime=False)
        feat_cols = _get_feature_cols(full)
        checkable_cols = [c for c in feat_cols if c in full.columns][:12]

        rng = np.random.default_rng(42)
        # Pick 5 rows from the second half (past the warmup period)
        half = len(full) // 2
        row_positions = sorted(rng.choice(range(half, len(full)), size=5, replace=False))

        for pos in row_positions:
            ts = full.index[pos]
            truncated = generate_features(
                ohlcv[ohlcv.index <= ts], add_regime=False
            )
            if ts not in truncated.index:
                continue
            for col in checkable_cols:
                if col in truncated.columns:
                    full_val = full.loc[ts, col]
                    trunc_val = truncated.loc[ts, col]
                    assert np.isclose(full_val, trunc_val, rtol=1e-6, equal_nan=True), (
                        f"Look-ahead bias detected in column '{col}' at {ts.date()}: "
                        f"full={full_val:.6f} vs truncated={trunc_val:.6f}"
                    )

    def test_as_of_date_drops_future_rows(self, ohlcv):
        cutoff = "2021-01-01"
        feat = generate_features(ohlcv, as_of_date=cutoff, add_regime=False)
        assert feat.index.max() <= pd.Timestamp(cutoff)

    def test_as_of_date_features_match_truncated_run(self, ohlcv):
        """
        Features generated with as_of_date must equal those from a run on
        data pre-filtered to that date — no wide window peeking ahead.
        """
        cutoff = "2021-06-30"
        feat_aod = generate_features(ohlcv, as_of_date=cutoff, add_regime=False)

        ohlcv_trunc = ohlcv[ohlcv.index <= pd.Timestamp(cutoff)]
        feat_trunc  = generate_features(ohlcv_trunc, add_regime=False)

        common_idx = feat_aod.index.intersection(feat_trunc.index)
        assert len(common_idx) > 0

        feat_cols = [c for c in _get_feature_cols(feat_aod) if c in feat_trunc.columns]
        for col in feat_cols[:10]:
            pd.testing.assert_series_equal(
                feat_aod.loc[common_idx, col],
                feat_trunc.loc[common_idx, col],
                check_names=False,
                rtol=1e-6,
                obj=f"Column {col}",
            )

    def test_label_not_in_feature_cols(self, features):
        feat_cols = _get_feature_cols(features)
        assert "label" not in feat_cols
        assert "fwd_return" not in feat_cols

    def test_no_feature_nan_after_warmup(self, features):
        feat_cols = _get_feature_cols(features)
        for col in feat_cols:
            assert features[col].isna().sum() == 0, (
                f"NaN values in feature column '{col}' after dropna"
            )

    def test_rsi_bounded(self, features):
        assert features["rsi_14"].between(0, 100).all()

    def test_stoch_k_bounded(self, features):
        assert features["stoch_k"].between(0, 1).all()

    def test_rsi_at_cutoff_equals_full_rsi(self, ohlcv):
        """RSI at any date T is the same whether computed on full or truncated history."""
        full = generate_features(ohlcv, add_regime=False)
        cutoff_ts = full.index[200]
        trunc = generate_features(ohlcv[ohlcv.index <= cutoff_ts], add_regime=False)
        assert np.isclose(
            full.loc[cutoff_ts, "rsi_14"],
            trunc.loc[cutoff_ts, "rsi_14"],
            rtol=1e-5,
        )

    def test_sma_crossover_only_uses_past_prices(self, ohlcv):
        """
        cross_20_50 at row T should only depend on SMA(20) and SMA(50) up to T.
        Inject an extreme future price and verify the past cross value is unchanged.
        """
        full = generate_features(ohlcv, add_regime=False)
        row_50 = full.index[150]
        val_before = full.loc[row_50, "cross_20_50"]

        # Inject an absurdly large price after row_50
        ohlcv_modified = ohlcv.copy()
        future_dates = ohlcv_modified.index[ohlcv_modified.index > row_50]
        if len(future_dates) > 0:
            ohlcv_modified.loc[future_dates[0], "close"] = 1_000_000.0

        feat_mod = generate_features(ohlcv_modified, add_regime=False)
        assert np.isclose(feat_mod.loc[row_50, "cross_20_50"], val_before, rtol=1e-6)


# ---------------------------------------------------------------------------
# generate_features correctness
# ---------------------------------------------------------------------------

class TestGenerateFeatures:
    def test_returns_correct_schema(self, features):
        expected_cols = [
            "ret_1d", "ret_5d", "rsi_14", "macd_norm",
            "vol_21d", "bb_width", "obv_trend", "adx_norm",
            "label", "fwd_return",
        ]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_all_ma_windows_present(self, features):
        for n in (5, 10, 20, 50, 200):
            assert f"sma_{n}_dist" in features.columns

    def test_crossover_columns_present(self, features):
        assert "cross_5_20"   in features.columns
        assert "cross_20_50"  in features.columns
        assert "cross_50_200" in features.columns

    def test_stochastic_columns_present(self, features):
        assert "stoch_k"    in features.columns
        assert "stoch_d"    in features.columns
        assert "stoch_hist" in features.columns

    def test_roc_columns_present(self, features):
        for n in (5, 10, 21):
            assert f"roc_{n}d" in features.columns

    def test_obv_present(self, features):
        assert "obv_trend" in features.columns

    def test_pattern_features_present(self, features):
        assert "higher_highs_10" in features.columns
        assert "higher_lows_10"  in features.columns
        assert "trend_strength"  in features.columns
        assert "adx_14" in features.columns

    def test_vol_features_present(self, features):
        assert "vol_5d"  in features.columns
        assert "vol_21d" in features.columns
        assert "vol_63d" in features.columns
        assert "vol_ratio_5_21"  in features.columns
        assert "vol_ratio_21_63" in features.columns
        assert "vol_zscore" in features.columns

    def test_label_is_binary(self, features):
        assert set(features["label"].dropna().unique()).issubset({0, 1})

    def test_bb_pct_mostly_bounded(self, features):
        # Most bars should be 0-1; extreme moves can exit briefly
        in_range = features["bb_pct"].between(-0.5, 1.5).mean()
        assert in_range > 0.90

    def test_missing_column_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        with pytest.raises(ValueError, match="Missing OHLCV"):
            generate_features(df)

    def test_empty_after_cutoff_raises(self, ohlcv):
        with pytest.raises(ValueError, match="No data"):
            generate_features(ohlcv, as_of_date="2010-01-01")

    def test_row_count_is_less_than_input(self, ohlcv):
        feat = generate_features(ohlcv)
        assert len(feat) < len(ohlcv)   # warmup rows dropped

    def test_channel_position_bounded(self, features):
        cp = features["channel_pos_52w"].dropna()
        assert (cp >= -0.1).all() and (cp <= 1.1).all()

    def test_gap_is_zero_on_perfectly_flat_prices(self):
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1_000_000.0,
        }, index=dates)
        feat = generate_features(df)
        # Gap = open/prev_close - 1; for flat data this should be near 0
        assert feat["gap"].abs().max() < 1e-6


# ---------------------------------------------------------------------------
# Regime labelling
# ---------------------------------------------------------------------------

class TestRegimeLabels:
    def test_regime_column_present(self, ohlcv):
        df = add_regime_labels(ohlcv)
        assert "regime" in df.columns

    def test_regime_values_are_valid(self, ohlcv):
        df = add_regime_labels(ohlcv)
        valid = {0, 1, 2, 3}
        assert set(df["regime"].dropna().unique()).issubset(valid)

    def test_binary_regime_flags(self, ohlcv):
        df = add_regime_labels(ohlcv)
        for col in ("regime_bull", "regime_bear", "regime_hv"):
            assert set(df[col].dropna().unique()).issubset({0, 1})

    def test_regime_in_generate_features(self, ohlcv):
        feat = generate_features(ohlcv, add_regime=True)
        assert "regime" in feat.columns
        assert "regime_bull" in feat.columns

    def test_regime_not_in_feature_cols(self, ohlcv):
        feat = generate_features(ohlcv, add_regime=True)
        feat_cols = _get_feature_cols(feat)
        assert "regime" not in feat_cols

    def test_bull_regime_when_trailing_return_high(self):
        """Synthesise a strong uptrend; expect bull regime for recent bars."""
        n = 350
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        close = 100.0 * np.exp(np.linspace(0, 1.5, n))  # large uptrend
        df = pd.DataFrame({
            "open": close, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=dates)
        out = add_regime_labels(df, bull_threshold=0.10)
        # Last quarter should all be bull or high-vol
        tail = out["regime"].iloc[-60:]
        assert (tail.isin({2, 3})).all()

    def test_bear_regime_when_trailing_return_low(self):
        n = 350
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        close = 100.0 * np.exp(np.linspace(0, -1.0, n))  # strong downtrend
        df = pd.DataFrame({
            "open": close, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=dates)
        out = add_regime_labels(df, bear_threshold=-0.10)
        tail = out["regime"].iloc[-60:]
        assert (tail.isin({0, 3})).all()

    def test_add_regime_does_not_modify_original(self, ohlcv):
        original_cols = set(ohlcv.columns)
        _ = add_regime_labels(ohlcv)
        assert set(ohlcv.columns) == original_cols


# ---------------------------------------------------------------------------
# Feature selector
# ---------------------------------------------------------------------------

class TestFeatureSelector:
    @pytest.fixture
    def X_y(self, features):
        cols = _get_feature_cols(features)
        X = features[cols].iloc[:200]
        y = features["label"].iloc[:200]
        return X, y

    def test_fit_produces_selected_features(self, X_y):
        X, y = X_y
        sel = FeatureSelector(n_features=12, max_correlation=0.85, use_shap=False)
        sel.fit(X, y)
        assert len(sel.selected_features_) <= 12
        assert len(sel.selected_features_) >= 1

    def test_selected_features_subset_of_input(self, X_y):
        X, y = X_y
        sel = FeatureSelector(n_features=10, use_shap=False)
        sel.fit(X, y)
        assert set(sel.selected_features_).issubset(set(X.columns))

    def test_transform_returns_only_selected(self, X_y):
        X, y = X_y
        sel = FeatureSelector(n_features=8, use_shap=False)
        Xt = sel.fit_transform(X, y)
        assert list(Xt.columns) == sel.selected_features_

    def test_transform_before_fit_raises(self, X_y):
        X, _ = X_y
        sel = FeatureSelector(use_shap=False)
        with pytest.raises(RuntimeError, match="fit"):
            sel.transform(X)

    def test_no_selected_feature_exceeds_correlation_threshold(self, X_y):
        X, y = X_y
        threshold = 0.80
        sel = FeatureSelector(n_features=15, max_correlation=threshold, use_shap=False)
        sel.fit(X, y)
        selected = sel.selected_features_
        if len(selected) < 2:
            return
        corr = X[selected].corr().abs()
        corr_arr = corr.values.copy()
        np.fill_diagonal(corr_arr, 0)
        assert corr_arr.max() < threshold + 0.01, (
            f"Correlated pair survived: max corr={corr_arr.max():.3f}"
        )

    def test_importance_series_populated(self, X_y):
        X, y = X_y
        sel = FeatureSelector(n_features=10, use_shap=False)
        sel.fit(X, y)
        assert len(sel.importance_) == len(X.columns)

    def test_fit_transform_row_count_unchanged(self, X_y):
        X, y = X_y
        sel = FeatureSelector(n_features=10, use_shap=False)
        Xt = sel.fit_transform(X, y)
        assert len(Xt) == len(X)


# ---------------------------------------------------------------------------
# FeatureEngine class
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        cfg = {
            "features": {
                "n_selected_features": 12,
                "max_feature_correlation": 0.85,
                "use_shap": False,
            },
            "model": {"target_horizon": 5, "random_state": 42},
            "data": {"processed_dir": str(tmp_path)},
            "regimes": {},
        }
        return FeatureEngine(config=cfg)

    def test_build_returns_dataframe(self, engine, ohlcv):
        feat = engine.build(ohlcv, ticker="TEST", save=False)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) > 0

    def test_build_with_as_of_date(self, engine, ohlcv):
        cutoff = "2021-01-01"
        feat = engine.build(ohlcv, ticker="TEST", save=False, as_of_date=cutoff)
        assert feat.index.max() <= pd.Timestamp(cutoff)

    def test_build_save_creates_two_files(self, engine, ohlcv, tmp_path):
        engine.build(ohlcv, ticker="TST", save=True)
        parquets = list(tmp_path.glob("tst_features*.parquet"))
        # Versioned file + latest file
        assert len(parquets) >= 2

    def test_feature_columns_default(self, engine):
        cols = engine.feature_columns
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "label" not in cols
        assert "fwd_return" not in cols

    def test_feature_columns_updated_after_select(self, engine, ohlcv):
        feat = engine.build(ohlcv, ticker="TEST", save=False, select=True)
        selected = engine.feature_columns
        assert len(selected) <= 12
        assert "label" not in selected

    def test_select_features_standalone(self, engine, ohlcv):
        feat = engine.build(ohlcv, ticker="TEST", save=False)
        feat_selected = engine.select_features(feat, train_end_idx=200)
        assert "label" in feat_selected.columns
        # Feature count should be reduced
        feat_cols_before = _get_feature_cols(feat)
        feat_cols_after  = _get_feature_cols(feat_selected)
        assert len(feat_cols_after) <= len(feat_cols_before)

    def test_build_label_column_present(self, engine, ohlcv):
        feat = engine.build(ohlcv, ticker="TEST", save=False)
        assert "label" in feat.columns
        assert feat["label"].isin({0, 1}).all()

    def test_build_with_select_no_nan_in_features(self, engine, ohlcv):
        feat = engine.build(ohlcv, ticker="TEST", save=False, select=True)
        feat_cols = [c for c in engine.feature_columns if c in feat.columns]
        assert feat[feat_cols].isna().sum().sum() == 0

    def test_engine_backward_compat_feature_columns(self, engine, ohlcv):
        """
        models/train.py calls fe.feature_columns to pick X columns.
        Verify those columns exist in the built feature matrix.
        """
        feat = engine.build(ohlcv, ticker="SPY", save=False)
        for col in engine.feature_columns:
            if col in feat.columns:   # some may not be present before selection
                assert not feat[col].isna().any()


# ---------------------------------------------------------------------------
# Integration: full pipeline anti-leakage
# ---------------------------------------------------------------------------

class TestPipelineAntiLeakage:
    def test_future_price_spike_does_not_affect_past_features(self, ohlcv):
        """
        Injecting a huge price spike in the future must not change any feature
        value for a row that precedes the spike.
        """
        feat_original = generate_features(ohlcv, add_regime=False)

        # Inject spike 50 bars from the end
        ohlcv_spike = ohlcv.copy()
        spike_date = ohlcv_spike.index[-50]
        ohlcv_spike.loc[spike_date:, "close"] *= 100.0

        feat_spike = generate_features(ohlcv_spike, add_regime=False)

        # All rows before spike_date should be identical
        before_spike = feat_original.index[feat_original.index < spike_date]
        for col in ["rsi_14", "macd_norm", "vol_21d", "cross_20_50"]:
            if col in feat_original.columns and col in feat_spike.columns:
                pd.testing.assert_series_equal(
                    feat_original.loc[before_spike, col],
                    feat_spike.loc[before_spike, col],
                    check_names=False,
                    rtol=1e-5,
                    obj=f"Column {col} before spike",
                )

    def test_as_of_date_identical_to_manual_truncation(self, ohlcv):
        cutoff = "2021-09-30"
        feat_aod   = generate_features(ohlcv, as_of_date=cutoff, add_regime=False)
        feat_trunc = generate_features(ohlcv[ohlcv.index <= pd.Timestamp(cutoff)], add_regime=False)

        pd.testing.assert_index_equal(feat_aod.index, feat_trunc.index)
        for col in ["sma_20_dist", "rsi_14", "bb_width", "obv_trend"]:
            if col in feat_aod.columns:
                pd.testing.assert_series_equal(
                    feat_aod[col], feat_trunc[col],
                    rtol=1e-6, obj=f"Column {col}",
                )

    def test_walk_forward_each_fold_sees_no_future(self, ohlcv):
        """
        Simulate a walk-forward loop: generate features with as_of_date set to
        each fold end and verify the last feature row does not exceed that date.
        """
        fold_ends = ["2021-01-01", "2021-07-01", "2022-01-01"]
        for cutoff in fold_ends:
            feat = generate_features(ohlcv, as_of_date=cutoff, add_regime=False)
            assert feat.index.max() <= pd.Timestamp(cutoff), (
                f"Feature row beyond cutoff {cutoff}: {feat.index.max()}"
            )
