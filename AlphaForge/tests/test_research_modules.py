"""
tests/test_research_modules.py
================================
Unit tests for execution + feature modules and the AI harness knowledge base:
  execution/almgren_chriss.py, features/fracdiff.py, features/scaler.py,
  harness/memory/knowledge_base.py
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_returns(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.01, n))


def _price_series(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.Series(np.maximum(prices, 10.0))


def _dated_price_series(n: int = 200, seed: int = 0) -> pd.Series:
    s = _price_series(n, seed)
    s.index = pd.date_range("2020-01-02", periods=n, freq="B")
    return s


# ===========================================================================
# execution/almgren_chriss.py
# ===========================================================================

class TestAlmgrenChriss:
    def test_trajectory_length(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10, sigma=1.0, eta=0.01, gamma=0.001)
        assert len(result.trajectory) == 11
        assert len(result.trade_list) == 10

    def test_trajectory_starts_at_X(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=5000, T=10, N=10)
        assert result.trajectory[0] == pytest.approx(5000.0, rel=1e-6)

    def test_trajectory_ends_near_zero(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10)
        assert abs(result.trajectory[-1]) < 1.0  # fully liquidated

    def test_trade_list_sums_to_X(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10)
        assert sum(result.trade_list) == pytest.approx(1000.0, rel=1e-5)

    def test_expected_cost_positive(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10, sigma=2.5, eta=0.01, gamma=0.001)
        assert result.expected_cost > 0

    def test_variance_cost_positive(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10, sigma=2.5)
        assert result.variance_cost >= 0

    def test_higher_risk_aversion_front_loads(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        # Higher lambda → more urgent → first trade larger
        r_low  = almgren_chriss_trajectory(X=1000, T=10, N=10, lam=1e-9)
        r_high = almgren_chriss_trajectory(X=1000, T=10, N=10, lam=1e-3)
        assert r_high.trade_list[0] >= r_low.trade_list[0]

    def test_zero_risk_aversion_uniform(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10, lam=0.0)
        # VWAP path: all trades should be ~equal
        assert result.trade_list.std() / result.trade_list.mean() < 0.05

    def test_to_dict(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=500, T=5, N=5)
        d = result.to_dict()
        assert "expected_cost" in d and "trade_list" in d

    def test_efficient_frontier(self):
        from execution.almgren_chriss import efficient_frontier
        frontier = efficient_frontier(X=1000, T=10, N=10, n_points=5)
        assert len(frontier) == 5
        assert all("expected_cost" in f and "variance_cost" in f for f in frontier)

    def test_adaptive_execution(self):
        from execution.almgren_chriss import adaptive_ac_execution
        prices = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 20))
        result = adaptive_ac_execution(X=1000, T=10, N=10, price_series=prices)
        assert result.trade_list.sum() == pytest.approx(1000.0, abs=10.0)

    def test_vwap_deviation(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=1000, T=10, N=10)
        prices = np.ones(10) * 100.0
        dev = result.vwap_deviation(prices)
        assert isinstance(dev, float)

    def test_negative_x_buy_direction(self):
        from execution.almgren_chriss import almgren_chriss_trajectory
        result = almgren_chriss_trajectory(X=-1000, T=10, N=10)
        assert result.trajectory[0] == pytest.approx(-1000.0, rel=1e-5)
        assert result.trade_list.sum() == pytest.approx(-1000.0, rel=1e-5)


# ===========================================================================
# features/fracdiff.py
# ===========================================================================

class TestFracdiff:
    def test_fracdiff_d1_approx_returns(self):
        from features.fracdiff import fracdiff
        prices = _price_series(100)
        fd = fracdiff(prices, d=1.0, threshold=1e-6)
        # d=1 → fractional diff ≈ first-difference
        pct_diff = prices.diff()
        # They won't be exactly equal but should be correlated
        valid = fd.dropna()
        assert len(valid) > 50

    def test_fracdiff_d0_identity(self):
        from features.fracdiff import fracdiff
        prices = _price_series(100)
        fd = fracdiff(prices, d=0.0)
        pd.testing.assert_series_equal(fd, prices, check_names=False)

    def test_fracdiff_output_length(self):
        from features.fracdiff import fracdiff
        prices = _price_series(200)
        fd = fracdiff(prices, d=0.4)
        assert len(fd) == len(prices)

    def test_fracdiff_leading_nans(self):
        from features.fracdiff import fracdiff
        prices = _price_series(200)
        fd = fracdiff(prices, d=0.4)
        # First few values should be NaN (warmup)
        assert fd.isna().any()

    def test_fracdiff_no_nans_at_end(self):
        from features.fracdiff import fracdiff
        prices = _price_series(200)
        fd = fracdiff(prices, d=0.4)
        assert not fd.iloc[-50:].isna().any()

    def test_fracdiff_df_applies_to_columns(self):
        from features.fracdiff import fracdiff_df
        df = pd.DataFrame({"close": _price_series(100), "volume": _price_series(100, seed=1)})
        out = fracdiff_df(df, d=0.5)
        assert "close_fd0.50" in out.columns
        assert "volume_fd0.50" in out.columns

    def test_fracdiff_invalid_d_raises(self):
        from features.fracdiff import fracdiff
        prices = _price_series(100)
        with pytest.raises(ValueError):
            fracdiff(prices, d=3.0)

    def test_add_fracdiff_features(self):
        from features.fracdiff import add_fracdiff_features
        n = 150
        df = pd.DataFrame({
            "close": _price_series(n),
            "open":  _price_series(n, seed=2),
        })
        out = add_fracdiff_features(df, d=0.4)
        assert "close_fd" in out.columns
        assert len(out) == n


# ===========================================================================
# features/scaler.py
# ===========================================================================

class TestFeatureScaler:
    def _df(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(5, 3, n),
            "f3": rng.exponential(2, n),
        })

    def test_fit_and_transform(self):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler(regime_aware=False)
        df = self._df()
        scaled = scaler.fit_transform(df)
        assert scaled.shape == df.shape
        assert scaler._fitted

    def test_median_near_zero_after_scaling(self):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler(regime_aware=False)
        df = self._df()
        scaled = scaler.fit_transform(df)
        for col in scaled.columns:
            assert abs(scaled[col].median()) < 0.5

    def test_transform_before_fit_returns_unchanged(self):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler(regime_aware=False)
        df = self._df()
        out = scaler.transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_regime_aware_fit(self):
        from features.scaler import FeatureScaler
        n = 300
        scaler = FeatureScaler(regime_aware=True, min_regime_samples=30)
        df = self._df(n=n)
        regimes = pd.Series(["bull"] * 150 + ["bear"] * 150, index=df.index)
        scaled = scaler.fit_transform(df, regimes=regimes)
        assert "bull" in scaler._regime

    def test_regime_transform(self):
        from features.scaler import FeatureScaler
        n = 300
        scaler = FeatureScaler(regime_aware=True, min_regime_samples=30)
        df = self._df(n=n)
        regimes = pd.Series(["bull"] * 150 + ["bear"] * 150, index=df.index)
        scaler.fit(df, regimes=regimes)
        test_df = self._df(n=10)
        scaled_bull = scaler.transform(test_df, regime="bull")
        scaled_bear = scaler.transform(test_df, regime="bear")
        # Bull and bear should produce different results
        assert not scaled_bull.equals(scaled_bear)

    def test_winsorization_clips_extremes(self):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler(regime_aware=False, winsor_low=5.0, winsor_high=95.0)
        df = self._df()
        df.iloc[0, 0] = 1e9  # extreme outlier
        scaled = scaler.fit_transform(df)
        assert scaled.iloc[0, 0] < 100  # should be clipped

    def test_to_dict(self):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler()
        scaler.fit(self._df())
        d = scaler.to_dict()
        assert "n_features" in d and d["fitted"] is True

    def test_save_and_load(self, tmp_path):
        from features.scaler import FeatureScaler
        scaler = FeatureScaler(regime_aware=False)
        df = self._df()
        scaler.fit(df)
        path = str(tmp_path / "scaler.joblib")
        scaler.save(path)
        loaded = FeatureScaler.load(path)
        assert loaded._fitted
        # Loaded scaler should produce same output
        out1 = scaler.transform(df)
        out2 = loaded.transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_artifact_path(self):
        from features.scaler import FeatureScaler
        path = FeatureScaler.artifact_path("models/artifacts", "SPY")
        assert "spy_scaler.joblib" in path


# ===========================================================================
# harness/memory/knowledge_base.py
# ===========================================================================

class TestHarnessKnowledgeBase:
    """Tests for harness/memory/knowledge_base.py â€” the AI harness JSON KB."""

    def _make_kb(self, tmp_path):
        """Create KB instance pointing at tmp_path to avoid touching real data."""
        import harness.memory.knowledge_base as kb_mod
        import harness.config as cfg_mod
        original = cfg_mod.MEMORY_DIR
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir(exist_ok=True)
        cfg_mod.MEMORY_DIR = kb_dir
        kb = kb_mod.KnowledgeBase.__new__(kb_mod.KnowledgeBase)
        kb.session_id = "test_session"
        kb._dir = kb_dir
        kb._dir.mkdir(parents=True, exist_ok=True)
        kb._index_path = kb._dir / "index.json"
        kb._index = []
        cfg_mod.MEMORY_DIR = original
        return kb

    def test_save_returns_8char_id(self, tmp_path):
        kb = self._make_kb(tmp_path)
        entry_id = kb.save("experiment", "Test hypothesis", {"config": {}, "results": {}})
        assert isinstance(entry_id, str) and len(entry_id) == 8

    def test_save_creates_json_file(self, tmp_path):
        kb = self._make_kb(tmp_path)
        entry_id = kb.save("finding", "RSI works in bull", {"insight": "RSI"}, tags=["rsi"])
        assert (kb._dir / f"{entry_id}.json").exists()

    def test_save_updates_index(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save("heuristic", "Shorter lookback in bear", {"rule": "short", "reason": "regime"})
        assert len(kb._index) == 1

    def test_save_experiment_extracts_metrics(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_experiment(
            hypothesis="Momentum 20-bar lookback",
            config={"features": ["momentum_20"], "horizon": 5},
            results={"sharpe": 1.2, "ann_return_pct": 15.0, "max_dd_pct": 8.0, "oos_sharpe": 0.9},
            verdict="ITERATE",
        )
        assert kb._index[0]["metrics"]["sharpe"] == pytest.approx(1.2)
        assert kb._index[0]["metrics"]["oos_sharpe"] == pytest.approx(0.9)

    def test_save_finding(self, tmp_path):
        kb = self._make_kb(tmp_path)
        eid = kb.save_finding("Alpha decays fast", "Past 5 bars consistently", tags=["alpha"])
        assert eid is not None
        assert kb._index[0]["type"] == "finding"

    def test_save_failure(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_failure("MACD long windows", "Extreme lag during regime changes")
        assert kb._index[0]["type"] == "failure"

    def test_save_promotion(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_promotion("Momentum_20d", {"features": ["mom"]}, {"sharpe": 1.1, "max_dd": 12.0})
        assert kb._index[0]["type"] == "promotion"

    def test_search_by_type_filters(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_experiment("Hypothesis A", {}, {"sharpe": 0.5, "oos_sharpe": 0.5}, "REJECT")
        kb.save_finding("Finding A", "Some insight")
        results = kb.search(entry_type="experiment")
        assert len(results) == 1 and results[0]["type"] == "experiment"

    def test_search_by_min_sharpe(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_experiment("Low", {}, {"sharpe": 0.3, "oos_sharpe": 0.3}, "REJECT")
        kb.save_experiment("High", {}, {"sharpe": 1.5, "oos_sharpe": 1.5}, "PROMOTE")
        results = kb.search(min_sharpe=1.0)
        assert len(results) == 1

    def test_get_returns_full_body(self, tmp_path):
        kb = self._make_kb(tmp_path)
        eid = kb.save("experiment", "Full body test", {"key": "value"})
        full = kb.get(eid)
        assert full is not None and full["body"]["key"] == "value"

    def test_get_nonexistent_returns_none(self, tmp_path):
        kb = self._make_kb(tmp_path)
        assert kb.get("nonexistent") is None

    def test_get_promotions(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_promotion("Strategy_A", {}, {"sharpe": 0.9, "max_dd": 15.0})
        kb.save_promotion("Strategy_B", {}, {"sharpe": 1.1, "max_dd": 10.0})
        promotions = kb.get_promotions()
        assert len(promotions) == 2

    def test_get_best_experiments_sorted_by_oos_sharpe(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_experiment("Low OOS", {}, {"sharpe": 0.5, "oos_sharpe": 0.5}, "REJECT")
        kb.save_experiment("High OOS", {}, {"sharpe": 1.2, "oos_sharpe": 1.2}, "PROMOTE")
        kb.save_experiment("Mid OOS", {}, {"sharpe": 0.8, "oos_sharpe": 0.8}, "ITERATE")
        best = kb.get_best_experiments(2)
        assert len(best) == 2
        assert best[0]["metrics"]["oos_sharpe"] >= best[1]["metrics"]["oos_sharpe"]

    def test_context_summary_empty_kb(self, tmp_path):
        kb = self._make_kb(tmp_path)
        summary = kb.context_summary()
        assert isinstance(summary, str) and len(summary) > 0

    def test_context_summary_with_data_contains_sections(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_promotion("TopStrategy", {}, {"sharpe": 1.1, "max_dd": 10.0})
        kb.save_failure("Failed approach", "Too many false signals")
        kb.save_heuristic("Use regime filter", "Reduces false positives")
        summary = kb.context_summary()
        assert "Promoted" in summary or "Dead-End" in summary or "Heuristic" in summary

    def test_stats_counts_by_type(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_experiment("E1", {}, {"sharpe": 0.5, "oos_sharpe": 0.5}, "REJECT")
        kb.save_experiment("E2", {}, {"sharpe": 0.7, "oos_sharpe": 0.7}, "ITERATE")
        kb.save_finding("F1", "insight")
        stats = kb.stats()
        assert stats["total"] == 3
        assert stats["by_type"]["experiment"] == 2
        assert stats["by_type"]["finding"] == 1

    def test_index_persisted_to_disk(self, tmp_path):
        import json
        kb = self._make_kb(tmp_path)
        kb.save("experiment", "Persisted", {"key": "val"})
        assert kb._index_path.exists()
        loaded = json.loads(kb._index_path.read_text())
        assert len(loaded) == 1

    def test_search_by_tags(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save("finding", "RSI finding", {"insight": "rsi works"}, tags=["rsi", "momentum"])
        kb.save("finding", "MACD finding", {"insight": "macd lags"}, tags=["macd"])
        results = kb.search(tags=["rsi"])
        assert len(results) == 1

    def test_save_experiment_tags_from_features(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_experiment(
            hypothesis="Test with features",
            config={"features": ["rsi_14", "macd"]},
            results={"sharpe": 0.7, "oos_sharpe": 0.7},
            verdict="ITERATE",
        )
        assert "rsi_14" in kb._index[0]["tags"]
        assert "macd" in kb._index[0]["tags"]
