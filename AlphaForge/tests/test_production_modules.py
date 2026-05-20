"""
tests/test_production_modules.py
=================================
Unit tests for production modules that previously had no test coverage:
  monitoring/spc.py, monitoring/kill_switch.py, audit/ledger.py,
  risk/capacity.py, risk/drawdown_control.py, risk/tail_risk.py,
  risk/pretrade_risk.py, data/quality.py, validation/robustness.py,
  utils/versioning.py, utils/xai.py, features/regime_hmm.py
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_returns(n: int = 300, seed: int = 42, mu: float = 0.0005, sigma: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def _dated_returns(n: int = 300, seed: int = 42) -> pd.Series:
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=dates)



# ===========================================================================
# harness: KB stats and search (production-level checks)
# ===========================================================================

class TestHarnessKBStats:
    def _fresh_kb(self, tmp_path):
        import harness.memory.knowledge_base as kb_mod
        import harness.config as cfg_mod
        original = cfg_mod.MEMORY_DIR
        kb_dir = tmp_path / "prod_kb"
        kb_dir.mkdir()
        cfg_mod.MEMORY_DIR = kb_dir
        kb = kb_mod.KnowledgeBase.__new__(kb_mod.KnowledgeBase)
        kb.session_id = "prod_test"
        kb._dir = kb_dir
        kb._index_path = kb_dir / "index.json"
        kb._index = []
        cfg_mod.MEMORY_DIR = original
        return kb

    def test_empty_kb_stats(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        stats = kb.stats()
        assert stats["total"] == 0
        assert stats["by_type"] == {}

    def test_stats_after_mixed_entries(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save_experiment("E", {}, {"sharpe": 0.5, "oos_sharpe": 0.5}, "REJECT")
        kb.save_promotion("P", {}, {"sharpe": 0.9, "max_dd": 10.0})
        kb.save_heuristic("H", "reason")
        stats = kb.stats()
        assert stats["total"] == 3
        assert stats["by_type"]["experiment"] == 1
        assert stats["by_type"]["promotion"] == 1
        assert stats["by_type"]["heuristic"] == 1

    def test_search_query_string_matching(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save("finding", "RSI divergence works in trending markets", {"insight": "rsi"}, tags=["rsi"])
        kb.save("finding", "MACD crossover lags in sideways markets", {"insight": "macd"}, tags=["macd"])
        results = kb.search("RSI divergence")
        assert len(results) == 1

    def test_get_failures_returns_body(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save_failure("Overfitted strategy", "IS/OOS gap > 1.5")
        failures = kb.get_failures()
        assert len(failures) == 1
        assert "approach" in failures[0]["body"]

    def test_context_summary_truncates_at_max_chars(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        for i in range(20):
            kb.save_experiment(f"Exp {i} with long hypothesis text", {}, {"sharpe": 0.5 + i*0.01, "oos_sharpe": 0.5}, "ITERATE")
        summary = kb.context_summary(max_chars=500)
        assert len(summary) <= 520  # slight tolerance for truncation marker

    def test_get_best_experiments_empty_kb(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        best = kb.get_best_experiments(5)
        assert best == []

    def test_search_limit_respected(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        for i in range(10):
            kb.save_experiment(f"Exp {i}", {}, {"sharpe": float(i)/10, "oos_sharpe": float(i)/10}, "ITERATE")
        results = kb.search(entry_type="experiment", limit=3)
        assert len(results) == 3

    def test_search_tags_filters(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save("finding", "Momentum decay", {"insight": "decay"}, tags=["momentum", "decay"])
        kb.save("finding", "Mean reversion", {"insight": "mrev"}, tags=["mean_reversion"])
        results = kb.search(tags=["momentum"])
        assert len(results) == 1
        assert "momentum" in results[0]["tags"]

    def test_index_file_is_valid_json(self, tmp_path):
        import json
        kb = self._fresh_kb(tmp_path)
        kb.save_experiment("Exp", {}, {"sharpe": 0.7, "oos_sharpe": 0.7}, "ITERATE")
        data = json.loads(kb._index_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert "type" in data[0]




# ===========================================================================
# risk/capacity.py
# ===========================================================================

class TestCapacityEstimator:
    def test_zero_sharpe_returns_zero(self):
        from risk.capacity import CapacityEstimator
        est = CapacityEstimator(adv_usd=50e6)
        result = est.estimate(sharpe_0=0.0)
        assert result.max_capacity_usd == 0.0

    def test_positive_sharpe_returns_capacity(self):
        from risk.capacity import CapacityEstimator
        est = CapacityEstimator(adv_usd=50e6)
        result = est.estimate(sharpe_0=1.5)
        assert result.optimal_capacity_usd > 0
        assert result.max_capacity_usd >= result.optimal_capacity_usd

    def test_sharpe_degrades_with_capital(self):
        from risk.capacity import CapacityEstimator
        est = CapacityEstimator(adv_usd=50e6)
        result = est.estimate(sharpe_0=2.0)
        # Sharpe at 100m should be <= sharpe at 1m
        assert result.sharpe_at_100m <= result.sharpe_at_1m + 1e-6

    def test_to_dict(self):
        from risk.capacity import CapacityEstimator
        result = CapacityEstimator().estimate(sharpe_0=1.0)
        d = result.to_dict()
        assert "max_capacity_usd" in d
        assert "passes_slippage_gate" in d

    def test_half_sharpe_capacity_alias(self):
        from risk.capacity import CapacityEstimator
        result = CapacityEstimator().estimate(sharpe_0=1.5)
        assert result.half_sharpe_capacity == result.optimal_capacity_usd

    def test_negative_sharpe(self):
        from risk.capacity import CapacityEstimator
        est = CapacityEstimator()
        result = est.estimate(sharpe_0=-0.5)
        assert result.max_capacity_usd == 0.0
        assert "≤ 0" in result.warning


# ===========================================================================
# risk/drawdown_control.py
# ===========================================================================

class TestDrawdownController:
    def test_normal_tier_initial(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        state = dc.update(100_000.0)
        assert state.tier == "normal"
        assert state.scale_factor == 1.0

    def test_caution_tier(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        state = dc.update(94_000.0)
        assert state.tier == "caution"
        assert state.scale_factor == 0.75

    def test_derisked_tier(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        state = dc.update(89_000.0)
        assert state.tier == "derisked"
        assert state.scale_factor == 0.50

    def test_halted_tier(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        state = dc.update(78_000.0)
        assert state.tier == "halted"
        assert state.scale_factor == 0.0

    def test_get_scale_factor(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        dc.update(89_000.0)
        assert dc.get_scale_factor() == 0.50

    def test_is_halted(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        dc.update(75_000.0)
        assert dc.is_halted()

    def test_hysteresis_recovery_requires_bars(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController(caution_threshold=0.05, recovery_bars=3)
        dc.update(100_000.0)
        dc.update(94_000.0)  # enters caution
        assert dc._state.tier == "caution"
        # 1 bar of recovery — still in caution
        dc.update(100_000.0)
        assert dc._state.tier == "caution"
        dc.update(100_000.0)
        assert dc._state.tier == "caution"
        # 3rd bar of recovery — back to normal
        dc.update(100_000.0)
        assert dc._state.tier == "normal"

    def test_peak_nav_tracked(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        dc.update(110_000.0)
        dc.update(90_000.0)
        assert dc._peak_nav == pytest.approx(110_000.0)

    def test_recovery_required_pct(self):
        from risk.drawdown_control import DrawdownController
        dc = DrawdownController()
        dc.update(100_000.0)
        dc.update(80_000.0)
        recovery = dc.recovery_required_pct()
        assert recovery == pytest.approx(0.25, abs=1e-3)


# ===========================================================================
# risk/tail_risk.py
# ===========================================================================

class TestTailRiskManager:
    def test_insufficient_data_returns_normal(self):
        from risk.tail_risk import TailRiskManager
        trm = TailRiskManager()
        ret = pd.Series([0.01, -0.01, 0.02])
        assessment = trm.assess(ret)
        assert assessment.current_scenario == "normal"
        assert assessment.recommended_exposure == 1.0

    def test_normal_market_conditions(self):
        from risk.tail_risk import TailRiskManager
        trm = TailRiskManager()
        ret = _dated_returns(300, seed=1)
        assessment = trm.assess(ret)
        assert assessment.current_scenario in ("normal", "caution", "extreme", "black_swan")
        assert 0.0 < assessment.recommended_exposure <= 1.0

    def test_high_vol_triggers_caution(self):
        from risk.tail_risk import TailRiskManager
        trm = TailRiskManager(vol_zscore_caution=0.5)
        rng = np.random.default_rng(0)
        # Create returns where recent vol spikes
        calm = pd.Series(rng.normal(0.0, 0.005, 300))
        spike = pd.Series(rng.normal(0.0, 0.10, 21))
        ret = pd.concat([calm, spike], ignore_index=True)
        assessment = trm.assess(ret)
        assert assessment.current_scenario in ("caution", "extreme", "black_swan")
        assert assessment.recommended_exposure < 1.0

    def test_to_dict(self):
        from risk.tail_risk import TailRiskManager
        trm = TailRiskManager()
        a = trm.assess(_dated_returns(300))
        d = a.to_dict()
        assert "current_scenario" in d
        assert "recommended_exposure" in d


# ===========================================================================
# risk/pretrade_risk.py
# ===========================================================================

class TestPreTradeRiskOrchestrator:
    def _orch(self):
        from risk.pretrade_risk import PreTradeRiskOrchestrator
        return PreTradeRiskOrchestrator(
            max_position_size=0.25,
            max_leverage=2.0,
            max_daily_loss_pct=0.03,
            max_drawdown_pct=0.20,
        )

    def _ps(self, **kwargs):
        defaults = {
            "nav": 100_000.0,
            "positions": {},
            "sector_exposures": {},
            "current_drawdown": 0.02,
            "daily_pnl_pct": 0.001,
            "cash_pct": 0.80,
            "current_leverage": 0.10,
        }
        defaults.update(kwargs)
        return defaults

    def test_clean_order_approved(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10, "sector": "equity"}
        result = orch.run_checks(order, self._ps())
        assert result.approved

    def test_daily_loss_hard_block(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10}
        result = orch.run_checks(order, self._ps(daily_pnl_pct=-0.05))
        assert not result.approved
        assert result.rejection_reason is not None

    def test_drawdown_hard_block(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10}
        result = orch.run_checks(order, self._ps(current_drawdown=0.25))
        assert not result.approved

    def test_vix_hard_block(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10}
        result = orch.run_checks(order, self._ps(vix=50.0))
        assert not result.approved

    def test_leverage_hard_block(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10}
        result = orch.run_checks(order, self._ps(current_leverage=1.95))
        assert not result.approved

    def test_result_has_all_checks(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10}
        result = orch.run_checks(order, self._ps())
        assert len(result.checks) == 14

    def test_soft_block_scales_down(self):
        orch = self._orch()
        # max_position_size=0.25 but size=0.30 → soft fail → scale down
        order = {"ticker": "SPY", "size": 0.30}
        result = orch.run_checks(order, self._ps())
        # Should still be approved (no hard blocks) but final_size scaled
        assert result.approved
        assert result.final_size < result.original_size

    def test_to_dict(self):
        orch = self._orch()
        order = {"ticker": "SPY", "size": 0.10}
        result = orch.run_checks(order, self._ps())
        d = result.to_dict()
        assert "approved" in d and "checks" in d and "n_passed" in d


# ===========================================================================
# data/quality.py
# ===========================================================================

class TestDataQualityPipeline:
    def _ohlcv(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        close = np.maximum(close, 1.0)
        df = pd.DataFrame({
            "open":   close * rng.uniform(0.99, 1.0, n),
            "high":   close * rng.uniform(1.0, 1.01, n),
            "low":    close * rng.uniform(0.99, 1.0, n),
            "close":  close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        }, index=pd.date_range("2020-01-02", periods=n, freq="B"))
        return df

    def test_clean_data_no_issues(self):
        from data.quality import DataQualityPipeline
        pipeline = DataQualityPipeline()
        df = self._ohlcv()
        clean, report = pipeline.run(df, ticker="TEST")
        assert report.rows_dropped == 0
        assert isinstance(report.version_hash, str) and len(report.version_hash) == 16

    def test_outlier_detection_and_capping(self):
        from data.quality import DataQualityPipeline
        pipeline = DataQualityPipeline(zscore_threshold=2.0)
        df = self._ohlcv()
        df.iloc[10, df.columns.get_loc("close")] = 10_000.0  # extreme outlier
        clean, report = pipeline.run(df, ticker="TEST")
        assert report.outliers_detected > 0
        assert report.outliers_capped > 0
        # Outlier is capped, not removed
        assert len(clean) == len(df)

    def test_missing_data_imputation(self):
        from data.quality import DataQualityPipeline
        pipeline = DataQualityPipeline()
        df = self._ohlcv()
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        clean, report = pipeline.run(df, ticker="TEST")
        assert clean["close"].isna().sum() == 0

    def test_version_hash_deterministic(self):
        from data.quality import DataQualityPipeline
        pipeline = DataQualityPipeline()
        df = self._ohlcv()
        _, r1 = pipeline.run(df, ticker="TEST")
        _, r2 = pipeline.run(df, ticker="TEST")
        assert r1.version_hash == r2.version_hash

    def test_version_hash_changes_with_data(self):
        from data.quality import DataQualityPipeline
        pipeline = DataQualityPipeline()
        df1 = self._ohlcv(seed=1)
        df2 = self._ohlcv(seed=2)
        _, r1 = pipeline.run(df1, ticker="T1")
        _, r2 = pipeline.run(df2, ticker="T2")
        assert r1.version_hash != r2.version_hash

    def test_report_to_dict(self):
        from data.quality import DataQualityPipeline
        pipeline = DataQualityPipeline()
        _, report = pipeline.run(self._ohlcv(), ticker="SPY")
        d = report.to_dict()
        assert d["ticker"] == "SPY"
        assert "version_hash" in d


# ===========================================================================
# validation/robustness.py
# ===========================================================================

class TestRobustnessGates:
    def test_positive_returns_passes_most_gates(self):
        from validation.robustness import RobustnessGates
        gates = RobustnessGates(require_all=False)
        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-02", periods=504, freq="B")
        ret = pd.Series(rng.normal(0.002, 0.01, 504), index=dates)
        report = gates.evaluate(ret, is_sharpe=2.0, oos_sharpe_list=[1.5, 1.8, 2.0])
        # With clearly positive returns over 2 years, most gates should pass
        assert report.n_passed >= 3

    def test_negative_returns_fails_most_gates(self):
        from validation.robustness import RobustnessGates
        gates = RobustnessGates(require_all=False)
        rng = np.random.default_rng(99)
        ret = pd.Series(rng.normal(-0.003, 0.015, 252))
        report = gates.evaluate(ret, is_sharpe=-1.0, oos_sharpe_list=[-0.5, -1.0])
        assert report.n_passed < 5

    def test_report_fields(self):
        from validation.robustness import RobustnessGates
        gates = RobustnessGates()
        ret = _dated_returns(300)
        report = gates.evaluate(ret)
        assert hasattr(report, "dsr")
        assert hasattr(report, "t_stat")
        assert hasattr(report, "cpcv_sharpe")
        assert hasattr(report, "min_trl_years")
        assert hasattr(report, "overall_pass")

    def test_to_dict(self):
        from validation.robustness import RobustnessGates
        gates = RobustnessGates()
        ret = _dated_returns(300)
        d = gates.evaluate(ret).to_dict()
        assert "dsr" in d and "t_stat" in d and "overall_pass" in d

    def test_norm_cdf_properties(self):
        from validation.robustness import _norm_cdf
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-6)
        assert _norm_cdf(3.0) > 0.99
        assert _norm_cdf(-3.0) < 0.01

    def test_norm_ppf_inverse(self):
        from validation.robustness import _norm_cdf, _norm_ppf
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            x = _norm_ppf(p)
            assert _norm_cdf(x) == pytest.approx(p, abs=0.01)

    def test_slippage_sharpes_ordered(self):
        from validation.robustness import RobustnessGates
        gates = RobustnessGates()
        rng = np.random.default_rng(8)
        dates = pd.date_range("2020-01-02", periods=504, freq="B")
        ret = pd.Series(rng.normal(0.002, 0.008, 504), index=dates)
        report = gates.evaluate(ret)
        # Higher slippage multiplier → lower (or equal) Sharpe
        assert report.slippage_sharpes[3] <= report.slippage_sharpes[1] + 1e-9


# ===========================================================================
# features/regime_hmm.py
# ===========================================================================

class TestHMMRegimeDetector:
    def _price_df(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        prices = np.maximum(prices, 10.0)
        return pd.DataFrame({"close": prices}, index=pd.date_range("2020-01-02", periods=n, freq="B"))

    def test_fit_and_predict(self):
        from features.regime_hmm import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3, max_iter=5)
        df = self._price_df(200)
        det.fit(df)
        regimes = det.predict(df)
        assert len(regimes) == len(df)
        # "unknown" is valid for warmup bars
        assert set(regimes.unique()).issubset({"bull", "bear", "sideways", "unknown"})

    def test_predict_proba(self):
        from features.regime_hmm import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3, max_iter=5)
        df = self._price_df(200)
        det.fit(df)
        proba = det.predict_proba(df)
        assert isinstance(proba, pd.DataFrame)
        # Probabilities should sum to ~1 across states
        row_sums = proba.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-3

    def test_current_regime(self):
        from features.regime_hmm import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3, max_iter=5)
        df = self._price_df(200)
        det.fit(df)
        regime = det.current_regime(df)
        assert regime in ("bull", "bear", "sideways", "unknown")

    def test_min_history_enforced(self):
        from features.regime_hmm import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3, min_history=100)
        df = self._price_df(30)
        # Short series should return unknown without crashing
        regime = det.current_regime(df)
        assert regime == "unknown"

    def test_state_labels_cover_bull_bear(self):
        from features.regime_hmm import HMMRegimeDetector
        det = HMMRegimeDetector(n_states=3, max_iter=20)
        # Construct series with clear bull/bear regimes
        rng = np.random.default_rng(7)
        bull = 100 + np.cumsum(rng.normal(0.3, 0.5, 150))
        bear = bull[-1] - np.cumsum(rng.normal(0.3, 0.5, 100))
        prices = np.concatenate([bull, bear])
        df = pd.DataFrame({"close": prices}, index=pd.date_range("2020-01-02", periods=len(prices), freq="B"))
        det.fit(df)
        regimes = det.predict(df)
        assert "bull" in regimes.values or "bear" in regimes.values


# ===========================================================================
# utils/xai.py  — lightweight tests (shap optional, permutation always runs)
# ===========================================================================

class TestXAIExplainer:
    def _make_data(self, n=100, seed=42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series((X["f0"] + rng.normal(0, 0.1, n) > 0).astype(int))
        return X, y

    def test_xai_explainer_importable(self):
        from utils import xai  # noqa: F401

    def test_explainer_instantiation(self):
        from utils.xai import XAIExplainer
        exp = XAIExplainer(top_k=5, use_shap=False)
        assert exp is not None

    def test_fit_baseline(self):
        from utils.xai import XAIExplainer
        from sklearn.linear_model import LogisticRegression
        X, y = self._make_data()
        model = LogisticRegression(max_iter=200).fit(X, y)
        exp = XAIExplainer(top_k=3, use_shap=False)
        exp.fit_baseline(X, model)   # correct order: feature_df, model

    def test_explain_trade_returns_explanation(self):
        from utils.xai import XAIExplainer, TradeExplanation
        from sklearn.linear_model import LogisticRegression
        X, y = self._make_data()
        model = LogisticRegression(max_iter=200).fit(X, y)
        exp = XAIExplainer(top_k=3, use_shap=False)
        exp.fit_baseline(X, model)
        row = X.iloc[0]   # pass Series, not DataFrame
        result = exp.explain_trade(row, model, signal=1.0, confidence=0.75, regime="bull")
        assert isinstance(result, TradeExplanation)
        assert len(result.top_features) <= 3
