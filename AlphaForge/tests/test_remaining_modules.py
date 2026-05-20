"""
tests/test_remaining_modules.py
=================================
Third batch of tests covering remaining untested modules:

  - risk/dynamic_position_sizing.py   → DynamicPositionSizer, SizingDecision
  - models/ensemble.py                → EnsembleModel
  - validation/walk_forward.py        → WalkForwardValidator, run_walk_forward
  - research/efficiency_optimizer.py  → EfficiencyOptimizer
  - features/momentum.py              → MomentumStrategy, compute_signals
  - features/cross_asset_cache.py     → CrossAssetCache
  - monitoring/drift_detector.py      → DriftDetector
  - research/post_mortem.py           → PostMortemEngine, PostMortemReport
"""

from __future__ import annotations

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _price_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    close = 100.0 * np.exp(rng.normal(0.0002, 0.012, n).cumsum())
    return pd.DataFrame({
        "open":   close * (1 - 0.002),
        "high":   close * (1 + 0.005),
        "low":    close * (1 - 0.005),
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _returns(n: int = 60, seed: int = 1, mu: float = 0.001, sigma: float = 0.015) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(rng.normal(mu, sigma, n), index=dates)


# ===========================================================================
# DynamicPositionSizer
# ===========================================================================

class TestDynamicPositionSizer:

    def _sizer(self, method="vol_target", **kw):
        from risk.dynamic_position_sizing import DynamicPositionSizer
        return DynamicPositionSizer(method=method, **kw)

    def test_below_confidence_gate_returns_zero(self):
        dps = self._sizer(min_confidence=0.60)
        dec = dps.size(signal=1.0, confidence=0.50, returns_series=None)
        assert dec.recommended_size == 0.0

    def test_vol_target_positive_signal_bounded(self):
        dps = self._sizer(method="vol_target", target_vol=0.15, max_position=0.25)
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=_returns(40))
        assert dec.recommended_size > 0.0
        assert dec.recommended_size <= 0.25

    def test_vol_target_negative_signal_negative_size(self):
        dps = self._sizer(method="vol_target", max_position=0.25)
        dec = dps.size(signal=-1.0, confidence=0.70, returns_series=None)
        assert dec.recommended_size < 0.0

    def test_kelly_method(self):
        from risk.dynamic_position_sizing import SizingMethod
        dps = self._sizer(method=SizingMethod.FRACTIONAL_KELLY, kelly_fraction=0.25)
        dec = dps.size(signal=0.8, confidence=0.75, returns_series=_returns())
        assert dec.method_used == "kelly"
        assert abs(dec.recommended_size) <= dps.max_position

    def test_confidence_edge_method(self):
        dps = self._sizer(method="confidence", min_confidence=0.55, max_position=0.25)
        dec = dps.size(signal=1.0, confidence=0.80, returns_series=None)
        assert dec.recommended_size > 0.0
        assert dec.recommended_size <= 0.25

    def test_regime_aware_bull(self):
        dps = self._sizer(method="regime")
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=_returns(), regime="bull")
        assert dec.method_used == "regime"
        assert dec.recommended_size > 0.0

    def test_regime_aware_bear_reduced_vs_bull(self):
        dps = self._sizer(method="regime", max_position=0.25)
        bull = dps.size(signal=1.0, confidence=0.80, returns_series=_returns(), regime="bull")
        bear = dps.size(signal=1.0, confidence=0.80, returns_series=_returns(), regime="bear")
        # bear multiplier 0.5 vs bull 1.0
        assert abs(bear.recommended_size) <= abs(bull.recommended_size) + 1e-6

    def test_drawdown_halt_returns_zero(self):
        dps = self._sizer(method="vol_target", dd_halt=0.20, drawdown_derisking=True)
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=None, current_drawdown=0.25)
        assert dec.recommended_size == 0.0

    def test_drawdown_caution_reduces_size(self):
        dps = self._sizer(method="vol_target", dd_caution=0.05, drawdown_derisking=True)
        no_dd = dps.size(signal=1.0, confidence=0.70, returns_series=None, current_drawdown=0.0)
        with_dd = dps.size(signal=1.0, confidence=0.70, returns_series=None, current_drawdown=0.07)
        assert with_dd.recommended_size <= no_dd.recommended_size + 1e-9

    def test_risk_budget_method(self):
        dps = self._sizer(method="risk_budget", max_portfolio_risk=0.02)
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=None, portfolio_vol=0.15)
        assert dec.recommended_size != 0.0
        assert abs(dec.recommended_size) <= dps.max_position

    def test_decision_has_reasoning_string(self):
        dps = self._sizer()
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=None)
        assert isinstance(dec.reasoning, str) and len(dec.reasoning) > 0

    def test_decision_risk_params_applied(self):
        dps = self._sizer()
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=None)
        assert "max_position" in dec.risk_params_applied

    def test_get_learnable_params_keys(self):
        dps = self._sizer()
        params = dps.get_learnable_params()
        for key in ("kelly_fraction", "target_vol", "max_position", "dd_caution"):
            assert key in params

    def test_set_learnable_params_clamps(self):
        dps = self._sizer()
        dps.set_learnable_params({"kelly_fraction": 5.0, "target_vol": -1.0})
        assert dps.kelly_fraction <= 1.0
        assert dps.target_vol >= 0.05

    def test_max_position_cap_enforced(self):
        dps = self._sizer(method="vol_target", max_position=0.10, target_vol=0.50)
        dec = dps.size(signal=1.0, confidence=0.90, returns_series=_returns(20))
        assert abs(dec.recommended_size) <= 0.10 + 1e-9

    def test_drawdown_derisking_disabled(self):
        dps = self._sizer(method="vol_target", drawdown_derisking=False)
        dec = dps.size(signal=1.0, confidence=0.70, returns_series=None, current_drawdown=0.25)
        # Should NOT be zero when derisking is disabled
        assert dec.recommended_size != 0.0


# ===========================================================================
# EnsembleModel
# ===========================================================================

class TestEnsembleModel:

    def _X_y(self, n: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"f{i}" for i in range(6)])
        y = pd.Series((rng.random(n) > 0.5).astype(int))
        return X, y

    def test_average_fit_predict_proba(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(method="average", use_xgb=True, use_rf=False, use_logreg=True, n_xgb=1)
        X, y = self._X_y()
        result = ens.fit(X, y)
        assert result.n_models >= 1
        probas = ens.predict_proba(X)
        assert probas.shape == (len(X),)
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_predict_signal_values_in_set(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(method="average", use_xgb=True, use_rf=False, use_logreg=True, n_xgb=1)
        X, y = self._X_y()
        ens.fit(X, y)
        sigs = ens.predict_signal(X, threshold=0.55)
        assert set(sigs).issubset({-1, 0, 1})

    def test_unfitted_returns_half_proba(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel()
        X, _ = self._X_y(50)
        probas = ens.predict_proba(X)
        assert np.allclose(probas, 0.5)

    def test_all_one_class_early_return(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(use_xgb=True, use_rf=False, use_logreg=False, n_xgb=1)
        X = pd.DataFrame(np.ones((50, 4)), columns=[f"f{i}" for i in range(4)])
        y = pd.Series(np.ones(50, dtype=int))
        result = ens.fit(X, y)
        assert result.n_models == 0

    def test_vote_method_proba_in_range(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(method="vote", use_xgb=True, use_rf=False, use_logreg=True, n_xgb=1)
        X, y = self._X_y()
        ens.fit(X, y)
        probas = ens.predict_proba(X)
        assert probas.shape == (len(X),)
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_stack_method(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(method="stack", use_xgb=True, use_rf=False, use_logreg=True, n_xgb=1)
        X, y = self._X_y(300)
        result = ens.fit(X, y)
        assert result.n_models >= 1
        probas = ens.predict_proba(X)
        assert probas.shape == (len(X),)

    def test_logreg_only(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(use_xgb=False, use_rf=False, use_logreg=True)
        X, y = self._X_y()
        result = ens.fit(X, y)
        assert result.n_models >= 1

    def test_fit_result_has_component_weights(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(use_xgb=True, use_rf=False, use_logreg=True, n_xgb=1)
        X, y = self._X_y()
        result = ens.fit(X, y)
        assert isinstance(result.component_weights, dict)

    def test_predict_proba_shape_matches_input(self):
        from models.ensemble import EnsembleModel
        ens = EnsembleModel(use_xgb=True, use_rf=False, use_logreg=False, n_xgb=1)
        X, y = self._X_y(200)
        ens.fit(X, y)
        probas = ens.predict_proba(X.iloc[:50])
        assert probas.shape == (50,)


# ===========================================================================
# WalkForwardValidator
# ===========================================================================

class TestWalkForwardValidator:

    def _flat_signal_fn(self, history, target):
        return pd.Series(0, index=target.index)

    def _momentum_fn(self, history, target):
        from features.momentum import MomentumStrategy
        strat = MomentumStrategy(fast_period=5, slow_period=10, rsi_period=5)
        return strat(history, target)

    def test_basic_run_produces_folds(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=80, test_days=20, embargo_days=5, min_folds=3)
        result = v.run(_price_df(400), self._flat_signal_fn, ticker="TEST")
        assert len(result.folds) >= 3

    def test_fold_timestamps_ordered(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=80, test_days=20, embargo_days=5, min_folds=2)
        result = v.run(_price_df(400), self._flat_signal_fn)
        for fold in result.folds:
            assert fold.train_start < fold.train_end
            assert fold.train_end < fold.test_start
            assert fold.test_start <= fold.test_end

    def test_aggregates_computed(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=80, test_days=20, embargo_days=5, min_folds=3)
        result = v.run(_price_df(400), self._flat_signal_fn)
        assert isinstance(result.mean_is_sharpe, float)
        assert isinstance(result.mean_oos_sharpe, float)
        assert result.overfitting_score >= 0.0

    def test_missing_close_column_raises(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=60, test_days=20, embargo_days=5, min_folds=2)
        bad_df = pd.DataFrame({"price": range(200)}, index=pd.bdate_range("2020-01-01", periods=200))
        with pytest.raises(ValueError, match="close"):
            v.run(bad_df, self._flat_signal_fn)

    def test_insufficient_data_raises(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=200, test_days=100, embargo_days=50, min_folds=5)
        with pytest.raises(ValueError):
            v.run(_price_df(200), self._flat_signal_fn)

    def test_oos_equity_not_empty_with_momentum(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=80, test_days=20, embargo_days=5, min_folds=3)
        result = v.run(_price_df(400), self._momentum_fn)
        assert not result.oos_equity.empty

    def test_run_walk_forward_convenience_wrapper(self):
        from validation.walk_forward import run_walk_forward
        result = run_walk_forward(
            _price_df(400),
            self._flat_signal_fn,
            ticker="TEST",
            train_days=80,
            test_days=20,
            embargo_days=5,
            min_folds=3,
            print_report=False,
        )
        assert len(result.folds) >= 3
        assert result.ticker == "TEST"

    def test_fold_result_metrics_dict(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=80, test_days=20, embargo_days=5, min_folds=2)
        result = v.run(_price_df(400), self._flat_signal_fn)
        fold = result.folds[0]
        assert isinstance(fold.is_metrics, dict)
        assert isinstance(fold.oos_metrics, dict)

    def test_combined_metrics_populated(self):
        from validation.walk_forward import WalkForwardValidator
        v = WalkForwardValidator(train_days=80, test_days=20, embargo_days=5, min_folds=3)
        result = v.run(_price_df(400), self._flat_signal_fn)
        assert isinstance(result.combined_metrics, dict)



# ===========================================================================
# harness/orchestrator.py — AlphaHarness structure tests
# ===========================================================================

class TestAlphaHarnessStructure:
    """Tests for harness/orchestrator.py structure and invariants."""

    def test_orchestrator_has_discover_method(self):
        from harness.orchestrator import AlphaHarness
        assert hasattr(AlphaHarness, "discover")

    def test_orchestrator_has_research_universe_method(self):
        from harness.orchestrator import AlphaHarness
        assert hasattr(AlphaHarness, "research_universe")

    def test_orchestrator_has_add_factor_method(self):
        from harness.orchestrator import AlphaHarness
        assert hasattr(AlphaHarness, "add_factor_and_test")

    def test_promote_threshold_in_orchestrator(self):
        """Sharpe >= 0.8 AND max DD <= 25% promote criteria must exist in source."""
        from harness.config import PROMOTE_SHARPE_THRESHOLD, PROMOTE_DD_LIMIT
        assert PROMOTE_SHARPE_THRESHOLD == 0.8
        assert PROMOTE_DD_LIMIT == 0.25

    def test_all_four_agents_referenced_in_orchestrator(self):
        import inspect
        from harness import orchestrator
        src = inspect.getsource(orchestrator)
        assert "StrategistAgent" in src
        assert "AnalystAgent" in src
        assert "CoderAgent" in src
        assert "ReviewerAgent" in src

    def test_knowledge_base_referenced_in_orchestrator(self):
        import inspect
        from harness import orchestrator
        src = inspect.getsource(orchestrator)
        assert "KnowledgeBase" in src

    def test_session_id_generated_on_init(self):
        """AlphaHarness sets a session_id on creation (requires API keys — check attribute structure only)."""
        import inspect
        from harness.orchestrator import AlphaHarness
        src = inspect.getsource(AlphaHarness.__init__)
        assert "session_id" in src

    def test_simulation_only_invariant(self):
        """The orchestrator must not contain any live broker connection code."""
        import inspect
        from harness import orchestrator
        src = inspect.getsource(orchestrator).lower()
        for forbidden in ("alpaca", "ibkr", "interactive brokers", "real order", "live_order"):
            assert forbidden not in src, f"Found forbidden term: {forbidden}"


# ===========================================================================
# harness: executor tool schemas
# ===========================================================================

class TestToolExecutorSchema:
    """Tests for harness/tools/executor.py bridge layer."""

    def test_executor_class_exists(self):
        from harness.tools.executor import ToolExecutor
        assert ToolExecutor is not None

    def test_executor_has_execute_method(self):
        from harness.tools.executor import ToolExecutor
        assert hasattr(ToolExecutor, "execute")

    def test_executor_references_run_backtest_tool(self):
        import inspect
        from harness.tools import executor
        src = inspect.getsource(executor)
        assert "run_backtest" in src

    def test_executor_references_train_model_tool(self):
        import inspect
        from harness.tools import executor
        src = inspect.getsource(executor)
        assert "train_model" in src

    def test_tool_registry_claude_format(self):
        from harness.tools.registry import CLAUDE_TOOLS
        for tool in CLAUDE_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema.get("type") == "object"

    def test_tool_registry_grok_format(self):
        from harness.tools.registry import GROK_TOOLS
        for tool in GROK_TOOLS:
            assert tool.get("type") == "function"
            fn = tool["function"]
            assert "name" in fn and "description" in fn

    def test_claude_and_grok_tool_counts_match(self):
        from harness.tools.registry import CLAUDE_TOOLS, GROK_TOOLS
        assert len(CLAUDE_TOOLS) == len(GROK_TOOLS)
