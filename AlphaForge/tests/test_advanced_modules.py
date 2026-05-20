"""
Tests for five advanced modules:
  - features/change_point.py
  - features/multi_timeframe.py
  - models/online_learner.py
  - monitoring/drift_handler.py
  - research/bayesian_optimizer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _price_series(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n)
    prices = 100 * np.cumprod(1 + rets)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="close")


def _returns_series(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rets, index=idx, name="ret")


def _returns_with_shift(n: int = 300, shift_at: int = 150, seed: int = 1) -> pd.Series:
    """Returns that have a mean shift in the second half."""
    rng = np.random.default_rng(seed)
    pre  = rng.normal(0.001, 0.01, shift_at)
    post = rng.normal(-0.005, 0.02, n - shift_at)
    arr = np.concatenate([pre, post])
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(arr, name="ret", index=idx)


def _ohlcv_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    close = _price_series(n, seed=seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.0, 0.01, n)),
        "low":    close * (1 - rng.uniform(0.0, 0.01, n)),
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=close.index)


# ── Minimal mock TrainedEnsembleModel ─────────────────────────────────────────

class _MockModel:
    """Minimal GBM-like model that satisfies the OnlineLearner interface."""

    def __init__(self):
        self._w = None

    def fit(self, X, y, sample_weight=None):
        self._w = np.zeros(X.shape[1])
        return self

    def predict_proba(self, X):
        scores = 1 / (1 + np.exp(-(X @ (self._w if self._w is not None else np.zeros(X.shape[1])))))
        return np.column_stack([1 - scores, scores])


class _MockEnsemble:
    def __init__(self, n_models: int = 3, n_features: int = 5):
        self.models = [_MockModel() for _ in range(n_models)]
        self.feature_columns = [f"f{i}" for i in range(n_features)]
        # Pre-fit with random data so predict_proba is callable
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, n_features))
        y = rng.integers(0, 2, 200)
        for m in self.models:
            m.fit(X, y)


# =============================================================================
# CHANGE POINT TESTS
# =============================================================================

class TestDetectChangePoints:
    def test_returns_list(self):
        from features.change_point import detect_change_points
        rets = _returns_series(200)
        result = detect_change_points(rets, penalty=5.0)
        assert isinstance(result, list)

    def test_empty_on_short_series(self):
        from features.change_point import detect_change_points
        rets = _returns_series(8)
        assert detect_change_points(rets) == []

    def test_indices_in_range(self):
        from features.change_point import detect_change_points
        rets = _returns_series(200)
        cps = detect_change_points(rets, penalty=2.0)
        for idx in cps:
            assert 0 <= idx < len(rets)

    def test_detects_shift_at_low_penalty(self):
        """A large structural shift should produce at least one change point."""
        from features.change_point import detect_change_points
        rng = np.random.default_rng(77)
        # Very pronounced shift: tiny std so the mean jump dwarfs variance noise
        pre  = pd.Series(rng.normal(0.00, 0.001, 100))
        post = pd.Series(rng.normal(0.10, 0.001, 100))
        rets = pd.concat([pre, post], ignore_index=True)
        cps = detect_change_points(rets, penalty=1.0)
        assert len(cps) >= 1

    def test_high_penalty_gives_few_cps(self):
        from features.change_point import detect_change_points
        rets = _returns_series(300)
        cps_hi  = detect_change_points(rets, penalty=100.0)
        cps_lo  = detect_change_points(rets, penalty=1.0)
        assert len(cps_hi) <= len(cps_lo)

    def test_sorted_output(self):
        from features.change_point import detect_change_points
        rets = _returns_series(300)
        cps = detect_change_points(rets, penalty=2.0)
        assert cps == sorted(cps)


class TestCusumChangePoints:
    def test_returns_series(self):
        from features.change_point import cusum_change_points
        rets = _returns_series(200)
        result = cusum_change_points(rets, threshold=4.0)
        assert isinstance(result, pd.Series)
        assert len(result) == len(rets)

    def test_dtype_bool(self):
        from features.change_point import cusum_change_points
        rets = _returns_series(100)
        result = cusum_change_points(rets)
        assert result.dtype == bool

    def test_alarm_on_large_shift(self):
        from features.change_point import cusum_change_points
        rng = np.random.default_rng(0)
        normal_part = pd.Series(rng.normal(0, 0.001, 100))
        shock_part  = pd.Series(rng.normal(0.2, 0.001, 50))
        rets = pd.concat([normal_part, shock_part], ignore_index=True)
        alarm = cusum_change_points(rets, threshold=2.0)
        assert alarm.any(), "Expected at least one alarm for large mean shift"

    def test_same_length_as_input(self):
        from features.change_point import cusum_change_points
        rets = _returns_series(150)
        result = cusum_change_points(rets)
        assert len(result) == 150

    def test_handles_all_nan(self):
        from features.change_point import cusum_change_points
        rets = pd.Series([np.nan] * 50)
        result = cusum_change_points(rets)
        assert len(result) == 50


class TestBayesianChangeProbabilities:
    def test_returns_series(self):
        from features.change_point import bayesian_change_point_probs
        rets = _returns_series(100)
        result = bayesian_change_point_probs(rets, hazard_rate=0.02)
        assert isinstance(result, pd.Series)
        assert len(result) == len(rets)

    def test_probabilities_in_unit_interval(self):
        from features.change_point import bayesian_change_point_probs
        rets = _returns_series(100)
        probs = bayesian_change_point_probs(rets)
        assert (probs >= 0).all() and (probs <= 1.01).all()

    def test_name_is_cp_prob(self):
        from features.change_point import bayesian_change_point_probs
        rets = _returns_series(50)
        probs = bayesian_change_point_probs(rets)
        assert probs.name == "cp_prob"


class TestAddChangePointFeatures:
    def test_adds_three_columns(self):
        from features.change_point import add_change_point_features
        df = _ohlcv_df(200)
        out = add_change_point_features(df, col="close")
        for col in ["cp_pelt", "cp_cusum", "cp_prob"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_modification_if_col_missing(self):
        from features.change_point import add_change_point_features
        df = _ohlcv_df(100)
        out = add_change_point_features(df, col="nonexistent")
        assert "cp_pelt" not in out.columns

    def test_cp_prob_is_float(self):
        from features.change_point import add_change_point_features
        df = _ohlcv_df(150)
        out = add_change_point_features(df, col="close")
        assert out["cp_prob"].dtype == float

    def test_cp_pelt_is_binary(self):
        from features.change_point import add_change_point_features
        df = _ohlcv_df(150)
        out = add_change_point_features(df, col="close")
        assert set(out["cp_pelt"].unique()).issubset({0, 1})


# =============================================================================
# MULTI TIMEFRAME TESTS
# =============================================================================

class TestMultiTimeframeConfirmation:
    def test_compute_all_returns_dataframe(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21, 63))
        close = _price_series(200)
        result = mtf.compute_all(close)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"mtf_5", "mtf_21", "mtf_63"}

    def test_signals_are_minus_one_zero_or_one(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21))
        close = _price_series(200)
        result = mtf.compute_all(close)
        for col in result.columns:
            assert set(result[col].unique()).issubset({-1, 0, 1})

    def test_confirm_returns_mtf_result(self):
        from features.multi_timeframe import MultiTimeframeConfirmation, MTFResult
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21))
        close = _price_series(200)
        r = mtf.confirm(close, base_signal=1)
        assert isinstance(r, MTFResult)
        assert r.signal in (-1, 0, 1)
        assert 0.0 <= r.confidence <= 1.0

    def test_confirm_with_require_all(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21), require_all=True)
        close = _price_series(200)
        r = mtf.confirm(close, base_signal=1)
        # require_all: confidence must be exactly 1.0 for confirmed=True
        if r.agree:
            assert r.confidence == 1.0

    def test_add_features_columns(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21, 63))
        df = _ohlcv_df(200)
        out = mtf.add_features(df, close_col="close")
        for col in ["mtf_5", "mtf_21", "mtf_63", "mtf_agree", "mtf_confirmed"]:
            assert col in out.columns, f"Missing: {col}"

    def test_add_features_agree_in_0_1(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21))
        df = _ohlcv_df(200)
        out = mtf.add_features(df)
        assert (out["mtf_agree"] >= 0).all() and (out["mtf_agree"] <= 1).all()

    def test_add_features_missing_col(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21))
        df = _ohlcv_df(100).drop(columns=["close"])
        out = mtf.add_features(df, close_col="close")
        assert "mtf_agree" not in out.columns

    def test_confirm_short_series_returns_zero(self):
        from features.multi_timeframe import MultiTimeframeConfirmation
        mtf = MultiTimeframeConfirmation(timeframes=(5, 21))
        close = _price_series(3)
        r = mtf.confirm(close, base_signal=1)
        assert r.signal in (-1, 0, 1)


# =============================================================================
# ONLINE LEARNER TESTS
# =============================================================================

def _make_xy(n: int = 250, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    return X, y



# ===========================================================================
# harness/agents/ — agent class interface tests
# ===========================================================================

class TestHarnessAgentInterfaces:
    """Verify agent classes expose the expected interface for tool-use orchestration."""

    def test_base_agent_has_add_user(self):
        from harness.agents.base import BaseAgent
        assert hasattr(BaseAgent, "add_user")

    def test_base_agent_has_add_assistant(self):
        from harness.agents.base import BaseAgent
        assert hasattr(BaseAgent, "add_assistant")

    def test_strategist_has_propose_experiment(self):
        from harness.agents.strategist import StrategistAgent
        assert hasattr(StrategistAgent, "propose_experiment")

    def test_strategist_has_synthesise(self):
        from harness.agents.strategist import StrategistAgent
        assert hasattr(StrategistAgent, "synthesise")

    def test_strategist_has_generate_research_plan(self):
        from harness.agents.strategist import StrategistAgent
        assert hasattr(StrategistAgent, "generate_research_plan")

    def test_analyst_has_call_method(self):
        from harness.agents.analyst import AnalystAgent
        assert hasattr(AnalystAgent, "call")

    def test_coder_is_claude_based(self):
        import inspect
        from harness.agents import coder
        src = inspect.getsource(coder)
        assert "claude" in src.lower() or "anthropic" in src.lower()

    def test_coder_look_ahead_rules_enforced(self):
        """CoderAgent source must reference look-ahead prevention rules."""
        import inspect
        from harness.agents import coder
        src = inspect.getsource(coder)
        assert "look" in src.lower() or "shift" in src.lower()

    def test_reviewer_has_call_method(self):
        from harness.agents.reviewer import ReviewerAgent
        assert hasattr(ReviewerAgent, "call")

    def test_reviewer_evaluates_sharpe(self):
        import inspect
        from harness.agents import reviewer
        src = inspect.getsource(reviewer)
        assert "sharpe" in src.lower()

    def test_all_agents_import_from_harness_base(self):
        from harness.agents.strategist import StrategistAgent
        from harness.agents.analyst import AnalystAgent
        from harness.agents.coder import CoderAgent
        from harness.agents.reviewer import ReviewerAgent
        from harness.agents.base import BaseAgent
        # Verify they share common base or at least all importable together
        assert all(cls is not None for cls in [StrategistAgent, AnalystAgent, CoderAgent, ReviewerAgent, BaseAgent])


# ===========================================================================
# harness/memory/knowledge_base.py — entry lifecycle
# ===========================================================================

class TestKBEntryLifecycle:
    """Integration-level tests: write entries, read them back, check summary."""

    def _fresh_kb(self, tmp_path):
        import harness.memory.knowledge_base as kb_mod
        import harness.config as cfg_mod
        original = cfg_mod.MEMORY_DIR
        kb_dir = tmp_path / "lifecycle_kb"
        kb_dir.mkdir()
        cfg_mod.MEMORY_DIR = kb_dir
        kb = kb_mod.KnowledgeBase.__new__(kb_mod.KnowledgeBase)
        kb.session_id = "lifecycle_test"
        kb._dir = kb_dir
        kb._index_path = kb._dir / "index.json"
        kb._index = []
        cfg_mod.MEMORY_DIR = original
        return kb

    def test_experiment_promote_roundtrip(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save_experiment(
            "Momentum 20-bar", {"horizon": 5},
            {"sharpe": 1.2, "oos_sharpe": 0.9, "max_dd_pct": 8.0},
            "PROMOTE",
        )
        kb.save_promotion("Momentum_20d", {}, {"sharpe": 0.9, "max_dd": 8.0})
        promos = kb.get_promotions()
        assert len(promos) == 1
        assert promos[0]["body"]["strategy_name"] == "Momentum_20d"

    def test_failure_appears_in_context_summary(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save_failure("MACD long windows", "Lags too much in regime transitions")
        summary = kb.context_summary()
        assert "Dead-End" in summary or "MACD" in summary

    def test_heuristic_appears_in_context_summary(self, tmp_path):
        kb = self._fresh_kb(tmp_path)
        kb.save_heuristic("Use regime filter", "Reduces false positives in bear markets")
        summary = kb.context_summary()
        assert "Heuristic" in summary or "regime" in summary.lower()

    def test_multiple_sessions_accumulate_in_index(self, tmp_path):
        import harness.memory.knowledge_base as kb_mod
        import harness.config as cfg_mod
        original = cfg_mod.MEMORY_DIR
        kb_dir = tmp_path / "multi_session"
        kb_dir.mkdir()
        cfg_mod.MEMORY_DIR = kb_dir

        # Session 1
        kb1 = kb_mod.KnowledgeBase.__new__(kb_mod.KnowledgeBase)
        kb1.session_id = "session_1"
        kb1._dir = kb_dir
        kb1._index_path = kb_dir / "index.json"
        kb1._index = []
        kb1.save_experiment("Exp A", {}, {"sharpe": 0.6, "oos_sharpe": 0.6}, "ITERATE")

        # Session 2 — reads index from disk
        kb2 = kb_mod.KnowledgeBase.__new__(kb_mod.KnowledgeBase)
        kb2.session_id = "session_2"
        kb2._dir = kb_dir
        kb2._index_path = kb_dir / "index.json"
        kb2._index = kb2._load_index()
        kb2.save_experiment("Exp B", {}, {"sharpe": 0.8, "oos_sharpe": 0.8}, "PROMOTE")

        assert len(kb2._index) == 2
        cfg_mod.MEMORY_DIR = original


# ---------------------------------------------------------------------------
# ── RL Experiment Bandit (UCB1) ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestExperimentBandit:
    """Tests for harness/rl_bandit.py — ExperimentBandit UCB1 implementation."""

    @pytest.fixture
    def bandit(self, tmp_path):
        from harness.rl_bandit import ExperimentBandit
        return ExperimentBandit(state_path=tmp_path / "bandit.json", exploration_c=1.0)

    def test_importable(self):
        from harness.rl_bandit import ExperimentBandit, ARMS, ARM_NAMES
        assert len(ARMS) >= 6
        assert len(ARM_NAMES) == len(ARMS)

    def test_select_arm_returns_valid_arm(self, bandit):
        from harness.rl_bandit import ARM_NAMES
        arm = bandit.select_arm()
        assert arm in ARM_NAMES

    def test_unvisited_arms_selected_first(self, bandit):
        from harness.rl_bandit import ARM_NAMES
        # All arms start unvisited; selection should return a valid arm
        arm = bandit.select_arm()
        s = bandit._state["arms"][arm]
        assert s["n"] == 0, "First selection should pick an unvisited arm"

    def test_update_increments_trial_count(self, bandit):
        arm = bandit.select_arm()
        bandit.update(arm, 0.75)
        assert bandit._state["arms"][arm]["n"] == 1
        assert bandit._state["total_trials"] == 1

    def test_update_records_reward(self, bandit):
        arm = bandit.select_arm()
        bandit.update(arm, 0.90)
        s = bandit._state["arms"][arm]
        assert abs(s["sum_reward"] - 0.90) < 1e-9
        assert abs(s["best"] - 0.90) < 1e-9

    def test_get_guidance_has_required_keys(self, bandit):
        guidance = bandit.get_guidance(iteration=1)
        for key in ("arm_name", "features", "model_params", "signal_threshold", "rationale"):
            assert key in guidance, f"Missing key: {key}"

    def test_get_guidance_features_is_list(self, bandit):
        guidance = bandit.get_guidance()
        assert isinstance(guidance["features"], list)
        assert len(guidance["features"]) >= 3

    def test_get_guidance_model_params_is_dict(self, bandit):
        guidance = bandit.get_guidance()
        mp = guidance["model_params"]
        assert isinstance(mp, dict)
        assert "n_estimators" in mp

    def test_ucb_prefers_high_reward_arm(self, tmp_path):
        from harness.rl_bandit import ExperimentBandit, ARM_NAMES
        # Use UCB1 explicitly — selection is deterministic when exploration_c is tiny
        bandit = ExperimentBandit(state_path=tmp_path / "b2.json", exploration_c=0.01, algorithm="ucb1")
        # Give all arms 1 trial so UCB is based on mean reward
        for arm in ARM_NAMES:
            bandit.update(arm, 0.1)
        # Now give one arm a very high reward
        target = ARM_NAMES[2]
        bandit.update(target, 2.0)
        selected = bandit.select_arm()
        assert selected == target, "UCB should prefer arm with highest mean reward when c is tiny"

    def test_persistence_across_instances(self, tmp_path):
        from harness.rl_bandit import ExperimentBandit
        state_path = tmp_path / "persist.json"
        b1 = ExperimentBandit(state_path=state_path)
        arm = b1.select_arm()
        b1.update(arm, 0.55)

        b2 = ExperimentBandit(state_path=state_path)
        assert b2._state["total_trials"] == 1
        assert b2._state["arms"][arm]["n"] == 1

    def test_bootstrap_from_empty_kb(self, tmp_path):
        from harness.rl_bandit import ExperimentBandit
        from unittest.mock import MagicMock
        bandit = ExperimentBandit(state_path=tmp_path / "b3.json")
        mock_kb = MagicMock()
        mock_kb.search.return_value = []
        n = bandit.bootstrap_from_kb(mock_kb)
        assert n == 0

    def test_bootstrap_from_kb_with_experiments(self, tmp_path):
        from harness.rl_bandit import ExperimentBandit
        from unittest.mock import MagicMock
        bandit = ExperimentBandit(state_path=tmp_path / "b4.json")
        mock_kb = MagicMock()
        mock_kb.search.return_value = [
            {"config": {"features": ["mom_12_1", "mom_5d", "sma_50_above_200"]},
             "results": {"sharpe": 0.7}},
            {"config": {"features": ["rsi_14", "bb_width", "vol_21d"]},
             "results": {"sharpe": 0.4}},
        ]
        n = bandit.bootstrap_from_kb(mock_kb)
        assert n >= 1  # at least one entry matched an arm
        assert bandit._state["total_trials"] >= 1

    def test_stats_summary_returns_string(self, bandit):
        summary = bandit.stats_summary()
        assert isinstance(summary, str)
        assert "Bandit" in summary and "total trials" in summary

    def test_infer_arm_returns_valid_or_none(self, bandit):
        from harness.rl_bandit import ARM_NAMES
        arm = bandit._infer_arm({"mom_12_1", "mom_5d", "sma_50_above_200"})
        assert arm is None or arm in ARM_NAMES

    def test_make_bandit_factory(self, tmp_path, monkeypatch):
        import harness.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "MEMORY_DIR", tmp_path)
        from harness.rl_bandit import make_bandit
        b = make_bandit(exploration_c=0.5)
        assert b.exploration_c == 0.5
        assert b.state_path.parent == tmp_path


# ---------------------------------------------------------------------------
# ── Web Search Tool ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestWebSearchTool:
    """Tests for the web_search tool in harness/tools/executor.py and registry."""

    def test_web_search_in_claude_tools(self):
        from harness.tools.registry import CLAUDE_TOOLS
        names = [t["name"] for t in CLAUDE_TOOLS]
        assert "web_search" in names

    def test_web_search_in_grok_tools(self):
        from harness.tools.registry import GROK_TOOLS
        names = [t["function"]["name"] for t in GROK_TOOLS]
        assert "web_search" in names

    def test_web_search_schema_has_query_required(self):
        from harness.tools.registry import CLAUDE_TOOLS
        tool = next(t for t in CLAUDE_TOOLS if t["name"] == "web_search")
        assert "query" in tool["input_schema"]["required"]

    def test_web_search_executor_offline_fallback(self, tmp_path):
        from harness.tools.executor import ToolExecutor
        import json, unittest.mock as um
        exec_ = ToolExecutor.__new__(ToolExecutor)
        exec_.kb = um.MagicMock()
        # Force offline fallback by mocking urlopen to raise
        with um.patch("urllib.request.urlopen", side_effect=OSError("no network")):
            result_str = exec_.execute("web_search", {"query": "momentum factor research"})
        result = json.loads(result_str)
        assert result.get("status") == "offline_fallback"
        assert len(result.get("results", [])) > 0

    def test_offline_fallback_momentum(self):
        from harness.tools.executor import _offline_fallback
        r = _offline_fallback("momentum factor equity")
        assert "momentum" in r["text"].lower()

    def test_offline_fallback_mean_reversion(self):
        from harness.tools.executor import _offline_fallback
        r = _offline_fallback("rsi mean reversion strategy")
        assert "reversion" in r["text"].lower() or "rsi" in r["text"].lower()

    def test_offline_fallback_regime(self):
        from harness.tools.executor import _offline_fallback
        r = _offline_fallback("market regime bull bear vix")
        assert "regime" in r["text"].lower() or "bull" in r["text"].lower()

    def test_offline_fallback_generic(self):
        from harness.tools.executor import _offline_fallback
        r = _offline_fallback("some random query about trading strategies")
        assert isinstance(r["text"], str) and len(r["text"]) > 50


# ---------------------------------------------------------------------------
# ── Demo Mode ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Integration tests for harness/demo_mode.py stub agents."""

    def test_demo_analyst_returns_string(self):
        from harness.demo_mode import DemoAnalystAgent
        agent = DemoAnalystAgent()
        result = agent.analyze_market("SPY", "2018-2023")
        assert isinstance(result, str) and len(result) > 50

    def test_demo_analyst_call_returns_string(self):
        from harness.demo_mode import DemoAnalystAgent
        agent = DemoAnalystAgent()
        result = agent.call("analyse SPY")
        assert isinstance(result, str) and len(result) > 20

    def test_demo_strategist_propose_returns_json_block(self):
        from harness.demo_mode import DemoStrategistAgent
        agent = DemoStrategistAgent()
        proposal = agent.propose_experiment("market context", "kb summary")
        assert "hypothesis" in proposal.lower()

    def test_demo_strategist_cycles_proposals(self):
        from harness.demo_mode import DemoStrategistAgent
        DemoStrategistAgent._proposal_idx = 0
        agent = DemoStrategistAgent()
        p1 = agent.propose_experiment("ctx", "kb")
        p2 = agent.propose_experiment("ctx", "kb")
        p3 = agent.propose_experiment("ctx", "kb")
        # After 3 proposals, cycle wraps
        p4 = agent.propose_experiment("ctx", "kb")
        assert p1 == p4

    def test_demo_strategist_synthesise_promote(self):
        from harness.demo_mode import DemoStrategistAgent
        agent = DemoStrategistAgent()
        review = "OOS Sharpe: 0.85\nMax Drawdown: 12.0%\nVerdict: PROMOTE"
        result = agent.synthesise(review, 1)
        assert "PROMOTE" in result

    def test_demo_strategist_synthesise_iterate(self):
        from harness.demo_mode import DemoStrategistAgent
        agent = DemoStrategistAgent()
        review = "OOS Sharpe: 0.55\nMax Drawdown: 18.0%\nVerdict: ITERATE"
        result = agent.synthesise(review, 1)
        assert "ITERATE" in result

    def test_demo_strategist_synthesise_reject(self):
        from harness.demo_mode import DemoStrategistAgent
        agent = DemoStrategistAgent()
        review = "OOS Sharpe: 0.10\nMax Drawdown: 30.0%\nVerdict: REJECT"
        result = agent.synthesise(review, 1)
        assert "REJECT" in result

    def test_demo_reviewer_analyze_returns_verdict(self):
        from harness.demo_mode import DemoReviewerAgent
        agent = DemoReviewerAgent()
        result = agent.analyze(
            hypothesis="test",
            config={},
            backtest_results={"sharpe": 0.85, "max_dd": -12.0},
            iteration=1,
        )
        assert "PROMOTE" in result

    def test_demo_coder_generate_factor_no_lookahead(self):
        from harness.demo_mode import DemoCoderAgent
        agent = DemoCoderAgent()
        result = agent.generate_factor("test_factor", "price trend", "")
        assert "look-ahead" in result.lower() or "no .shift(-" in result

    def test_extract_float_helper(self):
        from harness.demo_mode import _extract_float
        text = "Results: OOS Sharpe: 0.75, Max Drawdown: 18.5%"
        assert abs(_extract_float(text, "OOS Sharpe:", 0.0) - 0.75) < 1e-9
        assert abs(_extract_float(text, "Max Drawdown:", 0.0) - 18.5) < 1e-9

    def test_extract_float_missing_label(self):
        from harness.demo_mode import _extract_float
        assert _extract_float("no label here", "OOS Sharpe:", 0.42) == 0.42

    def test_demo_research_plan_contains_ticker(self):
        from harness.demo_mode import DemoStrategistAgent
        agent = DemoStrategistAgent()
        plan = agent.generate_research_plan("AAPL", "OOS Sharpe > 0.8", "")
        assert "AAPL" in plan

    def test_build_demo_harness_importable(self):
        from harness.demo_mode import build_demo_harness
        assert callable(build_demo_harness)


# ---------------------------------------------------------------------------
# ── Session Report ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestSessionReport:
    """Tests for harness/session_report.py markdown report generator."""

    @pytest.fixture
    def sample_log(self):
        return [
            {
                "iteration": 1,
                "config": {"hypothesis": "Momentum test", "_bandit_arm": "momentum_medium",
                           "features": ["mom_12_1", "sma_50_above_200"], "signal_threshold": 0.55,
                           "start": "2018-01-01", "end": "2023-12-31", "model_params": {}},
                "backtest": {"sharpe": 0.72, "max_dd": -18.0},
                "sharpe": 0.72,
                "promoted": False,
                "elapsed_s": 45.2,
            },
            {
                "iteration": 2,
                "config": {"hypothesis": "RSI mean reversion", "_bandit_arm": "mean_reversion_rsi",
                           "features": ["rsi_14", "bb_width"], "signal_threshold": 0.57,
                           "start": "2018-01-01", "end": "2023-12-31", "model_params": {}},
                "backtest": {"sharpe": 0.85, "max_dd": -12.0},
                "sharpe": 0.85,
                "promoted": True,
                "elapsed_s": 52.1,
            },
        ]

    @pytest.fixture
    def sample_promotions(self, sample_log):
        return [sample_log[1]]

    @pytest.fixture
    def sample_kb_stats(self):
        return {"total": 5, "by_type": {"experiment": 4, "promotion": 1}}

    def test_generate_returns_path(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log,
            promotions=sample_promotions,
            kb_stats=sample_kb_stats,
            bandit_summary="Arm stats here",
            ticker="SPY",
            session_id="test_001",
            goal="OOS Sharpe > 0.8",
            output_dir=tmp_path,
        )
        assert path.exists()
        assert path.suffix == ".md"

    def test_report_contains_session_id(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_xyz", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "sess_xyz" in content

    def test_report_contains_ticker(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="AAPL", session_id="sess_t", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "AAPL" in content

    def test_report_has_iteration_table(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_table", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## Iteration Results" in content
        assert "Sharpe" in content

    def test_report_promoted_section(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_promo", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## Promoted Strategies" in content
        assert "RSI mean reversion" in content

    def test_report_no_promotions_message(self, tmp_path, sample_log, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=[],
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_nopromo", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "No strategies met" in content or "No strategy met" in content

    def test_report_sharpe_convergence(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_conv", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## Sharpe Convergence" in content
        assert "best=" in content

    def test_report_bandit_section(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="arm=momentum_medium n=2 avg=0.72",
            ticker="SPY", session_id="sess_bandit", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## RL Bandit Learning" in content
        assert "momentum_medium" in content

    def test_report_kb_state_section(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_kb", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## Knowledge Base State" in content
        assert "Total entries" in content

    def test_report_recommendations(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_reco", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "## Recommendations" in content

    def test_report_executive_summary_best_sharpe(self, tmp_path, sample_log, sample_promotions, sample_kb_stats):
        from harness import session_report
        path = session_report.generate(
            session_log=sample_log, promotions=sample_promotions,
            kb_stats=sample_kb_stats, bandit_summary="",
            ticker="SPY", session_id="sess_exec", output_dir=tmp_path,
        )
        content = path.read_text(encoding="utf-8")
        assert "0.850" in content  # best sharpe 0.85


# ---------------------------------------------------------------------------
# ── KB context_summary with bandit ───────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestHarnessDashboard:
    """Structural tests for harness/harness_dashboard.py (no Streamlit runtime required)."""

    def test_dashboard_file_exists(self):
        from pathlib import Path
        p = Path(__file__).parent.parent / "harness" / "harness_dashboard.py"
        assert p.exists(), "harness_dashboard.py must exist"

    def test_dashboard_syntax(self):
        import py_compile
        from pathlib import Path
        p = Path(__file__).parent.parent / "harness" / "harness_dashboard.py"
        py_compile.compile(str(p), doraise=True)

    def test_dashboard_imports_config(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "RESULTS_DIR" in src
        assert "MEMORY_DIR" in src

    def test_dashboard_has_five_tabs(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        tabs = ["Overview", "Sessions", "Knowledge Base", "Bandit", "Reports"]
        for tab in tabs:
            assert tab in src, f"Tab '{tab}' not found in dashboard"

    def test_dashboard_has_run_demo_tab(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "Run Demo" in src

    def test_dashboard_loads_sessions(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "load_sessions" in src

    def test_dashboard_loads_kb(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "load_kb_index" in src

    def test_dashboard_loads_bandit(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "load_bandit_state" in src

    def test_dashboard_promote_threshold_reference(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "PROMOTE_SHARPE_THRESHOLD" in src

    def test_dashboard_simulation_only_note(self):
        src = (
            __import__("pathlib").Path(__file__).parent.parent
            / "harness" / "harness_dashboard.py"
        ).read_text(encoding="utf-8")
        assert "no real money" in src.lower() or "simulation only" in src.lower()


class TestKBFindSimilar:
    """Tests for KnowledgeBase.find_similar() deduplication method."""

    @pytest.fixture
    def fresh_kb(self, tmp_path, monkeypatch):
        import harness.memory.knowledge_base as kb_mod
        monkeypatch.setattr(kb_mod, "MEMORY_DIR", tmp_path)
        from harness.memory.knowledge_base import KnowledgeBase
        return KnowledgeBase()

    def test_find_similar_returns_list(self, fresh_kb):
        result = fresh_kb.find_similar(["mom_12_1", "sma_50_above_200"])
        assert isinstance(result, list)

    def test_find_similar_empty_features_returns_empty(self, fresh_kb):
        result = fresh_kb.find_similar([])
        assert result == []

    def test_find_similar_finds_matching_experiment(self, fresh_kb):
        fresh_kb.save_experiment(
            hypothesis="momentum test",
            config={"features": ["mom_12_1", "sma_50_above_200", "vol_21d"]},
            results={"sharpe": 0.6},
            verdict="MODERATE",
        )
        result = fresh_kb.find_similar(["mom_12_1", "sma_50_above_200", "vol_21d", "rsi_14"])
        assert len(result) >= 1
        assert any("momentum test" in r.get("title", "") for r in result)

    def test_find_similar_ignores_low_overlap(self, fresh_kb):
        fresh_kb.save_experiment(
            hypothesis="unrelated test",
            config={"features": ["atr_14", "dollar_volume"]},
            results={"sharpe": 0.3},
            verdict="WEAK",
        )
        # Query with completely different features → Jaccard < 0.5
        result = fresh_kb.find_similar(["mom_12_1", "sma_50_above_200", "rsi_14", "bb_width"])
        assert result == []

    def test_find_similar_top_n_limit(self, fresh_kb):
        feats = ["mom_12_1", "sma_50_above_200", "vol_21d"]
        for i in range(5):
            fresh_kb.save_experiment(
                hypothesis=f"exp {i}",
                config={"features": feats},
                results={"sharpe": 0.5},
                verdict="MODERATE",
            )
        result = fresh_kb.find_similar(feats, top_n=2)
        assert len(result) <= 2


class TestKBContextSummaryWithBandit:
    """Tests that context_summary correctly injects bandit intelligence."""

    def test_context_summary_without_bandit(self, tmp_path, monkeypatch):
        import harness.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "MEMORY_DIR", tmp_path)
        from harness.memory.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        summary = kb.context_summary()
        assert isinstance(summary, str)

    def test_context_summary_with_bandit_no_trials(self, tmp_path, monkeypatch):
        import harness.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "MEMORY_DIR", tmp_path)
        from harness.memory.knowledge_base import KnowledgeBase
        from harness.rl_bandit import ExperimentBandit
        kb = KnowledgeBase()
        bandit = ExperimentBandit(state_path=tmp_path / "b.json")
        summary = kb.context_summary(bandit=bandit)
        # No trials yet — bandit section should be absent
        assert "RL Bandit" not in summary

    def test_context_summary_with_bandit_shows_top_arms(self, tmp_path, monkeypatch):
        import harness.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "MEMORY_DIR", tmp_path)
        from harness.memory.knowledge_base import KnowledgeBase
        from harness.rl_bandit import ExperimentBandit, ARM_NAMES
        kb = KnowledgeBase()
        bandit = ExperimentBandit(state_path=tmp_path / "b2.json")
        # Give some arms rewards
        bandit.update(ARM_NAMES[0], 0.9)
        bandit.update(ARM_NAMES[1], 0.3)
        summary = kb.context_summary(bandit=bandit)
        assert "RL Bandit" in summary
        assert ARM_NAMES[0] in summary


# ---------------------------------------------------------------------------
# ── Backtest Tool Executor ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class TestRunBacktestTool:
    """Tests for ToolExecutor._tool_run_backtest — mocked to avoid I/O."""

    def _make_executor(self):
        """Return a ToolExecutor with a mocked KB (no disk I/O)."""
        from harness.tools.executor import ToolExecutor
        import unittest.mock as um
        ex = ToolExecutor.__new__(ToolExecutor)
        ex.kb = um.MagicMock()
        return ex

    def _fake_features(self, n: int = 200):
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        return pd.DataFrame(
            np.random.default_rng(0).standard_normal((n, 5)),
            index=idx,
            columns=["f1", "f2", "f3", "f4", "f5"],
        )

    def test_missing_model_returns_error(self, tmp_path):
        """When model file does not exist, result must contain 'error' key."""
        ex = self._make_executor()
        import unittest.mock as um
        feats = self._fake_features()
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"):  # prevent real download attempt
            result = ex._tool_run_backtest(ticker="FAKE", start="2018-01-01", end="2020-12-31")
        assert "error" in result

    def test_result_has_required_keys(self, tmp_path):
        """Successful backtest returns all expected output keys."""
        ex = self._make_executor()
        import unittest.mock as um
        import numpy as _np
        feats = self._fake_features(200)
        mock_model = um.MagicMock()
        mock_model.feature_columns = list(feats.columns)
        mock_model.predict.return_value = (_np.full(200, 0.6), None, None, None)
        mock_result = um.MagicMock()
        mock_result.metrics = {
            "sharpe_ratio": 0.72, "max_drawdown": 0.18,
            "ann_return_pct": 12.5, "total_trades": 10,
            "win_rate": 0.6, "total_costs_pct": 0.05,
        }
        model_path = tmp_path / "models" / "artifacts" / "spy_model.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"fake")
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"), \
             um.patch("joblib.load", return_value=mock_model), \
             um.patch("backtest.engine.run_backtest", return_value=mock_result), \
             um.patch("backtest.engine.CostModel") as mock_cm:
            mock_cm.for_stock.return_value = um.MagicMock()
            result = ex._tool_run_backtest(ticker="SPY", start="2018-01-01", end="2020-12-31")
        for key in ("sharpe", "ann_return", "max_dd", "n_trades", "win_rate",
                    "cost_drag", "ticker", "period", "signal_threshold"):
            assert key in result, f"Missing key: {key}"

    def test_metric_key_mapping_sharpe_ratio(self, tmp_path):
        """BacktestResult.metrics['sharpe_ratio'] must map to result['sharpe']."""
        ex = self._make_executor()
        import unittest.mock as um
        import numpy as _np
        feats = self._fake_features(200)
        mock_model = um.MagicMock()
        mock_model.feature_columns = list(feats.columns)
        mock_model.predict.return_value = (_np.full(200, 0.7), None, None, None)
        mock_result = um.MagicMock()
        mock_result.metrics = {"sharpe_ratio": 1.23, "max_drawdown": 0.0,
                               "total_trades": 0, "win_rate": 0.0, "total_costs_pct": 0.0}
        model_path = tmp_path / "models" / "artifacts" / "spy_model.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"x")
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"), \
             um.patch("joblib.load", return_value=mock_model), \
             um.patch("backtest.engine.run_backtest", return_value=mock_result), \
             um.patch("backtest.engine.CostModel") as mock_cm:
            mock_cm.for_stock.return_value = um.MagicMock()
            result = ex._tool_run_backtest(ticker="SPY")
        assert abs(result["sharpe"] - 1.23) < 1e-6

    def test_predict_tuple_unpacking(self, tmp_path):
        """When predict() returns a 4-tuple, index [0] is used as probabilities."""
        ex = self._make_executor()
        import unittest.mock as um
        import numpy as _np
        feats = self._fake_features(200)
        mock_model = um.MagicMock()
        mock_model.feature_columns = list(feats.columns)
        high_probas = _np.full(200, 0.9)   # all above any threshold → all signals = 1
        mock_model.predict.return_value = (high_probas, _np.zeros(200), None, None)
        mock_result = um.MagicMock()
        mock_result.metrics = {"sharpe_ratio": 0.5, "max_drawdown": 0.0,
                               "total_trades": 50, "win_rate": 0.5, "total_costs_pct": 0.0}
        model_path = tmp_path / "models" / "artifacts" / "spy_model.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"x")
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"), \
             um.patch("joblib.load", return_value=mock_model), \
             um.patch("backtest.engine.run_backtest", return_value=mock_result) as mock_bt, \
             um.patch("backtest.engine.CostModel") as mock_cm:
            mock_cm.for_stock.return_value = um.MagicMock()
            ex._tool_run_backtest(ticker="SPY", signal_threshold=0.55)
        # run_backtest was called with a signals series (first positional arg)
        sigs_arg = mock_bt.call_args[0][0]
        assert int(sigs_arg.sum()) == 200  # all 200 bars above 0.9 threshold

    def test_cost_model_called_without_args(self, tmp_path):
        """CostModel.for_stock() must be invoked with no arguments."""
        ex = self._make_executor()
        import unittest.mock as um
        import numpy as _np
        feats = self._fake_features(200)
        mock_model = um.MagicMock()
        mock_model.feature_columns = list(feats.columns)
        mock_model.predict.return_value = (_np.full(200, 0.6), None, None, None)
        mock_result = um.MagicMock()
        mock_result.metrics = {"sharpe_ratio": 0.5, "max_drawdown": 0.0,
                               "total_trades": 5, "win_rate": 0.5, "total_costs_pct": 0.0}
        model_path = tmp_path / "models" / "artifacts" / "spy_model.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"x")
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"), \
             um.patch("joblib.load", return_value=mock_model), \
             um.patch("backtest.engine.run_backtest", return_value=mock_result), \
             um.patch("backtest.engine.CostModel") as mock_cm:
            mock_cm.for_stock.return_value = um.MagicMock()
            ex._tool_run_backtest(ticker="SPY")
        mock_cm.for_stock.assert_called_once_with()  # no arguments

    def test_max_drawdown_scaled_to_percent(self, tmp_path):
        """max_drawdown metric (0–1 fraction) must be multiplied by 100 in result."""
        ex = self._make_executor()
        import unittest.mock as um
        import numpy as _np
        feats = self._fake_features(200)
        mock_model = um.MagicMock()
        mock_model.feature_columns = list(feats.columns)
        mock_model.predict.return_value = (_np.full(200, 0.6), None, None, None)
        mock_result = um.MagicMock()
        mock_result.metrics = {"sharpe_ratio": 0.5, "max_drawdown": 0.25,  # 0–1 fraction
                               "total_trades": 5, "win_rate": 0.5, "total_costs_pct": 0.0}
        model_path = tmp_path / "models" / "artifacts" / "spy_model.joblib"
        model_path.parent.mkdir(parents=True)
        model_path.write_bytes(b"x")
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"), \
             um.patch("joblib.load", return_value=mock_model), \
             um.patch("backtest.engine.run_backtest", return_value=mock_result), \
             um.patch("backtest.engine.CostModel") as mock_cm:
            mock_cm.for_stock.return_value = um.MagicMock()
            result = ex._tool_run_backtest(ticker="SPY")
        assert abs(result["max_dd"] - 25.0) < 0.01  # 0.25 × 100 = 25.0

    def test_insufficient_data_returns_error(self, tmp_path):
        """When filtered feature data has fewer than 50 bars, return error dict."""
        ex = self._make_executor()
        import unittest.mock as um
        feats = self._fake_features(10)  # only 10 bars
        with um.patch("features.feature_cache.get_or_compute_features", return_value=feats), \
             um.patch("harness.tools.executor.ROOT", tmp_path), \
             um.patch("data.ingest.DataIngestion"):
            result = ex._tool_run_backtest(ticker="SPY", start="2018-01-01", end="2018-01-15")
        assert "error" in result
        assert "Insufficient" in result["error"]

    def test_execute_dispatch_run_backtest(self, tmp_path):
        """ToolExecutor.execute('run_backtest', ...) dispatches to _tool_run_backtest."""
        ex = self._make_executor()
        import unittest.mock as um
        import json
        with um.patch.object(ex, "_tool_run_backtest", return_value={"sharpe": 0.5}) as mock_rb:
            result_str = ex.execute("run_backtest", {"ticker": "SPY"})
        mock_rb.assert_called_once_with(ticker="SPY")
        result = json.loads(result_str)
        assert result["sharpe"] == 0.5
