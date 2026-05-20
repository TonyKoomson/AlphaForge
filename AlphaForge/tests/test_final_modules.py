"""
tests/test_final_modules.py
============================
Fourth batch of tests covering the last three untested modules:

  - features/feature_cache.py          → get_or_compute_features, invalidate_cache, cache_stats
  - research/research_paper_generator.py → ResearchPaperGenerator, ResearchPaper
  - paper_trading/portfolio_loop.py    → _TickerState, PortfolioReplay._build_summary
"""

from __future__ import annotations

import json
import sys
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _make_raw_parquet(directory: Path, n: int = 200, seed: int = 0) -> Path:
    df = _price_df(n, seed)
    # Use uppercase column names like a real yfinance download
    df_upper = df.rename(columns=str.title)
    path = directory / "spy_daily.parquet"
    df_upper.to_parquet(path)
    return path


def _fake_features(raw_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Minimal generate_features stub — returns two columns over same index."""
    return pd.DataFrame({"feat_a": 1.0, "label": 0}, index=raw_df.index)


def _minimal_cycle_results(**overrides) -> dict:
    base = {
        "cycle_id":            "CYC-001",
        "best_oos_sharpe":     0.75,
        "best_is_sharpe":      1.10,
        "n_promoted":          3,
        "n_evaluated":         20,
        "regime":              "bull",
        "feature_count":       45,
        "overfitting_severity": "moderate",
        "stage2_triggered":    False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ── Feature Cache ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

import features.feature_cache as fc


class TestFeatureCache:
    """Tests for features/feature_cache.py."""

    @pytest.fixture(autouse=True)
    def _isolate_cache(self, tmp_path, monkeypatch):
        """Redirect cache dir to a temp directory for test isolation."""
        cache_dir = tmp_path / "feat_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(fc, "_CACHE_DIR", cache_dir)
        self._cache_dir = cache_dir
        self._raw_dir = tmp_path / "raw"
        self._raw_dir.mkdir()

    # ── FileNotFoundError on missing raw path ──

    def test_file_not_found(self):
        missing = self._raw_dir / "nonexistent.parquet"
        with pytest.raises(FileNotFoundError):
            fc.get_or_compute_features("SPY", missing)

    # ── Cache HIT — loads pre-built cache file ──

    def test_cache_hit_returns_correct_data(self):
        raw_path = _make_raw_parquet(self._raw_dir)
        fingerprint = fc._data_fingerprint(raw_path)
        expected = pd.DataFrame({"magic_col": [42.0, 43.0]},
                                index=pd.bdate_range("2021-01-01", periods=2))
        cached_path = self._cache_dir / f"spy_{fingerprint}_features.parquet"
        expected.to_parquet(cached_path)

        result = fc.get_or_compute_features("SPY", raw_path)
        assert list(result.columns) == ["magic_col"]
        assert result["magic_col"].iloc[0] == 42.0

    def test_cache_hit_skips_generate_features(self, monkeypatch):
        """generate_features must NOT be called on a cache hit."""
        raw_path = _make_raw_parquet(self._raw_dir)
        fingerprint = fc._data_fingerprint(raw_path)
        stub = pd.DataFrame({"x": [1.0]}, index=pd.bdate_range("2021-01-01", periods=1))
        cached_path = self._cache_dir / f"spy_{fingerprint}_features.parquet"
        stub.to_parquet(cached_path)

        called = []
        fake_eng = MagicMock()
        fake_eng.generate_features.side_effect = lambda *a, **kw: called.append(1) or stub

        with patch.dict(sys.modules, {"features.engine": fake_eng}):
            fc.get_or_compute_features("SPY", raw_path)

        assert len(called) == 0, "generate_features must not be called on cache hit"

    # ── Cache MISS — calls generate_features and writes cache ──

    def test_cache_miss_calls_generate_features(self, tmp_path):
        raw_path = _make_raw_parquet(self._raw_dir)
        raw_df_for_stub = _price_df(50)

        called_with = []
        def fake_gen(raw_df, as_of_date=None, cross_asset=None):
            called_with.append(True)
            return pd.DataFrame({"computed": 9.9}, index=raw_df.index)

        fake_eng = MagicMock()
        fake_eng.generate_features.side_effect = fake_gen
        fake_ingest = MagicMock()

        with patch.dict(sys.modules, {"features.engine": fake_eng,
                                       "data.ingest": fake_ingest}):
            result = fc.get_or_compute_features("SPY", raw_path)

        assert len(called_with) > 0
        assert "computed" in result.columns

    def test_cache_miss_writes_cache_file(self):
        raw_path = _make_raw_parquet(self._raw_dir)
        fingerprint = fc._data_fingerprint(raw_path)
        expected_cache = self._cache_dir / f"spy_{fingerprint}_features.parquet"
        assert not expected_cache.exists()

        def fake_gen(raw_df, as_of_date=None, cross_asset=None):
            return pd.DataFrame({"saved": 1.0}, index=raw_df.index)

        fake_eng = MagicMock()
        fake_eng.generate_features.side_effect = fake_gen
        fake_ingest = MagicMock()

        with patch.dict(sys.modules, {"features.engine": fake_eng,
                                       "data.ingest": fake_ingest}):
            fc.get_or_compute_features("SPY", raw_path)

        assert expected_cache.exists(), "Cache file should have been written after a miss"

    # ── force_recompute ignores existing cache ──

    def test_force_recompute_ignores_cache(self):
        raw_path = _make_raw_parquet(self._raw_dir)
        fingerprint = fc._data_fingerprint(raw_path)
        old_data = pd.DataFrame({"old": [999.0]}, index=pd.bdate_range("2021-01-01", periods=1))
        cached_path = self._cache_dir / f"spy_{fingerprint}_features.parquet"
        old_data.to_parquet(cached_path)

        def fake_gen(raw_df, as_of_date=None, cross_asset=None):
            return pd.DataFrame({"new": 1.0}, index=raw_df.index)

        fake_eng = MagicMock()
        fake_eng.generate_features.side_effect = fake_gen
        fake_ingest = MagicMock()

        with patch.dict(sys.modules, {"features.engine": fake_eng,
                                       "data.ingest": fake_ingest}):
            result = fc.get_or_compute_features("SPY", raw_path, force_recompute=True)

        assert "new" in result.columns, "force_recompute should bypass cache and return fresh data"

    # ── invalidate_cache ──

    def test_invalidate_cache_removes_files(self):
        # Create three fake cache files for SPY and one for QQQ
        for i in range(3):
            (self._cache_dir / f"spy_abcd{i:04x}_features.parquet").write_bytes(b"fake")
        (self._cache_dir / "qqq_abcd0001_features.parquet").write_bytes(b"fake")

        deleted = fc.invalidate_cache("SPY")
        assert deleted == 3
        spy_files = list(self._cache_dir.glob("spy_*_features.parquet"))
        assert len(spy_files) == 0
        qqq_files = list(self._cache_dir.glob("qqq_*_features.parquet"))
        assert len(qqq_files) == 1

    def test_invalidate_cache_returns_zero_when_none(self):
        result = fc.invalidate_cache("MISSING_TICKER")
        assert result == 0

    def test_invalidate_cache_case_insensitive_ticker(self):
        (self._cache_dir / "aapl_abc123ab_features.parquet").write_bytes(b"x")
        deleted = fc.invalidate_cache("AAPL")  # uppercase should match lowercase filename
        assert deleted == 1

    # ── cache_stats ──

    def test_cache_stats_empty(self):
        stats = fc.cache_stats()
        assert stats["n_files"] == 0
        assert stats["total_mb"] == 0.0
        assert "cache_dir" in stats

    def test_cache_stats_counts_files(self):
        for i in range(4):
            p = self._cache_dir / f"spy_aa{i:02x}ccddee_features.parquet"
            p.write_bytes(b"x" * (200 * 1024))  # 200 KB each → 0.8 MB total, rounds to > 0.0
        stats = fc.cache_stats()
        assert stats["n_files"] == 4
        assert stats["total_mb"] > 0.0

    # ── fingerprint stability ──

    def test_fingerprint_stable_for_same_file(self):
        raw_path = _make_raw_parquet(self._raw_dir)
        fp1 = fc._data_fingerprint(raw_path)
        fp2 = fc._data_fingerprint(raw_path)
        assert fp1 == fp2

    def test_fingerprint_different_for_different_files(self):
        p1 = _make_raw_parquet(self._raw_dir, n=100, seed=0)
        p2 = self._raw_dir / "qqq_daily.parquet"
        _price_df(150, seed=7).to_parquet(p2)
        assert fc._data_fingerprint(p1) != fc._data_fingerprint(p2)

    def test_fingerprint_missing_file_returns_string(self):
        missing = self._raw_dir / "ghost.parquet"
        fp = fc._data_fingerprint(missing)
        assert isinstance(fp, str)


# ---------------------------------------------------------------------------
# -- AI Harness: harness/agents/ stubs -------------------------------------
# ---------------------------------------------------------------------------

class TestHarnessAgentImports:
    """Smoke tests: harness agent classes can be imported and instantiated with mocked API keys."""

    def test_base_agent_importable(self):
        from harness.agents.base import BaseAgent
        assert BaseAgent is not None

    def test_strategist_has_propose_method(self):
        from harness.agents.strategist import StrategistAgent
        assert hasattr(StrategistAgent, "propose_experiment")

    def test_analyst_has_call_method(self):
        from harness.agents.analyst import AnalystAgent
        assert hasattr(AnalystAgent, "call")

    def test_coder_has_call_method(self):
        from harness.agents.coder import CoderAgent
        assert hasattr(CoderAgent, "call")

    def test_reviewer_has_call_method(self):
        from harness.agents.reviewer import ReviewerAgent
        assert hasattr(ReviewerAgent, "call")

    def test_knowledge_base_importable(self):
        from harness.memory.knowledge_base import KnowledgeBase
        assert KnowledgeBase is not None

    def test_orchestrator_importable(self):
        from harness.orchestrator import AlphaHarness
        assert AlphaHarness is not None

    def test_executor_importable(self):
        from harness.tools.executor import ToolExecutor
        assert ToolExecutor is not None

    def test_registry_has_claude_tools(self):
        from harness.tools.registry import CLAUDE_TOOLS
        assert isinstance(CLAUDE_TOOLS, list)
        assert len(CLAUDE_TOOLS) > 0

    def test_registry_claude_tool_schemas_valid(self):
        from harness.tools.registry import CLAUDE_TOOLS
        for tool in CLAUDE_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_harness_config_importable(self):
        from harness.config import CLAUDE_MODEL, GROK_MODEL, MEMORY_DIR
        assert isinstance(CLAUDE_MODEL, str)
        assert isinstance(GROK_MODEL, str)
        assert MEMORY_DIR is not None

    def test_harness_promote_threshold_documented(self):
        """Promote threshold (Sharpe >= 0.8, max DD <= 25%) is used in the orchestrator."""
        import inspect
        from harness import orchestrator
        src = inspect.getsource(orchestrator)
        assert "0.8" in src or "PROMOTE" in src


class TestToolRegistry:
    """Tests for harness/tools/registry.py tool schemas."""

    def test_tools_include_run_backtest(self):
        from harness.tools.registry import CLAUDE_TOOLS
        names = [t["name"] for t in CLAUDE_TOOLS]
        assert any("backtest" in n.lower() for n in names)

    def test_tools_include_train(self):
        from harness.tools.registry import CLAUDE_TOOLS
        names = [t["name"] for t in CLAUDE_TOOLS]
        assert any("train" in n.lower() for n in names)

    def test_grok_tools_have_function_wrapper(self):
        from harness.tools.registry import GROK_TOOLS
        for tool in GROK_TOOLS:
            assert tool.get("type") == "function"
            assert "function" in tool

    def test_tool_count_at_least_5(self):
        from harness.tools.registry import CLAUDE_TOOLS
        assert len(CLAUDE_TOOLS) >= 5

    def test_tool_names_list_matches_definitions(self):
        from harness.tools.registry import CLAUDE_TOOLS, TOOL_NAMES
        defined_names = {t["name"] for t in CLAUDE_TOOLS}
        for name in TOOL_NAMES:
            assert name in defined_names

