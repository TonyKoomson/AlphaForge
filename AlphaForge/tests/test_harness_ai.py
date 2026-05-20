"""
Tests for AI-technique improvements to the AlphaForge harness:
  1. harness/stats.py            — PSR, DSR, Sharpe t-stat
  2. harness/rl_bandit.py        — Thompson Sampling arm selection
  3. harness/memory/knowledge_base.py — TF-IDF semantic deduplication
  4. Integration: DSR/PSR in backtest result dict (executor)
"""
from __future__ import annotations

import math
import json
import tempfile
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────────────
# 1.  harness/stats.py — Statistical metrics
# ────────────────────────────────────────────────────────────────────────────

class TestHarnessStats:
    """Bailey & Lopez de Prado (2012) statistical metrics."""

    def test_norm_cdf_at_zero_is_half(self):
        from harness.stats import _norm_cdf
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_norm_cdf_monotone(self):
        from harness.stats import _norm_cdf
        assert _norm_cdf(-1.0) < _norm_cdf(0.0) < _norm_cdf(1.0)

    def test_sharpe_tstat_positive_sharpe_positive_t(self):
        from harness.stats import sharpe_tstat
        result = sharpe_tstat(sharpe=1.0, n_bars=1260)  # 5 years
        assert result["t_stat"] > 0
        assert result["p_value"] < 0.5
        assert result["n_years"] == pytest.approx(5.0, abs=0.1)

    def test_sharpe_tstat_zero_sharpe_p_near_half(self):
        from harness.stats import sharpe_tstat
        result = sharpe_tstat(sharpe=0.0, n_bars=1260)
        assert result["p_value"] == pytest.approx(0.5, abs=0.01)
        assert result["significant_at_05"] is False

    def test_sharpe_tstat_high_sharpe_significant(self):
        from harness.stats import sharpe_tstat
        result = sharpe_tstat(sharpe=2.0, n_bars=1260)
        assert result["significant_at_05"] is True
        assert result["p_value"] < 0.01

    def test_sharpe_tstat_insufficient_data(self):
        from harness.stats import sharpe_tstat
        result = sharpe_tstat(sharpe=1.5, n_bars=5)
        assert result["t_stat"] == 0.0
        assert result["p_value"] == 1.0

    def test_psr_positive_sharpe_above_half(self):
        from harness.stats import probabilistic_sharpe_ratio
        result = probabilistic_sharpe_ratio(sharpe=1.0, n_bars=1260)
        assert result["psr"] > 0.5
        assert result["sr_star"] == 0.0

    def test_psr_negative_sharpe_below_half(self):
        from harness.stats import probabilistic_sharpe_ratio
        result = probabilistic_sharpe_ratio(sharpe=-0.5, n_bars=1260)
        assert result["psr"] < 0.5

    def test_psr_larger_sample_higher_confidence(self):
        from harness.stats import probabilistic_sharpe_ratio
        psr_small = probabilistic_sharpe_ratio(sharpe=0.5, n_bars=252)["psr"]
        psr_large = probabilistic_sharpe_ratio(sharpe=0.5, n_bars=2520)["psr"]
        assert psr_large > psr_small

    def test_dsr_single_trial_equals_psr(self):
        from harness.stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio
        dsr = deflated_sharpe_ratio(sharpe=1.0, n_bars=1260, n_trials=1)
        psr = probabilistic_sharpe_ratio(sharpe=1.0, n_bars=1260)["psr"]
        # With 1 trial the SR* ≈ 0, so DSR should be close to PSR
        assert abs(dsr["dsr"] - psr) < 0.15

    def test_dsr_more_trials_lowers_score(self):
        from harness.stats import deflated_sharpe_ratio
        dsr1  = deflated_sharpe_ratio(sharpe=0.8, n_bars=1260, n_trials=1)["dsr"]
        dsr20 = deflated_sharpe_ratio(sharpe=0.8, n_bars=1260, n_trials=20)["dsr"]
        # After 20 trials the threshold SR* is higher, so DSR should be lower
        assert dsr20 < dsr1

    def test_dsr_interpretation_field_present(self):
        from harness.stats import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(sharpe=0.5, n_bars=1260, n_trials=5)
        assert "interpretation" in result
        assert result["interpretation"] in {
            "passes_multiple_testing_correction",
            "marginal_after_correction",
            "below_multiple_testing_threshold",
            "likely_false_positive",
        }

    def test_dsr_sr_star_increases_with_trials(self):
        from harness.stats import deflated_sharpe_ratio
        star5  = deflated_sharpe_ratio(sharpe=1.0, n_bars=1260, n_trials=5)["sr_star_ann"]
        star50 = deflated_sharpe_ratio(sharpe=1.0, n_bars=1260, n_trials=50)["sr_star_ann"]
        assert star50 > star5

    def test_dsr_result_keys(self):
        from harness.stats import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(sharpe=0.7, n_bars=1260, n_trials=3)
        for key in ("dsr", "sr_star_ann", "n_trials", "interpretation"):
            assert key in result


# ────────────────────────────────────────────────────────────────────────────
# 2.  Thompson Sampling Bandit
# ────────────────────────────────────────────────────────────────────────────

class TestThompsonSamplingBandit:
    """Gaussian Thompson Sampling (Chapelle & Li, 2011)."""

    def _make_bandit(self, tmp_path: Path, algorithm: str = "thompson") -> object:
        from harness.rl_bandit import ExperimentBandit, ARM_NAMES
        b = ExperimentBandit(
            state_path=tmp_path / "bandit_state.json",
            algorithm=algorithm,
        )
        return b

    def test_thompson_visits_all_arms_eventually(self, tmp_path):
        """With no prior data, all arms should be selected within N trials."""
        b = self._make_bandit(tmp_path, algorithm="thompson")
        from harness.rl_bandit import ARM_NAMES
        selected = set()
        for _ in range(100):
            arm = b.select_arm()
            selected.add(arm)
            b.update(arm, 0.1)
        assert len(selected) == len(ARM_NAMES)

    def test_thompson_exploits_high_reward_arm(self, tmp_path):
        """With very low posterior variance, Thompson should prefer the best arm."""
        b = self._make_bandit(tmp_path, algorithm="thompson")
        from harness.rl_bandit import ARM_NAMES
        target = ARM_NAMES[0]
        # Give target arm 30 high-reward observations to collapse posterior variance
        for _ in range(30):
            b.update(target, 2.0)
        # Give other arms 5 low-reward observations each
        for arm in ARM_NAMES[1:]:
            for _ in range(5):
                b.update(arm, 0.1)
        # With tight posterior on target, it should be selected most often
        counts = {arm: 0 for arm in ARM_NAMES}
        for _ in range(200):
            arm = b.select_arm()
            counts[arm] += 1
        assert counts[target] > counts[ARM_NAMES[1]]

    def test_thompson_tracks_sum_sq_reward(self, tmp_path):
        """sum_sq_reward should be updated on each call to update()."""
        b = self._make_bandit(tmp_path, algorithm="thompson")
        from harness.rl_bandit import ARM_NAMES
        arm = ARM_NAMES[0]
        b.update(arm, 1.0)
        b.update(arm, -0.5)
        state = b._state["arms"][arm]
        assert state["sum_sq_reward"] == pytest.approx(1.0 ** 2 + (-0.5) ** 2, abs=1e-9)

    def test_ucb1_still_works(self, tmp_path):
        """UCB1 algorithm should still function after refactor."""
        b = self._make_bandit(tmp_path, algorithm="ucb1")
        from harness.rl_bandit import ARM_NAMES
        # Seed all arms so UCB1 can compute scores (avoid infinite score for unvisited)
        for arm in ARM_NAMES:
            b.update(arm, 0.5)
        # High reward on target
        target = ARM_NAMES[0]
        for _ in range(10):
            b.update(target, 3.0)
        b_low_explore = b.__class__(
            state_path=tmp_path / "bandit_state.json",
            algorithm="ucb1",
            exploration_c=0.01,  # near-greedy
        )
        selected = b_low_explore.select_arm()
        assert selected == target

    def test_algorithm_stored_in_history(self, tmp_path):
        """History entries should record which algorithm was used."""
        b = self._make_bandit(tmp_path, algorithm="thompson")
        from harness.rl_bandit import ARM_NAMES
        arm = ARM_NAMES[0]
        b.update(arm, 0.5)
        history_entry = b._state["history"][-1]
        assert history_entry.get("algo") == "thompson"

    def test_persistence_preserves_sum_sq(self, tmp_path):
        """sum_sq_reward should persist across bandit instances."""
        b1 = self._make_bandit(tmp_path, algorithm="thompson")
        from harness.rl_bandit import ARM_NAMES
        arm = ARM_NAMES[0]
        b1.update(arm, 1.5)
        b1.update(arm, -0.5)
        expected = 1.5 ** 2 + (-0.5) ** 2

        b2 = self._make_bandit(tmp_path, algorithm="thompson")
        assert b2._state["arms"][arm]["sum_sq_reward"] == pytest.approx(expected, abs=1e-9)

    def test_stats_summary_shows_algorithm(self, tmp_path):
        """stats_summary() should mention the algorithm name."""
        b = self._make_bandit(tmp_path, algorithm="thompson")
        summary = b.stats_summary()
        assert "THOMPSON" in summary.upper()


# ────────────────────────────────────────────────────────────────────────────
# 3.  TF-IDF semantic deduplication in KnowledgeBase
# ────────────────────────────────────────────────────────────────────────────

class TestSemanticDeduplication:
    """TF-IDF cosine similarity for find_similar()."""

    @pytest.fixture()
    def fresh_kb(self, tmp_path, monkeypatch):
        import harness.memory.knowledge_base as kb_module
        monkeypatch.setattr(kb_module, "MEMORY_DIR", tmp_path)
        from harness.memory.knowledge_base import KnowledgeBase
        return KnowledgeBase(session_id="test")

    def test_find_similar_returns_feature_overlap(self, fresh_kb):
        """Jaccard similarity detects experiments sharing features."""
        fresh_kb.save_experiment(
            hypothesis="Momentum with RSI filter",
            config={"features": ["mom_12_1", "rsi_14", "vol_21d"]},
            results={"sharpe": 0.5},
            verdict="MODERATE",
        )
        fresh_kb.save_experiment(
            hypothesis="Volume liquidity strategy",
            config={"features": ["dollar_volume", "amihud_illiq", "obv"]},
            results={"sharpe": 0.3},
            verdict="WEAK",
        )
        # Query with features overlapping the first experiment
        similar = fresh_kb.find_similar(["mom_12_1", "rsi_14", "sma_50_dist"])
        assert len(similar) >= 1
        assert any("Momentum" in s.get("title", "") for s in similar)

    def test_find_similar_with_hypothesis_text(self, fresh_kb):
        """Semantic hypothesis matching via TF-IDF cosine similarity."""
        pytest.importorskip("sklearn")  # skip if sklearn not available
        fresh_kb.save_experiment(
            hypothesis="Momentum breakout with trend confirmation using SMA signals",
            config={"features": ["mom_12_1", "sma_50_above_200"]},
            results={"sharpe": 0.7},
            verdict="STRONG",
        )
        fresh_kb.save_experiment(
            hypothesis="Mean reversion using RSI oversold signals",
            config={"features": ["rsi_14", "bb_position"]},
            results={"sharpe": 0.4},
            verdict="MODERATE",
        )
        # Query with hypothesis similar to the first experiment
        similar = fresh_kb.find_similar(
            features=["mom_12_1"],
            hypothesis="momentum trend breakout SMA confirmation",
        )
        assert len(similar) >= 1
        assert any("Momentum" in s.get("title", "") for s in similar)

    def test_find_similar_no_false_positives_for_empty(self, fresh_kb):
        """Empty feature list + no hypothesis should return nothing."""
        fresh_kb.save_experiment(
            hypothesis="Test experiment",
            config={"features": ["mom_12_1"]},
            results={"sharpe": 0.5},
            verdict="MODERATE",
        )
        similar = fresh_kb.find_similar([], hypothesis="")
        assert similar == []

    def test_find_similar_threshold_filters_low_overlap(self, fresh_kb):
        """Entries with < 30% fused similarity should be excluded."""
        fresh_kb.save_experiment(
            hypothesis="Pure volatility contraction signal",
            config={"features": ["bb_width", "atr_14"]},
            results={"sharpe": 0.2},
            verdict="WEAK",
        )
        # Completely different features + different hypothesis
        similar = fresh_kb.find_similar(
            ["mom_12_1", "rsi_14", "dollar_volume"],
            hypothesis="momentum trend persistence",
        )
        # Should not match the volatility experiment (very low similarity)
        titles = [s.get("title", "") for s in similar]
        assert not any("volatility contraction" in t.lower() for t in titles)


# ────────────────────────────────────────────────────────────────────────────
# 4.  DSR integration in executor backtest result
# ────────────────────────────────────────────────────────────────────────────

class TestDSRInBacktestResult:
    """DSR/PSR fields injected into _tool_run_backtest output."""

    def test_executor_has_n_trials_attribute(self):
        """ToolExecutor should initialise _n_trials = 1."""
        from harness.tools.executor import ToolExecutor
        ex = ToolExecutor()
        assert hasattr(ex, "_n_trials")
        assert ex._n_trials == 1

    def test_stats_module_importable(self):
        """harness.stats should import without errors."""
        import harness.stats as stats
        assert callable(stats.sharpe_tstat)
        assert callable(stats.probabilistic_sharpe_ratio)
        assert callable(stats.deflated_sharpe_ratio)

    def test_sharpe_tstat_output_keys(self):
        """sharpe_tstat must return all required keys."""
        from harness.stats import sharpe_tstat
        result = sharpe_tstat(0.8, 1260)
        for key in ("t_stat", "p_value", "n_years", "significant_at_05"):
            assert key in result

    def test_psr_output_keys(self):
        from harness.stats import probabilistic_sharpe_ratio
        result = probabilistic_sharpe_ratio(0.8, 1260)
        for key in ("psr", "sr_star", "z_score", "interpretation"):
            assert key in result

    def test_dsr_output_keys(self):
        from harness.stats import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(0.8, 1260, n_trials=5)
        for key in ("dsr", "sr_star_ann", "n_trials", "interpretation"):
            assert key in result

    def test_dsr_bounded_zero_one(self):
        """DSR is a probability and must lie in [0, 1]."""
        from harness.stats import deflated_sharpe_ratio
        for sharpe in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            result = deflated_sharpe_ratio(sharpe, 1260, n_trials=10)
            assert 0.0 <= result["dsr"] <= 1.0, f"DSR out of range for SR={sharpe}"

    def test_psr_bounded_zero_one(self):
        """PSR is a probability and must lie in [0, 1]."""
        from harness.stats import probabilistic_sharpe_ratio
        for sharpe in [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]:
            result = probabilistic_sharpe_ratio(sharpe, 1260)
            assert 0.0 <= result["psr"] <= 1.0, f"PSR out of range for SR={sharpe}"


# ────────────────────────────────────────────────────────────────────────────
# 5.  minimum_backtest_length utility
# ────────────────────────────────────────────────────────────────────────────

class TestMinimumBacktestLength:
    """Bailey & Lopez de Prado (2012) minimum backtest length for DSR >= target."""

    def test_higher_sharpe_requires_fewer_bars(self):
        """A higher Sharpe needs fewer observations to reach the same DSR."""
        from harness.stats import minimum_backtest_length
        n_low  = minimum_backtest_length(0.5, n_trials=5)
        n_high = minimum_backtest_length(2.0, n_trials=5)
        assert n_high < n_low

    def test_more_trials_require_more_bars(self):
        """More trials inflate the SR* threshold, requiring more data."""
        from harness.stats import minimum_backtest_length
        n1 = minimum_backtest_length(1.0, n_trials=1)
        n5 = minimum_backtest_length(1.0, n_trials=5)
        assert n5 > n1

    def test_returns_positive_integer(self):
        """Return value is a positive integer (bar count)."""
        from harness.stats import minimum_backtest_length
        n = minimum_backtest_length(0.8, n_trials=10)
        assert isinstance(n, int) and n > 0

    def test_high_sharpe_short_horizon(self):
        """Sharpe=2.0 with 1 trial should be achievable in under 5 years."""
        from harness.stats import minimum_backtest_length
        n = minimum_backtest_length(2.0, n_trials=1, target_dsr=0.95)
        assert n <= 252 * 5  # at most 5 years of daily data

    def test_low_sharpe_many_trials_needs_very_long(self):
        """Sharpe=0.5 after 20 trials should need more than 5 years."""
        from harness.stats import minimum_backtest_length
        n = minimum_backtest_length(0.5, n_trials=20)
        assert n > 252 * 5  # more than 5 years needed
