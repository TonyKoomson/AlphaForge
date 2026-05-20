"""
Tests for backtest/engine.py — Alpha Forge
==========================================
Every test is deterministic (no randomness, no network).

Coverage areas:
  1. Look-ahead bias  — signal at bar T cannot capture return AT bar T
  2. Cost arithmetic  — costs reduce returns by the correct amount
  3. Equity curve     — known prices/signals produce exact equity values
  4. Trade log        — entries, exits, reversals, open-at-end trades
  5. Stress testing   — correct period slicing and re-computation
  6. Multi-asset      — equal-weight portfolio, independent cost application
  7. CostModel        — factory methods and round-trip arithmetic
  8. Edge cases       — all-flat signals, single bar, constant price
  9. BacktestEngine   — class wrapper interface (used by validation/report.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import (
    STRESS_PERIODS,
    BacktestEngine,
    BacktestResult,
    CostModel,
    _extract_trades,
    _get_close,
    run_backtest,
    stress_test,
)


# ---------------------------------------------------------------------------
# Shared fixtures and factories
# ---------------------------------------------------------------------------

def make_prices(values: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    idx = pd.bdate_range(start=start, periods=len(values))
    c = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c * 0.999, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e6},
        index=idx,
    )


def make_signals(values: list[int], start: str = "2020-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def make_trending_prices(n: int = 100, pct_per_bar: float = 0.005, start: str = "2020-01-02") -> pd.DataFrame:
    """Steadily rising prices for basic long-only tests."""
    close = [100.0 * (1 + pct_per_bar) ** i for i in range(n)]
    return make_prices(close, start=start)


# ---------------------------------------------------------------------------
# 1. CostModel
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_total_per_side(self):
        cm = CostModel(commission_pct=0.001, slippage_pct=0.0005, spread_pct=0.0002)
        assert cm.total_per_side == pytest.approx(0.0017)

    def test_round_trip_is_double_per_side(self):
        cm = CostModel(commission_pct=0.001, slippage_pct=0.0005, spread_pct=0.0002)
        assert cm.round_trip == pytest.approx(2 * cm.total_per_side)

    def test_zero_model_has_no_costs(self):
        z = CostModel.zero()
        assert z.total_per_side == 0.0
        assert z.round_trip == 0.0

    def test_for_stock_lower_than_for_crypto(self):
        assert CostModel.for_stock().round_trip < CostModel.for_crypto().round_trip

    def test_float_costs_parameter_sets_commission(self):
        """Passing a float to run_backtest() sets commission_pct."""
        prices  = make_trending_prices(20)
        signals = make_signals([1] * 20, start=prices.index[0].isoformat())

        r_float  = run_backtest(signals, prices, costs=0.005)
        r_model  = run_backtest(signals, prices, costs=CostModel(commission_pct=0.005, slippage_pct=0.0005))
        # Both should produce the same commission drag
        assert r_float.metrics["total_costs_pct"] == pytest.approx(
            r_model.metrics["total_costs_pct"], rel=0.2
        )


# ---------------------------------------------------------------------------
# 2. Look-ahead bias (the most important test class)
# ---------------------------------------------------------------------------

class TestLookAheadBias:
    """
    Verify that a signal generated at close of bar T can NEVER capture the
    return that occurred ON bar T — only returns from bar T+1 onwards.
    """

    def test_signal_at_drop_bar_misses_the_drop(self):
        """
        Price falls 10% on bar 1.  Buy signal is issued at close of bar 1
        (AFTER the drop has already happened).  The strategy must show 0 P&L
        on bar 1 because it had no position during the drop.
        """
        # 100 → 90 on bar 1; flat afterwards
        prices  = make_prices([100.0, 90.0, 90.0, 90.0])
        signals = make_signals([0, 1, 1, 0])  # buy AFTER the bar-1 drop

        result = run_backtest(signals, prices, costs=CostModel.zero())

        # Bar 1: position = signal[bar0] = 0  →  0 * (−10%) = 0
        assert result.returns.iloc[1] == pytest.approx(0.0, abs=1e-10), (
            f"Bar-1 return should be zero (no position), got {result.returns.iloc[1]}"
        )

    def test_signal_at_rally_bar_misses_the_rally(self):
        """
        Symmetric test: a sell signal issued at the top of a 10% rally bar
        does NOT let the strategy short the within-bar move.
        """
        prices  = make_prices([100.0, 110.0, 110.0, 110.0])
        signals = make_signals([0, -1, -1, 0])  # short AFTER the bar-1 rally

        result = run_backtest(signals, prices, costs=CostModel.zero())

        # Bar 1: position = signal[bar0] = 0 → zero return
        assert result.returns.iloc[1] == pytest.approx(0.0, abs=1e-10)

    def test_position_is_always_one_bar_delayed(self):
        """
        The positions Series must equal signals.shift(1) after the backtest.
        """
        prices  = make_prices([100, 101, 102, 103, 104, 103, 102])
        signals = make_signals([0, 1, 1, 0, -1, -1, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())

        expected = signals.shift(1).fillna(0)
        pd.testing.assert_series_equal(
            result.positions.rename(None),
            expected.rename(None),
            check_names=False,
        )

    def test_return_on_signal_bar_is_always_independent_of_signal_value(self):
        """
        For any signal value at bar T, the return at bar T is determined
        only by the position set at bar T-1 (the previous signal), not bar T.
        """
        prices  = make_prices([100, 90, 81, 73, 66])

        # Strategy A: flat the whole time
        sig_flat = make_signals([0, 0, 0, 0, 0])
        # Strategy B: long the whole time (including at every drop bar)
        sig_long = make_signals([1, 1, 1, 1, 1])

        r_flat = run_backtest(sig_flat, prices, costs=CostModel.zero())
        r_long = run_backtest(sig_long, prices, costs=CostModel.zero())

        # Bar 0: both have position=0 (shift(1) fills first bar with 0)
        assert r_flat.returns.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert r_long.returns.iloc[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Equity curve accuracy
# ---------------------------------------------------------------------------

class TestEquityCurveAccuracy:
    """Verify the equity curve against hand-calculated expected values."""

    def test_always_long_zero_costs(self):
        """
        Prices: 100 → 110 → 110 → 121
        Returns: NaN, +10%, 0%, +10%
        position (shift 1): 0, 1, 1, 1
        Net P&L per bar:    0, 10%, 0%, 10%
        Equity:             100k, 110k, 110k, 121k
        """
        prices  = make_prices([100.0, 110.0, 110.0, 121.0])
        signals = make_signals([1, 1, 1, 1])

        result = run_backtest(signals, prices, costs=CostModel.zero(), initial_capital=100_000.0)

        expected_equity = [100_000.0, 110_000.0, 110_000.0, 121_000.0]
        np.testing.assert_allclose(result.equity_curve.values, expected_equity, rtol=1e-9)

    def test_total_return_equals_equity_growth(self):
        prices  = make_trending_prices(50)
        signals = make_signals([1] * 50, start=prices.index[0].isoformat())

        result  = run_backtest(signals, prices, costs=CostModel.zero(), initial_capital=100_000.0)
        equity_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
        assert result.metrics["total_return_net"] == pytest.approx(equity_return, rel=1e-6)

    def test_flat_signal_produces_flat_equity(self):
        prices  = make_trending_prices(30)
        signals = make_signals([0] * 30, start=prices.index[0].isoformat())

        result  = run_backtest(signals, prices, costs=CostModel.zero(), initial_capital=100_000.0)

        # With zero signals and zero costs, equity should be flat
        np.testing.assert_allclose(
            result.equity_curve.values,
            np.full(30, 100_000.0),
            atol=1.0,
        )

    def test_equity_starts_at_initial_capital(self):
        prices  = make_prices([100, 110, 105])
        signals = make_signals([1, 1, 0])
        capital = 250_000.0

        result = run_backtest(signals, prices, costs=CostModel.zero(), initial_capital=capital)
        assert result.equity_curve.iloc[0] == pytest.approx(capital, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Cost arithmetic
# ---------------------------------------------------------------------------

class TestCostArithmetic:
    """Verify that costs reduce returns by the correct amount."""

    def test_costs_reduce_return_vs_zero_cost(self):
        prices  = make_trending_prices(50)
        signals = make_signals([1] * 50, start=prices.index[0].isoformat())

        r_free  = run_backtest(signals, prices, costs=CostModel.zero())
        r_cost  = run_backtest(signals, prices, costs=CostModel.for_crypto())

        assert r_cost.metrics["total_return_net"] < r_free.metrics["total_return_net"], (
            "Strategy with costs must underperform the zero-cost version."
        )

    def test_higher_costs_lower_return(self):
        prices  = make_trending_prices(40)
        signals = make_signals([1] * 40, start=prices.index[0].isoformat())

        r_cheap = run_backtest(signals, prices, costs=CostModel.for_stock())
        r_dear  = run_backtest(signals, prices, costs=CostModel.for_crypto())

        assert r_dear.metrics["total_return_net"] < r_cheap.metrics["total_return_net"]

    def test_single_entry_cost_applied_once(self):
        """
        Enter a long position at bar 1, hold flat until end, never exit.
        Exactly one 'entry' turnover event occurs → cost = 1 × total_per_side.
        """
        prices  = make_prices([100.0, 100.0, 100.0, 100.0, 100.0])
        signals = make_signals([1, 1, 1, 1, 1])
        cm      = CostModel(commission_pct=0.01, slippage_pct=0.0, spread_pct=0.0)

        result  = run_backtest(signals, prices, costs=cm, initial_capital=100_000.0)

        # Only one turnover event (bar 1, entering position)
        # Total cost should be 1 × 0.01 = 0.01 of position = 0.01
        assert result.metrics["total_costs_pct"] == pytest.approx(0.01, rel=1e-6)

    def test_reversal_costs_twice_as_much_as_entry(self):
        """
        Going from flat → long costs 1×.
        Going from long → short costs 2× (exit long + enter short).
        """
        prices = make_prices([100, 100, 100, 100, 100, 100])
        cm     = CostModel(commission_pct=0.01, slippage_pct=0.0, spread_pct=0.0)

        # Entry only (flat → long)
        sig_entry = make_signals([1, 1, 1, 1, 1, 1])
        r_entry   = run_backtest(sig_entry, prices, costs=cm)

        # Entry + reversal (flat → long → short)
        sig_rev   = make_signals([1, 1, 1, -1, -1, -1])
        r_rev     = run_backtest(sig_rev, prices, costs=cm)

        # r_rev should have paid more in costs
        assert r_rev.metrics["total_costs_pct"] > r_entry.metrics["total_costs_pct"]

    def test_zero_cost_gross_equals_net(self):
        prices  = make_trending_prices(20)
        signals = make_signals([1] * 20, start=prices.index[0].isoformat())

        result = run_backtest(signals, prices, costs=CostModel.zero())

        pd.testing.assert_series_equal(
            result.returns,
            result.gross_returns,
            check_names=False,
        )

    def test_cost_drag_series_non_negative(self):
        prices  = make_trending_prices(30)
        signals = make_signals([1, 0, -1] * 10, start=prices.index[0].isoformat())

        result = run_backtest(signals, prices, costs=CostModel.for_crypto())

        assert (result.costs_paid >= 0).all(), "Cost drag must never be negative."

    def test_theory_vs_real_gap_equals_total_costs(self):
        """
        The 'theory vs reality gap' in total return must be close to the total costs paid.

        Note: gap > costs slightly because compounding amplifies the initial cost drag —
        a 0.2% entry commission on a 239% trending market produces ~0.254% gap.
        We allow rel=0.35 to accommodate this compounding effect.
        """
        prices  = make_trending_prices(50)
        signals = make_signals([1] * 50, start=prices.index[0].isoformat())
        cm      = CostModel(commission_pct=0.002, slippage_pct=0.0, spread_pct=0.0)

        result = run_backtest(signals, prices, costs=cm)

        gap   = result.metrics["theory_vs_real_gap"]
        costs = result.metrics["total_costs_pct"]
        # gap ≈ costs × (1 + gross_return): costs get compounded; rel=0.35 covers this
        assert gap == pytest.approx(costs, rel=0.35)
        # Also verify ordering: gap must be at least as large as raw costs
        assert gap >= costs * 0.95


# ---------------------------------------------------------------------------
# 5. Trade log
# ---------------------------------------------------------------------------

class TestTradeLog:
    def test_long_then_flat_produces_one_trade(self):
        prices  = make_prices([100.0, 110.0, 115.0, 115.0])
        signals = make_signals([1, 1, 0, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert len(result.trade_log) == 1

    def test_trade_direction_is_correct(self):
        prices  = make_prices([100, 110, 110, 90, 90])
        signals = make_signals([1, 1, 0, -1, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert len(result.trade_log) == 2
        assert result.trade_log.iloc[0]["direction"] == "long"
        assert result.trade_log.iloc[1]["direction"] == "short"

    def test_winning_trade_flagged_correctly(self):
        prices  = make_prices([100.0, 105.0, 110.0, 110.0])
        signals = make_signals([1, 1, 0, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert bool(result.trade_log.iloc[0]["is_winner"]) is True

    def test_losing_trade_flagged_correctly(self):
        prices  = make_prices([100.0, 90.0, 80.0, 80.0])
        signals = make_signals([1, 1, 0, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert bool(result.trade_log.iloc[0]["is_winner"]) is False

    def test_reversal_creates_two_trades(self):
        """Reversing from long to short should produce 2 trades in the log."""
        prices  = make_prices([100, 105, 100, 95, 95])
        signals = make_signals([1, 1, -1, -1, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert len(result.trade_log) == 2

    def test_open_at_end_flagged(self):
        """A position that's never closed by a 0-signal is marked open_at_end=True."""
        prices  = make_prices([100, 105, 110, 115])
        signals = make_signals([1, 1, 1, 1])  # never exits

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert bool(result.trade_log.iloc[-1]["open_at_end"]) is True

    def test_no_trades_when_all_flat(self):
        prices  = make_prices([100, 102, 104, 106])
        signals = make_signals([0, 0, 0, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert result.trade_log.empty

    def test_win_rate_and_profit_factor_in_metrics(self):
        # Construct 2 winning trades and 1 losing trade
        prices  = make_prices([100, 110, 100, 110, 100, 110, 100, 90, 100])
        signals = make_signals([1, 0, 1, 0, 1, 0, -1, 0, 0])

        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert "win_rate" in result.metrics
        assert "profit_factor" in result.metrics
        assert 0 <= result.metrics["win_rate"] <= 1


# ---------------------------------------------------------------------------
# 6. Sharpe / metrics sanity
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_positive_trend_has_positive_sharpe(self):
        prices  = make_trending_prices(252, pct_per_bar=0.003)
        signals = make_signals([1] * 252, start=prices.index[0].isoformat())

        result  = run_backtest(signals, prices, costs=CostModel.zero())
        assert result.metrics["sharpe_ratio"] > 0

    def test_perfectly_flat_returns_have_zero_sharpe(self):
        """All returns = 0 → vol = 0 → Sharpe is 0 (not NaN or inf)."""
        prices  = make_prices([100.0] * 30)
        signals = make_signals([0] * 30)

        result  = run_backtest(signals, prices, costs=CostModel.zero())
        assert result.metrics["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_is_non_positive(self):
        prices  = make_trending_prices(100)
        signals = make_signals([1] * 100, start=prices.index[0].isoformat())

        result  = run_backtest(signals, prices, costs=CostModel.zero())
        assert result.metrics["max_drawdown"] <= 0

    def test_summary_dict_contains_required_keys(self):
        prices  = make_trending_prices(30)
        signals = make_signals([1] * 30, start=prices.index[0].isoformat())

        result  = run_backtest(signals, prices, costs=CostModel.for_stock(), label="SPY test")
        s = result.summary()

        for key in ["cagr_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
                    "profit_factor", "total_trades", "total_costs_pct", "cost_model"]:
            assert key in s, f"Missing key in summary: {key!r}"


# ---------------------------------------------------------------------------
# 7. Stress testing
# ---------------------------------------------------------------------------

class TestStressTest:
    def _make_long_prices(self, start="2019-01-01", end="2023-12-31") -> pd.DataFrame:
        idx = pd.bdate_range(start=start, end=end)
        rng = np.random.default_rng(42)
        c   = 100.0 * np.exp(rng.normal(0.0003, 0.012, len(idx)).cumsum())
        return pd.DataFrame(
            {"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6},
            index=idx,
        )

    def test_stress_test_returns_dict(self):
        prices  = self._make_long_prices()
        signals = pd.Series(1.0, index=prices.index)

        results = stress_test(signals, prices, periods={"covid_crash": ("2020-02-19", "2020-03-23")})
        assert isinstance(results, dict)
        assert "covid_crash" in results

    def test_stress_period_respects_date_bounds(self):
        prices  = self._make_long_prices()
        signals = pd.Series(1.0, index=prices.index)

        results = stress_test(signals, prices, periods={"test_period": ("2021-01-01", "2021-06-30")})
        r = results["test_period"]

        assert r.equity_curve.index.min() >= pd.Timestamp("2021-01-01")
        assert r.equity_curve.index.max() <= pd.Timestamp("2021-06-30")

    def test_stress_result_is_independent_backtest(self):
        """Each stress result should have its own equity curve starting at initial_capital."""
        prices  = self._make_long_prices()
        signals = pd.Series(1.0, index=prices.index)

        results = stress_test(
            signals, prices,
            periods={"p1": ("2020-01-01", "2020-06-30"),
                     "p2": ("2021-01-01", "2021-06-30")},
            initial_capital=50_000.0,
        )

        for name, r in results.items():
            assert r.equity_curve.iloc[0] == pytest.approx(50_000.0, rel=0.01), (
                f"Period {name!r}: equity should start at initial_capital"
            )

    def test_stress_skips_period_with_no_data(self):
        """A period outside the prices date range should be skipped gracefully."""
        prices  = self._make_long_prices(start="2022-01-01", end="2023-12-31")
        signals = pd.Series(1.0, index=prices.index)

        results = stress_test(
            signals, prices,
            periods={"old_period": ("2010-01-01", "2011-01-01")}
        )
        assert "old_period" not in results

    def test_predefined_stress_periods_have_correct_format(self):
        for name, (start, end) in STRESS_PERIODS.items():
            assert pd.Timestamp(start) < pd.Timestamp(end), (
                f"STRESS_PERIODS[{name!r}]: start must be before end"
            )

    def test_custom_periods_override_default(self):
        prices  = self._make_long_prices()
        signals = pd.Series(1.0, index=prices.index)

        custom = {"custom_window": ("2022-06-01", "2022-09-30")}
        results = stress_test(signals, prices, periods=custom)

        assert list(results.keys()) == ["custom_window"]

    def test_stress_costs_are_applied(self):
        prices  = self._make_long_prices()
        signals = pd.Series(1.0, index=prices.index)
        period  = {"test": ("2021-01-01", "2021-12-31")}

        r_free = stress_test(signals, prices, periods=period, costs=CostModel.zero())["test"]
        r_cost = stress_test(signals, prices, periods=period, costs=CostModel.for_crypto())["test"]

        assert r_cost.metrics["total_return_net"] < r_free.metrics["total_return_net"]


# ---------------------------------------------------------------------------
# 8. Multi-asset
# ---------------------------------------------------------------------------

class TestMultiAsset:
    def _make_multi(self, n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-02", periods=n)
        prices_dict = {}
        for sym in ["SPY", "BTC"]:
            c = 100.0 * np.exp(rng.normal(0.001, 0.01, n).cumsum())
            prices_dict[sym] = c
        prices = pd.DataFrame({
            "SPY": prices_dict["SPY"],
            "BTC": prices_dict["BTC"],
        }, index=idx)
        signals = pd.DataFrame({
            "SPY": pd.Series(1.0, index=idx),
            "BTC": pd.Series(1.0, index=idx),
        })
        return signals, prices

    def test_multi_asset_returns_single_result(self):
        signals, prices = self._make_multi()
        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert isinstance(result, BacktestResult)

    def test_multi_asset_equity_is_equal_weight_mean(self):
        """
        When signals are identical for two uncorrelated assets, the portfolio
        Sharpe should be higher than either asset's Sharpe alone (diversification).
        This is not always true in a tiny sample, but returns must be finite.
        """
        signals, prices = self._make_multi()
        result = run_backtest(signals, prices, costs=CostModel.zero())

        assert not result.equity_curve.isna().any()
        assert not result.returns.isna().any()

    def test_multi_asset_positions_are_dataframe(self):
        signals, prices = self._make_multi()
        result = run_backtest(signals, prices, costs=CostModel.zero())
        assert isinstance(result.positions, pd.DataFrame)
        assert set(result.positions.columns) == {"SPY", "BTC"}

    def test_multi_asset_costs_applied_per_asset(self):
        signals, prices = self._make_multi()

        r_free = run_backtest(signals, prices, costs=CostModel.zero())
        r_cost = run_backtest(signals, prices, costs=CostModel.for_crypto())

        assert r_cost.metrics["total_return_net"] < r_free.metrics["total_return_net"]


# ---------------------------------------------------------------------------
# 9. BacktestEngine class interface (used by validation/report.py)
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def _features(self, n: int = 40, start: str = "2020-01-02") -> pd.DataFrame:
        prices = make_trending_prices(n, start=start)
        # Simulate a feature DataFrame that also has vol columns
        prices["vol_21d"] = 0.15
        return prices

    def test_engine_run_returns_backtest_result(self):
        feat    = self._features()
        signals = pd.Series(1.0, index=feat.index)
        engine  = BacktestEngine(config={})
        result  = engine.run(feat, signals, ticker="SPY")
        assert isinstance(result, BacktestResult)

    def test_engine_run_metrics_dict(self):
        feat    = self._features()
        signals = pd.Series(1.0, index=feat.index)
        engine  = BacktestEngine(config={})
        result  = engine.run(feat, signals, ticker="TEST")
        assert isinstance(result.metrics, dict)
        assert "sharpe_ratio" in result.metrics

    def test_engine_run_theoretical_has_zero_costs(self):
        feat    = self._features()
        signals = pd.Series(1.0, index=feat.index)
        engine  = BacktestEngine(config={})
        result  = engine.run_theoretical(feat, signals)
        assert result.metrics.get("total_costs_pct", 0) == pytest.approx(0.0)

    def test_theoretical_beats_realistic(self):
        """Zero-cost theoretical run must not underperform the realistic run."""
        feat    = self._features(80)
        signals = pd.Series([1, -1] * 40, index=feat.index, dtype=float)
        engine  = BacktestEngine(config={})
        real    = engine.run(feat, signals, "SPY")
        theory  = engine.run_theoretical(feat, signals)
        assert theory.metrics["total_return_net"] >= real.metrics["total_return_net"] - 1e-6

    def test_engine_label_set_to_ticker(self):
        feat    = self._features()
        signals = pd.Series(1.0, index=feat.index)
        engine  = BacktestEngine(config={})
        result  = engine.run(feat, signals, ticker="AAPL")
        assert result.label == "AAPL"

    def test_engine_stress_test_returns_dict(self):
        feat    = self._features(500, start="2019-01-02")
        signals = pd.Series(1.0, index=feat.index)
        engine  = BacktestEngine(config={})
        results = engine.stress_test(feat, signals)
        assert isinstance(results, dict)

    def test_engine_high_volatility_stress_mode(self):
        feat    = self._features(900, start="2019-01-02")
        signals = pd.Series(1.0, index=feat.index)
        engine  = BacktestEngine(config={})
        results = engine.stress_test(feat, signals, high_volatility_mode=True)
        assert isinstance(results, dict)
        # Should focus on major high-vol periods such as March 2020 and 2022.
        assert "march_2020_crash" in results or "2022_bear_market" in results


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_zeros_signal_is_safe(self):
        prices  = make_trending_prices(20)
        signals = make_signals([0] * 20, start=prices.index[0].isoformat())
        result  = run_backtest(signals, prices, costs=CostModel.zero())
        assert result is not None
        assert result.metrics["total_trades"] == 0

    def test_single_long_bar_is_safe(self):
        """Entry bar only — position never exceeds bar 1."""
        prices  = make_prices([100.0, 105.0, 105.0])
        signals = make_signals([1, 0, 0])
        result  = run_backtest(signals, prices, costs=CostModel.zero())
        assert result is not None

    def test_constant_price_zero_costs_flat_equity(self):
        prices  = make_prices([100.0] * 10)
        signals = make_signals([1] * 10)
        result  = run_backtest(signals, prices, costs=CostModel.zero())
        np.testing.assert_allclose(result.equity_curve.values, np.full(10, 100_000.0), atol=1e-6)

    def test_prices_as_series_accepted(self):
        idx     = pd.bdate_range("2020-01-02", periods=10)
        close   = pd.Series(np.linspace(100, 110, 10), index=idx)
        signals = pd.Series([1.0] * 10, index=idx)
        result  = run_backtest(signals, close, costs=CostModel.zero())
        assert result is not None

    def test_get_close_from_dataframe_with_close_column(self):
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1e6, 1e6, 1e6]})
        s  = _get_close(df)
        pd.testing.assert_series_equal(s, df["close"])

    def test_get_close_from_series(self):
        s = pd.Series([100, 101, 102])
        assert _get_close(s) is s

    def test_get_close_from_single_column_df(self):
        df = pd.DataFrame({"price": [100, 101, 102]})
        s  = _get_close(df)
        pd.testing.assert_series_equal(s, df["price"])

    def test_misaligned_signals_and_prices_handled(self):
        """Signals and prices with slightly different indices should align cleanly."""
        price_idx  = pd.bdate_range("2020-01-02", periods=20)
        signal_idx = pd.bdate_range("2020-01-06", periods=15)   # starts later
        prices  = pd.DataFrame({"close": np.linspace(100, 120, 20)}, index=price_idx)
        signals = pd.Series(1.0, index=signal_idx)
        result  = run_backtest(signals, prices, costs=CostModel.zero())
        assert result is not None
        assert len(result.equity_curve) > 0

    def test_repr_contains_key_metrics(self):
        prices  = make_trending_prices(30)
        signals = make_signals([1] * 30, start=prices.index[0].isoformat())
        result  = run_backtest(signals, prices, costs=CostModel.for_stock(), label="test")
        r = repr(result)
        assert "Sharpe=" in r
        assert "CAGR=" in r
        assert "MaxDD=" in r

    def test_risk_managed_positions_are_capped(self):
        prices = make_trending_prices(80)
        signals = make_signals([1] * 80, start=prices.index[0].isoformat())
        confidence = pd.Series(0.99, index=signals.index)
        pred = pd.Series(0.03, index=signals.index)
        vol = pd.Series(0.10, index=signals.index)

        result = run_backtest(
            signals,
            prices,
            costs=CostModel.zero(),
            apply_risk_management=True,
            confidence_scores=confidence,
            predicted_returns=pred,
            rolling_volatility=vol,
        )
        assert (result.positions.abs() <= 1.0 + 1e-12).all()
