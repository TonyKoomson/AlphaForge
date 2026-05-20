"""
Backtesting Engine — Alpha Forge
=================================
Vectorised, look-ahead-proof backtesting with realistic transaction costs.

Execution model
---------------
Signals are generated at the **close** of bar T.
Execution is approximated as occurring at the **open of bar T+1**, achieved by
shifting the position series one bar forward (``signal.shift(1)``).

    signal[T]  → position effective from bar T+1 onwards

This guarantees that ANY return occurring WITHIN bar T is never captured by a
signal first observed at the close of bar T.  The invariant is validated in
the test suite (``tests/test_backtest.py::TestLookAheadBias``).

Cost model
----------
Transaction costs are applied as proportional drag each time the position
changes:

    cost_drag[t] = |Δposition[t]| × total_cost_per_side

where total_cost_per_side = commission + slippage + half_spread.

Turnover is fractional, so a reversal (long → short, Δ = 2) costs twice as
much as a simple entry or exit (Δ = 1).  Zero turnover bars carry zero cost.

Multi-asset
-----------
Pass DataFrames for both ``signals`` and ``prices``.  Each column is treated
as an independent asset with equal notional allocation.  Portfolio returns are
the equal-weighted mean across assets.

Stress testing
--------------
``stress_test()`` re-runs the backtest on pre-defined crisis/regime windows
so you can see where the strategy holds up and where it breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from risk.position_sizing import PositionSizer, high_volatility_stress_test_periods
from utils.helpers import compute_all_metrics, get_logger, load_config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pre-defined stress periods
# ---------------------------------------------------------------------------

STRESS_PERIODS: Dict[str, Tuple[str, str]] = {
    "covid_crash":         ("2020-02-19", "2020-03-23"),
    "covid_recovery":      ("2020-03-23", "2020-08-18"),
    "2022_bear_market":    ("2022-01-01", "2022-12-31"),
    "gfc_2008":            ("2007-10-09", "2009-03-09"),
    "dot_com_bust":        ("2000-03-10", "2002-10-09"),
    "q4_2018_selloff":     ("2018-10-01", "2018-12-24"),
    "taper_tantrum_2013":  ("2013-05-22", "2013-09-05"),
    "ukraine_invasion":    ("2022-02-24", "2022-03-16"),
}


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """
    Per-side transaction cost parameters.

    All values are expressed as a fraction of trade notional.
    ``total_per_side`` = commission + slippage + spread, applied once per
    unit of turnover (entry OR exit, not both).  A round-trip therefore
    costs ``2 × total_per_side``.
    """

    commission_pct: float = 0.001    # exchange fee / broker commission
    slippage_pct:   float = 0.0005   # market-impact & adverse selection
    spread_pct:     float = 0.0002   # half bid-ask spread

    @property
    def total_per_side(self) -> float:
        return self.commission_pct + self.slippage_pct + self.spread_pct

    @property
    def round_trip(self) -> float:
        return 2.0 * self.total_per_side

    @classmethod
    def for_stock(cls) -> "CostModel":
        """Typical US equity costs (~0.05% round-trip all-in)."""
        return cls(commission_pct=0.0005, slippage_pct=0.0002, spread_pct=0.0001)

    @classmethod
    def for_crypto(cls) -> "CostModel":
        """Typical crypto exchange costs (~0.20% round-trip all-in)."""
        return cls(commission_pct=0.001, slippage_pct=0.0005, spread_pct=0.0002)

    @classmethod
    def zero(cls) -> "CostModel":
        """No costs — use for theoretical / ideal comparison."""
        return cls(commission_pct=0.0, slippage_pct=0.0, spread_pct=0.0)

    def __str__(self) -> str:
        return (
            f"CostModel(commission={self.commission_pct*100:.3f}%, "
            f"slippage={self.slippage_pct*100:.3f}%, "
            f"spread={self.spread_pct*100:.3f}%, "
            f"round_trip={self.round_trip*100:.3f}%)"
        )


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Complete output of a single backtest run.

    Attributes
    ----------
    equity_curve  : portfolio value at each bar (starts at ``initial_capital``)
    returns       : net daily returns after transaction costs
    gross_returns : daily returns before costs (for theory-vs-reality comparison)
    positions     : position held at each bar (-1 / 0 / +1 or fraction)
    costs_paid    : cost drag per bar (as fraction of portfolio)
    trade_log     : one row per completed trade with entry/exit/P&L
    metrics       : dict of all performance statistics
    cost_model    : the cost parameters used in this run
    label         : optional identifier (e.g. "2022 bear" or "SPY realistic")
    signals       : original signal series (for auditing)
    """

    equity_curve:  pd.Series
    returns:       pd.Series
    gross_returns: pd.Series
    positions:     pd.Series
    costs_paid:    pd.Series
    trade_log:     pd.DataFrame
    metrics:       Dict[str, float]
    cost_model:    CostModel
    label:         str = ""
    signals:       pd.Series = field(default_factory=pd.Series)

    def summary(self) -> Dict[str, object]:
        """Return a compact summary suitable for printing or JSON serialisation."""
        m = self.metrics
        return {
            "label":              self.label or "unnamed",
            "period":             f"{self.equity_curve.index[0].date()} → {self.equity_curve.index[-1].date()}",
            "total_return_pct":   round(m.get("cagr", 0) * 100, 2),
            "cagr_pct":           round(m.get("cagr", 0) * 100, 2),
            "sharpe_ratio":       round(m.get("sharpe_ratio", 0), 3),
            "sortino_ratio":      round(m.get("sortino_ratio", 0), 3),
            "max_drawdown_pct":   round(m.get("max_drawdown", 0) * 100, 2),
            "max_dd_duration":    int(m.get("max_drawdown_duration_bars", 0)),
            "win_rate_pct":       round(m.get("win_rate", 0) * 100, 1),
            "profit_factor":      round(m.get("profit_factor", 0), 3),
            "total_trades":       int(m.get("total_trades", 0)),
            "total_costs_pct":    round(m.get("total_costs_pct", 0) * 100, 4),
            "cost_model":         str(self.cost_model),
        }

    def __repr__(self) -> str:
        m = self.metrics
        tag = f"[{self.label}]" if self.label else ""
        return (
            f"BacktestResult{tag} "
            f"Sharpe={m.get('sharpe_ratio', 0):.2f} | "
            f"CAGR={m.get('cagr', 0)*100:.1f}% | "
            f"MaxDD={m.get('max_drawdown', 0)*100:.1f}% | "
            f"Trades={m.get('total_trades', 0)} | "
            f"Costs={m.get('total_costs_pct', 0)*100:.2f}%"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_close(prices: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Extract a single close-price Series from various input formats."""
    if isinstance(prices, pd.Series):
        return prices
    if "close" in prices.columns:
        return prices["close"]
    if len(prices.columns) == 1:
        return prices.iloc[:, 0]
    raise ValueError(
        "Cannot determine close prices from prices DataFrame. "
        "Pass a Series or a DataFrame with a 'close' column."
    )


def _extract_trades(positions: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Walk the position series and emit one row per completed trade.

    A reversal (long → short) is split into two trades: exit long, enter short.
    An open position at the end of the series is closed at the last available price.
    """
    pos   = positions.reindex(close.index).fillna(0)
    close = close.reindex(pos.index)

    trades: List[dict] = []
    in_trade   = False
    entry_idx  = None
    entry_px   = np.nan
    direction  = 0

    for i, (ts, cur_pos) in enumerate(pos.items()):
        cur_px = close.iloc[i]

        if not in_trade:
            if cur_pos != 0:
                in_trade  = True
                entry_idx = ts
                entry_px  = cur_px
                direction = int(np.sign(cur_pos))
        else:
            cur_dir = int(np.sign(cur_pos)) if cur_pos != 0 else 0

            if cur_pos == 0 or cur_dir != direction:
                # Close current trade
                if entry_px > 0:
                    ret_pct = direction * (cur_px / entry_px - 1.0)
                else:
                    ret_pct = 0.0

                bars = i - pos.index.get_loc(entry_idx)
                trades.append({
                    "entry_date":  entry_idx,
                    "exit_date":   ts,
                    "direction":   "long" if direction == 1 else "short",
                    "entry_price": round(float(entry_px), 6),
                    "exit_price":  round(float(cur_px), 6),
                    "return_pct":  round(ret_pct * 100.0, 4),
                    "bars_held":   bars,
                    "open_at_end": False,
                })

                if cur_pos != 0:
                    # Reversal: immediately open the opposite trade
                    in_trade  = True
                    entry_idx = ts
                    entry_px  = cur_px
                    direction = cur_dir
                else:
                    in_trade = False

    # Close any open trade at end of series
    if in_trade and entry_px > 0:
        last_px  = close.iloc[-1]
        last_ts  = close.index[-1]
        ret_pct  = direction * (last_px / entry_px - 1.0)
        bars     = len(pos) - pos.index.get_loc(entry_idx) - 1
        trades.append({
            "entry_date":  entry_idx,
            "exit_date":   last_ts,
            "direction":   "long" if direction == 1 else "short",
            "entry_price": round(float(entry_px), 6),
            "exit_price":  round(float(last_px), 6),
            "return_pct":  round(ret_pct * 100.0, 4),
            "bars_held":   bars,
            "open_at_end": True,
        })

    _empty = pd.DataFrame(columns=[
        "entry_date", "exit_date", "direction",
        "entry_price", "exit_price", "return_pct", "bars_held", "open_at_end",
    ])
    if not trades:
        return _empty

    df = pd.DataFrame(trades)
    df["is_winner"] = df["return_pct"] > 0
    return df


def _trade_metrics(trade_log: pd.DataFrame) -> Dict[str, float]:
    """Compute win rate and profit factor from the trade log."""
    if trade_log.empty or "return_pct" not in trade_log.columns:
        return {"win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0,
                "avg_trade_return_pct": 0.0, "avg_bars_held": 0.0}

    wins   = trade_log[trade_log["return_pct"] > 0]
    losses = trade_log[trade_log["return_pct"] < 0]

    gross_profit = wins["return_pct"].sum()
    gross_loss   = abs(losses["return_pct"].sum())

    return {
        "win_rate":             float(len(wins) / len(trade_log)),
        "profit_factor":        float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "total_trades":         len(trade_log),
        "avg_trade_return_pct": float(trade_log["return_pct"].mean()),
        "avg_bars_held":        float(trade_log["bars_held"].mean()) if "bars_held" in trade_log else 0.0,
    }


def _vectorised_core(
    signals:          pd.Series,
    close:            pd.Series,
    cost_model:       CostModel,
    initial_capital:  float,
    periods_per_year: int,
) -> BacktestResult:
    """
    Single-asset vectorised backtest kernel.

    Look-ahead prevention
    ---------------------
    ``position = signal.shift(1).fillna(0)``

    A signal observed at the CLOSE of bar T sets the position that is EFFECTIVE
    from bar T+1.  Therefore the return on bar T is captured only by a signal
    that was set at bar T-1 or earlier — never by the signal at bar T itself.
    """
    # -- Align index -----------------------------------------------------------
    idx      = signals.index.intersection(close.index)
    signals  = signals.reindex(idx)
    close    = close.reindex(idx)

    if len(idx) < 2:
        raise ValueError("Backtest requires at least 2 aligned bars.")

    # -- Bar-delayed positions -------------------------------------------------
    position = signals.shift(1).fillna(0.0)

    # -- Daily returns ---------------------------------------------------------
    raw_return = close.pct_change().fillna(0.0)

    # -- Gross P&L (before costs) ---------------------------------------------
    gross_return = (position * raw_return).fillna(0.0)

    # -- Turnover (absolute change in position) --------------------------------
    # The first bar's turnover equals the opening position (entering from zero).
    first_turnover = pd.Series([abs(position.iloc[0])], index=[position.index[0]])
    rest_turnover  = position.diff().iloc[1:].abs()
    turnover       = pd.concat([first_turnover, rest_turnover])

    # -- Cost drag -------------------------------------------------------------
    cost_drag = (turnover * cost_model.total_per_side).fillna(0.0)

    # -- Net return ------------------------------------------------------------
    net_return = gross_return - cost_drag

    # -- Equity curve ----------------------------------------------------------
    equity = initial_capital * (1.0 + net_return).cumprod()

    # -- Performance metrics --------------------------------------------------
    metrics = compute_all_metrics(net_return, equity, periods_per_year)

    total_gross = (initial_capital * (1.0 + gross_return).cumprod()).iloc[-1]
    total_net   = equity.iloc[-1]
    metrics["total_costs_pct"]     = float(cost_drag.sum())
    metrics["total_return_gross"]  = float(total_gross / initial_capital - 1.0)
    metrics["total_return_net"]    = float(total_net  / initial_capital - 1.0)
    metrics["theory_vs_real_gap"]  = float(metrics["total_return_gross"] - metrics["total_return_net"])
    metrics["turnover_annualised"] = float(turnover.sum() / (len(turnover) / periods_per_year))

    # -- Trade log ------------------------------------------------------------
    trade_log = _extract_trades(position, close)
    metrics.update(_trade_metrics(trade_log))

    return BacktestResult(
        equity_curve  = equity,
        returns       = net_return,
        gross_returns = gross_return,
        positions     = position,
        costs_paid    = cost_drag,
        trade_log     = trade_log,
        metrics       = metrics,
        cost_model    = cost_model,
        signals       = signals,
    )


def _multi_asset_core(
    signals:          pd.DataFrame,
    prices:           pd.DataFrame,
    cost_model:       CostModel,
    initial_capital:  float,
    periods_per_year: int,
) -> BacktestResult:
    """
    Multi-asset vectorised backtest.
    Each asset is run independently then combined with equal notional weight.
    """
    assets    = signals.columns.tolist()
    n_assets  = len(assets)
    all_nets  = []
    all_gross = []
    all_costs = []
    all_pos   = []
    all_trades: List[pd.DataFrame] = []

    for asset in assets:
        sig   = signals[asset]
        close = _get_close(prices[asset] if isinstance(prices, pd.DataFrame) and asset in prices.columns else prices)
        r     = _vectorised_core(sig, close, cost_model, initial_capital / n_assets, periods_per_year)
        all_nets.append(r.returns)
        all_gross.append(r.gross_returns)
        all_costs.append(r.costs_paid)
        all_pos.append(r.positions.rename(asset))
        if not r.trade_log.empty:
            r.trade_log["asset"] = asset
            all_trades.append(r.trade_log)

    # Equal-weight portfolio
    common_idx   = all_nets[0].index
    net_return   = sum(s.reindex(common_idx).fillna(0) for s in all_nets) / n_assets
    gross_return = sum(s.reindex(common_idx).fillna(0) for s in all_gross) / n_assets
    cost_paid    = sum(s.reindex(common_idx).fillna(0) for s in all_costs) / n_assets

    equity    = initial_capital * (1.0 + net_return).cumprod()
    positions = pd.concat(all_pos, axis=1)
    trade_log = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    metrics = compute_all_metrics(net_return, equity, periods_per_year)
    metrics["total_costs_pct"]    = float(cost_paid.sum())
    metrics["total_return_gross"] = float((initial_capital * (1 + gross_return).cumprod()).iloc[-1] / initial_capital - 1)
    metrics["total_return_net"]   = float(equity.iloc[-1] / initial_capital - 1)
    metrics.update(_trade_metrics(trade_log))

    return BacktestResult(
        equity_curve  = equity,
        returns       = net_return,
        gross_returns = gross_return,
        positions     = positions,
        costs_paid    = cost_paid,
        trade_log     = trade_log,
        metrics       = metrics,
        cost_model    = cost_model,
    )


# ---------------------------------------------------------------------------
# Public module-level API
# ---------------------------------------------------------------------------

def run_backtest(
    signals:          Union[pd.Series, pd.DataFrame],
    prices:           Union[pd.Series, pd.DataFrame],
    costs:            Union[CostModel, float, None] = None,
    slippage:         float = 0.0005,
    initial_capital:  float = 100_000.0,
    periods_per_year: int   = 252,
    label:            str   = "",
    apply_risk_management: bool = False,
    predicted_returns: Optional[Union[pd.Series, pd.DataFrame]] = None,
    confidence_scores: Optional[Union[pd.Series, pd.DataFrame]] = None,
    rolling_volatility: Optional[Union[pd.Series, pd.DataFrame]] = None,
    risk_config: Optional[dict] = None,
) -> BacktestResult:
    """
    Run a vectorised backtest with realistic transaction costs.

    Parameters
    ----------
    signals : pd.Series or pd.DataFrame
        Bar-level signals: -1 (short), 0 (flat), +1 (long).
        DataFrame columns are treated as independent assets.
    prices : pd.Series or pd.DataFrame
        OHLCV DataFrame (must contain a 'close' column) or a plain close-price
        Series.  For multi-asset, column names must match those in ``signals``.
    costs : CostModel | float | None
        Transaction cost spec.
        - ``None``  → default ``CostModel`` (0.10% commission + slippage)
        - ``float`` → treated as commission_pct per side
        - ``CostModel`` → used directly
        Use ``CostModel.for_stock()`` for equities, ``CostModel.for_crypto()``
        for crypto, or ``CostModel.zero()`` for a theoretical baseline.
    slippage : float
        Per-side slippage fraction (only used when ``costs`` is None or float).
    initial_capital : float
        Starting portfolio value.
    periods_per_year : int
        Used to annualise returns and volatility (252 for daily equity data).
    label : str
        Optional name for this run (shown in repr and summary).

    Returns
    -------
    BacktestResult
        Contains equity curve, net/gross returns, positions, cost drag, trade
        log, and a full metrics dict.

    Examples
    --------
    >>> from backtest.engine import run_backtest, CostModel
    >>> result = run_backtest(signals, prices, costs=CostModel.for_stock())
    >>> print(result)
    >>> print(result.summary())
    """
    # Resolve cost model
    if costs is None:
        cost_model = CostModel(commission_pct=0.001, slippage_pct=slippage, spread_pct=0.0002)
    elif isinstance(costs, (int, float)):
        cost_model = CostModel(commission_pct=float(costs), slippage_pct=slippage, spread_pct=0.0002)
    else:
        cost_model = costs

    logger.info(
        "run_backtest | signals=%s | cost=%s | capital=%.0f",
        signals.shape, cost_model, initial_capital,
    )

    if apply_risk_management:
        sizer = PositionSizer(config=risk_config if risk_config is not None else load_config())
        if isinstance(signals, pd.DataFrame):
            sized = signals.copy()
            for col in sized.columns:
                px_col = prices[col] if isinstance(prices, pd.DataFrame) and col in prices.columns else _get_close(prices)
                pr_col = predicted_returns[col] if isinstance(predicted_returns, pd.DataFrame) and col in predicted_returns.columns else None
                cf_col = confidence_scores[col] if isinstance(confidence_scores, pd.DataFrame) and col in confidence_scores.columns else None
                rv_col = rolling_volatility[col] if isinstance(rolling_volatility, pd.DataFrame) and col in rolling_volatility.columns else None
                sized[col] = sizer.compute_position_series(
                    signals=sized[col],
                    prices=px_col,
                    rolling_vol=rv_col,
                    predicted_returns=pr_col,
                    confidence_scores=cf_col,
                    initial_equity=initial_capital,
                )
            signals = sized
        else:
            close = _get_close(prices)
            signals = sizer.compute_position_series(
                signals=signals,
                prices=close,
                rolling_vol=rolling_volatility if isinstance(rolling_volatility, pd.Series) else None,
                predicted_returns=predicted_returns if isinstance(predicted_returns, pd.Series) else None,
                confidence_scores=confidence_scores if isinstance(confidence_scores, pd.Series) else None,
                initial_equity=initial_capital,
            )

    if isinstance(signals, pd.DataFrame):
        result = _multi_asset_core(signals, prices, cost_model, initial_capital, periods_per_year)
    else:
        close  = _get_close(prices)
        result = _vectorised_core(signals, close, cost_model, initial_capital, periods_per_year)

    result.label = label

    n_trades = result.metrics.get("total_trades", 0)
    sharpe   = result.metrics.get("sharpe_ratio", 0)
    logger.info("Backtest complete: %s", result)
    return result


def stress_test(
    signals:          Union[pd.Series, pd.DataFrame],
    prices:           Union[pd.Series, pd.DataFrame],
    periods:          Optional[Dict[str, Tuple[str, str]]] = None,
    costs:            Union[CostModel, float, None] = None,
    slippage:         float = 0.0005,
    initial_capital:  float = 100_000.0,
    periods_per_year: int   = 252,
    min_bars:         int   = 10,
    high_volatility_mode: bool = False,
) -> Dict[str, BacktestResult]:
    """
    Re-run the backtest on each stress period and return a dict of results.

    Parameters
    ----------
    periods : dict mapping name → (start_date, end_date) strings, or None to
              use the built-in ``STRESS_PERIODS`` catalogue.
    min_bars : skip any period with fewer bars than this threshold.

    Returns
    -------
    dict mapping period name → BacktestResult

    Examples
    --------
    >>> from backtest.engine import stress_test, STRESS_PERIODS
    >>> results = stress_test(signals, prices)
    >>> for name, r in results.items():
    ...     print(name, r.metrics["sharpe_ratio"])
    """
    if periods is None:
        periods = high_volatility_stress_test_periods() if high_volatility_mode else STRESS_PERIODS

    close_ref = _get_close(prices) if isinstance(prices, pd.Series) else _get_close(prices)
    out: Dict[str, BacktestResult] = {}

    for name, (start, end) in periods.items():
        ts_start = pd.Timestamp(start)
        ts_end   = pd.Timestamp(end)

        # Slice signals
        if isinstance(signals, pd.DataFrame):
            sig_slice = signals.loc[(signals.index >= ts_start) & (signals.index <= ts_end)]
        else:
            sig_slice = signals.loc[(signals.index >= ts_start) & (signals.index <= ts_end)]

        # Slice prices
        if isinstance(prices, pd.DataFrame):
            px_slice = prices.loc[(prices.index >= ts_start) & (prices.index <= ts_end)]
        else:
            px_slice = prices.loc[(prices.index >= ts_start) & (prices.index <= ts_end)]

        if len(sig_slice) < min_bars:
            logger.warning("Stress period %r has only %d bars (< %d) — skipping.", name, len(sig_slice), min_bars)
            continue

        try:
            result = run_backtest(
                sig_slice, px_slice,
                costs=costs, slippage=slippage,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
                label=name,
            )
            out[name] = result
            logger.info(
                "Stress [%-25s]: Sharpe=%.2f  MaxDD=%.1f%%  Return=%.1f%%",
                name,
                result.metrics.get("sharpe_ratio", 0),
                result.metrics.get("max_drawdown", 0) * 100,
                result.metrics.get("total_return_net", 0) * 100,
            )
        except Exception as exc:
            logger.warning("Stress period %r failed: %s", name, exc)

    return out


# ---------------------------------------------------------------------------
# Adversarial Stress Testing
# ---------------------------------------------------------------------------

def adversarial_stress_test(
    signals:          Union[pd.Series, pd.DataFrame],
    prices:           Union[pd.Series, pd.DataFrame],
    n_perturbations:  int   = 50,
    noise_std:        float = 0.05,
    costs:            Union[CostModel, float, None] = None,
    slippage:         float = 0.0005,
    initial_capital:  float = 100_000.0,
    seed:             int   = 42,
) -> dict:
    """
    Evaluate strategy robustness by perturbing signals with noise and
    bootstrapping returns.

    Three adversarial tests:

    1. Signal Noise      — add Gaussian noise to raw signal probabilities
                           and re-threshold. Simulates prediction uncertainty.
    2. Return Shuffle    — randomly permute return blocks (block bootstrap)
                           to generate worst-case path alternatives.
    3. Synthetic Crash   — inject a sudden -10% / -20% / -30% drawdown at a
                           random bar and measure recovery Sharpe.

    Returns a summary dict with:
        base_sharpe, noise_sharpe_mean, noise_sharpe_std,
        bootstrap_sharpe_5th, bootstrap_sharpe_95th,
        crash_sharpes, fragility_score  (0 = robust, 1 = fragile)
    """
    rng = np.random.default_rng(seed)

    # Baseline
    base = run_backtest(signals, prices, costs=costs, slippage=slippage,
                        initial_capital=initial_capital)
    base_sharpe = base.metrics.get("sharpe_ratio", 0.0)

    close = _get_close(prices) if isinstance(prices, pd.Series) else _get_close(prices)

    # ── 1. Signal noise perturbation ──────────────────────────────────────
    sig_arr = np.asarray(signals, dtype=float)
    noise_sharpes: list[float] = []
    for _ in range(n_perturbations):
        noisy = sig_arr + rng.normal(0, noise_std, size=sig_arr.shape)
        noisy_sig = pd.Series(np.sign(noisy).astype(int), index=signals.index)
        try:
            r = run_backtest(noisy_sig, prices, costs=costs, slippage=slippage,
                             initial_capital=initial_capital)
            noise_sharpes.append(r.metrics.get("sharpe_ratio", 0.0))
        except Exception:
            pass

    noise_mean = float(np.mean(noise_sharpes)) if noise_sharpes else base_sharpe
    noise_std_val = float(np.std(noise_sharpes)) if noise_sharpes else 0.0

    # ── 2. Block bootstrap ────────────────────────────────────────────────
    rets = close.pct_change().fillna(0).values
    n    = len(rets)
    block = max(5, n // 20)   # ~5% of series per block
    bootstrap_sharpes: list[float] = []
    for _ in range(n_perturbations):
        idx = np.concatenate([
            np.arange(i, min(i + block, n))
            for i in rng.choice(n, size=n // block + 1)
        ])[:n]
        shuffled_rets = rets[idx]
        shuffled_prices = pd.Series(
            close.iloc[0] * np.cumprod(1 + shuffled_rets),
            index=close.index,
        )
        try:
            r = run_backtest(signals, shuffled_prices, costs=costs, slippage=slippage,
                             initial_capital=initial_capital)
            bootstrap_sharpes.append(r.metrics.get("sharpe_ratio", 0.0))
        except Exception:
            pass

    bs5  = float(np.percentile(bootstrap_sharpes, 5))  if bootstrap_sharpes else base_sharpe
    bs95 = float(np.percentile(bootstrap_sharpes, 95)) if bootstrap_sharpes else base_sharpe

    # ── 3. Synthetic crash injection ──────────────────────────────────────
    crash_sizes = {"crash_10pct": -0.10, "crash_20pct": -0.20, "crash_30pct": -0.30}
    crash_sharpes: dict[str, float] = {}
    crash_idx = int(len(close) * 0.50)  # inject crash at midpoint
    for name, drop in crash_sizes.items():
        px_arr    = close.values.copy().astype(float)
        shock     = np.ones(len(px_arr))
        shock[crash_idx] = 1.0 + drop
        px_crashed = pd.Series(
            px_arr[0] * np.cumprod(shock * (px_arr / np.roll(px_arr, 1))),
            index=close.index,
        )
        px_crashed.iloc[0] = close.iloc[0]
        try:
            r = run_backtest(signals, px_crashed, costs=costs, slippage=slippage,
                             initial_capital=initial_capital)
            crash_sharpes[name] = round(r.metrics.get("sharpe_ratio", 0.0), 3)
        except Exception:
            crash_sharpes[name] = 0.0

    # ── Fragility score (0 = robust, 1 = fragile) ─────────────────────────
    # Measures how much the strategy degrades under adversarial conditions.
    worst_sharpe = min(noise_mean - 2 * noise_std_val, bs5, min(crash_sharpes.values(), default=0.0))
    fragility = float(np.clip((base_sharpe - worst_sharpe) / (abs(base_sharpe) + 1e-9), 0.0, 1.0))

    result = {
        "base_sharpe":          round(base_sharpe, 3),
        "noise_sharpe_mean":    round(noise_mean, 3),
        "noise_sharpe_std":     round(noise_std_val, 3),
        "bootstrap_sharpe_5th": round(bs5, 3),
        "bootstrap_sharpe_95th":round(bs95, 3),
        "crash_sharpes":        crash_sharpes,
        "fragility_score":      round(fragility, 3),
        "n_perturbations":      n_perturbations,
    }
    logger.info(
        "AdversarialStress: base=%.2f  noise_mean=%.2f±%.2f  bs5=%.2f  fragility=%.3f",
        base_sharpe, noise_mean, noise_std_val, bs5, fragility,
    )
    return result


# ---------------------------------------------------------------------------
# BacktestEngine — class wrapper (keeps validation/report.py interface intact)
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Class wrapper around the vectorised backtest functions.
    Accepts the same ``run(features, signals, ticker)`` interface used by
    ``validation/report.py`` and ``main.py``.
    """

    def __init__(self, config: Optional[dict] = None, cost_model: Optional[CostModel] = None) -> None:
        self.cfg        = config if config is not None else load_config()
        self._bt_cfg    = self.cfg.get("backtest", {})
        self.cost_model = cost_model or CostModel(
            commission_pct = self._bt_cfg.get("commission_pct", 0.001),
            slippage_pct   = self._bt_cfg.get("slippage_pct",   0.0005),
            spread_pct     = self._bt_cfg.get("spread_pct",     0.0002),
        )
        self._capital   = self._bt_cfg.get("initial_capital", 100_000.0)

    # ------------------------------------------------------------------
    # Primary interface (used by validation/report.py and main.py)
    # ------------------------------------------------------------------

    def run(
        self,
        features:        pd.DataFrame,
        signals:         pd.Series,
        ticker:          str          = "",
        initial_capital: float        = 0.0,
    ) -> BacktestResult:
        """
        Run a realistic backtest.  ``features`` must contain a 'close' column.
        """
        capital = initial_capital if initial_capital > 0 else self._capital
        result  = run_backtest(
            signals, features,
            costs           = self.cost_model,
            initial_capital = capital,
            label           = ticker,
            apply_risk_management = self._bt_cfg.get("apply_risk_management", True),
            predicted_returns = features["predicted_return"] if "predicted_return" in features.columns else None,
            confidence_scores = features["confidence_score"] if "confidence_score" in features.columns else None,
            rolling_volatility = features["vol_21d"] if "vol_21d" in features.columns else None,
            risk_config = self.cfg,
        )
        return result

    def run_theoretical(
        self,
        features: pd.DataFrame,
        signals:  pd.Series,
    ) -> BacktestResult:
        """
        Zero-cost idealized backtest — used to compute the theory-vs-reality gap.
        """
        result = run_backtest(
            signals, features,
            costs           = CostModel.zero(),
            initial_capital = self._capital,
            label           = "theoretical (zero cost)",
        )
        return result

    def stress_test(
        self,
        features: pd.DataFrame,
        signals:  pd.Series,
        periods:  Optional[Dict[str, Tuple[str, str]]] = None,
        high_volatility_mode: bool = False,
    ) -> Dict[str, BacktestResult]:
        """Run stress tests across pre-defined market regimes."""
        return stress_test(
            signals, features,
            periods         = periods,
            costs           = self.cost_model,
            initial_capital = self._capital,
            high_volatility_mode=high_volatility_mode,
        )
