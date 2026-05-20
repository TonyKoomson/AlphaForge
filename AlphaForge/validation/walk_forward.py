"""
Purged walk-forward cross-validation (PWFCV).

Split layout per fold
---------------------
  |<--- TRAIN (train_days) --->|<- EMBARGO (embargo_days) ->|<- TEST (test_days) ->|

The embargo gap is excluded from both training and testing so that overlapping
return windows in the training labels cannot leak information about the test period.

Overfitting score
-----------------
  overfitting_score = max(0, (mean_IS_sharpe - mean_OOS_sharpe) / |mean_IS_sharpe|)

  0   → no detectable overfitting
  1   → OOS edge has fully disappeared
  >1  → model performs worse OOS than a zero-edge strategy

Usage
-----
    from validation.walk_forward import WalkForwardValidator, run_walk_forward
    from features.momentum import MomentumStrategy

    strategy = MomentumStrategy()
    result = run_walk_forward(prices_df, strategy)
    result.print_report()
    result.plot()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtest.engine import CostModel, run_backtest
from utils.helpers import compute_all_metrics, get_logger

logger = get_logger(__name__)
console = Console()

# Type alias: signal_fn(history, target) → pd.Series of {-1, 0, 1}
SignalFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    # In-sample metrics (train window, zero costs to measure pure IS edge)
    is_metrics: dict[str, float] = field(default_factory=dict)
    # Out-of-sample metrics (test window, with realistic costs)
    oos_metrics: dict[str, float] = field(default_factory=dict)

    is_equity: pd.Series = field(default_factory=pd.Series)
    oos_equity: pd.Series = field(default_factory=pd.Series)
    oos_signals: pd.Series = field(default_factory=pd.Series)

    @property
    def is_sharpe(self) -> float:
        return self.is_metrics.get("sharpe_ratio", 0.0)

    @property
    def oos_sharpe(self) -> float:
        return self.oos_metrics.get("sharpe_ratio", 0.0)

    @property
    def is_cagr(self) -> float:
        return self.is_metrics.get("cagr", 0.0)

    @property
    def oos_cagr(self) -> float:
        return self.oos_metrics.get("cagr", 0.0)


@dataclass
class WalkForwardResult:
    ticker: str
    folds: list[FoldResult] = field(default_factory=list)
    oos_equity: pd.Series = field(default_factory=pd.Series)  # stitched OOS equity
    cost_model: CostModel = field(default_factory=CostModel.for_stock)

    # Aggregate metrics across all folds
    mean_is_sharpe: float = 0.0
    mean_oos_sharpe: float = 0.0
    mean_is_cagr: float = 0.0
    mean_oos_cagr: float = 0.0
    overfitting_score: float = 0.0

    # Full-OOS combined backtest metrics
    combined_metrics: dict[str, float] = field(default_factory=dict)

    def _compute_aggregates(self) -> None:
        if not self.folds:
            return
        is_sharpes = [f.is_sharpe for f in self.folds]
        oos_sharpes = [f.oos_sharpe for f in self.folds]
        self.mean_is_sharpe = float(np.mean(is_sharpes))
        self.mean_oos_sharpe = float(np.mean(oos_sharpes))
        self.mean_is_cagr = float(np.mean([f.is_cagr for f in self.folds]))
        self.mean_oos_cagr = float(np.mean([f.oos_cagr for f in self.folds]))

        denom = abs(self.mean_is_sharpe)
        if denom > 1e-9:
            self.overfitting_score = max(
                0.0, (self.mean_is_sharpe - self.mean_oos_sharpe) / denom
            )
        else:
            self.overfitting_score = 0.0

    def print_report(self) -> None:
        """Print a rich formatted report to the console."""
        self._compute_aggregates()

        console.print(f"\n[bold cyan]Walk-Forward Validation Report — {self.ticker}[/]")
        console.print(f"[dim]Folds: {len(self.folds)} | Cost model: {self.cost_model}[/]\n")

        # Per-fold table
        fold_table = Table(title="Per-Fold Results", expand=False)
        fold_table.add_column("Fold", style="cyan", justify="right")
        fold_table.add_column("Train window")
        fold_table.add_column("Test window")
        fold_table.add_column("IS Sharpe", justify="right")
        fold_table.add_column("OOS Sharpe", justify="right")
        fold_table.add_column("IS CAGR %", justify="right")
        fold_table.add_column("OOS CAGR %", justify="right")
        fold_table.add_column("OOS MaxDD %", justify="right")

        for f in self.folds:
            oos_mdd = f.oos_metrics.get("max_drawdown", 0.0) * 100
            is_sharpe_color = "green" if f.is_sharpe > 0.5 else "yellow" if f.is_sharpe > 0 else "red"
            oos_sharpe_color = "green" if f.oos_sharpe > 0.5 else "yellow" if f.oos_sharpe > 0 else "red"
            fold_table.add_row(
                str(f.fold_index + 1),
                f"{f.train_start.date()} → {f.train_end.date()}",
                f"{f.test_start.date()} → {f.test_end.date()}",
                f"[{is_sharpe_color}]{f.is_sharpe:.2f}[/]",
                f"[{oos_sharpe_color}]{f.oos_sharpe:.2f}[/]",
                f"{f.is_cagr * 100:.1f}%",
                f"{f.oos_cagr * 100:.1f}%",
                f"{oos_mdd:.1f}%",
            )
        console.print(fold_table)

        # Summary table
        summary_table = Table(title="Aggregate Summary", expand=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")

        oos_color = "green" if self.mean_oos_sharpe > 0.5 else "yellow" if self.mean_oos_sharpe > 0 else "red"
        of_score = self.overfitting_score
        of_color = "green" if of_score < 0.3 else "yellow" if of_score < 0.7 else "red"

        summary_table.add_row("Mean IS Sharpe", f"{self.mean_is_sharpe:.2f}")
        summary_table.add_row("Mean OOS Sharpe", f"[{oos_color}]{self.mean_oos_sharpe:.2f}[/]")
        summary_table.add_row("Mean IS CAGR", f"{self.mean_is_cagr * 100:.1f}%")
        summary_table.add_row("Mean OOS CAGR", f"{self.mean_oos_cagr * 100:.1f}%")
        summary_table.add_row(
            "Overfitting Score",
            f"[{of_color}]{of_score:.2f}[/] (0=none, 1=full degradation)",
        )

        if self.combined_metrics:
            combined_sharpe = self.combined_metrics.get("sharpe_ratio", 0.0)
            combined_mdd = self.combined_metrics.get("max_drawdown", 0.0) * 100
            combined_cagr = self.combined_metrics.get("cagr", 0.0) * 100
            summary_table.add_row("─" * 20, "─" * 12)
            summary_table.add_row("Combined OOS Sharpe", f"{combined_sharpe:.2f}")
            summary_table.add_row("Combined OOS CAGR", f"{combined_cagr:.1f}%")
            summary_table.add_row("Combined OOS MaxDD", f"{combined_mdd:.1f}%")

        console.print(summary_table)

        # Verdict
        if self.mean_oos_sharpe >= 0.5 and of_score < 0.5:
            verdict = "[bold green]PASS[/] — Strategy shows genuine OOS edge with acceptable IS/OOS degradation."
        elif self.mean_oos_sharpe > 0 and of_score < 0.8:
            verdict = "[bold yellow]MARGINAL[/] — Some OOS edge, but degradation suggests overfitting risk."
        else:
            verdict = "[bold red]FAIL[/] — OOS performance does not support real-world deployment."

        console.print(f"\nVerdict: {verdict}\n")

    def plot(self, save_path: Optional[str] = None) -> None:
        """Plot stitched OOS equity curve with fold boundaries."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not available — skipping plot")
            return

        if self.oos_equity.empty:
            logger.warning("No OOS equity to plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"Walk-Forward Validation — {self.ticker}", fontsize=14, fontweight="bold")

        ax_eq, ax_dd = axes

        # Equity curve
        ax_eq.plot(self.oos_equity.index, self.oos_equity.values, color="#2196F3", linewidth=1.5, label="OOS Equity")
        ax_eq.set_ylabel("Portfolio Value ($)")
        ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_eq.grid(alpha=0.3)
        ax_eq.legend(loc="upper left")

        # Fold boundary lines
        for f in self.folds:
            ax_eq.axvline(f.test_start, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        # Drawdown
        rolling_max = self.oos_equity.cummax()
        drawdown = (self.oos_equity - rolling_max) / rolling_max * 100
        ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.4)
        ax_dd.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=0.8)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.grid(alpha=0.3)

        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Walk-forward plot saved to %s", save_path)
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Fold generator
# ---------------------------------------------------------------------------

def _generate_folds(
    index: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    embargo_days: int,
    min_folds: int,
    step_days: Optional[int],
) -> list[tuple[int, int, int, int]]:
    """
    Yields (train_start_i, train_end_i, test_start_i, test_end_i) index positions.

    The walk advances by `step_days` bars each fold (default = test_days).
    """
    if step_days is None:
        step_days = test_days

    n = len(index)
    folds = []

    fold_start = 0
    while True:
        train_end = fold_start + train_days - 1
        embargo_end = train_end + embargo_days
        test_start = embargo_end + 1
        test_end = test_start + test_days - 1

        if test_end >= n:
            break

        folds.append((fold_start, train_end, test_start, test_end))
        fold_start += step_days

    if len(folds) < min_folds:
        raise ValueError(
            f"Only {len(folds)} folds generated but min_folds={min_folds}. "
            f"Increase the date range or decrease train/test/embargo windows."
        )

    return folds


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Purged walk-forward cross-validator for trading strategies.

    Parameters
    ----------
    train_days : Trading days per training window (default 126 ≈ 6 months).
    test_days  : Trading days per test window    (default  21 ≈ 1 month).
    embargo_days : Bars excluded between train end and test start (default 21).
    min_folds  : Minimum number of folds required (raises if not met).
    step_days  : Bars to advance the window each fold (default = test_days).
    cost_model : CostModel used for OOS and combined backtests.
    initial_capital : Starting capital for each fold and combined backtest.
    """

    def __init__(
        self,
        train_days: int = 126,
        test_days: int = 21,
        embargo_days: int = 21,
        min_folds: int = 5,
        step_days: Optional[int] = None,
        cost_model: Optional[CostModel] = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.min_folds = min_folds
        self.step_days = step_days
        self.cost_model = cost_model or CostModel.for_stock()
        self.initial_capital = initial_capital

    def run(
        self,
        prices: pd.DataFrame,
        signal_fn: SignalFn,
        ticker: str = "ASSET",
    ) -> WalkForwardResult:
        """
        Execute the walk-forward validation.

        Parameters
        ----------
        prices : DataFrame with at least a 'close' column, DatetimeIndex.
        signal_fn : Callable(history_df, target_df) → pd.Series of {-1, 0, 1}.
                    history_df provides warmup data; target_df is the test window.
        ticker : Label used in reporting.

        Returns
        -------
        WalkForwardResult with per-fold and aggregate metrics.
        """
        prices = prices.sort_index()
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must have a 'close' column")

        index = prices.index
        fold_positions = _generate_folds(
            index,
            train_days=self.train_days,
            test_days=self.test_days,
            embargo_days=self.embargo_days,
            min_folds=self.min_folds,
            step_days=self.step_days,
        )

        logger.info(
            "Walk-forward: %d folds | train=%d embargo=%d test=%d bars",
            len(fold_positions),
            self.train_days,
            self.embargo_days,
            self.test_days,
        )

        result = WalkForwardResult(ticker=ticker, cost_model=self.cost_model)
        all_oos_signals: list[pd.Series] = []
        all_oos_prices: list[pd.Series] = []

        for fold_idx, (ts_i, te_i, tti_i, tte_i) in enumerate(fold_positions):
            train_df = prices.iloc[ts_i : te_i + 1]
            test_df = prices.iloc[tti_i : tte_i + 1]

            train_start = index[ts_i]
            train_end = index[te_i]
            test_start = index[tti_i]
            test_end = index[tte_i]

            logger.debug(
                "Fold %d: train %s→%s, test %s→%s",
                fold_idx + 1,
                train_start.date(),
                train_end.date(),
                test_start.date(),
                test_end.date(),
            )

            # Generate IS signals (train window only, no history warmup)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                is_signals = signal_fn(train_df.iloc[:0], train_df)  # empty history
                oos_signals = signal_fn(train_df, test_df)

            train_close = train_df["close"]
            test_close = test_df["close"]

            # IS metrics — zero costs to measure raw edge
            is_bt = run_backtest(
                is_signals,
                train_close,
                costs=CostModel.zero(),
                initial_capital=self.initial_capital,
                label=f"fold{fold_idx}_IS",
            )

            # OOS metrics — realistic costs
            oos_bt = run_backtest(
                oos_signals,
                test_close,
                costs=self.cost_model,
                initial_capital=self.initial_capital,
                label=f"fold{fold_idx}_OOS",
            )

            fold = FoldResult(
                fold_index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                is_metrics=is_bt.metrics,
                oos_metrics=oos_bt.metrics,
                is_equity=is_bt.equity_curve,
                oos_equity=oos_bt.equity_curve,
                oos_signals=oos_signals,
            )
            result.folds.append(fold)

            all_oos_signals.append(oos_signals)
            all_oos_prices.append(test_close)

        # Stitch OOS equity and compute combined metrics
        if all_oos_signals:
            combined_signals = pd.concat(all_oos_signals)
            combined_prices = pd.concat(all_oos_prices)
            combined_bt = run_backtest(
                combined_signals,
                combined_prices,
                costs=self.cost_model,
                initial_capital=self.initial_capital,
                label="combined_OOS",
            )
            result.oos_equity = combined_bt.equity_curve
            result.combined_metrics = combined_bt.metrics

        result._compute_aggregates()

        logger.info(
            "Walk-forward complete: IS Sharpe=%.2f OOS Sharpe=%.2f Overfitting=%.2f",
            result.mean_is_sharpe,
            result.mean_oos_sharpe,
            result.overfitting_score,
        )
        return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_walk_forward(
    prices: pd.DataFrame,
    signal_fn: SignalFn,
    ticker: str = "ASSET",
    train_days: int = 126,
    test_days: int = 21,
    embargo_days: int = 21,
    min_folds: int = 5,
    cost_model: Optional[CostModel] = None,
    initial_capital: float = 100_000.0,
    print_report: bool = True,
    plot: bool = False,
    save_plot: Optional[str] = None,
) -> WalkForwardResult:
    """
    One-call entry point for purged walk-forward validation.

    Parameters
    ----------
    prices : DataFrame with 'close' column and DatetimeIndex.
    signal_fn : Callable(history_df, target_df) → pd.Series of {-1, 0, 1}.
    ticker : Label for the asset being tested.
    train_days : Bars per training window (default 126 ≈ 6 months).
    test_days : Bars per test window (default 21 ≈ 1 month).
    embargo_days : Purge gap between train and test (default 21 ≈ 1 month).
    min_folds : Raise if fewer folds can be formed (default 5).
    cost_model : CostModel for OOS evaluation (default for_stock()).
    initial_capital : Starting capital per fold (default 100_000).
    print_report : Print rich summary table (default True).
    plot : Show matplotlib equity curve (default False).
    save_plot : Path to save the equity-curve figure (optional).

    Returns
    -------
    WalkForwardResult
    """
    validator = WalkForwardValidator(
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        min_folds=min_folds,
        cost_model=cost_model,
        initial_capital=initial_capital,
    )
    result = validator.run(prices, signal_fn, ticker=ticker)

    if print_report:
        result.print_report()

    if plot or save_plot:
        result.plot(save_path=save_plot)

    return result
