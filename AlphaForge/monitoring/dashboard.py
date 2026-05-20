"""
AlphaForge — Monitoring Dashboard (programmatic / CLI output)

Provides a lightweight Dashboard class used by main.py commands to print
backtest metrics and save equity curve plots.  This is NOT the Streamlit
research harness dashboard (see harness/harness_dashboard.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class Dashboard:
    """
    Minimal CLI dashboard — prints formatted metrics and optionally saves plots.

    Parameters
    ----------
    config : dict
        The global config dict (passed through but not currently used).
    """

    def __init__(self, config: dict | None = None) -> None:
        self._cfg = config or {}

    # ── Public API ────────────────────────────────────────────────────────────

    def print_metrics(self, equity_curve: pd.Series, label: str = "Backtest") -> None:
        """Print a formatted metrics summary to the terminal."""
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            metrics = self._compute_metrics(equity_curve)
            table = Table(
                title=f"[bold cyan]{label}[/bold cyan]",
                show_header=True,
                header_style="bold",
            )
            table.add_column("Metric", style="dim", min_width=24)
            table.add_column("Value", justify="right")
            for k, v in metrics.items():
                color = "green" if isinstance(v, float) and v > 0 else "white"
                table.add_row(k, f"[{color}]{v}[/]")
            console.print(table)
        except ImportError:
            metrics = self._compute_metrics(equity_curve)
            print(f"\n=== {label} ===")
            for k, v in metrics.items():
                print(f"  {k:<26} {v}")

    def plot(
        self,
        equity_curve: pd.Series,
        title: str = "Equity Curve",
        save_path: str | None = None,
    ) -> None:
        """
        Display or save an equity curve chart.

        Requires matplotlib.  If ``save_path`` is None the chart is shown
        interactively; otherwise it is written to the given path.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("[Dashboard] matplotlib not installed — cannot plot equity curve.")
            return

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity_curve.index, equity_curve.values, color="#3b82f6", linewidth=1.5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(equity_curve: pd.Series) -> dict[str, Any]:
        """Derive standard performance metrics from a NAV series."""
        import numpy as np

        if equity_curve is None or len(equity_curve) < 2:
            return {"Error": "Insufficient data"}

        ret = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_years      = len(ret) / 252
        cagr         = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
        vol          = ret.std() * (252 ** 0.5)
        sharpe       = (ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() > 0 else 0.0
        running_max  = equity_curve.cummax()
        drawdowns    = (equity_curve - running_max) / running_max
        max_dd       = drawdowns.min()

        return {
            "Total Return":    f"{total_return * 100:.2f}%",
            "CAGR":            f"{cagr * 100:.2f}%",
            "Volatility (ann)": f"{vol * 100:.2f}%",
            "Sharpe Ratio":    f"{sharpe:.3f}",
            "Max Drawdown":    f"{max_dd * 100:.2f}%",
            "Bars":            str(len(equity_curve)),
            "Start":           str(equity_curve.index[0].date()),
            "End":             str(equity_curve.index[-1].date()),
        }
