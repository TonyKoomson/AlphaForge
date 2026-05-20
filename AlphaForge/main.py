"""
Alpha Forge — CLI entry point.

All commands operate in simulation / backtesting mode only.
No real broker connections. No live trading. No real money.

Usage
-----
    python main.py --help
    python main.py ingest    --ticker SPY --start 2018-01-01
    python main.py features  --ticker SPY
    python main.py train     --ticker SPY
    python main.py backtest  --ticker SPY --start 2020-01-01 --end 2024-01-01
    python main.py validate  --ticker SPY
    python main.py baseline  --ticker SPY
    python main.py paper-trade --ticker SPY --replay-start 2023-01-01
    python main.py live      --ticker SPY --duration 30d
    python main.py dashboard --ticker SPY
    python main.py report    --ticker SPY
    python main.py drift-check --ticker SPY
    python main.py overfit-check --ticker SPY --n-tests 50
    python main.py research-v2   --ticker SPY --cycles 5 [--use-llm]
    python main.py config-show

Whole-market universe commands:
    python main.py fetch-universe  --name sp500
    python main.py train-universe  --universe sp500 [--resume] [--max 50]
    python main.py universe-trade  --universe sp500 --top-n 20 --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for → ✓ █ etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="alpha",
    help=(
        "[bold cyan]Alpha Forge[/] — AI trading strategy research & validation platform.\n\n"
        "[bold yellow]IMPORTANT:[/] Simulation and backtesting ONLY. "
        "No real money. No live trading. No broker connections."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]Alpha Forge[/] v{_VERSION}")
        raise typer.Exit()


@app.callback()
def _global_options(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Alpha Forge — AI trading strategy research & validation platform."""


def _load_cfg(config_path: str) -> dict:
    from utils.helpers import load_config
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/] {config_path}")
        console.print("[dim]Create one with: cp config.yaml.example config.yaml[/]")
        raise typer.Exit(1)
    return cfg


def _banner(title: str, cfg: dict, ticker: str = "", extra: str = "") -> None:
    data_cfg = cfg.get("data", {})
    lines = [f"[bold]{title}[/]"]
    if ticker:
        lines.append(f"Ticker: [cyan]{ticker}[/]")
    lines.append(f"Source: [dim]{data_cfg.get('source', 'yfinance')}[/]   "
                 f"Frequency: [dim]{data_cfg.get('frequency', '1d')}[/]")
    if extra:
        lines.append(extra)
    console.print(Panel("\n".join(lines), expand=False))


def _make_spinner(description: str) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _load_data_and_features(
    cfg: dict,
    ticker: str,
    start: str,
    end: str,
    save_features: bool = False,
) -> tuple:
    """Shared data → feature pipeline with progress display."""
    import pandas as pd
    from data.ingest import DataIngestion
    from features.engine import FeatureEngine

    with _make_spinner("") as prog:
        t = prog.add_task(f"Fetching data for [cyan]{ticker}[/]…", total=None)
        df = DataIngestion(config=cfg).get_data(ticker, as_of_date=end)
        df = df[df.index >= pd.Timestamp(start)]
        prog.update(t, description=f"Building features for [cyan]{ticker}[/]…")
        engine = FeatureEngine(config=cfg)
        feat_df = engine.build(df, ticker=ticker, save=save_features)

    console.print(
        f"[green]OK[/] Data: [bold]{len(df)}[/] bars  "
        f"([dim]{df.index.min().date()} → {df.index.max().date()}[/])   "
        f"Features: [bold]{feat_df.shape[1]}[/] columns"
    )
    return df, feat_df, engine


def _load_signals(cfg: dict, feat_df, engine, ticker: str = "") -> tuple:
    """Load trained model and generate signals with progress display."""
    import numpy as np
    import pandas as pd
    from models.train import ModelTrainer
    from paper_trading.loop import _apply_signal_filters, _regime_conditional_signals

    with _make_spinner("") as prog:
        prog.add_task("Generating signals…", total=None)
        trainer = ModelTrainer(config=cfg)
        if ticker:
            trainer.load(ticker)

    bt_cfg = cfg.get("backtest", {})
    feat_cols = [c for c in engine.feature_columns if c in feat_df.columns]
    probas = trainer.predict_proba(feat_df[feat_cols])

    _default_rt = {
        "bull":     {"long": 0.53, "short": 0.80},
        "sideways": {"long": 0.62, "short": 0.62},
        "bear":     {"long": 0.78, "short": 0.53},
        "high_vol": {"long": 0.68, "short": 0.68},
        "unknown":  {"long": 0.65, "short": 0.65},
    }
    raw_signals = _regime_conditional_signals(
        probas=probas,
        index=feat_df.index,
        features=feat_df,
        regime_thresholds=bt_cfg.get("regime_thresholds", _default_rt),
        fallback_threshold=bt_cfg.get("signal_threshold", 0.65),
        trend_filter_200ma=bt_cfg.get("trend_filter_200ma", True),
    )
    signals = _apply_signal_filters(
        raw_signals,
        min_holding_bars=bt_cfg.get("min_holding_bars", 5),
        confirm_bars=bt_cfg.get("signal_confirm_bars", 2),
    )
    return trainer, feat_cols, probas, signals


def _assert_data_downloaded(cfg: dict, ticker: str) -> None:
    """Check cached Parquet exists and warn gracefully if not."""
    from data.ingest import DataIngestion
    ing = DataIngestion(config=cfg)
    path = ing._symbol_path(ticker)
    if not path.exists():
        console.print(
            f"[yellow]No cached data for[/] [cyan]{ticker}[/]. "
            f"Run: [bold]python main.py ingest --ticker {ticker}[/]"
        )
        raise typer.Exit(1)


def _assert_model_exists(cfg: dict, ticker: str) -> None:
    """Check a trained model artefact exists."""
    from utils.helpers import ensure_dir
    artifacts_dir = Path(cfg.get("model", {}).get("artifacts_dir", "models/artifacts"))
    model_path = artifacts_dir / f"{ticker.lower()}_model.joblib"
    if not model_path.exists():
        console.print(
            f"[yellow]No trained model for[/] [cyan]{ticker}[/]. "
            f"Run: [bold]python main.py train --ticker {ticker}[/]"
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    ticker: str = typer.Option(..., "--ticker", "-t",
                               help="Ticker symbol, e.g. [cyan]SPY[/] or [cyan]BTC-USD[/]"),
    start:  str  = typer.Option("2018-01-01", "--start",  help="Start date YYYY-MM-DD"),
    end:    Optional[str] = typer.Option(None, "--end",   help="End date (default: today)"),
    force:  bool = typer.Option(False, "--force",         help="Re-download even if cached"),
    config_path: str = typer.Option("config.yaml", "--config", help="Path to config.yaml"),
) -> None:
    """Download and cache daily OHLCV data for a symbol."""
    from data.ingest import DataIngestion
    cfg = _load_cfg(config_path)
    _banner("Data Ingestion", cfg, ticker)

    with _make_spinner("") as prog:
        prog.add_task(f"Downloading [cyan]{ticker}[/]…", total=None)
        ing = DataIngestion(config=cfg)
        result = ing.download_data([ticker], start_date=start, end_date=end, force_download=force)

    df = result[ticker]
    path = ing._symbol_path(ticker)

    table = Table(show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Ticker",   f"[cyan]{ticker}[/]")
    table.add_row("Rows",     f"[bold]{len(df):,}[/]")
    table.add_row("Date range", f"{df.index.min().date()} → {df.index.max().date()}")
    table.add_row("Cached at", str(path))
    console.print(Panel(table, title="[green]Ingest complete[/]", expand=False))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@app.command()
def features(
    ticker:  str = typer.Option(..., "--ticker", "-t"),
    start:   str = typer.Option("2018-01-01", "--start"),
    end:     str = typer.Option("2024-01-01", "--end"),
    select:  bool = typer.Option(False, "--select",
                                  help="Run feature selection (MI + SHAP + correlation pruning)"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Engineer features from cached OHLCV data and save to data/processed/."""
    cfg = _load_cfg(config_path)
    _assert_data_downloaded(cfg, ticker)
    _banner("Feature Engineering", cfg, ticker, f"Period: {start} → {end}")

    _, feat_df, engine = _load_data_and_features(cfg, ticker, start, end, save_features=True)

    if select:
        with _make_spinner("") as prog:
            prog.add_task("Running feature selection…", total=None)
            feat_df = engine.select_features(feat_df)
        console.print(f"[green]OK[/] Selected [bold]{len(engine.feature_columns)}[/] features after pruning")

    console.print(f"[dim]Feature columns:[/] {', '.join(engine.feature_columns[:8])}…")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    ticker:  str  = typer.Option(..., "--ticker", "-t"),
    start:   Optional[str] = typer.Option(None, "--start",
                                          help="Override training start (default: config)"),
    end:     Optional[str] = typer.Option(None, "--end",
                                          help="Override training end (default: config)"),
    fast:    bool = typer.Option(False, "--fast/--no-fast",
                                 help="Fast mode: fewer folds, smaller ensemble, histogram trees"),
    no_cache: bool = typer.Option(False, "--no-cache",
                                  help="Recompute features even if cache exists"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Train the prediction model on the processed feature matrix.

    --fast  Activates fast-train mode: 2-3x fewer folds, 2 ensemble members,
            early stopping. Ideal for hyperparameter sweeps and universe-scale
            training. Use standard mode for final champion models.
    """
    import os, time, pandas as pd
    from data.ingest import DataIngestion
    from features.engine import FeatureEngine
    from features.feature_cache import get_or_compute_features
    from models.train import ModelTrainer

    cfg = _load_cfg(config_path)

    # Propagate fast mode into config and env
    if fast:
        cfg.setdefault("model", {})["fast_mode"] = True
        os.environ["ALPHAFORGE_FAST_MODE"] = "1"
        console.print("[yellow]Fast mode ON[/] — fewer folds, smaller ensemble, hist trees")
    else:
        os.environ.pop("ALPHAFORGE_FAST_MODE", None)

    _assert_data_downloaded(cfg, ticker)
    _banner("Model Training", cfg, ticker,
            f"Algorithm: [cyan]{cfg.get('model', {}).get('algorithm', 'xgboost')}[/]")

    data_cfg = cfg["data"]
    train_end   = end   or data_cfg["end_date"]
    train_start = start or data_cfg["start_date"]

    # --- Feature engineering with cache ---
    t_feat = time.time()
    raw_dir  = Path(cfg.get("data", {}).get("cache_dir", "data/raw"))
    raw_path = raw_dir / f"{ticker.lower()}_daily.parquet"

    if raw_path.exists():
        console.print(f"[dim]Feature cache check for {ticker}...[/]", end=" ")
        with _make_spinner("") as prog:
            prog.add_task("", total=None)
            feat_df = get_or_compute_features(
                ticker=ticker,
                raw_path=raw_path,
                config=cfg,
                as_of_date=train_end,
                force_recompute=no_cache,
            )
    else:
        # Fallback: use legacy FeatureEngine.build() path
        with _make_spinner("") as prog:
            prog.add_task(f"Building features for [cyan]{ticker}[/]…", total=None)
            df = DataIngestion(config=cfg).get_data(ticker, as_of_date=train_end)
            df = df[df.index >= pd.Timestamp(train_start)]
            feat_df = FeatureEngine(config=cfg).build(df, ticker=ticker)

    feat_time = time.time() - t_feat
    console.print(
        f"[green]Features:[/] {feat_df.shape[0]:,} rows x {feat_df.shape[1]} cols  "
        f"[dim]({feat_time:.1f}s)[/]"
    )

    # --- Model training ---
    t_train = time.time()
    with _make_spinner("") as prog:
        prog.add_task("Training model…", total=None)
        metrics = ModelTrainer(config=cfg).train(feat_df, ticker=ticker)
    train_time = time.time() - t_train
    console.print(f"[dim]Training time: {train_time:.1f}s[/]")

    table = Table(title="[bold]Training Complete — OOS Metrics[/]", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        color = "green" if isinstance(v, float) and v > 0.55 else "white"
        table.add_row(k.replace("_", " ").title(), f"[{color}]{v}[/]")
    console.print(table)
    # Machine-readable line for parallel_train.py subprocess parsing
    _sharpe_val = metrics.get("sharpe_like", metrics.get("Sharpe Like", 0.0))
    print(f"CV_SHARPE={_sharpe_val}", flush=True)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@app.command()
def backtest(
    ticker:    str   = typer.Option(..., "--ticker", "-t"),
    start:     str   = typer.Option("2020-01-01", "--start"),
    end:       str   = typer.Option("2024-01-01", "--end"),
    capital:   float = typer.Option(100_000.0,    "--capital"),
    costs:     str   = typer.Option("stock",      "--costs",
                                    help="Cost preset: stock | crypto | zero"),
    save_plot: Optional[str] = typer.Option(None, "--save-plot",
                                             help="Save equity chart to path"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Run a realistic backtest with transaction costs. Simulation only."""
    from backtest.engine import BacktestEngine, CostModel
    from monitoring.dashboard import Dashboard

    cfg = _load_cfg(config_path)
    _assert_data_downloaded(cfg, ticker)
    _assert_model_exists(cfg, ticker)
    _banner("Backtest", cfg, ticker,
            f"Period: [bold]{start}[/] → [bold]{end}[/]   Capital: [bold]${capital:,.0f}[/]")

    cost_map = {"stock": CostModel.for_stock(), "crypto": CostModel.for_crypto(), "zero": CostModel.zero()}
    cost_model = cost_map.get(costs.lower(), CostModel.for_stock())

    _, feat_df, engine = _load_data_and_features(cfg, ticker, start, end)

    # Use the same regime-conditional signal pipeline as paper trading
    # (confirmation filter + regime thresholds) so backtest results are
    # consistent with live/replay results.
    trainer, feat_cols, probas, signals = _load_signals(cfg, feat_df, engine, ticker=ticker)

    with _make_spinner("") as prog:
        prog.add_task("Running backtest…", total=None)
        result = BacktestEngine(config=cfg).run(feat_df, signals, ticker, initial_capital=capital)

    dash = Dashboard(config=cfg)
    dash.print_metrics(result.equity_curve, label=f"{ticker} Backtest")

    # Cost summary
    m = result.metrics
    extras = Table(show_header=False, box=None)
    extras.add_column(style="dim", min_width=24)
    extras.add_column(justify="right")
    extras.add_row("Total trades",     str(int(m.get("total_trades", 0))))
    extras.add_row("Total cost drag",  f"{m.get('total_costs_pct', 0)*100:.3f}%")
    extras.add_row("Theory vs reality gap", f"{m.get('theory_vs_real_gap', 0)*100:.2f}%")
    extras.add_row("Cost preset used", costs)
    console.print(Panel(extras, title="[dim]Cost & Execution Details[/]", expand=False))

    if save_plot:
        dash.plot(result.equity_curve, title=f"{ticker} Backtest", save_path=save_plot)
        console.print(f"[green]OK[/] Chart saved: {save_plot}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@app.command()
def validate(
    ticker:  str = typer.Option(..., "--ticker", "-t"),
    output:  str = typer.Option("reports/", "--output", "-o",
                                 help="Directory to write report files"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Run purged walk-forward validation and generate theory-vs-reality report."""
    import pandas as pd
    from data.ingest import DataIngestion
    from features.engine import FeatureEngine
    from validation.report import ValidationReport

    cfg = _load_cfg(config_path)
    _assert_data_downloaded(cfg, ticker)
    _assert_model_exists(cfg, ticker)
    cfg["validation"]["metrics_output_dir"] = output
    _banner("Walk-Forward Validation", cfg, ticker,
            f"Folds: [bold]{cfg['validation'].get('n_splits', 5)}[/]   "
            f"Output: [dim]{output}[/]")

    data_cfg = cfg["data"]
    with _make_spinner("") as prog:
        prog.add_task(f"Loading data for [cyan]{ticker}[/]…", total=None)
        df = DataIngestion(config=cfg).get_data(ticker, as_of_date=data_cfg["end_date"])
        df = df[df.index >= pd.Timestamp(data_cfg["start_date"])]
        prog.add_task(f"Building features…", total=None)
        feat_df = FeatureEngine(config=cfg).build(df, ticker=ticker)

    ValidationReport(config=cfg).run(feat_df, ticker=ticker)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    ticker:        str   = typer.Option(..., "--ticker", "-t"),
    start:         str   = typer.Option("2018-01-01", "--start"),
    end:           str   = typer.Option("2024-01-01", "--end"),
    fast:          int   = typer.Option(20,   "--fast",          help="Fast MA window"),
    slow:          int   = typer.Option(50,   "--slow",          help="Slow MA window"),
    rsi_period:    int   = typer.Option(14,   "--rsi-period"),
    rsi_threshold: float = typer.Option(50.0, "--rsi-threshold"),
    allow_short: bool    = typer.Option(True, "--allow-short/--long-only"),
    train_days:    int   = typer.Option(126,  "--train-days",    help="Walk-forward train window (bars)"),
    test_days:     int   = typer.Option(21,   "--test-days",     help="Walk-forward test window (bars)"),
    embargo_days:  int   = typer.Option(21,   "--embargo-days",  help="Purge embargo gap (bars)"),
    capital:       float = typer.Option(100_000.0, "--capital"),
    plot:          bool  = typer.Option(False, "--plot",         help="Show equity curve"),
    save_plot: Optional[str] = typer.Option(None, "--save-plot", help="Save plot to path"),
    config_path: str     = typer.Option("config.yaml", "--config"),
) -> None:
    """Run MA-crossover + RSI baseline through purged walk-forward validation."""
    import pandas as pd
    from data.ingest import DataIngestion
    from features.momentum import MomentumStrategy
    from validation.walk_forward import run_walk_forward
    from backtest.engine import CostModel

    cfg = _load_cfg(config_path)
    _assert_data_downloaded(cfg, ticker)
    _banner("Momentum Baseline", cfg, ticker,
            f"SMA {fast}/{slow} + RSI({rsi_period})   Period: {start} → {end}")

    with _make_spinner("") as prog:
        prog.add_task(f"Loading data for [cyan]{ticker}[/]…", total=None)
        df = DataIngestion(config=cfg).get_data(ticker, as_of_date=end)
        df = df[df.index >= pd.Timestamp(start)]

    run_walk_forward(
        prices=df,
        signal_fn=MomentumStrategy(
            fast_period=fast, slow_period=slow,
            rsi_period=rsi_period, rsi_threshold=rsi_threshold,
            allow_short=allow_short,
        ),
        ticker=ticker,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        cost_model=CostModel.for_stock(),
        initial_capital=capital,
        print_report=True,
        plot=plot,
        save_plot=save_plot,
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@app.command()
def report(
    ticker:     str  = typer.Option(..., "--ticker", "-t"),
    output:     str  = typer.Option("reports/", "--output", "-o",
                                     help="Output directory for report files"),
    format_:    str  = typer.Option("json", "--format",
                                     help="Output format: json | markdown"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Generate a full validation report: walk-forward, theory vs reality, regime analysis.
    Writes JSON (and optionally Markdown) to the output directory.
    """
    import json
    import pandas as pd
    from datetime import datetime
    from data.ingest import DataIngestion
    from features.engine import FeatureEngine
    from validation.report import ValidationReport
    from utils.helpers import ensure_dir

    cfg = _load_cfg(config_path)
    _assert_data_downloaded(cfg, ticker)
    _assert_model_exists(cfg, ticker)
    cfg["validation"]["metrics_output_dir"] = output
    out_dir = ensure_dir(output)

    _banner("Validation Report", cfg, ticker,
            f"Format: [cyan]{format_}[/]   Output: [dim]{output}[/]")

    data_cfg = cfg["data"]
    with _make_spinner("") as prog:
        prog.add_task(f"Loading data for [cyan]{ticker}[/]…", total=None)
        df = DataIngestion(config=cfg).get_data(ticker, as_of_date=data_cfg["end_date"])
        df = df[df.index >= pd.Timestamp(data_cfg["start_date"])]
        prog.add_task("Building features…", total=None)
        feat_df = FeatureEngine(config=cfg).build(df, ticker=ticker)

    results = ValidationReport(config=cfg).run(feat_df, ticker=ticker)

    if format_.lower() == "markdown":
        md_path = _write_markdown_report(results, ticker, out_dir)
        console.print(f"[green]OK[/] Markdown report: [bold]{md_path}[/]")

    json_files = sorted(out_dir.glob(f"{ticker.lower()}_validation_*.json"))
    if json_files:
        console.print(f"[green]OK[/] JSON report: [bold]{json_files[-1]}[/]")


def _write_markdown_report(results: dict, ticker: str, out_dir: Path) -> Path:
    """Render the validation result dict as a Markdown file."""
    from datetime import datetime
    summary = results.get("summary", {})
    tvr     = results.get("theory_vs_reality", {})
    wf      = results.get("walk_forward", [])

    lines = [
        f"# Alpha Forge — Validation Report: {ticker}",
        f"\n_Generated: {results.get('generated_at', 'unknown')}_\n",
        "---\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean OOS Sharpe | `{summary.get('mean_oos_sharpe', 0):.3f}` |",
        f"| Mean OOS CAGR   | `{summary.get('mean_oos_cagr', 0)*100:.1f}%` |",
        f"| Mean OOS MaxDD  | `{summary.get('mean_oos_max_drawdown', 0)*100:.1f}%` |",
        f"| Theory vs Reality Sharpe Gap | `{summary.get('theory_vs_reality_sharpe_gap_pct', 0):.1f}%` |",
        f"| Edge Survives Costs | `{'YES' if summary.get('edge_survives_costs') else 'NO'}` |",
        f"| Folds with Positive Sharpe | `{summary.get('n_folds_positive_sharpe', 0)}` |",
        f"\n**Verdict:** {summary.get('verdict', '—')}\n",
        "---\n",
        "## Walk-Forward Fold Results\n",
        "| Fold | Train Start | Test Start | Test End | Sharpe | CAGR | Max DD |",
        "|------|-------------|------------|----------|--------|------|--------|",
    ]
    for r in wf:
        lines.append(
            f"| {r.get('fold','')} "
            f"| {r.get('train_start','')} "
            f"| {r.get('test_start','')} "
            f"| {r.get('test_end','')} "
            f"| {r.get('sharpe_ratio',0):.3f} "
            f"| {r.get('cagr',0)*100:.1f}% "
            f"| {r.get('max_drawdown',0)*100:.1f}% |"
        )
    lines += [
        "\n---\n",
        "## Theory vs Reality\n",
        f"| | Theoretical | Realistic |",
        f"|---|---|---|",
        f"| Sharpe | `{tvr.get('theoretical',{}).get('sharpe_ratio',0):.3f}` "
        f"| `{tvr.get('realistic',{}).get('sharpe_ratio',0):.3f}` |",
        f"| CAGR   | `{tvr.get('theoretical',{}).get('cagr',0)*100:.1f}%` "
        f"| `{tvr.get('realistic',{}).get('cagr',0)*100:.1f}%` |",
        f"\n**Sharpe gap (costs):** `{tvr.get('sharpe_gap_pct',0):.1f}%`\n",
        "---\n",
        "> This report is for research purposes only. "
        "No investment advice. No real capital involved.\n",
    ]

    from datetime import datetime
    fname = f"{ticker.lower()}_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    path  = out_dir / fname
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# paper-trade
# ---------------------------------------------------------------------------

@app.command(name="paper-trade")
def paper_trade(
    ticker:        str   = typer.Option(..., "--ticker", "-t",
                                        help="Ticker symbol, e.g. SPY"),
    replay_start:  str   = typer.Option("2023-01-01", "--replay-start",
                                        help="Start date for paper trading replay (ISO format)"),
    capital:       float = typer.Option(100_000.0, "--capital",
                                        help="Starting simulated capital"),
    config_path:   str   = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Simulate paper trading bar-by-bar from replay_start to today.

    Loads the trained model, fetches market data (yfinance), applies the full
    risk management stack, and logs every bar to:
      logs/paper_trading/{ticker}_paper_trades.csv
      logs/paper_trading/{ticker}_equity_curve.csv

    Results are visible in the dashboard (Live Markets and Home tabs).

    This is SIMULATION ONLY — no real broker, no real money.

    Examples:
        python main.py paper-trade --ticker SPY
        python main.py paper-trade --ticker SPY --replay-start 2022-01-01 --capital 50000
    """
    import pandas as pd
    from datetime import date
    from data.ingest import DataIngestion
    from features.engine import FeatureEngine
    from models.train import ModelTrainer
    from paper_trading.simulator import run_paper_trade

    cfg = _load_cfg(config_path)
    _assert_data_downloaded(cfg, ticker)
    _assert_model_exists(cfg, ticker)

    _banner("Paper Trading Simulation", cfg, ticker,
            f"Replay from: [cyan]{replay_start}[/]   Capital: [bold]${capital:,.0f}[/]")

    today = date.today().isoformat()
    with _make_spinner("") as prog:
        prog.add_task(f"Fetching latest data for [cyan]{ticker}[/]…", total=None)
        df = DataIngestion(config=cfg).get_data(ticker, as_of_date=today)
        prog.add_task("Building features…", total=None)
        engine = FeatureEngine(config=cfg)
        feat_df = engine.build(df, ticker=ticker, save=False)

    console.print(
        f"[green]OK[/] Data: [bold]{len(df)}[/] bars  "
        f"([dim]{df.index.min().date()} → {df.index.max().date()}[/])"
    )

    with _make_spinner("") as prog:
        prog.add_task("Loading model…", total=None)
        trainer = ModelTrainer(config=cfg)
        trainer.load(ticker)

    with _make_spinner("") as prog:
        prog.add_task("Running bar-by-bar simulation…", total=None)
        result = run_paper_trade(
            features_df=feat_df,
            ticker=ticker,
            model=trainer,
            config=cfg,
            initial_capital=capital,
            replay_start=replay_start,
        )

    from rich.table import Table as _Table
    tbl = _Table(title=f"Paper Trading Results — {ticker}", show_header=True,
                 header_style="bold cyan")
    tbl.add_column("Metric", style="bold", min_width=25)
    tbl.add_column("Value", justify="right", min_width=15)
    color = "green" if result.total_return_pct >= 0 else "red"
    tbl.add_row("Final NAV",     f"${result.final_nav:,.2f}")
    tbl.add_row("Total Return",  f"[{color}]{result.total_return_pct:+.2f}%[/]")
    tbl.add_row("Trades Entered", str(result.n_trades))
    tbl.add_row("Win Rate",       f"{result.win_rate:.1%}")
    tbl.add_row("Bars simulated", str(len(result.trades)))
    console.print(tbl)

    from paper_trading.simulator import LOG_DIR
    console.print(
        f"\n[green]Logs written to:[/] [dim]{LOG_DIR}/{ticker.lower()}_paper_trades.csv[/]"
    )
    console.print("[dim]Open the dashboard to see live results.[/]")
    console.print(
        "\n[bold yellow]SIMULATION ONLY[/] — No real orders. No real money. No broker connections."
    )


# ---------------------------------------------------------------------------
# overfit-check
# ---------------------------------------------------------------------------

@app.command(name="overfit-check")
def overfit_check(
    ticker:      str = typer.Option(..., "--ticker", "-t",
                                    help="Ticker used in the prior `alpha report` run"),
    n_tests:     int = typer.Option(50, "--n-tests",
                                    help="Total strategy configurations trialled "
                                         "(hyperparameter search + variants)"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Detect overfitting and multiple-testing bias from the latest validation report.

    Loads the most recent <ticker>_report_*.json written by 'alpha report' and
    runs the full overfitting detection pipeline: IS/OOS gap, Deflated Sharpe
    Ratio, Probability of Backtest Overfitting, feature importance stability,
    and multiple testing penalty.

    Run 'alpha report --ticker <TICKER>' first if no report exists yet.
    """
    import json
    from validation.overfitting_detector import detect_overfitting, print_report as print_overfit
    from utils.helpers import ensure_dir

    cfg     = _load_cfg(config_path)
    out_dir = Path(cfg.get("validation", {}).get("metrics_output_dir", "reports"))

    _banner("Overfitting Detection", cfg, ticker,
            f"Trials: [bold]{n_tests}[/]   Source: [dim]{out_dir}[/]")

    # Locate most recent report JSON for this ticker
    json_files = sorted(
        out_dir.glob(f"{ticker.lower()}_report_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not json_files:
        console.print(
            f"[yellow]No validation report found for[/] [cyan]{ticker}[/].\n"
            f"Generate one first: [bold]python main.py report --ticker {ticker}[/]"
        )
        raise typer.Exit(1)

    latest = json_files[-1]
    console.print(f"[dim]Loading: {latest.name}[/]")

    with latest.open(encoding="utf-8") as fh:
        rep = json.load(fh)

    strats    = rep.get("strategies", {})
    is_m      = strats.get("ml_theoretical",    strats.get("baseline_theoretical", {}))
    oos_m     = strats.get("ml_realistic",       strats.get("baseline_realistic",   {}))
    wf        = rep.get("walk_forward", [])
    feat_imp  = rep.get("feature_importance", {})
    n_features = max(len(feat_imp), 1)
    oos_bars   = rep.get("n_bars", 252) // max(len(wf) + 1, 2)

    with _make_spinner("") as prog:
        prog.add_task("Running overfitting analysis…", total=None)
        overfit_report = detect_overfitting(
            in_sample_metrics     = is_m,
            out_of_sample_metrics = oos_m,
            feature_count         = n_features,
            number_of_tests       = n_tests,
            fold_metrics          = wf,
            feature_importances   = [feat_imp] if feat_imp else None,
            oos_bars              = oos_bars,
        )

    print_overfit(overfit_report, ticker=ticker)

    # Exit with non-zero code if HIGH or SEVERE — useful in CI pipelines
    if overfit_report.severity in ("HIGH", "SEVERE"):
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# robustness-check
# ---------------------------------------------------------------------------

@app.command(name="robustness-check")
def robustness_check(
    ticker:      str  = typer.Option(..., "--ticker", "-t",
                                     help="Ticker symbol (must have a trained model + walk-forward data)"),
    require_all: bool = typer.Option(False, "--require-all",
                                     help="Require all 6 gates to pass (default: need ≥ 4/6)"),
    config_path: str  = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Run the 6-gate robustness check before deploying to paper trading.

    Gates: DSR, CPCV, Parameter Stability, Slippage Scaling, t-stat, MinTRL.
    Exits with code 1 if the strategy fails (can be used as a pre-deployment gate).
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from data.ingest import DataIngestion
    from features.engine import FeatureEngine
    from models.train import ModelTrainer
    from backtest.engine import run_backtest, CostModel
    from validation.robustness import RobustnessGates

    cfg   = _load_cfg(config_path)
    rg_cfg = dict(cfg.get("robustness_gates", {}))
    rg_cfg["require_all"] = require_all
    gates = RobustnessGates(**rg_cfg)

    console.print(f"\n[bold cyan]Running robustness gates for [green]{ticker}[/green]…[/bold cyan]")
    console.print("[dim]Step 1/3: Loading data and features[/dim]")

    try:
        data_cfg = cfg["data"]
        df = DataIngestion(config=cfg).get_data(ticker, as_of_date=data_cfg["end_date"])
        df = df[df.index >= pd.Timestamp(data_cfg["start_date"])]
        feat_df = FeatureEngine(config=cfg).build(df, ticker=ticker)
    except Exception as exc:
        console.print(f"[bold red]Data loading failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print("[dim]Step 2/3: Generating OOS signals and running backtest[/dim]")

    try:
        trainer = ModelTrainer(config=cfg)
        trainer.load(ticker)
        feat_cols = [c for c in trainer.feature_columns if c in feat_df.columns] \
                    if hasattr(trainer, "feature_columns") else \
                    [c for c in feat_df.columns if c not in ("label", "close", "open", "high", "low", "volume")]
        X = feat_df[feat_cols]
        probas = trainer.predict_proba(X)
        threshold = float(cfg.get("model", {}).get("confidence_threshold", 0.55))
        signals = pd.Series(
            np.where(probas > threshold, 1, np.where(probas < 1 - threshold, -1, 0)),
            index=X.index, dtype=int,
        )
        # Use last 30% of data as OOS
        split = int(len(feat_df) * 0.70)
        oos_close   = feat_df["close"].iloc[split:]
        oos_signals = signals.iloc[split:]
        bt = run_backtest(oos_signals, oos_close, costs=CostModel.for_stock(),
                          initial_capital=100_000.0, label="robustness_oos")
        oos_returns = bt.equity_curve.pct_change().dropna()
    except Exception as exc:
        console.print(f"[bold red]Model inference failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # Load fold Sharpes from most recent training metrics JSON
    fold_sharpes: list[float] = []
    is_sharpe: float = 0.0
    try:
        artifacts_dir = Path("models/artifacts")
        candidates = sorted(artifacts_dir.glob("training_metrics_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            tm = json.loads(candidates[0].read_text())
            folds = tm.get("fold_results", [])
            fold_sharpes = [f["ml_strategy_metrics"]["sharpe_like"]
                            for f in folds if "ml_strategy_metrics" in f]
            is_sharpes   = [f["model_metrics"].get("roc_auc", 0.5) * 2 - 1
                            for f in folds if "model_metrics" in f]
            is_sharpe    = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    except Exception:
        pass

    console.print("[dim]Step 3/3: Evaluating robustness gates[/dim]")
    bt_cfg  = cfg.get("backtest", {})
    report  = gates.evaluate(
        oos_returns=oos_returns,
        is_sharpe=is_sharpe,
        oos_sharpe_list=fold_sharpes,
        commission_pct=float(bt_cfg.get("commission_pct", 0.001)),
        slippage_pct=float(bt_cfg.get("slippage_pct", 0.0005)),
    )

    # Print results table
    from rich.table import Table
    table = Table(title=f"Robustness Gates — {ticker}", show_header=True,
                  header_style="bold cyan")
    table.add_column("Gate",   style="bold", width=22)
    table.add_column("Value",  justify="right", width=12)
    table.add_column("Pass?",  justify="center", width=8)
    gate_rows = [
        ("DSR",                f"{report.dsr:.4f}",               report.dsr_pass),
        ("CPCV Sharpe",        f"{report.cpcv_sharpe:.4f}",       report.cpcv_pass),
        ("Param Stability CV", f"{report.param_stability_cv:.4f}", report.param_stability_pass),
        ("Slippage 3×",        f"{report.slippage_sharpes.get(3, 0):.4f}", report.slippage_pass),
        ("t-stat",             f"{report.t_stat:.4f}",             report.t_stat_pass),
        ("MinTRL (years)",     f"{report.min_trl_years:.2f}",      report.min_trl_pass),
    ]
    for name, val, passed in gate_rows:
        icon = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, val, icon)
    console.print(table)

    verdict    = report.overall_pass
    n_gates    = len([report.dsr_pass, report.cpcv_pass, report.param_stability_pass,
                      report.slippage_pass, report.t_stat_pass, report.min_trl_pass,
                      getattr(report, "equity_r2_pass", False),
                      getattr(report, "positive_folds_pass", False)])
    passed_str = f"{report.n_passed}/{n_gates}"
    if verdict:
        console.print(f"\n[bold green]PASS[/bold green] — {passed_str} gates passed. Strategy is robust.\n")
    else:
        console.print(f"\n[bold red]FAIL[/bold red] — {passed_str} gates passed. Do not deploy.\n")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# fetch-universe  (download S&P 500 / NASDAQ 100 / ETF ticker lists)
# ---------------------------------------------------------------------------

@app.command(name="fetch-universe")
def fetch_universe_cmd(
    name: str = typer.Option(
        "sp500",
        "--name", "-n",
        help="Universe to fetch: sp500 | nasdaq100 | russell1000 | etfs | all",
    ),
) -> None:
    """
    Download a stock universe ticker list and save it to data/universes/.

    Examples:
        python main.py fetch-universe --name sp500
        python main.py fetch-universe --name nasdaq100
        python main.py fetch-universe --name all
    """
    from tools.fetch_universe import save_universe, _FETCHERS

    targets = list(_FETCHERS) if name == "all" else [name]
    for target in targets:
        try:
            path = save_universe(target, verbose=False)
            import pandas as pd
            df = pd.read_csv(path)
            console.print(f"[green]Saved[/] [cyan]{target}[/]: {len(df)} tickers -> {path}")
        except Exception as exc:
            console.print(f"[red]Failed {target}:[/] {exc}")


# ---------------------------------------------------------------------------
# precompute-features  (pre-fill feature cache for a universe using subprocesses)
# ---------------------------------------------------------------------------

@app.command(name="precompute-features")
def precompute_features_cmd(
    universe:    str  = typer.Option("sp500", "--universe", "-u"),
    workers:     int  = typer.Option(4, "--workers", "-w", help="Number of parallel subprocesses"),
    config_path: str  = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Pre-compute and cache features for all tickers in a universe using parallel subprocesses.

    Run this once before universe-trade to avoid slow first-run feature computation.
    Features are cached to disk — subsequent calls are instant.

    Examples:
        python main.py precompute-features --universe sp500 --workers 4
        python main.py precompute-features --universe sp500 --workers 8
    """
    import concurrent.futures, subprocess, sys
    from tools.fetch_universe import load_universe

    try:
        tickers = load_universe(universe)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    cfg = _load_cfg(config_path)
    raw_dir = Path(cfg.get("data", {}).get("cache_dir", "data/raw"))

    # Filter to tickers that have raw data downloaded
    available = [t for t in tickers if (raw_dir / f"{t.lower()}_daily.parquet").exists()]
    console.print(
        f"[cyan]Pre-computing features for {len(available)}/{len(tickers)} tickers[/] "
        f"({len(tickers) - len(available)} skip — no raw data)\n"
        f"[dim]Workers: {workers} | Config: {config_path}[/]"
    )

    cwd = str(Path(__file__).parent)

    def _compute_one(ticker: str) -> tuple[str, bool, str]:
        """Launch feature computation in a separate Python process (bypasses GIL)."""
        raw_path_fwd = str(raw_dir / f"{ticker.lower()}_daily.parquet").replace("\\", "/")
        script = (
            "import sys; sys.path.insert(0,'.');"
            "from features.feature_cache import get_or_compute_features;"
            "from pathlib import Path;"
            f"get_or_compute_features(ticker='{ticker}',"
            f"raw_path=Path('{raw_path_fwd}'),"
            "config=None)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
            cwd=cwd,
        )
        ok = result.returncode == 0
        err = result.stderr.strip()[-120:] if result.stderr else ""
        return ticker, ok, err

    ok_count = fail_count = 0
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as prog:
        task = prog.add_task(f"Computing features ({workers} workers)...", total=len(available))
        # ThreadPoolExecutor: each thread runs a subprocess → true CPU parallelism, no pickle issues
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_compute_one, t): t for t in available}
            for fut in concurrent.futures.as_completed(futures):
                ticker, ok, err = fut.result()
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    if err:
                        logger.debug("Feature compute failed for %s: %s", ticker, err)
                prog.advance(task)

    console.print(
        f"\n[green]Done:[/] {ok_count} computed, "
        f"[{'red' if fail_count else 'dim'}]{fail_count} failed[/], "
        f"{len(tickers) - len(available)} skipped (no data)"
    )


# ---------------------------------------------------------------------------
# train-universe  (train models for every ticker in a universe file)
# ---------------------------------------------------------------------------

@app.command(name="train-universe")
def train_universe_cmd(
    universe:    str  = typer.Option(
        "sp500",
        "--universe", "-u",
        help="Universe name (sp500/nasdaq100/etfs) or path to a CSV file with a 'ticker' column",
    ),
    start:       str  = typer.Option("2018-01-01", "--start"),
    end:         str  = typer.Option("2024-01-01", "--end"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest",
                                     help="Skip data download (use cached Parquet)"),
    resume:      bool = typer.Option(True,  "--resume/--no-resume",
                                     help="Skip tickers that already have a trained model"),
    max_tickers: int  = typer.Option(0, "--max",
                                     help="Train at most this many tickers (0 = all)"),
    fast:        bool = typer.Option(False, "--fast/--no-fast",
                                     help="Fast-train mode: fewer folds, smaller ensemble"),
    config_path: str  = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Train models for every ticker in a universe (e.g., all S&P 500 stocks).

    This is the large-scale version of train-all. Use --resume (default) to
    skip already-trained tickers and continue from where you left off.

    Examples:
        python main.py train-universe --universe sp500
        python main.py train-universe --universe nasdaq100 --max 50
        python main.py train-universe --universe data/universes/custom.csv --resume
    """
    import subprocess, time, re
    from tools.fetch_universe import load_universe

    cfg = _load_cfg(config_path)
    artifacts_dir = Path(cfg.get("model", {}).get("artifacts_dir", "models/artifacts"))

    try:
        ticker_list = load_universe(universe)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        console.print(f"Run: [bold]python main.py fetch-universe --name {universe}[/]")
        raise typer.Exit(1)

    if max_tickers > 0:
        ticker_list = ticker_list[:max_tickers]

    if resume:
        before = len(ticker_list)
        ticker_list = [
            t for t in ticker_list
            if not (artifacts_dir / f"{t.lower()}_model.joblib").exists()
        ]
        skipped = before - len(ticker_list)
        if skipped:
            console.print(f"[dim]Skipping {skipped} already-trained tickers (--resume)[/]")

    console.print(Panel(
        f"[bold]Universe Training[/]\n"
        f"Universe: [cyan]{universe}[/] | "
        f"Tickers to train: [green]{len(ticker_list)}[/]\n"
        f"Window: {start} to {end}",
        expand=False,
    ))

    if not ticker_list:
        console.print("[green]All tickers already trained.[/]")
        return

    results = []
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    ok_count = 0
    fail_count = 0

    for i, ticker in enumerate(ticker_list, 1):
        console.print(f"[dim][{i}/{len(ticker_list)}][/] [cyan]{ticker}[/]", end="  ")
        t0 = time.time()

        if not skip_ingest:
            r = subprocess.run(
                [sys.executable, "main.py", "ingest",
                 "--ticker", ticker, "--start", start, "--end", end],
                capture_output=True, env=env,
            )
            if r.returncode != 0:
                console.print("[red]INGEST FAIL[/]")
                results.append((ticker, "INGEST_FAILED", 0.0))
                fail_count += 1
                continue

        train_cmd = [sys.executable, "main.py", "train", "--ticker", ticker]
        if fast:
            train_cmd.append("--fast")
        r = subprocess.run(train_cmd, capture_output=True, env=env)
        elapsed = time.time() - t0

        if r.returncode != 0:
            err = (r.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
            reason = err[-1][:60] if err else "unknown"
            console.print(f"[red]FAIL[/] ({reason})")
            results.append((ticker, "TRAIN_FAILED", elapsed))
            fail_count += 1
            continue

        out = (r.stdout or b"").decode("utf-8", errors="replace")
        cv_m = re.search(r"CV_SHARPE=([\d.]+)", out)
        if not cv_m:
            cv_m = re.search(r"CV Sharpe[:\s]+([\d.]+)", out)
        cv = float(cv_m.group(1)) if cv_m else 0.0
        console.print(f"[green]OK[/] CV={cv:.2f}  ({elapsed:.0f}s)")
        results.append((ticker, f"OK  CV={cv:.3f}", elapsed))
        ok_count += 1

    total_time = sum(e for _, _, e in results)
    console.print(
        f"\n[bold]Done.[/] {ok_count} trained | {fail_count} failed | "
        f"Total time: {total_time/60:.1f}min\n"
        f"Next: [bold]python main.py universe-trade --universe {universe}[/]"
    )


# ---------------------------------------------------------------------------
# universe-trade  (ranking-based whole-market portfolio simulation)
# ---------------------------------------------------------------------------

@app.command(name="universe-trade")
def universe_trade_cmd(
    universe:     str   = typer.Option(
        "sp500",
        "--universe", "-u",
        help="Universe name or path to CSV with 'ticker' column",
    ),
    top_n:        int   = typer.Option(10, "--top-n",
                                        help="Number of stocks to hold at once"),
    replay_start: str   = typer.Option("2020-01-01", "--start"),
    replay_end:   str   = typer.Option("2023-12-31", "--end"),
    capital:      float = typer.Option(100_000.0, "--capital"),
    confidence_weighted: bool = typer.Option(
        True, "--confidence-weighted/--equal-weight",
        help="Weight positions by model confidence instead of equal weight",
    ),
    min_confidence: float = typer.Option(
        0.52, "--min-confidence",
        help="Min confidence score threshold; calibrated scores cluster at 0.5469 so 0.52 admits all liquid stocks",
    ),
    rebalance_threshold: float = typer.Option(
        0.05, "--rebalance-threshold",
        help="Only rebalance if position deviates >N fraction from target (0.05 = 5%%)",
    ),
    bear_min_confidence: float = typer.Option(
        0.52, "--bear-min-confidence",
        help="Min confidence in bear regime (same as min_confidence to leave bear guard to trailing stop)",
    ),
    trailing_stop_pct: float = typer.Option(
        0.08, "--trailing-stop",
        help="Trailing stop distance as fraction of trail high (0.08 = 8%%)",
    ),
    min_holding_bars: int = typer.Option(
        3, "--min-holding-bars",
        help="Minimum bars to hold before a ranking exit (reduces churn)",
    ),
    use_momentum_filter: bool = typer.Option(
        False, "--momentum-filter/--no-momentum-filter",
        help="Only buy stocks where SMA50 > SMA200 (uptrend filter); off by default",
    ),
    market_breadth_threshold: float = typer.Option(
        0.0, "--breadth-threshold",
        help="Halve positions when fewer than this fraction of stocks are in SMA uptrend "
             "(0 = disabled, recommended to start with 0.25 for all_us_stocks)",
    ),
    vol_adjusted_ranking: bool = typer.Option(
        False, "--vol-adj-rank/--no-vol-adj-rank",
        help="Rank by confidence/vol_21d (risk-adjusted); off by default",
    ),
    use_raw_scores: bool = typer.Option(
        False, "--raw-scores/--calibrated-scores",
        help="Use raw ensemble scores for threshold filtering (calibrated=default since raw scores "
             "are biased toward low-vol defensive stocks for cross-sectional ranking)",
    ),
    ranking_factor: str = typer.Option(
        "liquidity", "--ranking-factor",
        help="Stock selection criterion: 'model' (confidence score), 'momentum' (12-1 month self-relative rank), "
             "'cross_momentum' (true cross-sectional 12-1 month rank — use for ETF rotation), "
             "'liquidity' (largest-cap proxy via dollar volume), 'composite' (model+momentum blend)",
    ),
    min_dollar_volume: float = typer.Option(
        500_000_000.0, "--min-dollar-volume",
        help="Minimum daily dollar volume (close*volume) in dollars to filter investable stocks "
             "(500M default selects mega-cap stocks; 0 = disabled)",
    ),
    spy_timing_method: str = typer.Option(
        "sma50_200", "--spy-timing",
        help="SPY market timing signal: 'sma50_200' (Golden/Death Cross, smoothest), "
             "'price_200' (price vs SMA200, faster exit), 'price_50' (price vs SMA50, fastest), "
             "'none' (disabled — use for ETF rotation where some ETFs hedge bear markets)",
    ),
    stop_loss_pct: float = typer.Option(
        0.0, "--stop-loss",
        help="Hard per-position stop loss as fraction of entry price (0.10 = 10%%). "
             "0 = disabled. Fires before trailing stop, bypasses min_holding_bars.",
    ),
    stop_loss_cooldown: int = typer.Option(
        10, "--stop-loss-cooldown",
        help="Bars to exclude a stopped-out stock from re-entry (default 10 = 2 weeks).",
    ),
    config_path:  str   = typer.Option("config.yaml", "--config"),
) -> None:
    """
    Run a ranking-based universe portfolio simulation.

    Each day, scores all trained tickers, picks the top-N by predicted
    confidence, and holds them in confidence-weighted positions.

    Prerequisites: run train-universe first.

    Examples:
        python main.py universe-trade --universe sp500
        python main.py universe-trade --universe etfs --top-n 5 --start 2020-01-01
        python main.py universe-trade --universe nasdaq100 --top-n 15 --min-confidence 0.65
        python main.py universe-trade --universe all_us_stocks --top-n 20 --no-momentum-filter
    """
    from tools.fetch_universe import load_universe
    from paper_trading.universe_portfolio import UniversePortfolio

    cfg = _load_cfg(config_path)

    try:
        ticker_list = load_universe(universe)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    port = UniversePortfolio(
        universe_tickers=ticker_list,
        top_n=top_n,
        capital=capital,
        config=cfg,
        confidence_weighted=confidence_weighted,
        min_confidence=min_confidence,
        rebalance_threshold=rebalance_threshold,
        bear_min_confidence=bear_min_confidence,
        trailing_stop_pct=trailing_stop_pct,
        min_holding_bars=min_holding_bars,
        use_momentum_filter=use_momentum_filter,
        market_breadth_threshold=market_breadth_threshold,
        vol_adjusted_ranking=vol_adjusted_ranking,
        use_raw_scores=use_raw_scores,
        ranking_factor=ranking_factor,
        min_dollar_volume=min_dollar_volume,
        spy_timing_method=spy_timing_method,
        stop_loss_pct=stop_loss_pct,
        stop_loss_cooldown=stop_loss_cooldown,
    )

    if len(port.models) < top_n:
        console.print(
            f"[yellow]Only {len(port.models)} models available.[/] "
            f"Train more with: [bold]python main.py train-universe --universe {universe}[/]"
        )

    port.run(replay_start, replay_end)


# ---------------------------------------------------------------------------
# train-universal  (cross-sectional model for all-universe scoring)
# ---------------------------------------------------------------------------

@app.command(name="train-universal")
def train_universal_cmd(
    config_path: str = typer.Option("config.yaml", "--config"),
    max_tickers: int = typer.Option(
        0, "--max-tickers",
        help="Limit to N tickers (0 = all cached tickers)",
    ),
    max_rows_per_ticker: int = typer.Option(
        500, "--max-rows-per-ticker",
        help="Sample at most N bars per ticker (keeps training set manageable for large universes).",
    ),
    train_end: str = typer.Option(
        "", "--train-end",
        help="Exclude data after this date (YYYY-MM-DD) so the simulation window is out-of-sample. "
             "E.g. --train-end 2021-12-31 then simulate from 2022-01-01 onward.",
    ),
) -> None:
    """
    Train a universal cross-sectional model on all cached tickers.

    Pools feature data from every locally cached stock and trains one model
    using the same walk-forward purged CV as per-ticker training.  The result
    is saved as 'universal_model.joblib' and is automatically picked up by
    'universe-trade' as a fallback for tickers without a dedicated model.

    Run this once after ingesting data for a large universe so that every
    ticker with raw price data can be scored without a per-ticker model.

    Examples:
        python main.py train-universal
        python main.py train-universal --max-tickers 50
        python main.py train-universal --max-rows-per-ticker 300
        python main.py train-universal --train-end 2021-12-31
    """
    from pathlib import Path as _Path

    import pandas as pd

    from features.feature_cache import get_or_compute_features
    from models.train import train_model

    cfg = _load_cfg(config_path)
    raw_dir = _Path(cfg.get("data", {}).get("cache_dir", "data/raw"))
    artifacts_dir = _Path(cfg.get("model", {}).get("artifacts_dir", "models/artifacts"))

    parquets = sorted(raw_dir.glob("*_daily.parquet"))
    if max_tickers > 0:
        parquets = parquets[:max_tickers]

    cutoff_ts = pd.Timestamp(train_end) if train_end.strip() else None
    cutoff_note = f" | train cutoff: [yellow]{train_end}[/]" if cutoff_ts else ""
    console.print(Panel(
        f"[bold]Training Universal Cross-Sectional Model[/]\n"
        f"Pooling [cyan]{len(parquets)}[/] cached tickers -> one global model{cutoff_note}",
        expand=False,
    ))

    from datetime import timedelta as _td
    import concurrent.futures as _cf

    def _load_one(args: tuple) -> tuple[pd.DataFrame | None, str, str]:
        idx_i, parquet_path = args
        ticker = parquet_path.stem.replace("_daily", "").upper()
        try:
            feats = get_or_compute_features(
                ticker=ticker, raw_path=parquet_path, config=cfg
            )
            if feats is None or len(feats) < 100:
                return None, ticker, "too few bars"
            if cutoff_ts is not None:
                feats = feats[feats.index <= cutoff_ts]
                if len(feats) < 100:
                    return None, ticker, "too few bars before cutoff"
            if max_rows_per_ticker > 0 and len(feats) > max_rows_per_ticker:
                step = len(feats) // max_rows_per_ticker
                feats = feats.iloc[::step].head(max_rows_per_ticker)
            feats = feats.copy()
            feats.index = feats.index + _td(seconds=idx_i)
            return feats, ticker, ""
        except Exception as exc:
            return None, ticker, str(exc)

    console.print(f"[dim]Loading features (16 parallel threads)...[/]")
    all_frames: list[pd.DataFrame] = []
    with _cf.ThreadPoolExecutor(max_workers=16) as _pool:
        _futs = {_pool.submit(_load_one, (i, p)): (i, p) for i, p in enumerate(parquets)}
        for _fut in _cf.as_completed(_futs):
            feats, ticker, err = _fut.result()
            if feats is not None:
                all_frames.append(feats)
                console.print(f"  [green]{ticker}[/]  {len(feats):,} bars")
            elif err:
                console.print(f"  [yellow]{ticker}[/] skipped ({err})")

    if not all_frames:
        console.print("[red]No data available. Run data ingestion first.[/]")
        raise typer.Exit(1)

    # Sort by date so walk-forward CV folds are temporally ordered
    pooled = pd.concat(all_frames, axis=0).sort_index()
    console.print(
        f"\nPooled: [bold]{len(pooled):,}[/] bars from [bold]{len(all_frames)}[/] tickers"
    )

    console.print("[cyan]Training universal model (this may take a few minutes)...[/]")
    result = train_model(
        features_df=pooled,
        target_horizon=int(cfg.get("model", {}).get("target_horizon", 5)),
        config=cfg,
        confidence_threshold=float(cfg.get("model", {}).get("confidence_threshold", 0.65)),
        ticker="universal",
    )

    model_path = artifacts_dir / "universal_model.joblib"
    scaler_path = artifacts_dir / "universal_scaler.joblib"
    # train_model already saves {ticker}_model.joblib → universal_model.joblib
    # and {ticker}_scaler.joblib → universal_scaler.joblib
    console.print()
    if model_path.exists():
        console.print(f"[green]Universal model ->[/] {model_path}")
    if scaler_path.exists():
        console.print(f"[green]Universal scaler ->[/] {scaler_path}")

    report   = result.get("report", {})
    ml_m     = report.get("comparison", {}).get("ml", {})
    base_m   = report.get("comparison", {}).get("baseline", {})
    console.print(
        f"\nCV Sharpe   ML: [bold]{ml_m.get('sharpe', 0):.3f}[/]  "
        f"Base: {base_m.get('sharpe', 0):.3f}\n"
        f"CV Hit Rate ML: {ml_m.get('hit_rate', 0):.1%}\n"
        f"\nReady: [bold]python main.py universe-trade --universe all_us_stocks[/]"
    )


# ---------------------------------------------------------------------------
# config-show
# ---------------------------------------------------------------------------

@app.command(name="config-show")
def config_show(
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Display the active configuration settings."""
    cfg = _load_cfg(config_path)

    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        table = Table(title=f"[bold cyan]{section}[/]", show_header=False, box=None)
        table.add_column(style="dim", min_width=30)
        table.add_column()
        for k, v in values.items():
            table.add_row(k, str(v))
        console.print(table)
        console.print()


# ---------------------------------------------------------------------------
# Analysis commands — expose tools/ scripts as proper CLI commands
# ---------------------------------------------------------------------------

@app.command(name="compare-metrics")
def compare_metrics(
    n: int = typer.Option(6, "--n", help="Number of most recent training runs to compare"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Compare CV Sharpe across the most recent N training runs."""
    import glob, json as _json
    cfg = _load_cfg(config_path)
    artifacts_dir = cfg.get("model", {}).get("artifacts_dir", "models/artifacts")
    files = sorted(glob.glob(f"{artifacts_dir}/training_metrics_*.json"))
    if not files:
        console.print("[red]No training metrics found. Run 'train' first.[/]")
        raise typer.Exit(1)

    table = Table(title="Training Metrics Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Timestamp",    min_width=15)
    table.add_column("CV Sharpe",    justify="right", min_width=10)
    table.add_column("Features",     justify="right", min_width=10)
    table.add_column("Folds",        justify="right", min_width=6)
    table.add_column("Best OOS",     justify="right", min_width=10)

    for f in files[-n:]:
        try:
            tm = _json.loads(Path(f).read_text())
            folds = tm.get("fold_results", [])
            sharpes = [x.get("ml_strategy_metrics", {}).get("sharpe_like", 0) for x in folds]
            avg_sharpe = sum(sharpes) / max(len(sharpes), 1)
            best_oos = max(sharpes) if sharpes else 0.0
            ts = Path(f).stem.replace("training_metrics_", "")
            feat_count = tm.get("feature_count", "?")
            color = "green" if avg_sharpe > 1.0 else "yellow" if avg_sharpe > 0.5 else "red"
            table.add_row(
                ts, f"[{color}]{avg_sharpe:.3f}[/]",
                str(feat_count), str(len(folds)), f"{best_oos:.3f}",
            )
        except Exception as exc:
            table.add_row(Path(f).stem, f"[red]ERR: {exc}[/]", "?", "?", "?")
    console.print(table)


@app.command(name="top-features")
def top_features(
    n: int = typer.Option(20, "--n", help="Number of top features to display"),
    config_path: str = typer.Option("config.yaml", "--config"),
) -> None:
    """Display the top N most important features from the latest training run (SHAP or XGB gain)."""
    import glob, json as _json
    cfg = _load_cfg(config_path)
    artifacts_dir = cfg.get("model", {}).get("artifacts_dir", "models/artifacts")
    files = sorted(glob.glob(f"{artifacts_dir}/training_metrics_*.json"))
    if not files:
        console.print("[red]No training metrics found. Run 'train' first.[/]")
        raise typer.Exit(1)

    tm = _json.loads(Path(files[-1]).read_text())
    ts = Path(files[-1]).stem.replace("training_metrics_", "")

    shap_imp = tm.get("feature_importance_shap", {})
    gain_imp = tm.get("feature_importance_gain", {})
    imp = shap_imp or gain_imp
    imp_type = "SHAP" if shap_imp else "XGB gain"

    if not imp:
        console.print("[yellow]No feature importances in metrics file.[/]")
        raise typer.Exit(0)

    top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:n]
    table = Table(
        title=f"Top {n} Features by {imp_type} — {ts}",
        show_header=True, header_style="bold cyan",
    )
    table.add_column("#",        justify="right", min_width=3)
    table.add_column("Feature",  min_width=35)
    table.add_column("Score",    justify="right", min_width=12)

    for i, (feat, val) in enumerate(top, 1):
        color = "green" if i <= 5 else "yellow" if i <= 10 else "white"
        table.add_row(str(i), f"[{color}]{feat}[/]", f"{val:.6f}")

    console.print(table)
    selected = tm.get("selected_features", [])
    if selected:
        console.print(f"\n[dim]Total selected features in model: {len(selected)}[/]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
