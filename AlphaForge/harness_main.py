"""
AlphaForge AI Harness — CLI Entry Point

Usage
-----
  py harness_main.py discover                         # discover strategies for SPY
  py harness_main.py discover --ticker AAPL --iter 5
  py harness_main.py universe --universe sp500 --top-n 10
  py harness_main.py factor --name reversal_5d        # generate + test a new alpha factor
  py harness_main.py status                           # show knowledge base stats
  py harness_main.py promote-list                     # list all promoted strategies
  py harness_main.py replay --session <session-id>    # review a past session log
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_utf8_out = (
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
console = Console(file=_utf8_out, highlight=False)

app = typer.Typer(
    name="alphaforge-harness",
    help="AlphaForge AI Harness — multi-agent strategy research (simulation only, no real money)",
    add_completion=False,
)


def _check_keys() -> bool:
    """Warn if API keys are missing and return True if we can proceed."""
    from harness.config import validate_keys
    missing = validate_keys()
    if missing:
        console.print(
            Panel(
                f"[yellow]Missing environment variables:[/] {', '.join(missing)}\n\n"
                "Set them before running:\n"
                f"  [dim]$env:{missing[0]} = 'your-key-here'[/]\n\n"
                "Get keys at:\n"
                "  Anthropic: https://console.anthropic.com/\n"
                "  xAI (Grok): https://console.x.ai/",
                title="[bold red]API Keys Required[/]",
                border_style="red",
                expand=False,
            )
        )
        return False
    return True


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command(name="discover")
def discover_cmd(
    ticker: str = typer.Option("SPY", "--ticker", "-t",
                               help="Asset to research (e.g. SPY, AAPL, QQQ)"),
    iterations: int = typer.Option(5, "--iter", "-n",
                                   help="Number of research iterations"),
    goal: str = typer.Option(
        "Find a strategy with OOS Sharpe > 0.8 and max drawdown < 20%",
        "--goal", help="Research objective statement",
    ),
) -> None:
    """
    Run the AI research loop to discover trading strategies for a single ticker.

    The Strategist (Claude) proposes experiments, the Analyst (Grok) provides
    market context, the Coder (Claude) can generate new factors, and the
    Reviewer (Claude) evaluates each result.
    """
    if not _check_keys():
        raise typer.Exit(1)

    from harness.orchestrator import AlphaHarness
    harness = AlphaHarness()
    harness.discover(ticker=ticker.upper(), iterations=iterations, goal=goal)


@app.command(name="universe")
def universe_cmd(
    universe: str = typer.Option("sp500", "--universe", "-u",
                                 help="Universe: sp500, nasdaq100, etfs, all_us_stocks"),
    top_n: int    = typer.Option(10, "--top-n", help="Number of stocks to hold"),
    start: str    = typer.Option("2020-01-01", "--start"),
    end: str      = typer.Option("2023-12-31", "--end"),
    iterations: int = typer.Option(3, "--iter", "-n",
                                   help="Number of ranking configurations to test"),
) -> None:
    """
    AI-driven universe portfolio strategy research.

    Tests multiple ranking factors (momentum, model, liquidity) for a universe
    of stocks, guided by the Analyst and Strategist agents.
    """
    if not _check_keys():
        raise typer.Exit(1)

    from harness.orchestrator import AlphaHarness
    harness = AlphaHarness()
    harness.research_universe(
        universe=universe, top_n=top_n,
        start=start, end=end, iterations=iterations,
    )


@app.command(name="factor")
def factor_cmd(
    name: str       = typer.Option(..., "--name", "-n", help="Factor name (snake_case)"),
    hypothesis: str = typer.Option(..., "--hypothesis", "-h",
                                   help="Economic rationale for the factor"),
    ticker: str     = typer.Option("SPY", "--ticker", "-t",
                                   help="Asset to test the factor on"),
) -> None:
    """
    Generate a new alpha factor via Claude and immediately backtest it.

    The Coder (Claude) writes look-ahead-free Python code, validates it,
    integrates it into the feature pipeline, and runs a backtest.
    """
    if not _check_keys():
        raise typer.Exit(1)

    from harness.orchestrator import AlphaHarness
    harness = AlphaHarness()
    harness.add_factor_and_test(
        factor_name=name, hypothesis=hypothesis, ticker=ticker.upper()
    )


@app.command(name="status")
def status_cmd() -> None:
    """Show knowledge base statistics and recent experiment summary."""
    from harness.memory.knowledge_base import KnowledgeBase
    from harness.config import MEMORY_DIR, RESULTS_DIR

    kb    = KnowledgeBase()
    stats = kb.stats()

    console.print()
    console.print(Panel(
        f"[bold]Knowledge Base[/]  [dim]{MEMORY_DIR}[/]",
        title="AlphaForge AI Harness — Status",
        expand=False,
    ))

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(justify="right")
    t.add_row("Total entries:", f"[bold]{stats.get('total', 0)}[/]")
    for etype, count in stats.get("by_type", {}).items():
        color = {"promotion": "green", "failure": "red", "experiment": "cyan"}.get(etype, "white")
        t.add_row(f"  {etype}:", f"[{color}]{count}[/]")
    console.print(t)

    best = kb.get_best_experiments(3)
    if best:
        console.print("\n[bold]Top Experiments (by OOS Sharpe)[/]")
        for exp in best:
            b = exp.get("body", {})
            m = exp.get("metrics", {})
            console.print(
                f"  [green]{m.get('oos_sharpe', 0):.2f}[/] Sharpe | "
                f"{b.get('hypothesis','?')[:60]}"
            )

    promotions = kb.get_promotions()
    if promotions:
        console.print(f"\n[bold green]{len(promotions)} promoted strategies[/]")
        for p in promotions[-5:]:
            b = p.get("body", {})
            m = p.get("metrics", {})
            console.print(f"  [green]>[/] {b.get('strategy_name','?')}: Sharpe={m.get('sharpe','?')}")

    # Session logs
    logs = sorted(RESULTS_DIR.glob("session_*.json"), reverse=True)
    if logs:
        console.print(f"\n[dim]{len(logs)} session logs in {RESULTS_DIR}[/]")
        console.print(f"[dim]Latest: {logs[0].name}[/]")


@app.command(name="promote-list")
def promote_list_cmd() -> None:
    """List all promoted strategies from the knowledge base."""
    from harness.memory.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    promotions = kb.get_promotions()

    if not promotions:
        console.print("[yellow]No promoted strategies yet. Run 'discover' to start.[/]")
        return

    t = Table(title=f"{len(promotions)} Promoted Strategies",
              show_header=True, header_style="bold green")
    t.add_column("Strategy", width=40)
    t.add_column("Sharpe",   justify="right")
    t.add_column("Max DD",   justify="right")
    t.add_column("Session",  style="dim")

    for p in promotions:
        b = p.get("body", {})
        m = p.get("metrics", {})
        t.add_row(
            b.get("strategy_name", "?")[:38],
            f"{m.get('sharpe', 0):.2f}",
            f"{m.get('max_dd', 0):.1f}%",
            p.get("session", "?"),
        )
    console.print(t)


@app.command(name="replay")
def replay_cmd(
    session: str = typer.Argument(..., help="Session ID to replay (from logs/harness/)"),
) -> None:
    """Print a summary of a past research session."""
    from harness.config import RESULTS_DIR

    log_path = RESULTS_DIR / f"session_{session}.json"
    if not log_path.exists():
        console.print(f"[red]Session log not found: {log_path}[/]")
        raise typer.Exit(1)

    log = json.loads(log_path.read_text())
    console.print(Panel(f"Session: {session}  |  Iterations: {len(log)}",
                        title="[bold]Session Replay[/]", expand=False))

    for entry in log:
        i = entry.get("iteration", "?")
        sharpe = entry.get("sharpe", 0)
        promoted = entry.get("promoted", False)
        hyp = entry.get("config", {}).get("hypothesis", "?")[:55]
        flag = "[bold green]PROMOTED[/]" if promoted else ""
        console.print(f"  [{i:02d}] Sharpe={sharpe:+.3f}  {flag}  {hyp}")


@app.command(name="bandit-stats")
def bandit_stats_cmd() -> None:
    """Show RL bandit arm statistics — which experiment archetypes are winning."""
    from harness.rl_bandit import make_bandit, ARMS
    from harness.memory.knowledge_base import KnowledgeBase

    kb     = KnowledgeBase()
    bandit = make_bandit()
    n_boot = bandit.bootstrap_from_kb(kb)

    console.print()
    console.print(Panel(
        f"[bold]UCB1 Experiment Bandit[/]\n"
        f"[dim]Bootstrapped {n_boot} experiments from KB[/]",
        title="RL Bandit Stats",
        expand=False,
    ))

    t = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    t.add_column("Archetype",   width=26)
    t.add_column("Trials",      justify="right", width=7)
    t.add_column("Avg Sharpe",  justify="right", width=10)
    t.add_column("Best",        justify="right", width=8)
    t.add_column("Description", style="dim")

    state = bandit._state["arms"]
    for arm_name in sorted(state, key=lambda a: state[a]["sum_reward"] / max(state[a]["n"], 1), reverse=True):
        s   = state[arm_name]
        n   = s["n"]
        avg = s["sum_reward"] / n if n > 0 else 0.0
        best = s.get("best", float("nan"))
        color = "green" if avg > 0.6 else "yellow" if avg > 0.3 else "dim"
        t.add_row(
            arm_name,
            str(n),
            f"[{color}]{avg:+.3f}[/]" if n > 0 else "[dim]—[/]",
            f"{best:+.3f}" if n > 0 else "—",
            ARMS[arm_name]["description"][:45],
        )

    console.print(t)
    console.print(f"\n[dim]Total trials: {bandit._state['total_trials']} | "
                  f"State: {bandit.state_path}[/]")


@app.command(name="dashboard")
def dashboard_cmd(
    port: int = typer.Option(8502, "--port", "-p", help="Streamlit port"),
) -> None:
    """
    Launch the AlphaForge Harness dashboard (Streamlit).

    Displays session results, knowledge base state, RL bandit learning curves,
    promoted strategies, and session reports. Auto-refreshes every 30 seconds.
    """
    import subprocess
    dashboard_path = Path(__file__).parent / "harness" / "harness_dashboard.py"
    console.print(Panel(
        f"[bold]Launching Harness Dashboard[/] at [cyan]http://localhost:{port}[/]\n"
        "[dim]Press Ctrl+C to stop.[/]",
        title="AlphaForge Harness Dashboard",
        border_style="cyan",
        expand=False,
    ))
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


@app.command(name="demo")
def demo_cmd(
    ticker: str     = typer.Option("SPY", "--ticker", "-t", help="Asset to research"),
    iterations: int = typer.Option(3, "--iter", "-n", help="Number of research iterations"),
    goal: str       = typer.Option(
        "Find a strategy with OOS Sharpe > 0.8 and max drawdown < 20%",
        "--goal", help="Research objective",
    ),
) -> None:
    """
    Run the AI harness in demo mode — no API keys required.

    Real backtests and training run via the actual AlphaForge infrastructure.
    Only the LLM calls (Strategist, Analyst, Coder, Reviewer) are replaced with
    pre-written, realistic domain-appropriate responses.

    Useful for testing the harness without API credentials.
    """
    console.print(Panel(
        "[bold yellow]DEMO MODE[/] — LLM responses are pre-written stubs.\n"
        "Real backtests and training run via the AlphaForge engine.\n"
        "[dim]Set ANTHROPIC_API_KEY + XAI_API_KEY to run with real AI agents.[/]",
        title="[bold]Demo Mode[/]",
        border_style="yellow",
        expand=False,
    ))
    from harness.demo_mode import build_demo_harness
    harness = build_demo_harness()
    harness.discover(ticker=ticker.upper(), iterations=iterations, goal=goal)


@app.command(name="kb-search")
def kb_search_cmd(
    query: str      = typer.Argument("", help="Search query (empty = list recent)"),
    entry_type: str = typer.Option("", "--type", "-t",
                                   help="Filter by type: experiment, promotion, failure, heuristic, finding"),
    min_sharpe: float = typer.Option(0.0, "--min-sharpe", help="Minimum OOS Sharpe filter"),
    limit: int      = typer.Option(10, "--limit", "-n", help="Max results"),
) -> None:
    """Search the knowledge base for experiments, promotions, and findings."""
    from harness.memory.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    results = kb.search(
        query=query,
        entry_type=entry_type or None,
        min_sharpe=min_sharpe if min_sharpe > 0 else None,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No matching entries found.[/]")
        return

    t = Table(
        title=f"{len(results)} KB Result(s) for '{query or 'all'}'",
        show_header=True, header_style="bold cyan",
    )
    t.add_column("ID",    width=9, style="dim")
    t.add_column("Type",  width=12)
    t.add_column("Title", width=50)
    t.add_column("Sharpe", justify="right", width=8)
    t.add_column("Session", width=16, style="dim")

    type_colors = {
        "promotion": "green", "failure": "red",
        "experiment": "cyan", "finding": "yellow", "heuristic": "magenta",
    }
    for entry in results:
        etype  = entry.get("type", "?")
        color  = type_colors.get(etype, "white")
        sharpe = entry.get("metrics", {}).get("oos_sharpe") or entry.get("metrics", {}).get("sharpe")
        sharpe_str = f"{sharpe:+.3f}" if sharpe is not None else "—"
        t.add_row(
            entry.get("id", "?"),
            f"[{color}]{etype}[/]",
            entry.get("title", "?")[:48],
            sharpe_str,
            entry.get("session", "?")[:16],
        )
    console.print(t)


@app.command(name="setup")
def setup_cmd() -> None:
    """Check environment, install dependencies, and validate API keys."""
    console.print(Panel(
        "Checking AlphaForge AI Harness setup...",
        title="[bold]Setup Check[/]",
        expand=False,
    ))

    # Check Python packages
    packages = {"anthropic": "Claude API", "openai": "Grok (xAI) API"}
    for pkg, label in packages.items():
        try:
            __import__(pkg)
            console.print(f"  [green]OK[/]  {pkg} ({label})")
        except ImportError:
            console.print(f"  [red]MISSING[/]  {pkg} — run: pip install {pkg}")

    # Check API keys
    from harness.config import validate_keys, CLAUDE_MODEL, GROK_MODEL
    missing = validate_keys()
    if not missing:
        console.print(f"  [green]OK[/]  ANTHROPIC_API_KEY set (model: {CLAUDE_MODEL})")
        console.print(f"  [green]OK[/]  XAI_API_KEY set (model: {GROK_MODEL})")
    else:
        for k in missing:
            console.print(f"  [red]MISSING[/]  {k}")

    # Check data
    from harness.config import ROOT
    raw_dir = ROOT / "data" / "raw"
    n_files = len(list(raw_dir.glob("*.parquet"))) if raw_dir.exists() else 0
    console.print(f"  [{'green' if n_files > 0 else 'yellow'}]{'OK' if n_files > 0 else 'WARN'}[/]"
                  f"  data/raw: {n_files} cached ticker files")

    # Check models
    arts_dir = ROOT / "models" / "artifacts"
    n_models = len(list(arts_dir.glob("*.joblib"))) if arts_dir.exists() else 0
    console.print(f"  [{'green' if n_models > 0 else 'yellow'}]{'OK' if n_models > 0 else 'WARN'}[/]"
                  f"  models/artifacts: {n_models} trained models")

    console.print()
    if not missing and n_files > 0:
        console.print("[green]Ready![/] Run: [bold]py harness_main.py discover[/]")
    elif not missing:
        console.print(
            "[yellow]No data cached.[/] Run first:\n"
            "  [bold]py main.py ingest --ticker SPY[/]\n"
            "  [bold]py main.py train --ticker SPY[/]\n"
            "Then: [bold]py harness_main.py discover[/]"
        )


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_global_banner() -> None:
    from harness.config import CLAUDE_MODEL, GROK_MODEL
    console.print(Panel(
        f"[bold white]AlphaForge AI Harness[/]\n"
        f"[dim]Strategist + Coder + Reviewer:[/] [cyan]{CLAUDE_MODEL}[/]\n"
        f"[dim]Analyst:[/] [blue]{GROK_MODEL}[/] [dim](xAI)[/]\n"
        f"[red bold]Simulation only — no real broker, no real money[/]",
        border_style="cyan",
        expand=False,
    ))


if __name__ == "__main__":
    _print_global_banner()
    app()
