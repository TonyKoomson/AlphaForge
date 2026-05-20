"""
AlphaForge AI Harness — Orchestrator

The central coordinator of the multi-agent research loop.

Loop (per iteration):
  1. Analyst  (Grok)   — market context for the target asset
  2. Strategist (Claude) — proposes an experiment (uses KB + analyst context)
  3. Coder   (Claude)  — generates any new alpha factors if needed
  4. [System] — runs training + backtesting
  5. Reviewer (Claude)  — evaluates results, produces verdict
  6. Strategist (Claude) — synthesises: PROMOTE / ITERATE / REJECT
  7. [System] — saves to knowledge base, updates session log
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich import box as rich_box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from harness.config import (
    RESULTS_DIR, PROMOTE_SHARPE_THRESHOLD, PROMOTE_DD_LIMIT,
    DEFAULT_ITERATIONS, DEFAULT_TICKER,
)
from harness.memory.knowledge_base import KnowledgeBase
from harness.tools.executor import ToolExecutor
from harness.agents.strategist import StrategistAgent
from harness.agents.analyst    import AnalystAgent
from harness.agents.coder      import CoderAgent
from harness.agents.reviewer   import ReviewerAgent
from harness.rl_bandit import ExperimentBandit, make_bandit

_utf8_out = (
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
console = Console(file=_utf8_out, highlight=False)


class AlphaHarness:
    """
    Multi-agent AI harness for autonomous trading strategy research.

    Usage
    -----
    harness = AlphaHarness()
    harness.discover(ticker="SPY", iterations=5)
    harness.research_universe(universe="sp500", top_n=10, iterations=3)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.kb         = KnowledgeBase(session_id=self.session_id)
        self.executor   = ToolExecutor(knowledge_base=self.kb)

        # Agents (share the same executor so tools update the same KB)
        self.strategist = StrategistAgent(executor=self.executor)
        self.analyst    = AnalystAgent(executor=self.executor)
        self.coder      = CoderAgent(executor=self.executor)
        self.reviewer   = ReviewerAgent(executor=self.executor)

        # RL bandit: learns which experiment archetypes yield highest OOS Sharpe
        self.bandit = make_bandit(exploration_c=1.0)
        n_loaded = self.bandit.bootstrap_from_kb(self.kb)
        if n_loaded:
            console.print(f"[dim]Bandit: bootstrapped {n_loaded} prior experiments from KB[/]")

        # Attach console logger to executor for rich display
        self.executor._log_fn = self._log_tool

        self._session_log: list[dict] = []
        self._promotions:  list[dict] = []
        self._goal: str = ""

    # ── Main entry points ─────────────────────────────────────────────────────

    def discover(
        self,
        ticker: str = DEFAULT_TICKER,
        iterations: int = DEFAULT_ITERATIONS,
        goal: str = "Find a strategy with OOS Sharpe > 0.8 and max drawdown < 20%",
    ) -> list[dict]:
        """
        Single-ticker strategy discovery loop.

        Returns the list of promoted strategies found.
        Early stopping: halts after 3 consecutive promoted strategies
        (the goal has been achieved with high confidence).
        """
        self._goal = goal
        self._print_banner(f"Strategy Discovery: {ticker}", iterations)
        kb_summary = self.kb.context_summary(bandit=self.bandit)

        # Step 0: Research plan from Strategist
        self._section("Drafting research plan", "Strategist")
        plan = self.strategist.generate_research_plan(ticker, goal, kb_summary)
        console.print(Panel(plan[:1500], title="[bold cyan]🗺  Research Plan[/]", expand=False, border_style="cyan"))

        consecutive_promoted = 0
        for i in range(1, iterations + 1):
            self._section(f"Iteration {i}/{iterations}", "Orchestrator")
            result = self._run_iteration(ticker, i)
            self._session_log.append(result)

            if result.get("promoted"):
                self._promotions.append(result)
                consecutive_promoted += 1
                console.print()
                console.print(Panel(
                    Align.center(
                        f"[bold bright_green]  STRATEGY APPROVED!  [/]\n\n"
                        f"[white]Score {result['sharpe']:+.3f}  ·  {self._stars(result['sharpe'])}[/]\n"
                        f"[dim]{result['config'].get('hypothesis','')[:60]}[/]"
                    ),
                    border_style="bright_green",
                    padding=(1, 4),
                ))
                if consecutive_promoted >= 3:
                    console.print(Panel(
                        Align.center(
                            "[bold bright_green]  Research goal achieved!  [/]\n"
                            "[dim]3 winning strategies found — stopping early.[/]"
                        ),
                        border_style="bright_green",
                    ))
                    break
            else:
                consecutive_promoted = 0
                sharpe = result.get("sharpe", 0) or 0
                if sharpe > 0:
                    console.print(f"\n  [yellow]Promising but not quite there yet. Score: {sharpe:+.3f}  Refining…[/]")
                else:
                    console.print(f"\n  [dim]Score {sharpe:+.3f} — this approach won't work. Trying something different.[/]")

        self._print_summary()
        self._save_session_log()
        return self._promotions

    def research_universe(
        self,
        universe: str = "sp500",
        top_n: int = 10,
        start: str = "2020-01-01",
        end: str   = "2023-12-31",
        iterations: int = 3,
    ) -> dict:
        """
        Universe-level portfolio strategy discovery.
        Uses the Analyst for context, Strategist for ranking config, then runs universe-trade.
        """
        self._print_banner(f"Universe Research: {universe} top-{top_n}", iterations)

        # Get market context
        self._section("Market Analysis", "Analyst")
        period = f"{start} to {end}"
        context = self.analyst.analyze_market(ticker=f"{universe} universe", period=period)
        console.print(Panel(context[:1200], title="[bold blue]Market Context[/]", expand=False))

        best_result: dict = {}
        best_sharpe  = -999.0

        for i in range(1, iterations + 1):
            self._section(f"Universe Iteration {i}/{iterations}", "Orchestrator")

            # Strategist proposes config
            kb_sum = self.kb.context_summary()
            proposal_text = self.strategist.propose_experiment(context, kb_sum)
            console.print(Panel(proposal_text[:1000], title="[cyan]Strategist Proposal[/]", expand=False))

            # Extract ranking factor from proposal (default to cross_momentum)
            ranking_factor = "cross_momentum"
            spy_timing     = "sma50_200"
            stop_loss      = 0.12 if "bear" in context.lower() or "2022" in period else 0.0

            for line in proposal_text.lower().split("\n"):
                if "ranking" in line and "momentum" in line:
                    ranking_factor = "cross_momentum"
                elif "ranking" in line and "model" in line:
                    ranking_factor = "model"
                elif "ranking" in line and "liquidity" in line:
                    ranking_factor = "liquidity"

            # Run universe simulation
            self._log_tool("run_universe_trade", {
                "universe": universe, "top_n": top_n,
                "ranking_factor": ranking_factor, "stop_loss_pct": stop_loss,
            })
            result_str = self.executor.execute("run_universe_trade", {
                "universe": universe, "top_n": top_n,
                "start": start, "end": end,
                "ranking_factor": ranking_factor,
                "spy_timing": spy_timing,
                "stop_loss_pct": stop_loss,
            })
            result = json.loads(result_str)
            self._print_results_table(result)

            # Reviewer evaluates
            self._section(f"Review — Iteration {i}", "Reviewer")
            review = self.reviewer.analyze(
                hypothesis=f"Universe {universe}, top-{top_n}, ranking={ranking_factor}",
                config={"universe": universe, "top_n": top_n, "ranking_factor": ranking_factor},
                backtest_results=result,
            )
            console.print(Panel(review[:1200], title="[bold yellow]Review[/]", expand=False))

            sharpe = result.get("sharpe", 0.0) or 0.0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_result = result

            self.kb.save_experiment(
                hypothesis=f"Universe {universe} top-{top_n} {ranking_factor}",
                config={"universe": universe, "top_n": top_n, "ranking_factor": ranking_factor,
                        "spy_timing": spy_timing, "stop_loss": stop_loss},
                results=result,
                verdict="MODERATE" if sharpe > 0.5 else "WEAK",
            )

        self._print_summary()
        return best_result

    def add_factor_and_test(
        self,
        factor_name: str,
        hypothesis: str,
        ticker: str = DEFAULT_TICKER,
    ) -> dict:
        """
        One-shot: Coder generates a new factor, then backtest immediately.
        """
        self._print_banner(f"New Factor: {factor_name}", 1)

        # Analyst context
        context = self.analyst.assess_factor(factor_name, hypothesis, ticker)

        # Coder generates factor
        self._section("Factor Generation", "Coder")
        code_result = self.coder.generate_factor(factor_name, hypothesis, context)
        console.print(Panel(code_result[:1000], title="[cyan]Factor Code[/]", expand=False))

        # Train and backtest
        self._section("Training & Backtest", "Orchestrator")
        train_result = json.loads(self.executor.execute("train_model", {"ticker": ticker}))
        bt_result    = json.loads(self.executor.execute("run_backtest", {"ticker": ticker}))
        self._print_results_table(bt_result)

        # Review
        review = self.reviewer.analyze(
            hypothesis=hypothesis, config={"factor": factor_name, "ticker": ticker},
            backtest_results=bt_result,
        )
        console.print(Panel(review[:1000], title="[bold yellow]Review[/]", expand=False))

        return {"factor": factor_name, "backtest": bt_result, "review": review}

    # ── Internal iteration ────────────────────────────────────────────────────

    def _run_iteration(self, ticker: str, iteration: int) -> dict:
        """Execute one full research iteration and return a result dict."""
        t_start = time.time()

        # 0. Bandit: select experiment archetype for this iteration
        bandit_guidance = self.bandit.get_guidance(iteration)
        arm_name = bandit_guidance["arm_name"]
        console.print(
            f"  [dim]AI chose approach:[/] [magenta]{arm_name}[/] "
            f"[dim]— {bandit_guidance['description'][:60]}[/]"
        )
        if bandit_guidance.get("top_archetypes"):
            top = bandit_guidance["top_archetypes"]
            top_str = ", ".join(
                f"{t['arm']}([green]{t['avg_sharpe']:+.2f}[/])" for t in top
            )
            console.print(f"  [dim]Best so far: {top_str}[/]")

        # 1. Market context (Analyst / Grok)
        self._section(f"Step 1 of 6  ·  Reading the market — Experiment {iteration}", "Analyst")
        context = self.analyst.analyze_market(ticker)
        console.print(Panel(context[:800], title="[bold blue]📊  Market Analysis[/]", expand=False, border_style="bright_blue"))

        # 2. Propose experiment (Strategist / Claude) — bandit guidance injected
        self._section(f"Step 2 of 6  ·  Designing the strategy — Experiment {iteration}", "Strategist")
        kb_summary = self.kb.context_summary(bandit=self.bandit)
        # Deduplication: warn Strategist if suggested features are similar to prior experiments
        suggested_feats = bandit_guidance.get("features", [])
        similar = self.kb.find_similar(
            suggested_feats,
            top_n=2,
            hypothesis=bandit_guidance.get("hypothesis_hint", ""),
        )
        dedup_hint = ""
        if similar:
            dedup_lines = []
            for s in similar:
                m = s.get("metrics", {})
                dedup_lines.append(
                    f"  - {s.get('title','?')[:60]} "
                    f"(Sharpe={m.get('oos_sharpe', m.get('sharpe', 0)):.2f}, "
                    f"tags={s.get('tags',[])})"
                )
            dedup_hint = (
                "\n\n## ⚠ Similar Prior Experiments (avoid redundancy)\n"
                + "\n".join(dedup_lines)
                + "\nConsider meaningfully differentiating your proposal (different features, threshold, or hypothesis)."
            )

        bandit_hint = (
            f"\n\n## RL Bandit Guidance (Thompson Sampling)\n"
            f"{bandit_guidance['rationale']}\n"
            f"Suggested archetype: **{arm_name}** — {bandit_guidance['description']}\n"
            f"Hypothesis hint: {bandit_guidance['hypothesis_hint']}\n"
            f"Suggested features: {bandit_guidance['features']}\n"
            f"Suggested model params: {bandit_guidance['model_params']}\n"
            f"Suggested signal threshold: {bandit_guidance['signal_threshold']}\n\n"
            "You may adopt, adapt, or override this suggestion based on market context and KB findings."
            + dedup_hint
        )
        proposal_text = self.strategist.propose_experiment(context + bandit_hint, kb_summary)
        console.print(Panel(proposal_text[:1000], title="[bold cyan]🎯  Strategy Proposal[/]", expand=False, border_style="cyan"))

        # Extract experiment config from proposal text
        config = self._extract_json(proposal_text) or {
            "ticker": ticker,
            "hypothesis": bandit_guidance["hypothesis_hint"],
            "features": bandit_guidance["features"],
            "model_params": bandit_guidance["model_params"],
            "signal_threshold": bandit_guidance["signal_threshold"],
            "start": "2018-01-01", "end": "2023-12-31",
        }
        config["_bandit_arm"] = arm_name
        config.setdefault("ticker", ticker)
        config.setdefault("start",  "2018-01-01")
        config.setdefault("end",    "2023-12-31")

        # 3. Training
        self._section(f"Step 3 of 6  ·  Training the AI model — Experiment {iteration}", "Orchestrator")
        self._log_tool("train_model", {"ticker": config["ticker"]})
        train_str    = self.executor.execute("train_model", {
            "ticker": config["ticker"],
            "start":  config.get("start", "2018-01-01"),
            "end":    config.get("end",   "2023-12-31"),
        })
        train_result = json.loads(train_str)
        oos_sh = train_result.get("oos_sharpe", "?")
        oos_label = f"{oos_sh:+.3f}" if isinstance(oos_sh, (int, float)) else str(oos_sh)
        console.print(f"  [dim]Model trained  ·  Quality score: [white]{oos_label}[/][/]")

        # 4. Backtest
        self._section(f"Step 4 of 6  ·  Historical simulation — Experiment {iteration}", "Orchestrator")
        self._log_tool("run_backtest", {"ticker": config["ticker"]})
        bt_str    = self.executor.execute("run_backtest", {
            "ticker":           config["ticker"],
            "start":            config.get("start", "2018-01-01"),
            "end":              config.get("end",   "2023-12-31"),
            "features":         config.get("features"),
            "signal_threshold": config.get("signal_threshold", 0.55),
        })
        bt_result = json.loads(bt_str)
        self._print_results_table(bt_result)

        # 5. Review (Reviewer / Claude)
        self._section(f"Step 5 of 6  ·  Quality review — Experiment {iteration}", "Reviewer")
        review = self.reviewer.analyze(
            hypothesis=config.get("hypothesis", ""),
            config=config,
            backtest_results=bt_result,
            validation_results=train_result,
            iteration=iteration,
        )
        console.print(Panel(review[:1200], title="[bold yellow]🔍  Quality Review[/]", expand=False, border_style="yellow"))

        # 6. Synthesise (Strategist / Claude)
        self._section(f"Step 6 of 6  ·  Decision — Experiment {iteration}", "Strategist")
        decision = self.strategist.synthesise(review, iteration)
        console.print(Panel(decision[:800], title="[bold magenta]📋  Decision[/]", expand=False, border_style="magenta"))

        # 7. Determine if promoted
        sharpe = bt_result.get("sharpe", 0.0) or 0.0
        max_dd = abs(bt_result.get("max_dd", 100.0) or 100.0)
        promoted = (sharpe >= PROMOTE_SHARPE_THRESHOLD and max_dd <= PROMOTE_DD_LIMIT * 100)

        # Save to KB
        verdict = "STRONG" if promoted else ("MODERATE" if sharpe > 0.4 else "WEAK")
        entry_id = self.kb.save_experiment(
            hypothesis=config.get("hypothesis", ""),
            config=config,
            results=bt_result,
            verdict=verdict,
        )
        if promoted:
            self.kb.save_promotion(
                strategy_name=config.get("hypothesis", f"iter_{iteration}"),
                config=config,
                metrics={"sharpe": sharpe, "max_dd": max_dd},
            )

        # Bandit: update arm reward with this iteration's OOS Sharpe
        self.bandit.update(arm_name, sharpe)
        console.print(
            f"  [dim]Bandit updated:[/] arm=[magenta]{arm_name}[/] sharpe=[cyan]{sharpe:+.3f}[/]"
        )

        elapsed = time.time() - t_start
        return {
            "iteration":  iteration,
            "config":     config,
            "backtest":   bt_result,
            "training":   train_result,
            "review":     review,
            "decision":   decision,
            "promoted":   promoted,
            "sharpe":     sharpe,
            "elapsed_s":  round(elapsed, 1),
            "kb_id":      entry_id,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict | None:
        """Extract the first JSON block from agent text output."""
        import re
        pattern = r"```json\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        # Try bare JSON object
        idx = text.find("{")
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass
        return None

    def _log_tool(self, name: str, args: dict) -> None:
        _TOOL_LABELS = {
            "train_model":        "  Training AI model on historical data…",
            "run_backtest":       "  Running historical simulation…",
            "run_walk_forward":   "  Running walk-forward quality check…",
            "kb_search":          "  Searching strategy library…",
            "kb_save":            "  Saving strategy to library…",
            "run_universe_trade": "  Simulating portfolio across universe…",
        }
        label = _TOOL_LABELS.get(name, f"  Running: {name}")
        console.print(f"[dim]{label}[/dim]")

    def _section(self, title: str, agent: str) -> None:
        _AGENT_DISPLAY = {
            "Strategist":   ("🎯", "Strategy Designer",  "cyan"),
            "Analyst":      ("📊", "Market Analyst",     "bright_blue"),
            "Coder":        ("⚙ ", "AI Engineer",        "green"),
            "Reviewer":     ("🔍", "Quality Reviewer",   "yellow"),
            "Orchestrator": ("🤖", "System",             "bright_magenta"),
        }
        emoji, name, color = _AGENT_DISPLAY.get(agent, ("●", agent, "white"))
        console.print()
        console.rule(
            f"[bold {color}]{emoji}  {name}[/]  [dim]{title}[/dim]",
            style=f"dim {color}",
        )

    def _print_banner(self, title: str, iterations: int) -> None:
        n_str  = f"{iterations} experiment{'s' if iterations != 1 else ''}"
        lines  = (
            f"[bold white]  AlphaForge — AI Strategy Research  [/bold white]\n\n"
            f"[cyan]  {title}[/cyan]\n\n"
            f"[dim]  {n_str}  ·  Session {self.session_id[-8:]}[/dim]\n"
            f"[dim]  Simulation only — no real money involved[/dim]"
        )
        console.print()
        console.print(Panel(
            Align.center(lines),
            border_style="bright_cyan",
            padding=(1, 6),
        ))

    def _score_bar(self, value: float, width: int = 10) -> str:
        """Return a colored ASCII progress bar for a 0-1 score."""
        clamped = max(0.0, min(1.0, value))
        filled  = int(round(clamped * width))
        bar     = "█" * filled + "░" * (width - filled)
        if clamped >= 0.7:
            color = "bright_green"
        elif clamped >= 0.4:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]{bar}[/]"

    def _stars(self, sharpe: float) -> str:
        if sharpe >= 0.8:  return "[bright_green]★★★★★[/]"
        if sharpe >= 0.6:  return "[green]★★★★[dim]★[/][/]"
        if sharpe >= 0.4:  return "[yellow]★★★[dim]★★[/][/]"
        if sharpe >= 0.2:  return "[yellow]★★[dim]★★★[/][/]"
        return "[red]★[dim]★★★★[/][/]"

    def _print_results_table(self, result: dict) -> None:
        sharpe   = float(result.get("sharpe",     0) or 0)
        ret      = float(result.get("ann_return",  0) or 0)
        max_dd   = abs(float(result.get("max_dd",  0) or 0))
        win_rate = float(result.get("win_rate",    0) or 0)
        trades   = result.get("n_trades", "—")
        cost_pct = float(result.get("cost_drag",   0) or 0)

        sc      = "bright_green" if sharpe >= 0.6 else "green" if sharpe >= 0.4 else "yellow" if sharpe >= 0.2 else "red"
        ret_c   = "green" if ret > 0 else "red"
        dd_c    = "green" if max_dd < 20 else "yellow" if max_dd < 35 else "red"
        win_c   = "green" if win_rate > 0.52 else "yellow" if win_rate >= 0.48 else "red"

        t = Table(
            title="[bold]Results[/]",
            show_header=True,
            header_style="bold dim",
            box=rich_box.ROUNDED,
            padding=(0, 2),
            border_style="dim",
            width=62,
        )
        t.add_column("Metric",           style="dim", width=20)
        t.add_column("Value",            justify="right", width=12)
        t.add_column("Rating",           justify="left",  width=14)
        t.add_column("",                 justify="center", width=4)

        ok, warn, fail = "[bright_green]✓[/]", "[yellow]~[/]", "[red]✗[/]"

        t.add_row(
            "Performance Score",
            f"[{sc}]{sharpe:+.3f}[/]",
            self._stars(sharpe),
            ok if sharpe >= 0.5 else warn if sharpe > 0 else fail,
        )
        t.add_row(
            "Yearly Return",
            f"[{ret_c}]{ret * 100:+.1f}%[/]",
            self._score_bar(max(0, min(ret * 2.5, 1))),
            ok if ret > 0.05 else warn if ret > 0 else fail,
        )
        t.add_row(
            "Worst Drawdown",
            f"[{dd_c}]-{max_dd:.1f}%[/]",
            self._score_bar(max(0, 1 - max_dd / 50)),
            ok if max_dd < 20 else warn if max_dd < 35 else fail,
        )
        t.add_row(
            "Win Rate",
            f"[{win_c}]{win_rate * 100:.1f}%[/]",
            self._score_bar(win_rate),
            ok if win_rate > 0.52 else warn if win_rate >= 0.45 else fail,
        )
        t.add_row(
            "Number of Trades",
            str(trades),
            "",
            "",
        )
        if cost_pct:
            t.add_row(
                "Trading Costs",
                f"[dim]{cost_pct * 100:.2f}%[/]",
                "",
                "",
            )
        console.print(t)

    def _print_summary(self) -> None:
        total    = len(self._session_log)
        promoted = len(self._promotions)
        kb_stats = self.kb.stats()
        sharpes  = [r.get("sharpe", 0) or 0 for r in self._session_log]
        best_sh  = max(sharpes, default=0)
        avg_sh   = sum(sharpes) / len(sharpes) if sharpes else 0

        console.print()
        console.rule("[bold bright_cyan]  Session Complete  [/]", style="bright_cyan")
        console.print()

        # Main summary panel
        verdict_line: str
        if promoted:
            verdict_line = (
                f"[bold bright_green]  {promoted} strategy{'ies' if promoted > 1 else 'y'} "
                f"approved for deployment!  [/]"
            )
            border = "bright_green"
        else:
            verdict_line = "[bold yellow]  No strategies met the approval threshold this session.[/]"
            border = "yellow"

        summary_text = (
            f"{verdict_line}\n\n"
            f"  [dim]Experiments run:[/dim]    [white]{total}[/]\n"
            f"  [dim]Best score:[/dim]         [white]{self._stars(best_sh)}  {best_sh:+.3f}[/]\n"
            f"  [dim]Avg score:[/dim]          {self._stars(avg_sh)}  [dim]{avg_sh:+.3f}[/]\n"
            f"  [dim]Library entries:[/dim]    [white]{kb_stats.get('total', 0)}[/]"
        )
        console.print(Panel(summary_text, border_style=border, padding=(1, 2)))

        # Promoted strategies
        if self._promotions:
            console.print()
            console.print("[bold bright_green]  Approved Strategies[/]")
            for i, p in enumerate(self._promotions, 1):
                hyp = p["config"].get("hypothesis", "?")[:65]
                sh  = p["sharpe"]
                dd  = abs(p["backtest"].get("max_dd", 0) or 0)
                console.print(
                    f"  [bright_green]{i}.[/]  {hyp}\n"
                    f"     [dim]Score {sh:+.3f}  ·  Max loss -{dd:.1f}%[/]"
                )

        # What the AI learned
        arms = getattr(self.bandit, "_state", {}).get("arms", {})
        tried = {k: v for k, v in arms.items() if v.get("n", 0) > 0}
        if tried:
            console.print()
            console.print("[bold dim]  What the AI learned this session[/]")
            best_arm = max(tried, key=lambda k: tried[k]["sum_reward"] / max(tried[k]["n"], 1))
            ba_avg   = tried[best_arm]["sum_reward"] / tried[best_arm]["n"]
            console.print(
                f"  Best approach: [cyan]{best_arm}[/]  "
                f"(avg score [green]{ba_avg:+.3f}[/])\n"
                f"  [dim]{len(tried)} strategy type{'s' if len(tried)>1 else ''} explored[/]"
            )

    def _save_session_log(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = RESULTS_DIR / f"session_{self.session_id}.json"
        log_path.write_text(json.dumps(self._session_log, indent=2, default=str))
        console.print(f"\n[dim]Session log saved to: {log_path}[/]")

        # Generate markdown report
        try:
            from harness import session_report
            report_path = session_report.generate(
                session_log=self._session_log,
                promotions=self._promotions,
                kb_stats=self.kb.stats(),
                bandit_summary=self.bandit.stats_summary(),
                ticker=self._session_log[0]["config"].get("ticker", "SPY") if self._session_log else "SPY",
                session_id=self.session_id,
                goal=self._goal,
            )
            console.print(f"[dim]Session report saved to: {report_path}[/]")
        except Exception as exc:
            console.print(f"[dim yellow]Session report skipped: {exc}[/]")
