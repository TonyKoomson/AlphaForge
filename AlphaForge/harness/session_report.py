"""
AlphaForge AI Harness — Session Report Generator

Produces a human-readable Markdown report after each research session.
The report captures:
  - Session metadata (ID, ticker, iterations, timestamp)
  - Per-iteration results table (Sharpe, max DD, verdict, bandit arm)
  - Promoted strategies list
  - RL bandit learning curve (arm reward history)
  - Knowledge base state (counts by type, top experiments)
  - Recommendations for next session

Reports are saved to:  logs/harness/reports/session_{session_id}.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def generate(
    session_log: list[dict],
    promotions: list[dict],
    kb_stats: dict,
    bandit_summary: str,
    ticker: str,
    session_id: str,
    goal: str = "",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Write a Markdown session report and return the file path.

    Parameters
    ----------
    session_log   : list of per-iteration result dicts from AlphaHarness
    promotions    : list of promoted strategy dicts
    kb_stats      : dict from KnowledgeBase.stats()
    bandit_summary: str from ExperimentBandit.stats_summary()
    ticker        : research target asset
    session_id    : harness session identifier
    goal          : research goal string
    output_dir    : where to write the report (default: logs/harness/reports/)
    """
    if output_dir is None:
        from harness.config import RESULTS_DIR
        output_dir = RESULTS_DIR / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# AlphaForge AI Harness — Session Report",
        "",
        f"**Session ID:** `{session_id}`  ",
        f"**Ticker:** {ticker}  ",
        f"**Goal:** {goal or 'OOS Sharpe > 0.8, max DD < 25%'}  ",
        f"**Generated:** {now}  ",
        f"**Iterations:** {len(session_log)}  ",
        f"**Promoted:** {len(promotions)}  ",
        "",
        "---",
        "",
    ]

    # ── Executive Summary ─────────────────────────────────────────────────────
    best_sharpe = max((r.get("sharpe", 0) for r in session_log), default=0.0)
    lines += [
        "## Executive Summary",
        "",
        f"The harness completed **{len(session_log)} research iterations** on **{ticker}**. "
        f"The best out-of-sample Sharpe achieved was **{best_sharpe:.3f}**. "
        + (f"**{len(promotions)} strategy(ies) met the promotion threshold** "
           "(OOS Sharpe ≥ 0.8 AND max drawdown ≤ 25%)."
           if promotions else
           "**No strategy met the promotion threshold** in this session — "
           "findings saved to the knowledge base for subsequent sessions."),
        "",
    ]

    # ── Iteration Results Table ───────────────────────────────────────────────
    lines += [
        "## Iteration Results",
        "",
        "| Iter | Hypothesis (truncated) | Sharpe | Max DD | Promoted | Bandit Arm | Time (s) |",
        "|------|------------------------|--------|--------|----------|------------|----------|",
    ]
    for r in session_log:
        hyp      = r.get("config", {}).get("hypothesis", "—")[:40].replace("|", "\\|")
        sharpe   = r.get("sharpe", 0.0)
        max_dd   = abs(r.get("backtest", {}).get("max_dd", 0.0))
        promoted = "✅ YES" if r.get("promoted") else "—"
        arm      = r.get("config", {}).get("_bandit_arm", "—")
        elapsed  = r.get("elapsed_s", 0)
        sr_color = "**" if sharpe >= 0.8 else ""
        lines.append(
            f"| {r.get('iteration','?'):>4} "
            f"| {hyp:<40} "
            f"| {sr_color}{sharpe:+.3f}{sr_color} "
            f"| {max_dd:.1f}% "
            f"| {promoted} "
            f"| {arm} "
            f"| {elapsed:.0f}s |"
        )
    lines.append("")

    # ── Promoted Strategies ───────────────────────────────────────────────────
    if promotions:
        lines += ["## Promoted Strategies", ""]
        for i, p in enumerate(promotions, 1):
            cfg = p.get("config", {})
            bt  = p.get("backtest", {})
            lines += [
                f"### {i}. {cfg.get('hypothesis', 'Strategy')[:70]}",
                "",
                f"- **OOS Sharpe:** {p.get('sharpe', 0):.3f}",
                f"- **Max Drawdown:** {abs(bt.get('max_dd', 0)):.1f}%",
                f"- **Features:** {cfg.get('features', [])}",
                f"- **Model params:** {json.dumps(cfg.get('model_params', {}), indent=None)}",
                f"- **Signal threshold:** {cfg.get('signal_threshold', 0.55)}",
                f"- **Backtest period:** {cfg.get('start','?')} → {cfg.get('end','?')}",
                "",
            ]
    else:
        lines += [
            "## Promoted Strategies",
            "",
            "_No strategies met the promotion threshold (OOS Sharpe ≥ 0.8 AND max DD ≤ 25%) "
            "in this session. The knowledge base has been updated with findings for the next run._",
            "",
        ]

    # ── Statistical Significance Summary ────────────────────────────────────
    sig_rows = []
    for r in session_log:
        bt = r.get("backtest", {}) or {}
        psr_v = bt.get("psr")
        dsr_v = bt.get("dsr")
        if psr_v is not None and dsr_v is not None:
            sig_rows.append({
                "iter":     r.get("iteration", "?"),
                "sharpe":   r.get("sharpe", 0.0),
                "p_value":  bt.get("sharpe_p_value", "—"),
                "psr":      psr_v,
                "dsr":      dsr_v,
                "verdict":  bt.get("dsr_verdict", "—"),
                "n_trials": bt.get("dsr_n_trials", "—"),
            })
    if sig_rows:
        lines += [
            "## Statistical Significance (PSR / DSR)",
            "",
            "Bailey & Lopez de Prado (2012) metrics for each backtest result:",
            "",
            "| Iter | Sharpe | p-value | PSR | DSR | n_trials | Verdict |",
            "|------|--------|---------|-----|-----|----------|---------|",
        ]
        for row in sig_rows:
            p_str = row['p_value'] if isinstance(row['p_value'], str) else f"{row['p_value']:.3f}"
            lines.append(
                f"| {row['iter']:>4} "
                f"| {row['sharpe']:+.3f} "
                f"| {p_str} "
                f"| {row['psr']:.3f} "
                f"| {row['dsr']:.3f} "
                f"| {row['n_trials']} "
                f"| {row['verdict']} |"
            )
        lines += ["", "_DSR ≥ 0.95 required for PROMOTE. DSR accounts for multiple-testing inflation._", ""]

    # ── RL Bandit Learning ────────────────────────────────────────────────────
    lines += [
        "## RL Bandit Learning",
        "",
        "The Thompson Sampling bandit learns which experiment archetypes yield higher OOS Sharpe.",
        "Arm selection samples from posterior distributions — exploration is implicit via posterior uncertainty.",
        "",
        "```",
        bandit_summary,
        "```",
        "",
    ]

    # ── Knowledge Base State ──────────────────────────────────────────────────
    by_type = kb_stats.get("by_type", {})
    lines += [
        "## Knowledge Base State",
        "",
        f"- **Total entries:** {kb_stats.get('total', 0)}",
        f"- **Experiments:** {by_type.get('experiment', 0)}",
        f"- **Promotions:** {by_type.get('promotion', 0)}",
        f"- **Failures:** {by_type.get('failure', 0)}",
        f"- **Findings:** {by_type.get('finding', 0)}",
        f"- **Heuristics:** {by_type.get('heuristic', 0)}",
        "",
    ]

    # ── Sharpe Convergence ────────────────────────────────────────────────────
    if len(session_log) > 1:
        sharpes = [r.get("sharpe", 0.0) for r in session_log]
        running_best = []
        best = -999.0
        for s in sharpes:
            best = max(best, s)
            running_best.append(best)
        lines += [
            "## Sharpe Convergence",
            "",
            "Best-so-far OOS Sharpe per iteration:",
            "",
            "```",
        ]
        for i, (s, rb) in enumerate(zip(sharpes, running_best), 1):
            bar = "█" * max(0, int(max(s, 0) * 20))
            lines.append(f"  Iter {i:2d}: {s:+.3f}  (best={rb:.3f})  {bar}")
        lines += ["```", ""]

    # ── Recommendations ───────────────────────────────────────────────────────
    lines += ["## Recommendations for Next Session", ""]
    if promotions:
        lines += [
            f"- ✅ {len(promotions)} strategy(ies) promoted — run `py harness_main.py promote-list` to review",
            "- Consider testing promoted strategy on different tickers (QQQ, AAPL) to check generalisability",
            "- Run paper-trading simulation: `py main.py paper-trade --ticker {ticker}`",
        ]
    else:
        # Find best Sharpe iteration for recommendation
        best_iter = max(session_log, key=lambda r: r.get("sharpe", 0), default=None)
        if best_iter and best_iter.get("sharpe", 0) >= 0.3:
            arm = best_iter.get("config", {}).get("_bandit_arm", "momentum_medium")
            lines += [
                f"- Best iteration achieved Sharpe {best_iter['sharpe']:.3f} with archetype `{arm}`",
                f"- Continue with `py harness_main.py discover --ticker {ticker} --iter 5` — bandit will exploit this archetype",
                "- Consider extending backtest window to 2015-2023 for more regime diversity",
            ]
        else:
            lines += [
                "- Results below threshold — bandit will explore different archetypes next session",
                "- Try running with `--iter 8` for more exploration coverage",
                f"- Check `py harness_main.py bandit-stats` to see which archetypes need exploration",
            ]
    lines += [
        f"- View this report: `logs/harness/reports/session_{session_id}.md`",
        f"- View session JSON: `logs/harness/session_{session_id}.json`",
        "",
        "---",
        "",
        "_AlphaForge AI Harness — Simulation Only, No Real Money_",
        "",
    ]

    # ── Write to disk ─────────────────────────────────────────────────────────
    report_path = output_dir / f"session_{session_id}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
