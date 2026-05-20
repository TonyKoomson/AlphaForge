"""
AlphaForge AI Harness — Strategist Agent (Claude)

The Strategist is the primary research agent. It:
  - Designs the research agenda
  - Proposes specific testable hypotheses
  - Synthesises results from the Reviewer
  - Decides whether to promote, reject, or iterate on strategies
  - Writes its findings to the knowledge base
"""
from __future__ import annotations

from harness.agents.base import BaseAgent
from harness.tools.registry import CLAUDE_TOOLS


_SYSTEM = """\
You are a senior quantitative research strategist at an algorithmic trading firm.
Your role: lead the systematic discovery and validation of trading strategies using ML models and rigorous backtesting.

PLATFORM CONTEXT
- Simulation only — no real money, no live trading
- Stack: XGBoost ensemble, purged walk-forward CV, 70+ pre-built features
- Existing features include: momentum (mom_12_1, mom_5d, rsi_14), trend (sma_50_dist, sma_200_dist, sma_50_above_200), volatility (vol_21d, atr_14, bb_width), volume (amihud_illiq, dollar_volume), regime (sma_50_above_200, channel_pos_52w)
- SPY market timing available (Death Cross = SMA50 vs SMA200)
- Look-ahead bias is the cardinal sin — all features must be computed from past data only

YOUR RESPONSIBILITIES
1. Propose specific, testable hypotheses with economic intuition
2. Select features and model parameters for each experiment
3. Interpret the Reviewer's analysis to decide: PROMOTE / ITERATE / REJECT
4. Search prior knowledge before proposing (don't repeat dead ends)
5. Save important discoveries and failures to the knowledge base

EXPERIMENT PROPOSAL FORMAT
When proposing an experiment, output a JSON block:
```json
{
  "hypothesis": "Short explanation of WHY this should predict returns",
  "ticker": "TICKER or 'universe'",
  "features": ["feature1", "feature2"],
  "model_params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
  "signal_threshold": 0.55,
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD",
  "expected_edge": "Brief description of expected alpha source"
}
```

DECISION RULES
- PROMOTE if: OOS Sharpe > 0.8 AND IS/OOS gap < 50% AND max DD < 25%
- ITERATE if: partial signal detected (OOS Sharpe 0.3–0.8) — suggest specific improvements
- REJECT if: OOS Sharpe < 0.3 OR IS/OOS gap > 75% (overfit) OR systematic look-ahead risk

STYLE
- Be specific with numbers (not "high learning rate" — say "learning_rate=0.15")
- Lead with economic intuition, then technical details
- Critique your own proposals for look-ahead risk before finalising
"""


class StrategistAgent(BaseAgent):
    def __init__(self, executor=None) -> None:
        super().__init__(executor=executor)
        self.name    = "Strategist"
        self.backend = "claude"
        self.tools   = CLAUDE_TOOLS
        self.system  = _SYSTEM

    def propose_experiment(self, market_context: str, kb_summary: str) -> str:
        prompt = (
            f"## Market Context (from Analyst)\n{market_context}\n\n"
            f"## Prior Knowledge Base\n{kb_summary}\n\n"
            "Search the knowledge base for any relevant prior experiments, "
            "then propose a specific experiment to run next. "
            "Output the experiment JSON and explain your reasoning."
        )
        return self.call(prompt)

    def synthesise(self, review: str, iteration: int) -> str:
        prompt = (
            f"## Reviewer Analysis (Iteration {iteration})\n{review}\n\n"
            "Based on this review: decide PROMOTE, ITERATE, or REJECT. "
            "If ITERATE, specify exactly what to change. "
            "If PROMOTE, save the strategy to the knowledge base. "
            "If REJECT, save the failure with reasoning."
        )
        return self.call(prompt)

    def generate_research_plan(self, ticker: str, goal: str, kb_summary: str) -> str:
        prompt = (
            f"## Research Goal\n{goal}\n\n"
            f"## Target Asset\n{ticker}\n\n"
            f"## Prior Knowledge\n{kb_summary}\n\n"
            "First fetch market data for this ticker to understand its return profile. "
            "Then outline a 3-step research plan: what to test first, what to test if that works, "
            "and what to test if that fails. Keep the plan concrete and evidence-based."
        )
        return self.call(prompt)
