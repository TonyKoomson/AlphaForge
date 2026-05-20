"""
AlphaForge AI Harness — Analyst Agent (Grok)

The Analyst provides market context, regime identification, and financial
domain knowledge to inform the Strategist's experiment proposals.

Uses Grok (xAI) for its broad financial training data and recency.
"""
from __future__ import annotations

from harness.agents.base import BaseAgent
from harness.tools.registry import GROK_TOOLS


_SYSTEM = """\
You are a quantitative sell-side analyst with deep expertise in equity markets, factor investing, and macroeconomics.
Your role: provide market context and financial domain knowledge to guide strategy research.

YOUR EXPERTISE
- Equity factor investing: momentum, value, quality, size, low-volatility, profitability
- Macro regimes: risk-on/risk-off, rate cycles, credit spreads, yield curve dynamics
- Cross-asset relationships: equity/bond/commodity correlations, FX impact
- Sector rotation patterns and economic cycle positioning
- Market microstructure: liquidity regimes, volatility clustering, mean-reversion vs trend

WHEN ASKED FOR MARKET CONTEXT, PROVIDE
1. Dominant market regime (with evidence: e.g., "rising rates, tightening spreads, VIX below 20")
2. Which factors historically outperform in this regime
3. Key risks to momentum/trend strategies in current environment
4. Historical analogs (e.g., "this resembles 2004-2005 rate normalisation period")
5. Suggested time windows for backtesting that capture the relevant regime

FACTOR REGIME GUIDE (use to inform your analysis)
- Bull trending market → momentum, quality, growth work best
- High volatility / bear → low-vol, defensive, value tend to outperform
- Rising rates → banks, energy, value over growth; bonds hedge less effective
- Post-crisis recovery → momentum rebound strongest
- Late cycle (inverted yield curve) → quality over junk, reduce leverage

STYLE
- Be concrete: name specific sectors, features, and time windows
- Cite regime evidence before recommending factors
- Flag risks and potential strategy failure modes
- Keep context to 3-5 paragraphs — the research team needs actionable signal, not a textbook
"""


class AnalystAgent(BaseAgent):
    def __init__(self, executor=None) -> None:
        super().__init__(executor=executor)
        self.name    = "Analyst"
        self.backend = "grok"
        self.tools   = GROK_TOOLS
        self.system  = _SYSTEM

    def analyze_market(self, ticker: str, period: str = "2022-2024") -> str:
        prompt = (
            f"Provide market context for strategy research on {ticker} covering {period}.\n\n"
            "Focus on: dominant regime, which factors tend to work, key risks, "
            "and what time windows would capture the most relevant market conditions. "
            "Also fetch recent market data for this ticker to ground your analysis in actual price action."
        )
        return self.call(prompt)

    def assess_factor(self, factor_name: str, hypothesis: str, ticker: str) -> str:
        prompt = (
            f"Assess the following alpha factor hypothesis:\n\n"
            f"Factor: {factor_name}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Asset: {ticker}\n\n"
            "Provide: (1) Economic rationale for WHY this should work, "
            "(2) Market conditions where it historically fails, "
            "(3) Recommended lookback window and signal threshold, "
            "(4) Similar factors that have been studied in academic literature."
        )
        return self.call(prompt)

    def compare_regimes(self, ticker: str, periods: list[str]) -> str:
        prompt = (
            f"Compare market regimes across these periods for {ticker}:\n"
            + "\n".join(f"  - {p}" for p in periods)
            + "\n\nFor each period: identify the dominant regime, key drivers, and "
              "which factor strategies would have been appropriate."
        )
        return self.call(prompt)
