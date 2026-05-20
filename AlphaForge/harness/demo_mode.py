"""
AlphaForge AI Harness — Demo Mode

Pre-written agent responses that simulate the full four-agent research
loop without requiring real API keys.

Real backtests and model training run against the actual AlphaForge
infrastructure; only the agent conversation is pre-written.
"""
from __future__ import annotations

import random
from typing import Optional


# ── Analyst responses (plain, narrative-driven) ──────────────────────────────

_ANALYST_RESPONSES = [
    """\
Here's what's happening in the {ticker} market right now:

The market is in a **transitional phase** — prices are moving above their long-term average
(200-day trend line) but momentum is slowing down slightly. Volatility is running about 20%
above normal, which means the market is a bit choppy.

**What this means for our strategy:**
- Trend-following should work well here — the overall direction is up
- Short-term reversals (buy the dip) are getting riskier than usual
- Strategies that combine price momentum with a market-health check tend to do well

**The opportunity I see:**
12-month momentum is one of the most reliable patterns in financial markets.
When prices have been rising over the past year AND the market is in an uptrend,
the odds of continuation are historically around 58-62%. That's not huge, but
repeated consistently it adds up to real outperformance.

**Recommended test period:** 2018–2023 covers a bull market, a crash, a recovery,
and a rate-hike bear market. If a strategy survives all of that, it's genuinely robust.
""",
    """\
Market update for {ticker}:

We're in what I'd call a **late-cycle bull market** — things have been going up
for a while, but some warning signs are appearing (inverted yield curve, rising costs).

**Best strategies in this environment:**
- Focus on quality: companies (and ETFs) with strong fundamentals tend to hold up
- Use volatility as a filter — avoid trading when markets are extremely turbulent
- Momentum still works but needs a safety check to avoid big crashes

**What I recommend testing:**
A strategy that combines 12-month momentum with a market health check.
The idea: only take buy signals when the 50-day average is above the 200-day average.
This simple filter alone historically cuts drawdowns by 30-40%.

**Why the 2018-2023 period?**
It's the toughest recent test we have: COVID crash (-34% in a month),
a full recovery, then a rate-hike bear market (-25% in 2022). If a strategy
works across all of those, it's the real deal.
""",
    """\
{ticker} market analysis — here's my read:

**The market is trending strongly upward.** 72% of stocks are above their
200-day moving average, which is a classic bull market signal. Low volatility
(VIX near 14) means the market is confident.

**What works in this environment:**
Strong bull markets are where momentum strategies shine brightest.
The core idea: winners keep winning. Stocks/ETFs that have done well over
the past 12 months tend to continue outperforming for another 3-6 months.

**My recommended approach:**
- Look at 12-month past performance as the main signal
- Add a trend health check: only buy when prices are above their 200-day average
- Use a machine learning model to combine multiple signals and find the best entry points

The historical equivalent of this market? 2013 and 2017 — both years where
simple momentum + trend strategies returned 20-30%. Let's see if we can replicate that.
""",
]

_PROPOSAL_RESPONSES = [
    """\
Based on the market analysis, here's the strategy I want to test:

**The idea:** "12-month momentum with trend confirmation"

When a market has been rising for 12 months AND is currently in an uptrend
(50-day average above 200-day average), the probability of further gains is
statistically elevated. We'll train an AI model to detect these moments and
generate buy/sell signals.

**Why this should work:**
Institutional investors follow trends because they have to — large funds
can't exit positions instantly, so price momentum reflects real capital flows.
The trend filter (50-day vs 200-day) acts as a risk manager, cutting exposure
during genuine bear markets like 2022.

```json
{
  "hypothesis": "12-month momentum with SMA50/200 regime filter outperforms buy-and-hold by selecting only trend-aligned entries",
  "ticker": "SPY",
  "features": ["mom_12_1", "mom_5d", "sma_50_above_200", "channel_pos_52w", "vol_21d"],
  "model_params": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
  "signal_threshold": 0.55,
  "start": "2018-01-01",
  "end": "2023-12-31",
  "expected_edge": "Trend persistence over 12-month horizon combined with regime filter reduces false entries"
}
```

**No data snooping:** All signals look only at past data. The AI never sees future prices.
""",
    """\
I searched the strategy library and found a prior momentum test that scored well.
Building on that, here's my next experiment:

**The idea:** "Momentum + oversold dip buying in uptrends"

Pure trend-following exits positions when prices dip, which often means selling
at the worst moment. Instead, we'll add an RSI signal: when a market is in an
uptrend BUT has temporarily dipped to oversold levels, that's actually a great entry point.

```json
{
  "hypothesis": "RSI oversold dips within a bull trend offer high-probability bounce entries",
  "ticker": "SPY",
  "features": ["rsi_14", "sma_50_above_200", "bb_width", "vol_21d", "mom_12_1"],
  "model_params": {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9},
  "signal_threshold": 0.57,
  "start": "2018-01-01",
  "end": "2023-12-31",
  "expected_edge": "Oversold readings within uptrend have historically offered limited downside and strong mean-reversion upside"
}
```

**The intuition:** Think of this as "buy the dip, but only when the overall trend is your friend."
""",
    """\
After reviewing what's worked so far, I want to test a different angle:

**The idea:** "Trade only when market conditions are perfectly aligned"

Instead of always being in the market, this strategy is highly selective.
It only takes positions when BOTH conditions are true:
1. The market is in a clear uptrend (50-day > 200-day)
2. Volatility is low (below the 25th percentile historically)

This drastically reduces the number of trades but should dramatically improve
the quality of each one. Think of it as waiting for the "easy trades."

```json
{
  "hypothesis": "Selective entry when trend and low-volatility conditions both align produces cleaner signals",
  "ticker": "SPY",
  "features": ["sma_50_above_200", "sma_50_dist", "vol_21d", "mom_12_1", "dollar_volume"],
  "model_params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.04, "reg_lambda": 1.5},
  "signal_threshold": 0.55,
  "start": "2018-01-01",
  "end": "2023-12-31",
  "expected_edge": "Dual filter (trend + low volatility) eliminates most losing trades at the cost of fewer total trades"
}
```

**Why this matters:** Fewer, better trades can outperform more frequent, noisier ones
— especially after accounting for trading costs.
""",
]

_REVIEW_RESPONSES = [
    """\
## Strategy Review — Experiment {iteration}

**Overall assessment:** {verdict_text}

**Performance breakdown:**
- Score: {sharpe:.3f} (for reference: anything above 0.8 is excellent, 0.5-0.8 is good, below 0.3 needs work)
- Worst loss period: {max_dd:.1f}% (target: keep this below 25%)
- The strategy was consistently profitable or not across different time periods: {fold_verdict}

**Is this genuine or just luck?**
✓ All signals look only at past data — no cheating by peeking at future prices
✓ The top predictive factors are economically sensible (momentum, trend health) — not random noise
✓ Results were consistent across different market conditions tested separately

**What could be better:**
The strategy is sensitive to the 2022 bear market. Most trend-following strategies
take a hit during that period — the question is whether recovery comes quickly enough.

**My recommendation:** {recommendation}
""",
]

_SYNTHESIS_RESPONSES = [
    "APPROVED — Score {sharpe:.3f} is excellent, max loss {max_dd:.1f}% is within limits. Saving to strategy library.",
    "PROMISING — Score {sharpe:.3f} shows a real edge. Suggest: (1) raise signal threshold to 0.58 for cleaner signals, (2) add volatility filter to reduce 2022 exposure, (3) test longer history back to 2015.",
    "INTERESTING — Score {sharpe:.3f} has potential. Try: reduce tree depth to 3 (may be slightly overfit), add trading volume as a liquidity check, test a 3-day forward horizon instead of 5-day.",
    "NOT YET — Score {sharpe:.3f} doesn't show enough edge after accounting for trading costs. Logging this as a failed approach so we don't test it again.",
]


# ── DemoAgent: wraps any BaseAgent with pre-written responses ─────────────────

class DemoAgentMixin:
    """
    Replaces live API calls with pre-written realistic responses.
    All tool calls (training, backtesting) still run for real.
    """
    _demo_call_count: int = 0

    def call(self, user_message: str, max_tool_rounds: int = 8) -> str:
        self.add_user(user_message)
        response = self._demo_response(user_message)
        self.add_assistant(response)
        return response

    def _demo_response(self, prompt: str) -> str:
        raise NotImplementedError


class DemoAnalystAgent(DemoAgentMixin):
    name = "Market Analyst (DEMO)"

    def analyze_market(self, ticker: str, period: str = "2018-2023") -> str:
        return random.choice(_ANALYST_RESPONSES).format(ticker=ticker, period=period)

    def assess_factor(self, factor_name: str, hypothesis: str, ticker: str) -> str:
        return (
            f"Economic assessment of '{factor_name}':\n\n"
            f"{hypothesis}\n\n"
            "This factor is grounded in sound market microstructure reasoning. "
            "The signal uses only backward-looking data — no future information is used."
        )

    def compare_regimes(self, ticker: str, periods: list) -> str:
        return (
            "Market comparison across periods:\n"
            "• 2019-2021 (bull market): Momentum strategies excelled\n"
            "• 2022 (bear market): Defensive, low-volatility approaches held up best\n"
            "• 2023 (recovery): Momentum resumed, trend-following worked well again"
        )

    def _demo_response(self, prompt: str) -> str:
        return random.choice(_ANALYST_RESPONSES).format(ticker="SPY", period="2018-2023")

    def call(self, user_message: str, max_tool_rounds: int = 8) -> str:
        self.add_user(user_message)
        resp = self._demo_response(user_message)
        self.add_assistant(resp)
        return resp

    def add_user(self, msg: str) -> None: pass
    def add_assistant(self, msg: str) -> None: pass


class DemoStrategistAgent(DemoAgentMixin):
    name = "Strategy Designer (DEMO)"
    _proposal_idx: int = 0

    def propose_experiment(self, market_context: str, kb_summary: str) -> str:
        resp = _PROPOSAL_RESPONSES[self._proposal_idx % len(_PROPOSAL_RESPONSES)]
        type(self)._proposal_idx += 1
        return resp

    def synthesise(self, review: str, iteration: int) -> str:
        sharpe = _extract_float(review, "Score:", 0.55)
        max_dd = _extract_float(review, "Worst loss period:", 12.0)
        if sharpe >= 0.8 and max_dd <= 25.0:
            return _SYNTHESIS_RESPONSES[0].format(sharpe=sharpe, max_dd=max_dd)
        elif sharpe >= 0.3:
            return _SYNTHESIS_RESPONSES[iteration % 2 + 1].format(sharpe=sharpe, max_dd=max_dd)
        else:
            return _SYNTHESIS_RESPONSES[3].format(sharpe=sharpe, max_dd=max_dd)

    def generate_research_plan(self, ticker: str, goal: str, kb_summary: str) -> str:
        return (
            f"Research plan for {ticker}\n\n"
            f"Goal: {goal}\n\n"
            "Step 1 — Test the classic approach\n"
            "  Try 12-month momentum with a trend health filter. This is the most\n"
            "  well-documented pattern in financial markets and a solid baseline.\n\n"
            "Step 2 — If Step 1 works well: Add volatility awareness\n"
            "  Overlay a volatility filter to reduce exposure during turbulent periods.\n"
            "  This should cut drawdowns without sacrificing much return.\n\n"
            "Step 3 — If Step 2 falls short: Try dip-buying within uptrends\n"
            "  Test RSI oversold signals filtered to only fire in uptrending markets.\n"
            "  A completely different mechanism to Step 1 and 2.\n\n"
            "We'll stop early if we find 3 strategies that all meet our quality bar —\n"
            "that's strong evidence we've genuinely found something that works."
        )

    def _demo_response(self, prompt: str) -> str:
        return self.propose_experiment(prompt, "")

    def call(self, user_message: str, max_tool_rounds: int = 8) -> str:
        self.add_user(user_message)
        resp = self._demo_response(user_message)
        self.add_assistant(resp)
        return resp

    def add_user(self, msg: str) -> None: pass
    def add_assistant(self, msg: str) -> None: pass


class DemoCoderAgent(DemoAgentMixin):
    name = "AI Engineer (DEMO)"

    def generate_factor(self, factor_name: str, hypothesis: str, analyst_context: str) -> str:
        return (
            f"New signal: `{factor_name}`\n\n"
            f"Idea: {hypothesis[:80]}\n\n"
            "```python\n"
            f"def compute_{factor_name}(df: pd.DataFrame) -> pd.Series:\n"
            f'    """Measures {factor_name}: {hypothesis[:50]}"""\n'
            "    # Uses only historical data — no future prices used\n"
            "    short_avg = df['close'].rolling(20).mean()\n"
            "    long_avg  = df['close'].rolling(60).mean()\n"
            "    return (short_avg / long_avg - 1).fillna(0.0)\n"
            "```\n\n"
            "Safety checks:\n"
            "✓ Only looks backward — no future data leakage\n"
            "✓ Handles missing values gracefully\n"
            "✓ Returns a clean numeric series"
        )

    def _demo_response(self, prompt: str) -> str:
        return self.generate_factor("custom_factor", "Price trend ratio", "")

    def call(self, user_message: str, max_tool_rounds: int = 8) -> str:
        self.add_user(user_message)
        resp = self._demo_response(user_message)
        self.add_assistant(resp)
        return resp

    def add_user(self, msg: str) -> None: pass
    def add_assistant(self, msg: str) -> None: pass


class DemoReviewerAgent(DemoAgentMixin):
    name = "Quality Reviewer (DEMO)"

    def analyze(
        self,
        hypothesis: str = "",
        config: Optional[dict] = None,
        backtest_results: Optional[dict] = None,
        validation_results: Optional[dict] = None,
        iteration: int = 1,
    ) -> str:
        bt      = backtest_results or {}
        sharpe  = float(bt.get("sharpe", 0.55))
        max_dd  = abs(float(bt.get("max_dd", 12.0)))

        if sharpe >= 0.8 and max_dd <= 25:
            verdict_text = "This strategy passes all our quality checks. It shows a genuine edge."
            fold_verdict  = "Consistently profitable across all test windows."
            recommendation = "Approve for deployment — this is a strong result."
        elif sharpe >= 0.3:
            verdict_text = "Promising but needs refinement before approval."
            fold_verdict  = "Mostly profitable but inconsistent in some periods."
            recommendation = "Iterate — adjust the signal threshold and retest."
        else:
            verdict_text = "This configuration doesn't show enough edge to be useful."
            fold_verdict  = "Inconsistent across test periods — likely noise, not signal."
            recommendation = "Reject — log as a failure and try a different approach."

        return _REVIEW_RESPONSES[0].format(
            iteration=iteration,
            sharpe=sharpe,
            max_dd=max_dd,
            verdict_text=verdict_text,
            fold_verdict=fold_verdict,
            recommendation=recommendation,
        )

    def _demo_response(self, prompt: str) -> str:
        return self.analyze()

    def call(self, user_message: str, max_tool_rounds: int = 8) -> str:
        self.add_user(user_message)
        resp = self._demo_response(user_message)
        self.add_assistant(resp)
        return resp

    def add_user(self, msg: str) -> None: pass
    def add_assistant(self, msg: str) -> None: pass


# ── Helper ────────────────────────────────────────────────────────────────────

def _extract_float(text: str, label: str, default: float) -> float:
    idx = text.find(label)
    if idx < 0:
        return default
    rest = text[idx + len(label):].strip()
    for token in rest.split():
        cleaned = token.rstrip(",%")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return default


def build_demo_harness(session_id: Optional[str] = None):
    """
    Build an AlphaHarness with all four agents replaced by demo stubs.
    The executor, KB, and bandit are REAL — only the agent responses are pre-written.
    """
    from harness.orchestrator import AlphaHarness

    harness = AlphaHarness(session_id=session_id)

    harness.strategist = DemoStrategistAgent()
    harness.analyst    = DemoAnalystAgent()
    harness.coder      = DemoCoderAgent()
    harness.reviewer   = DemoReviewerAgent()

    harness.strategist._executor = harness.executor
    harness.analyst._executor    = harness.executor
    harness.coder._executor      = harness.executor
    harness.reviewer._executor   = harness.executor

    return harness
