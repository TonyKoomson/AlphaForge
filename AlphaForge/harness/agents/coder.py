"""
AlphaForge AI Harness — Coder Agent (Claude)

The Coder generates new alpha factor Python code, validates it for
look-ahead bias, and integrates it into the feature pipeline.
"""
from __future__ import annotations

from harness.agents.base import BaseAgent
from harness.tools.registry import CLAUDE_TOOLS


_SYSTEM = """\
You are a quantitative developer specialising in feature engineering for ML trading models.
Your role: implement new alpha factors as look-ahead-free Python code and integrate them into the feature pipeline.

LOOK-AHEAD BIAS RULES — CRITICAL
1. NEVER use `.shift(-N)` for N > 0 — this accesses future data
2. NEVER use `pct_change(-N)` with negative N
3. All rolling windows compute backwards in time: `.rolling(N)` looks at the last N bars
4. Use `.shift(1)` or `.shift(N)` (positive) to avoid using the current bar's close in signals
5. NEVER use `expanding(min_periods=0).apply(lambda x: x.tail())` patterns that peek forward
6. After computing a feature at bar T, it should only use data from bars T and earlier

FUNCTION SIGNATURE REQUIREMENT
Every factor must be a standalone function:
```python
def compute_<name>(df: pd.DataFrame) -> pd.Series:
    '''Economic intuition: <why this predicts returns>'''
    # df has columns: open, high, low, close, volume
    # Must return a pd.Series with the same index as df
    ...
```

AVAILABLE COLUMNS IN df
- close, open, high, low, volume (OHLCV)
- Use only these inputs — no external data

CODING STYLE
- Use pandas vectorised operations (no row-by-row loops)
- Handle edge cases: use .fillna(0) or .clip() where appropriate
- Use `min_periods` in rolling to avoid NaN at the start
- Document the economic intuition in the docstring
- Keep functions under 30 lines

FACTOR CATEGORIES TO CONSIDER
- Momentum: rate-of-change over different lookbacks
- Mean-reversion: deviation from moving average, RSI-like oscillators
- Volatility: realised vol, vol-of-vol, Parkinson estimator
- Volume: turnover anomaly, volume surprise, Amihud illiquidity
- Trend: channel breakout, 52-week high proximity
- Quality: price stability, drawdown recovery speed

AFTER GENERATING CODE
- Use the add_alpha_factor tool to validate and register the factor
- Report whether integration succeeded and what the factor name is
"""


class CoderAgent(BaseAgent):
    def __init__(self, executor=None) -> None:
        super().__init__(executor=executor)
        self.name    = "Coder"
        self.backend = "claude"
        self.tools   = CLAUDE_TOOLS
        self.system  = _SYSTEM

    def generate_factor(self, factor_name: str, hypothesis: str, analyst_context: str = "") -> str:
        prompt = (
            f"Generate a new alpha factor called '{factor_name}'.\n\n"
            f"Hypothesis: {hypothesis}\n"
            + (f"\nAnalyst context:\n{analyst_context}\n" if analyst_context else "")
            + "\nWrite the `compute_{name}` function, validate it is look-ahead free, "
              "then call add_alpha_factor to register it. "
              "Return the factor name and a brief explanation of how it was implemented."
        )
        return self.call(prompt)

    def improve_factor(self, factor_name: str, current_code: str, feedback: str) -> str:
        prompt = (
            f"Improve the alpha factor '{factor_name}'.\n\n"
            f"Current implementation:\n```python\n{current_code}\n```\n\n"
            f"Feedback from review:\n{feedback}\n\n"
            "Generate an improved version and register it via add_alpha_factor."
        )
        return self.call(prompt)

    def audit_features(self, feature_list: list[str]) -> str:
        prompt = (
            "Audit the following features for potential look-ahead bias:\n\n"
            + "\n".join(f"  - {f}" for f in feature_list)
            + "\n\nFor each feature: confirm it is purely backward-looking, "
              "or flag specific concerns. Focus on features with unusual names "
              "or that reference ranks, quantiles, or cross-sectional computations."
        )
        return self.call(prompt)
