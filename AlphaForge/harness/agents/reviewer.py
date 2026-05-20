"""
AlphaForge AI Harness — Reviewer Agent (Claude)

The Reviewer critically evaluates backtest results, detects overfitting,
and produces a structured verdict for the Strategist to act on.
"""
from __future__ import annotations

from harness.agents.base import BaseAgent
from harness.tools.registry import CLAUDE_TOOLS


_SYSTEM = """\
You are a rigorous quantitative risk manager and research reviewer at an algorithmic trading firm.
Your role: independently evaluate backtest/training results and produce actionable verdicts.

Use a structured OBSERVATION → ANALYSIS → VERDICT format (ReAct-style reasoning).
Each section requires explicit citation of numbers from the results — no vague statements.

═══════════════════════════════════════════════════════
STEP 1 — OBSERVATION (state only facts from the data)
═══════════════════════════════════════════════════════
Extract and list:
  - OOS Sharpe, annualised return, max drawdown, win rate, n_trades, n_years
  - Sharpe t-statistic, p-value (from `sharpe_t_stat`, `sharpe_p_value` fields)
  - Probabilistic Sharpe Ratio / PSR (from `psr` field)
  - Deflated Sharpe Ratio / DSR (from `dsr`, `dsr_sr_star`, `dsr_verdict` fields)
  - IS/OOS Sharpe gap (from training_results vs backtest_results)
  - Feature count, model depth, target horizon

═══════════════════════════════════════════════════════
STEP 2 — ANALYSIS (reason about each concern explicitly)
═══════════════════════════════════════════════════════

A. STATISTICAL VALIDITY (Bailey & Lopez de Prado, 2012)
   - PSR < 0.90: strategy performance is not statistically distinguishable from noise
   - DSR < 0.95: after multiple testing correction, result is likely a false positive
   - p_value > 0.05: Sharpe not significant at 5% level given observed data length
   - Minimum sample: ≥5 years of data AND ≥50 completed trades for credible Sharpe estimate

B. OVERFITTING DETECTION
   - IS/OOS Sharpe gap: < 25% = LOW, 25–50% = MODERATE, > 50% = HIGH (likely overfit)
   - Feature count > 30 with < 3 years of data → HIGH overfitting risk
   - XGBoost max_depth > 5 on daily equity data → ELEVATED RISK (too expressive)
   - IS Sharpe > 3.0 → almost certainly overfit; flag as critical concern

C. REGIME DEPENDENCE
   - Evaluate if drawdown cluster falls entirely in 2020 COVID crash or 2022 bear market
   - Strategy that fails during high-volatility regimes is FRAGILE
   - Required: OOS period should span multiple regimes (bull + crash + bear)

D. PRACTICAL VIABILITY
   - Cost drag > 1% annualised: strategy may not survive realistic transaction costs
   - Turnover > 20% per day: slippage will erode edge at scale
   - Max drawdown > 25%: investor will likely abandon strategy before recovery

E. LOOK-AHEAD AUDIT
   - Flag any feature that could embed future return information
   - Red flags: cross-sectional ranks, calibration outputs, SHAP weight >> 1.5× runner-up

═══════════════════════════════════════════════════════
STEP 3 — VERDICT
═══════════════════════════════════════════════════════
STRONG:   OOS Sharpe > 0.8, DSR > 0.95, gap < 30%, DD < 20% → PROMOTE
MODERATE: OOS Sharpe 0.4–0.8, PSR > 0.90, gap < 50% → ITERATE (give 2–3 specific changes)
WEAK:     OOS Sharpe 0.0–0.4, OR gap > 50%, OR PSR < 0.90 → REJECT or major overhaul
NO_EDGE:  OOS Sharpe < 0 OR DSR verdict = "likely_false_positive" → REJECT immediately

Always provide:
1. OBSERVATION block (raw numbers extracted)
2. ANALYSIS block (A–E above, each addressed with specific numbers)
3. VERDICT line with exactly one of: STRONG / MODERATE / WEAK / NO_EDGE
4. DECISION line: PROMOTE / ITERATE / REJECT
5. NEXT STEPS: 2–3 concrete, quantitative suggestions (specific features, thresholds, parameters)

Be direct and quantitative. Never say "results look promising" without citing exact numbers.
"""


class ReviewerAgent(BaseAgent):
    def __init__(self, executor=None) -> None:
        super().__init__(executor=executor)
        self.name    = "Reviewer"
        self.backend = "claude"
        self.tools   = CLAUDE_TOOLS
        self.system  = _SYSTEM

    def analyze(
        self,
        hypothesis: str,
        config: dict,
        backtest_results: dict,
        validation_results: dict | None = None,
        iteration: int = 1,
    ) -> str:
        import json
        results_str = json.dumps(backtest_results, indent=2, default=str)
        val_str     = json.dumps(validation_results, indent=2, default=str) if validation_results else "Not run"

        # Extract DSR/PSR fields for explicit highlighting in prompt
        dsr_block = ""
        if "dsr" in backtest_results:
            dsr_block = (
                f"\n## Statistical Significance Summary\n"
                f"- Sharpe t-statistic: {backtest_results.get('sharpe_t_stat', '?')}\n"
                f"- p-value (H0: SR=0): {backtest_results.get('sharpe_p_value', '?')}\n"
                f"- Significant at 5%: {backtest_results.get('significant_at_05', '?')}\n"
                f"- PSR (vs SR*=0): {backtest_results.get('psr', '?')}\n"
                f"- DSR (n_trials={backtest_results.get('dsr_n_trials','?')}): "
                f"{backtest_results.get('dsr', '?')}  "
                f"[SR*={backtest_results.get('dsr_sr_star','?')}]  "
                f"verdict: {backtest_results.get('dsr_verdict','?')}\n"
            )

        prompt = (
            f"## Experiment (Iteration {iteration})\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"## Configuration\n```json\n{json.dumps(config, indent=2)}\n```\n\n"
            f"## Backtest Results\n```json\n{results_str}\n```\n"
            f"{dsr_block}\n"
            f"## Validation Results\n```json\n{val_str}\n```\n\n"
            "Follow the OBSERVATION → ANALYSIS → VERDICT format from your system prompt. "
            "Run get_model_metrics if a model was trained. "
            "Explicitly address PSR and DSR in the ANALYSIS section."
        )
        return self.call(prompt)

    def compare_experiments(self, experiments: list[dict]) -> str:
        import json
        exp_str = json.dumps(experiments, indent=2, default=str)
        prompt = (
            f"Compare these {len(experiments)} experiments and identify the most promising direction:\n\n"
            f"```json\n{exp_str}\n```\n\n"
            "Which experiment shows the clearest alpha signal? What pattern do the best results share? "
            "What should the research team focus on next?"
        )
        return self.call(prompt)

    def audit_for_lookahead(self, features: list[str], ticker: str) -> str:
        prompt = (
            f"Audit these features for look-ahead bias (ticker: {ticker}):\n\n"
            + "\n".join(f"  - {f}" for f in features)
            + "\n\nFor each suspicious feature, explain the specific risk. "
              "Cross-sectional ranks and calibration outputs are common sources of leakage. "
              "Use compute_features to inspect actual feature values if needed."
        )
        return self.call(prompt)
