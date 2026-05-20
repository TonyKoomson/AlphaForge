"""
AlphaForge AI Harness — Tool Registry

Defines every tool available to agents in both:
  - Anthropic format  (for Claude)
  - OpenAI format     (for Grok via xAI API)
"""
from __future__ import annotations

# ── Tool definitions (canonical) ──────────────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "name": "run_backtest",
        "description": (
            "Run a backtesting simulation for a given ticker and strategy configuration. "
            "Returns Sharpe ratio, annualised return, max drawdown, and trade statistics. "
            "Use this to validate a strategy hypothesis on historical data."
        ),
        "properties": {
            "ticker":      {"type": "string",  "description": "Stock/ETF ticker (e.g. 'SPY', 'AAPL')"},
            "start":       {"type": "string",  "description": "Backtest start date YYYY-MM-DD"},
            "end":         {"type": "string",  "description": "Backtest end date YYYY-MM-DD"},
            "features":    {"type": "array",   "items": {"type": "string"},
                            "description": "Feature column names to include (from feature engine)"},
            "model_params":{"type": "object",  "description": "XGBoost hyperparameters (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_lambda, reg_alpha)"},
            "signal_threshold": {"type": "number", "description": "Min predicted probability to take a position (0.5–0.7)"},
            "use_spy_timing":   {"type": "boolean", "description": "Apply SPY SMA50/200 Death Cross market timing (default true)"},
        },
        "required": ["ticker", "start", "end"],
    },
    {
        "name": "train_model",
        "description": (
            "Train an XGBoost ensemble model on a ticker with purged walk-forward cross-validation. "
            "Returns OOS Sharpe, IS/OOS gap, feature importances, and saves the model artifact. "
            "Use this before run_backtest to get a trained model."
        ),
        "properties": {
            "ticker":      {"type": "string",  "description": "Ticker to train on"},
            "start":       {"type": "string",  "description": "Training data start YYYY-MM-DD"},
            "end":         {"type": "string",  "description": "Training data end YYYY-MM-DD (as-of date)"},
            "features":    {"type": "array",   "items": {"type": "string"},
                            "description": "Feature columns to include. Leave empty for auto-selection."},
            "model_params":{"type": "object",  "description": "XGBoost hyperparameters"},
            "target_horizon": {"type": "integer", "description": "Forward return horizon in bars (default 5)"},
        },
        "required": ["ticker"],
    },
    {
        "name": "fetch_market_data",
        "description": (
            "Fetch OHLCV price data for a ticker from the data cache or yfinance. "
            "Returns summary statistics (date range, mean return, annualised vol, Sharpe of buy-hold). "
            "Use to understand the asset before proposing a strategy."
        ),
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol"},
            "start":  {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end":    {"type": "string", "description": "End date YYYY-MM-DD"},
        },
        "required": ["ticker"],
    },
    {
        "name": "compute_features",
        "description": (
            "Generate the full feature matrix for a ticker and return a summary "
            "(available feature names, sample values, correlation with forward returns). "
            "Use this to explore what signals are available before selecting features for training."
        ),
        "properties": {
            "ticker":  {"type": "string", "description": "Ticker symbol"},
            "start":   {"type": "string", "description": "Start date"},
            "end":     {"type": "string", "description": "End date"},
            "top_n":   {"type": "integer", "description": "Number of top features to show by IC (default 20)"},
        },
        "required": ["ticker"],
    },
    {
        "name": "add_alpha_factor",
        "description": (
            "Add a new alpha factor (feature) to the feature pipeline by supplying Python source code. "
            "The code must define a function `compute_<name>(df) -> pd.Series` that uses only past data. "
            "The factor is validated for look-ahead bias before being integrated. "
            "Returns success/failure and the factor name."
        ),
        "properties": {
            "factor_name": {"type": "string", "description": "Snake_case name for the factor (e.g. 'reversal_5d')"},
            "python_code": {"type": "string", "description": "Complete Python function definition"},
            "hypothesis":  {"type": "string", "description": "Economic rationale for why this factor should predict returns"},
        },
        "required": ["factor_name", "python_code", "hypothesis"],
    },
    {
        "name": "run_validation",
        "description": (
            "Run full walk-forward cross-validation and overfitting detection on a ticker's trained model. "
            "Returns per-fold metrics, IS/OOS Sharpe gap, overfitting score, and a verdict "
            "(STRONG / MODERATE / WEAK / NO_EDGE)."
        ),
        "properties": {
            "ticker": {"type": "string", "description": "Ticker with a trained model artifact"},
            "end":    {"type": "string", "description": "Validation end date (as-of date)"},
        },
        "required": ["ticker"],
    },
    {
        "name": "run_universe_trade",
        "description": (
            "Run a ranking-based universe portfolio simulation across many tickers simultaneously. "
            "Selects top-N tickers by a ranking factor (momentum, model confidence, or liquidity) "
            "and simulates rebalancing daily. Returns aggregate portfolio metrics."
        ),
        "properties": {
            "universe":       {"type": "string",  "description": "Universe name: 'sp500', 'nasdaq100', 'etfs', 'all_us_stocks'"},
            "top_n":          {"type": "integer", "description": "Number of tickers to hold simultaneously"},
            "start":          {"type": "string",  "description": "Simulation start date"},
            "end":            {"type": "string",  "description": "Simulation end date"},
            "ranking_factor": {"type": "string",  "description": "Ranking method: 'model', 'cross_momentum', 'liquidity', 'composite'"},
            "spy_timing":     {"type": "string",  "description": "SPY timing: 'sma50_200', 'price_200', 'none'"},
            "stop_loss_pct":  {"type": "number",  "description": "Hard stop loss fraction (0 = disabled, 0.10 = 10%)"},
        },
        "required": ["universe", "top_n", "start", "end"],
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the knowledge base for prior experiments, findings, and heuristics. "
            "Use to avoid repeating failed approaches and to build on prior discoveries."
        ),
        "properties": {
            "query":      {"type": "string",  "description": "Natural language query"},
            "entry_type": {"type": "string",  "description": "Filter by type: 'experiment', 'finding', 'heuristic', 'failure', 'promotion'"},
            "min_sharpe": {"type": "number",  "description": "Only return experiments with OOS Sharpe >= this"},
            "limit":      {"type": "integer", "description": "Max results to return (default 5)"},
        },
        "required": ["query"],
    },
    {
        "name": "save_finding",
        "description": "Save an insight, heuristic, or finding to the knowledge base for future reference.",
        "properties": {
            "entry_type": {"type": "string", "description": "Type: 'finding', 'heuristic', or 'failure'"},
            "title":      {"type": "string", "description": "Short title (< 100 chars)"},
            "body":       {"type": "string", "description": "Full description of the insight or rule"},
            "tags":       {"type": "array",  "items": {"type": "string"}, "description": "Relevant tags"},
        },
        "required": ["entry_type", "title", "body"],
    },
    {
        "name": "get_model_metrics",
        "description": (
            "Load saved training metrics for a ticker's model artifact. "
            "Returns: feature importances (SHAP), fold-by-fold Sharpe, training date, feature count."
        ),
        "properties": {
            "ticker": {"type": "string", "description": "Ticker with a trained model"},
        },
        "required": ["ticker"],
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for market research, factor investing literature, or macro context. "
            "Returns a concise summary of search results. Use to gather current market regime context, "
            "recent academic findings on factors, or macro data that informs strategy design."
        ),
        "properties": {
            "query": {"type": "string", "description": "Search query (e.g. 'momentum factor performance 2023 rising rates')"},
            "context": {"type": "string", "description": "Optional: 'factor_research', 'macro_context', 'sector_rotation', 'regime_analysis'"},
        },
        "required": ["query"],
    },
]


# ── Format converters ─────────────────────────────────────────────────────────

def _to_anthropic(tool: dict) -> dict:
    """Convert canonical tool def → Anthropic `tools` format."""
    return {
        "name":        tool["name"],
        "description": tool["description"],
        "input_schema": {
            "type":       "object",
            "properties": tool["properties"],
            "required":   tool.get("required", []),
        },
    }


def _to_openai(tool: dict) -> dict:
    """Convert canonical tool def → OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name":        tool["name"],
            "description": tool["description"],
            "parameters": {
                "type":       "object",
                "properties": tool["properties"],
                "required":   tool.get("required", []),
            },
        },
    }


# ── Public exports ────────────────────────────────────────────────────────────

CLAUDE_TOOLS: list[dict] = [_to_anthropic(t) for t in _TOOLS]
GROK_TOOLS:   list[dict] = [_to_openai(t) for t in _TOOLS]
LOCAL_TOOLS:  list[dict] = GROK_TOOLS  # Ollama uses OpenAI-compatible tool format


def get_tool_schema(name: str, fmt: str = "anthropic") -> dict | None:
    """Return the schema for a single named tool."""
    for t in _TOOLS:
        if t["name"] == name:
            return _to_anthropic(t) if fmt == "anthropic" else _to_openai(t)
    return None


TOOL_NAMES: set[str] = {t["name"] for t in _TOOLS}
