"""
AlphaForge AI Harness — Configuration
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── API credentials (set via environment variables) ──────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
XAI_API_KEY: str       = os.environ.get("XAI_API_KEY", "")

# ── Model selection ───────────────────────────────────────────────────────────
# Claude: primary reasoning (strategy, code, review)
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Grok: market context, financial domain knowledge
# xAI API is OpenAI-compatible; override with XAI_MODEL env var
GROK_MODEL: str    = os.environ.get("GROK_MODEL", "grok-3")
XAI_BASE_URL: str  = "https://api.x.ai/v1"

# Local LLM via Ollama (OpenAI-compatible endpoint at localhost:11434)
# Install: https://ollama.com  then: ollama pull deepseek-r1:7b
# Set OLLAMA_MODEL env var to switch models, e.g. "deepseek-r1:14b", "qwen2.5:14b"
OLLAMA_MODEL: str    = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MAX_TOKENS: int = 4096

# ── Agent behaviour ───────────────────────────────────────────────────────────
CLAUDE_MAX_TOKENS: int = 8192
GROK_MAX_TOKENS: int   = 4096
MAX_TOOL_ROUNDS: int   = 8      # max back-and-forth tool call rounds per agent turn
CONTEXT_WINDOW: int    = 20     # max messages kept in each agent's rolling history

# ── Research loop ─────────────────────────────────────────────────────────────
DEFAULT_ITERATIONS: int   = 5
DEFAULT_TICKER: str       = "SPY"
DEFAULT_UNIVERSE: str     = "sp500"
PROMOTE_SHARPE_THRESHOLD  = 0.8   # minimum OOS Sharpe to promote a strategy
PROMOTE_DD_LIMIT          = 0.25  # maximum max-drawdown fraction to promote

# ── Paths ─────────────────────────────────────────────────────────────────────
HARNESS_DIR   = ROOT / "harness"
MEMORY_DIR    = HARNESS_DIR / "memory" / "data"
RESULTS_DIR   = ROOT / "logs" / "harness"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_keys(backend: str = "claude") -> list[str]:
    """Return list of missing API key names for the given backend (empty = all present)."""
    missing = []
    if backend in ("claude", "all") and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if backend in ("grok", "all") and not XAI_API_KEY:
        missing.append("XAI_API_KEY")
    # Ollama is local — no API key required
    return missing
