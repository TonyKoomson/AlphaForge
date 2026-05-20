"""
AlphaForge AI Harness — Base Agent

Provides the shared foundation for all agents:
  - API client management (Anthropic for Claude, OpenAI-compat for Grok)
  - Rolling message history
  - Tool-use agentic loop (call → execute → feed result back → repeat)
  - Rich terminal output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from harness.config import (
    ANTHROPIC_API_KEY, XAI_API_KEY,
    CLAUDE_MODEL, GROK_MODEL, XAI_BASE_URL,
    CLAUDE_MAX_TOKENS, GROK_MAX_TOKENS,
    OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_MAX_TOKENS,
    MAX_TOOL_ROUNDS, CONTEXT_WINDOW,
)
from utils.helpers import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """
    Abstract base for all harness agents.

    Sub-classes set:
      self.name      — display name shown in UI
      self.backend   — "claude" or "grok"
      self.tools     — tool schemas in the correct format
      self.system    — system prompt string
    """

    def __init__(self, executor=None) -> None:
        self.name:    str  = "Agent"
        self.backend: str  = "claude"   # or "grok"
        self.tools:   list = []
        self.system:  str  = ""
        self._history: list[dict] = []
        self._executor = executor       # ToolExecutor (injected by orchestrator)

        # Lazy-initialise API clients
        self._claude_client = None
        self._grok_client   = None
        self._ollama_client = None

    # ── API clients ───────────────────────────────────────────────────────────

    @property
    def claude(self):
        if self._claude_client is None:
            try:
                from anthropic import Anthropic
                self._claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        return self._claude_client

    @property
    def grok(self):
        if self._grok_client is None:
            try:
                from openai import OpenAI
                self._grok_client = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._grok_client

    @property
    def ollama(self):
        """OpenAI-compatible client pointing at a local Ollama server."""
        if self._ollama_client is None:
            try:
                from openai import OpenAI
                # Ollama doesn't need a real API key; "ollama" is a placeholder
                self._ollama_client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._ollama_client

    # ── History management ────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})

    def _trim_history(self) -> None:
        if len(self._history) > CONTEXT_WINDOW:
            self._history = self._history[-CONTEXT_WINDOW:]

    def clear_history(self) -> None:
        self._history = []

    # ── Core call ─────────────────────────────────────────────────────────────

    def call(self, user_message: str, max_tool_rounds: int = MAX_TOOL_ROUNDS) -> str:
        """
        Send a message to the agent and return its final text response.
        Handles the full agentic tool-use loop internally.
        """
        self.add_user(user_message)

        if self.backend == "claude":
            return self._call_claude(max_tool_rounds)
        elif self.backend == "grok":
            return self._call_grok(max_tool_rounds)
        elif self.backend == "ollama":
            return self._call_ollama(max_tool_rounds)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ── Claude agentic loop ───────────────────────────────────────────────────

    def _call_claude(self, max_rounds: int) -> str:
        from anthropic.types import ToolUseBlock, TextBlock

        messages = list(self._history)

        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {
                "model":      CLAUDE_MODEL,
                "max_tokens": CLAUDE_MAX_TOKENS,
                "system":     self.system,
                "messages":   messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools

            resp = self.claude.messages.create(**kwargs)

            # Collect text and tool calls
            text_parts:  list[str]  = []
            tool_calls:  list[dict] = []

            for block in resp.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append({
                        "id":    block.id,
                        "name":  block.name,
                        "input": block.input,
                    })

            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": resp.content})

            if not tool_calls or resp.stop_reason == "end_turn":
                final_text = "\n".join(text_parts)
                self.add_assistant(final_text)
                return final_text

            # Execute tools and feed results back
            tool_results = []
            for tc in tool_calls:
                self._log_tool_call(tc["name"], tc["input"])
                result_str = self._execute_tool(tc["name"], tc["input"])
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tc["id"],
                    "content":     result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        # Max rounds reached — return whatever text we have
        self.add_assistant("[max tool rounds reached]")
        return "[max tool rounds reached]"

    # ── Grok (OpenAI-compat) agentic loop ────────────────────────────────────

    def _call_grok(self, max_rounds: int) -> str:
        # Build messages for OpenAI format (system as first message)
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self._history)

        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {
                "model":      GROK_MODEL,
                "max_tokens": GROK_MAX_TOKENS,
                "messages":   messages,
            }
            if self.tools:
                kwargs["tools"] = self.tools

            resp = self.grok.chat.completions.create(**kwargs)
            msg  = resp.choices[0].message

            # If no tool calls, return text
            if not msg.tool_calls:
                text = msg.content or ""
                messages.append({"role": "assistant", "content": text})
                self.add_assistant(text)
                return text

            # Add assistant message with tool calls
            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            # Execute tools and feed results
            for tc in msg.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                self._log_tool_call(tc.function.name, tool_input)
                result_str = self._execute_tool(tc.function.name, tool_input)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

        self.add_assistant("[max tool rounds reached]")
        return "[max tool rounds reached]"

    # ── Ollama (local LLM) agentic loop ──────────────────────────────────────

    def _call_ollama(self, max_rounds: int) -> str:
        """
        Call a locally-running Ollama model via its OpenAI-compatible API.

        Supports tool calling for models that implement it (e.g. qwen2.5, mistral-nemo).
        DeepSeek-R1 has limited tool support — if no tool calls come back the loop
        still returns the text response, so it degrades gracefully.
        """
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self._history)

        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {
                "model":      OLLAMA_MODEL,
                "max_tokens": OLLAMA_MAX_TOKENS,
                "messages":   messages,
            }
            # Only pass tools if the model supports function calling
            if self.tools:
                kwargs["tools"] = self.tools

            try:
                resp = self.ollama.chat.completions.create(**kwargs)
            except Exception as exc:
                err = str(exc)
                # Some models reject tools — retry without them
                if "tool" in err.lower() and "tools" in kwargs:
                    kwargs.pop("tools")
                    resp = self.ollama.chat.completions.create(**kwargs)
                else:
                    raise

            msg = resp.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                messages.append({"role": "assistant", "content": text})
                self.add_assistant(text)
                return text

            # Tool calls — same protocol as Grok
            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                self._log_tool_call(tc.function.name, tool_input)
                result_str = self._execute_tool(tc.function.name, tool_input)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

        self.add_assistant("[max tool rounds reached]")
        return "[max tool rounds reached]"

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, input_: dict) -> str:
        if self._executor is None:
            return json.dumps({"error": "No executor attached to agent"})
        return self._executor.execute(name, input_)

    def _log_tool_call(self, name: str, input_: dict) -> None:
        """Log tool calls to console (overridden by orchestrator for rich display)."""
        short = json.dumps(input_, default=str)[:120]
        logger.debug("[%s] tool_call: %s(%s)", self.name, name, short)
