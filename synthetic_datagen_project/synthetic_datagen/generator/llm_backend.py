"""
generator/llm_backend.py
------------------------
Shared LLM backend for all generator and planner components.

Implements the .complete(prompt) -> str interface used by:
  - PlannerAgent (narrative layer) — replaces DeterministicNarrativeBackend
  - UserProxyAgent — generates user utterances
  - AssistantAgent — generates assistant utterances
  - OfflineExecutor — generates realistic tool outputs
  - Legacy PlannerAgent — generates user goals
"""

from __future__ import annotations

import os


class AnthropicLLMBackend:
    """
    Anthropic Claude backend. Implements .complete(prompt) -> str.

    All generator and planner components accept any object with this
    interface, so this class can be swapped for a test stub without
    touching any other file.

    Args:
        model:       Claude model ID. Defaults to claude-haiku-4-5-20251001.
        temperature: Sampling temperature. 0.7 balances variety and consistency.
        api_key:     Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        max_tokens:  Max tokens per completion. 2048 is the safe minimum for
                     narrative JSON (multi-step chains can exceed 1024 tokens).
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            ) from exc

        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = __import__("anthropic").Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def complete(self, prompt: str) -> str:
        """
        Send a single-turn prompt and return the response text.
        Matches the interface expected by narrative.call_llm() and all
        generator components.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
