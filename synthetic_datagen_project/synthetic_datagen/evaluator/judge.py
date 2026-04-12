"""
evaluator/judge.py
------------------
Provider-abstracted LLM judge client.

The judge scores each conversation on three dimensions using structured output
(Claude tool use) so that downstream parsing is reliable.

Architecture:
  - JudgeClient          — abstract base; swap provider by subclassing
  - AnthropicJudgeClient — Claude-backed implementation (default)
  - JudgePromptBuilder   — assembles system + user prompt from a conversation record

Structured output strategy:
  Claude's tool use API is used with tool_choice forced to the "submit_scores"
  tool. This guarantees the response is always a valid JSON object matching the
  score schema — no free-text parsing required.

Rate-limit / retry policy:
  - 0.5 s sleep between successful calls (configurable via call_delay_s)
  - Up to max_retries attempts on parse or provider error
  - Distinguishes judge_error_type: parse_failure | schema_validation_failure | provider_error
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RawJudgeResult:
    """Raw output from one judge call before validation."""
    tool_correctness: float | None = None
    task_completion: float | None = None
    naturalness: float | None = None
    reasoning: str = ""
    judge_model: str = ""
    scored_at: str = ""
    error: str | None = None
    judge_error_type: str | None = None  # parse_failure | schema_validation_failure | provider_error


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class JudgePromptBuilder:
    """Builds the system + user prompt for the LLM judge."""

    SYSTEM = (
        "You are an expert evaluator of AI assistant conversations that involve tool use. "
        "Your task is to score the quality of a conversation where an AI assistant uses "
        "tools to complete a user's request. Be objective and consistent. "
        "Use the submit_scores tool to return your evaluation."
    )

    SCORE_RUBRIC = """\
Score the conversation on three dimensions (1.0 – 5.0):

1. tool_correctness — Did the assistant call the right tools with valid arguments?
   Were arguments grounded in prior step outputs (not hallucinated)?
   Were all required parameters provided?
   5 = All calls correct, arguments chain from prior outputs, required params present.
   3 = Most calls correct; minor hallucinated arguments.
   1 = Wrong tools, required params missing, or IDs not carried from prior outputs.

2. task_completion — Did the conversation resolve the user's original request?
   5 = Final response directly addresses the request with real values from tool outputs.
   3 = Partial completion; some steps accomplished but request not fully resolved.
   1 = Conversation ends without resolving the user's request.

3. naturalness — Does the conversation flow like a real human–assistant exchange?
   Are clarification questions appropriate and well-timed?
   5 = Natural flow; clarifications sensible; user replies follow from questions.
   3 = Slightly robotic but functional.
   1 = Clarification questions are random; user replies do not follow from questions.
"""

    def build(self, record: dict) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for this conversation record."""
        user_prompt = self._format_user_prompt(record)
        return self.SYSTEM, user_prompt

    def _format_user_prompt(self, record: dict) -> str:
        tool_schemas_text = self._format_tool_schemas(record)
        conversation_text = self._format_conversation(record)

        return (
            f"Evaluate the following conversation.\n\n"
            f"AVAILABLE TOOLS:\n{tool_schemas_text}\n\n"
            f"CONVERSATION:\n{conversation_text}\n\n"
            f"{self.SCORE_RUBRIC}\n"
            f"Call the submit_scores tool with your evaluation."
        )

    def _format_tool_schemas(self, record: dict) -> str:
        """Format tool schemas from tool_calls in the record."""
        seen: set[str] = set()
        lines: list[str] = []
        for tc in record.get("tool_calls", []):
            name = tc.get("name", "unknown")
            if name in seen:
                continue
            seen.add(name)
            params = tc.get("parameters", {})
            param_keys = ", ".join(params.keys()) if params else "(none)"
            lines.append(f"  - {name}  params: [{param_keys}]")
        return "\n".join(lines) if lines else "  (no tools used)"

    def _format_conversation(self, record: dict) -> str:
        """Format messages into a readable conversation transcript."""
        lines: list[str] = []
        for msg in record.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    tc_parts.append(
                        f"{tc.get('name', '?')}({tc.get('parameters', {})})"
                    )
                lines.append(f"[{role}] <tool_calls> {' | '.join(tc_parts)}")
                if content:
                    lines.append(f"         {content}")
            else:
                # Truncate very long tool outputs to keep prompt manageable
                display = content[:500] + "..." if len(content) > 500 else content
                lines.append(f"[{role}] {display}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class JudgeClient(abc.ABC):
    """Abstract judge client. Implement score() to add a new provider."""

    @abc.abstractmethod
    def score(self, record: dict) -> RawJudgeResult:
        """Score one conversation record. Returns RawJudgeResult."""


# ---------------------------------------------------------------------------
# Anthropic (Claude) implementation
# ---------------------------------------------------------------------------

# Tool schema for structured output — the model MUST call this tool.
_SUBMIT_SCORES_TOOL = {
    "name": "submit_scores",
    "description": (
        "Submit quality scores for the evaluated conversation. "
        "Call this tool with your assessment of the three dimensions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_correctness": {
                "type": "number",
                "description": (
                    "Score 1.0–5.0: correctness of tool selection and argument grounding. "
                    "5=all correct and grounded, 1=wrong tools or hallucinated args."
                ),
            },
            "task_completion": {
                "type": "number",
                "description": (
                    "Score 1.0–5.0: how completely the user's request was resolved. "
                    "5=fully resolved with real values, 1=request unresolved."
                ),
            },
            "naturalness": {
                "type": "number",
                "description": (
                    "Score 1.0–5.0: naturalness of conversation flow and clarifications. "
                    "5=natural human-assistant flow, 1=robotic or incoherent."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief justification for the scores (2–4 sentences).",
            },
        },
        "required": ["tool_correctness", "task_completion", "naturalness", "reasoning"],
    },
}


class AnthropicJudgeClient(JudgeClient):
    """
    Claude-backed judge using tool use for guaranteed structured output.

    Uses tool_choice={"type": "tool", "name": "submit_scores"} to force
    the model to always return a parseable score object.

    Args:
        model:         Claude model ID to use for judging.
        max_retries:   Retry attempts on failure (default 3).
        call_delay_s:  Sleep between successful calls to avoid rate limits (default 0.5).
        api_key:       Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        call_delay_s: float = 0.5,
        api_key: str | None = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.max_retries = max_retries
        self.call_delay_s = call_delay_s
        self._api_key = api_key
        self._client: Any = None  # lazy init
        self._prompt_builder = JudgePromptBuilder()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                ) from e

            # Resolve API key: constructor arg > env var > .env file
            import os
            from pathlib import Path

            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                # Try to load from .env in the project root
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("ANTHROPIC_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break

            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def score(self, record: dict) -> RawJudgeResult:
        """Score one conversation. Retries up to max_retries on failure."""
        import datetime

        system_prompt, user_prompt = self._prompt_builder.build(record)
        last_error: str = ""
        last_error_type: str = "provider_error"

        for attempt in range(self.max_retries):
            try:
                result = self._call_api(system_prompt, user_prompt)
                time.sleep(self.call_delay_s)
                return result
            except _ParseFailure as e:
                last_error = str(e)
                last_error_type = "parse_failure"
            except _SchemaValidationFailure as e:
                last_error = str(e)
                last_error_type = "schema_validation_failure"
            except Exception as e:
                last_error = str(e)
                last_error_type = "provider_error"

            if attempt < self.max_retries - 1:
                time.sleep(self.call_delay_s * (attempt + 1))  # backoff

        # All retries exhausted
        return RawJudgeResult(
            judge_model=self.model,
            scored_at=datetime.datetime.utcnow().isoformat(),
            error=last_error,
            judge_error_type=last_error_type,
        )

    def _call_api(self, system_prompt: str, user_prompt: str) -> RawJudgeResult:
        """Single API call. Raises _ParseFailure or _SchemaValidationFailure on bad output."""
        import datetime

        client = self._get_client()

        response = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system_prompt,
            tools=[_SUBMIT_SCORES_TOOL],
            tool_choice={"type": "tool", "name": "submit_scores"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract tool use block
        tool_input = self._extract_tool_input(response)
        scores = self._validate_schema(tool_input)

        return RawJudgeResult(
            tool_correctness=scores["tool_correctness"],
            task_completion=scores["task_completion"],
            naturalness=scores["naturalness"],
            reasoning=scores.get("reasoning", ""),
            judge_model=self.model,
            scored_at=datetime.datetime.utcnow().isoformat(),
        )

    def _extract_tool_input(self, response: Any) -> dict:
        """Pull the tool_use input dict out of the API response."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_scores":
                return block.input
        raise _ParseFailure(
            f"No submit_scores tool_use block in response. "
            f"stop_reason={response.stop_reason}"
        )

    def _validate_schema(self, tool_input: dict) -> dict:
        """Validate that all required score fields are present and in range."""
        required = ["tool_correctness", "task_completion", "naturalness"]
        for field_name in required:
            if field_name not in tool_input:
                raise _SchemaValidationFailure(f"Missing field: {field_name}")
            val = tool_input[field_name]
            if not isinstance(val, (int, float)):
                raise _SchemaValidationFailure(
                    f"Field {field_name} must be numeric, got {type(val)}"
                )
            # Clamp to valid range rather than rejecting — avoids needless retries
            tool_input[field_name] = max(1.0, min(5.0, float(val)))
        return tool_input


# ---------------------------------------------------------------------------
# Internal exceptions (not part of public API)
# ---------------------------------------------------------------------------

class _ParseFailure(Exception):
    """Raised when the API response cannot be parsed as a tool call."""


class _SchemaValidationFailure(Exception):
    """Raised when tool_input fields are missing or out of range."""
