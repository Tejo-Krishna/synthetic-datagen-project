"""
evaluator/repairer.py
---------------------
Surgical conversation repair using an LLM.

Repair strategy:
  1. Identify the lowest-scoring dimension from judge feedback.
  2. Ask the repair model to rewrite only the parts of the conversation
     responsible for that dimension's failure (surgical repair).
  3. Re-score the repaired conversation.
  4. If surgical repair fails after attempt 1, ask for a full rewrite of
     the dialogue while keeping the same tool sequence (fallback repair).
  5. After max_attempts total, mark the record as passed=False and keep it.

Why surgical before full-rewrite:
  Many failures are local — a bad final response, an awkward clarification
  turn, or one tool call with a hallucinated argument. A targeted fix is
  cheaper and preserves the rest of the conversation, which may be correct.
  Full rewrite is only used when the conversation is structurally broken
  end-to-end.

The repairer does NOT regenerate from the original SampledChain (that would
require the graph/sampler pipeline). It repairs the conversation text in-place
via LLM, keeping the tool sequence and metadata intact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from synthetic_datagen.evaluator.judge import JudgeClient, RawJudgeResult
from synthetic_datagen.evaluator.scorer import JudgeScores, ScoreValidator, attach_scores


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """
    Outcome of one repair cycle.

    Attributes:
        record:          The final record (repaired or original).
        scores:          Final judge scores after repair attempts.
        repair_attempts: Number of repair attempts made.
        repair_history:  List of (attempt, dimension_targeted, outcome) dicts.
        repaired:        True if at least one repair produced a passing record.
    """
    record: dict
    scores: JudgeScores
    repair_attempts: int
    repair_history: list[dict] = field(default_factory=list)
    repaired: bool = False


# ---------------------------------------------------------------------------
# Repair prompt builders
# ---------------------------------------------------------------------------

def _build_surgical_repair_prompt(
    record: dict,
    scores: JudgeScores,
    target_dimension: str,
) -> str:
    """
    Build a repair prompt targeting one failing dimension.

    Returns a user-facing prompt string for the repair LLM call.
    """
    dimension_guidance = {
        "tool_correctness": (
            "The conversation has poor tool correctness. "
            "Specifically: fix tool call arguments so they are grounded in prior step outputs "
            "(not hallucinated), ensure all required parameters are provided, and make sure "
            "the correct endpoints are called for the task."
        ),
        "task_completion": (
            "The conversation has poor task completion. "
            "Specifically: rewrite the final assistant message so it directly addresses the "
            "user's original request using concrete values from the tool outputs. "
            "Do not end with vague statements — reference actual results."
        ),
        "naturalness": (
            "The conversation has poor naturalness. "
            "Specifically: rewrite the clarification turns so they are contextually appropriate "
            "and the user replies logically follow from the assistant's questions. "
            "Make the dialogue feel like a real human–assistant exchange."
        ),
        "mean": (
            "The conversation scores low overall. "
            "Improve tool call grounding, ensure the task is resolved, "
            "and make the dialogue flow more naturally."
        ),
    }

    guidance = dimension_guidance.get(
        target_dimension,
        "Improve the overall quality of the conversation."
    )

    messages_json = json.dumps(record.get("messages", []), indent=2)
    tool_calls_json = json.dumps(record.get("tool_calls", []), indent=2)
    tool_outputs_json = json.dumps(record.get("tool_outputs", []), indent=2)

    return f"""\
You are repairing a synthetic AI assistant conversation for a training dataset.

JUDGE FEEDBACK:
  Dimension targeted: {target_dimension}
  Judge score:        {getattr(scores, target_dimension, 'N/A')} / 5.0
  Judge reasoning:    {scores.reasoning}

REPAIR INSTRUCTION:
{guidance}

TOOL CALLS (do not change endpoint names or sequence):
{tool_calls_json}

TOOL OUTPUTS (do not change these):
{tool_outputs_json}

ORIGINAL MESSAGES:
{messages_json}

Return a JSON object with one key "messages" containing the repaired messages array.
Keep the same roles and tool_calls structure. Only modify the content of text turns
to fix the identified issue. Do not change endpoint names or tool output values.

Example output format:
{{"messages": [
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "...", "tool_calls": [...]}},
  ...
]}}"""


def _build_full_rewrite_prompt(record: dict, scores: JudgeScores) -> str:
    """
    Build a full-rewrite repair prompt when surgical repair has failed.

    The tool sequence and outputs are preserved; only the dialogue text is rewritten.
    """
    tool_calls_json = json.dumps(record.get("tool_calls", []), indent=2)
    tool_outputs_json = json.dumps(record.get("tool_outputs", []), indent=2)

    return f"""\
You are rewriting a synthetic AI assistant conversation for a training dataset.
A previous targeted repair attempt failed. Please rewrite the full dialogue from scratch.

JUDGE FEEDBACK (all dimensions):
  tool_correctness: {scores.tool_correctness} / 5.0
  task_completion:  {scores.task_completion} / 5.0
  naturalness:      {scores.naturalness} / 5.0
  reasoning:        {scores.reasoning}

REQUIREMENTS:
- Keep exactly the same tool call sequence (endpoint names and order must not change).
- Keep exactly the same tool output values.
- Arguments in each tool call must be grounded in prior step outputs, not hallucinated.
- The final assistant message must reference real values from tool outputs.
- Clarification questions must be contextually appropriate.
- The conversation must feel natural and resolve the user's request.

TOOL SEQUENCE TO FOLLOW (do not reorder or add steps):
{tool_calls_json}

TOOL OUTPUTS (use these exact values in the dialogue):
{tool_outputs_json}

Return a JSON object with one key "messages" containing the full rewritten messages array.
Include all roles: user, assistant (with tool_calls where appropriate), and tool responses.

Example output format:
{{"messages": [
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "...", "tool_calls": [...]}},
  {{"role": "tool", "name": "...", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]}}"""


# ---------------------------------------------------------------------------
# Repairer
# ---------------------------------------------------------------------------

class ConversationRepairer:
    """
    Repairs conversations that fail quality thresholds.

    Uses the same judge client to re-score after each repair attempt.
    Uses a separate repair_model (defaults to the judge model) for rewriting.

    Args:
        judge_client:   The JudgeClient used for scoring.
        validator:      ScoreValidator with the same thresholds as evaluate.
        repair_model:   Claude model ID for repair calls (default: same as judge).
        max_attempts:   Maximum repair attempts before giving up (default 2).
        call_delay_s:   Sleep between API calls (default 0.5).
        api_key:        Anthropic API key (falls back to env var).
    """

    def __init__(
        self,
        judge_client: JudgeClient,
        validator: ScoreValidator,
        repair_model: str | None = None,
        max_attempts: int = 2,
        call_delay_s: float = 0.5,
        api_key: str | None = None,
    ):
        self.judge_client = judge_client
        self.validator = validator
        self.repair_model = repair_model or getattr(judge_client, "model", "claude-haiku-4-5-20251001")
        self.max_attempts = max_attempts
        self.call_delay_s = call_delay_s
        self._api_key = api_key
        self._anthropic_client: Any = None

    def repair(self, record: dict, initial_scores: JudgeScores) -> RepairResult:
        """
        Attempt to repair a failing conversation.

        Attempt 1: surgical — target the lowest-scoring dimension.
        Attempt 2: full rewrite — if surgical repair still fails.

        Always returns a RepairResult; never raises.
        """
        current_record = record
        current_scores = initial_scores
        repair_history: list[dict] = []

        for attempt in range(1, self.max_attempts + 1):
            is_final_attempt = (attempt == self.max_attempts)

            # Choose repair strategy
            if attempt == 1:
                target_dim = self.validator.lowest_scoring_dimension(current_scores)
                strategy = "surgical"
                prompt = _build_surgical_repair_prompt(
                    current_record, current_scores, target_dim or "mean"
                )
            else:
                target_dim = "all"
                strategy = "full_rewrite"
                prompt = _build_full_rewrite_prompt(current_record, current_scores)

            # Attempt repair
            repaired_messages, repair_error = self._call_repair_model(prompt)

            if repair_error or repaired_messages is None:
                repair_history.append({
                    "attempt": attempt,
                    "strategy": strategy,
                    "dimension_targeted": target_dim,
                    "outcome": "repair_call_failed",
                    "error": repair_error,
                })
                if is_final_attempt:
                    break
                continue

            # Build updated record with repaired messages
            repaired_record = dict(current_record)
            repaired_record["messages"] = repaired_messages

            # Re-score
            raw_result = self.judge_client.score(repaired_record)
            new_scores = self.validator.validate(raw_result)

            repair_history.append({
                "attempt": attempt,
                "strategy": strategy,
                "dimension_targeted": target_dim,
                "outcome": "passed" if new_scores.passed else "still_failing",
                "scores": {
                    "tool_correctness": new_scores.tool_correctness,
                    "task_completion": new_scores.task_completion,
                    "naturalness": new_scores.naturalness,
                    "mean_score": new_scores.mean_score,
                },
            })

            current_record = repaired_record
            current_scores = new_scores

            if new_scores.passed:
                return RepairResult(
                    record=attach_scores(current_record, current_scores),
                    scores=current_scores,
                    repair_attempts=attempt,
                    repair_history=repair_history,
                    repaired=True,
                )

        # All attempts exhausted — return original with failure metadata
        final_record = attach_scores(current_record, current_scores)
        final_record["metadata"] = dict(final_record.get("metadata", {}))
        final_record["metadata"]["repair_attempts"] = len(repair_history)
        final_record["metadata"]["repair_history"] = repair_history
        return RepairResult(
            record=final_record,
            scores=current_scores,
            repair_attempts=len(repair_history),
            repair_history=repair_history,
            repaired=False,
        )

    def _call_repair_model(self, prompt: str) -> tuple[list[dict] | None, str | None]:
        """
        Call the repair LLM and parse the repaired messages array.

        Returns (messages, None) on success, (None, error_str) on failure.
        """
        try:
            client = self._get_anthropic_client()
            response = client.messages.create(
                model=self.repair_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            time.sleep(self.call_delay_s)

            raw_text = response.content[0].text if response.content else ""
            messages = self._parse_messages_from_response(raw_text)
            return messages, None

        except Exception as e:
            return None, str(e)

    def _parse_messages_from_response(self, text: str) -> list[dict] | None:
        """
        Extract the messages array from the repair model's JSON response.

        Handles both clean JSON and JSON embedded in markdown code blocks.
        """
        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            inner = "\n".join(lines[1:])
            if inner.endswith("```"):
                inner = inner[:-3].rstrip()
            text = inner.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed, dict):
            return None
        messages = parsed.get("messages")
        if not isinstance(messages, list):
            return None
        # Basic sanity: each item should have a role
        if not all(isinstance(m, dict) and "role" in m for m in messages):
            return None
        return messages

    def _get_anthropic_client(self) -> Any:
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                ) from e
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._anthropic_client = anthropic.Anthropic(**kwargs)
        return self._anthropic_client
