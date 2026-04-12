"""
evaluator/scorer.py
-------------------
Score parsing, gated pass/fail validation, and record attachment.

Gated pass rule (all conditions must hold):
  - tool_correctness >= THRESHOLD_TOOL_CORRECTNESS  (default 3.5)
  - task_completion  >= THRESHOLD_TASK_COMPLETION   (default 3.5)
  - naturalness      >= THRESHOLD_NATURALNESS        (default 3.0)
  - mean score       >= threshold                    (default 3.5)

Why gated rather than mean-only:
  A conversation with broken tool calls (tool_correctness=1.0) and perfect
  naturalness (5.0) averages to 3.0 but is useless as training data.
  The hard floors on tool_correctness and task_completion prevent this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthetic_datagen.evaluator.judge import RawJudgeResult


# ---------------------------------------------------------------------------
# Thresholds (module-level constants — easy to override in tests)
# ---------------------------------------------------------------------------

THRESHOLD_TOOL_CORRECTNESS: float = 3.5
THRESHOLD_TASK_COMPLETION: float = 3.5
THRESHOLD_NATURALNESS: float = 3.0
THRESHOLD_MEAN: float = 3.5


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class JudgeScores:
    """
    Parsed and validated scores for one conversation.

    Attributes:
        tool_correctness:  1.0–5.0, correctness of tool calls and argument grounding.
        task_completion:   1.0–5.0, how fully the user's request was resolved.
        naturalness:       1.0–5.0, naturalness of conversation flow.
        mean_score:        Simple average of the three dimensions.
        reasoning:         Judge's brief justification.
        judge_model:       Model ID that produced the scores.
        scored_at:         ISO timestamp of scoring.
        passed:            True if all gated thresholds are met.
        failed_gates:      List of dimension names that failed their threshold.
        error:             Set if judging failed; scores will be None.
        judge_error_type:  parse_failure | schema_validation_failure | provider_error
    """
    tool_correctness: float | None
    task_completion: float | None
    naturalness: float | None
    mean_score: float | None
    reasoning: str
    judge_model: str
    scored_at: str
    passed: bool
    failed_gates: list[str] = field(default_factory=list)
    error: str | None = None
    judge_error_type: str | None = None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ScoreValidator:
    """
    Applies the gated pass/fail rule to a RawJudgeResult.

    Instantiate once and reuse — holds no mutable state.
    """

    def __init__(
        self,
        threshold_tool_correctness: float = THRESHOLD_TOOL_CORRECTNESS,
        threshold_task_completion: float = THRESHOLD_TASK_COMPLETION,
        threshold_naturalness: float = THRESHOLD_NATURALNESS,
        threshold_mean: float = THRESHOLD_MEAN,
    ):
        self.threshold_tool_correctness = threshold_tool_correctness
        self.threshold_task_completion = threshold_task_completion
        self.threshold_naturalness = threshold_naturalness
        self.threshold_mean = threshold_mean

    def validate(self, raw: RawJudgeResult) -> JudgeScores:
        """Convert a RawJudgeResult into a validated JudgeScores."""
        # Judge call itself failed
        if raw.error or raw.tool_correctness is None:
            return JudgeScores(
                tool_correctness=None,
                task_completion=None,
                naturalness=None,
                mean_score=None,
                reasoning=raw.reasoning,
                judge_model=raw.judge_model,
                scored_at=raw.scored_at,
                passed=False,
                failed_gates=["judge_error"],
                error=raw.error,
                judge_error_type=raw.judge_error_type,
            )

        tc = raw.tool_correctness
        comp = raw.task_completion
        nat = raw.naturalness
        mean = (tc + comp + nat) / 3.0

        failed_gates: list[str] = []
        if tc < self.threshold_tool_correctness:
            failed_gates.append("tool_correctness")
        if comp < self.threshold_task_completion:
            failed_gates.append("task_completion")
        if nat < self.threshold_naturalness:
            failed_gates.append("naturalness")
        if mean < self.threshold_mean:
            failed_gates.append("mean")

        return JudgeScores(
            tool_correctness=tc,
            task_completion=comp,
            naturalness=nat,
            mean_score=round(mean, 4),
            reasoning=raw.reasoning,
            judge_model=raw.judge_model,
            scored_at=raw.scored_at,
            passed=len(failed_gates) == 0,
            failed_gates=failed_gates,
        )

    def lowest_scoring_dimension(self, scores: JudgeScores) -> str | None:
        """
        Return the name of the dimension with the lowest score.
        Used by ConversationRepairer to target surgical repair.
        Returns None if scores are unavailable.
        """
        if scores.tool_correctness is None:
            return None
        dims = {
            "tool_correctness": scores.tool_correctness,
            "task_completion": scores.task_completion,
            "naturalness": scores.naturalness,
        }
        return min(dims, key=dims.get)


# ---------------------------------------------------------------------------
# Record attachment
# ---------------------------------------------------------------------------

def attach_scores(record: dict, scores: JudgeScores) -> dict:
    """
    Return a new record dict with judge_scores and passed fields attached.

    Does not mutate the original record.
    """
    updated = dict(record)
    updated["judge_scores"] = {
        "tool_correctness": scores.tool_correctness,
        "task_completion": scores.task_completion,
        "naturalness": scores.naturalness,
        "mean_score": scores.mean_score,
        "reasoning": scores.reasoning,
        "judge_model": scores.judge_model,
        "scored_at": scores.scored_at,
        "passed": scores.passed,
        "failed_gates": scores.failed_gates,
        "error": scores.error,
        "judge_error_type": scores.judge_error_type,
    }
    # Top-level passed flag for easy filtering
    updated["passed"] = scores.passed
    return updated
