"""
evaluator/
----------
LLM-as-judge quality evaluation pipeline.

Public API:
  - JudgeClient          — abstract provider interface
  - AnthropicJudgeClient — Claude-backed judge using tool use for structured output
  - JudgeScores          — parsed score dataclass
  - ScoreValidator       — gated pass/fail rule
  - ConversationRepairer — surgical + fallback repair logic
  - EvaluationReport     — aggregated results
  - generate_report      — build report from scored records
"""

from synthetic_datagen.evaluator.judge import JudgeClient, AnthropicJudgeClient
from synthetic_datagen.evaluator.scorer import JudgeScores, ScoreValidator, attach_scores
from synthetic_datagen.evaluator.repairer import ConversationRepairer
from synthetic_datagen.evaluator.report import EvaluationReport, generate_report

__all__ = [
    "JudgeClient",
    "AnthropicJudgeClient",
    "JudgeScores",
    "ScoreValidator",
    "attach_scores",
    "ConversationRepairer",
    "EvaluationReport",
    "generate_report",
]
