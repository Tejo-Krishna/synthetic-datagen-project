"""
Planner Agent — Data Models
All dataclasses representing the contracts between the Sampler, Planner,
and downstream agents (User-proxy, Assistant, Validator, Orchestrator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

ConversationStyleEnum = Literal[
    "direct",
    "exploratory",
    "underspecified",
    "preference_driven",
    "correction_heavy",
    "comparison_oriented",
    "goal_driven",
]

VALID_CONVERSATION_STYLES: tuple[str, ...] = (
    "direct",
    "exploratory",
    "underspecified",
    "preference_driven",
    "correction_heavy",
    "comparison_oriented",
    "goal_driven",
)


# ---------------------------------------------------------------------------
# Upstream contract: SampledToolChain (produced by Sampler Agent)
# ---------------------------------------------------------------------------

@dataclass
class SampledStep:
    step_index: int
    tool_id: str
    endpoint_id: str
    depends_on_steps: list[int] = field(default_factory=list)
    # Optional enrichment fields from sampler
    tool_name: Optional[str] = None
    endpoint_name: Optional[str] = None
    input_parameter_refs: list[str] = field(default_factory=list)
    output_field_refs: list[str] = field(default_factory=list)
    role_in_chain: Optional[str] = None


@dataclass
class SampledToolChain:
    chain_id: str
    seed: int
    pattern_type: str
    steps: list[SampledStep]
    # Optional enrichment fields
    domain_hint: Optional[str] = None
    concept_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry metadata contract (planner-facing, minimal)
# ---------------------------------------------------------------------------

@dataclass
class RegistryParameterMetadata:
    name: str
    required: bool
    type: Optional[str] = None
    description: Optional[str] = None
    # "user" | "derived_from_previous_step" | "either"
    source_hint: Optional[str] = None


@dataclass
class RegistryResponseField:
    name: str
    type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class RegistryEndpointMetadata:
    tool_id: str
    endpoint_id: str
    tool_name: Optional[str] = None
    endpoint_name: Optional[str] = None
    description: Optional[str] = None
    parameters: list[RegistryParameterMetadata] = field(default_factory=list)
    response_fields: list[RegistryResponseField] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Corpus memory contract (retrieved before planning)
# ---------------------------------------------------------------------------

@dataclass
class CorpusSummary:
    content: str                          # human-readable summary text
    conversation_id: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    pattern_type: Optional[str] = None
    domain: Optional[str] = None
    conversation_style: Optional[str] = None


# ---------------------------------------------------------------------------
# StructuredConversationPlan — Planner output / downstream contract
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    step_index: int
    tool_id: str
    endpoint_id: str
    depends_on_steps: list[int]
    purpose: str
    user_intent: str
    assistant_intent: str
    expected_output_usage: Optional[str]
    may_require_clarification: bool
    clarification_reason: Optional[str]


@dataclass
class ClarificationPoint:
    before_step: int
    reason: str
    missing_or_ambiguous_fields: list[str]
    question_goal: str


@dataclass
class SummarySeedFields:
    """
    Draft metadata for post-validation corpus writeback.
    This is NOT the final corpus summary — the orchestrator builds that
    from the actual validated conversation after execution.
    """
    domain: str
    pattern_type: str
    tools_used: list[str]
    conversation_style: str
    planned_clarification_count: int


@dataclass
class StructuredConversationPlan:
    # Orchestrator-injected IDs (echoed, never invented by planner)
    plan_id: str
    chain_id: str
    seed: int
    # Narrative fields (LLM-generated)
    domain: str
    user_goal: str
    # Structural fields
    pattern_type: str
    conversation_style: ConversationStyleEnum
    style_notes: str                          # top-level, alongside conversation_style
    tools_used: list[str]
    steps: list[PlanStep]
    clarification_points: list[ClarificationPoint]
    # Draft fields for post-validation corpus writeback
    summary_seed_fields: SummarySeedFields


# ---------------------------------------------------------------------------
# Planner result envelope (returned to orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class PlannerResult:
    success: bool
    plan: Optional[StructuredConversationPlan]
    retrieved_corpus_summaries: list[CorpusSummary]
    num_retries_used: int
    error_code: Optional[str]
    error_message: Optional[str]


# ---------------------------------------------------------------------------
# Typed planner errors
# ---------------------------------------------------------------------------

class PlannerError(Exception):
    """Base class for all planner errors."""
    error_code: str = "PLANNER_ERROR"


class PlannerInvalidInputError(PlannerError):
    """
    Sampled tool chain is malformed, empty, or structurally invalid.
    Orchestrator should resample.
    """
    error_code = "PLANNER_INVALID_INPUT"


class PlannerChainTooShortError(PlannerError):
    """
    Sampled chain has fewer steps than the configured minimum.
    Orchestrator should resample.
    """
    error_code = "PLANNER_CHAIN_TOO_SHORT"


class PlannerOutputValidationError(PlannerError):
    """
    Generated StructuredConversationPlan failed self-validation.
    Orchestrator should retry, then resample if retries exhausted.
    """
    error_code = "PLANNER_OUTPUT_VALIDATION"


class PlannerRetryExhaustedError(PlannerError):
    """
    All retry attempts failed. Orchestrator should resample the chain.
    """
    error_code = "PLANNER_RETRY_EXHAUSTED"


class PlannerConfigError(PlannerError):
    """
    Planner is misconfigured (missing required dependency, bad settings).
    Orchestrator should hard stop.
    """
    error_code = "PLANNER_CONFIG_ERROR"


class PlannerDependencyError(PlannerError):
    """
    A required external dependency (registry, memory backend) is
    unavailable and strict mode is enabled.
    Orchestrator should hard stop.
    """
    error_code = "PLANNER_DEPENDENCY_ERROR"


# ---------------------------------------------------------------------------
# Orchestrator retry/resample policy table
# ---------------------------------------------------------------------------

RESAMPLE_ON: frozenset[str] = frozenset({
    "PLANNER_INVALID_INPUT",
    "PLANNER_CHAIN_TOO_SHORT",
    "PLANNER_RETRY_EXHAUSTED",
    "PLANNER_OUTPUT_VALIDATION",
})

HARD_STOP_ON: frozenset[str] = frozenset({
    "PLANNER_CONFIG_ERROR",
    "PLANNER_DEPENDENCY_ERROR",
})
