"""
common/types.py
---------------
Shared runtime dataclasses used across sampler, planner, executor, and validator.

Ownership rules:
  - Sampler creates ClarificationStep(reason="missing_required_param", ...)
  - Planner may add ClarificationStep(reason="intent_ambiguity", missing_params=[])
  - ClarificationStep lives here so both components can use it without circular imports.

No Pydantic here — plain stdlib dataclasses only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Field-level data flow between endpoints
# ---------------------------------------------------------------------------

@dataclass
class FieldMapping:
    """
    Structural declaration that a response field from one endpoint
    can fill a parameter of the next endpoint.

    These are schema-level linkages, not concrete runtime values.
    Concrete values are resolved only at executor time.
    """
    source_field: str   # field name in from_endpoint's response output
    target_param: str   # parameter name in to_endpoint's input schema


# ---------------------------------------------------------------------------
# Transitions between endpoints in a chain
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """
    One step-to-step link in a sampled chain.

    Carries structural information needed by the Planner, Executor, and Validator.
    Does NOT carry:
      - edge weight (stays on ProjectedEdge, used only during sampling)
      - provenance_path (stays on ProjectedEdge, used only for debugging/artifacts)
    """
    from_endpoint: str
    to_endpoint: str
    edge_type: str                        # "data_link" | "semantic" | "category"
    field_mappings: list[FieldMapping]    # explicit source->target field flows
    matched_concepts: list[str]           # concept nodes matched in heterogeneous graph
    is_executable: bool = True            # False if semantic/category with unsourced required params


# ---------------------------------------------------------------------------
# Parallel branch support
# ---------------------------------------------------------------------------

@dataclass
class ParallelBranch:
    """
    One branch in a parallel sampling pattern.

    Used when pattern_type == "parallel". The Planner serializes branches
    naturally into conversation turns — it does not need to re-infer
    branch membership from a flat endpoint list.
    """
    branch_id: str
    endpoint_ids: list[str]
    transitions: list[Transition]


# ---------------------------------------------------------------------------
# Clarification step metadata
# ---------------------------------------------------------------------------

@dataclass
class ClarificationStep:
    """
    Marks a step in the chain that requires clarification before proceeding.

    Two reasons:
      - "missing_required_param": detected by Sampler from schema analysis
      - "intent_ambiguity": injected by Planner from goal framing analysis

    For "intent_ambiguity", missing_params is always [].
    """
    step_index: int
    reason: Literal["missing_required_param", "intent_ambiguity"]
    missing_params: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sampled chain — the structural skeleton for one conversation
# ---------------------------------------------------------------------------

@dataclass
class SampledChain:
    """
    The output of the SamplerAgent. Represents the structural backbone
    of one synthetic conversation.

    Sequential chains:
      - root_endpoint_id is None
      - branches is None
      - merge_endpoint_id is None
      - endpoint_ids and transitions carry the full chain

    Parallel chains:
      - root_endpoint_id is the shared entry point
      - branches holds two ParallelBranch objects
      - merge_endpoint_id is the convergence endpoint
      - endpoint_ids holds the flattened all-endpoints view for metrics/dedup

    The Planner consumes branches directly — it does not re-infer
    branch structure from pattern_type alone.
    """
    # Core chain content
    endpoint_ids: list[str]               # flattened view, always present, used for metrics/dedup
    tool_ids: list[str]                   # deduplicated ordered tool names
    transitions: list[Transition]         # sequential transitions or cross-branch merge links
    pattern_type: str                     # "sequential" | "multi_tool" | "clarification_first" | "parallel"
    sampling_mode: str                    # how the chain was generated (may differ from pattern_type)

    # Clarification metadata
    clarification_steps: list[ClarificationStep] = field(default_factory=list)

    # Parallel-only fields (None for sequential chains)
    root_endpoint_id: str | None = None
    branches: list[ParallelBranch] | None = None
    merge_endpoint_id: str | None = None

    # ---------------------------------------------------------------------------
    # Convenience properties
    # ---------------------------------------------------------------------------

    @property
    def requires_clarification(self) -> bool:
        """True if any step in this chain needs clarification."""
        return len(self.clarification_steps) > 0

    @property
    def num_clarification_questions(self) -> int:
        """Count of clarification steps — used in output metadata."""
        return len(self.clarification_steps)

    @property
    def is_parallel(self) -> bool:
        return self.pattern_type == "parallel" and self.branches is not None

    @property
    def num_distinct_tools(self) -> int:
        return len(set(self.tool_ids))


# ---------------------------------------------------------------------------
# ConversationState — shared communication object between all generator agents
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """
    Shared state object passed between all generator agents.

    Replaces loose function arguments as the inter-agent communication
    protocol. Each agent reads from and writes to this object:

      Planner    → writes: plan
      UserProxy  → writes: messages (user turns)
      Assistant  → writes: messages (assistant turns + tool_calls)
      Executor   → writes: tool_outputs, session, grounded_steps, non_first_steps
      Validator  → reads:  messages, tool_calls, tool_outputs, metadata

    Grounding warnings are accumulated here so they can be surfaced
    at the end of generation rather than silently swallowed.
    """
    conversation_id: str

    # Set by Planner
    plan: object = None                        # StructuredConversationPlan

    # Accumulated by Generator agents
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    tool_outputs: list = field(default_factory=list)

    # Set by Executor
    session: object = None                     # SessionState

    # Grounding tracking
    clarification_count: int = 0
    grounded_steps: int = 0
    non_first_steps: int = 0

    # Inline grounding warnings (populated by grounding check)
    grounding_warnings: list = field(default_factory=list)

    @property
    def memory_grounding_rate(self) -> float | None:
        if self.non_first_steps == 0:
            return None
        return self.grounded_steps / self.non_first_steps

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for artifact output."""
        return {
            "endpoint_ids": self.endpoint_ids,
            "tool_ids": self.tool_ids,
            "pattern_type": self.pattern_type,
            "sampling_mode": self.sampling_mode,
            "requires_clarification": self.requires_clarification,
            "num_clarification_questions": self.num_clarification_questions,
            "clarification_steps": [
                {
                    "step_index": cs.step_index,
                    "reason": cs.reason,
                    "missing_params": cs.missing_params,
                }
                for cs in self.clarification_steps
            ],
            "transitions": [
                {
                    "from_endpoint": t.from_endpoint,
                    "to_endpoint": t.to_endpoint,
                    "edge_type": t.edge_type,
                    "is_executable": t.is_executable,
                    "field_mappings": [
                        {"source_field": fm.source_field, "target_param": fm.target_param}
                        for fm in t.field_mappings
                    ],
                    "matched_concepts": t.matched_concepts,
                }
                for t in self.transitions
            ],
            "root_endpoint_id": self.root_endpoint_id,
            "merge_endpoint_id": self.merge_endpoint_id,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "endpoint_ids": b.endpoint_ids,
                    "transitions": [
                        {
                            "from_endpoint": t.from_endpoint,
                            "to_endpoint": t.to_endpoint,
                            "edge_type": t.edge_type,
                            "is_executable": t.is_executable,
                            "field_mappings": [
                                {"source_field": fm.source_field, "target_param": fm.target_param}
                                for fm in t.field_mappings
                            ],
                            "matched_concepts": t.matched_concepts,
                        }
                        for t in b.transitions
                    ],
                }
                for b in (self.branches or [])
            ] or None,
        }
