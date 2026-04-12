"""
planner/__init__.py
--------------------
Public API surface for the planner package.

Two distinct planning layers live in this package:

  1. Legacy planner (planner.py)
     - PlannerAgent        → converts SampledChain into ConversationPlan (generator-facing)
     - ConversationPlan    → turn-based plan consumed by assistant.py, user_proxy.py
     - TurnPlan            → one turn inside a ConversationPlan
     Imported by: generator/assistant.py, generator/user_proxy.py, cli/main.py

  2. Structured planner (agent.py + models.py + scaffold.py + narrative.py + validator.py)
     - StructuredPlannerAgent      → converts SampledToolChain into StructuredConversationPlan
     - StructuredConversationPlan  → rich plan with PlanStep, ClarificationPoint, SummarySeedFields
     - PlannerResult               → typed result envelope returned to orchestrator
     Imported by: orchestrator, future downstream agents

  3. Shared utilities
     - registry_adapter   → SampledChain → SampledToolChain, ToolRegistry → planner registry
     - config             → PlannerConfig, load_planner_config
"""

# ---------------------------------------------------------------------------
# Legacy planner — generator-facing (keep these imports stable)
# ---------------------------------------------------------------------------
from .planner import (
    PlannerAgent,
    ConversationPlan,
    TurnPlan,
)

# ---------------------------------------------------------------------------
# Structured planner — orchestrator-facing
# ---------------------------------------------------------------------------
from .agent import PlannerAgent as StructuredPlannerAgent
from .models import (
    StructuredConversationPlan,
    PlannerResult,
    PlanStep,
    ClarificationPoint,
    SummarySeedFields,
    SampledToolChain,
    SampledStep,
    CorpusSummary,
    VALID_CONVERSATION_STYLES,
    RESAMPLE_ON,
    HARD_STOP_ON,
    # Error types
    PlannerError,
    PlannerInvalidInputError,
    PlannerChainTooShortError,
    PlannerOutputValidationError,
    PlannerRetryExhaustedError,
    PlannerConfigError,
    PlannerDependencyError,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from .config import PlannerConfig, load_planner_config

# ---------------------------------------------------------------------------
# Registry adapter — boundary between common types and planner internals
# ---------------------------------------------------------------------------
from .registry_adapter import (
    adapt_sampled_chain,
    adapt_sampled_chain_safe,
    build_planner_registry,
    validate_adaptation,
    ChainAdaptationError,
)

__all__ = [
    # Legacy planner
    "PlannerAgent",
    "ConversationPlan",
    "TurnPlan",
    # Structured planner
    "StructuredPlannerAgent",
    "StructuredConversationPlan",
    "PlannerResult",
    "PlanStep",
    "ClarificationPoint",
    "SummarySeedFields",
    "SampledToolChain",
    "SampledStep",
    "CorpusSummary",
    "VALID_CONVERSATION_STYLES",
    "RESAMPLE_ON",
    "HARD_STOP_ON",
    # Errors
    "PlannerError",
    "PlannerInvalidInputError",
    "PlannerChainTooShortError",
    "PlannerOutputValidationError",
    "PlannerRetryExhaustedError",
    "PlannerConfigError",
    "PlannerDependencyError",
    # Config
    "PlannerConfig",
    "load_planner_config",
    # Adapter
    "adapt_sampled_chain",
    "adapt_sampled_chain_safe",
    "build_planner_registry",
    "validate_adaptation",
    "ChainAdaptationError",
]
