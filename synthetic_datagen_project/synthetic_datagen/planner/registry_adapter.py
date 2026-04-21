"""
planner/registry_adapter.py
----------------------------
The single boundary layer between project-wide types and planner internals.

This is the ONLY file that is allowed to import from both:
  - synthetic_datagen.common.types   (project-wide canonical schema)
  - synthetic_datagen.planner.models (planner-internal schema)

No other file in the planner package should import from common.types directly,
and no file outside the planner package should import from planner.models directly.

Responsibilities:
  A. adapt_sampled_chain()     — SampledChain -> SampledToolChain
  B. build_planner_registry()  — ToolRegistry -> planner registry dict
  C. Field normalization, defaults, dependency inference, validation

Why this exists:
  SampledChain (common/types.py) is the project-wide sampler output.
  SampledToolChain (planner/models.py) is the planner's internal contract.
  They differ in structure:
    - SampledChain uses endpoint_ids (flat list) + transitions (edges)
    - SampledToolChain uses steps (indexed) with depends_on_steps (explicit deps)
  This adapter converts between the two cleanly without polluting either schema.

Future compatibility:
  If SampledChain gains new fields, add them here.
  If the planner's internal contract changes, update only this file.
  Neither schema needs to know about the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Project-wide canonical types (sampler output)
from synthetic_datagen.common.types import (
    SampledChain,
    Transition,
    ClarificationStep,
    ParallelBranch,
    FieldMapping,
)

# Planner-internal types
from synthetic_datagen.planner.models import (
    SampledToolChain,
    SampledStep,
    RegistryEndpointMetadata,
    RegistryParameterMetadata,
    RegistryResponseField,
)

if TYPE_CHECKING:
    from synthetic_datagen.graph.registry import ToolRegistry
    from synthetic_datagen.planner.models import ClarificationPoint


# ---------------------------------------------------------------------------
# User-provided parameter names
# These mirror sampler/config.py user_natural_params and are used to infer
# source_hint="user" for registry parameters.
# Kept in sync manually — or load from SamplerConfig if available.
# ---------------------------------------------------------------------------

_USER_NATURAL_PARAMS: frozenset[str] = frozenset({
    "query", "city", "date", "location", "origin", "destination",
    "source", "target", "language", "country", "keyword", "term",
    "text", "name", "category", "type", "from_date", "to_date",
    "start_date", "end_date",
})


# ---------------------------------------------------------------------------
# A. SampledChain → SampledToolChain
# ---------------------------------------------------------------------------

class ChainAdaptationError(ValueError):
    """Raised when a SampledChain cannot be converted to SampledToolChain."""


def adapt_sampled_chain(
    chain: SampledChain,
    chain_id: str,
    seed: int,
) -> SampledToolChain:
    """
    Convert a SampledChain (common/types.py) into a SampledToolChain
    (planner/models.py) that the PlannerAgent can consume.

    Args:
        chain:    SampledChain produced by SamplerAgent
        chain_id: stable identifier assigned by the orchestrator
                  (SampledChain has no chain_id field — orchestrator owns it)
        seed:     reproducibility seed assigned by the orchestrator

    Returns:
        SampledToolChain with:
          - one SampledStep per endpoint_id
          - depends_on_steps derived from Transition.from_endpoint ordering
          - optional field refs populated from Transition.field_mappings
          - concept_tags from matched_concepts across all transitions
          - domain_hint from the first tool_id (best-effort)

    Raises:
        ChainAdaptationError: if endpoint_ids is empty or chain_id/seed invalid
    """
    if not chain_id or not chain_id.strip():
        raise ChainAdaptationError("chain_id must not be empty.")
    if not isinstance(seed, int):
        raise ChainAdaptationError(f"seed must be an int, got {type(seed).__name__}.")
    if not chain.endpoint_ids:
        raise ChainAdaptationError("SampledChain has no endpoint_ids — cannot adapt.")

    # Build a lookup: endpoint_id -> index in the flat endpoint list
    endpoint_index: dict[str, int] = {
        eid: idx for idx, eid in enumerate(chain.endpoint_ids)
    }

    # Build a lookup: endpoint_id -> list of step indices it depends on
    # A step depends on any earlier step that transitions INTO it.
    depends_on: dict[str, list[int]] = {eid: [] for eid in chain.endpoint_ids}
    for t in chain.transitions:
        if t.to_endpoint in endpoint_index and t.from_endpoint in endpoint_index:
            from_idx = endpoint_index[t.from_endpoint]
            to_idx   = endpoint_index[t.to_endpoint]
            if from_idx < to_idx:
                depends_on[t.to_endpoint].append(from_idx)

    # Build a lookup: endpoint_id -> output field refs (from transition field_mappings)
    # These are fields this endpoint produces that later steps consume.
    output_refs: dict[str, list[str]] = {eid: [] for eid in chain.endpoint_ids}
    input_refs:  dict[str, list[str]] = {eid: [] for eid in chain.endpoint_ids}
    for t in chain.transitions:
        for fm in t.field_mappings:
            if t.from_endpoint in output_refs:
                if fm.source_field not in output_refs[t.from_endpoint]:
                    output_refs[t.from_endpoint].append(fm.source_field)
            if t.to_endpoint in input_refs:
                if fm.target_param not in input_refs[t.to_endpoint]:
                    input_refs[t.to_endpoint].append(fm.target_param)

    # Collect concept_tags from all transitions
    concept_tags: list[str] = []
    seen_concepts: set[str] = set()
    for t in chain.transitions:
        for concept in t.matched_concepts:
            if concept not in seen_concepts:
                concept_tags.append(concept)
                seen_concepts.add(concept)

    # Determine domain_hint from all tool_ids so the narrative layer can
    # derive a coherent domain instead of just seeing the first tool.
    domain_hint: str | None = ", ".join(chain.tool_ids) if chain.tool_ids else None

    # Determine role_in_chain for each endpoint
    def _role(idx: int, total: int, eid: str) -> str:
        if idx == 0:
            return "entry"
        if idx == total - 1:
            return "terminal"
        # Check if this is a merge point (parallel chains)
        if chain.merge_endpoint_id and eid == chain.merge_endpoint_id:
            return "merge"
        # Check if this is a branch root
        if chain.root_endpoint_id and eid == chain.root_endpoint_id:
            return "branch_root"
        return "intermediate"

    total = len(chain.endpoint_ids)

    # Assemble SampledStep list
    steps: list[SampledStep] = []
    for idx, endpoint_id in enumerate(chain.endpoint_ids):
        # endpoint_id format: "tool_name::endpoint_name"
        parts = endpoint_id.split("::", 1)
        tool_id      = parts[0] if len(parts) == 2 else endpoint_id
        endpoint_name = parts[1] if len(parts) == 2 else endpoint_id

        step = SampledStep(
            step_index=idx,
            tool_id=tool_id,
            endpoint_id=endpoint_id,
            depends_on_steps=sorted(set(depends_on[endpoint_id])),
            tool_name=tool_id,
            endpoint_name=endpoint_name,
            input_parameter_refs=input_refs[endpoint_id],
            output_field_refs=output_refs[endpoint_id],
            role_in_chain=_role(idx, total, endpoint_id),
        )
        steps.append(step)

    return SampledToolChain(
        chain_id=chain_id,
        seed=seed,
        pattern_type=chain.pattern_type,
        steps=steps,
        domain_hint=domain_hint,
        concept_tags=concept_tags,
    )


def adapt_sampled_chain_safe(
    chain: SampledChain,
    chain_id: str,
    seed: int,
) -> tuple[SampledToolChain | None, str | None]:
    """
    Non-raising version of adapt_sampled_chain().
    Returns (SampledToolChain, None) on success or (None, error_message) on failure.
    Useful for orchestrator loops that handle errors without try/except noise.
    """
    try:
        return adapt_sampled_chain(chain, chain_id, seed), None
    except ChainAdaptationError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# B. ToolRegistry → planner registry dict
# ---------------------------------------------------------------------------

def build_planner_registry(
    tool_registry: "ToolRegistry",
    user_natural_params: frozenset[str] | None = None,
) -> dict[tuple[str, str], RegistryEndpointMetadata]:
    """
    Convert a ToolRegistry (graph/registry.py) into the planner's registry
    format: dict[(tool_id, endpoint_id) -> RegistryEndpointMetadata].

    The planner uses this dict to:
      - detect clarification candidates (which params need user input)
      - enrich narrative prompts with endpoint descriptions
      - score dependency chains for argument filling

    Args:
        tool_registry:       ToolRegistry from graph/registry.py
        user_natural_params: set of parameter names that typically come from
                             the user (e.g. "query", "location"). Defaults to
                             the project-wide _USER_NATURAL_PARAMS set.
                             Pass sampler_config.user_natural_params for full alignment.

    Returns:
        dict keyed by (tool_id, endpoint_id) — the exact format PlannerAgent expects.
    """
    natural_params = user_natural_params or _USER_NATURAL_PARAMS
    result: dict[tuple[str, str], RegistryEndpointMetadata] = {}

    for endpoint_id, endpoint in tool_registry.endpoints_by_id.items():
        tool_id = endpoint.tool_name

        # Convert NormalizedParameter -> RegistryParameterMetadata
        parameters: list[RegistryParameterMetadata] = []
        for p in endpoint.parameters:
            source_hint = _infer_source_hint(
                param_name=p.name,
                required=p.required,
                natural_params=natural_params,
                endpoint_returns_fields=endpoint.returns_fields,
            )
            parameters.append(RegistryParameterMetadata(
                name=p.name,
                required=p.required,
                type=p.type,
                description=p.description,
                source_hint=source_hint,
            ))

        # Convert returns_fields -> RegistryResponseField list
        response_fields: list[RegistryResponseField] = [
            RegistryResponseField(
                name=field_name,
                type=endpoint.returns_types.get(field_name),
                description=None,  # not available in current registry schema
            )
            for field_name in sorted(endpoint.returns_fields)
        ]

        key = (tool_id, endpoint_id)
        result[key] = RegistryEndpointMetadata(
            tool_id=tool_id,
            endpoint_id=endpoint_id,
            tool_name=endpoint.tool_name,
            endpoint_name=endpoint.name,
            description=endpoint.description,
            parameters=parameters,
            response_fields=response_fields,
        )

    return result


def _infer_source_hint(
    param_name: str,
    required: bool,
    natural_params: frozenset[str],
    endpoint_returns_fields: set[str],
) -> str:
    """
    Infer the source_hint for a parameter:
      "user"                     — param name is in the natural/user-provided set
      "derived_from_previous_step" — param name matches a field a prior endpoint returns
      "either"                   — could come from either source
      (None is used when truly unknown — returned as "either" here as safest default)

    Logic:
      1. If the param name is in user_natural_params → "user"
      2. If the param name matches a known response field from any endpoint → "derived_from_previous_step"
      3. Otherwise → "either" (safest default for clarification detection)
    """
    name_lower = param_name.lower()

    if name_lower in natural_params:
        return "user"

    # Check if this param name appears as a response field of any endpoint
    # (cross-endpoint field matching — rough heuristic but effective)
    if param_name in endpoint_returns_fields:
        return "derived_from_previous_step"

    # Structural suffixes that strongly suggest derivation
    derived_suffixes = ("_id", "_key", "_token", "_ref", "_code", "_handle", "_uuid")
    if any(name_lower.endswith(s) for s in derived_suffixes):
        return "derived_from_previous_step"

    return "either"


# ---------------------------------------------------------------------------
# C. Validation helpers
# ---------------------------------------------------------------------------

def validate_adaptation(adapted: SampledToolChain, original: SampledChain) -> list[str]:
    """
    Validate that an adapted SampledToolChain faithfully represents
    the original SampledChain. Returns a list of error strings (empty = valid).

    Checks:
      - step count matches endpoint count
      - all original endpoint_ids are present
      - step indices are contiguous from 0
      - dependencies only point to earlier steps
      - pattern_type is preserved
    """
    errors: list[str] = []

    if len(adapted.steps) != len(original.endpoint_ids):
        errors.append(
            f"Step count mismatch: adapted has {len(adapted.steps)} steps, "
            f"original has {len(original.endpoint_ids)} endpoint_ids."
        )

    adapted_endpoint_ids = {s.endpoint_id for s in adapted.steps}
    for eid in original.endpoint_ids:
        if eid not in adapted_endpoint_ids:
            errors.append(f"Original endpoint_id '{eid}' missing from adapted steps.")

    indices = [s.step_index for s in adapted.steps]
    if sorted(indices) != list(range(len(adapted.steps))):
        errors.append(f"Adapted step indices are not contiguous: {sorted(indices)}")

    index_set = set(indices)
    for step in adapted.steps:
        for dep in step.depends_on_steps:
            if dep not in index_set:
                errors.append(
                    f"Step {step.step_index} depends on step {dep} which does not exist."
                )
            elif dep >= step.step_index:
                errors.append(
                    f"Step {step.step_index} depends on step {dep} which is not earlier."
                )

    if adapted.pattern_type != original.pattern_type:
        errors.append(
            f"pattern_type mismatch: adapted='{adapted.pattern_type}', "
            f"original='{original.pattern_type}'."
        )

    return errors


# ---------------------------------------------------------------------------
# D. Clarification bridge — one-way, explicit
# ---------------------------------------------------------------------------
# ClarificationPoint (planner internal) → ClarificationStep (common/types.py)
#
# This is the ONLY direction supported. Downstream generator code
# (assistant.py, user_proxy.py) depends on ClarificationStep from
# common/types.py and must never import planner-internal types.
# This function is the single crossing point — keep it here.

def clarification_point_to_step(cp: "ClarificationPoint") -> ClarificationStep:
    """
    Convert a planner-internal ClarificationPoint into a ClarificationStep
    that the generator layer (assistant.py, user_proxy.py) can consume.

    Mapping:
        ClarificationPoint.before_step             → ClarificationStep.step_index
        ClarificationPoint.missing_or_ambiguous_fields → ClarificationStep.missing_params
        ClarificationPoint.reason                  → mapped to Literal reason:
            any reason text → "missing_required_param" (default)
            if "ambiguity" in reason text → "intent_ambiguity"

    Args:
        cp: ClarificationPoint from planner/models.py

    Returns:
        ClarificationStep from common/types.py — safe for use in generator layer
    """
    reason_text = (cp.reason or "").lower()
    if "ambiguity" in reason_text or "intent" in reason_text:
        reason: str = "intent_ambiguity"
        missing_params: list[str] = []
    else:
        reason = "missing_required_param"
        missing_params = list(cp.missing_or_ambiguous_fields)

    return ClarificationStep(
        step_index=cp.before_step,
        reason=reason,  # type: ignore[arg-type]
        missing_params=missing_params,
    )


def clarification_points_to_steps(
    cps: list["ClarificationPoint"],
) -> list[ClarificationStep]:
    """
    Batch convert a list of ClarificationPoints to ClarificationSteps.
    Preserves order. Safe to call with an empty list.
    """
    return [clarification_point_to_step(cp) for cp in cps]
