"""
Planner Agent — Self-Validation
Rule-based checks run after every StructuredConversationPlan is generated,
before it is returned to the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    StructuredConversationPlan,
    SampledToolChain,
    VALID_CONVERSATION_STYLES,
    PlannerOutputValidationError,
)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


def validate_sampled_tool_chain(chain: SampledToolChain, min_steps: int = 1) -> ValidationResult:
    """
    Validate the SampledToolChain before planning begins.
    Failures here warrant an immediate resample, not a retry.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not chain.steps:
        errors.append("SampledToolChain has no steps.")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    if len(chain.steps) < min_steps:
        errors.append(
            f"Chain has {len(chain.steps)} step(s); minimum required is {min_steps}."
        )

    # Step indices must be contiguous starting from 0
    indices = [s.step_index for s in chain.steps]
    expected = list(range(len(chain.steps)))
    if sorted(indices) != expected:
        errors.append(
            f"Step indices are not contiguous. Got {sorted(indices)}, expected {expected}."
        )

    # Dependencies must point only to earlier steps
    index_set = set(indices)
    for step in chain.steps:
        for dep in step.depends_on_steps:
            if dep not in index_set:
                errors.append(
                    f"Step {step.step_index} depends on step {dep}, which does not exist."
                )
            elif dep >= step.step_index:
                errors.append(
                    f"Step {step.step_index} depends on step {dep}, which is not earlier."
                )

    # Tool and endpoint IDs must be non-empty strings
    for step in chain.steps:
        if not step.tool_id or not step.tool_id.strip():
            errors.append(f"Step {step.step_index} has an empty tool_id.")
        if not step.endpoint_id or not step.endpoint_id.strip():
            errors.append(f"Step {step.step_index} has an empty endpoint_id.")

    if not chain.chain_id or not chain.chain_id.strip():
        errors.append("SampledToolChain has an empty chain_id.")

    if not chain.pattern_type or not chain.pattern_type.strip():
        warnings.append("SampledToolChain has no pattern_type; planner will infer.")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_conversation_plan(
    plan: StructuredConversationPlan,
    source_chain: SampledToolChain,
    min_steps: int = 1,
    min_distinct_tools: int = 1,
) -> ValidationResult:
    """
    Validate a generated StructuredConversationPlan against structural rules.
    Failures here warrant a retry (and eventual resample if retries exhausted).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- ID consistency ---
    if plan.chain_id != source_chain.chain_id:
        errors.append(
            f"plan.chain_id '{plan.chain_id}' does not match "
            f"source chain_id '{source_chain.chain_id}'."
        )

    if plan.seed != source_chain.seed:
        errors.append(
            f"plan.seed {plan.seed} does not match source seed {source_chain.seed}."
        )

    if not plan.plan_id or not plan.plan_id.strip():
        errors.append("plan.plan_id is empty; it must be injected by the orchestrator.")

    # --- Step count ---
    if len(plan.steps) < min_steps:
        errors.append(
            f"Plan has {len(plan.steps)} step(s); minimum required is {min_steps}."
        )

    # --- All sampled steps must be preserved ---
    sampled_pairs = {(s.tool_id, s.endpoint_id, s.step_index) for s in source_chain.steps}
    plan_pairs = {(s.tool_id, s.endpoint_id, s.step_index) for s in plan.steps}
    missing = sampled_pairs - plan_pairs
    if missing:
        errors.append(f"Plan is missing sampled steps: {missing}.")

    # --- Step indices must be contiguous ---
    plan_indices = [s.step_index for s in plan.steps]
    expected = list(range(len(plan.steps)))
    if sorted(plan_indices) != expected:
        errors.append(
            f"Plan step indices are not contiguous. "
            f"Got {sorted(plan_indices)}, expected {expected}."
        )

    # --- Dependencies must point only to earlier steps ---
    plan_index_set = set(plan_indices)
    for step in plan.steps:
        for dep in step.depends_on_steps:
            if dep not in plan_index_set:
                errors.append(
                    f"Plan step {step.step_index} depends on step {dep}, which does not exist."
                )
            elif dep >= step.step_index:
                errors.append(
                    f"Plan step {step.step_index} depends on step {dep}, which is not earlier."
                )

    # --- Clarification flag consistency ---
    for step in plan.steps:
        if step.may_require_clarification and not step.clarification_reason:
            errors.append(
                f"Step {step.step_index} is marked may_require_clarification=True "
                f"but has no clarification_reason."
            )

    # --- Clarification points must reference valid steps ---
    for cp in plan.clarification_points:
        if cp.before_step not in plan_index_set:
            errors.append(
                f"ClarificationPoint references before_step={cp.before_step}, "
                f"which is not a valid step index."
            )
        if not cp.missing_or_ambiguous_fields:
            warnings.append(
                f"ClarificationPoint before_step={cp.before_step} has no "
                f"missing_or_ambiguous_fields listed."
            )

    # --- conversation_style must be from controlled enum ---
    if plan.conversation_style not in VALID_CONVERSATION_STYLES:
        errors.append(
            f"conversation_style '{plan.conversation_style}' is not in the controlled enum. "
            f"Valid values: {VALID_CONVERSATION_STYLES}."
        )

    # --- tools_used must match distinct tool_ids in steps ---
    distinct_tools_in_steps = sorted({s.tool_id for s in plan.steps})
    tools_used_sorted = sorted(plan.tools_used)
    if tools_used_sorted != distinct_tools_in_steps:
        errors.append(
            f"tools_used {tools_used_sorted} does not match distinct tools in steps "
            f"{distinct_tools_in_steps}."
        )

    if len(distinct_tools_in_steps) < min_distinct_tools:
        warnings.append(
            f"Plan uses only {len(distinct_tools_in_steps)} distinct tool(s); "
            f"consider sampling a richer chain."
        )

    # --- summary_seed_fields consistency ---
    ssf = plan.summary_seed_fields
    if ssf.domain != plan.domain:
        errors.append(
            f"summary_seed_fields.domain '{ssf.domain}' does not match plan.domain '{plan.domain}'."
        )
    if ssf.pattern_type != plan.pattern_type:
        errors.append(
            f"summary_seed_fields.pattern_type '{ssf.pattern_type}' does not match "
            f"plan.pattern_type '{plan.pattern_type}'."
        )
    if sorted(ssf.tools_used) != tools_used_sorted:
        errors.append(
            f"summary_seed_fields.tools_used {sorted(ssf.tools_used)} does not match "
            f"plan.tools_used {tools_used_sorted}."
        )
    if ssf.conversation_style != plan.conversation_style:
        errors.append(
            f"summary_seed_fields.conversation_style '{ssf.conversation_style}' does not match "
            f"plan.conversation_style '{plan.conversation_style}'."
        )

    # --- Required narrative fields must be non-empty ---
    for field_name, value in [
        ("domain", plan.domain),
        ("user_goal", plan.user_goal),
        ("pattern_type", plan.pattern_type),
        ("style_notes", plan.style_notes),
    ]:
        if not value or not value.strip():
            errors.append(f"plan.{field_name} is empty.")

    for step in plan.steps:
        for field_name, value in [
            ("purpose", step.purpose),
            ("user_intent", step.user_intent),
            ("assistant_intent", step.assistant_intent),
        ]:
            if not value or not value.strip():
                errors.append(f"Step {step.step_index}.{field_name} is empty.")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def raise_if_invalid(result: ValidationResult, context: str = "") -> None:
    """Raise PlannerOutputValidationError if the ValidationResult is not valid."""
    if not result.valid:
        prefix = f"[{context}] " if context else ""
        joined = "; ".join(result.errors)
        raise PlannerOutputValidationError(f"{prefix}Validation failed: {joined}")
