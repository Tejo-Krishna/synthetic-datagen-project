"""
Planner Agent — Rule-Based Scaffold Builder
Deterministic structural layer. No LLM involved.
Builds the step scaffold, detects clarification candidates,
and derives novelty hints from corpus memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    SampledToolChain,
    RegistryEndpointMetadata,
    CorpusSummary,
    PlanStep,
    ClarificationPoint,
    VALID_CONVERSATION_STYLES,
)


# ---------------------------------------------------------------------------
# Step scaffold
# ---------------------------------------------------------------------------

def build_step_scaffold(chain: SampledToolChain) -> list[PlanStep]:
    """
    Build a base list of PlanStep objects from the sampled chain.
    Structural fields are copied exactly; narrative fields are left as
    empty placeholders to be filled by the LLM narrative layer.
    """
    steps: list[PlanStep] = []
    for s in sorted(chain.steps, key=lambda x: x.step_index):
        steps.append(
            PlanStep(
                step_index=s.step_index,
                tool_id=s.tool_id,
                endpoint_id=s.endpoint_id,
                depends_on_steps=list(s.depends_on_steps),
                # Narrative placeholders — filled by LLM
                purpose="",
                user_intent="",
                assistant_intent="",
                expected_output_usage=None,
                may_require_clarification=False,
                clarification_reason=None,
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Clarification candidate detection
# ---------------------------------------------------------------------------

@dataclass
class ClarificationCandidate:
    step_index: int
    missing_or_ambiguous_fields: list[str]
    reason: str


def detect_clarification_candidates(
    chain: SampledToolChain,
    registry: dict[tuple[str, str], RegistryEndpointMetadata] | None,
) -> list[ClarificationCandidate]:
    """
    Identify steps where clarification is likely needed.

    Uses registry metadata when available:
    - Required parameters with source_hint="user" are likely missing from the
      user's initial request and may need clarification.
    - Parameters with source_hint="either" may need clarification depending
      on whether a prior step provides them.

    Degrades gracefully when registry is None or an endpoint is unlisted.
    """
    if not registry:
        return []

    candidates: list[ClarificationCandidate] = []
    prior_output_fields: set[str] = set()

    for step in sorted(chain.steps, key=lambda x: x.step_index):
        key = (step.tool_id, step.endpoint_id)
        meta = registry.get(key)
        if not meta:
            # Update prior outputs from optional sampler hints, then continue
            prior_output_fields.update(step.output_field_refs)
            continue

        ambiguous: list[str] = []

        for param in meta.parameters:
            if not param.required:
                continue

            hint = (param.source_hint or "").lower()

            if hint == "derived_from_previous_step":
                # Expected from a prior step — only flag if not available
                if param.name not in prior_output_fields:
                    ambiguous.append(param.name)

            elif hint == "user":
                # Must come from the user — always a clarification candidate
                ambiguous.append(param.name)

            elif hint == "either":
                # Available from prior step or user; flag only if not in prior outputs
                if param.name not in prior_output_fields:
                    ambiguous.append(param.name)

            else:
                # No hint — treat as potentially user-provided
                ambiguous.append(param.name)

        if ambiguous:
            candidates.append(
                ClarificationCandidate(
                    step_index=step.step_index,
                    missing_or_ambiguous_fields=ambiguous,
                    reason=(
                        f"Required parameter(s) {ambiguous} for endpoint "
                        f"'{step.endpoint_id}' may need user input."
                    ),
                )
            )

        # Track what this step produces for downstream steps
        for rf in meta.response_fields:
            prior_output_fields.add(rf.name)
        # Also track from sampler hints if available
        prior_output_fields.update(step.output_field_refs)

    return candidates


def build_clarification_points(
    candidates: list[ClarificationCandidate],
) -> list[ClarificationPoint]:
    """Convert raw ClarificationCandidates into ConversationPlan ClarificationPoints."""
    return [
        ClarificationPoint(
            before_step=c.step_index,
            reason=c.reason,
            missing_or_ambiguous_fields=c.missing_or_ambiguous_fields,
            question_goal=(
                f"Establish value(s) for: {', '.join(c.missing_or_ambiguous_fields)}."
            ),
        )
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# Diversity / novelty steering
# ---------------------------------------------------------------------------

@dataclass
class NoveltyHints:
    avoid_domains: list[str]
    avoid_tool_combos: list[frozenset[str]]
    avoid_styles: list[str]
    avoid_pattern_types: list[str]
    suggested_style: str | None


def derive_novelty_hints(summaries: list[CorpusSummary]) -> NoveltyHints:
    """
    Analyse prior corpus summaries to steer the planner away from duplicates.
    Returns hints the LLM narrative layer can use for diversification.
    """
    seen_domains: list[str] = []
    seen_combos: list[frozenset[str]] = []
    seen_styles: list[str] = []
    seen_patterns: list[str] = []

    for s in summaries:
        if s.domain:
            seen_domains.append(s.domain.lower())
        if s.tools:
            seen_combos.append(frozenset(t.lower() for t in s.tools))
        if s.conversation_style:
            seen_styles.append(s.conversation_style.lower())
        if s.pattern_type:
            seen_patterns.append(s.pattern_type.lower())

    # Suggest the least-used conversation style from the controlled enum
    style_counts = {style: seen_styles.count(style) for style in VALID_CONVERSATION_STYLES}
    suggested_style: str | None = min(style_counts, key=lambda k: style_counts[k])

    return NoveltyHints(
        avoid_domains=list(dict.fromkeys(seen_domains)),      # deduplicated, order-preserved
        avoid_tool_combos=list({frozenset(c) for c in seen_combos}),
        avoid_styles=list(dict.fromkeys(seen_styles)),
        avoid_pattern_types=list(dict.fromkeys(seen_patterns)),
        suggested_style=suggested_style,
    )
