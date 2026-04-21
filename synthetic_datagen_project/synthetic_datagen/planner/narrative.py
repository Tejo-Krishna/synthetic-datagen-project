"""
Planner Agent — LLM Narrative Layer
Responsible ONLY for generating narrative fields.
Never touches structural fields (tool IDs, step indices, dependencies).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .models import (
    SampledToolChain,
    RegistryEndpointMetadata,
    CorpusSummary,
    PlanStep,
    ClarificationPoint,
    SummarySeedFields,
    VALID_CONVERSATION_STYLES,
)
from .scaffold import NoveltyHints


# ---------------------------------------------------------------------------
# Narrative request / response shapes
# ---------------------------------------------------------------------------

@dataclass
class NarrativeRequest:
    """Everything the LLM needs to generate narrative fields."""
    seed: int
    chain: SampledToolChain
    scaffold_steps: list[PlanStep]
    clarification_points: list[ClarificationPoint]
    novelty_hints: NoveltyHints
    registry: dict[tuple[str, str], RegistryEndpointMetadata] | None
    corpus_summaries: list[CorpusSummary]


@dataclass
class NarrativeOutput:
    """Raw LLM output before being merged into the ConversationPlan."""
    domain: str
    user_goal: str
    conversation_style: str
    style_notes: str
    # Per-step narrative, keyed by step_index
    step_narratives: dict[int, "StepNarrative"]


@dataclass
class StepNarrative:
    step_index: int
    purpose: str
    user_intent: str
    assistant_intent: str
    expected_output_usage: str | None
    may_require_clarification: bool
    clarification_reason: str | None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _format_corpus_summaries(summaries: list[CorpusSummary]) -> str:
    if not summaries:
        return "None."
    lines = []
    for i, s in enumerate(summaries, 1):
        lines.append(
            f"{i}. Domain: {s.domain or 'unknown'} | "
            f"Tools: {', '.join(s.tools) or 'unknown'} | "
            f"Style: {s.conversation_style or 'unknown'} | "
            f"Pattern: {s.pattern_type or 'unknown'}\n"
            f"   Summary: {s.content}"
        )
    return "\n".join(lines)


def _format_novelty_hints(hints: NoveltyHints) -> str:
    parts = []
    if hints.avoid_domains:
        parts.append(f"- Avoid these domains (already used): {hints.avoid_domains}")
    if hints.avoid_styles:
        parts.append(f"- Avoid these conversation styles (already used): {hints.avoid_styles}")
    if hints.avoid_pattern_types:
        parts.append(f"- Avoid these pattern types (already used): {hints.avoid_pattern_types}")
    if hints.suggested_style:
        parts.append(f"- Suggested conversation_style (least used so far): {hints.suggested_style}")
    return "\n".join(parts) if parts else "No diversity constraints — be creative."


def _format_steps(steps: list[PlanStep], registry: dict | None) -> str:
    lines = []
    for s in steps:
        reg_key = (s.tool_id, s.endpoint_id)
        endpoint_desc = ""
        params_info = ""
        if registry and reg_key in registry:
            meta = registry[reg_key]
            endpoint_desc = f" — {meta.description or meta.endpoint_name or ''}"
            if meta.parameters:
                param_names = [p.name for p in meta.parameters if p.required]
                opt_names = [p.name for p in meta.parameters if not p.required]
                if param_names:
                    params_info += f"\n    required params: {param_names}"
                if opt_names:
                    params_info += f"\n    optional params: {opt_names}"
            if meta.response_fields:
                field_names = sorted(f.name for f in meta.response_fields)
                params_info += f"\n    returns: {field_names[:8]}"
        lines.append(
            f"  Step {s.step_index}: {s.endpoint_id}{endpoint_desc}"
            f"{params_info}"
            f"\n    depends_on: {s.depends_on_steps}"
        )
    return "\n".join(lines)


def _format_clarification_points(cps: list[ClarificationPoint]) -> str:
    if not cps:
        return "None detected."
    lines = []
    for cp in cps:
        lines.append(
            f"  Before step {cp.before_step}: fields={cp.missing_or_ambiguous_fields}, "
            f"reason='{cp.reason}'"
        )
    return "\n".join(lines)


def build_narrative_prompt(req: NarrativeRequest) -> str:
    n_steps = len(req.scaffold_steps)
    return f"""You are generating narrative fields for a structured ConversationPlan.

SEED: {req.seed}
PATTERN TYPE: {req.chain.pattern_type or "unknown"}
DOMAIN HINT: {req.chain.domain_hint or "none"}
CONCEPT TAGS: {req.chain.concept_tags or []}

[Prior corpus conversations]
{_format_corpus_summaries(req.corpus_summaries)}

[Diversity guidance]
{_format_novelty_hints(req.novelty_hints)}

[Tool chain steps — DO NOT change these]
{_format_steps(req.scaffold_steps, req.registry)}

[Clarification candidates — mark relevant steps accordingly]
{_format_clarification_points(req.clarification_points)}

Your task:
Generate ONLY the narrative fields listed below for this conversation plan.
Do NOT alter the tool chain, step indices, dependencies, tool IDs, or endpoint IDs.

Conversation style MUST be one of:
{list(VALID_CONVERSATION_STYLES)}

Return ONLY valid JSON with this exact structure:
{{
  "domain": "...",
  "user_goal": "...",
  "conversation_style": "...",
  "style_notes": "...",
  "steps": [
    {{
      "step_index": 0,
      "purpose": "...",
      "user_intent": "...",
      "assistant_intent": "...",
      "expected_output_usage": "...",
      "may_require_clarification": false,
      "clarification_reason": null
    }}
  ]
}}

Rules:
- domain: a realistic real-world domain (e.g. "travel planning", "e-commerce", "healthcare scheduling")
- ORDERING IS FIXED AND AUTHORITATIVE: The step indices above define the exact execution order. You MUST write the user_goal and each step's purpose so that executing these tools in exactly this order — step 0 first, then step 1, etc. — is the natural, correct way to achieve the goal. Do NOT write a goal that implies a different order. If a tool appears early in the chain, find a reason why the user would naturally want that done first.
- user_goal: a SPECIFIC, realistic sentence describing the user's complete end-to-end goal that ONLY asks for features the listed tools can actually deliver (check required/optional params and returns above). Do NOT mention features not in the tool schemas (e.g. if create_event has no attendees param, don't ask the user to invite people; if a search returns a list, don't claim booking was done). The goal must be concrete enough that ALL {n_steps} steps are clearly necessary. Include real specifics: destinations, dates, IDs, or constraints that the tools actually accept.
- conversation_style: exactly one value from the enum above
- style_notes: one sentence describing how the user communicates in this scenario
- For each step: purpose describes what the step achieves; user_intent is what the user wants at that moment; assistant_intent is what the assistant will do; expected_output_usage explains how this step's returned fields (see "returns") feed into the NEXT step's required params — this is how the chain connects; null if last step
- Mark may_require_clarification=true only for steps where a required param is not present in the user_goal and cannot be inferred — if the user_goal contains enough context to infer the value, set may_require_clarification=false
- Be diverse: avoid near-duplicates of prior corpus conversations shown above
"""


# ---------------------------------------------------------------------------
# LLM call (pluggable backend)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, llm_backend: Any) -> str:
    """
    Call the LLM backend with the given prompt.
    The backend must implement: .complete(prompt: str) -> str
    This keeps the narrative layer decoupled from any specific LLM client.
    """
    return llm_backend.complete(prompt)


# DeterministicNarrativeBackend has been removed.
# All narrative generation requires a real LLM backend (AnthropicLLMBackend
# or any object implementing .complete(prompt: str) -> str).
# Passing None or a stub will cause PlannerAgent to raise PlannerConfigError.

class DeterministicNarrativeBackend:
    """
    Removed stub — kept only so existing imports don't break at module load time.
    Instantiating this class raises immediately. Pass an AnthropicLLMBackend
    (or any object with .complete(prompt: str) -> str) instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "DeterministicNarrativeBackend has been removed. "
            "Use AnthropicLLMBackend from synthetic_datagen.generator.llm_backend instead."
        )

    def complete(self, prompt: str) -> str:  # pragma: no cover
        raise RuntimeError("DeterministicNarrativeBackend has been removed.")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown fences if present, then return the JSON string."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return raw


def parse_narrative_response(raw: str, scaffold_steps: list[PlanStep]) -> NarrativeOutput:
    """
    Parse and validate the LLM's JSON response into a NarrativeOutput.
    Raises ValueError with a descriptive message if parsing fails.
    """
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM narrative response is not valid JSON: {e}\nRaw: {raw[:500]}")

    required_top = {"domain", "user_goal", "conversation_style", "style_notes", "steps"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(f"LLM narrative response missing top-level fields: {missing}")

    if data["conversation_style"] not in VALID_CONVERSATION_STYLES:
        raise ValueError(
            f"LLM returned invalid conversation_style '{data['conversation_style']}'. "
            f"Valid: {VALID_CONVERSATION_STYLES}"
        )

    scaffold_indices = {s.step_index for s in scaffold_steps}
    step_narratives: dict[int, StepNarrative] = {}

    for raw_step in data["steps"]:
        idx = raw_step.get("step_index")
        if idx is None:
            raise ValueError(f"Narrative step missing step_index: {raw_step}")
        if idx not in scaffold_indices:
            raise ValueError(
                f"Narrative step_index {idx} not in scaffold indices {scaffold_indices}."
            )
        for field in ("purpose", "user_intent", "assistant_intent"):
            if not raw_step.get(field, "").strip():
                raise ValueError(f"Narrative step {idx} has empty field '{field}'.")

        may_clarify = bool(raw_step.get("may_require_clarification", False))
        clarify_reason = raw_step.get("clarification_reason") or None
        if may_clarify and not clarify_reason:
            raise ValueError(
                f"Narrative step {idx} has may_require_clarification=True but no clarification_reason."
            )

        step_narratives[idx] = StepNarrative(
            step_index=idx,
            purpose=raw_step["purpose"].strip(),
            user_intent=raw_step["user_intent"].strip(),
            assistant_intent=raw_step["assistant_intent"].strip(),
            expected_output_usage=(raw_step.get("expected_output_usage") or "").strip() or None,
            may_require_clarification=may_clarify,
            clarification_reason=clarify_reason,
        )

    # Every scaffold step must have a narrative
    for s in scaffold_steps:
        if s.step_index not in step_narratives:
            raise ValueError(
                f"Narrative response is missing step_index {s.step_index}."
            )

    return NarrativeOutput(
        domain=data["domain"].strip(),
        user_goal=data["user_goal"].strip(),
        conversation_style=data["conversation_style"],
        style_notes=data["style_notes"].strip(),
        step_narratives=step_narratives,
    )


# ---------------------------------------------------------------------------
# Merge scaffold + narrative → enriched steps + summary seed fields
# ---------------------------------------------------------------------------

def merge_narrative_into_steps(
    scaffold_steps: list[PlanStep],
    narrative: NarrativeOutput,
    clarification_points: list[ClarificationPoint],
) -> list[PlanStep]:
    """
    Produce final PlanStep list by merging structural scaffold with LLM narrative.
    Structural fields are always taken from the scaffold — never from the LLM.
    """
    # Build a lookup: step_index -> ClarificationPoint reason text
    cp_reason_by_step: dict[int, str] = {
        cp.before_step: cp.reason for cp in clarification_points
    }
    enriched: list[PlanStep] = []

    for s in scaffold_steps:
        n = narrative.step_narratives[s.step_index]
        # Determine whether this step requires clarification.
        # Priority: scaffold flag > LLM flag > clarification detection hit
        cp_flagged = s.step_index in cp_reason_by_step
        may_clarify = s.may_require_clarification or n.may_require_clarification or cp_flagged

        # Populate clarification_reason — must be non-None whenever may_clarify is True.
        # Pull from scaffold, then LLM, then the ClarificationPoint reason text.
        clarify_reason = (
            s.clarification_reason
            or n.clarification_reason
            or (cp_reason_by_step.get(s.step_index) if cp_flagged else None)
        )

        enriched.append(
            PlanStep(
                # Structural — from scaffold only
                step_index=s.step_index,
                tool_id=s.tool_id,
                endpoint_id=s.endpoint_id,
                depends_on_steps=s.depends_on_steps,
                # Narrative — from LLM
                purpose=n.purpose,
                user_intent=n.user_intent,
                assistant_intent=n.assistant_intent,
                expected_output_usage=n.expected_output_usage,
                may_require_clarification=may_clarify,
                clarification_reason=clarify_reason,
            )
        )

    return enriched


def build_summary_seed_fields(
    domain: str,
    pattern_type: str,
    tools_used: list[str],
    conversation_style: str,
    clarification_points: list[ClarificationPoint],
) -> SummarySeedFields:
    return SummarySeedFields(
        domain=domain,
        pattern_type=pattern_type,
        tools_used=sorted(tools_used),
        conversation_style=conversation_style,
        planned_clarification_count=len(clarification_points),
    )
