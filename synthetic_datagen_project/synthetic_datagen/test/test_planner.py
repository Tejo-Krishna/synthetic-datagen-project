"""
Planner Agent — Tests
Covers: dataclasses, validator, scaffold, narrative parser, and PlannerAgent end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planner.models import (
    SampledStep,
    SampledToolChain,
    RegistryParameterMetadata,
    RegistryEndpointMetadata,
    CorpusSummary,
    PlanStep,
    ClarificationPoint,
    SummarySeedFields,
    StructuredConversationPlan,
    PlannerResult,
    PlannerInvalidInputError,
    PlannerChainTooShortError,
    PlannerOutputValidationError,
    VALID_CONVERSATION_STYLES,
    RESAMPLE_ON,
    HARD_STOP_ON,
)
from planner.validator import (
    validate_sampled_tool_chain,
    validate_conversation_plan,
    raise_if_invalid,
)
from planner.scaffold import (
    build_step_scaffold,
    detect_clarification_candidates,
    build_clarification_points,
    derive_novelty_hints,
)
from planner.narrative import parse_narrative_response, build_summary_seed_fields
from planner.agent import PlannerAgent, PlannerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chain(n_steps: int = 3, pattern: str = "sequential") -> SampledToolChain:
    steps = []
    for i in range(n_steps):
        steps.append(SampledStep(
            step_index=i,
            tool_id=f"tool_{chr(65 + i)}",       # tool_A, tool_B, tool_C ...
            endpoint_id=f"ep_{i}",
            depends_on_steps=list(range(i)),       # each step depends on all prior
        ))
    return SampledToolChain(
        chain_id="chain_test_001",
        seed=42,
        pattern_type=pattern,
        steps=steps,
    )


def make_registry(chain: SampledToolChain) -> dict:
    registry = {}
    for s in chain.steps:
        registry[(s.tool_id, s.endpoint_id)] = RegistryEndpointMetadata(
            tool_id=s.tool_id,
            endpoint_id=s.endpoint_id,
            description=f"Endpoint {s.endpoint_id}",
            parameters=[
                RegistryParameterMetadata(
                    name="query",
                    required=True,
                    source_hint="user",
                ),
                RegistryParameterMetadata(
                    name="context_id",
                    required=True,
                    source_hint="derived_from_previous_step",
                ),
            ],
        )
    return registry


def make_valid_plan(chain: SampledToolChain, plan_id: str = "plan_001") -> StructuredConversationPlan:
    steps = [
        PlanStep(
            step_index=i,
            tool_id=f"tool_{chr(65 + i)}",
            endpoint_id=f"ep_{i}",
            depends_on_steps=list(range(i)),
            purpose=f"Purpose of step {i}",
            user_intent=f"User wants step {i}",
            assistant_intent=f"Assistant handles step {i}",
            expected_output_usage=None if i == 2 else f"Used by step {i + 1}",
            may_require_clarification=False,
            clarification_reason=None,
        )
        for i in range(3)
    ]
    return StructuredConversationPlan(
        plan_id=plan_id,
        chain_id=chain.chain_id,
        seed=chain.seed,
        domain="travel planning",
        user_goal="Book a flight and hotel for a business trip.",
        pattern_type=chain.pattern_type,
        conversation_style="goal_driven",
        style_notes="User is direct and goal-oriented.",
        tools_used=["tool_A", "tool_B", "tool_C"],
        steps=steps,
        clarification_points=[],
        summary_seed_fields=SummarySeedFields(
            domain="travel planning",
            pattern_type=chain.pattern_type,
            tools_used=["tool_A", "tool_B", "tool_C"],
            conversation_style="goal_driven",
            planned_clarification_count=0,
        ),
    )


# ---------------------------------------------------------------------------
# Validator tests — SampledToolChain
# ---------------------------------------------------------------------------

class TestValidateSampledToolChain:
    def test_valid_chain(self):
        chain = make_chain()
        result = validate_sampled_tool_chain(chain)
        assert result.valid

    def test_empty_chain(self):
        chain = SampledToolChain(chain_id="c1", seed=1, pattern_type="seq", steps=[])
        result = validate_sampled_tool_chain(chain)
        assert not result.valid
        assert any("no steps" in e.lower() for e in result.errors)

    def test_non_contiguous_indices(self):
        chain = make_chain()
        chain.steps[1].step_index = 5
        result = validate_sampled_tool_chain(chain)
        assert not result.valid

    def test_invalid_dependency_out_of_range(self):
        chain = make_chain()
        chain.steps[0].depends_on_steps = [99]
        result = validate_sampled_tool_chain(chain)
        assert not result.valid

    def test_forward_dependency(self):
        chain = make_chain()
        chain.steps[0].depends_on_steps = [1]   # depends on a later step
        result = validate_sampled_tool_chain(chain)
        assert not result.valid

    def test_too_short(self):
        chain = make_chain(n_steps=1)
        result = validate_sampled_tool_chain(chain, min_steps=3)
        assert not result.valid

    def test_missing_tool_id(self):
        chain = make_chain()
        chain.steps[0].tool_id = ""
        result = validate_sampled_tool_chain(chain)
        assert not result.valid


# ---------------------------------------------------------------------------
# Validator tests — ConversationPlan
# ---------------------------------------------------------------------------

class TestValidateConversationPlan:
    def test_valid_plan(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        result = validate_conversation_plan(plan, chain)
        assert result.valid

    def test_chain_id_mismatch(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.chain_id = "wrong_chain"
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_seed_mismatch(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.seed = 999
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_missing_sampled_step(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.steps = plan.steps[:2]   # drop the last step
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_invalid_conversation_style(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.conversation_style = "invalid_style"
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_tools_used_mismatch(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.tools_used = ["tool_X", "tool_Y"]   # wrong
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_clarification_flag_without_reason(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.steps[0].may_require_clarification = True
        plan.steps[0].clarification_reason = None
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_clarification_point_invalid_step(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.clarification_points = [
            ClarificationPoint(
                before_step=99,
                reason="Missing info",
                missing_or_ambiguous_fields=["field_x"],
                question_goal="Ask about field_x",
            )
        ]
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_summary_seed_domain_mismatch(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.summary_seed_fields.domain = "wrong domain"
        result = validate_conversation_plan(plan, chain)
        assert not result.valid

    def test_raise_if_invalid(self):
        chain = make_chain()
        plan = make_valid_plan(chain)
        plan.chain_id = "wrong"
        result = validate_conversation_plan(plan, chain)
        with pytest.raises(PlannerOutputValidationError):
            raise_if_invalid(result)


# ---------------------------------------------------------------------------
# Scaffold tests
# ---------------------------------------------------------------------------

class TestScaffold:
    def test_build_step_scaffold_preserves_structure(self):
        chain = make_chain(3)
        steps = build_step_scaffold(chain)
        assert len(steps) == 3
        for i, s in enumerate(steps):
            assert s.step_index == i
            assert s.tool_id == f"tool_{chr(65 + i)}"
            assert s.purpose == ""   # narrative not yet filled

    def test_clarification_detection_with_registry(self):
        chain = make_chain(3)
        registry = make_registry(chain)
        candidates = detect_clarification_candidates(chain, registry)
        # step 0 has no prior steps, so "derived_from_previous_step" param is ambiguous
        assert any(c.step_index == 0 for c in candidates)

    def test_clarification_detection_no_registry(self):
        chain = make_chain(3)
        candidates = detect_clarification_candidates(chain, None)
        assert candidates == []

    def test_novelty_hints_suggest_least_used_style(self):
        summaries = [
            CorpusSummary(content="s1", conversation_style="direct"),
            CorpusSummary(content="s2", conversation_style="direct"),
            CorpusSummary(content="s3", conversation_style="exploratory"),
        ]
        hints = derive_novelty_hints(summaries)
        # "direct" x2, "exploratory" x1 — least used is one of the remaining 5
        assert hints.suggested_style not in ("direct",)
        assert hints.suggested_style in VALID_CONVERSATION_STYLES

    def test_novelty_hints_empty_corpus(self):
        hints = derive_novelty_hints([])
        assert hints.avoid_domains == []
        assert hints.suggested_style in VALID_CONVERSATION_STYLES


# ---------------------------------------------------------------------------
# Narrative parser tests
# ---------------------------------------------------------------------------

class TestNarrativeParser:
    def _make_scaffold(self, n=2):
        chain = make_chain(n)
        return build_step_scaffold(chain)

    def _valid_response(self, n=2):
        steps = [
            {
                "step_index": i,
                "purpose": f"Fetch result {i}",
                "user_intent": f"User wants result {i}",
                "assistant_intent": f"Assistant calls endpoint {i}",
                "expected_output_usage": None if i == n - 1 else f"Used in step {i + 1}",
                "may_require_clarification": False,
                "clarification_reason": None,
            }
            for i in range(n)
        ]
        return json.dumps({
            "domain": "e-commerce",
            "user_goal": "Find and purchase a laptop.",
            "conversation_style": "goal_driven",
            "style_notes": "User is concise and task-focused.",
            "steps": steps,
        })

    def test_valid_parse(self):
        scaffold = self._make_scaffold(2)
        output = parse_narrative_response(self._valid_response(2), scaffold)
        assert output.domain == "e-commerce"
        assert output.conversation_style == "goal_driven"
        assert len(output.step_narratives) == 2

    def test_invalid_json(self):
        scaffold = self._make_scaffold(2)
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_narrative_response("not json", scaffold)

    def test_missing_top_level_field(self):
        scaffold = self._make_scaffold(2)
        data = json.loads(self._valid_response(2))
        del data["domain"]
        with pytest.raises(ValueError, match="missing top-level fields"):
            parse_narrative_response(json.dumps(data), scaffold)

    def test_invalid_style(self):
        scaffold = self._make_scaffold(2)
        data = json.loads(self._valid_response(2))
        data["conversation_style"] = "bad_style"
        with pytest.raises(ValueError, match="invalid conversation_style"):
            parse_narrative_response(json.dumps(data), scaffold)

    def test_missing_step(self):
        scaffold = self._make_scaffold(2)
        data = json.loads(self._valid_response(2))
        data["steps"] = data["steps"][:1]   # only step 0
        with pytest.raises(ValueError, match="missing step_index 1"):
            parse_narrative_response(json.dumps(data), scaffold)

    def test_fenced_json_stripped(self):
        scaffold = self._make_scaffold(2)
        fenced = f"```json\n{self._valid_response(2)}\n```"
        output = parse_narrative_response(fenced, scaffold)
        assert output.domain == "e-commerce"

    def test_clarification_flag_without_reason_rejected(self):
        scaffold = self._make_scaffold(2)
        data = json.loads(self._valid_response(2))
        data["steps"][0]["may_require_clarification"] = True
        data["steps"][0]["clarification_reason"] = None
        with pytest.raises(ValueError, match="no clarification_reason"):
            parse_narrative_response(json.dumps(data), scaffold)


# ---------------------------------------------------------------------------
# PlannerAgent end-to-end tests (mock LLM and memory)
# ---------------------------------------------------------------------------

class MockLLM:
    """Returns a valid narrative for whatever chain it receives."""
    def complete(self, prompt: str) -> str:
        # Extract step count from prompt (hacky but fine for tests)
        import re
        matches = re.findall(r"Step (\d+):", prompt)
        n = max(int(m) for m in matches) + 1 if matches else 1
        steps = [
            {
                "step_index": i,
                "purpose": f"Purpose {i}",
                "user_intent": f"User intent {i}",
                "assistant_intent": f"Assistant intent {i}",
                "expected_output_usage": None,
                "may_require_clarification": False,
                "clarification_reason": None,
            }
            for i in range(n)
        ]
        return json.dumps({
            "domain": "healthcare scheduling",
            "user_goal": "Schedule an appointment with a specialist.",
            "conversation_style": "exploratory",
            "style_notes": "User is uncertain and asks follow-up questions.",
            "steps": steps,
        })


class MockMemory:
    def __init__(self, summaries=None):
        self._summaries = summaries or []

    def search(self, query: str, scope: str, top_k: int):
        results = self._summaries[:top_k]
        if results and isinstance(results[0], CorpusSummary):
            return [
                {
                    "memory": summary.content,
                    "metadata": {
                        "conversation_id": summary.conversation_id,
                        "tools": summary.tools,
                        "pattern_type": summary.pattern_type,
                        "domain": summary.domain,
                        "conversation_style": summary.conversation_style,
                    },
                }
                for summary in results
            ]
        return results

    def query(self, scope: str, limit: int):
        return self._summaries[:limit]


class TestPlannerAgentEndToEnd:
    def _agent(self, registry=None, config=None):
        return PlannerAgent(
            llm_backend=MockLLM(),
            memory_store=MockMemory(),
            registry=registry,
            config=config or PlannerConfig(),
        )

    def test_successful_plan(self):
        chain = make_chain(3)
        agent = self._agent()
        result = agent.plan(chain, plan_id="plan_e2e_001")
        assert result.success
        assert result.plan is not None
        assert result.plan.plan_id == "plan_e2e_001"
        assert result.plan.chain_id == chain.chain_id
        assert result.plan.seed == chain.seed
        assert len(result.plan.steps) == 3
        assert result.plan.conversation_style in VALID_CONVERSATION_STYLES

    def test_plan_id_echoed(self):
        chain = make_chain(2)
        agent = self._agent()
        result = agent.plan(chain, plan_id="orchestrator_assigned_id")
        assert result.plan.plan_id == "orchestrator_assigned_id"

    def test_empty_plan_id_fails(self):
        chain = make_chain(2)
        agent = self._agent()
        result = agent.plan(chain, plan_id="")
        assert not result.success
        assert result.error_code == "PLANNER_INVALID_INPUT"

    def test_empty_chain_fails(self):
        chain = SampledToolChain(chain_id="c1", seed=1, pattern_type="seq", steps=[])
        agent = self._agent()
        result = agent.plan(chain, plan_id="p1")
        assert not result.success
        assert result.error_code in RESAMPLE_ON

    def test_chain_too_short_fails(self):
        chain = make_chain(1)
        agent = self._agent(config=PlannerConfig(min_steps=3))
        result = agent.plan(chain, plan_id="p1")
        assert not result.success
        assert result.error_code == "PLANNER_CHAIN_TOO_SHORT"

    def test_with_corpus_summaries(self):
        chain = make_chain(2)
        memory = MockMemory(summaries=[
            CorpusSummary(
                content="Prior conversation about travel.",
                domain="travel",
                tools=["tool_A"],
                pattern_type="sequential",
                conversation_style="direct",
            )
        ])
        agent = PlannerAgent(llm_backend=MockLLM(), memory_store=memory)
        result = agent.plan(chain, plan_id="p2")
        assert result.success
        assert len(result.retrieved_corpus_summaries) == 1

    def test_missing_llm_raises_config_error(self):
        with pytest.raises(Exception):
            PlannerAgent(llm_backend=None, memory_store=MockMemory())

    def test_missing_memory_raises_config_error(self):
        with pytest.raises(Exception):
            PlannerAgent(llm_backend=MockLLM(), memory_store=None)

    def test_resample_on_policy_coverage(self):
        """Ensure resample-on codes include all retriable error types."""
        assert "PLANNER_INVALID_INPUT" in RESAMPLE_ON
        assert "PLANNER_CHAIN_TOO_SHORT" in RESAMPLE_ON
        assert "PLANNER_RETRY_EXHAUSTED" in RESAMPLE_ON
        assert "PLANNER_OUTPUT_VALIDATION" in RESAMPLE_ON

    def test_hard_stop_on_policy_coverage(self):
        assert "PLANNER_CONFIG_ERROR" in HARD_STOP_ON
        assert "PLANNER_DEPENDENCY_ERROR" in HARD_STOP_ON

    def test_summary_seed_fields_consistent(self):
        chain = make_chain(3)
        agent = self._agent()
        result = agent.plan(chain, plan_id="p3")
        assert result.success
        plan = result.plan
        ssf = plan.summary_seed_fields
        assert ssf.domain == plan.domain
        assert ssf.pattern_type == plan.pattern_type
        assert sorted(ssf.tools_used) == sorted(plan.tools_used)
        assert ssf.conversation_style == plan.conversation_style
