"""
generator/executor.py
---------------------
Offline Executor — required by the PDF as a first-class component.

Responsibilities:
  - Validate arguments against endpoint schema
  - Emit LLM-generated mock outputs consistent with the actual call arguments
  - Maintain lightweight session state so later calls reference earlier outputs
  - Resolve argument values using the 4-step precedence policy

Value resolution precedence (at executor time only):
  1. Explicit user input from conversation
  2. Transition.field_mappings from prior step output
  3. Session memory retrieval
  4. Default/mock fallback

No concrete values are pre-filled by Sampler or Planner.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from synthetic_datagen.common.types import Transition, FieldMapping
from synthetic_datagen.graph.registry import ToolRegistry, Endpoint, NormalizedParameter
from synthetic_datagen.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """Record of one tool call's inputs and outputs."""
    step_index: int
    endpoint_id: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    was_grounded: bool = False


@dataclass
class SessionState:
    """Lightweight session state for one conversation."""
    conversation_id: str
    steps: list[StepOutput] = field(default_factory=list)
    accumulated_fields: dict[str, Any] = field(default_factory=dict)

    def record_step(self, step: StepOutput) -> None:
        self.steps.append(step)
        self._flatten_into(step.output, self.accumulated_fields)

    def _flatten_into(self, obj: Any, target: dict, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                target[k] = v
                if isinstance(v, (dict, list)):
                    self._flatten_into(v, target, k)
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            self._flatten_into(obj[0], target, prefix)

    def get_field(self, field_name: str) -> Any | None:
        return self.accumulated_fields.get(field_name)


# ---------------------------------------------------------------------------
# Type-based argument fallback
# ---------------------------------------------------------------------------
# Used only when all 4 precedence sources are exhausted. No pre-written
# example pools — type and enum constraints are the only guides.

_TYPE_DEFAULTS: dict[str, Any] = {
    "string":  "value",
    "integer": 1,
    "number":  1.0,
    "boolean": True,
    "array":   [],
    "object":  {},
    "unknown": "value",
}


def _mock_value_for_param(param: NormalizedParameter, rng: random.Random | None = None) -> Any:
    """Return a type-appropriate fallback value for a parameter."""
    if param.enum:
        return (rng or random).choice(param.enum)
    if param.default is not None:
        return param.default
    return _TYPE_DEFAULTS.get(param.type, "value")


# ---------------------------------------------------------------------------
# LLM-backed mock output generator
# ---------------------------------------------------------------------------

def _generate_mock_output_llm(
    endpoint: Endpoint,
    arguments: dict,
    llm_backend: Any,
    user_goal: str | None = None,
) -> dict:
    """
    Ask the LLM to generate a realistic response JSON for this endpoint call.

    The prompt includes the endpoint description, actual call arguments, and
    optional user_goal so results are grounded in user intent (e.g. a restaurant
    search with user_goal mentioning "Italian" returns Italian restaurants even
    if no cuisine param was passed).

    Falls back to _generate_mock_output_minimal on any parse failure.
    """
    schema = endpoint.returns_schema or {}
    goal_line = f"User's overall goal: {user_goal}\n" if user_goal else ""
    prompt = (
        f"Generate a realistic JSON API response for the following endpoint call.\n\n"
        f"Endpoint: {endpoint.endpoint_id}\n"
        f"Description: {endpoint.description or 'API endpoint'}\n"
        f"{goal_line}"
        f"Request parameters: {json.dumps(arguments)}\n"
        f"Response schema: {json.dumps(schema)}\n\n"
        f"Requirements:\n"
        f"- Values must be consistent with the request parameters AND the user's goal\n"
        f"  (e.g. if goal mentions 'Italian restaurant', return Italian restaurants;\n"
        f"   if goal mentions 'Paris', use Parisian names and addresses)\n"
        f"- Use realistic values: real-sounding names, plausible prices, "
        f"valid date strings, proper ID formats\n"
        f"- Match the response schema structure exactly\n"
        f"- Include a status field set to 'ok' or 'success' if the schema has one\n\n"
        f"Return ONLY a valid JSON object — no explanation, no markdown fences."
    )
    try:
        raw = llm_backend.complete(prompt).strip()
        # Strip markdown fences if the LLM adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return _generate_mock_output_minimal(schema, arguments)


def _generate_mock_output_minimal(schema: Any, arguments: dict) -> dict:
    """
    Fallback: build a minimal valid response from the schema using only
    type-based defaults. Used when LLM call or JSON parsing fails.
    """
    if not schema:
        return {"status": "ok", "result": "success"}
    if isinstance(schema, dict):
        result: dict = {}
        for key, value in schema.items():
            if key in arguments:
                result[key] = arguments[key]
            elif isinstance(value, str):
                result[key] = arguments.get(key, "value")
            elif isinstance(value, (int, float)):
                result[key] = value
            elif isinstance(value, bool):
                result[key] = value
            elif isinstance(value, list):
                result[key] = []
            elif isinstance(value, dict):
                result[key] = _generate_mock_output_minimal(value, arguments)
            else:
                result[key] = value
        return result
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Offline Executor
# ---------------------------------------------------------------------------

class OfflineExecutor:
    """
    Validates and executes tool calls offline.

    Mock outputs are generated by the LLM so they are contextually grounded
    in the actual call arguments (e.g. a Paris hotel search returns Parisian
    hotel names, not generic placeholder strings).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm_backend: Any,
        memory_store: MemoryStore | None = None,
        seed: int | None = None,
    ):
        self.registry = registry
        self.llm = llm_backend
        self.memory = memory_store
        self.rng = random.Random(seed)

    def execute_step(
        self,
        endpoint_id: str,
        user_inputs: dict[str, Any],
        session: SessionState,
        transition: Transition | None = None,
        step_index: int = 0,
        user_goal: str | None = None,
    ) -> StepOutput:
        """
        Execute one tool call step.

        Args:
            endpoint_id:  which endpoint to call
            user_inputs:  values explicitly provided by the user
            session:      current session state
            transition:   transition leading to this step (for field_mappings)
            step_index:   position in the chain (0 = first step)

        Returns:
            StepOutput with validated arguments and LLM-generated output
        """
        ep = self.registry.get_endpoint(endpoint_id)
        if ep is None:
            raise ValueError(f"[executor] Unknown endpoint: {endpoint_id}")

        arguments, was_grounded = self._resolve_arguments(
            endpoint=ep,
            user_inputs=user_inputs,
            session=session,
            transition=transition,
            step_index=step_index,
        )

        validation_errors = self._validate_arguments(ep, arguments)
        if validation_errors:
            for param in ep.parameters:
                if param.required and param.name not in arguments:
                    arguments[param.name] = _mock_value_for_param(param, self.rng)

        output = _generate_mock_output_llm(ep, arguments, self.llm, user_goal=user_goal)

        step = StepOutput(
            step_index=step_index,
            endpoint_id=endpoint_id,
            arguments=arguments,
            output=output,
            was_grounded=was_grounded,
        )

        session.record_step(step)

        if self.memory:
            self.memory.add(
                content=json.dumps(output),
                scope=f"session_{session.conversation_id}",
                metadata={
                    "conversation_id": session.conversation_id,
                    "step": step_index,
                    "endpoint": endpoint_id,
                },
            )

        return step

    def _resolve_arguments(
        self,
        endpoint: Endpoint,
        user_inputs: dict[str, Any],
        session: SessionState,
        transition: Transition | None,
        step_index: int,
    ) -> tuple[dict[str, Any], bool]:
        """
        Resolve argument values using the 4-step precedence policy.

        Returns: (arguments dict, was_grounded by memory)
        """
        arguments: dict[str, Any] = {}
        was_grounded = False
        memory_field_cache: dict[str, Any] = {}

        if self.memory and step_index > 0:
            step_results = self.memory.search(
                query=f"{endpoint.name} {endpoint.intent}",
                scope=f"session_{session.conversation_id}",
                top_k=5,
            )
            if step_results:
                was_grounded = True

                retrieved_entries = "\n".join(
                    r.get("memory", "") for r in step_results if r.get("memory")
                )
                argument_filling_prompt = (
                    f"[Memory context]\n{retrieved_entries}\n\n"
                    f"Given the above context and the current tool schema, "
                    f"fill in the arguments for {endpoint.name}."
                )
                if not hasattr(session, "_memory_prompts"):
                    session._memory_prompts = []
                session._memory_prompts.append({
                    "step": step_index,
                    "endpoint": endpoint.endpoint_id,
                    "prompt": argument_filling_prompt,
                })

                for r in step_results:
                    try:
                        mem_content = json.loads(r.get("memory", "{}"))
                        if isinstance(mem_content, dict):
                            memory_field_cache.update(mem_content)
                    except (json.JSONDecodeError, TypeError):
                        pass

        for param in endpoint.parameters:
            if not param.required and param.name not in user_inputs:
                if param.default is not None:
                    arguments[param.name] = param.default
                continue

            # Priority 1: Explicit user input
            if param.name in user_inputs:
                arguments[param.name] = user_inputs[param.name]
                continue

            # Priority 2: field_mappings from prior transition
            if transition and step_index > 0:
                for fm in transition.field_mappings:
                    if fm.target_param == param.name:
                        field_value = session.get_field(fm.source_field)
                        if field_value is not None:
                            arguments[param.name] = field_value
                            break

            if param.name in arguments:
                continue

            # Priority 2.5: Direct accumulated_fields lookup
            if step_index > 0:
                direct_value = session.accumulated_fields.get(param.name)
                if direct_value is not None:
                    arguments[param.name] = direct_value
                    continue

            # Priority 3: Extract from step-level memory cache
            if step_index > 0 and param.name in memory_field_cache:
                arguments[param.name] = memory_field_cache[param.name]
                continue

            # Priority 4: LLM-generated contextual fallback, then type-default
            arguments[param.name] = self._llm_fallback_value(param, endpoint, arguments)

        return arguments, was_grounded

    def _llm_fallback_value(
        self,
        param: NormalizedParameter,
        endpoint: Endpoint,
        partial_arguments: dict[str, Any],
    ) -> Any:
        """
        Generate a realistic value for a required param using the LLM when
        no value was found via the 4-step precedence chain.

        This replaces the generic _mock_value_for_param("string") → "value"
        fallback that makes tool arguments look synthetic.
        """
        if param.enum:
            return (self.rng or random).choice(param.enum)
        if param.default is not None:
            return param.default

        # Only call LLM for string params — integers/booleans need no embellishment
        if param.type not in ("string", "unknown"):
            return _TYPE_DEFAULTS.get(param.type, "value")

        context_str = json.dumps(partial_arguments) if partial_arguments else "none"
        prompt = (
            f"An API call is being made to endpoint '{endpoint.endpoint_id}'.\n"
            f"Endpoint description: {endpoint.description or 'API endpoint'}\n"
            f"Other argument values already resolved: {context_str}\n\n"
            f"The parameter '{param.name}' (type: string) has no value yet.\n"
            f"Parameter description: {param.description or param.name}\n\n"
            f"Return ONLY a realistic, specific string value for this parameter "
            f"that is consistent with the other argument values. "
            f"No explanation, no quotes, just the value."
        )
        try:
            result = self.llm.complete(prompt).strip().strip('"').strip("'").strip()
            return result if result else _TYPE_DEFAULTS.get(param.type, "value")
        except Exception:
            return _TYPE_DEFAULTS.get(param.type, "value")

    def _validate_arguments(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
    ) -> list[str]:
        """Validate arguments against endpoint schema. Returns list of errors."""
        errors: list[str] = []
        for param in endpoint.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required param: {param.name}")
            if param.enum and param.name in arguments:
                if str(arguments[param.name]) not in [str(e) for e in param.enum]:
                    errors.append(f"Invalid enum value for {param.name}: {arguments[param.name]}")
        return errors

    def create_session(self, conversation_id: str) -> SessionState:
        """Create a new session for a conversation."""
        return SessionState(conversation_id=conversation_id)
