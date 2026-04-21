"""
cli/main.py
-----------
CLI for the offline synthetic conversation generator.

Commands:
  build     — build registry, heterogeneous graph, projected graph, write artifacts
  generate  — generate synthetic conversations using built artifacts
  evaluate  — score conversations with LLM-as-judge, optionally repair failing ones
  validate  — validate a generated JSONL dataset (structural checks only)
  metrics   — compute diversity and quality metrics on generated data

Usage:
  python -m synthetic_datagen.cli.main build
  python -m synthetic_datagen.cli.main generate --n 50 --seed 42
  python -m synthetic_datagen.cli.main generate --n 50 --seed 42 --no-corpus-memory
  python -m synthetic_datagen.cli.main evaluate --input output/conversations.jsonl
  python -m synthetic_datagen.cli.main evaluate --input output/conversations.jsonl --repair
  python -m synthetic_datagen.cli.main validate --input output/conversations.jsonl
  python -m synthetic_datagen.cli.main metrics --input output/conversations.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from the project root (synthetic_datagen_project/.env)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — rely on environment variables being set externally

def _write_json(path: Path, data: dict) -> None:
    """Write a dict to a JSON file with consistent formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)



def _build_pipeline(data_path: str | None = None):
    """Build and return all pipeline components from seed data."""
    from synthetic_datagen.toolbench.ingest import load_seed_tools
    from synthetic_datagen.graph.registry import build_registry
    from synthetic_datagen.graph.heterogeneous_graph import build_heterogeneous_graph
    from synthetic_datagen.graph.projected_graph import build_projected_graph
    from synthetic_datagen.sampler.config import load_sampler_config

    result = load_seed_tools(data_path)
    registry = build_registry(result)
    hetero = build_heterogeneous_graph(registry)
    projected = build_projected_graph(registry, hetero)
    config = load_sampler_config()
    return registry, hetero, projected, config


# ---------------------------------------------------------------------------
# build command
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> None:
    """Build and persist registry, graphs, and manifest artifacts."""
    from synthetic_datagen.graph.registry import summarize_registry
    from synthetic_datagen.graph.heterogeneous_graph import summarize_graph
    from synthetic_datagen.graph.projected_graph import summarize_projected
    import time

    print("[build] Starting build...")
    t0 = time.time()

    registry, hetero, projected, config = _build_pipeline(args.data)

    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Write registry.json
    registry_out = {
        "version": "1.0",
        "tools": [
            {
                "tool_id": t.tool_id,
                "tool_name": t.name,
                "category": t.category,
                "description": t.description,
                "endpoint_ids": t.endpoint_ids,
            }
            for t in registry.tools_by_id.values()
        ],
        "endpoints": [
            {
                "endpoint_id": ep.endpoint_id,
                "tool_id": ep.tool_name,
                "name": ep.name,
                "description": ep.description,
                "category": ep.category,
                "intent": ep.intent,
                "method": ep.method,
                "parameters": [
                    {"name": p.name, "type": p.type, "required": p.required,
                     "description": p.description, "enum": p.enum}
                    for p in ep.parameters
                ],
                "returns_schema": ep.returns_schema,
                "returns_fields": sorted(ep.returns_fields),
                "returns_types": ep.returns_types,
                "tags": ep.tags,
            }
            for ep in registry.endpoints_by_id.values()
        ],
    }
    _write_json(artifacts_dir / "registry.json", registry_out)
    print(f"[build] Wrote registry.json ({len(registry_out['endpoints'])} endpoints)")

    # Write heterogeneous_graph.json
    _write_json(artifacts_dir / "heterogeneous_graph.json", hetero.to_dict())
    print(f"[build] Wrote heterogeneous_graph.json "
          f"({hetero.node_count()} nodes, {hetero.edge_count()} edges)")

    # Write projected_graph.json
    _write_json(artifacts_dir / "projected_graph.json", projected.to_dict())
    print(f"[build] Wrote projected_graph.json "
          f"({projected.node_count} nodes, {projected.edge_count} edges)")

    # Write build_manifest.json
    elapsed = time.time() - t0
    manifest = {
        "version": "1.0",
        "tool_count": registry.tool_count,
        "endpoint_count": registry.endpoint_count,
        "heterogeneous_node_count": hetero.node_count(),
        "heterogeneous_edge_count": hetero.edge_count(),
        "projected_node_count": projected.node_count,
        "projected_edge_count": projected.edge_count,
        "entry_node_count": len(projected.entry_nodes),
        "build_time_seconds": round(elapsed, 2),
        "artifacts": {
            "registry": str(artifacts_dir / "registry.json"),
            "heterogeneous_graph": str(artifacts_dir / "heterogeneous_graph.json"),
            "projected_graph": str(artifacts_dir / "projected_graph.json"),
        },
    }
    _write_json(artifacts_dir / "build_manifest.json", manifest)
    print(f"[build] Wrote build_manifest.json")
    print(f"[build] Done in {elapsed:.2f}s")
    print(summarize_registry(registry))
    print(summarize_projected(projected))


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic conversations using built artifacts."""
    from synthetic_datagen.sampler.sampler import SamplerAgent
    from synthetic_datagen.planner.agent import PlannerAgent as StructuredPlannerAgent
    from synthetic_datagen.planner.config import load_planner_config
    from synthetic_datagen.planner.registry_adapter import (
        build_planner_registry,
        adapt_sampled_chain,
        clarification_points_to_steps,
    )
    from synthetic_datagen.generator.llm_backend import AnthropicLLMBackend
    from synthetic_datagen.generator.user_proxy import UserProxyAgent
    from synthetic_datagen.generator.assistant import AssistantAgent
    from synthetic_datagen.generator.executor import OfflineExecutor
    from synthetic_datagen.generator.validator import ConversationValidator
    from synthetic_datagen.generator.writer import DatasetWriter
    from synthetic_datagen.memory.store import MemoryStore

    corpus_memory_enabled = not args.no_corpus_memory
    seed = args.seed
    n = args.n
    mode = args.mode
    inline_evaluate = getattr(args, "evaluate", False)

    # Coverage tracker — counts domain and pattern usage across conversations
    # Used to show diversity steering in action and surface underrepresented combinations
    coverage: dict[str, dict[str, int]] = {"domains": {}, "patterns": {}}

    print(f"[generate] Generating {n} conversations | seed={seed} | mode={mode} | "
          f"corpus_memory={'enabled' if corpus_memory_enabled else 'disabled'}"
          + (f" | inline_evaluate=True | judge={args.judge_model}" if inline_evaluate else ""))

    # Build pipeline
    registry, hetero, projected, config = _build_pipeline(args.data)

    # Initialize components
    memory = MemoryStore(use_mem0=True)  # mem0-backed per PDF; falls back if mem0ai not installed
    sampler = SamplerAgent(projected, registry, config)

    # Shared LLM backend — used by planner, user proxy, assistant, and executor.
    # Model can be overridden via --judge-model arg; default is claude-haiku-4-5-20251001.
    judge_model = getattr(args, "judge_model", None) or "claude-haiku-4-5-20251001"
    llm_backend = AnthropicLLMBackend(model=judge_model)

    # Structured planner setup — narrative.py builds the corpus-grounded prompt
    planner_config = load_planner_config()
    planner_registry = build_planner_registry(
        tool_registry=registry,
        user_natural_params=frozenset(config.user_natural_params),
    )
    planner = StructuredPlannerAgent(
        llm_backend=llm_backend,
        memory_store=memory,
        registry=planner_registry,
        config=planner_config,
    )
    user_proxy = UserProxyAgent(registry, llm_backend=llm_backend, seed=seed)
    assistant = AssistantAgent(registry, llm_backend=llm_backend, seed=seed)
    executor = OfflineExecutor(registry, llm_backend=llm_backend, memory_store=memory, seed=seed)
    validator = ConversationValidator()

    # Optional inline LLM-as-judge
    inline_judge = None
    inline_score_validator = None
    if inline_evaluate:
        from synthetic_datagen.evaluator.judge import AnthropicJudgeClient
        from synthetic_datagen.evaluator.scorer import ScoreValidator
        inline_judge = AnthropicJudgeClient(model=args.judge_model, max_retries=3)
        inline_score_validator = ScoreValidator()

    output_path = Path(args.output)
    writer = DatasetWriter(output_path)
    output_path.unlink(missing_ok=True)  # start fresh

    generated = 0
    rejected = 0
    conv_index = 0

    # Sample chains
    try:
        chains = sampler.sample_mixed(n=n * 2, seed=seed)  # oversample for rejection
    except Exception as e:
        print(f"[generate] Warning: {e}")
        chains = sampler.sample_chains(n=n, mode=mode, seed=seed)

    for chain in chains:
        if generated >= n:
            break

        conv_index += 1
        conversation_id = f"conv_{seed}_{conv_index:04d}"

        # Reject chains whose tools span incompatible scenario groups before
        # spending any LLM calls on them.
        if not _chain_is_domain_coherent(chain.endpoint_ids):
            rejected += 1
            if args.verbose:
                print(f"[generate] Skipped {conversation_id}: incoherent domain mix "
                      f"{chain.endpoint_ids}")
            continue

        try:
            record = _generate_one_conversation(
                conversation_id=conversation_id,
                chain=chain,
                planner=planner,
                planner_registry=planner_registry,
                user_proxy=user_proxy,
                assistant=assistant,
                executor=executor,
                memory=memory,
                corpus_memory_enabled=corpus_memory_enabled,
                seed=seed,
                user_natural_params=config.user_natural_params,
                llm_backend=llm_backend,
            )

            # Validate
            result = validator.validate(record)
            if not result.passed:
                rejected += 1
                if args.verbose:
                    print(f"[generate] Rejected {conversation_id}: {result.errors}")
                continue

            # Inline LLM-as-judge scoring (if --evaluate flag set)
            if inline_judge is not None:
                from synthetic_datagen.evaluator.scorer import attach_scores
                try:
                    raw_result = inline_judge.score(record)
                    scores = inline_score_validator.validate(raw_result)
                    record = attach_scores(record, scores)
                except Exception as e:
                    if args.verbose:
                        print(f"[generate] Judge error on {conversation_id}: {e}")

            writer.write(record)
            generated += 1

            # Update coverage tracker
            plan_domain = record["metadata"].get("domain", "General")
            plan_pattern = record["metadata"].get("pattern_type", "unknown")
            coverage["domains"][plan_domain] = coverage["domains"].get(plan_domain, 0) + 1
            coverage["patterns"][plan_pattern] = coverage["patterns"].get(plan_pattern, 0) + 1

            # Write to corpus memory after successful generation
            if corpus_memory_enabled:
                memory.add(
                    content=(
                        f"Tools: {', '.join(chain.tool_ids)}. "
                        f"Domain: {plan_domain}. "
                        f"Pattern: {chain.pattern_type}."
                    ),
                    scope="corpus",
                    metadata={
                        "conversation_id": conversation_id,
                        "tools": chain.tool_ids,
                        "pattern_type": chain.pattern_type,
                    },
                )

            if generated % 10 == 0:
                print(f"[generate] {generated}/{n} generated...")

        except Exception as e:
            if args.verbose:
                print(f"[generate] Error on {conversation_id}: {e}")
            rejected += 1
            continue

    print(f"[generate] Done: {generated} generated, {rejected} rejected")
    print(f"[generate] Output: {output_path}")

    # Print coverage report — shows diversity steering in action
    print(f"\n[coverage] Domain distribution ({len(coverage['domains'])} domains):")
    for domain, count in sorted(coverage["domains"].items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {domain:<30} {count:>3}  {bar}")

    print(f"\n[coverage] Pattern distribution:")
    for pattern, count in sorted(coverage["patterns"].items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {pattern:<30} {count:>3}  {bar}")

    # Highlight underrepresented domains (appeared only once)
    underrepresented = [d for d, c in coverage["domains"].items() if c == 1]
    if underrepresented:
        print(f"\n[coverage] Underrepresented domains (1 conversation each): {underrepresented}")


# ---------------------------------------------------------------------------
# Domain cohesion filter — rejects chains that combine incompatible tool groups
# ---------------------------------------------------------------------------

# Map tool_id (the namespace before "::") to a scenario group.
# Tools in the same or adjacent groups work together naturally.
_TOOL_SCENARIO_GROUP: dict[str, str] = {
    "flight_search":    "travel",
    "hotel_booking":    "travel",
    "maps_geocoding":   "travel",
    "weather_api":      "travel",
    "currency_exchange":"travel",
    "restaurant_finder":"lifestyle",
    "recipe_finder":    "lifestyle",
    "event_ticketing":  "lifestyle",
    "calendar_events":  "productivity",
    "messaging_service":"communication",
    "translation_service": "communication",
    "user_profile":     "account",
    "stock_market":     "finance",
    "news_api":         "finance",
    "job_search":       "career",
    "ecommerce_search": "commerce",
}

# Groups that should never appear together in a single chain.
# A chain is incoherent if ANY incompatible pair is present.
_INCOMPATIBLE_GROUP_PAIRS: set[frozenset] = {
    frozenset({"finance", "lifestyle"}),    # stock trading + dining
    frozenset({"finance", "travel"}),       # stock trading + flights/hotels
    frozenset({"finance", "commerce"}),     # stock trading + e-commerce
    frozenset({"career",  "lifestyle"}),    # job search + restaurant booking
    frozenset({"career",  "travel"}),       # job search + flights/hotels
    frozenset({"career",  "commerce"}),     # job search + e-commerce shopping
}


def _chain_is_domain_coherent(endpoint_ids: list[str]) -> bool:
    """
    Return True if the chain's tools form a coherent user scenario.

    Rejects chains that span incompatible scenario groups (e.g. stock market
    combined with restaurant booking — no realistic user does both in one flow).
    Cross-group combinations that ARE realistic (travel + productivity, travel +
    lifestyle) are allowed.
    """
    groups = {
        _TOOL_SCENARIO_GROUP.get(eid.split("::")[0], "other")
        for eid in endpoint_ids
    }
    groups.discard("other")          # unknown tools are neutral
    groups.discard("productivity")   # calendar/tasks are universal
    groups.discard("communication")  # messaging is universal
    groups.discard("account")        # profile is universal

    for g1 in groups:
        for g2 in groups:
            if g1 < g2 and frozenset({g1, g2}) in _INCOMPATIBLE_GROUP_PAIRS:
                return False
    return True


def _extract_params_from_conversation(
    llm_backend,
    messages: list[dict],
    endpoint_id: str,
    param_names: list[str],
    endpoint_description: str = "",
    param_descriptions: dict | None = None,
) -> dict:
    """
    Extract machine-readable parameter values from the conversation history.

    Called before each tool step so user-provided values flow into executor
    Priority 1 (user_inputs) rather than hitting the type-based fallback.
    Only returns params that can be confidently inferred — omits uncertain ones
    so the executor's field_mappings and accumulated_fields still apply.
    """
    if not param_names or not messages:
        return {}

    # Only use user messages for extraction — tool outputs contain values for
    # other tools (e.g. directions origin) that would pollute extraction for
    # the current tool. IDs from prior steps are handled by accumulated_fields.
    history_text = "\n".join(
        f"USER: {m['content']}"
        for m in messages
        if m.get("content") and m["role"] == "user"
    )
    tool_name = endpoint_id.split("::")[-1].replace("_", " ")
    desc_line = f"What this tool does: {endpoint_description}\n" if endpoint_description else ""

    # Include param descriptions when available for better disambiguation
    pd = param_descriptions or {}
    params_list = "\n".join(
        f"- {p}" + (f" ({pd[p]})" if p in pd else "")
        for p in param_names
    )

    prompt = (
        f"You are extracting API call parameters from a conversation.\n\n"
        f"Tool: {tool_name}\n"
        f"{desc_line}"
        f"Parameters needed for THIS tool call:\n{params_list}\n\n"
        f"What the user said:\n{history_text}\n\n"
        f"IMPORTANT: Only extract values that clearly apply to the '{tool_name}' tool "
        f"specifically — do not confuse values meant for other tools in the conversation "
        f"(e.g. don't use a map destination as a flight destination if they are different).\n\n"
        f"Return a JSON object with ONLY the parameters you can confidently extract. "
        f"Use proper types: strings as strings, numbers as numbers, ISO 8601 for dates/times. "
        f"Omit any parameter you are not sure about. "
        f"Return ONLY valid JSON — no explanation, no markdown fences."
    )
    try:
        raw = llm_backend.complete(prompt).strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        extracted = json.loads(raw)
        if isinstance(extracted, dict):
            return extracted
    except Exception:
        pass
    return {}


def _llm_needs_clarification(
    llm_backend,
    conversation_so_far: list[dict],
    missing_params: list[str],
) -> bool:
    """
    Ask the LLM whether the assistant should ask a clarification question.

    Returns True if the information is genuinely missing from the conversation
    so far. Returns False (suppress clarification) if the user already provided
    the information implicitly or explicitly.
    """
    if not missing_params:
        return False

    history_text = "\n".join(
        f"{m['role'].upper()}: {m.get('content', '')}"
        for m in conversation_so_far
        if m.get("content")
    )
    params_text = ", ".join(p.replace("_", " ") for p in missing_params)

    prompt = (
        f"You are deciding whether an AI assistant needs to ask the user for more information.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"The assistant is considering asking for: {params_text}\n\n"
        f"Question: Based on the conversation above, has the user already provided "
        f"enough information to infer '{params_text}', either explicitly or implicitly?\n\n"
        f"Answer with exactly one word: YES (user already provided it, skip clarification) "
        f"or NO (information is genuinely missing, ask for it)."
    )
    try:
        answer = llm_backend.complete(prompt).strip().upper()
        # YES means user already provided it → no clarification needed
        return "YES" not in answer
    except Exception:
        return True  # safe default: ask if unsure


def _generate_one_conversation(
    conversation_id: str,
    chain,
    planner,
    planner_registry: dict,
    user_proxy,
    assistant,
    executor,
    memory,
    corpus_memory_enabled: bool,
    seed: int | None,
    user_natural_params: set | None = None,
    llm_backend=None,
) -> dict:
    """Generate one complete conversation from a SampledChain."""
    from synthetic_datagen.generator.writer import DatasetWriter
    from synthetic_datagen.planner.registry_adapter import (
        adapt_sampled_chain,
        clarification_points_to_steps,
    )

    # Adapt SampledChain → SampledToolChain for the structured planner
    adapted_chain = adapt_sampled_chain(
        chain=chain,
        chain_id=conversation_id,
        seed=seed if seed is not None else 0,
    )

    # Plan — narrative.py builds the corpus-grounded prompt and sends to LLM
    plan_result = planner.plan(adapted_chain, plan_id=conversation_id)
    if not plan_result.success:
        raise RuntimeError(
            f"Planner failed [{plan_result.error_code}]: {plan_result.error_message}"
        )
    plan = plan_result.plan

    # Bridge ClarificationPoints → ClarificationSteps for the generator layer
    all_clarif = clarification_points_to_steps(plan.clarification_points)

    # Create session state
    session = executor.create_session(conversation_id)

    # ConversationState — shared communication object between all generator agents
    from synthetic_datagen.common.types import ConversationState
    state = ConversationState(conversation_id=conversation_id, plan=plan, session=session)

    messages = []
    tool_calls_log = []
    tool_outputs_log = []
    clarification_count = 0
    grounded_steps = 0
    non_first_steps = 0

    # Accumulates resolved param values from every clarification answer.
    # Passed to every executor.execute_step so tool arguments are consistent
    # with what the user said — no LLM parsing needed because user_proxy
    # generated both the utterance and its canonical value from the same source.
    persistent_user_inputs: dict = {}

    # Opening user message
    user_turn = user_proxy.generate_initial_request(plan)
    messages.append({"role": "user", "content": user_turn.content})

    def _step_purpose(step_idx: int) -> str | None:
        """Look up the step purpose from the plan for context-aware clarification."""
        return next((s.purpose for s in plan.steps if s.step_index == step_idx), None)

    # Keep the opening user message for context in clarification answers
    opening_message = user_turn.content

    # Handle step-0 clarification — gate with LLM to avoid asking for info
    # the user already provided in their opening message.
    step0_clarifs = [cs for cs in all_clarif if cs.step_index == 0]
    if step0_clarifs:
        cs0 = step0_clarifs[0]
        should_ask = (
            llm_backend is None
            or _llm_needs_clarification(
                llm_backend,
                conversation_so_far=messages,
                missing_params=cs0.missing_params or [],
            )
        )
        if should_ask:
            ast_turn = assistant.ask_clarification(cs0, step_purpose=_step_purpose(0))
            messages.append({"role": "assistant", "content": ast_turn.content})
            clarification_count += 1

            user_resp = user_proxy.answer_clarification(
                cs0, plan,
                original_request=opening_message,
                accumulated_fields=session.accumulated_fields,
            )
            messages.append({"role": "user", "content": user_resp.content})
            persistent_user_inputs.update(user_resp.resolved_params)

    # Execute each endpoint in the chain
    for step_idx, endpoint_id in enumerate(chain.endpoint_ids):
        # Check for mid-chain clarification.
        # Skip if the required params are already resolved via earlier clarifications
        # or accumulated session fields — avoids redundant clarification questions.
        mid_clarifs = [cs for cs in all_clarif
                      if cs.step_index == step_idx and step_idx > 0]
        if mid_clarifs:
            cs = mid_clarifs[0]
            already_resolved = (
                all(
                    p in persistent_user_inputs or p in session.accumulated_fields
                    for p in (cs.missing_params or [])
                )
                if cs.missing_params else False
            )
            if not already_resolved:
                # LLM gate — suppress if the conversation already contains the info
                should_ask = (
                    llm_backend is None
                    or _llm_needs_clarification(
                        llm_backend,
                        conversation_so_far=messages,
                        missing_params=cs.missing_params or [],
                    )
                )
                if should_ask:
                    ast_turn = assistant.ask_clarification(cs, step_purpose=_step_purpose(step_idx))
                    messages.append({"role": "assistant", "content": ast_turn.content})
                    clarification_count += 1

                    user_resp = user_proxy.answer_clarification(
                        cs, plan,
                        original_request=opening_message,
                        accumulated_fields=session.accumulated_fields,
                    )
                    messages.append({"role": "user", "content": user_resp.content})
                    persistent_user_inputs.update(user_resp.resolved_params)

        # Get transition for this step
        transition = chain.transitions[step_idx - 1] if (step_idx > 0 and step_idx - 1 < len(chain.transitions)) else None

        # Extract any user-stated param values from the conversation so far.
        # This ensures that values the user mentioned in their opening message
        # (or in clarification answers) flow into the executor's Priority 1
        # even when clarification was suppressed by the LLM gate.
        if llm_backend is not None:
            ep = executor.registry.get_endpoint(endpoint_id)
            if ep:
                param_names = [p.name for p in ep.parameters if p.required]
                missing_from_inputs = [
                    p for p in param_names
                    if p not in persistent_user_inputs
                    and p not in session.accumulated_fields
                ]
                if missing_from_inputs:
                    param_descs = {
                        p.name: p.description
                        for p in ep.parameters
                        if p.description
                    }
                    extracted = _extract_params_from_conversation(
                        llm_backend, messages, endpoint_id, missing_from_inputs,
                        endpoint_description=ep.description or "",
                        param_descriptions=param_descs,
                    )
                    persistent_user_inputs.update(extracted)

        # Execute tool call — pass accumulated user inputs so clarification
        # answers flow into actual argument values (Priority 1 in executor)
        if step_idx > 0:
            non_first_steps += 1

        step_out = executor.execute_step(
            endpoint_id=endpoint_id,
            user_inputs=persistent_user_inputs,
            session=session,
            transition=transition,
            step_index=step_idx,
            user_goal=plan.user_goal,
        )

        if step_idx > 0 and step_out.was_grounded:
            grounded_steps += 1
            state.grounded_steps += 1
        if step_idx > 0:
            state.non_first_steps += 1

        # Inline grounding check — verify next step's required params are resolvable
        next_idx = step_idx + 1
        if next_idx < len(chain.endpoint_ids):
            next_endpoint_id = chain.endpoint_ids[next_idx]
            next_ep = executor.registry.get_endpoint(next_endpoint_id)
            if next_ep:
                accumulated = set(session.accumulated_fields.keys())
                next_transition = chain.transitions[step_idx] if step_idx < len(chain.transitions) else None
                mapped_params = set()
                if next_transition:
                    mapped_params = {fm.target_param for fm in next_transition.field_mappings}
                natural = user_natural_params or set()
                # Params covered by a ClarificationStep at this step index
                # (the conversation asked for clarification — offline executor
                # can't resolve ID from natural language, so mock is used; not a gap)
                clarified_params = set()
                for cs in chain.clarification_steps:
                    if cs.step_index == next_idx:
                        clarified_params.update(cs.missing_params)
                missing = [
                    p.name for p in next_ep.parameters
                    if p.required
                    and p.name not in accumulated
                    and p.name not in mapped_params
                    and p.name.lower() not in natural
                    and p.name not in clarified_params
                ]
                if missing:
                    warning = (f"step {next_idx} ({next_endpoint_id}) has unresolvable "
                               f"required params: {missing} — will use mock fallback")
                    state.grounding_warnings.append(warning)

        # Assistant emits tool call — use plan step's assistant_intent as preamble
        # when the narrative backend has generated a meaningful one.
        plan_step = next((s for s in plan.steps if s.step_index == step_idx), None)
        preamble = None
        if (plan_step and plan_step.assistant_intent
                and "calls the appropriate endpoint" not in plan_step.assistant_intent
                and "take care of the" not in plan_step.assistant_intent.lower()):
            preamble = plan_step.assistant_intent
        ast_tool_turn = assistant.emit_tool_call(endpoint_id, step_out.arguments, preamble=preamble)
        messages.append({
            "role": "assistant",
            "content": ast_tool_turn.content,
            "tool_calls": ast_tool_turn.tool_calls,
        })

        # Tool output
        messages.append({
            "role": "tool",
            "name": endpoint_id,
            "content": json.dumps(step_out.output),
        })

        tool_calls_log.append({
            "name": endpoint_id,
            "parameters": step_out.arguments,
        })
        tool_outputs_log.append({
            "name": endpoint_id,
            "output": step_out.output,
        })

    # Final response
    final = assistant.generate_final_response(plan, session.steps)
    messages.append({"role": "assistant", "content": final.content})

    # Surface inline grounding warnings
    if state.grounding_warnings:
        for w in state.grounding_warnings:
            print(f"[grounding] {conversation_id}: {w}")

    # Compute memory_grounding_rate
    if non_first_steps == 0:
        memory_grounding_rate = None
    else:
        memory_grounding_rate = grounded_steps / non_first_steps

    return DatasetWriter.build_record(
        conversation_id=conversation_id,
        messages=messages,
        tool_calls=tool_calls_log,
        tool_outputs=tool_outputs_log,
        chain=chain,
        domain=plan.domain,
        memory_grounding_rate=memory_grounding_rate,
        corpus_memory_enabled=corpus_memory_enabled,
        seed=seed,
        num_clarification_questions=clarification_count,
    )


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Score conversations with LLM-as-judge and write evaluated.jsonl.

    Pipeline:
      1. Load records from --input JSONL.
      2. For each record, call AnthropicJudgeClient.score() (structured output
         via Claude tool use — satisfies the PDF structured-output requirement).
      3. Validate scores against gated thresholds.
      4. If --repair is set and record fails: attempt surgical repair (up to
         --max-repairs attempts), re-score after each attempt.
      5. Attach judge_scores and passed flag to each record.
      6. Write all records (pass and fail) to --output JSONL.
      7. Print aggregated EvaluationReport and exit non-zero if quality
         assertion fails (useful for CI / end-to-end test assertion).
    """
    from synthetic_datagen.evaluator.judge import AnthropicJudgeClient
    from synthetic_datagen.evaluator.scorer import ScoreValidator, attach_scores
    from synthetic_datagen.evaluator.repairer import ConversationRepairer
    from synthetic_datagen.evaluator.report import generate_report, print_report

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[evaluate] File not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold

    # Load records
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[evaluate] JSON parse error, skipping line: {e}")

    if not records:
        print("[evaluate] No records found.")
        sys.exit(1)

    print(f"[evaluate] Scoring {len(records)} conversations | "
          f"model={args.model} | threshold={threshold} | repair={args.repair}")

    # Build components
    judge = AnthropicJudgeClient(
        model=args.model,
        max_retries=3,
        call_delay_s=args.delay,
    )
    validator = ScoreValidator(
        threshold_tool_correctness=threshold,
        threshold_task_completion=threshold,
        threshold_naturalness=max(threshold - 0.5, 1.0),  # naturalness is 0.5 more forgiving
        threshold_mean=threshold,
    )
    repairer = ConversationRepairer(
        judge_client=judge,
        validator=validator,
        max_attempts=args.max_repairs,
        call_delay_s=args.delay,
    ) if args.repair else None

    evaluated: list[dict] = []
    passed_count = failed_count = repaired_count = error_count = 0

    for i, record in enumerate(records, 1):
        conv_id = record.get("metadata", {}).get("conversation_id", f"record_{i}")

        # Score
        raw_result = judge.score(record)
        scores = validator.validate(raw_result)

        if scores.error:
            error_count += 1
            if args.verbose:
                print(f"[evaluate] Judge error on {conv_id}: {scores.error}")

        if scores.passed:
            final_record = attach_scores(record, scores)
            passed_count += 1
        elif repairer and not scores.error:
            # Attempt repair
            repair_result = repairer.repair(record, scores)
            final_record = repair_result.record
            if repair_result.repaired:
                repaired_count += 1
                passed_count += 1
                if args.verbose:
                    print(f"[evaluate] Repaired {conv_id} in {repair_result.repair_attempts} attempt(s)")
            else:
                failed_count += 1
                if args.verbose:
                    print(f"[evaluate] Repair failed for {conv_id}: "
                          f"failed_gates={scores.failed_gates}")
        else:
            final_record = attach_scores(record, scores)
            failed_count += 1
            if args.verbose:
                print(f"[evaluate] Failed {conv_id}: failed_gates={scores.failed_gates}")

        evaluated.append(final_record)

        if i % 10 == 0:
            print(f"[evaluate] {i}/{len(records)} scored...")

    # Write output
    output_path.unlink(missing_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in evaluated:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"[evaluate] Written to {output_path}")
    print(f"[evaluate] Passed: {passed_count} | Failed: {failed_count} | "
          f"Repaired: {repaired_count} | Judge errors: {error_count}")

    # Build and print report
    report = generate_report(
        evaluated,
        threshold_mean=threshold,
        threshold_tool_correctness=threshold,
        threshold_task_completion=threshold,
        threshold_naturalness=max(threshold - 0.5, 1.0),
    )
    print_report(report)

    # Write report JSON alongside output
    report_path = output_path.parent / "evaluation_report.json"
    _write_json(report_path, {
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "judge_errors": report.judge_errors,
        "repaired": report.repaired,
        "repair_attempted": report.repair_attempted,
        "pass_rate": report.pass_rate,
        "mean_tool_correctness": report.mean_tool_correctness,
        "mean_task_completion": report.mean_task_completion,
        "mean_naturalness": report.mean_naturalness,
        "mean_overall": report.mean_overall,
        "thresholds": {
            "mean": report.threshold_mean,
            "tool_correctness": report.threshold_tool_correctness,
            "task_completion": report.threshold_task_completion,
            "naturalness": report.threshold_naturalness,
        },
        "by_domain": report.by_domain,
        "by_pattern": report.by_pattern,
    })
    print(f"[evaluate] Report written to {report_path}")

    # Exit non-zero if quality assertion fails (for CI / e2e test assertion)
    assertion_failed = (
        report.mean_overall is None
        or report.mean_overall < threshold
        or (report.mean_tool_correctness or 0) < threshold
        or (report.mean_task_completion or 0) < threshold
    )
    if assertion_failed:
        print(f"[evaluate] Quality assertion FAILED: mean_overall={report.mean_overall}")
        sys.exit(2)


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a generated JSONL dataset."""
    from synthetic_datagen.generator.validator import ConversationValidator

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[validate] File not found: {input_path}")
        sys.exit(1)

    validator = ConversationValidator()
    passed = failed = 0
    all_errors: list[str] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                result = validator.validate(record)
                if result.passed:
                    passed += 1
                else:
                    failed += 1
                    all_errors.extend(result.errors)
            except json.JSONDecodeError as e:
                failed += 1
                all_errors.append(f"JSON parse error: {e}")

    total = passed + failed
    print(f"[validate] Results: {passed}/{total} passed, {failed} failed")
    if all_errors:
        print(f"[validate] Errors ({len(all_errors)}):")
        for err in all_errors[:10]:
            print(f"  - {err}")


# ---------------------------------------------------------------------------
# metrics command
# ---------------------------------------------------------------------------

def _compute_metrics_dict(records: list[dict]) -> dict:
    """Compute metrics for a set of records and return as dict."""
    meta = [r.get("metadata", {}) for r in records]

    tool_call_counts = [m.get("num_tool_calls", 0) for m in meta]
    clarif_counts = [m.get("num_clarification_questions", 0) for m in meta]
    grounding_rates = [m.get("memory_grounding_rate") for m in meta
                      if m.get("memory_grounding_rate") is not None]
    multi_tool = sum(1 for m in meta if m.get("num_distinct_tools", 0) >= 2)
    multi_step = sum(1 for m in meta if m.get("num_tool_calls", 0) >= 3)

    patterns: dict[str, int] = {}
    for m in meta:
        p = m.get("pattern_type", "unknown")
        patterns[p] = patterns.get(p, 0) + 1

    domains: dict[str, int] = {}
    for m in meta:
        d = m.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1

    buckets: dict[str, int] = {}
    for m in meta:
        tools_key = ",".join(sorted(m.get("tool_ids_used", [])))
        pattern = m.get("pattern_type", "unknown")
        key = f"{tools_key}|{pattern}"
        buckets[key] = buckets.get(key, 0) + 1

    entropy = _compute_entropy(list(buckets.values()), len(records))
    max_entropy = math.log(len(buckets)) if len(buckets) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    import random
    rng = random.Random(42)
    all_tool_sets = [set(m.get("tool_ids_used", [])) for m in meta]
    avg_jaccard = 0.0
    if len(all_tool_sets) >= 2:
        pairs = min(100, len(all_tool_sets) * (len(all_tool_sets) - 1) // 2)
        jaccard_sum = 0.0
        sampled = 0
        for _ in range(pairs):
            i, j = rng.sample(range(len(all_tool_sets)), 2)
            a, b = all_tool_sets[i], all_tool_sets[j]
            union = len(a | b)
            if union > 0:
                jaccard_sum += 1.0 - len(a & b) / union
                sampled += 1
        avg_jaccard = jaccard_sum / sampled if sampled > 0 else 0.0

    return {
        "total": len(records),
        "multi_step": multi_step,
        "multi_tool": multi_tool,
        "avg_tool_calls": sum(tool_call_counts) / len(tool_call_counts) if tool_call_counts else 0,
        "avg_clarifications": sum(clarif_counts) / len(clarif_counts) if clarif_counts else 0,
        "avg_grounding_rate": sum(grounding_rates) / len(grounding_rates) if grounding_rates else None,
        "patterns": patterns,
        "domains": domains,
        "unique_buckets": len(buckets),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "avg_jaccard_dissimilarity": avg_jaccard,
    }


def _print_metrics(label: str, metrics: dict) -> None:
    """Pretty-print metrics dict."""
    print(f"\n[metrics] === {label} ===")
    print(f"  Total conversations:         {metrics['total']}")
    print(f"  Multi-step (>=3):            {metrics['multi_step']} ({100*metrics['multi_step']//metrics['total'] if metrics['total'] > 0 else 0}%)")
    print(f"  Multi-tool (>=2):            {metrics['multi_tool']} ({100*metrics['multi_tool']//metrics['total'] if metrics['total'] > 0 else 0}%)")
    print(f"  Avg tool calls:              {metrics['avg_tool_calls']:.1f}")
    print(f"  Avg clarifications:          {metrics['avg_clarifications']:.2f}")
    if metrics['avg_grounding_rate'] is not None:
        print(f"  Avg grounding rate:          {metrics['avg_grounding_rate']:.3f}")
    print(f"\n  === Pattern Distribution ===")
    for p, count in sorted(metrics['patterns'].items(), key=lambda x: -x[1]):
        print(f"    {p}: {count} ({100*count//metrics['total'] if metrics['total'] > 0 else 0}%)")
    print(f"\n  === Domain Distribution ===")
    for d, count in sorted(metrics['domains'].items(), key=lambda x: -x[1]):
        print(f"    {d}: {count}")
    print(f"\n  === Diversity Metrics ===")
    print(f"    Unique (tool_ids, pattern) buckets: {metrics['unique_buckets']}")
    print(f"    Entropy (primary metric):           {metrics['entropy']:.4f}")
    print(f"    Normalized entropy (0-1):           {metrics['normalized_entropy']:.4f}")
    print(f"    Avg pairwise Jaccard dissimilarity: {metrics['avg_jaccard_dissimilarity']:.4f}")


def cmd_metrics(args: argparse.Namespace) -> None:
    """Compute diversity and quality metrics on generated dataset."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[metrics] File not found: {input_path}")
        sys.exit(1)

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("[metrics] No records found")
        return

    compare_path = getattr(args, "compare", None)
    if compare_path:
        # Load comparison file
        compare_records = []
        cmp_path = Path(compare_path)
        if not cmp_path.exists():
            print(f"[metrics] Comparison file not found: {cmp_path}")
            sys.exit(1)
        with open(cmp_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        compare_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not compare_records:
            print("[metrics] No records found in comparison file")
            sys.exit(1)

        # Compute metrics for both
        metrics_a = _compute_metrics_dict(records)
        metrics_b = _compute_metrics_dict(compare_records)

        print(f"[metrics] Analyzing {len(records)} conversations (file 1) vs {len(compare_records)} conversations (file 2)...")
        _print_metrics(f"File 1: {input_path.name}", metrics_a)
        _print_metrics(f"File 2: {cmp_path.name}", metrics_b)
        return

    print(f"[metrics] Analyzing {len(records)} conversations...")

    # Use helper to compute and print metrics
    metrics = _compute_metrics_dict(records)
    _print_metrics("Single File Analysis", metrics)

    corpus_enabled = [r.get("metadata", {}) for r in records if r.get("metadata", {}).get("corpus_memory_enabled")]
    corpus_disabled = [r.get("metadata", {}) for r in records if not r.get("metadata", {}).get("corpus_memory_enabled")]
    if corpus_enabled and corpus_disabled:
        print(f"\n[metrics] === Corpus Memory Comparison (within file) ===")
        print(f"  With corpus memory:    {len(corpus_enabled)} conversations")
        print(f"  Without corpus memory: {len(corpus_disabled)} conversations")


def _compute_entropy(counts: list[int], total: int) -> float:
    """Compute Shannon entropy from bucket counts."""
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> None:
    """Per-conversation PDF compliance breakdown with tool call sequence detail."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[inspect] File not found: {input_path}")
        sys.exit(1)

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[inspect] JSON parse error: {e}")

    if not records:
        print("[inspect] No records found.")
        return

    print(f"\n[inspect] Analysing {len(records)} conversations from {input_path}\n")
    print(f"{'ID':<20} {'Tools':>5} {'Distinct':>8} {'Clarif':>6} {'Grounding':>9} {'MultiStep':>9} {'MultiTool':>9} {'Status':>6}")
    print("-" * 90)

    pass_count = fail_count = warn_count = 0
    total_tool_calls = total_clarifications = total_distinct_tools = 0
    grounding_rates: list[float] = []
    multi_step_ok = multi_tool_ok = clarif_present = 0
    domain_counts: dict[str, int] = {}
    pattern_counts: dict[str, int] = {}
    tool_combo_counts: dict[str, int] = {}

    for record in records:
        meta = record.get("metadata", {})
        tool_calls = record.get("tool_calls", [])
        conv_id = meta.get("conversation_id", "unknown")[-16:]

        n_tools = len(tool_calls)
        distinct = set(tc.get("name", "").split("::")[0] for tc in tool_calls)
        n_distinct = len(distinct)
        n_clarif = meta.get("num_clarification_questions", 0)
        grounding = meta.get("memory_grounding_rate")
        domain = meta.get("domain", "unknown")
        pattern = meta.get("pattern_type", "unknown")

        ok_multi_step = n_tools >= 3
        ok_multi_tool = n_distinct >= 2
        sampled_chain = meta.get("endpoint_ids", [])
        actual_endpoints = [tc.get("name", "") for tc in tool_calls]
        ok_chain = not sampled_chain or actual_endpoints == sampled_chain

        grounding_str = f"{grounding:.3f}" if grounding is not None else "null"

        if not ok_multi_step or not ok_multi_tool or not ok_chain:
            status = "FAIL"
            fail_count += 1
        elif n_clarif == 0:
            status = "WARN"
            warn_count += 1
        else:
            status = "OK"
            pass_count += 1

        print(
            f"{conv_id:<20} {n_tools:>5} {n_distinct:>8} {n_clarif:>6} "
            f"{grounding_str:>9} "
            f"{'YES' if ok_multi_step else 'NO':>9} "
            f"{'YES' if ok_multi_tool else 'NO':>9} "
            f"{status:>6}"
        )

        if args.verbose:
            print(f"   domain={domain}  pattern={pattern}")
            print(f"   tools used: {sorted(distinct)}")
            print(f"   tool call sequence (grounding chain):")

            tool_outputs = record.get("tool_outputs", [])
            endpoint_ids = meta.get("endpoint_ids", [])

            # Build field_mappings lookup from metadata if available
            field_mappings = meta.get("field_mappings_per_step", {})

            for i, tc in enumerate(tool_calls):
                name = tc.get("name", "?")
                params = tc.get("parameters", {})
                output = tool_outputs[i]["output"] if i < len(tool_outputs) else {}
                output_fields = list(output.keys()) if isinstance(output, dict) else []

                if i == 0:
                    print(f"     step {i}: {name}  [FIRST STEP]")
                    print(f"             → input:  {json.dumps(params)}")
                    print(f"             → output fields: {output_fields}")
                else:
                    # Detect which params came from prior step output
                    prior_output = tool_outputs[i - 1]["output"] if i - 1 < len(tool_outputs) else {}
                    prior_fields = set(prior_output.keys()) if isinstance(prior_output, dict) else set()
                    grounded_params = {k: v for k, v in params.items() if k in prior_fields and v == prior_output.get(k)}
                    mock_params = {k: v for k, v in params.items() if k not in prior_fields}

                    if grounded_params:
                        grounded_from = ", ".join(f"{k}←step{i-1}" for k in grounded_params)
                        print(f"     step {i}: {name}  [GROUNDED via {grounded_from}]")
                    else:
                        print(f"     step {i}: {name}  [grounded via session memory]")

                    for k, v in params.items():
                        if k in grounded_params:
                            print(f"             ← {k}: {repr(v)}  (from step {i-1} output)")
                        else:
                            print(f"               {k}: {repr(v)}")
                    print(f"             → output fields: {output_fields}")

            if sampled_chain and actual_endpoints != sampled_chain:
                print(f"   CHAIN MISMATCH:")
                print(f"     expected: {sampled_chain}")
                print(f"     actual:   {actual_endpoints}")
            print()

        total_tool_calls += n_tools
        total_distinct_tools += n_distinct
        total_clarifications += n_clarif
        if ok_multi_step:
            multi_step_ok += 1
        if ok_multi_tool:
            multi_tool_ok += 1
        if n_clarif > 0:
            clarif_present += 1
        if grounding is not None:
            grounding_rates.append(grounding)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        combo_key = "+".join(sorted(distinct))
        tool_combo_counts[combo_key] = tool_combo_counts.get(combo_key, 0) + 1

    n = len(records)
    print("-" * 90)
    print(f"\n{'='*20} DATASET QUALITY SUMMARY {'='*20}\n")
    print(f"  Conversations analysed:      {n}")
    print(f"  PASS (all checks):           {pass_count}  ({100*pass_count//n}%)")
    print(f"  WARN (no clarifications):    {warn_count}  ({100*warn_count//n}%)")
    print(f"  FAIL (hard check failed):    {fail_count}  ({100*fail_count//n}%)")
    print(f"\n  --- Averages ---")
    print(f"  Avg tool calls per conv:     {total_tool_calls/n:.1f}")
    print(f"  Avg distinct tools per conv: {total_distinct_tools/n:.1f}")
    print(f"  Avg clarifications per conv: {total_clarifications/n:.2f}")
    if grounding_rates:
        print(f"  Avg memory grounding rate:   {sum(grounding_rates)/len(grounding_rates):.3f}")
    else:
        print(f"  Memory grounding rate:       not available")
    print(f"\n  --- Pattern Distribution ---")
    for p, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {p:<30} {count:>3}  ({100*count//n}%)")
    print(f"\n  --- Domain Distribution ---")
    for d, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {d:<30} {count:>3}")
    print(f"\n  --- Tool Combination Diversity ---")
    unique_combos = len(tool_combo_counts)
    print(f"  Unique tool combinations:    {unique_combos}")
    print(f"  Combination coverage:        {100*unique_combos//n}% of conversations unique")
    if args.verbose:
        print(f"\n  Top 10 tool combinations:")
        for combo, count in sorted(tool_combo_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {combo[:60]:<60}  x{count}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic_datagen",
        description="Offline synthetic multi-agent tool-use conversation generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build registry and graph artifacts")
    p_build.add_argument("--data", default=None, help="Path to seed tools JSON")
    p_build.add_argument("--artifacts", default="artifacts", help="Artifacts output directory")

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic conversations")
    p_gen.add_argument("--n", type=int, default=50, help="Number of conversations to generate")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p_gen.add_argument("--mode", default="sequential",
                       choices=["sequential", "multi_tool", "clarification_first", "parallel", "mixed"],
                       help="Sampling mode (default: sequential)")
    p_gen.add_argument("--output", default="output/conversations.jsonl",
                       help="Output JSONL file path")
    p_gen.add_argument("--data", default=None, help="Path to seed tools JSON")
    p_gen.add_argument(
        "--no-corpus-memory",
        "--no-cross-conversation-steering",
        action="store_true",
        help="Disable corpus memory / cross-conversation steering (for Run A in diversity experiment)",
    )
    p_gen.add_argument("--evaluate", action="store_true",
                       help="Score each conversation with LLM-as-judge inline (requires ANTHROPIC_API_KEY)")
    p_gen.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                       help="Judge model ID when --evaluate is set (default: claude-haiku-4-5-20251001)")
    p_gen.add_argument("--verbose", action="store_true", help="Show rejected conversations")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Score conversations with LLM-as-judge")
    p_eval.add_argument("--input", required=True, help="Path to conversations JSONL")
    p_eval.add_argument("--output", default="output/evaluated.jsonl",
                        help="Output path for evaluated JSONL (default: output/evaluated.jsonl)")
    p_eval.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Judge model ID (default: claude-haiku-4-5-20251001)")
    p_eval.add_argument("--threshold", type=float, default=3.5,
                        help="Pass threshold 1.0–5.0 (default: 3.5)")
    p_eval.add_argument("--repair", action="store_true",
                        help="Attempt LLM repair on failing conversations")
    p_eval.add_argument("--max-repairs", type=int, default=2,
                        help="Max repair attempts per conversation (default: 2)")
    p_eval.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    p_eval.add_argument("--verbose", action="store_true",
                        help="Log per-conversation pass/fail decisions")

    # validate
    p_val = sub.add_parser("validate", help="Validate a generated JSONL dataset")
    p_val.add_argument("--input", required=True, help="Path to JSONL file to validate")

    # metrics
    p_met = sub.add_parser("metrics", help="Compute metrics on a generated dataset")
    p_met.add_argument("--input", required=True, help="Path to JSONL file")
    p_met.add_argument("--compare", default=None, help="Optional second JSONL file for side-by-side comparison")

    # inspect
    p_insp = sub.add_parser("inspect", help="Per-conversation PDF compliance breakdown")
    p_insp.add_argument("--input", required=True, help="Path to JSONL file to inspect")
    p_insp.add_argument("--verbose", action="store_true",
                        help="Show tool call sequence per conversation")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    main()
