"""
Planner Agent — Main Entry Point
Orchestrates the full planning pipeline:
  1. Validate input chain
  2. Query corpus memory
  3. Build rule-based structural scaffold
  4. Detect clarification candidates
  5. Derive novelty hints
  6. Generate LLM narrative
  7. Merge scaffold + narrative
  8. Self-validate
  9. Retry on failure
 10. Return typed PlannerResult

Wiring notes for this project:
  - Accepts SampledChain from synthetic_datagen.common.types (your sampler output).
    Use adapt_sampled_chain() from registry_adapter.py to convert before calling plan().
  - memory_store must be a MemoryStore from synthetic_datagen.memory.store.
    Uses .search(query, scope, top_k) — NOT .query().
  - registry is a dict[(tool_id, endpoint_id) -> RegistryEndpointMetadata].
    Use build_planner_registry() from registry_adapter.py to convert ToolRegistry.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .models import (
    SampledToolChain,
    RegistryEndpointMetadata,
    CorpusSummary,
    StructuredConversationPlan,
    PlannerResult,
    PlannerError,
    PlannerInvalidInputError,
    PlannerChainTooShortError,
    PlannerOutputValidationError,
    PlannerRetryExhaustedError,
    PlannerConfigError,
    PlannerDependencyError,
    RESAMPLE_ON,
    HARD_STOP_ON,
)
from .scaffold import (
    build_step_scaffold,
    detect_clarification_candidates,
    build_clarification_points,
    derive_novelty_hints,
)
from .narrative import (
    NarrativeRequest,
    build_narrative_prompt,
    call_llm,
    parse_narrative_response,
    merge_narrative_into_steps,
    build_summary_seed_fields,
)
from .validator import (
    validate_sampled_tool_chain,
    validate_conversation_plan,
    raise_if_invalid,
)
from .config import PlannerConfig, load_planner_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory store interface (duck-typed)
# ---------------------------------------------------------------------------
# Expects synthetic_datagen.memory.store.MemoryStore which implements:
#   .search(query: str, scope: str, top_k: int) -> list[dict]
#
# Each dict has keys: id, memory, metadata, score.
# The planner converts these dicts into CorpusSummary objects internally.


def _dict_to_corpus_summary(entry: dict) -> CorpusSummary:
    """
    Convert a MemoryStore search result dict into a CorpusSummary.

    MemoryStore.search() returns dicts with keys: id, memory, metadata, score.
    The planner's internal CorpusSummary shape is extracted from metadata
    where available, with the memory string as the content field.
    """
    metadata = entry.get("metadata", {})
    return CorpusSummary(
        content=entry.get("memory", ""),
        conversation_id=metadata.get("conversation_id"),
        tools=metadata.get("tools", []),
        pattern_type=metadata.get("pattern_type"),
        domain=metadata.get("domain"),
        conversation_style=metadata.get("conversation_style"),
    )


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Converts a SampledToolChain into a structured StructuredConversationPlan.

    Dependencies:
        llm_backend   — any object with .complete(prompt: str) -> str
        memory_store  — MemoryStore from synthetic_datagen.memory.store
                        (must implement .search(query, scope, top_k) -> list[dict])
        registry      — dict[(tool_id, endpoint_id) -> RegistryEndpointMetadata] | None
                        Build this with registry_adapter.build_planner_registry()
        config        — PlannerConfig from planner/config.py
                        Load with load_planner_config() or pass directly.

    Typical usage:
        from synthetic_datagen.planner.config import load_planner_config
        from synthetic_datagen.planner.registry_adapter import (
            adapt_sampled_chain,
            build_planner_registry,
        )

        config           = load_planner_config()
        planner_registry = build_planner_registry(tool_registry)
        planner          = PlannerAgent(
            llm_backend=my_llm,
            memory_store=memory_store,
            registry=planner_registry,
            config=config,
        )

        # Convert SampledChain → SampledToolChain before calling plan()
        adapted_chain = adapt_sampled_chain(sampled_chain, chain_id="c001", seed=42)
        result        = planner.plan(adapted_chain, plan_id="plan_001")
    """

    def __init__(
        self,
        llm_backend: Any,
        memory_store: Any,
        registry: Optional[dict[tuple[str, str], RegistryEndpointMetadata]] = None,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        if llm_backend is None:
            raise PlannerConfigError("llm_backend is required.")
        if memory_store is None:
            raise PlannerConfigError("memory_store is required.")

        self.llm = llm_backend
        self.memory = memory_store
        self.registry = registry
        self.config = config or PlannerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        chain: SampledToolChain,
        plan_id: str,
        corpus_query_limit: int = 10,
    ) -> PlannerResult:
        """
        Main entry point. Returns a PlannerResult always — never raises.
        The orchestrator inspects result.success and result.error_code to decide
        whether to retry, resample, or hard stop.

        Args:
            chain:               SampledToolChain (use adapt_sampled_chain() to
                                 convert from SampledChain if needed)
            plan_id:             Orchestrator-assigned ID — must not be empty
            corpus_query_limit:  How many corpus memory entries to retrieve
        """
        corpus_summaries: list[CorpusSummary] = []

        try:
            # --- Step 0: Validate injected plan_id ---
            if not plan_id or not plan_id.strip():
                raise PlannerInvalidInputError(
                    "plan_id must be injected by the orchestrator and must not be empty."
                )

            # --- Step 1: Validate input chain ---
            self._validate_input_chain(chain)

            # --- Step 2: Query corpus memory ---
            corpus_summaries = self._query_corpus_memory(corpus_query_limit)

            # --- Steps 3–8: Attempt planning with retries ---
            plan = self._plan_with_retries(
                chain=chain,
                plan_id=plan_id,
                corpus_summaries=corpus_summaries,
            )

            return PlannerResult(
                success=True,
                plan=plan,
                retrieved_corpus_summaries=corpus_summaries,
                num_retries_used=0,
                error_code=None,
                error_message=None,
            )

        except PlannerError as e:
            error_code = getattr(e, "error_code", "PLANNER_ERROR")
            logger.error("PlannerAgent failed: [%s] %s", error_code, e)
            return PlannerResult(
                success=False,
                plan=None,
                retrieved_corpus_summaries=corpus_summaries,
                num_retries_used=0,
                error_code=error_code,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _check_plan_coherence(
        self,
        user_goal: str,
        enriched_steps: list,
    ) -> tuple[bool, str | None]:
        """
        Ask the LLM whether the plan is semantically coherent.

        Checks two things:
          1. Execution order — does each step logically follow from the previous?
             (e.g. get_job_details should come before create_event for an interview)
          2. Tool relevance — does every tool actually contribute to the user_goal?
             (e.g. get_product_details returning headphones in a flight-booking chain)

        Returns (True, None) if coherent, (False, reason) if not.
        Defaults to (True, None) on any LLM error so generation is never blocked
        by a transient API failure.
        """
        step_lines = "\n".join(
            f"  Step {s.step_index}: {s.endpoint_id} — {s.purpose}"
            for s in enriched_steps
        )
        prompt = (
            f"You are reviewing a planned AI assistant conversation for CRITICAL failures only.\n\n"
            f"IMPORTANT CONTEXT: The tool execution order (step indices) is FIXED and cannot be changed. "
            f"The narrative has been written to justify this order. Accept the ordering as given.\n\n"
            f"User's goal: {user_goal}\n\n"
            f"Planned tool execution order:\n{step_lines}\n\n"
            f"Reply INCOHERENT only if BOTH of these are true simultaneously:\n"
            f"  A. A tool is from a completely unrelated domain (e.g. job_search in a restaurant chain)\n"
            f"  AND\n"
            f"  B. No reasonable interpretation of the user_goal can connect that tool to the goal\n\n"
            f"Reply COHERENT for everything else, including:\n"
            f"- Ordering you might personally prefer differently\n"
            f"- Steps that seem redundant but could have a reason\n"
            f"- Tools that are adjacent to the goal domain (e.g. weather + travel, geocoding + directions)\n"
            f"- Any case where the step purposes plausibly justify the order\n\n"
            f"If the plan passes: reply with exactly: COHERENT\n"
            f"If a tool is truly from an unrelated domain: reply with: INCOHERENT: <one sentence>\n"
            f"No other text."
        )
        try:
            answer = self.llm.complete(prompt).strip()
            if answer.upper().startswith("INCOHERENT"):
                issue = answer.split(":", 1)[-1].strip() if ":" in answer else answer
                return False, issue
            return True, None
        except Exception:
            return True, None  # fail open — never block on LLM error

    def _validate_input_chain(self, chain: SampledToolChain) -> None:
        result = validate_sampled_tool_chain(chain, min_steps=self.config.min_steps)
        if result.warnings:
            for w in result.warnings:
                logger.warning("Input chain warning: %s", w)
        if not result.valid:
            if any("minimum required" in e for e in result.errors):
                raise PlannerChainTooShortError("; ".join(result.errors))
            raise PlannerInvalidInputError("; ".join(result.errors))

    def _query_corpus_memory(self, limit: int) -> list[CorpusSummary]:
        """
        Query corpus memory using MemoryStore.search().

        FIX APPLIED: uses .search(query, scope, top_k) to match
        synthetic_datagen.memory.store.MemoryStore's actual interface.
        The original .query(scope, limit) call has been replaced.
        """
        try:
            raw_results = self.memory.search(
                query=self.config.corpus_query_term,
                scope="corpus",
                top_k=limit,
            )
            return [_dict_to_corpus_summary(entry) for entry in raw_results]
        except Exception as e:
            if self.config.memory_strict_mode:
                raise PlannerDependencyError(
                    f"Memory backend unavailable (strict mode): {e}"
                ) from e
            logger.warning("Corpus memory query failed (degraded): %s", e)
            return []

    def _plan_with_retries(
        self,
        chain: SampledToolChain,
        plan_id: str,
        corpus_summaries: list[CorpusSummary],
    ) -> StructuredConversationPlan:
        last_error: Exception | None = None
        retries_used = 0

        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                retries_used += 1
                logger.info("Planner retry %d/%d", attempt, self.config.max_retries)

            try:
                plan = self._build_plan(
                    chain=chain,
                    plan_id=plan_id,
                    corpus_summaries=corpus_summaries,
                )
                vr = validate_conversation_plan(
                    plan=plan,
                    source_chain=chain,
                    min_steps=self.config.min_steps,
                    min_distinct_tools=self.config.min_distinct_tools,
                )
                if vr.warnings:
                    for w in vr.warnings:
                        logger.warning("Plan validation warning: %s", w)
                raise_if_invalid(vr, context=f"attempt={attempt}")
                return plan

            except PlannerOutputValidationError as e:
                last_error = e
                logger.warning("Plan validation failed on attempt %d: %s", attempt, e)
                continue

            except (ValueError, KeyError) as e:
                last_error = PlannerOutputValidationError(str(e))
                logger.warning("Narrative parse error on attempt %d: %s", attempt, e)
                continue

        raise PlannerRetryExhaustedError(
            f"Planning failed after {self.config.max_retries + 1} attempt(s). "
            f"Last error: {last_error}"
        )

    def _build_plan(
        self,
        chain: SampledToolChain,
        plan_id: str,
        corpus_summaries: list[CorpusSummary],
    ) -> StructuredConversationPlan:

        # --- Registry availability check ---
        if self.registry is None and self.config.registry_strict_mode:
            raise PlannerDependencyError(
                "Registry is required in strict mode but was not provided."
            )

        # --- Rule-based scaffold ---
        scaffold_steps = build_step_scaffold(chain)

        # --- Clarification detection ---
        # Skipped if use_registry_metadata=False in config (registry not required)
        if self.config.use_registry_metadata:
            candidates = detect_clarification_candidates(chain, self.registry)
        else:
            candidates = []
        clarification_points = build_clarification_points(candidates)

        # --- Diversity hints ---
        novelty_hints = derive_novelty_hints(corpus_summaries)

        # --- LLM narrative ---
        req = NarrativeRequest(
            seed=chain.seed,
            chain=chain,
            scaffold_steps=scaffold_steps,
            clarification_points=clarification_points,
            novelty_hints=novelty_hints,
            registry=self.registry,
            corpus_summaries=corpus_summaries,
        )
        prompt = build_narrative_prompt(req)
        raw_response = call_llm(prompt, self.llm)
        narrative = parse_narrative_response(raw_response, scaffold_steps)

        # --- Merge scaffold + narrative ---
        enriched_steps = merge_narrative_into_steps(
            scaffold_steps, narrative, clarification_points
        )
        tools_used = sorted({s.tool_id for s in enriched_steps})
        summary_seed = build_summary_seed_fields(
            domain=narrative.domain,
            pattern_type=chain.pattern_type,
            tools_used=tools_used,
            conversation_style=narrative.conversation_style,
            clarification_points=clarification_points,
        )

        # --- Coherence check ---
        # Ask the LLM whether the execution order makes sense and whether
        # the user_goal is achievable with the listed tools. If not, raise
        # so the retry loop can attempt a fresh narrative or abandon the chain.
        coherent, issue = self._check_plan_coherence(
            user_goal=narrative.user_goal,
            enriched_steps=enriched_steps,
        )
        if not coherent:
            raise PlannerOutputValidationError(
                f"Plan coherence check failed: {issue}"
            )

        return StructuredConversationPlan(
            plan_id=plan_id,
            chain_id=chain.chain_id,
            seed=chain.seed,
            domain=narrative.domain,
            user_goal=narrative.user_goal,
            pattern_type=chain.pattern_type,
            conversation_style=narrative.conversation_style,
            style_notes=narrative.style_notes,
            tools_used=tools_used,
            steps=enriched_steps,
            clarification_points=clarification_points,
            summary_seed_fields=summary_seed,
        )
