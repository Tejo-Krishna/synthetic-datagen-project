"""
sampler/sampler.py
------------------
SamplerAgent — the structural exploration engine.

Responsibilities:
  - Load projected graph + registry + sampler config
  - Use a seeded local RNG (never the global random state)
  - Select valid start nodes
  - Dispatch to strategy implementations
  - Enforce chain constraints (length, distinct tools, uniqueness)
  - Detect missing_required_param clarification steps
  - Produce SampledChain objects

Must NOT:
  - Generate conversation text
  - Fill concrete argument values
  - Call memory
  - Execute tools
  - Detect intent_ambiguity (that's the Planner's job)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Iterator

from synthetic_datagen.common.types import SampledChain, ClarificationStep
from synthetic_datagen.graph.projected_graph import ProjectedGraph
from synthetic_datagen.graph.registry import ToolRegistry
from synthetic_datagen.sampler.config import SamplerConfig, load_sampler_config
from synthetic_datagen.sampler.strategies import (
    run_strategy,
    WalkResult,
    _detect_clarification_steps,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SamplerExhaustedError(Exception):
    """
    Raised when the retry budget is exhausted without producing
    the requested number of valid unique chains.

    Attributes:
        requested: number of chains requested
        produced: number of valid chains produced before exhaustion
    """
    def __init__(self, requested: int, produced: int):
        self.requested = requested
        self.produced = produced
        super().__init__(
            f"SamplerAgent exhausted retry budget: "
            f"produced {produced}/{requested} valid unique chains. "
            f"Try increasing max_retries, relaxing constraints, or expanding the seed data."
        )


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

def _classify_pattern(
    walk: WalkResult,
    registry: ToolRegistry,
    config: SamplerConfig,
) -> str:
    """
    Classify the pattern type of a completed walk.

    Priority order:
      1. parallel   — explicit branch structure
      2. search_then_action — search/retrieve step followed by create/execute
      3. information_then_decision — retrieve steps followed by compare/select
      4. multi_tool_chain — multiple distinct tools
      5. sequential_multi_step — default

    sampling_mode and pattern_type may differ (e.g. multi_tool mode
    might produce a sequential pattern if tool diversity is low).
    """
    if len(walk.endpoint_ids) <= config.short_chain_max_length:
        return "short_chain"

    if walk.branches is not None:
        return "parallel"

    endpoints = walk.endpoint_ids
    if not endpoints:
        return "sequential_multi_step"

    intents = []
    for eid in endpoints:
        ep = registry.get_endpoint(eid)
        intents.append(ep.intent if ep else "unknown")

    tools = [eid.split("::")[0] for eid in endpoints]

    # search_then_action: search/retrieve followed by create/execute
    has_search = any(i in ("search", "retrieve", "lookup") for i in intents)
    has_action = any(i in ("create", "execute", "update", "delete") for i in intents)
    if has_search and has_action:
        # Ensure action comes after search
        first_search = next((i for i, intent in enumerate(intents) if intent in ("search", "retrieve")), None)
        first_action = next((i for i, intent in enumerate(intents) if intent in ("create", "execute")), None)
        if first_search is not None and first_action is not None and first_action > first_search:
            return "search_then_action"

    # information_then_decision: retrieve + compare/select
    has_info = any(i in ("retrieve", "search") for i in intents)
    has_decision = any(i in ("compare", "summarize", "select") for i in intents)
    if has_info and has_decision:
        return "information_then_decision"

    # multi_tool_chain
    if len(set(tools)) > 1:
        return "multi_tool_chain"

    return "sequential_multi_step"


# ---------------------------------------------------------------------------
# Uniqueness key
# ---------------------------------------------------------------------------

def _chain_key(chain: SampledChain) -> tuple:
    """
    Uniqueness key for a SampledChain.
    Richer than endpoint IDs alone — same endpoints with different
    clarification positions or patterns are considered distinct.
    """
    return (
        tuple(chain.endpoint_ids),
        chain.pattern_type,
        tuple(cs.step_index for cs in chain.clarification_steps),
    )


# ---------------------------------------------------------------------------
# WalkResult -> SampledChain assembly
# ---------------------------------------------------------------------------

def _assemble_chain(
    walk: WalkResult,
    registry: ToolRegistry,
    config: SamplerConfig,
) -> SampledChain:
    """Convert a WalkResult into a fully assembled SampledChain."""

    # Detect clarification steps from all endpoints in the flattened view
    clarification_steps = _detect_clarification_steps(
        endpoint_ids=walk.endpoint_ids,
        transitions=walk.transitions,
        registry=registry,
        user_natural_params=config.user_natural_params,
    )

    # Deduplicated ordered tool list
    tool_ids = list(dict.fromkeys(eid.split("::")[0] for eid in walk.endpoint_ids))

    # Classify pattern
    pattern_type = _classify_pattern(walk, registry, config)

    return SampledChain(
        endpoint_ids=walk.endpoint_ids,
        tool_ids=tool_ids,
        transitions=walk.transitions,
        pattern_type=pattern_type,
        sampling_mode=walk.mode,
        clarification_steps=clarification_steps,
        root_endpoint_id=walk.root_endpoint_id,
        branches=walk.branches,
        merge_endpoint_id=walk.merge_endpoint_id,
    )


# ---------------------------------------------------------------------------
# SamplerAgent
# ---------------------------------------------------------------------------

class SamplerAgent:
    """
    Graph-driven structural exploration engine.

    Produces SampledChain objects from the projected sampler graph.
    Uses a local seeded RNG — never touches global random state.

    Usage:
        agent = SamplerAgent(projected, registry)
        chain = agent.sample_chain(mode="sequential", seed=42)
        chains = agent.sample_chains(n=50, mode="sequential", seed=42)
    """

    def __init__(
        self,
        projected: ProjectedGraph,
        registry: ToolRegistry,
        config: SamplerConfig | None = None,
    ):
        self.projected = projected
        self.registry = registry
        self.config = config or load_sampler_config()

        if not projected.nodes:
            raise ValueError("SamplerAgent: projected graph has no nodes")
        if not registry.endpoints_by_id:
            raise ValueError("SamplerAgent: registry has no endpoints")

    def sample_chain(
        self,
        mode: str = "sequential",
        seed: int | None = None,
    ) -> SampledChain:
        """
        Sample a single valid SampledChain.

        Args:
            mode: sampling mode — sequential, multi_tool, clarification_first, parallel
            seed: optional seed for reproducibility

        Returns:
            A valid SampledChain satisfying all constraints.

        Raises:
            ValueError: if mode is not supported
            SamplerExhaustedError: if retry budget exhausted without valid chain
        """
        if not self.config.is_mode_supported(mode):
            raise ValueError(
                f"Mode '{mode}' not supported. Supported: {self.config.supported_modes}"
            )

        rng = random.Random(seed)

        for attempt in range(self.config.max_retries):
            walk = run_strategy(mode, self.projected, self.registry, self.config, rng)
            if walk is not None:
                chain = _assemble_chain(walk, self.registry, self.config)
                if self._is_valid(chain):
                    return chain

        raise SamplerExhaustedError(requested=1, produced=0)

    def sample_chains(
        self,
        n: int,
        mode: str = "sequential",
        seed: int | None = None,
        unique: bool | None = None,
    ) -> list[SampledChain]:
        """
        Sample n valid SampledChain objects.

        Args:
            n: number of chains to produce
            mode: sampling mode
            seed: optional seed — same seed + same graph = same ordered set of chains
            unique: override config.unique_chains for this call

        Returns:
            List of n valid chains (fewer if graph is too sparse and budget exhausted)

        Raises:
            SamplerExhaustedError: if retry budget exhausted with fewer than n chains
        """
        if not self.config.is_mode_supported(mode):
            raise ValueError(f"Mode '{mode}' not supported.")

        enforce_unique = unique if unique is not None else self.config.unique_chains
        rng = random.Random(seed)
        seen: set[tuple] = set()
        chains: list[SampledChain] = []
        total_attempts = 0
        max_total = self.config.max_retries * n

        while len(chains) < n and total_attempts < max_total:
            total_attempts += 1
            walk = run_strategy(mode, self.projected, self.registry, self.config, rng)
            if walk is None:
                continue

            chain = _assemble_chain(walk, self.registry, self.config)
            if not self._is_valid(chain):
                continue

            if enforce_unique:
                key = _chain_key(chain)
                if key in seen:
                    continue
                seen.add(key)

            chains.append(chain)

        if len(chains) < n:
            print(
                f"[sampler] Warning: produced {len(chains)}/{n} chains "
                f"(retry budget exhausted after {total_attempts} attempts). "
                f"Consider expanding seed data or relaxing constraints."
            )
            if len(chains) == 0:
                raise SamplerExhaustedError(requested=n, produced=0)

        return chains

    def sample_mixed(
        self,
        n: int,
        seed: int | None = None,
        mode_weights: dict[str, float] | None = None,
    ) -> list[SampledChain]:
        """
        Sample n chains using a weighted mix of all supported modes.

        Args:
            n: total number of chains
            seed: reproducibility seed
            mode_weights: optional {mode: weight} dict (default: equal weights)

        Returns:
            Mixed list of SampledChain objects.
        """
        modes = self.config.supported_modes
        if mode_weights is None:
            weights = [1.0] * len(modes)
        else:
            weights = [mode_weights.get(m, 1.0) for m in modes]

        rng = random.Random(seed)
        seen: set[tuple] = set()
        chains: list[SampledChain] = []
        total_attempts = 0
        max_total = self.config.max_retries * n

        while len(chains) < n and total_attempts < max_total:
            total_attempts += 1
            mode = rng.choices(modes, weights=weights, k=1)[0]
            use_short_chain = rng.random() < self.config.short_chain_weight

            if use_short_chain:
                short_config = replace(
                    self.config,
                    min_chain_length=1,
                    max_chain_length=self.config.short_chain_max_length,
                    min_distinct_tools=1,
                )
                walk = run_strategy(mode, self.projected, self.registry, short_config, rng)
            else:
                walk = run_strategy(mode, self.projected, self.registry, self.config, rng)

            if walk is None:
                continue

            chain = _assemble_chain(walk, self.registry, self.config)
            if not self._is_valid(chain):
                continue

            if self.config.unique_chains:
                key = _chain_key(chain)
                if key in seen:
                    continue
                seen.add(key)

            chains.append(chain)

        return chains

    def _is_valid(self, chain: SampledChain) -> bool:
        """Check all hard constraints on a chain."""
        # Length constraint
        if chain.pattern_type == "short_chain":
            if len(chain.endpoint_ids) < 1:
                return False
            if len(chain.endpoint_ids) > self.config.short_chain_max_length:
                return False
        else:
            if len(chain.endpoint_ids) < self.config.min_chain_length:
                return False
            if len(chain.endpoint_ids) > self.config.max_chain_length:
                return False

            # Distinct tools constraint
            if len(set(chain.tool_ids)) < self.config.min_distinct_tools:
                return False

        # Category coherence — reject chains spanning more than max_distinct_categories
        # distinct categories (spec requires rejecting 4+ category chains).
        # Account is the only exempt category — it's authentication/profile and
        # naturally pairs with any domain.
        _UTILITY_CATEGORIES = {"Account"}
        categories = set()
        for eid in chain.endpoint_ids:
            ep = self.registry.get_endpoint(eid)
            if ep and ep.category not in _UTILITY_CATEGORIES:
                categories.add(ep.category)
        if len(categories) > self.config.max_distinct_categories:
            return False

        # Dominant-category coherence: for multi-category chains, at least one
        # non-utility category must appear 2+ times. This ensures a clear "primary"
        # domain (e.g. Travel=2 + Finance=1 is coherent; Career=1+Travel=1+Shopping=1
        # is not — all three are minority categories with no dominant theme).
        category_counts: dict[str, int] = {}
        for eid in chain.endpoint_ids:
            ep = self.registry.get_endpoint(eid)
            if ep and ep.category not in _UTILITY_CATEGORIES:
                category_counts[ep.category] = category_counts.get(ep.category, 0) + 1

        if len(category_counts) > 1:
            # Multi-category chain must have a dominant category (≥2 tools)
            if max(category_counts.values()) < 2:
                return False

        # Entry-tool coherence — the first tool's category must be the dominant one
        # so the user's opening request matches the first tool call.
        if chain.endpoint_ids and category_counts:
            first_ep = self.registry.get_endpoint(chain.endpoint_ids[0])
            if first_ep and first_ep.category not in _UTILITY_CATEGORIES:
                max_count = max(category_counts.values())
                dominant_categories = {c for c, cnt in category_counts.items() if cnt == max_count}
                if first_ep.category not in dominant_categories:
                    return False

        # Clarification-first constraint
        if not self.config.allow_clarification_first:
            if chain.clarification_steps and chain.clarification_steps[0].step_index == 0:
                return False

        # All endpoint IDs must exist in registry
        for eid in chain.endpoint_ids:
            if self.registry.get_endpoint(eid) is None:
                return False

        return True

    def iter_chains(
        self,
        mode: str = "sequential",
        seed: int | None = None,
    ) -> Iterator[SampledChain]:
        """
        Infinite iterator of valid unique chains. Useful for streaming generation.
        Respects uniqueness config. Stops when graph is exhausted.
        """
        rng = random.Random(seed)
        seen: set[tuple] = set()
        consecutive_failures = 0
        max_failures = self.config.max_retries

        while consecutive_failures < max_failures:
            walk = run_strategy(mode, self.projected, self.registry, self.config, rng)
            if walk is None:
                consecutive_failures += 1
                continue

            chain = _assemble_chain(walk, self.registry, self.config)
            if not self._is_valid(chain):
                consecutive_failures += 1
                continue

            if self.config.unique_chains:
                key = _chain_key(chain)
                if key in seen:
                    consecutive_failures += 1
                    continue
                seen.add(key)

            consecutive_failures = 0
            yield chain

    def stats(self) -> dict:
        """Return summary statistics about the sampler's graph."""
        return {
            "node_count": self.projected.node_count,
            "edge_count": self.projected.edge_count,
            "entry_node_count": len(self.projected.entry_nodes),
            "tool_count": self.registry.tool_count,
            "endpoint_count": self.registry.endpoint_count,
            "config": repr(self.config),
        }
