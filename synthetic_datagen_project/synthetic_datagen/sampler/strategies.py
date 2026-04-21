"""
sampler/strategies.py
---------------------
Walk strategies for the SamplerAgent.

Each strategy takes the projected graph, registry, config, and a seeded RNG,
and returns a raw walk result that the SamplerAgent uses to build a SampledChain.

Strategies:
  - sequential:          weighted linear walk, prefers data_link edges
  - multi_tool:          linear walk with cross-tool bias enforced
  - clarification_first: allows start nodes with unsourced required params
  - parallel:            branch-merge pattern with explicit ParallelBranch structure

No strategy generates text, fills arguments, or calls memory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from synthetic_datagen.common.types import (
    FieldMapping, Transition, ParallelBranch, ClarificationStep, SampledChain
)
from synthetic_datagen.graph.projected_graph import ProjectedGraph, ProjectedEdge
from synthetic_datagen.graph.registry import ToolRegistry
from synthetic_datagen.sampler.config import SamplerConfig


# ---------------------------------------------------------------------------
# Walk result — internal intermediate before SampledChain is assembled
# ---------------------------------------------------------------------------

@dataclass
class WalkResult:
    """
    Raw output of a strategy walk.
    The SamplerAgent converts this into a SampledChain.
    """
    endpoint_ids: list[str]
    transitions: list[Transition]
    branches: list[ParallelBranch] | None = None
    root_endpoint_id: str | None = None
    merge_endpoint_id: str | None = None
    mode: str = "sequential"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _effective_weight(
    edge: ProjectedEdge,
    current_tool: str,
    cross_tool_bias: float,
) -> float:
    """
    Apply multiplicative cross-tool bias when edge crosses tool boundary.
    effective_weight = edge.weight * (1 + cross_tool_bias) if crossing tools
    """
    target_tool = edge.to_endpoint.split("::")[0]
    if target_tool != current_tool:
        return edge.weight * (1.0 + cross_tool_bias)
    return edge.weight


def _select_neighbor(
    edges: list[ProjectedEdge],
    visited: set[str],
    current_tool: str,
    cross_tool_bias: float,
    rng: random.Random,
    exclude_tools: set[str] | None = None,
) -> ProjectedEdge | None:
    """
    Weighted random selection of next edge, excluding visited endpoints.
    Returns None if no valid candidates exist.
    """
    candidates = [
        e for e in edges
        if e.to_endpoint not in visited
        and (exclude_tools is None or e.to_endpoint.split("::")[0] not in exclude_tools)
    ]
    if not candidates:
        return None

    weights = [_effective_weight(e, current_tool, cross_tool_bias) for e in candidates]
    total = sum(weights)
    if total <= 0:
        return rng.choice(candidates)

    return rng.choices(candidates, weights=weights, k=1)[0]


def _build_transition(edge: ProjectedEdge) -> Transition:
    """Convert a ProjectedEdge into a Transition (strips weight and provenance)."""
    # Determine executability: data_link edges are always executable
    # semantic/category edges are executable only if they have field mappings
    # (which only data_link edges do, so this is effectively: data_link=True, others depend)
    is_executable = (edge.edge_type == "data_link") or bool(edge.field_mappings)

    return Transition(
        from_endpoint=edge.from_endpoint,
        to_endpoint=edge.to_endpoint,
        edge_type=edge.edge_type,
        field_mappings=edge.field_mappings,
        matched_concepts=edge.matched_concepts,
        is_executable=is_executable,
        # weight and provenance_path intentionally NOT copied
    )


def _detect_clarification_steps(
    endpoint_ids: list[str],
    transitions: list[Transition],
    registry: ToolRegistry,
    user_natural_params: set[str],
) -> list[ClarificationStep]:
    """
    Detect steps that require missing_required_param clarification.

    For each step, determine which required params are:
      - covered by prior step's field_mappings (from transitions)
      - user-natural (user provides at chain start)
      - neither — needs clarification

    Only detects missing_required_param. Planner handles intent_ambiguity.
    """
    clarification_steps: list[ClarificationStep] = []

    # Track which fields have been produced so far
    produced_fields: set[str] = set()

    for step_idx, endpoint_id in enumerate(endpoint_ids):
        ep = registry.get_endpoint(endpoint_id)
        if ep is None:
            continue

        required_params = {p.name for p in ep.parameters if p.required}
        if not required_params:
            continue

        # Fields available from prior transitions
        if step_idx > 0:
            # Add return fields of the prior endpoint
            prior_ep = registry.get_endpoint(endpoint_ids[step_idx - 1])
            if prior_ep:
                produced_fields.update(prior_ep.returns_fields)
            # Add explicitly mapped fields from the prior transition (if exists)
            if step_idx - 1 < len(transitions):
                prior_transition = transitions[step_idx - 1]
                for fm in prior_transition.field_mappings:
                    produced_fields.add(fm.target_param)

        # Determine unsourced required params
        unsourced: list[str] = []
        for param_name in required_params:
            if param_name in produced_fields:
                continue
            if param_name.lower() in user_natural_params:
                continue
            unsourced.append(param_name)

        if unsourced:
            clarification_steps.append(ClarificationStep(
                step_index=step_idx,
                reason="missing_required_param",
                missing_params=unsourced,
            ))

    return clarification_steps


# ---------------------------------------------------------------------------
# Strategy 1: Sequential
# ---------------------------------------------------------------------------

def sequential_walk(
    projected: ProjectedGraph,
    registry: ToolRegistry,
    config: SamplerConfig,
    rng: random.Random,
    start_node: str | None = None,
) -> WalkResult | None:
    """
    Weighted linear walk. Prefers data_link edges.
    Enforces min_chain_length, max_chain_length, min_distinct_tools.
    """
    target_length = rng.randint(config.min_chain_length, config.max_chain_length)

    # Choose start
    if start_node:
        current = start_node
    elif projected.entry_nodes:
        current = rng.choice(projected.entry_nodes)
    else:
        current = rng.choice(list(projected.nodes.keys()))

    visited: set[str] = {current}
    chain: list[str] = [current]
    transitions: list[Transition] = []

    for _ in range(target_length - 1):
        edges = projected.get_neighbors(current)
        current_tool = current.split("::")[0]

        edge = _select_neighbor(edges, visited, current_tool, config.cross_tool_bias, rng)
        if edge is None:
            break

        transitions.append(_build_transition(edge))
        current = edge.to_endpoint
        visited.add(current)
        chain.append(current)

    # Enforce minimum length
    if len(chain) < config.min_chain_length:
        return None

    # Enforce minimum distinct tools
    tools_used = list(dict.fromkeys(eid.split("::")[0] for eid in chain))
    if len(set(tools_used)) < config.min_distinct_tools:
        return None

    return WalkResult(
        endpoint_ids=chain,
        transitions=transitions,
        mode="sequential",
    )


# ---------------------------------------------------------------------------
# Strategy 2: Multi-tool
# ---------------------------------------------------------------------------

def multi_tool_walk(
    projected: ProjectedGraph,
    registry: ToolRegistry,
    config: SamplerConfig,
    rng: random.Random,
    start_node: str | None = None,
) -> WalkResult | None:
    """
    Walk that actively biases toward crossing tool boundaries at each step.
    Ensures min_distinct_tools by tracking and favoring unseen tools.
    """
    target_length = rng.randint(config.min_chain_length, config.max_chain_length)

    if start_node:
        current = start_node
    elif projected.entry_nodes:
        current = rng.choice(projected.entry_nodes)
    else:
        current = rng.choice(list(projected.nodes.keys()))

    visited: set[str] = {current}
    chain: list[str] = [current]
    transitions: list[Transition] = []
    tools_seen: set[str] = {current.split("::")[0]}

    for step in range(target_length - 1):
        edges = projected.get_neighbors(current)
        current_tool = current.split("::")[0]

        # For first few steps, strongly prefer crossing tools
        need_new_tool = len(tools_seen) < config.min_distinct_tools

        # Build candidates — prefer edges to unseen tools when needed
        candidates_new_tool = [
            e for e in edges
            if e.to_endpoint not in visited and e.to_endpoint.split("::")[0] not in tools_seen
        ]
        candidates_any = [
            e for e in edges
            if e.to_endpoint not in visited
        ]

        pool = candidates_new_tool if (need_new_tool and candidates_new_tool) else candidates_any
        if not pool:
            break

        weights = [_effective_weight(e, current_tool, config.cross_tool_bias) for e in pool]
        total = sum(weights)
        if total <= 0:
            edge = rng.choice(pool)
        else:
            edge = rng.choices(pool, weights=weights, k=1)[0]

        transitions.append(_build_transition(edge))
        current = edge.to_endpoint
        visited.add(current)
        chain.append(current)
        tools_seen.add(current.split("::")[0])

    if len(chain) < config.min_chain_length:
        return None

    tools_used = list(dict.fromkeys(eid.split("::")[0] for eid in chain))
    if len(set(tools_used)) < config.min_distinct_tools:
        return None

    return WalkResult(
        endpoint_ids=chain,
        transitions=transitions,
        mode="multi_tool",
    )


# ---------------------------------------------------------------------------
# Strategy 3: Clarification-first
# ---------------------------------------------------------------------------

def clarification_first_walk(
    projected: ProjectedGraph,
    registry: ToolRegistry,
    config: SamplerConfig,
    rng: random.Random,
) -> WalkResult | None:
    """
    Walk that deliberately starts from a node with unsourced required params.
    This produces chains that require clarification at step 0.
    Falls back to sequential if no clarification-requiring start is found.
    """
    user_natural = config.user_natural_params

    # Find endpoints that need clarification at start (not entry-eligible)
    clarification_starts = [
        eid for eid in projected.nodes
        if eid not in projected.entry_nodes
        and projected.get_neighbors(eid)  # must have outgoing edges
    ]

    if clarification_starts:
        start = rng.choice(clarification_starts)
    elif projected.entry_nodes:
        start = rng.choice(projected.entry_nodes)
    else:
        start = rng.choice(list(projected.nodes.keys()))

    # Do a sequential walk from this start
    result = sequential_walk(projected, registry, config, rng, start_node=start)
    if result:
        result.mode = "clarification_first"
    return result


# ---------------------------------------------------------------------------
# Strategy 4: Parallel (minimal branch-merge)
# ---------------------------------------------------------------------------

def parallel_walk(
    projected: ProjectedGraph,
    registry: ToolRegistry,
    config: SamplerConfig,
    rng: random.Random,
) -> WalkResult | None:
    """
    Minimal parallel pattern:
      root -> [branch_A, branch_B] -> merge

    The root endpoint's neighbors are split into two independent branches.
    Both branches feed a shared merge endpoint that connects to both.

    Returns a WalkResult with explicit branches and merge_endpoint_id.
    """
    # Choose a root node (entry-eligible with at least 3 distinct neighbors)
    eligible_roots = [
        eid for eid in (projected.entry_nodes or list(projected.nodes.keys()))
        if len(set(e.to_endpoint for e in projected.get_neighbors(eid))) >= 3
    ]

    if not eligible_roots:
        # Fall back to any node with enough neighbors
        eligible_roots = [
            eid for eid in projected.nodes
            if len(set(e.to_endpoint for e in projected.get_neighbors(eid))) >= 3
        ]

    if not eligible_roots:
        return None

    root = rng.choice(eligible_roots)
    root_tool = root.split("::")[0]

    # Get root's neighbors as branch candidates
    root_edges = projected.get_neighbors(root)
    if len(root_edges) < 2:
        return None

    # Shuffle and pick two distinct-tool branch endpoints if possible
    shuffled = list(root_edges)
    rng.shuffle(shuffled)

    branch_edge_a: ProjectedEdge | None = None
    branch_edge_b: ProjectedEdge | None = None

    for e in shuffled:
        if branch_edge_a is None:
            branch_edge_a = e
        elif e.to_endpoint != branch_edge_a.to_endpoint:
            # Prefer different tools for the two branches
            if e.to_endpoint.split("::")[0] != branch_edge_a.to_endpoint.split("::")[0]:
                branch_edge_b = e
                break
            elif branch_edge_b is None:
                branch_edge_b = e

    if branch_edge_a is None or branch_edge_b is None:
        return None

    branch_a_ep = branch_edge_a.to_endpoint
    branch_b_ep = branch_edge_b.to_endpoint

    # Find a merge endpoint reachable from both branches
    neighbors_a = {e.to_endpoint for e in projected.get_neighbors(branch_a_ep)}
    neighbors_b = {e.to_endpoint for e in projected.get_neighbors(branch_b_ep)}
    merge_candidates = neighbors_a & neighbors_b - {root, branch_a_ep, branch_b_ep}

    if not merge_candidates:
        # Try one hop further
        for n_a in list(neighbors_a):
            for e in projected.get_neighbors(n_a):
                if e.to_endpoint in neighbors_b and e.to_endpoint not in {root, branch_a_ep, branch_b_ep}:
                    merge_candidates.add(e.to_endpoint)

    if not merge_candidates:
        # Use the highest-weight neighbor of either branch as merge
        all_candidates = list(neighbors_a | neighbors_b - {root, branch_a_ep, branch_b_ep})
        if not all_candidates:
            return None
        merge_ep = rng.choice(all_candidates)
    else:
        merge_ep = rng.choice(list(merge_candidates))

    # Build branches
    trans_a = _build_transition(branch_edge_a)
    trans_b = _build_transition(branch_edge_b)

    # Merge transitions from both branches
    merge_edge_from_a = projected.get_edge(branch_a_ep, merge_ep)
    merge_edge_from_b = projected.get_edge(branch_b_ep, merge_ep)
    merge_transition = None
    if merge_edge_from_a:
        merge_transition = _build_transition(merge_edge_from_a)
    elif merge_edge_from_b:
        merge_transition = _build_transition(merge_edge_from_b)

    branch_a = ParallelBranch(
        branch_id="branch_a",
        endpoint_ids=[branch_a_ep],
        transitions=[trans_a],
    )
    branch_b = ParallelBranch(
        branch_id="branch_b",
        endpoint_ids=[branch_b_ep],
        transitions=[trans_b],
    )

    # Flattened endpoint_ids view for metrics/dedup
    all_ep_ids = [root, branch_a_ep, branch_b_ep, merge_ep]

    # Cross-branch transitions for the main transitions list
    root_to_merge_transitions = []
    if merge_transition:
        root_to_merge_transitions.append(merge_transition)

    # Check tool diversity
    tools_used = list(dict.fromkeys(eid.split("::")[0] for eid in all_ep_ids))
    if len(set(tools_used)) < config.min_distinct_tools:
        return None

    return WalkResult(
        endpoint_ids=all_ep_ids,
        transitions=root_to_merge_transitions,
        branches=[branch_a, branch_b],
        root_endpoint_id=root,
        merge_endpoint_id=merge_ep,
        mode="parallel",
    )


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

STRATEGY_MAP: dict[str, Callable] = {
    "sequential":          sequential_walk,
    "multi_tool":          multi_tool_walk,
    "clarification_first": clarification_first_walk,
    "parallel":            parallel_walk,
}


def run_strategy(
    mode: str,
    projected: ProjectedGraph,
    registry: ToolRegistry,
    config: SamplerConfig,
    rng: random.Random,
) -> WalkResult | None:
    """Dispatch to the correct strategy by mode name."""
    fn = STRATEGY_MAP.get(mode)
    if fn is None:
        raise ValueError(f"Unknown sampling mode: '{mode}'. Supported: {list(STRATEGY_MAP.keys())}")
    return fn(projected, registry, config, rng)
