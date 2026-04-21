"""
graph/heterogeneous_graph.py


Node types:
  - Tool
  - Endpoint
  - Parameter
  - ResponseField
  - Concept/Tag

Edge types:
  - has_endpoint:      Tool -> Endpoint
  - has_parameter:     Endpoint -> Parameter
  - returns_field:     Endpoint -> ResponseField
  - maps_to_concept:   Parameter/ResponseField -> Concept
  - shares_concept:    used to derive projected edges

This graph is the canonical rich representation.
The Sampler does NOT traverse it directly — it uses the projected graph.
This graph exists for correctness, explainability, and edge provenance.

Config: loads config/graph_config.yaml for semantic_groups.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from synthetic_datagen.graph.registry import ToolRegistry, Endpoint


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

NodeType = Literal["tool", "endpoint", "parameter", "response_field", "concept"]


@dataclass
class HeteroNode:
    """A node in the heterogeneous graph."""
    node_id: str
    node_type: NodeType
    label: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

HeteroEdgeType = Literal[
    "has_endpoint",      # Tool -> Endpoint
    "has_parameter",     # Endpoint -> Parameter
    "returns_field",     # Endpoint -> ResponseField
    "maps_to_concept",   # Parameter/ResponseField -> Concept
]


@dataclass
class HeteroEdge:
    """A directed edge in the heterogeneous graph."""
    source: str          # source node_id
    target: str          # target node_id
    edge_type: HeteroEdgeType
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Heterogeneous Graph container
# ---------------------------------------------------------------------------

@dataclass
class HeterogeneousGraph:
    """
    The full heterogeneous graph with 5 node types.

    nodes: node_id -> HeteroNode
    edges: list of directed HeteroEdge
    adjacency: node_id -> list of (target_id, edge_type)
    """
    nodes: dict[str, HeteroNode] = field(default_factory=dict)
    edges: list[HeteroEdge] = field(default_factory=list)
    adjacency: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def add_node(self, node: HeteroNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: HeteroEdge) -> None:
        self.edges.append(edge)
        self.adjacency.setdefault(edge.source, []).append((edge.target, edge.edge_type))

    def get_nodes_of_type(self, node_type: NodeType) -> list[HeteroNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_neighbors(self, node_id: str, edge_type: str | None = None) -> list[tuple[str, str]]:
        """Return (target_id, edge_type) pairs for a node."""
        neighbors = self.adjacency.get(node_id, [])
        if edge_type:
            return [(t, et) for t, et in neighbors if et == edge_type]
        return neighbors

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for artifact output."""
        return {
            "version": "1.0",
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "label": n.label,
                    "metadata": n.metadata,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "metadata": e.metadata,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HeterogeneousGraph":
        """Deserialize from artifact dict."""
        g = cls()
        for nd in data.get("nodes", []):
            g.add_node(HeteroNode(
                node_id=nd["node_id"],
                node_type=nd["node_type"],
                label=nd["label"],
                metadata=nd.get("metadata", {}),
            ))
        for ed in data.get("edges", []):
            g.add_edge(HeteroEdge(
                source=ed["source"],
                target=ed["target"],
                edge_type=ed["edge_type"],
                metadata=ed.get("metadata", {}),
            ))
        return g


# ---------------------------------------------------------------------------
# Default semantic groups (fallback when YAML is missing)
# ---------------------------------------------------------------------------

DEFAULT_SEMANTIC_GROUPS: dict[str, list[str]] = {
    "location":   ["location", "city", "destination", "place", "address", "origin", "source", "region", "country", "area"],
    "time":       ["date", "time", "schedule", "appointment", "datetime", "departure", "arrival", "deadline", "period", "duration"],
    "person":     ["person", "user", "customer", "client", "guest", "passenger", "name", "profile", "account"],
    "item":       ["item", "product", "sku", "goods", "service", "listing", "offering", "package"],
    "identifier": ["id", "identifier", "handle", "key", "ref", "code", "token", "number",
                   "booking_id", "reservation_id", "flight_id", "hotel_id", "order_id", "product_id", "event_id"],
    "query":      ["query", "keyword", "search_term", "term", "phrase", "text", "input"],
    "financial":  ["price", "cost", "amount", "budget", "rate", "fee", "total", "currency", "payment"],
    "contact":    ["email", "phone", "message", "notification", "alert", "contact"],
}

_DEFAULT_GRAPH_CONFIG_PATH = Path(__file__).parent.parent / "config" / "graph_config.yaml"


def _load_semantic_groups(config_path: Path | None = None) -> dict[str, list[str]]:
    """Load semantic groups from YAML, fall back to defaults."""
    path = config_path or _DEFAULT_GRAPH_CONFIG_PATH

    if not path.exists():
        return DEFAULT_SEMANTIC_GROUPS.copy()

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        groups = data.get("semantic_groups", {})
        if groups and isinstance(groups, dict):
            return {k: [str(v).lower() for v in vs] for k, vs in groups.items()}
    except Exception as e:
        print(f"[hetero_graph] Warning: could not load graph_config.yaml ({e}), using defaults")

    return DEFAULT_SEMANTIC_GROUPS.copy()


# ---------------------------------------------------------------------------
# Concept matching helper
# ---------------------------------------------------------------------------

def _find_concepts(token: str, semantic_groups: dict[str, list[str]]) -> list[str]:
    """Return concept names that match the given token."""
    token_lower = token.lower()
    matched = []
    for concept, keywords in semantic_groups.items():
        for kw in keywords:
            if kw == token_lower or kw in token_lower or token_lower in kw:
                matched.append(concept)
                break
    return matched


# ---------------------------------------------------------------------------
# Node ID helpers — consistent naming scheme
# ---------------------------------------------------------------------------

def tool_node_id(tool_name: str) -> str:
    return f"tool::{tool_name}"

def endpoint_node_id(endpoint_id: str) -> str:
    return f"endpoint::{endpoint_id}"

def param_node_id(endpoint_id: str, param_name: str) -> str:
    return f"param::{endpoint_id}::{param_name}"

def respfield_node_id(endpoint_id: str, field_name: str) -> str:
    return f"respfield::{endpoint_id}::{field_name}"

def concept_node_id(concept_name: str) -> str:
    return f"concept::{concept_name}"


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_heterogeneous_graph(
    registry: ToolRegistry,
    graph_config_path: Path | None = None,
) -> HeterogeneousGraph:
    """
    Build the full heterogeneous graph from a normalized ToolRegistry.

    Constructs:
      - Tool nodes
      - Endpoint nodes
      - Parameter nodes (required and optional)
      - ResponseField nodes (from returns_fields)
      - Concept/Tag nodes (from semantic_groups config)

    Connects them with typed edges:
      - Tool -> Endpoint (has_endpoint)
      - Endpoint -> Parameter (has_parameter)
      - Endpoint -> ResponseField (returns_field)
      - Parameter -> Concept (maps_to_concept)
      - ResponseField -> Concept (maps_to_concept)
    """
    semantic_groups = _load_semantic_groups(graph_config_path)
    graph = HeterogeneousGraph()

    # Pre-create all concept nodes
    for concept_name in semantic_groups:
        cid = concept_node_id(concept_name)
        if cid not in graph.nodes:
            graph.add_node(HeteroNode(
                node_id=cid,
                node_type="concept",
                label=concept_name,
                metadata={"keywords": semantic_groups[concept_name]},
            ))

    # Build nodes for each tool and its endpoints
    for tool_id, tool_record in registry.tools_by_id.items():
        # Tool node
        tid = tool_node_id(tool_id)
        graph.add_node(HeteroNode(
            node_id=tid,
            node_type="tool",
            label=tool_id,
            metadata={
                "category": tool_record.category,
                "description": tool_record.description,
            },
        ))

        for endpoint_id in tool_record.endpoint_ids:
            ep = registry.get_endpoint(endpoint_id)
            if ep is None:
                continue

            # Endpoint node
            eid = endpoint_node_id(endpoint_id)
            graph.add_node(HeteroNode(
                node_id=eid,
                node_type="endpoint",
                label=ep.name,
                metadata={
                    "tool_id": tool_id,
                    "category": ep.category,
                    "intent": ep.intent,
                    "method": ep.method,
                    "description": ep.description,
                },
            ))

            # Tool -> Endpoint edge
            graph.add_edge(HeteroEdge(
                source=tid,
                target=eid,
                edge_type="has_endpoint",
            ))

            # Parameter nodes
            for param in ep.parameters:
                pid = param_node_id(endpoint_id, param.name)
                graph.add_node(HeteroNode(
                    node_id=pid,
                    node_type="parameter",
                    label=param.name,
                    metadata={
                        "type": param.type,
                        "required": param.required,
                        "description": param.description,
                    },
                ))

                # Endpoint -> Parameter edge
                graph.add_edge(HeteroEdge(
                    source=eid,
                    target=pid,
                    edge_type="has_parameter",
                    metadata={"required": param.required},
                ))

                # Parameter -> Concept edges
                for concept in _find_concepts(param.name, semantic_groups):
                    graph.add_edge(HeteroEdge(
                        source=pid,
                        target=concept_node_id(concept),
                        edge_type="maps_to_concept",
                    ))

            # ResponseField nodes (from returns_fields)
            for field_name in ep.returns_fields:
                rfid = respfield_node_id(endpoint_id, field_name)
                graph.add_node(HeteroNode(
                    node_id=rfid,
                    node_type="response_field",
                    label=field_name,
                    metadata={
                        "type": ep.returns_types.get(field_name, "unknown"),
                        "endpoint_id": endpoint_id,
                    },
                ))

                # Endpoint -> ResponseField edge
                graph.add_edge(HeteroEdge(
                    source=eid,
                    target=rfid,
                    edge_type="returns_field",
                ))

                # ResponseField -> Concept edges
                for concept in _find_concepts(field_name, semantic_groups):
                    graph.add_edge(HeteroEdge(
                        source=rfid,
                        target=concept_node_id(concept),
                        edge_type="maps_to_concept",
                    ))

    return graph


def summarize_graph(graph: HeterogeneousGraph) -> str:
    """Human-readable summary of the heterogeneous graph."""
    type_counts: dict[str, int] = {}
    for node in graph.nodes.values():
        type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

    edge_counts: dict[str, int] = {}
    for edge in graph.edges:
        edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1

    lines = [
        f"HeterogeneousGraph:",
        f"  Total nodes: {graph.node_count()}",
        f"  Total edges: {graph.edge_count()}",
        "  Node types:",
    ]
    for ntype, count in sorted(type_counts.items()):
        lines.append(f"    {ntype}: {count}")
    lines.append("  Edge types:")
    for etype, count in sorted(edge_counts.items()):
        lines.append(f"    {etype}: {count}")
    return "\n".join(lines)
