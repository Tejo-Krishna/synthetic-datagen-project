"""
graph/projected_graph.py
-------------------------
Derives a traversal-friendly endpoint-to-endpoint projected graph
from the heterogeneous graph.

The Sampler walks THIS graph, not the heterogeneous graph directly.
Each projected edge carries provenance from the heterogeneous graph.

Edge types in order of strength:
  - data_link (1.0): response field from A directly fills a required param in B
  - semantic (0.45): A and B share a concept node via their fields/params
  - category (0.2): A and B belong to the same tool category

ProjectedEdge fields:
  - weight: used ONLY during sampling, NOT copied to SampledChain.Transition
  - provenance_path: stored in artifact for debugging, NOT copied to SampledChain

Config: loads config/graph_config.yaml for edge weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from synthetic_datagen.common.types import FieldMapping
from synthetic_datagen.graph.registry import ToolRegistry, Endpoint
from synthetic_datagen.graph.heterogeneous_graph import (
    HeterogeneousGraph,
    endpoint_node_id,
    param_node_id,
    respfield_node_id,
    concept_node_id,
)


# ---------------------------------------------------------------------------
# ProjectedEdge — internal graph edge (never exposed in SampledChain)
# ---------------------------------------------------------------------------

@dataclass
class ProjectedEdge:
    """
    One endpoint-to-endpoint edge in the sampler graph.

    weight and provenance_path are graph-internal.
    The Sampler copies only: from_endpoint, to_endpoint, edge_type,
    field_mappings, matched_concepts into SampledChain.Transition.
    """
    from_endpoint: str
    to_endpoint: str
    edge_type: Literal["data_link", "semantic", "category"]
    weight: float                           # used only during sampling
    field_mappings: list[FieldMapping]      # source_field -> target_param
    matched_concepts: list[str]             # concept nodes that created this edge
    provenance_path: list[str] | None = None  # full het. graph path, for debugging


# ---------------------------------------------------------------------------
# ProjectedGraph container
# ---------------------------------------------------------------------------

@dataclass
class ProjectedGraph:
    """
    Endpoint-to-endpoint projected graph.

    nodes: endpoint_id -> node metadata
    adjacency: endpoint_id -> list[ProjectedEdge]
    entry_nodes: pre-computed eligible start nodes
    """
    nodes: dict[str, dict] = field(default_factory=dict)    # endpoint_id -> metadata
    adjacency: dict[str, list[ProjectedEdge]] = field(default_factory=dict)
    entry_nodes: list[str] = field(default_factory=list)

    def add_node(self, endpoint_id: str, metadata: dict) -> None:
        self.nodes[endpoint_id] = metadata

    def add_edge(self, edge: ProjectedEdge) -> None:
        self.adjacency.setdefault(edge.from_endpoint, []).append(edge)

    def get_neighbors(self, endpoint_id: str) -> list[ProjectedEdge]:
        return self.adjacency.get(endpoint_id, [])

    def has_edge(self, from_ep: str, to_ep: str) -> bool:
        return any(e.to_endpoint == to_ep for e in self.adjacency.get(from_ep, []))

    def get_edge(self, from_ep: str, to_ep: str) -> ProjectedEdge | None:
        for e in self.adjacency.get(from_ep, []):
            if e.to_endpoint == to_ep:
                return e
        return None

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.adjacency.values())

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for artifact output."""
        return {
            "version": "1.0",
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entry_node_count": len(self.entry_nodes),
            "nodes": [
                {"endpoint_id": eid, **meta}
                for eid, meta in self.nodes.items()
            ],
            "edges": [
                {
                    "from_endpoint": e.from_endpoint,
                    "to_endpoint": e.to_endpoint,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "field_mappings": [
                        {"source_field": fm.source_field, "target_param": fm.target_param}
                        for fm in e.field_mappings
                    ],
                    "matched_concepts": e.matched_concepts,
                    "provenance_path": e.provenance_path,
                }
                for edges in self.adjacency.values()
                for e in edges
            ],
            "entry_nodes": self.entry_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectedGraph":
        """Deserialize from artifact dict."""
        g = cls()
        for nd in data.get("nodes", []):
            eid = nd.pop("endpoint_id")
            g.add_node(eid, nd)
        for ed in data.get("edges", []):
            g.add_edge(ProjectedEdge(
                from_endpoint=ed["from_endpoint"],
                to_endpoint=ed["to_endpoint"],
                edge_type=ed["edge_type"],
                weight=ed["weight"],
                field_mappings=[
                    FieldMapping(fm["source_field"], fm["target_param"])
                    for fm in ed.get("field_mappings", [])
                ],
                matched_concepts=ed.get("matched_concepts", []),
                provenance_path=ed.get("provenance_path"),
            ))
        g.entry_nodes = data.get("entry_nodes", [])
        return g


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------

DEFAULT_EDGE_WEIGHTS = {
    "data_link": 1.0,
    "semantic":  0.45,
    "category":  0.2,
}

DEFAULT_USER_NATURAL_PARAMS = {
    # Basic search/navigation params
    "query", "city", "date", "location", "origin", "destination",
    "source", "target", "language", "country", "keyword", "term",
    "text", "name", "category", "type", "from_date", "to_date",
    "start_date", "end_date",
    # Date/time params users specify directly
    "check_in", "check_out", "departure_date", "return_date",
    "start_datetime", "end_datetime", "time", "datetime",
    # Identity/contact params users provide
    "guest_name", "passenger_name", "buyer_email", "passenger_email",
    "email", "address", "recipient", "sender",
    # Financial params users specify
    "from_currency", "to_currency", "amount", "currency", "budget",
    # Quantity/preference params
    "quantity", "party_size", "passengers", "preferences",
    "job_title", "topic", "subject", "message", "symbol",
}

_DEFAULT_GRAPH_CONFIG_PATH = Path(__file__).parent.parent / "config" / "graph_config.yaml"


def _load_edge_weights(config_path: Path | None = None) -> dict[str, float]:
    """Load edge weights from YAML, fall back to defaults."""
    path = config_path or _DEFAULT_GRAPH_CONFIG_PATH
    if not path.exists():
        return DEFAULT_EDGE_WEIGHTS.copy()
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        weights = data.get("edge_weights", {})
        if weights:
            return {
                "data_link": float(weights.get("data_link", 1.0)),
                "semantic":  float(weights.get("semantic", 0.45)),
                "category":  float(weights.get("category", 0.2)),
            }
    except Exception as e:
        print(f"[projected_graph] Warning: could not load graph_config.yaml ({e}), using defaults")
    return DEFAULT_EDGE_WEIGHTS.copy()


# ---------------------------------------------------------------------------
# Edge construction helpers
# ---------------------------------------------------------------------------

def _build_data_link_edges(
    source_ep: Endpoint,
    target_ep: Endpoint,
    weight: float,
    hetero_graph: HeterogeneousGraph,
) -> list[ProjectedEdge]:
    """
    Build data_link edges: source response fields that match target required params.

    A match occurs when:
      - Exact name match: source field name == target required param name
      - Suffix match: one name is a suffix/substring of the other
    """
    edges: list[ProjectedEdge] = []

    target_required_params = {p.name for p in target_ep.parameters if p.required}
    if not target_required_params:
        return edges

    field_mappings: list[FieldMapping] = []
    matched_fields: set[str] = set()

    for field_name in source_ep.returns_fields:
        for param_name in target_required_params:
            if _fields_match(field_name, param_name):
                field_mappings.append(FieldMapping(
                    source_field=field_name,
                    target_param=param_name,
                ))
                matched_fields.add(field_name)

    if not field_mappings:
        return edges

    # Build provenance path through heterogeneous graph
    prov: list[str] = []
    src_eid = endpoint_node_id(source_ep.endpoint_id)
    tgt_eid = endpoint_node_id(target_ep.endpoint_id)
    for fm in field_mappings[:1]:  # one example path
        rf_nid = respfield_node_id(source_ep.endpoint_id, fm.source_field)
        p_nid = param_node_id(target_ep.endpoint_id, fm.target_param)
        prov = [src_eid, rf_nid, p_nid, tgt_eid]

    edges.append(ProjectedEdge(
        from_endpoint=source_ep.endpoint_id,
        to_endpoint=target_ep.endpoint_id,
        edge_type="data_link",
        weight=weight,
        field_mappings=field_mappings,
        matched_concepts=[],
        provenance_path=prov,
    ))
    return edges


def _fields_match(field_name: str, param_name: str) -> bool:
    """Check if a response field name matches a parameter name."""
    fn = field_name.lower().replace("_", "").replace("-", "")
    pn = param_name.lower().replace("_", "").replace("-", "")

    # Exact match
    if fn == pn:
        return True

    # One is a suffix of the other
    if fn.endswith(pn) or pn.endswith(fn):
        return True

    # One contains the other (minimum length 3 to avoid trivial matches)
    if len(fn) >= 3 and len(pn) >= 3:
        if fn in pn or pn in fn:
            return True

    return False


def _build_semantic_edges(
    source_ep: Endpoint,
    target_ep: Endpoint,
    weight: float,
    hetero_graph: HeterogeneousGraph,
) -> list[ProjectedEdge]:
    """
    Build semantic edges: source and target share concept nodes via their
    response fields and parameters respectively.
    """
    # Get concepts reachable from source's response fields
    source_concepts: set[str] = set()
    for field_name in source_ep.returns_fields:
        rf_nid = respfield_node_id(source_ep.endpoint_id, field_name)
        for target_nid, edge_type in hetero_graph.get_neighbors(rf_nid, "maps_to_concept"):
            concept = target_nid.replace("concept::", "")
            source_concepts.add(concept)

    if not source_concepts:
        return []

    # Get concepts reachable from target's required parameters
    target_concepts: set[str] = set()
    for param in target_ep.parameters:
        if param.required:
            p_nid = param_node_id(target_ep.endpoint_id, param.name)
            for target_nid, edge_type in hetero_graph.get_neighbors(p_nid, "maps_to_concept"):
                concept = target_nid.replace("concept::", "")
                target_concepts.add(concept)

    shared_concepts = source_concepts & target_concepts
    if not shared_concepts:
        return []

    return [ProjectedEdge(
        from_endpoint=source_ep.endpoint_id,
        to_endpoint=target_ep.endpoint_id,
        edge_type="semantic",
        weight=weight,
        field_mappings=[],
        matched_concepts=sorted(shared_concepts),
        provenance_path=[
            endpoint_node_id(source_ep.endpoint_id),
            f"concept::{next(iter(shared_concepts))}",
            endpoint_node_id(target_ep.endpoint_id),
        ],
    )]


def _build_category_edges(
    source_ep: Endpoint,
    target_ep: Endpoint,
    weight: float,
) -> list[ProjectedEdge]:
    """
    Build category edges: source and target belong to the same category.
    Only add if no stronger edge already exists between them.
    """
    if source_ep.category != target_ep.category:
        return []

    return [ProjectedEdge(
        from_endpoint=source_ep.endpoint_id,
        to_endpoint=target_ep.endpoint_id,
        edge_type="category",
        weight=weight,
        field_mappings=[],
        matched_concepts=[],
        provenance_path=None,
    )]


# ---------------------------------------------------------------------------
# Entry node detection
# ---------------------------------------------------------------------------

def _is_entry_eligible(
    endpoint: Endpoint,
    user_natural_params: set[str],
) -> bool:
    """
    An endpoint is entry-eligible if:
      - It has no required parameters, OR
      - All its required parameters are user-natural (user can provide them directly)
    """
    required = [p for p in endpoint.parameters if p.required]
    if not required:
        return True
    return all(p.name.lower() in user_natural_params for p in required)


# ---------------------------------------------------------------------------
# Main projected graph builder
# ---------------------------------------------------------------------------

def build_projected_graph(
    registry: ToolRegistry,
    hetero_graph: HeterogeneousGraph,
    graph_config_path: Path | None = None,
    user_natural_params: set[str] | None = None,
) -> ProjectedGraph:
    """
    Derive the projected endpoint-to-endpoint sampler graph from the
    heterogeneous graph and registry.

    For each ordered pair of endpoints (A, B), attempt to create:
      1. A data_link edge (strongest) — if A's response fills B's required params
      2. A semantic edge (medium) — if A and B share concept nodes
      3. A category edge (weakest) — if A and B are in the same category

    Only the strongest applicable edge type is kept per (A, B) pair.
    Self-loops are excluded.
    """
    weights = _load_edge_weights(graph_config_path)
    natural_params = user_natural_params or DEFAULT_USER_NATURAL_PARAMS

    projected = ProjectedGraph()

    # Add all endpoints as nodes
    all_endpoint_ids = registry.all_endpoint_ids()
    for eid in all_endpoint_ids:
        ep = registry.get_endpoint(eid)
        if ep:
            projected.add_node(eid, {
                "tool_id": ep.tool_name,
                "intent": ep.intent,
                "category": ep.category,
                "name": ep.name,
            })

    # Build edges between all pairs
    endpoints = [registry.get_endpoint(eid) for eid in all_endpoint_ids]
    endpoints = [ep for ep in endpoints if ep is not None]

    for source_ep in endpoints:
        for target_ep in endpoints:
            # No self-loops
            if source_ep.endpoint_id == target_ep.endpoint_id:
                continue

            # Try data_link first (strongest)
            data_edges = _build_data_link_edges(
                source_ep, target_ep, weights["data_link"], hetero_graph
            )
            if data_edges:
                for e in data_edges:
                    projected.add_edge(e)
                continue

            # Try semantic (medium)
            sem_edges = _build_semantic_edges(
                source_ep, target_ep, weights["semantic"], hetero_graph
            )
            if sem_edges:
                for e in sem_edges:
                    projected.add_edge(e)
                continue

            # Try category (weakest)
            cat_edges = _build_category_edges(
                source_ep, target_ep, weights["category"]
            )
            for e in cat_edges:
                projected.add_edge(e)

    # Pre-compute entry nodes
    projected.entry_nodes = [
        eid for eid in all_endpoint_ids
        if _is_entry_eligible(registry.get_endpoint(eid), natural_params)
    ]

    return projected


def summarize_projected(graph: ProjectedGraph) -> str:
    """Human-readable summary of the projected graph."""
    edge_type_counts: dict[str, int] = {}
    for edges in graph.adjacency.values():
        for e in edges:
            edge_type_counts[e.edge_type] = edge_type_counts.get(e.edge_type, 0) + 1

    lines = [
        f"ProjectedGraph:",
        f"  Endpoint nodes: {graph.node_count}",
        f"  Total edges:    {graph.edge_count}",
        f"  Entry nodes:    {len(graph.entry_nodes)}",
        "  Edge types:",
    ]
    for etype, count in sorted(edge_type_counts.items()):
        lines.append(f"    {etype}: {count}")
    return "\n".join(lines)
