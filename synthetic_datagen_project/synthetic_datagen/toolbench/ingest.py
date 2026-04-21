"""
toolbench/ingest.py
-------------------
Raw ingestion layer. Reads ToolBench-format JSON and produces raw structured
objects with NO interpretation, NO normalization, and NO config dependency.

Responsibilities:
  - Parse raw tool/endpoint JSON into typed raw dataclasses
  - Preserve returns_raw as an unparsed string
  - Handle missing/inconsistent fields defensively

Must NOT:
  - Infer intent
  - Parse returns_raw
  - Normalize categories
  - Build graph edges
  - Load any YAML config

This module is deterministic and config-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Raw dataclasses — faithful representation of what the JSON contains
# ---------------------------------------------------------------------------

@dataclass
class RawParameter:
    """One input parameter as it appears in the raw JSON."""
    name: str
    type: str
    description: str
    required: bool
    enum: list[str] = field(default_factory=list)
    default: Any = None


@dataclass
class RawEndpoint:
    """
    One API endpoint as it appears in the raw JSON.
    returns_raw is preserved as the original string from the JSON.
    The Registry is the first and only layer that parses it.
    """
    name: str
    description: str
    method: str
    required_parameters: list[RawParameter]
    optional_parameters: list[RawParameter]
    returns_raw: str
    tool_name: str
    tool_description: str
    raw_category: str


@dataclass
class RawTool:
    """One tool (API group) as it appears in the raw JSON."""
    name: str
    description: str
    raw_category: str
    homepage: str
    endpoints: list[RawEndpoint]


@dataclass
class IngestResult:
    """Output of the ingestion step."""
    tools: list[RawTool]
    endpoints: list[RawEndpoint]
    source_path: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default data path
# ---------------------------------------------------------------------------

_DEFAULT_SEED_PATH = Path(__file__).parent.parent / "data" / "seed_tools.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_seed_tools(path: str | Path | None = None) -> IngestResult:
    """
    Load raw tools from a ToolBench-format JSON file.
    Falls back to the bundled seed_tools.json if path is None.
    """
    data_path = Path(path) if path else _DEFAULT_SEED_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"[ingest] Seed tools file not found: {data_path}"
        )

    with open(data_path, encoding="utf-8") as f:
        try:
            raw_json = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"[ingest] Failed to parse JSON from {data_path}: {e}") from e

    return parse_seed_tools(raw_json, source_path=str(data_path))


def parse_seed_tools(raw_json: Any, source_path: str = "<unknown>") -> IngestResult:
    """
    Parse a raw JSON payload into RawTool / RawEndpoint structures.
    Accepts list of tools, dict with "tools" key, or single tool dict.
    """
    warnings: list[str] = []

    if isinstance(raw_json, list):
        raw_tools = raw_json
    elif isinstance(raw_json, dict) and "tools" in raw_json:
        raw_tools = raw_json["tools"]
    elif isinstance(raw_json, dict) and ("tool_name" in raw_json or "name" in raw_json):
        raw_tools = [raw_json]
    else:
        warnings.append(f"[ingest] Unrecognized top-level structure in {source_path}")
        return IngestResult(tools=[], endpoints=[], source_path=source_path, warnings=warnings)

    tools: list[RawTool] = []
    all_endpoints: list[RawEndpoint] = []

    for i, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, dict):
            warnings.append(f"[ingest] Tool at index {i} is not a dict — skipping")
            continue
        try:
            tool, tool_warnings = _parse_tool(raw_tool)
            tools.append(tool)
            all_endpoints.extend(tool.endpoints)
            warnings.extend(tool_warnings)
        except Exception as e:
            tool_id = raw_tool.get("tool_name", raw_tool.get("name", f"index_{i}"))
            warnings.append(f"[ingest] Failed to parse tool '{tool_id}': {e}")

    return IngestResult(
        tools=tools,
        endpoints=all_endpoints,
        source_path=source_path,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_tool(raw: dict) -> tuple[RawTool, list[str]]:
    warnings: list[str] = []

    tool_name = raw.get("tool_name") or raw.get("name") or "unknown_tool"
    description = raw.get("tool_description") or raw.get("description") or ""
    raw_category = raw.get("category") or raw.get("tool_category") or "general"
    homepage = raw.get("home_url") or raw.get("homepage") or ""

    raw_endpoint_list = (
        raw.get("api_list") or raw.get("apis") or raw.get("endpoints") or []
    )

    if not raw_endpoint_list:
        warnings.append(f"[ingest] Tool '{tool_name}' has no endpoints")

    endpoints: list[RawEndpoint] = []
    for j, raw_ep in enumerate(raw_endpoint_list):
        if not isinstance(raw_ep, dict):
            warnings.append(f"[ingest] Endpoint {j} in '{tool_name}' is not a dict — skipping")
            continue
        try:
            ep = _parse_endpoint(raw_ep, tool_name=tool_name,
                                 tool_description=description, raw_category=raw_category)
            endpoints.append(ep)
        except Exception as e:
            ep_name = raw_ep.get("name", f"index_{j}")
            warnings.append(f"[ingest] Failed to parse endpoint '{ep_name}' in '{tool_name}': {e}")

    return RawTool(
        name=tool_name,
        description=description,
        raw_category=raw_category,
        homepage=homepage,
        endpoints=endpoints,
    ), warnings


def _parse_endpoint(raw: dict, tool_name: str, tool_description: str, raw_category: str) -> RawEndpoint:
    name = raw.get("name") or raw.get("endpoint_name") or "unknown_endpoint"
    description = raw.get("description") or ""
    method = (raw.get("method") or "GET").upper()

    raw_required = raw.get("required_parameters") or []
    raw_optional = raw.get("optional_parameters") or []

    required_params = [_parse_parameter(p, required=True) for p in raw_required if isinstance(p, dict)]
    optional_params = [_parse_parameter(p, required=False) for p in raw_optional if isinstance(p, dict)]

    returns_raw = (
        raw.get("template_response") or raw.get("returns") or raw.get("response") or "{}"
    )
    if isinstance(returns_raw, dict):
        returns_raw = json.dumps(returns_raw)
    elif not isinstance(returns_raw, str):
        returns_raw = "{}"

    return RawEndpoint(
        name=name,
        description=description,
        method=method,
        required_parameters=required_params,
        optional_parameters=optional_params,
        returns_raw=returns_raw,
        tool_name=tool_name,
        tool_description=tool_description,
        raw_category=raw_category,
    )


def _parse_parameter(raw: dict, required: bool) -> RawParameter:
    return RawParameter(
        name=raw.get("name") or "unnamed_param",
        type=raw.get("type") or "string",
        description=raw.get("description") or "",
        required=required,
        enum=raw.get("enum") or [],
        default=raw.get("default"),
    )


def summarize(result: IngestResult) -> str:
    """Human-readable summary of an IngestResult."""
    lines = [
        f"Ingested from: {result.source_path}",
        f"  Tools:     {len(result.tools)}",
        f"  Endpoints: {len(result.endpoints)}",
    ]
    if result.warnings:
        lines.append(f"  Warnings:  {len(result.warnings)}")
        for w in result.warnings:
            lines.append(f"    {w}")
    return "\n".join(lines)
