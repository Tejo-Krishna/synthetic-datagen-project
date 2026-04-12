"""
graph/registry.py
-----------------
Normalization boundary. The first and only layer that interprets raw ingest output.

Responsibilities:
  - Convert RawEndpoint -> normalized Endpoint
  - Parse returns_raw into returns_schema, returns_fields, returns_types
  - Infer endpoint intent from config/intent_rules.yaml
  - Normalize category and tool metadata
  - Build fast lookup indexes

Config: loads config/intent_rules.yaml (with in-code defaults as fallback).
Input:  IngestResult from toolbench/ingest.py
Output: ToolRegistry with normalized Endpoint objects and indexes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Normalized Endpoint — the output type of the Registry
# ---------------------------------------------------------------------------

@dataclass
class NormalizedParameter:
    """A normalized parameter with correct required flag."""
    name: str
    type: str
    description: str
    required: bool
    enum: list[str] = field(default_factory=list)
    default: Any = None


@dataclass
class Endpoint:
    """
    Fully normalized endpoint object. This is what all downstream components
    (Graph builder, Sampler, Executor, Validator) consume.

    After the Registry produces this, no downstream component should need
    to re-parse or re-derive any field.
    """
    endpoint_id: str                    # "tool_name::endpoint_name" — canonical ID
    name: str
    description: str
    method: str
    tool_name: str
    category: str                       # normalized category (Title Case)
    intent: str                         # inferred: search/retrieve/create/execute/etc.
    tags: list[str]                     # keywords extracted from name+description

    parameters: list[NormalizedParameter]

    returns_raw: str                    # original string (kept for reference)
    returns_schema: dict                # parsed dict from returns_raw
    returns_fields: set[str]            # all field names, fully flattened
    returns_types: dict[str, str]       # field_name -> inferred type string


@dataclass
class ToolRecord:
    """Registry record for one tool."""
    tool_id: str
    name: str
    description: str
    category: str
    endpoint_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ToolRegistry — the output of the Registry build step
# ---------------------------------------------------------------------------

@dataclass
class ToolRegistry:
    """
    Fully normalized registry of all tools and endpoints.

    Provides fast lookups used by Graph builder, Sampler, and Executor.
    """
    # Primary lookup
    endpoints_by_id: dict[str, Endpoint] = field(default_factory=dict)
    tools_by_id: dict[str, ToolRecord] = field(default_factory=dict)

    # Secondary indexes
    by_category: dict[str, list[str]] = field(default_factory=dict)   # category -> [endpoint_ids]
    by_tool: dict[str, list[str]] = field(default_factory=dict)        # tool_id -> [endpoint_ids]
    by_intent: dict[str, list[str]] = field(default_factory=dict)      # intent -> [endpoint_ids]

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        return self.endpoints_by_id.get(endpoint_id)

    def get_tool(self, tool_id: str) -> ToolRecord | None:
        return self.tools_by_id.get(tool_id)

    def all_endpoint_ids(self) -> list[str]:
        return list(self.endpoints_by_id.keys())

    def endpoints_in_category(self, category: str) -> list[str]:
        return self.by_category.get(category, [])

    def endpoints_for_tool(self, tool_id: str) -> list[str]:
        return self.by_tool.get(tool_id, [])

    def endpoints_with_intent(self, intent: str) -> list[str]:
        return self.by_intent.get(intent, [])

    @property
    def tool_count(self) -> int:
        return len(self.tools_by_id)

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoints_by_id)


# ---------------------------------------------------------------------------
# Default intent rules (fallback when YAML is missing)
# ---------------------------------------------------------------------------

DEFAULT_INTENT_RULES = [
    {"intent": "search",    "priority": 100, "keywords": ["search", "find", "query", "list", "lookup", "browse", "discover", "explore"]},
    {"intent": "retrieve",  "priority": 90,  "keywords": ["get", "fetch", "retrieve", "details", "info", "show", "read", "load", "view"]},
    {"intent": "create",    "priority": 80,  "keywords": ["book", "create", "make", "add", "schedule", "post", "submit", "register", "reserve", "order", "buy", "purchase"]},
    {"intent": "update",    "priority": 75,  "keywords": ["update", "edit", "modify", "change", "patch", "set", "put"]},
    {"intent": "delete",    "priority": 72,  "keywords": ["delete", "remove", "cancel", "unsubscribe", "clear"]},
    {"intent": "execute",   "priority": 70,  "keywords": ["convert", "translate", "calculate", "compute", "run", "process", "generate", "send", "check"]},
    {"intent": "compare",   "priority": 60,  "keywords": ["compare", "rank", "sort", "filter", "select", "choose", "recommend", "suggest"]},
    {"intent": "summarize", "priority": 50,  "keywords": ["summarize", "aggregate", "report", "analyze", "review", "describe"]},
]

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "intent_rules.yaml"


# ---------------------------------------------------------------------------
# Intent rule loading and inference
# ---------------------------------------------------------------------------

def _load_intent_rules(config_path: Path | None = None) -> list[dict]:
    """
    Load intent rules from YAML. Falls back to DEFAULT_INTENT_RULES if
    the file is missing or malformed.
    Rules are sorted by priority descending at load time.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    rules = DEFAULT_INTENT_RULES.copy()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "intent_rules" in data:
                loaded = data["intent_rules"]
                # Validate each rule has required fields
                valid = []
                for r in loaded:
                    if isinstance(r, dict) and "intent" in r and "keywords" in r:
                        valid.append({
                            "intent": r["intent"],
                            "priority": r.get("priority", 0),
                            "keywords": [str(k).lower() for k in r.get("keywords", [])],
                        })
                if valid:
                    rules = valid
        except Exception as e:
            print(f"[registry] Warning: could not load intent_rules.yaml ({e}), using defaults")

    # Sort descending by priority — never trust YAML list order
    rules.sort(key=lambda r: r["priority"], reverse=True)
    return rules


def infer_intent(name: str, description: str, rules: list[dict]) -> str:
    """
    Infer the intent of an endpoint from its name and description.
    Applies rules in priority order. First keyword match wins.
    Returns "unknown" if no rule matches.
    """
    text = f"{name} {description}".lower()
    # Tokenize to whole words to avoid partial matches
    tokens = set(re.findall(r'[a-z]+', text))

    for rule in rules:  # already sorted by priority
        for kw in rule["keywords"]:
            if kw in tokens or kw in text:
                return rule["intent"]

    return "unknown"


# ---------------------------------------------------------------------------
# returns_raw parsing helpers
# ---------------------------------------------------------------------------

def _parse_returns(returns_raw: str) -> tuple[dict, set[str], dict[str, str]]:
    """
    Parse a raw JSON string into:
      - returns_schema: the parsed dict
      - returns_fields: all field names, fully flattened (including nested)
      - returns_types:  field_name -> inferred type string

    Handles malformed JSON gracefully.
    """
    try:
        schema = json.loads(returns_raw) if returns_raw and returns_raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        schema = {}

    if not isinstance(schema, dict):
        schema = {}

    fields: set[str] = set()
    types: dict[str, str] = {}

    _extract_fields(schema, fields, types, prefix="")

    return schema, fields, types


def _extract_fields(obj: Any, fields: set[str], types: dict[str, str], prefix: str) -> None:
    """
    Recursively extract all field names from a JSON object.
    Supports flat fields and one level of nesting into lists/dicts.
    field names stored without prefix for easy parameter matching.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            fields.add(key)
            types[key] = _infer_json_type(value)
            # Recurse into nested dicts
            if isinstance(value, dict):
                _extract_fields(value, fields, types, prefix=key)
            # Recurse into list items (use first item as representative)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                _extract_fields(value[0], fields, types, prefix=key)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        _extract_fields(obj[0], fields, types, prefix=prefix)


def _infer_json_type(value: Any) -> str:
    """Infer a type string from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


# ---------------------------------------------------------------------------
# Category normalization
# ---------------------------------------------------------------------------

_CATEGORY_ALIASES: dict[str, str] = {
    "travel": "Travel",
    "weather": "Weather",
    "maps": "Maps",
    "finance": "Finance",
    "news": "News",
    "shopping": "Shopping",
    "productivity": "Productivity",
    "language": "Language",
    "food": "Food",
    "career": "Career",
    "entertainment": "Entertainment",
    "communication": "Communication",
    "account": "Account",
    "general": "General",
}


def _normalize_category(raw: str) -> str:
    """Normalize a raw category string to Title Case."""
    if not raw:
        return "General"
    normalized = raw.strip().lower()
    return _CATEGORY_ALIASES.get(normalized, raw.strip().title())


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = {"a", "an", "the", "and", "or", "for", "in", "on", "at", "to",
               "of", "is", "are", "was", "be", "by", "with", "from", "get",
               "set", "use", "can", "will", "that", "this", "as", "its"}


def _extract_tags(name: str, description: str) -> list[str]:
    """Extract meaningful keyword tags from endpoint name and description."""
    text = f"{name} {description}".lower()
    tokens = re.findall(r'[a-z]+', text)
    seen: set[str] = set()
    tags: list[str] = []
    for token in tokens:
        if len(token) >= 3 and token not in _STOP_WORDS and token not in seen:
            tags.append(token)
            seen.add(token)
    return tags[:20]  # cap at 20 tags


# ---------------------------------------------------------------------------
# Main registry builder
# ---------------------------------------------------------------------------

def build_registry(
    ingest_result: Any,
    intent_config_path: Path | None = None,
) -> ToolRegistry:
    """
    Build a ToolRegistry from an IngestResult.

    This is the normalization boundary — by the time data leaves this function,
    all raw strings are parsed, all intents are inferred, and all indexes are built.
    """
    intent_rules = _load_intent_rules(intent_config_path)

    registry = ToolRegistry()

    for raw_tool in ingest_result.tools:
        tool_id = raw_tool.name
        category = _normalize_category(raw_tool.raw_category)

        tool_record = ToolRecord(
            tool_id=tool_id,
            name=raw_tool.name,
            description=raw_tool.description,
            category=category,
        )

        for raw_ep in raw_tool.endpoints:
            endpoint_id = f"{tool_id}::{raw_ep.name}"

            # Parse returns_raw once
            returns_schema, returns_fields, returns_types = _parse_returns(raw_ep.returns_raw)

            # Infer intent
            intent = infer_intent(raw_ep.name, raw_ep.description, intent_rules)

            # Extract tags
            tags = _extract_tags(raw_ep.name, raw_ep.description)

            # Normalize parameters
            params: list[NormalizedParameter] = []
            for rp in raw_ep.required_parameters:
                params.append(NormalizedParameter(
                    name=rp.name, type=rp.type, description=rp.description,
                    required=True, enum=rp.enum, default=rp.default,
                ))
            for rp in raw_ep.optional_parameters:
                params.append(NormalizedParameter(
                    name=rp.name, type=rp.type, description=rp.description,
                    required=False, enum=rp.enum, default=rp.default,
                ))

            endpoint = Endpoint(
                endpoint_id=endpoint_id,
                name=raw_ep.name,
                description=raw_ep.description,
                method=raw_ep.method,
                tool_name=tool_id,
                category=category,
                intent=intent,
                tags=tags,
                parameters=params,
                returns_raw=raw_ep.returns_raw,
                returns_schema=returns_schema,
                returns_fields=returns_fields,
                returns_types=returns_types,
            )

            # Register endpoint
            registry.endpoints_by_id[endpoint_id] = endpoint
            tool_record.endpoint_ids.append(endpoint_id)

            # Update indexes
            registry.by_category.setdefault(category, []).append(endpoint_id)
            registry.by_tool.setdefault(tool_id, []).append(endpoint_id)
            registry.by_intent.setdefault(intent, []).append(endpoint_id)

        registry.tools_by_id[tool_id] = tool_record

    return registry


def summarize_registry(registry: ToolRegistry) -> str:
    """Human-readable summary of the registry."""
    lines = [
        f"ToolRegistry:",
        f"  Tools:     {registry.tool_count}",
        f"  Endpoints: {registry.endpoint_count}",
        f"  Categories: {sorted(registry.by_category.keys())}",
        f"  Intents:    {sorted(registry.by_intent.keys())}",
    ]
    return "\n".join(lines)
