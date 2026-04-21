"""
sampler/config.py
-----------------
SamplerConfig dataclass and YAML-backed loader.

Owned by the Sampler. No other component should load this config.

Since pydantic may not be available offline, we use plain dataclasses
with manual validation — same guarantees, no external dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Default values — used when YAML is missing or a key is absent
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "min_chain_length": 3,
    "max_chain_length": 5,
    "min_distinct_tools": 2,
    "max_distinct_categories": 3,
    "max_retries": 200,
    "unique_chains": True,
    "allow_clarification_first": True,
    "cross_tool_bias": 0.3,
    "supported_modes": ["sequential", "multi_tool", "clarification_first", "parallel"],
    # Short chain support — produces 2-3 turn conversations to satisfy
    # the "varied conversation lengths" dataset property
    "short_chain_max_length": 2,   # max tool calls in a "short" chain (1 or 2)
    "short_chain_weight": 0.10,    # fraction of sample_mixed output that uses short chains
    "user_natural_params": [
        "query", "city", "date", "location", "origin", "destination",
        "source", "target", "language", "country", "keyword", "term",
        "text", "name", "category", "type", "from_date", "to_date",
        "start_date", "end_date",
        "check_in", "check_out", "departure_date", "return_date",
        "start_datetime", "end_datetime", "time", "datetime",
        "guest_name", "passenger_name", "buyer_email", "passenger_email",
        "email", "address", "recipient", "sender",
        "from_currency", "to_currency", "amount", "currency", "budget",
        "quantity", "party_size", "passengers", "preferences",
        "job_title", "topic", "subject", "message", "symbol",
    ],
}

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "sampler_config.yaml"

SUPPORTED_MODES = {"sequential", "multi_tool", "clarification_first", "parallel"}


# ---------------------------------------------------------------------------
# SamplerConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class SamplerConfig:
    """
    All sampler behavior parameters.

    Loaded from config/sampler_config.yaml with in-code defaults as fallback.
    Every field has a default so the system works with zero config files.
    """
    min_chain_length: int = 3
    max_chain_length: int = 5
    min_distinct_tools: int = 2
    max_distinct_categories: int = 3
    max_retries: int = 200
    unique_chains: bool = True
    allow_clarification_first: bool = True
    cross_tool_bias: float = 0.3
    supported_modes: list[str] = field(default_factory=lambda: list(SUPPORTED_MODES))
    user_natural_params: set[str] = field(default_factory=lambda: set(_DEFAULTS["user_natural_params"]))
    # Short chain support
    short_chain_max_length: int = 2
    short_chain_weight: float = 0.25

    def validate(self) -> None:
        """Validate config values. Raises ValueError on invalid config."""
        if self.min_chain_length < 1:
            raise ValueError(f"min_chain_length must be >= 1, got {self.min_chain_length}")
        if self.max_chain_length < self.min_chain_length:
            raise ValueError(
                f"max_chain_length ({self.max_chain_length}) must be >= "
                f"min_chain_length ({self.min_chain_length})"
            )
        if self.min_distinct_tools < 1:
            raise ValueError(f"min_distinct_tools must be >= 1, got {self.min_distinct_tools}")
        if self.max_distinct_categories < 1:
            raise ValueError(f"max_distinct_categories must be >= 1, got {self.max_distinct_categories}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        if self.cross_tool_bias < 0:
            raise ValueError(f"cross_tool_bias must be >= 0, got {self.cross_tool_bias}")
        if not self.supported_modes:
            raise ValueError("supported_modes must not be empty")
        unknown_modes = set(self.supported_modes) - SUPPORTED_MODES
        if unknown_modes:
            raise ValueError(f"Unknown sampling modes: {unknown_modes}")
        if self.short_chain_max_length < 1:
            raise ValueError(f"short_chain_max_length must be >= 1, got {self.short_chain_max_length}")
        if not 0.0 <= self.short_chain_weight <= 1.0:
            raise ValueError(f"short_chain_weight must be in [0, 1], got {self.short_chain_weight}")

    def is_mode_supported(self, mode: str) -> bool:
        return mode in self.supported_modes

    def __repr__(self) -> str:
        return (
            f"SamplerConfig("
            f"chain={self.min_chain_length}-{self.max_chain_length}, "
            f"short_chain_max={self.short_chain_max_length}, "
            f"short_chain_weight={self.short_chain_weight}, "
            f"min_tools={self.min_distinct_tools}, "
            f"max_categories={self.max_distinct_categories}, "
            f"retries={self.max_retries}, "
            f"clarification_first={self.allow_clarification_first}, "
            f"modes={self.supported_modes})"
        )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_sampler_config(config_path: Path | str | None = None) -> SamplerConfig:
    """
    Load SamplerConfig from YAML.

    Loading pattern:
      1. Start with in-code defaults
      2. Load YAML if file exists
      3. Merge YAML over defaults (YAML wins, defaults fill missing keys)
      4. Validate the resolved config
      5. Return typed SamplerConfig

    Returns default config if YAML is missing — never raises on missing file.
    """
    resolved = _DEFAULTS.copy()

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

            if isinstance(yaml_data, dict):
                for key in _DEFAULTS:
                    if key in yaml_data:
                        resolved[key] = yaml_data[key]
        except Exception as e:
            print(f"[sampler_config] Warning: could not load {path} ({e}), using defaults")
    else:
        print(f"[sampler_config] Config not found at {path}, using defaults")

    config = SamplerConfig(
        min_chain_length=int(resolved["min_chain_length"]),
        max_chain_length=int(resolved["max_chain_length"]),
        min_distinct_tools=int(resolved["min_distinct_tools"]),
        max_distinct_categories=int(resolved.get("max_distinct_categories", _DEFAULTS["max_distinct_categories"])),
        max_retries=int(resolved["max_retries"]),
        unique_chains=bool(resolved["unique_chains"]),
        allow_clarification_first=bool(resolved["allow_clarification_first"]),
        cross_tool_bias=float(resolved["cross_tool_bias"]),
        supported_modes=list(resolved["supported_modes"]),
        user_natural_params=set(str(p) for p in resolved["user_natural_params"]),
        short_chain_max_length=int(resolved.get("short_chain_max_length", 2)),
        short_chain_weight=float(resolved.get("short_chain_weight", 0.25)),
    )

    config.validate()
    return config
