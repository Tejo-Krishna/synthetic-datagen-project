"""
generator/writer.py
-------------------
JSONL dataset writer. Writes one conversation record per line.

Output format matches the PDF's required metadata fields plus
additional fields needed for diversity metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetWriter:
    """Writes conversation records to a JSONL file."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, record: dict) -> None:
        """Append one conversation record to the JSONL file."""
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._count += 1

    def write_batch(self, records: list[dict]) -> None:
        """Write multiple records."""
        for record in records:
            self.write(record)

    @property
    def records_written(self) -> int:
        return self._count

    @staticmethod
    def build_record(
        conversation_id: str,
        messages: list[dict],
        tool_calls: list[dict],
        tool_outputs: list[dict],
        chain: Any,
        domain: str,
        memory_grounding_rate: float | None,
        corpus_memory_enabled: bool,
        seed: int | None,
        num_clarification_questions: int,
    ) -> dict:
        """
        Build a complete JSONL record with all required PDF metadata fields.

        Required metadata (per PDF):
          - seed
          - tool_ids_used
          - num_turns
          - num_clarification_questions
          - memory_grounding_rate
          - corpus_memory_enabled

        Additional fields for diversity metrics:
          - pattern_type
          - domain
          - endpoint_ids (for chain alignment validation)
        """
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
            "metadata": {
                "conversation_id": conversation_id,
                "seed": seed,
                "tool_ids_used": chain.tool_ids if chain else [],
                "num_turns": len(messages),
                "num_clarification_questions": num_clarification_questions,
                "memory_grounding_rate": memory_grounding_rate,
                "corpus_memory_enabled": corpus_memory_enabled,
                "pattern_type": chain.pattern_type if chain else "unknown",
                "sampling_mode": chain.sampling_mode if chain else "unknown",
                "domain": domain,
                "endpoint_ids": chain.endpoint_ids if chain else [],
                "num_tool_calls": len(tool_calls),
                "num_distinct_tools": len(set(chain.tool_ids)) if chain else 0,
            },
        }
