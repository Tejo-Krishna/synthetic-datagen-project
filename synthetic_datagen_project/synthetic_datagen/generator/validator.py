"""
generator/validator.py
----------------------
Conversation Validator Agent — validates the final conversation artifact.

Validates against the FINAL conversation, not just the sampled chain.
May reject conversations and trigger regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of validating one conversation."""
    conversation_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


class ConversationValidator:
    """Validates generated conversations for structural and semantic correctness."""

    def validate(self, conversation: dict) -> ValidationResult:
        """
        Validate a conversation record.

        Checks:
          - has_messages: conversation has messages
          - has_tool_calls: at least one tool call present
          - multi_step_or_short: either >= 3 tool calls (standard) or
            pattern_type == "short_chain" with 1–2 tool calls
          - multi_tool: at least 2 distinct tools (waived for short_chain pattern)
          - chain_alignment: tool calls match sampled chain
          - metadata_complete: all required metadata fields present
          - tool_outputs_present: all tool calls have corresponding outputs
        """
        conv_id = conversation.get("metadata", {}).get("conversation_id", "unknown")
        result = ValidationResult(conversation_id=conv_id, passed=True)
        messages = conversation.get("messages", [])
        tool_calls = conversation.get("tool_calls", [])
        tool_outputs = conversation.get("tool_outputs", [])
        metadata = conversation.get("metadata", {})

        pattern_type = metadata.get("pattern_type", "")
        is_short_chain = pattern_type == "short_chain"

        # Check: has messages
        result.checks["has_messages"] = len(messages) > 0
        if not result.checks["has_messages"]:
            result.errors.append("Conversation has no messages")

        # Check: has tool calls
        result.checks["has_tool_calls"] = len(tool_calls) > 0
        if not result.checks["has_tool_calls"]:
            result.errors.append("Conversation has no tool calls")

        # Check: multi_step — waived for short_chain pattern
        if is_short_chain:
            # Short chains need 1–2 tool calls; this is intentionally shorter
            result.checks["multi_step"] = 1 <= len(tool_calls) <= 2
            if not result.checks["multi_step"]:
                result.errors.append(
                    f"Short chain has {len(tool_calls)} tool calls (expected 1–2)"
                )
        else:
            result.checks["multi_step"] = len(tool_calls) >= 3
            if not result.checks["multi_step"]:
                result.errors.append(f"Only {len(tool_calls)} tool calls (minimum 3 required)")

        # Check: multi_tool — waived for short_chain pattern (single-tool chains are valid)
        distinct_tools = set(tc.get("name", "").split("::")[0] for tc in tool_calls)
        if is_short_chain:
            result.checks["multi_tool"] = len(distinct_tools) >= 1
        else:
            result.checks["multi_tool"] = len(distinct_tools) >= 2
            if not result.checks["multi_tool"]:
                result.errors.append(f"Only {len(distinct_tools)} distinct tools (minimum 2 required)")

        # Check: tool_outputs_present
        result.checks["tool_outputs_present"] = len(tool_outputs) == len(tool_calls)
        if not result.checks["tool_outputs_present"]:
            result.warnings.append(
                f"Tool outputs count ({len(tool_outputs)}) != tool calls count ({len(tool_calls)})"
            )

        # Check: metadata_complete
        required_meta = ["seed", "tool_ids_used", "num_turns", "num_clarification_questions",
                         "memory_grounding_rate", "corpus_memory_enabled"]
        missing_meta = [k for k in required_meta if k not in metadata]
        result.checks["metadata_complete"] = len(missing_meta) == 0
        if missing_meta:
            result.warnings.append(f"Missing metadata fields: {missing_meta}")

        # Check: chain alignment
        sampled_chain = metadata.get("endpoint_ids", [])
        if sampled_chain and tool_calls:
            actual_endpoints = [tc.get("name", "") for tc in tool_calls]
            result.checks["chain_alignment"] = actual_endpoints == sampled_chain
            if not result.checks["chain_alignment"]:
                result.warnings.append("Tool call order does not match sampled chain")
        else:
            result.checks["chain_alignment"] = True

        # Hard checks determine overall pass/fail
        hard_checks = ["has_messages", "has_tool_calls", "multi_step", "multi_tool"]
        result.passed = all(result.checks.get(c, True) for c in hard_checks)

        return result
