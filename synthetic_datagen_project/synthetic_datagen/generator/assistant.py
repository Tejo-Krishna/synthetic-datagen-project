"""
generator/assistant.py
----------------------
Assistant Agent — simulates the tool-using assistant.

Generates clarification questions, emits tool calls, interprets outputs,
and produces final responses. Behaves like the model being trained.

Initially template-driven for determinism and structural correctness.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from synthetic_datagen.common.types import ClarificationStep
from synthetic_datagen.graph.registry import ToolRegistry
from synthetic_datagen.generator.executor import StepOutput
from synthetic_datagen.planner import StructuredConversationPlan


@dataclass
class AssistantTurn:
    """One assistant turn in the conversation."""
    role: str = "assistant"
    content: str = ""
    tool_calls: list[dict] | None = None  # list of {name, parameters} dicts


class AssistantAgent:
    """Simulates the tool-using assistant agent."""

    def __init__(self, registry: ToolRegistry, seed: int | None = None):
        self.registry = registry
        self.rng = random.Random(seed)

    def ask_clarification(
        self,
        clarification: ClarificationStep,
        step_purpose: str | None = None,
    ) -> AssistantTurn:
        """Generate a clarification question with varied natural phrasing.

        step_purpose: optional sentence describing what the next step does
                      (e.g. "Book a hotel for your stay."). When provided,
                      the question is prefixed with a brief context phrase
                      so the parameter ask doesn't feel abrupt.
        """
        if clarification.reason == "intent_ambiguity":
            questions = [
                "I'd be happy to help! Could you tell me a bit more about what you're looking to accomplish?",
                "To make sure I understand correctly, could you clarify what you need?",
                "I want to help you with the right task. Could you provide more details about your goal?",
                "Sure, I can help with that. What specifically are you trying to accomplish?",
                "Happy to assist! Could you give me a bit more context about your request?",
            ]
            return AssistantTurn(content=self.rng.choice(questions))

        # Build a context prefix from the step purpose so the clarification
        # feels motivated rather than abrupt.
        prefix = self._purpose_prefix(step_purpose) if step_purpose else ""

        # missing_required_param
        if clarification.missing_params:
            param_phrases = [p.replace("_", " ") for p in clarification.missing_params]

            if len(param_phrases) == 1:
                p = param_phrases[0]
                single_templates = [
                    f"{prefix}could you share your {p}?",
                    f"{prefix}I'll need your {p}. Could you provide that?",
                    f"{prefix}could you let me know your {p}?",
                    f"{prefix}please share your {p} and I'll take care of the rest.",
                ]
                # Capitalise the first letter
                q = self.rng.choice(single_templates)
                q = q[0].upper() + q[1:]
            else:
                listed = ", ".join(param_phrases[:-1]) + f" and {param_phrases[-1]}"
                multi_templates = [
                    f"{prefix}could you provide your {listed}?",
                    f"{prefix}I'll need a few details: your {listed}.",
                    f"{prefix}could you share: {listed}?",
                    f"{prefix}I'll need your {listed} to proceed.",
                ]
                q = self.rng.choice(multi_templates)
                q = q[0].upper() + q[1:]
            return AssistantTurn(content=q)

        return AssistantTurn(content="Could you provide some additional details?")

    def _purpose_prefix(self, purpose: str) -> str:
        """
        Convert a step purpose like "Book a hotel for your stay." into a
        contextual lead-in phrase like "To book your hotel, ".
        """
        p = purpose.strip().rstrip(".")
        # Convert third-person to second-person
        p = p.replace("the user's", "your").replace("for the user", "for you")
        p_lower = p.lower()
        if p_lower.startswith(("book", "complete", "purchase", "create", "make")):
            return f"To {p_lower}, "
        elif p_lower.startswith(("search", "look up", "retrieve", "find", "fetch", "check")):
            return f"To {p_lower}, "
        elif p_lower.startswith(("save", "update", "store")):
            return f"To {p_lower}, "
        elif p_lower.startswith(("handle", "execute", "run")):
            return f"For this step, "
        return f"For {p_lower}, "

    def emit_tool_call(
        self,
        endpoint_id: str,
        arguments: dict,
        preamble: str | None = None,
    ) -> AssistantTurn:
        """Generate an assistant turn that emits a tool call."""
        ep = self.registry.get_endpoint(endpoint_id)
        tool_name = ep.name if ep else endpoint_id.split("::")[-1]

        if preamble:
            content = preamble
        else:
            content = self._preamble_for_tool(tool_name, ep.intent if ep else "retrieve")

        return AssistantTurn(
            content=content,
            tool_calls=[{
                "name": endpoint_id,
                "parameters": arguments,
            }],
        )

    def interpret_tool_output(
        self,
        step: StepOutput,
        is_final: bool = False,
    ) -> AssistantTurn:
        """Generate an assistant turn that interprets tool output."""
        ep = self.registry.get_endpoint(step.endpoint_id)
        tool_name = ep.name if ep else step.endpoint_id.split("::")[-1]

        if is_final:
            return AssistantTurn(content=self._final_summary(step))
        else:
            return AssistantTurn(content=self._intermediate_summary(tool_name, step.output))

    def generate_final_response(
        self,
        plan: StructuredConversationPlan,
        all_steps: list[StepOutput],
    ) -> AssistantTurn:
        """Generate a grounded final response referencing actual tool output values."""
        # Fields considered meaningful for surface-level reference
        _MEANINGFUL_FIELDS = {
            "flight_id", "airline", "price", "departure_time", "arrival_time",
            "hotel_id", "hotel_name", "room_type", "check_in", "check_out",
            "reservation_id", "confirmation", "confirmation_number",
            "restaurant_id", "restaurant_name", "cuisine", "rating",
            "order_id", "tracking_number", "item_name",
            "job_id", "company", "job_title", "salary",
            "event_id", "event_name", "venue", "date",
            "temperature", "forecast", "weather_description",
            "rate", "converted_amount", "from_currency", "to_currency",
            "symbol", "current_price", "change_percent",
            "recipe_id", "recipe_name", "cuisine_type",
            "ticket_id", "seat", "total_price",
        }
        _SKIP_FIELDS = {"status", "ok", "result", "success", "message", "error", "code"}

        sentences = []
        for step in all_steps:
            # Extract top 2 meaningful values from this step's output
            values = []
            for k, v in step.output.items():
                if k in _SKIP_FIELDS:
                    continue
                if k in _MEANINGFUL_FIELDS and not isinstance(v, (dict, list)):
                    values.append((k.replace("_", " "), v))
                if len(values) >= 2:
                    break

            if values:
                parts = ", ".join(f"{k}: {v}" for k, v in values)
                endpoint_label = step.endpoint_id.split("::")[-1].replace("_", " ")
                sentences.append(f"From {endpoint_label} — {parts}.")

        if sentences:
            body = " ".join(sentences)
            closings = [
                "Is there anything else you'd like to know?",
                "Let me know if you have any questions.",
                "Feel free to ask if you need anything else.",
            ]
            closing = self.rng.choice(closings)

            openers = [
                f"I've completed your request. Here's a summary of what I found: {body} {closing}",
                f"All done! Here are the results: {body} {closing}",
                f"Here's what I found for you: {body} {closing}",
            ]
            content = self.rng.choice(openers)
        else:
            # Fallback if no meaningful fields found
            tool_names = list(set(s.endpoint_id.split("::")[0] for s in all_steps))
            fallbacks = [
                f"I've completed all the steps successfully using {', '.join(tool_names)}. Let me know if you need anything else.",
                f"All done! I've retrieved the information you need. Let me know if you'd like to follow up.",
                f"Your request has been completed. Everything went through successfully. Let me know if you have questions.",
            ]
            content = self.rng.choice(fallbacks)

        return AssistantTurn(content=content)

    def _preamble_for_tool(self, tool_name: str, intent: str) -> str:
        """Generate a natural preamble before a tool call."""
        tool_label = tool_name.replace("_", " ")
        preambles = {
            "search":   [
                "Let me search for that information.",
                "I'll look that up for you.",
                "Searching now — one moment.",
                f"Let me check what's available.",
            ],
            "retrieve": [
                "Let me get the details.",
                "I'll fetch that information now.",
                "Looking that up for you.",
                "Let me pull that up.",
            ],
            "create":   [
                "I'll proceed with the booking.",
                "Let me complete that for you.",
                "Confirming that now.",
                "I'll take care of that.",
            ],
            "execute":  [
                "I'll run that calculation.",
                "Let me process that.",
                "Computing that for you.",
                "On it — processing now.",
            ],
            "update":   [
                "Updating that for you.",
                "I'll apply those changes.",
                "Let me save that.",
            ],
        }
        options = preambles.get(intent, [
            f"I'll use the {tool_label} tool.",
            f"Let me handle that with {tool_label}.",
        ])
        return self.rng.choice(options)

    def _intermediate_summary(self, tool_name: str, output: dict) -> str:
        """Summarize intermediate tool output briefly."""
        # Extract one meaningful value to mention
        _SKIP = {"status", "ok", "result", "success", "message", "error", "code"}
        _MEANINGFUL = {
            "reservation_id", "confirmation", "booking_id", "order_id",
            "converted_amount", "rate", "current_price", "temperature",
            "forecast", "name", "title",
        }
        mention = None
        for k, v in output.items():
            if k in _SKIP or isinstance(v, (dict, list)):
                continue
            if k in _MEANINGFUL:
                mention = f"{k.replace('_', ' ')}: {v}"
                break

        if mention:
            transitions = [
                f"Got it — {mention}.",
                f"Done — {mention}.",
                f"That went through. {mention.capitalize()}.",
            ]
        else:
            transitions = [
                "That's done.",
                "Got it.",
                "Done.",
            ]
        return self.rng.choice(transitions)

    def _final_summary(self, step: StepOutput) -> str:
        """Summarize the final step output."""
        tool_name = step.endpoint_id.split("::")[-1].replace("_", " ")
        return f"The {tool_name} call completed successfully. Here are the results."
