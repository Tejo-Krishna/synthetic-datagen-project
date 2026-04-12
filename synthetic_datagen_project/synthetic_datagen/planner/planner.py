"""
planner/planner.py
------------------
Planner Agent — converts a SampledChain into a ConversationPlan.

Responsibilities:
  - Turn a structural chain into a plausible user goal
  - Decide whether to inject intent_ambiguity clarification steps
  - Consult corpus memory before planning (for diversity)
  - Preserve the sampled chain as the structural backbone
  - Stage conversation turns

Must NOT:
  - Invent a different tool chain than the one sampled
  - Fill concrete argument values (Executor's job)
  - Call tools or execute anything
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from synthetic_datagen.common.types import (
    SampledChain, ClarificationStep, Transition
)
from synthetic_datagen.graph.registry import ToolRegistry
from synthetic_datagen.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# ConversationPlan — output of the Planner
# ---------------------------------------------------------------------------

@dataclass
class TurnPlan:
    """Plan for one turn in the conversation."""
    turn_index: int
    turn_type: str          # "user_request" | "clarification_ask" | "tool_call" | "final_response"
    endpoint_id: str | None  # which endpoint to call (if tool_call)
    description: str        # what should happen in this turn


@dataclass
class ConversationPlan:
    """
    Structured plan for one synthetic conversation.
    The Generator uses this to produce actual dialogue.
    """
    conversation_id: str
    chain: SampledChain
    user_goal: str                          # plausible user request motivating the chain
    domain: str                             # domain/category of the conversation
    turns: list[TurnPlan]
    clarification_steps: list[ClarificationStep]  # includes both sampler + planner injections
    corpus_memory_used: bool = False        # whether corpus memory influenced this plan
    seed: int | None = None


# ---------------------------------------------------------------------------
# Goal templates per domain
# ---------------------------------------------------------------------------

_GOAL_TEMPLATES: dict[str, list[str]] = {
    "Travel": [
        "Help me plan a trip to {destination}",
        "I need to book a flight and hotel for my upcoming trip",
        "Can you help me organize travel arrangements?",
        "I'm planning a vacation and need help with flights and accommodation",
    ],
    "Weather": [
        "What's the weather like in {location}?",
        "I need weather information to plan my outdoor activities",
        "Can you check the forecast for my destination?",
    ],
    "Finance": [
        "Help me analyze some financial information",
        "I need help with currency conversion and financial data",
        "Can you look up some stock and financial data for me?",
    ],
    "Shopping": [
        "I'm looking to buy {item} and want to compare options",
        "Help me find the best product for my needs",
        "I need help researching products before making a purchase",
    ],
    "Food": [
        "Help me find a good restaurant for tonight",
        "I need a recipe for {dish}",
        "Can you help me plan a dinner?",
    ],
    "Productivity": [
        "Help me schedule a meeting",
        "I need to organize my calendar",
        "Can you help me set up an appointment?",
    ],
    "Career": [
        "I'm looking for a new job in {field}",
        "Help me research career opportunities",
        "Can you help me find job listings and salary information?",
    ],
    "Entertainment": [
        "I want to find tickets for an event",
        "Help me find something fun to do this weekend",
        "Can you help me book event tickets?",
    ],
    "News": [
        "What's the latest news about {topic}?",
        "Help me catch up on current events",
        "Can you search for recent news articles?",
    ],
    "Maps": [
        "Help me find directions and nearby places",
        "I need to navigate to a location",
        "Can you help me find places near me?",
    ],
}

_AMBIGUOUS_OPENERS: list[str] = [
    "I need some help with something",
    "Can you assist me with a task?",
    "I have a few things I need to do",
    "Help me out with this",
    "I'm trying to figure out what to do",
]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Converts a SampledChain into a ConversationPlan.

    Uses corpus memory (if enabled) to diversify the user goals and
    conversation scenarios across the generated dataset.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory_store: MemoryStore | None = None,
        corpus_memory_enabled: bool = True,
        seed: int | None = None,
    ):
        self.registry = registry
        self.memory = memory_store
        self.corpus_memory_enabled = corpus_memory_enabled
        self.rng = random.Random(seed)

    def plan(
        self,
        chain: SampledChain,
        conversation_id: str,
        seed: int | None = None,
    ) -> ConversationPlan:
        """
        Convert a SampledChain into a ConversationPlan.

        Steps:
          1. Consult corpus memory for diversity guidance
          2. Choose a user goal appropriate for the chain's domain
          3. Optionally inject intent_ambiguity clarification at step 0
          4. Stage turn-by-turn plan
          5. Return ConversationPlan
        """
        if seed is not None:
            self.rng = random.Random(seed)

        # Step 1: Consult corpus memory
        corpus_context = []
        corpus_prompt_prefix = ""
        if self.corpus_memory_enabled and self.memory:
            domain = self._infer_domain(chain)
            corpus_context = self.memory.search(
                query=f"{domain} tool conversation",
                scope="corpus",
                top_k=3,
            )

        # Build the corpus planning prompt the PDF requires:
        # [Prior conversations in corpus]
        # {retrieved_summaries}
        # Given the above, plan a new diverse conversation using the
        # following tool chain: {proposed_tool_chain}
        if corpus_context:
            summaries = "\n".join(
                f"- {e.get('memory', '')}" for e in corpus_context if e.get("memory")
            )
            tool_chain_desc = ", ".join(chain.tool_ids)
            corpus_prompt_prefix = (
                f"[Prior conversations in corpus]\n{summaries}\n\n"
                f"Given the above, plan a new diverse conversation using the "
                f"following tool chain: {tool_chain_desc}\n"
            )
            # Store on plan for audit — used by structured planner's narrative layer
            self._last_corpus_prompt = corpus_prompt_prefix

        # Step 2: Choose user goal
        domain = self._infer_domain(chain)
        user_goal = self._choose_goal(chain, domain, corpus_context)

        # Step 3: Decide on intent_ambiguity injection
        clarification_steps = list(chain.clarification_steps)
        should_add_ambiguity = self._should_add_ambiguity(chain, corpus_context)

        if should_add_ambiguity:
            ambiguity_step = ClarificationStep(
                step_index=0,
                reason="intent_ambiguity",
                missing_params=[],
            )
            # Only add if not already a clarification at step 0
            if not any(cs.step_index == 0 for cs in clarification_steps):
                clarification_steps.insert(0, ambiguity_step)
                user_goal = self.rng.choice(_AMBIGUOUS_OPENERS)

        # Step 4: Stage turns
        turns = self._stage_turns(chain, clarification_steps)

        return ConversationPlan(
            conversation_id=conversation_id,
            chain=chain,
            user_goal=user_goal,
            domain=domain,
            turns=turns,
            clarification_steps=clarification_steps,
            corpus_memory_used=bool(corpus_context),
            seed=seed,
        )

    def _infer_domain(self, chain: SampledChain) -> str:
        """Infer the primary domain from the chain's tool categories."""
        categories: dict[str, int] = {}
        for eid in chain.endpoint_ids:
            ep = self.registry.get_endpoint(eid)
            if ep:
                categories[ep.category] = categories.get(ep.category, 0) + 1
        if not categories:
            return "General"
        return max(categories, key=categories.get)

    def _choose_goal(
        self,
        chain: SampledChain,
        domain: str,
        corpus_context: list[dict],
    ) -> str:
        """Choose a plausible user goal for the chain."""
        # Get recently used goals to avoid repetition
        recent_goals: set[str] = set()
        for entry in corpus_context:
            content = entry.get("memory", "")
            if "goal:" in content.lower():
                recent_goals.add(content[:80])

        templates = _GOAL_TEMPLATES.get(domain, [
            f"Help me with a {domain.lower()} related task",
            f"I need assistance with {domain.lower()}",
        ])

        # Prefer templates not recently used
        available = [t for t in templates if t not in recent_goals] or templates
        goal = self.rng.choice(available)

        # Fill in simple placeholders from endpoint names
        goal = goal.replace("{destination}", "Paris")
        goal = goal.replace("{location}", "New York")
        goal = goal.replace("{item}", "laptop")
        goal = goal.replace("{dish}", "pasta carbonara")
        goal = goal.replace("{field}", "software engineering")
        goal = goal.replace("{topic}", "technology")

        return goal

    def _should_add_ambiguity(
        self,
        chain: SampledChain,
        corpus_context: list[dict],
    ) -> bool:
        """
        Decide whether to inject an intent_ambiguity clarification.
        Adds ambiguity ~25% of the time, more if corpus memory suggests
        prior conversations have been too direct.
        """
        base_probability = 0.25

        # If corpus memory shows many direct conversations, increase ambiguity
        if corpus_context:
            direct_count = sum(
                1 for e in corpus_context
                if "ambiguity" not in e.get("memory", "").lower()
            )
            if direct_count >= len(corpus_context) * 0.7:
                base_probability = 0.4

        return self.rng.random() < base_probability

    def _stage_turns(
        self,
        chain: SampledChain,
        clarification_steps: list[ClarificationStep],
    ) -> list[TurnPlan]:
        """Stage the conversation into ordered turns."""
        turns: list[TurnPlan] = []
        turn_idx = 0

        # Clarification steps at index 0 come before any tool calls
        step0_clarifications = [cs for cs in clarification_steps if cs.step_index == 0]

        # Turn 0: user request
        turns.append(TurnPlan(
            turn_index=turn_idx,
            turn_type="user_request",
            endpoint_id=None,
            description="User states their goal or request",
        ))
        turn_idx += 1

        # If step 0 clarification, add clarification exchange
        if step0_clarifications:
            turns.append(TurnPlan(
                turn_index=turn_idx,
                turn_type="clarification_ask",
                endpoint_id=None,
                description=f"Assistant asks for clarification: {step0_clarifications[0].reason}",
            ))
            turn_idx += 1
            turns.append(TurnPlan(
                turn_index=turn_idx,
                turn_type="user_request",
                endpoint_id=None,
                description="User provides clarification",
            ))
            turn_idx += 1

        # Tool call turns
        for step_idx, endpoint_id in enumerate(chain.endpoint_ids):
            # Mid-chain clarifications
            mid_clarifs = [cs for cs in clarification_steps
                          if cs.step_index == step_idx and step_idx > 0]
            if mid_clarifs:
                turns.append(TurnPlan(
                    turn_index=turn_idx,
                    turn_type="clarification_ask",
                    endpoint_id=None,
                    description=f"Assistant needs more info before step {step_idx}",
                ))
                turn_idx += 1
                turns.append(TurnPlan(
                    turn_index=turn_idx,
                    turn_type="user_request",
                    endpoint_id=None,
                    description="User provides missing information",
                ))
                turn_idx += 1

            ep = self.registry.get_endpoint(endpoint_id)
            ep_name = ep.name if ep else endpoint_id
            turns.append(TurnPlan(
                turn_index=turn_idx,
                turn_type="tool_call",
                endpoint_id=endpoint_id,
                description=f"Assistant calls {ep_name}",
            ))
            turn_idx += 1

        # Final response
        turns.append(TurnPlan(
            turn_index=turn_idx,
            turn_type="final_response",
            endpoint_id=None,
            description="Assistant synthesizes results and completes the task",
        ))

        return turns

    def write_to_corpus_memory(
        self,
        plan: ConversationPlan,
        conversation_id: str,
    ) -> None:
        """
        Write a summary of a completed conversation to corpus memory.
        Called after conversation is generated and validated.
        """
        if not self.memory or not self.corpus_memory_enabled:
            return

        summary = (
            f"Tools: {', '.join(plan.chain.tool_ids)}. "
            f"Domain: {plan.domain}. "
            f"Pattern: {plan.chain.pattern_type}. "
            f"Goal: {plan.user_goal[:60]}. "
            f"Clarification: {plan.chain.requires_clarification}."
        )

        self.memory.add(
            content=summary,
            scope="corpus",
            metadata={
                "conversation_id": conversation_id,
                "tools": plan.chain.tool_ids,
                "pattern_type": plan.chain.pattern_type,
                "domain": plan.domain,
            },
        )
