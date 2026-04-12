"""
Planner Agent — LLM Narrative Layer
Responsible ONLY for generating narrative fields.
Never touches structural fields (tool IDs, step indices, dependencies).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .models import (
    SampledToolChain,
    RegistryEndpointMetadata,
    CorpusSummary,
    PlanStep,
    ClarificationPoint,
    SummarySeedFields,
    VALID_CONVERSATION_STYLES,
)
from .scaffold import NoveltyHints


# ---------------------------------------------------------------------------
# Narrative request / response shapes
# ---------------------------------------------------------------------------

@dataclass
class NarrativeRequest:
    """Everything the LLM needs to generate narrative fields."""
    seed: int
    chain: SampledToolChain
    scaffold_steps: list[PlanStep]
    clarification_points: list[ClarificationPoint]
    novelty_hints: NoveltyHints
    registry: dict[tuple[str, str], RegistryEndpointMetadata] | None
    corpus_summaries: list[CorpusSummary]


@dataclass
class NarrativeOutput:
    """Raw LLM output before being merged into the ConversationPlan."""
    domain: str
    user_goal: str
    conversation_style: str
    style_notes: str
    # Per-step narrative, keyed by step_index
    step_narratives: dict[int, "StepNarrative"]


@dataclass
class StepNarrative:
    step_index: int
    purpose: str
    user_intent: str
    assistant_intent: str
    expected_output_usage: str | None
    may_require_clarification: bool
    clarification_reason: str | None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _format_corpus_summaries(summaries: list[CorpusSummary]) -> str:
    if not summaries:
        return "None."
    lines = []
    for i, s in enumerate(summaries, 1):
        lines.append(
            f"{i}. Domain: {s.domain or 'unknown'} | "
            f"Tools: {', '.join(s.tools) or 'unknown'} | "
            f"Style: {s.conversation_style or 'unknown'} | "
            f"Pattern: {s.pattern_type or 'unknown'}\n"
            f"   Summary: {s.content}"
        )
    return "\n".join(lines)


def _format_novelty_hints(hints: NoveltyHints) -> str:
    parts = []
    if hints.avoid_domains:
        parts.append(f"- Avoid these domains (already used): {hints.avoid_domains}")
    if hints.avoid_styles:
        parts.append(f"- Avoid these conversation styles (already used): {hints.avoid_styles}")
    if hints.avoid_pattern_types:
        parts.append(f"- Avoid these pattern types (already used): {hints.avoid_pattern_types}")
    if hints.suggested_style:
        parts.append(f"- Suggested conversation_style (least used so far): {hints.suggested_style}")
    return "\n".join(parts) if parts else "No diversity constraints — be creative."


def _format_steps(steps: list[PlanStep], registry: dict | None) -> str:
    lines = []
    for s in steps:
        reg_key = (s.tool_id, s.endpoint_id)
        endpoint_desc = ""
        if registry and reg_key in registry:
            meta = registry[reg_key]
            endpoint_desc = f" ({meta.description or meta.endpoint_name or ''})"
        lines.append(
            f"  Step {s.step_index}: tool={s.tool_id}, endpoint={s.endpoint_id}{endpoint_desc}, "
            f"depends_on={s.depends_on_steps}"
        )
    return "\n".join(lines)


def _format_clarification_points(cps: list[ClarificationPoint]) -> str:
    if not cps:
        return "None detected."
    lines = []
    for cp in cps:
        lines.append(
            f"  Before step {cp.before_step}: fields={cp.missing_or_ambiguous_fields}, "
            f"reason='{cp.reason}'"
        )
    return "\n".join(lines)


def build_narrative_prompt(req: NarrativeRequest) -> str:
    return f"""You are generating narrative fields for a structured ConversationPlan.

SEED: {req.seed}
PATTERN TYPE: {req.chain.pattern_type or "unknown"}
DOMAIN HINT: {req.chain.domain_hint or "none"}
CONCEPT TAGS: {req.chain.concept_tags or []}

[Prior corpus conversations]
{_format_corpus_summaries(req.corpus_summaries)}

[Diversity guidance]
{_format_novelty_hints(req.novelty_hints)}

[Tool chain steps — DO NOT change these]
{_format_steps(req.scaffold_steps, req.registry)}

[Clarification candidates — mark relevant steps accordingly]
{_format_clarification_points(req.clarification_points)}

Your task:
Generate ONLY the narrative fields listed below for this conversation plan.
Do NOT alter the tool chain, step indices, dependencies, tool IDs, or endpoint IDs.

Conversation style MUST be one of:
{list(VALID_CONVERSATION_STYLES)}

Return ONLY valid JSON with this exact structure:
{{
  "domain": "...",
  "user_goal": "...",
  "conversation_style": "...",
  "style_notes": "...",
  "steps": [
    {{
      "step_index": 0,
      "purpose": "...",
      "user_intent": "...",
      "assistant_intent": "...",
      "expected_output_usage": "...",
      "may_require_clarification": false,
      "clarification_reason": null
    }}
  ]
}}

Rules:
- domain: a realistic real-world domain (e.g. "travel planning", "e-commerce", "healthcare scheduling")
- user_goal: one clear sentence describing what the user wants to accomplish
- conversation_style: exactly one value from the enum above
- style_notes: one sentence describing how the user communicates in this scenario
- For each step: purpose describes what the step achieves; user_intent is what the user wants at that moment; assistant_intent is what the assistant will do; expected_output_usage explains how the output feeds into later steps (null if last step or unused)
- Mark may_require_clarification=true only for steps in the clarification candidates list, or where you identify genuine ambiguity
- Be diverse: avoid near-duplicates of prior corpus conversations shown above
"""


# ---------------------------------------------------------------------------
# LLM call (pluggable backend)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, llm_backend: Any) -> str:
    """
    Call the LLM backend with the given prompt.
    The backend must implement: .complete(prompt: str) -> str
    This keeps the narrative layer decoupled from any specific LLM client.
    """
    return llm_backend.complete(prompt)


# ---------------------------------------------------------------------------
# Deterministic stub backend (used when config.use_llm = False)
# ---------------------------------------------------------------------------

class DeterministicNarrativeBackend:
    """
    Offline stub backend. Produces valid, structurally correct narrative
    without any LLM call. Used when PlannerConfig.use_llm = False.

    Derives domain and user goal from the actual tool names in the chain
    so the narrative is coherent with what the tools actually do.
    Output always passes parse_narrative_response() validation.
    """

    # Maps substrings of tool names to (domain, goal_templates)
    _TOOL_DOMAIN_MAP: list[tuple[tuple[str, ...], str, list[str]]] = [
        (
            ("flight", "airline", "travel", "hotel", "booking", "trip", "itinerary"),
            "travel planning",
            [
                "Book flights and hotels for an upcoming trip.",
                "Plan travel arrangements including flights and accommodation.",
                "Find and book travel options for a vacation.",
            ],
        ),
        (
            ("weather", "forecast", "climate"),
            "weather information",
            [
                "Check the weather forecast before planning outdoor activities.",
                "Get current weather conditions for a destination.",
                "Look up weather information to plan a trip.",
            ],
        ),
        (
            ("currency", "exchange", "stock", "finance", "invest", "bank", "crypto"),
            "financial services",
            [
                "Check financial data and perform a currency conversion.",
                "Look up stock prices and make an informed financial decision.",
                "Research financial options and act on the results.",
            ],
        ),
        (
            ("shop", "product", "order", "cart", "store", "purchase", "retail", "ecommerce", "shopping"),
            "e-commerce",
            [
                "Find and purchase a product online.",
                "Research products and place an order.",
                "Compare product options and complete a purchase.",
            ],
        ),
        (
            ("restaurant", "recipe", "food", "meal", "cuisine", "dining", "menu", "delivery"),
            "food and dining",
            [
                "Find a good restaurant and make a reservation.",
                "Search for a recipe and get cooking instructions.",
                "Explore dining options and place an order.",
            ],
        ),
        (
            ("job", "career", "salary", "resume", "hiring", "recruitment", "jobboard"),
            "career services",
            [
                "Search for job listings and research salary ranges.",
                "Find career opportunities and apply for a position.",
                "Look up job postings and get company details.",
            ],
        ),
        (
            ("event", "ticket", "concert", "show", "entertainment", "movie", "ticketing"),
            "entertainment",
            [
                "Find tickets for an upcoming event.",
                "Browse entertainment options and book tickets.",
                "Search for events and make a booking.",
            ],
        ),
        (
            ("news", "article", "headline", "media", "press", "newsapi"),
            "news and media",
            [
                "Search for the latest news on a topic of interest.",
                "Find recent articles and get a summary of current events.",
                "Look up news articles to stay informed.",
            ],
        ),
        (
            ("map", "direction", "location", "navigation", "place", "address"),
            "maps and navigation",
            [
                "Find directions and nearby places of interest.",
                "Look up a location and get navigation details.",
                "Search for places near a destination.",
            ],
        ),
        (
            ("calendar", "schedule", "task", "reminder", "productivity", "meeting"),
            "productivity",
            [
                "Schedule a meeting and set up calendar reminders.",
                "Organize tasks and manage a schedule.",
                "Create calendar events and track upcoming commitments.",
            ],
        ),
        (
            ("email", "message", "communication", "contact", "sms", "chat"),
            "communication",
            [
                "Send a message and track communication history.",
                "Look up contact details and send an email.",
                "Compose and send a message to a contact.",
            ],
        ),
    ]

    _DEFAULT_DOMAIN = "general assistance"
    _DEFAULT_GOALS = [
        "Complete a multi-step task using the available tools.",
        "Find relevant information and take a follow-up action.",
        "Retrieve details and confirm the results.",
    ]

    _STYLES = [
        "direct", "exploratory", "underspecified",
        "preference_driven", "correction_heavy", "comparison_oriented", "goal_driven",
    ]
    _STYLE_NOTES = [
        "User communicates clearly and directly.",
        "User explores options before committing.",
        "User provides partial information and clarifies when prompted.",
        "User has strong preferences and states them upfront.",
        "User corrects the assistant when results are not quite right.",
        "User wants to compare options before deciding.",
        "User is focused on completing a specific goal efficiently.",
    ]

    def _best_domain_for_tool(self, tool_name: str) -> str | None:
        """Return the domain that best matches a single tool name."""
        tool_lower = tool_name.lower()
        best_domain: str | None = None
        best_score = 0
        for keywords, domain, _goals in self._TOOL_DOMAIN_MAP:
            score = sum(1 for kw in keywords if kw in tool_lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain

    def _derive_domain_and_goals(
        self, tool_names: list[str]
    ) -> tuple[str, list[str], bool]:
        """
        Match tool names against the domain map and return the best-fit
        (domain, goal_templates, is_multi_domain) tuple.

        Scoring: each tool contributes 1 point to its best-matching domain.
        This prevents multi-keyword domains (entertainment) from unfairly
        dominating single-keyword ones (weather).

        is_multi_domain is True when 2+ domains share the top score — signals
        that a generic multi-task goal should be used instead of a domain-specific one.
        """
        domain_scores: dict[str, int] = {}
        domain_goals: dict[str, list[str]] = {}

        # Deduplicate tool names — duplicate tools shouldn't inflate scores
        seen_tools: set[str] = set()
        for tool in tool_names:
            tool_lower = tool.lower()
            if tool_lower in seen_tools:
                continue
            seen_tools.add(tool_lower)
            best_domain = self._best_domain_for_tool(tool_lower)
            if best_domain:
                domain_scores[best_domain] = domain_scores.get(best_domain, 0) + 1
                # Store goals for this domain
                for kws, d, goals in self._TOOL_DOMAIN_MAP:
                    if d == best_domain and d not in domain_goals:
                        domain_goals[d] = goals

        if not domain_scores:
            return self._DEFAULT_DOMAIN, self._DEFAULT_GOALS, False

        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]

        # Detect multi-domain: only true ties (same score) trigger multi-domain
        tied_domains = [d for d, s in domain_scores.items() if s == max_score]
        is_multi_domain = len(tied_domains) >= 2

        return best_domain, domain_goals.get(best_domain, self._DEFAULT_GOALS), is_multi_domain

    # Maps first-tool substrings to opening user goals — goal should naturally
    # lead the assistant to call that tool FIRST.
    _FIRST_TOOL_GOALS: list[tuple[tuple[str, ...], str]] = [
        (("convert", "currency", "exchange"), "I need to convert some currency."),
        (("search_flight", "find_flight", "flight_search"), "I'd like to search for available flights."),
        (("book_flight",), "I'd like to book a flight for my trip."),
        (("hotel", "accommodation"), "I need to find and book a hotel."),
        (("weather",), "I'd like to check the weather forecast."),
        (("restaurant", "dining"), "I'm looking for a restaurant recommendation."),
        (("recipe",), "I need help finding a recipe."),
        (("stock", "market", "ticker"), "I'd like to check some stock prices."),
        (("news", "article", "headline"), "I want to search for recent news."),
        (("job", "career", "salary"), "I'm looking for job opportunities."),
        (("event", "ticket", "concert"), "I'd like to find tickets for an event."),
        (("map", "direction", "geocod"), "I need help with directions or finding a location."),
        (("calendar", "schedule", "appointment"), "I'd like to schedule an appointment."),
        (("translate", "translation"), "I need to translate some text."),
        (("product", "shop", "ecommerce"), "I'm looking to search for a product online."),
        (("profile", "account", "user"), "I need to look up some account information."),
    ]

    def _goal_for_first_tool(self, first_tool: str) -> str | None:
        """Return a user goal that naturally leads to calling first_tool."""
        tool_lower = first_tool.lower()
        for keywords, goal in self._FIRST_TOOL_GOALS:
            if any(kw in tool_lower for kw in keywords):
                return goal
        return None

    # Maps endpoint name substrings → (purpose, assistant_intent) describing what the step does.
    # assistant_intent is shown as the assistant's preamble before calling the tool.
    _ENDPOINT_STEP_MAP: list[tuple[tuple[str, ...], str, str]] = [
        (("book_hotel", "hotel_booking", "reserve_hotel"),
         "Book a hotel for the user's stay.",
         "I'll book the hotel for you now."),
        (("hotel",),
         "Look up hotel details or availability.",
         "Let me check the hotel information for you."),
        (("book_flight", "flight_booking"),
         "Book a flight for the user's trip.",
         "I'll go ahead and book the flight for you."),
        (("search_flight", "flight_search", "find_flight", "get_flight_details", "flight_details"),
         "Search for available flights.",
         "Let me search for available flights."),
        (("flight",),
         "Handle a flight-related request.",
         "Looking up flight information now."),
        (("weather_forecast", "get_forecast", "get_weather"),
         "Check the weather forecast.",
         "Let me pull up the weather forecast for you."),
        (("current_weather", "weather_current"),
         "Get the current weather conditions.",
         "Checking the current weather conditions."),
        (("weather",),
         "Retrieve weather information.",
         "Fetching weather data for you."),
        (("convert_currency", "currency_convert"),
         "Convert the currency amount.",
         "I'll convert the currency for you."),
        (("currency", "exchange_rate"),
         "Look up currency or exchange rate information.",
         "Looking up the exchange rate now."),
        (("stock", "get_stock", "stock_price", "company_financials"),
         "Retrieve financial market data.",
         "Let me fetch the latest market data."),
        (("update_preference", "save_preference", "update_profile", "save_profile"),
         "Save the user's preferences to their profile.",
         "I'll save your preferences now."),
        (("user_profile", "get_profile", "profile"),
         "Look up or update the user's profile.",
         "Let me access your profile."),
        (("search_restaurant", "find_restaurant", "restaurant_search"),
         "Search for restaurants matching the user's preferences.",
         "Searching for restaurants for you."),
        (("get_restaurant", "restaurant_details", "restaurant_menu"),
         "Retrieve restaurant details or menu.",
         "Let me pull up the restaurant information."),
        (("make_reservation", "book_restaurant", "restaurant_reservation"),
         "Make a restaurant reservation.",
         "I'll book the restaurant reservation for you."),
        (("restaurant",),
         "Handle a restaurant-related request.",
         "Looking up restaurant information."),
        (("search_job", "job_search", "find_job"),
         "Search for job listings.",
         "Let me search for relevant job listings."),
        (("get_job", "job_details"),
         "Retrieve job listing details.",
         "Fetching the job details for you."),
        (("salary",),
         "Look up salary information.",
         "Checking salary data now."),
        (("search_event", "event_search", "find_event"),
         "Search for events or shows.",
         "Let me search for available events."),
        (("purchase_ticket", "book_ticket", "buy_ticket"),
         "Purchase tickets for the event.",
         "I'll book the tickets for you now."),
        (("event",),
         "Handle an event-related request.",
         "Looking up event information."),
        (("search_news", "news_search", "get_news"),
         "Search for recent news articles.",
         "Searching the latest news for you."),
        (("get_article", "article_details"),
         "Retrieve a news article.",
         "Fetching the article for you."),
        (("search_product", "product_search", "find_product"),
         "Search for products online.",
         "Searching for products that match your request."),
        (("get_product", "product_details"),
         "Retrieve product details.",
         "Let me pull up the product information."),
        (("add_to_cart", "purchase", "order_product"),
         "Add the item to cart or complete the purchase.",
         "Adding that to your cart now."),
        (("translate", "translation"),
         "Translate the text.",
         "I'll translate that for you."),
        (("detect_language", "identify_language"),
         "Detect the language of the text.",
         "Let me identify the language for you."),
        (("geocode", "directions", "nearby", "maps"),
         "Look up location or navigation information.",
         "Checking the location details for you."),
        (("schedule", "calendar", "create_event"),
         "Create a calendar event or schedule.",
         "I'll add that to your calendar."),
        (("send_email", "compose_email", "email"),
         "Send or compose an email message.",
         "Sending the email for you now."),
    ]

    def _step_description(self, endpoint_name: str, step_idx: int, n_steps: int) -> tuple[str, str]:
        """Return (purpose, assistant_intent) for the given endpoint name."""
        ep_lower = endpoint_name.lower()
        for keywords, purpose, intent in self._ENDPOINT_STEP_MAP:
            if any(kw in ep_lower for kw in keywords):
                return purpose, intent
        # Generic fallback
        label = endpoint_name.replace("_", " ")
        return (
            f"Complete the {label} step toward the user goal.",
            f"I'll take care of the {label} for you.",
        )

    def complete(self, prompt: str) -> str:
        import json, re, hashlib

        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)

        # Extract tool names and endpoint names from the prompt's tool chain section
        # Step lines appear as "Step N: tool=tool_name, endpoint=tool::endpoint_name, ..."
        step_tool_matches = re.findall(r"Step\s+\d+:\s+tool=([a-zA-Z0-9_]+)", prompt)
        step_endpoint_matches = re.findall(r"endpoint=([a-zA-Z0-9_:]+)", prompt)
        all_tool_matches = re.findall(r"tool=([a-zA-Z0-9_]+)", prompt)
        # Also parse DOMAIN HINT line
        hint_match = re.search(r"DOMAIN HINT:\s*(.+)", prompt)
        hint_tools = []
        if hint_match:
            hint_tools = [t.strip() for t in hint_match.group(1).split(",")]

        all_tools = step_tool_matches or all_tool_matches
        domain, domain_goals, is_multi_domain = self._derive_domain_and_goals(all_tools + hint_tools)

        # Derive user_goal from the FIRST tool so the opening message naturally
        # leads into the first tool call. For multi-domain chains use a generic
        # multi-task framing so later tool calls don't seem random.
        first_tool = (step_tool_matches or hint_tools or [""])[0]
        first_tool_goal = self._goal_for_first_tool(first_tool)

        if is_multi_domain:
            # Generic goals that work for any combination of tools
            multi_task_goals = [
                "I have a few different tasks I need help with today.",
                "I need assistance with several things.",
                "I have some tasks to get done — could you help me work through them?",
                "I need to take care of a few things. Can you help?",
                "I have a couple of tasks I need to complete.",
            ]
            goal = multi_task_goals[h % len(multi_task_goals)]
            # Use "general assistance" as domain so the opener matches
            domain = "general assistance"
        elif first_tool_goal:
            goal = first_tool_goal
            # Override domain to match the first tool so opener and goal are coherent.
            # Without this, domain could be "news and media" while goal says "search for a product."
            first_tool_lower = first_tool.lower()
            for kws, d, _goals in self._TOOL_DOMAIN_MAP:
                if any(kw in first_tool_lower for kw in kws):
                    domain = d
                    break
        else:
            goal = domain_goals[h % len(domain_goals)]

        style = self._STYLES[h % len(self._STYLES)]
        note = self._STYLE_NOTES[self._STYLES.index(style)]

        step_matches = re.findall(r"Step (\d+):", prompt)
        n_steps = max(int(m) for m in step_matches) + 1 if step_matches else 1

        # Build per-step descriptions using endpoint name when available
        # step_endpoint_matches are in order of steps
        steps = []
        for i in range(n_steps):
            is_last = i == n_steps - 1
            # Get endpoint name for this step (strip tool:: prefix if present)
            ep_full = step_endpoint_matches[i] if i < len(step_endpoint_matches) else ""
            ep_name = ep_full.split("::")[-1] if "::" in ep_full else ep_full
            purpose, assistant_intent = self._step_description(ep_name, i, n_steps)
            steps.append({
                "step_index": i,
                "purpose": purpose,
                "user_intent": f"User wants the assistant to complete step {i}.",
                "assistant_intent": assistant_intent,
                "expected_output_usage": (
                    None if is_last
                    else f"Output from step {i} feeds into step {i + 1}."
                ),
                "may_require_clarification": False,
                "clarification_reason": None,
            })

        return json.dumps({
            "domain": domain,
            "user_goal": goal,
            "conversation_style": style,
            "style_notes": note,
            "steps": steps,
        })


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip markdown fences if present, then return the JSON string."""
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return raw


def parse_narrative_response(raw: str, scaffold_steps: list[PlanStep]) -> NarrativeOutput:
    """
    Parse and validate the LLM's JSON response into a NarrativeOutput.
    Raises ValueError with a descriptive message if parsing fails.
    """
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM narrative response is not valid JSON: {e}\nRaw: {raw[:500]}")

    required_top = {"domain", "user_goal", "conversation_style", "style_notes", "steps"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(f"LLM narrative response missing top-level fields: {missing}")

    if data["conversation_style"] not in VALID_CONVERSATION_STYLES:
        raise ValueError(
            f"LLM returned invalid conversation_style '{data['conversation_style']}'. "
            f"Valid: {VALID_CONVERSATION_STYLES}"
        )

    scaffold_indices = {s.step_index for s in scaffold_steps}
    step_narratives: dict[int, StepNarrative] = {}

    for raw_step in data["steps"]:
        idx = raw_step.get("step_index")
        if idx is None:
            raise ValueError(f"Narrative step missing step_index: {raw_step}")
        if idx not in scaffold_indices:
            raise ValueError(
                f"Narrative step_index {idx} not in scaffold indices {scaffold_indices}."
            )
        for field in ("purpose", "user_intent", "assistant_intent"):
            if not raw_step.get(field, "").strip():
                raise ValueError(f"Narrative step {idx} has empty field '{field}'.")

        may_clarify = bool(raw_step.get("may_require_clarification", False))
        clarify_reason = raw_step.get("clarification_reason") or None
        if may_clarify and not clarify_reason:
            raise ValueError(
                f"Narrative step {idx} has may_require_clarification=True but no clarification_reason."
            )

        step_narratives[idx] = StepNarrative(
            step_index=idx,
            purpose=raw_step["purpose"].strip(),
            user_intent=raw_step["user_intent"].strip(),
            assistant_intent=raw_step["assistant_intent"].strip(),
            expected_output_usage=(raw_step.get("expected_output_usage") or "").strip() or None,
            may_require_clarification=may_clarify,
            clarification_reason=clarify_reason,
        )

    # Every scaffold step must have a narrative
    for s in scaffold_steps:
        if s.step_index not in step_narratives:
            raise ValueError(
                f"Narrative response is missing step_index {s.step_index}."
            )

    return NarrativeOutput(
        domain=data["domain"].strip(),
        user_goal=data["user_goal"].strip(),
        conversation_style=data["conversation_style"],
        style_notes=data["style_notes"].strip(),
        step_narratives=step_narratives,
    )


# ---------------------------------------------------------------------------
# Merge scaffold + narrative → enriched steps + summary seed fields
# ---------------------------------------------------------------------------

def merge_narrative_into_steps(
    scaffold_steps: list[PlanStep],
    narrative: NarrativeOutput,
    clarification_points: list[ClarificationPoint],
) -> list[PlanStep]:
    """
    Produce final PlanStep list by merging structural scaffold with LLM narrative.
    Structural fields are always taken from the scaffold — never from the LLM.
    """
    # Build a lookup: step_index -> ClarificationPoint reason text
    cp_reason_by_step: dict[int, str] = {
        cp.before_step: cp.reason for cp in clarification_points
    }
    enriched: list[PlanStep] = []

    for s in scaffold_steps:
        n = narrative.step_narratives[s.step_index]
        # Determine whether this step requires clarification.
        # Priority: scaffold flag > LLM flag > clarification detection hit
        cp_flagged = s.step_index in cp_reason_by_step
        may_clarify = s.may_require_clarification or n.may_require_clarification or cp_flagged

        # Populate clarification_reason — must be non-None whenever may_clarify is True.
        # Pull from scaffold, then LLM, then the ClarificationPoint reason text.
        clarify_reason = (
            s.clarification_reason
            or n.clarification_reason
            or (cp_reason_by_step.get(s.step_index) if cp_flagged else None)
        )

        enriched.append(
            PlanStep(
                # Structural — from scaffold only
                step_index=s.step_index,
                tool_id=s.tool_id,
                endpoint_id=s.endpoint_id,
                depends_on_steps=s.depends_on_steps,
                # Narrative — from LLM
                purpose=n.purpose,
                user_intent=n.user_intent,
                assistant_intent=n.assistant_intent,
                expected_output_usage=n.expected_output_usage,
                may_require_clarification=may_clarify,
                clarification_reason=clarify_reason,
            )
        )

    return enriched


def build_summary_seed_fields(
    domain: str,
    pattern_type: str,
    tools_used: list[str],
    conversation_style: str,
    clarification_points: list[ClarificationPoint],
) -> SummarySeedFields:
    return SummarySeedFields(
        domain=domain,
        pattern_type=pattern_type,
        tools_used=sorted(tools_used),
        conversation_style=conversation_style,
        planned_clarification_count=len(clarification_points),
    )
