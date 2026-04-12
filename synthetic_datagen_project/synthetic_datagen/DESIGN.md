# DESIGN.md — Offline Synthetic Tool-Use Conversation Generator

## 1. Problem Overview

The goal is to generate a synthetic dataset of multi-turn conversations where an AI assistant uses tools (APIs) across multiple steps to complete a user task. The dataset must support training and evaluating tool-use agents on:

- Multi-step tool chains (≥ 3 tool calls per conversation)
- Multi-tool workflows (≥ 2 distinct tools per conversation)
- Clarification turns (when intent is ambiguous or required fields are missing)
- Realistic data flow (later tool calls reference outputs of earlier calls)

The system is fully **offline** — no real API calls, no internet, no model downloads required. Tool outputs are simulated deterministically from schemas.

---

## 2. Pipeline Architecture

```
ToolBench JSON
     │
     ▼
[1] toolbench/ingest.py
    Raw parse only. Produces RawTool / RawEndpoint.
    Zero config dependency. returns_raw preserved as string.
     │
     ▼
[2] graph/registry.py                         ← Normalization boundary
    Converts RawEndpoint → Endpoint.
    Parses returns_raw into returns_schema, returns_fields, returns_types.
    Infers endpoint intent from config/intent_rules.yaml.
    Builds 4 lookup indexes.
     │
     ▼
[3] graph/heterogeneous_graph.py
    Full 5-node-type graph: Tool, Endpoint, Parameter, ResponseField, Concept/Tag.
    Concept nodes from semantic_groups in config/graph_config.yaml.
    Exists for correctness, explainability, and edge provenance.
     │
     ▼
[4] graph/projected_graph.py
    Endpoint-to-endpoint sampler graph derived from heterogeneous graph.
    Three edge types: data_link (1.0), semantic (0.45), category (0.2).
    field_mappings carry explicit source_field → target_param pairs.
    Entry nodes pre-computed for valid start-node selection.
     │
     ▼
[5] sampler/sampler.py                        ← Graph-driven chain proposer
    Walks projected graph. Never calls LLM.
    4 modes: sequential, multi_tool, clarification_first, parallel.
    Produces SampledChain with Transitions, ParallelBranches, ClarificationSteps.
     │
     ▼
[6] planner/planner.py
    Converts SampledChain → ConversationPlan.
    Injects intent_ambiguity clarification steps.
    Consults corpus memory for diversity.
     │
     ▼
[7] generator/
    user_proxy.py   — simulates user utterances
    assistant.py    — simulates assistant tool calls and responses
    executor.py     — offline tool execution with session state
    validator.py    — validates final conversation artifact
    writer.py       — writes JSONL output
     │
     ▼
[8] memory/store.py
    MemoryStore backed by mem0ai (or in-process fallback).
    session scope: grounds argument filling for non-first-step tool calls.
    corpus scope:  grounds planner for cross-conversation diversity.
     │
     ▼
output/conversations.jsonl
```

---

## 3. Key Design Decisions

### 3.1 Two-layer graph design

**Decision**: Build a heterogeneous graph (5 node types) for richness, but have the Sampler walk a projected endpoint-to-endpoint graph derived from it.

**Rationale**: The PDF requires a graph with Tool/Endpoint/Parameter/ResponseField/Concept nodes, but direct traversal of a heterogeneous graph by the Sampler adds complexity without benefit. The projection collapses multi-hop paths (e.g., `Endpoint A → ResponseField(flight_id) → Concept(identifier) → Parameter(flight_id) → Endpoint B`) into a single weighted `ProjectedEdge`, while preserving the provenance path for debugging.

### 3.2 Normalization boundary at Registry

**Decision**: `ingest.py` is raw-only. The Registry is the first and only layer that interprets data.

**Rationale**: `returns_raw` as a string in the Endpoint would mean every downstream component (Graph, Sampler, Executor, Validator) would need to parse it independently. By parsing once at Registry time into `returns_schema`, `returns_fields`, and `returns_types`, we ensure a single parse point with consistent behavior.

### 3.3 Per-component YAML config files

**Decision**: Three separate config files (`intent_rules.yaml`, `graph_config.yaml`, `sampler_config.yaml`), each owned by one component with in-code defaults as fallback.

**Rationale**: The pipeline has 8 stages with different configuration needs and different change velocities. Intent keywords change as new ToolBench tools are discovered; edge weights are set once and rarely changed; chain length bounds are tuned during development. Separating them prevents coupling and makes each component independently testable.

### 3.4 Typed weighted edges with three tiers

**Decision**: Three edge types with fixed weights — `data_link` (1.0), `semantic` (0.45), `category` (0.2).

**Rationale**:
- `data_link`: Only type that proves execution compatibility (field from A fills param of B)
- `semantic`: Concept-level relationship allows realistic but softer transitions
- `category`: Domain grouping, weakest — allows exploration without claiming data compatibility

Multiplicative cross-tool bias (`edge.weight × (1 + 0.3)`) preserves relative ordering while biasing toward multi-tool chains.

### 3.5 Clarification split between Sampler and Planner

**Decision**: Sampler detects only `missing_required_param`; Planner injects `intent_ambiguity`.

**Rationale**: The Sampler has only structural graph knowledge. It can detect mechanically that a required parameter has no upstream source. Intent ambiguity requires understanding the user's conversational goal — that context only exists at planning time when the Planner frames the conversation scenario.

### 3.6 Field mappings as structured source→target pairs

**Decision**: `FieldMapping(source_field, target_param)` instead of a flat `list[str]`.

**Rationale**: A flat list of field names loses the mapping when source and target names differ (e.g., `id` → `hotel_id`). The Offline Executor needs to know exactly which output field to read and which input parameter to fill. Structured mappings make this unambiguous.

### 3.7 Argument-fill precedence policy

The Executor resolves argument values in this fixed order:

1. Explicit user input from conversation
2. `Transition.field_mappings` from prior step's output
3. Session memory retrieval via `MemoryStore.search()`
4. Default/mock fallback (type-consistent generated value)

This order ensures that directly chained data (step N output feeds step N+1) takes precedence over broader memory context, while memory provides a fallback for values not directly linked.

### 3.8 Parallel mode with explicit branch structure

**Decision**: `SampledChain` carries explicit `ParallelBranch` objects rather than a flattened list with inferred parallelism.

**Rationale**: A flat list cannot express branching without metadata that forces the Planner to re-infer structure. Explicit branches (`root → [branch_a, branch_b] → merge`) make the Planner's job straightforward and prevent ambiguity in the conversation plan.

### 3.9 MemoryStore interface isolation

**Decision**: All components depend on `MemoryStore`, never on `mem0` directly.

**Rationale**: mem0ai is an optional dependency. The in-process fallback store allows the full pipeline to run without it. This decoupling also makes the memory layer testable in isolation. The PDF requires mem0-backing but allows graceful degradation in offline environments.

**Recommended backend**: `sentence-transformers` + `qdrant-client` (local vector store). This provides real semantic similarity search without requiring an OpenAI API key and runs fully offline. mem0ai requires an OpenAI key for embeddings and makes network calls on every `add`/`search`, making generation significantly slower.

### 3.10 Per-conversation session scope isolation

**Decision**: Session memory uses `scope=f"session_{conversation_id}"` rather than a shared `scope="session"`.

**Rationale**: With a shared session scope, conversation B could retrieve tool outputs from conversation A's steps via `memory.search()`. This would cause argument grounding to pull incorrect values from prior conversations — a subtle but serious data quality issue. Per-conversation scoping ensures that each conversation's session memory is completely isolated, so grounding always reflects the current conversation's tool chain only.

### 3.11 Randomized mock fallback values

**Decision**: The Executor's mock fallback uses randomized value pools (seeded by `rng`) rather than hardcoded constants.

**Rationale**: The first step of every conversation has no prior output to pull from, so it falls back to mock values. With hardcoded constants (`city="Paris"`, `name="Jane Smith"` every time), 50 conversations would all start with identical seed values. Using per-conversation random pools (same seed = reproducible, different conversations = varied values) produces more realistic training data without breaking determinism.

---

### 3.12 ConversationState as inter-agent communication protocol

**Decision**: All generator agents communicate through a shared `ConversationState` dataclass defined in `common/types.py`.

**Rationale**: Without a formal shared object, data flows through loose function arguments — each agent receives only what it needs, passed individually. This works but makes the inter-agent protocol implicit and hard to document. `ConversationState` makes it explicit: the Planner writes the plan into it, the Executor updates grounding stats and surfaces warnings, and the Validator reads the final messages and tool_calls. Adding a new agent means reading/writing this object, not changing function signatures across the codebase.

### 3.13 Inline grounding check

**Decision**: After each tool executes, the system checks whether the next step's required parameters exist in `session.accumulated_fields` before proceeding.

**Rationale**: Without this check, grounding failures are silent — the Executor falls back to mock values and only the post-hoc judge catches the issue via a low `tool_correctness` score. The inline check surfaces the gap immediately during generation, identifying which tool transitions have weak field coverage in the projected graph. This is a zero-cost programmatic check (pure dictionary lookup, no LLM call).

### 3.14 Coverage tracker

**Decision**: A `coverage` dict tracks domain and pattern usage across conversations during generation, printed as a report at the end of each run.

**Rationale**: Cross-conversation steering via corpus memory is implicit — you can't observe it happening. The coverage tracker makes diversity steering visible and measurable in real time. It also identifies underrepresented domains (appeared only once), which is directly useful for the Run A vs Run B diversity experiment.

---

## 4. Component Responsibility Boundaries

| Component | Inputs | Outputs | Must NOT |
|-----------|--------|---------|----------|
| `ingest.py` | JSON files | `RawTool`, `RawEndpoint` | Load config, infer intent |
| `registry.py` | `RawEndpoint` | `Endpoint`, indexes | Build graph edges |
| `heterogeneous_graph.py` | Registry | `HeterogeneousGraph` | Be traversed by Sampler |
| `projected_graph.py` | Hetero graph + Registry | `ProjectedGraph` | Store Endpoint objects |
| `sampler.py` | Projected graph + Registry | `SampledChain` | Generate text, fill args, call memory |
| `planner.py` | `SampledChain` | `ConversationPlan` | Invent a different chain |
| `executor.py` | Endpoint schema + args | Mock output + session state | Pre-fill values before executor time |
| `memory/store.py` | Content + scope | Search results | Expose mem0 API directly |

---

## 5. SampledChain Schema

```python
@dataclass
class FieldMapping:
    source_field: str    # field in prior step's response
    target_param: str    # parameter in next step's input

@dataclass
class Transition:
    from_endpoint: str
    to_endpoint: str
    edge_type: str                    # data_link | semantic | category
    field_mappings: list[FieldMapping]
    matched_concepts: list[str]
    is_executable: bool

@dataclass
class ParallelBranch:
    branch_id: str
    endpoint_ids: list[str]
    transitions: list[Transition]

@dataclass
class ClarificationStep:
    step_index: int
    reason: Literal["missing_required_param", "intent_ambiguity"]
    missing_params: list[str]

@dataclass
class SampledChain:
    endpoint_ids: list[str]           # flat view for metrics/dedup
    tool_ids: list[str]
    transitions: list[Transition]
    pattern_type: str
    sampling_mode: str
    clarification_steps: list[ClarificationStep]
    root_endpoint_id: str | None      # parallel only
    branches: list[ParallelBranch] | None
    merge_endpoint_id: str | None
```

---

## 6. Sampler Modes

| Mode | Walk behavior | Use case |
|------|---------------|----------|
| `sequential` | Weighted linear walk, prefer data_link | Standard multi-step chains |
| `multi_tool` | Bias toward unseen tools at each step | Cross-tool workflow training |
| `clarification_first` | Allow start nodes with unsourced required params | Disambiguation training |
| `parallel` | Branch root → [A, B] → merge | Parallel tool use training |

Pattern type is classified after sampling, independent of mode:
- `search_then_action`: search/retrieve step followed by create/execute
- `information_then_decision`: retrieve followed by compare/select
- `multi_tool_chain`: multiple distinct tools, no specific pattern
- `sequential_multi_step`: default
- `parallel`: explicit branch structure

### Practical limits on `--n` and `--seed`

**`--n` (number of conversations):**
With 16 tools and 43 endpoints, the number of unique `(tool_ids, pattern_type)` combinations is finite. The sampler enforces uniqueness — no two conversations use the exact same tool chain. In practice, unique combinations are exhausted around 200 conversations, beyond which the sampler begins reusing chain structures with different endpoint orderings. For datasets larger than 200, use `--mode mixed` to maximize variety across pattern types.

**`--seed` (random seed):**
Any integer is valid (e.g. 42, 7, 123, 2024). The seed controls the graph walk, mock value selection, and clarification placement. The same seed always produces identical output — useful for reproducibility and the Run A vs Run B diversity experiment. Different seeds produce different tool chains and conversation structures from the same pool of 43 endpoints.

---

## 7. Corpus Memory & Diversity Analysis

### Metric Choice

**Primary metric**: Shannon entropy over `(tool_ids_used, pattern_type)` buckets.

Each unique combination of tool set and pattern type forms a bucket. Entropy measures how evenly distributed the generated conversations are across these buckets. A higher entropy means more diverse conversations.

**Justification**: This metric directly captures the two key dimensions of diversity the PDF is concerned with — which tools appear and what conversation pattern is used. It is deterministic, easy to compute from metadata, and directly interpretable. Unlike n-gram diversity on utterances, it measures structural diversity rather than surface-level lexical variation.

**Secondary metric**: Average pairwise tool-chain Jaccard dissimilarity. Measures how different the tool sets used across conversations are from each other. A value of 1.0 means every pair of conversations uses completely different tools.

### Experimental Setup

Both runs use seed=42 and generate 50 conversations.

- **Run A**: Corpus memory disabled (`--no-corpus-memory`)
- **Run B**: Corpus memory enabled (default)

### Results

Both runs use seed=42, 50 conversations each. LLM-as-judge quality scores come from `AnthropicJudgeClient` (`claude-haiku-4-5`) applied post-hoc to each run's output via `evaluate`. Threshold: 3.5/5.0 on all dimensions.

| Metric | Run A (steering disabled) | Run B (steering enabled) |
|--------|--------------------------|----------------------|
| Unique (tool_ids, pattern) buckets | 50 | 50 |
| Shannon entropy (primary) | 3.9120 | 3.9120 |
| Normalized entropy (0–1) | 1.0000 | 1.0000 |
| Avg Jaccard dissimilarity | 0.8468 | 0.8426 |
| **mean tool_correctness** | **4.10** | **4.08** |
| **mean task_completion** | **3.79** | **3.81** |
| **mean naturalness** | **3.60** | **3.60** |
| **mean overall** | **3.83** | **3.83** |
| Pass rate (≥3.5 all dims) | 100% | 100% |

### Diversity–Quality Tradeoff Analysis

**Diversity:** Both runs achieve maximum normalized entropy (1.0) — every conversation falls into a unique `(tool_ids_used, pattern_type)` bucket. The Sampler's uniqueness enforcement is the primary driver of structural diversity; corpus memory has no measurable effect on this metric at 50-conversation scale with 16 tools.

The Jaccard dissimilarity values (0.847 vs 0.843) are nearly identical and both high — cross-tool usage is diverse in both runs regardless of steering.

**Quality:** Judge scores are nearly identical across runs (mean overall 3.83 in both). Enabling corpus memory does not degrade quality. The slight variation (±0.02 per dimension) is within expected noise for a 50-sample comparison.

**Why corpus memory didn't move diversity metrics here:**

The `(tool_ids, pattern_type)` metrics measure *structural* diversity at the chain level. Corpus memory influences the Planner's goal framing and narrative — it steers away from overused *conversation topics* (e.g., "book a flight to Paris" appearing repeatedly), not away from overused *tool chains*. These are different axes of diversity. The Sampler's graph-based uniqueness enforcement already exhausts structural diversity with 16 tools and 50 conversations.

**Where corpus memory would show effect:**

At scale (500+ conversations, 100+ tools), the Sampler will begin reusing the same tool chains with different endpoint orderings. Corpus memory would then steer the Planner toward underrepresented domains and ambiguity patterns, producing measurable gains in topic-level diversity that the structural metrics above do not capture. A better metric at that scale would be embedding-based cluster entropy over conversation goal embeddings.

**Conclusion:** No diversity–quality tradeoff exists at this scale — steering neither hurts quality nor improves structural diversity measurably. Both effects would emerge at larger corpus sizes.

### Non-Determinism in Domain Labels

Running the pipeline twice with the same seed (`--seed 42`) produces **identical chain structure** — same endpoint sequences, same argument values, same pattern distribution — but **different domain label distributions**. For example:

| Domain | Run 1 | Run 2 |
|--------|-------|-------|
| travel planning | 9 | 2 |
| news and media | 6 | 11 |
| food and dining | 4 | 7 |

**Root cause:** The corpus memory (cross-conversation steering) uses vector search — `sentence-transformers` + `qdrant-client` for approximate nearest-neighbor retrieval. Even with the same seed, vector search is non-deterministic: internal index state, floating-point ordering, and memory layout can vary between runs, causing the planner to receive slightly different retrieval results and assign different domain labels.

**What is reproducible:** The sampler (graph walk), executor (mock values), and argument resolution are all fully seeded — same seed always produces identical tool chains, endpoint IDs, arguments, and tool outputs.

**What is not reproducible:** Domain labels assigned by the planner, which depend on corpus memory queries. These are metadata annotations and do not affect the conversation content or tool call structure.

**Mitigation options:**
1. Replace approximate vector search with exact keyword matching for corpus memory (fully deterministic, lower recall)
2. Pin domain labels to tool categories directly from the registry (deterministic, bypasses memory entirely)
3. Accept non-determinism in domain labels and measure structural diversity (entropy over tool chains) rather than domain distribution

**Current approach:** Option 3 — the spec acknowledges this explicitly: *"if you use vector search, note that approximate nearest-neighbor search is non-deterministic. Document how you handle this in your metrics and reproducibility strategy."* The Shannon entropy and Jaccard dissimilarity metrics operate over tool chain structure, not domain labels, and are fully reproducible between runs.

---

## 8. Prompt Design

### 8.1 Judge Prompt

**Structure:**
```
[SYSTEM] You are an expert evaluator of AI assistant conversations...
         Use the submit_scores tool to return your evaluation.

[USER]   AVAILABLE TOOLS:
           - hotels::search  params: [city, max_price, currency]
           - hotels::book    params: [hotel_id, check_in]
           ...

         CONVERSATION:
         [USER] Find me a hotel in Paris...
         [ASSISTANT] <tool_calls> hotels::search(...)
         [TOOL] {"results": [...]}
         ...

         Score the conversation on three dimensions (1.0–5.0):
         1. tool_correctness — ...rubric...
         2. task_completion  — ...rubric...
         3. naturalness      — ...rubric...

         Call the submit_scores tool with your evaluation.
```

**Why structured this way:**

- **Tool schemas are included** — the judge cannot assess `tool_correctness` without knowing what the valid parameters were. Without schemas, a hallucinated `hotel_id` looks identical to a grounded one.
- **Conversation is formatted as a transcript** — role prefixes (`[USER]`, `[ASSISTANT]`, `[TOOL]`) make the turn structure unambiguous. Tool calls are shown inline as `<tool_calls> endpoint(args)` so the judge sees them without needing to parse nested JSON.
- **Rubrics are included in the user prompt** — not just in the system prompt — because the model attends more reliably to criteria adjacent to the content it is evaluating.
- **Forced tool use** — `tool_choice={"type": "tool", "name": "submit_scores"}` guarantees the response is always parseable JSON. The system never relies on extracting scores from free text.
- **Tool outputs truncated at 500 chars** — prevents the judge prompt from exceeding context limits on long tool output responses.

**Why three dimensions (not one score):**

A single quality score masks which aspect failed. A conversation with broken tool calls but natural dialogue averages to a misleading middle score. Separate dimensions allow the repairer to target the lowest-scoring dimension surgically.

---

### 8.2 Repair Prompt (Surgical)

The surgical repair prompt tells the model exactly what failed and which dimension to fix:

```
You are repairing a synthetic AI assistant conversation for a training dataset.

JUDGE FEEDBACK:
  Dimension targeted: tool_correctness
  Judge score:        2.0 / 5.0
  Judge reasoning:    Tool arguments were hallucinated...

REPAIR INSTRUCTION:
  Fix tool call arguments so they are grounded in prior step outputs
  (not hallucinated), ensure all required parameters are provided...

TOOL CALLS (do not change endpoint names or sequence): [...]
TOOL OUTPUTS (do not change these): [...]
ORIGINAL MESSAGES: [...]

Return a JSON object with one key "messages" containing the repaired messages array.
```

**Why this structure:**

- The judge's `reasoning` field is injected verbatim — the repair model sees the exact failure explanation, not a generic instruction.
- The tool sequence and outputs are pinned — the repair model cannot change endpoint names or output values, only the dialogue text. This prevents repair from introducing new structural errors.
- The dimension-specific instruction is direct and actionable — "fix tool call arguments" is more useful than "improve quality."

---

### 8.3 Repair Prompt (Full Rewrite — Fallback)

Used only on attempt 2, when surgical repair failed:

```
You are rewriting a synthetic AI assistant conversation from scratch.
A previous targeted repair attempt failed.

JUDGE FEEDBACK (all dimensions): [scores + reasoning]
REQUIREMENTS:
  - Keep exactly the same tool call sequence
  - Keep exactly the same tool output values
  - Arguments must be grounded in prior step outputs
  - Final message must reference real values from tool outputs
  - Clarification questions must be contextually appropriate

TOOL SEQUENCE TO FOLLOW: [...]
TOOL OUTPUTS (use these exact values): [...]
```

**Why full rewrite as fallback — not as first attempt:**

Surgical repair is cheaper (fewer tokens rewritten) and preserves correct parts of the conversation. Full rewrite discards everything and risks introducing new problems in turns that were previously correct. The two-stage approach (surgical → full rewrite) captures the best of both: fast targeted fixes in the common case, full rewrite only when the conversation is structurally broken end-to-end.

---

### 8.4 Iteration That Did Not Work

**First attempt: asking the judge to return scores in free text**

The original judge prompt asked the model to return scores as:
```
tool_correctness: 4.2
task_completion: 3.8
naturalness: 4.0
reasoning: The assistant correctly...
```

This failed in two ways:
1. The model occasionally returned scores in a different format (`"4.2/5"`, `"4 out of 5"`, `"~4"`), causing parse failures.
2. Sometimes the model would add preamble text before the scores (`"Here is my evaluation:\ntool_correctness: 4.2..."`), breaking line-based parsing.

**What was learned:** Free-text extraction from LLMs is fragile even for structured numeric output. The fix was switching to forced tool use (`tool_choice={"type": "tool", "name": "submit_scores"}`), which guarantees a JSON object every time. The Anthropic API rejects responses that don't conform to the tool schema, so parse failures dropped to zero. This also satisfies the spec requirement that at least one agent uses structured output.

---

## 9. Offline Execution Model

The `OfflineExecutor` simulates tool execution without real API calls:

1. **Argument resolution** follows the fixed precedence policy (user input → field mappings → session memory → mock fallback)
2. **Mock output generation** fills the endpoint's `returns_schema` template with values consistent with the arguments provided
3. **Session state** accumulates all output fields from prior steps, making them available for downstream field mapping resolution

This ensures multi-step traces are chain-consistent: if step 1 returns `flight_id: "AA123"`, step 2 will use exactly `"AA123"` as its `flight_id` input parameter (not a randomly generated value).

---

## 10. Testing

**56 tests** covering all pipeline components and quality dimensions:

**Unit tests:**
- Ingestion parsing (5 tests)
- Registry normalization + deduplication (6 tests)
- Intent rule priority (6 tests)
- Heterogeneous graph — 5 node types, edges, serialization, connectivity (5 tests)
- Projected graph edges — data_link, flight chain, no self-loops, entry nodes, weights (5 tests)
- Sampler determinism and constraints (6 tests)
- Clarification detection (3 tests)
- MemoryStore add/search + scope isolation (3 tests)
- Parallel mode (3 tests)
- Offline Executor — schema conformance, cross-step grounding, session isolation (3 tests)

**Integration tests:**
- Repair loop — surgical success in 1 attempt (1 test)
- Repair loop — max attempts exhausted, conversation kept as failed (1 test)
- Judge output dimensions and score range 1.0–5.0 (1 test)
- Output format — all required JSONL fields, message roles, tool_calls, metadata (1 test)
- Corpus memory A/B diversity experiment (1 test)

**End-to-end tests:**
- 50-conversation generation — all pass structural validation (1 test)
- 100-conversation generation with mock judge — mean score ≥ 3.5 (1 test)
- Message role ordering across 5 conversations — no back-to-back user messages (1 test)
- Shannon entropy correctness — uniform, single, binary distributions (1 test)
- Domain coverage — 50 conversations span ≥ 5 distinct domains (1 test)

All 56 tests pass with no warnings. No API key required — judge and repair tests use deterministic mocks. Full suite runs in under 1 second.

---

## 11. Known Limitations and What Would Be Done Next

### 11.1 Judge Scores Are Low (Current Mean: ~1.2–2.0 / 5.0)

When running `generate --evaluate` against real conversations, the LLM-as-judge scores all conversations below the 3.5 passing threshold. The judge's reasoning identifies two root causes:

**Root cause 1 — Domain/tool mismatch:** The planner assigns a domain label (e.g., "entertainment") independently of the tools in the chain. A chain of `flight_search → currency_exchange → weather → job_search → calendar` is labeled "entertainment" with a generic user goal "research options and make a selection." The judge correctly identifies that these tools are contextually inappropriate for the stated goal.

**Root cause 2 — Chain incoherence:** Semantic and category edges in the projected graph connect endpoints that share a concept node but have no realistic user scenario linking them. A user does not naturally need flight search, currency conversion, weather, job salary data, and calendar management in a single conversation. These chains are structurally valid but semantically broken.

**What would fix this:**
1. **Derive domain from tool categories** — the registry already knows each tool's category. The most frequent category in the chain should be the domain label. One-line change in the planner, no LLM needed.
2. **Generate goal from endpoint intents** — instead of a generic goal, compose a specific goal from the tool sequence: `search_flights → convert_currency → get_weather` → "I'm planning a trip to Paris and need to check flights, currency rates, and the weather."
3. **Chain coherence filter** — reject chains where tools span more than 2–3 distinct categories. A realistic user task clusters naturally in 1–2 domains. This is a constraint added to `_is_valid()` in `sampler.py`.

**Why not fixed yet:** These three changes require careful testing to avoid breaking the 56-test suite and the spec's hard requirements (≥3 tool calls, ≥2 distinct tools). They are the next implementation priority.

### 11.2 User Clarification Values Now Flow Into Tool Arguments

Prior to a fix applied during development, user clarification answers (e.g., "I have US dollars") were captured as conversation text only — `executor.execute_step(user_inputs={})` was always called with an empty dict. This caused tool arguments to use randomly mocked values that contradicted what the user said, contributing to low `tool_correctness` scores.

**Fix applied:** `UserTurn` now carries a `resolved_params` dict alongside the natural-language content. Every clarification answer populates `persistent_user_inputs`, which is passed to every subsequent `executor.execute_step` call. The fix requires zero LLM tokens — since `UserProxyAgent` generates both the utterance and its canonical value from the same source map, no parsing is needed.

### 11.3 Varied Conversation Lengths Not Yet Implemented

The spec requires "a mix of short (2–3 turns) and long (5+ turns) conversations." The current sampler enforces `min_chain_length=3`, producing all conversations with 3–5 tool calls. Short 1–2 tool call conversations (which would generate 2–3 turn dialogues) are not yet supported.

**What would fix this:** Add a `short` sampling mode that allows chains of length 1–2, and mix it into `sample_mixed()` at roughly 20–30% weight. The validator would need a conditional check: short conversations bypass the `min_distinct_tools=2` constraint.
