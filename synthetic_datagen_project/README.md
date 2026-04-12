# Offline Synthetic Multi-Agent Tool-Use Conversation Generator

An offline system that generates synthetic multi-turn conversations containing multi-step, multi-tool tool-use traces, grounded in tool schemas from ToolBench. Suitable for training and evaluating tool-use agents.

## How to Run This Project

### Step 1 — Install dependencies (once)
```bash
pip3 install pyyaml anthropic sentence-transformers qdrant-client
pip install mem0ai 
```

### Step 2 — Set your API key (every terminal session)
```bash
export $(cat .env | xargs)
```

### Step 3 — Build graph artifacts (once)
```bash
python3 -m synthetic_datagen.cli.main build
```

### Step 4 — Generate varied conversations with judge scoring
```bash
python3 -m synthetic_datagen.cli.main generate --n 50 --seed 44 --mode mixed --evaluate
```
source /Users/suresh/data-generator/.venv/bin/activate
python3 -m synthetic_datagen.cli.main generate --n 20 --seed 42 --no-cross-conversation-steering --output output/run_a.jsonl

### Step 5 — Inspect output
```bash
python3 -m synthetic_datagen.cli.main inspect --input output/conversations.jsonl
```

### Step 6 — Diversity metrics
```bash
python3 -m synthetic_datagen.cli.main metrics --input output/conversations.jsonl
```

### Step 7 — Run tests
```bash
python3 -m pytest synthetic_datagen/test/test_pipeline.py -v
```

---

## Architecture

```
ToolBench JSON
     │
     ▼
toolbench/ingest.py          ← Raw parse only (no interpretation)
     │
     ▼
graph/registry.py            ← Normalization boundary
     │
     ▼
graph/heterogeneous_graph.py ← 5-node-type graph
     │
     ▼
graph/projected_graph.py     ← Endpoint-to-endpoint sampler graph
     │
     ▼
sampler/sampler.py           ← Graph-driven chain proposer (4 modes)
     │
     ▼
planner/agent.py             ← Conversation planner + corpus memory
     │
     ▼
generator/                   ← UserProxy, Assistant, OfflineExecutor, Validator
     │
     ▼
evaluator/                   ← LLM-as-judge + repair loop
     │
     ▼
output/conversations.jsonl   ← JSONL dataset with judge_scores
```

---

## Installation

Python 3.11+ required.

```bash
pip install pyyaml anthropic sentence-transformers qdrant-client
pip uninstall mem0ai -y     # mem0ai requires OpenAI key and makes network calls — use local vector store instead
```

**Memory backend priority** (automatic, no config needed):
1. `mem0ai` — if installed and `OPENAI_API_KEY` is set (slow, requires OpenAI)
2. `sentence-transformers` + `qdrant-client` — local vector search, no API key needed (recommended)
3. In-process keyword store — zero dependencies, always available as last fallback

---

## CLI Reference

### `build`
Ingests ToolBench data and builds all derived artifacts.

```bash
python3 -m synthetic_datagen.cli.main build [--data PATH] [--artifacts DIR]
```

Writes: `artifacts/registry.json`, `heterogeneous_graph.json`, `projected_graph.json`, `build_manifest.json`

---

### `generate`
Generates synthetic conversations. Optionally scores inline with LLM-as-judge.

```bash
python3 -m synthetic_datagen.cli.main generate \
  --n 50 \
  --seed 42 \
  --mode mixed \
  --output output/conversations.jsonl \
  [--evaluate] \
  [--judge-model claude-haiku-4-5-20251001] \
  [--no-corpus-memory] \
  [--verbose]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | 50 | Number of conversations. With 16 tools and 43 endpoints, unique tool combinations are exhausted around 200 conversations — beyond that the sampler reuses chain structures |
| `--seed` | 42 | Any integer. Same seed always produces identical output. Different seeds produce different tool chains and mock values |
| `--mode` | sequential | `sequential`, `multi_tool`, `clarification_first`, `parallel`, `mixed` |
| `--output` | output/conversations.jsonl | Output JSONL path |
| `--evaluate` | off | Score each conversation with LLM-as-judge inline (requires `ANTHROPIC_API_KEY`). Adds one Anthropic API call per conversation |
| `--judge-model` | claude-haiku-4-5-20251001 | Judge model when `--evaluate` is set |
| `--no-corpus-memory` | off | Disable cross-conversation steering (Run A in diversity experiment) |
| `--verbose` | off | Log rejected conversations |

---

### `evaluate`
Scores conversations with LLM-as-judge. Optionally repairs failing conversations.

```bash
python3 -m synthetic_datagen.cli.main evaluate \
  --input output/conversations.jsonl \
  --output output/evaluated.jsonl \
  [--repair] \
  [--threshold 3.5] \
  [--model claude-haiku-4-5-20251001] \
  [--verbose]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input JSONL |
| `--output` | output/evaluated.jsonl | Output JSONL with scores attached |
| `--threshold` | 3.5 | Pass threshold (1.0–5.0) |
| `--repair` | off | Attempt LLM repair on failing conversations |
| `--max-repairs` | 2 | Max repair attempts per conversation |
| `--model` | claude-haiku-4-5-20251001 | Judge model ID |
| `--verbose` | off | Log per-conversation decisions |

Writes: `evaluated.jsonl` + `evaluation_report.json`
Exits with code `2` if mean quality assertion fails (useful for CI).

---

### `validate`
Structural validation only (no LLM required).

```bash
python3 -m synthetic_datagen.cli.main validate --input output/conversations.jsonl
```

---

### `metrics`
Computes diversity metrics (Shannon entropy + Jaccard dissimilarity).

```bash
python3 -m synthetic_datagen.cli.main metrics --input output/conversations.jsonl
```

---

### `inspect`
Per-conversation compliance breakdown. Use `--verbose` to see the full grounding chain — which field from which step fed the next step's arguments.

```bash
python3 -m synthetic_datagen.cli.main inspect --input output/conversations.jsonl
python3 -m synthetic_datagen.cli.main inspect --input output/conversations.jsonl --verbose
```

`--verbose` shows the data lineage across steps:
```
step 0: flight_search::search_flights  [FIRST STEP]
        → input:  {"origin": "JFK", "destination": "CDG"}
        → output fields: [flight_id, price, airline]

step 1: flight_booking::book_flight  [GROUNDED via flight_id←step0]
        ← flight_id: 'FL247'  (from step 0 output)
        → output fields: [booking_id, status]
```

---

## Output Format

Each JSONL record:

```json
{
  "conversation_id": "conv_42_0001",
  "messages": [
    {"role": "user", "content": "Find me a hotel in Paris under 200 euros"},
    {"role": "assistant", "content": "What dates are you looking for?"},
    {"role": "user", "content": "April 11th"},
    {"role": "assistant", "content": "Let me search.", "tool_calls": [
      {"name": "hotels::search", "parameters": {"city": "Paris", "max_price": 200}}
    ]},
    {"role": "tool", "name": "hotels::search", "content": "{\"results\": [...]}"},
    {"role": "assistant", "content": "I found Hotel du Marais at 175 EUR. Booking now.", "tool_calls": [
      {"name": "hotels::book", "parameters": {"hotel_id": "htl_881", "check_in": "2026-04-11"}}
    ]},
    {"role": "tool", "name": "hotels::book", "content": "{\"booking_id\": \"bk_3391\"}"},
    {"role": "assistant", "content": "Booked! Confirmation: bk_3391."}
  ],
  "tool_calls": [
    {"name": "hotels::search", "parameters": {"city": "Paris", "max_price": 200}},
    {"name": "hotels::book",   "parameters": {"hotel_id": "htl_881", "check_in": "2026-04-11"}},
    {"name": "hotels::confirm","parameters": {"booking_id": "bk_3391"}}
  ],
  "tool_outputs": [
    {"name": "hotels::search",  "output": {"results": [{"hotel_id": "htl_881", "price": 175}]}},
    {"name": "hotels::book",    "output": {"booking_id": "bk_3391", "status": "confirmed"}},
    {"name": "hotels::confirm", "output": {"confirmation_number": "CONF-9921"}}
  ],
  "judge_scores": {
    "tool_correctness": 4.5,
    "task_completion": 4.0,
    "naturalness": 4.0,
    "mean_score": 4.17,
    "reasoning": "Arguments grounded from prior outputs. Task resolved with confirmation.",
    "judge_model": "claude-haiku-4-5-20251001",
    "passed": true,
    "failed_gates": []
  },
  "passed": true,
  "metadata": {
    "conversation_id": "conv_42_0001",
    "seed": 42,
    "tool_ids_used": ["hotels"],
    "num_turns": 8,
    "num_clarification_questions": 1,
    "memory_grounding_rate": 1.0,
    "corpus_memory_enabled": true,
    "pattern_type": "search_then_action",
    "sampling_mode": "sequential",
    "domain": "Travel",
    "endpoint_ids": ["hotels::search", "hotels::book", "hotels::confirm"],
    "num_tool_calls": 3,
    "num_distinct_tools": 1
  }
}
```

### Metadata Schema

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Random seed used for generation |
| `tool_ids_used` | list[str] | Deduplicated tool names used |
| `num_turns` | int | Total message count |
| `num_clarification_questions` | int | Number of clarification turns |
| `memory_grounding_rate` | float\|null | Fraction of non-first steps where session memory returned at least one result before argument filling. `null` if only one tool call |
| `corpus_memory_enabled` | bool | Whether cross-conversation steering was active |
| `pattern_type` | str | `sequential_multi_step`, `search_then_action`, `multi_tool_chain`, `parallel` |
| `sampling_mode` | str | Sampler mode used |
| `domain` | str | Planner-assigned domain |
| `endpoint_ids` | list[str] | Full endpoint IDs in chain order |
| `num_tool_calls` | int | Total tool calls |
| `num_distinct_tools` | int | Number of distinct tools used |

---

## Quality & Observability Features

### Inline Grounding Check
During generation, after each tool executes, the system programmatically checks whether the next step's required parameters exist in `session.accumulated_fields`. If a required param cannot be resolved from prior outputs, a warning is logged immediately:
```
[grounding] conv_42_0003: step 2 (currency_exchange::convert) has unresolvable
  required params: [from_currency] — will use mock fallback
```
This surfaces grounding gaps during generation rather than waiting for the judge to score `tool_correctness` low.

### ConversationState — Inter-Agent Communication Protocol
All generator agents communicate through a shared `ConversationState` object defined in `common/types.py`. The Planner writes the plan into it, the Executor updates grounding stats and warnings, and the Validator reads the final messages and tool_calls from it. This makes the inter-agent protocol explicit and documentable.

### Coverage Tracker
After every `generate` run, a coverage report is printed showing domain and pattern distribution in real time:
```
[coverage] Domain distribution (10 domains):
  travel planning                    9  █████████
  food and dining                    7  ███████
  ...
[coverage] Underrepresented domains (1 conversation each): ['news_and_media']
```
This makes diversity steering visible — you can see which domains are underrepresented and how corpus memory influences the distribution.

### Grounding Chain in `inspect --verbose`
The `inspect --verbose` command shows the exact data lineage across tool steps — which field from which step fed the next step's arguments. See the `inspect` command reference above.

---

## Running Tests

```bash
# Run all 56 tests (no API key required — judge/repair tests use mocks)
python3 -m pytest synthetic_datagen/test/test_pipeline.py -v

# Run specific test groups
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "repair" -v
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "e2e" -v
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "executor" -v
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "output_format" -v
```

**Test coverage by component:**

| Component | Tests |
|---|---|
| Ingestion | 5 |
| Registry normalization + deduplication | 6 |
| Intent rule priority | 6 |
| Heterogeneous graph | 5 |
| Projected graph | 5 |
| Sampler determinism + constraints | 6 |
| Clarification detection | 3 |
| MemoryStore | 3 |
| Parallel mode | 3 |
| Offline Executor (schema, grounding, isolation) | 3 |
| Judge scores + output format | 2 |
| Corpus memory A/B + diversity metrics | 2 |
| Repair loop integration | 2 |
| End-to-end (50 conv, 100 conv + judge, domains, entropy, roles) | 5 |
| **Total** | **56** |

---

## Diversity Experiment (Run A vs Run B)

```bash
# Run A: cross-conversation steering disabled
python3 -m synthetic_datagen.cli.main generate --n 50 --seed 42 \
  --no-corpus-memory --output output/run_a.jsonl

# Run B: cross-conversation steering enabled
python3 -m synthetic_datagen.cli.main generate --n 50 --seed 42 \
  --output output/run_b.jsonl

# Compute and compare metrics
python3 -m synthetic_datagen.cli.main metrics --input output/run_a.jsonl
python3 -m synthetic_datagen.cli.main metrics --input output/run_b.jsonl
```

See `DESIGN.md` Section 7 for numeric results and analysis.

---

## Project Structure

```
synthetic_datagen_project/
├── README.md                     ← This file
├── .env                          ← ANTHROPIC_API_KEY (not committed)
├── .gitignore
└── synthetic_datagen/
    ├── DESIGN.md                 ← Architecture, decisions, analysis
    ├── config/                   ← Per-component YAML configs
    ├── data/seed_tools.json      ← 16 tools, 43 endpoints (ToolBench format)
    ├── common/types.py           ← Shared dataclasses
    ├── toolbench/ingest.py       ← Raw JSON parser
    ├── graph/
    │   ├── registry.py           ← Normalization boundary
    │   ├── heterogeneous_graph.py← 5-node-type graph
    │   └── projected_graph.py    ← Endpoint-to-endpoint sampler graph
    ├── sampler/                  ← Graph-driven chain sampler
    ├── planner/                  ← Conversation planner + corpus memory
    ├── generator/                ← UserProxy, Assistant, OfflineExecutor
    ├── evaluator/                ← LLM-as-judge + repair loop
    │   ├── judge.py              ← AnthropicJudgeClient (tool use)
    │   ├── scorer.py             ← Gated pass/fail validation
    │   ├── repairer.py           ← Surgical + full-rewrite repair
    │   └── report.py             ← Aggregated metrics report
    ├── memory/store.py           ← MemoryStore (mem0-backed with fallbacks)
    ├── cli/main.py               ← CLI entry point
    ├── test/test_pipeline.py     ← 56 tests
    ├── artifacts/                ← Built graph artifacts
    └── output/                   ← Generated JSONL datasets
```
