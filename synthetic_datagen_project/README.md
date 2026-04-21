# Offline Synthetic Multi-Agent Tool-Use Conversation Generator

An offline system that generates synthetic multi-turn conversations with multi-step, multi-tool tool-use traces, grounded in tool schemas from ToolBench. Suitable for training and evaluating tool-use agents.

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Python | 3.11+ |
| Anthropic API key | Required for `evaluate` only — not needed for `build`, `generate`, `validate`, `metrics` |

**Memory backend** (automatic, no config needed):
- `sentence-transformers` + `qdrant-client` — local vector search, no API key required
- In-process keyword store — zero dependencies, fallback if vector backend unavailable

---

## Quickstart

### Step 1 — Install dependencies

```bash
pip3 install pyyaml anthropic sentence-transformers qdrant-client
```

### Step 2 — Set your API key

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

Export it every terminal session:
```bash
export $(cat .env | xargs)
```

### Step 3 — Build graph artifacts

```bash
python3 -m synthetic_datagen.cli.main build
```

### Step 4 — Generate conversations

**Run A** — cross-conversation steering disabled:
```bash
python3 -m synthetic_datagen.cli.main generate \
  --n 20 --seed 42 \
  --no-cross-conversation-steering \
  --output output/run_a.jsonl
```

**Run B** — cross-conversation steering enabled:
```bash
python3 -m synthetic_datagen.cli.main generate \
  --n 20 --seed 42 \
  --output output/run_b.jsonl
```

### Step 5 — Validate output

```bash
python3 -m synthetic_datagen.cli.main validate --input output/run_a.jsonl
python3 -m synthetic_datagen.cli.main validate --input output/run_b.jsonl
```

### Step 6 — Evaluate with LLM-as-judge

> Makes one Anthropic API call per conversation (20 conversations = 20 API calls).

```bash
python3 -m synthetic_datagen.cli.main evaluate \
  --input output/run_a.jsonl \
  --output output/run_a_eval.jsonl \
  --model claude-haiku-4-5-20251001 \
  --threshold 3.5

python3 -m synthetic_datagen.cli.main evaluate \
  --input output/run_b.jsonl \
  --output output/run_b_eval.jsonl \
  --model claude-haiku-4-5-20251001 \
  --threshold 3.5
```

### Step 7 — Compare diversity metrics

```bash
python3 -m synthetic_datagen.cli.main metrics \
  --input output/run_a.jsonl \
  --compare output/run_b.jsonl
```

### Step 8 — Inspect grounding chain (optional)

```bash
python3 -m synthetic_datagen.cli.main inspect --input output/run_b.jsonl --verbose
```

### Step 9 — Run tests

```bash
python3 -m pytest synthetic_datagen/test/test_pipeline.py -v
```


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
  --n 20 \
  --seed 42 \
  --mode mixed \
  --output output/conversations.jsonl \
  [--evaluate] \
  [--judge-model claude-haiku-4-5-20251001] \
  [--no-cross-conversation-steering] \
  [--verbose]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | 50 | Number of conversations to generate |
| `--seed` | 42 | Random seed — same seed always produces identical output |
| `--mode` | sequential | `sequential`, `multi_tool`, `clarification_first`, `parallel`, `mixed` |
| `--output` | output/conversations.jsonl | Output JSONL path |
| `--evaluate` | off | Score inline with LLM-as-judge (requires `ANTHROPIC_API_KEY`) |
| `--judge-model` | claude-haiku-4-5-20251001 | Model used when `--evaluate` is set |
| `--no-cross-conversation-steering` | off | Disable corpus memory steering — use for Run A. Also accepted as `--no-corpus-memory` |
| `--verbose` | off | Log rejected conversations |

> With 16 tools and 43 endpoints, unique chain combinations are exhausted around 200 conversations. Use `--mode mixed` for larger datasets.

---

### `evaluate`

Scores conversations with LLM-as-judge. Optionally repairs failing conversations.

```bash
python3 -m synthetic_datagen.cli.main evaluate \
  --input output/conversations.jsonl \
  --output output/evaluated.jsonl \
  --model claude-haiku-4-5-20251001 \
  --threshold 3.5 \
  [--repair] \
  [--verbose]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input JSONL path |
| `--output` | output/evaluated.jsonl | Output JSONL with scores attached |
| `--model` | claude-haiku-4-5-20251001 | Judge model ID |
| `--threshold` | 3.5 | Pass threshold (1.0–5.0) |
| `--repair` | off | Attempt LLM repair on failing conversations |
| `--max-repairs` | 2 | Max repair attempts per conversation |
| `--verbose` | off | Log per-conversation decisions |

Writes: `evaluated.jsonl` + `evaluation_report.json`

---

### `validate`

Structural validation only — no LLM required.

```bash
python3 -m synthetic_datagen.cli.main validate --input output/conversations.jsonl
```

---

### `metrics`

Computes diversity metrics (Shannon entropy + Jaccard dissimilarity). Use `--compare` to diff two runs side by side.

```bash
python3 -m synthetic_datagen.cli.main metrics \
  --input output/run_a.jsonl \
  --compare output/run_b.jsonl
```

---

### `inspect`

Per-conversation compliance breakdown. `--verbose` shows the exact data lineage across tool steps.

```bash
python3 -m synthetic_datagen.cli.main inspect --input output/conversations.jsonl --verbose
```

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

Each JSONL record contains:

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
    {"role": "assistant", "content": "Booked! Confirmation: bk_3391."}
  ],
  "tool_calls": [
    {"name": "hotels::search", "parameters": {"city": "Paris", "max_price": 200}},
    {"name": "hotels::book",   "parameters": {"hotel_id": "htl_881", "check_in": "2026-04-11"}}
  ],
  "tool_outputs": [
    {"name": "hotels::search", "output": {"results": [{"hotel_id": "htl_881", "price": 175}]}},
    {"name": "hotels::book",   "output": {"booking_id": "bk_3391", "status": "confirmed"}}
  ],
  "judge_scores": {
    "tool_correctness": 4.5,
    "task_completion": 4.0,
    "naturalness": 4.0,
    "mean_score": 4.17,
    "reasoning": "Arguments grounded from prior outputs. Task resolved with confirmation.",
    "judge_model": "claude-haiku-4-5-20251001",
    "passed": true
  },
  "passed": true,
  "metadata": {
    "seed": 42,
    "tool_ids_used": ["hotels"],
    "num_turns": 6,
    "num_clarification_questions": 1,
    "memory_grounding_rate": 1.0,
    "corpus_memory_enabled": true,
    "pattern_type": "search_then_action",
    "sampling_mode": "sequential",
    "domain": "travel planning",
    "endpoint_ids": ["hotels::search", "hotels::book"],
    "num_tool_calls": 2,
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
| `memory_grounding_rate` | float\|null | Fraction of non-first steps grounded from prior outputs. `null` if only one tool call |
| `corpus_memory_enabled` | bool | Whether cross-conversation steering was active |
| `pattern_type` | str | `sequential_multi_step`, `search_then_action`, `multi_tool_chain`, `parallel` |
| `sampling_mode` | str | Sampler mode used |
| `domain` | str | Planner-assigned domain |
| `endpoint_ids` | list[str] | Full endpoint IDs in chain order |
| `num_tool_calls` | int | Total tool calls |
| `num_distinct_tools` | int | Number of distinct tools used |

---

## Running Tests

```bash
# Full suite — no API key required (judge/repair tests use mocks)
python3 -m pytest synthetic_datagen/test/test_pipeline.py -v

# Run specific groups
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "repair" -v
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "e2e" -v
python3 -m pytest synthetic_datagen/test/test_pipeline.py -k "executor" -v
```

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

## Project Structure

```
synthetic_datagen_project/
├── README.md
├── .env                           ← ANTHROPIC_API_KEY (not committed)
├── .gitignore
└── synthetic_datagen/
    ├── DESIGN.md                  ← Architecture, decisions, analysis
    ├── config/                    ← Per-component YAML configs
    ├── data/seed_tools.json       ← 16 tools, 43 endpoints (ToolBench format)
    ├── common/types.py            ← Shared dataclasses (ConversationState)
    ├── toolbench/ingest.py        ← Raw JSON parser
    ├── graph/
    │   ├── registry.py            ← Normalization boundary
    │   ├── heterogeneous_graph.py ← 5-node-type graph
    │   └── projected_graph.py     ← Endpoint-to-endpoint sampler graph
    ├── sampler/                   ← Graph-driven chain sampler (4 modes)
    ├── planner/                   ← Conversation planner + corpus memory
    ├── generator/                 ← UserProxy, Assistant, OfflineExecutor
    ├── evaluator/
    │   ├── judge.py               ← AnthropicJudgeClient (forced tool use)
    │   ├── scorer.py              ← Gated pass/fail validation
    │   ├── repairer.py            ← Surgical + full-rewrite repair
    │   └── report.py              ← Aggregated metrics report
    ├── memory/store.py            ← MemoryStore (vector search with fallbacks)
    ├── cli/main.py                ← CLI entry point
    ├── test/test_pipeline.py      ← 56 tests
    ├── artifacts/                 ← Built graph artifacts
    └── output/                    ← Generated JSONL datasets
```
