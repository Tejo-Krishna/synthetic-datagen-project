# Offline Synthetic Multi-Agent Tool-Use Conversation Generator

An offline system that generates synthetic multi-turn conversations containing multi-step, multi-tool tool-use traces, grounded in tool schemas from ToolBench. Suitable for training and evaluating tool-use agents.

## Architecture

```
ToolBench JSON
     │
     ▼
toolbench/ingest.py       ← Raw parse only
     │
     ▼
graph/registry.py         ← Normalization boundary
     │
     ▼
graph/heterogeneous_graph.py   ← 5-node-type graph (PDF requirement)
     │
     ▼
graph/projected_graph.py       ← Endpoint-to-endpoint sampler graph
     │
     ▼
sampler/sampler.py        ← Graph-driven chain proposer (4 modes)
     │
     ▼
planner/planner.py        ← Conversation planner + corpus memory
     │
     ▼
generator/                ← User-proxy, assistant, executor, validator
     │
     ▼
memory/store.py           ← Session + corpus memory (mem0-backed)
     │
     ▼
output/conversations.jsonl  ← JSONL dataset
```

## Installation

```bash
pip install pyyaml
# Optional (for vector-backed memory):
pip install mem0ai
```

Python 3.11+ required. All other dependencies are stdlib.

## Quick Start

```bash
# 1. Build registry and graph artifacts
python -m synthetic_datagen.cli.main build

# 2. Generate 50 conversations (Run B: corpus memory enabled)
python -m synthetic_datagen.cli.main generate --n 50 --seed 42 --output output/run_b.jsonl

# 3. Generate without corpus memory (Run A: for diversity comparison)
python -m synthetic_datagen.cli.main generate --n 50 --seed 42 --no-corpus-memory --output output/run_a.jsonl

# 4. Validate generated conversations
python -m synthetic_datagen.cli.main validate --input output/run_b.jsonl

# 5. Compute diversity metrics
python -m synthetic_datagen.cli.main metrics --input output/run_b.jsonl
```

## CLI Reference

### `build`
Builds and persists registry, heterogeneous graph, projected graph, and manifest.

```bash
python -m synthetic_datagen.cli.main build [--data PATH] [--artifacts DIR]
```

- `--data`: path to ToolBench JSON (default: `data/seed_tools.json`)
- `--artifacts`: output directory (default: `artifacts/`)

Writes:
- `artifacts/registry.json`
- `artifacts/heterogeneous_graph.json`
- `artifacts/projected_graph.json`
- `artifacts/build_manifest.json`

### `generate`
Generates synthetic conversations using the built pipeline.

```bash
python -m synthetic_datagen.cli.main generate \
  --n 50 \
  --seed 42 \
  --mode sequential \
  --output output/conversations.jsonl \
  [--no-corpus-memory] \
  [--verbose]
```

- `--n`: number of conversations (default: 50)
- `--seed`: random seed for reproducibility (default: 42)
- `--mode`: sampling mode — `sequential`, `multi_tool`, `clarification_first`, `parallel`, `mixed` (default: `sequential`)
- `--output`: output JSONL file
- `--no-corpus-memory`: disable corpus memory (for Run A in diversity experiment)
- `--verbose`: show rejected conversations

### `validate`
Validates a generated JSONL dataset.

```bash
python -m synthetic_datagen.cli.main validate --input output/conversations.jsonl
```

Checks:
- Multi-step traces (≥ 3 tool calls)
- Multi-tool traces (≥ 2 distinct tools)
- Metadata completeness
- Tool call / output alignment

### `metrics`
Computes diversity and quality metrics.

```bash
python -m synthetic_datagen.cli.main metrics --input output/conversations.jsonl
```

Reports:
- Multi-step / multi-tool counts
- Pattern distribution
- Domain distribution
- **Primary**: Entropy over `(tool_ids_used, pattern_type)` buckets
- **Secondary**: Pairwise tool-chain Jaccard dissimilarity

## Running Tests

```bash
# Run all tests (no pytest required)
python synthetic_datagen/tests/test_pipeline.py

# With pytest (if installed)
pytest synthetic_datagen/tests/test_pipeline.py -v
```

## Output Format

Each JSONL record contains:

```json
{
  "conversation_id": "conv_42_0001",
  "messages": [
    {"role": "user", "content": "Help me plan a trip to Paris"},
    {"role": "assistant", "content": "Let me search for flights.", "tool_calls": [...]},
    {"role": "tool", "name": "flight_search::search_flights", "content": "{...}"},
    ...
  ],
  "tool_calls": [
    {"name": "flight_search::search_flights", "parameters": {"origin": "JFK", ...}}
  ],
  "tool_outputs": [
    {"name": "flight_search::search_flights", "output": {"flights": [...]}}
  ],
  "metadata": {
    "seed": 42,
    "tool_ids_used": ["flight_search", "hotel_booking", "weather_api"],
    "num_turns": 9,
    "num_clarification_questions": 2,
    "memory_grounding_rate": 0.0,
    "corpus_memory_enabled": true,
    "pattern_type": "search_then_action",
    "sampling_mode": "sequential",
    "domain": "Travel"
  }
}
```

## Configuration

| File | Owned by | Controls |
|------|----------|---------|
| `config/intent_rules.yaml` | Registry | Keyword-to-intent inference rules |
| `config/graph_config.yaml` | Graph builder | Edge weights, semantic concept groups |
| `config/sampler_config.yaml` | Sampler | Chain length, modes, diversity params |

Each component falls back to in-code defaults if its config file is missing.

## Project Structure

```
synthetic_datagen/
├── config/                   # Per-component YAML configs
├── common/types.py           # Shared runtime dataclasses
├── data/seed_tools.json      # 16 tools, 43 endpoints
├── toolbench/ingest.py       # Raw JSON parser
├── graph/
│   ├── registry.py           # Normalization boundary
│   ├── heterogeneous_graph.py
│   └── projected_graph.py
├── sampler/
│   ├── config.py
│   ├── strategies.py         # 4 sampling modes
│   └── sampler.py
├── planner/planner.py
├── generator/
│   ├── user_proxy.py
│   ├── assistant.py
│   ├── executor.py           # Offline tool execution
│   ├── validator.py
│   └── writer.py
├── memory/store.py           # MemoryStore (mem0-backed)
├── cli/main.py
├── artifacts/                # Built artifacts
├── output/                   # Generated datasets
└── tests/test_pipeline.py
```
