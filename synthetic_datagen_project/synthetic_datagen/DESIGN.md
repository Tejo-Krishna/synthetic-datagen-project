# DESIGN.md — Offline Synthetic Tool-Use Conversation Generator

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║              OFFLINE SYNTHETIC TOOL-USE CONVERSATION GENERATOR                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 · BUILD                              python -m synthetic_datagen build  │
└─────────────────────────────────────────────────────────────────────────────────┘

   data/seed_tools.json
   (16 tools · 43 endpoints)
            │
            ▼
   ┌──────────────────────┐
   │  toolbench/ingest    │  Raw parse only — zero interpretation
   └──────────┬───────────┘  RawTool · RawEndpoint
              │
              ▼
   ┌──────────────────────┐  ◄─── Normalization boundary
   │   graph/registry     │  Parses returns_schema · returns_fields · returns_types
   │                      │  Infers intent from intent_rules.yaml
   └──────────┬───────────┘  Builds 4 lookup indexes
              │
              ▼
   ┌──────────────────────────────────────┐
   │   graph/heterogeneous_graph          │
   │                                      │
   │   Tool ── Endpoint ── Parameter      │  5 node types
   │              │                       │  Full provenance
   │         ResponseField ── Concept     │  Never walked directly
   └──────────┬───────────────────────────┘
              │  collapses multi-hop paths
              ▼
   ┌──────────────────────────────────────┐
   │   graph/projected_graph              │
   │                                      │
   │   Endpoint ──────────────► Endpoint  │
   │         data_link   (1.00)           │  3 edge types
   │         semantic    (0.45)           │  field_mappings on data_link
   │         category    (0.20)           │  entry nodes pre-computed
   └──────────┬───────────────────────────┘
              │
              ▼
   artifacts/  registry.json · heterogeneous_graph.json
               projected_graph.json · build_manifest.json


┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 · GENERATE                        python -m synthetic_datagen generate  │
└─────────────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────────────────────┐
                         │         memory/store.py            │
                         │                                    │
                         │  session_{conv_id}                 │  per-conversation
                         │  └── tool outputs & arg values     │  argument grounding
                         │                                    │
                         │  corpus                            │  cross-conversation
                         │  └── goals · domains · patterns    │  topic steering
                         │                                    │
                         │  sentence-transformers + qdrant    │
                         │  (in-process keyword fallback)     │
                         └──────┬────────────────────┬────────┘
                                │ corpus queries      │ session writes/reads
                                │                     │
   ┌──────────────────────┐     │                     │
   │  sampler/            │     │                     │
   │  SamplerAgent        │     ▼                     │
   │                      │  ┌─────────────────────┐  │
   │  strategies.py       │  │  planner/           │  │
   │  · sequential        │  │  PlannerAgent       │  │
   │  · multi_tool        │  │                     │  │
   │  · clarification     │  │  1. validate chain  │  │
   │    _first            │  │  2. query corpus    │  │
   │  · parallel          │  │  3. build scaffold  │  │
   │                      │  │  4. detect ambiguity│  │
   │  sampler_config.yaml │  │  5. derive hints    │  │
   └──────────┬───────────┘  │  6. gen narrative   │  │
              │               │  7. merge + validate│  │
              │  SampledChain │  8. retry on fail   │  │
              └───────────────►                     │  │
                              │  Deterministic      │  │
                              │  NarrativeBackend   │  │
                              └──────────┬──────────┘  │
                                         │              │
                              StructuredConversationPlan│
                                         │              │
                    ┌────────────────────▼──────────────┴──────────────────────┐
                    │                  ConversationState                        │
                    │           inter-agent communication object                │
                    │                                                           │
                    │   .plan            ◄── written by PlannerAgent            │
                    │   .messages        ◄── written by UserProxy + Assistant   │
                    │   .tool_calls      ◄── written by AssistantAgent          │
                    │   .tool_outputs    ◄── written by OfflineExecutor         │
                    │   .grounding_warnings ◄── written by OfflineExecutor      │
                    └──────────────────────┬────────────────────────────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────┐
               │                           │                       │
               ▼                           ▼                       ▼
   ┌───────────────────┐     ┌─────────────────────┐   ┌─────────────────────┐
   │  generator/       │     │  generator/         │   │  generator/         │
   │  UserProxyAgent   │     │  AssistantAgent     │   │  OfflineExecutor    │
   │                   │     │                     │   │                     │
   │  · user messages  │     │  · tool call text   │   │  arg resolution:    │
   │  · clarification  │     │  · preambles        │   │  1. user input      │
   │    questions      │     │  · intermediate     │   │  2. field_mappings  │
   │  · resolved_params│     │    summaries        │   │  3. session memory  │
   │    (canonical     │     │  · final response   │   │  4. mock fallback   │
   │    key-values for │     │    referencing      │   │                     │
   │    executor)      │     │    real outputs     │   │  inline grounding   │
   └───────────────────┘     └─────────────────────┘   │  check after step  │
                                                        └─────────────────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  generator/            │
                              │  ConversationValidator  │
                              │                        │
                              │  · ≥ 3 tool calls      │
                              │  · ≥ 2 distinct tools  │
                              │  · role ordering       │
                              │  · short_chain exempt  │
                              └────────────┬───────────┘
                                           │ pass / reject + resample
                                           ▼
                              ┌────────────────────────┐
                              │  generator/writer.py   │
                              └────────────┬───────────┘
                                           │
                              output/conversations.jsonl


┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 · EVALUATE                        python -m synthetic_datagen evaluate  │
└─────────────────────────────────────────────────────────────────────────────────┘

   output/conversations.jsonl
            │
            ▼
   ┌──────────────────────────────────────┐
   │  evaluator/judge.py                  │
   │  AnthropicJudgeClient                │
   │                                      │
   │  prompt: tool schemas + transcript   │
   │  tool_choice = submit_scores         │  ◄── forced tool use
   │                                      │       always parseable JSON
   │  tool_correctness  (1.0 – 5.0)       │
   │  task_completion   (1.0 – 5.0)       │
   │  naturalness       (1.0 – 5.0)       │
   └───────────────────┬──────────────────┘
                       │
                       ▼
   ┌──────────────────────────────────────┐
   │  evaluator/scorer.py                 │
   │  pass: mean ≥ 3.5 · all dims ≥ 3.5  │
   └────────────┬─────────────────────────┘
                │
       ┌────────┴────────┐
     PASS              FAIL
       │                 │
       │                 ▼
       │    ┌────────────────────────────┐
       │    │  evaluator/repairer.py     │
       │    │                            │
       │    │  attempt 1 — surgical      │
       │    │  · target lowest dim       │
       │    │  · inject judge reasoning  │
       │    │  · pin tool sequence       │
       │    │                            │
       │    │  attempt 2 — full rewrite  │
       │    │  · rebuild all dialogue    │
       │    │  · pin tool sequence       │
       │    └──────────┬─────────────────┘
       │               │
       └───────┬───────┘
               ▼
   ┌──────────────────────────────────────┐
   │  evaluator/report.py                 │
   │  evaluation_report.json              │
   │                                      │
   │  pass rate · mean scores             │
   │  by domain · by pattern              │
   │  repair success rate                 │
   └──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 · COMPARE                          python -m synthetic_datagen metrics  │
└─────────────────────────────────────────────────────────────────────────────────┘

   run_a.jsonl  (--no-cross-conversation-steering)
   run_b.jsonl  (corpus memory enabled)
            │
            ▼
   Shannon entropy over (tool_ids, pattern_type) buckets
   Average pairwise Jaccard dissimilarity
            │
            ▼
   Run A vs Run B · diversity + quality side-by-side


┌─────────────────────────────────────────────────────────────────────────────────┐
│  DATA CONTRACTS                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

   SamplerAgent        →  SampledChain               common/types.py
                          FieldMapping · Transition
                          ParallelBranch · ClarificationStep

   PlannerAgent        →  StructuredConversationPlan  planner/models.py

   OfflineExecutor     →  StepOutput                 generator/executor.py

   All agents         ↔   ConversationState           common/types.py
                          single shared object · one per conversation
```

---

**System design & architecture decisions**
---------------
Two- layered approach
I chose to construct a heterogeneous graph with five node types — Tool, Endpoint, Parameter, ResponseField, and Concept as  I wanted to capture the full structural richness of the API landscape.  In my exploration, I found that allowing the Sampler to traverse this complex graph directly would add unnecessary difficulty without improving the quality of the chains.  

So, I projected the graph into a simpler endpoint-to-endpoint view. In this projection, I collapsed multi-hop paths  for eg Endpoint A → ResponseField(flight_id) -> Concept(identifier) -> Parameter(flight_id) -> Endpoint B into a single weighted edge. I wanted both simplicity for the sampler and traceability for deeper investigation.

---

Normalization Boundary at Registry
I deliberately drew a boundary in the system design. ingest.py handled only raw data, while the Registry became the first and only layer responsible for interpreting that data.  

I considered the alternative  leaving `returns_raw` as an unparsed string on the Endpoint object  but my analysis showed that this would force every downstream component (Graph, Sampler, Executor, Validator) to parse it independently. That would lead to duplicated logic and inconsistent behavior.  

By parsing once at Registry time into `returns_schema`, `returns_fields`, and `returns_types`, I established a single canonical interpretation* This ensured that all later stages could rely on the same consistent view without re-parsing. My reasoning here was guided by a research principle: minimize redundancy, enforce consistency, and create a reliable foundation for experimentation.

---

Per-Component YAML Configuration Files
I tried to split the configuration into three separate YAML files — intent_rules.yaml, graph_config.yaml, and sampler_config.yaml — each owned by exactly one component, with in-code defaults serving as a fallback. As the pipeline had eight stages with fundamentally different configuration needs and different rates of change. Intent keywords evolved as new ToolBench tools were discovered; edge weights were calibrated once and rarely touched again. Chain length bounds were tuned iteratively during development. Coupling these into a single file would have made independent testing difficult and created unnecessary merge conflicts across development branches.

---
Typed Weighted Edges with Three Tiers
I decided to go for three edge types with fixed weights: data_link at 1.0, semantic at 0.45, and category at 0.2. 
I simplified the system into three tiers based on how well the "tools" actually fit together:

Data Links (1.0): The strongest connection. Data from A fits perfectly into B.

Semantic Links (0.45): Moderate strength. They share similar topics or language but aren't a guaranteed fit.

Category Links (0.2): The weakest signal. They are just in the same general group.

To encourage the system to mix and match different tools, I added a 30% "cross-tool" bonus. This makes transitions between different tools more likely while keeping the original ranking of the three connection types.

-------------------------
Sampler and Planner
I splitted clarifications between two parts based on what they could see. The Sampler spots missing data because it can see where the connections in the "tool chain" are broken. The Planner would handle confusing goals because it is the only part that understands the user's actual conversation.


 Argument-Fill Precedence Policy
I  defined a fixed resolution order for how the Executor filled argument values. This ordering ensured that directly chained data  where step N's output fed step N+1's input  always took precedence over broader memory context. Memory served as a fallback for values that were not directly linked through the chain, while mock values acted as a last resort to keep execution moving forward.
--

Memory

Though the component in the pipeline depended on the MemoryStore abstraction, I did not join on mem0 directly.I used in-process fallback store. This isolation  made the memory layer independently testable. While the project specification called for mem0-backing, I designed for graceful degradation in offline environments.  I used sentence-transformers combined with qdrant-client for local vector storage  this provided genuine semantic similarity search without requiring an OpenAI API key and ran fully offline. In contrast, mem0ai would have required an OpenAI key for embeddings and made network calls on every add and search operation, which would have slowed generation significantly.

-----
Per-Conversation Session Scope Isolation

I gave each chat its own private memory folder using a unique ID instead of sharing one big folder. This prevents a "data leak" where the system might accidentally grab an answer from a different conversation by mistake. By keeping them separate, the system only uses information from the current conversation to fill in details.

---

I used same placeholder names and cities every time, the system uses a random generator to fill in missing info. As I felt  every conversation starts with no previous data, and if we used the same "Jane Smith" and "Paris" examples every time, our data would be boring and repetitive. By using random seeds, I get variety across all our conversations while still being able to recreate the exact same results if we need to test them again. For Reproducibility: I used "seed" for the randomness, so that I  can still track exactly how a specific result happened.

---
ConversationState as Inter-Agent Communication Protocol
I used a single, shared "status file" called ConversationState so that every agent can talk to the others. Instead of passing messy, individual notes back and forth, every agent reads from and writes to this one master document. It helped as the Planner writes the goal, the Executor adds the results, and the Validator checks the final work. Also It made the rules of the conversation official and easy for developers to follow.

---
Inline grounding

The system uses a spot check after every tool is used to make sure the next step has the data it needs. Without this, the system would silently fail and use fake placeholder data, and I wouldn't find out until the very end. By checking the data immediately, I could  see exactly where a connection was broken without needing to use an expensive AI to figure it out.

---
Coverage

I kept a running tally to track which topics and patterns were used during each session, then printed a final report at the end. Without this, it was tough to tell if the system was actually being diverse or just repeating the same few ideas. This made it much easier to compare different runs and prove that one was more diverse than the other.It helped spotting gaps and easy comparisons



----
 **Prompt Design**

Structure:

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
         1. tool_correctness ...rubric...
         2. task_completion  ..rubric...
         3. naturalness      ...rubric...

         Call the submit_scores tool with your evaluation.
```

Why structured this way:**

 Tool schemas are included  the judge cannot assess `tool_correctness` without knowing what the valid parameters were. Without schemas, a hallucinated `hotel_id` looks identical to a grounded one.

I tried to Force the tool use `tool_choice={"type": "tool", "name": "submit_scores"}` so that it  guarantees that the response is always parseable JSON. The system never relies on extracting scores from free text.


Why three dimensions (not one score):

A single quality score masks which aspect failed. A conversation with broken tool calls but natural dialogue averages to a misleading middle score. Separate dimensions allow the repairer to target the lowest-scoring dimension surgically.

---

Repair Prompt (Surgical)

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

Why this structure:

 The judge's `reasoning` field was injected to verbatim the repair model sees the exact failure explanation, not a generic instruction.
 The tool sequence and outputs were pinned  the repair model cannot change endpoint names or output values, only the dialogue text. This prevents repair from introducing new structural errors.
The dimension-specific instruction was direct and actionable fix tool call arguments is more useful than "improve quality."

---

 Repair Prompt (Full Rewrite  Fallback)

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

Why full rewrite as fallback  not as first attempt:**

Surgical repair is cheaper (fewer tokens rewritten) and preserves correct parts of the conversation. Full rewrite discards everything and risks introducing new problems in turns that were previously correct. The two-stage approach (surgical → full rewrite) captures the best of both: fast targeted fixes in the common case, full rewrite only when the conversation is structurally broken end-to-end.

---

 Iteration That Did Not Work

First attempt: asking the judge to return scores in free text

The original judge prompt asked the model to return scores as:
```
tool_correctness: 4.2
task_completion: 3.8
naturalness: 4.0
reasoning: The assistant correctly...
```

This failed in two ways:
1. The model occasionally returned scores in a different format (`"4.2/5"`, `"4 out of 5"`, `"~4"`), causing parse failures.
2. Sometimes the model would add preamble text before the scores

What I Free-text extraction from LLMs is fragile even for structured numeric output. The fix was switching to forced tool use (`tool_choice={"type": "tool", "name": "submit_scores"}`), which guarantees a JSON object every time. The Anthropic API rejects responses that don't conform to the tool schema, so parse failures dropped to zero. This also satisfies the spec requirement that at least one agent uses structured output.
----------

**Corpus Memory & Diversity Analysis**

I learned that the best way to measure variety is to track two things which tools were used and what conversation style was followed. By grouping every conversation into buckets based on these two factors, I could calculate entropy. It  is way of saying how evenly spread out the conversations are across those buckets. High entropy means  getting a great mix of different scenarios rather than the same one over and over.

I also used a metric called Jaccard dissimilarity to see how much the tool sets overlapped between any two conversations. If the score is 1.0, it means every single conversation used a completely unique set of tools. This approach is much better than just checking for different words because it proves the actual structure of the work is changing, not just the vocabulary.

In short:
The Bucket Method: Measures how well we spread conversations across different tool/style combinations.

Entropy: A single score that tells me if the overall collection is diverse (High = Good).

Jaccard Score: A specific check to ensure tools aren't being repeated too often across different sessions.