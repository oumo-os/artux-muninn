# 🧠 memory_module

**A hybrid, adaptive memory system for local AI agents.**  
Structured short-term and long-term memory, entity ledgers, semantic recall, and forgetting — backed entirely by SQLite.  
Ships with LLM tool call definitions so your agent can manage its own memory.

---

## Overview

`memory_module` implements a cognitively-inspired memory architecture for personal AI agents. Rather than maintaining a flat conversation log, it models memory the way minds do — with ephemeral short-term perception that consolidates into persistent long-term knowledge, entities that evolve over time, and memories that fade unless reinforced.

The agent's LLM (Claude, GPT-4o, or any tool-calling model) drives the memory system autonomously via tool calls. You wire it up once; the model decides when to record, recall, and consolidate.

This is not a database. It is a structured cognitive model.

```
Perception → Signatures → STM → [Compression] → LTM → [Recall / Decay / Archive]
                                                    ↕
                                               Entities
                                          (living ledgers)
```

---

## Features

- **Two-tier memory** — Short-Term Memory (STM) for immediate context; Long-Term Memory (LTM) for persistent, queryable knowledge
- **Entity ledgers** — People, objects, and concepts are historical narrative records. Contradictions are preserved and sourced.
- **Authority-weighted conflict resolution** — Corrections from designated anchors outweigh peer claims, which outweigh self-reports
- **Hybrid recall** — Structured filters (entity, topic, concept, time) blended with semantic cosine similarity via embeddings
- **LLM tool call interface** — Ready-to-use schemas for both Anthropic and OpenAI tool calling APIs, plus a `ToolExecutor` that dispatches LLM tool calls to memory operations
- **Forgetting** — Confidence decays exponentially over time (`confidence × e^(−λt)`). Weak memories are archived as scars; stale scars are purged.
- **STM compression** — Raw temporal segments are periodically summarised (plug in your LLM call here)
- **Graceful degradation** — Runs without `sentence-transformers`; falls back to TF-IDF bag-of-words embedding
- **Zero required external dependencies** — Pure Python + SQLite stdlib

---

## Installation

```bash
git clone https://github.com/yourname/memory_module
cd memory_module
pip install -r requirements.txt
```

For semantic recall quality (strongly recommended):

```bash
pip install sentence-transformers
```

---

## Using Memory as LLM Tools

This is the primary integration pattern. You give your LLM the memory tool definitions, and the model calls them autonomously — deciding when to record observations, recall past context, create entities, and consolidate important facts.

### Anthropic (Claude)

```python
import anthropic
from memory_module import MemoryAgent, get_tools, ToolExecutor

client   = anthropic.Anthropic()
agent    = MemoryAgent("memory.db")
executor = ToolExecutor(agent)
tools    = get_tools(format="anthropic")   # ready to pass to client.messages.create()

SYSTEM = """You are a personal AI assistant with persistent memory.
Use your memory tools proactively:
  - record_stm after every user message and important response
  - recall before answering anything that may involve past context
  - consolidate_ltm when you learn something durable (name, preference, fact)
  - create_entity for significant people or things; resolve_entity first
"""

messages = []

def chat(user_input: str) -> str:
    agent.record_stm(f"User: {user_input}")
    messages.append({"role": "user", "content": user_input})

    # Agentic loop — keep going until no more tool calls
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        # Execute all tool calls, feed results back
        tool_results = executor.run_anthropic(response.content)
        messages.append({"role": "user", "content": tool_results})

    return next(b.text for b in response.content if hasattr(b, "text"))

print(chat("Hi, my name is Musa and I work on robotics."))
print(chat("What do you remember about me?"))
```

### OpenAI (GPT-4o, etc.)

```python
from openai import OpenAI
from memory_module import MemoryAgent, get_tools, ToolExecutor

client   = OpenAI()
agent    = MemoryAgent("memory.db")
executor = ToolExecutor(agent)
tools    = get_tools(format="openai")   # ready to pass to client.chat.completions.create()

messages = []

def chat(user_input: str) -> str:
    agent.record_stm(f"User: {user_input}")
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=tools,
            messages=[{"role": "system", "content": "You are a helpful assistant with persistent memory. Use your memory tools proactively."}] + messages,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            break

        tool_results = executor.run_openai(msg.tool_calls)
        messages.extend(tool_results)

    return response.choices[0].message.content

print(chat("Hi, my name is Musa and I work on robotics."))
print(chat("What do you remember about me?"))
```

### Available Memory Tools

These are the tools the LLM can call. Their schemas are defined in `tools.py` and served via `get_tools()`.

| Tool | When the LLM should use it |
|---|---|
| `recall(query, top_k?)` | Before answering anything that may depend on past context |
| `record_stm(content)` | After every user message and any notable event |
| `consolidate_ltm(narrative, class_type?, topics?, confidence?)` | When something durable is learned — name, preference, decision |
| `create_entity(name, description, topics?)` | First time a significant person or thing is encountered |
| `observe_entity(entity_id, observation, authority?, memory_ref?)` | When something new is learned about a known entity |
| `resolve_entity(clues, top_k?)` | Before creating a new entity — check if it already exists |
| `get_stm_window()` | To inspect the current immediate context |

### Injecting Memory Context into the System Prompt

For stronger grounding, inject the STM window and pre-recalled context before the model even makes a tool call:

```python
def build_system_prompt(user_message: str) -> str:
    relevant = agent.recall(user_message, top_k=3)
    stm      = agent.get_stm_window()

    return f"""You are a personal AI assistant with persistent memory.

Current short-term context:
{stm}

Relevant long-term memory:
{chr(10).join(r.entry.content for r in relevant) or "Nothing recalled yet."}

Use your memory tools to record new information and update your knowledge.
"""
```


---

## Source File References & Re-Resolution

Memory summaries lose detail. When the agent first perceives an image, it stores a text description — but that description can only contain what was explicitly noted at capture time. When a user later asks a specific detail question, the answer may not be in the summary.

The file reference system solves this by keeping a pointer back to the original source file (image, audio, PDF, video frame, etc.) and returning it alongside recalled memories. The calling agent — a VLM, ASR model, PDF reader, or whatever is appropriate — then re-interrogates the original file with a focused question.

The memory module's role is minimal and precise: store the path, surface it on recall, hand it back. It never reads files itself.

### Registering a Source File

```python
from memory_module import MemoryAgent

agent = MemoryAgent("memory.db")

# 1. Capture: get initial VLM description at perception time
initial_description = vlm.describe("living_room_001.jpg", "Describe this scene.")

# 2. Store the text description as an LTM entry
entry = agent.consolidate_ltm(
    narrative=f"Room scene: {initial_description}",
    topics=["room", "table", "objects"],
)

# 3. Register the source file, linked to the LTM entry
ref = agent.register_file(
    path="/captures/living_room_001.jpg",
    media_type="image",
    summary=initial_description,
    re_resolve_hint=(
        "List EVERY object on the table. For each: colour, "
        "position, size relative to other objects, any text visible."
    ),
    linked_entry_ids=[entry.id],
)
```

### Re-Resolution on Recall

When a memory is recalled, any associated file refs are returned alongside it:

```python
results = agent.recall("what was on the table")

for r in results:
    print(r.entry.content)          # text summary from LTM
    for fref in r.file_refs:        # source files attached to this memory
        print(fref.path)            # → /captures/living_room_001.jpg
        print(fref.storage_tier)    # hot | cold | purged
```

When the detail question can't be answered from the summary, get the re-resolution payload and let the appropriate model handle it:

```python
# User: "what else was on the table besides the doll?"
# The summary only mentioned a doll — time to go back to the source.

payload = agent.re_resolve_payload(fref.id)
# → {
#     "path":              "/captures/living_room_001.jpg",
#     "media_type":        "image",
#     "handler":           "VLM (vision language model)",
#     "current_summary":   "A wooden table with a red doll...",
#     "re_resolve_hint":   "List EVERY object on the table...",
#     "accessible":        True,
#     "storage_tier":      "hot",
# }

if payload["accessible"]:
    enriched = vlm.describe(payload["path"], payload["re_resolve_hint"])

    # Write the richer description back so future recalls benefit
    agent.update_file_summary(fref.id, enriched)

    # Optionally store a new LTM entry with the enriched detail
    agent.store_ltm(
        content=f"Re-resolved table: {enriched}",
        topics=["table", "objects", "room"],
        confidence=0.97,
    )
```

### Supported Media Types

| `media_type` | Handler hint | Typical use |
|---|---|---|
| `image` | VLM | Photos, screenshots, scans |
| `audio` | ASR or audio-language model | Voice memos, recordings |
| `pdf` | PDF reader / document QA | Documents, reports, forms |
| `video` | Video-language model or frame extractor | Video clips, surveillance |
| `document` | Document reader | .docx, .txt, .html |
| `data` | Data analysis tool | .csv, .json, spreadsheets |
| `other` | Application-specific handler | Any custom file type |

### Storage Tier Management

Files move through three tiers as maintenance policy dictates:

```python
# Move to cold storage (cheaper, slower — path may become a remote URI)
agent.move_file_to_cold(ref.id, "/archive/2024/living_room_001.jpg")
# or: "s3://my-bucket/archive/living_room_001.jpg"

# Permanently delete (tombstone kept for provenance)
agent.purge_file(ref.id)

# List all files by tier
hot_images  = agent.list_files(tier="hot",  media_type="image")
cold_files  = agent.list_files(tier="cold")
```

The `re_resolve_payload` will always tell the calling agent whether the file is currently accessible before it tries to open it. Purged files return `None`.

### LLM Tool Interface

Two additional tools are available for LLM-driven agents:

| Tool | When the LLM should call it |
|---|---|
| `re_resolve_file(file_ref_id)` | When a recalled memory has an attached file and the user needs detail the text summary can't provide |
| `update_file_summary(file_ref_id, new_summary)` | After re-interrogating a file with a VLM/ASR — write the richer result back |

```python
tools = get_tools(format="anthropic")   # includes re_resolve_file + update_file_summary
```

The LLM receives `file_refs` in recall results and can call `re_resolve_file` autonomously when it recognises that a detail question requires going back to the source.

---

## Manual / Direct API

You can also drive the memory system directly from your own code, without going through the LLM tool interface:

```python
from memory_module import MemoryAgent

agent = MemoryAgent("memory.db")

# Short-Term Memory
agent.record_stm("User said: hi, my name is Musa")
agent.record_stm("Sam said: he is Musa, not Kyle")
print(agent.get_stm_window())

# Entities — living historical ledgers
musa = agent.create_entity("described self as Kyle", name="Musa")
agent.observe_entity(musa.id, "confirmed name is Musa", authority="self")
agent.correct_entity(musa.id, "he is Musa not Kyle", correcting_entity_id=sam.id)

# Long-Term Memory
agent.consolidate_ltm(
    narrative="User's name is Musa. Initially said Kyle — corrected by Sam.",
    topics=["identity"],
    entities=[musa.id],
    confidence=0.95,
)

# Recall — hybrid structured + semantic
for r in agent.recall("who is Musa"):
    print(f"{r.score:.3f}  {r.entry.content}")

# Forgetting
agent.run_decay()
agent.run_maintenance()

print(agent.status())
# → {'stm_segments': 2, 'ltm_entries': 1, 'ltm_avg_confidence': 0.95,
#    'entities': 1, 'archive_scars': 0}
```

---

## Package Structure

```
memory_module/
├── __init__.py            # Public API: MemoryAgent, get_tools, ToolExecutor
├── agent.py               # MemoryAgent — unified interface for all memory operations
├── tools.py               # LLM tool schemas (Anthropic + OpenAI) + ToolExecutor dispatcher
├── models.py              # Dataclasses: STMSegment, Entity, LTMEntry, Concept, Association, …
├── db.py                  # SQLite schema, connection manager, serialisation helpers
├── stm.py                 # Short-Term Memory: sliding window + compression
├── ltm.py                 # Long-Term Memory: store, concepts, associations, archive
├── entities.py            # Entity ledger: narrative append, conflict resolution, fuzzy resolution
├── recall.py              # Hybrid recall: query parser + structured filter + semantic ranking
├── forgetting.py          # Confidence decay, maintenance, reinforcement
├── embeddings.py          # sentence-transformers with TF-IDF fallback
├── example.py             # Full lifecycle walkthrough (no LLM required)
├── example_anthropic.py   # Complete Anthropic agent loop with memory tools
├── example_openai.py      # Complete OpenAI agent loop with memory tools
└── requirements.txt
```

---

## Architecture

### Short-Term Memory (STM)

STM stores a sliding window of recent temporal segments:

```
[consN  ----  t1  ----  t2  ----  t3]
```

`tX` are raw input segments; `consN` is a compressed narrative summary. When the window exceeds `max_segments` (default: 10), the oldest segments are compressed via `compress_fn`. Swap in your LLM call for production-quality summaries:

```python
def llm_compress(texts: list[str]) -> str:
    return call_llm("Summarise:\n" + "\n".join(texts))

agent = MemoryAgent("memory.db", compress_fn=llm_compress)
```

### Long-Term Memory (LTM)

LTM entries are rich, persistent records with semantic embeddings:

| Field | Description |
|---|---|
| `class_type` | `event`, `assertion`, `decision`, `procedure`, `observation` |
| `content` | Narrative summary |
| `entities` | Referenced entity IDs |
| `topics` | Keywords and subjects |
| `concepts` | Cognitive triples: `operator:subject:focus` |
| `confidence` | Relevance/truth weight (0.0–1.0), decays over time |
| `embedding` | Semantic vector for recall |

### Entities — Living Historical Ledgers

Each entity maintains an append-only narrative that preserves every claim, correction, and observation with its source:

```
described self as Kyle [auth:1] [m.ref:t1]
[ent.ref:sam-id] [auth:4] [dispute:sam-id] he is Musa not Kyle [m.ref:t2]
[ent.ref:john-id] [auth:4] [dispute:john-id] addressed him as Musa [m.ref:t3]
```

Authority tiers determine how to weight competing claims:

| Authority | Weight | Example |
|---|---|---|
| `self` | 1 | Subject's own claim |
| `peer` | 2 | Another person's observation |
| `system` | 3 | Automated sensor or log |
| `anchor` | 4 | Designated authoritative source |

### Concepts

Cognitive triples encode the *structure* of an observation for precise retrieval:

```
what:Musa:identity      →  What is Musa's identity?
dispute:Kyle:identity   →  There is a dispute about Kyle's identity
where:John:location     →  Where is John?
```

A recall query beginning with `what` preferentially matches entries carrying `what:…` concept triples.

### Recall

Five-stage hybrid pipeline:

1. **Query parsing** — extract operator (`who/what/where/…`), subjects, quoted topics, time constraints
2. **Structured filter** — entity matches, topic matches, concept operator matches, time bracket
3. **Semantic ranking** — embed query, compute cosine similarity against LTM embeddings
4. **Score blending** — `score = (w_sem × semantic + (1-w_sem) × struct_boost) × confidence`
5. **Reinforcement** — recalled entries get a small confidence boost

### Forgetting

```
confidence(t) = confidence₀ × e^(−λ × elapsed_days)
```

Default `λ = 0.01` → ~50% decay after 70 days. Entries below the archive threshold become scars; scars older than the TTL are permanently deleted. Scars can be rehydrated if a topic resurfaces.

---

## Full API Reference

### MemoryAgent

**STM**

| Method | Description |
|---|---|
| `record_stm(segment)` | Append a temporal segment |
| `forget_stm(segment_id)` | Remove a segment by ID |
| `get_stm_window()` | Return formatted STM string |

**LTM**

| Method | Description |
|---|---|
| `consolidate_ltm(narrative?, ...)` | Compress STM → persist to LTM |
| `store_ltm(content, ...)` | Directly store an LTM entry |
| `recall(query, top_k)` | Hybrid recall → `list[RecallResult]` |

**Entities**

| Method | Description |
|---|---|
| `create_entity(description, name, topics)` | Create a new entity ledger |
| `resolve_entity(clues, top_k)` | Fuzzy semantic search over entities |
| `observe_entity(id, text, authority, ...)` | Append a sourced observation |
| `correct_entity(id, text, correcting_id)` | Record an authoritative correction |

**Associations & Concepts**

| Method | Description |
|---|---|
| `link_entities(id1, id2, relation, confidence)` | Create a directed association |
| `add_concept(operator, subject, focus, ...)` | Register a cognitive triple |
| `infer_relationships()` | Return all associations |

**Lifecycle**

| Method | Description |
|---|---|
| `run_decay()` | Apply time-based confidence decay |
| `run_maintenance()` | Archive weak entries; purge old scars |
| `reinforce(entry_id, amount)` | Manually boost confidence |
| `get_archive()` | List archive scars |
| `rehydrate(archive_id)` | Pull a scar back into active LTM |
| `status()` | Memory state snapshot dict |

### Tools

| Export | Description |
|---|---|
| `get_tools(format)` | Returns tool schemas for `"anthropic"` or `"openai"` |
| `ToolExecutor(agent)` | Dispatcher: binds tool call results to MemoryAgent |
| `executor.run_anthropic(content)` | Processes Anthropic `response.content` blocks → tool_result list |
| `executor.run_openai(tool_calls)` | Processes OpenAI `message.tool_calls` → tool message list |
| `executor.execute(name, input_dict)` | Execute a single tool by name (provider-agnostic) |

---

## Running the Examples

```bash
# No LLM required — demonstrates the full lifecycle directly
python memory_module/example.py

# With Claude (requires ANTHROPIC_API_KEY)
python memory_module/example_anthropic.py

# With GPT-4o (requires OPENAI_API_KEY)
python memory_module/example_openai.py
```

---

## Tuning

**Forgetting rate:**

```python
from memory_module.forgetting import ForgettingEngine

agent.forgetting = ForgettingEngine(
    agent.db, agent.ltm,
    decay_lambda=0.005,       # slower: ~140 days to 50%
    archive_threshold=0.15,
    archive_ttl_days=730,
)
```

**Recall balance:**

```python
from memory_module.recall import RecallQuery
from datetime import datetime

q = RecallQuery(
    raw="where was John last month",
    operator="where",
    subjects=["John"],
    after=datetime(2024, 3, 1),
    semantic_weight=0.7,
    top_k=5,
)
results = agent.recall_engine.recall(q)
```

**Scheduled maintenance:**

```python
import schedule

schedule.every().day.at("03:00").do(agent.run_decay)
schedule.every().week.do(agent.run_maintenance)
```

---

## Design Philosophy

**Memory is metabolic.** It breathes — growing through consolidation, shrinking through forgetting, updating through reinforcement.

**Entities are chronicles.** A person in this system is a ledger of everything ever claimed about them, by whom, and when. Truth is derived from history, not imposed.

**Forgetting is intentional.** Rarely recalled memories fade; frequently used ones stay sharp. The archive ensures nothing is permanently lost until it has been dormant long enough to be considered irrelevant.

**Truth is confidence-weighted.** Contradictions are a feature. Multiple competing claims coexist, each weighted by authority and recency.

**Social intelligence emerges from provenance.** Who said what, when, with what authority — this is what allows an agent to reason about trust and evolving understanding over time.

---

## Contributing

Impactful areas for improvement:

- **Automated concept extraction** — LLM-based extraction of `operator:subject:focus` triples from narratives
- **Entity auto-resolution** — smarter deduplication and merging of entity records
- **Async support** — `aiosqlite` backend for async agent frameworks
- **Vector index backend** — `chromadb` or `faiss` for large-scale semantic search
- **Multi-agent memory sharing** — shared entity pool with per-agent episodic isolation

---

## License

MIT
