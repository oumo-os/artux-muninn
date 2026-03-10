# 🧠 Muninn — Memory Module

**A standalone, hybrid STM/LTM memory system for AI agents.**  
Structured short-term and long-term memory, entity ledgers, semantic recall, source references, and forgetting — backed entirely by SQLite.  
Ships with LLM tool call definitions so your agent can manage its own memory autonomously.

> Muninn is part of the [Artux cognitive stack](https://github.com/oumo-os/artux-huginn), but is fully self-contained and has no dependency on Huginn. Drop it into any agent project.

---

## Overview

Rather than maintaining a flat conversation log, Muninn models memory the way minds do — with ephemeral short-term perception that consolidates into persistent long-term knowledge, entities that evolve over time, and memories that fade unless reinforced.

```
Perception → STM (raw events + consN) → [consolidation] → LTM
                                                            ↕
                                                        Entities
                                                   (living ledgers)
```

Raw STM events are never deleted by compression. They stay as ground truth until a consolidation agent explicitly flushes them after verifying their LTM entries are written. This separation is what makes Muninn safe for multi-agent use — Sagax can update its rolling narrative summary without racing Logos' consolidation pass.

---

## Features

- **Two-tier memory** — STM for immediate context; LTM for persistent, queryable knowledge
- **Append-only STM** — raw events survive compression; flushed only after verified LTM writes
- **Flush watermark** — consolidation agents resume exactly where they left off across restarts
- **Typed events** — `source`, `event_type`, `payload`, `confidence` on every STM event; works with plain text too
- **Entity ledgers** — people, objects, and concepts are historical narrative records; contradictions preserved with authority weights
- **Hybrid recall** — structured filters (entity, topic, concept, time) blended with semantic cosine similarity
- **LLM tool call interface** — ready-to-use schemas for Anthropic and OpenAI; `ToolExecutor` dispatches tool calls to memory operations
- **Source references** — link LTM entries back to original files (images, audio, PDFs, URLs); surface on recall for re-interrogation
- **Forgetting** — confidence decays exponentially (`confidence × e^{−λt}`); weak entries archived as scars; stale scars purged
- **Graceful degradation** — runs without `sentence-transformers`; falls back to TF-IDF bag-of-words embeddings
- **Zero required external dependencies** — pure Python + SQLite stdlib

---

## Installation

```bash
git clone https://github.com/oumo-os/artux-muninn
cd artux-muninn
pip install -r requirements.txt
```

For semantic recall quality (strongly recommended):

```bash
pip install sentence-transformers
```

---

## Quick Start

```python
from memory_module import MemoryAgent

agent = MemoryAgent("agent.db")

# Record a perception event
agent.record_stm("User said: my name is Musa, I work on robotics",
                 source="user", event_type="speech")

# Create an entity ledger for Musa
musa = agent.create_entity("described self as Musa", name="Musa",
                           topics=["identity"])
agent.observe_entity(musa.id, "works on robotics", authority="self")

# Consolidate to LTM — writes per-event entries, then a period summary, then flushes
agent.consolidate_ltm(topics=["identity", "robotics"], entities=[musa.id])

# Recall
for r in agent.recall("who works on robotics"):
    print(f"{r.score:.3f}  {r.entry.content}")

# Forgetting
agent.run_decay()        # apply time-based confidence decay
agent.run_maintenance()  # archive weak entries, purge stale scars

print(agent.status())
# → {'stm_segments': 0, 'ltm_entries': 2, 'ltm_avg_confidence': 1.0,
#    'entities': 1, 'archive_scars': 0, 'sources': {'total': 0, 'by_type': {}}}
```

---

## Using Memory as LLM Tools

Give your LLM the tool definitions; it decides when to record, recall, and consolidate.

### Anthropic (Claude)

```python
import anthropic
from memory_module import MemoryAgent, get_tools, ToolExecutor

client   = anthropic.Anthropic()
agent    = MemoryAgent("memory.db")
executor = ToolExecutor(agent)
tools    = get_tools(format="anthropic")

SYSTEM = """You are a personal AI assistant with persistent memory.
Use your memory tools proactively:
  - record_stm after every user message and important observation
  - recall before answering anything that may involve past context
  - consolidate_ltm when you learn something durable (name, preference, fact)
  - create_entity for significant people or things; resolve_entity first
"""

messages = []

def chat(user_input: str) -> str:
    agent.record_stm(user_input, source="user", event_type="speech")
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

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
tools    = get_tools(format="openai")

messages = []

def chat(user_input: str) -> str:
    agent.record_stm(user_input, source="user", event_type="speech")
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=tools,
            messages=[{"role": "system", "content": "You have persistent memory. Use your tools."}] + messages,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            break

        messages.extend(executor.run_openai(msg.tool_calls))

    return response.choices[0].message.content
```

### Available Memory Tools

| Tool | When to use |
|---|---|
| `recall(query, top_k?)` | Before answering anything that may depend on past context |
| `record_stm(content)` | After every user message and notable observation |
| `consolidate_ltm(narrative?, class_type?, topics?, confidence?)` | When something durable is learned |
| `create_entity(name, description, topics?)` | First time a significant person/object is encountered |
| `observe_entity(entity_id, observation, authority?, memory_ref?)` | When something new is learned about a known entity |
| `resolve_entity(clues, top_k?)` | Before creating — check if the entity already exists |
| `get_stm_window()` | To inspect current immediate context |
| `record_source(location, type?, description?, meta?)` | Register an external file/URL that backs a memory |
| `update_source_description(source_id, new_description)` | After re-interrogating a source for richer detail |

### Injecting Memory Context into the System Prompt

Pre-populate the LLM's context before it even makes a tool call:

```python
def build_system_prompt(user_message: str) -> str:
    relevant = agent.recall(user_message, top_k=4)
    stm      = agent.get_stm_window()

    return f"""You are a personal AI assistant with persistent memory.

Current short-term context:
{stm or "(empty)"}

Relevant long-term memory:
{chr(10).join(r.entry.content for r in relevant) or "(nothing recalled yet)"}

Use your memory tools to record new information and update your knowledge.
"""
```

---

## STM Event Model

Every STM record is an `STMSegment`. Plain text works fine; the typed fields are optional and add richness for agents that care about event provenance.

```python
agent.record_stm(
    "Kettle reached 95°C",
    source     = "sensor",          # who produced this: user | system | tool | sensor | agent
    event_type = "sensor",          # what kind: speech | tool_call | tool_result | sensor | internal | output | …
    payload    = {"temp_c": 95,     # structured JSON for typed events
                  "device": "kettle-01"},
    confidence = 0.98,              # producer-assigned confidence (0.0–1.0)
)
```

The `payload` field on a consN segment automatically carries:
```json
{"last_event_id": "...", "event_count_folded": 12}
```

---

## STM Flush Model

Compression and flushing are intentionally separate operations.

**`compress()` / `compress_head(retain)`** — updates the rolling consN narrative. Raw events stay untouched. Call this as often as you like; it cannot lose data.

**`flush_up_to(event_id)`** — deletes raw events up to and including the given ID and advances the `flush_watermark`. Call this **only after LTM entries are verified written**.

`consolidate_ltm()` handles both in sequence for the common single-agent case:

```
compress_head(retain_tail) → write per-segment LTM entries → write period entry → flush_up_to(head[-1].id)
```

For a multi-agent setup where a background consolidation agent manages its own flush:

```python
# Consolidation agent startup: resume from watermark
last = agent.get_flush_watermark()   # None on first run

# Poll for unprocessed events
new_events = agent.stm.get_events_after(last)
for event in new_events:
    # ... write your own LTM entries ...
    pass

# Flush only after all writes confirmed
if new_events:
    agent.flush_stm_up_to(new_events[-1].id)
```

### Attention-gate polling

```python
# Lightweight agent that polls STM for new events to triage
last_seen_id = None
while True:
    new_events = agent.stm.get_events_after(last_seen_id)
    if new_events:
        triage(new_events)
        last_seen_id = new_events[-1].id
    sleep(0.005)
```

---

## Source References

Link external files (images, audio, PDFs, URLs) to LTM entries. On recall, source refs are returned alongside the entry — the calling agent can re-interrogate the original file when the text summary isn't sufficient.

```python
# At perception time: store text summary + register the source file
entry = agent.consolidate_ltm(
    narrative="Room scene: wooden table, red doll on left, blue cup on right.",
    topics=["room", "table", "objects"],
)
agent.record_and_attach_source(
    ltm_entry_id = entry.id,
    location     = "/captures/living_room_001.jpg",
    type         = "image",
    description  = "Living room. Red doll on left side of table. Blue cup on right.",
    meta         = {"width": 1920, "height": 1080},
)

# At recall time
results = agent.recall("what was on the table")
for r in results:
    print(r.entry.content)
    for ref in r.sources:
        print(f"  Source: {ref.location}  ({ref.type})")
        # Re-interrogate the original file if the summary isn't enough:
        # enriched = vlm.describe(ref.location, "List every object on the table.")
        # agent.update_source_description(ref.id, enriched)
```

### Supported source types

| `type` | Typical use |
|---|---|
| `image` | Photos, screenshots, scans |
| `audio` | Voice memos, recordings |
| `video` | Video clips, surveillance frames |
| `pdf` | Documents, reports, forms |
| `webpage` | URLs, articles |
| `file` | Any other local file |
| `remote` | API endpoints, S3 objects |

---

## Architecture

### Short-Term Memory (STM)

```
[consN  ----  t1  ----  t2  ----  t3  ----  t4]
  ↑ rolling narrative           ↑ raw events (ground truth)
  updated by compress()         never deleted by compress()
  last_event_id bookmark        flushed only by flush_up_to()
```

The consN `payload` carries `last_event_id` — the last raw event folded into the current summary. Use `get_events_after(consN.payload["last_event_id"])` to get the new-event window.

The `flush_watermark` (stored in `stm_meta`) tracks the last event_id deleted by `flush_up_to()`. Independent of `last_event_id` — they may diverge significantly without error.

### Long-Term Memory (LTM)

| Field | Description |
|---|---|
| `class_type` | `event` `assertion` `decision` `procedure` `observation` `skill` `tool` |
| `content` | Narrative summary |
| `entities` | Referenced entity IDs |
| `topics` | Keywords and subjects |
| `concepts` | Cognitive triples: `operator:subject:focus` |
| `confidence` | Relevance/truth weight (0.0–1.0), decays over time |
| `embedding` | Semantic vector for recall |

### Entities — Living Historical Ledgers

```
described self as Kyle [auth:1] [m.ref:t1]
[ent.ref:sam-id] [auth:4] [dispute:sam-id] he is Musa not Kyle [m.ref:t2]
```

Authority tiers:
| Authority | Weight | Example |
|---|---|---|
| `self` | 1 | Subject's own claim |
| `peer` | 2 | Another person's observation |
| `system` | 3 | Automated sensor |
| `anchor` | 4 | Designated authoritative source |

### Recall Pipeline

1. Query parsing — extract operator (`who/what/where/…`), subjects, topics, time constraints
2. Structured filter — entity, topic, concept, time bracket
3. Semantic ranking — cosine similarity over LTM embeddings
4. Score blending — `score = (w_sem × semantic + (1−w_sem) × struct_boost) × confidence`
5. Reinforcement — recalled entries get a small confidence boost

### Forgetting

```
confidence(t) = confidence₀ × e^{−λ × elapsed_days}
```

Default `λ = 0.01` → ~50% decay after 70 days. Entries below the archive threshold become scars; stale scars are permanently deleted after the TTL. Scars can be rehydrated if a topic resurfaces.

---

## Package Structure

```
memory_module/
├── __init__.py            # Public API
├── agent.py               # MemoryAgent — unified interface for all operations
├── tools.py               # LLM tool schemas (Anthropic + OpenAI) + ToolExecutor dispatcher
├── models.py              # Dataclasses: STMSegment, Entity, LTMEntry, SourceRef, …
├── db.py                  # SQLite schema, connection manager, migration helpers
├── stm.py                 # STM: sliding window, compression, flush watermark
├── ltm.py                 # LTM: store, concepts, associations, archive
├── entities.py            # Entity ledger: narrative append, conflict resolution, fuzzy matching
├── recall.py              # Hybrid recall: query parser + structured filter + semantic ranking
├── forgetting.py          # Confidence decay, maintenance, reinforcement
├── embeddings.py          # sentence-transformers with TF-IDF fallback
├── sources.py             # Source references: register, attach, retrieve
├── example.py             # Full lifecycle walkthrough (no LLM required)
├── example_anthropic.py   # Anthropic agent loop with memory tools
├── example_openai.py      # OpenAI agent loop with memory tools
├── demo_live.py           # Live perceptual agent (cloud: Anthropic + Whisper)
├── demo_local.py          # Live perceptual agent (local: faster-whisper + Ollama)
├── demo_moonshine.py      # Live perceptual agent (offline: Moonshine ONNX + SmolVLM + Ollama)
└── requirements.txt
```

---

## Full API Reference

### MemoryAgent

**STM**

| Method | Description |
|---|---|
| `record_stm(content, source?, event_type?, payload?, confidence?)` | Append an event to STM |
| `forget_stm(segment_id)` | Remove a segment by ID |
| `get_stm_window()` | Return formatted STM string (consN + raw tail) |
| `get_flush_watermark()` | Last event_id that was flushed; None if nothing flushed yet |
| `flush_stm_up_to(event_id)` | Delete raw events ≤ event_id; advance watermark |

**LTM**

| Method | Description |
|---|---|
| `consolidate_ltm(narrative?, class_type?, topics?, entities?, confidence?, retain_tail?, per_segment?)` | Compress STM head → write LTM entries → flush head |
| `store_ltm(content, class_type?, topics?, entities?, confidence?)` | Directly store an LTM entry |
| `recall(query, top_k?)` | Hybrid recall → `list[RecallResult]` |

**Entities**

| Method | Description |
|---|---|
| `create_entity(description, name?, topics?)` | Create a new entity ledger |
| `resolve_entity(clues, top_k?)` | Fuzzy semantic search over entities |
| `observe_entity(entity_id, observation, authority?, memory_ref?, source_entity_id?)` | Append a sourced observation |
| `correct_entity(entity_id, correction, correcting_entity_id, memory_ref?)` | Record an authoritative correction |

**Sources**

| Method | Description |
|---|---|
| `record_source(location, type?, description?, captured_at?, meta?)` | Register a source asset |
| `attach_source(source_id, ltm_entry_id)` | Link a source to an LTM entry |
| `record_and_attach_source(ltm_entry_id, location, ...)` | Register + attach in one call |
| `sources_for_entry(ltm_entry_id)` | Return all sources for an LTM entry |
| `update_source_description(source_id, new_description)` | Enrich description after re-interrogation |

**Lifecycle**

| Method | Description |
|---|---|
| `run_decay()` | Apply time-based confidence decay |
| `run_maintenance()` | Archive weak entries; purge stale scars |
| `reinforce(entry_id, amount?)` | Manually boost an entry's confidence |
| `get_archive()` | List archived scars |
| `rehydrate(archive_id)` | Pull a scar back into active LTM |
| `status()` | Memory state snapshot dict |

### Tools

| Export | Description |
|---|---|
| `get_tools(format)` | Returns tool schemas: `"anthropic"` or `"openai"` |
| `ToolExecutor(agent)` | Dispatcher: routes LLM tool calls to MemoryAgent |
| `executor.run_anthropic(content)` | Process Anthropic `response.content` → tool_result list |
| `executor.run_openai(tool_calls)` | Process OpenAI `message.tool_calls` → message list |
| `executor.execute(name, input_dict)` | Execute a single tool by name |

---

## Running the Examples

```bash
# No LLM required — full lifecycle walkthrough
python memory_module/example.py

# With Claude (requires ANTHROPIC_API_KEY)
python memory_module/example_anthropic.py

# With GPT-4o (requires OPENAI_API_KEY)
python memory_module/example_openai.py

# Live perceptual agent — cloud (requires ANTHROPIC_API_KEY + OPENAI_API_KEY)
python memory_module/demo_live.py

# Live perceptual agent — local (requires Ollama + llava + llama3.2)
python memory_module/demo_local.py

# Live perceptual agent — fully offline (requires Ollama + Moonshine ONNX + pip packages)
python memory_module/demo_moonshine.py
```

---

## Tuning

**Forgetting rate:**

```python
from memory_module.forgetting import ForgettingEngine

agent.forgetting = ForgettingEngine(
    agent.db, agent.ltm,
    decay_lambda    = 0.005,   # slower: ~140 days to 50%
    archive_threshold = 0.15,
    archive_ttl_days  = 730,
)
```

**Recall balance:**

```python
from memory_module.recall import RecallQuery
from datetime import datetime

q = RecallQuery(
    raw            = "where was John last month",
    operator       = "where",
    subjects       = ["John"],
    after          = datetime(2024, 3, 1),
    semantic_weight = 0.7,
    top_k          = 5,
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

**Raw events are ground truth.** Compression updates the narrative shorthand; it never destroys the record. The flush cursor is explicit and independently controlled.

**Entities are chronicles.** A person is a ledger of everything ever claimed about them, by whom, and when. Truth is derived from history, not imposed.

**Forgetting is intentional.** Rarely recalled memories fade; frequently used ones stay sharp. The archive ensures nothing is permanently lost until dormant long enough to be irrelevant.

**Confidence is not certainty.** Contradictions coexist, each weighted by authority and recency. Social intelligence emerges from provenance.

---

## Relationship to Huginn

Muninn is the memory store. [Huginn](https://github.com/oumo-os/artux-huginn) is the cognitive layer built on top of it — Exilis (attention gate), Sagax (reasoning agent), Logos (background consolidation), and the Orchestrator.

| | Muninn | Huginn |
|---|---|---|
| **Does** | Store, retrieve, decay, archive | Ingest, reason, plan, consolidate, synthesise skills |
| **Writes LTM** | Never (passive store) | Yes — via Logos |
| **Can run alone** | Yes | No — requires Muninn |

Huginn agents interact with Muninn exclusively through `MemoryAgent`'s public API. No Huginn concepts (Logos, Sagax, Exilis, ASC, HTM) leak into Muninn.

---

## License

MIT
