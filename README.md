# 🧠 Muninn — Memory Module

**A standalone, hybrid STM/LTM memory system for AI agents.**  
Structured short-term and long-term memory, entity ledgers, surgical multi-signal recall, source references, and forgetting — backed entirely by SQLite.  
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
                                                            ↕
                                                    Recall Engine
                                          (concept · entity · association
                                           · topic · semantic · scar)
```

Raw STM events are never deleted by compression. They stay as ground truth until a consolidation agent explicitly flushes them after verifying their LTM entries are written. This separation makes Muninn safe for multi-agent use — a reasoning agent can update its rolling narrative without racing a background consolidation agent's flush pass.

---

## Recall — Muninn's Defining Strength

Most memory systems offer a search box. Muninn offers a surgical instrument.

The design premise is that **the LLM is the query parser**. When a reasoning agent needs to retrieve something, it has already reasoned about *what kind of thing* it needs — the subject, the cognitive frame, the relevant topics. Muninn takes those conclusions as direct structured inputs and applies them with precision across five complementary signals simultaneously. No NLP guesswork on Muninn's side.

### The problem topics solve — and why they matter most

Consider two scenarios where neither keyword search nor embedding similarity is sufficient:

**Scenario A — latent project identity.**  
A long conversation covers LLMs, orchestrators, triage loops, token streams, and attention gates. The name of the software project being discussed never comes up — participants already know what they're building. A future `recall(subject="my_project")` query finds nothing. The content is rich and detailed, but it contains no vocabulary that bridges to the project name. No embedding model can infer that connection from the text alone, because the speaker never made it explicit.

**Scenario B — latent object identity.**  
Someone refers to their car exclusively as "Betty" or "her" across dozens of conversations. When the agent later searches for car maintenance records, other more semantically on-topic memories surface — actual discussions about cars, garages, servicing. "Betty needs her oil changed" sits buried near the bottom. The sentence does not embed near automotive concepts; the embedding model reads it as being about a person, and ranks it accordingly. The agent answers: *"I don't seem to have any record of car maintenance."* The memory exists. It was never surfaced.

And if it did surface by some chance — the agent reads a sentence about someone named Betty receiving an action. It has no basis to know Betty is a car. It dismisses the entry as a misclassification: a figure of speech, a nickname for a person, a joke. Retrieval buried it. Interpretation would have rejected it anyway.

This is what topics prevent. **Topics are out-of-band semantic annotations** — associations and identities that the content cannot carry itself and that no retrieval or interpretation step can recover post-hoc. When an agent with longitudinal context writes `topics=["car", "vehicle", "maintenance"]` onto every Betty entry, the connection is preserved explicitly. A future `recall(topics=["car"])` surfaces those entries with a direct structural signal. The agent no longer needs to infer what Betty is — the annotation already resolved it.

The requirement is that the annotating agent has seen enough to make the connection. An agent processing a single conversation where Betty is first mentioned cannot know she is a car. An agent reading events across many sessions — watching "Betty" appear consistently in contexts of fuel costs, tyre changes, oil checks, and parking — can recognise the pattern and annotate accordingly. This is why a background consolidation agent with longitudinal scope matters: it is the only component with enough accumulated evidence to establish these latent identities and preserve them.

Once topics are written correctly, recall is genuinely surgical. The project conversation surfaces for the project name query not because the content mentions it, but because the annotation carries it. Betty surfaces for `recall(topics=["car"])` not because she sounds like a car, but because the pattern was recognised across time and written down.

```
recall(
    operator       = "when",           # cognitive frame — matches concept triples
    subject        = "John",           # resolved to entity ID, followed into associations
    topics         = ["car",           # exact tag match against write-time annotations
                      "maintenance"],  #   — the only reliable path to content whose
                                       #     vocabulary doesn't match the query
    semantic_query = "last time John   # embedding similarity — sees everything,
                      did car work",   #   excludes nothing, bridges vocabulary gaps
    time_range     = {"after":         # hard SQL gate — the only exclusive filter
                      "2026-01-01"},
    include_scars  = False,
)
```

### Five signals, one ranked result

**1. Topic exact match (the out-of-band annotation layer)**  
Topics written at consolidation time are matched directly against the query's `topics` list. This is the *only* signal that can surface entries whose content is completely invisible to both keyword and embedding search — content where the relevant concept was never spoken or written, but was recognised and annotated by an agent with broader context. An entry tagged `topics=["car", "maintenance"]` surfaces exactly for `recall(topics=["car"])` regardless of whether the word "car" appears anywhere in the entry text. No other signal could find it.

**2. Concept triple lookup (highest structural precision)**  
At consolidation time, a consolidation agent writes concept triples: `what:john:lighting_preference`. At recall time, an `operator="what", subject="john"` query does a direct JOIN against those triples. Entries written with surgical intent are retrieved with surgical precision — no keyword guessing, no embedding approximation.

**3. Direct entity reference**  
When a subject name is given, Muninn resolves it to an entity ID. Any LTM entry that explicitly references that entity ID in its `entities` field gets a significant boost. This is exact — not proximity-based.

**4. Association traversal (1-hop relational depth)**  
Muninn walks the associations graph one hop from the resolved entity. If John is associated with a "movie night" event entity, entries that reference the movie night entity are scored as if they partially reference John — even if "John" never appears in their text. This is what gives Muninn relational depth rather than flat recall.

**5. Semantic similarity (fallback net)**  
The embedding similarity search sees *every* candidate that passed the hard gates. It excludes nothing. An entry about "a reptile spotted on the north wall" surfaces for a "gecko on the wall" query because the embedding space knows they're related — even though no keyword overlaps. Structural signals add to scores; only the time bracket and confidence threshold are exclusive.

**Scar hydration**  
With `include_scars=True`, Muninn also searches the archive of faded memories — entries that decayed below the confidence threshold. Scar results are returned alongside active entries, clearly flagged, with reduced score. The agent can decide whether a dim memory is relevant without permanently committing to rehydrating it.

### Score anatomy

```
score = (0.65 × semantic_similarity + 0.35 × struct_boost) × entry.confidence

struct_boost  — normalised with diminishing returns from:
  concept_triple  0.8   exact operator+subject triple match
  entity_ref      0.6   direct entity ID in entry.entities
  topic tag       0.5   exact tag written at consolidation
  assoc_hop       0.3   entity reachable via association graph
  topic_content   0.15  topic word appears in entry text
  subject_content 0.10  subject name appears in entry text
```

All struct signals are additive and use diminishing returns (`1 − 1/(1 + raw_score)`), so ten weak signals don't beat one strong one. Recalled entries get a small confidence reinforcement boost so frequently accessed knowledge stays sharp.

### Write precision is the multiplier

Recall quality scales directly with how carefully entries are written. When a consolidation agent tags topics precisely, links entity IDs, and records concept triples, every one of those decisions improves all future recall for those entries. An entry written with:

```python
agent.store_ltm(
    "Discussed distributed orchestration, triage loops, attention gates, and token routing.",
    class_type = "observation",
    topics     = ["my_project", "orchestrator", "triage"],   # ← latent identity preserved
    entities   = [project_entity_id],
    confidence = 0.92,
)
```

…is retrievable even though the project name appears nowhere in the content. A future `recall(subject="my_project")` or `recall(topics=["my_project"])` finds it precisely. Without those topic annotations, it is permanently invisible to any retrieval method — the vocabulary bridge simply does not exist in the content.

---

## Features

- **Surgical multi-signal recall** — concept triples, entity references, association traversal, topic exact match, and semantic similarity as additive scoring signals; only the time bracket and confidence threshold exclude
- **Two-tier memory** — STM for immediate context; LTM for persistent, queryable knowledge
- **Append-only STM** — raw events survive compression; flushed only after verified LTM writes
- **Flush watermark** — consolidation agents resume exactly where they left off across restarts
- **Typed events** — `source`, `event_type`, `payload`, `confidence` on every STM event
- **Entity ledgers** — people, objects, and concepts are historical narrative records; contradictions preserved with authority weights
- **Association graph** — directed, weighted relationships between entities; traversed at recall time
- **Concept triples** — `operator:subject:focus` cognitive frames written at consolidation, matched exactly at recall
- **LLM tool call interface** — structured-parameter schemas for Anthropic and OpenAI; `ToolExecutor` dispatches tool calls to memory operations
- **Source references** — link LTM entries back to original files (images, audio, PDFs, URLs); surfaced on recall for re-interrogation
- **Scar hydration** — faded memories survive in the archive; surfaced on demand with `include_scars=True`
- **Forgetting** — confidence decays exponentially (`confidence × e^{−λt}`); weak entries archived as scars; stale scars purged
- **Graceful degradation** — runs without `sentence-transformers`; falls back to TF-IDF bag-of-words embeddings
- **Zero required external dependencies** — pure Python + SQLite stdlib

---

## Installation

```bash
git clone https://github.com/oumo-os/artux-muninn
cd artux-muninn
```

Muninn has no mandatory third-party dependencies — it runs on pure Python and SQLite. Install an embedding backend for meaningful semantic recall:

### Option A — llama-cpp-python + local GGUF (recommended)

Any GGUF embedding model works. Good choices:

| Model | Dims | Notes |
|---|---|---|
| `nomic-embed-text-v1.5` | 768 | Strong general-purpose, widely available |
| `mxbai-embed-large-v1` | 1024 | High accuracy |
| `bge-small-en-v1.5` | 384 | Fast, low memory |
| `all-MiniLM-L6-v2` | 384 | Widely available |

```bash
# CPU
pip install llama-cpp-python

# GPU — CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# GPU — Apple Silicon
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

Then point Muninn at the model — either via environment variable (picked up automatically by every agent):

```bash
export MUNINN_EMBEDDING_MODEL=/models/nomic-embed-text-v1.5.Q8_0.gguf
```

Or per-agent in code:

```python
agent = MemoryAgent("agent.db",
                    embedding_model_path="/models/nomic-embed-text-v1.5.Q8_0.gguf",
                    n_gpu_layers=-1)    # -1 = all layers on GPU, 0 = CPU only
```

### Option B — sentence-transformers

Downloads `all-MiniLM-L6-v2` (~80 MB) on first use:

```bash
pip install sentence-transformers
```

### No embedding library

Muninn falls back to a TF-IDF bag-of-words vector. Structural signals (topics, entity references, concept triples) still work fully. Semantic similarity will not bridge vocabulary gaps — see the Betty scenario in the Recall section for why that matters.

---

## Quick Start

```python
from memory_module import MemoryAgent, RecallQuery

agent = MemoryAgent("agent.db")

# Record a perception event
agent.record_stm("User said: my name is Musa, I work on robotics",
                 source="user", event_type="speech")

# Create an entity ledger
musa = agent.create_entity("described self as Musa", name="Musa",
                           topics=["identity"])
agent.observe_entity(musa.id, "works on robotics", authority="self")

# Consolidate to LTM — writes per-event entries, then a period summary, then flushes
entry = agent.consolidate_ltm(
    topics   = ["identity", "robotics"],
    entities = [musa.id],
)

# Write a concept triple so future recall is surgically precise
agent.add_concept("who", "Musa", "identity",
                  ltm_entry_id=entry.id, entity_id=musa.id)

# Recall with structured parameters
results = agent.recall(RecallQuery(
    operator       = "who",
    subject        = "Musa",
    semantic_query = "who is Musa and what do they do",
))
for r in results:
    print(f"{r.score:.3f}  {r.entry.content}")

# Forgetting
agent.run_decay()        # apply time-based confidence decay
agent.run_maintenance()  # archive weak entries, purge stale scars
```

---

## Using Memory as LLM Tools

Give your LLM the tool definitions; it reasons about what it needs and calls `recall` with the structured parameters it has already determined.

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
  - recall before answering anything that may involve past context.
    Provide structured parameters you have already reasoned about:
      operator, subject, topics, semantic_query, time_range, include_scars.
  - record_stm after every user message and important observation.
  - consolidate_ltm when you learn something durable. Tag topics precisely,
    link entity IDs, and write concept triples — this directly improves
    future recall accuracy.
  - create_entity for significant people or things; resolve_entity first.
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
            messages=[{"role": "system",
                       "content": "You have persistent memory. Use structured recall parameters."}]
                     + messages,
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
| `recall(operator?, subject?, topics?, semantic_query?, time_range?, include_scars?)` | Before answering anything that may involve past context. Provide whichever structured parameters you have already reasoned about. |
| `record_stm(content)` | After every user message and notable observation |
| `consolidate_ltm(narrative?, class_type?, topics?, entities?, confidence?)` | When something durable is learned — precision at write time pays dividends at recall time |
| `create_entity(name, description, topics?)` | First time a significant person/object is encountered |
| `observe_entity(entity_id, observation, authority?, memory_ref?)` | When something new is learned about a known entity |
| `resolve_entity(clues, top_k?)` | Before creating — check if the entity already exists |
| `get_stm_window()` | To inspect current immediate context |
| `record_source(location, type?, description?, meta?)` | Register an external file/URL that backs a memory |
| `update_source_description(source_id, new_description)` | After re-interrogating a source for richer detail |

### Pre-populating Context Before Tool Calls

For agents that want to inject relevant memory into the system prompt before the LLM even makes a tool call:

```python
from memory_module import RecallQuery

def build_system_prompt(user_message: str, entity_id: str = None) -> str:
    # Build a structured pre-fetch query from what we already know
    q = RecallQuery(
        subject        = entity_id,          # if we know who we're talking to
        semantic_query = user_message,        # what they just said
        top_k          = 4,
    )
    relevant = agent.recall(q)
    stm      = agent.get_stm_window()

    return f"""You are a personal AI assistant with persistent memory.

Current short-term context:
{stm or "(empty)"}

Relevant long-term memory:
{chr(10).join(
    f"• {r.entry.content}" + (f" [sources: {len(r.sources)}]" if r.sources else "")
    for r in relevant
) or "(nothing recalled yet)"}

Use recall with structured parameters before answering anything from memory.
"""
```

---

## STM Event Model

Every STM record is an `STMSegment`. Plain text works; the typed fields add richness for agents that care about event provenance.

```python
agent.record_stm(
    "Kettle reached 95°C",
    source     = "sensor",          # user | system | tool | sensor | agent
    event_type = "sensor",          # speech | tool_call | tool_result | sensor | internal | output
    payload    = {"temp_c": 95,     # structured JSON for typed events
                  "device": "kettle-01"},
    confidence = 0.98,
)
```

The `payload` field on a consN segment carries:
```json
{"last_event_id": "...", "event_count_folded": 12}
```

---

## STM Flush Model

Compression and flushing are intentionally separate operations.

**`compress()` / `compress_head(retain)`** — updates the rolling consN narrative. Raw events stay untouched. Call as often as needed; it cannot lose data.

**`flush_up_to(event_id)`** — deletes raw events up to and including the given ID, advances the `flush_watermark`. Call **only after LTM entries are verified written**.

`consolidate_ltm()` handles both in sequence for the common single-agent case:

```
compress_head(retain_tail) → write per-segment LTM entries → write period entry → flush_up_to(head[-1].id)
```

For a multi-agent setup where a background consolidation agent manages its own flush:

```python
last = agent.get_flush_watermark()   # None on first run

new_events = agent.stm.get_events_after(last)
for event in new_events:
    agent.store_ltm(event.content, class_type="observation",
                    confidence=event.confidence)

if new_events:
    agent.flush_stm_up_to(new_events[-1].id)
```

### Attention-gate polling

```python
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

Link external files to LTM entries. On recall, source refs are returned alongside the entry — the calling agent can re-interrogate the original file when the text summary isn't sufficient.

```python
entry = agent.consolidate_ltm(
    narrative = "Room scene: wooden table, red doll on left, blue cup on right.",
    topics    = ["room", "table", "objects"],
)
agent.record_and_attach_source(
    ltm_entry_id = entry.id,
    location     = "/captures/living_room_001.jpg",
    type         = "image",
    description  = "Living room. Red doll on left side of table. Blue cup on right.",
    meta         = {"width": 1920, "height": 1080},
)

results = agent.recall(RecallQuery(topics=["room"], semantic_query="what was on the table"))
for r in results:
    for ref in r.sources:
        print(f"Source: {ref.location}  ({ref.type})")
        # Re-interrogate when the summary isn't enough:
        # enriched = vlm.describe(ref.location, "List every object on the table.")
        # agent.update_source_description(ref.id, enriched)
```

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
| `entities` | Referenced entity IDs — resolved at recall time |
| `topics` | Tag vocabulary — matched exactly at recall time |
| `concepts` | Cognitive triples: `operator:subject:focus` — matched by `recall(operator=, subject=)` |
| `confidence` | Relevance/truth weight (0.0–1.0), decays over time |
| `embedding` | Semantic vector — cosine similarity at recall time |

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

```
RecallQuery(operator, subject, topics, semantic_query, time_range, include_scars)
     │
     ├─ 1. SQL hard gates ──── confidence ≥ threshold, timestamp in range
     │                         (only exclusive filters — everything else scores)
     │
     ├─ 2. Concept tier ─────── JOIN concepts WHERE operator+subject match
     │                           → concept_tier_ids (highest-precision candidates)
     │
     ├─ 3. Subject resolution ── entity name → entity IDs
     │                           → entity_ids
     │
     ├─ 4. Association expand ── 1-hop graph walk from entity_ids
     │                           → assoc_entity_ids
     │
     ├─ 5. Score all candidates ─ for every entry above hard gates:
     │       concept_triple  0.8  if entry.id in concept_tier_ids
     │       entity_ref      0.6  if entity_ids ∩ entry.entities
     │       topic_tag       0.5  if topic in entry.topics
     │       assoc_hop       0.3  if assoc_entity_ids ∩ entry.entities
     │       topic_content   0.15 if topic in entry.content
     │       subject_content 0.10 if subject in entry.content
     │       semantic        cosine_similarity(embed(semantic_query), entry.embedding)
     │
     ├─ 6. Blend ────────────── score = (0.65 × sem + 0.35 × struct) × confidence
     │
     ├─ 7. Sort + truncate ──── top_k results
     │
     ├─ 8. Reinforce ─────────── confidence += 0.02 (capped at 1.0)
     │
     ├─ 9. Attach sources ────── SourceRefs hydrated onto each result
     │
     └─ 10. Scar hydration ───── if include_scars=True, archive searched and merged
```

### Forgetting

```
confidence(t) = confidence₀ × e^{−λ × elapsed_days}
```

Default `λ = 0.01` → ~50% decay after 70 days. Entries below the archive threshold become scars; stale scars are permanently deleted after the TTL. Scars can be rehydrated via `include_scars=True` in recall.

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
├── recall.py              # Recall engine: concept tier, entity resolution, association traversal,
│                          #   topic match, semantic similarity, scar hydration
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
| `recall(query, top_k?)` | Multi-signal recall → `list[RecallResult]`; accepts `RecallQuery` or plain string |
| `add_concept(operator, subject, focus, ltm_entry_id?, entity_id?)` | Write a concept triple |
| `link_entities(entity1_id, entity2_id, relation, confidence?)` | Create a directed association |

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

### RecallQuery

```python
from memory_module import RecallQuery

RecallQuery(
    operator        = "who",           # cognitive frame — matches concept triples
    subject         = "Musa",          # name or entity_id — resolved automatically
    topics          = ["identity"],    # exact tag match
    semantic_query  = "who is Musa",   # embedding similarity
    after           = datetime(...),   # hard time gate
    before          = datetime(...),   # hard time gate
    min_confidence  = 0.0,             # hard confidence gate
    include_scars   = False,           # search archive too
    top_k           = 5,
    semantic_weight = 0.65,            # blend ratio (0=struct only, 1=semantic only)
)
```

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

**Recall signal weights:**

```python
from memory_module import RecallQuery

# Emphasise structured signals (precise vocabulary, known entities)
q = RecallQuery(
    operator       = "where",
    subject        = "John",
    after          = datetime(2026, 1, 1),
    semantic_query = "where was John in January",
    semantic_weight = 0.3,   # lean on struct signals
    top_k          = 5,
)

# Emphasise semantic (open-ended question, uncertain vocabulary)
q = RecallQuery(
    semantic_query  = "ambient observations from the living room",
    semantic_weight = 0.9,   # lean on embedding similarity
    top_k           = 10,
)
```

**Forgetting rate:**

```python
from memory_module.forgetting import ForgettingEngine

agent.forgetting = ForgettingEngine(
    agent.db, agent.ltm,
    decay_lambda      = 0.005,   # slower: ~140 days to 50%
    archive_threshold = 0.15,
    archive_ttl_days  = 730,
)
```

**Scheduled maintenance:**

```python
import schedule

schedule.every().day.at("03:00").do(agent.run_decay)
schedule.every().week.do(agent.run_maintenance)
```

---

## Design Philosophy

**The LLM is the query parser.** Muninn does not do NLP on a search string. The reasoning agent has already determined what it needs — operator, subject, topics. Muninn takes those conclusions and executes them with precision across five complementary signals.

**Topics are out-of-band semantic annotations, not keywords.** A conversation about a software project may never mention the project's name. An entry about "Betty" needing maintenance may never mention "car" — and unlike the project scenario, Betty's entries will not even surface near the top of an automotive search, because the embedding model reads those sentences as being about a person, not a vehicle. They get buried below more semantically relevant results. Topics are the only mechanism that preserves these latent identities and makes entries retrievable by concepts they never explicitly contain. No retrieval or interpretation step can recover that connection post-hoc — the annotation has to be written by an agent that already knows it. This is why a background consolidation agent with longitudinal scope is architecturally important: only an agent reading events across many sessions can recognise that Betty is a car and annotate accordingly. A session-scoped agent cannot — it hasn't seen enough.

**Write precision is the multiplier.** Every topic tag, entity ID, and concept triple written at consolidation time is a future recall signal. Imprecise writes mean imprecise recall. Precise writes mean surgical recall — entries surfacing for queries whose vocabulary has no overlap with their content.

**Structural signals reward precision; semantic ensures breadth.** Topic tags and concept triples are exact — they only fire when the annotation matches. Semantic similarity is the fallback net — it sees everything, excludes nothing, and surfaces related entries that share meaning without sharing words. Both are necessary. Neither is sufficient alone.

**Memory is metabolic.** It breathes — growing through consolidation, shrinking through forgetting, sharpening through reinforcement.

**Raw events are ground truth.** Compression updates the narrative shorthand; it never destroys the record. The flush cursor is explicit and independently controlled.

**Entities are chronicles.** A person is a ledger of everything ever claimed about them, by whom, and when. Truth is derived from history, not imposed.

**Forgetting is intentional.** Rarely recalled memories fade; frequently used ones stay sharp. Nothing is permanently lost until it has been dormant long enough to be irrelevant — and even then, `include_scars=True` can surface it.

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
