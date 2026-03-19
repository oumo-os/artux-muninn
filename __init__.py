"""
memory_module — Hybrid STM/LTM Memory System for AI Agents
===========================================================

A standalone, SQLite-backed cognitive memory module. Drop it into any
Python agent project — it has no opinion about which agents use it or how
they are structured.

Core concepts
-------------
STM  Short-Term Memory: an append-only log of raw events. Agents read it
     as a sliding awareness window. A rolling consN narrative summary sits
     at the head for quick context. Raw events are never deleted by the
     compression step — they are flushed only after a caller has verified
     they are durably written to LTM.

LTM  Long-Term Memory: persistent, queryable entries with semantic
     embeddings, confidence decay, entity associations, and source refs.

Entities  Living historical ledgers — append-only narratives that preserve
          every claim, correction, and observation with its source.

Recall    Hybrid: structured filters (entity, topic, concept, time) blended
          with cosine-similarity ranking over semantic embeddings.

Forgetting  Confidence decays exponentially (confidence × e^{−λt}).
            Weak entries are archived as scars; stale scars are purged.

Quick start
-----------
    from memory_module import MemoryAgent

    agent = MemoryAgent("agent.db")

    # Record a perception event
    agent.record_stm("User said: my name is Musa",
                     source="user", event_type="speech")

    # Create an entity ledger
    musa = agent.create_entity("described self as Musa", name="Musa",
                               topics=["identity"])
    agent.observe_entity(musa.id, "works on robotics", authority="self")

    # Consolidate to LTM
    agent.consolidate_ltm(topics=["identity", "robotics"],
                          entities=[musa.id])

    # Recall
    for r in agent.recall("who works on robotics"):
        print(r.score, r.entry.content)

    # Check flush cursor (for background consolidation agents)
    print(agent.get_flush_watermark())

Using LLM tool schemas
----------------------
    from memory_module import get_tools, ToolExecutor

    tools    = get_tools(format="anthropic")  # or "openai"
    executor = ToolExecutor(agent)

    # Pass `tools` to your LLM; call executor.run_anthropic(response.content)
    # or executor.run_openai(msg.tool_calls) to dispatch results.

Plugging in a custom compressor
--------------------------------
    def llm_compress(texts: list[str]) -> str:
        return call_your_llm("Summarise concisely:\\n" + "\\n".join(texts))

    agent = MemoryAgent("agent.db", compress_fn=llm_compress)
"""

from .agent import MemoryAgent
from .tools import get_tools, ToolExecutor
from .models import (
    STMSegment,
    Signature,
    Entity,
    LTMEntry,
    Concept,
    Association,
    ArchiveEntry,
    SourceRef,
)
from .db import Database
from .embeddings import SEMANTIC_AVAILABLE
from .recall import RecallResult, RecallQuery

__all__ = [
    # Primary interface
    "MemoryAgent",
    # LLM tool interface
    "get_tools",
    "ToolExecutor",
    # Data models
    "STMSegment",
    "Signature",
    "Entity",
    "LTMEntry",
    "Concept",
    "Association",
    "ArchiveEntry",
    "SourceRef",
    "RecallResult",
    "RecallQuery",
    # Low-level access
    "Database",
    # Feature flag
    "SEMANTIC_AVAILABLE",
]
