"""
agent.py — MemoryAgent: the unified tool interface for the Memory Module.

Usage
-----
    from memory_module import MemoryAgent

    agent = MemoryAgent("memory.db")
    agent.record_stm("User said: hello, my name is Musa")
    results = agent.recall("who is Musa")
"""

from __future__ import annotations
from datetime import datetime
from typing import Callable, Optional

from .db import Database
from .models import (
    LTMEntry, Entity, Concept, Association, Signature, ArchiveEntry, SourceRef
)
from .stm import STMManager
from .ltm import LTMManager
from .entities import EntityManager
from .recall import RecallEngine, RecallResult
from .forgetting import ForgettingEngine
from .sources import SourceManager


class MemoryAgent:
    """
    Single entry point for all memory operations.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    compress_fn : callable, optional
        Function(list[str]) -> str used to summarise STM segments.
        Pass your LLM call here for AI-quality compression.
        Defaults to a simple join (useful for testing).
    max_stm_segments : int
        Number of raw STM segments before auto-compression fires.
    decay_lambda : float
        Controls how fast LTM confidence decays (higher = faster).
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        compress_fn: Optional[Callable[[list[str]], str]] = None,
        max_stm_segments: int = 10,
        decay_lambda: float = 0.01,
    ):
        self.db = Database(db_path)
        self.ltm = LTMManager(self.db)
        self.entities = EntityManager(self.db)
        self.sources = SourceManager(self.db)
        self.stm = STMManager(
            self.db,
            max_segments=max_stm_segments,
            compress_fn=compress_fn,
        )
        self.recall_engine = RecallEngine(
            self.db, self.ltm, self.entities, source_mgr=self.sources
        )
        self.forgetting = ForgettingEngine(self.db, self.ltm, decay_lambda=decay_lambda)

    # ==================================================================
    # §7 Tool Interface — exact names from the spec
    # ==================================================================

    # --- Recall --------------------------------------------------------

    def recall(self, query: str, top_k: int = 10) -> list[RecallResult]:
        """
        Hybrid recall: structured + semantic.
        Returns ranked RecallResult objects with provenance.
        Each result includes any SourceRefs attached to the LTM entry.
        """
        return self.recall_engine.recall(query, top_k=top_k)

    # --- STM -----------------------------------------------------------

    def record_stm(
        self,
        segment: str,
        source: str = "",
        event_type: str = "",
        payload: Optional[dict] = None,
        confidence: float = 1.0,
    ):
        """
        Append a raw event to Short-Term Memory.

        Parameters
        ----------
        segment     : Text content of the event.
        source      : Who produced this event ("user", "system", "tool", "sensor", …).
        event_type  : Category ("speech", "tool_call", "tool_result", "sensor", …).
        payload     : Structured metadata dict for typed events.
        confidence  : Producer confidence in this event (0.0–1.0).
        """
        return self.stm.record(
            segment,
            source=source,
            event_type=event_type,
            payload=payload,
            confidence=confidence,
        )

    def forget_stm(self, segment_id: str) -> None:
        """Remove a specific STM segment by ID."""
        self.stm.forget(segment_id)

    def get_stm_window(self) -> str:
        """Return the current STM window as a formatted narrative string."""
        return self.stm.get_window()

    # --- LTM -----------------------------------------------------------

    def consolidate_ltm(
        self,
        narrative: Optional[str] = None,
        class_type: str = "event",
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
        confidence: float = 1.0,
        retain_tail: int = 3,
        per_segment: bool = True,
    ) -> LTMEntry:
        """
        Consolidate STM into LTM, then flush the consolidated events.

        When called without an explicit narrative this method:

          1. Calls compress_head(retain_tail) — updates the consN narrative
             to cover the head segments, marks its last_event_id bookmark.
             Raw events are NOT deleted at this step.

          2. Optionally writes one LTM entry per raw head segment
             (per_segment=True, default).  This ensures every event has its
             own LTM entry independent of what the compress_fn chose to keep
             in its narrative — the "gecko on the wall" is preserved even if
             the rolling summary doesn't mention it.

          3. Writes one "period" LTM entry from the compressed narrative.

          4. Calls flush_up_to(head[-1].id) — deletes the head raw events
             and advances the flush_watermark.  The tail raw segments remain
             live in STM.

        If `narrative` is supplied explicitly, the STM is left entirely
        untouched and that narrative is stored directly (same as before).

        Parameters
        ----------
        retain_tail : int
            Number of newest raw segments to keep live after flush. Default 3.
        per_segment : bool
            Write an individual LTM entry for each flushed raw segment.
        """
        if narrative is not None:
            return self.ltm.consolidate_from_stm(
                narrative=narrative,
                class_type=class_type,
                entities=entities or [],
                topics=topics or [],
                concepts=concepts or [],
                confidence=confidence,
            )

        # 1. Update consN with the head, get head segments back (not yet deleted)
        new_cons, head_segments = self.stm.compress_head(retain=retain_tail)

        # 2. Per-segment LTM entries for every flushed raw segment
        if per_segment:
            for seg in head_segments:
                if seg.is_compression:
                    continue
                self.ltm.consolidate_from_stm(
                    narrative=seg.content,
                    class_type=class_type,
                    entities=entities or [],
                    topics=topics or [],
                    concepts=concepts or [],
                    confidence=seg.confidence if seg.confidence < 1.0 else confidence,
                )

        # 3. Period LTM entry from the compressed narrative
        period_narrative = (new_cons.content if new_cons
                            else self.stm.get_window() or "(no content)")
        period_entry = self.ltm.consolidate_from_stm(
            narrative=period_narrative,
            class_type=class_type,
            entities=entities or [],
            topics=topics or [],
            concepts=concepts or [],
            confidence=confidence,
        )

        # 4. Flush the head events now that LTM entries are written
        if head_segments:
            self.stm.flush_up_to(head_segments[-1].id)

        return period_entry

    def flush_stm_up_to(self, event_id: str) -> int:
        """
        Explicitly flush STM raw events up to and including `event_id`.

        Use this when a separate consolidation agent (e.g. a background Logos
        process) has already written its own LTM entries and just needs to
        advance the flush cursor.

        Returns the number of segments deleted.
        """
        return self.stm.flush_up_to(event_id)

    def get_flush_watermark(self) -> Optional[str]:
        """
        Return the event_id of the last raw STM segment that was flushed,
        or None if nothing has been flushed yet.

        Useful for consolidation agents that need to resume from where they
        left off across restarts.
        """
        return self.stm.get_flush_watermark()

    # --- Entities ------------------------------------------------------

    def create_entity(
        self,
        description: str,
        name: str = "",
        topics: list[str] | None = None,
    ) -> Entity:
        """Create a new entity with an initial narrative description."""
        return self.entities.create(
            name=name, initial_content=description, topics=topics
        )

    def resolve_entity(self, clues: str, top_k: int = 3) -> list[tuple[Entity, float]]:
        """Find existing entities matching a natural-language description."""
        return self.recall_engine.recall_entities(clues, top_k=top_k)

    def observe_entity(
        self,
        entity_id: str,
        observation: str,
        memory_ref: str = "",
        source_entity_id: str = "",
        authority: str = "peer",
    ) -> Entity:
        """Append a sourced observation to an entity's ledger."""
        return self.entities.append_observation(
            entity_id, observation,
            memory_ref=memory_ref,
            source_entity_id=source_entity_id,
            authority=authority,
        )

    def correct_entity(
        self,
        entity_id: str,
        correction: str,
        correcting_entity_id: str,
        memory_ref: str = "",
    ) -> Entity:
        """Record an authoritative correction to an entity's identity."""
        return self.entities.record_correction(
            entity_id, correction,
            correcting_entity_id=correcting_entity_id,
            memory_ref=memory_ref,
        )

    # --- Signatures ----------------------------------------------------

    def associate_signature(
        self,
        content: str,
        modality: str = "text",
        entity_ids: list[str] | None = None,
        topics: list[str] | None = None,
        confidence: float = 1.0,
    ) -> Signature:
        """Create a signature and associate it with entities."""
        sig = Signature(
            modality=modality,
            content=content,
            topics=topics or [],
            entity_ids=entity_ids or [],
            confidence=confidence,
        )
        return self.ltm.record_signature(sig)

    # --- Associations --------------------------------------------------

    def link_entities(
        self,
        entity1: str,
        entity2: str,
        relation: str,
        confidence: float = 1.0,
    ) -> Association:
        """Create a directed association between two entity IDs."""
        return self.ltm.link(entity1, entity2, relation, confidence)

    def infer_relationships(self) -> list[Association]:
        """Return all stored associations."""
        return self.ltm.get_associations()

    # ==================================================================
    # Source References
    # ==================================================================

    def record_source(
        self,
        location: str,
        type: str = "file",
        description: str = "",
        captured_at: Optional[datetime] = None,
        meta: dict | None = None,
    ) -> SourceRef:
        """
        Register an external knowledge source (image, audio, PDF, webpage, etc.)

        Parameters
        ----------
        location    : File path or URL pointing to the source asset.
        type        : image | audio | video | pdf | webpage | file | remote
        description : Agent-derived text summary of the source at record time.
                      This is what ends up in LTM text; the source is only
                      re-examined if the agent decides the description is
                      insufficient to answer a query.
        captured_at : When the source was originally created/captured.
        meta        : Freeform metadata dict.
                      e.g. {"width": 1920, "height": 1080, "mime": "image/jpeg"}
                           {"duration_s": 42, "language": "en"}
                           {"page_count": 5, "author": "Alice"}
        """
        return self.sources.record(
            location=location,
            type=type,
            description=description,
            captured_at=captured_at,
            meta=meta,
        )

    def attach_source(self, source_id: str, ltm_entry_id: str) -> None:
        """Link an existing source to an LTM entry (many-to-many)."""
        self.sources.attach(source_id, ltm_entry_id)

    def record_and_attach_source(
        self,
        ltm_entry_id: str,
        location: str,
        type: str = "file",
        description: str = "",
        captured_at: Optional[datetime] = None,
        meta: dict | None = None,
    ) -> SourceRef:
        """
        Convenience: register a source and immediately attach it to an LTM entry.
        This is the most common single-call pattern when consolidating a perception.
        """
        return self.sources.record_and_attach(
            ltm_entry_id=ltm_entry_id,
            location=location,
            type=type,
            description=description,
            captured_at=captured_at,
            meta=meta,
        )

    def sources_for_entry(self, ltm_entry_id: str) -> list[SourceRef]:
        """Return all sources attached to a given LTM entry."""
        return self.sources.for_entry(ltm_entry_id)

    def update_source_description(
        self, source_id: str, new_description: str
    ) -> Optional[SourceRef]:
        """
        Update a source's description after re-interrogation by a VLM/ASR/reader.
        Call this so future recalls benefit from the richer detail.
        """
        ref = self.sources.get(source_id)
        if ref is None:
            return None
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE sources SET description = ? WHERE id = ?",
                (new_description, source_id),
            )
        ref.description = new_description
        return ref

    # ==================================================================
    # Convenience helpers
    # ==================================================================

    def add_concept(
        self,
        operator: str,
        subject: str,
        focus: str,
        ltm_entry_id: str | None = None,
        entity_id: str | None = None,
    ) -> Concept:
        """Register a cognitive concept triple (operator:subject:focus)."""
        concept = Concept(
            operator=operator,
            subject=subject,
            focus=focus,
            ltm_entry_id=ltm_entry_id,
            entity_id=entity_id,
        )
        return self.ltm.add_concept(concept)

    def store_ltm(
        self,
        content: str,
        class_type: str = "assertion",
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
        confidence: float = 1.0,
    ) -> LTMEntry:
        """Directly persist an LTM entry without going through STM."""
        entry = LTMEntry(
            class_type=class_type,
            content=content,
            entities=entities or [],
            topics=topics or [],
            concepts=concepts or [],
            confidence=confidence,
        )
        return self.ltm.store(entry)

    def run_decay(self) -> int:
        """Apply time-based confidence decay to all LTM entries."""
        return self.forgetting.run_decay()

    def run_maintenance(self) -> dict:
        """Archive weak entries and purge stale scars."""
        return self.forgetting.run_maintenance()

    def reinforce(self, entry_id: str, amount: float = 0.1) -> Optional[float]:
        """Manually boost an LTM entry's confidence."""
        return self.forgetting.reinforce(entry_id, amount)

    def get_archive(self) -> list[ArchiveEntry]:
        """List all archived (scar) entries."""
        return self.ltm.get_archive()

    def rehydrate(self, archive_id: str) -> Optional[LTMEntry]:
        """Pull an archived scar back into active LTM."""
        return self.ltm.rehydrate(archive_id)

	def get_raw_events(self, after_id: Optional[str] = None, limit: int = 100) -> list[STMSegment]:
	    """
	    Return raw STM events after `after_id`, up to `limit`.
	    Pass after_id=get_flush_watermark() to get only unprocessed events.
	    """
	    events = self.stm.get_events_after(after_id)
	    return events[:limit]
    
    
    # ==================================================================
    # Debug / introspection
    # ==================================================================

    def status(self) -> dict:
        """Return a snapshot of current memory state."""
        ltm_all = self.ltm.get_all()
        archive = self.ltm.get_archive()
        all_sources = self.sources.all()
        return {
            "stm_segments": self.stm.count(),
            "ltm_entries": len(ltm_all),
            "ltm_avg_confidence": (
                sum(e.confidence for e in ltm_all) / len(ltm_all)
                if ltm_all else 0.0
            ),
            "entities": len(self.entities.all()),
            "archive_scars": len(archive),
            "sources": {
                "total": len(all_sources),
                "by_type": {
                    t: sum(1 for s in all_sources if s.type == t)
                    for t in set(s.type for s in all_sources)
                }
            }
        }
