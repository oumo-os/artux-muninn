"""
recall.py — Hybrid recall engine.

The LLM is the query parser. It reads the conversation, reasons about
what it needs to know, and calls recall() with explicit structured
parameters it already determined. This module is the surgical backend
that executes those parameters precisely.

Pipeline
--------
1.  SQL hard gates     — confidence threshold + time bracket.
                         These are the only exclusive filters.

2.  Concept tier       — JOIN the concepts table on operator+subject.
                         Entries with matching concept triples are
                         marked as tier-1 candidates (highest precision).

3.  Candidate pool     — All entries above the hard gates.

4.  Struct scoring     — For each candidate, compute a boost from:
                           • topic tag exact match
                           • direct entity_id reference in entry.entities
                           • association traversal (1-hop from queried entities)
                           • concept-tier membership

5.  Semantic scoring   — embed(semantic_query) → cosine similarity.
                         This sees ALL candidates from the hard gate;
                         no keyword gate ever excludes an entry.

6.  Score blend        — (w_sem × semantic + (1-w_sem) × struct) × confidence

7.  Sort + truncate    — top_k results returned.

8.  Reinforce          — Recalled entries get a small confidence boost.

9.  Attach sources     — SourceRefs hydrated onto each result.

10. Scar hydration     — If include_scars=True, archive is searched
                         with the same semantic query and merged in,
                         marked with a flag so the caller knows.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .db import Database, from_json
from .models import LTMEntry, Entity, SourceRef, ArchiveEntry
from .embeddings import embed, cosine_similarity
from .ltm import LTMManager
from .entities import EntityManager
from .sources import SourceManager


# Recognised query operators (same vocabulary as concept triples)
OPERATORS = frozenset({"what", "who", "when", "how", "where", "why", "dispute", "find"})


@dataclass
class RecallQuery:
    """
    Structured recall query. Every field is optional — the LLM fills in
    whichever fields it has reasoned about. Fields left as None are skipped.

    operator        — Cognitive frame: what / who / when / how / where / why / dispute.
                      Matches entries that have a concept triple with this operator.
    subject         — Name or entity_id of the primary subject.
                      Accepts a name string ("John") or a UUID entity_id.
                      The backend resolves names to entity IDs automatically.
    topics          — List of topic tags to match against entry.topics.
    semantic_query  — Freeform natural language for embedding similarity.
                      Can be the same as or different from subject/topics.
                      Leave None to skip semantic ranking.
    after / before  — Hard time bracket (ISO date strings or datetime objects).
    min_confidence  — Hard lower bound on entry confidence (default 0.0).
    include_scars   — If True, also search the archive for forgotten memories.
    top_k           — Maximum results to return (default 5).
    semantic_weight — Blend ratio: 0.0 = struct only, 1.0 = semantic only (default 0.65).
    """
    # Structured signals (LLM-provided)
    operator:        Optional[str]       = None
    subject:         Optional[str]       = None   # name or entity_id
    topics:          list[str]           = field(default_factory=list)
    # Semantic signal
    semantic_query:  Optional[str]       = None
    # Hard gates
    after:           Optional[datetime]  = None
    before:          Optional[datetime]  = None
    min_confidence:  float               = 0.0
    # Options
    include_scars:   bool                = False
    top_k:           int                 = 5
    semantic_weight: float               = 0.65


@dataclass
class RecallResult:
    """
    A single recall result.

    entry         — The LTM entry.
    score         — Combined score (0.0–1.0).
    match_reasons — Human-readable list of signals that contributed.
    sources       — External assets (images, audio, PDFs, URLs) that backed
                    this memory. Re-examine these when the text summary is
                    insufficient to answer the question.
    from_archive  — True if this result was hydrated from the scar archive.
    """
    entry:         LTMEntry
    score:         float
    match_reasons: list[str]
    sources:       list[SourceRef] = field(default_factory=list)
    from_archive:  bool            = False


class RecallEngine:
    def __init__(
        self,
        db: Database,
        ltm: LTMManager,
        entities: EntityManager,
        source_mgr: Optional[SourceManager] = None,
    ):
        self.db         = db
        self.ltm        = ltm
        self.entities   = entities
        self.source_mgr = source_mgr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(
        self,
        query:   "str | RecallQuery",
        top_k:   int = 5,
    ) -> list[RecallResult]:
        """
        Execute a recall query and return ranked results.

        Accepts either:
          - A RecallQuery with explicit structured parameters (preferred —
            the LLM fills these in from its reasoning).
          - A plain string, treated as semantic_query only (convenient for
            direct programmatic use without an LLM).
        """
        if isinstance(query, str):
            q = RecallQuery(semantic_query=query, top_k=top_k)
        else:
            q = query
            q.top_k = top_k

        # 1. Resolve subject name → entity IDs
        entity_ids = self._resolve_subject(q.subject)

        # 2. SQL hard gates → candidate pool
        candidates = self._hard_gate(q)
        if not candidates:
            results = []
        else:
            # 3. Concept tier — entries with matching concept triples
            concept_tier_ids = self._concept_tier(q)

            # 4. Association expansion — entity IDs reachable via associations
            assoc_entity_ids = self._association_expand(entity_ids)

            # 5. Score every candidate
            results = self._score_and_rank(
                q, candidates, entity_ids, assoc_entity_ids, concept_tier_ids
            )

        # 6. Scar hydration
        if q.include_scars and q.semantic_query:
            scar_results = self._hydrate_scars(q)
            # Merge, de-duplicate by entry id, re-sort
            seen = {r.entry.id for r in results}
            results.extend(r for r in scar_results if r.entry.id not in seen)
            results.sort(key=lambda r: r.score, reverse=True)

        # 7. Truncate, reinforce, attach sources
        final = results[:q.top_k]
        for r in final:
            if not r.from_archive:
                self._reinforce(r.entry.id)
            if self.source_mgr is not None:
                r.sources = self.source_mgr.for_entry(r.entry.id)

        return final

    def recall_entities(
        self,
        query:     str,
        top_k:     int   = 5,
        threshold: float = 0.3,
    ) -> list[tuple[Entity, float]]:
        """Semantic recall against the entity store."""
        return self.entities.resolve(query, threshold=threshold, top_k=top_k)

    # ------------------------------------------------------------------
    # Subject resolution
    # ------------------------------------------------------------------

    def _resolve_subject(self, subject: Optional[str]) -> set[str]:
        """
        Resolve the subject parameter to a set of entity IDs.

        If subject looks like a UUID, treat it as an entity_id directly.
        Otherwise, search entity names (case-insensitive, partial match).
        Returns empty set if subject is None or no entities are found.
        """
        if not subject:
            return set()

        # If it looks like a UUID, use directly
        if len(subject) == 36 and subject.count("-") == 4:
            return {subject}

        # Name lookup
        matches = self.entities.get_by_name(subject)
        return {e.id for e in matches}

    # ------------------------------------------------------------------
    # SQL hard gate
    # ------------------------------------------------------------------

    def _hard_gate(self, q: RecallQuery) -> list[LTMEntry]:
        """
        Return every active LTM entry that passes the hard gates.
        Only confidence threshold and time bracket are exclusive here.
        """
        sql    = "SELECT * FROM ltm_entries WHERE confidence >= ?"
        params: list = [q.min_confidence]

        if q.after:
            ts = q.after.isoformat() if isinstance(q.after, datetime) else q.after
            sql += " AND timestamp >= ?"
            params.append(ts)
        if q.before:
            ts = q.before.isoformat() if isinstance(q.before, datetime) else q.before
            sql += " AND timestamp <= ?"
            params.append(ts)

        with self.db.connection() as conn:
            rows = conn.execute(sql + " ORDER BY timestamp DESC", params).fetchall()

        return [LTMManager._row_to_entry(r) for r in rows]

    # ------------------------------------------------------------------
    # Concept tier
    # ------------------------------------------------------------------

    def _concept_tier(self, q: RecallQuery) -> set[str]:
        """
        Return the set of LTM entry IDs that have a concept triple
        matching the operator and/or subject. These get a significant
        struct boost — Logos wrote them with surgical intent.
        """
        if not q.operator and not q.subject:
            return set()

        sql    = "SELECT ltm_entry_id FROM concepts WHERE ltm_entry_id IS NOT NULL"
        params: list = []

        if q.operator and q.operator in OPERATORS:
            sql += " AND operator = ?"
            params.append(q.operator)

        if q.subject:
            sql += " AND (subject LIKE ? OR focus LIKE ?)"
            params.extend([f"%{q.subject}%", f"%{q.subject}%"])

        with self.db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return {r["ltm_entry_id"] for r in rows}

    # ------------------------------------------------------------------
    # Association expansion (1-hop)
    # ------------------------------------------------------------------

    def _association_expand(self, entity_ids: set[str]) -> set[str]:
        """
        Given a set of entity IDs, return the set of entity IDs reachable
        via one hop in the associations graph (in either direction).

        This gives relational depth: if John is associated with 'Home Alone
        movie night' and you query for John, entries that reference the
        movie night entity are scored as if they partially reference John.
        """
        if not entity_ids:
            return set()

        related: set[str] = set()
        placeholders = ",".join("?" * len(entity_ids))
        ids_list = list(entity_ids)

        with self.db.connection() as conn:
            # Forward: queried entity is the source
            rows = conn.execute(
                f"SELECT target_id FROM associations WHERE source_id IN ({placeholders})",
                ids_list,
            ).fetchall()
            related.update(r["target_id"] for r in rows)

            # Reverse: queried entity is the target
            rows = conn.execute(
                f"SELECT source_id FROM associations WHERE target_id IN ({placeholders})",
                ids_list,
            ).fetchall()
            related.update(r["source_id"] for r in rows)

        return related - entity_ids   # exclude the seed IDs themselves

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_and_rank(
        self,
        q:                RecallQuery,
        candidates:       list[LTMEntry],
        entity_ids:       set[str],
        assoc_entity_ids: set[str],
        concept_tier_ids: set[str],
    ) -> list[RecallResult]:
        """
        Score every candidate using struct signals + semantic similarity,
        blend, and sort. No entry is excluded at this stage.
        """
        query_vec = embed(q.semantic_query) if q.semantic_query else None
        topics_lower = {t.lower() for t in q.topics}
        results: list[RecallResult] = []

        for entry in candidates:
            reasons: list[str] = []
            raw_struct = 0.0

            entry_topics_lower = {t.lower() for t in entry.topics}
            entry_content_lower = entry.content.lower()
            entry_entity_set = set(entry.entities)

            # ── Concept tier (highest precision) ──────────────────────
            if entry.id in concept_tier_ids:
                reasons.append("concept_triple")
                raw_struct += 0.8

            # ── Topic exact match ──────────────────────────────────────
            for t in topics_lower:
                if t in entry_topics_lower:
                    reasons.append(f"topic:{t}")
                    raw_struct += 0.5

            # ── Direct entity reference ────────────────────────────────
            direct_hit = entity_ids & entry_entity_set
            if direct_hit:
                reasons.append(f"entity_ref:{len(direct_hit)}")
                raw_struct += 0.6 * len(direct_hit)

            # ── Association hop ────────────────────────────────────────
            # Entry references an entity that is 1-hop from the queried entity.
            assoc_hit = assoc_entity_ids & entry_entity_set
            if assoc_hit:
                reasons.append(f"assoc_hop:{len(assoc_hit)}")
                raw_struct += 0.3 * len(assoc_hit)

            # ── Topic appears in content (softer) ─────────────────────
            for t in topics_lower:
                if t not in entry_topics_lower and t in entry_content_lower:
                    reasons.append(f"topic_in_content:{t}")
                    raw_struct += 0.15

            # ── Subject name in content (softest structural signal) ────
            if q.subject and q.subject.lower() in entry_content_lower:
                reasons.append("subject_in_content")
                raw_struct += 0.10

            # ── Normalise struct to 0–1 with diminishing returns ───────
            struct = 1.0 - (1.0 / (1.0 + raw_struct)) if raw_struct > 0 else 0.0

            # ── Semantic similarity ────────────────────────────────────
            if query_vec is not None and entry.embedding:
                sem = cosine_similarity(query_vec, entry.embedding)
            else:
                sem = 0.0

            # ── Blend × confidence ────────────────────────────────────
            score = (
                q.semantic_weight * sem
                + (1.0 - q.semantic_weight) * struct
            ) * entry.confidence

            results.append(RecallResult(
                entry=entry, score=score, match_reasons=reasons
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Scar hydration
    # ------------------------------------------------------------------

    def _hydrate_scars(self, q: RecallQuery) -> list[RecallResult]:
        """
        Search the archive (scar store) using semantic similarity and
        return matching entries marked as from_archive=True.

        These are memories that faded below the confidence threshold but
        may still be relevant when explicitly requested.
        """
        if not q.semantic_query:
            return []

        query_vec = embed(q.semantic_query)

        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM archive WHERE rehydrated = 0 ORDER BY timestamp DESC"
            ).fetchall()

        scored: list[RecallResult] = []
        for row in rows:
            # Scars don't have embeddings — score by keyword overlap as proxy
            content = row["content"].lower()
            query_lower = q.semantic_query.lower()

            # Simple overlap score for scars (no embedding)
            words = [w for w in query_lower.split() if len(w) > 3]
            overlap = sum(1 for w in words if w in content) / max(len(words), 1)

            if overlap < 0.15:
                continue   # too distant to bother surfacing

            # Build a lightweight LTMEntry proxy for the scar
            scar_entry = LTMEntry(
                id        = row["original_id"],
                class_type = row["original_type"],
                content    = row["content"],
                confidence = 0.3,   # scars have reduced confidence on hydration
            )
            scored.append(RecallResult(
                entry        = scar_entry,
                score        = overlap * 0.3,   # scars rank below active entries
                match_reasons= [f"scar:overlap={overlap:.2f}"],
                from_archive  = True,
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:q.top_k]

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------

    def _reinforce(self, entry_id: str, boost: float = 0.02) -> None:
        """Slightly raise confidence of a recalled entry (capped at 1.0)."""
        entry = self.ltm.get(entry_id)
        if entry:
            self.ltm.update_confidence(entry_id, min(1.0, entry.confidence + boost))
