"""
recall.py — Hybrid recall engine.

Recall pipeline (mirrors §5 of the spec):
  1. Parse the query for operator, subjects, topics, time bracket.
  2. Apply structured filters (entity, topic, concept, time).
  3. Compute semantic similarity via embeddings.
  4. Combine scores (structured match boost + semantic similarity).
  5. Return ranked results with provenance.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, field
from datetime import datetime
from typing import Optional

from .db import Database, from_json
from .models import LTMEntry, Entity, SourceRef
from .embeddings import embed, cosine_similarity
from .ltm import LTMManager
from .entities import EntityManager
from .sources import SourceManager


# Recognised query operators
OPERATORS = {"what", "who", "when", "how", "where", "why", "dispute", "find"}


@dataclass
class RecallQuery:
    raw: str
    operator: Optional[str] = None
    subjects: list[str] = None          # type: ignore[assignment]
    topics: list[str] = None            # type: ignore[assignment]
    after: Optional[datetime] = None
    before: Optional[datetime] = None
    min_confidence: float = 0.0
    top_k: int = 10
    semantic_weight: float = 0.5        # blend ratio (0=structured only, 1=semantic only)

    def __post_init__(self):
        if self.subjects is None:
            self.subjects = []
        if self.topics is None:
            self.topics = []


@dataclass
class RecallResult:
    entry: LTMEntry
    score: float
    match_reasons: list[str]
    sources: list[SourceRef] = field(default_factory=list)
    """
    External knowledge sources that backed this memory at record time.
    Each SourceRef carries a `location` (file path or URL) and a
    `description` (what the agent derived from it then).

    If the text in `entry.content` doesn't satisfy the calling agent,
    it should re-examine the source at `location` using the appropriate
    tool (VLM for images, ASR for audio, PDF reader, web fetcher, etc.).
    """
    file_refs: list = field(default_factory=list)  # FileRef objects attached to this entry


def _parse_query(raw: str) -> RecallQuery:
    """
    Very lightweight query parser.
    Recognises:
      - leading operator keyword  (what / who / when / …)
      - quoted phrases as explicit topics
      - after:YYYY-MM-DD / before:YYYY-MM-DD constraints
      - everything else → subjects/topics by keyword split
    """
    q = RecallQuery(raw=raw)
    text = raw.lower().strip()

    # Temporal constraints
    after_m = re.search(r"after:(\d{4}-\d{2}-\d{2})", text)
    before_m = re.search(r"before:(\d{4}-\d{2}-\d{2})", text)
    if after_m:
        q.after = datetime.strptime(after_m.group(1), "%Y-%m-%d")
        text = text.replace(after_m.group(0), "")
    if before_m:
        q.before = datetime.strptime(before_m.group(1), "%Y-%m-%d")
        text = text.replace(before_m.group(0), "")

    # Operator
    for op in OPERATORS:
        if text.startswith(op):
            q.operator = op
            text = text[len(op):].strip(" :?")
            break

    # Quoted topics
    quoted = re.findall(r'"([^"]+)"', text)
    q.topics.extend(quoted)
    text = re.sub(r'"[^"]+"', "", text).strip()

    # Remaining words → subjects
    words = [w for w in re.split(r"\W+", text) if len(w) > 2]
    q.subjects.extend(words)

    return q


class RecallEngine:
    def __init__(self, db: Database, ltm: LTMManager, entities: EntityManager,
                 source_mgr=None):
        self.db = db
        self.ltm = ltm
        self.entities = entities
        self.source_mgr = source_mgr  # SourceManager | None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(self, query: str | RecallQuery, top_k: int = 10) -> list[RecallResult]:
        """
        Main recall entry point.
        Accepts either a raw string or a pre-built RecallQuery.
        Returns up to `top_k` results ranked by combined score.
        Each result includes any SourceRefs attached to the LTM entry —
        the caller decides whether to re-examine the source for more detail.
        """
        if isinstance(query, str):
            q = _parse_query(query)
            q.top_k = top_k
        else:
            q = query

        # Step 1: structured candidate set
        candidates = self._structured_filter(q)

        # Step 2: semantic re-ranking
        results = self._semantic_rank(q, candidates)

        # Step 3: reinforce confidence of recalled entries
        for r in results:
            self._reinforce(r.entry.id)

        # Step 4: attach source refs from SourceManager
        if self.source_mgr is not None:
            for r in results:
                r.sources = self.source_mgr.for_entry(r.entry.id)

        return results[:q.top_k]

    def recall_entities(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[Entity, float]]:
        """Semantic recall against the entity store."""
        return self.entities.resolve(query, threshold=threshold, top_k=top_k)

    # ------------------------------------------------------------------
    # Structured filter
    # ------------------------------------------------------------------

    def _structured_filter(self, q: RecallQuery) -> list[tuple[LTMEntry, list[str]]]:
        """
        Return (entry, reasons) pairs that pass at least one hard filter.
        An empty filter set returns all entries above min_confidence.
        """
        sql = "SELECT * FROM ltm_entries WHERE confidence >= ?"
        params: list = [q.min_confidence]

        if q.after:
            sql += " AND timestamp >= ?"
            params.append(q.after.isoformat())
        if q.before:
            sql += " AND timestamp <= ?"
            params.append(q.before.isoformat())

        with self.db.connection() as conn:
            rows = conn.execute(sql + " ORDER BY timestamp DESC", params).fetchall()

        entries = [LTMManager._row_to_entry(r) for r in rows]

        results: list[tuple[LTMEntry, list[str]]] = []
        for entry in entries:
            reasons: list[str] = []

            # Topic filter
            if q.topics:
                entry_topics = [t.lower() for t in entry.topics]
                for t in q.topics:
                    if t.lower() in entry_topics or t.lower() in entry.content.lower():
                        reasons.append(f"topic:{t}")

            # Subject / entity name filter
            if q.subjects:
                for s in q.subjects:
                    if s in entry.content.lower():
                        reasons.append(f"subject:{s}")

            # Operator / concept filter
            if q.operator:
                for c in entry.concepts:
                    if c.startswith(q.operator):
                        reasons.append(f"concept:{c}")

            # Include if any hit, or if no filters are active
            has_filters = bool(q.topics or q.subjects or q.operator)
            if reasons or not has_filters:
                results.append((entry, reasons))

        return results

    # ------------------------------------------------------------------
    # Semantic ranking
    # ------------------------------------------------------------------

    def _semantic_rank(
        self,
        q: RecallQuery,
        candidates: list[tuple[LTMEntry, list[str]]],
    ) -> list[RecallResult]:
        if not candidates:
            return []

        query_vec = embed(q.raw)
        scored: list[RecallResult] = []

        for entry, reasons in candidates:
            # Semantic similarity
            if entry.embedding:
                sem = cosine_similarity(query_vec, entry.embedding)
            else:
                sem = 0.0

            # Structured score (each reason adds a small boost)
            struct = min(1.0, len(reasons) * 0.25)

            # Blended score weighted by confidence
            blend = (
                q.semantic_weight * sem
                + (1 - q.semantic_weight) * struct
            ) * entry.confidence

            scored.append(RecallResult(entry=entry, score=blend, match_reasons=reasons))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Reinforcement on recall
    # ------------------------------------------------------------------

    def _reinforce(self, entry_id: str, boost: float = 0.02) -> None:
        """Slightly raise confidence of a recalled entry (capped at 1.0)."""
        entry = self.ltm.get(entry_id)
        if entry:
            new_conf = min(1.0, entry.confidence + boost)
            self.ltm.update_confidence(entry_id, new_conf)
