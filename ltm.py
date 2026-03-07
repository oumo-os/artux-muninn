"""
ltm.py — Long-Term Memory manager.

Handles:
  • Persistence of LTMEntry records with embeddings
  • Concept and Association CRUD
  • LTM consolidation triggered from STM compression
  • Signature recording and entity linking
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Optional

from .db import Database, to_json, from_json
from .models import LTMEntry, Concept, Association, Signature, ArchiveEntry
from .embeddings import embed


class LTMManager:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # LTM Entries
    # ------------------------------------------------------------------

    def store(self, entry: LTMEntry) -> LTMEntry:
        """Persist an LTM entry (generates embedding if missing)."""
        if entry.embedding is None:
            entry.embedding = embed(entry.content)
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ltm_entries "
                "(id, class_type, content, entities, topics, concepts, "
                "associations, confidence, timestamp, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.class_type,
                    entry.content,
                    to_json(entry.entities),
                    to_json(entry.topics),
                    to_json(entry.concepts),
                    to_json(entry.associations),
                    entry.confidence,
                    entry.timestamp.isoformat(),
                    to_json(entry.embedding),
                ),
            )
        return entry

    def get(self, entry_id: str) -> Optional[LTMEntry]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM ltm_entries WHERE id = ?", (entry_id,)
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_all(self, min_confidence: float = 0.0) -> list[LTMEntry]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ltm_entries WHERE confidence >= ? "
                "ORDER BY timestamp DESC",
                (min_confidence,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def update_confidence(self, entry_id: str, confidence: float) -> None:
        confidence = max(0.0, min(1.0, confidence))
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE ltm_entries SET confidence = ? WHERE id = ?",
                (confidence, entry_id),
            )

    def delete(self, entry_id: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM ltm_entries WHERE id = ?", (entry_id,))

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate_from_stm(
        self,
        narrative: str,
        class_type: str = "event",
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
        confidence: float = 1.0,
        min_confidence_to_archive: float = 0.3,
    ) -> LTMEntry:
        """
        Create a new LTM entry from an STM compression narrative.
        Low-confidence entries are archived as scars instead of persisted.
        """
        entry = LTMEntry(
            class_type=class_type,
            content=narrative,
            entities=entities or [],
            topics=topics or [],
            concepts=concepts or [],
            confidence=confidence,
        )

        if confidence >= min_confidence_to_archive:
            return self.store(entry)
        else:
            # Demote to archive scar
            self.archive_entry(
                content=narrative,
                original_type="ltm",
                original_id=entry.id,
                reason="low-confidence on consolidation",
            )
            return entry   # returned but not stored in ltm_entries

    # ------------------------------------------------------------------
    # Concepts
    # ------------------------------------------------------------------

    def add_concept(self, concept: Concept) -> Concept:
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO concepts "
                "(id, operator, subject, focus, ltm_entry_id, entity_id, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    concept.id,
                    concept.operator,
                    concept.subject,
                    concept.focus,
                    concept.ltm_entry_id,
                    concept.entity_id,
                    concept.timestamp.isoformat(),
                ),
            )
        return concept

    def get_concepts(
        self,
        operator: str | None = None,
        subject: str | None = None,
    ) -> list[Concept]:
        query = "SELECT * FROM concepts WHERE 1=1"
        params: list = []
        if operator:
            query += " AND operator = ?"
            params.append(operator)
        if subject:
            query += " AND subject LIKE ?"
            params.append(f"%{subject}%")
        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_concept(r) for r in rows]

    # ------------------------------------------------------------------
    # Associations
    # ------------------------------------------------------------------

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float = 1.0,
    ) -> Association:
        assoc = Association(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            confidence=confidence,
        )
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO associations "
                "(id, source_id, target_id, relation, confidence, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    assoc.id,
                    assoc.source_id,
                    assoc.target_id,
                    assoc.relation,
                    assoc.confidence,
                    assoc.timestamp.isoformat(),
                ),
            )
        return assoc

    def get_associations(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        relation: str | None = None,
    ) -> list[Association]:
        query = "SELECT * FROM associations WHERE 1=1"
        params: list = []
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        if relation:
            query += " AND relation = ?"
            params.append(relation)
        with self.db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_association(r) for r in rows]

    # ------------------------------------------------------------------
    # Signatures
    # ------------------------------------------------------------------

    def record_signature(self, sig: Signature) -> Signature:
        if sig.embedding is None:
            sig.embedding = embed(sig.content)
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO signatures "
                "(id, modality, content, embedding, timestamp, confidence, "
                "topics, entity_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sig.id,
                    sig.modality,
                    sig.content,
                    to_json(sig.embedding),
                    sig.timestamp.isoformat(),
                    sig.confidence,
                    to_json(sig.topics),
                    to_json(sig.entity_ids),
                ),
            )
        return sig

    # ------------------------------------------------------------------
    # Archive (Scar)
    # ------------------------------------------------------------------

    def archive_entry(
        self,
        content: str,
        original_type: str,
        original_id: str,
        reason: str,
    ) -> ArchiveEntry:
        scar = ArchiveEntry(
            content=content,
            original_type=original_type,
            original_id=original_id,
            reason=reason,
        )
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO archive "
                "(id, content, original_type, original_id, reason, "
                "timestamp, rehydrated) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    scar.id,
                    scar.content,
                    scar.original_type,
                    scar.original_id,
                    scar.reason,
                    scar.timestamp.isoformat(),
                    0,
                ),
            )
        return scar

    def rehydrate(self, archive_id: str) -> Optional[LTMEntry]:
        """Pull an archived entry back into active LTM."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM archive WHERE id = ?", (archive_id,)
            ).fetchone()
        if not row:
            return None

        entry = LTMEntry(
            id=row["original_id"],
            class_type="observation",
            content=row["content"],
            confidence=0.5,
        )
        self.store(entry)

        with self.db.connection() as conn:
            conn.execute(
                "UPDATE archive SET rehydrated = 1 WHERE id = ?", (archive_id,)
            )
        return entry

    def get_archive(self) -> list[ArchiveEntry]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM archive ORDER BY timestamp DESC").fetchall()
        return [self._row_to_archive(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal row mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row) -> LTMEntry:
        emb_raw = row["embedding"]
        emb = from_json(emb_raw) if emb_raw else None
        return LTMEntry(
            id=row["id"],
            class_type=row["class_type"],
            content=row["content"],
            entities=from_json(row["entities"]),
            topics=from_json(row["topics"]),
            concepts=from_json(row["concepts"]),
            associations=from_json(row["associations"]),
            confidence=row["confidence"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            embedding=emb,
        )

    @staticmethod
    def _row_to_concept(row) -> Concept:
        return Concept(
            id=row["id"],
            operator=row["operator"],
            subject=row["subject"],
            focus=row["focus"],
            ltm_entry_id=row["ltm_entry_id"],
            entity_id=row["entity_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    @staticmethod
    def _row_to_association(row) -> Association:
        return Association(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=row["relation"],
            confidence=row["confidence"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    @staticmethod
    def _row_to_archive(row) -> ArchiveEntry:
        return ArchiveEntry(
            id=row["id"],
            content=row["content"],
            original_type=row["original_type"],
            original_id=row["original_id"],
            reason=row["reason"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            rehydrated=bool(row["rehydrated"]),
        )
