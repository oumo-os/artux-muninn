"""
entities.py — Entity management: creation, resolution, narrative updates,
and authority-weighted conflict handling.

Each Entity is a living historical ledger.  Contradictions are preserved
as sourced narrative lines; confidence is updated by authority weight.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Optional

from .db import Database, to_json, from_json
from .models import Entity
from .embeddings import embed, cosine_similarity


# Authority tiers (higher = more trusted)
AUTHORITY = {
    "self":     1,
    "peer":     2,
    "system":   3,
    "anchor":   4,   # explicitly designated authoritative source
}


class EntityManager:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str = "",
        initial_content: str = "",
        topics: list[str] | None = None,
    ) -> Entity:
        entity = Entity(
            name=name,
            content=initial_content,
            topics=topics or [],
        )
        entity.embedding = embed(initial_content or name)
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO entities "
                "(id, name, content, topics, embedding, created_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.id,
                    entity.name,
                    entity.content,
                    to_json(entity.topics),
                    to_json(entity.embedding),
                    entity.created_at.isoformat(),
                    entity.last_seen.isoformat(),
                ),
            )
        return entity

    def get(self, entity_id: str) -> Optional[Entity]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_by_name(self, name: str) -> list[Entity]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM entities WHERE name LIKE ?", (f"%{name}%",)
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def all(self) -> list[Entity]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM entities").fetchall()
        return [self._row_to_entity(r) for r in rows]

    def update(self, entity: Entity) -> None:
        entity.embedding = embed(entity.content or entity.name)
        with self.db.connection() as conn:
            conn.execute(
                "UPDATE entities SET name=?, content=?, topics=?, "
                "embedding=?, last_seen=? WHERE id=?",
                (
                    entity.name,
                    entity.content,
                    to_json(entity.topics),
                    to_json(entity.embedding),
                    entity.last_seen.isoformat(),
                    entity.id,
                ),
            )

    def delete(self, entity_id: str) -> None:
        with self.db.connection() as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))

    # ------------------------------------------------------------------
    # Narrative & Conflict
    # ------------------------------------------------------------------

    def append_observation(
        self,
        entity_id: str,
        observation: str,
        memory_ref: str = "",
        source_entity_id: str = "",
        authority: str = "peer",
    ) -> Entity:
        """
        Append a sourced observation to an entity's narrative ledger.
        Authority tier is recorded inline so conflict resolution can weigh
        competing claims later.
        """
        entity = self.get(entity_id)
        if entity is None:
            raise ValueError(f"Entity {entity_id} not found")

        auth_weight = AUTHORITY.get(authority, 2)
        entity.append_narrative(
            f"[auth:{auth_weight}] {observation}",
            memory_ref=memory_ref,
            entity_ref=source_entity_id,
        )
        self.update(entity)
        return entity

    def record_correction(
        self,
        entity_id: str,
        correction: str,
        correcting_entity_id: str,
        memory_ref: str = "",
        authority: str = "anchor",
    ) -> Entity:
        """
        Record that `correcting_entity_id` disputes or corrects a prior claim.
        Authority defaults to 'anchor' since corrections carry strong signal.
        """
        tag = f"dispute:{correcting_entity_id}"
        return self.append_observation(
            entity_id,
            f"[{tag}] {correction}",
            memory_ref=memory_ref,
            source_entity_id=correcting_entity_id,
            authority=authority,
        )

    # ------------------------------------------------------------------
    # Resolution (fuzzy match an unknown description to an existing entity)
    # ------------------------------------------------------------------

    def resolve(
        self,
        clues: str,
        threshold: float = 0.4,
        top_k: int = 3,
    ) -> list[tuple[Entity, float]]:
        """
        Given a natural-language clue string, return the top-k matching
        entities ranked by embedding cosine similarity.
        Returns list of (Entity, score) pairs.
        """
        query_vec = embed(clues)
        candidates = self.all()
        scored: list[tuple[Entity, float]] = []
        for ent in candidates:
            if ent.embedding:
                sim = cosine_similarity(query_vec, ent.embedding)
                if sim >= threshold:
                    scored.append((ent, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entity(row) -> Entity:
        emb_raw = row["embedding"]
        emb = from_json(emb_raw) if emb_raw else None
        return Entity(
            id=row["id"],
            name=row["name"] or "",
            content=row["content"] or "",
            topics=from_json(row["topics"]),
            embedding=emb,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
        )
