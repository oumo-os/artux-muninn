"""
sources.py — Source Reference management.

A SourceRef is a pointer to an external knowledge asset (image, audio,
PDF, webpage, remote URL, etc.) that was the perceptual basis for one or
more LTM entries.

The memory module stores the reference and the agent-derived text summary
from the time of recording.  It returns both alongside LTM entries during
recall.  The calling agent (LLM, VLM, ASR system, etc.) decides whether
the summary is sufficient or whether to re-examine the original source.

Design principle:
    The memory module is an index, not a file system.
    It stores `location` (path or URL) and returns it.
    It never reads, fetches, or processes the source itself.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Optional

from .db import Database, from_json
from .models import SourceRef


class SourceManager:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # Create & store
    # ------------------------------------------------------------------

    def record(
        self,
        location: str,
        type: str = "file",
        description: str = "",
        captured_at: Optional[datetime] = None,
        meta: dict | None = None,
    ) -> SourceRef:
        """
        Register an external knowledge source.

        Parameters
        ----------
        location    : File path or URL pointing to the source asset.
        type        : One of image | audio | video | pdf | webpage | file | remote
        description : Agent-derived text summary of the source content at
                      record time.  This is what ends up in LTM; the source
                      is re-examined only if the agent decides the summary
                      is insufficient.
        captured_at : When the source was originally created/captured.
                      Defaults to now.
        meta        : Freeform metadata dict (e.g. {"width": 1920, "height": 1080,
                      "duration_s": 42, "page_count": 5, "mime": "image/jpeg"}).
        """
        ref = SourceRef(
            type=type,
            location=location,
            description=description,
            captured_at=captured_at or datetime.utcnow(),
            meta=meta or {},
        )
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO sources "
                "(id, type, location, description, captured_at, recorded_at, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ref.id,
                    ref.type,
                    ref.location,
                    ref.description,
                    ref.captured_at.isoformat(),
                    ref.recorded_at.isoformat(),
                    json.dumps(ref.meta),
                ),
            )
        return ref

    def attach(self, source_id: str, ltm_entry_id: str) -> None:
        """
        Link an existing source to an LTM entry (many-to-many).
        Safe to call multiple times — uses INSERT OR IGNORE.
        """
        with self.db.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ltm_sources (ltm_entry_id, source_id) "
                "VALUES (?, ?)",
                (ltm_entry_id, source_id),
            )

    def record_and_attach(
        self,
        ltm_entry_id: str,
        location: str,
        type: str = "file",
        description: str = "",
        captured_at: Optional[datetime] = None,
        meta: dict | None = None,
    ) -> SourceRef:
        """Convenience: record a new source and immediately attach it to an LTM entry."""
        ref = self.record(location, type, description, captured_at, meta)
        self.attach(ref.id, ltm_entry_id)
        return ref

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get(self, source_id: str) -> Optional[SourceRef]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?", (source_id,)
            ).fetchone()
        return self._row(row) if row else None

    def for_entry(self, ltm_entry_id: str) -> list[SourceRef]:
        """Return all sources attached to a given LTM entry."""
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT s.* FROM sources s
                JOIN ltm_sources ls ON s.id = ls.source_id
                WHERE ls.ltm_entry_id = ?
                ORDER BY s.captured_at ASC
                """,
                (ltm_entry_id,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def entries_for_source(self, source_id: str) -> list[str]:
        """Return all LTM entry IDs that reference a given source."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT ltm_entry_id FROM ltm_sources WHERE source_id = ?",
                (source_id,),
            ).fetchall()
        return [r["ltm_entry_id"] for r in rows]

    def find_by_type(self, type: str) -> list[SourceRef]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE type = ? ORDER BY captured_at DESC",
                (type,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def find_by_location(self, location: str) -> Optional[SourceRef]:
        """Look up an existing source by its exact location (path or URL)."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE location = ?", (location,)
            ).fetchone()
        return self._row(row) if row else None

    def all(self) -> list[SourceRef]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM sources ORDER BY captured_at DESC").fetchall()
        return [self._row(r) for r in rows]

    # ------------------------------------------------------------------
    # Detach / delete
    # ------------------------------------------------------------------

    def detach(self, source_id: str, ltm_entry_id: str) -> None:
        """Remove the link between a source and an LTM entry."""
        with self.db.connection() as conn:
            conn.execute(
                "DELETE FROM ltm_sources WHERE ltm_entry_id = ? AND source_id = ?",
                (ltm_entry_id, source_id),
            )

    def delete(self, source_id: str) -> None:
        """
        Delete a source record and all its LTM links.
        Does NOT delete the actual file or URL — that is the agent's responsibility.
        """
        with self.db.connection() as conn:
            conn.execute("DELETE FROM ltm_sources WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row(row) -> SourceRef:
        return SourceRef(
            id=row["id"],
            type=row["type"],
            location=row["location"],
            description=row["description"],
            captured_at=datetime.fromisoformat(row["captured_at"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            meta=json.loads(row["meta"] or "{}"),
        )
