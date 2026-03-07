"""
stm.py — Short-Term Memory manager.

Maintains a sliding window of temporal segments:
    [consN ---- t1 ---- t2 ---- t3]

When the window exceeds `max_segments`, the oldest non-compression
segments are folded into a new consN via a pluggable `compress_fn`.
"""

from __future__ import annotations
from datetime import datetime
from typing import Callable, Optional

from .db import Database, to_json, from_json
from .models import STMSegment


# Default: keep 10 raw segments before compressing
DEFAULT_MAX_SEGMENTS = 10


class STMManager:
    """
    Manages the Short-Term Memory store.

    compress_fn: optional callable(list[str]) -> str
        Receives a list of segment contents and returns a compressed
        narrative.  If None, a simple concatenation is used (useful for
        testing without an LLM).
    """

    def __init__(
        self,
        db: Database,
        max_segments: int = DEFAULT_MAX_SEGMENTS,
        compress_fn: Optional[Callable[[list[str]], str]] = None,
    ):
        self.db = db
        self.max_segments = max_segments
        self.compress_fn = compress_fn or self._default_compress

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, content: str) -> STMSegment:
        """Append a new temporal segment and trigger compression if needed."""
        seg = STMSegment(content=content)
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO stm_segments (id, content, timestamp, is_compression) "
                "VALUES (?, ?, ?, ?)",
                (seg.id, seg.content, seg.timestamp.isoformat(), 0),
            )
        self._maybe_compress()
        return seg

    def forget(self, segment_id: str) -> None:
        """Remove a specific segment from STM."""
        with self.db.connection() as conn:
            conn.execute("DELETE FROM stm_segments WHERE id = ?", (segment_id,))

    def clear(self) -> None:
        """Wipe all STM segments (destructive — use with care)."""
        with self.db.connection() as conn:
            conn.execute("DELETE FROM stm_segments")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> list[STMSegment]:
        """Return all segments ordered oldest → newest."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stm_segments ORDER BY timestamp ASC"
            ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_window(self) -> str:
        """
        Return the current STM window as a single narrative string:
        [consN][t1][t2]…
        Useful as context injection into an LLM prompt.
        """
        segments = self.get_all()
        parts = []
        for seg in segments:
            if seg.is_compression:
                parts.append(f"[SUMMARY: {seg.content}]")
            else:
                parts.append(f"[{seg.timestamp.strftime('%H:%M:%S')}] {seg.content}")
        return "\n".join(parts)

    def count(self) -> int:
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM stm_segments WHERE is_compression = 0"
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self) -> Optional[STMSegment]:
        """
        Force compression of all current raw segments into a consN.
        Returns the new compression segment, or None if nothing to compress.
        """
        raw = self._get_raw_segments()
        if not raw:
            return None

        # Build compressed narrative
        texts = [s.content for s in raw]
        summary = self.compress_fn(texts)

        cons = STMSegment(content=summary, is_compression=True)

        with self.db.connection() as conn:
            # Remove previous compressions (we fold them in)
            conn.execute(
                "DELETE FROM stm_segments WHERE is_compression = 1"
            )
            # Remove the raw segments we just compressed
            ids = tuple(s.id for s in raw)
            conn.execute(
                f"DELETE FROM stm_segments WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            # Insert the new consN
            conn.execute(
                "INSERT INTO stm_segments (id, content, timestamp, is_compression) "
                "VALUES (?, ?, ?, ?)",
                (cons.id, cons.content, cons.timestamp.isoformat(), 1),
            )

        return cons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        if self.count() >= self.max_segments:
            self.compress()

    def _get_raw_segments(self) -> list[STMSegment]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stm_segments ORDER BY timestamp ASC"
            ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    @staticmethod
    def _default_compress(texts: list[str]) -> str:
        """Naive fallback: join all texts with separators."""
        return " | ".join(texts)

    @staticmethod
    def _row_to_segment(row) -> STMSegment:
        return STMSegment(
            id=row["id"],
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            is_compression=bool(row["is_compression"]),
        )
