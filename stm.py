"""
stm.py — Short-Term Memory manager.

Maintains a sliding window of temporal segments:
    [consN ---- t1 ---- t2 ---- t3]

Rolling compression (triggered by _maybe_compress):
    When raw count hits max_segments, ALL raw segments + existing consN
    are folded into a new consN.  This is the continuous awareness window —
    older details fade naturally because the LLM compress_fn will deprioritise
    them as new events dominate.

LTM consolidation (compress_head):
    Splits raw into head (older) and tail (newest N).  Only the head is
    folded into consN and flushed.  The tail stays live as raw segments so
    the most recent events retain their individual identity and aren't
    silently absorbed into a narrative before the caller has had a chance
    to inspect or persist them.

    This prevents the "gecko on the wall" problem: an event 2 minutes ago
    with high LTM relevance doesn't get buried in a compression just because
    a lot happened afterward.
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
        """Return all segments (consN + raw) ordered oldest → newest."""
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
        """Number of raw (non-compression) segments."""
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM stm_segments WHERE is_compression = 0"
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self) -> Optional[STMSegment]:
        """
        Rolling compression: fold ALL raw segments (plus existing consN) into
        a new consN.  Called automatically by _maybe_compress.

        Returns the new consN segment, or None if there were no raw segments.
        """
        raw = self._get_raw_segments()
        if not raw:
            return None

        # Fold existing consN text into the new compression so no context is
        # silently dropped between rolling cycles.
        texts = self._prepend_existing_cons([s.content for s in raw])
        summary = self.compress_fn(texts)
        cons = STMSegment(content=summary, is_compression=True)

        with self.db.connection() as conn:
            conn.execute("DELETE FROM stm_segments WHERE is_compression = 1")
            ids = tuple(s.id for s in raw)
            conn.execute(
                f"DELETE FROM stm_segments WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            conn.execute(
                "INSERT INTO stm_segments (id, content, timestamp, is_compression) "
                "VALUES (?, ?, ?, ?)",
                (cons.id, cons.content, cons.timestamp.isoformat(), 1),
            )

        return cons

    def compress_head(self, retain: int = 3) -> tuple[Optional[STMSegment], list[STMSegment]]:
        """
        LTM-consolidation compression: compress the head (older raw segments)
        while leaving the `retain` newest raw segments live in STM.

        Layout before:  [consN?] [t1 t2 t3 … tN-retain | tN-retain+1 … tN]
                                  ←————————— head ————→   ←——— tail ———→
        Layout after:   [new_consN(old_consN+head)]  [tN-retain+1 … tN]

        Parameters
        ----------
        retain : int
            Number of newest raw segments to leave untouched in STM.
            Defaults to 3. Pass 0 to flush everything (equivalent to compress()).

        Returns
        -------
        (new_consN, head_segments)
            new_consN       — the new compression segment (None if head was empty)
            head_segments   — the raw segments that were folded and removed
        """
        raw = self._get_raw_segments()

        if retain >= len(raw):
            # Everything fits in the tail — nothing to flush.
            return None, []

        head = raw[:-retain] if retain > 0 else raw[:]
        # tail stays in STM untouched — we don't touch its rows at all

        texts = self._prepend_existing_cons([s.content for s in head])
        summary = self.compress_fn(texts)
        new_cons = STMSegment(content=summary, is_compression=True)

        with self.db.connection() as conn:
            # Replace old consN
            conn.execute("DELETE FROM stm_segments WHERE is_compression = 1")
            # Remove only the head segments
            ids = tuple(s.id for s in head)
            conn.execute(
                f"DELETE FROM stm_segments WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            # Insert new consN
            conn.execute(
                "INSERT INTO stm_segments (id, content, timestamp, is_compression) "
                "VALUES (?, ?, ?, ?)",
                (new_cons.id, new_cons.content, new_cons.timestamp.isoformat(), 1),
            )

        return new_cons, head

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        if self.count() >= self.max_segments:
            self.compress()

    def _get_raw_segments(self) -> list[STMSegment]:
        """Return only raw (non-compression) segments, oldest → newest."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stm_segments "
                "WHERE is_compression = 0 "
                "ORDER BY timestamp ASC"
            ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def _get_compression_segment(self) -> Optional[STMSegment]:
        """Return the current consN, or None if there isn't one."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM stm_segments WHERE is_compression = 1 LIMIT 1"
            ).fetchone()
        return self._row_to_segment(row) if row else None

    def _prepend_existing_cons(self, texts: list[str]) -> list[str]:
        """
        Prepend the current consN content to `texts` so that rolling or
        head compressions never silently discard the accumulated narrative.
        """
        cons = self._get_compression_segment()
        if cons:
            return [cons.content] + texts
        return texts

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
