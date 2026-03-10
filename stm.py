"""
stm.py — Short-Term Memory manager.

Window layout:
    [consN  ----  t1  ----  t2  ----  t3]

Two independent operations — deliberately separated:

compress() / compress_head()
    Update the rolling consN narrative (and its last_event_id bookmark).
    Raw events are NEVER deleted here.  The consN is intentionally lossy;
    raw events remain as ground truth until explicitly flushed.

flush_up_to(event_id)
    Delete raw events up to and including the named event, then advance
    the flush_watermark.  Call this only after LTM entries have been
    durably written for those events.

consolidate_ltm() in agent.py uses both in sequence:
    1. compress_head(retain) → narrative + returns head segments
    2. (caller writes LTM entries)
    3. flush_up_to(head[-1].id) → delete flushed events, advance watermark

This separation means:
  • A lightweight agent can update consN frequently for context freshness
    without touching the event log.
  • A heavier consolidation agent can flush independently, on its own
    schedule, after verifying LTM writes.
  • Muninn is not aware of any specific agent architecture — the cursor
    is generic (flush_watermark) and the flush call is a plain method.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Callable, Optional

from .db import Database
from .models import STMSegment


DEFAULT_MAX_SEGMENTS = 10

# stm_meta keys
_KEY_WATERMARK = "flush_watermark"     # last event_id whose raw segment was deleted


class STMManager:
    """
    Manages the Short-Term Memory store.

    compress_fn : callable(list[str]) -> str
        Receives a list of text contents and returns a compressed narrative.
        If None, a simple join is used (suitable for testing without an LLM).
    """

    def __init__(
        self,
        db: Database,
        max_segments: int = DEFAULT_MAX_SEGMENTS,
        compress_fn: Optional[Callable[[list[str]], str]] = None,
    ):
        self.db           = db
        self.max_segments = max_segments
        self.compress_fn  = compress_fn or self._default_compress

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        content: str,
        source: str = "",
        event_type: str = "",
        payload: Optional[dict] = None,
        confidence: float = 1.0,
    ) -> STMSegment:
        """
        Append a new raw event and trigger consN update if needed.

        Parameters
        ----------
        source      : Who produced this event ("user", "system", "tool", …).
        event_type  : What kind of event ("speech", "tool_call", "sensor", …).
        payload     : Structured metadata dict for typed events.
        confidence  : Producer-assigned confidence (0.0–1.0).
        """
        seg = STMSegment(
            content    = content,
            source     = source,
            event_type = event_type,
            payload    = payload or {},
            confidence = confidence,
        )
        with self.db.connection() as conn:
            conn.execute(
                "INSERT INTO stm_segments "
                "(id, content, timestamp, is_compression, source, event_type, payload, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    seg.id,
                    seg.content,
                    seg.timestamp.isoformat(),
                    0,
                    seg.source,
                    seg.event_type,
                    json.dumps(seg.payload),
                    seg.confidence,
                ),
            )
        self._maybe_compress()
        return seg

    def forget(self, segment_id: str) -> None:
        """Remove a specific segment from STM by ID."""
        with self.db.connection() as conn:
            conn.execute("DELETE FROM stm_segments WHERE id = ?", (segment_id,))

    def clear(self) -> None:
        """Wipe all STM segments and reset the watermark.  Use with care."""
        with self.db.connection() as conn:
            conn.execute("DELETE FROM stm_segments")
            conn.execute("DELETE FROM stm_meta WHERE key = ?", (_KEY_WATERMARK,))

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
        Return the current STM window as a formatted string for prompt injection:
            [SUMMARY: …] [HH:MM:SS] t1 [HH:MM:SS] t2 …
        """
        parts = []
        for seg in self.get_all():
            if seg.is_compression:
                parts.append(f"[SUMMARY: {seg.content}]")
            else:
                parts.append(f"[{seg.timestamp.strftime('%H:%M:%S')}] {seg.content}")
        return "\n".join(parts)

    def count(self) -> int:
        """Number of raw (non-consN) segments currently in STM."""
        with self.db.connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM stm_segments WHERE is_compression = 0"
            ).fetchone()[0]

    def get_events_after(self, event_id: Optional[str]) -> list[STMSegment]:
        """
        Return raw events that arrived after the given event_id (by timestamp).
        Pass None to get all raw events.
        Useful for attention-gate agents that poll for new events to triage.
        """
        if event_id is None:
            return self._get_raw_segments()
        with self.db.connection() as conn:
            ts_row = conn.execute(
                "SELECT timestamp FROM stm_segments WHERE id = ?", (event_id,)
            ).fetchone()
            if not ts_row:
                return self._get_raw_segments()
            rows = conn.execute(
                "SELECT * FROM stm_segments "
                "WHERE is_compression = 0 AND timestamp > ? "
                "ORDER BY timestamp ASC",
                (ts_row["timestamp"],),
            ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    # ------------------------------------------------------------------
    # Watermark
    # ------------------------------------------------------------------

    def get_flush_watermark(self) -> Optional[str]:
        """
        Return the event_id of the last raw segment that was flushed.
        None means nothing has been flushed yet.
        """
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT value FROM stm_meta WHERE key = ?", (_KEY_WATERMARK,)
            ).fetchone()
        return row["value"] if row else None

    def flush_up_to(self, event_id: str) -> int:
        """
        Delete all raw segments up to and including `event_id` (by timestamp),
        then advance the flush_watermark.

        This is the only place raw events are deleted.
        Call this only after LTM entries for those events have been written.

        Returns the number of raw segments deleted.
        """
        with self.db.connection() as conn:
            ts_row = conn.execute(
                "SELECT timestamp FROM stm_segments WHERE id = ?", (event_id,)
            ).fetchone()
            if not ts_row:
                return 0

            target_ts = ts_row["timestamp"]

            n = conn.execute(
                "SELECT COUNT(*) FROM stm_segments "
                "WHERE is_compression = 0 AND timestamp <= ?",
                (target_ts,),
            ).fetchone()[0]

            conn.execute(
                "DELETE FROM stm_segments "
                "WHERE is_compression = 0 AND timestamp <= ?",
                (target_ts,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO stm_meta (key, value) VALUES (?, ?)",
                (_KEY_WATERMARK, event_id),
            )

        return n

    # ------------------------------------------------------------------
    # Compression  (consN update — no deletion)
    # ------------------------------------------------------------------

    def compress(self) -> Optional[STMSegment]:
        """
        Rolling consN update: fold ALL raw segments (+ existing consN) into
        a new consN.  Raw events are NOT deleted.

        Called automatically by _maybe_compress when the window fills up.
        Returns the new consN segment, or None if there were no raw segments.
        """
        raw = self._get_raw_segments()
        if not raw:
            return None
        return self._write_cons(raw)

    def compress_head(self, retain: int = 3) -> tuple[Optional[STMSegment], list[STMSegment]]:
        """
        LTM-consolidation consN update: fold the head (older raw segments)
        into a new consN, leave the `retain` newest raw segments untouched.

        Raw events are NOT deleted here — call flush_up_to(head[-1].id) after
        writing LTM entries for the head segments.

        Layout before:  [consN?] [t1 … t(N-retain) | t(N-retain+1) … tN]
                                  ←————— head ——————  ←—— tail ———→
        Layout after:   [new_consN] [t(N-retain+1) … tN]   (all raw still present)

        Returns
        -------
        (new_consN, head_segments)
            new_consN     — updated consN (None if head was empty)
            head_segments — the raw segments included in the new consN
                            (not yet deleted — caller decides when to flush)
        """
        raw = self._get_raw_segments()
        if retain >= len(raw):
            return None, []

        head = raw[:-retain] if retain > 0 else raw[:]
        new_cons = self._write_cons(head)
        return new_cons, head

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        if self.count() >= self.max_segments:
            self.compress()

    def _write_cons(self, head: list[STMSegment]) -> STMSegment:
        """
        Build a new consN from `head` (folding in the existing consN text),
        write it to the DB, and return the new STMSegment.
        Raw events in `head` are left untouched.
        """
        texts   = self._prepend_existing_cons([s.content for s in head])
        summary = self.compress_fn(texts)

        last_id = head[-1].id if head else ""
        cons_payload = {
            "last_event_id":     last_id,
            "event_count_folded": len(head),
        }
        cons = STMSegment(
            content        = summary,
            is_compression = True,
            event_type     = "internal",
            payload        = cons_payload,
        )

        with self.db.connection() as conn:
            conn.execute("DELETE FROM stm_segments WHERE is_compression = 1")
            conn.execute(
                "INSERT INTO stm_segments "
                "(id, content, timestamp, is_compression, source, event_type, payload, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cons.id,
                    cons.content,
                    cons.timestamp.isoformat(),
                    1,
                    cons.source,
                    cons.event_type,
                    json.dumps(cons.payload),
                    cons.confidence,
                ),
            )
        return cons

    def _get_raw_segments(self) -> list[STMSegment]:
        """Return only raw (non-consN) segments, oldest → newest."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stm_segments "
                "WHERE is_compression = 0 "
                "ORDER BY timestamp ASC"
            ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def _get_compression_segment(self) -> Optional[STMSegment]:
        """Return the current consN, or None if absent."""
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM stm_segments WHERE is_compression = 1 LIMIT 1"
            ).fetchone()
        return self._row_to_segment(row) if row else None

    def _prepend_existing_cons(self, texts: list[str]) -> list[str]:
        """
        Prepend the current consN text so rolling updates never silently
        discard the accumulated narrative.
        """
        cons = self._get_compression_segment()
        return ([cons.content] + texts) if cons else texts

    @staticmethod
    def _default_compress(texts: list[str]) -> str:
        return " | ".join(texts)

    @staticmethod
    def _row_to_segment(row) -> STMSegment:
        raw_payload = row["payload"] if "payload" in row.keys() else "{}"
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        return STMSegment(
            id             = row["id"],
            content        = row["content"],
            timestamp      = datetime.fromisoformat(row["timestamp"]),
            is_compression = bool(row["is_compression"]),
            source         = row["source"]     if "source"     in row.keys() else "",
            event_type     = row["event_type"] if "event_type" in row.keys() else "",
            payload        = payload,
            confidence     = row["confidence"] if "confidence" in row.keys() else 1.0,
        )
