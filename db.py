"""
db.py — SQLite persistence layer.
Handles schema creation, JSON serialisation for list fields,
and a thread-safe connection context manager.
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
-- Short-Term Memory --------------------------------------------------------
-- Raw events are NEVER deleted by consN updates.
-- Deletion happens only via flush_up_to(), which advances the flush_watermark
-- and removes events that have been verified-consolidated to LTM.
CREATE TABLE IF NOT EXISTS stm_segments (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    is_compression  INTEGER DEFAULT 0,
    source          TEXT    DEFAULT '',   -- who produced this event
    event_type      TEXT    DEFAULT '',   -- speech | tool_call | tool_result | sensor | internal | …
    payload         TEXT    DEFAULT '{}', -- JSON: structured metadata; consN stores last_event_id here
    confidence      REAL    DEFAULT 1.0   -- producer-assigned confidence
);

-- STM key-value meta (flush watermark, session state, …) -------------------
-- Used by callers to persist cursor state without polluting the event table.
CREATE TABLE IF NOT EXISTS stm_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL DEFAULT ''
);

-- Perceptual layer ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS signatures (
    id          TEXT PRIMARY KEY,
    modality    TEXT NOT NULL DEFAULT 'text',
    content     TEXT NOT NULL,
    embedding   TEXT,                   -- JSON list[float] | NULL
    timestamp   TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    topics      TEXT DEFAULT '[]',      -- JSON list[str]
    entity_ids  TEXT DEFAULT '[]'
);

-- Entity ledger -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT DEFAULT '',
    content     TEXT NOT NULL DEFAULT '',
    topics      TEXT DEFAULT '[]',
    embedding   TEXT,
    created_at  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

-- Long-Term Memory ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS ltm_entries (
    id          TEXT PRIMARY KEY,
    class_type  TEXT NOT NULL DEFAULT 'assertion',
    content     TEXT NOT NULL,
    entities    TEXT DEFAULT '[]',
    topics      TEXT DEFAULT '[]',
    concepts    TEXT DEFAULT '[]',
    associations TEXT DEFAULT '[]',
    confidence  REAL DEFAULT 1.0,
    timestamp   TEXT NOT NULL,
    embedding   TEXT
);

-- Source References --------------------------------------------------------
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL DEFAULT 'file',
    location        TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    captured_at     TEXT NOT NULL,
    recorded_at     TEXT NOT NULL,
    meta            TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ltm_sources (
    ltm_entry_id    TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    PRIMARY KEY (ltm_entry_id, source_id)
);

CREATE TABLE IF NOT EXISTS concepts (
    id              TEXT PRIMARY KEY,
    operator        TEXT NOT NULL,
    subject         TEXT NOT NULL,
    focus           TEXT NOT NULL,
    ltm_entry_id    TEXT,
    entity_id       TEXT,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS associations (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    relation    TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    timestamp   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS archive (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    original_type   TEXT NOT NULL,
    original_id     TEXT NOT NULL,
    reason          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    rehydrated      INTEGER DEFAULT 0
);

-- Indexes -------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_stm_ts        ON stm_segments(timestamp);
CREATE INDEX IF NOT EXISTS idx_stm_type      ON stm_segments(event_type);
CREATE INDEX IF NOT EXISTS idx_sources_type  ON sources(type);
CREATE INDEX IF NOT EXISTS idx_ltm_sources_ltm ON ltm_sources(ltm_entry_id);
CREATE INDEX IF NOT EXISTS idx_ltm_sources_src ON ltm_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_ltm_conf      ON ltm_entries(confidence);
CREATE INDEX IF NOT EXISTS idx_ltm_ts        ON ltm_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_ltm_type      ON ltm_entries(class_type);
CREATE INDEX IF NOT EXISTS idx_entity_name   ON entities(name);
CREATE INDEX IF NOT EXISTS idx_concept_op    ON concepts(operator, subject);
CREATE INDEX IF NOT EXISTS idx_assoc_src     ON associations(source_id);
CREATE INDEX IF NOT EXISTS idx_assoc_tgt     ON associations(target_id);
CREATE INDEX IF NOT EXISTS idx_archive_orig  ON archive(original_id);
"""

# Columns added after the initial schema — applied to existing databases.
_MIGRATIONS = [
    "ALTER TABLE stm_segments ADD COLUMN source      TEXT DEFAULT ''",
    "ALTER TABLE stm_segments ADD COLUMN event_type  TEXT DEFAULT ''",
    "ALTER TABLE stm_segments ADD COLUMN payload     TEXT DEFAULT '{}'",
    "ALTER TABLE stm_segments ADD COLUMN confidence  REAL DEFAULT 1.0",
]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def to_json(value) -> str:
    if value is None:
        return "[]"
    return json.dumps(value)


def from_json(value: str | None):
    if value is None:
        return []
    return json.loads(value)


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Thin wrapper around sqlite3.  All callers use `db.connection()` as a
    context manager; commits on success, rolls back on exception.

    For ":memory:" databases a single persistent connection is held open
    for the lifetime of the instance — SQLite in-memory databases vanish
    when their last connection closes.
    """

    def __init__(self, db_path: str = "memory.db"):
        # ":memory:" is SQLite's in-process database — Path.resolve() would
        # turn it into a real file path (/:memory:) which is not what we want.
        self.db_path      = db_path if db_path == ":memory:" else str(Path(db_path).resolve())
        self._mem_conn    = None   # persistent connection for :memory: mode only
        self._init()

    def _init(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            # Apply additive migrations to existing databases.
            # SQLite has no IF NOT EXISTS on ALTER TABLE — we swallow
            # the OperationalError that fires when a column already exists.
            for sql in _MIGRATIONS:
                try:
                    conn.execute(sql)
                except Exception:
                    pass  # column already present

    def close(self) -> None:
        """Explicitly close a persistent in-memory connection."""
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        if self.db_path == ":memory:":
            # Reuse the single persistent connection; don't close it on exit.
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
                self._mem_conn.execute("PRAGMA foreign_keys=ON")
            conn = self._mem_conn
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            # deliberately no conn.close() here
        else:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
