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
CREATE TABLE IF NOT EXISTS stm_segments (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    is_compression  INTEGER DEFAULT 0
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
-- A source is any external knowledge asset (file, URL, etc.) that backs
-- one or more LTM entries.  Stored once; linked to many entries via the
-- join table below.
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL DEFAULT 'file',
    location        TEXT NOT NULL,          -- file path or URL
    description     TEXT NOT NULL DEFAULT '',
    captured_at     TEXT NOT NULL,          -- when the source was created/captured
    recorded_at     TEXT NOT NULL,          -- when it was attached to memory
    meta            TEXT DEFAULT '{}'       -- JSON freeform metadata
);

-- Many-to-many: one source can back multiple LTM entries and vice-versa
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

-- Associations --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS associations (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    relation    TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    timestamp   TEXT NOT NULL
);

-- External Archive ----------------------------------------------------------
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
CREATE INDEX IF NOT EXISTS idx_sources_type    ON sources(type);
CREATE INDEX IF NOT EXISTS idx_ltm_sources_ltm ON ltm_sources(ltm_entry_id);
CREATE INDEX IF NOT EXISTS idx_ltm_sources_src ON ltm_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_stm_ts        ON stm_segments(timestamp);
CREATE INDEX IF NOT EXISTS idx_ltm_conf      ON ltm_entries(confidence);
CREATE INDEX IF NOT EXISTS idx_ltm_ts        ON ltm_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_ltm_type      ON ltm_entries(class_type);
CREATE INDEX IF NOT EXISTS idx_entity_name   ON entities(name);
CREATE INDEX IF NOT EXISTS idx_concept_op    ON concepts(operator, subject);
CREATE INDEX IF NOT EXISTS idx_assoc_src     ON associations(source_id);
CREATE INDEX IF NOT EXISTS idx_assoc_tgt     ON associations(target_id);
CREATE INDEX IF NOT EXISTS idx_archive_orig  ON archive(original_id);
"""


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
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = str(Path(db_path).resolve())
        self._init()

    def _init(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
