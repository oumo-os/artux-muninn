"""
models.py — Core dataclasses for the Memory Module.
All structures mirror the spec: STM segments, LTM entries,
Entities, Signatures, Concepts, Associations, and Archive entries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Short-Term Memory
# ---------------------------------------------------------------------------

@dataclass
class STMSegment:
    """A single temporal snapshot in short-term memory."""
    id: str = field(default_factory=_uid)
    content: str = ""
    timestamp: datetime = field(default_factory=_now)
    is_compression: bool = False   # True → this is a consN summary


# ---------------------------------------------------------------------------
# Perception Layer
# ---------------------------------------------------------------------------

@dataclass
class Signature:
    """Atomic perceptual capture (voice, visual, behavioral)."""
    id: str = field(default_factory=_uid)
    modality: str = "text"          # voice | visual | behavioral | text
    content: str = ""
    embedding: Optional[list[float]] = None
    timestamp: datetime = field(default_factory=_now)
    confidence: float = 1.0
    topics: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)   # resolved entities


# ---------------------------------------------------------------------------
# Entity Layer
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    A living historical ledger for a person, object, or concept.
    `content` is an append-only narrative string; contradictions are
    preserved inline with [m.ref] and [ent.ref] markers.
    """
    id: str = field(default_factory=_uid)
    name: str = ""                  # canonical / best-known label
    content: str = ""               # narrative ledger
    topics: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=_now)
    last_seen: datetime = field(default_factory=_now)

    def append_narrative(self, text: str, memory_ref: str = "",
                         entity_ref: str = "") -> None:
        """Append a sourced observation to the narrative ledger."""
        parts = []
        if entity_ref:
            parts.append(f"[ent.ref:{entity_ref}]")
        parts.append(text)
        if memory_ref:
            parts.append(f"[m.ref:{memory_ref}]")
        self.content += (" " if self.content else "") + " ".join(parts)
        self.last_seen = _now()


# ---------------------------------------------------------------------------
# Long-Term Memory
# ---------------------------------------------------------------------------

@dataclass
class LTMEntry:
    """
    Persistent memory record in the Long-Term Memory store.
    class_type: event | assertion | decision | procedure | observation
    """
    id: str = field(default_factory=_uid)
    class_type: str = "assertion"
    content: str = ""
    entities: list[str] = field(default_factory=list)      # entity IDs
    topics: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)       # "op:subj:focus"
    associations: list[str] = field(default_factory=list)   # association IDs
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=_now)
    embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Cognitive Structure
# ---------------------------------------------------------------------------

@dataclass
class Concept:
    """
    Cognitive frame: operator : subject : focus
    e.g.  what:Kyle:identity  |  dispute:Kyle:identity  |  where:John:location
    """
    id: str = field(default_factory=_uid)
    operator: str = ""      # what | who | when | how | where | why | dispute
    subject: str = ""
    focus: str = ""
    ltm_entry_id: Optional[str] = None
    entity_id: Optional[str] = None
    timestamp: datetime = field(default_factory=_now)

    @property
    def triple(self) -> str:
        return f"{self.operator}:{self.subject}:{self.focus}"


# ---------------------------------------------------------------------------
# Associations
# ---------------------------------------------------------------------------

@dataclass
class Association:
    """
    Directed, weighted relationship between two IDs (entity, LTM, concept…).
    relation: participant-in | describes | disputes | addresses | refers-to | …
    """
    id: str = field(default_factory=_uid)
    source_id: str = ""
    target_id: str = ""
    relation: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=_now)


# ---------------------------------------------------------------------------
# Source References  (external knowledge sources backing an LTM entry)
# ---------------------------------------------------------------------------

# All recognised source types.  Extend freely.
SOURCE_TYPES = {
    "image",    # local image file (jpg, png, webp, …)
    "audio",    # local audio file (mp3, wav, …)
    "video",    # local video file
    "pdf",      # local PDF document
    "webpage",  # remote URL (HTML page, article, …)
    "file",     # any other local file
    "remote",   # any other remote resource (API endpoint, S3 object, …)
}


@dataclass
class SourceRef:
    """
    A pointer to an external knowledge source that backs one or more LTM entries.

    `description` is what the agent (LLM/VLM) derived from the source at
    record time — this is the text that ends up in the LTM content summary.
    When recall returns this ref alongside an LTM entry, the calling agent
    can decide whether the description is sufficient or whether to re-examine
    the source at `location` for more detail.

    `location` is a file path OR a URL — the memory module does not care
    which.  It stores and returns it; the agent resolves it.

    Example:
        SourceRef(
            type="image",
            location="/data/perception/2024-03-01T14:22:00_living_room.jpg",
            description="Living room table. Red doll on left side. Blue cup on right.",
            captured_at=datetime(2024, 3, 1, 14, 22),
        )
    """
    id: str = field(default_factory=_uid)
    type: str = "file"                   # one of SOURCE_TYPES
    location: str = ""                   # file path or URL
    description: str = ""               # agent-derived text summary at record time
    captured_at: datetime = field(default_factory=_now)   # when the source was created
    recorded_at: datetime = field(default_factory=_now)   # when it was attached to memory
    meta: dict = field(default_factory=dict)              # freeform: duration, size, page_count, …


# ---------------------------------------------------------------------------
# External Archive
# ---------------------------------------------------------------------------

@dataclass
class ArchiveEntry:
    """
    Holds raw/low-confidence data evicted from active memory.
    Linked back to its original record via original_id.
    Can be rehydrated if later deemed relevant.
    """
    id: str = field(default_factory=_uid)
    content: str = ""
    original_type: str = ""   # stm | ltm | signature
    original_id: str = ""
    reason: str = ""          # low-confidence | size | maintenance
    timestamp: datetime = field(default_factory=_now)
    rehydrated: bool = False
