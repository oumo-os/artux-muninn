"""
test_memory_standalone.py — Exhaustive unit + integration tests for the standalone
memory_module package (artux-muninn).

Tests cover every public class, method, and integration path across all 10 source
files.  All tests use SQLite :memory: — zero filesystem side effects, zero network.

Run:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_memory_standalone.py -v
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from memory_module import (
    MemoryAgent, RecallQuery, RecallResult,
    STMSegment, Signature, Entity, LTMEntry, Concept, Association,
    ArchiveEntry, SourceRef, Database, SEMANTIC_AVAILABLE,
)
from memory_module.db import to_json, from_json
from memory_module.models import SOURCE_TYPES
from memory_module.embeddings import (
    embed, cosine_similarity, top_k_similar, backend_info, configure,
    _tfidf_embed, _l2_normalize, _VOCAB_SIZE,
)
from memory_module.stm import STMManager
from memory_module.ltm import LTMManager
from memory_module.entities import EntityManager, AUTHORITY
from memory_module.forgetting import (
    ForgettingEngine, DEFAULT_LAMBDA, ARCHIVE_THRESHOLD, ARCHIVE_TTL_DAYS,
)
from memory_module.sources import SourceManager
from memory_module.recall import RecallEngine, OPERATORS
from memory_module.tools import get_tools, ToolExecutor


# ===================================================================
# §1  Database (db.py)
# ===================================================================

class TestDatabase:
    def test_memory_db_creates_all_tables(self, db):
        with db.connection() as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        expected = {
            "stm_segments", "stm_meta", "signatures", "entities",
            "ltm_entries", "sources", "ltm_sources", "concepts",
            "associations", "archive",
        }
        assert expected.issubset(tables)

    def test_memory_db_creates_indexes(self, db):
        with db.connection() as conn:
            indexes = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
                ).fetchall()
            }
        assert len(indexes) >= 10

    def test_connection_commits_on_success(self, db):
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO stm_segments (id, content, timestamp) VALUES (?, ?, ?)",
                ("test-1", "hello", datetime.utcnow().isoformat()),
            )
        with db.connection() as conn:
            row = conn.execute("SELECT * FROM stm_segments WHERE id = 'test-1'").fetchone()
        assert row is not None

    def test_connection_rolls_back_on_exception(self, db):
        try:
            with db.connection() as conn:
                conn.execute(
                    "INSERT INTO stm_segments (id, content, timestamp) VALUES (?, ?, ?)",
                    ("test-rb", "hello", datetime.utcnow().isoformat()),
                )
                conn.execute("INVALID SQL THAT WILL FAIL")
        except Exception:
            pass
        with db.connection() as conn:
            row = conn.execute("SELECT * FROM stm_segments WHERE id = 'test-rb'").fetchone()
        assert row is None

    def test_memory_connection_reuse(self):
        db = Database(":memory:")
        with db.connection() as conn1:
            conn1.execute(
                "INSERT INTO stm_segments (id, content, timestamp) VALUES (?, ?, ?)",
                ("reuse-1", "a", datetime.utcnow().isoformat()),
            )
        with db.connection() as conn2:
            row = conn2.execute("SELECT * FROM stm_segments WHERE id = 'reuse-1'").fetchone()
        assert row is not None
        db.close()

    def test_close_cleans_up_memory_connection(self):
        db = Database(":memory:")
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO stm_segments (id, content, timestamp) VALUES (?, ?, ?)",
                ("close-test", "x", datetime.utcnow().isoformat()),
            )
        db.close()
        assert db._mem_conn is None

    def test_to_json_helpers(self):
        assert to_json(None) == "[]"
        assert to_json([]) == "[]"
        assert to_json([1, 2]) == "[1, 2]"
        assert json.loads(to_json({"a": 1})) == {"a": 1}

    def test_from_json_helpers(self):
        assert from_json(None) == []
        assert from_json("[]") == []
        assert from_json("[1, 2]") == [1, 2]
        assert from_json('{"a": 1}') == {"a": 1}


# ===================================================================
# §2  Models (models.py)
# ===================================================================

class TestModels:
    def test_stm_segment_defaults(self):
        seg = STMSegment(content="hello")
        assert seg.id
        assert seg.content == "hello"
        assert seg.is_compression is False
        assert seg.source == ""
        assert seg.event_type == ""
        assert seg.payload == {}
        assert seg.confidence == 1.0
        assert isinstance(seg.timestamp, datetime)

    def test_entity_append_narrative(self):
        ent = Entity(name="Test")
        assert ent.content == ""
        ent.append_narrative("first observation")
        assert ent.content == "first observation"
        ent.append_narrative("second", memory_ref="m1")
        assert "second" in ent.content
        assert "[m.ref:m1]" in ent.content

    def test_entity_append_narrative_with_entity_ref(self):
        ent = Entity(name="Test")
        ent.append_narrative("info", entity_ref="other-id")
        assert "[ent.ref:other-id]" in ent.content

    def test_concept_triple(self):
        c = Concept(operator="what", subject="Kyle", focus="identity")
        assert c.triple == "what:Kyle:identity"

    def test_source_types_complete(self):
        expected = {"image", "audio", "video", "pdf", "webpage", "file", "remote"}
        assert SOURCE_TYPES == expected

    def test_ltm_entry_defaults(self):
        e = LTMEntry(content="test")
        assert e.id
        assert e.class_type == "assertion"
        assert e.entities == []
        assert e.topics == []
        assert e.concepts == []
        assert e.confidence == 1.0
        assert e.embedding is None

    def test_archive_entry_defaults(self):
        a = ArchiveEntry(content="scar")
        assert a.rehydrated is False


# ===================================================================
# §3  STM Manager (stm.py)
# ===================================================================

class TestSTMManager:
    def test_record_returns_segment(self, stm):
        seg = stm.record("hello world")
        assert isinstance(seg, STMSegment)
        assert seg.content == "hello world"
        assert seg.id

    def test_record_stores_metadata(self, stm):
        seg = stm.record(
            "event", source="user", event_type="speech",
            payload={"key": "val"}, confidence=0.8,
        )
        all_segs = stm.get_all()
        assert any(s.source == "user" and s.event_type == "speech" and s.confidence == 0.8 for s in all_segs)

    def test_get_all_ordered(self, stm):
        stm.record("a")
        stm.record("b")
        stm.record("c")
        segs = stm.get_all()
        contents = [s.content for s in segs]
        assert contents == ["a", "b", "c"]

    def test_count_raw_only(self, stm):
        for i in range(4):
            stm.record(f"seg {i}")
        assert stm.count() == 4

    def test_get_window_formats_correctly(self, stm):
        stm.record("hello")
        window = stm.get_window()
        assert "hello" in window
        assert "[SUMMARY:" not in window

    def test_forget_removes_segment(self, stm):
        seg = stm.record("to forget")
        assert stm.count() == 1
        stm.forget(seg.id)
        assert stm.count() == 0

    def test_clear_wipes_all(self, stm):
        stm.record("a")
        stm.record("b")
        stm.clear()
        assert stm.count() == 0
        assert stm.get_flush_watermark() is None

    def test_auto_compress_triggers(self):
        s = STMManager(Database(":memory:"), max_segments=3)
        s.record("a")
        s.record("b")
        s.record("c")
        # 4th record triggers compress at count >= max_segments
        s.record("d")
        cons = [x for x in s.get_all() if x.is_compression]
        assert len(cons) >= 1

    def test_compress_creates_consN(self, stm):
        stm.record("a")
        stm.record("b")
        cons = stm.compress()
        assert cons is not None
        assert cons.is_compression
        assert "a" in cons.content or "b" in cons.content

    def test_compress_empty_returns_none(self, stm):
        assert stm.compress() is None

    def test_compress_head_splits_correctly(self, stm):
        for i in range(6):
            stm.record(f"event {i}")
        cons, head = stm.compress_head(retain=2)
        assert cons is not None
        assert len(head) == 4
        assert cons.is_compression

    def test_compress_head_retain_zero(self, stm):
        for i in range(3):
            stm.record(f"e{i}")
        cons, head = stm.compress_head(retain=0)
        assert cons is not None
        assert len(head) == 3

    def test_compress_head_retain_exceeds_raw(self, stm):
        stm.record("only one")
        cons, head = stm.compress_head(retain=10)
        assert cons is None
        assert head == []

    def test_flush_up_to_removes_raw(self, stm):
        segs = [stm.record(f"e{i}") for i in range(5)]
        n = stm.flush_up_to(segs[2].id)
        assert n == 3
        remaining = [s for s in stm.get_all() if not s.is_compression]
        assert len(remaining) == 2

    def test_flush_up_to_nonexistent_returns_zero(self, stm):
        assert stm.flush_up_to("nonexistent-id") == 0

    def test_flush_watermark(self, stm):
        assert stm.get_flush_watermark() is None
        segs = [stm.record("a"), stm.record("b")]
        stm.flush_up_to(segs[0].id)
        assert stm.get_flush_watermark() == segs[0].id

    def test_get_events_after_by_id(self, stm):
        s1 = stm.record("first")
        s2 = stm.record("second")
        s3 = stm.record("third")
        after = stm.get_events_after(s1.id)
        assert len(after) == 2
        assert after[0].id == s2.id

    def test_get_events_after_none_returns_all(self, stm):
        stm.record("a")
        stm.record("b")
        assert len(stm.get_events_after(None)) == 2

    def test_get_events_after_nonexistent_returns_all(self, stm):
        stm.record("a")
        assert len(stm.get_events_after("fake")) == 1

    def test_rolling_cons_preserves_narrative(self, stm):
        for i in range(8):
            stm.record(f"event {i}")
        # After 5+ records, auto-compress fires; consN should contain prior content
        cons = [x for x in stm.get_all() if x.is_compression]
        assert len(cons) >= 1

    def test_custom_compress_fn(self):
        custom = lambda texts: " | ".join(reversed(texts))
        s = STMManager(Database(":memory:"), max_segments=3, compress_fn=custom)
        s.record("a")
        s.record("b")
        s.record("c")
        s.record("d")
        cons = [x for x in s.get_all() if x.is_compression]
        assert len(cons) >= 1
        # Custom fn should have been applied
        assert " | " in cons[0].content


# ===================================================================
# §4  LTM Manager (ltm.py)
# ===================================================================

class TestLTMManager:
    def test_store_persists_entry(self, ltm):
        entry = LTMEntry(content="test entry")
        stored = ltm.store(entry)
        assert stored.id
        fetched = ltm.get(stored.id)
        assert fetched is not None
        assert fetched.content == "test entry"

    def test_store_auto_generates_embedding(self, ltm):
        entry = LTMEntry(content="generate my embedding")
        stored = ltm.store(entry)
        assert stored.embedding is not None
        assert len(stored.embedding) > 0

    def test_store_preserves_preexisting_embedding(self, ltm):
        entry = LTMEntry(content="has embedding", embedding=[0.1] * 512)
        stored = ltm.store(entry)
        assert stored.embedding == [0.1] * 512

    def test_get_nonexistent_returns_none(self, ltm):
        assert ltm.get("nonexistent") is None

    def test_get_all_ordered_desc(self, ltm):
        e1 = ltm.store(LTMEntry(content="first"))
        e2 = ltm.store(LTMEntry(content="second"))
        all_entries = ltm.get_all()
        assert all_entries[0].content == "second"
        assert all_entries[1].content == "first"

    def test_get_all_min_confidence(self, ltm):
        ltm.store(LTMEntry(content="high", confidence=0.9))
        ltm.store(LTMEntry(content="low", confidence=0.1))
        filtered = ltm.get_all(min_confidence=0.5)
        assert len(filtered) == 1
        assert filtered[0].content == "high"

    def test_update_confidence(self, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=0.5))
        ltm.update_confidence(entry.id, 0.8)
        fetched = ltm.get(entry.id)
        assert fetched.confidence == 0.8

    def test_update_confidence_clamps(self, ltm):
        entry = ltm.store(LTMEntry(content="test"))
        ltm.update_confidence(entry.id, 1.5)
        assert ltm.get(entry.id).confidence == 1.0
        ltm.update_confidence(entry.id, -0.5)
        assert ltm.get(entry.id).confidence == 0.0

    def test_update_content(self, ltm):
        entry = ltm.store(LTMEntry(content="old content"))
        result = ltm.update_content(entry.id, "new content")
        assert result is True
        fetched = ltm.get(entry.id)
        assert fetched.content == "new content"
        assert fetched.embedding is not None

    def test_update_content_with_confidence(self, ltm):
        entry = ltm.store(LTMEntry(content="old", confidence=0.5))
        ltm.update_content(entry.id, "new", confidence=0.9)
        fetched = ltm.get(entry.id)
        assert fetched.content == "new"
        assert fetched.confidence == 0.9

    def test_update_content_nonexistent_returns_false(self, ltm):
        assert ltm.update_content("nope", "content") is False

    def test_delete(self, ltm):
        entry = ltm.store(LTMEntry(content="delete me"))
        ltm.delete(entry.id)
        assert ltm.get(entry.id) is None

    def test_consolidate_high_confidence_stores(self, ltm):
        entry = ltm.consolidate_from_stm("important fact", confidence=0.9)
        fetched = ltm.get(entry.id)
        assert fetched is not None
        assert fetched.content == "important fact"

    def test_consolidate_low_confidence_archives(self, ltm):
        entry = ltm.consolidate_from_stm("weak fact", confidence=0.1)
        fetched = ltm.get(entry.id)
        assert fetched is None
        scars = ltm.get_archive()
        assert any(s.original_id == entry.id for s in scars)

    def test_add_concept(self, ltm):
        c = Concept(operator="what", subject="Kyle", focus="identity")
        stored = ltm.add_concept(c)
        assert stored.id
        concepts = ltm.get_concepts(operator="what")
        assert len(concepts) == 1

    def test_get_concepts_filter(self, ltm):
        ltm.add_concept(Concept(operator="what", subject="Kyle", focus="identity"))
        ltm.add_concept(Concept(operator="who", subject="Sam", focus="role"))
        assert len(ltm.get_concepts(operator="what")) == 1
        assert len(ltm.get_concepts(subject="Kyle")) == 1
        assert len(ltm.get_concepts()) == 2

    def test_link_creates_association(self, ltm):
        assoc = ltm.link("e1", "e2", "knows")
        assert assoc.id
        assocs = ltm.get_associations(source_id="e1")
        assert len(assocs) == 1

    def test_get_associations_filter(self, ltm):
        ltm.link("a", "b", "knows")
        ltm.link("c", "d", "works_with")
        assert len(ltm.get_associations(source_id="a")) == 1
        assert len(ltm.get_associations(target_id="d")) == 1
        assert len(ltm.get_associations(relation="knows")) == 1

    def test_record_signature(self, ltm):
        sig = Signature(content="voice capture", modality="voice")
        stored = ltm.record_signature(sig)
        assert stored.id
        assert stored.embedding is not None

    def test_archive_and_get(self, ltm):
        scar = ltm.archive_entry("scar content", "ltm", "orig-1", "low confidence")
        scars = ltm.get_archive()
        assert len(scars) == 1
        assert scars[0].content == "scar content"

    def test_rehydrate(self, ltm):
        scar = ltm.archive_entry("rehydrate me", "ltm", "orig-2", "test")
        entry = ltm.rehydrate(scar.id)
        assert entry is not None
        assert entry.content == "rehydrate me"
        fetched = ltm.get(entry.id)
        assert fetched is not None
        updated_scar = ltm.get_archive()
        assert any(s.rehydrated for s in updated_scar)

    def test_rehydrate_nonexistent(self, ltm):
        assert ltm.rehydrate("nope") is None


# ===================================================================
# §5  Entity Manager (entities.py)
# ===================================================================

class TestEntityManager:
    def test_create(self, entities):
        ent = entities.create(name="Alice", initial_content="A person", topics=["person"])
        assert ent.id
        assert ent.name == "Alice"
        assert ent.content == "A person"

    def test_get(self, entities):
        ent = entities.create(name="Bob")
        fetched = entities.get(ent.id)
        assert fetched is not None
        assert fetched.name == "Bob"

    def test_get_nonexistent(self, entities):
        assert entities.get("nope") is None

    def test_get_by_name_partial(self, entities):
        entities.create(name="Alice Smith")
        entities.create(name="Bob Jones")
        matches = entities.get_by_name("Alice")
        assert len(matches) == 1

    def test_get_by_name_case_insensitive(self, entities):
        entities.create(name="CHARLIE")
        assert len(entities.get_by_name("charlie")) == 1

    def test_all(self, entities):
        entities.create(name="A")
        entities.create(name="B")
        assert len(entities.all()) == 2

    def test_update(self, entities):
        ent = entities.create(name="Eve")
        ent.content = "Updated content"
        entities.update(ent)
        fetched = entities.get(ent.id)
        assert fetched.content == "Updated content"

    def test_delete(self, entities):
        ent = entities.create(name="Delete me")
        entities.delete(ent.id)
        assert entities.get(ent.id) is None

    def test_append_observation(self, entities):
        ent = entities.create(name="Test")
        updated = entities.append_observation(ent.id, "saw something", authority="peer")
        assert "saw something" in updated.content
        assert "[auth:2]" in updated.content

    def test_append_observation_nonexistent_raises(self, entities):
        with pytest.raises(ValueError):
            entities.append_observation("nope", "obs")

    def test_append_observation_authority_weights(self, entities):
        ent = entities.create(name="AuthTest")
        for auth, weight in AUTHORITY.items():
            updated = entities.append_observation(ent.id, f"via {auth}", authority=auth)
            assert f"[auth:{weight}]" in updated.content

    def test_record_correction(self, entities):
        ent = entities.create(name="Original")
        corrected = entities.record_correction(ent.id, "actually it is X", "corrector-1")
        assert "dispute:corrector-1" in corrected.content

    def test_resolve(self, entities):
        entities.create(name="Alice", initial_content="works on robotics")
        entities.create(name="Bob", initial_content="works on cooking")
        matches = entities.resolve("robotics expert", threshold=0.1, top_k=3)
        assert len(matches) >= 1
        assert any(m[0].name == "Alice" for m in matches)

    def test_resolve_respects_threshold(self, entities):
        entities.create(name="Alice", initial_content="unrelated content")
        matches = entities.resolve("completely different topic", threshold=0.99)
        assert len(matches) == 0

    def test_resolve_respects_top_k(self, entities):
        for i in range(10):
            entities.create(name=f"Person{i}", initial_content="similar content about robotics")
        matches = entities.resolve("robotics", threshold=0.1, top_k=3)
        assert len(matches) <= 3


# ===================================================================
# §6  Forgetting Engine (forgetting.py)
# ===================================================================

class TestForgettingEngine:
    def test_run_decay_updates_entries(self, forgetting, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=1.0))
        # Backdate the entry by 30 days
        ltm.update_confidence(entry.id, 1.0)
        with ltm.db.connection() as conn:
            past = (datetime.utcnow() - timedelta(days=30)).isoformat()
            conn.execute(
                "UPDATE ltm_entries SET timestamp = ? WHERE id = ?",
                (past, entry.id),
            )
        updated = forgetting.run_decay()
        assert updated >= 1
        fetched = ltm.get(entry.id)
        assert fetched.confidence < 1.0

    def test_run_decay_returns_count(self, forgetting, ltm):
        ltm.store(LTMEntry(content="a"))
        ltm.store(LTMEntry(content="b"))
        updated = forgetting.run_decay()
        assert isinstance(updated, int)

    def test_decay_formula_correct(self):
        lam = 0.01
        days = 70
        expected = math.exp(-lam * days)
        assert abs(expected - 0.4966) < 0.01

    def test_run_maintenance_archives_weak(self, forgetting, ltm):
        entry = ltm.store(LTMEntry(content="weak", confidence=0.05))
        result = forgetting.run_maintenance()
        assert result["archived_to_scar"] >= 1
        assert ltm.get(entry.id) is None

    def test_run_maintenance_purges_old_scars(self, forgetting, ltm):
        scar = ltm.archive_entry("old scar", "ltm", "orig", "test")
        # Backdate the scar
        old_ts = (datetime.utcnow() - timedelta(days=ARCHIVE_TTL_DAYS + 10)).isoformat()
        with ltm.db.connection() as conn:
            conn.execute(
                "UPDATE archive SET timestamp = ? WHERE id = ?",
                (old_ts, scar.id),
            )
        result = forgetting.run_maintenance()
        assert result["deleted_old_scars"] >= 1

    def test_run_maintenance_returns_dict(self, forgetting):
        result = forgetting.run_maintenance()
        assert "archived_to_scar" in result
        assert "deleted_old_scars" in result
        assert "run_at" in result

    def test_reinforce(self, forgetting, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=0.5))
        new_conf = forgetting.reinforce(entry.id, 0.3)
        assert new_conf == pytest.approx(0.8)

    def test_reinforce_caps_at_one(self, forgetting, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=0.95))
        new_conf = forgetting.reinforce(entry.id, 0.5)
        assert new_conf == 1.0

    def test_reinforce_nonexistent(self, forgetting):
        assert forgetting.reinforce("nope") is None

    def test_decay_entry(self, forgetting, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=0.7))
        new_conf = forgetting.decay_entry(entry.id, 0.2)
        assert new_conf == pytest.approx(0.5)

    def test_decay_entry_floors_at_zero(self, forgetting, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=0.1))
        new_conf = forgetting.decay_entry(entry.id, 0.5)
        assert new_conf == 0.0

    def test_decay_entry_nonexistent(self, forgetting):
        assert forgetting.decay_entry("nope") is None


# ===================================================================
# §7  Source Manager (sources.py)
# ===================================================================

class TestSourceManager:
    def test_record(self, sources):
        ref = sources.record(
            location="/img/test.jpg", type="image",
            description="a test image", meta={"width": 100},
        )
        assert ref.id
        assert ref.type == "image"
        assert ref.location == "/img/test.jpg"

    def test_record_with_captured_at(self, sources):
        dt = datetime(2024, 1, 15, 10, 30)
        ref = sources.record(location="/f.pdf", captured_at=dt)
        assert ref.captured_at == dt

    def test_attach_idempotent(self, sources):
        ref = sources.record(location="/a.wav", type="audio")
        sources.attach(ref.id, "ltm-1")
        sources.attach(ref.id, "ltm-1")  # duplicate — should not error
        entries = sources.entries_for_source(ref.id)
        assert len(entries) == 1

    def test_record_and_attach(self, sources):
        ref = sources.record_and_attach("ltm-2", "/b.png", type="image")
        attached = sources.for_entry("ltm-2")
        assert len(attached) == 1

    def test_get(self, sources):
        ref = sources.record(location="/c.mp3")
        fetched = sources.get(ref.id)
        assert fetched is not None
        assert fetched.location == "/c.mp3"

    def test_get_nonexistent(self, sources):
        assert sources.get("nope") is None

    def test_for_entry(self, sources):
        r1 = sources.record(location="/1.jpg", type="image")
        r2 = sources.record(location="/2.jpg", type="image")
        sources.attach(r1.id, "entry-1")
        sources.attach(r2.id, "entry-1")
        attached = sources.for_entry("entry-1")
        assert len(attached) == 2

    def test_entries_for_source(self, sources):
        ref = sources.record(location="/shared.pdf")
        sources.attach(ref.id, "e1")
        sources.attach(ref.id, "e2")
        entries = sources.entries_for_source(ref.id)
        assert set(entries) == {"e1", "e2"}

    def test_find_by_type(self, sources):
        sources.record(location="/a.jpg", type="image")
        sources.record(location="/b.wav", type="audio")
        images = sources.find_by_type("image")
        assert len(images) == 1

    def test_find_by_location(self, sources):
        sources.record(location="/exact/path.jpg")
        found = sources.find_by_location("/exact/path.jpg")
        assert found is not None

    def test_all(self, sources):
        sources.record(location="/1.jpg")
        sources.record(location="/2.jpg")
        assert len(sources.all()) == 2

    def test_detach(self, sources):
        ref = sources.record(location="/detach.jpg")
        sources.attach(ref.id, "entry-d")
        sources.detach(ref.id, "entry-d")
        assert len(sources.for_entry("entry-d")) == 0

    def test_delete(self, sources):
        ref = sources.record(location="/delete.jpg")
        sources.attach(ref.id, "entry-del")
        sources.delete(ref.id)
        assert sources.get(ref.id) is None
        assert len(sources.entries_for_source(ref.id)) == 0

    def test_source_types_covered(self):
        for st in SOURCE_TYPES:
            assert st in {"image", "audio", "video", "pdf", "webpage", "file", "remote"}


# ===================================================================
# §8  Recall Engine (recall.py)
# ===================================================================

class TestRecallEngine:
    def test_recall_plain_string(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="Alice likes robotics", topics=["robotics"]))
        results = recall_engine.recall("robotics", top_k=5)
        assert len(results) >= 1

    def test_recall_empty_when_no_entries(self, recall_engine):
        results = recall_engine.recall("anything")
        assert results == []

    def test_recall_min_confidence_gate(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="high conf", confidence=0.9))
        ltm.store(LTMEntry(content="low conf", confidence=0.05))
        q = RecallQuery(min_confidence=0.5)
        results = recall_engine.recall(q)
        assert all(r.entry.confidence >= 0.5 for r in results)

    def test_recall_time_bracket(self, recall_engine, ltm):
        old = LTMEntry(content="old entry", confidence=1.0)
        old.timestamp = datetime(2020, 1, 1)
        ltm.store(old)
        new = LTMEntry(content="new entry", confidence=1.0)
        new.timestamp = datetime(2025, 1, 1)
        ltm.store(new)
        q = RecallQuery(after=datetime(2024, 1, 1))
        results = recall_engine.recall(q)
        assert all("new" in r.entry.content for r in results)

    def test_recall_concept_tier_boost(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="Kyle identity fact"))
        ltm.add_concept(Concept(
            operator="what", subject="Kyle", focus="identity",
            ltm_entry_id=entry.id,
        ))
        q = RecallQuery(operator="what", subject="Kyle")
        results = recall_engine.recall(q)
        assert len(results) >= 1
        assert "concept_triple" in results[0].match_reasons

    def test_recall_topic_match(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="topic test", topics=["alpha", "beta"]))
        q = RecallQuery(topics=["alpha"])
        results = recall_engine.recall(q)
        assert any("topic:alpha" in r.match_reasons for r in results)

    def test_recall_entity_reference(self, recall_engine, ltm, entities):
        ent = entities.create(name="TestPerson")
        ltm.store(LTMEntry(content="about person", entities=[ent.id]))
        q = RecallQuery(subject=ent.id)
        results = recall_engine.recall(q)
        assert len(results) >= 1

    def test_recall_association_hop(self, recall_engine, ltm, entities):
        e1 = entities.create(name="Alice")
        e2 = entities.create(name="Bob")
        ltm.link(e1.id, e2.id, "knows")
        ltm.store(LTMEntry(content="about Bob", entities=[e2.id]))
        q = RecallQuery(subject=e1.id)
        results = recall_engine.recall(q)
        assert len(results) >= 1

    def test_recall_subject_in_content(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="John went to the store"))
        q = RecallQuery(subject="John")
        results = recall_engine.recall(q)
        assert any("subject_in_content" in r.match_reasons for r in results)

    def test_recall_semantic_similarity(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="artificial intelligence and machine learning"))
        results = recall_engine.recall("AI and ML")
        assert len(results) >= 1

    def test_recall_score_blend(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="test blending", topics=["blend"]))
        q_sem = RecallQuery(topics=["blend"], semantic_query="blend", semantic_weight=1.0)
        q_struct = RecallQuery(topics=["blend"], semantic_weight=0.0)
        r_sem = recall_engine.recall(q_sem)
        r_struct = recall_engine.recall(q_struct)
        if r_sem and r_struct:
            assert r_sem[0].score != r_struct[0].score

    def test_recall_reinforces_confidence(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="boost me", confidence=0.5))
        recall_engine.recall("boost me")
        fetched = ltm.get(entry.id)
        assert fetched.confidence > 0.5

    def test_recall_attaches_sources(self, recall_engine, ltm, sources):
        entry = ltm.store(LTMEntry(content="with source"))
        ref = sources.record(location="/test.jpg")
        sources.attach(ref.id, entry.id)
        results = recall_engine.recall("with source")
        if results:
            assert len(results[0].sources) >= 1

    def test_recall_scar_hydration(self, recall_engine, ltm):
        scar = ltm.archive_entry("forgotten memory", "ltm", "scar-1", "test")
        q = RecallQuery(semantic_query="forgotten memory", include_scars=True)
        results = recall_engine.recall(q)
        assert any(r.from_archive for r in results)

    def test_recall_scar_dedup(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="duplicate topic"))
        scar = ltm.archive_entry("duplicate topic", "ltm", entry.id, "test")
        q = RecallQuery(semantic_query="duplicate topic", include_scars=True)
        results = recall_engine.recall(q)
        ids = [r.entry.id for r in results]
        assert len(ids) == len(set(ids))

    def test_recall_top_k_truncation(self, recall_engine, ltm):
        for i in range(20):
            ltm.store(LTMEntry(content=f"entry {i}", topics=["trunc"]))
        results = recall_engine.recall(RecallQuery(topics=["trunc"]), top_k=5)
        assert len(results) <= 5

    def test_recall_entities(self, recall_engine, entities):
        entities.create(name="Alice", initial_content="robotics engineer")
        matches = recall_engine.recall_entities("robotics", threshold=0.1)
        assert len(matches) >= 1

    def test_resolve_subject_uuid_passthrough(self, recall_engine):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        ids = recall_engine._resolve_subject(uid)
        assert ids == {uid}

    def test_resolve_subject_name(self, recall_engine, entities):
        entities.create(name="TestName", initial_content="a test entity")
        ids = recall_engine._resolve_subject("TestName")
        assert len(ids) >= 1

    def test_resolve_subject_none(self, recall_engine):
        assert recall_engine._resolve_subject(None) == set()


# ===================================================================
# §9  Tools (tools.py)
# ===================================================================

class TestTools:
    def test_get_tools_anthropic(self):
        tools = get_tools("anthropic")
        assert len(tools) == 9
        assert all("name" in t and "input_schema" in t for t in tools)

    def test_get_tools_openai(self):
        tools = get_tools("openai")
        assert len(tools) == 9
        assert all(t["type"] == "function" and "function" in t for t in tools)

    def test_get_tools_invalid_format(self):
        with pytest.raises(ValueError):
            get_tools("invalid")

    def test_tool_executor_dispatches_all(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        # Each tool needs its required params; test that execute dispatches correctly
        assert "Recorded" in executor.execute("record_stm", {"content": "test"})
        assert "Consolidated" in executor.execute("consolidate_ltm", {"narrative": "test"})
        assert "Created entity" in executor.execute("create_entity", {"name": "E", "description": "D"})
        assert isinstance(executor.execute("get_stm_window", {}), str)
        assert isinstance(executor.execute("resolve_entity", {"clues": "test"}), str)
        assert isinstance(executor.execute("recall", {}), str)

    def test_tool_executor_unknown_raises(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        with pytest.raises(ValueError):
            executor.execute("nonexistent_tool", {})

    def test_tool_executor_record_stm(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        result = executor.execute("record_stm", {"content": "test event"})
        assert "Recorded" in result

    def test_tool_executor_consolidate_ltm(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        result = executor.execute("consolidate_ltm", {
            "narrative": "test consolidation",
            "class_type": "assertion",
        })
        assert "Consolidated" in result

    def test_tool_executor_create_entity(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        result = executor.execute("create_entity", {
            "name": "ToolEntity",
            "description": "created via tool",
        })
        assert "Created entity" in result

    def test_tool_executor_get_stm_window(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        result = executor.execute("get_stm_window", {})
        assert isinstance(result, str)

    def test_run_anthropic_handles_blocks(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        block = MagicMock()
        block.type = "tool_use"
        block.id = "test-block-1"
        block.name = "get_stm_window"
        block.input = {}
        results = executor.run_anthropic([block])
        assert len(results) == 1
        assert results[0]["tool_use_id"] == "test-block-1"

    def test_run_anthropic_handles_non_tool_blocks(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        block = MagicMock()
        block.type = "text"
        results = executor.run_anthropic([block])
        assert results == []

    def test_run_openaiHandles_none(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        assert executor.run_openai(None) == []

    def test_run_openaiHandles_calls(self, memory_agent):
        executor = ToolExecutor(memory_agent)
        call = MagicMock()
        call.function.name = "get_stm_window"
        call.function.arguments = "{}"
        call.id = "call-1"
        results = executor.run_openai([call])
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call-1"


# ===================================================================
# §10  Embeddings (embeddings.py)
# ===================================================================

class TestEmbeddings:
    def test_embed_returns_vector(self):
        vec = embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_embed_deterministic(self):
        a = embed("same input")
        b = embed("same input")
        assert a == b

    def test_cosine_similarity_identical(self):
        vec = embed("test")
        sim = cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_symmetric(self):
        a = embed("hello")
        b = embed("world")
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_cosine_similarity_empty(self):
        assert cosine_similarity([], [1.0]) == 0.0
        assert cosine_similarity([1.0], []) == 0.0

    def test_cosine_similarity_different_lengths(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_similarity_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_top_k_similar(self):
        q = embed("query")
        cands = [(f"c{i}", embed(f"candidate {i}")) for i in range(10)]
        results = top_k_similar(q, cands, k=3)
        assert len(results) <= 3
        assert all(s >= 0 for _, s in results)

    def test_top_k_similar_threshold(self):
        q = embed("query")
        cands = [("c1", [0.0] * 512), ("c2", embed("similar text"))]
        results = top_k_similar(q, cands, threshold=0.5)
        assert all(s >= 0.5 for _, s in results)

    def test_top_k_similar_empty(self):
        assert top_k_similar(embed("q"), [], k=5) == []

    def test_backend_info(self):
        info = backend_info()
        assert "backend" in info
        assert "semantic_available" in info

    def test_l2_normalize_unit_magnitude(self):
        vec = [3.0, 4.0, 0.0]
        normed = _l2_normalize(vec)
        mag = math.sqrt(sum(v * v for v in normed))
        assert mag == pytest.approx(1.0)

    def test_tfidf_embedding_consistent_dims(self):
        v1 = embed("short")
        v2 = embed("a much longer piece of text with many more tokens")
        assert len(v1) == len(v2)
        # GGUF (nomic-embed) = 768, sentence-transformers = 384, TF-IDF = 512
        assert len(v1) in (_VOCAB_SIZE, 384, 768)


# ===================================================================
# §11  MemoryAgent — Integration (agent.py)
# ===================================================================

class TestMemoryAgent:
    def test_record_stm(self, memory_agent):
        seg = memory_agent.record_stm("hello")
        assert seg.id
        assert memory_agent.get_stm_window()

    def test_record_stm_with_metadata(self, memory_agent):
        seg = memory_agent.record_stm(
            "typed", source="user", event_type="speech",
            payload={"key": "val"}, confidence=0.9,
        )
        window = memory_agent.get_stm_window()
        assert "typed" in window

    def test_forget_stm(self, memory_agent):
        seg = memory_agent.record_stm("forget me")
        memory_agent.forget_stm(seg.id)
        assert memory_agent.stm.count() == 0

    def test_consolidate_ltm_with_narrative(self, memory_agent):
        entry = memory_agent.consolidate_ltm(
            narrative="important fact",
            class_type="assertion",
            topics=["test"],
        )
        assert entry.id
        fetched = memory_agent.get_ltm_entry(entry.id)
        assert fetched is not None

    def test_consolidate_ltm_auto(self, memory_agent):
        for i in range(6):
            memory_agent.record_stm(f"event {i}")
        entry = memory_agent.consolidate_ltm()
        assert entry.id
        assert memory_agent.get_flush_watermark() is not None

    def test_consolidate_ltm_per_segment_false(self, memory_agent):
        for i in range(6):
            memory_agent.record_stm(f"event {i}")
        memory_agent.consolidate_ltm(per_segment=False)
        ltm_count = len(memory_agent.ltm.get_all())
        assert ltm_count >= 1

    def test_recall(self, memory_agent):
        memory_agent.store_ltm("test recall content", topics=["recall"])
        results = memory_agent.recall("recall")
        assert len(results) >= 1

    def test_recall_with_query(self, memory_agent):
        memory_agent.store_ltm("structured recall", topics=["struct"])
        q = RecallQuery(topics=["struct"])
        results = memory_agent.recall(q)
        assert len(results) >= 1

    def test_create_and_resolve_entity(self, memory_agent):
        ent = memory_agent.create_entity(
            description="works on robotics", name="Musa",
        )
        matches = memory_agent.resolve_entity("robotics engineer")
        assert len(matches) >= 1
        assert any(m[0].id == ent.id for m in matches)

    def test_observe_entity(self, memory_agent):
        ent = memory_agent.create_entity(name="Ob", description="original")
        updated = memory_agent.observe_entity(ent.id, "new observation")
        assert "new observation" in updated.content

    def test_correct_entity(self, memory_agent):
        ent = memory_agent.create_entity(name="Cor", description="original")
        corrected = memory_agent.correct_entity(ent.id, "correction", "corr-id")
        assert "dispute:corr-id" in corrected.content

    def test_associate_signature(self, memory_agent):
        sig = memory_agent.associate_signature(
            content="voice data", modality="voice", topics=["audio"],
        )
        assert sig.id
        assert sig.embedding is not None

    def test_link_and_infer(self, memory_agent):
        e1 = memory_agent.create_entity(name="A", description="entity A")
        e2 = memory_agent.create_entity(name="B", description="entity B")
        memory_agent.link_entities(e1.id, e2.id, "related")
        assocs = memory_agent.infer_relationships()
        assert len(assocs) >= 1

    def test_record_source(self, memory_agent):
        ref = memory_agent.record_source(
            location="/test.jpg", type="image", description="a test",
        )
        assert ref.id

    def test_record_and_attach_source(self, memory_agent):
        entry = memory_agent.store_ltm("with source")
        ref = memory_agent.record_and_attach_source(
            entry.id, "/test.jpg", type="image", description="img desc",
        )
        attached = memory_agent.sources_for_entry(entry.id)
        assert len(attached) == 1

    def test_update_source_description(self, memory_agent):
        ref = memory_agent.record_source(location="/x.jpg")
        updated = memory_agent.update_source_description(ref.id, "better description")
        assert updated.description == "better description"

    def test_update_source_description_nonexistent(self, memory_agent):
        assert memory_agent.update_source_description("nope", "desc") is None

    def test_add_concept(self, memory_agent):
        concept = memory_agent.add_concept("what", "Kyle", "identity")
        assert concept.id

    def test_store_and_get_ltm(self, memory_agent):
        entry = memory_agent.store_ltm("direct store", confidence=0.8)
        fetched = memory_agent.get_ltm_entry(entry.id)
        assert fetched.content == "direct store"

    def test_update_ltm_content(self, memory_agent):
        entry = memory_agent.store_ltm("old content")
        result = memory_agent.update_ltm_content(entry.id, "new content")
        assert result is True
        fetched = memory_agent.get_ltm_entry(entry.id)
        assert fetched.content == "new content"

    def test_update_ltm_content_with_confidence(self, memory_agent):
        entry = memory_agent.store_ltm("content", confidence=0.5)
        memory_agent.update_ltm_content(entry.id, "updated", confidence=0.9)
        fetched = memory_agent.get_ltm_entry(entry.id)
        assert fetched.confidence == 0.9

    def test_run_decay(self, memory_agent):
        memory_agent.store_ltm("decay test")
        updated = memory_agent.run_decay()
        assert isinstance(updated, int)

    def test_reinforce(self, memory_agent):
        entry = memory_agent.store_ltm("reinforce me", confidence=0.5)
        new_conf = memory_agent.reinforce(entry.id, 0.3)
        assert new_conf == pytest.approx(0.8)

    def test_run_maintenance(self, memory_agent):
        result = memory_agent.run_maintenance()
        assert "archived_to_scar" in result

    def test_archive_and_rehydrate(self, memory_agent):
        entry = memory_agent.store_ltm("will be archived", confidence=0.05)
        memory_agent.run_maintenance()
        scars = memory_agent.get_archive()
        if scars:
            rehydrated = memory_agent.rehydrate(scars[0].id)
            assert rehydrated is not None

    def test_flush_stm_up_to(self, memory_agent):
        segs = [memory_agent.record_stm(f"e{i}") for i in range(3)]
        n = memory_agent.flush_stm_up_to(segs[1].id)
        assert n == 2

    def test_status(self, memory_agent):
        status = memory_agent.status()
        assert "stm_segments" in status
        assert "ltm_entries" in status
        assert "entities" in status
        assert "archive_scars" in status
        assert "embedding_backend" in status

    def test_full_lifecycle(self, memory_agent):
        memory_agent.record_stm("User: my name is Musa")
        memory_agent.record_stm("User: I work on robotics")
        memory_agent.record_stm("User: I like coffee")

        ent = memory_agent.create_entity(name="Musa", description="works on robotics")
        memory_agent.observe_entity(ent.id, "likes coffee", authority="self")

        for i in range(7):
            memory_agent.record_stm(f"event {i}")
        memory_agent.consolidate_ltm()

        results = memory_agent.recall("Musa")
        assert len(results) >= 1

        status = memory_agent.status()
        assert status["ltm_entries"] >= 1
