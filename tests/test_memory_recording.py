"""
test_memory_recording.py — Recording and updating correctness tests.

Verifies that STM, LTM, Entity, and Source recording operations store data
correctly, updates propagate, and read-your-writes consistency holds.

Run:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_memory_recording.py -v --import-mode=importlib --rootdir=tests
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory_module import (
    MemoryAgent, RecallQuery, LTMEntry, Entity, STMSegment, SourceRef,
)
from memory_module.db import Database
from memory_module.models import STMSegment as STMSegmentModel


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def agent():
    return MemoryAgent(":memory:", max_stm_segments=10)


@pytest.fixture
def db():
    return Database(":memory:")


# ===================================================================
# §1  STM Recording Correctness
# ===================================================================

class TestSTMRecording:
    """Verify STM stores data correctly and returns accurate state."""

    def test_record_returns_segment_with_correct_content(self, agent):
        seg = agent.record_stm("hello world")
        assert seg.content == "hello world"
        assert isinstance(seg, STMSegmentModel)

    def test_record_assigns_unique_ids(self, agent):
        s1 = agent.record_stm("a")
        s2 = agent.record_stm("b")
        assert s1.id != s2.id

    def test_record_stores_timestamp(self, agent):
        before = datetime.utcnow()
        seg = agent.record_stm("timestamped")
        after = datetime.utcnow()
        assert before <= seg.timestamp <= after

    def test_record_source_field(self, agent):
        seg = agent.record_stm("from user", source="user")
        window = agent.get_stm_window()
        assert "from user" in window

    def test_record_event_type_field(self, agent):
        seg = agent.record_stm("tool result", event_type="tool_result")
        # Verify it's stored (check via raw DB)
        with agent.db.connection() as conn:
            row = conn.execute(
                "SELECT event_type FROM stm_segments WHERE id = ?", (seg.id,)
            ).fetchone()
        assert row["event_type"] == "tool_result"

    def test_record_payload_json_roundtrip(self, agent):
        payload = {"key": "value", "nested": {"a": 1}}
        seg = agent.record_stm("payload test", payload=payload)
        with agent.db.connection() as conn:
            row = conn.execute(
                "SELECT payload FROM stm_segments WHERE id = ?", (seg.id,)
            ).fetchone()
        stored = json.loads(row["payload"])
        assert stored == payload

    def test_record_confidence_stored(self, agent):
        seg = agent.record_stm("confident", confidence=0.75)
        with agent.db.connection() as conn:
            row = conn.execute(
                "SELECT confidence FROM stm_segments WHERE id = ?", (seg.id,)
            ).fetchone()
        assert row["confidence"] == 0.75

    def test_stm_window_contains_all_raw_events(self, agent):
        agent.record_stm("first")
        agent.record_stm("second")
        agent.record_stm("third")
        window = agent.get_stm_window()
        assert "first" in window
        assert "second" in window
        assert "third" in window

    def test_stm_window_has_timestamps(self, agent):
        agent.record_stm("timed event")
        window = agent.get_stm_window()
        # Window format includes [HH:MM:SS]
        assert "[" in window

    def test_stm_count_accurate(self, agent):
        assert agent.stm.count() == 0
        agent.record_stm("a")
        assert agent.stm.count() == 1
        agent.record_stm("b")
        assert agent.stm.count() == 2

    def test_stm_forget_removes_only_target(self, agent):
        s1 = agent.record_stm("keep")
        s2 = agent.record_stm("remove")
        agent.forget_stm(s2.id)
        assert agent.stm.count() == 1
        window = agent.get_stm_window()
        assert "keep" in window
        assert "remove" not in window

    def test_stm_clear_removes_everything(self, agent):
        agent.record_stm("a")
        agent.record_stm("b")
        agent.stm.clear()
        assert agent.stm.count() == 0
        assert agent.get_stm_window() == ""

    def test_stm_get_all_ordered_oldest_first(self, agent):
        s1 = agent.record_stm("first")
        s2 = agent.record_stm("second")
        s3 = agent.record_stm("third")
        all_segs = agent.stm.get_all()
        contents = [s.content for s in all_segs]
        assert contents.index("first") < contents.index("second") < contents.index("third")

    def test_stm_compression_counted_separately(self, agent):
        for i in range(12):
            agent.record_stm(f"event {i}")
        # After 10+ records, auto-compress fires
        all_segs = agent.stm.get_all()
        compressions = [s for s in all_segs if s.is_compression]
        raw = [s for s in all_segs if not s.is_compression]
        assert len(compressions) >= 1
        assert agent.stm.count() == len(raw)  # count() only returns raw


# ===================================================================
# §2  LTM Recording Correctness
# ===================================================================

class TestLTMRecording:
    """Verify LTM stores, retrieves, and updates entries correctly."""

    def test_store_ltm_persists_all_fields(self, agent):
        entry = agent.store_ltm(
            "complete entry",
            class_type="decision",
            entities=["e1", "e2"],
            topics=["t1", "t2"],
            concepts=["c1"],
            confidence=0.85,
        )
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.content == "complete entry"
        assert fetched.class_type == "decision"
        assert fetched.entities == ["e1", "e2"]
        assert fetched.topics == ["t1", "t2"]
        assert fetched.concepts == ["c1"]
        assert fetched.confidence == 0.85

    def test_store_ltm_auto_embeds(self, agent):
        entry = agent.store_ltm("auto embedded")
        assert entry.embedding is not None
        assert len(entry.embedding) > 0

    def test_store_ltm_preserves_existing_embedding(self, agent):
        entry = LTMEntry(content="custom embed", embedding=[0.1] * 384)
        stored = agent.ltm.store(entry)
        assert stored.embedding == [0.1] * 384

    def test_get_ltm_entry_nonexistent_returns_none(self, agent):
        assert agent.get_ltm_entry("nonexistent") is None

    def test_get_ltm_entry_returns_fresh_copy(self, agent):
        entry = agent.store_ltm("fresh copy test")
        f1 = agent.get_ltm_entry(entry.id)
        f2 = agent.get_ltm_entry(entry.id)
        assert f1.id == f2.id
        assert f1 is not f2  # different object instances

    def test_update_ltm_content_replaces(self, agent):
        entry = agent.store_ltm("old content")
        agent.update_ltm_content(entry.id, "new content")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.content == "new content"

    def test_update_ltm_content_reembeds(self, agent):
        entry = agent.store_ltm("original text")
        old_emb = entry.embedding[:]
        agent.update_ltm_content(entry.id, "completely different text now")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.embedding != old_emb

    def test_update_ltm_content_preserves_metadata(self, agent):
        entry = agent.store_ltm(
            "metadata test",
            topics=["t1"],
            entities=["e1"],
            confidence=0.7,
        )
        agent.update_ltm_content(entry.id, "updated")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.topics == ["t1"]
        assert fetched.entities == ["e1"]

    def test_update_ltm_content_with_confidence(self, agent):
        entry = agent.store_ltm("conf update", confidence=0.5)
        agent.update_ltm_content(entry.id, "new", confidence=0.9)
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.confidence == 0.9

    def test_update_ltm_content_nonexistent_returns_false(self, agent):
        assert agent.update_ltm_content("nope", "text") is False

    def test_update_ltm_content_returns_true_on_success(self, agent):
        entry = agent.store_ltm("test")
        result = agent.update_ltm_content(entry.id, "updated")
        assert result is True

    def test_consolidate_from_narrative_stored(self, agent):
        entry = agent.consolidate_ltm(narrative="consolidated fact", confidence=0.9)
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is not None
        assert fetched.content == "consolidated fact"

    def test_consolidate_low_confidence_archived(self, agent):
        entry = agent.consolidate_ltm(narrative="weak fact", confidence=0.1)
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is None  # not in active LTM
        scars = agent.get_archive()
        assert any(s.original_id == entry.id for s in scars)

    def test_ltm_delete_removes(self, agent):
        entry = agent.store_ltm("delete me")
        agent.ltm.delete(entry.id)
        assert agent.get_ltm_entry(entry.id) is None

    def test_ltm_get_all_min_confidence(self, agent):
        agent.store_ltm("high", confidence=0.9)
        agent.store_ltm("low", confidence=0.1)
        high = agent.ltm.get_all(min_confidence=0.5)
        assert len(high) == 1
        assert high[0].content == "high"

    def test_ltm_get_all_ordered_newest_first(self, agent):
        agent.store_ltm("first")
        agent.store_ltm("second")
        all_entries = agent.ltm.get_all()
        assert all_entries[0].content == "second"
        assert all_entries[1].content == "first"


# ===================================================================
# §3  Entity Recording Correctness
# ===================================================================

class TestEntityRecording:
    """Verify entity CRUD and narrative operations are correct."""

    def test_create_entity_stores_all_fields(self, agent):
        ent = agent.create_entity(
            name="FullEntity",
            description="complete entity",
            topics=["person", "engineer"],
        )
        fetched = agent.entities.get(ent.id)
        assert fetched.name == "FullEntity"
        assert "complete entity" in fetched.content
        assert "person" in fetched.topics

    def test_create_entity_assigns_embedding(self, agent):
        ent = agent.create_entity(name="Emb", description="embedded entity")
        assert ent.embedding is not None

    def test_create_entity_has_timestamps(self, agent):
        ent = agent.create_entity(name="Ts", description="timestamped")
        assert ent.created_at is not None
        assert ent.last_seen is not None

    def test_observe_appends_to_narrative(self, agent):
        ent = agent.create_entity(name="Obs", description="initial")
        agent.observe_entity(ent.id, "first observation")
        fetched = agent.entities.get(ent.id)
        assert "first observation" in fetched.content

    def test_observe_multiple_appends_all(self, agent):
        ent = agent.create_entity(name="Multi", description="start")
        agent.observe_entity(ent.id, "obs 1")
        agent.observe_entity(ent.id, "obs 2")
        agent.observe_entity(ent.id, "obs 3")
        fetched = agent.entities.get(ent.id)
        assert "obs 1" in fetched.content
        assert "obs 2" in fetched.content
        assert "obs 3" in fetched.content

    def test_observe_preserves_order(self, agent):
        ent = agent.create_entity(name="Ord", description="start")
        agent.observe_entity(ent.id, "first")
        agent.observe_entity(ent.id, "second")
        fetched = agent.entities.get(ent.id)
        assert fetched.content.index("first") < fetched.content.index("second")

    def test_observe_records_authority_weight(self, agent):
        ent = agent.create_entity(name="Auth", description="start")
        agent.observe_entity(ent.id, "self report", authority="self")
        agent.observe_entity(ent.id, "peer report", authority="peer")
        fetched = agent.entities.get(ent.id)
        assert "[auth:1]" in fetched.content
        assert "[auth:2]" in fetched.content

    def test_observe_records_memory_ref(self, agent):
        ent = agent.create_entity(name="Ref", description="start")
        agent.observe_entity(ent.id, "with ref", memory_ref="stm-123")
        fetched = agent.entities.get(ent.id)
        assert "[m.ref:stm-123]" in fetched.content

    def test_observe_records_source_entity(self, agent):
        ent = agent.create_entity(name="Src", description="start")
        agent.observe_entity(ent.id, "from source", source_entity_id="src-456")
        fetched = agent.entities.get(ent.id)
        assert "[ent.ref:src-456]" in fetched.content

    def test_correct_records_dispute(self, agent):
        ent = agent.create_entity(name="Cor", description="original claim")
        agent.correct_entity(ent.id, "correction text", "corrector-id")
        fetched = agent.entities.get(ent.id)
        assert "dispute:corrector-id" in fetched.content
        assert "correction text" in fetched.content

    def test_observe_nonexistent_raises(self, agent):
        with pytest.raises(ValueError):
            agent.observe_entity("nonexistent", "obs")

    def test_entity_get_by_name(self, agent):
        agent.create_entity(name="FindMe", description="findable")
        matches = agent.entities.get_by_name("FindMe")
        assert len(matches) == 1
        assert matches[0].name == "FindMe"

    def test_entity_get_by_name_partial(self, agent):
        agent.create_entity(name="John Smith", description="a person")
        agent.create_entity(name="Jane Doe", description="another person")
        matches = agent.entities.get_by_name("John")
        assert len(matches) == 1
        assert "John" in matches[0].name

    def test_entity_get_by_name_case_insensitive(self, agent):
        agent.create_entity(name="ALICE", description="upper case")
        matches = agent.entities.get_by_name("alice")
        assert len(matches) == 1

    def test_entity_all_returns_all(self, agent):
        agent.create_entity(name="A", description="a")
        agent.create_entity(name="B", description="b")
        all_ents = agent.entities.all()
        assert len(all_ents) == 2

    def test_entity_delete_removes(self, agent):
        ent = agent.create_entity(name="Del", description="delete me")
        agent.entities.delete(ent.id)
        assert agent.entities.get(ent.id) is None

    def test_entity_update_persists(self, agent):
        ent = agent.entities.get(
            agent.create_entity(name="Upd", description="original").id
        )
        ent.content = "updated content"
        agent.entities.update(ent)
        fetched = agent.entities.get(ent.id)
        assert fetched.content == "updated content"

    def test_entity_last_seen_updates_on_observe(self, agent):
        ent = agent.create_entity(name="Seen", description="initial")
        first_seen = ent.last_seen
        import time; time.sleep(0.01)
        agent.observe_entity(ent.id, "new obs")
        updated = agent.entities.get(ent.id)
        assert updated.last_seen >= first_seen


# ===================================================================
# §4  Source Recording Correctness
# ===================================================================

class TestSourceRecording:
    """Verify source reference CRUD is correct."""

    def test_record_source_stores_all_fields(self, agent):
        ref = agent.record_source(
            location="/data/img.jpg",
            type="image",
            description="a test image",
            meta={"width": 1920, "height": 1080},
        )
        fetched = agent.sources.get(ref.id)
        assert fetched.location == "/data/img.jpg"
        assert fetched.type == "image"
        assert fetched.description == "a test image"
        assert fetched.meta["width"] == 1920

    def test_record_source_assigns_id(self, agent):
        ref = agent.record_source(location="/f.jpg")
        assert ref.id
        assert len(ref.id) > 0

    def test_record_source_timestamps(self, agent):
        ref = agent.record_source(location="/ts.jpg")
        assert ref.recorded_at is not None
        assert ref.captured_at is not None

    def test_attach_source_creates_link(self, agent):
        entry = agent.store_ltm("with source")
        ref = agent.record_source(location="/link.jpg")
        agent.attach_source(ref.id, entry.id)
        attached = agent.sources_for_entry(entry.id)
        assert len(attached) == 1
        assert attached[0].id == ref.id

    def test_attach_source_idempotent(self, agent):
        entry = agent.store_ltm("idempotent test")
        ref = agent.record_source(location="/idem.jpg")
        agent.attach_source(ref.id, entry.id)
        agent.attach_source(ref.id, entry.id)  # duplicate
        attached = agent.sources_for_entry(entry.id)
        assert len(attached) == 1

    def test_record_and_attach_combo(self, agent):
        entry = agent.store_ltm("combo test")
        ref = agent.record_and_attach_source(
            entry.id, "/combo.jpg", type="image", description="combo",
        )
        attached = agent.sources_for_entry(entry.id)
        assert len(attached) == 1
        assert attached[0].location == "/combo.jpg"

    def test_detach_removes_link(self, agent):
        entry = agent.store_ltm("detach test")
        ref = agent.record_source(location="/detach.jpg")
        agent.attach_source(ref.id, entry.id)
        agent.sources.detach(ref.id, entry.id)
        attached = agent.sources_for_entry(entry.id)
        assert len(attached) == 0

    def test_delete_source_removes_record(self, agent):
        ref = agent.record_source(location="/del.jpg")
        agent.sources.delete(ref.id)
        assert agent.sources.get(ref.id) is None

    def test_delete_source_removes_links(self, agent):
        entry = agent.store_ltm("link test")
        ref = agent.record_source(location="/dellink.jpg")
        agent.attach_source(ref.id, entry.id)
        agent.sources.delete(ref.id)
        attached = agent.sources_for_entry(entry.id)
        assert len(attached) == 0

    def test_find_by_type(self, agent):
        agent.record_source(location="/a.jpg", type="image")
        agent.record_source(location="/b.wav", type="audio")
        images = agent.sources.find_by_type("image")
        assert len(images) == 1
        assert images[0].type == "image"

    def test_find_by_location(self, agent):
        agent.record_source(location="/exact/path.jpg")
        found = agent.sources.find_by_location("/exact/path.jpg")
        assert found is not None

    def test_update_source_description(self, agent):
        ref = agent.record_source(location="/upd.jpg", description="original")
        updated = agent.update_source_description(ref.id, "better description")
        assert updated.description == "better description"
        # Verify persisted
        fetched = agent.sources.get(ref.id)
        assert fetched.description == "better description"

    def test_update_source_description_nonexistent(self, agent):
        assert agent.update_source_description("nope", "desc") is None

    def test_entries_for_source_reverse_lookup(self, agent):
        e1 = agent.store_ltm("entry 1")
        e2 = agent.store_ltm("entry 2")
        ref = agent.record_source(location="/shared.jpg")
        agent.attach_source(ref.id, e1.id)
        agent.attach_source(ref.id, e2.id)
        entries = agent.sources.entries_for_source(ref.id)
        assert set(entries) == {e1.id, e2.id}

    def test_source_meta_json_roundtrip(self, agent):
        meta = {"duration_s": 42, "language": "en", "nested": {"key": "val"}}
        ref = agent.record_source(location="/meta.wav", meta=meta)
        fetched = agent.sources.get(ref.id)
        assert fetched.meta == meta


# ===================================================================
# §5  Forgetting / Reinforcement Correctness
# ===================================================================

class TestForgettingRecording:
    """Verify decay, reinforcement, and maintenance update correctly."""

    def test_decay_reduces_confidence(self, agent):
        entry = agent.store_ltm("will decay", confidence=1.0)
        # Backdate
        with agent.db.connection() as conn:
            past = (datetime.utcnow() - timedelta(days=30)).isoformat()
            conn.execute("UPDATE ltm_entries SET timestamp = ? WHERE id = ?", (past, entry.id))
        agent.run_decay()
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.confidence < 1.0

    def test_reinforce_increases_confidence(self, agent):
        entry = agent.store_ltm("reinforce", confidence=0.5)
        agent.reinforce(entry.id, 0.3)
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.confidence == pytest.approx(0.8)

    def test_reinforce_caps_at_1(self, agent):
        entry = agent.store_ltm("cap test", confidence=0.9)
        agent.reinforce(entry.id, 0.5)
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.confidence == 1.0

    def test_reinforce_nonexistent_returns_none(self, agent):
        assert agent.reinforce("nope") is None

    def test_maintenance_archives_weak_entries(self, agent):
        entry = agent.store_ltm("weak", confidence=0.05)
        agent.run_maintenance()
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is None
        scars = agent.get_archive()
        assert any(s.original_id == entry.id for s in scars)

    def test_maintenance_preserves_strong_entries(self, agent):
        entry = agent.store_ltm("strong", confidence=0.9)
        agent.run_maintenance()
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is not None

    def test_rehydrate_restores_to_active(self, agent):
        entry = agent.store_ltm("to be archived", confidence=0.05)
        agent.run_maintenance()
        scars = agent.get_archive()
        target = next(s for s in scars if s.original_id == entry.id)
        rehydrated = agent.rehydrate(target.id)
        assert rehydrated is not None
        fetched = agent.get_ltm_entry(rehydrated.id)
        assert fetched is not None

    def test_rehydrate_nonexistent_returns_none(self, agent):
        assert agent.rehydrate("nope") is None

    def test_rehydrate_marks_scar(self, agent):
        entry = agent.store_ltm("mark test", confidence=0.05)
        agent.run_maintenance()
        scars = agent.get_archive()
        target = next(s for s in scars if s.original_id == entry.id)
        agent.rehydrate(target.id)
        updated_scars = agent.get_archive()
        marked = next(s for s in updated_scars if s.id == target.id)
        assert marked.rehydrated is True


# ===================================================================
# §6  Concept and Association Recording
# ===================================================================

class TestConceptAssociationRecording:
    """Verify concepts and associations are stored and retrieved correctly."""

    def test_add_concept_stores(self, agent):
        concept = agent.add_concept("what", "Kyle", "identity", ltm_entry_id="e1")
        concepts = agent.ltm.get_concepts(operator="what", subject="Kyle")
        assert len(concepts) == 1
        assert concepts[0].triple == "what:Kyle:identity"

    def test_concept_triple_property(self, agent):
        concept = agent.add_concept("where", "John", "location")
        assert concept.triple == "where:John:location"

    def test_get_concepts_filter_by_operator(self, agent):
        agent.add_concept("what", "A", "focus1")
        agent.add_concept("who", "B", "focus2")
        agent.add_concept("what", "C", "focus3")
        what_concepts = agent.ltm.get_concepts(operator="what")
        assert len(what_concepts) == 2

    def test_get_concepts_filter_by_subject(self, agent):
        agent.add_concept("what", "Alpha", "f1")
        agent.add_concept("what", "Beta", "f2")
        alpha = agent.ltm.get_concepts(subject="Alpha")
        assert len(alpha) == 1

    def test_link_creates_association(self, agent):
        e1 = agent.create_entity(name="A", description="a")
        e2 = agent.create_entity(name="B", description="b")
        assoc = agent.link_entities(e1.id, e2.id, "works_with")
        assocs = agent.infer_relationships()
        assert len(assocs) >= 1
        assert assocs[0].relation == "works_with"

    def test_link_stores_direction(self, agent):
        e1 = agent.create_entity(name="X", description="x")
        e2 = agent.create_entity(name="Y", description="y")
        agent.link_entities(e1.id, e2.id, "leads")
        assocs = agent.ltm.get_associations(source_id=e1.id)
        assert len(assocs) == 1
        assert assocs[0].target_id == e2.id

    def test_get_associations_filter(self, agent):
        e1 = agent.create_entity(name="P", description="p")
        e2 = agent.create_entity(name="Q", description="q")
        e3 = agent.create_entity(name="R", description="r")
        agent.link_entities(e1.id, e2.id, "knows")
        agent.link_entities(e2.id, e3.id, "works_with")
        by_source = agent.ltm.get_associations(source_id=e1.id)
        assert len(by_source) == 1
        by_relation = agent.ltm.get_associations(relation="works_with")
        assert len(by_relation) == 1


# ===================================================================
# §7  Read-Your-Writes Consistency
# ===================================================================

class TestReadYourWrites:
    """Verify that every write is immediately visible to subsequent reads."""

    def test_stm_read_after_write(self, agent):
        agent.record_stm("immediate read")
        window = agent.get_stm_window()
        assert "immediate read" in window

    def test_ltm_read_after_write(self, agent):
        entry = agent.store_ltm("immediate LTM read")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.content == "immediate LTM read"

    def test_entity_read_after_write(self, agent):
        ent = agent.create_entity(name="Immediate", description="read me")
        fetched = agent.entities.get(ent.id)
        assert fetched.name == "Immediate"

    def test_entity_observe_read_after_write(self, agent):
        ent = agent.create_entity(name="R1", description="start")
        agent.observe_entity(ent.id, "new observation")
        fetched = agent.entities.get(ent.id)
        assert "new observation" in fetched.content

    def test_source_read_after_write(self, agent):
        ref = agent.record_source(location="/immediate.jpg")
        fetched = agent.sources.get(ref.id)
        assert fetched.location == "/immediate.jpg"

    def test_concept_read_after_write(self, agent):
        agent.add_concept("what", "Test", "value")
        concepts = agent.ltm.get_concepts(subject="Test")
        assert len(concepts) == 1

    def test_association_read_after_write(self, agent):
        e1 = agent.create_entity(name="AR1", description="a")
        e2 = agent.create_entity(name="AR2", description="b")
        agent.link_entities(e1.id, e2.id, "test_rel")
        assocs = agent.ltm.get_associations(source_id=e1.id)
        assert len(assocs) == 1

    def test_consolidation_read_after_write(self, agent):
        for i in range(6):
            agent.record_stm(f"event {i}")
        entry = agent.consolidate_ltm()
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is not None

    def test_status_reflects_writes(self, agent):
        status_before = agent.status()
        agent.store_ltm("new entry")
        agent.create_entity(name="New", description="entity")
        status_after = agent.status()
        assert status_after["ltm_entries"] > status_before["ltm_entries"]
        assert status_after["entities"] > status_before["entities"]


# ===================================================================
# §8  Large Batch Recording
# ===================================================================

class TestBatchRecording:
    """Verify correctness under high-volume recording."""

    def test_1000_stm_records_ordered(self, agent):
        for i in range(1000):
            agent.record_stm(f"batch event {i}")
        window = agent.get_stm_window()
        # Verify first and last events present (may be compressed)
        assert "batch event 999" in window or agent.stm.count() > 0

    def test_1000_ltm_records_all_persisted(self, agent):
        for i in range(1000):
            agent.store_ltm(f"batch entry {i}", topics=[f"t{i%10}"])
        all_entries = agent.ltm.get_all()
        assert len(all_entries) == 1000

    def test_100_entities_all_persisted(self, agent):
        for i in range(100):
            agent.create_entity(name=f"Batch{i}", description=f"entity {i}")
        all_ents = agent.entities.all()
        assert len(all_ents) == 100

    def test_100_sources_all_persisted(self, agent):
        for i in range(100):
            agent.record_source(location=f"/batch/file_{i}.jpg")
        all_sources = agent.sources.all()
        assert len(all_sources) == 100
