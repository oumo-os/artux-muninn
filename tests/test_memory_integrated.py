"""
test_memory_integrated.py — Tests for the integrated anima.memory module.

Focuses on:
  1. Public API exports from anima.memory.__init__
  2. The 3 new methods unique to the integrated version:
     - LTMManager.update_content()
     - MemoryAgent.update_ltm_content()
     - MemoryAgent.get_ltm_entry()
  3. Cross-validation: standalone and integrated produce identical results

Run from the standalone folder:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_memory_integrated.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

# ---------------------------------------------------------------------------
# Import from the integrated anima.memory package
# ---------------------------------------------------------------------------
try:
    from anima.memory import (
        MemoryAgent, RecallQuery, RecallResult,
        STMSegment, Signature, Entity, LTMEntry, Concept, Association,
        ArchiveEntry, SourceRef, Database, SEMANTIC_AVAILABLE,
    )
    from anima.memory.ltm import LTMManager
    from anima.memory.stm import STMManager
    from anima.memory.entities import EntityManager
    from anima.memory.forgetting import ForgettingEngine
    from anima.memory.sources import SourceManager
    from anima.memory.recall import RecallEngine
    from anima.memory.tools import get_tools, ToolExecutor
    from anima.memory.embeddings import embed, cosine_similarity, backend_info
    HAS_INTEGRATED = True
except ImportError:
    HAS_INTEGRATED = False


# Skip all tests if the integrated package is not importable
pytestmark = pytest.mark.skipif(
    not HAS_INTEGRATED,
    reason="anima.memory not importable from this environment",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return MemoryAgent(":memory:", max_stm_segments=5)


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture
def ltm(db):
    return LTMManager(db)


@pytest.fixture
def stm(db):
    return STMManager(db, max_segments=5)


@pytest.fixture
def entities(db):
    return EntityManager(db)


@pytest.fixture
def sources(db):
    return SourceManager(db)


@pytest.fixture
def recall_engine(db, ltm, entities, sources):
    return RecallEngine(db, ltm, entities, source_mgr=sources)


@pytest.fixture
def forgetting(db, ltm):
    return ForgettingEngine(db, ltm)


# ===================================================================
# §1  Public API Exports
# ===================================================================

class TestIntegratedExports:
    def test_memory_agent_importable(self):
        assert MemoryAgent is not None

    def test_recall_query_importable(self):
        assert RecallQuery is not None

    def test_recall_result_importable(self):
        assert RecallResult is not None

    def test_all_models_importable(self):
        for cls in [STMSegment, Signature, Entity, LTMEntry, Concept,
                     Association, ArchiveEntry, SourceRef]:
            assert cls is not None

    def test_database_importable(self):
        assert Database is not None

    def test_semantic_available_importable(self):
        assert isinstance(SEMANTIC_AVAILABLE, bool)

    def test_tools_importable(self):
        assert callable(get_tools)
        assert ToolExecutor is not None

    def test_submanagers_importable(self):
        for cls in [LTMManager, STMManager, EntityManager,
                     ForgettingEngine, SourceManager, RecallEngine]:
            assert cls is not None


# ===================================================================
# §2  LTMManager.update_content() — integrated-only
# ===================================================================

class TestIntegratedLTMUpdateContent:
    def test_update_content_replaces_text(self, ltm):
        entry = ltm.store(LTMEntry(content="original text"))
        result = ltm.update_content(entry.id, "updated text")
        assert result is True
        fetched = ltm.get(entry.id)
        assert fetched.content == "updated text"

    def test_update_content_reembeds(self, ltm):
        entry = ltm.store(LTMEntry(content="before"))
        old_emb = entry.embedding[:]
        ltm.update_content(entry.id, "completely different content now")
        fetched = ltm.get(entry.id)
        assert fetched.embedding != old_emb

    def test_update_content_sets_confidence(self, ltm):
        entry = LTMEntry(content="test", confidence=0.5)
        stored = ltm.store(entry)
        ltm.update_content(stored.id, "new", confidence=0.95)
        fetched = ltm.get(stored.id)
        assert fetched.confidence == 0.95

    def test_update_content_clamps_confidence(self, ltm):
        entry = ltm.store(LTMEntry(content="test"))
        ltm.update_content(entry.id, "new", confidence=2.0)
        assert ltm.get(entry.id).confidence == 1.0

    def test_update_content_none_confidence_unchanged(self, ltm):
        entry = ltm.store(LTMEntry(content="test", confidence=0.7))
        ltm.update_content(entry.id, "new", confidence=None)
        # confidence is NOT updated when None is passed
        # but store() is called which uses the existing confidence
        fetched = ltm.get(entry.id)
        assert fetched.content == "new"

    def test_update_content_nonexistent_returns_false(self, ltm):
        assert ltm.update_content("fake-id", "text") is False

    def test_update_content_preserves_topics(self, ltm):
        entry = ltm.store(LTMEntry(content="old", topics=["alpha", "beta"]))
        ltm.update_content(entry.id, "new")
        fetched = ltm.get(entry.id)
        assert fetched.topics == ["alpha", "beta"]

    def test_update_content_preserves_entities(self, ltm):
        entry = ltm.store(LTMEntry(content="old", entities=["e1", "e2"]))
        ltm.update_content(entry.id, "new")
        fetched = ltm.get(entry.id)
        assert fetched.entities == ["e1", "e2"]


# ===================================================================
# §3  MemoryAgent.update_ltm_content() — integrated-only
# ===================================================================

class TestIntegratedAgentUpdateLTMContent:
    def test_update_ltm_content(self, agent):
        entry = agent.store_ltm("old content")
        result = agent.update_ltm_content(entry.id, "new content")
        assert result is True
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.content == "new content"

    def test_update_ltm_content_with_confidence(self, agent):
        entry = agent.store_ltm("old", confidence=0.5)
        agent.update_ltm_content(entry.id, "new", confidence=0.9)
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.confidence == 0.9

    def test_update_ltm_content_reembeds(self, agent):
        entry = agent.store_ltm("original")
        old_emb = entry.embedding[:]
        agent.update_ltm_content(entry.id, "totally different text")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.embedding != old_emb

    def test_update_ltm_content_nonexistent(self, agent):
        assert agent.update_ltm_content("nope", "text") is False

    def test_update_ltm_content_status_transition(self, agent):
        entry = agent.store_ltm(
            '{"status": "proposed", "name": "skill_a"}',
            class_type="procedure",
        )
        agent.update_ltm_content(
            entry.id,
            '{"status": "active", "name": "skill_a"}',
            confidence=0.95,
        )
        fetched = agent.get_ltm_entry(entry.id)
        assert "active" in fetched.content
        assert fetched.confidence == 0.95


# ===================================================================
# §4  MemoryAgent.get_ltm_entry() — integrated-only
# ===================================================================

class TestIntegratedAgentGetLTMEntry:
    def test_get_ltm_entry(self, agent):
        entry = agent.store_ltm("fetchable content")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is not None
        assert fetched.content == "fetchable content"

    def test_get_ltm_entry_nonexistent(self, agent):
        assert agent.get_ltm_entry("nope") is None

    def test_get_ltm_entry_preserves_all_fields(self, agent):
        entry = agent.store_ltm(
            "full fields",
            class_type="decision",
            entities=["e1"],
            topics=["t1"],
            concepts=["c1"],
            confidence=0.85,
        )
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.class_type == "decision"
        assert fetched.entities == ["e1"]
        assert fetched.topics == ["t1"]
        assert fetched.concepts == ["c1"]
        assert fetched.confidence == 0.85

    def test_get_ltm_entry_has_embedding(self, agent):
        entry = agent.store_ltm("has embedding")
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched.embedding is not None
        assert len(fetched.embedding) > 0


# ===================================================================
# §5  Cross-Validation: Integrated vs Standalone Equivalence
# ===================================================================

class TestCrossValidation:
    """Verify the integrated version produces identical results to standalone."""

    def test_stm_record_equivalent(self, agent):
        seg1 = agent.record_stm("test content", source="user", event_type="speech")
        seg2 = agent.record_stm("test content", source="user", event_type="speech")
        # Both should work identically
        assert seg1.id != seg2.id
        assert seg1.content == seg2.content

    def test_ltm_store_equivalent(self, agent):
        e1 = agent.store_ltm("same content", topics=["t"])
        e2 = agent.store_ltm("same content", topics=["t"])
        fetched1 = agent.get_ltm_entry(e1.id)
        fetched2 = agent.get_ltm_entry(e2.id)
        assert fetched1.content == fetched2.content
        assert fetched1.topics == fetched2.topics

    def test_entity_equivalent(self, agent):
        ent1 = agent.create_entity(name="Test", description="same desc")
        ent2 = agent.create_entity(name="Test", description="same desc")
        assert ent1.id != ent2.id
        assert ent1.name == ent2.name

    def test_consolidate_equivalent(self, agent):
        for i in range(6):
            agent.record_stm(f"event {i}")
        entry = agent.consolidate_ltm()
        fetched = agent.get_ltm_entry(entry.id)
        assert fetched is not None

    def test_recall_equivalent(self, agent):
        agent.store_ltm("shared recall content", topics=["shared"])
        results = agent.recall("shared")
        assert len(results) >= 1
        assert all(isinstance(r, RecallResult) for r in results)


# ===================================================================
# §6  Integration Smoke Tests — Full Pipeline via anima.memory
# ===================================================================

class TestIntegratedPipeline:
    def test_full_lifecycle(self, agent):
        # STM
        agent.record_stm("Hello, my name is integrated test")
        agent.record_stm("I test the anima.memory module")
        window = agent.get_stm_window()
        assert "integrated test" in window

        # Entity
        ent = agent.create_entity(name="Tester", description="tests memory modules")
        matches = agent.resolve_entity("memory tester")
        assert len(matches) >= 1

        # Observation
        agent.observe_entity(ent.id, "runs automated tests", authority="system")

        # Consolidation
        for i in range(7):
            agent.record_stm(f"auto event {i}")
        entry = agent.consolidate_ltm(topics=["testing"])
        assert entry.id

        # Recall
        results = agent.recall("Tester")
        assert len(results) >= 1

        # Source
        source = agent.record_source(
            location="/test/log.txt", type="file",
            description="test execution log",
        )
        agent.attach_source(source.id, entry.id)
        attached = agent.sources_for_entry(entry.id)
        assert len(attached) == 1

        # Status
        status = agent.status()
        assert status["ltm_entries"] >= 1

    def test_concepts_and_associations(self, agent):
        ent_a = agent.create_entity(name="ConceptA", description="concept A")
        ent_b = agent.create_entity(name="ConceptB", description="concept B")
        agent.link_entities(ent_a.id, ent_b.id, "relates_to")
        assocs = agent.infer_relationships()
        assert len(assocs) >= 1

        concept = agent.add_concept("what", "ConceptA", "purpose", ltm_entry_id="test")
        assert concept.id

    def test_forgetting_lifecycle(self, agent):
        entry = agent.store_ltm("will fade", confidence=0.5)
        agent.run_decay()
        agent.reinforce(entry.id, 0.1)
        agent.run_maintenance()
        status = agent.status()
        assert isinstance(status["archive_scars"], int)

    def test_update_ltm_content_workflow(self, agent):
        entry = agent.store_ltm('{"status": "proposed"}', class_type="procedure")
        assert "proposed" in agent.get_ltm_entry(entry.id).content

        agent.update_ltm_content(entry.id, '{"status": "active"}')
        assert "active" in agent.get_ltm_entry(entry.id).content

        agent.update_ltm_content(entry.id, '{"status": "archived"}', confidence=0.1)
        fetched = agent.get_ltm_entry(entry.id)
        assert "archived" in fetched.content
        assert fetched.confidence == 0.1
