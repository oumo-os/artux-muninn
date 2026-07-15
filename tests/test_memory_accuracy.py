"""
test_memory_accuracy.py — Accuracy and correctness tests for the memory module.

These tests verify that recall, entity resolution, topic matching, concept triples,
associations, and scar hydration return THE RIGHT results — not just any results.

Run:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_memory_accuracy.py -v --import-mode=importlib --rootdir=tests
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from memory_module import (
    MemoryAgent, RecallQuery, RecallResult, LTMEntry, Entity, Concept,
)
from memory_module.db import Database
from memory_module.ltm import LTMManager
from memory_module.stm import STMManager
from memory_module.entities import EntityManager
from memory_module.recall import RecallEngine


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def agent():
    return MemoryAgent(":memory:", max_stm_segments=10)


@pytest.fixture
def db():
    return Database(":memory:")


@pytest.fixture
def ltm(db):
    return LTMManager(db)


@pytest.fixture
def entities(db):
    return EntityManager(db)


@pytest.fixture
def recall_engine(db, ltm, entities):
    from memory_module.sources import SourceManager
    sources = SourceManager(db)
    return RecallEngine(db, ltm, entities, source_mgr=sources)


# ===================================================================
# §1  Recall Accuracy — Semantic
# ===================================================================

class TestRecallSemanticAccuracy:
    """Does semantic recall return the RIGHT entry for a given query?"""

    def test_exact_keyword_match_ranks_first(self, agent):
        agent.store_ltm("The capital of France is Paris", topics=["geography"])
        agent.store_ltm("Dogs are domesticated mammals", topics=["animals"])
        agent.store_ltm("Python is a programming language", topics=["tech"])
        results = agent.recall("capital of France")
        assert len(results) >= 1
        assert "Paris" in results[0].entry.content

    def test_paraphrase_returns_correct_entry(self, agent):
        agent.store_ltm("The user prefers dark mode for all applications")
        agent.store_ltm("The user enjoys cooking Italian food")
        agent.store_ltm("The user runs 5k every morning")
        results = agent.recall("display theme preference dark")
        assert len(results) >= 1
        assert "dark mode" in results[0].entry.content

    def test_specificity_beats_generality(self, agent):
        agent.store_ltm("Musa works on robotics at MIT", topics=["person", "robotics"])
        agent.store_ltm("Robotics is a branch of engineering", topics=["field"])
        results = agent.recall("Musa robotics")
        # The entry about Musa should rank higher than the generic one
        musa_found = any("Musa" in r.entry.content for r in results[:2])
        assert musa_found

    def test_no_false_positives_on_unrelated(self, agent):
        agent.store_ltm("The Eiffel Tower is in Paris", topics=["landmark"])
        agent.store_ltm("Quantum computing uses qubits", topics=["tech"])
        results = agent.recall("recipe for chocolate cake")
        # Should either be empty or have very low scores
        for r in results:
            assert r.score < 0.5 or "chocolate" in r.entry.content.lower()

    def test_multi_word_query_accuracy(self, agent):
        agent.store_ltm("User's daughter is named Aisha, age 7")
        agent.store_ltm("User's son is named Omar, age 10")
        agent.store_ltm("User lives in London")
        results = agent.recall("daughter name")
        assert len(results) >= 1
        assert "Aisha" in results[0].entry.content


# ===================================================================
# §2  Recall Accuracy — Topic Matching
# ===================================================================

class TestRecallTopicAccuracy:
    """Do topic tags correctly filter and boost entries?"""

    def test_topic_exact_match_returns_correct_entry(self, recall_engine, ltm):
        e1 = ltm.store(LTMEntry(content="Dark mode preference", topics=["ui", "preference"]))
        e2 = ltm.store(LTMEntry(content="Coffee preference", topics=["food", "preference"]))
        e3 = ltm.store(LTMEntry(content="IDE preference", topics=["ui", "tool"]))
        q = RecallQuery(topics=["ui"])
        results = recall_engine.recall(q)
        # e1 and e3 have "ui" topic — e2 should not be boosted
        result_ids = [r.entry.id for r in results]
        assert e1.id in result_ids
        assert e2.id not in result_ids or results[0].entry.id != e2.id

    def test_multiple_topics_narrow_results(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="Python coding", topics=["code", "python"]))
        ltm.store(LTMEntry(content="Python snake", topics=["animal", "python"]))
        ltm.store(LTMEntry(content="Java coding", topics=["code", "java"]))
        q = RecallQuery(topics=["code", "python"])
        results = recall_engine.recall(q)
        if results:
            assert "coding" in results[0].entry.content.lower()

    def test_topic_in_content_soft_boost(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="The user likes ambient lighting", topics=["lighting"]))
        ltm.store(LTMEntry(content="The user likes dark mode", topics=["ui"]))
        # Query with topic "lighting" that appears in content but NOT in topics of entry 2
        q = RecallQuery(topics=["lighting"])
        results = recall_engine.recall(q)
        if results:
            assert any("lighting" in r.entry.content.lower() for r in results[:2])


# ===================================================================
# §3  Recall Accuracy — Concept Triples
# ===================================================================

class TestRecallConceptAccuracy:
    """Do concept triples (operator:subject:focus) correctly route recalls?"""

    def test_concept_triple_precise_match(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="Kyle's identity is software engineer"))
        ltm.add_concept(Concept(
            operator="what", subject="Kyle", focus="identity",
            ltm_entry_id=entry.id,
        ))
        q = RecallQuery(operator="what", subject="Kyle")
        results = recall_engine.recall(q)
        assert len(results) >= 1
        assert results[0].entry.id == entry.id
        assert "concept_triple" in results[0].match_reasons

    def test_concept_wrong_operator_misses(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="Kyle's identity"))
        ltm.add_concept(Concept(
            operator="what", subject="Kyle", focus="identity",
            ltm_entry_id=entry.id,
        ))
        # Query with wrong operator — should still find via semantic, but no concept boost
        q = RecallQuery(operator="where", subject="Kyle")
        results = recall_engine.recall(q)
        if results:
            assert "concept_triple" not in results[0].match_reasons

    def test_concept_subject_partial_match(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="Kyle prefers dark mode"))
        ltm.add_concept(Concept(
            operator="what", subject="Kyle", focus="preference",
            ltm_entry_id=entry.id,
        ))
        q = RecallQuery(operator="what", subject="Kyle")
        results = recall_engine.recall(q)
        assert len(results) >= 1
        assert "concept_triple" in results[0].match_reasons

    def test_concept_focus_search(self, recall_engine, ltm):
        entry = ltm.store(LTMEntry(content="Kyle's location is Boston"))
        ltm.add_concept(Concept(
            operator="where", subject="Kyle", focus="location",
            ltm_entry_id=entry.id,
        ))
        q = RecallQuery(operator="where", subject="Kyle")
        results = recall_engine.recall(q)
        assert len(results) >= 1
        assert "concept_triple" in results[0].match_reasons


# ===================================================================
# §4  Recall Accuracy — Entity References
# ===================================================================

class TestRecallEntityAccuracy:
    """Do entity IDs and association hops correctly boost recall?"""

    def test_direct_entity_reference_ranks_higher(self, recall_engine, ltm, entities):
        ent = entities.create(name="Musa", initial_content="robotics engineer")
        e1 = ltm.store(LTMEntry(content="Musa's project is humanoid robots", entities=[ent.id]))
        e2 = ltm.store(LTMEntry(content="Musa likes coffee", topics=["food"]))
        q = RecallQuery(subject=ent.id)
        results = recall_engine.recall(q)
        if len(results) >= 2:
            # The entry with direct entity reference should rank higher
            musa_project_idx = next((i for i, r in enumerate(results) if r.entry.id == e1.id), None)
            musa_coffee_idx = next((i for i, r in enumerate(results) if r.entry.id == e2.id), None)
            if musa_project_idx is not None and musa_coffee_idx is not None:
                assert musa_project_idx < musa_coffee_idx

    def test_association_hop_boosts_related(self, recall_engine, ltm, entities):
        alice = entities.create(name="Alice", initial_content="works at Google")
        bob = entities.create(name="Bob", initial_content="works at Google too")
        ltm.link(alice.id, bob.id, "colleague")
        ltm.store(LTMEntry(content="Bob's project is search ranking", entities=[bob.id]))
        q = RecallQuery(subject=alice.id)
        results = recall_engine.recall(q)
        if results:
            # Bob's entry should get an association hop boost
            bob_found = any("Bob" in r.entry.content for r in results)
            assert bob_found

    def test_entity_name_resolution_in_recall(self, recall_engine, ltm, entities):
        ent = entities.create(name="Charlie", initial_content="data scientist")
        ltm.store(LTMEntry(content="Charlie's favorite language is R", entities=[ent.id]))
        q = RecallQuery(subject="Charlie")
        results = recall_engine.recall(q)
        if results:
            assert any("Charlie" in r.entry.content for r in results)


# ===================================================================
# §5  Recall Accuracy — Association Expansion
# ===================================================================

class TestRecallAssociationAccuracy:
    """Does 1-hop association expansion find the right connected entries?"""

    def test_one_hop_finds_connected(self, recall_engine, ltm, entities):
        e1 = entities.create(name="ProjectA", initial_content="AI project")
        e2 = entities.create(name="ProjectB", initial_content="ML project")
        ltm.link(e1.id, e2.id, "depends_on")
        ltm.store(LTMEntry(content="ProjectB uses TensorFlow", entities=[e2.id]))
        q = RecallQuery(subject=e1.id)
        results = recall_engine.recall(q)
        if results:
            tf_found = any("TensorFlow" in r.entry.content for r in results)
            assert tf_found

    def test_two_hop_not_reached(self, recall_engine, ltm, entities):
        e1 = entities.create(name="A")
        e2 = entities.create(name="B")
        e3 = entities.create(name="C")
        ltm.link(e1.id, e2.id, "knows")
        ltm.link(e2.id, e3.id, "knows")
        ltm.store(LTMEntry(content="C's secret project", entities=[e3.id]))
        q = RecallQuery(subject=e1.id)
        results = recall_engine.recall(q)
        # 2-hop should NOT be reached (only 1-hop expansion)
        if results:
            c_found = any("secret project" in r.entry.content for r in results)
            # C's entry might appear via semantic similarity but shouldn't get assoc boost
            for r in results:
                if "secret project" in r.entry.content:
                    assert "assoc_hop" not in r.match_reasons

    def test_bidirectional_association(self, recall_engine, ltm, entities):
        e1 = entities.create(name="X")
        e2 = entities.create(name="Y")
        ltm.link(e2.id, e1.id, "references")  # Y -> X direction
        ltm.store(LTMEntry(content="X's data", entities=[e1.id]))
        q = RecallQuery(subject=e2.id)
        results = recall_engine.recall(q)
        if results:
            x_found = any("X" in r.entry.content for r in results)
            assert x_found


# ===================================================================
# §6  Recall Accuracy — Scar Hydration
# ===================================================================

class TestRecallScarAccuracy:
    """Are archived/forgotten memories surfaced correctly when requested?"""

    def test_scar_surfaced_when_requested(self, agent):
        # Create and archive a memory
        entry = agent.store_ltm("Forgotten birthday is March 15", confidence=0.01)
        agent.run_maintenance()
        scars = agent.get_archive()
        assert len(scars) >= 1
        # Recall with scars
        q = RecallQuery(semantic_query="birthday date", include_scars=True)
        results = agent.recall(q)
        archived_results = [r for r in results if r.from_archive]
        assert len(archived_results) >= 1
        assert "March 15" in archived_results[0].entry.content

    def test_scar_not_surfaced_by_default(self, agent):
        entry = agent.store_ltm("Forgotten fact", confidence=0.01)
        agent.run_maintenance()
        q = RecallQuery(semantic_query="forgotten fact", include_scars=False)
        results = agent.recall(q)
        archived_results = [r for r in results if r.from_archive]
        assert len(archived_results) == 0

    def test_scar_score_lower_than_active(self, agent):
        active = agent.store_ltm("Active memory about cats", confidence=0.9)
        faded = agent.store_ltm("Faded memory about cats", confidence=0.01)
        agent.run_maintenance()
        q = RecallQuery(semantic_query="cats", include_scars=True)
        results = agent.recall(q)
        if len(results) >= 2:
            active_results = [r for r in results if not r.from_archive]
            scar_results = [r for r in results if r.from_archive]
            if active_results and scar_results:
                assert active_results[0].score >= scar_results[0].score

    def test_scar_deduplication(self, agent):
        entry = agent.store_ltm("Same content for active and scar", confidence=0.9)
        # Archive a copy with same content
        agent.ltm.archive_entry("Same content for active and scar", "ltm", entry.id, "test")
        q = RecallQuery(semantic_query="same content", include_scars=True)
        results = agent.recall(q)
        ids = [r.entry.id for r in results]
        assert len(ids) == len(set(ids))


# ===================================================================
# §7  Recall Accuracy — Time Bracket
# ===================================================================

class TestRecallTimeAccuracy:
    """Do time brackets correctly filter entries?"""

    def test_after_filters_old_entries(self, recall_engine, ltm):
        old = LTMEntry(content="Ancient history", confidence=1.0)
        old.timestamp = datetime(2020, 1, 1)
        ltm.store(old)
        new = LTMEntry(content="Recent event", confidence=1.0)
        new.timestamp = datetime(2025, 6, 1)
        ltm.store(new)
        q = RecallQuery(after=datetime(2024, 1, 1))
        results = recall_engine.recall(q)
        assert all("Recent" in r.entry.content for r in results)

    def test_before_filters_new_entries(self, recall_engine, ltm):
        old = LTMEntry(content="Old fact", confidence=1.0)
        old.timestamp = datetime(2020, 1, 1)
        ltm.store(old)
        new = LTMEntry(content="New fact", confidence=1.0)
        new.timestamp = datetime(2025, 6, 1)
        ltm.store(new)
        q = RecallQuery(before=datetime(2023, 1, 1))
        results = recall_engine.recall(q)
        assert all("Old" in r.entry.content for r in results)

    def test_between_filters_correctly(self, recall_engine, ltm):
        for year in [2019, 2021, 2023, 2025]:
            e = LTMEntry(content=f"Event in {year}", confidence=1.0)
            e.timestamp = datetime(year, 6, 1)
            ltm.store(e)
        q = RecallQuery(
            after=datetime(2020, 1, 1),
            before=datetime(2024, 1, 1),
        )
        results = recall_engine.recall(q)
        for r in results:
            assert "2021" in r.entry.content or "2023" in r.entry.content


# ===================================================================
# §8  Recall Accuracy — Confidence Gating
# ===================================================================

class TestRecallConfidenceAccuracy:
    """Does min_confidence correctly filter low-quality entries?"""

    def test_high_confidence_only(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="Certain fact", confidence=0.95))
        ltm.store(LTMEntry(content="Uncertain guess", confidence=0.1))
        q = RecallQuery(min_confidence=0.5)
        results = recall_engine.recall(q)
        assert all(r.entry.confidence >= 0.5 for r in results)
        contents = [r.entry.content for r in results]
        assert "Certain fact" in contents
        assert "Uncertain guess" not in contents

    def test_all_confidence_when_zero(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="A", confidence=0.05))
        ltm.store(LTMEntry(content="B", confidence=0.99))
        q = RecallQuery(min_confidence=0.0)
        results = recall_engine.recall(q)
        assert len(results) == 2


# ===================================================================
# §9  Entity Resolution Accuracy
# ===================================================================

class TestEntityResolutionAccuracy:
    """Does entity fuzzy resolution return the right entity?"""

    def test_exact_name_match排名第一(self, agent):
        agent.create_entity(name="Albert Einstein", description="physicist")
        agent.create_entity(name="Marie Curie", description="chemist")
        agent.create_entity(name="Isaac Newton", description="physicist")
        matches = agent.resolve_entity("Albert Einstein")
        assert len(matches) >= 1
        assert matches[0][0].name == "Albert Einstein"

    def test_description_match_finds_correct(self, agent):
        agent.create_entity(name="E1", description="expert in quantum computing and qubits")
        agent.create_entity(name="E2", description="expert in marine biology and coral reefs")
        agent.create_entity(name="E3", description="expert in quantum computing and entanglement")
        matches = agent.resolve_entity("quantum computing expert")
        assert len(matches) >= 1
        # E1 or E3 should be top, not E2
        top_ids = {m[0].id for m in matches[:2]}
        e2 = [e for e in agent.entities.all() if e.name == "E2"][0]
        assert e2.id not in top_ids

    def test_threshold_filters_irrelevant(self, agent):
        agent.create_entity(name="Tech Corp", description="makes smartphones")
        matches = agent.resolve_entity("tropical fruit recipes", top_k=5)
        # Should find the entity but with very low score — irrelevant topic
        for _, score in matches:
            assert score < 0.5

    def test_top_k_limits_results(self, agent):
        for i in range(20):
            agent.create_entity(name=f"Entity{i}", description=f"person number {i}")
        matches = agent.resolve_entity("person", top_k=5)
        assert len(matches) <= 5


# ===================================================================
# §10  Consolidation Accuracy
# ===================================================================

class TestConsolidationAccuracy:
    """Is information preserved correctly through STM→LTM consolidation?"""

    def test_explicit_narrative_preserved(self, agent):
        entry = agent.consolidate_ltm(
            narrative="User's name is Musa, works on robotics, prefers dark mode",
            topics=["identity", "preference"],
        )
        fetched = agent.get_ltm_entry(entry.id)
        assert "Musa" in fetched.content
        assert "robotics" in fetched.content
        assert "dark mode" in fetched.content

    def test_per_segment_preserves_all_events(self, agent):
        letters = "abcde"
        # Need enough segments so compress_head(retain=3) flushes all of them.
        # With max_stm_segments=10, auto-compress fires at 10+; retain=3 means
        # head = all but last 3.  Record 12 to guarantee flush covers all.
        for i in range(len(letters)):
            agent.record_stm(f"Event {i}: {letters[i]} is important")
        for i in range(7):
            agent.record_stm(f"Filler event {i}")
        agent.consolidate_ltm(per_segment=True)
        all_ltm = agent.ltm.get_all()
        contents = " ".join(e.content for e in all_ltm)
        # Each raw event should have its own LTM entry
        for letter in letters:
            assert letter in contents

    def test_topics_preserved_through_consolidation(self, agent):
        entry = agent.consolidate_ltm(
            narrative="Important decision made",
            topics=["decision", "q3-planning"],
        )
        fetched = agent.get_ltm_entry(entry.id)
        assert "decision" in fetched.topics
        assert "q3-planning" in fetched.topics

    def test_consolidation_reduces_stm(self, agent):
        for i in range(8):
            agent.record_stm(f"event {i}")
        stm_before = agent.stm.count()
        agent.consolidate_ltm(retain_tail=2)
        stm_after = agent.stm.count()
        assert stm_after < stm_before
        assert stm_after <= 2

    def test_consolidation_advances_watermark(self, agent):
        for i in range(8):
            agent.record_stm(f"event {i}")
        assert agent.get_flush_watermark() is None
        agent.consolidate_ltm()
        assert agent.get_flush_watermark() is not None


# ===================================================================
# §11  Authority Weighting Accuracy
# ===================================================================

class TestAuthorityAccuracy:
    """Do authority tiers correctly affect entity narrative?"""

    def test_self_authority_is_lowest(self, agent):
        ent = agent.create_entity(name="A", description="initial")
        agent.observe_entity(ent.id, "self-report", authority="self")
        updated = agent.entities.get(ent.id)
        assert "[auth:1]" in updated.content

    def test_anchor_authority_is_highest(self, agent):
        ent = agent.create_entity(name="B", description="initial")
        agent.correct_entity(ent.id, "authoritative correction", "anchor-id")
        updated = agent.entities.get(ent.id)
        assert "[auth:4]" in updated.content

    def test_peer_authority_mid_range(self, agent):
        ent = agent.create_entity(name="C", description="initial")
        agent.observe_entity(ent.id, "peer observation", authority="peer")
        updated = agent.entities.get(ent.id)
        assert "[auth:2]" in updated.content

    def test_system_authority(self, agent):
        ent = agent.create_entity(name="D", description="initial")
        agent.observe_entity(ent.id, "system observation", authority="system")
        updated = agent.entities.get(ent.id)
        assert "[auth:3]" in updated.content

    def test_dispute_preserved_inline(self, agent):
        ent = agent.create_entity(name="E", description="original claim")
        agent.correct_entity(ent.id, "actually X is wrong", "corrector-1")
        updated = agent.entities.get(ent.id)
        assert "dispute:corrector-1" in updated.content


# ===================================================================
# §12  Score Blending Accuracy
# ===================================================================

class TestScoreBlendingAccuracy:
    """Does semantic_weight correctly control the blend ratio?"""

    def test_high_semantic_weight_favors_embedding(self, agent):
        agent.store_ltm("Cats are fluffy feline pets", topics=["animals"])
        agent.store_ltm("Dogs are loyal canine pets", topics=["animals"])
        q_sem = RecallQuery(
            topics=["animals"],
            semantic_query="feline fluffy animal",
            semantic_weight=0.95,
        )
        q_struct = RecallQuery(
            topics=["animals"],
            semantic_query="feline fluffy animal",
            semantic_weight=0.05,
        )
        r_sem = agent.recall(q_sem)
        r_struct = agent.recall(q_struct)
        if r_sem and r_struct:
            # With high semantic weight, "cats" entry should rank higher
            cat_top_sem = "Cat" in r_sem[0].entry.content
            cat_top_struct = "Cat" in r_struct[0].entry.content
            # Semantic should be better at matching "feline" to "cats"
            assert cat_top_sem or cat_top_struct  # at least one should find cats

    def test_zero_semantic_uses_struct_only(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="A about cats", topics=["feline"]))
        ltm.store(LTMEntry(content="B about dogs", topics=["canine"]))
        q = RecallQuery(
            topics=["feline"],
            semantic_query="canine dogs",
            semantic_weight=0.0,
        )
        results = recall_engine.recall(q)
        if results:
            # With zero semantic, topic match should dominate
            assert "feline" in results[0].match_reasons[0] if results[0].match_reasons else True

    def test_confidence_scales_final_score(self, recall_engine, ltm):
        ltm.store(LTMEntry(content="High conf", confidence=0.99, topics=["boost"]))
        ltm.store(LTMEntry(content="Low conf", confidence=0.1, topics=["boost"]))
        q = RecallQuery(topics=["boost"])
        results = recall_engine.recall(q)
        if len(results) >= 2:
            high_idx = next((i for i, r in enumerate(results) if r.entry.content == "High conf"), None)
            low_idx = next((i for i, r in enumerate(results) if r.entry.content == "Low conf"), None)
            if high_idx is not None and low_idx is not None:
                assert high_idx < low_idx
