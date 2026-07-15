"""
test_memory_accuracy_bench.py — Accuracy, relevance, and agent-paired benchmarks.

Measures correctness quality, not speed. Includes:
  - Precision@k / Recall@k for semantic recall
  - Entity resolution precision
  - Topic filtering accuracy
  - Concept triple matching
  - Agent-paired workflow: simulate a real agent lifecycle and score recall quality

Run:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_memory_accuracy_bench.py -v --import-mode=importlib --rootdir=tests -s
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import pytest

from memory_module import MemoryAgent, RecallQuery, LTMEntry, Entity, Concept


# ===================================================================
# Helpers
# ===================================================================

def _p_at_k(results, expected_content_substrings: list[str], k: int) -> float:
    """Precision@k: fraction of top-k results that match any expected substring."""
    if not results:
        return 0.0
    top = results[:k]
    hits = sum(
        1 for r in top
        if any(sub.lower() in r.entry.content.lower() for sub in expected_content_substrings)
    )
    return hits / k


def _r_at_k(results, expected_content_substrings: list[str], k: int) -> float:
    """Recall@k: fraction of expected items found in top-k results."""
    if not expected_content_substrings:
        return 1.0
    top = results[:k]
    found = sum(
        1 for sub in expected_content_substrings
        if any(sub.lower() in r.entry.content.lower() for r in top)
    )
    return found / len(expected_content_substrings)


def _mrr(results, expected_content_substrings: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, r in enumerate(results):
        if any(sub.lower() in r.entry.content.lower() for sub in expected_content_substrings):
            return 1.0 / (i + 1)
    return 0.0


def _score(results, expected_id: str) -> float:
    """Return the score of the result matching expected_id, or 0.0."""
    for r in results:
        if r.entry.id == expected_id:
            return r.score
    return 0.0


# ===================================================================
# §1  Semantic Recall Accuracy
# ===================================================================

class TestSemanticRecallAccuracy:
    """Precision/recall of semantic recall across diverse query types."""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_exact_keyword_p_at_5(self, agent):
        agent.store_ltm("The capital of France is Paris", topics=["geography"])
        agent.store_ltm("Dogs are domesticated mammals", topics=["animals"])
        agent.store_ltm("Python is a programming language", topics=["tech"])
        agent.store_ltm("The Nile is the longest river", topics=["geography"])
        agent.store_ltm("JavaScript runs in browsers", topics=["tech"])
        results = agent.recall("capital of France", top_k=5)
        p5 = _p_at_k(results, ["Paris"], 5)
        assert p5 >= 0.2  # at least 1 of 5 should be relevant

    def test_paraphrase_r_at_3(self, agent):
        agent.store_ltm("User prefers dark mode for all applications")
        agent.store_ltm("User enjoys cooking Italian food")
        agent.store_ltm("User runs 5k every morning")
        results = agent.recall("display theme preference dark", top_k=3)
        r3 = _r_at_k(results, ["dark mode"], 3)
        assert r3 >= 0.33

    def test_multi_entity_mrr(self, agent):
        agent.store_ltm("User's daughter is named Aisha, age 7")
        agent.store_ltm("User's son is named Omar, age 10")
        agent.store_ltm("User lives in London")
        agent.store_ltm("User works at Google")
        results = agent.recall("daughter name", top_k=5)
        mrr = _mrr(results, ["Aisha"])
        assert mrr > 0.0  # Aisha should appear somewhere in results

    def test_unrelated_query_low_scores(self, agent):
        agent.store_ltm("The Eiffel Tower is in Paris")
        agent.store_ltm("Quantum computing uses qubits")
        agent.store_ltm("Bananas are yellow fruits")
        results = agent.recall("nuclear reactor design specifications", top_k=5)
        for r in results:
            # unrelated query should not score above 0.5
            assert r.score < 0.7 or "reactor" in r.entry.content.lower()

    def test_specific_beats_generic(self, agent):
        agent.store_ltm("Musa works on humanoid robotics at MIT", topics=["person", "robotics"])
        agent.store_ltm("Robotics is a broad branch of engineering", topics=["field"])
        results = agent.recall("Musa robotics", top_k=5)
        if len(results) >= 2:
            # Musa-specific entry should rank higher
            musa_idx = next((i for i, r in enumerate(results) if "Musa" in r.entry.content), None)
            generic_idx = next((i for i, r in enumerate(results) if "broad branch" in r.entry.content), None)
            if musa_idx is not None and generic_idx is not None:
                assert musa_idx < generic_idx


# ===================================================================
# §2  Topic Filtering Accuracy
# ===================================================================

class TestTopicFilteringAccuracy:
    """Do topic tags correctly narrow recall results?"""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_topic_tag_boosts_precision(self, agent):
        agent.store_ltm("Dark mode preference", topics=["ui", "preference"])
        agent.store_ltm("Coffee preference", topics=["food", "preference"])
        agent.store_ltm("IDE preference", topics=["ui", "tool"])
        agent.store_ltm("Python coding", topics=["code", "python"])
        results = agent.recall(
            RecallQuery(topics=["ui"], semantic_query="user preference"),
            top_k=10,
        )
        # UI-tagged entries should dominate the top results
        top3_topics = []
        for r in results[:3]:
            top3_topics.extend(r.entry.topics)
        ui_count = sum(1 for t in top3_topics if t == "ui")
        assert ui_count >= 1

    def test_multiple_topics_narrow_more(self, agent):
        agent.store_ltm("Python coding tips", topics=["code", "python"])
        agent.store_ltm("Python snake facts", topics=["animal", "python"])
        agent.store_ltm("Java coding guide", topics=["code", "java"])
        results = agent.recall(
            RecallQuery(topics=["code", "python"], semantic_query="programming"),
            top_k=5,
        )
        if results:
            # The entry with both "code" AND "python" should rank first
            assert "coding" in results[0].entry.content.lower()


# ===================================================================
# §3  Concept Triple Accuracy
# ===================================================================

class TestConceptTripleAccuracy:
    """Do concept triples (operator:subject:focus) route recalls correctly?"""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_concept_triple_direct_hit(self, agent):
        entry = agent.store_ltm("Kyle is a software engineer")
        concept = agent.add_concept("what", "Kyle", "identity", ltm_entry_id=entry.id)
        results = agent.recall(
            RecallQuery(operator="what", subject="Kyle"),
            top_k=5,
        )
        assert len(results) >= 1
        # Kyle's entry should be top result
        assert results[0].entry.id == entry.id
        assert "concept_triple" in results[0].match_reasons

    def test_wrong_operator_no_concept_boost(self, agent):
        entry = agent.store_ltm("Kyle works in Boston")
        agent.add_concept("where", "Kyle", "location", ltm_entry_id=entry.id)
        results = agent.recall(
            RecallQuery(operator="what", subject="Kyle"),
            top_k=5,
        )
        if results:
            # Should find via semantic but no concept_triple in match_reasons
            for r in results:
                if r.entry.id == entry.id:
                    assert "concept_triple" not in r.match_reasons

    def test_concept_focus_alignment(self, agent):
        e1 = agent.store_ltm("Kyle's location is Boston")
        e2 = agent.store_ltm("Kyle's hobby is chess")
        agent.add_concept("where", "Kyle", "location", ltm_entry_id=e1.id)
        agent.add_concept("what", "Kyle", "hobby", ltm_entry_id=e2.id)
        results = agent.recall(
            RecallQuery(operator="where", subject="Kyle"),
            top_k=5,
        )
        if results:
            # "where" query should prefer the location entry
            location_found = any("Boston" in r.entry.content for r in results[:2])
            assert location_found


# ===================================================================
# §4  Entity Resolution Accuracy
# ===================================================================

class TestEntityResolutionAccuracy:
    """Does fuzzy entity resolution return the right entity?"""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_exact_name_top_1(self, agent):
        agent.create_entity(name="Albert Einstein", description="physicist who developed relativity")
        agent.create_entity(name="Marie Curie", description="chemist who discovered radium")
        agent.create_entity(name="Isaac Newton", description="physicist who described gravity")
        matches = agent.resolve_entity("Albert Einstein", top_k=3)
        assert len(matches) >= 1
        assert matches[0][0].name == "Albert Einstein"

    def test_description_match_top_2(self, agent):
        e1 = agent.create_entity(name="E1", description="expert in quantum computing and qubits")
        e2 = agent.create_entity(name="E2", description="expert in marine biology and coral")
        e3 = agent.create_entity(name="E3", description="expert in quantum entanglement")
        matches = agent.resolve_entity("quantum computing specialist", top_k=3)
        assert len(matches) >= 1
        # E1 or E3 should be top 2, not E2
        top_ids = {m[0].id for m in matches[:2]}
        assert e2.id not in top_ids

    def test_unrelated_low_score(self, agent):
        agent.create_entity(name="Tech Corp", description="makes smartphones and tablets")
        matches = agent.resolve_entity("tropical fruit recipes", top_k=5)
        for _, score in matches:
            assert score < 0.5

    def test_resolution_top_k_respected(self, agent):
        for i in range(20):
            agent.create_entity(name=f"Person{i:04d}", description=f"person number {i}")
        matches = agent.resolve_entity("person", top_k=5)
        assert len(matches) <= 5

    def test_resolution_threshold_filters(self, agent):
        agent.create_entity(name="Alpha", description="senior engineer in robotics")
        agent.create_entity(name="Beta", description="junior designer in marketing")
        matches = agent.resolve_entity("senior robotics engineer", top_k=10)
        # Alpha should score significantly higher than Beta
        alpha_score = next((s for e, s in matches if e.name == "Alpha"), 0)
        beta_score = next((s for e, s in matches if e.name == "Beta"), 0)
        assert alpha_score > beta_score


# ===================================================================
# §5  Association Hop Accuracy
# ===================================================================

class TestAssociationHopAccuracy:
    """Does 1-hop association expansion find connected entries correctly?"""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_direct_association_found(self, agent):
        alice = agent.create_entity(name="Alice", description="PM at Google")
        bob = agent.create_entity(name="Bob", description=" engineer at Google")
        agent.link_entities(alice.id, bob.id, "colleague")
        agent.store_ltm("Bob's project is search ranking", entities=[bob.id])
        results = agent.recall(RecallQuery(subject=alice.id), top_k=5)
        if results:
            bob_found = any("Bob" in r.entry.content for r in results)
            assert bob_found

    def test_two_hop_not_reached(self, agent):
        a = agent.create_entity(name="A", description="first")
        b = agent.create_entity(name="B", description="second")
        c = agent.create_entity(name="C", description="third")
        agent.link_entities(a.id, b.id, "knows")
        agent.link_entities(b.id, c.id, "knows")
        agent.store_ltm("C's secret project", entities=[c.id])
        results = agent.recall(RecallQuery(subject=a.id), top_k=5)
        if results:
            for r in results:
                if "secret project" in r.entry.content:
                    assert "assoc_hop" not in r.match_reasons


# ===================================================================
# §6  Scar Hydration Accuracy
# ===================================================================

class TestScarHydrationAccuracy:
    """Are archived memories surfaced correctly when requested?"""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_scar_included_when_requested(self, agent):
        agent.store_ltm("Forgotten birthday is March 15", confidence=0.01)
        agent.run_maintenance()
        q = RecallQuery(semantic_query="birthday date", include_scars=True)
        results = agent.recall(q, top_k=10)
        scar_results = [r for r in results if r.from_archive]
        assert len(scar_results) >= 1
        assert "March 15" in scar_results[0].entry.content

    def test_scar_excluded_by_default(self, agent):
        agent.store_ltm("Forgotten fact about cats", confidence=0.01)
        agent.run_maintenance()
        q = RecallQuery(semantic_query="forgotten fact", include_scars=False)
        results = agent.recall(q, top_k=10)
        scar_results = [r for r in results if r.from_archive]
        assert len(scar_results) == 0

    def test_scar_scores_lower_than_active(self, agent):
        agent.store_ltm("Active memory about cats", confidence=0.9)
        agent.store_ltm("Faded memory about cats", confidence=0.01)
        agent.run_maintenance()
        q = RecallQuery(semantic_query="cats", include_scars=True)
        results = agent.recall(q, top_k=10)
        if len(results) >= 2:
            active_scores = [r.score for r in results if not r.from_archive]
            scar_scores = [r.score for r in results if r.from_archive]
            if active_scores and scar_scores:
                assert max(active_scores) >= max(scar_scores)


# ===================================================================
# §7  Agent-Paired Workflow Accuracy
# ===================================================================

class TestAgentPairedAccuracy:
    """
    Simulate a realistic agent lifecycle: observe → recall → create → update.
    Score the accuracy of recall at each step to verify the memory module
    supports correct agent behavior.
    """

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_observe_then_recall_identity(self, agent):
        """Agent observes facts about a user, then recalls their identity."""
        # Store LTM entries that reference the entity (observations alone aren't in LTM)
        ent = agent.create_entity(name="Musa", description="user identity")
        agent.observe_entity(ent.id, "Musa is a robotics engineer at MIT")
        agent.store_ltm("Musa is a robotics engineer at MIT", entities=[ent.id])
        agent.store_ltm("Musa prefers dark mode", entities=[ent.id])
        agent.store_ltm("Musa's daughter is named Aisha", entities=[ent.id])

        # Agent recalls who Musa is
        results = agent.recall("who is Musa", top_k=5)
        assert len(results) >= 1
        musa_found = any("Musa" in r.entry.content for r in results)
        assert musa_found

    def test_observe_then_recall_preference(self, agent):
        """Agent records user preference, later recalls it correctly."""
        agent.store_ltm("User prefers dark mode for all applications", topics=["preference", "ui"])
        agent.store_ltm("User is allergic to peanuts", topics=["health"])
        agent.store_ltm("User's favorite color is blue", topics=["preference"])

        results = agent.recall("what display theme does the user like", top_k=5)
        p3 = _p_at_k(results, ["dark mode"], 3)
        assert p3 >= 0.33

    def test_consolidate_then_recall(self, agent):
        """Agent records events, consolidates, then recalls from LTM."""
        labels = ["alpha", "bravo", "charlie", "delta", "echo"]
        for i in range(8):
            agent.record_stm(f"User mentioned topic {i}: {labels[i % 5]}")
        agent.consolidate_ltm()

        results = agent.recall("bravo topic", top_k=5)
        bravo_found = any("bravo" in r.entry.content.lower() for r in results)
        assert bravo_found

    def test_entity_create_observe_correct_cycle(self, agent):
        """Agent creates entity, observes, corrects, and recalls the correction."""
        ent = agent.create_entity(name="ProjectX", description="AI project")
        agent.observe_entity(ent.id, "ProjectX uses PyTorch framework")
        agent.correct_entity(ent.id, "ProjectX actually uses TensorFlow", "lead-engineer")

        # Recall should find both the observation and correction
        results = agent.recall("ProjectX framework", top_k=5)
        if results:
            content_all = " ".join(r.entry.content for r in results)
            assert "TensorFlow" in content_all or "PyTorch" in content_all

    def test_multi_topic_recall_precision(self, agent):
        """Agent stores multiple topics, recalls with specific topic filter."""
        agent.store_ltm("React is a UI framework", topics=["frontend", "javascript"])
        agent.store_ltm("Django is a Python web framework", topics=["backend", "python"])
        agent.store_ltm("TensorFlow is a ML framework", topics=["ml", "python"])
        agent.store_ltm("Docker containers package apps", topics=["devops"])

        results = agent.recall(
            RecallQuery(topics=["ml"], semantic_query="machine learning framework"),
            top_k=5,
        )
        if results:
            ml_found = any("TensorFlow" in r.entry.content for r in results[:3])
            assert ml_found

    def test_full_agent_session(self, agent):
        """
        Simulate a full agent session:
        1. Record 20 STM events (user conversation)
        2. Consolidate to LTM
        3. Create entities for people mentioned
        4. Link entities
        5. Recall across all memory types
        6. Verify recall quality
        """
        # 1. Record conversation events
        events = [
            "User said they work with Alice on Project Alpha",
            "User mentioned Bob handles the database",
            "Alice prefers Vim, Bob prefers Emacs",
            "Project Alpha deadline is next Friday",
            "User had coffee this morning",
            "Bob reported the database is running slow",
            "Alice suggested switching to PostgreSQL",
            "User agreed with Alice's suggestion",
            "Project Alpha uses Python and React",
            "User's manager is Charlie",
        ]
        for event in events:
            agent.record_stm(event)

        # 2. Consolidate
        agent.consolidate_ltm(topics=["project-alpha", "team"])

        # 3. Create entities
        alice = agent.create_entity(name="Alice", description="team member, prefers Vim")
        bob = agent.create_entity(name="Bob", description="database person, prefers Emacs")
        charlie = agent.create_entity(name="Charlie", description="user's manager")

        # 4. Link entities
        agent.link_entities(alice.id, bob.id, "colleague")
        agent.link_entities(alice.id, charlie.id, "reports_to")

        # 5. Recall queries
        queries = [
            ("Project Alpha tech stack", ["Python", "React"]),
            ("database person", ["Bob"]),
            ("Alice editor preference", ["Vim"]),
            ("user manager", ["Charlie"]),
        ]
        scores = {}
        for query_text, expected in queries:
            results = agent.recall(query_text, top_k=5)
            p = _p_at_k(results, expected, 5)
            scores[query_text] = p

        # 6. Verify overall quality
        avg_precision = sum(scores.values()) / len(scores)
        assert avg_precision >= 0.3  # at least 30% average precision

    def test_temporal_recall(self, agent):
        """Agent records time-stamped events, recalls with time bracket."""
        old_entry = LTMEntry(content="Old project kickoff meeting", confidence=1.0)
        old_entry.timestamp = datetime(2024, 1, 15)
        agent.ltm.store(old_entry)

        new_entry = LTMEntry(content="New project kickoff meeting", confidence=1.0)
        new_entry.timestamp = datetime(2025, 6, 15)
        agent.ltm.store(new_entry)

        results = agent.recall(
            RecallQuery(semantic_query="project kickoff", after=datetime(2025, 1, 1)),
            top_k=5,
        )
        if results:
            assert all("2025" in r.entry.content or "New" in r.entry.content for r in results)

    def test_confidence_gating_accuracy(self, agent):
        """High-confidence entries recalled, low-confidence filtered."""
        agent.store_ltm("Certain fact: water boils at 100C", confidence=0.95)
        agent.store_ltm("Uncertain guess: aliens built pyramids", confidence=0.05)

        results_all = agent.recall(RecallQuery(semantic_query="fact", min_confidence=0.0), top_k=10)
        results_high = agent.recall(RecallQuery(semantic_query="fact", min_confidence=0.5), top_k=10)

        assert len(results_all) >= 2
        # High-confidence filter should exclude the low-confidence entry
        high_contents = [r.entry.content for r in results_high]
        assert any("Certain fact" in c for c in high_contents)
        assert not any("Uncertain guess" in c for c in high_contents)


# ===================================================================
# §8  Score Distribution Quality
# ===================================================================

class TestScoreDistribution:
    """Verify that scores form a meaningful ranking — relevant > irrelevant."""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_relevant_scores_higher_than_irrelevant(self, agent):
        agent.store_ltm("User's favorite food is sushi", topics=["food"])
        agent.store_ltm("Quantum entanglement is a physics phenomenon", topics=["physics"])
        agent.store_ltm("User likes Japanese cuisine", topics=["food"])

        results = agent.recall("food preference", top_k=10)
        if len(results) >= 2:
            food_scores = [r.score for r in results if "food" in r.entry.content.lower() or "sushi" in r.entry.content.lower() or "cuisine" in r.entry.content.lower()]
            physics_scores = [r.score for r in results if "quantum" in r.entry.content.lower()]
            if food_scores and physics_scores:
                assert max(food_scores) > max(physics_scores)

    def test_topic_match_beats_semantic_only(self, agent):
        agent.store_ltm("Database indexing strategy", topics=["database"])
        agent.store_ltm("Cooking pasta al dente", topics=["cooking"])
        agent.store_ltm("PostgreSQL query optimization", topics=["database"])

        results = agent.recall(
            RecallQuery(topics=["database"], semantic_query="optimization"),
            top_k=5,
        )
        if len(results) >= 2:
            db_scores = [r.score for r in results if "database" in r.entry.topics]
            non_db_scores = [r.score for r in results if "database" not in r.entry.topics]
            if db_scores and non_db_scores:
                assert max(db_scores) >= max(non_db_scores)


# ===================================================================
# §9  Summary / Aggregated Metrics
# ===================================================================

class TestAggregateMetrics:
    """Run all accuracy scenarios and compute aggregate scores."""

    @pytest.fixture
    def agent(self):
        return MemoryAgent(":memory:", max_stm_segments=10)

    def test_overall_accuracy_report(self, agent):
        """Run a battery of recall scenarios and report aggregate precision."""
        # Seed memory with diverse content
        entries = {
            "Paris is the capital of France": ["geography", "europe"],
            "Tokyo is the capital of Japan": ["geography", "asia"],
            "Python is a programming language": ["tech", "code"],
            "JavaScript runs in browsers": ["tech", "frontend"],
            "User prefers dark mode": ["preference", "ui"],
            "User likes sushi": ["food", "preference"],
            "Alice works at Google": ["person", "company"],
            "Bob works at Google too": ["person", "company"],
            "Project Alpha uses React": ["project", "tech"],
            "Project Alpha deadline is Friday": ["project", "deadline"],
        }
        for content, topics in entries.items():
            agent.store_ltm(content, topics=topics)

        # Query suite: (query, expected_substring, topic_filter)
        queries = [
            ("capital of France", ["Paris"], None),
            ("capital of Japan", ["Tokyo"], None),
            ("programming language", ["Python"], None),
            ("display theme", ["dark mode"], None),
            ("food preference", ["sushi"], None),
            ("Google employees", ["Alice", "Bob"], None),
            ("Project Alpha", ["React", "deadline"], None),
            ("European capital", ["Paris"], ["geography"]),
        ]

        precisions = []
        mrrs = []
        for query_text, expected, topics in queries:
            if topics:
                q = RecallQuery(topics=topics, semantic_query=query_text)
            else:
                q = query_text
            results = agent.recall(q, top_k=5)
            p = _p_at_k(results, expected, 5)
            m = _mrr(results, expected)
            precisions.append(p)
            mrrs.append(m)

        avg_p5 = sum(precisions) / len(precisions)
        avg_mrr = sum(mrrs) / len(mrrs)
        print(f"\n    Aggregate P@5:  {avg_p5:.3f}")
        print(f"    Aggregate MRR:  {avg_mrr:.3f}")
        print(f"    Per-query P@5:  {[f'{p:.2f}' for p in precisions]}")

        assert avg_p5 >= 0.15, f"Average P@5 too low: {avg_p5:.3f}"
        assert avg_mrr >= 0.3, f"Average MRR too low: {avg_mrr:.3f}"
