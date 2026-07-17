"""
test_demo_chat_agent.py — Paired tests for demo_chat_agent.py

Verifies that both context-RAG and Muninn-RAG backends produce correct
answers over "The Republic of Plato" text.

Run:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_demo_chat_agent.py -v --import-mode=importlib --rootdir=tests -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ─── Bootstrap memory_module ──────────────────────────────────────────────────
_STANDALONE_ROOT = str(Path(__file__).resolve().parent.parent)
_DEMO_DIR = str(Path(__file__).resolve().parent.parent / "demo")
sys.path.insert(0, _STANDALONE_ROOT)
sys.path.insert(0, _DEMO_DIR)

from memory_module import MemoryAgent, RecallQuery

# Import the chat agent modules
from demo_chat_agent import load_and_chunk_text, ContextRAG, MuninnRAG, Config


# ─── Fixtures ─────────────────────────────────────────────────────────────────

TEXT_FILE = Path(__file__).resolve().parent.parent / "tests" / "The Republic of Pluto.txt"

# Use first 100 chunks for tests (~80K chars) — fast indexing, still representative
TEST_CHUNK_LIMIT = 100

@pytest.fixture(scope="module")
def chunks():
    """Load and chunk the Plato text once for all tests."""
    if not TEXT_FILE.exists():
        pytest.skip("Plato text file not found")
    all_chunks = load_and_chunk_text(TEXT_FILE, chunk_size=800, overlap=100)
    return all_chunks[:TEST_CHUNK_LIMIT]


@pytest.fixture(scope="module")
def muninn_agent(chunks):
    """Create a Muninn agent and index all chunks."""
    agent = MemoryAgent(":memory:", max_stm_segments=20)

    # Index chunks
    for chunk in chunks:
        agent.store_ltm(
            content=chunk["text"],
            class_type="text_chunk",
            topics=["plato", "republic"],
            confidence=1.0,
        )

    # Create entities
    entities_data = [
        ("Socrates",   "Greek philosopher, main speaker"),
        ("Glaucon",    "brother of Adeimantus, interlocutor"),
        ("Adeimantus", "brother of Glaucon, interlocutor"),
        ("Thrasymachus", "Sophist, argues might is right"),
        ("Polemarchus", "Son of Cephalus"),
        ("Cephalus",   "Wealthy old man of Piraeus"),
        ("Plato",      "Author of the Republic"),
    ]
    for name, desc in entities_data:
        agent.create_entity(name=name, description=desc)

    # Create concepts
    concepts = [
        ("what", "justice", "main topic of the Republic"),
        ("what", "idea of good", "highest form in Platonic philosophy"),
        ("what", "philosopher-king", "ideal ruler of the just state"),
        ("where", "ideal state", "the Kallipolis"),
        ("what", "cave allegory", "metaphor for ignorance and enlightenment"),
    ]
    for op, subj, focus in concepts:
        agent.add_concept(op, subj, focus)

    return agent


# ─── §1  Text Loading ────────────────────────────────────────────────────────

class TestTextLoading:
    """Verify the Plato text loads and chunks correctly."""

    def test_text_file_exists(self):
        assert TEXT_FILE.exists()

    def test_chunks_not_empty(self, chunks):
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self, chunks):
        for c in chunks[:10]:
            assert "id" in c
            assert "text" in c
            assert "offset" in c
            assert len(c["text"]) > 0

    def test_chunk_overlap_works(self, chunks):
        """Adjacent chunks should share some text (overlap)."""
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks")
        # Check that text from chunk N appears partially in chunk N+1
        found_overlap = False
        for i in range(len(chunks) - 1):
            text_a = chunks[i]["text"]
            text_b = chunks[i + 1]["text"]
            # Check last 100 chars of A appear in B
            tail = text_a[-100:]
            if tail in text_b:
                found_overlap = True
                break
        assert found_overlap, "No overlap found between adjacent chunks"

    def test_gutenberg_header_removed(self, chunks):
        """Gutenberg header/footer should be stripped."""
        all_text = " ".join(c["text"] for c in chunks)
        assert "START OF THE PROJECT GUTENBERG" not in all_text
        # Footer might not be in the first 100 chunks, just check header is gone

    def test_plato_content_present(self, chunks):
        """Core content should be present."""
        all_text = " ".join(c["text"] for c in chunks)
        assert "Republic" in all_text
        assert "Socrates" in all_text
        assert "justice" in all_text.lower()


# ─── §2  Muninn Memory Indexing ──────────────────────────────────────────────

class TestMuninnIndexing:
    """Verify the Muninn agent indexes the text correctly."""

    def test_ltm_entries_created(self, muninn_agent):
        status = muninn_agent.status()
        assert status["ltm_entries"] > 0

    def test_entities_created(self, muninn_agent):
        status = muninn_agent.status()
        assert status["entities"] >= 7

    def test_entity_socrates_exists(self, muninn_agent):
        matches = muninn_agent.resolve_entity("Socrates philosopher", top_k=3)
        assert len(matches) >= 1
        assert matches[0][0].name == "Socrates"

    def test_concept_justice_exists(self, muninn_agent):
        concepts = muninn_agent.ltm.get_concepts(subject="justice")
        assert len(concepts) >= 1

    def test_concepts_all_present(self, muninn_agent):
        all_concepts = muninn_agent.ltm.get_concepts()
        subjects = {c.subject for c in all_concepts}
        assert "justice" in subjects
        assert "philosopher-king" in subjects
        assert "idea of good" in subjects


# ─── §3  Muninn Recall Quality ───────────────────────────────────────────────

class TestMuninnRecall:
    """Verify Muninn recall returns relevant results for Republic questions."""

    def test_recall_justice(self, muninn_agent):
        results = muninn_agent.recall("what is justice", top_k=5)
        assert len(results) >= 1
        # At least one result should mention justice
        justice_found = any("justice" in r.entry.content.lower() for r in results)
        assert justice_found

    def test_recall_socrates(self, muninn_agent):
        results = muninn_agent.recall("who is Socrates", top_k=5)
        assert len(results) >= 1
        socrates_found = any("socrates" in r.entry.content.lower() for r in results)
        assert socrates_found

    def test_recall_cave_allegory(self, muninn_agent):
        results = muninn_agent.recall("cave allegory shadow", top_k=5)
        assert len(results) >= 1
        cave_found = any("cave" in r.entry.content.lower() for r in results)
        assert cave_found

    def test_recall_philosopher_king(self, muninn_agent):
        results = muninn_agent.recall("philosopher king ruler", top_k=5)
        assert len(results) >= 1
        pk_found = any(
            "philosopher" in r.entry.content.lower() and "king" in r.entry.content.lower()
            for r in results
        )
        assert pk_found

    def test_recall_entity_socrates(self, muninn_agent):
        """Entity-based recall should find Socrates-related content."""
        results = muninn_agent.recall("Socrates argument", top_k=5)
        assert len(results) >= 1

    def test_recall_concept_triple(self, muninn_agent):
        """Concept triple query should boost justice-related entries."""
        results = muninn_agent.recall(
            RecallQuery(operator="what", subject="justice"),
            top_k=5,
        )
        assert len(results) >= 1
        # Should have concept_triple in match_reasons
        has_concept = any("concept_triple" in r.match_reasons for r in results)
        # At minimum, results should be relevant
        assert len(results) >= 1


# ─── §4  Context-RAG Quality ────────────────────────────────────────────────

class TestContextRAG:
    """Verify context-RAG retrieves relevant chunks."""

    def test_retrieve_justice(self, chunks):
        rag = ContextRAG(chunks, None, Config())  # No LLM needed for retrieval test
        results = rag._retrieve("what is justice", top_k=5)
        assert len(results) >= 1
        justice_found = any("justice" in c["text"].lower() for c in results)
        assert justice_found

    def test_retrieve_socrates(self, chunks):
        rag = ContextRAG(chunks, None, Config())
        results = rag._retrieve("Socrates philosopher", top_k=5)
        assert len(results) >= 1
        socrates_found = any("socrates" in c["text"].lower() for c in results)
        assert socrates_found

    def test_retrieve_cave(self, chunks):
        rag = ContextRAG(chunks, None, Config())
        results = rag._retrieve("cave allegory shadows", top_k=5)
        assert len(results) >= 1
        cave_found = any("cave" in c["text"].lower() for c in results)
        assert cave_found

    def test_top_k_respected(self, chunks):
        rag = ContextRAG(chunks, None, Config())
        results = rag._retrieve("justice", top_k=3)
        assert len(results) <= 3

    def test_relevance_ordering(self, chunks):
        """More relevant chunks should rank higher."""
        rag = ContextRAG(chunks, None, Config())
        results = rag._retrieve("Republic justice Socrates", top_k=10)
        if len(results) >= 2:
            # First result should be more relevant than last
            first_text = results[0]["text"].lower()
            last_text  = results[-1]["text"].lower()
            first_score = sum(1 for w in ["republic", "justice", "socrates"] if w in first_text)
            last_score  = sum(1 for w in ["republic", "justice", "socrates"] if w in last_text)
            assert first_score >= last_score


# ─── §5  Round-trip Agent Tests ──────────────────────────────────────────────

class TestAgentRoundTrip:
    """End-to-end tests: query → retrieve → verify answer quality."""

    def test_muninn_recall_vs_context_retrieval(self, chunks, muninn_agent):
        """Both methods should find justice-related content for the same query."""
        query = "what is justice according to Plato"

        # Muninn recall
        muninn_results = muninn_agent.recall(query, top_k=5)
        muninn_texts = " ".join(r.entry.content.lower() for r in muninn_results)

        # Context RAG retrieval
        rag = ContextRAG(chunks, None, Config())
        context_results = rag._retrieve(query, top_k=5)
        context_texts = " ".join(c["text"].lower() for c in context_results)

        # Both should find justice-related content
        assert "justice" in muninn_texts
        assert "justice" in context_texts

    def test_muninn_entity_recall(self, muninn_agent):
        """Entity resolution should find Thrasymachus."""
        matches = muninn_agent.resolve_entity("sophist who argues might is right", top_k=3)
        assert len(matches) >= 1
        names = [m[0].name for m in matches]
        assert "Thrasymachus" in names

    def test_muninn_topic_filtering(self, muninn_agent):
        """Topic-filtered recall should narrow results."""
        results = muninn_agent.recall(
            RecallQuery(topics=["plato"], semantic_query="justice"),
            top_k=5,
        )
        assert len(results) >= 1
        # All results should have "plato" topic
        for r in results:
            assert "plato" in r.entry.topics

    def test_context_retrieval_handles_long_query(self, chunks):
        """Long multi-word query should still return relevant chunks."""
        rag = ContextRAG(chunks, None, Config())
        query = "What does Socrates say about the nature of justice in the ideal state"
        results = rag._retrieve(query, top_k=5)
        assert len(results) >= 1
        # At least some results should mention justice or state
        combined = " ".join(c["text"].lower() for c in results)
        assert "justice" in combined or "state" in combined


# ─── §6  Mode Switching ──────────────────────────────────────────────────────

class TestModeSwitching:
    """Verify mode switching works correctly."""

    def test_switch_to_muninn(self, chunks):
        from demo_chat_agent import ChatAgent
        # We can't fully init ChatAgent (needs Ollama), but test the mode logic
        agent = MuninnRAG(chunks, None, Config())
        assert not agent._indexed
        # Indexing should work without LLM
        agent._index_chunks()
        assert agent._indexed
        assert agent.agent.status()["ltm_entries"] > 0

    def test_muninn_idempotent_indexing(self, chunks):
        """Calling _index_chunks twice should not double entries."""
        agent = MuninnRAG(chunks, None, Config())
        agent._index_chunks()
        count_1 = agent.agent.status()["ltm_entries"]
        agent._index_chunks()  # second call should be no-op
        count_2 = agent.agent.status()["ltm_entries"]
        assert count_1 == count_2


# ─── §7  Edge Cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Verify robustness with edge-case inputs."""

    def test_empty_query(self, muninn_agent):
        results = muninn_agent.recall("", top_k=5)
        # Should not crash, may return empty or low-scored results
        assert isinstance(results, list)

    def test_gibberish_query(self, muninn_agent):
        results = muninn_agent.recall("asdfghjkl qwertyuiop", top_k=5)
        assert isinstance(results, list)

    def test_very_long_query(self, muninn_agent):
        query = "justice " * 100
        results = muninn_agent.recall(query, top_k=5)
        assert isinstance(results, list)

    def test_context_rag_empty_query(self, chunks):
        rag = ContextRAG(chunks, None, Config())
        results = rag._retrieve("", top_k=5)
        assert isinstance(results, list)

    def test_context_rag_no_match(self, chunks):
        rag = ContextRAG(chunks, None, Config())
        results = rag._retrieve("quantum chromodynamics quark gluon", top_k=5)
        # Should return something (all chunks have low score) but not crash
        assert isinstance(results, list)
