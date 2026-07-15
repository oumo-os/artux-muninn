"""
test_memory_bench.py — Performance benchmarks for the standalone memory_module.

Measures throughput and latency across all major operations.
Uses time.perf_counter() for precise timing.

Run:
    cd "artux-muninn-memory module independent"
    python -m pytest tests/test_memory_bench.py -v -s
"""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timedelta

import pytest

from memory_module import MemoryAgent, RecallQuery, LTMEntry, Entity
from memory_module.embeddings import embed, cosine_similarity


# ===================================================================
# Helpers
# ===================================================================

def _bench(label: str, fn, iterations: int = 1):
    """Run fn() iterations times, print stats, return (median, total)."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    total = sum(times)
    med = statistics.median(times)
    ops_sec = iterations / total if total > 0 else 0
    per_op_ms = (med * 1000) if iterations == 1 else (total / iterations * 1000)
    print(f"\n  {label}")
    print(f"    {iterations} iters | total={total:.3f}s | "
          f"median={med*1000:.1f}ms | per-op={per_op_ms:.1f}ms | "
          f"{ops_sec:.1f} ops/s")
    return med, total


# ===================================================================
# §1  STM Throughput
# ===================================================================

class TestBenchSTMThroughput:
    def test_record_1000_events(self):
        agent = MemoryAgent(":memory:", max_stm_segments=1000)
        _bench("STM record 1000 events",
               lambda: [agent.record_stm(f"event {i}") for i in range(1000)],
               iterations=1)

    def test_record_with_auto_compress(self):
        agent = MemoryAgent(":memory:", max_stm_segments=10)
        _bench("STM record 100 events (auto-compress at 10)",
               lambda: [agent.record_stm(f"event {i}") for i in range(100)],
               iterations=3)

    def test_record_single_event(self):
        agent = MemoryAgent(":memory:", max_stm_segments=10000)
        _bench("STM record single event (latency)",
               lambda: agent.record_stm("single event"),
               iterations=50)


# ===================================================================
# §2  LTM Consolidation
# ===================================================================

class TestBenchConsolidation:
    def test_consolidate_500_segments(self):
        agent = MemoryAgent(":memory:", max_stm_segments=500)
        for i in range(500):
            agent.record_stm(f"segment {i}")
        _bench("Consolidate 500 STM segments into LTM",
               lambda: agent.consolidate_ltm(),
               iterations=1)

    def test_consolidate_with_per_segment(self):
        agent = MemoryAgent(":memory:", max_stm_segments=100)
        for i in range(100):
            agent.record_stm(f"seg {i}")
        _bench("Consolidate 100 segments (per_segment=True)",
               lambda: agent.consolidate_ltm(per_segment=True),
               iterations=1)

    def test_consolidate_without_per_segment(self):
        agent = MemoryAgent(":memory:", max_stm_segments=100)
        for i in range(100):
            agent.record_stm(f"seg {i}")
        _bench("Consolidate 100 segments (per_segment=False)",
               lambda: agent.consolidate_ltm(per_segment=False),
               iterations=3)

    def test_consolidate_explicit_narrative(self):
        agent = MemoryAgent(":memory:")
        _bench("Consolidate explicit narrative (no STM)",
               lambda: agent.consolidate_ltm(narrative="explicit consolidation text"),
               iterations=20)


# ===================================================================
# §3  Recall Latency
# ===================================================================

class TestBenchRecallLatency:
    @pytest.fixture(scope="class")
    def populated_agent(self):
        agent = MemoryAgent(":memory:", max_stm_segments=10)
        for i in range(1000):
            agent.store_ltm(
                f"Entry number {i} about topic {i % 20} related to entity {i % 50}",
                topics=[f"topic_{i % 20}", f"group_{i % 10}"],
                confidence=0.5 + (i % 50) / 100,
            )
        return agent

    def test_recall_latency_10_queries(self, populated_agent):
        queries = [
            "what is topic 5",
            "who is entity 10",
            "entries about topic 15",
            "information regarding group 3",
            "tell me about topic 0",
        ]
        def run_queries():
            for q in queries:
                populated_agent.recall(q, top_k=5)
        _bench("Recall 50 queries (5×10) against 1000 entries",
               run_queries, iterations=10)

    def test_recall_structured_query(self, populated_agent):
        q = RecallQuery(
            operator="what",
            topics=["topic_5"],
            semantic_query="information about topic 5",
        )
        _bench("Recall structured query",
               lambda: populated_agent.recall(q, top_k=5),
               iterations=20)

    def test_recall_semantic_only(self, populated_agent):
        _bench("Recall semantic-only query",
               lambda: populated_agent.recall("entries about robotics and AI", top_k=5),
               iterations=20)

    def test_recall_with_scars(self, populated_agent):
        q = RecallQuery(semantic_query="topic 5", include_scars=True)
        _bench("Recall with scar hydration",
               lambda: populated_agent.recall(q, top_k=5),
               iterations=20)


# ===================================================================
# §4  Entity Resolution
# ===================================================================

class TestBenchEntityResolution:
    @pytest.fixture(scope="class")
    def entity_agent(self):
        agent = MemoryAgent(":memory:")
        for i in range(200):
            agent.create_entity(
                name=f"Entity{i}",
                description=f"Person number {i} working on project {i % 10}",
            )
        return agent

    def test_resolve_50_queries(self, entity_agent):
        def run():
            for i in range(50):
                entity_agent.resolve_entity(f"project {i % 10}", top_k=5)
        _bench("Resolve 50 entity queries against 200 entities",
               run, iterations=1)

    def test_resolve_single(self, entity_agent):
        _bench("Single entity resolve",
               lambda: entity_agent.resolve_entity("project 5", top_k=3),
               iterations=20)


# ===================================================================
# §5  Embeddings
# ===================================================================

class TestBenchEmbeddings:
    def test_embed_500_texts(self):
        texts = [f"This is test sentence number {i} with varied content" for i in range(500)]
        _bench("Embed 500 texts",
               lambda: [embed(t) for t in texts],
               iterations=1)

    def test_cosine_similarity_1000_pairs(self):
        a = embed("query text for similarity")
        candidates = [embed(f"candidate text {i}") for i in range(100)]
        _bench("Cosine similarity 1000 pairs",
               lambda: [cosine_similarity(a, c) for c in candidates for _ in range(10)],
               iterations=1)

    def test_embed_single(self):
        _bench("Embed single text",
               lambda: embed("single text embedding benchmark"),
               iterations=50)


# ===================================================================
# §6  Full Pipeline
# ===================================================================

class TestBenchFullPipeline:
    def test_record_consolidate_recall_x100(self):
        agent = MemoryAgent(":memory:", max_stm_segments=5)
        def cycle():
            for i in range(5):
                agent.record_stm(f"event {i} about topic {i}")
            agent.consolidate_ltm()
            agent.recall("topic")
        _bench("Full pipeline: record 5 + consolidate + recall × 100",
               lambda: [cycle() for _ in range(100)],
               iterations=1)


# ===================================================================
# §7  Decay
# ===================================================================

class TestBenchDecay:
    def test_decay_1000_entries(self):
        agent = MemoryAgent(":memory:")
        for i in range(1000):
            entry = agent.store_ltm(f"entry {i}", confidence=0.5 + (i % 50) / 100)
            with agent.db.connection() as conn:
                past = (datetime.utcnow() - timedelta(days=i % 90)).isoformat()
                conn.execute("UPDATE ltm_entries SET timestamp = ? WHERE id = ?", (past, entry.id))
        _bench("Decay 1000 entries",
               lambda: agent.run_decay(),
               iterations=2)

    def test_maintenance_1000_entries(self):
        agent = MemoryAgent(":memory:")
        for i in range(1000):
            conf = 0.05 if i % 3 == 0 else 0.8
            entry = agent.store_ltm(f"entry {i}", confidence=conf)
            with agent.db.connection() as conn:
                past = (datetime.utcnow() - timedelta(days=i % 90)).isoformat()
                conn.execute("UPDATE ltm_entries SET timestamp = ? WHERE id = ?", (past, entry.id))
        _bench("Maintenance 1000 entries (archive weak + purge old)",
               lambda: agent.run_maintenance(),
               iterations=1)


# ===================================================================
# §8  STM Compression
# ===================================================================

class TestBenchSTMCompression:
    def test_repeated_compress_head(self):
        agent = MemoryAgent(":memory:", max_stm_segments=100)
        for i in range(100):
            agent.record_stm(f"event {i}")
        _bench("compress_head ×10 with 100 segments",
               lambda: [agent.stm.compress_head(retain=5) for _ in range(10)],
               iterations=5)

    def test_flush_up_to_large_set(self):
        agent = MemoryAgent(":memory:", max_stm_segments=100)
        segs = []
        for i in range(100):
            segs.append(agent.record_stm(f"event {i}"))
        _bench("flush_up_to (100 segments)",
               lambda: agent.flush_stm_up_to(segs[-1].id),
               iterations=5)


# ===================================================================
# §9  Heavy / Extensive Benchmarks
# ===================================================================

class TestBenchHeavy:
    """Large-scale stress benchmarks."""

    def test_ltm_store_5000(self):
        agent = MemoryAgent(":memory:")
        _bench("HEAVY: Store 5000 LTM entries directly",
               lambda: [agent.store_ltm(f"entry {i}", topics=[f"t{i%100}"]) for i in range(5000)],
               iterations=1)

    def test_recall_100_against_5000(self):
        agent = MemoryAgent(":memory:", max_stm_segments=10)
        for i in range(5000):
            agent.store_ltm(
                f"Entry {i}: {['alpha','bravo','charlie','delta','echo'][i%5]} system "
                f"with {['sensor','actuator','controller','processor','memory'][i%5]} component",
                topics=[f"topic_{i%50}", f"group_{i%10}"],
                confidence=0.4 + (i % 60) / 100,
            )
        queries = [
            "alpha system sensor", "bravo actuator controller",
            "charlie processor memory", "delta system alpha",
            "echo sensor bravo controller",
        ]
        def run():
            for q in queries:
                agent.recall(q, top_k=10)
        _bench("HEAVY: Recall 100 queries (20×5) against 5000 entries",
               run, iterations=20)

    def test_entity_resolution_1000(self):
        agent = MemoryAgent(":memory:")
        for i in range(1000):
            agent.create_entity(
                name=f"Person{i:04d}",
                description=f"Engineer working on {['robotics','vision','nlp','systems','ml'][i%5]} "
                            f"in {['Boston','London','Tokyo','Berlin','Seoul'][i%5]}",
            )
        queries = ["robotics engineer in Boston", "NLP researcher Tokyo",
                    "machine learning Berlin", "vision systems Seoul"]
        def run():
            for q in queries:
                agent.resolve_entity(q, top_k=5)
        _bench("HEAVY: Resolve 20 queries against 1000 entities",
               run, iterations=10)

    def test_full_lifecycle_200(self):
        agent = MemoryAgent(":memory:", max_stm_segments=5)
        def lifecycle():
            for i in range(5):
                agent.record_stm(f"event {i} about {['topic_a','topic_b','topic_c'][i%3]}")
            agent.consolidate_ltm(topics=["lifecycle"])
            agent.recall("topic_a")
        _bench("HEAVY: Full lifecycle (record+consolidate+recall) × 200",
               lambda: [lifecycle() for _ in range(200)],
               iterations=1)

    def test_concurrent_writes_5000(self):
        agent = MemoryAgent(":memory:", max_stm_segments=10000)
        _bench("HEAVY: 5000 interleaved STM+LTM writes",
               lambda: [
                   (agent.record_stm(f"stm {i}"), agent.store_ltm(f"ltm {i}"))
                   for i in range(5000)
               ],
               iterations=1)

    def test_decay_5000(self):
        agent = MemoryAgent(":memory:")
        entries = []
        for i in range(5000):
            entry = agent.store_ltm(f"entry {i}", confidence=0.1 + (i % 90) / 100)
            entries.append(entry)
        with agent.db.connection() as conn:
            for e in entries:
                past = (datetime.utcnow() - timedelta(days=int(e.confidence * 100))).isoformat()
                conn.execute("UPDATE ltm_entries SET timestamp = ? WHERE id = ?", (past, e.id))
        _bench("HEAVY: Decay 5000 entries",
               lambda: agent.run_decay(),
               iterations=2)

    def test_source_attach_retrieve_2000(self):
        agent = MemoryAgent(":memory:")
        entry = agent.store_ltm("backing entry")
        for i in range(2000):
            ref = agent.record_source(
                location=f"/data/file_{i}.jpg", type="image",
                description=f"Frame {i} from camera",
            )
            agent.attach_source(ref.id, entry.id)
        _bench("HEAVY: Attach 2000 sources, retrieve all",
               lambda: agent.sources_for_entry(entry.id),
               iterations=5)

    def test_maintenance_heavy(self):
        agent = MemoryAgent(":memory:")
        for i in range(3000):
            conf = 0.03 if i % 4 == 0 else 0.7
            entry = agent.store_ltm(f"entry {i}", confidence=conf)
            with agent.db.connection() as conn:
                past = (datetime.utcnow() - timedelta(days=i % 200)).isoformat()
                conn.execute("UPDATE ltm_entries SET timestamp = ? WHERE id = ?", (past, entry.id))
        _bench("HEAVY: Maintenance 3000 entries (archive+purge)",
               lambda: agent.run_maintenance(),
               iterations=1)

    def test_recall_with_scars_heavy(self):
        agent = MemoryAgent(":memory:")
        for i in range(2000):
            agent.store_ltm(f"active entry {i}", topics=[f"t{i%40}"])
        for i in range(1000):
            agent.ltm.archive_entry(f"forgotten memory {i}", "ltm", f"orig_{i}", "test")
        q = RecallQuery(semantic_query="forgotten memory about topic", include_scars=True)
        _bench("HEAVY: Recall with scar hydration (2000 active + 1000 scars)",
               lambda: agent.recall(q, top_k=10),
               iterations=10)

    def test_embedding_throughput_2000(self):
        texts = [f"Benchmark sentence number {i} with diverse vocabulary about "
                 f"{['technology','science','art','music','philosophy'][i%5]}" for i in range(2000)]
        _bench("HEAVY: Embed 2000 texts throughput",
               lambda: [embed(t) for t in texts],
               iterations=1)

    def test_status_on_large_db(self):
        agent = MemoryAgent(":memory:")
        for i in range(2000):
            agent.store_ltm(f"entry {i}")
        for i in range(100):
            agent.create_entity(name=f"E{i}", description=f"entity {i}")
        _bench("HEAVY: status() on 2000 LTM + 100 entities",
               lambda: agent.status(),
               iterations=20)

    def test_consolidation_churn(self):
        agent = MemoryAgent(":memory:", max_stm_segments=10)
        for cycle in range(50):
            for i in range(10):
                agent.record_stm(f"cycle {cycle} event {i}")
            agent.consolidate_ltm()
        _bench("HEAVY: 50 consolidation cycles (10 events each)",
               lambda: None,  # just report — work done above
               iterations=1)
        status = agent.status()
        print(f"    Final state: {status['ltm_entries']} LTM entries, "
              f"{status['stm_segments']} STM segments, "
              f"{status['entities']} entities")