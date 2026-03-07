"""
example.py — Demonstrates the full Memory Module lifecycle.

Run with:  python example.py
"""

from memory_module import MemoryAgent

# -----------------------------------------------------------------------
# 1. Initialise
# -----------------------------------------------------------------------
agent = MemoryAgent(
    db_path="demo_memory.db",
    max_stm_segments=5,      # compress after 5 raw segments
    decay_lambda=0.01,
)

print("=== Memory Module Demo ===\n")

# -----------------------------------------------------------------------
# 2. Perception → STM recording
# -----------------------------------------------------------------------
agent.record_stm("User said: 'hi, I'm Kyle'")
agent.record_stm("User appears cheerful and curious")
agent.record_stm("Sam said: 'he is Musa, not Kyle'")
agent.record_stm("John addressed him as Musa")
agent.record_stm("User said: 'yeah, Musa is correct, I work on robotics'")

print("[STM Window]\n" + agent.get_stm_window())
print()

# -----------------------------------------------------------------------
# 3. Entity creation
# -----------------------------------------------------------------------
musa = agent.create_entity(
    description="described self as Kyle, later confirmed as Musa",
    name="Musa",
    topics=["identity", "robotics"],
)
print(f"[Entity Created] {musa.name} — id: {musa.id[:8]}…\n")

sam = agent.create_entity("Sam, a peer who corrected Musa's initial name", name="Sam")
john = agent.create_entity("John, addressed user as Musa — anchor authority", name="John")

# -----------------------------------------------------------------------
# 4. Observations and conflict resolution
# -----------------------------------------------------------------------
# t1: self-claim
agent.observe_entity(musa.id, "described self as Kyle", memory_ref="t1",
                     authority="self")

# t2: peer correction (Sam)
agent.correct_entity(musa.id, "he is Musa not Kyle",
                     correcting_entity_id=sam.id, memory_ref="t2")

# t3: anchor confirmation (John)
agent.correct_entity(musa.id, "addressed him as Musa",
                     correcting_entity_id=john.id, memory_ref="t3")

print("[Entity Narrative]\n" + agent.entities.get(musa.id).content)
print()

# -----------------------------------------------------------------------
# 5. Concepts
# -----------------------------------------------------------------------
agent.add_concept("what", "Musa", "identity", entity_id=musa.id)
agent.add_concept("dispute", "Kyle", "identity", entity_id=musa.id)
agent.add_concept("what", "Musa", "occupation", entity_id=musa.id)

# -----------------------------------------------------------------------
# 6. Associations
# -----------------------------------------------------------------------
agent.link_entities(sam.id, musa.id, "disputes")
agent.link_entities(john.id, musa.id, "addresses")

# -----------------------------------------------------------------------
# 7. LTM consolidation
# -----------------------------------------------------------------------
ltm_entry = agent.consolidate_ltm(
    narrative=(
        "User initially identified as Kyle (self-claim, t1). "
        "Sam corrected: his name is Musa (t2). "
        "John addressed him as Musa (t3). "
        "User confirmed: Musa, works on robotics."
    ),
    class_type="event",
    entities=[musa.id, sam.id, john.id],
    topics=["identity", "name-correction", "robotics"],
    concepts=["what:Musa:identity", "dispute:Kyle:identity"],
    confidence=0.95,
)
print(f"[LTM Consolidated] entry id: {ltm_entry.id[:8]}… confidence: {ltm_entry.confidence}\n")

# Store additional assertion
agent.store_ltm(
    content="Musa works on robotics — self-reported with confidence",
    class_type="assertion",
    entities=[musa.id],
    topics=["occupation", "robotics"],
    confidence=0.85,
)

# -----------------------------------------------------------------------
# 8. Recall
# -----------------------------------------------------------------------
print("[Recall: 'who is Musa']")
results = agent.recall("who is Musa", top_k=5)
for r in results:
    print(f"  score={r.score:.3f}  [{r.entry.class_type}] {r.entry.content[:80]}…")
    if r.match_reasons:
        print(f"    reasons: {r.match_reasons}")
print()

print("[Recall: 'what does Musa do for work']")
results = agent.recall("what does Musa do for work", top_k=3)
for r in results:
    print(f"  score={r.score:.3f}  {r.entry.content[:80]}…")
print()

# -----------------------------------------------------------------------
# 9. Entity recall
# -----------------------------------------------------------------------
print("[Entity Recall: 'person who corrected the name mistake']")
matches = agent.resolve_entity("person who corrected the name mistake")
for ent, score in matches:
    print(f"  {ent.name} — score={score:.3f}")
print()

# -----------------------------------------------------------------------
# 10. Forgetting cycle
# -----------------------------------------------------------------------
n_updated = agent.run_decay()
report = agent.run_maintenance()
print(f"[Decay] {n_updated} entries updated")
print(f"[Maintenance] {report}")
print()

# -----------------------------------------------------------------------
# 11. Status summary
# -----------------------------------------------------------------------
print("[Status]")
status = agent.status()
for k, v in status.items():
    print(f"  {k}: {v}")
