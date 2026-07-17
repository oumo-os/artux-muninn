#!/usr/bin/env python3
"""
test_function_calling_recall.py — Automated recall QA battery for
the function-calling memory agent.

Workflow
--------
1. Feed book chunks through the LLM agent (same tool-calling loop as
   demo_function_calling.py).  The LLM autonomously decides what to
   record, consolidate, and which entities to create.
2. After all chunks are processed, run a battery of recall queries
   directly against the MemoryAgent (not through the LLM).
3. For each query, check whether relevant content appears in the
   top-K results.  Report pass/fail and aggregate metrics.

Usage
-----
    cd "artux-muninn-memory module independent"
    python tests/test_function_calling_recall.py           \
        --chat-model M:/Dev/projects/models/Qwen3.5-0.8B-Q4_K_M.gguf  \
        --embed-model M:/Dev/projects/models/nomic-embed-text-v1.5.Q8_0.gguf \
        --chunk-size 10 --max-chunks 2

Output
------
  A summary table:
    character-questions:    12/15 passed  (80.0%)
    concept-questions:       8/10 passed  (80.0%)
    ...
    OVERALL:  55/75 passed  (73.3%)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

_TESTS_DIR  = Path(__file__).resolve().parent
_STANDALONE = _TESTS_DIR.parent
_DEMO_DIR   = _STANDALONE / "demo"
_MODELS_DIR  = Path(os.environ.get("MODELS_DIR", r"M:\Dev\projects\models"))

# Set embedding model before ANY memory_module import
os.environ.setdefault(
    "MUNINN_EMBEDDING_MODEL",
    str(_MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf"),
)

sys.path.insert(0, str(_STANDALONE))
sys.path.insert(0, str(_DEMO_DIR))
sys.path.insert(0, str(_TESTS_DIR))

from _bootstrap import ensure_memory_module
ensure_memory_module(_STANDALONE)

from text_utils import load_chunks


# ─── Colour ─────────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    CYAN   = "\033[36m"
    DIM    = "\033[2m"


# ─── Questions ─────────────────────────────────────────────────────────────

QUESTIONS: list[dict] = [
    # ── Characters ──────────────────────────────────────────────────────
    {"q": "Who is the main speaker in The Republic?",                   "cat": "characters", "kw": ["Socrates"]},
    {"q": "Who was the wealthy old man hosting the discussion?",        "cat": "characters", "kw": ["Cephalus"]},
    {"q": "Who was Thrasymachus arguing with?",                         "cat": "characters", "kw": ["Socrates", "Thrasymachus"]},
    {"q": "Who challenged Socrates about the definition of justice?",   "cat": "characters", "kw": ["Thrasymachus"]},
    {"q": "Who was Glaucon in relation to the discussion?",             "cat": "characters", "kw": ["Glaucon"]},
    {"q": "Who was Adeimantus?",                                        "cat": "characters", "kw": ["Adeimantus"]},
    {"q": "Who was Polemarchus?",                                       "cat": "characters", "kw": ["Polemarchus"]},
    {"q": "Who challenged Socrates at the beginning with a fierce argument?", "cat": "characters", "kw": ["Thrasymachus"]},
    {"q": "Who is the son of Ariston mentioned in the Republic?",       "cat": "characters", "kw": ["Glaucon", "Adeimantus"]},
    {"q": "Who was the goddess Bendis?",                                "cat": "characters", "kw": ["Bendis"]},
    {"q": "Who is the author of The Republic?",                         "cat": "characters", "kw": ["Plato"]},
    {"q": "Who was the sophist known for arguing that justice is the advantage of the stronger?", "cat": "characters", "kw": ["Thrasymachus"]},
    {"q": "Who was the quiet participant Cleitophon?",                  "cat": "characters", "kw": ["Cleitophon"]},
    {"q": "Who was Er?",                                                "cat": "characters", "kw": ["Er"]},
    {"q": "Who was Cephalus' son?",                                     "cat": "characters", "kw": ["Polemarchus"]},

    # ── Concepts ────────────────────────────────────────────────────────
    {"q": "What is the main subject of The Republic?",                  "cat": "concepts",  "kw": ["justice"]},
    {"q": "What is the allegory of the cave?",                         "cat": "concepts",  "kw": ["cave"]},
    {"q": "What is the divided line?",                                  "cat": "concepts",  "kw": ["divided", "line"]},
    {"q": "What are the Forms or Ideas in Plato's philosophy?",         "cat": "concepts",  "kw": ["form", "idea"]},
    {"q": "What is the ideal State or Kallipolis?",                     "cat": "concepts",  "kw": ["ideal", "state", "kallipolis"]},
    {"q": "What is the philosopher-king?",                              "cat": "concepts",  "kw": ["philosopher", "king"]},
    {"q": "What is the theory of the tripartite soul?",                 "cat": "concepts",  "kw": ["soul"]},
    {"q": "What is the myth of Er?",                                    "cat": "concepts",  "kw": ["Er"]},
    {"q": "What is the ship of state analogy?",                         "cat": "concepts",  "kw": ["ship", "state"]},
    {"q": "What is the ring of Gyges?",                                 "cat": "concepts",  "kw": ["Gyges"]},
    {"q": "What is the education of the guardians?",                    "cat": "concepts",  "kw": ["guardian"]},
    {"q": "What is the noble lie?",                                     "cat": "concepts",  "kw": ["noble", "lie"]},
    {"q": "What is the censorship of poetry in the ideal state?",       "cat": "concepts",  "kw": ["poetry", "poet"]},
    {"q": "What is the concept of the golden mean?",                    "cat": "concepts",  "kw": ["mean", "moderation"]},
    {"q": "What is the doctrine of ideas?",                             "cat": "concepts",  "kw": ["idea"]},
    {"q": "What is the number of the State or the Platonic number?",    "cat": "concepts",  "kw": ["number"]},
    {"q": "What is the theory of anamnesis or recollection?",           "cat": "concepts",  "kw": ["recollection", "anamnesis"]},
    {"q": "What is the second education for philosophers?",             "cat": "concepts",  "kw": ["dialectic", "philosopher"]},
    {"q": "What is the concept of justice in the individual?",          "cat": "concepts",  "kw": ["justice", "soul"]},
    {"q": "What is the decline of constitutions or regimes?",           "cat": "concepts",  "kw": ["timocracy", "oligarchy", "democracy"]},

    # ── Books & Structure ───────────────────────────────────────────────
    {"q": "What is discussed in Book I of the Republic?",               "cat": "books",     "kw": ["Thrasymachus", "justice"]},
    {"q": "What is the main topic of Book II?",                         "cat": "books",     "kw": ["Glaucon", "Gyges"]},
    {"q": "What is discussed in Book IV?",                              "cat": "books",     "kw": ["justice", "soul", "city"]},
    {"q": "What is discussed in Book V?",                               "cat": "books",     "kw": ["women", "children", "community"]},
    {"q": "What is discussed in Book VI?",                              "cat": "books",     "kw": ["philosopher", "good"]},
    {"q": "What is discussed in Book VII?",                             "cat": "books",     "kw": ["cave", "education"]},
    {"q": "What is discussed in Book VIII?",                            "cat": "books",     "kw": ["timocracy", "oligarchy"]},
    {"q": "What is discussed in Book IX?",                              "cat": "books",     "kw": ["tyrant", "tyranny"]},
    {"q": "What is discussed in Book X?",                               "cat": "books",     "kw": ["poetry", "immortality", "Er"]},
    {"q": "What are the four declining forms of government?",           "cat": "books",     "kw": ["timocracy", "oligarchy", "democracy", "tyranny"]},

    # ── Places & Scenes ─────────────────────────────────────────────────
    {"q": "Where does the discussion in the Republic take place?",      "cat": "places",    "kw": ["Piraeus"]},
    {"q": "What festival was being held when the discussion started?",  "cat": "places",    "kw": ["Bendis", "festival"]},
    {"q": "What is the Piraeus?",                                       "cat": "places",    "kw": ["Piraeus"]},

    # ── Key Arguments ───────────────────────────────────────────────────
    {"q": "What is Thrasymachus' definition of justice?",               "cat": "arguments", "kw": ["advantage", "stronger"]},
    {"q": "What is Glaucon's challenge about justice?",                 "cat": "arguments", "kw": ["Gyges", "ring"]},
    {"q": "What is the analogy of the sun in the Republic?",            "cat": "arguments", "kw": ["sun", "good"]},
    {"q": "What is the line analogy in the Republic?",                  "cat": "arguments", "kw": ["divided", "line"]},
    {"q": "What is the cave allegory about?",                           "cat": "arguments", "kw": ["cave", "shadow"]},
    {"q": "What is the proof of immortality in the Republic?",          "cat": "arguments", "kw": ["immortal", "soul"]},
    {"q": "What does Socrates say about the poets?",                    "cat": "arguments", "kw": ["poet", "banish"]},
    {"q": "What is the marriage number?",                               "cat": "arguments", "kw": ["number", "marriage"]},
    {"q": "What is the comparison of the tyrant to the king?",          "cat": "arguments", "kw": ["tyrant", "king"]},
    {"q": "What is the reward of justice discussed at the end?",        "cat": "arguments", "kw": ["reward", "afterlife"]},

    # ── Scholar / Analysis ──────────────────────────────────────────────
    {"q": "What is the Jowett introduction about?",                     "cat": "analysis",  "kw": ["Jowett"]},
    {"q": "What is the analysis of Greek poetry in the Republic?",      "cat": "analysis",  "kw": ["poetry", "Homer"]},
    {"q": "What is the Spartan influence on the Republic?",             "cat": "analysis",  "kw": ["Sparta", "Lacedaemon"]},
    {"q": "What is the Pythagorean influence on Plato?",                "cat": "analysis",  "kw": ["Pythagoras"]},
]


# ─── Tool-calling feed logic (reused from demo) ──────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant engaged in a conversation. "
    "The user is sharing passages from a text with you.\n\n"
    "You have access to memory tools:\n"
    "  - record_stm: record a short-term memory entry after each turn\n"
    "  - consolidate_ltm: persist important facts to long-term memory\n"
    "  - create_entity: create an entity for a person or concept\n"
    "  - observe_entity: append a new observation to an existing entity\n"
    "  - resolve_entity: search for an entity by name or description\n"
    "  - recall: search long-term memory for relevant information\n"
    "  - get_stm_window: view current short-term memory contents\n\n"
    "Use these tools naturally as you would rely on your own memory. "
    "Respond thoughtfully to what the user shares."
)


def get_tool_definitions():
    """Return OpenAI-format tool definitions from memory_module.tools."""
    from memory_module.tools import get_tools
    return get_tools(format="openai")


def _trim_messages(messages, max_keep=10):
    """Trim conversation history to prevent context overflow."""
    if len(messages) <= max_keep + 1:
        return messages
    return [messages[0]] + messages[-(max_keep):]


def process_message_with_tools(llm, messages, executor, tools, config):
    """Send to LLM, handle tool calls in loop, return text response.

    Supports both native JSON function calling and XML-style tool_call blocks
    (fallback for small models like Qwen3.5-0.8B).
    """
    from xml_parser import parse_tool_calls
    max_loops = getattr(config, "max_tool_loops", 8)
    for _ in range(max_loops):
        trimmed = _trim_messages(messages)
        resp = llm.create_chat_completion(
            messages=trimmed,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=getattr(config, "max_tokens", 256),
            temperature=getattr(config, "temperature", 0.4),
            top_p=getattr(config, "top_p", 0.9),
        )
        choice = resp["choices"][0]
        msg = choice["message"]
        content = (msg.get("content") or "").strip()

        # Path 1: native JSON tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_list = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]}}
                for tc in tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tc_list,
            })
            results = executor.run_openai(tool_calls)
            messages.extend(results)
            continue

        # Path 2: XML/bracket fallback
        xml_calls = parse_tool_calls(content) if content else []
        if xml_calls:
            messages.append({"role": "assistant", "content": content})
            for tc in xml_calls:
                name = tc["name"]
                args = tc["arguments"]
                try:
                    result = executor.execute(name, args)
                except Exception as e:
                    result = f"Error: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"xml-{name}",
                    "content": result,
                })
            continue

        # Path 3: plain text
        messages.append({"role": "assistant", "content": content})
        return content
    return "(tool loop limit)"


# ─── Config simple class ─────────────────────────────────────────────────

class TestConfig:
    def __init__(self, **kw):
        self.chat_model = kw.get("chat_model", "")
        self.embed_model = kw.get("embed_model", "")
        self.text_file = kw.get("text_file", "")
        self.chunk_size = kw.get("chunk_size", 10)
        self.max_chunks = kw.get("max_chunks", 0)
        self.n_ctx = kw.get("n_ctx", 4096)
        self.n_gpu_layers = kw.get("n_gpu_layers", 0)
        self.max_tokens = kw.get("max_tokens", 256)
        self.temperature = kw.get("temperature", 0.4)
        self.top_p = kw.get("top_p", 0.9)
        self.max_tool_loops = kw.get("max_tool_loops", 8)


# ─── Recall test suite ───────────────────────────────────────────────────

class RecallTestSuite:
    """Run the recall QA battery against an in-memory agent."""

    def __init__(self):
        self.results: list[dict] = []

    def run_question(self, q: dict, agent) -> dict:
        """
        Run a single recall query. Returns:
          {"question": ..., "cat": ..., "passed": bool, "top_results": [...]}
        """
        query = q["q"]
        expected = [kw.lower() for kw in q["kw"]]
        top_k = 5

        results = agent.recall(query, top_k=top_k)

        top_texts = []
        found_any = False
        for r in results:
            content_lower = r.entry.content.lower()
            top_texts.append({
                "score": r.score,
                "content_preview": r.entry.content[:200],
            })
            # Check if ANY expected keyword appears in this result
            if any(kw in content_lower for kw in expected):
                found_any = True

        return {
            "question": query,
            "cat": q["cat"],
            "expected_keywords": q["kw"],
            "passed": found_any,
            "num_results": len(results),
            "top_results": top_texts,
        }

    def run_all(self, agent) -> dict:
        """Run all questions and return aggregated report."""
        for q in QUESTIONS:
            r = self.run_question(q, agent)
            self.results.append(r)

        return self.summarize()

    def summarize(self) -> dict:
        """Aggregate results by category and overall."""
        by_cat: dict[str, list[dict]] = {}
        for r in self.results:
            by_cat.setdefault(r["cat"], []).append(r)

        lines = []
        total = len(self.results)
        total_passed = sum(1 for r in self.results if r["passed"])

        lines.append(f"\n{C.BOLD}{'Category':<20} {'Passed':>10} {'Total':>6} {'Rate':>8}{C.RESET}")
        lines.append("-" * 48)

        for cat in sorted(by_cat):
            items = by_cat[cat]
            pass_ct = sum(1 for r in items if r["passed"])
            rate = pass_ct / len(items) * 100
            lines.append(f"  {cat:<18} {pass_ct:>4}/{len(items):<4} {rate:>6.1f}%")

        if total:
            rate = total_passed / total * 100
            lines.append("-" * 48)
            lines.append(f"  {'OVERALL':<18} {total_passed:>4}/{total:<4} {rate:>6.1f}%")

        return {"summary": "\n".join(lines), "results": self.results, "total": total, "passed": total_passed}

    def detailed_report(self, max_fail: int = 10) -> str:
        """Show failed queries in detail."""
        failures = [r for r in self.results if not r["passed"]]
        if not failures:
            return "All questions passed."

        lines = [f"\n{C.RED}Failed questions ({len(failures)}):{C.RESET}"]
        for r in failures[:max_fail]:
            lines.append(f"  {C.RED}[FAIL]{C.RESET} [{r['cat']}] {r['question']}")
            lines.append(f"      expected: {r['expected_keywords']}")
            if r["top_results"]:
                lines.append(f"      top score: {r['top_results'][0]['score']:.3f}")
                lines.append(f"      top text: {r['top_results'][0]['content_preview'][:120]}...")
            else:
                lines.append(f"      (no results)")
        return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Automated recall QA battery")
    parser.add_argument("--chat-model", default=str(_MODELS_DIR / "LFM2.5-230M-Q8_0.gguf"))
    parser.add_argument("--embed-model", default=str(_MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf"))
    parser.add_argument("--text", default=str(_STANDALONE / "tests" / "The Republic of Pluto.txt"))
    parser.add_argument("--chunk-size", type=int, default=10, help="Paragraphs per user message")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks processed (0=all)")
    parser.add_argument("--quiet", action="store_true", help="Suppress agent replies")
    args = parser.parse_args()

    config = TestConfig(
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        text_file=args.text,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )

    # Validate
    for label, path in [("Chat model", config.chat_model),
                        ("Embed model", config.embed_model),
                        ("Text file", config.text_file)]:
        if not Path(path).exists():
            print(f"[ERROR] {label} not found: {path}")
            sys.exit(1)

    # ── Load models ────────────────────────────────────────────────────
    from llama_cpp import Llama
    print(f"{C.BLUE}Loading chat model...{C.RESET}")
    llm = Llama(
        model_path=config.chat_model,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        verbose=False,
    )

    os.environ["MUNINN_EMBEDDING_MODEL"] = config.embed_model
    from memory_module import MemoryAgent
    from memory_module.tools import ToolExecutor

    print(f"{C.BLUE}Initialising memory module...{C.RESET}")
    agent = MemoryAgent(":memory:", max_stm_segments=50)
    executor = ToolExecutor(agent)
    tools = get_tool_definitions()

    # ── Load chunks ────────────────────────────────────────────────────
    print(f"{C.BLUE}Loading text...{C.RESET}")
    chunks = load_chunks(config.text_file, chunk_size=config.chunk_size)
    if config.max_chunks and config.max_chunks < len(chunks):
        chunks = chunks[:config.max_chunks]
    print(f"{C.CYAN}Loaded {len(chunks)} chunks.{C.RESET}")

    # ── Feed chunks (fresh context per chunk to avoid token overflow) ───
    t0 = time.time()

    for idx, chunk in enumerate(chunks):
        n = idx + 1
        preview = chunk["text"][:100].replace("\n", " ") + "..."
        if not args.quiet:
            print(f"\n{C.CYAN}-- Chunk {n}/{len(chunks)} --{C.RESET} {preview}")

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": chunk["text"]})

        reply = process_message_with_tools(
            llm, messages, executor, tools, config
        )

        if not args.quiet:
            print(f"{C.CYAN}Agent:{C.RESET} {reply[:200]}...")

    elapsed = time.time() - t0
    print(f"\n{C.GREEN}Feed complete: {len(chunks)} chunks in {elapsed:.1f}s{C.RESET}")

    # ── Run recall battery ─────────────────────────────────────────────
    print(f"\n{C.BOLD}{'='*50}{C.RESET}")
    print(f"{C.BOLD}Running recall QA battery ({len(QUESTIONS)} questions)...{C.RESET}")

    suite = RecallTestSuite()
    report = suite.run_all(agent)

    print(report["summary"])
    print(suite.detailed_report(max_fail=10))

    # Print memory stats
    s = agent.status()
    print(f"\n{C.DIM}Memory stats: {s.get('ltm_entries', '?')} LTM, "
          f"{s.get('entities', '?')} entities, "
          f"{s.get('stm_raw', '?')} STM segments{C.RESET}")

    # Exit code based on pass rate
    total = report["total"]
    passed = report["passed"]
    rate = passed / total if total else 0

    if rate >= 0.5:
        print(f"\n{C.GREEN}THRESHOLD PASSED: {rate*100:.1f}% >= 50%{C.RESET}")
        sys.exit(0)
    else:
        print(f"\n{C.RED}THRESHOLD FAILED: {rate*100:.1f}% < 50%{C.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
