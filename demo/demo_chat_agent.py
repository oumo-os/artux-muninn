#!/usr/bin/env python3
"""
demo_chat_agent.py — Chat agent over "The Republic of Plato" with RAG vs Muninn toggle.

Loads the Plato text, chunks it, and lets you ask questions using either:
  1. CONTEXT mode — plain sliding-window context (traditional RAG)
  2. MUNINN mode  — Muninn memory module with semantic recall + entities

Switch modes at runtime with: /mode context  or  /mode muninn

──────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────
    python demo_chat_agent.py

    # Requires Ollama running locally:
    ollama serve
    ollama pull llama3.2

    # Optional env:
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_LLM=llama3.2
"""

import os
import sys
import json
import textwrap
import datetime
from pathlib import Path
from typing import Optional

_DEMO_DIR   = Path(__file__).resolve().parent
_STANDALONE = _DEMO_DIR.parent

# ─── Dependency checks ────────────────────────────────────────────────────────

def _require(pkg, install_hint):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"\n[ERROR] Missing: {pkg}")
        print(f"        Install: {install_hint}")
        sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    _require("openai", "pip install openai")

# Memory module
sys.path.insert(0, str(_STANDALONE))
sys.path.insert(0, str(_DEMO_DIR))
from _bootstrap import ensure_memory_module
ensure_memory_module(_STANDALONE)
from memory_module import MemoryAgent, RecallQuery, get_tools, ToolExecutor


# ─── Configuration ────────────────────────────────────────────────────────────

class Config:
    OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_LLM   = os.environ.get("OLLAMA_LLM", "llama3.2")
    DB_PATH      = "chat_agent_memory.db"
    TEXT_FILE    = _STANDALONE / "tests" / "The Republic of Pluto.txt"
    CHUNK_SIZE   = 800     # characters per chunk
    CHUNK_OVERLAP = 100
    CONTEXT_WINDOW = 8     # max context chunks for RAG mode
    STM_MAX      = 20
    MAX_TOKENS   = 1024
    TEMPERATURE  = 0.4


# ─── Text loader ──────────────────────────────────────────────────────────────

def load_and_chunk_text(filepath: Path, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Load a text file and split into overlapping chunks with metadata."""
    raw = filepath.read_text(encoding="utf-8", errors="replace")

    # Strip Gutenberg header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = raw.find(start_marker)
    end   = raw.find(end_marker)
    if start != -1:
        raw = raw[start + len(start_marker):]
    if end != -1:
        raw = raw[:end]
    raw = raw.strip()

    # Split into chunks
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(raw):
        end_i = min(i + chunk_size, len(raw))
        text  = raw[i:end_i].strip()
        if text:
            chunks.append({
                "id":      f"chunk_{chunk_id:04d}",
                "text":    text,
                "offset":  i,
                "length":  len(text),
            })
            chunk_id += 1
        i += chunk_size - overlap

    return chunks


# ─── Colour terminal ──────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    USER    = "\033[33m"
    AGENT   = "\033[32m"
    MODE    = "\033[36m"
    MEMORY  = "\033[35m"
    ERROR   = "\033[31m"
    INFO    = "\033[34m"


# ─── LLM client ──────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, config: Config):
        self.cfg    = config
        self.client = OpenAI(
            base_url=f"{config.OLLAMA_HOST}/v1",
            api_key="ollama",
        )

    def chat(self, messages: list[dict], system: str = "") -> str:
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        resp = self.client.chat.completions.create(
            model=self.cfg.OLLAMA_LLM,
            messages=full,
            temperature=self.cfg.TEMPERATURE,
            max_tokens=self.cfg.MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()


# ─── Context-mode RAG (plain sliding window) ──────────────────────────────────

class ContextRAG:
    """Traditional RAG: concatenate top-K relevant chunks into context window."""

    def __init__(self, chunks: list[dict], llm: LLMClient, config: Config):
        self.chunks = chunks
        self.llm    = llm
        self.cfg    = config

    def _retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        """Simple TF-IDF-like keyword matching (no embeddings needed)."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words & chunk_words)
            # Also boost by substring match
            if query.lower() in chunk["text"].lower():
                overlap += 5
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def answer(self, query: str) -> str:
        relevant = self._retrieve(query, self.cfg.CONTEXT_WINDOW)
        context  = "\n\n---\n\n".join(
            f"[Chunk {c['id']}] {c['text']}" for c in relevant
        )
        system = (
            "You are a helpful assistant answering questions about "
            "'The Republic of Plato' by Plato (translated by Benjamin Jowett). "
            "Answer based ONLY on the provided context. "
            "If the context doesn't contain enough information, say so.\n\n"
            f"── Context ──\n{context}"
        )
        return self.llm.chat(
            messages=[{"role": "user", "content": query}],
            system=system,
        )


# ─── Muninn-mode (memory module) ─────────────────────────────────────────────

class MuninnRAG:
    """Muninn memory module: store chunks as LTM entries, use semantic recall."""

    def __init__(self, chunks: list[dict], llm: LLMClient, config: Config):
        self.chunks = chunks
        self.llm    = llm
        self.cfg    = config
        self.agent  = MemoryAgent(
            config.DB_PATH,
            max_stm_segments=config.STM_MAX,
        )
        self.executor = ToolExecutor(self.agent)
        self.tools    = get_tools(format="openai")
        self.messages: list[dict] = []
        self._indexed = False

    def _index_chunks(self):
        """Store all text chunks as LTM entries with topic tags."""
        if self._indexed:
            return

        print(f"{C.MEMORY}Indexing {len(self.chunks)} chunks into Muninn memory…{C.RESET}")
        for i, chunk in enumerate(self.chunks):
            # Determine section from content
            section = "general"
            lower = chunk["text"].lower()
            if "book i" in lower or "book 1" in lower:
                section = "book-1"
            elif "book ii" in lower or "book 2" in lower:
                section = "book-2"
            elif "book iii" in lower or "book 3" in lower:
                section = "book-3"
            elif "book iv" in lower or "book 4" in lower:
                section = "book-4"
            elif "book v" in lower or "book 5" in lower:
                section = "book-5"
            elif "book vi" in lower or "book 6" in lower:
                section = "book-6"
            elif "book vii" in lower or "book 7" in lower:
                section = "book-7"
            elif "book viii" in lower or "book 8" in lower:
                section = "book-8"
            elif "book ix" in lower or "book 9" in lower:
                section = "book-9"
            elif "book x" in lower or "book 10" in lower:
                section = "book-10"
            elif "introduction" in lower or "analysis" in lower:
                section = "introduction"

            self.agent.store_ltm(
                content=chunk["text"],
                class_type="text_chunk",
                topics=["plato", "republic", section],
                confidence=1.0,
            )

            if (i + 1) % 50 == 0:
                print(f"  … indexed {i + 1}/{len(self.chunks)}")

        # Create key entities
        entities_data = [
            ("Socrates",   "Greek philosopher, main speaker in the Republic"),
            ("Glaucon",    "Son of Ariston, brother of Adeimantus, interlocutor"),
            ("Adeimantus", "Son of Ariston, brother of Glaucon, interlocutor"),
            ("Thrasymachus", "Chalcedonian sophist, argues might is right"),
            ("Polemarchus", "Son of Cephalus, inherits the argument"),
            ("Cephalus",   "Wealthy old man of Piraeus, patriarch"),
            ("Plato",      "Author of the Republic, student of Socrates"),
        ]
        for name, desc in entities_data:
            self.agent.create_entity(name=name, description=desc)

        # Create concept triples for key ideas
        concepts = [
            ("what", "justice", "main topic of the Republic"),
            ("what", "idea of good", "highest form in Platonic philosophy"),
            ("what", "philosopher-king", "ideal ruler of the just state"),
            ("where", "ideal state", "the Kallipolis, city of pigs"),
            ("what", "cave allegory", "metaphor for ignorance and enlightenment"),
        ]
        for op, subj, focus in concepts:
            self.agent.add_concept(op, subj, focus)

        self._indexed = True
        print(f"{C.MEMORY}Indexing complete. {self.agent.status()['ltm_entries']} LTM entries.{C.RESET}\n")

    def answer(self, query: str) -> str:
        self._index_chunks()

        # Use Muninn recall
        results = self.agent.recall(query, top_k=5)

        # Build context from recalled entries
        context_parts = []
        for r in results:
            source_info = ""
            if r.sources:
                source_info = f" [sources: {', '.join(s.location for s in r.sources)}]"
            context_parts.append(
                f"[{r.entry.class_type}] (score={r.score:.3f}) {r.entry.content[:400]}…{source_info}"
            )
        context = "\n\n".join(context_parts)

        # Also check entities
        entity_matches = self.agent.resolve_entity(query, top_k=3)
        entity_info = ""
        if entity_matches:
            entity_info = "\n\n── Related Entities ──\n" + "\n".join(
                f"  • {e.name}: {e.content[:200]}" for e, s in entity_matches if s > 0.2
            )

        system = (
            "You are a helpful assistant answering questions about "
            "'The Republic of Plato' by Plato (translated by Benjamin Jowett). "
            "Answer based on the Muninn memory recall results below. "
            "If recall doesn't contain enough information, say so.\n\n"
            f"── Muninn Recall Results ──\n{context}"
            f"{entity_info}"
        )

        return self.llm.chat(
            messages=[{"role": "user", "content": query}],
            system=system,
        )


# ─── Chat agent ──────────────────────────────────────────────────────────────

class ChatAgent:
    def __init__(self, config: Config):
        self.cfg   = config
        self.llm   = LLMClient(config)
        self.mode  = "context"   # "context" | "muninn"

        # Load text
        print(f"{C.INFO}Loading text from {config.TEXT_FILE.name}…{C.RESET}")
        self.chunks = load_and_chunk_text(
            config.TEXT_FILE, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        print(f"{C.INFO}Loaded {len(self.chunks)} chunks "
              f"({sum(c['length'] for c in self.chunks):,} chars){C.RESET}\n")

        # Init both backends
        self.context_rag = ContextRAG(self.chunks, self.llm, config)
        self.muninn_rag  = MuninnRAG(self.chunks, self.llm, config)

    def chat(self, query: str) -> str:
        if self.mode == "context":
            return self.context_rag.answer(query)
        else:
            return self.muninn_rag.answer(query)

    def switch_mode(self, mode: str):
        mode = mode.strip().lower()
        if mode in ("context", "c"):
            self.mode = "context"
        elif mode in ("muninn", "m"):
            self.mode = "muninn"
            # Ensure indexed
            self.muninn_rag._index_chunks()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'context' or 'muninn'.")


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner(config: Config):
    print(f"""
{C.BOLD}╔════════════════════════════════════════════════════════════╗
║    chat_agent  ·  Republic of Plato  ·  RAG vs Muninn      ║
╚════════════════════════════════════════════════════════════╝{C.RESET}

{C.MODE}MODE: context{C.RESET}  — plain sliding-window context (traditional RAG)
{C.MEMORY}MODE: muninn{C.RESET}   — Muninn memory module with semantic recall

Commands:
  {C.DIM}/mode context{C.RESET}  — switch to context-window RAG
  {C.DIM}/mode muninn{C.RESET}   — switch to Muninn memory recall
  {C.DIM}/status{C.RESET}        — show memory stats (muninn mode)
  {C.DIM}/history{C.RESET}       — show STM window (muninn mode)
  {C.DIM}/chunks{C.RESET}        — show loaded chunk count
  {C.DIM}/quit{C.RESET}          — exit

Text: The Republic of Plato (Jowett translation)
LLM  : {config.OLLAMA_LLM}
───────────────────────────────────────────────────────────────
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = Config()

    if not config.TEXT_FILE.exists():
        print(f"{C.ERROR}Text file not found: {config.TEXT_FILE}{C.RESET}")
        print("Place 'The Republic of Pluto.txt' in the tests/ directory.")
        sys.exit(1)

    print_banner(config)
    agent = ChatAgent(config)

    while True:
        try:
            mode_label = f"{C.MODE}context{C.RESET}" if agent.mode == "context" else f"{C.MEMORY}muninn{C.RESET}"
            user_input = input(f"{C.USER}You ({mode_label}):{C.RESET} ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            break

        elif user_input.lower() == "/status":
            if agent.mode == "muninn":
                s = agent.muninn_rag.agent.status()
                print(f"\n{C.INFO}── Muninn Memory Status ──{C.RESET}")
                for k, v in s.items():
                    print(f"  {k:<24}: {v}")
                print()
            else:
                print(f"{C.DIM}Status only available in muninn mode.{C.RESET}\n")

        elif user_input.lower() == "/history":
            if agent.mode == "muninn":
                window = agent.muninn_rag.agent.get_stm_window()
                print(f"\n{C.INFO}── STM Window ──{C.RESET}")
                print(window or "  (empty)")
                print()
            else:
                print(f"{C.DIM}History only available in muninn mode.{C.RESET}\n")

        elif user_input.lower() == "/chunks":
            print(f"{C.INFO}Loaded {len(agent.chunks)} chunks "
                  f"({sum(c['length'] for c in agent.chunks):,} chars){C.RESET}\n")

        elif user_input.lower().startswith("/mode"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{C.INFO}Current mode: {agent.mode}{C.RESET}")
                print(f"{C.DIM}Usage: /mode context  or  /mode muninn{C.RESET}\n")
            else:
                try:
                    agent.switch_mode(parts[1])
                    mode_label = f"{C.MODE}context{C.RESET}" if agent.mode == "context" else f"{C.MEMORY}muninn{C.RESET}"
                    print(f"{C.INFO}Switched to {mode_label}{C.RESET}\n")
                except ValueError as e:
                    print(f"{C.ERROR}{e}{C.RESET}\n")

        elif user_input.lower().startswith("/"):
            print(f"{C.DIM}Unknown command: {user_input}{C.RESET}\n")

        else:
            print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
            try:
                reply = agent.chat(user_input)
                print(reply)
                # Record to STM in muninn mode
                if agent.mode == "muninn":
                    agent.muninn_rag.agent.record_stm(f"User: {user_input}")
                    agent.muninn_rag.agent.record_stm(f"Agent: {reply[:300]}")
            except Exception as e:
                print(f"{C.ERROR}Error: {e}{C.RESET}")
            print()


if __name__ == "__main__":
    main()
