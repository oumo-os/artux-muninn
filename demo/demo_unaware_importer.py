#!/usr/bin/env python3
"""
demo_unaware_importer.py — Agent that "hears" a book without knowing it.

A chat agent receives paragraphs from The Republic of Plato as if they're
normal user messages.  The agent doesn't know it's importing a book — it
just responds to each paragraph naturally.  Over time, STM fills up,
auto-consolidates to LTM, and the user can switch to Muninn recall to
query the accumulated knowledge.

Components
──────────
  LLM         → llama-cpp-python + Qwen3.5-0.8B GGUF (local, CPU)
  Embeddings  → llama-cpp-python + nomic-embed-text GGUF (local, CPU)
  Memory      → memory_module (SQLite :memory:)

─────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────
    cd "artux-muninn-memory module independent"
    python demo/demo_unaware_importer.py

    # Or specify paths explicitly:
    python demo/demo_unaware_importer.py \\
        --chat-model "M:/Dev/projects/models/Qwen3.5-0.8B-Q4_K_M.gguf" \\
        --embed-model "M:/Dev/projects/models/nomic-embed-text-v1.5.Q8_0.gguf" \\
        --text "../tests/The Republic of Pluto.txt"

Commands at the prompt:
    /mode context   — switch to plain chat-history mode
    /mode muninn    — switch to Muninn memory recall mode
    /status         — show memory stats
    /history        — show current STM window
    /pause          — pause auto-feeding (resume with Enter)
    /skip [N]       — skip N paragraphs (default 1)
    /jump [N]       — jump to paragraph N
    /quit           — exit
"""

import os
import sys
import time
import argparse
import datetime
from pathlib import Path
from typing import Optional

# ─── Path setup ──────────────────────────────────────────────────────────────

_DEMO_DIR    = Path(__file__).resolve().parent
_STANDALONE  = _DEMO_DIR.parent
_MODELS_DIR  = Path(os.environ.get("MODELS_DIR", r"M:\Dev\projects\models"))

# Set embedding model before ANY memory_module import
os.environ.setdefault(
    "MUNINN_EMBEDDING_MODEL",
    str(_MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf"),
)

# Ensure memory_module and demo utils are importable
sys.path.insert(0, str(_STANDALONE))
sys.path.insert(0, str(_DEMO_DIR))

from _bootstrap import ensure_memory_module
ensure_memory_module(_STANDALONE)

# ─── Lightweight imports (no heavy deps yet) ────────────────────────────────

from text_utils import load_paragraphs, detect_section, detect_people


# ─── Colour terminal ─────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    USER    = "\033[33m"
    AGENT   = "\033[32m"
    SYSTEM  = "\033[36m"
    MEMORY  = "\033[35m"
    INFO    = "\033[34m"
    ERROR   = "\033[31m"
    PARA    = "\033[90m"


# ─── Configuration ───────────────────────────────────────────────────────────

class Config:
    # Models
    CHAT_MODEL    = str(_MODELS_DIR / "Qwen3.5-0.8B-Q4_K_M.gguf")
    EMBED_MODEL   = str(_MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf")
    N_CTX         = 4096
    N_GPU_LAYERS  = 0

    # Text source
    TEXT_FILE     = str(_STANDALONE / "tests" / "The Republic of Pluto.txt")

    # Memory
    DB_PATH       = ":memory:"
    STM_MAX       = 20
    CONSOLIDATE_N = 8

    # Generation
    MAX_TOKENS    = 256
    TEMPERATURE   = 0.4
    TOP_P         = 0.9


# ─── Lazy-loaded singletons ─────────────────────────────────────────────────

_llm_singleton = None
_agent_singleton = None


def get_llm(config: Config):
    """Lazy-load the LLM singleton."""
    global _llm_singleton
    if _llm_singleton is None:
        from llama_cpp import Llama
        print(f"{C.INFO}Loading chat model: {Path(config.CHAT_MODEL).name}...{C.RESET}")
        _llm_singleton = Llama(
            model_path=config.CHAT_MODEL,
            n_ctx=config.N_CTX,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False,
        )
        print(f"{C.INFO}Chat model loaded.{C.RESET}\n")
    return _llm_singleton


def get_agent(config: Config):
    """Lazy-load the MemoryAgent singleton."""
    global _agent_singleton
    if _agent_singleton is None:
        os.environ["MUNINN_EMBEDDING_MODEL"] = config.EMBED_MODEL
        from memory_module import MemoryAgent
        print(f"{C.MEMORY}Initialising memory module...{C.RESET}")
        _agent_singleton = MemoryAgent(
            config.DB_PATH,
            max_stm_segments=config.STM_MAX,
        )
        print(f"{C.MEMORY}Memory ready.{C.RESET}\n")
    return _agent_singleton


# ─── Agent modes ─────────────────────────────────────────────────────────────

class ContextMode:
    """Plain sliding-window context (traditional RAG)."""

    def __init__(self, config: Config, max_history: int = 16):
        self.cfg = config
        self.history: list[dict] = []
        self.max_history = max_history

    def respond(self, user_message: str) -> str:
        llm = get_llm(self.cfg)
        self.history.append({"role": "user", "content": user_message})

        trimmed = self.history[-self.max_history:]

        system = (
            "You are a helpful, articulate assistant engaged in a conversation. "
            "The other participant is sharing passages from a text with you. "
            "Respond thoughtfully -- acknowledge what they shared, offer insights, "
            "ask clarifying questions, or continue the thread of ideas. "
            "Be natural and conversational, not robotic."
        )

        resp = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}] + trimmed,
            max_tokens=self.cfg.MAX_TOKENS,
            temperature=self.cfg.TEMPERATURE,
            top_p=self.cfg.TOP_P,
        )
        reply = resp["choices"][0]["message"]["content"].strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply


class MuninnMode:
    """Muninn memory module with semantic recall."""

    def __init__(self, config: Config):
        self.cfg = config

    def respond(self, user_message: str) -> str:
        llm = get_llm(self.cfg)
        agent = get_agent(self.cfg)

        # Recall relevant memories
        results = agent.recall(user_message, top_k=5)

        # Build context from recalled entries
        context_parts = []
        for r in results:
            context_parts.append(
                f"[score={r.score:.3f}] {r.entry.content[:500]}"
            )
        context = "\n\n".join(context_parts) if context_parts else "(no relevant memories found)"

        # Entity resolution
        entity_matches = agent.resolve_entity(user_message, top_k=3)
        entity_info = ""
        if entity_matches:
            entity_info = "\n\n-- Related Entities --\n" + "\n".join(
                f"  * {e.name}: {e.content[:200]}" for e, s in entity_matches if s > 0.2
            )

        system = (
            "You are a helpful assistant with access to a memory of past conversations. "
            "Use the recalled memories below to answer the user's question. "
            "If the memories don't contain enough information, say so.\n\n"
            f"-- Recalled Memories --\n{context}"
            f"{entity_info}"
        )

        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.cfg.MAX_TOKENS,
            temperature=self.cfg.TEMPERATURE,
            top_p=self.cfg.TOP_P,
        )
        return resp["choices"][0]["message"]["content"].strip()


# ─── Main demo ───────────────────────────────────────────────────────────────

class UnawareImporter:
    """
    Main demo controller.

    Feeds paragraphs from the text as simulated user messages.
    The agent responds naturally, unaware it's importing a book.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.mode_name = "context"

        # Load text
        print(f"{C.INFO}Loading text from {Path(config.TEXT_FILE).name}...{C.RESET}")
        self.paragraphs = load_paragraphs(config.TEXT_FILE)
        print(f"{C.INFO}Loaded {len(self.paragraphs)} paragraphs.{C.RESET}\n")

        # Init modes (LLM and agent lazy-loaded on first use)
        self.context_mode = ContextMode(config)
        self.muninn_mode  = MuninnMode(config)
        self.current_mode = self.context_mode

        # State
        self.para_index   = 0
        self.total_paras  = len(self.paragraphs)
        self.paused       = False
        self._consolidation_count = 0

    # ── Paragraph feeding ──────────────────────────────────────────────

    def feed_next_paragraph(self) -> Optional[str]:
        """
        Feed the next paragraph as a user message.
        Returns the paragraph text, or None if no more paragraphs.
        """
        if self.para_index >= self.total_paras:
            return None

        para = self.paragraphs[self.para_index]
        self.para_index += 1
        return para["text"]

    def process_paragraph(self, text: str) -> str:
        """
        Process a paragraph: agent responds, exchange recorded to STM.
        Returns the agent's reply.
        """
        reply = self.current_mode.respond(text)

        # Record to STM
        agent = get_agent(self.cfg)
        agent.record_stm(f"User: {text[:500]}")
        agent.record_stm(f"Agent: {reply[:500]}")

        # Track consolidation
        self._consolidation_count += 1
        if self._consolidation_count >= self.cfg.CONSOLIDATE_N:
            self._auto_consolidate()
            self._consolidation_count = 0

        return reply

    def _auto_consolidate(self):
        """Auto-consolidate STM to LTM."""
        try:
            agent = get_agent(self.cfg)
            entry = agent.consolidate_ltm(
                class_type="conversation",
                confidence=0.85,
            )
            print(f"{C.MEMORY}  [auto-consolidated -> LTM {entry.id[:8]}...]{C.RESET}")
        except Exception as e:
            print(f"{C.ERROR}  [consolidation error: {e}]{C.RESET}")

    # ── Commands ───────────────────────────────────────────────────────

    def handle_command(self, cmd: str) -> bool:
        """
        Handle a /command.  Returns True if the demo should continue,
        False if it should exit.
        """
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg     = parts[1] if len(parts) > 1 else ""

        if command == "/quit" or command == "/exit" or command == "/q":
            return False

        elif command == "/mode":
            mode = arg.strip().lower()
            if mode in ("context", "c"):
                self.mode_name = "context"
                self.current_mode = self.context_mode
                print(f"{C.SYSTEM}Switched to context mode (plain chat history){C.RESET}")
            elif mode in ("muninn", "m"):
                self.mode_name = "muninn"
                self.current_mode = self.muninn_mode
                print(f"{C.SYSTEM}Switched to muninn mode (memory recall){C.RESET}")
            else:
                print(f"{C.ERROR}Unknown mode: {mode}. Use 'context' or 'muninn'.{C.RESET}")

        elif command == "/status":
            agent = get_agent(self.cfg)
            s = agent.status()
            print(f"\n{C.SYSTEM}-- Memory Status --{C.RESET}")
            for k, v in s.items():
                print(f"  {k:<24}: {v}")
            print(f"  {'paragraph_index':<24}: {self.para_index}/{self.total_paras}")
            print(f"  {'mode':<24}: {self.mode_name}")
            print()

        elif command == "/history":
            agent = get_agent(self.cfg)
            window = agent.get_stm_window()
            print(f"\n{C.SYSTEM}-- STM Window --{C.RESET}")
            print(window or "  (empty)")
            print()

        elif command == "/pause":
            self.paused = True
            print(f"{C.SYSTEM}Paused. Press Enter to resume.{C.RESET}")

        elif command == "/resume":
            self.paused = False
            print(f"{C.SYSTEM}Resumed.{C.RESET}")

        elif command == "/skip":
            n = int(arg) if arg.isdigit() else 1
            self.para_index = min(self.para_index + n, self.total_paras)
            print(f"{C.SYSTEM}Skipped {n} paragraph(s). Now at {self.para_index}/{self.total_paras}.{C.RESET}")

        elif command == "/jump":
            if arg.isdigit():
                target = int(arg)
                self.para_index = min(target, self.total_paras)
                print(f"{C.SYSTEM}Jumped to paragraph {self.para_index}/{self.total_paras}.{C.RESET}")
            else:
                print(f"{C.ERROR}Usage: /jump <paragraph_number>{C.RESET}")

        elif command == "/consolidate":
            self._auto_consolidate()

        elif command == "/recall":
            if arg:
                agent = get_agent(self.cfg)
                results = agent.recall(arg, top_k=5)
                print(f"\n{C.SYSTEM}-- Recall: \"{arg}\" --{C.RESET}")
                for i, r in enumerate(results):
                    print(f"  {i+1}. [{r.score:.3f}] {r.entry.content[:200]}...")
                print()
            else:
                print(f"{C.ERROR}Usage: /recall <query>{C.RESET}")

        elif command == "/entities":
            agent = get_agent(self.cfg)
            entities = agent.entities.all()
            print(f"\n{C.SYSTEM}-- Entities ({len(entities)}) --{C.RESET}")
            for e in entities[:20]:
                print(f"  * {e.name}: {e.content[:100]}")
            if len(entities) > 20:
                print(f"  ... and {len(entities) - 20} more")
            print()

        else:
            print(f"{C.ERROR}Unknown command: {command}{C.RESET}")

        return True

    # ── Interactive loop ───────────────────────────────────────────────

    def run(self):
        """Run the interactive demo."""
        self._print_banner()

        while True:
            # Check if we've run out of paragraphs
            if self.para_index >= self.total_paras:
                print(f"\n{C.SYSTEM}All {self.total_paras} paragraphs have been processed.{C.RESET}")
                print(f"{C.SYSTEM}You can still ask questions in muninn mode, or /quit.{C.RESET}\n")

            # Get next paragraph
            para_text = self.feed_next_paragraph()
            if para_text is None:
                # No more paragraphs — just wait for user input
                try:
                    user_input = input(f"{C.USER}You:{C.RESET} ").strip()
                except EOFError:
                    break
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue
                # Treat as a question in muninn mode
                print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
                reply = self.current_mode.respond(user_input)
                print(reply)
                agent = get_agent(self.cfg)
                agent.record_stm(f"User asked: {user_input}")
                agent.record_stm(f"Agent replied: {reply[:500]}")
                print()
                continue

            # Show paragraph preview
            preview = para_text[:120].replace('\n', ' ') + ("…" if len(para_text) > 120 else "")
            print(f"\n{C.PARA}── Paragraph {self.para_index}/{self.total_paras} ──{C.RESET}")
            print(f"{C.PARA}{preview}{C.RESET}")

            # Process the paragraph
            print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
            reply = self.process_paragraph(para_text)
            print(reply)
            print()

            # Check for user interruption (non-blocking)
            # In a real scenario you'd use select/threading, but for simplicity
            # we just process paragraphs sequentially with a small delay
            time.sleep(0.1)

    def _print_banner(self):
        print(f"""
{C.BOLD}================================================================
  demo  --  unaware importer  --  Republic of Plato
================================================================{C.RESET}

{C.PARA}The agent doesn't know it's importing a book.{C.RESET}
Paragraphs arrive as simulated user messages.
The agent just responds naturally, building memory over time.

{C.SYSTEM}MODE: context{C.RESET} -- plain chat history
{C.MEMORY}MODE: muninn{C.RESET}  -- memory recall from accumulated knowledge

{C.INFO}Text:{C.RESET}     {Path(self.cfg.TEXT_FILE).name} ({self.total_paras} paragraphs)
{C.INFO}Chat LLM:{C.RESET} {Path(self.cfg.CHAT_MODEL).name}
{C.INFO}Embeddings:{C.RESET} {Path(self.cfg.EMBED_MODEL).name}
{C.INFO}Memory:{C.RESET}    SQLite {self.cfg.DB_PATH}

Commands:
  {C.DIM}/mode context{C.RESET}   -- switch to plain chat-history mode
  {C.DIM}/mode muninn{C.RESET}    -- switch to Muninn memory recall
  {C.DIM}/status{C.RESET}         -- show memory stats
  {C.DIM}/history{C.RESET}        -- show STM window
  {C.DIM}/skip [N]{C.RESET}       -- skip N paragraphs
  {C.DIM}/jump [N]{C.RESET}       -- jump to paragraph N
  {C.DIM}/recall <query>{C.RESET} -- manually query memories
  {C.DIM}/entities{C.RESET}       -- list known entities
  {C.DIM}/consolidate{C.RESET}    -- force consolidation
  {C.DIM}/pause{C.RESET}          -- pause auto-feeding
  {C.DIM}/quit{C.RESET}           -- exit
--------------------------------------------------------------
""")


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demo: agent imports a book without knowing it"
    )
    parser.add_argument("--chat-model", default=Config.CHAT_MODEL,
                        help="Path to GGUF chat model")
    parser.add_argument("--embed-model", default=Config.EMBED_MODEL,
                        help="Path to GGUF embedding model")
    parser.add_argument("--text", default=Config.TEXT_FILE,
                        help="Path to text file")
    parser.add_argument("--db", default=Config.DB_PATH,
                        help="SQLite database path")
    parser.add_argument("--consolidate-every", type=int, default=Config.CONSOLIDATE_N,
                        help="Auto-consolidate every N paragraphs")
    parser.add_argument("--max-stm", type=int, default=Config.STM_MAX,
                        help="Max STM segments before compression")
    args = parser.parse_args()

    config = Config()
    config.CHAT_MODEL    = args.chat_model
    config.EMBED_MODEL   = args.embed_model
    config.TEXT_FILE     = args.text
    config.DB_PATH       = args.db
    config.CONSOLIDATE_N = args.consolidate_every
    config.STM_MAX       = args.max_stm

    # Verify model files exist
    for label, path in [("Chat model", config.CHAT_MODEL),
                        ("Embed model", config.EMBED_MODEL),
                        ("Text file", config.TEXT_FILE)]:
        if not Path(path).exists():
            print(f"{C.ERROR}[ERROR] {label} not found: {path}{C.RESET}")
            sys.exit(1)

    demo = UnawareImporter(config)
    try:
        demo.run()
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Shutting down...{C.RESET}")
        agent = get_agent(demo.cfg)
        n = agent.run_decay()
        report = agent.run_maintenance()
        print(f"{C.MEMORY}Decay: {n} entries. Maintenance: {report}{C.RESET}")


if __name__ == "__main__":
    main()
