#!/usr/bin/env python3
"""
demo_function_calling.py — LLM agent with memory tools, processing a book.

The agent receives passages from a text as user messages and has access to
memory tools (record_stm, consolidate_ltm, create_entity, etc.) via
OpenAI-compatible function calling.  It autonomously decides what to remember,
what entities to create, and when to consolidate — the script never records
to memory directly.

Usage
-----
    cd "artux-muninn-memory module independent"
    python demo/demo_function_calling.py

Options:
    --chat-model PATH   GGUF chat model path
    --embed-model PATH  GGUF embedding model path
    --text PATH         Text file to process
    --chunk-size N      Paragraphs per user message (default 5)
    --max-chunks N      Stop after N chunks (default unlimited)
    --skip-intro        Skip chunks before the first BOOK header
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

_DEMO_DIR    = Path(__file__).resolve().parent
_STANDALONE  = _DEMO_DIR.parent
_MODELS_DIR  = Path(os.environ.get("MODELS_DIR", r"M:\Dev\projects\models"))

# Set embedding model before ANY memory_module import
os.environ.setdefault(
    "MUNINN_EMBEDDING_MODEL",
    str(_MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf"),
)

sys.path.insert(0, str(_STANDALONE))
sys.path.insert(0, str(_DEMO_DIR))

from _bootstrap import ensure_memory_module
ensure_memory_module(_STANDALONE)

from text_utils import load_chunks


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    USER    = "\033[33m"
    AGENT   = "\033[32m"
    SYSTEM  = "\033[36m"
    MEMORY  = "\033[35m"
    TOOL    = "\033[35m"
    INFO    = "\033[34m"
    ERROR   = "\033[31m"
    PARA    = "\033[90m"


class Config:
    CHAT_MODEL    = str(_MODELS_DIR / "LFM2.5-230M-Q8_0.gguf")
    EMBED_MODEL   = str(_MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf")
    N_CTX         = 4096
    N_GPU_LAYERS  = 0
    TEXT_FILE     = str(_STANDALONE / "tests" / "The Republic of Pluto.txt")
    DB_PATH       = ":memory:"
    STM_MAX       = 50
    CHUNK_SIZE    = 5
    MAX_CHUNKS    = 0
    MAX_TOKENS    = 256
    TEMPERATURE   = 0.4
    TOP_P         = 0.9
    MAX_TOOL_LOOPS = 8


# ─── Lazy singletons ───────────────────────────────────────────────────────

_llm_singleton = None
_agent_singleton = None
_executor_singleton = None


def get_llm(config: Config):
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


def get_executor(config: Config):
    global _executor_singleton
    if _executor_singleton is None:
        from memory_module.tools import ToolExecutor
        agent = get_agent(config)
        _executor_singleton = ToolExecutor(agent)
    return _executor_singleton


# ─── Tool-calling loop ─────────────────────────────────────────────────────

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


def get_tool_definitions() -> list[dict]:
    """Return OpenAI-format tool definitions from memory_module.tools."""
    from memory_module.tools import get_tools
    return get_tools(format="openai")


def _trim_messages(messages, max_keep=10):
    """Trim conversation history to prevent context overflow.

    Keeps the system prompt (first message) and the most recent max_keep messages.
    """
    if len(messages) <= max_keep + 1:
        return messages
    return [messages[0]] + messages[-(max_keep):]


def _truncate_after_chunk(messages):
    """After a chunk is fully processed, keep only the system prompt.

    Conversation history is NOT needed — the memory system (STM/LTM/entities)
    retains all information. Dropping history prevents context overflow.
    """
    return [messages[0]]


def _xml_tool_calls(text):
    """Parse tool calls from any format: XML, bracket notation, etc."""
    from xml_parser import parse_tool_calls
    return parse_tool_calls(text)


def process_message_with_tools(
    llm,
    messages: list[dict],
    executor,
    tools: list[dict],
    config: Config,
) -> str:
    """
    Send messages to LLM, handle tool calls in a loop, return final text response.

    Supports both native JSON function calling and XML-style ``<tool_call>`` blocks
    (fallback for small models like Qwen3.5-0.8B).

    Mutates *messages* in place — the conversation history is extended with
    assistant + tool result messages as they happen.
    """
    for _ in range(config.MAX_TOOL_LOOPS):
        trimmed = _trim_messages(messages)
        resp = llm.create_chat_completion(
            messages=trimmed,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )

        choice = resp["choices"][0]
        msg = choice["message"]
        content = (msg.get("content") or "").strip()

        # Path 1: native JSON tool calls (larger models)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls
                ],
            })
            results = executor.run_openai(tool_calls)
            messages.extend(results)
            for tc in tool_calls:
                name = tc["function"]["name"]
                print(f"  {C.TOOL}>> tool: {name}{C.RESET}")
            continue

        # Path 2: XML-style tool_call fallback (Qwen3.5-0.8B etc.)
        xml_calls = _xml_tool_calls(content) if content else []
        if xml_calls:
            messages.append({"role": "assistant", "content": content})
            for tc in xml_calls:
                name = tc["name"]
                args = tc["arguments"]
                try:
                    result = executor.execute(name, args)
                    print(f"  {C.TOOL}>> tool: {name}{C.RESET}")
                except Exception as e:
                    result = f"Error: {e}"
                    print(f"  {C.TOOL}>> tool: {name} FAILED: {e}{C.RESET}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"xml-{name}",
                    "content": result,
                })
            continue

        # Path 3: plain text response
        messages.append({"role": "assistant", "content": content})
        return content

    return "(tool loop limit reached)"


# ─── Main demo ─────────────────────────────────────────────────────────────

class FunctionCallingDemo:
    def __init__(self, config: Config):
        self.cfg = config
        self.llm = get_llm(config)
        self.executor = get_executor(config)
        self.tools = get_tool_definitions()

        # Load chunks
        print(f"{C.INFO}Loading text from {Path(config.TEXT_FILE).name}...{C.RESET}")
        self.chunks = load_chunks(config.TEXT_FILE, chunk_size=config.CHUNK_SIZE)
        total_chunks = len(self.chunks)
        limit = config.MAX_CHUNKS
        if limit and limit < total_chunks:
            self.chunks = self.chunks[:limit]
            total_chunks = limit
        print(f"{C.INFO}Loaded {total_chunks} chunks ({config.CHUNK_SIZE} paragraphs each).{C.RESET}\n")

        # Conversation history
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # State
        self.chunk_index = 0
        self.total_chunks = total_chunks
        self.paused = False

    # ── Auto-feed ──────────────────────────────────────────────────────

    def feed_all(self):
        """Feed all chunks automatically."""
        self._print_banner()

        while self.chunk_index < self.total_chunks:
            if self.paused:
                time.sleep(0.5)
                continue

            chunk = self.chunks[self.chunk_index]
            self._process_chunk(chunk)

            # Small delay so output is readable
            time.sleep(0.3)

    def feed_one(self, chunk: dict) -> str:
        """Feed a single chunk and return the agent's response."""
        return self._process_chunk(chunk)

    def _process_chunk(self, chunk: dict) -> str:
        """Process one chunk through the tool-calling loop."""
        n = chunk["id"] + 1
        text = chunk["text"]
        preview = text[:120].replace("\n", " ") + ("..." if len(text) > 120 else "")

        print(f"\n{C.PARA}-- Chunk {n}/{self.total_chunks} (paras {chunk['start_pid']}-{chunk['end_pid']}) --{C.RESET}")
        print(f"{C.PARA}{preview}{C.RESET}")

        self.messages.append({"role": "user", "content": text})

        print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
        reply = process_message_with_tools(
            self.llm, self.messages, self.executor, self.tools, self.cfg
        )
        print(reply)
        print()

        # Truncate history to prevent context overflow across chunks
        self.messages = _truncate_after_chunk(self.messages)

        self.chunk_index += 1
        return reply

    # ── Interactive Q&A ────────────────────────────────────────────────

    def handle_command(self, cmd: str) -> bool:
        """Handle a /command. Returns False if should exit."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            return False

        elif command == "/status":
            agent = get_agent(self.cfg)
            s = agent.status()
            print(f"\n{C.SYSTEM}-- Memory Status --{C.RESET}")
            for k, v in s.items():
                print(f"  {k:<24}: {v}")
            print(f"  {'chunks_processed':<24}: {self.chunk_index}/{self.total_chunks}")
            print()
            return True

        elif command == "/history":
            agent = get_agent(self.cfg)
            window = agent.get_stm_window()
            print(f"\n{C.SYSTEM}-- STM Window --{C.RESET}")
            print(window or "  (empty)")
            print()
            return True

        elif command == "/recall":
            if arg:
                agent = get_agent(self.cfg)
                results = agent.recall(arg, top_k=5)
                print(f"\n{C.SYSTEM}-- Recall: {arg} --{C.RESET}")
                for i, r in enumerate(results):
                    print(f"  {i+1}. [{r.score:.3f}] {r.entry.content[:200]}...")
                print()
            else:
                print(f"{C.ERROR}Usage: /recall <query>{C.RESET}")
            return True

        elif command == "/entities":
            agent = get_agent(self.cfg)
            entities = agent.entities.all()
            print(f"\n{C.SYSTEM}-- Entities ({len(entities)}) --{C.RESET}")
            for e in entities[:20]:
                print(f"  * {e.name}: {e.content[:100]}")
            if len(entities) > 20:
                print(f"  ... and {len(entities) - 20} more")
            print()
            return True

        elif command == "/feed":
            # Resume feeding after interactive Q&A
            print(f"{C.SYSTEM}Resuming feed...{C.RESET}")
            while self.chunk_index < self.total_chunks:
                chunk = self.chunks[self.chunk_index]
                self._process_chunk(chunk)
                time.sleep(0.3)
            print(f"{C.SYSTEM}All chunks processed. Back to Q&A.{C.RESET}")
            return True

        elif command == "/pause":
            self.paused = True
            print(f"{C.SYSTEM}Paused. Use /resume to continue.{C.RESET}")
            return True

        elif command == "/resume":
            self.paused = False
            print(f"{C.SYSTEM}Resumed.{C.RESET}")
            return True

        elif command == "/skip":
            n = int(arg) if arg.isdigit() else 1
            self.chunk_index = min(self.chunk_index + n, self.total_chunks)
            print(f"{C.SYSTEM}Skipped to chunk {self.chunk_index}/{self.total_chunks}.{C.RESET}")
            return True

        elif command == "/jump":
            if arg.isdigit():
                target = int(arg)
                self.chunk_index = min(target, self.total_chunks)
                print(f"{C.SYSTEM}Jumped to chunk {self.chunk_index}/{self.total_chunks}.{C.RESET}")
            else:
                print(f"{C.ERROR}Usage: /jump <chunk_number>{C.RESET}")
            return True

        else:
            print(f"{C.ERROR}Unknown command: {command}{C.RESET}")
            return True

    def run_interactive(self):
        """Interactive Q&A after all chunks are processed."""
        print(f"\n{C.SYSTEM}====== Q&A Mode ======{C.RESET}")
        agent = get_agent(self.cfg)
        s = agent.status()
        print(f"Memory: {s.get('ltm_entries', '?')} LTM entries, {s.get('entities', '?')} entities")
        print(f"Type a question, /recall <query>, or /commands. /quit to exit.\n")

        while True:
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

            # Send as a normal message (with tools)
            print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
            self.messages.append({"role": "user", "content": user_input})
            reply = process_message_with_tools(
                self.llm, self.messages, self.executor, self.tools, self.cfg
            )
            print(reply)
            print()

    def _print_banner(self):
        info = f"""
{C.BOLD}================================================================
  demo  --  function-calling agent  --  {Path(self.cfg.TEXT_FILE).name}
================================================================{C.RESET}

{C.PARA}The agent receives text passages as user messages and autonomously
decides what to remember using memory tools.{C.RESET}

{C.INFO}Chunks:{C.RESET}     {self.total_chunks} ({self.cfg.CHUNK_SIZE} paragraphs each)
{C.INFO}Chat LLM:{C.RESET}  {Path(self.cfg.CHAT_MODEL).name}
{C.INFO}Tools:{C.RESET}     record_stm, consolidate_ltm, create_entity,
           observe_entity, resolve_entity, recall, get_stm_window,
           record_source, update_source_description

Commands:
  {C.DIM}/status{C.RESET}    -- show memory stats
  {C.DIM}/history{C.RESET}   -- show STM window
  {C.DIM}/recall <q>{C.RESET} -- manually query memory
  {C.DIM}/entities{C.RESET}  -- list known entities
  {C.DIM}/pause{C.RESET}     -- pause feed
  {C.DIM}/skip [N]{C.RESET}  -- skip N chunks
  {C.DIM}/quit{C.RESET}      -- exit
--------------------------------------------------------------
        """
        print(info.strip())


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Demo: function-calling agent with memory tools")
    parser.add_argument("--chat-model", default=Config.CHAT_MODEL)
    parser.add_argument("--embed-model", default=Config.EMBED_MODEL)
    parser.add_argument("--text", default=Config.TEXT_FILE)
    parser.add_argument("--chunk-size", type=int, default=Config.CHUNK_SIZE)
    parser.add_argument("--max-chunks", type=int, default=Config.MAX_CHUNKS)
    args = parser.parse_args()

    config = Config()
    config.CHAT_MODEL   = args.chat_model
    config.EMBED_MODEL  = args.embed_model
    config.TEXT_FILE    = args.text
    config.CHUNK_SIZE   = args.chunk_size
    config.MAX_CHUNKS   = args.max_chunks

    for label, path in [("Chat model", config.CHAT_MODEL),
                        ("Embed model", config.EMBED_MODEL),
                        ("Text file", config.TEXT_FILE)]:
        if not Path(path).exists():
            print(f"{C.ERROR}[ERROR] {label} not found: {path}{C.RESET}")
            sys.exit(1)

    demo = FunctionCallingDemo(config)
    try:
        demo.feed_all()
        demo.run_interactive()
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Shutting down...{C.RESET}")


if __name__ == "__main__":
    main()
