"""
example_anthropic.py — Full agent loop using memory tools with Claude.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-...
    python example_anthropic.py
"""

import anthropic
from memory_module import MemoryAgent
from memory_module.tools import get_tools, ToolExecutor

# ── Setup ────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

agent    = MemoryAgent("agent_memory.db", max_stm_segments=10)
executor = ToolExecutor(agent)
tools    = get_tools(format="anthropic")

SYSTEM = """You are a personal AI assistant with persistent memory.
You have access to memory tools — use them proactively:
  - Record every user turn and important assistant response with record_stm.
  - Use recall before answering questions that might touch past context.
  - Consolidate important facts (names, preferences, events) with consolidate_ltm.
  - Create entities for significant people or things you encounter.
  - resolve_entity before creating a new entity — it may already exist.

Be natural. Don't announce every memory operation. Just do it.
"""


# ── Agent loop ───────────────────────────────────────────────────────────────

def chat(user_input: str, messages: list) -> str:
    # Record user turn immediately
    agent.record_stm(f"User: {user_input}")

    messages.append({"role": "user", "content": user_input})

    # Inject current STM window into a fresh context retrieval
    relevant = agent.recall(user_input, top_k=3)
    stm_ctx  = agent.get_stm_window()

    # Agentic tool-use loop
    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM + f"\n\nCurrent STM:\n{stm_ctx}"
                          + (f"\n\nRelevant LTM:\n" +
                             "\n".join(r.entry.content for r in relevant)
                             if relevant else ""),
            tools=tools,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we're done
        if response.stop_reason != "tool_use":
            break

        # Execute all tool calls and feed results back
        tool_results = executor.run_anthropic(response.content)
        messages.append({"role": "user", "content": tool_results})

    # Extract final text
    final = next(
        (b.text for b in response.content if hasattr(b, "text")), ""
    )

    # Record assistant response
    agent.record_stm(f"Assistant: {final[:200]}")

    return final


# ── Demo conversation ────────────────────────────────────────────────────────

if __name__ == "__main__":
    messages = []

    turns = [
        "Hi! My name is Musa.",
        "I work on robotics at a university lab.",
        "What do you remember about me so far?",
        "I prefer concise responses by the way.",
        "What's my name and what do I do?",
    ]

    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        reply = chat(user_msg, messages)
        print(f"Assistant: {reply}")

    print("\n── Memory Status ──")
    print(agent.status())
