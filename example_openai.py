"""
example_openai.py — Full agent loop using memory tools with GPT-4o.

Run:
    pip install openai
    export OPENAI_API_KEY=sk-...
    python example_openai.py
"""

import json
from openai import OpenAI
from memory_module import MemoryAgent
from memory_module.tools import get_tools, ToolExecutor

# ── Setup ────────────────────────────────────────────────────────────────────

client   = OpenAI()
agent    = MemoryAgent("agent_memory.db", max_stm_segments=10)
executor = ToolExecutor(agent)
tools    = get_tools(format="openai")

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
    agent.record_stm(f"User: {user_input}")

    relevant = agent.recall(user_input, top_k=3)
    stm_ctx  = agent.get_stm_window()

    system_with_ctx = (
        SYSTEM
        + f"\n\nCurrent STM:\n{stm_ctx}"
        + (f"\n\nRelevant LTM:\n" + "\n".join(r.entry.content for r in relevant)
           if relevant else "")
    )

    messages.append({"role": "user", "content": user_input})

    # Agentic tool-use loop
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=tools,
            messages=[{"role": "system", "content": system_with_ctx}] + messages,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            break

        # Execute tool calls and append results
        tool_results = executor.run_openai(msg.tool_calls)
        messages.extend(tool_results)

    final = response.choices[0].message.content or ""
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
