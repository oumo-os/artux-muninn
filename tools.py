"""
tools.py — LLM-callable tool definitions for the Memory Module.

Provides two things:
  1. Tool schemas in both Anthropic and OpenAI formats, ready to pass
     directly to your LLM API call.
  2. A ToolExecutor that receives tool_use blocks / tool_calls from
     the LLM response and dispatches them to the MemoryAgent.

Typical usage
-------------
    from memory_module import MemoryAgent
    from memory_module.tools import get_tools, ToolExecutor

    agent = MemoryAgent("memory.db")
    executor = ToolExecutor(agent)

    # --- Anthropic ---
    tools = get_tools(format="anthropic")
    response = client.messages.create(model=..., tools=tools, messages=...)
    results = executor.run_anthropic(response.content)

    # --- OpenAI ---
    tools = get_tools(format="openai")
    response = client.chat.completions.create(model=..., tools=tools, messages=...)
    results = executor.run_openai(response.choices[0].message.tool_calls)
"""

from __future__ import annotations
import json
from typing import Any

from .agent import MemoryAgent


# ────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ────────────────────────────────────────────────────────────────────────────

_TOOL_SPECS = [
    {
        "name": "recall",
        "description": (
            "Search long-term memory using a natural language query. "
            "Combines structured filtering (entity, topic, concept, time) "
            "with semantic similarity. Returns the most relevant memory entries. "
            "Use this when you need to retrieve facts, past events, or information "
            "about a person or topic from previous interactions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query. Optionally prefix with an operator "
                        "(who / what / where / when / how / why / dispute) and use "
                        "quoted phrases for exact topic matches. "
                        "Supports after:YYYY-MM-DD and before:YYYY-MM-DD constraints. "
                        "Examples: 'who is Musa', 'what does Musa do for work', "
                        "'where was John after:2024-01-01'"
                    )
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "record_stm",
        "description": (
            "Record a new observation into Short-Term Memory. "
            "Call this after every user turn, assistant turn, or notable event "
            "to maintain a running context window. "
            "STM is automatically compressed into summaries when it grows large."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The content to record. Be concise but specific. "
                        "Examples: 'User said: my name is Musa', "
                        "'User confirmed they work on robotics', "
                        "'System: user emotion appears frustrated'"
                    )
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "consolidate_ltm",
        "description": (
            "Consolidate important information from the current session into "
            "Long-Term Memory so it persists across future conversations. "
            "Use this when something significant has been learned or confirmed — "
            "a person's name, a preference, an important event, a decision made. "
            "Do not over-consolidate; focus on durable, reusable facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "narrative": {
                    "type": "string",
                    "description": (
                        "A clear, self-contained summary of what should be remembered. "
                        "Write it so it makes sense when read in isolation, months later. "
                        "Example: 'User's name is Musa (initially said Kyle, corrected by "
                        "Sam and John). Works on robotics. Cheerful and curious personality.'"
                    )
                },
                "class_type": {
                    "type": "string",
                    "enum": ["event", "assertion", "decision", "procedure", "observation"],
                    "description": "Type of memory entry.",
                    "default": "assertion"
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords for filtering. Example: ['identity', 'robotics']"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in this information (0.0–1.0). Default 0.9.",
                    "default": 0.9
                }
            },
            "required": ["narrative"]
        }
    },
    {
        "name": "create_entity",
        "description": (
            "Create a new entity (person, object, or concept) in memory. "
            "Call this the first time you encounter a significant person or thing "
            "that you expect to reference again in the future. "
            "The entity becomes a living historical ledger — observations can be "
            "appended to it over time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The canonical name or label for this entity."
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Initial description or observation about this entity. "
                        "Example: 'Described self as Musa. Works on robotics. Cheerful.'"
                    )
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Associated keywords. Example: ['person', 'robotics']"
                }
            },
            "required": ["name", "description"]
        }
    },
    {
        "name": "observe_entity",
        "description": (
            "Append a new observation to an existing entity's historical ledger. "
            "Use this when you learn something new about a known person or thing. "
            "Contradictions are preserved — do not worry about overwriting old information. "
            "You must have the entity_id from a prior create_entity or resolve_entity call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The entity's unique ID."
                },
                "observation": {
                    "type": "string",
                    "description": "What was observed or learned."
                },
                "authority": {
                    "type": "string",
                    "enum": ["self", "peer", "system", "anchor"],
                    "description": (
                        "Source authority. 'self' = subject self-reported, "
                        "'peer' = another person said it, "
                        "'system' = automated/verified source, "
                        "'anchor' = explicitly trusted authority."
                    ),
                    "default": "peer"
                },
                "memory_ref": {
                    "type": "string",
                    "description": "Optional: reference to the STM segment this came from (e.g. 't3')."
                }
            },
            "required": ["entity_id", "observation"]
        }
    },
    {
        "name": "resolve_entity",
        "description": (
            "Search for an existing entity in memory using a natural language description. "
            "Use this before creating a new entity to check if one already exists, "
            "or when you need to find the entity_id for a person mentioned in conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "clues": {
                    "type": "string",
                    "description": (
                        "Natural language description to match against known entities. "
                        "Example: 'the person who works on robotics', 'Musa', "
                        "'the user who mentioned a name correction'"
                    )
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of candidate matches to return (default: 3).",
                    "default": 3
                }
            },
            "required": ["clues"]
        }
    },
    {
        "name": "get_stm_window",
        "description": (
            "Retrieve the current Short-Term Memory window as a formatted string. "
            "Use this to inspect what the agent currently holds in immediate context, "
            "or to manually review recent events before consolidation."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "record_source",
        "description": (
            "Register an external knowledge source — an image, audio file, PDF, "
            "webpage, or any other asset — and attach it to an LTM entry. "
            "Call this immediately after consolidating a perception that came from "
            "a non-text source, so future recall can return the source location "
            "alongside the text summary. "
            "The memory module stores the location and the description you provide. "
            "It does NOT read or process the source itself. "
            "The description should be what you already derived from the source. "
            "If a future recall returns this source and the text is insufficient "
            "to answer the question, the agent should re-examine the source at "
            "the returned location using the appropriate tool (VLM, ASR, PDF reader, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ltm_entry_id": {
                    "type": "string",
                    "description": "The LTM entry ID this source backs. Get this from consolidate_ltm."
                },
                "location": {
                    "type": "string",
                    "description": (
                        "File path or URL of the source asset. "
                        "Examples: '/data/images/2024-03-01_living_room.jpg', "
                        "'https://example.com/article', "
                        "'s3://my-bucket/audio/meeting_2024.mp3'"
                    )
                },
                "type": {
                    "type": "string",
                    "enum": ["image", "audio", "video", "pdf", "webpage", "file", "remote"],
                    "description": "Type of the source asset.",
                    "default": "file"
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Your current text summary of the source content — "
                        "what you already know from it. Be specific: include objects, "
                        "positions, colours, people, timestamps, key facts. "
                        "Example: 'Living room table at 14:22. Red doll on left side. "
                        "Blue ceramic cup on right. Yellow notepad in centre.'"
                    )
                },
                "meta": {
                    "type": "object",
                    "description": (
                        "Optional freeform metadata dict. "
                        "Images: {'width': 1920, 'height': 1080}. "
                        "Audio: {'duration_s': 142, 'language': 'en'}. "
                        "PDFs: {'page_count': 12}."
                    )
                }
            },
            "required": ["ltm_entry_id", "location", "description"]
        }
    },
    {
        "name": "update_source_description",
        "description": (
            "Update the stored description of a source after re-examining it with "
            "a VLM, ASR system, PDF reader, or web fetcher. "
            "Call this after getting richer detail from the source so future recalls "
            "benefit from the improved description without re-examining the source again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "The source ID returned by record_source."
                },
                "new_description": {
                    "type": "string",
                    "description": "The updated, richer description of the source content."
                }
            },
            "required": ["source_id", "new_description"]
        }
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Format conversion
# ────────────────────────────────────────────────────────────────────────────

def get_tools(format: str = "anthropic") -> list[dict]:
    """
    Return tool definitions ready to pass to your LLM API.

    format: "anthropic" | "openai"
    """
    if format == "anthropic":
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in _TOOL_SPECS
        ]
    elif format == "openai":
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
            }
            for t in _TOOL_SPECS
        ]
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'anthropic' or 'openai'.")


# ────────────────────────────────────────────────────────────────────────────
# Tool Executor
# ────────────────────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Receives tool_use blocks / tool_calls from an LLM response
    and dispatches them to the MemoryAgent.

    Returns results as plain strings suitable for feeding back
    to the LLM as tool_result / assistant messages.
    """

    def __init__(self, agent: MemoryAgent):
        self.agent = agent

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a single tool call by name with the given input dict.
        Returns a plain string result.
        Raises ValueError for unknown tool names.
        """
        dispatch = {
            "recall":                    self._recall,
            "record_stm":                self._record_stm,
            "consolidate_ltm":           self._consolidate_ltm,
            "create_entity":             self._create_entity,
            "observe_entity":            self._observe_entity,
            "resolve_entity":            self._resolve_entity,
            "get_stm_window":            self._get_stm_window,
            "record_source":             self._record_source,
            "update_source_description": self._update_source_description,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown memory tool: {tool_name!r}")
        return fn(tool_input)

    # ------------------------------------------------------------------
    # Anthropic response handling
    # ------------------------------------------------------------------

    def run_anthropic(
        self,
        content_blocks: list,
    ) -> list[dict]:
        """
        Process all tool_use blocks from an Anthropic response.
        Returns a list of tool_result dicts ready to send back as
        the next user message:

            messages.append({"role": "user", "content": results})

        Usage:
            response = client.messages.create(...)
            results  = executor.run_anthropic(response.content)
            messages += [
                {"role": "assistant", "content": response.content},
                {"role": "user",      "content": results},
            ]
        """
        results = []
        for block in content_blocks:
            if hasattr(block, "type") and block.type == "tool_use":
                try:
                    output = self.execute(block.name, block.input)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
                except Exception as e:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    })
        return results

    # ------------------------------------------------------------------
    # OpenAI response handling
    # ------------------------------------------------------------------

    def run_openai(self, tool_calls: list | None) -> list[dict]:
        """
        Process all tool_calls from an OpenAI response.
        Returns a list of tool-role messages ready to append to messages:

            messages.append(response.choices[0].message)
            messages += executor.run_openai(response.choices[0].message.tool_calls)

        Usage:
            response    = client.chat.completions.create(...)
            tool_calls  = response.choices[0].message.tool_calls
            results     = executor.run_openai(tool_calls)
        """
        if not tool_calls:
            return []
        results = []
        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments)
                output = self.execute(call.function.name, args)
                results.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output,
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"Error: {e}",
                })
        return results

    # ------------------------------------------------------------------
    # Individual tool implementations
    # ------------------------------------------------------------------

    def _recall(self, inp: dict) -> str:
        query = inp["query"]
        top_k = inp.get("top_k", 5)
        results = self.agent.recall(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] (confidence={r.entry.confidence:.2f}, score={r.score:.2f})\n"
                f"    {r.entry.content}"
            )
            if r.match_reasons:
                lines.append(f"    matched on: {', '.join(r.match_reasons)}")
            if r.sources:
                lines.append(f"    sources ({len(r.sources)}):")
                for src in r.sources:
                    lines.append(
                        f"      [{src.type}] id={src.id}\n"
                        f"        location: {src.location}\n"
                        f"        captured: {src.captured_at.strftime('%Y-%m-%d %H:%M')}\n"
                        f"        summary:  {src.description[:200]}"
                        + (" [re-examine for more detail]" if len(src.description) < 80 else "")
                    )
        return "\n".join(lines)

    def _record_stm(self, inp: dict) -> str:
        seg = self.agent.record_stm(inp["content"])
        return f"Recorded STM segment (id: {seg.id[:8]}…)"

    def _consolidate_ltm(self, inp: dict) -> str:
        entry = self.agent.consolidate_ltm(
            narrative=inp["narrative"],
            class_type=inp.get("class_type", "assertion"),
            topics=inp.get("topics", []),
            confidence=inp.get("confidence", 0.9),
        )
        return (
            f"Consolidated to LTM (id: {entry.id[:8]}…, "
            f"confidence: {entry.confidence})"
        )

    def _create_entity(self, inp: dict) -> str:
        entity = self.agent.create_entity(
            description=inp["description"],
            name=inp["name"],
            topics=inp.get("topics", []),
        )
        return (
            f"Created entity '{entity.name}' (id: {entity.id}). "
            f"Use this id for future observe_entity calls."
        )


    def _observe_entity(self, inp: dict) -> str:
        entity = self.agent.observe_entity(
            entity_id=inp["entity_id"],
            observation=inp["observation"],
            memory_ref=inp.get("memory_ref", ""),
            authority=inp.get("authority", "peer"),
        )
        return f"Observation appended to entity '{entity.name}'."

    def _resolve_entity(self, inp: dict) -> str:
        matches = self.agent.resolve_entity(
            inp["clues"], top_k=inp.get("top_k", 3)
        )
        if not matches:
            return "No matching entities found."
        lines = []
        for entity, score in matches:
            lines.append(
                f"  id={entity.id}  name='{entity.name}'  score={score:.2f}\n"
                f"    {entity.content[:120]}…"
            )
        return "Matching entities:\n" + "\n".join(lines)

    def _get_stm_window(self, inp: dict) -> str:
        window = self.agent.get_stm_window()
        return window if window else "STM is currently empty."

    def _record_source(self, inp: dict) -> str:
        from datetime import datetime
        ref = self.agent.record_and_attach_source(
            ltm_entry_id=inp["ltm_entry_id"],
            location=inp["location"],
            type=inp.get("type", "file"),
            description=inp.get("description", ""),
            meta=inp.get("meta"),
        )
        return (
            f"Source registered and attached.\n"
            f"  source_id : {ref.id}\n"
            f"  type      : {ref.type}\n"
            f"  location  : {ref.location}\n"
            f"  captured  : {ref.captured_at.strftime('%Y-%m-%d %H:%M')}\n"
            f"Use source_id to update the description after re-examination."
        )

    def _update_source_description(self, inp: dict) -> str:
        ref = self.agent.update_source_description(
            inp["source_id"], inp["new_description"]
        )
        if ref is None:
            return f"Source {inp['source_id']} not found."
        return (
            f"Source description updated (id: {ref.id[:8]}…).\n"
            f"  location: {ref.location}\n"
            f"  new description: {ref.description[:200]}"
        )


