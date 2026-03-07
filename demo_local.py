#!/usr/bin/env python3
"""
demo_local.py — Fully offline perceptual agent with persistent memory.

Identical to demo_live.py in behaviour, but every component runs locally:

  Microphone  → faster-whisper (local STT)
  Webcam      → Ollama VLM  e.g. llava / moondream (local vision)
  LLM agent   → Ollama LLM  e.g. llama3.2 / mistral-nemo (local inference)
  Embeddings  → sentence-transformers all-MiniLM-L6-v2 (local, falls back to TF-IDF)
  Memory      → SQLite  (always local)

Nothing leaves the machine.  No API keys required.

──────────────────────────────────────────────────────────────────────
Requirements
──────────────────────────────────────────────────────────────────────
  1. Install Ollama:  https://ollama.com
     ollama serve                        # must be running

  2. Pull models:
     ollama pull llama3.2                # or mistral-nemo, qwen2.5, etc.
     ollama pull llava                   # or moondream, bakllava, llava-phi3

  3. Python packages:
     pip install faster-whisper sounddevice numpy scipy opencv-python openai

     # Semantic embeddings (recommended — falls back to TF-IDF without it):
     pip install sentence-transformers

──────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────
  python demo_local.py

  # Optional env overrides:
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_LLM=llama3.2
  OLLAMA_VLM=llava

Commands at the prompt:
  status   — memory stats
  history  — current STM window
  models   — show active model names
  decay    — run memory decay + maintenance
  quit     — exit

──────────────────────────────────────────────────────────────────────
Tool-calling compatibility
──────────────────────────────────────────────────────────────────────
Ollama exposes an OpenAI-compatible API so we use the `openai` Python
client pointed at localhost.

Models with reliable tool/function-call support:
  llama3.2  llama3.1  mistral-nemo  qwen2.5  qwen2.5-coder
  phi4  hermes3  command-r

If your chosen model doesn't format tool calls correctly, the agent
falls back to a prompt-based JSON extraction path automatically.
"""

import os
import sys
import json
import time
import wave
import base64
import queue
import re
import signal
import tempfile
import threading
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import numpy as np

# ─── Dependency checks ────────────────────────────────────────────────────────

def _require(pkg, install_hint, attr=None):
    try:
        mod = __import__(pkg)
        return getattr(mod, attr) if attr else mod
    except ImportError:
        print(f"\n[ERROR] Missing package: {pkg}")
        print(f"        pip install {install_hint}")
        sys.exit(1)

sd  = _require("sounddevice", "sounddevice")
cv2 = _require("cv2",         "opencv-python")

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("\n[ERROR] faster-whisper is required for local STT.")
    print("        pip install faster-whisper")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("\n[ERROR] openai client is required to talk to Ollama's API.")
    print("        pip install openai")
    sys.exit(1)

# Memory module
sys.path.insert(0, str(Path(__file__).parent.parent))
from memory_module import MemoryAgent, get_tools, ToolExecutor
from memory_module.embeddings import SEMANTIC_AVAILABLE


# ─── Configuration ────────────────────────────────────────────────────────────

class Config:
    # Ollama
    OLLAMA_HOST         = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_LLM          = os.environ.get("OLLAMA_LLM",  "llama3.2")
    OLLAMA_VLM          = os.environ.get("OLLAMA_VLM",  "llava")

    # Memory
    DB_PATH             = "local_demo_memory.db"
    CAPTURES_DIR        = Path("./captures_local")

    # Audio
    SAMPLE_RATE         = 16_000
    AUDIO_CHUNK_SECS    = 5
    SILENCE_RMS_THRESH  = 0.008
    WHISPER_MODEL       = "base.en"         # tiny.en / base.en / small.en / medium.en
    WHISPER_DEVICE      = "cpu"             # "cuda" if you have a GPU
    WHISPER_COMPUTE     = "int8"            # int8 / float16 / float32

    # Video
    VIDEO_INTERVAL_SECS = 15               # slightly longer than cloud — local VLM is slower
    WEBCAM_INDEX        = 0
    FRAME_QUALITY       = 80

    # LLM inference
    MAX_TOKENS          = 1024
    TEMPERATURE         = 0.4
    TOOL_TIMEOUT_SECS   = 60               # generous timeout for local inference

    # Memory behaviour
    STM_MAX_SEGMENTS    = 20
    AUTO_CONSOLIDATE_N  = 8


# ─── Colour terminal ──────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    AUDIO   = "\033[36m"
    VIDEO   = "\033[35m"
    AGENT   = "\033[32m"
    USER    = "\033[33m"
    ERROR   = "\033[31m"
    INFO    = "\033[34m"

def log(kind: str, msg: str, colour: str = C.INFO):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{colour}[{kind} {ts}]{C.RESET} {msg}")


# ─── Ollama client wrapper ────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around the OpenAI-compatible Ollama API.
    Handles both tool-calling models and fallback JSON extraction
    for models that don't emit tool_calls correctly.
    """

    def __init__(self, config: Config):
        self.cfg    = config
        self.client = OpenAI(
            base_url=f"{config.OLLAMA_HOST}/v1",
            api_key="ollama",       # Ollama ignores this but openai client requires it
        )
        self._tool_calling_works: Optional[bool] = None  # discovered at runtime

    # ── Chat with automatic tool-call fallback ────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> tuple[str, list[dict]]:
        """
        Send a chat turn with tools.

        Returns:
            (final_text, tool_calls_made)

        Where tool_calls_made is a list of {"name": ..., "arguments": {...}} dicts
        extracted from the model response (either native tool_calls or JSON fallback).

        The caller is responsible for executing tool calls and feeding results back.
        This method handles one round of tool elicitation, not the full agentic loop.
        """
        sys_msg = {"role": "system", "content": system}
        full_messages = [sys_msg] + messages

        # First, try with native tool calling if not known to fail
        if self._tool_calling_works is not False:
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.OLLAMA_LLM,
                    messages=full_messages,
                    tools=tools,
                    temperature=self.cfg.TEMPERATURE,
                    timeout=self.cfg.TOOL_TIMEOUT_SECS,
                )
                msg   = resp.choices[0].message
                calls = msg.tool_calls or []

                if calls:
                    self._tool_calling_works = True
                    parsed = [
                        {
                            "id":        c.id,
                            "name":      c.function.name,
                            "arguments": json.loads(c.function.arguments),
                        }
                        for c in calls
                    ]
                    return (msg.content or ""), parsed

                # Model responded with text but no tool calls — that's fine too
                self._tool_calling_works = True
                return (msg.content or ""), []

            except Exception as e:
                # Could be a model that doesn't support tools parameter at all
                log("LLM", f"Native tool call failed ({e}), switching to JSON fallback", C.DIM)
                self._tool_calling_works = False

        # Fallback: ask model to respond with JSON-encoded tool calls
        return self._json_fallback(full_messages, tools)

    def chat_plain(self, messages: list[dict], system: str) -> str:
        """Plain (no tools) chat — for compression, vision description, etc."""
        resp = self.client.chat.completions.create(
            model=self.cfg.OLLAMA_LLM,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=self.cfg.TEMPERATURE,
            timeout=self.cfg.TOOL_TIMEOUT_SECS,
        )
        return resp.choices[0].message.content.strip()

    def describe_image(self, b64_image: str, prompt: str) -> str:
        """Send an image to the VLM for description."""
        resp = self.client.chat.completions.create(
            model=self.cfg.OLLAMA_VLM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.2,
            timeout=self.cfg.TOOL_TIMEOUT_SECS,
        )
        return resp.choices[0].message.content.strip()

    # ── JSON fallback tool calling ────────────────────────────────────

    _FALLBACK_SYSTEM_SUFFIX = """
── Tool Calling Instructions ──
You have access to memory tools. To call a tool, output ONLY a JSON object:

{"tool": "<tool_name>", "arguments": {<args>}}

To call multiple tools, output a JSON array:

[{"tool": "<tool_name>", "arguments": {<args>}}, ...]

After all tool calls are made and you have the information you need,
output your final response as plain text (no JSON).

If you don't need to call any tools, just respond in plain text directly.
"""

    def _tool_schema_summary(self, tools: list[dict]) -> str:
        """Compact tool reference for the fallback prompt."""
        lines = ["Available tools:"]
        for t in tools:
            fn   = t.get("function", t)   # handles both OpenAI and raw formats
            name = fn.get("name", "?")
            desc = fn.get("description", "")[:100]
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_str = ", ".join(
                f"{k}{'*' if k in required else '?'}"
                for k in params
            )
            lines.append(f"  {name}({param_str}) — {desc}")
        return "\n".join(lines)

    def _json_fallback(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[str, list[dict]]:
        tool_summary = self._tool_schema_summary(tools)
        system = tool_summary + self._FALLBACK_SYSTEM_SUFFIX

        resp = self.client.chat.completions.create(
            model=self.cfg.OLLAMA_LLM,
            messages=messages,
            system=system,
            temperature=self.cfg.TEMPERATURE,
            timeout=self.cfg.TOOL_TIMEOUT_SECS,
        )
        raw = resp.choices[0].message.content or ""
        return self._extract_tool_calls(raw)

    @staticmethod
    def _extract_tool_calls(raw: str) -> tuple[str, list[dict]]:
        """
        Parse tool calls from raw LLM output using bracket-depth scanning.
        Handles nested JSON objects correctly (e.g. {"arguments": {"query": "..."}}).
        Returns (remaining_text, list_of_parsed_calls).
        """
        calls     = []
        consumed  = set()   # character ranges already parsed

        def find_json_spans(text: str) -> list[tuple[int, int]]:
            """Find start/end indices of all top-level JSON objects and arrays."""
            spans = []
            i = 0
            while i < len(text):
                if text[i] in ('{', '['):
                    opener = text[i]
                    closer = '}' if opener == '{' else ']'
                    depth  = 0
                    in_str = False
                    esc    = False
                    start  = i
                    for j in range(i, len(text)):
                        ch = text[j]
                        if esc:
                            esc = False
                        elif ch == '\\' and in_str:
                            esc = True
                        elif ch == '"':
                            in_str = not in_str
                        elif not in_str:
                            if ch == opener:
                                depth += 1
                            elif ch == closer:
                                depth -= 1
                                if depth == 0:
                                    spans.append((start, j + 1))
                                    i = j + 1
                                    break
                    else:
                        i += 1
                else:
                    i += 1
            return spans

        spans = find_json_spans(raw)
        removed_ranges = []

        for start, end in spans:
            candidate = raw[start:end]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    calls.append({
                        "id":        f"fallback-{len(calls)}",
                        "name":      parsed["tool"],
                        "arguments": parsed.get("arguments", {}),
                    })
                    removed_ranges.append((start, end))
                elif isinstance(parsed, list):
                    found_any = False
                    for item in parsed:
                        if isinstance(item, dict) and "tool" in item:
                            calls.append({
                                "id":        f"fallback-{len(calls)}",
                                "name":      item["tool"],
                                "arguments": item.get("arguments", {}),
                            })
                            found_any = True
                    if found_any:
                        removed_ranges.append((start, end))
            except (json.JSONDecodeError, KeyError):
                continue

        # Reconstruct remaining text by removing parsed JSON spans
        remaining = raw
        for start, end in sorted(removed_ranges, reverse=True):
            remaining = remaining[:start] + remaining[end:]

        return remaining.strip(), calls


# ─── Perception event ─────────────────────────────────────────────────────────

@dataclass
class PerceptionEvent:
    kind: str
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    source_location: Optional[str] = None
    source_meta: dict = field(default_factory=dict)


# ─── Audio thread ─────────────────────────────────────────────────────────────

class AudioThread(threading.Thread):
    def __init__(self, perc_queue: queue.Queue, config: Config):
        super().__init__(daemon=True, name="AudioThread")
        self.queue      = perc_queue
        self.cfg        = config
        self.stop_event = threading.Event()

        log("AUDIO", f"Loading Whisper ({config.WHISPER_MODEL})…", C.INFO)
        self._whisper = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )
        log("AUDIO", "Whisper ready.", C.INFO)

    def run(self):
        log("AUDIO", "Microphone active — listening…", C.AUDIO)
        while not self.stop_event.is_set():
            try:
                self._record_and_process()
            except Exception as e:
                log("AUDIO", f"Error: {e}", C.ERROR)
                time.sleep(1)

    def stop(self):
        self.stop_event.set()

    def _record_and_process(self):
        n_samples = int(self.cfg.AUDIO_CHUNK_SECS * self.cfg.SAMPLE_RATE)
        audio = sd.rec(n_samples, samplerate=self.cfg.SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.cfg.SILENCE_RMS_THRESH:
            return

        text = self._transcribe(audio)
        if not text:
            return

        log("AUDIO", f'"{text}"', C.AUDIO)
        self.queue.put(PerceptionEvent(kind="audio", content=text))

    def _transcribe(self, audio: np.ndarray) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        try:
            segments, _ = self._whisper.transcribe(tmp.name, beam_size=5)
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            os.unlink(tmp.name)


# ─── Video thread ─────────────────────────────────────────────────────────────

class VideoThread(threading.Thread):
    VISION_PROMPT = (
        "Describe this webcam frame precisely and concisely. "
        "Include: all visible objects and their exact positions "
        "(left/right/centre/foreground/background/on top of/next to), "
        "colours, any people and what they are doing, "
        "spatial relationships between objects, and anything unusual. "
        "Be factual and specific. 2-4 sentences."
    )

    def __init__(self, perc_queue: queue.Queue, config: Config,
                 ollama: OllamaClient):
        super().__init__(daemon=True, name="VideoThread")
        self.queue      = perc_queue
        self.cfg        = config
        self.ollama     = ollama
        self.stop_event = threading.Event()
        config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        log("VIDEO", f"Webcam active — describing every {self.cfg.VIDEO_INTERVAL_SECS}s "
                     f"via {self.cfg.OLLAMA_VLM}…", C.VIDEO)
        while not self.stop_event.is_set():
            try:
                self._capture_and_process()
            except Exception as e:
                log("VIDEO", f"Error: {e}", C.ERROR)
            for _ in range(self.cfg.VIDEO_INTERVAL_SECS * 4):
                if self.stop_event.is_set():
                    break
                time.sleep(0.25)

    def stop(self):
        self.stop_event.set()

    def _capture_and_process(self):
        cap = cv2.VideoCapture(self.cfg.WEBCAM_INDEX)
        try:
            if not cap.isOpened():
                log("VIDEO", "Cannot open webcam — skipping", C.ERROR)
                return
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            return

        ts_str    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.cfg.CAPTURES_DIR / f"frame_{ts_str}.jpg"
        cv2.imwrite(str(save_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.cfg.FRAME_QUALITY])

        _, buf    = cv2.imencode(".jpg", frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, self.cfg.FRAME_QUALITY])
        b64_image = base64.b64encode(buf).decode("utf-8")
        h, w      = frame.shape[:2]

        description = self.ollama.describe_image(b64_image, self.VISION_PROMPT)
        if not description:
            return

        log("VIDEO", description[:120] + ("…" if len(description) > 120 else ""), C.VIDEO)
        self.queue.put(PerceptionEvent(
            kind="video",
            content=description,
            source_location=str(save_path.resolve()),
            source_meta={"width": w, "height": h, "mime": "image/jpeg",
                         "saved_at": ts_str, "vlm": self.cfg.OLLAMA_VLM},
        ))


# ─── Perception coordinator ───────────────────────────────────────────────────

class PerceptionCoordinator(threading.Thread):
    def __init__(self, perc_queue: queue.Queue, agent: MemoryAgent,
                 ollama: OllamaClient, config: Config):
        super().__init__(daemon=True, name="Coordinator")
        self.queue      = perc_queue
        self.agent      = agent
        self.ollama     = ollama
        self.cfg        = config
        self.stop_event = threading.Event()
        self._count     = 0
        self._last_video: Optional[PerceptionEvent] = None
        self._lock      = threading.Lock()

    def run(self):
        while not self.stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.5)
                self._process(event)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log("MEM", f"Coordinator error: {e}", C.ERROR)

    def stop(self):
        self.stop_event.set()

    def _process(self, event: PerceptionEvent):
        ts = event.timestamp.strftime("%H:%M:%S")
        with self._lock:
            if event.kind == "audio":
                self.agent.record_stm(f"[{ts}] Heard: {event.content}")
            elif event.kind == "video":
                self.agent.record_stm(f"[{ts}] Saw: {event.content}")
                self._last_video = event

            self._count += 1
            if self._count >= self.cfg.AUTO_CONSOLIDATE_N:
                self._auto_consolidate()
                self._count = 0

    def _auto_consolidate(self):
        try:
            entry = self.agent.consolidate_ltm(
                class_type="observation",
                confidence=0.85,
            )
            log("MEM", f"Auto-consolidated → {entry.id[:8]}…", C.INFO)

            if self._last_video and self._last_video.source_location:
                self.agent.record_and_attach_source(
                    ltm_entry_id=entry.id,
                    location=self._last_video.source_location,
                    type="image",
                    description=self._last_video.content,
                    captured_at=self._last_video.timestamp,
                    meta=self._last_video.source_meta,
                )
                log("MEM", f"Source attached: "
                            f"{Path(self._last_video.source_location).name}", C.INFO)
        except Exception as e:
            log("MEM", f"Auto-consolidation error: {e}", C.ERROR)


# ─── Agent session ────────────────────────────────────────────────────────────

AGENT_SYSTEM = """\
You are a perceptual AI agent with real-time persistent memory running fully offline.

You receive a continuous stream of:
  • Audio: transcribed speech from a microphone (via Whisper)
  • Video: descriptions of webcam frames (via local VLM), with saved image paths

Use your memory tools to answer questions about what you have seen and heard.

When answering:
1. Use recall() first before answering anything involving past observations.
2. If a recall result includes a source image path and the stored description
   is not detailed enough to answer the question, say so explicitly:
   name the image path and what spatial or visual detail you would need
   from re-examining it. The user's system can then pass the image to
   a VLM for deeper interrogation and call update_source_description()
   with the richer detail.
3. Be honest about uncertainty. Don't hallucinate details not in memory.
4. Be concise. Don't narrate tool use — just use the tools and answer.
"""

# OpenAI tool format for Ollama
def _get_ollama_tools() -> list[dict]:
    return get_tools(format="openai")


class AgentSession:
    def __init__(self, agent: MemoryAgent, ollama: OllamaClient, config: Config):
        self.agent    = agent
        self.ollama   = ollama
        self.cfg      = config
        self.executor = ToolExecutor(agent)
        self.tools    = _get_ollama_tools()
        self.messages: list[dict] = []

    def chat(self, user_input: str) -> str:
        stm_ctx  = self.agent.get_stm_window()
        relevant = self.agent.recall(user_input, top_k=4)
        ltm_ctx  = "\n".join(
            f"• [{r.entry.class_type}] {r.entry.content}"
            + (f"\n  sources: {', '.join(s.location for s in r.sources)}"
               if r.sources else "")
            for r in relevant
        )

        system = AGENT_SYSTEM
        if stm_ctx:
            system += f"\n\n── Current STM ──\n{stm_ctx}"
        if ltm_ctx:
            system += f"\n\n── Relevant LTM ──\n{ltm_ctx}"

        self.messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep going while the model is making tool calls
        max_rounds = 6   # guard against runaway loops with local models
        for _ in range(max_rounds):
            text, calls = self.ollama.chat_with_tools(
                messages=self.messages,
                tools=self.tools,
                system=system,
            )

            if not calls:
                # No tool calls — final answer
                self.messages.append({"role": "assistant", "content": text})
                return text

            # Execute tool calls and collect results
            tool_result_msgs = []
            for call in calls:
                log("LLM", f"→ {call['name']}({list(call['arguments'].keys())})", C.DIM)
                try:
                    result = self.executor.execute(call["name"], call["arguments"])
                except Exception as e:
                    result = f"Error: {e}"

                # Append in OpenAI tool result format
                tool_result_msgs.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })

            # Add assistant turn + tool results to history
            self.messages.append({
                "role": "assistant",
                "content": text or "",
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"]),
                        },
                    }
                    for c in calls
                ],
            })
            self.messages.extend(tool_result_msgs)

        # If we exhausted max_rounds, do one final plain response
        log("LLM", "Max tool rounds reached — generating final response", C.DIM)
        return self.ollama.chat_plain(self.messages, system)


# ─── STM compression using local LLM ──────────────────────────────────────────

def make_compress_fn(ollama: OllamaClient) -> callable:
    def compress(texts: list[str]) -> str:
        joined = "\n".join(f"  • {t}" for t in texts)
        return ollama.chat_plain(
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise these perceptual observations into a single "
                    f"concise narrative (3-5 sentences). Preserve key objects, "
                    f"people, positions, colours, and spatial relationships.\n\n"
                    f"{joined}"
                ),
            }],
            system="You are a memory compression assistant. Be concise and factual.",
        )
    return compress


# ─── Startup checks ───────────────────────────────────────────────────────────

def check_ollama(config: Config) -> None:
    """Verify Ollama is running and required models are available."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_HOST}/api/tags", timeout=3
        ) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError):
        print(f"\n{C.ERROR}[ERROR] Cannot reach Ollama at {config.OLLAMA_HOST}{C.RESET}")
        print("        Make sure Ollama is installed and running:")
        print("          ollama serve")
        sys.exit(1)

    available = {m["name"].split(":")[0] for m in data.get("models", [])}
    llm_base  = config.OLLAMA_LLM.split(":")[0]
    vlm_base  = config.OLLAMA_VLM.split(":")[0]

    missing = []
    if llm_base not in available:
        missing.append(f"ollama pull {config.OLLAMA_LLM}")
    if vlm_base not in available:
        missing.append(f"ollama pull {config.OLLAMA_VLM}")

    if missing:
        print(f"\n{C.ERROR}[ERROR] Missing Ollama models:{C.RESET}")
        for cmd in missing:
            print(f"          {cmd}")
        sys.exit(1)

    log("INIT", f"Ollama OK  LLM={config.OLLAMA_LLM}  VLM={config.OLLAMA_VLM}", C.INFO)


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner(config: Config):
    emb_status = (
        f"sentence-transformers ({C.AGENT}semantic{C.RESET})"
        if SEMANTIC_AVAILABLE
        else f"TF-IDF fallback ({C.ERROR}install sentence-transformers for better recall{C.RESET})"
    )
    print(f"""
{C.BOLD}╔════════════════════════════════════════════════════════╗
║      memory_module  ·  local offline perceptual agent   ║
╚════════════════════════════════════════════════════════╝{C.RESET}

{C.AUDIO}■ cyan{C.RESET}    = audio  (Whisper {config.WHISPER_MODEL})
{C.VIDEO}■ magenta{C.RESET} = video  ({config.OLLAMA_VLM})
{C.INFO}■ blue{C.RESET}    = memory operations
{C.AGENT}■ green{C.RESET}   = agent  ({config.OLLAMA_LLM})

Embeddings  : {emb_status}
Database    : {config.DB_PATH}
Captures    : {config.CAPTURES_DIR}/
Ollama host : {config.OLLAMA_HOST}

Commands: status | history | models | decay | quit
──────────────────────────────────────────────────────────
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = Config()

    # Startup checks
    check_ollama(config)

    print_banner(config)

    # Ollama client shared across all components
    ollama = OllamaClient(config)

    # Memory agent with local LLM compression
    agent = MemoryAgent(
        config.DB_PATH,
        compress_fn=make_compress_fn(ollama),
        max_stm_segments=config.STM_MAX_SEGMENTS,
    )
    session = AgentSession(agent, ollama, config)

    # Perception threads
    perc_queue   = queue.Queue()
    audio_thread = AudioThread(perc_queue, config)
    video_thread = VideoThread(perc_queue, config, ollama)
    coordinator  = PerceptionCoordinator(perc_queue, agent, ollama, config)

    audio_thread.start()
    video_thread.start()
    coordinator.start()

    # Graceful shutdown
    def shutdown(sig=None, frame=None):
        print(f"\n{C.DIM}Shutting down…{C.RESET}")
        audio_thread.stop()
        video_thread.stop()
        coordinator.stop()
        n = agent.run_decay()
        report = agent.run_maintenance()
        log("MEM", f"Decay: {n} entries updated. {report}", C.INFO)
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"{C.DIM}Everything is local. No data leaves this machine.{C.RESET}\n")

    # Interactive chat loop
    while True:
        try:
            user_input = input(f"{C.USER}You:{C.RESET} ").strip()
        except EOFError:
            shutdown()
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            shutdown()
            break

        elif user_input.lower() == "status":
            s = agent.status()
            print(f"\n{C.INFO}── Memory Status ──{C.RESET}")
            for k, v in s.items():
                print(f"  {k:<24}: {v}")
            print()

        elif user_input.lower() == "history":
            window = agent.get_stm_window()
            print(f"\n{C.INFO}── STM Window ──{C.RESET}")
            print(window or "  (empty)")
            print()

        elif user_input.lower() == "models":
            print(f"\n{C.INFO}── Active Models ──{C.RESET}")
            print(f"  LLM : {config.OLLAMA_LLM}")
            print(f"  VLM : {config.OLLAMA_VLM}")
            print(f"  STT : Whisper {config.WHISPER_MODEL} ({config.WHISPER_DEVICE})")
            print(f"  Embeddings: {'sentence-transformers' if SEMANTIC_AVAILABLE else 'TF-IDF fallback'}")
            print(f"  Tool calling: {'native' if ollama._tool_calling_works else 'JSON fallback' if ollama._tool_calling_works is False else 'not yet tested'}")
            print()

        elif user_input.lower() == "decay":
            n = agent.run_decay()
            r = agent.run_maintenance()
            log("MEM", f"Decay: {n} entries. Maintenance: {r}", C.INFO)

        else:
            print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
            try:
                reply = session.chat(user_input)
                print(reply)
                agent.record_stm(f"User asked: {user_input}")
                agent.record_stm(f"Agent replied: {reply[:300]}")
            except Exception as e:
                print(f"{C.ERROR}Error: {e}{C.RESET}")
            print()


if __name__ == "__main__":
    main()
