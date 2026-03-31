#!/usr/bin/env python3
"""
demo_moonshine.py — Fully offline perceptual agent.
Uses Moonshine STT, SmolVLM vision, and raw requests to Ollama.
No openai package required.

Components
──────────
  Microphone  → Moonshine ONNX  (fast, tiny, CPU-native STT)
  Webcam      → SmolVLM         (local HuggingFace vision model)
  LLM agent   → Ollama          (via raw requests, native /api/chat)
  Embeddings  → sentence-transformers / TF-IDF fallback
  Memory      → SQLite

──────────────────────────────────────────────────────────────────────
Install
──────────────────────────────────────────────────────────────────────
  pip install moonshine-onnx sounddevice numpy scipy
  pip install transformers accelerate Pillow torch
  pip install opencv-python requests

  # Ollama must be running with at least one LLM:
  ollama serve
  ollama pull llama3.2       # or qwen2.5, mistral-nemo, etc.

  # SmolVLM downloads automatically from HuggingFace on first run (~2 GB).
  # Moonshine ONNX models download automatically on first run (~100 MB).

──────────────────────────────────────────────────────────────────────
Environment overrides
──────────────────────────────────────────────────────────────────────
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_LLM=llama3.2
  SMOLVLM_MODEL=HuggingFaceTB/SmolVLM-Instruct        # 2 GB
  SMOLVLM_MODEL=HuggingFaceTB/SmolVLM-500M-Instruct   # 1 GB, faster
  MOONSHINE_MODEL=moonshine/base                       # or moonshine/tiny

──────────────────────────────────────────────────────────────────────
Key difference from demo_local.py
──────────────────────────────────────────────────────────────────────
  • Moonshine ONNX instead of faster-whisper
      Moonshine is purpose-built for real-time transcription on CPU.
      It is 5× faster than Whisper-tiny at equivalent accuracy on
      short utterances, and runs entirely via ONNX Runtime (no PyTorch
      required for the STT component).

  • SmolVLM instead of Ollama-hosted VLM
      SmolVLM runs in-process via HuggingFace Transformers.
      This means the VLM doesn't need to be loaded into Ollama,
      freeing GPU/RAM for the LLM.  SmolVLM-Instruct at 2B params
      describes scenes accurately.  The 500M variant is usable on
      CPU-only machines.

  • requests instead of openai package
      We call Ollama's native /api/chat endpoint directly.
      The native API returns tool call arguments as a parsed dict
      (not a JSON string like the OpenAI-compat endpoint does).
      This removes the openai package as a dependency entirely.
"""

import os
import sys
import json
import time
import wave
import base64
import queue
import signal
import tempfile
import threading
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests as req

# ─── Dependency checks ────────────────────────────────────────────────────────

def _require(pkg, pip_name=None, attr=None, extra=None):
    try:
        mod = __import__(pkg)
        return getattr(mod, attr) if attr else mod
    except ImportError:
        name = pip_name or pkg
        hint = f"pip install {name}"
        if extra:
            hint += f"\n        {extra}"
        print(f"\n[ERROR] Missing: {pkg}")
        print(f"        {hint}")
        sys.exit(1)

sd  = _require("sounddevice", "sounddevice")
cv2 = _require("cv2",         "opencv-python")

try:
    from moonshine_onnx import MoonshineOnnxModel
except ImportError:
    print("\n[ERROR] moonshine-onnx not installed.")
    print("        pip install moonshine-onnx")
    sys.exit(1)

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch
    from PIL import Image
except ImportError:
    print("\n[ERROR] SmolVLM requires: transformers accelerate torch Pillow")
    print("        pip install transformers accelerate torch Pillow")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from memory_module import MemoryAgent, get_tools, ToolExecutor
from memory_module.embeddings import SEMANTIC_AVAILABLE


# ─── Configuration ────────────────────────────────────────────────────────────

class Config:
    # Ollama (LLM only — VLM is handled by SmolVLM in-process)
    OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_LLM      = os.environ.get("OLLAMA_LLM",  "llama3.2")

    # Moonshine STT
    MOONSHINE_MODEL = os.environ.get("MOONSHINE_MODEL", "moonshine/base")
    # moonshine/tiny  →  ~50 MB,  fastest, slightly lower accuracy
    # moonshine/base  →  ~100 MB, good balance  ← default
    # moonshine/small →  ~200 MB, best accuracy

    # SmolVLM
    SMOLVLM_MODEL   = os.environ.get(
        "SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-Instruct"
    )
    # HuggingFaceTB/SmolVLM-Instruct      → ~2 GB, best quality
    # HuggingFaceTB/SmolVLM-500M-Instruct → ~1 GB, CPU-friendly
    SMOLVLM_DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    SMOLVLM_DTYPE   = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Memory
    DB_PATH         = "moonshine_demo_memory.db"
    CAPTURES_DIR    = Path("./captures_moonshine")

    # Audio
    SAMPLE_RATE     = 16_000        # Moonshine expects 16 kHz
    AUDIO_CHUNK_SECS = 5
    SILENCE_RMS_THRESH = 0.008

    # Video
    VIDEO_INTERVAL_SECS = 15
    WEBCAM_INDEX    = 0
    FRAME_QUALITY   = 80

    # Inference
    MAX_NEW_TOKENS  = 1024
    TEMPERATURE     = 0.4
    TIMEOUT_SECS    = 90            # generous for local inference

    # Memory
    STM_MAX_SEGMENTS  = 20
    AUTO_CONSOLIDATE_N = 8


# ─── Terminal colours ─────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m";  BOLD = "\033[1m";  DIM = "\033[2m"
    AUDIO = "\033[36m"; VIDEO = "\033[35m"; AGENT = "\033[32m"
    USER  = "\033[33m"; ERROR = "\033[31m"; INFO  = "\033[34m"

def log(kind: str, msg: str, colour: str = C.INFO):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{colour}[{kind} {ts}]{C.RESET} {msg}")


# ─── Ollama client via raw requests ──────────────────────────────────────────
#
# We call the native /api/chat endpoint, NOT the OpenAI-compat /v1/ endpoint.
#
# Native API differences from OpenAI-compat:
#   • arguments in tool_calls is already a parsed dict (not a JSON string)
#   • the "tools" key format is identical to OpenAI
#   • stream:false gives a single JSON response body
#   • no "choices" wrapper — response has "message" directly

class OllamaClient:
    """
    Thin requests-based wrapper for Ollama's native /api/chat endpoint.
    Handles tool calling with the same JSON fallback as demo_local.py.
    """

    def __init__(self, config: Config):
        self.cfg  = config
        self.base = config.OLLAMA_HOST.rstrip("/")
        self._tool_calling_works: Optional[bool] = None

    # ── Core request ──────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base}{endpoint}"
        try:
            resp = req.post(
                url, json=payload,
                timeout=self.cfg.TIMEOUT_SECS,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except req.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base}. Is 'ollama serve' running?"
            )
        except req.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {self.cfg.TIMEOUT_SECS}s. "
                "Try a smaller model or increase TIMEOUT_SECS."
            )

    # ── Chat with tools ───────────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> tuple[str, list[dict]]:
        """
        One round of tool elicitation.
        Returns (text_content, list_of_tool_calls).

        tool_calls entries: {"name": str, "arguments": dict}
        Arguments are already parsed dicts — no json.loads() needed.
        """
        full_messages = [{"role": "system", "content": system}] + messages

        if self._tool_calling_works is not False:
            try:
                data = self._post("/api/chat", {
                    "model":    self.cfg.OLLAMA_LLM,
                    "messages": full_messages,
                    "tools":    tools,
                    "stream":   False,
                    "options":  {"temperature": self.cfg.TEMPERATURE},
                })
                msg   = data.get("message", {})
                calls = msg.get("tool_calls") or []

                if calls:
                    self._tool_calling_works = True
                    # Native API: arguments is already a dict
                    parsed = [
                        {
                            "name":      c["function"]["name"],
                            "arguments": c["function"]["arguments"],  # already dict
                        }
                        for c in calls
                    ]
                    return msg.get("content") or "", parsed

                self._tool_calling_works = True
                return msg.get("content") or "", []

            except Exception as e:
                log("LLM", f"Native tool call error ({e}) — switching to JSON fallback", C.DIM)
                self._tool_calling_works = False

        return self._json_fallback(full_messages, tools)

    def chat_plain(self, messages: list[dict], system: str) -> str:
        """Plain chat without tools — for compression, etc."""
        data = self._post("/api/chat", {
            "model":    self.cfg.OLLAMA_LLM,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream":   False,
            "options":  {"temperature": self.cfg.TEMPERATURE},
        })
        return (data.get("message", {}).get("content") or "").strip()

    def list_models(self) -> list[str]:
        """Return model names available in this Ollama instance."""
        try:
            data = req.get(f"{self.base}/api/tags", timeout=5).json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── JSON fallback ─────────────────────────────────────────────────

    _FALLBACK_SUFFIX = """
── Tool Instructions ──
To call a tool, respond with ONLY a JSON object:
  {"tool": "<name>", "arguments": {<args>}}

For multiple tools, use a JSON array:
  [{"tool": "<name>", "arguments": {<args>}}, ...]

After all needed tool calls, respond with plain text.
If no tools are needed, respond with plain text directly.
"""

    def _tool_schema_summary(self, tools: list[dict]) -> str:
        lines = ["Available tools:"]
        for t in tools:
            fn       = t.get("function", t)
            name     = fn.get("name", "?")
            desc     = fn.get("description", "")[:90]
            props    = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            params   = ", ".join(
                f"{k}{'*' if k in required else '?'}" for k in props
            )
            lines.append(f"  {name}({params}) — {desc}")
        return "\n".join(lines)

    def _json_fallback(
        self, messages: list[dict], tools: list[dict]
    ) -> tuple[str, list[dict]]:
        system = self._tool_schema_summary(tools) + self._FALLBACK_SUFFIX
        data   = self._post("/api/chat", {
            "model":    self.cfg.OLLAMA_LLM,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream":   False,
            "options":  {"temperature": self.cfg.TEMPERATURE},
        })
        raw = (data.get("message", {}).get("content") or "").strip()
        return _extract_json_tool_calls(raw)


# ─── JSON tool-call extraction (bracket-depth scanner) ────────────────────────

def _find_json_spans(text: str) -> list[tuple[int, int]]:
    """Find start/end indices of all top-level JSON objects and arrays."""
    spans, i = [], 0
    while i < len(text):
        if text[i] in ('{', '['):
            opener = text[i]
            closer = '}' if opener == '{' else ']'
            depth = in_str = esc = 0
            start = i
            for j in range(i, len(text)):
                ch = text[j]
                if esc:
                    esc = False
                elif ch == '\\' and in_str:
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == opener:   depth += 1
                    elif ch == closer: depth -= 1
                    if depth == 0:
                        spans.append((start, j + 1))
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1
    return spans


def _extract_json_tool_calls(raw: str) -> tuple[str, list[dict]]:
    """
    Extract tool call dicts from a raw LLM response string.
    Returns (remaining_plain_text, list_of_calls).
    Each call: {"name": str, "arguments": dict}
    """
    calls, removed = [], []
    for start, end in _find_json_spans(raw):
        try:
            parsed = json.loads(raw[start:end])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "tool" in parsed:
            calls.append({"name": parsed["tool"],
                           "arguments": parsed.get("arguments", {})})
            removed.append((start, end))

        elif isinstance(parsed, list):
            hits = [i for i in parsed if isinstance(i, dict) and "tool" in i]
            if hits:
                for item in hits:
                    calls.append({"name": item["tool"],
                                  "arguments": item.get("arguments", {})})
                removed.append((start, end))

    remaining = raw
    for s, e in sorted(removed, reverse=True):
        remaining = remaining[:s] + remaining[e:]
    return remaining.strip(), calls


# ─── Moonshine STT ────────────────────────────────────────────────────────────

class MoonshineTranscriber:
    """
    Wrapper around MoonshineOnnxModel.

    Moonshine is designed for streaming / real-time transcription on CPU.
    It uses ONNX Runtime under the hood — no PyTorch required for inference.

    Model sizes:
      moonshine/tiny  — ~50 MB,  RTF < 0.1 on modern CPU
      moonshine/base  — ~100 MB, RTF < 0.15, better accuracy  ← default
      moonshine/small — ~200 MB, near Whisper-small quality
    """

    def __init__(self, model_name: str = "moonshine/base"):
        log("STT", f"Loading Moonshine ({model_name})…", C.INFO)
        self._model = MoonshineOnnxModel(model_name=model_name)
        log("STT", "Moonshine ready.", C.INFO)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 16 kHz mono audio array.
        Returns stripped transcription string, or empty string on failure.
        """
        # Moonshine expects shape (1, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        tokens = self._model.generate(audio)
        texts  = self._model.tokenizer.decode_batch(tokens)
        return texts[0].strip() if texts else ""


# ─── SmolVLM vision ───────────────────────────────────────────────────────────

class SmolVLM:
    """
    In-process SmolVLM for webcam frame description.

    SmolVLM-Instruct (2B params) runs via HuggingFace Transformers.
    SmolVLM-500M-Instruct is a smaller option for CPU-only machines.

    Loading is lazy — the model isn't pulled into RAM until the first
    frame arrives, so startup is fast even on slow machines.
    """

    VISION_PROMPT = (
        "Describe this webcam frame precisely and concisely. "
        "Include all visible objects and their exact positions "
        "(left / right / centre / foreground / background / on top of / next to), "
        "colours, any people and what they are doing, spatial relationships "
        "between objects, and anything unusual or notable. "
        "Be factual and specific. 2-4 sentences."
    )

    def __init__(self, model_id: str, device: str, dtype):
        self._model_id = model_id
        self._device   = device
        self._dtype    = dtype
        self._processor = None
        self._model     = None
        self._lock      = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            log("VLM", f"Loading {self._model_id} on {self._device}…", C.INFO)
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                torch_dtype=self._dtype,
                device_map="auto" if self._device == "cuda" else None,
            )
            if self._device == "cpu":
                self._model = self._model.to("cpu")
            log("VLM", "SmolVLM ready.", C.INFO)

    def describe(self, frame_bgr: np.ndarray) -> str:
        """
        Describe a BGR OpenCV frame.
        Returns a plain text description string.
        """
        self._ensure_loaded()

        # OpenCV BGR → PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Build chat-template message
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.VISION_PROMPT},
            ],
        }]

        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=prompt,
            images=[pil_image],
            return_tensors="pt",
        )
        # Move inputs to the same device as model
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return text.strip()


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
        self._stt       = MoonshineTranscriber(config.MOONSHINE_MODEL)

    def run(self):
        log("AUDIO", "Microphone active — listening…", C.AUDIO)
        while not self.stop_event.is_set():
            try:
                self._record_and_process()
            except Exception as e:
                log("AUDIO", f"Error: {e}", C.ERROR)
                time.sleep(1)

    def stop(self): self.stop_event.set()

    def _record_and_process(self):
        n = int(self.cfg.AUDIO_CHUNK_SECS * self.cfg.SAMPLE_RATE)
        audio = sd.rec(n, samplerate=self.cfg.SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        if np.sqrt(np.mean(audio ** 2)) < self.cfg.SILENCE_RMS_THRESH:
            return

        text = self._stt.transcribe(audio)
        if not text:
            return

        log("AUDIO", f'"{text}"', C.AUDIO)
        self.queue.put(PerceptionEvent(kind="audio", content=text))


# ─── Video thread ─────────────────────────────────────────────────────────────

class VideoThread(threading.Thread):
    def __init__(self, perc_queue: queue.Queue, config: Config, vlm: SmolVLM):
        super().__init__(daemon=True, name="VideoThread")
        self.queue      = perc_queue
        self.cfg        = config
        self.vlm        = vlm
        self.stop_event = threading.Event()
        config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        log("VIDEO", f"Webcam active — describing every {self.cfg.VIDEO_INTERVAL_SECS}s "
                     f"via SmolVLM…", C.VIDEO)
        while not self.stop_event.is_set():
            try:
                self._capture_and_process()
            except Exception as e:
                log("VIDEO", f"Error: {e}", C.ERROR)
            for _ in range(self.cfg.VIDEO_INTERVAL_SECS * 4):
                if self.stop_event.is_set():
                    break
                time.sleep(0.25)

    def stop(self): self.stop_event.set()

    def _capture_and_process(self):
        cap = cv2.VideoCapture(self.cfg.WEBCAM_INDEX)
        try:
            if not cap.isOpened():
                log("VIDEO", "Cannot open webcam", C.ERROR)
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

        h, w = frame.shape[:2]

        # SmolVLM describes the frame in-process
        description = self.vlm.describe(frame)
        if not description:
            return

        log("VIDEO", description[:120] + ("…" if len(description) > 120 else ""), C.VIDEO)
        self.queue.put(PerceptionEvent(
            kind="video",
            content=description,
            source_location=str(save_path.resolve()),
            source_meta={
                "width": w, "height": h, "mime": "image/jpeg",
                "saved_at": ts_str, "vlm": self.cfg.SMOLVLM_MODEL,
            },
        ))


# ─── Perception coordinator ───────────────────────────────────────────────────

class PerceptionCoordinator(threading.Thread):
    def __init__(self, perc_queue: queue.Queue, agent: MemoryAgent,
                 config: Config):
        super().__init__(daemon=True, name="Coordinator")
        self.queue      = perc_queue
        self.agent      = agent
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

    def stop(self): self.stop_event.set()

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
                self._consolidate()
                self._count = 0

    def _consolidate(self):
        try:
            entry = self.agent.consolidate_ltm(
                class_type="observation", confidence=0.85
            )
            log("MEM", f"Consolidated → {entry.id[:8]}…", C.INFO)
            if self._last_video and self._last_video.source_location:
                self.agent.record_and_attach_source(
                    ltm_entry_id=entry.id,
                    location=self._last_video.source_location,
                    type="image",
                    description=self._last_video.content,
                    captured_at=self._last_video.timestamp,
                    meta=self._last_video.source_meta,
                )
                log("MEM", f"Source: {Path(self._last_video.source_location).name}", C.INFO)
        except Exception as e:
            log("MEM", f"Consolidation error: {e}", C.ERROR)


# ─── Agent session ────────────────────────────────────────────────────────────

AGENT_SYSTEM = """\
You are a perceptual AI agent with real-time persistent memory, running entirely offline.

Perception inputs:
  • Audio: transcribed speech via Moonshine STT
  • Video: webcam frame descriptions via SmolVLM, with saved image paths

Use your memory tools when answering questions:
1. recall() before answering anything involving past observations.
2. If a recall result includes a source image path and the stored description
   is not detailed enough, name the path and say what additional spatial or
   visual detail you would need from re-examining it.
3. Be honest about what you don't know. Don't invent details.
4. Be concise. Use tools silently — don't narrate tool usage.
"""


class AgentSession:
    def __init__(self, agent: MemoryAgent, ollama: OllamaClient, config: Config):
        self.agent    = agent
        self.ollama   = ollama
        self.cfg      = config
        self.executor = ToolExecutor(agent)
        # Use OpenAI format — Ollama's /api/chat accepts the same tool schema
        self.tools    = get_tools(format="openai")
        self.messages: list[dict] = []

    def chat(self, user_input: str) -> str:
        stm      = self.agent.get_stm_window()
        relevant = self.agent.recall(user_input, top_k=4)
        ltm_ctx  = "\n".join(
            f"• [{r.entry.class_type}] {r.entry.content}"
            + (f"\n  sources: {', '.join(s.location for s in r.sources)}"
               if r.sources else "")
            for r in relevant
        )

        system = AGENT_SYSTEM
        if stm:
            system += f"\n\n── STM ──\n{stm}"
        if ltm_ctx:
            system += f"\n\n── Relevant LTM ──\n{ltm_ctx}"

        self.messages.append({"role": "user", "content": user_input})

        max_rounds = 6
        for _ in range(max_rounds):
            text, calls = self.ollama.chat_with_tools(
                messages=self.messages,
                tools=self.tools,
                system=system,
            )

            if not calls:
                self.messages.append({"role": "assistant", "content": text})
                return text

            for call in calls:
                log("LLM", f"→ {call['name']}({list(call['arguments'].keys())})", C.DIM)
                try:
                    result = self.executor.execute(call["name"], call["arguments"])
                except Exception as e:
                    result = f"Error: {e}"

                # Append as Ollama native tool result format
                self.messages.append({
                    "role": "assistant",
                    "content": text or "",
                    "tool_calls": [{
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        }
                    }],
                })
                self.messages.append({
                    "role": "tool",
                    "content": result,
                })

        log("LLM", "Max rounds reached — final response", C.DIM)
        return self.ollama.chat_plain(self.messages, system)


# ─── STM compression ──────────────────────────────────────────────────────────

def make_compress_fn(ollama: OllamaClient) -> callable:
    def compress(texts: list[str]) -> str:
        return ollama.chat_plain(
            messages=[{
                "role": "user",
                "content": (
                    "Summarise these perceptual observations into a single "
                    "concise narrative (3-5 sentences). Preserve objects, "
                    "positions, colours, spatial relationships.\n\n"
                    + "\n".join(f"  • {t}" for t in texts)
                ),
            }],
            system="You are a memory compression assistant. Be concise and factual.",
        )
    return compress


# ─── Startup checks ───────────────────────────────────────────────────────────

def check_ollama(config: Config) -> None:
    try:
        resp = req.get(f"{config.OLLAMA_HOST}/api/tags", timeout=4)
        data = resp.json()
    except Exception:
        print(f"\n{C.ERROR}[ERROR] Cannot reach Ollama at {config.OLLAMA_HOST}{C.RESET}")
        print("        Run:  ollama serve")
        sys.exit(1)

    available = {m["name"].split(":")[0] for m in data.get("models", [])}
    llm_base  = config.OLLAMA_LLM.split(":")[0]

    if llm_base not in available:
        print(f"\n{C.ERROR}[ERROR] LLM not found: {config.OLLAMA_LLM}{C.RESET}")
        print(f"        Run:  ollama pull {config.OLLAMA_LLM}")
        print(f"        Available: {', '.join(sorted(available)) or '(none)'}")
        sys.exit(1)

    log("INIT", f"Ollama OK  LLM={config.OLLAMA_LLM}", C.INFO)
    log("INIT", f"SmolVLM={config.SMOLVLM_MODEL}  device={config.SMOLVLM_DEVICE}", C.INFO)
    log("INIT", f"Moonshine={config.MOONSHINE_MODEL}", C.INFO)


# ─── Banner ───────────────────────────────────────────────────────────────────

def print_banner(config: Config):
    emb = ("sentence-transformers (semantic)" if SEMANTIC_AVAILABLE
           else f"TF-IDF fallback {C.DIM}(pip install sentence-transformers){C.RESET}")
    print(f"""
{C.BOLD}╔══════════════════════════════════════════════════════════╗
║   memory_module · moonshine + smolvlm · offline agent    ║
╚══════════════════════════════════════════════════════════╝{C.RESET}

{C.AUDIO}■ cyan{C.RESET}    audio  Moonshine {config.MOONSHINE_MODEL}
{C.VIDEO}■ magenta{C.RESET} video  SmolVLM   {config.SMOLVLM_MODEL}  [{config.SMOLVLM_DEVICE}]
{C.INFO}■ blue{C.RESET}    memory SQLite + {emb}
{C.AGENT}■ green{C.RESET}   agent  Ollama    {config.OLLAMA_LLM}

Database  : {config.DB_PATH}
Captures  : {config.CAPTURES_DIR}/

Commands: status | history | models | decay | quit
──────────────────────────────────────────────────────────────
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = Config()
    check_ollama(config)
    print_banner(config)

    ollama = OllamaClient(config)

    # SmolVLM — shared between VideoThread and AgentSession
    # (lazy-loads on first frame, so startup is immediate)
    vlm = SmolVLM(
        model_id=config.SMOLVLM_MODEL,
        device=config.SMOLVLM_DEVICE,
        dtype=config.SMOLVLM_DTYPE,
    )

    agent   = MemoryAgent(
        config.DB_PATH,
        compress_fn=make_compress_fn(ollama),
        max_stm_segments=config.STM_MAX_SEGMENTS,
    )
    session = AgentSession(agent, ollama, config)

    perc_queue   = queue.Queue()
    audio_thread = AudioThread(perc_queue, config)
    video_thread = VideoThread(perc_queue, config, vlm)
    coordinator  = PerceptionCoordinator(perc_queue, agent, config)

    audio_thread.start()
    video_thread.start()
    coordinator.start()

    def shutdown(sig=None, frame=None):
        print(f"\n{C.DIM}Shutting down…{C.RESET}")
        audio_thread.stop()
        video_thread.stop()
        coordinator.stop()
        n = agent.run_decay()
        r = agent.run_maintenance()
        log("MEM", f"Decay: {n} entries. {r}", C.INFO)
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"{C.DIM}Nothing leaves this machine.{C.RESET}\n")

    while True:
        try:
            user_input = input(f"{C.USER}You:{C.RESET} ").strip()
        except EOFError:
            shutdown()
            break

        if not user_input:
            continue

        match user_input.lower():
            case "quit":
                shutdown()

            case "status":
                s = agent.status()
                print(f"\n{C.INFO}── Memory Status ──{C.RESET}")
                for k, v in s.items():
                    print(f"  {k:<24}: {v}")
                print()

            case "history":
                w = agent.get_stm_window()
                print(f"\n{C.INFO}── STM Window ──{C.RESET}")
                print(w or "  (empty)")
                print()

            case "models":
                tc = ("native" if ollama._tool_calling_works
                      else "JSON fallback" if ollama._tool_calling_works is False
                      else "not yet tested")
                print(f"\n{C.INFO}── Active Models ──{C.RESET}")
                print(f"  STT        : Moonshine {config.MOONSHINE_MODEL}")
                print(f"  VLM        : {config.SMOLVLM_MODEL}  [{config.SMOLVLM_DEVICE}]")
                print(f"               {'loaded' if vlm._model else 'not yet loaded (lazy)'}")
                print(f"  LLM        : {config.OLLAMA_LLM} via Ollama")
                print(f"  Tool calls : {tc}")
                print(f"  Embeddings : {'sentence-transformers' if SEMANTIC_AVAILABLE else 'TF-IDF'}")
                print()

            case "decay":
                n = agent.run_decay()
                r = agent.run_maintenance()
                log("MEM", f"Decay: {n} entries. {r}", C.INFO)

            case _:
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
