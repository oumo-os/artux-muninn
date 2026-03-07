#!/usr/bin/env python3
"""
demo_live.py — Live perceptual agent with persistent memory.

Two real-time perception streams feed the memory module simultaneously:

  • Microphone  → Whisper transcription → STM segments
  • Webcam      → Claude Vision description → STM + SourceRef (saved frame)

An LLM agent (Claude) uses memory tools to answer your questions about
what it has seen and heard, and can signal when a stored description
isn't detailed enough and the source image should be re-examined.

──────────────────────────────────────────────────────────────────────
Installation
──────────────────────────────────────────────────────────────────────
    pip install anthropic openai sounddevice numpy scipy opencv-python

    # Optional: local Whisper instead of OpenAI API
    pip install faster-whisper

──────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...        # only needed if USE_LOCAL_WHISPER=False

    python demo_live.py

    # Then just talk and look at the camera.
    # Type questions at the prompt to query your memories.
    # Type 'status', 'history', or 'quit' for controls.

──────────────────────────────────────────────────────────────────────
Architecture
──────────────────────────────────────────────────────────────────────

    [Mic] ──► AudioThread ──► Whisper ──────────────────────► Queue
    [Cam] ──► VideoThread ──► Claude Vision ──► save frame ──► Queue
                                                                 │
                                                    PerceptionCoordinator
                                                    • record_stm()
                                                    • record_and_attach_source()
                                                    • auto-consolidate every N events
                                                                 │
                                                          MemoryAgent (SQLite)
                                                                 │
                                                        AgentSession (LLM)
                                                        • recall()
                                                        • tool loop
                                                        • chat replies
"""

import os
import sys
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

# ─── Dependency checks with helpful messages ─────────────────────────────────

def _require(pkg, install_hint):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"\n[ERROR] Missing: {pkg}")
        print(f"        Install: {install_hint}")
        sys.exit(1)

sd  = _require("sounddevice",  "pip install sounddevice")
cv2 = _require("cv2",          "pip install opencv-python")

try:
    import anthropic as _anthropic
except ImportError:
    _require("anthropic", "pip install anthropic")

import anthropic

# Whisper: try local faster-whisper first, fall back to OpenAI API
try:
    from faster_whisper import WhisperModel as _FW
    USE_LOCAL_WHISPER = True
except ImportError:
    USE_LOCAL_WHISPER = False

if not USE_LOCAL_WHISPER:
    try:
        from openai import OpenAI as _OAI
    except ImportError:
        _require("openai", "pip install openai  # or: pip install faster-whisper")

# Memory module (must be in PYTHONPATH or same parent directory)
sys.path.insert(0, str(Path(__file__).parent.parent))
from memory_module import MemoryAgent, get_tools, ToolExecutor


# ─── Configuration ────────────────────────────────────────────────────────────

class Config:
    # Memory
    DB_PATH             = "live_demo_memory.db"
    CAPTURES_DIR        = Path("./captures")        # where webcam frames are saved

    # Audio
    SAMPLE_RATE         = 16_000                    # Hz — Whisper expects 16kHz
    AUDIO_CHUNK_SECS    = 5                         # record this many seconds per chunk
    SILENCE_RMS_THRESH  = 0.008                     # below this → skip transcription
    LOCAL_WHISPER_MODEL = "base.en"                 # faster-whisper model size
    OPENAI_WHISPER_MODEL= "whisper-1"

    # Video
    VIDEO_INTERVAL_SECS = 12                        # describe a new frame every N seconds
    WEBCAM_INDEX        = 0                         # change if you have multiple cameras
    FRAME_QUALITY       = 85                        # JPEG quality for saved frames

    # LLM
    AGENT_MODEL         = "claude-opus-4-6"         # the agent / VLM model
    MAX_TOKENS          = 1024

    # Memory behaviour
    STM_MAX_SEGMENTS    = 20                        # compress STM after this many segments
    AUTO_CONSOLIDATE_N  = 8                         # consolidate to LTM every N perceptions


# ─── Perception event ─────────────────────────────────────────────────────────

@dataclass
class PerceptionEvent:
    kind: str                           # "audio" | "video"
    content: str                        # transcription or VLM description
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    source_location: Optional[str] = None   # saved frame path (video only)
    source_meta: dict = field(default_factory=dict)


# ─── Colour terminal helpers ──────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    AUDIO   = "\033[36m"     # cyan
    VIDEO   = "\033[35m"     # magenta
    AGENT   = "\033[32m"     # green
    USER    = "\033[33m"     # yellow
    ERROR   = "\033[31m"     # red
    INFO    = "\033[34m"     # blue

def log(kind: str, msg: str, colour: str = C.INFO):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"{colour}[{kind} {ts}]{C.RESET} {msg}")


# ─── Audio perception ─────────────────────────────────────────────────────────

class AudioThread(threading.Thread):
    """
    Continuously records microphone audio in fixed-duration chunks.
    Silent chunks are discarded; speech chunks are transcribed and
    pushed to the perception queue.
    """

    def __init__(self, perc_queue: queue.Queue, config: Config):
        super().__init__(daemon=True, name="AudioThread")
        self.queue  = perc_queue
        self.cfg    = config
        self.stop_event = threading.Event()

        # Initialise transcription backend
        if USE_LOCAL_WHISPER:
            log("AUDIO", f"Loading local Whisper ({config.LOCAL_WHISPER_MODEL})…", C.INFO)
            self._whisper = _FW(config.LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8")
            self._transcribe = self._local_transcribe
        else:
            self._oai = _OAI()
            self._transcribe = self._openai_transcribe

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

        # Skip silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.cfg.SILENCE_RMS_THRESH:
            return

        # Transcribe
        text = self._transcribe(audio)
        if not text:
            return

        log("AUDIO", f'"{text}"', C.AUDIO)
        self.queue.put(PerceptionEvent(
            kind="audio",
            content=text,
        ))

    def _save_wav(self, audio: np.ndarray) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return tmp.name

    def _local_transcribe(self, audio: np.ndarray) -> str:
        wav_path = self._save_wav(audio)
        try:
            segments, _ = self._whisper.transcribe(wav_path, beam_size=5)
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            os.unlink(wav_path)

    def _openai_transcribe(self, audio: np.ndarray) -> str:
        wav_path = self._save_wav(audio)
        try:
            with open(wav_path, "rb") as f:
                result = self._oai.audio.transcriptions.create(
                    model=self.cfg.OPENAI_WHISPER_MODEL,
                    file=f,
                )
            return result.text.strip()
        finally:
            os.unlink(wav_path)


# ─── Video perception ─────────────────────────────────────────────────────────

class VideoThread(threading.Thread):
    """
    Captures a webcam frame every VIDEO_INTERVAL_SECS seconds.
    Each frame is:
      1. Saved to disk (CAPTURES_DIR) as the persistent source reference.
      2. Described by Claude Vision.
      3. Pushed to the perception queue with the saved path as source_location.

    The calling agent (via recall) can later re-examine the image at
    source_location for richer spatial detail if the stored description
    is not sufficient.
    """

    def __init__(self, perc_queue: queue.Queue, config: Config,
                 anthropic_client: anthropic.Anthropic):
        super().__init__(daemon=True, name="VideoThread")
        self.queue   = perc_queue
        self.cfg     = config
        self.client  = anthropic_client
        self.stop_event = threading.Event()
        self.cfg.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        log("VIDEO", f"Webcam active — describing every {self.cfg.VIDEO_INTERVAL_SECS}s…", C.VIDEO)
        while not self.stop_event.is_set():
            try:
                self._capture_and_process()
            except Exception as e:
                log("VIDEO", f"Error: {e}", C.ERROR)
            # Sleep in small increments so stop_event is checked responsively
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
                log("VIDEO", "Cannot open webcam — skipping frame", C.ERROR)
                return
            ret, frame = cap.read()
        finally:
            cap.release()

        if not ret or frame is None:
            return

        # Save frame to disk as source reference
        ts_str    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.cfg.CAPTURES_DIR / f"frame_{ts_str}.jpg"
        cv2.imwrite(str(save_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.cfg.FRAME_QUALITY])

        # Encode for VLM
        _, buf    = cv2.imencode(".jpg", frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, self.cfg.FRAME_QUALITY])
        b64_image = base64.b64encode(buf).decode("utf-8")
        h, w      = frame.shape[:2]

        # Describe with Claude Vision
        description = self._describe(b64_image)
        if not description:
            return

        log("VIDEO", f"{description[:120]}…" if len(description) > 120 else description, C.VIDEO)

        self.queue.put(PerceptionEvent(
            kind="video",
            content=description,
            source_location=str(save_path.resolve()),
            source_meta={"width": w, "height": h, "mime": "image/jpeg",
                         "saved_at": ts_str},
        ))

    def _describe(self, b64_image: str) -> str:
        response = self.client.messages.create(
            model=self.cfg.AGENT_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg",
                                   "data": b64_image},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this webcam frame concisely but precisely. "
                            "Include: all visible objects and their positions "
                            "(left/right/centre/foreground/background), colours, "
                            "any people and what they are doing, spatial relationships "
                            "between objects, and anything unusual or notable. "
                            "Be factual. 2-4 sentences."
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text.strip()


# ─── Perception coordinator ───────────────────────────────────────────────────

class PerceptionCoordinator(threading.Thread):
    """
    Reads PerceptionEvents from the queue and feeds them into the
    memory module.  Also triggers periodic LTM consolidation so
    memories build up even without the user asking questions.
    """

    def __init__(self, perc_queue: queue.Queue, agent: MemoryAgent,
                 config: Config):
        super().__init__(daemon=True, name="Coordinator")
        self.queue      = perc_queue
        self.agent      = agent
        self.cfg        = config
        self.stop_event = threading.Event()
        self._count     = 0          # perceptions since last auto-consolidation
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
                # Record speech to STM
                self.agent.record_stm(f"[{ts}] Heard: {event.content}")

            elif event.kind == "video":
                # Record visual description to STM
                seg = self.agent.record_stm(f"[{ts}] Saw: {event.content}")

                # Register the saved frame as a source reference
                # We attach it via STM-then-consolidate later,
                # so we store it temporarily on the event for the consolidation step
                event._pending_source = True

            self._count += 1

            # Periodic auto-consolidation
            if self._count >= self.cfg.AUTO_CONSOLIDATE_N:
                self._auto_consolidate(event if event.kind == "video" else None)
                self._count = 0

    def _auto_consolidate(self, last_video_event: Optional[PerceptionEvent] = None):
        """Compress STM into an LTM entry, attaching any pending video source."""
        try:
            entry = self.agent.consolidate_ltm(
                class_type="observation",
                confidence=0.85,
            )
            log("MEM", f"Auto-consolidated → LTM entry {entry.id[:8]}…", C.INFO)

            # Attach the most recent video source to this LTM entry
            if last_video_event and last_video_event.source_location:
                self.agent.record_and_attach_source(
                    ltm_entry_id=entry.id,
                    location=last_video_event.source_location,
                    type="image",
                    description=last_video_event.content,
                    captured_at=last_video_event.timestamp,
                    meta=last_video_event.source_meta,
                )
                log("MEM", f"Source attached: {Path(last_video_event.source_location).name}", C.INFO)
        except Exception as e:
            log("MEM", f"Auto-consolidation failed: {e}", C.ERROR)


# ─── Agent session ────────────────────────────────────────────────────────────

AGENT_SYSTEM = """\
You are a perceptual AI agent with real-time persistent memory.

You receive a continuous stream of two perception channels:
  • Audio: transcribed speech from a microphone
  • Video: descriptions of webcam frames (with source image paths)

Your memory tools let you:
  - recall()               — retrieve relevant past perceptions and observations
  - record_stm()           — note important real-time events
  - consolidate_ltm()      — persist durable knowledge across sessions
  - create_entity()        — track people and objects as historical ledgers
  - observe_entity()       — append new observations to a known entity
  - resolve_entity()       — find an entity by description
  - record_source()        — register a perception source (image, audio file, etc.)
  - update_source_description() — enrich a source's stored description after re-examination

When answering questions:
1. Always use recall() first before replying from memory.
2. If a recall result includes a source (an image path), and the stored text
   description is not detailed enough to answer the question fully, say so
   explicitly: name the source path and explain what additional detail you
   would need from re-examining it (e.g. "I would need to re-examine
   captures/frame_20240301_142200.jpg — specifically the spatial layout
   of objects on the table — to answer more precisely").
3. Be honest about uncertainty. If you don't have a memory, say so.
4. Be concise. Don't narrate your tool calls — just use them and answer.
"""


class AgentSession:
    def __init__(self, agent: MemoryAgent, config: Config):
        self.agent    = agent
        self.cfg      = config
        self.client   = anthropic.Anthropic()
        self.executor = ToolExecutor(agent)
        self.tools    = get_tools(format="anthropic")
        self.messages: list[dict] = []

    def chat(self, user_input: str) -> str:
        """Send a user message, run the tool loop, return the final response."""
        # Pre-fetch context to inject alongside tools
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
            system += f"\n\n── Current STM window ──\n{stm_ctx}"
        if ltm_ctx:
            system += f"\n\n── Pre-fetched relevant LTM ──\n{ltm_ctx}"

        self.messages.append({"role": "user", "content": user_input})

        # Agentic tool-use loop
        while True:
            response = self.client.messages.create(
                model=self.cfg.AGENT_MODEL,
                max_tokens=self.cfg.MAX_TOKENS,
                system=system,
                tools=self.tools,
                messages=self.messages,
            )
            self.messages.append({
                "role": "assistant",
                "content": response.content,
            })

            if response.stop_reason != "tool_use":
                break

            tool_results = self.executor.run_anthropic(response.content)
            self.messages.append({
                "role": "user",
                "content": tool_results,
            })

        return next(
            (b.text for b in response.content if hasattr(b, "text")),
            "(no response)"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_banner():
    print(f"""
{C.BOLD}╔══════════════════════════════════════════════════════╗
║         memory_module  ·  live perceptual agent       ║
╚══════════════════════════════════════════════════════╝{C.RESET}

{C.AUDIO}■ cyan{C.RESET}    = audio perception (microphone → Whisper)
{C.VIDEO}■ magenta{C.RESET} = video perception (webcam → Claude Vision)
{C.INFO}■ blue{C.RESET}    = memory operations
{C.AGENT}■ green{C.RESET}   = agent responses

Whisper backend : {"local (faster-whisper)" if USE_LOCAL_WHISPER else "OpenAI API"}
Memory database : {Config.DB_PATH}
Captures saved  : {Config.CAPTURES_DIR}/

Commands at the prompt:
  {C.DIM}status{C.RESET}    — show memory stats
  {C.DIM}history{C.RESET}   — show current STM window
  {C.DIM}decay{C.RESET}     — run memory decay + maintenance
  {C.DIM}quit{C.RESET}      — exit
  {C.DIM}(anything else){C.RESET} — ask the agent

──────────────────────────────────────────────────────────
""")


def main():
    print_banner()

    # ── Initialise memory ──────────────────────────────────────────────
    client = anthropic.Anthropic()

    def llm_compress(texts: list[str]) -> str:
        """LLM-quality STM compression."""
        joined = "\n".join(f"  • {t}" for t in texts)
        resp = client.messages.create(
            model=Config.AGENT_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarise these perceptual observations into a single "
                    f"concise narrative (3-5 sentences). Preserve key objects, "
                    f"people, positions, and spatial relationships.\n\n{joined}"
                ),
            }],
        )
        return resp.content[0].text.strip()

    agent   = MemoryAgent(
        Config.DB_PATH,
        compress_fn=llm_compress,
        max_stm_segments=Config.STM_MAX_SEGMENTS,
    )
    session = AgentSession(agent, Config)

    # ── Start perception threads ───────────────────────────────────────
    perc_queue   = queue.Queue()
    audio_thread = AudioThread(perc_queue, Config)
    video_thread = VideoThread(perc_queue, Config, client)
    coordinator  = PerceptionCoordinator(perc_queue, agent, Config)

    audio_thread.start()
    video_thread.start()
    coordinator.start()

    # ── Graceful shutdown ──────────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        print(f"\n{C.DIM}Shutting down…{C.RESET}")
        audio_thread.stop()
        video_thread.stop()
        coordinator.stop()
        # Final decay pass
        n = agent.run_decay()
        report = agent.run_maintenance()
        log("MEM", f"Decay: {n} entries updated. Maintenance: {report}", C.INFO)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Interactive chat loop ──────────────────────────────────────────
    print(f"{C.DIM}Perceptions are being recorded. Ask anything, or just wait and observe.{C.RESET}\n")

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
            print(f"  STM segments       : {s['stm_segments']}")
            print(f"  LTM entries        : {s['ltm_entries']}")
            print(f"  LTM avg confidence : {s['ltm_avg_confidence']:.3f}")
            print(f"  Entities           : {s['entities']}")
            print(f"  Archive scars      : {s['archive_scars']}")
            print(f"  Sources            : {s['sources']}")
            print()

        elif user_input.lower() == "history":
            window = agent.get_stm_window()
            print(f"\n{C.INFO}── Current STM Window ──{C.RESET}")
            print(window or "  (empty)")
            print()

        elif user_input.lower() == "decay":
            n = agent.run_decay()
            report = agent.run_maintenance()
            log("MEM", f"Decay ran: {n} entries updated. {report}", C.INFO)

        else:
            # Route to agent
            print(f"{C.AGENT}Agent:{C.RESET} ", end="", flush=True)
            try:
                reply = session.chat(user_input)
                print(reply)
                # Record this exchange to STM
                agent.record_stm(f"User asked: {user_input}")
                agent.record_stm(f"Agent replied: {reply[:300]}")
            except Exception as e:
                print(f"{C.ERROR}Error: {e}{C.RESET}")
            print()


if __name__ == "__main__":
    main()
