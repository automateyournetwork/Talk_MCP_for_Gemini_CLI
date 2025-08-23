#!/usr/bin/env python3
"""
MCP Server: Talk mode (PyAudio + Whisper) with tools:
  - talk_start(blocking?, device_index?, energy_gate?, end_sil_ms?, max_utter_ms?, ...)
  - talk_status()
  - talk_stop()
  - list_devices()
  - tts_speak(text, voice?, model?, fmt?)

Pure MCP (stdio). No playback. No Gemini calls here.
- talk_start: record → transcribe → return transcript only
- tts_speak: text → audio file (absolute path returned)
"""

import os, sys, time, wave, math, queue, shutil, tempfile, subprocess, logging, threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pyaudio

# MCP server (FastMCP wrapper)
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

# ---------- logging to stderr (NEVER stdout) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr
)
log = logging.getLogger("TalkMCP")

# ---------- defaults (env-overridable) ----------
DEF_RATE         = int(os.getenv("TALK_RATE", "16000"))
DEF_CHANNELS     = int(os.getenv("TALK_CHANNELS", "1"))
DEF_CHUNK        = int(os.getenv("TALK_CHUNK", "1024"))
# Friendlier for typical WSLg mics
DEF_ENERGY_GATE  = float(os.getenv("TALK_ENERGY_GATE", "30.0"))      # was 150.0
DEF_MIN_TALK_MS  = int(os.getenv("TALK_MIN_TALK_MS", "350"))         # was 500
DEF_END_SIL_MS   = int(os.getenv("TALK_END_SIL_MS", "1500"))         # was 900
DEF_MAX_UTTER_MS = int(os.getenv("TALK_MAX_UTTER_MS", "20000"))
DEF_PRE_ROLL_MS  = int(os.getenv("TALK_PRE_ROLL_MS", "300"))         # was 200
DEF_DEVICE_INDEX = os.getenv("TALK_DEVICE_INDEX")  # optional string/int

WHISPER_BIN   = os.getenv("WHISPER_BIN", "whisper")   # if present, prefer CLI
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
LANG          = os.getenv("TALK_LANG", "")
TMP_DIR       = os.getenv("TALK_TMP", "")  # optional

DEBUG_ENERGY  = os.getenv("TALK_DEBUG_ENERGY", "0") == "1"

# ---------- TTS defaults ----------
TTS_MODEL  = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE  = os.getenv("TTS_VOICE", "alloy")

# ---------- helpers ----------
def _rms_energy(pcm: bytes) -> float:
    import struct
    n = len(pcm) // 2
    if n == 0: return 0.0
    vals = struct.unpack("<" + "h"*n, pcm)
    s2 = sum(v*v for v in vals)
    return math.sqrt(s2 / n)

def _write_wav(path: Path, frames: List[bytes], rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

def _have_whisper_cli() -> bool:
    return shutil.which(WHISPER_BIN) is not None

def _transcribe_cli(wav: Path) -> str:
    cmd = [WHISPER_BIN, str(wav), "--model", WHISPER_MODEL, "--fp16", "False", "--output_format", "txt"]
    if LANG: cmd += ["--language", LANG]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        txt = Path(str(wav) + ".txt")
        return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    except subprocess.CalledProcessError as e:
        log.error("whisper CLI failed: %s", e)
        return ""

def _transcribe_py(wav: Path) -> str:
    try:
        import whisper  # pip install -U openai-whisper
        model = whisper.load_model(WHISPER_MODEL)
        res = model.transcribe(str(wav), language=LANG or None, fp16=False)
        return (res.get("text") or "").strip()
    except Exception as e:
        log.error("whisper (py) failed: %s", e)
        return ""

def _transcribe(wav: Path) -> str:
    return _transcribe_cli(wav) if _have_whisper_cli() else _transcribe_py(wav)

def _ms() -> int:
    return int(round(time.time() * 1000))

# ---------- OpenAI TTS ----------
def _tts_openai(text: str, out_path: Path, model: str, voice: str) -> tuple[bool, str]:
    """
    Synthesize speech with OpenAI TTS → write to out_path.
    Requires OPENAI_API_KEY and `pip install openai>=1.0`.
    Returns (ok, error_message).
    """
    try:
        from openai import OpenAI
    except Exception as e:
        log.error("OpenAI SDK not installed: %s", e)
        return False, f"import_error: {e}"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "missing OPENAI_API_KEY in this process"

    try:
        client = OpenAI(api_key=api_key)
        # No 'format' arg; rely on SDK default container (mp3) and stream to file
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
        ) as resp:
            resp.stream_to_file(str(out_path))
        return True, ""
    except Exception as e:
        log.error("TTS error: %s", e)
        return False, f"{type(e).__name__}: {e}"

# ---------- engine ----------
class TalkEngine:
    def __init__(self):
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.running = False
        self.rate, self.channels, self.chunk = DEF_RATE, DEF_CHANNELS, DEF_CHUNK
        self.energy_gate = DEF_ENERGY_GATE
        self.min_talk_ms, self.end_sil_ms = DEF_MIN_TALK_MS, DEF_END_SIL_MS
        self.max_utter_ms, self.pre_roll_ms = DEF_MAX_UTTER_MS, DEF_PRE_ROLL_MS
        self.device_index = int(DEF_DEVICE_INDEX) if DEF_DEVICE_INDEX not in (None, "") else None

        self.tmp_base: Optional[Path] = None
        self.last_transcript: str = ""
        self.last_reply: str = ""   # kept for backward compat; unused in STT-only mode
        self.error: Optional[str] = None

    def start(self,
              rate: Optional[int] = None,
              channels: Optional[int] = None,
              chunk: Optional[int] = None,
              energy_gate: Optional[float] = None,
              min_talk_ms: Optional[int] = None,
              end_sil_ms: Optional[int] = None,
              max_utter_ms: Optional[int] = None,
              pre_roll_ms: Optional[int] = None,
              device_index: Optional[int] = None,
              blocking: bool = True) -> Dict[str, Any]:
        with self._lock:
            if self.running:
                return {"ok": False, "msg": "already running"}

            # apply overrides
            if rate is not None: self.rate = rate
            if channels is not None: self.channels = channels
            if chunk is not None: self.chunk = chunk
            if energy_gate is not None: self.energy_gate = energy_gate
            if min_talk_ms is not None: self.min_talk_ms = min_talk_ms
            if end_sil_ms is not None: self.end_sil_ms = end_sil_ms
            if max_utter_ms is not None: self.max_utter_ms = max_utter_ms
            if pre_roll_ms is not None: self.pre_roll_ms = pre_roll_ms
            if device_index is not None: self.device_index = device_index

            self.last_transcript = ""
            self.last_reply = ""
            self.error = None
            self._stop.clear()

            self.tmp_base = Path(TMP_DIR) if TMP_DIR else Path(tempfile.mkdtemp(prefix="talk_"))
            self.tmp_base.mkdir(parents=True, exist_ok=True)

            self.running = True
            self._t = threading.Thread(target=self._run_once, daemon=True)
            self._t.start()

        if blocking:
            self._t.join()

        out = {
            "ok": True,
            "msg": "finished" if blocking else "started",
            "running": self.running,
            "params": self._params(),
            "transcript": self.last_transcript,
            "reply": self.last_reply,  # always "" in STT-only mode
            "error": self.error,
        }
        return out

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "params": self._params(),
                "transcript": self.last_transcript,
                "reply": self.last_reply,
                "error": self.error,
            }

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self.running:
                return {"ok": True, "msg": "already stopped"}
            self._stop.set()
            t = self._t
        if t and t.is_alive():
            t.join(timeout=5)
        with self._lock:
            self.running = False
        return {"ok": True, "msg": "stopped"}

    def list_devices(self) -> Dict[str, Any]:
        pa = pyaudio.PyAudio()
        devices = []
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append({
                        "index": i,
                        "name": info.get("name", ""),
                        "rate_default": int(info.get("defaultSampleRate", 0)),
                        "max_input_channels": int(info.get("maxInputChannels", 0)),
                    })
        finally:
            pa.terminate()
        return {"devices": devices}

    def _params(self) -> Dict[str, Any]:
        return {
            "rate": self.rate, "channels": self.channels, "chunk": self.chunk,
            "energy_gate": self.energy_gate, "min_talk_ms": self.min_talk_ms,
            "end_sil_ms": self.end_sil_ms, "max_utter_ms": self.max_utter_ms,
            "pre_roll_ms": self.pre_roll_ms, "device_index": self.device_index,
        }

    def _auto_pick_device(self, pa: pyaudio.PyAudio) -> Optional[int]:
        try:
            candidates = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    name = (info.get("name") or "").lower()
                    score = 0
                    if "wslg" in name: score += 3
                    if "default" in name: score += 2
                    if "microphone" in name: score += 1
                    candidates.append((score, i))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]
        except Exception:
            pass
        return None

    def _run_once(self) -> None:
        try:
            pa = pyaudio.PyAudio()
            kwargs = dict(format=pyaudio.paInt16, channels=self.channels,
                          rate=self.rate, input=True, frames_per_buffer=self.chunk)
            if self.device_index is None:
                picked = self._auto_pick_device(pa)
                if picked is not None:
                    self.device_index = picked
                    log.info(f"Auto-selected input device index={self.device_index}")
            if self.device_index is not None:
                kwargs["input_device_index"] = int(self.device_index)
            stream = pa.open(**kwargs)

            speaking = False
            started_at = 0
            last_voice_ms = 0
            frames: List[bytes] = []

            pre = queue.Queue()
            pre_bytes_target = int(self.rate * (self.pre_roll_ms / 1000.0)) * 2

            while not self._stop.is_set():
                data = stream.read(self.chunk, exception_on_overflow=False)
                energy = _rms_energy(data)
                now_ms = _ms()

                if DEBUG_ENERGY:
                    log.debug(f"energy={energy:.2f} speaking={speaking}")

                pre.put(data)
                # trim pre-roll queue to size
                while True:
                    total = 0
                    for x in list(pre.queue):
                        total += len(x)
                        if total > pre_bytes_target:
                            break
                    if total <= pre_bytes_target:
                        break
                    try: pre.get_nowait()
                    except Exception: break

                if energy >= self.energy_gate:
                    if not speaking:
                        speaking = True
                        started_at = now_ms
                        frames = list(pre.queue)
                    last_voice_ms = now_ms
                    frames.append(data)
                else:
                    if speaking and last_voice_ms and (now_ms - last_voice_ms) >= self.end_sil_ms:
                        dur = now_ms - started_at
                        speaking = False
                        if dur >= self.min_talk_ms and frames:
                            wav = (self.tmp_base / f"utt_{started_at}.wav")
                            _write_wav(wav, frames, self.rate, self.channels)
                            text = _transcribe(wav)
                            with self._lock:
                                self.last_transcript = text
                        break

                if speaking and (now_ms - started_at) >= self.max_utter_ms:
                    speaking = False
                    if frames:
                        wav = (self.tmp_base / f"utt_{started_at}.wav")
                        _write_wav(wav, frames, self.rate, self.channels)
                        text = _transcribe(wav)
                        with self._lock:
                            self.last_transcript = text
                    break

                time.sleep(0.001)

            try:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            except Exception:
                pass

        except Exception as e:
            log.exception("TalkEngine error: %s", e)
            with self._lock:
                self.error = str(e)
        finally:
            with self._lock:
                self.running = False

# ---------- MCP server & tools ----------
mcp = FastMCP("talk")
_engine = TalkEngine()

@mcp.tool()
def talk_start(
    rate: Optional[int] = None,
    channels: Optional[int] = None,
    chunk: Optional[int] = None,
    energy_gate: Optional[float] = None,
    min_talk_ms: Optional[int] = None,
    end_sil_ms: Optional[int] = None,
    max_utter_ms: Optional[int] = None,
    pre_roll_ms: Optional[int] = None,
    device_index: Optional[int] = None,
    blocking: bool = True,
) -> Dict[str, Any]:
    """
    Capture one utterance with VAD and return the transcript only.
    No Gemini or TTS here — the agent will handle reply + audio.
    """
    out = _engine.start(
        rate, channels, chunk, energy_gate, min_talk_ms, end_sil_ms,
        max_utter_ms, pre_roll_ms, device_index, blocking
    )

    # If non-blocking, poll until finished
    if not blocking and out.get("running"):
        for _ in range(30):
            time.sleep(0.3)
            st = _engine.status()
            if not st.get("running"):
                out.update(st)
                break

    # STT-only result
    transcript = out.get("transcript") or _engine.last_transcript
    out["reply"] = ""                 # leave empty; Gemini will reply
    out["output"] = transcript or ""  # convenience field
    return out

@mcp.tool()
def talk_status() -> Dict[str, Any]:
    """Return current state, parameters, last transcript, last reply (unused), and error (if any)."""
    return _engine.status()

@mcp.tool()
def talk_stop() -> Dict[str, Any]:
    """Stop capture if running."""
    return _engine.stop()

@mcp.tool()
def list_devices() -> Dict[str, Any]:
    """List available input devices with indices."""
    return _engine.list_devices()

# -------- standalone TTS tool (no playback) --------
@mcp.tool()
def tts_speak(
    text: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Text → speech using OpenAI TTS. No playback.
    Returns { ok, path (absolute), error }.
    """
    if not text or not text.strip():
        return {"ok": False, "path": "", "error": "empty text"}

    v = voice or TTS_VOICE
    m = model or TTS_MODEL

    base = Path(TMP_DIR) if TMP_DIR else Path(tempfile.mkdtemp(prefix="talk_"))
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / f"tts_{_ms()}.mp3"   # fixed container

    ok, err = _tts_openai(text, out_path, model=m, voice=v)
    return {
        "ok": bool(ok),
        "path": str(out_path.resolve()) if ok else "",
        "error": err,
    }

if __name__ == "__main__":
    log.info("Starting Talk MCP (stdio)…")
    # stdio transport; DO NOT print to stdout
    mcp.run(transport="stdio")
