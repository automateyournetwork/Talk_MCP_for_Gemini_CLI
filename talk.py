#!/usr/bin/env python3
"""
MCP Server: Talk mode (PyAudio + Whisper) with tools:
  - talk_start(blocking?, device_index?, energy_gate?, end_sil_ms?, max_utter_ms?, ..., send_to_gemini?)
  - talk_status()
  - talk_stop()
  - list_devices()

Pure MCP (stdio). No Gemini HTTP/CLI calls except when send_to_gemini=true.
"""

import os, sys, time, wave, math, queue, shutil, tempfile, subprocess, logging, threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import shlex
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

GEMINI_BIN    = os.getenv("GEMINI_BIN", "gemini")     # for send_to_gemini=true
DEBUG_ENERGY  = os.getenv("TALK_DEBUG_ENERGY", "0") == "1"

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

def _run_gemini_cli_verbose(prompt_text: str) -> Dict[str, Any]:
    """
    Call Gemini CLI with the transcript. Supports 3 modes:
      1. GEMINI_USE_STDIN=1   → echo text into stdin of gemini
      2. GEMINI_ARGS template → args with {text} placeholder
      3. Default              → gemini --prompt "<text>"
    Returns: {"ok": bool, "stdout": str, "stderr": str,
              "returncode": int, "cmd": str}
    """
    bin_path = shutil.which(GEMINI_BIN) or GEMINI_BIN

    def result(ok, stdout, stderr, rc, cmd_list):
        return {
            "ok": ok,
            "stdout": (stdout or "").strip(),
            "stderr": (stderr or "").strip(),
            "returncode": rc,
            "cmd": " ".join(shlex.quote(c) for c in cmd_list),
        }

    try:
        # Mode 1: send transcript via stdin
        if os.getenv("GEMINI_USE_STDIN", "0") == "1":
            cmd = [bin_path]
            p = subprocess.run(
                cmd,
                input=prompt_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return result(p.returncode == 0, p.stdout, p.stderr, p.returncode, cmd)

        # Mode 2: args template
        args_tpl = os.getenv("GEMINI_ARGS", "").strip()
        if args_tpl:
            filled = args_tpl.replace("{text}", prompt_text)
            cmd = [bin_path, *shlex.split(filled)]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result(p.returncode == 0, p.stdout, p.stderr, p.returncode, cmd)

        # Mode 3: default
        cmd = [bin_path, "--prompt", prompt_text]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result(p.returncode == 0, p.stdout, p.stderr, p.returncode, cmd)

    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
            "returncode": -1,
            "cmd": bin_path,
        }

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
        self.last_reply: str = ""
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
            "reply": self.last_reply,
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
    Capture one utterance with VAD, send transcript to Gemini CLI, and return both.
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

    transcript = out.get("transcript") or _engine.last_transcript
    if transcript:
        g = _run_gemini_cli_verbose(transcript)
        _engine.last_reply = g.get("stdout", "").strip()
        out["reply"] = _engine.last_reply
        out["gemini"] = g

    # Always give a nice unified output
    out["output"] = out.get("reply") or transcript or ""
    return out

@mcp.tool()
def talk_status() -> Dict[str, Any]:
    """Return current state, parameters, last transcript, last reply, and error (if any)."""
    return _engine.status()

@mcp.tool()
def talk_stop() -> Dict[str, Any]:
    """Stop capture if running."""
    return _engine.stop()

@mcp.tool()
def list_devices() -> Dict[str, Any]:
    """List available input devices with indices."""
    return _engine.list_devices()

if __name__ == "__main__":
    log.info("Starting Talk MCP (stdio)…")
    # stdio transport; DO NOT print to stdout
    mcp.run(transport="stdio")
