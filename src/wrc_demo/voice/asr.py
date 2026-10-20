"""faster-whisper-backed speech-to-text for the dashboard mic button.

Design notes:
- faster-whisper is a POST-INSTALL pip dependency (same pattern as
  pyorbbecsdk2 -- do NOT add it to environment.yml; its ctranslate2 chain
  is heavy and CI/mock tests must not need it). Import happens lazily
  inside :meth:`WhisperASR._load`, never at module import.
- Transcription runs in whatever thread calls it. The stream server calls
  it from an HTTP handler thread, so a slow CPU decode blocks that one
  viewer request only, never the arm/skills (ThreadingHTTPServer).
- Audio arrives as a browser MediaRecorder blob (webm/opus or mp4/aac
  depending on the browser); faster-whisper decodes via PyAV, which
  handles both containers without conversion.
"""

from __future__ import annotations

import io
import json
import sys
import time


class ASRUnavailable(RuntimeError):
    """Raised when the engine is missing or transcription fails."""


class WhisperASR:
    """Lazy whisper holder; construct cheaply, pay the import on first use."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language or None  # "" in YAML -> None -> auto-detect
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ASRUnavailable(
                "faster-whisper is not installed; run "
                "`PYTHONNOUSERSITE=1 pip install faster-whisper` in the "
                "wrc-demo env (post-install step, see CLAUDE.md)"
            ) from e
        print(f"[voice] loading whisper model '{self.model_name}' "
              f"({self.device}/{self.compute_type})...", file=sys.stderr)
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type,
        )
        return self._model

    def transcribe(self, audio: bytes, mime: str = "audio/webm") -> str:
        """Transcribe an audio blob; returns '' when no speech was found."""
        if not audio:
            return ""
        model = self._load()
        try:
            segments, info = model.transcribe(
                io.BytesIO(audio),
                language=self.language,
                vad_filter=True,          # trims lead-in/lead-out silence
                beam_size=1,              # greedy: interactive latency
            )
        except Exception as e:  # decode or model error -> one clear message
            raise ASRUnavailable(f"transcription failed: {e}") from e
        text = " ".join(seg.text.strip() for seg in segments).strip()
        print(f"[voice] {info.language} p={info.language_probability:.2f} "
              f"-> {text!r}", file=sys.stderr)
        return text


class RemoteASR:
    """Transcribe by POSTing raw audio to a whisper HTTP server.

    Point ``voice.backend: remote`` at `scripts/serve_whisper_agx.py`
    running on the AGX: this machine (Jetson) keeps its CPU for
    perception + the 500 Hz control loop, the AGX burns its cores on
    whisper. Same ``fn(audio, mime) -> str`` contract as WhisperASR, so
    StreamServer.set_asr_fn cannot tell them apart.
    """

    def __init__(self, url: str, timeout_s: float = 90.0):
        self.url = url
        self.timeout_s = timeout_s  # 30 s clip on a busy CPU can take a while

    def transcribe(self, audio: bytes, mime: str = "audio/webm") -> str:
        if not audio:
            return ""
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.url, data=audio, method="POST",
            headers={"Content-Type": mime or "audio/webm"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                payload = json.loads(r.read().decode())
        except Exception as e:
            raise ASRUnavailable(
                f"remote ASR unreachable ({self.url}): {e}"
            ) from e
        text = str(payload.get("text", "") or "")
        print(f"[voice] remote -> {text!r}", file=sys.stderr)
        return text


def _voice_get(voice_cfg):
    if voice_cfg is None:
        return lambda k, d=None: d
    return voice_cfg.get if hasattr(voice_cfg, "get") else (lambda k, d=None: d)


def _health_ok(url: str, timeout_s: float = 2.5) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_remote_voice_services(voice_cfg) -> bool:
    """Health-check the AGX voice service; ssh-start it when down.

    The AGX hosts one process for both /asr and /tts (see
    scripts/serve_whisper_agx.py). After an AGX reboot it is NOT running
    -- this closes that gap at demo startup: probe /health, and when it is
    down, ``ssh <voice.ssh_host> '~/projects/wrc-asr/start.sh'`` (the
    start script's pid guard makes double-starts harmless) then poll until
    healthy. Returns True when healthy at the end; never raises -- voice
    is a convenience, the demo must run without it. An AGX that is powered
    off cannot be started from here: ssh fails fast and we bail out.
    """
    get = _voice_get(voice_cfg)
    if not bool(get("enabled", False)):
        return False
    base = None
    if str(get("backend", "local")).lower() == "remote":
        base = str(get("url", "") or "")
    spk = get("speak", None)
    if not base and spk is not None:
        base = str(spk.get("tts_url", "") or "")
    if not base:
        return False
    root = base.rsplit("/", 1)[0]  # http://ip:port
    health = f"{root}/health"
    if _health_ok(health):
        return True
    ssh_host = str(get("ssh_host", "agx"))
    print(f"[voice] AGX voice service down ({health}); "
          f"starting it via ssh {ssh_host}...", file=sys.stderr)
    import subprocess

    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             ssh_host, "~/projects/wrc-asr/start.sh"],
            check=False, capture_output=True, timeout=20,
        )
    except Exception as e:
        print(f"[voice] could not reach {ssh_host} ({e}); voice disabled "
              f"this session", file=sys.stderr)
        return False
    if r.returncode != 0:
        print(f"[voice] ssh start failed (rc={r.returncode}); voice "
              f"disabled this session", file=sys.stderr)
        return False
    for _ in range(15):  # whisper model load takes a few seconds
        time.sleep(2)
        if _health_ok(health):
            print("[voice] AGX voice service is up", file=sys.stderr)
            return True
    print("[voice] AGX voice service did not come up; voice disabled this "
          "session", file=sys.stderr)
    return False


def make_asr(voice_cfg) -> WhisperASR | RemoteASR | None:
    """Factory from the ``voice:`` config block (a Cfg or dict).

    Returns None (with a stderr note) when disabled or misconfigured --
    the demo keeps running without the mic button wired.
    """
    if voice_cfg is None:
        return None
    get = voice_cfg.get if hasattr(voice_cfg, "get") else (lambda k, d=None: d)
    if not bool(get("enabled", False)):
        return None
    if str(get("backend", "local")).lower() == "remote":
        url = str(get("url", "") or "")
        if not url:
            print("[voice] backend: remote but no voice.url set -- "
                  "voice input disabled", file=sys.stderr)
            return None
        return RemoteASR(url, timeout_s=float(get("timeout_s", 90.0)))
    return WhisperASR(
        model_name=str(get("model", "base")),
        device=str(get("device", "cpu")),
        compute_type=str(get("compute_type", "int8")),
        language=get("language") or None,
    )
