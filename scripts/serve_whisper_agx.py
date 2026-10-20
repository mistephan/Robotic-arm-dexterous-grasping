#!/usr/bin/env python3
"""Voice services for the AGX: whisper ASR (/asr) + Piper TTS (/tts).

The Jetson drives the arm (500 Hz control loop), the cameras and YOLO;
burning its CPU cores on whisper/piper steals cycles from all of that.
This script is the ``voice.backend: remote`` / ``voice.speak.tts_url``
endpoint: a stdlib-only HTTP server, same deploy story as the GraspGen-X
server that already lives on the AGX.

Setup (once, on the AGX):

    ssh agx
    cd ~/projects/wrc-asr
    python3 -m venv venv
    venv/bin/pip install faster-whisper piper-tts
    # voice models (one-time, needs internet):
    mkdir -p voices && cd voices
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

Run:

    ./start.sh          # serve_whisper_agx.py --port 8001 --model base

Then in the Jetson's configs/demo.yaml:

    voice:
      enabled: true
      backend: remote
      url: http://192.168.50.1:8001/asr
      speak:
        enabled: true
        tts_url: http://192.168.50.1:8001/tts

Contract:
    POST /asr  raw audio bytes (audio/webm | mp4 | wav) -> {"text": "..."}
    POST /tts  {"text": "...", "lang": "zh"|"en"(optional, auto by script)}
               -> audio/wav bytes
    GET  /health -> {"ok": true, ...}
Errors answer JSON so the dashboard/speaker never choke on HTML pages.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import io

_MAX_BODY = 16 * 1024 * 1024  # ~4 min of opus; the dashboard caps at 30 s
_MAX_TTS_CHARS = 400         # announcements are one-liners, not audiobooks
_model = None
_lock = threading.Lock()  # ctranslate2 decode under one lock: no GPU thrash
_piper = {}                # lang -> PiperVoice, lazy per language
_piper_lock = threading.Lock()

_ASR_PROMPT = (
    "机器人桌面操作。中文或 English commands. "
    "香蕉 banana，芒果 mango，青柠 lime，藤篮 wicker basket。"
    "拿取，抓取，放置，pick up and place."
)

def _decode(audio: bytes, language: str | None):
    segments, info = _model.transcribe(
        io.BytesIO(audio), language=language, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300}, beam_size=5,
        condition_on_previous_text=False, initial_prompt=_ASR_PROMPT,
    )
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    score = (sum(float(getattr(s, "avg_logprob", -10.0)) for s in segments)
             / len(segments) if segments else -10.0)
    return text, info, score


def _load(model_name: str, device: str, compute_type: str):
    global _model
    from faster_whisper import WhisperModel

    print(f"[whisper-agx] loading model '{model_name}' "
          f"({device}/{compute_type})...", file=sys.stderr)
    _model = WhisperModel(model_name, device=device, compute_type=compute_type)


def _piper_voice(voice_path: str):
    key = voice_path
    with _piper_lock:
        if key in _piper:
            return _piper[key]
        from piper import PiperVoice

        print(f"[whisper-agx] loading piper voice '{voice_path}'...",
              file=sys.stderr)
        _piper[key] = PiperVoice.load(voice_path)
        return _piper[key]


def _tts_wav(text: str, lang: str, zh_voice: str, en_voice: str,
             volume: float = 0.8) -> bytes:
    """Synthesize one sentence to in-memory wav via piper.

    piper>=1.3 API: ``synthesize(text)`` yields AudioChunk objects (raw
    int16 + stream metadata); we assemble the wav container ourselves.
    The USB speaker has no ALSA mixer, so volume is applied here:
    peak-normalize UP to ``volume`` (0.8 = 80% of full scale), gain capped
    at 4x so a near-silent clip can't blast garbage.
    """
    import array
    import io
    import wave

    path = zh_voice if lang == "zh" else en_voice
    voice = _piper_voice(path)
    chunks = list(voice.synthesize(text))
    if not chunks:
        raise RuntimeError("piper produced no audio")
    samples = array.array("h")
    samples.frombytes(b"".join(c.audio_int16_bytes for c in chunks))
    peak = max(1, max(abs(s) for s in samples))
    gain = min(4.0, (volume * 32767.0) / peak)
    if gain > 1.01:  # amplify only; never make a loud clip louder
        for i in range(len(samples)):
            v = int(samples[i] * gain)
            samples[i] = 32767 if v > 32767 else (-32768 if v < -32768 else v)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(getattr(chunks[0], "channels", 0) or 1)
        w.setsampwidth(getattr(chunks[0], "sample_width", 0) or 2)
        w.setframerate(chunks[0].sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def make_handler(args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # quiet unless debugging
            pass

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/health":
                body = json.dumps({
                    "ok": True, "model": args.model,
                    "tts": bool(args.zh_voice and args.en_voice),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404, "POST audio to /asr, text to /tts")

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") == "/tts":
                return self._tts()
            if self.path.rstrip("/") != "/asr":
                body = json.dumps({"text": "", "error": "unknown path"}).encode()
                return self._json(body, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            if n <= 0 or n > _MAX_BODY:
                return self._json(
                    json.dumps({"text": "", "error": "audio too large or empty"}).encode(), 400)
            audio = self.rfile.read(n)
            t0 = time.monotonic()
            try:
                with _lock:
                    language = None if args.language == "auto" else args.language
                    text, info, score = _decode(audio, language)
                    if language is None and (
                        info.language not in {"zh", "en"}
                        or float(info.language_probability) < 0.60
                    ):
                        candidates = [_decode(audio, "zh"), _decode(audio, "en")]
                        text, info, score = max(candidates, key=lambda item: item[2])
                    if score < -1.25:
                        text = ""
            except Exception as e:
                return self._json(
                    json.dumps({"text": "", "error": f"transcription failed: {e}"[:200]}).encode(), 500)
            print(f"[whisper-agx] {info.language} p={info.language_probability:.2f} "
                  f"score={score:.2f} {time.monotonic() - t0:.1f}s -> {text!r}",
                  file=sys.stderr)
            return self._json(json.dumps({"text": text}).encode(), 200)

        def _tts(self):
            """POST /tts -- {"text", "lang"?} -> audio/wav bytes."""
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(json.dumps({"error": "bad JSON body"}).encode(), 400)
            text = str(req.get("text", "")).strip()
            if not text or len(text) > _MAX_TTS_CHARS:
                return self._json(json.dumps(
                    {"error": f"text must be 1..{_MAX_TTS_CHARS} chars"}).encode(), 400)
            lang = str(req.get("lang", "") or "").lower()
            if lang not in ("zh", "en"):  # auto: CJK anywhere -> zh voice
                lang = "zh" if any("一" <= c <= "鿿" for c in text) else "en"
            t0 = time.monotonic()
            try:
                wav = _tts_wav(text, lang, args.zh_voice, args.en_voice,
                               volume=float(args.volume))
            except FileNotFoundError as e:
                return self._json(json.dumps(
                    {"error": f"voice model missing: {e}"}).encode(), 500)
            except Exception as e:
                return self._json(json.dumps(
                    {"error": f"tts failed: {e}"[:200]}).encode(), 500)
            print(f"[whisper-agx] tts {lang} {time.monotonic() - t0:.1f}s "
                  f"{len(wav)}B -> {text[:60]!r}", file=sys.stderr)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)

        def _json(self, body: bytes, code: int):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="base",
                   help="tiny | base | small | ... (whisper model size)")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--compute-type", default="int8",
                   help="int8 (cpu) | float16 (cuda)")
    p.add_argument("--language", default="auto",
                   help="auto | zh | en | ...")
    p.add_argument("--zh-voice", default="voices/zh_CN-huayan-medium.onnx",
                   help="piper voice for Chinese announcements")
    p.add_argument("--en-voice", default="voices/en_US-lessac-medium.onnx",
                   help="piper voice for English announcements")
    p.add_argument("--volume", default="0.8", type=float,
                   help="peak-normalize announcements to this fraction of "
                        "full scale (speaker has no ALSA mixer)")
    args = p.parse_args()

    _load(args.model, args.device, args.compute_type)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(args))
    httpd.daemon_threads = True
    print(f"[whisper-agx] serving on :{args.port} (POST /asr, GET /health)",
          file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
