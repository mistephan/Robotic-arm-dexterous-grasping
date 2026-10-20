"""Voice input (push-to-talk -> POST /asr -> faster-whisper) contract tests.

No faster-whisper installation required anywhere in this file: the route is
exercised with a fake transcription fn, and WhisperASR's lazy-import design
is pinned explicitly (mock stack must stay free of the heavy dependency).
"""

import json
import sys
import urllib.error
import urllib.request

import pytest

from wrc_demo.apps.stream_server import StreamServer, _extract_multipart_audio
from wrc_demo.voice.asr import WhisperASR, make_asr


class _FakeStream:
    name = "mock"

    def latest(self):
        return None

    def overlay(self):
        return [], "idle"


class _FakeRig:
    names = ["mock"]

    def get(self, _n):
        return _FakeStream()

    def stats(self):
        return {"mock": {"fps": 0.0}}


def _multipart(audio: bytes, field: str = "audio", filename: str = "talk.webm"):
    boundary = "----wrctestboundary"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: audio/webm\r\n\r\n"
    ).encode()
    return (
        boundary,
        head + audio + f"\r\n--{boundary}--\r\n".encode(),
    )


@pytest.fixture
def server():
    srv = StreamServer(_FakeRig(), state_fn=lambda: {}, port=0)
    srv.start()
    yield srv
    srv.stop()


def _post(url, body, headers):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_asr_route_transcribes_via_wired_fn(server):
    seen = {}

    def fake_asr(audio, mime):
        seen["audio"], seen["mime"] = audio, mime
        return "把粉色方块放到托盘上"

    server.set_asr_fn(fake_asr)
    boundary, body = _multipart(b"\x1aE\xdf\xa3" + b"0" * 64)
    status, payload = _post(
        f"http://127.0.0.1:{server.port}/asr", body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert status == 200
    assert payload["text"] == "把粉色方块放到托盘上"
    assert seen["audio"].startswith(b"\x1aE\xdf\xa3") and seen["mime"] == "audio/webm"


def test_asr_route_reports_voice_availability_in_state(server):
    import urllib.request as ur

    with ur.urlopen(f"http://127.0.0.1:{server.port}/state", timeout=5) as r:
        assert json.loads(r.read().decode())["voice"] is False
    server.set_asr_fn(lambda a, m: "hi")
    with ur.urlopen(f"http://127.0.0.1:{server.port}/state", timeout=5) as r:
        assert json.loads(r.read().decode())["voice"] is True


def test_asr_route_501_when_not_wired(server):
    boundary, body = _multipart(b"x" * 32)
    with pytest.raises(urllib.error.HTTPError) as ei:
        _post(f"http://127.0.0.1:{server.port}/asr", body,
              {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert ei.value.code == 501
    assert "voice input not configured" in ei.value.read().decode()


def test_asr_route_400_on_wrong_content_type(server):
    server.set_asr_fn(lambda a, m: "hi")
    with pytest.raises(urllib.error.HTTPError) as ei:
        _post(f"http://127.0.0.1:{server.port}/asr", b"{}",
              {"Content-Type": "application/json"})
    assert ei.value.code == 400


def test_asr_route_500_json_when_engine_fails(server):
    def boom(_a, _m):
        raise RuntimeError("faster-whisper is not installed; pip install it")

    server.set_asr_fn(boom)
    boundary, body = _multipart(b"x" * 32)
    with pytest.raises(urllib.error.HTTPError) as ei:
        _post(f"http://127.0.0.1:{server.port}/asr", body,
              {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert ei.value.code == 500
    assert "pip install" in json.loads(ei.value.read().decode())["error"]


def test_extract_multipart_audio_skips_fields_takes_file():
    audio = b"FAKEAUDIODATA"
    boundary = "----wrctestboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="kind"\r\n\r\nvoice\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="talk.webm"\r\n'
        f"Content-Type: audio/webm\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    got, mime = _extract_multipart_audio(
        f"multipart/form-data; boundary={boundary}", body)
    assert got == audio and mime == "audio/webm"


def test_extract_multipart_audio_garbage_returns_none():
    assert _extract_multipart_audio("multipart/form-data; boundary=x", b"junk") == (None, None)


def test_whisper_asr_is_lazy_and_skips_empty_audio():
    # Empty blob -> "" WITHOUT importing faster-whisper (mock-stack guarantee:
    # constructing and calling must never pull the heavy dependency).
    asr = WhisperASR(model_name="base")
    assert asr.transcribe(b"") == ""
    assert "faster_whisper" not in sys.modules


def test_whisper_asr_joins_segments_via_injected_model():
    class _Seg:
        def __init__(self, text):
            self.text = text

    class _Info:
        language, language_probability = "zh", 0.93

    class _FakeModel:
        def transcribe(self, buf, **kw):
            assert kw.get("vad_filter") is True
            return iter([_Seg(" pick the pink "), _Seg("object ")]), _Info()

    asr = WhisperASR()
    asr._model = _FakeModel()
    assert asr.transcribe(b"xx") == "pick the pink object"


def test_make_asr_disabled_returns_none_and_config_passthrough():
    assert make_asr(None) is None
    assert make_asr({"enabled": False}) is None
    asr = make_asr({"enabled": True, "model": "small", "language": ""})
    assert isinstance(asr, WhisperASR)
    assert asr.model_name == "small" and asr.language is None  # "" -> auto


def test_demo_yaml_wires_voice_block():
    from conftest import REPO  # noqa: F401  (documents the import path used)
    from wrc_demo.config import load_demo_config
    from wrc_demo.voice.asr import RemoteASR

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    asr = make_asr(cfg.get("voice", None))
    # shipped default is the AGX remote backend (deployed 2026-09-17);
    # flip voice.backend to local to run whisper on the Jetson instead.
    assert isinstance(asr, RemoteASR)
    assert asr.url.startswith("http://192.168.50.1:8001")


def test_dashboard_template_has_push_to_talk_mic_button():
    from wrc_demo.apps.stream_server import _INDEX_HTML

    html = _INDEX_HTML.format(tiles="")
    assert 'id="micbtn"' in html
    assert "getUserMedia" in html
    assert "MediaRecorder" in html
    assert "pointerdown" in html and "pointerup" in html
    assert "micStop" in html and "/asr" in html
    # transcription fills the box for human review -- never auto-sends
    assert "box.value = r.text" in html
    # 30 s hard cap (user decision 2026-09-17; was 10 s)
    assert "setTimeout(micStop, 30000)" in html


# ── remote backend (whisper on the AGX) ─────────────────────────────────


class _StubASRServer:
    """Minimal stand-in for scripts/serve_whisper_agx.py."""

    def __init__(self, status=200, payload=b'{"text": "pick the pink object"}'):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                outer.seen = (self.rfile.read(n),
                              self.headers.get("Content-Type"))
                body = outer.payload
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.status, self.payload = status, payload
        self.seen = None
        import threading

        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_remote_asr_posts_raw_audio_and_parses_text():
    stub = _StubASRServer()
    try:
        from wrc_demo.voice.asr import RemoteASR

        asr = RemoteASR(f"http://127.0.0.1:{stub.port}/asr", timeout_s=5)
        assert asr.transcribe(b"\x1aE\xdf\xa3pad", "audio/webm") == "pick the pink object"
        assert stub.seen[0].startswith(b"\x1aE\xdf\xa3")
        assert stub.seen[1] == "audio/webm"
        assert asr.transcribe(b"") == ""  # empty shortcut, no request
    finally:
        stub.close()


def test_remote_asr_raises_asrunavailable_when_down():
    from wrc_demo.voice.asr import ASRUnavailable, RemoteASR

    asr = RemoteASR("http://127.0.0.1:1/asr", timeout_s=1)
    with pytest.raises(ASRUnavailable, match="remote ASR unreachable"):
        asr.transcribe(b"xx")


def test_make_asr_remote_backend_needs_url():
    assert make_asr({"enabled": True, "backend": "remote"}) is None
    asr = make_asr({"enabled": True, "backend": "remote",
                    "url": "http://192.168.50.1:8001/asr"})
    from wrc_demo.voice.asr import RemoteASR

    assert isinstance(asr, RemoteASR)
    assert asr.url == "http://192.168.50.1:8001/asr"


# ── AGX voice service auto-start at demo boot ───────────────────────────


def test_ensure_remote_noop_when_healthy(monkeypatch):
    from wrc_demo.voice import asr as asr_mod

    probes = []
    monkeypatch.setattr(asr_mod, "_health_ok",
                        lambda url, t=2.5: probes.append(url) or True)

    import subprocess as sp

    ssh_calls = []
    monkeypatch.setattr(sp, "run",
                        lambda *a, **k: ssh_calls.append(a) or None)
    assert asr_mod.ensure_remote_voice_services(
        {"enabled": True, "backend": "remote",
         "url": "http://192.168.50.1:8001/asr"}) is True
    assert len(probes) == 1 and not ssh_calls  # healthy: one probe, no ssh


def test_ensure_remote_starts_via_ssh_when_down(monkeypatch):
    from wrc_demo.voice import asr as asr_mod

    flips = {"n": 0}

    def fake_health(url, t=2.5):
        flips["n"] += 1
        return flips["n"] > 1  # down at first probe, up after ssh start

    monkeypatch.setattr(asr_mod, "_health_ok", fake_health)
    monkeypatch.setattr(asr_mod.time, "sleep", lambda s: None)

    import subprocess as sp

    class R:
        returncode = 0

    monkeypatch.setattr(sp, "run", lambda *a, **k: R())
    assert asr_mod.ensure_remote_voice_services(
        {"enabled": True, "speak": {"tts_url": "http://10.0.0.9:8001/tts"}}) is True


def test_ensure_remote_bails_fast_when_ssh_fails(monkeypatch):
    from wrc_demo.voice import asr as asr_mod

    monkeypatch.setattr(asr_mod, "_health_ok", lambda url, t=2.5: False)

    import subprocess as sp

    class R:
        returncode = 255  # ssh: no route / auth failure

    monkeypatch.setattr(sp, "run", lambda *a, **k: R())
    assert asr_mod.ensure_remote_voice_services(
        {"enabled": True, "backend": "remote",
         "url": "http://192.168.50.1:8001/asr"}) is False


def test_ensure_remote_disabled_is_free():
    from wrc_demo.voice.asr import ensure_remote_voice_services

    assert ensure_remote_voice_services(None) is False
    assert ensure_remote_voice_services({"enabled": False}) is False


# ── TLS (mic access needs https on LAN addresses) ───────────────────────


def test_stream_server_tls_serves_https():
    import shutil
    import ssl
    import tempfile
    from pathlib import Path

    if shutil.which("openssl") is None:
        pytest.skip("openssl binary not available")

    with tempfile.TemporaryDirectory() as td:
        cert = Path(td) / "cert.pem"
        key = Path(td) / "key.pem"
        srv = StreamServer(_FakeRig(), state_fn=lambda: {}, port=0,
                           tls=True, tls_cert=cert, tls_key=key)
        try:
            srv.start()
            assert srv.url.startswith("https://")
            assert cert.exists() and key.exists()  # generated + cached
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(f"https://127.0.0.1:{srv.port}/",
                                        context=ctx, timeout=5) as r:
                assert r.status == 200
                assert b"micbtn" in r.read()
            # plain http no longer speaks to this port
            with pytest.raises(Exception):
                urllib.request.urlopen(
                    f"http://127.0.0.1:{srv.port}/", timeout=5)
        finally:
            srv.stop()


def test_stream_server_plain_http_when_tls_off():
    srv = StreamServer(_FakeRig(), state_fn=lambda: {}, port=0)
    try:
        srv.start()
        assert srv.url.startswith("http://")
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/state",
                                    timeout=5) as r:
            assert r.status == 200
    finally:
        srv.stop()
