"""Voice announcements: key runtime events -> Piper TTS (AGX) -> USB speaker.

``Speaker.say(text)`` only queues; a daemon worker synthesizes on the AGX
(``POST /tts`` -> wav bytes) and plays through ALSA (``aplay``). None of it
touches the skill/control threads -- narration must never block motion.

Event selection: ``announce_text(kind, text)`` maps the small closed set of
English runtime events to fixed Chinese phrases (deterministic, no LLM);
anything unmapped stays silent ("key events only", user decision
2026-09-17). Unmapped-but-explicit ``say()`` calls (task echo) are spoken
verbatim -- the AGX picks the zh/en voice by script detection.

Failure policy: AGX down / aplay missing / bad ALSA device -> log one line,
drop the announcement, keep the demo running.
"""

from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time


# (pattern, announcement) -- first match wins. Order matters: specific
# failure causes before the generic "grasp FAILED".
_PHRASES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"jaws closed on air", re.I), "抓取失败，夹爪没有抓到物体"),
    (re.compile(r"pick attempt \d+ failed", re.I), "抓取失败，准备重试"),
    (re.compile(r"grasped\b|grasp .*success", re.I), "抓取成功"),
    (re.compile(r"grasp .*FAILED", re.I), "抓取失败"),
    (re.compile(r"^placed ", re.I), "已放置到位"),
    (re.compile(r"^released ", re.I), "已松开物体"),
)


def announce_text(kind: str, text: str) -> str | None:
    """Narration event -> phrase to speak; None means stay silent."""
    if kind == "outcome":
        for pat, phrase in _PHRASES:
            if pat.search(text):
                return phrase
        return None
    if kind == "note":
        low = text.lower()
        if "estop" in low or "e-stop" in low:
            return "急停触发"
        if "safetyviolation" in low or "unsafe" in low or "out of limits" in low:
            return "动作被安全系统拦截"
    return None


def _aplay_player(device: str):
    """Default player factory: aplay on the configured ALSA device."""

    def play(wav: bytes) -> None:
        subprocess.run(
            ["aplay", "-D", device, "-q", "-"],
            input=wav, timeout=60, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return play


def _http_synthesize(url: str, timeout_s: float):
    """Default synth factory: POST {"text"} to the AGX piper service."""

    def synth(text: str) -> bytes:
        import json
        import urllib.request

        req = urllib.request.Request(
            url, method="POST", headers={"Content-Type": "application/json"},
            data=json.dumps({"text": text}).encode(),
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            wav = r.read()
        if not wav.startswith(b"RIFF"):
            raise RuntimeError("tts response is not a wav file")
        return wav

    return synth


class Speaker:
    """Queued announcer; safe to call from any thread (skills, dashboard)."""

    def __init__(
        self,
        tts_url: str,
        alsa_device: str = "plughw:CARD=Device",
        timeout_s: float = 30.0,
        synth=None,
        player=None,
    ):
        self.tts_url = tts_url
        self.alsa_device = alsa_device
        self._synth = synth or _http_synthesize(tts_url, timeout_s)
        self._play = player or _aplay_player(alsa_device)
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._last: tuple[str, float] = ("", 0.0)
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="voice-tts")
        self._thread.start()

    def say(self, text: str) -> bool:
        """Queue an announcement verbatim; False if dropped/debounced."""
        text = (text or "").strip()
        if not text or not self._thread.is_alive():
            return False
        now = time.monotonic()
        if text == self._last[0] and now - self._last[1] < 3.0:
            return False  # identical announcement within 3 s: say it once
        self._last = (text, now)
        try:
            self._q.put_nowait(text)
            return True
        except queue.Full:
            print("[voice-tts] busy, dropped announcement", file=sys.stderr)
            return False

    def observe(self, kind: str, text: str) -> None:
        """memory.add hook: speaks only mapped key events."""
        phrase = announce_text(kind, text)
        if phrase:
            self.say(phrase)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._play(self._synth(text))
            except Exception as e:
                print(f"[voice-tts] announcement dropped: {e}", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def set_hw_volume(alsa_device: str, value: str) -> bool:
    """Re-apply the speaker's HARDWARE mixer level (alsamixer equivalent).

    Manual alsamixer changes live in the USB device and reset when it
    loses power (replug / host reboot); demo.yaml's ``hw_volume`` makes
    the level survive by re-applying it at every demo start. Derives the
    card NAME from alsa_device (``plughw:CARD=Device`` -> ``Device``) --
    names are stable, indices shift on every replug.
    """
    import subprocess

    if "CARD=" not in alsa_device:
        print(f"[voice-tts] cannot derive card name from {alsa_device!r}; "
              f"hw volume not applied", file=sys.stderr)
        return False
    name = alsa_device.split("CARD=")[-1].split(",")[0]
    try:
        r = subprocess.run(
            ["amixer", "-q", "-D", f"hw:CARD={name}", "sset", "Speaker", value],
            capture_output=True, timeout=5,
        )
    except Exception as e:
        print(f"[voice-tts] hw volume set failed: {e}", file=sys.stderr)
        return False
    ok = r.returncode == 0
    print(f"[voice-tts] hw volume {name!r} Speaker -> {value}: "
          f"{'ok' if ok else r.stderr.decode()[:80]}", file=sys.stderr)
    return ok


def make_speaker(voice_cfg) -> Speaker | None:
    """Factory from the ``voice.speak:`` block; None when disabled."""
    if voice_cfg is None:
        return None
    get = voice_cfg.get if hasattr(voice_cfg, "get") else (lambda k, d=None: d)
    spk = get("speak", None)
    if spk is None or not bool(spk.get("enabled", False)):
        return None
    url = str(spk.get("tts_url", "") or "")
    if not url:
        print("[voice-tts] speak.enabled but speak.tts_url missing -- "
              "announcements disabled", file=sys.stderr)
        return None
    device = str(spk.get("alsa_device", "plughw:CARD=Device"))
    hw_volume = str(spk.get("hw_volume", "") or "")
    if hw_volume:
        set_hw_volume(device, hw_volume)  # best effort, never fatal
    return Speaker(
        url,
        alsa_device=device,
        timeout_s=float(spk.get("timeout_s", 30.0)),
    )
