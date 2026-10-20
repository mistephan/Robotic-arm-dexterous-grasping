"""Voice announcements (key events -> Speaker -> AGX piper -> USB speaker).

Unit-level only: HTTP synthesis and aplay are injected fakes, so nothing
here touches the network, audio hardware, or the AGX. The piper server
side is exercised live on the AGX (see scripts/serve_whisper_agx.py).
"""

import time

from wrc_demo.voice.tts import Speaker, announce_text, make_speaker


def test_announce_text_maps_key_outcomes_to_chinese_phrases():
    assert announce_text("outcome", "grasp 'lime' FAILED: jaws closed on air") \
        == "抓取失败，夹爪没有抓到物体"
    assert announce_text("outcome", "pick attempt 1 failed: IK no convergence") \
        == "抓取失败，准备重试"
    assert announce_text("outcome", "placed 'pink object' at [0.2, 0.1, 0.05]") \
        == "已放置到位"
    assert announce_text("outcome", "released 'pink object'") == "已松开物体"


def test_announce_text_silent_for_unmapped_and_observations():
    # unmapped outcomes and routine observations must NOT speak
    assert announce_text("outcome", "IK OK (pos 1.2mm)") is None
    assert announce_text("observation", "pink object at [0.25, -0.01, 0.05]") is None
    assert announce_text("action", "picking pink object") is None


def test_announce_text_safety_notes():
    assert announce_text("note", "USER pressed STOP -- aborting") is None
    assert announce_text("note", "SafetyViolation: pregrasp target unsafe") \
        == "动作被安全系统拦截"
    assert announce_text("note", "e-stop latched by operator") == "急停触发"


def _fake_speaker(texts):
    played = []

    def synth(text):
        texts.append(text)
        return b"RIFF" + b"\x00" * 32  # wav magic + padding

    def player(wav):
        played.append(wav)

    spk = Speaker("http://tts/test", synth=synth, player=player)
    return spk, synth, player, played


def test_speaker_speaks_queued_text_end_to_end():
    spk, synth, player, played = _fake_speaker([])
    assert spk.say("任务完成")
    for _ in range(50):
        if played:
            break
        time.sleep(0.1)
    spk.stop()
    assert played and played[0].startswith(b"RIFF")


def test_speaker_debounces_and_rejects_empty():
    spk, _, _, played = _fake_speaker([])
    assert not spk.say("") and not spk.say("   ")
    assert spk.say("任务完成")
    assert not spk.say("任务完成")  # identical within 3 s
    spk.stop()
    time.sleep(0.2)
    assert not played or len(played) == 1


def test_speaker_survives_synth_failure():
    def boom(_text):
        raise RuntimeError("AGX down")

    spk = Speaker("http://tts/test", synth=boom, player=lambda w: None)
    assert spk.say("任务完成")      # queued fine
    time.sleep(0.6)                # worker hits the failure, logs, lives
    # different text (same text within 3 s is debounced by design);
    # say() returning True proves the worker thread is still accepting
    assert spk.say("依然存活")       # still alive and accepting
    spk.stop()


def test_make_speaker_disabled_and_missing_url():
    assert make_speaker(None) is None
    assert make_speaker({"speak": {"enabled": False}}) is None
    assert make_speaker({"speak": {"enabled": True}}) is None  # no url
    spk = make_speaker({"speak": {
        "enabled": True, "tts_url": "http://192.168.50.1:8001/tts",
        "alsa_device": "plughw:CARD=Device"}})
    try:
        assert spk.tts_url.endswith("/tts") and spk.alsa_device == "plughw:CARD=Device"
    finally:
        spk.stop()


def test_demo_yaml_wires_speak_block(monkeypatch):
    from wrc_demo.config import load_demo_config
    from wrc_demo.voice import tts as tts_mod

    hw = []
    monkeypatch.setattr(tts_mod, "set_hw_volume",
                        lambda dev, val: hw.append((dev, val)) or True)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    spk = make_speaker(cfg.get("voice", None))
    try:
        assert spk is not None
        assert spk.tts_url == "http://192.168.50.1:8001/tts"
        # hw_volume from demo.yaml is applied at construction (patched
        # here so the unit test never touches the real mixer)
        assert hw == [("plughw:CARD=Device", "92%")]
    finally:
        if spk is not None:
            spk.stop()


def test_set_hw_volume_builds_name_based_amixer_command(monkeypatch):
    import subprocess as sp

    from wrc_demo.voice.tts import set_hw_volume

    calls = []

    class R:
        returncode = 0
        stderr = b""

    monkeypatch.setattr(sp, "run",
                        lambda cmd, **kw: calls.append(cmd) or R())
    assert set_hw_volume("plughw:CARD=Device,DEV=0", "92%") is True
    assert calls == [["amixer", "-q", "-D", "hw:CARD=Device",
                      "sset", "Speaker", "92%"]]
    # no CARD= in the device string -> refuse instead of touching card 0
    assert set_hw_volume("default", "92%") is False


def test_set_hw_volume_failure_is_not_fatal(monkeypatch):
    import subprocess as sp

    from wrc_demo.voice.tts import set_hw_volume

    class R:
        returncode = 1
        stderr = b"amixer: Unable to find simple control"

    monkeypatch.setattr(sp, "run", lambda cmd, **kw: R())
    assert set_hw_volume("plughw:CARD=Device", "92%") is False
