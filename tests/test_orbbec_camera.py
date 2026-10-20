"""Tests for the OrbbecCamera backend (fake SDK injection only)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from wrc_demo.perception.camera_base import CameraError, make_camera
from wrc_demo.perception.orbbec_camera import list_orbbec_devices
from wrc_demo.config import Cfg


def test_orbbec_camera_uses_lazy_import(monkeypatch):
    """Constructing the backend must not trigger pyorbbecsdk import."""
    # Make sure pyorbbecsdk isn't importable
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    cam = make_camera(Cfg({"type": "orbbec", "serial": "ABC"}))
    assert cam.has_depth is True


def test_orbbec_camera_raises_without_sdk(monkeypatch):
    """Without the SDK installed, open() must raise a CameraError."""
    from wrc_demo.perception import orbbec_camera as oc

    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)

    cam = oc.OrbbecCamera(Cfg({"type": "orbbec", "serial": "ABC"}))
    with pytest.raises(CameraError):
        cam.open()


def test_orbbec_camera_open_and_grab(fake_orbbec_sdk):
    cam = make_camera(Cfg({"type": "orbbec", "width": 640, "height": 480}))
    cam.open()
    try:
        f = cam.get_frame()
    finally:
        cam.close()
    assert f.rgb.shape == (480, 640, 3)
    assert f.depth_m.shape == (480, 640)
    assert f.depth_m.dtype == np.float32
    # The fake SDK returns 600 mm depth on every pixel.
    assert np.allclose(f.depth_m, 0.6, atol=1e-6)
    assert f.K.shape == (3, 3)
    assert f.K[0, 0] > 0
    assert f.K[1, 1] > 0
    assert f.depth_source == "sensor"


def test_orbbec_camera_grab_calls_align(fake_orbbec_sdk):
    cam = make_camera(Cfg({"type": "orbbec"}))
    cam.open()
    try:
        cam.get_frame()
    finally:
        cam.close()
    # The fake AlignFilter records the number of process() calls.
    af = cam._align
    assert af is not None
    assert af.process_calls >= 1


def test_orbbec_camera_lifecycle_close(fake_orbbec_sdk):
    cam = make_camera(Cfg({"type": "orbbec"}))
    cam.open()
    cam.close()
    assert cam._pipeline is None


def test_list_orbbec_devices_empty(fake_orbbec_sdk):
    out = list_orbbec_devices()
    assert out == []


def test_list_orbbec_devices_returns_serials(fake_orbbec_sdk, monkeypatch):
    # Inject a single fake device via the (fake) Context. Reach into the
    # fake SDK namespace directly to access the helper classes.
    fake_sdk = sys.modules["pyorbbecsdk"]
    DeviceInfo = fake_sdk.DeviceInfo
    ctx_factory = fake_sdk.Context
    # Replace the Context constructor with a closure that returns our
    # desired list of devices.
    dev_info = DeviceInfo(name="Gemini2", serial="AY3Z331006L")
    class _Dev:
        def get_device_info(self):
            return dev_info
    monkeypatch.setattr(fake_sdk, "Context", lambda: ctx_factory([_Dev()]))
    out = list_orbbec_devices()
    assert len(out) == 1
    assert out[0]["serial"] == "AY3Z331006L"


# ─────────────────────────────────────────────────────────────────────────
# Regression: Pipeline must actually bind to the requested serial.
# (HANDOFF §11 issue 12: pyorbbecsdk v2 has no set_device_serial_number;
# the previous code silently fell back to the first enumerated device,
# which manifested as "I asked for the top camera but the wrist view is
# showing" whenever the binding in cameras.json didn't match the USB
# enumeration order.)
# ─────────────────────────────────────────────────────────────────────────


def _make_fake_devices(fake_sdk, serials):
    """Build a list of fake Devices each carrying its own serial.

    NOTE: this uses default-arg capture so each ``_D`` instance holds a
    *distinct* ``DeviceInfo``; naive closure-over-loop-variable would
    alias every instance to the last serial in the list.
    """
    DeviceInfo = fake_sdk.DeviceInfo
    devs = []
    for s in serials:
        def _make(di=s):
            info = DeviceInfo(name="Gemini2", serial=di)
            class _D:
                def get_device_info(self):
                    return info
            return _D()
        devs.append(_make())
    return devs


def test_orbbec_camera_binds_pipeline_to_requested_serial(fake_orbbec_sdk, monkeypatch):
    fake_sdk = sys.modules["pyorbbecsdk"]
    Context = fake_sdk.Context

    devs = _make_fake_devices(fake_sdk, ["AY3794300W4", "AY3Z331006L"])
    monkeypatch.setattr(fake_sdk, "Context", lambda: Context(devs))

    # Spy on Pipeline(device) so we can see what was passed.
    import wrc_demo.perception.orbbec_camera as oc
    seen = {"device": None}
    RealPipeline = fake_sdk.Pipeline
    def _spy_pipeline(device=None):
        seen["device"] = device
        return RealPipeline(device)
    monkeypatch.setattr(fake_sdk, "Pipeline", _spy_pipeline)

    cam = make_camera(Cfg({"type": "orbbec", "serial": "AY3Z331006L"}))
    cam.open()
    try:
        # The Device passed to Pipeline must be the AY3Z331006L one.
        assert seen["device"] is not None
        assert seen["device"].get_device_info().get_serial_number() == "AY3Z331006L"
        # cam.serial echoes the requested serial.
        assert cam.serial == "AY3Z331006L"
    finally:
        cam.close()


def test_orbbec_camera_raises_when_serial_not_present(fake_orbbec_sdk, monkeypatch):
    """If the requested serial isn't in the enumerated set, open() must
    raise CameraError immediately (no silent fallback to first device)."""
    fake_sdk = sys.modules["pyorbbecsdk"]
    Context = fake_sdk.Context
    devs = _make_fake_devices(fake_sdk, ["AY3794300W4"])  # AY3Z331006L absent
    monkeypatch.setattr(fake_sdk, "Context", lambda: Context(devs))

    cam = make_camera(Cfg({"type": "orbbec", "serial": "AY3Z331006L"}))
    with pytest.raises(CameraError) as ei:
        cam.open()
    assert "AY3Z331006L" in str(ei.value)
    assert "not found" in str(ei.value)


def test_orbbec_camera_detects_serial_mismatch_after_open(fake_orbbec_sdk, monkeypatch):
    """If the SDK's Pipeline(device) silently opens a different device
    (some buggy SDK paths ignore the device arg), OrbbecCamera.open()
    must catch the mismatch and raise -- otherwise we get the
    wrong-camera bug back."""
    fake_sdk = sys.modules["pyorbbecsdk"]
    Context = fake_sdk.Context
    Pipeline = fake_sdk.Pipeline

    devs = _make_fake_devices(fake_sdk, ["AY3794300W4", "AY3Z331006L"])
    monkeypatch.setattr(fake_sdk, "Context", lambda: Context(devs))

    # Build a "lying" Pipeline subclass that ignores its device arg and
    # always binds to the FIRST device (mimicking a broken SDK).
    class _LyingPipeline(Pipeline):
        def __init__(self, device=None):
            # Always pick the first device regardless of what was asked.
            super().__init__(devs[0])
    monkeypatch.setattr(fake_sdk, "Pipeline", _LyingPipeline)

    cam = make_camera(Cfg({"type": "orbbec", "serial": "AY3Z331006L"}))
    with pytest.raises(CameraError) as ei:
        cam.open()
    msg = str(ei.value)
    assert "serial mismatch" in msg.lower() or "mismatch" in msg.lower()
    assert "AY3Z331006L" in msg
    assert "AY3794300W4" in msg
