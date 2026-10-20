import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# wrc_demo ``__init__.py`` injects ``wrc_demo/sdk/`` into ``sys.path`` so
# ``reBotArm_control_py`` (the vendored motor SDK) is importable. The
# vendored SDK + ``WrcGripper`` (in ``wrc_demo.control.gripper``) are
# the only arm-control dependencies — wrc_demo no longer references
# rebot_grasp at runtime (see ADR-0005).

from wrc_demo.config import load_demo_config  # noqa: E402

URDF = REPO / "assets" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
# The shipped URDF is authored in the mirrored joint convention (q_asset = -q_local);
# the SDK and every q constant in this repo are local. The flip is baked into
# the model at load time (see Kinematics.__init__), so every downstream
# consumer (FK, IK, limits, safety) speaks local convention.
JOINT_SIGNS = [-1, -1, -1, -1, -1, -1]


def has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401

        return True
    except ImportError:
        return False


needs_pin = pytest.mark.skipif(
    not has_pinocchio() or not URDF.exists(),
    reason="pinocchio or RS model not available",
)


@pytest.fixture
def demo_cfg():
    return load_demo_config(camera="mock", arm="mock", llm="mock")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────
# Fake Orbbec SDK injection (PLAN ADR-6: tests never import the real SDK).
# ─────────────────────────────────────────────────────────────────────────

class _FakeIntrinsic:
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.dcoeff = []


class _FakeColorStreamProfile:
    def __init__(self, w: int, h: int, fx: float, fy: float, cx: float, cy: float) -> None:
        self._w, self._h = w, h
        self._intrinsic = _FakeIntrinsic(fx, fy, cx, cy)

    def get_intrinsic(self):
        return self._intrinsic

    def is_video_stream_profile(self):
        return True

    def is_accel_stream_profile(self):
        return False

    def is_gyro_stream_profile(self):
        return False

    def is_lidar_stream_profile(self):
        return False

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_fps(self):
        return 30

    def get_format(self):
        return 0

    def get_type(self):
        return 0


class _FakeFrame:
    def __init__(self, w: int, h: int, payload: bytes, value_scale: float = 1.0) -> None:
        self._w, self._h = w, h
        self._payload = payload
        self._value_scale = value_scale

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_data(self):
        return self._payload


class _FakeColorFrame(_FakeFrame):
    def __init__(self, w: int, h: int, bgr: np.ndarray) -> None:
        super().__init__(w, h, bgr.tobytes())
        self._bgr_shape = bgr.shape

    def get_height(self):  # type: ignore[override]
        return self._bgr_shape[0]


class _FakeDepthFrame(_FakeFrame):
    def __init__(self, w: int, h: int, depth_mm: np.ndarray, depth_scale: float) -> None:
        super().__init__(w, h, depth_mm.astype(np.uint16).tobytes(), value_scale=depth_scale)
        self._depth_shape = depth_mm.shape

    def get_height(self):  # type: ignore[override]
        return self._depth_shape[0]


class _FakeDepthSensor:
    def get_depth_scale(self):
        return 0.001


class _FakeStreamProfileList:
    def __init__(self, profiles: list) -> None:
        self._profiles = profiles

    def get_count(self):
        return len(self._profiles)

    def get_stream_profile_by_index(self, i: int):
        return self._profiles[i]


class _FakeFrameSet:
    def __init__(self, color: _FakeColorFrame | None, depth: _FakeDepthFrame | None) -> None:
        self._color, self._depth = color, depth

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth


class _FakeAlignFilter:
    def __init__(self, target) -> None:
        self._target = target
        self.process_calls = 0

    def process(self, frames):
        self.process_calls += 1
        return frames


class _FakePipeline:
    # Test profiles the picker may pick from. Mirrors the Gemini 2
    # advertised resolutions (subset).
    _ADVERTISED_RES = [
        (1920, 1080),
        (1280, 720),
        (640, 480),
    ]

    def __init__(self, device=None) -> None:
        # pyorbbecsdk v2 binds the pipeline to a specific device via the
        # ``Pipeline(device)`` constructor (see official example
        # examples/beginner/01_hello_camera.py:96). Old code used a
        # non-existent ``set_device_serial_number`` -- we mirror the v2
        # API here.
        self._cfg = None
        self.started = False
        self._counter = 0
        self._color_profile = None
        self._device = device

    def set_device_serial_number(self, s: str) -> None:
        # Legacy v1-ish API. The current OrbbecCamera code no longer
        # calls this, but keep it as a no-op so unrelated tests don't
        # break if some legacy helper still pokes at it.
        if self._device is not None:
            return  # already bound by constructor
        # Synthesize a stub Device so the post-start verification can
        # read back a serial via get_device_info().
        self._device = _FakeDevice(_FakeDeviceInfo(name="FakeGemini2", serial=s))

    def get_device(self) -> "_FakeDevice | None":
        return self._device

    @staticmethod
    def get_supported_stream_profiles(_sensor_type):
        return None

    def _all_profiles(self):
        out = []
        for (w, h) in self._ADVERTISED_RES:
            fx, fy = float(w), float(w)
            out.append(_FakeColorStreamProfile(w, h, fx, fy, w / 2.0, h / 2.0))
        return out

    def get_stream_profile_list(self, sensor_type):
        return _FakeStreamProfileList(self._all_profiles())

    def start(self, cfg):
        self._cfg = cfg
        self.started = True
        # Pick the profile the user requested via cfg._w/_h; fall back
        # to the first available profile.
        w_req = int(getattr(cfg, "_w", 1280))
        h_req = int(getattr(cfg, "_h", 720))
        chosen = None
        for prof in self._all_profiles():
            if prof.get_width() == w_req and prof.get_height() == h_req:
                chosen = prof
                break
        if chosen is None:
            chosen = self._all_profiles()[0]
        self._color_profile = chosen
        # Return value is the opened "Device" handle; we re-use the
        # module-level _FakeDevice class (it now exposes both
        # get_device_info and get_depth_sensor).
        return _FakeDevice(
            _FakeDeviceInfo(
                name="FakeGemini2",
                serial=self._device.get_device_info().get_serial_number()
                if self._device is not None and hasattr(self._device, "get_device_info")
                else "",
            )
        )

    def stop(self) -> None:
        self.started = False

    def wait_for_frames(self, _timeout_ms: int):
        self._counter += 1
        # Use the picked profile's w/h so the frame matches what
        # OrbbecCamera thinks it configured.
        if self._color_profile is not None:
            w = self._color_profile._w
            h = self._color_profile._h
        else:
            w, h = 1280, 720
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[..., 0] = (self._counter * 7) % 256
        bgr[..., 1] = 200
        bgr[..., 2] = 200
        depth_mm = np.full((h, w), 600, dtype=np.uint16)  # 600 mm = 0.6 m
        return _FakeFrameSet(
            _FakeColorFrame(w, h, bgr), _FakeDepthFrame(w, h, depth_mm, 0.001)
        )


class _FakeConfig:
    def __init__(self) -> None:
        self._serial = ""
        self._align = 0
        self._w = 1280
        self._h = 720
        self._streams: list = []
        self._profiles: list = []

    def enable_stream(self, profile, **_kw) -> None:
        # v2 SDK accepts a single StreamProfile. Track it so callers can
        # introspect what was configured.
        self._profiles.append(profile)
        if hasattr(profile, "get_width"):
            self._w, self._h = profile.get_width(), profile.get_height()

    def set_device_serial_number(self, s: str) -> None:
        self._serial = s

    def set_align_mode(self, m) -> None:
        self._align = m


class _FakeDeviceInfo:
    def __init__(self, name: str, serial: str, pid: int = 0, vid: int = 0) -> None:
        self._name, self._serial, self._pid, self._vid = name, serial, pid, vid

    def get_name(self):
        return self._name

    def get_serial_number(self):
        return self._serial

    def get_pid(self):
        return self._pid

    def get_vid(self):
        return self._vid


class _FakeDevice:
    def __init__(self, info: _FakeDeviceInfo) -> None:
        self._info = info

    def get_device_info(self) -> _FakeDeviceInfo:
        return self._info

    def get_depth_sensor(self) -> _FakeDepthSensor:
        return _FakeDepthSensor()


class _FakeDeviceList:
    def __init__(self, items: list[_FakeDevice]) -> None:
        self._items = items

    def get_count(self):
        return len(self._items)

    def get_device_by_index(self, i: int):
        return self._items[i]

    def __getitem__(self, i: int):
        return self._items[i]


class _FakeContext:
    def __init__(self, devices: list[_FakeDevice] | None = None) -> None:
        self._devices = _FakeDeviceList(devices or [])

    def query_devices(self):
        return self._devices


class FakeOrbbecSDK(types.SimpleNamespace):
    """Drop-in replacement for the ``pyorbbecsdk`` module.

    Exposes the v2 surface (>= pyorbbecsdk 2.x): OBSensorType,
    OBStreamType, OBFormat, OBAlignMode, a Pipeline.get_stream_profile_list
    that returns StreamProfileList, and a Config whose enable_stream
    takes a single StreamProfile. Mirrors the real SDK closely enough
    for ``OrbbecCamera.open()`` to drive without a device.
    """

    Pipeline = _FakePipeline
    Config = _FakeConfig
    AlignFilter = _FakeAlignFilter
    Context = _FakeContext
    DeviceInfo = _FakeDeviceInfo
    Device = _FakeDevice
    OBSensorType = types.SimpleNamespace(
        COLOR_SENSOR=0, DEPTH_SENSOR=1, IR_SENSOR=2,
    )
    OBStreamType = types.SimpleNamespace(
        COLOR_STREAM=0, DEPTH_STREAM=1, IR_STREAM=2,
    )
    OBFormat = types.SimpleNamespace(
        RGB=0, BGR=1, Y16=2, YUYV=3, MJPG=4, Y14=5, RLE=6,
    )
    OBAlignMode = types.SimpleNamespace(
        DISABLE=0, HW_MODE=1, SW_MODE=2,
    )


@pytest.fixture
def fake_orbbec_sdk(monkeypatch):
    """Replace ``pyorbbecsdk`` with ``FakeOrbbecSDK`` for the duration of
    the test. Returns the fake module so tests can introspect state.
    """
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", FakeOrbbecSDK())
    return sys.modules["pyorbbecsdk"]


# ─────────────────────────────────────────────────────────────────────────
# WrcGripper + RebotArmEndPose mocks (post-ADR-0005).
# Tests that exercise SkillRuntime with the rebot_grasp bridge use this
# fixture to replace the SDK objects without needing real hardware.
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def wrc_arm_mocks():
    """Replace RebotArmEndPose + WrcGripper with MagicMocks.

    Returns ``(fake_controller, fake_gripper)`` so tests can introspect
    calls. The bridge module imports the SDK under ``if TYPE_CHECKING``
    only, so the live modules don't need to be mocked at import time.

    Pre-set return values for the typical happy-path call sequence.
    """
    fake_controller = MagicMock()
    fake_controller.move_to_traj.return_value = True

    fake_gripper = MagicMock()
    fake_gripper.grasp.return_value = True
    fake_gripper.open_gripper.return_value = None
    fake_gripper.release_gripper.return_value = None
    fake_gripper.MAX_DISTANCE_M = 0.09
    fake_gripper.set_position_target.return_value = None

    return fake_controller, fake_gripper
