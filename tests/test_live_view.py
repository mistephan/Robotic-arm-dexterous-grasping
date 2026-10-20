"""FrameHub tests (headless: show=False -> pure frame pump, no GUI)."""

import numpy as np

from wrc_demo.apps.live_view import FrameHub, depth_colormap
from wrc_demo.perception.mock_camera import MockCamera
from wrc_demo.types import Detection


def test_hub_pumps_and_serves_fresh_frames():
    hub = FrameHub(MockCamera(), show=False, rate_hz=60.0)
    hub.open()
    try:
        hub.warm_up(3)
        f1 = hub.get_frame()
        f2 = hub.get_frame()
        assert f1.rgb.shape == (480, 640, 3)
        assert f2.t >= f1.t  # second call waited for a NEWER frame
        assert hub.has_depth
    finally:
        hub.close()


def test_hub_overlay_is_thread_safe_noop_headless():
    hub = FrameHub(MockCamera(), show=False, rate_hz=60.0)
    hub.open()
    try:
        det = Detection("cube", 0.9, np.array([1, 2, 3, 4], dtype=np.float32))
        hub.set_overlay(detections=[det], status="grasping cube")
        hub.get_frame()  # still serves frames with overlays set
    finally:
        hub.close()


def test_hub_close_is_clean_and_idempotent():
    hub = FrameHub(MockCamera(), show=False)
    hub.open()
    hub.close()
    hub.close()  # second close must not raise


def test_runtime_status_reaches_hub(demo_cfg, tmp_path):
    """build_runtime(view=...) wires the hub; skill calls push status."""
    import pytest

    pytest.importorskip("pinocchio")
    from wrc_demo.apps.demo import build_runtime
    from wrc_demo.apps import live_view

    # Patch FrameHub to headless so the test never needs a display.
    orig_init = live_view.FrameHub.__init__

    def headless_init(self, camera, **kw):
        kw["show"] = False
        orig_init(self, camera, **kw)

    live_view.FrameHub.__init__ = headless_init
    try:
        runtime, arm = build_runtime(demo_cfg, tmp_path / "run", view=True)
        try:
            assert hasattr(runtime.camera, "set_overlay")
            runtime.execute("get_observation", {})
            assert "get_observation" in runtime.camera._overlay_status
            assert len(runtime.camera._overlay_dets) >= 1  # red cube drawn
        finally:
            runtime.camera.close()
            arm.disconnect()
    finally:
        live_view.FrameHub.__init__ = orig_init


def test_depth_colormap_invalid_black():
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, 0] = 1.0
    img = depth_colormap(depth, max_m=2.0)
    assert img.shape == (4, 4, 3)
    assert (img[1, 1] == 0).all()  # invalid pixel black
    assert img[0, 0].sum() > 0
