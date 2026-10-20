"""Regressions from the 2026-07-18 adversarial review of the livestreaming +
fast-path work (45-agent multi-lens workflow; 33 confirmed findings)."""

import threading
import time

import numpy as np
import pytest

from conftest import needs_pin

from wrc_demo.config import Cfg, load_demo_config
from wrc_demo.memory.beliefs import BeliefStore
from wrc_demo.memory.episodic import EpisodicMemory
from wrc_demo.perception.colors import center_bbox_mask, classify_hsv, color_matches
from wrc_demo.perception.detector import MockDetector
from wrc_demo.perception.grounding import Extrinsics, localize_object
from wrc_demo.perception.mock_camera import MockCamera, synthetic_tabletop
from wrc_demo.perception.stream import CameraStream
from wrc_demo.types import Detection


# ── watchdog heartbeat cluster (critical) ─────────────────────────────────


class _FakeHarness:
    def __init__(self):
        self.beats = 0
        self.last = None

    def heartbeat(self):
        self.beats += 1
        self.last = time.monotonic()


def _watcher(harness, label="nothing matches", classes=("cube",)):
    from wrc_demo.perception.depth_provider import DepthProvider
    from wrc_demo.perception.world import LockedDetector, WatchedCamera, WorldWatcher

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    stream = CameraStream(MockCamera(), name="cam", rate_hz=60.0)
    stream.open()
    watcher = WorldWatcher(
        [WatchedCamera(
            stream=stream,
            depth=DepthProvider(cfg.camera),
            extrinsics=Extrinsics.from_config(cfg.camera.extrinsics),
        )],
        LockedDetector(MockDetector(label=label)),
        BeliefStore(),
        classes=list(classes),
        rate_hz=30.0,
        harness=harness,
    )
    watcher.start()
    return watcher, stream


def _wait_beats(harness, n, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and harness.beats < n:
        time.sleep(0.02)
    return harness.beats >= n


def test_watcher_heartbeats_with_empty_scene():
    """An empty/undetectable table must NOT starve the watchdog: wave and
    move_home would be blocked 5 s after startup otherwise."""
    harness = _FakeHarness()
    watcher, stream = _watcher(harness, label="unicorn", classes=("cube",))
    try:
        assert _wait_beats(harness, 3), "no heartbeats despite frames flowing"
    finally:
        watcher.stop()
        stream.close()


def test_watcher_keeps_heartbeating_while_paused():
    """pick_and_place outlasts watchdog_s on the real (paced) arm; the pause
    must only stop belief fusion, never the heartbeat."""
    harness = _FakeHarness()
    watcher, stream = _watcher(harness, label="red cube", classes=("cube",))
    try:
        assert _wait_beats(harness, 2)
        with watcher.paused():
            before = harness.beats
            assert _wait_beats(harness, before + 3), "heartbeat stopped during pause"
    finally:
        watcher.stop()
        stream.close()


def test_paused_watcher_does_not_fuse_beliefs():
    from wrc_demo.perception.depth_provider import DepthProvider
    from wrc_demo.perception.world import LockedDetector, WatchedCamera, WorldWatcher

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    stream = CameraStream(MockCamera(), name="cam", rate_hz=60.0)
    stream.open()
    beliefs = BeliefStore()
    watcher = WorldWatcher(
        [WatchedCamera(stream=stream, depth=DepthProvider(cfg.camera),
                       extrinsics=Extrinsics.from_config(cfg.camera.extrinsics))],
        LockedDetector(MockDetector(label="red cube")),
        beliefs, classes=["cube"], rate_hz=30.0,
    )
    with watcher.paused():
        watcher.start()
        time.sleep(0.4)
        assert beliefs.find("red cube") is None, "fused beliefs while paused"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and beliefs.find("red cube") is None:
        time.sleep(0.05)
    assert beliefs.find("red cube") is not None  # resumes after release
    watcher.stop()
    stream.close()


# ── execute() robustness (critical) ───────────────────────────────────────


@needs_pin
def test_execute_survives_ok_false_without_error_key(tmp_path):
    from wrc_demo.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    try:
        runtime.skill_bare_fail = lambda: {"ok": False}  # no 'error' key
        result = runtime.execute("bare_fail", {})
        assert result == {"ok": False}  # must not raise KeyError
        # the real reproducer: sort with nothing graspable
        result = runtime.execute("sort_by_color", {})
        assert "error" in result or result.get("ok"), result
    finally:
        shutdown_runtime(runtime, arm)


# ── color tolerance (major) ───────────────────────────────────────────────


def test_color_boundary_tolerance():
    # saturated h=10..12 now classifies orange but must still MATCH red
    assert classify_hsv(11, 250, 200) == "orange"
    assert color_matches("red", "orange")
    assert color_matches("red", "pink")
    assert not color_matches("red", "green")
    assert color_matches("blue", None)  # unknown is never a hard veto
    # widened red band
    assert classify_hsv(9, 250, 200) == "red"
    assert classify_hsv(171, 250, 200) == "red"


def test_center_bbox_mask_ignores_box_edges():
    m = center_bbox_mask((100, 100), [10, 10, 90, 90], frac=0.5)
    assert m[50, 50] and not m[12, 12]  # center in, corner out


def test_beliefs_neighbor_color_tier():
    store = BeliefStore()
    store.update("cup", np.array([0.3, 0.1, 0.02]), 0.9, color="orange")
    # no exact red belief -> the orange (neighbor) one is accepted
    assert store.find("red object").label == "cup"
    store.update("box", np.array([0.2, -0.1, 0.02]), 0.9, color="red")
    # exact tier outranks neighbor tier
    assert store.find("red object").label == "box"


# ── localize branches (previously uncovered) ──────────────────────────────


def _two_object_frame():
    """Synthetic frame: red box (left) + pink box (right), both with depth."""
    frame = synthetic_tabletop()  # red box at (280..360, 200..260)
    frame.rgb[200:260, 420:500] = (203, 105, 255)  # hot pink, BGR
    frame.depth_m[200:260, 420:500] = 0.55
    red_mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
    red_mask[200:260, 280:360] = True
    pink_mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
    pink_mask[200:260, 420:500] = True
    dets = [
        Detection("box", 0.9, np.array([280, 200, 360, 260], dtype=np.float32), mask=red_mask),
        Detection("box", 0.8, np.array([420, 200, 500, 260], dtype=np.float32), mask=pink_mask),
    ]
    return frame, dets


def test_localize_vocab_color_filter_picks_pink():
    frame, dets = _two_object_frame()
    detector = MockDetector(detections=dets)
    extr = Extrinsics(mode="eye_to_hand", T=np.eye(4))
    fix = localize_object(
        frame, "pink object", detector, extr, vocab=["box"], color="pink"
    )
    # the pink box is on the right (larger x in pixels -> different cam x)
    assert fix.detection.mask[230, 460]
    # and the color filter rejects red when asked for pink
    fix_red = localize_object(frame, "red object", detector, extr, vocab=["box"], color="red")
    assert fix_red.detection.mask[230, 300]


def test_localize_near_xyz_disambiguates():
    frame, dets = _two_object_frame()
    detector = MockDetector(detections=dets)
    extr = Extrinsics(mode="eye_to_hand", T=np.eye(4))
    # identity extrinsics: base == camera frame; the two boxes differ in x
    left = localize_object(frame, "box", detector, extr, vocab=["box"],
                           near_xyz=dets[0] and localize_object(
                               frame, "box", detector, extr, prompts=["box"]).position)
    assert left is not None
    a = localize_object(frame, "box", detector, extr, vocab=["box"],
                        near_xyz=np.array([-0.1, 0.0, 0.55]))
    b = localize_object(frame, "box", detector, extr, vocab=["box"],
                        near_xyz=np.array([0.15, 0.0, 0.55]))
    assert abs(float(a.position[0]) - float(b.position[0])) > 0.05


# ── episodic memory thread-safety (major) ─────────────────────────────────


def test_episodic_memory_concurrent_add_and_digest():
    mem = EpisodicMemory(horizon_s=0.05)  # aggressive pruning
    stop = threading.Event()
    errors = []

    def writer():
        while not stop.is_set():
            mem.add("note", "x" * 20)

    def reader():
        while not stop.is_set():
            try:
                mem.digest(max_lines=10)
                mem.events()
            except RuntimeError as e:  # deque mutated during iteration
                errors.append(e)
                return

    threads = [threading.Thread(target=writer) for _ in range(2)] + [
        threading.Thread(target=reader) for _ in range(2)
    ]
    for t in threads:
        t.start()
    time.sleep(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    assert not errors, f"unsynchronized access crashed: {errors[0]}"


# ── reflex compound rejection (major) ─────────────────────────────────────


def test_reflex_rejects_compound_commands():
    from wrc_demo.agent.reflex import parse_command

    assert parse_command("wave and then pick up the knife") is None
    assert parse_command("grab the cup then drop it in the bin") is None
    assert parse_command("wave") is None
    assert parse_command("wave at the crowd") is None
    # the explicit two-clause pick..place form is still one reflex
    plan = parse_command("pick up the cube and put it in the bowl")
    assert plan is not None and plan.calls[0][0] == "pick_and_place"
