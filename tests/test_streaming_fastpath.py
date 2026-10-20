"""Livestreaming + fast-path tests: colors, reflex grammar, experience
memory, camera rig, world watcher, MJPEG server, and the end-to-end
"pick and place pink object" reflex on the mock stack."""

import json
import time
import urllib.request

import numpy as np
import pytest

from conftest import needs_pin

from wrc_demo.agent.reflex import ExperienceMemory, FastPlanner, parse_command
from wrc_demo.config import Cfg, load_demo_config
from wrc_demo.memory.beliefs import BeliefStore
from wrc_demo.perception.colors import classify_hsv, mask_color, parse_color_query
from wrc_demo.perception.mock_camera import MockCamera, synthetic_tabletop
from wrc_demo.perception.object_names import canonical_destination
from wrc_demo.perception.stream import CameraRig, CameraStream


# ── colors ────────────────────────────────────────────────────────────────


def test_classify_hsv_bands():
    assert classify_hsv(3, 220, 200) == "red"
    assert classify_hsv(160, 200, 220) == "pink"    # magenta band
    assert classify_hsv(5, 100, 220) == "pink"      # light low-sat red
    assert classify_hsv(60, 200, 180) == "green"
    assert classify_hsv(115, 200, 180) == "blue"
    assert classify_hsv(0, 20, 230) == "white"
    assert classify_hsv(0, 20, 100) == "gray"
    assert classify_hsv(0, 200, 30) == "black"


def test_mask_color_on_synthetic_scene():
    frame = synthetic_tabletop()
    red = frame.rgb[:, :, 2].astype(int) - frame.rgb[:, :, 0].astype(int) > 60
    assert mask_color(frame.rgb, red) == "red"
    pink = np.zeros(frame.rgb.shape[:2], dtype=bool)
    img = frame.rgb.copy()
    img[10:40, 10:40] = (203, 105, 255)  # BGR hot pink
    pink[10:40, 10:40] = True
    assert mask_color(img, pink) == "pink"
    assert mask_color(frame.rgb, None) is None


def test_parse_color_query():
    assert parse_color_query("pink object") == ("pink", None)
    assert parse_color_query("the red mug") == ("red", "mug")
    assert parse_color_query("bottle") == (None, "bottle")
    assert parse_color_query("objeto rosa") == ("pink", None)
    assert parse_color_query("thing") == (None, None)


# ── beliefs with color ────────────────────────────────────────────────────


def test_beliefs_resolve_color_queries():
    store = BeliefStore()
    store.update("cup", np.array([0.3, 0.1, 0.02]), 0.9, color="pink")
    store.update("bottle", np.array([0.3, -0.1, 0.05]), 0.9, color="green")
    hit = store.find("pink object")
    assert hit is not None and hit.label == "cup"
    assert store.find("green bottle").label == "bottle"
    assert store.find("blue object") is None
    # untagged beliefs stay eligible when nothing matches the color
    store.update("box", np.array([0.2, 0.0, 0.03]), 0.8)
    assert store.find("blue object").label == "box"
    assert store.summary()[0]["color"] in {"pink", "green", None}


# ── reflex grammar ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,tool,obj,dest",
    [
        ("pick and place pink object", "pick_and_place", "pink object", None),
        ("Pick up the red cube and put it in the bowl", "pick_and_place", "red cube", "bowl"),
        ("put the pink cube in the bowl", "pick_and_place", "pink cube", "bowl"),
        ("please can you grab the bottle", "grasp_object", "bottle", None),
        ("pick up the banana", "grasp_object", "banana", None),
        ("pick up the mango", "grasp_object", "mango", None),
        ("coge y coloca el objeto rosa", "pick_and_place", "objeto rosa", None),
        ("place it on the plate", "place_on_object", None, "plate"),
        ("go home", "move_home", None, None),
        ("look around", "get_observation", None, None),
        ("stack the red cube on the blue box", "pick_and_place", "red cube", "blue box"),
        ("release", "open_gripper", None, None),
        ("现在 pick up and place the banana", "pick_and_place", "banana", None),
        ("拿取香蕉并放置", "pick_and_place", "banana", None),
        ("拿取香蕉,并放进藤篮里。", "pick_and_place", "banana", "wicker basket"),
        ("拿起青柠并放进藤篮里", "pick_and_place", "lime", "wicker basket"),
        ("把芒果放到篮子里", "pick_and_place", "mango", "wicker basket"),
        ("把香蕉放到青柠左边", "pick_and_place", "banana", "left of lime"),
        ("拿起芒果并放到青柠右侧", "pick_and_place", "mango", "right of lime"),
        ("捡起苹果，并把它放在香蕉的右侧", "pick_and_place", "red apple", "right of banana"),
        ("捡起苹果然后放到藤篮里", "pick_and_place", "red apple", "wicker basket"),
        ("张开灵巧手", "hand_action", None, None),
        ("五指收缩", "hand_action", None, None),
        ("食指捏合", "hand_action", None, None),
    ],
)
def test_reflex_parses_routine_commands(text, tool, obj, dest):
    plan = parse_command(text)
    assert plan is not None, text
    name, args = plan.calls[0]
    assert name == tool
    if obj is not None:
        assert args.get("object", args.get("label")) == obj
    if dest is not None:
        assert args.get("destination", args.get("label")) == dest


def test_chinese_container_aliases_and_collective_command():
    assert canonical_destination("筐里") == "wicker basket"
    assert canonical_destination("水果框") == "wicker basket"
    assert parse_command("将桌面上所有的水果放到筐里") is None


def test_reflex_rejects_novel_tasks():
    assert parse_command("sort the objects by size, largest first") is None
    assert parse_command("stop") is None  # e-stop is NOT a reflex plan
    assert parse_command("") is None


def test_reflex_rejects_chinese_compound_commands():
    """A trailing "，然后/接着/之后…" clause is a SECOND command, not part of
    the destination — the reflex must decline so the LLM tier plans the
    multi-step task. (2026-09-22: the tail used to leak into the
    destination and waste a localize cycle before falling back anyway.
    The English \\b compound guard cannot match between Han codepoints.)"""
    for text in (
        "拿起苹果然后放到篮子里，然后拿起香蕉",
        "把香蕉放进藤篮里，接着拿起芒果",
        "捡起苹果放到青柠左边之后回到原位",
    ):
        assert parse_command(text) is None, text


# ── experience memory ─────────────────────────────────────────────────────


def test_experience_memory_recall_and_persistence(tmp_path):
    path = tmp_path / "exp.json"
    em = ExperienceMemory(path)
    calls = [("pick_and_place", {"object": "toy dinosaur"})]
    em.record("hand me the toy dinosaur", calls, True, 8.0)
    hit = em.recall("hand me the toy dinosaur")
    assert hit is not None and hit["calls"][0][0] == "pick_and_place"
    # near-identical phrasing still hits; unrelated does not
    assert em.recall("hand me that toy dinosaur") is not None
    assert em.recall("calibrate the camera") is None
    # reload from disk
    em2 = ExperienceMemory(path)
    assert len(em2) == 1 and em2.recall("hand me the toy dinosaur") is not None
    # failures only demote, never store fresh
    em2.record("throw the cup", [("grasp_object", {"label": "cup"})], False, 3.0)
    assert em2.recall("throw the cup") is None


def test_fast_planner_prefers_reflex_over_experience(tmp_path):
    em = ExperienceMemory(tmp_path / "exp.json")
    em.record("pick and place pink object", [("move_home", {})], True, 1.0)
    plan = FastPlanner(em).plan("pick and place pink object")
    assert plan.source == "reflex"  # grammar wins; memory is the fallback
    assert plan.calls[0][0] == "pick_and_place"


# ── camera rig + watcher ──────────────────────────────────────────────────


def _mock_stream(name="cam0"):
    return CameraStream(MockCamera(), name=name, rate_hz=60.0)


def test_camera_rig_streams_n_cameras():
    rig = CameraRig([_mock_stream("over"), _mock_stream("side")])
    rig.open()
    try:
        rig.warm_up(3)
        f1 = rig.get("over").get_frame()
        f2 = rig.get("side").get_frame()
        assert f1.rgb.shape == (480, 640, 3) and f2.has_depth
        assert rig.primary.name == "over"
        stats = rig.stats()
        assert set(stats) == {"over", "side"} and stats["over"]["fps"] > 0
        with pytest.raises(KeyError):
            rig.get("nope")
    finally:
        rig.close()


def test_world_watcher_populates_beliefs_with_color(demo_cfg):
    from wrc_demo.perception.detector import MockDetector
    from wrc_demo.perception.depth_provider import DepthProvider
    from wrc_demo.perception.grounding import Extrinsics
    from wrc_demo.perception.world import LockedDetector, WatchedCamera, WorldWatcher

    stream = _mock_stream()
    stream.open()
    beliefs = BeliefStore()
    watcher = WorldWatcher(
        [WatchedCamera(
            stream=stream,
            depth=DepthProvider(demo_cfg.camera),
            extrinsics=Extrinsics.from_config(demo_cfg.camera.extrinsics),
        )],
        LockedDetector(MockDetector(label="red cube")),
        beliefs,
        classes=["cube"],
        rate_hz=30.0,
    )
    watcher.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and beliefs.find("red cube") is None:
            time.sleep(0.05)
        b = beliefs.find("red cube")
        assert b is not None, "watcher never fused the mock detection"
        assert b.color == "red"
        assert beliefs.find("red object").label == "red cube"
        with watcher.paused():
            assert watcher.is_paused
        assert not watcher.is_paused
    finally:
        watcher.stop()
        stream.close()


# ── MJPEG dashboard ───────────────────────────────────────────────────────


def test_stream_server_serves_state_snapshot_and_index():
    from wrc_demo.apps.stream_server import StreamServer

    rig = CameraRig([_mock_stream("over")])
    rig.open()
    server = StreamServer(rig, state_fn=lambda: {"agent_status": "testing"}, port=0)
    server.start()
    try:
        rig.warm_up(2)
        base = f"http://127.0.0.1:{server.port}"
        state = json.loads(urllib.request.urlopen(f"{base}/state", timeout=5).read())
        assert state["agent_status"] == "testing" and "over" in state["cameras"]
        jpeg = urllib.request.urlopen(f"{base}/snapshot/over.jpg", timeout=5).read()
        assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
        html = urllib.request.urlopen(f"{base}/", timeout=5).read().decode()
        assert "/stream/over" in html and "robot narration" in html
    finally:
        server.stop()
        rig.close()


# ── end-to-end reflex on the mock stack ───────────────────────────────────


@needs_pin
def test_pick_and_place_red_object_end_to_end(tmp_path):
    """The demo's money path: a color query resolved against the live world
    model, grasped, placed at the drop zone -- no LLM anywhere."""
    from wrc_demo.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    # Mock arm's home_q is in a different basin than rebot_rs's; the
    # production `approach_pitch_rad: 1.2` (tilted) pregrasp is unreachable
    # from the mock seed. Pin legacy top-down for the e2e tests; the
    # pitch=1.2 path is covered separately by the real-hardware demo.
    # Isolate grasp memory under tmp_path so a stale ~/.wrc_demo/...json
    # (with a learned z_nudge) doesn't push the pregrasp into a reach
    # dead-zone (per CLAUDE.md "Persistent learned state").
    import math
    cfg._data["grasp"]["approach_pitch_rad"] = math.pi / 2
    cfg._data["grasp"]["memory_path"] = str(tmp_path / "grasp_memory.json")
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    arm.object_stop_frac = 0.5  # jaws jam halfway: an object is in the gripper
    try:
        # the watcher needs one detector pass; don't race it
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red object") is None:
            time.sleep(0.05)
        t0 = time.monotonic()
        result = runtime.execute("pick_and_place", {"object": "red object"})
        assert result.get("ok"), result
        assert result["picked"] == "red object"
        assert result["duration_s"] < 30
        assert runtime.held_object is None
        # the fix must have come from the color-aware resolution
        assert runtime.beliefs.find("red object") is not None  # re-registered at drop zone
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_orchestrator_reflex_path_never_calls_llm(tmp_path):
    from wrc_demo.agent.orchestrator import AgentOrchestrator
    from wrc_demo.apps.demo import build_runtime, shutdown_runtime

    class ExplodingLLM:
        supports_vision = False

        def chat(self, *a, **kw):
            raise AssertionError("LLM must not be called on the reflex path")

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    import math
    cfg._data["grasp"]["approach_pitch_rad"] = math.pi / 2
    cfg._data["grasp"]["memory_path"] = str(tmp_path / "grasp_memory.json")
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    arm.object_stop_frac = 0.5
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        agent = AgentOrchestrator(
            ExplodingLLM(), runtime, decompose=False,
            fast_planner=FastPlanner(ExperienceMemory(tmp_path / "exp.json")),
        )
        report = agent.run_task("pick and place the red object")
        assert report.success, report.summary
        assert report.path == "reflex"
        assert report.duration_s > 0
    finally:
        shutdown_runtime(runtime, arm)
