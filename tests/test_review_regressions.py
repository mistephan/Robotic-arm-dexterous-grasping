"""Regression tests for defects found in the adversarial review pass."""

from types import SimpleNamespace
import time

import numpy as np
import pytest

from wrc_demo.agent.llm import LLMResponse, MockLLM, ToolCall
from wrc_demo.agent.orchestrator import AgentOrchestrator, _as_bool
from wrc_demo.agent.trace import TraceLogger
from wrc_demo.memory.beliefs import BeliefStore
from wrc_demo.perception.detector import MockDetector
from wrc_demo.perception.grounding import Extrinsics, localize_object
from wrc_demo.perception.mock_camera import synthetic_tabletop
from wrc_demo.safety.harness import SafetyHarness, SafetyLimits
from wrc_demo.skills.runtime import SkillRuntime
from wrc_demo.types import Detection, SafetyViolation


T_CAM2BASE = np.array(
    [
        [0.0, -1.0, 0.0, 0.28],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.00],
    ]
)


def _mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_spatial_hint_left_picks_left_object():
    """'left' must select the object with the LARGER base-frame y."""
    f = synthetic_tabletop()
    h, w = f.rgb.shape[:2]
    # Two cups: one left in the image, one right. With this camera transform
    # base y = -x_cam, so the image-left cup (smaller u) has larger base y.
    d_img_left = Detection("cup", 0.9, np.array([100, 200, 160, 260], np.float32),
                           mask=_mask(h, w, 100, 200, 160, 260))
    d_img_right = Detection("cup", 0.9, np.array([480, 200, 540, 260], np.float32),
                            mask=_mask(h, w, 480, 200, 540, 260))
    det = MockDetector(detections=[d_img_left, d_img_right])
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)

    fix_left = localize_object(f, "cup", det, extr, spatial_hint="left")
    fix_right = localize_object(f, "cup", det, extr, spatial_hint="right")
    assert fix_left.position[1] > fix_right.position[1]

    # near/far discriminate along base x = -y_cam, i.e. image rows.
    d_img_top = Detection("cup", 0.9, np.array([300, 60, 360, 120], np.float32),
                          mask=_mask(h, w, 300, 60, 360, 120))
    d_img_bottom = Detection("cup", 0.9, np.array([300, 360, 360, 420], np.float32),
                             mask=_mask(h, w, 300, 360, 360, 420))
    det2 = MockDetector(detections=[d_img_top, d_img_bottom])
    fix_near = localize_object(f, "cup", det2, extr, spatial_hint="near")
    fix_far = localize_object(f, "cup", det2, extr, spatial_hint="far")
    assert fix_near.position[0] < fix_far.position[0]


def test_hand_eye_npz_baseline_format(tmp_path):
    """The baseline saves mode as a 1-element string array; must load."""
    from wrc_demo.config import Cfg

    npz = tmp_path / "hand_eye.npz"
    T = np.eye(4)
    T[:3, 3] = [0.01, 0.02, 0.03]
    np.savez(npz, T_result=T, mode=np.array(["eye_in_hand"]), n_samples=15)
    extr = Extrinsics.from_config(
        Cfg({"hand_eye_npz": str(npz)}), fk_tcp2base=lambda: np.eye(4)
    )
    assert extr.mode == "eye_in_hand"
    assert np.allclose(extr.cam_to_base(), T)


class FakeKin:
    n_joints = 6
    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([[0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])

    def ik(self, T, q_init, retries=0):
        from types import SimpleNamespace
        return SimpleNamespace(
            q=np.asarray(q_init, dtype=float), success=True, error=0.0, iterations=1,
        )


def _limits(**kw):
    base = dict(
        workspace_min=np.array([-1.0, -1.0, -0.5]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
        table_z=0.0,
        table_clearance=0.02,
        max_joint_vel=10.0,
        watchdog_s=0.0,  # instantly stale
    )
    base.update(kw)
    return SafetyLimits(**base)


def test_watchdog_checked_at_motion_start_not_per_waypoint():
    h = SafetyHarness(_limits(), kinematics=FakeKin())
    # Stale perception: starting a motion must fail...
    with pytest.raises(SafetyViolation, match="watchdog"):
        h.begin_motion()
    # ...but once begun (fresh heartbeat), per-waypoint approve()s do not
    # re-check freshness even as time passes.
    h.heartbeat()
    h2 = SafetyHarness(_limits(watchdog_s=1e-3), kinematics=FakeKin())
    h2.heartbeat()
    h2.begin_motion()
    import time

    time.sleep(0.01)  # now stale relative to 1 ms watchdog
    q = np.array([0.3, 0.0, 0.2, 0, 0, 0])
    assert h2.approve(q, q, 0.02)  # mid-motion: skip watchdog, OK
    h2.end_motion()
    # Outside a motion, approve() re-checks the watchdog and vetoes
    # with a reason string (not a raise).
    assert not h2.approve(q, q, 0.02)
    assert "watchdog" in h2.violations[-1]


def test_escape_upward_from_below_clearance():
    h = SafetyHarness(_limits(watchdog_s=1e9), kinematics=FakeKin())
    below = np.array([0.3, 0.0, 0.005, 0, 0, 0])
    slightly_up = np.array([0.3, 0.0, 0.012, 0, 0, 0])
    deeper = np.array([0.3, 0.0, 0.001, 0, 0, 0])
    # No exemption: moving UP from below the floor is allowed...
    assert h.approve(below, slightly_up, dt=1e9)
    # ...but descending further is not.
    assert not h.approve(below, deeper, dt=1e9)
    assert "table" in h.violations[-1]


def test_belief_top_z_used_for_placement():
    store = BeliefStore()
    b = store.update("bowl", [0.3, 0.0, 0.04], conf=0.9, extent=np.array([0.2, 0.06, 0.05]),
                     top_z=0.08)
    assert b.top_z == 0.08
    # A later observation without top_z keeps the previous one.
    store.update("bowl", [0.3, 0.0, 0.04], conf=0.9)
    assert store.find("bowl").top_z == 0.08


def test_as_bool_string_false():
    assert _as_bool(True) is True
    assert _as_bool("true") is True
    assert _as_bool("false") is False
    assert _as_bool("no") is False
    assert _as_bool(0) is False


class _StubRuntime:
    def __init__(self):
        from wrc_demo.memory import EpisodicMemory

        self.memory = EpisodicMemory()
        self.last_frame = None
        self.executed = []
        self.finished_summary = None

        class _T:
            def finish(_trace, s):
                self.finished_summary = s

            def record_event(self, name, payload):
                pass

        self.trace = _T()

    def execute(self, name, args):
        self.executed.append(name)
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True}

    def frame_jpeg(self):
        return None


class _FailingPickRuntime(_StubRuntime):
    def execute(self, name, args):
        self.executed.append(name)
        if name == "pick_and_place":
            return {"ok": False, "stage": "grasp", "error": "failed after 3 attempts"}
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True}


class _HeldPickRuntime(_StubRuntime):
    def execute(self, name, args):
        self.executed.append(name)
        if name == "grasp_object":
            return {"ok": True, "held": args.get("label", "object")}
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True}


def test_o6_hand_open_clears_held_object_state():
    rt = object.__new__(SkillRuntime)
    rt.held_object = "banana"
    rt._held_det_label = "banana"
    rt._held_color = "yellow"
    rt._held_support_offset_m = 0.02
    rt.memory = _StubRuntime().memory
    rt.arm = SimpleNamespace(
        end_effector=SimpleNamespace(
            open=lambda: {"ok": True, "action": "open"},
            close_hand=lambda: {"ok": True},
            power_grasp=lambda: {"ok": True},
            pinch=lambda: {"ok": True},
        )
    )

    result = rt.skill_hand_action("open")

    assert result["ok"]
    assert rt.held_object is None
    assert rt._held_det_label is None
    assert rt._held_color is None
    assert rt._held_support_offset_m is None


def test_o6_reconcile_checks_tactile_before_already_holding_rejection():
    rt = object.__new__(SkillRuntime)
    rt.held_object = "mango"
    rt._held_det_label = "mango"
    rt._held_color = "orange"
    rt._held_support_offset_m = 0.03
    rt._held_verified_t = time.monotonic() - 30.0
    rt._held_no_contact_checks = 2
    rt.memory = _StubRuntime().memory
    events = []
    rt.trace = SimpleNamespace(
        record_event=lambda name, payload: events.append((name, payload)),
        record_exception=lambda *args: None,
    )
    rt.arm = SimpleNamespace(
        end_effector=SimpleNamespace(
            holding_contact=lambda: {
                "contacted": [], "pressure": {}, "sample_count": 2,
            }
        )
    )

    rt._reconcile_held()

    assert rt.held_object is None
    assert events[0][0] == "held_object_tactile_check"


def test_o6_recent_verified_hold_ignores_transient_zero_tactile():
    rt = object.__new__(SkillRuntime)
    rt.held_object = "banana"
    rt._held_det_label = "banana"
    rt._held_color = "yellow"
    rt._held_support_offset_m = 0.03
    rt._held_R_level = np.eye(3)
    rt._held_long_tool = None
    rt._held_verified_t = time.monotonic()
    rt._held_no_contact_checks = 0
    rt.memory = _StubRuntime().memory
    events = []
    rt.trace = SimpleNamespace(
        record_event=lambda name, payload: events.append((name, payload)),
        record_exception=lambda *args: None,
    )
    rt.arm = SimpleNamespace(
        end_effector=SimpleNamespace(
            holding_contact=lambda: {
                "contacted": [], "pressure": {}, "sample_count": 2,
            }
        )
    )

    rt._reconcile_held()

    assert rt.held_object == "banana"
    assert rt._held_no_contact_checks == 0
    assert any(name == "held_contact_loss_ignored" for name, _ in events)

def test_trace_exception_goes_to_log_not_console(tmp_path, capsys):
    trace = TraceLogger(tmp_path)
    try:
        raise RuntimeError("private hardware detail")
    except RuntimeError as exc:
        trace.record_exception("grasp_object", exc)

    assert capsys.readouterr().err == ""
    log = (tmp_path / "runtime_errors.log").read_text()
    assert "grasp_object" in log
    assert "RuntimeError: private hardware detail" in log


def test_grasp_localization_writes_json_event_and_csv(tmp_path):
    trace = TraceLogger(tmp_path)
    trace.record_grasp_localization({
        "timestamp": 1.0, "grasp_call_index": 2, "pick_attempt": 3,
        "frame_id": 42, "query": "banana", "detected_as": "banana",
        "source": "fresh_detection", "confidence": 0.91,
        "base_x_m": 0.31, "base_y_m": -0.04, "base_z_m": 0.025,
        "cloud_top_z_m": 0.05, "cloud_bottom_z_m": 0.0,
        "n_points": 500, "extent_0_m": 0.18, "extent_1_m": 0.04,
        "extent_2_m": 0.03, "bbox_x0_px": 10, "bbox_y0_px": 20,
        "bbox_x1_px": 110, "bbox_y1_px": 80,
    })

    assert "grasp_object_localization" in (tmp_path / "trace.jsonl").read_text()
    csv_text = (tmp_path / "grasp_localizations.csv").read_text()
    assert "base_x_m" in csv_text
    assert "banana" in csv_text


def test_llm_failure_returns_report_instead_of_crashing_demo():
    class _BrokenLLM:
        supports_vision = False

        def chat(self, *args, **kwargs):
            raise RuntimeError("Input length (4318) exceeds maximum (4096)")

    rt = _StubRuntime()
    report = AgentOrchestrator(
        _BrokenLLM(), rt, decompose=False, max_steps=2
    ).run_task("pick up the mango")

    assert not report.success
    assert "LLM request failed" in report.summary
    assert "success: false" in rt.finished_summary


def test_failed_pick_and_place_is_not_reissued_by_llm():
    rt = _FailingPickRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("pick_and_place", {"object": "lime"})]),
        LLMResponse(tool_calls=[ToolCall("pick_and_place", {"object": "lime"})]),
    ])
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task(
        "pick and place lime"
    )
    assert not report.success
    assert rt.executed == ["pick_and_place", "task_done"]
    assert rt.finished_summary is not None
    assert "task: pick and place lime" in rt.finished_summary
    assert "success: false" in rt.finished_summary


def test_plain_pickup_cannot_be_upgraded_to_pick_and_place_by_llm():
    rt = _StubRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("pick_and_place", {"object": "mango"})]),
        LLMResponse(tool_calls=[ToolCall(
            "task_done", {"success": True, "summary": "Mango held"}
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=4).run_task(
        "pick up the mango"
    )

    assert report.success
    assert rt.executed == ["grasp_object", "task_done"]


def test_successful_recovery_grasp_auto_completes_hold_only_task():
    rt = _HeldPickRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("grasp_object", {"label": "banana"})]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=8).run_task(
        "pick up the banana"
    )

    assert report.success
    assert rt.executed == ["grasp_object", "task_done"]
    assert report.steps == 1
    assert report.tool_log[-1]["forced"] is True


def test_parallel_tool_calls_all_get_results():
    """Extra tool calls are answered (protocol) but not executed."""
    rt = _StubRuntime()
    llm = MockLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("get_observation", {}, id="a"),
                    ToolCall("move_home", {}, id="b"),
                ]
            ),
            LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "x"}, id="c")]),
        ]
    )
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task("t")
    assert report.success
    assert rt.executed == ["get_observation", "task_done"]  # move_home NOT executed
    # Second request must contain tool results for BOTH call ids a and b.
    msgs = llm.requests[1]["messages"]
    tool_ids = [m.get("tool_call_id") for m in msgs if m.get("role") == "tool"]
    assert "a" in tool_ids and "b" in tool_ids


def test_task_done_string_false_reports_failure():
    rt = _StubRuntime()
    llm = MockLLM(
        [LLMResponse(tool_calls=[ToolCall("task_done", {"success": "false", "summary": "s"})])]
    )
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=3).run_task("t")
    assert report.success is False


def test_image_pruning_keeps_only_latest():
    msgs = [
        {"role": "user", "content": "a", "images": [b"old"]},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c", "images": [b"new"]},
    ]
    pruned = AgentOrchestrator._prune_images(msgs)
    assert "images" not in pruned[0]
    assert pruned[2]["images"] == [b"new"]
    # Original list untouched.
    assert msgs[0]["images"] == [b"old"]


def test_o6_closed_hand_zero_tactile_is_contact_migration_not_release():
    rt = object.__new__(SkillRuntime)
    rt.held_object = "banana"
    rt._held_det_label = "banana"
    rt._held_color = "yellow"
    rt._held_support_offset_m = 0.03
    rt._held_verified_t = time.monotonic() - 30.0
    rt._held_no_contact_checks = 2
    rt.memory = _StubRuntime().memory
    rt.cfg = SimpleNamespace(grasp={
        "held_contact_grace_s": 0.0,
        "held_contact_loss_checks": 3,
    })
    events = []
    rt.trace = SimpleNamespace(
        record_event=lambda name, payload: events.append((name, payload)),
        record_exception=lambda *args: None,
    )
    rt.arm = SimpleNamespace(
        end_effector=SimpleNamespace(
            holding_contact=lambda: {
                "contacted": [], "pressure": {}, "sample_count": 2,
                "commanded_open": False,
            }
        )
    )

    rt._reconcile_held()

    assert rt.held_object == "banana"
    assert rt._held_no_contact_checks == 0
    assert any(name == "held_contact_migrated" for name, _ in events)


def test_batch_task_allows_different_objects_to_same_destination():
    rt = _StubRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place", {"object": "banana", "destination": "left basket"}
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place", {"object": "mango", "destination": "left basket"}
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "task_done", {"success": True, "summary": "All fruit moved"}
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task(
        "put all fruit on the table into the left basket"
    )

    assert report.success
    assert rt.executed == ["pick_and_place", "pick_and_place", "task_done"]


def test_exact_duplicate_pick_and_place_is_suppressed():
    rt = _StubRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place", {"object": "banana", "destination": "left basket"}
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place", {"object": " banana ", "destination": "LEFT  basket"}
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=4).run_task(
        "put the banana into the left basket"
    )

    assert report.success
    assert rt.executed == ["pick_and_place", "task_done"]


def test_batch_pending_list_overrides_premature_task_done():
    class BatchRuntime(_StubRuntime):
        def execute(self, name, args):
            self.executed.append(name)
            if name == "list_objects":
                return {"ok": True, "objects": [
                    {"label": "banana", "state": "visible"},
                    {"label": "mango", "state": "visible"},
                    {"label": "wicker basket", "state": "visible"},
                ]}
            if name == "task_done":
                return {"ok": True, "task_complete": True}
            return {"ok": True}

    rt = BatchRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("list_objects", {})]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place", {"object": "banana", "destination": "left basket"}
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "task_done", {"success": True, "summary": "done"}
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "task_done", {"success": True, "summary": "all done"}
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=6).run_task(
        "put all fruit on the table into the left basket"
    )

    assert report.success
    assert rt.executed == [
        "list_objects", "pick_and_place", "pick_and_place", "task_done"
    ]


def test_chinese_basket_task_overrides_llm_box_destination():
    class BasketRuntime(_StubRuntime):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, name, args):
            self.executed.append(name)
            self.calls.append((name, dict(args)))
            if name == "list_objects":
                return {"ok": True, "objects": [
                    {"label": "banana", "state": "visible"},
                    {"label": "wicker basket", "state": "visible"},
                ]}
            if name == "task_done":
                return {"ok": True, "task_complete": True}
            return {"ok": True}

    rt = BasketRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("list_objects", {})]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place", {"object": "banana", "destination": "box"}
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "task_done", {"success": True, "summary": "done"}
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task(
        "将桌面上所有的水果放到筐里"
    )

    assert report.success
    pick_call = next(args for name, args in rt.calls if name == "pick_and_place")
    assert pick_call["destination"] == "wicker basket"


def test_batch_rejects_remembered_alias_after_initial_list_is_complete():
    class SnapshotRuntime(_StubRuntime):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, name, args):
            self.executed.append(name)
            self.calls.append((name, dict(args)))
            if name == "list_objects":
                return {"ok": True, "objects": [
                    {"label": "banana", "state": "visible"},
                    {"label": "mango", "state": "remembered"},
                    {"label": "wicker basket", "state": "visible"},
                ]}
            if name == "task_done":
                return {"ok": True, "task_complete": True}
            return {"ok": True}

    rt = SnapshotRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("list_objects", {})]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place",
            {"object": "banana", "destination": "wicker basket"},
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place",
            {"object": "mango", "destination": "wicker basket"},
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task(
        "将桌面上所有的水果放到筐里"
    )

    assert report.success
    picks = [args for name, args in rt.calls if name == "pick_and_place"]
    assert [item["object"] for item in picks] == ["banana"]
    # Bootstrap executed list locally; the LLM's later re-observe and the
    # suppressed mango retry close the task — mango never executes.
    assert rt.executed.count("list_objects") == 2
    assert rt.executed[-1] == "task_done"


def test_batch_autopilot_skips_per_fruit_llm_round_trips():
    """2026-09-22 log 165241: each batch fruit paid a ~6.5 s LLM round trip
    even though the frozen pending list is authoritative. The autopilot
    synthesizes the next pick_and_place locally — the LLM is chatted only
    for the first pick and the closing task_done, never per fruit."""
    class AutopilotRuntime(_StubRuntime):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, name, args):
            self.executed.append(name)
            self.calls.append((name, dict(args)))
            if name == "list_objects":
                return {"ok": True, "objects": [
                    {"label": "banana", "state": "visible"},
                    {"label": "red apple", "state": "visible"},
                    {"label": "mango", "state": "visible"},
                    {"label": "wicker basket", "state": "visible"},
                ]}
            if name == "task_done":
                return {"ok": True, "task_complete": True}
            return {"ok": True}

    rt = AutopilotRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("list_objects", {})]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place",
            {"object": "banana", "destination": "wicker basket"},
        )]),
        LLMResponse(tool_calls=[ToolCall(
            "task_done", {"success": True, "summary": "all fruit moved"}
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=8).run_task(
        "将桌面上所有的水果放到筐里"
    )

    assert report.success
    picks = [args for name, args in rt.calls if name == "pick_and_place"]
    assert [item["object"] for item in picks] == [
        "banana", "red apple", "mango",
    ]
    # Bootstrap list (local) + 3 picks, then the LLM's scripted list call
    # executes as a no-op re-observe before the closing task_done.
    assert rt.executed == [
        "list_objects", "pick_and_place", "pick_and_place",
        "pick_and_place", "list_objects", "task_done",
    ]
    # The whole batch needed exactly 2 LLM requests: the closing pair
    # (list + task_done from the script). Bootstrap + autopilot covered
    # the opening list and all three picks with zero LLM round trips.
    assert len(llm.requests) == 2


def test_batch_autopilot_failure_reports_failed_fruit_without_retry():
    """A failed autopilot pick moves the fruit to the failed list (one
    outer attempt per fruit — pick_and_place already owns the internal
    retry budget) and the task ends with an honest failure summary. No
    blind retry loop on the frozen list."""
    class FlakyRuntime(_StubRuntime):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, name, args):
            self.executed.append(name)
            self.calls.append((name, dict(args)))
            if name == "list_objects":
                return {"ok": True, "objects": [
                    {"label": "banana", "state": "visible"},
                    {"label": "mango", "state": "visible"},
                    {"label": "wicker basket", "state": "visible"},
                ]}
            if name == "task_done":
                return {"ok": True, "task_complete": True}
            if name == "pick_and_place" and args.get("object") == "mango":
                return {"ok": False, "error": "SkillError: no mango detected"}
            return {"ok": True}

    rt = FlakyRuntime()
    llm = MockLLM([
        LLMResponse(tool_calls=[ToolCall("list_objects", {})]),
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place",
            {"object": "banana", "destination": "wicker basket"},
        )]),
        # The LLM tries to retry mango; the exhausted-list guard converts
        # it to an honest closing task_done instead of re-executing.
        LLMResponse(tool_calls=[ToolCall(
            "pick_and_place",
            {"object": "mango", "destination": "wicker basket"},
        )]),
    ])

    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=8).run_task(
        "将桌面上所有的水果放到筐里"
    )

    assert report.success is False
    assert "mango" in report.summary
    picks = [args for name, args in rt.calls if name == "pick_and_place"]
    assert [item["object"] for item in picks] == ["banana", "mango"]
    assert rt.executed[-1] == "task_done"
    # Bootstrap covered list + both picks' decisions; the LLM was chatted
    # twice at the tail (its scripted list + the suppressed mango retry).
    assert len(llm.requests) == 2
