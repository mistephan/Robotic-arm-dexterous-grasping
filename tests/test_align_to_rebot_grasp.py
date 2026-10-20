"""Tests for the rebot_grasp-aligned skill_grasp_object bridge path.

The legacy ``skill_grasp_object`` uses SafeArm.move_joints / move_cartesian +
_close_two_stage + air-grasp verification. The new bridge path mirrors
``rebot_grasp/scripts/main.py::execute_grasp`` exactly:

    open_gripper → move_to_traj(pre, 2.0) → move_to_traj(grasp, 1.5)
              → width-limited grip → move_to_traj(home, 3.0)

The legacy path is still exercised by existing tests with mock arm (no
bridge); these tests assert the bridge path is used when ``runtime.bridge``
is set.
"""

from unittest.mock import MagicMock, patch, call as mock_call

import numpy as np
import pinocchio as pin
import pytest

from wrc_demo.config import load_demo_config
from wrc_demo.skills.runtime import SkillRuntime
from wrc_demo.types import SkillError


# ──────────────── helpers ────────────────


def _grasp_pose(position=(0.30, -0.10, 0.05), rpy=(0.0, 1.2, 0.0)):
    """Build a Grasp-like MagicMock with .position / .rotation / .width_m / .approach."""
    grasp = MagicMock()
    grasp.position = np.array(position, dtype=float)
    grasp.rotation = pin.rpy.rpyToMatrix(*rpy)
    grasp.width_m = 0.030
    grasp.approach = np.array([0.0, 0.0, -1.0])
    grasp.quality = 0.9
    grasp.label = "banana"
    grasp.pregrasp_position = lambda offset_m: (
        grasp.position - grasp.approach * offset_m
    )
    return grasp


def _fix():
    fix = MagicMock()
    fix.detection.label = "banana"
    fix.detection.mask = None
    fix.detection.bbox = np.array([100, 100, 200, 200])
    fix.position = np.array([0.30, -0.10, 0.05])
    return fix


def _frame():
    """A minimal Frame mock whose .rgb is a real yellow BGR image (so
    detection_color can extract a color name)."""
    frame = MagicMock()
    frame.rgb = np.full((480, 640, 3), [0, 255, 255], dtype=np.uint8)  # yellow BGR
    return frame


def _make_runtime_with_bridge(bridge):
    """Build a SkillRuntime wired to use the bridge path."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    arm = MagicMock()
    arm.grasp_bridge = bridge
    arm.harness = MagicMock()
    arm.harness.vet_pose.return_value = None  # always safe by default
    arm.harness.allow_grasp_descent = MagicMock()
    arm.harness.clear_grasp_exemption = MagicMock()
    # Off-ready start pose: the production ready-skip (bridge_motion_skipped)
    # fires when the measured joints are already within home_noop_tol_rad of
    # the ready target — zeros would collide with the mocked IK target and
    # silently eat the ready move this file's sequence tests assert on.
    arm.get_state.return_value.q = np.full(6, 0.05)

    # kin.fk is called for home_q → 6D; return a real matrix so
    # pin.rpy.matrixToRpy succeeds. (0.25, 0, 0.35) with rpy=(0, π/2, 0)
    # matches wrc_demo's home_q FK approximation.
    kin = MagicMock()
    home_T = np.eye(4)
    home_T[:3, 3] = [0.25, 0.0, 0.35]
    home_T[:3, :3] = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)
    kin.fk.return_value = home_T
    ik_ok = MagicMock(success=True, q=np.zeros(6))
    kin.ik.return_value = ik_ok

    rt = SkillRuntime(
        camera=MagicMock(),
        depth_provider=MagicMock(),
        detector=MagicMock(),
        extrinsics=MagicMock(),
        kin=kin,
        safe_arm=arm,
        memory=MagicMock(),
        beliefs=MagicMock(),
        trace=MagicMock(),
        cfg=cfg,
    )
    rt._reconcile_held = MagicMock()
    rt._localize = MagicMock(return_value=(_frame(), _fix()))
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])
    rt._max_width = 0.095
    return rt


def _patched_select_grasp(rt):
    """Context manager: stub select_grasp to return canned grasp + q seeds."""
    q_pre = np.array([0.1, 0.2, 0.3, -0.1, 0.0, 0.0])
    q_grasp = np.array([0.1, 0.2, 0.4, -0.2, 0.0, 0.0])
    return patch(
        "wrc_demo.skills.runtime.select_grasp",
        return_value=(_grasp_pose(), q_pre, q_grasp),
    )


# ──────────────── tests ────────────────


def test_bridge_path_calls_open_then_pre_then_grasp_then_home():
    """Bridge flow: ready(3.0) → open → pre(2.0) → grasp(1.5) → width-limited grip → ready(3.0)."""
    bridge = MagicMock()
    bridge.last_grasp_telemetry.return_value = {
        "stop_reason": "tactile_contact", "contacted": ["thumb", "index"],
    }
    rt = _make_runtime_with_bridge(bridge)

    with _patched_select_grasp(rt):
        rt.skill_grasp_object("banana")

    # open first (after ready)
    bridge.open_gripper.assert_called_once()
    # then width-limited, material-aware grasp
    bridge.grasp.assert_called_once_with(
        force=0.35, min_width_m=0.012, close_force=0.70,
    )
    # Ready reuses vetted IK through SafeArm; only pregrasp and grasp stay Cartesian.
    durations = [c.kwargs.get("duration") for c in bridge.move_to_traj.call_args_list]
    assert durations == [2.0, 1.5]
    joint_durations = [c.kwargs.get("duration_s") for c in rt.arm.move_joints.call_args_list]
    assert joint_durations == [3.0]


def test_bridge_path_pregrasp_position_is_offset_from_grasp():
    """pregrasp z should use the configured 0.05 m offset (pregrasp_offset_m)."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)

    with _patched_select_grasp(rt):
        rt.skill_grasp_object("banana")

    # Ready moves are joint-space; bridge calls are pregrasp(0), grasp(1).
    pre_call = bridge.move_to_traj.call_args_list[0]
    assert pre_call.args[2] == pytest.approx(0.05 + 0.05)
    assert pre_call.kwargs["duration"] == 2.0

    grasp_call = bridge.move_to_traj.call_args_list[1]
    assert grasp_call.args[2] == pytest.approx(0.05)
    assert grasp_call.kwargs["duration"] == 1.5


def test_bridge_path_reuses_selector_ik_for_pregrasp_and_grasp():
    """Execution must not re-solve IK after selector already vetted q targets."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)
    # This fixture has no configured ready_q. Only the empty-hand ready move
    # solves IK; held retreat preserves the current FK rotation.
    rt.kin.ik.side_effect = [
        MagicMock(success=True, q=np.zeros(6)),
        MagicMock(success=True, q=np.zeros(6)),
    ]

    with _patched_select_grasp(rt):
        result = rt.skill_grasp_object("banana")

    assert result["ok"] is True
    assert rt.kin.ik.call_count == 1


def test_bridge_path_empty_grasp_returns_failure_dict():
    """bridge.grasp returns False (no contact) → ok=False, no exception."""
    bridge = MagicMock()
    bridge.move_to_traj.return_value = True
    bridge.grasp.return_value = False
    rt = _make_runtime_with_bridge(bridge)

    with _patched_select_grasp(rt):
        result = rt.skill_grasp_object("banana")

    assert result["ok"] is False
    assert "empty grasp" in result["error"].lower()


def test_bridge_path_success_sets_held_object_and_marks_belief():
    """On successful grasp, runtime.held_object is set + belief removed."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)

    fix = _fix()
    rt._localize = MagicMock(return_value=(_frame(), fix))

    with _patched_select_grasp(rt):
        result = rt.skill_grasp_object("banana")

    assert result["ok"] is True
    assert rt.held_object == "banana"
    rt.beliefs.mark_removed.assert_called_once()


def test_bridge_carry_runtime_error_does_not_reclassify_verified_grasp():
    """A post-contact CLIK failure is a carry failure, not a grasp failure."""
    bridge = MagicMock()
    bridge.last_grasp_telemetry.return_value = {
        "stop_reason": "tactile_contact", "contacted": ["thumb", "index"],
    }
    rt = _make_runtime_with_bridge(bridge)
    rt._move_held_to_carry = MagicMock(
        side_effect=RuntimeError("CLIK tracking failed")
    )

    with _patched_select_grasp(rt):
        result = rt.skill_grasp_object("banana")

    assert result["ok"] is True
    assert result["grip_verified"] is True
    assert rt.held_object == "banana"
    assert any(
        "grasp still succeeded" in call.args[1]
        for call in rt.memory.add.call_args_list
    )

def test_legacy_path_used_when_bridge_is_none():
    """If bridge is None (mock arm), the legacy v5 path runs.

    We don't run the full legacy flow (needs harness limits mocks);
    instead we patch harness limits + early helpers so we can confirm
    the dispatch reached the legacy branch.
    """
    rt = _make_runtime_with_bridge(None)
    rt.bridge = None

    # Make harness limits return real numeric values so the legacy code
    # can compute z_floor / vet_pose etc.
    rt.arm.harness.limits.workspace_min = np.array([-1.0, -1.0, 0.0])
    rt.arm.harness.limits.table_z = 0.0
    rt.arm.harness.limits.max_joint_vel = 1.0
    rt.arm.harness.vet_pose = MagicMock(return_value=None)
    rt.arm.harness.allow_grasp_descent = MagicMock()
    rt.arm.harness.clear_grasp_exemption = MagicMock()

    # Patch perception + IK helpers so we don't need a real environment.
    rt._plan_grasps = MagicMock(return_value=[])  # empty → legacy raises
    rt.kin.fk.return_value = np.eye(4)
    rt.kin.ik.return_value = MagicMock(success=True, q=np.zeros(6))

    with pytest.raises(SkillError, match="no executable grasp"):
        rt.skill_grasp_object("banana")

    rt._plan_grasps.assert_called_once()
    assert rt.bridge is None


# ──────────────── pick_and_place bridge path ────────────────


def _fix_at(position, label="banana"):
    fix = MagicMock()
    fix.detection.label = label
    fix.detection.mask = None
    fix.detection.bbox = np.array([100, 100, 200, 200])
    fix.position = np.array(position, dtype=float)
    return fix


def _make_pick_and_place_runtime(bridge, *, already_held=None):
    """Build a SkillRuntime for pick_and_place tests."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    arm = MagicMock()
    arm.grasp_bridge = bridge
    arm.harness = MagicMock()
    arm.harness.vet_pose.return_value = None  # always safe by default
    arm.harness.allow_grasp_descent = MagicMock()
    arm.harness.clear_grasp_exemption = MagicMock()
    # Off-ready start (see _make_runtime_with_bridge): keeps the Step-0 ready
    # move in the asserted [3.0, 2.0, 3.0] joint-duration sequence.
    arm.get_state.return_value.q = np.full(6, 0.05)

    kin = MagicMock()
    home_T = np.eye(4)
    home_T[:3, 3] = [0.25, 0.0, 0.35]
    home_T[:3, :3] = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)
    kin.fk.return_value = home_T
    ik_ok = MagicMock(success=True, q=np.zeros(6))
    kin.ik.return_value = ik_ok

    rt = SkillRuntime(
        camera=MagicMock(),
        depth_provider=MagicMock(),
        detector=MagicMock(),
        extrinsics=MagicMock(),
        kin=kin,
        safe_arm=arm,
        memory=MagicMock(),
        beliefs=MagicMock(),
        trace=MagicMock(),
        cfg=cfg,
    )
    rt._reconcile_held = MagicMock()
    rt._reobserve = MagicMock()
    rt.observe = MagicMock(return_value=_frame())
    rt._max_width = 0.095
    rt.held_object = already_held
    return rt


def test_pick_and_place_bridge_path_calls_full_sequence():
    """Bridge pick_and_place: same-frame detect → grasp → home → place → home."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    # Stubs: same-frame detect returns banana + box
    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        result = rt.skill_pick_and_place("banana", destination="yellow box")

    # Held place executes the exact IK solution vetted by the object-level
    # constraint; on this fixture every candidate ties (constant FK), so the
    # identity candidate -- the carried rotation -- wins.
    durations = [c.kwargs.get("duration") for c in bridge.move_to_traj.call_args_list]
    assert durations == [2.0, 1.5]
    joint_durations = [c.kwargs.get("duration_s") for c in rt.arm.move_joints.call_args_list]
    assert joint_durations == [3.0, 2.0, 3.0]
    rt.arm.move_cartesian.assert_not_called()
    assert bridge.open_gripper.call_args_list == [
        mock_call(distance_m=0.03), mock_call(timeout=6.0),
    ]
    bridge.grasp.assert_called_once_with(
        force=0.35, min_width_m=0.012, close_force=0.70,
    )
    bridge.release_with_async_close.assert_not_called()
    bridge.close_gripper.assert_not_called()
    # 2026-09-22: no idle close after a verified release — the hand stays
    # open at the clear ready pose instead of close/reopen per placement.
    events = [c.args[0] for c in rt.trace.record_event.call_args_list]
    assert "gripper_left_open_after_place" in events


def test_safe_bridge_move_skips_ready_when_already_at_target():
    """Ready-skip: measured joints within home_noop_tol_rad of the ready
    target → no motion, bridge_motion_skipped traced, True returned."""
    bridge = MagicMock()
    rt = _make_runtime_with_bridge(bridge)
    rt.arm.get_state.return_value.q = np.zeros(6)  # exactly the IK-ready target
    ready6d = rt._ready_pose_6d()

    assert rt._safe_bridge_move(ready6d, duration=3.0, where="ready") is True
    rt.arm.move_joints.assert_not_called()
    events = [c.args[0] for c in rt.trace.record_event.call_args_list]
    assert "bridge_motion_skipped" in events


def test_pick_and_place_bridge_path_uses_same_frame_for_both_targets():
    """Both banana and box must come from ONE observe() call."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # observe() called exactly once (same-frame detect)
    rt.observe.assert_called_once()
    # _single_localize called twice (once for each label) on the same frame
    assert rt._single_localize.call_count == 2
    # Both calls received the same frame argument
    frames_passed = [c.args[0] for c in rt._single_localize.call_args_list]
    assert frames_passed[0] is frames_passed[1], (
        "banana and box should be detected on the SAME frame"
    )


def test_pick_and_place_bridge_path_uses_configured_release_height():
    """Place move_to_traj uses the configured safe destination height."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    search_event = next(
        c for c in rt.trace.record_event.call_args_list
        if c.args[:1] == ("place_orientation_search",)
    )
    assert search_event.args[1]["target"][2] == pytest.approx(0.14)
    place_call = next(
        c for c in rt.arm.move_joints.call_args_list
        if c.kwargs.get("duration_s") == 2.0
    )
    assert np.allclose(place_call.args[0], np.zeros(6))


def test_bridge_pick_failure_returns_home(monkeypatch):
    """A bridge failure after partial motion must park the arm."""
    from wrc_demo.types import SafetyViolation

    rt = _make_pick_and_place_runtime(MagicMock())
    rt.skill_move_home = MagicMock(return_value={"at": "home"})
    rt.arm.harness.estopped = False
    monkeypatch.setattr(
        rt, "_skill_pick_and_place_bridge",
        MagicMock(side_effect=SafetyViolation("target outside workspace")),
    )

    with pytest.raises(SafetyViolation, match="outside workspace"):
        rt.skill_pick_and_place("banana", destination="wicker basket")

    rt.skill_move_home.assert_called_once_with(park_gripper=False)


@pytest.mark.parametrize("text, expected", [
    ("left of lime", ("left", "lime")),
    ("left side of the lime", ("left", "lime")),
    ("right of mango", ("right", "mango")),
    ("青柠左边", ("left", "lime")),
    ("芒果右侧", ("right", "mango")),
])
def test_parse_relative_destination(text, expected):
    assert SkillRuntime._parse_relative_destination(text) == expected


def test_o6_home_parks_hand_open_not_power_closed():
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)
    rt._refresh_motion_watchdog = MagicMock()
    rt.arm.end_effector = MagicMock()
    result = rt.skill_move_home()
    rt.arm.end_effector.open.assert_called_once_with()
    bridge.close_gripper.assert_not_called()
    assert result["gripper"] == "open"


def test_failure_home_does_not_close_uncertain_gripper():
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)
    rt._refresh_motion_watchdog = MagicMock()
    result = rt.skill_move_home(park_gripper=False)
    bridge.close_gripper.assert_not_called()
    assert result["gripper"] == "unchanged"


def test_relative_place_uses_robot_base_y_and_checks_workspace():
    rt = _make_pick_and_place_runtime(MagicMock())
    rt.arm.harness.limits.workspace_min = np.array([0.10, -0.15, -0.01])
    rt.arm.harness.limits.workspace_max = np.array([0.50, 0.30, 0.55])
    rt._localize = MagicMock(return_value=(_frame(), _fix_at([0.30, 0.10, 0.04], "lime")))
    assert rt._relative_place_xy("lime", "left") == pytest.approx((0.30, 0.18))
    assert rt._relative_place_xy("lime", "right") == pytest.approx((0.30, 0.02))
    event = rt.trace.record_event.call_args_list[0]
    assert event.args[1]["convention"] == "+Y=left, -Y=right"
    rt._localize.return_value = (_frame(), _fix_at([0.30, 0.27, 0.04], "lime"))
    with pytest.raises(SkillError, match="outside workspace"):
        rt._relative_place_xy("lime", "left")


def test_relative_place_uses_measured_object_extents_and_clearance():
    rt = _make_pick_and_place_runtime(MagicMock())
    rt.arm.harness.limits.workspace_min = np.array([0.10, -0.20, -0.01])
    rt.arm.harness.limits.workspace_max = np.array([0.50, 0.30, 0.55])
    fix = _fix_at([0.30, 0.10, 0.04], "banana")
    fix.points = np.array([[0.30, 0.07, 0.04], [0.30, 0.10, 0.04],
                           [0.30, 0.13, 0.04]])
    rt._held_xy_half_extent_m = np.array([0.04, 0.04])
    rt._localize = MagicMock(return_value=(_frame(), fix))
    x, y = rt._relative_place_xy("banana", "right")
    payload = rt.trace.record_event.call_args.args[1]
    assert x == pytest.approx(0.30)
    assert payload["offset_m"] > 0.08
    assert payload["clearance_m"] == pytest.approx(
        float(rt.cfg.grasp.get("relative_place_clearance_m", 0.025))
    )
    assert y == pytest.approx(0.10 - payload["offset_m"])


def test_pick_and_place_bridge_path_returns_failure_if_no_box():
    """If box not detected, raise SkillError before grasp."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    rt._single_localize = MagicMock(side_effect=[banana_fix, None])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with pytest.raises(SkillError, match="yellow box"):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # Bridge.grasp NOT called (we never reached the grasp stage)
    bridge.grasp.assert_not_called()


def test_pick_and_place_bridge_path_clears_held_after_release():
    """After release, held_object is None + belief updated to drop point."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    assert rt.held_object is None
    # Belief was updated to the drop point
    assert rt.beliefs.update.called


def test_pick_and_place_bridge_path_skipped_when_legacy_no_destination():
    """Bridge path requires destination (set.py flow is pick-and-place); without
    it, the legacy path runs.
    """
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)
    rt._localize = MagicMock(side_effect=SkillError("no banana detected"))

    try:
        rt.skill_pick_and_place("banana", destination=None)  # no destination
    except (SkillError, Exception):
        pass  # legacy flow may raise; we just want to verify bridge NOT used

    # Bridge methods were never called (no destination → legacy path)
    bridge.open_gripper.assert_not_called()
    bridge.grasp.assert_not_called()
    bridge.move_to_traj.assert_not_called()


def test_pick_and_place_bridge_warns_when_box_top_above_release_height():
    """If box top is higher than the configured release_z=0.14, log a warning."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    # Box top is 0.15 m — well above 0.14 release height
    box_fix.points = np.array([
        [0.35, 0.10, 0.10],
        [0.35, 0.11, 0.15],  # highest point
    ])
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # memory.add was called with a "warning" event mentioning the collision risk
    warning_calls = [
        c for c in rt.memory.add.call_args_list
        if c.args[0] == "warning"
    ]
    assert len(warning_calls) >= 1, (
        f"expected at least one warning event, got {rt.memory.add.call_args_list}"
    )
    msg = warning_calls[0].args[1]
    assert "box top_z" in msg
    assert "0.15" in msg


def test_pick_and_place_bridge_no_warning_when_box_top_below_release_height():
    """If box top is below 0.14 (e.g. flat box), no warning."""
    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.05], label="yellow box")
    # Box top is 0.05 m — below 0.08 release height
    box_fix.points = np.array([[0.35, 0.10, 0.05]])
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    with _patched_select_grasp(rt):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # No "warning" event for box top
    warning_calls = [
        c for c in rt.memory.add.call_args_list
        if c.args[0] == "warning" and "box top_z" in c.args[1]
    ]
    assert len(warning_calls) == 0


def test_pick_and_place_bridge_keeps_held_on_place_failure():
    """A failure before release keeps held state for a safe retry."""
    from wrc_demo.types import SafetyViolation

    bridge = MagicMock()
    rt = _make_pick_and_place_runtime(bridge)

    banana_fix = _fix_at([0.30, -0.10, 0.05], label="banana")
    box_fix = _fix_at([0.35, 0.10, 0.10], label="yellow box")
    box_fix.points = np.array([[0.35, 0.10, 0.10]])
    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])
    rt._plan_grasps = MagicMock(return_value=[_grasp_pose()])

    # Fail the rotation-preserving held-object place motion.
    def fail_place(_q, duration_s=0.0):
        if duration_s == 2.0:
            raise SafetyViolation("place motion failed")
        return True
    rt.arm.move_joints.side_effect = fail_place

    with _patched_select_grasp(rt):
        with pytest.raises(SafetyViolation, match="place motion"):
            rt.skill_pick_and_place("banana", destination="yellow box")

    assert rt.held_object == "banana"
    assert rt._held_det_label == "banana"
    assert rt._held_R_level is not None


def test_wicker_basket_destination_retries_detector_aliases():
    rt = object.__new__(SkillRuntime)
    fix = MagicMock()
    rt._single_localize = MagicMock(side_effect=[None, fix])
    rt.trace = MagicMock()
    frame = MagicMock()

    result = rt._localize_destination(frame, "wicker basket")

    assert result is fix
    assert rt._single_localize.call_args_list == [
        mock_call(frame, "wicker basket"),
        mock_call(frame, "basket"),
    ]
