"""Object-level place orientation constraint (task space, NOT wrist lock).

``skill_place_at`` and the bridge pick-and-place Step4 used to freeze the
carried TCP rotation at place. That over-constrained IK (yaw about world
Z and small tilts are legitimately free while an object stays level) and
made place fail on reachable destinations. The replacement constrains
the OBJECT: its up axis -- captured at grasp close, the one moment it is
known level on the table -- must stay within ``max_object_tilt_deg`` of
world +Z. Yaw/tilt candidates are searched; the winner has the lowest
weighted joint travel minus a joint-limit-margin bonus and must pass the
harness vet. Elongated fruit additionally keeps its long axis near the
table plane (``|long . z| < sin(max_tilt)``).

See ``_place_rotation_candidates`` / ``_search_held_place_rotation`` /
``_capture_held_reference`` in ``wrc_demo/skills/runtime.py``.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from wrc_demo.config import load_demo_config
from wrc_demo.skills.runtime import (
    SkillRuntime,
    _place_rotation_candidates,
    _rot_x,
    _rot_z,
)
from wrc_demo.types import SkillError, make_transform

from conftest import REPO, needs_pin

YAWS = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 180]
TILTS = [0, 5, -5, 10, -10]
TILT_AXES = [0, 90]
MAX_TILT = np.radians(12.0)
Z_HAT = np.array([0.0, 0.0, 1.0])

O6_URDF = (
    REPO / "assets/urdf/rebot_b601_linkerhand_o6/rebot_b601_linkerhand_o6.urdf"
)
# configs/arms/rebot_rs_o6.yaml home_q: base and wrist neutral.
O6_HOME_Q = np.array([0.0, 0.1684, 0.6226, -0.4543, 0.0, 0.0])


class _IkResult:
    """Minimal stand-in for the harness IKResult."""

    def __init__(self, success=True, q=None):
        self.success = success
        self.q = np.zeros(6) if q is None else np.asarray(q, dtype=float)
        self.position_error_m = 0.0
        self.rotation_error_rad = 0.0


def _stub_runtime(kin, cfg=None):
    cfg = cfg or load_demo_config(camera="mock", arm="mock", llm="mock")
    arm = MagicMock()
    arm.harness = MagicMock()
    arm.harness.vet_pose.return_value = None  # safe by default
    arm.get_state.return_value.q = np.zeros(6)
    return SkillRuntime(
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


# ── pure candidate math ───────────────────────────────────────────────


def test_candidates_identity_first_count_and_orthonormal():
    """Nothing is filtered without a long axis; identity is first."""
    R_level = _rot_z(0.4) @ _rot_x(0.2)
    cands = _place_rotation_candidates(R_level, YAWS, TILTS, MAX_TILT)
    expected = len(YAWS) * (1 + (len(TILTS) - 1) * len(TILT_AXES))
    assert len(cands) == expected
    yaw0, tilt0 = cands[0][1], cands[0][2]
    assert (yaw0, tilt0) == (0.0, 0.0)
    assert np.allclose(cands[0][0], R_level, atol=1e-12)
    for R, _, _, _ in cands:
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_candidate_up_axis_tilt_equals_tilt_candidate():
    """The object up-axis tilt off world +Z is exactly |tilt_deg| for every
    yaw (Rz preserves the polar angle) and never exceeds the budget."""
    R_level = _rot_z(-0.3) @ _rot_x(0.5)
    u_tool = R_level.T @ Z_HAT
    cands = _place_rotation_candidates(R_level, YAWS, TILTS, MAX_TILT)
    assert cands
    for R, _yaw, tilt, _tilt_axis in cands:
        up = R @ u_tool
        tilt_rad = float(np.arccos(np.clip(up @ Z_HAT, -1.0, 1.0)))
        assert tilt_rad == pytest.approx(abs(np.radians(tilt)), abs=1e-9)
        assert np.degrees(tilt_rad) <= 12.0 + 1e-6
    # No 45/90-deg flips can be generated: the tilt set caps at 10 deg.
    assert max(abs(t) for _, _, t, _ in cands) <= 10.0


def test_long_axis_filter():
    """Elongated-fruit constraint: a horizontal long axis never violates;
    a vertical one (object held nose-up) can never be placed level and
    must yield ZERO candidates instead of a 90-deg flip."""
    R_level = _rot_z(0.2)
    # Banana lying on the table: long axis horizontal in the tool frame.
    horizontal = R_level.T @ np.array([1.0, 0.0, 0.0])
    cands = _place_rotation_candidates(
        R_level, YAWS, TILTS, MAX_TILT, long_tool=horizontal,
    )
    expected = len(YAWS) * (1 + (len(TILTS) - 1) * len(TILT_AXES))
    assert len(cands) == expected
    # Object grasped nose-up: |(R @ long).z| = cos(tilt) >= sin(12 deg)
    # for every candidate -> all filtered -> place reports unreachable.
    vertical = R_level.T @ Z_HAT
    cands = _place_rotation_candidates(
        R_level, YAWS, TILTS, MAX_TILT, long_tool=vertical,
    )
    assert cands == []


# ── search on stub kinematics ─────────────────────────────────────────


def test_search_picks_feasible_yaw_over_infeasible_identity():
    """The exact failure this change fixes: the carried (identity)
    rotation has no IK solution at the destination, a yaw-rotated
    candidate does -- the search must return it instead of failing."""
    R_level = _rot_z(0.4) @ _rot_x(0.2)
    kin = MagicMock()
    kin.fk.return_value = make_transform(R_level, np.array([0.25, 0.0, 0.30]))
    only_feasible = _rot_z(np.radians(-30.0)) @ R_level

    def ik(T_target, q_init, retries=0, **kwargs):
        R = np.asarray(T_target, dtype=float)[:3, :3]
        return _IkResult(success=bool(np.allclose(R, only_feasible, atol=1e-9)))

    kin.ik.side_effect = ik
    kin.joint_limits = (np.full(6, -3.0), np.full(6, 3.0))
    rt = _stub_runtime(kin)
    rt._held_R_level = R_level.copy()

    found = rt._search_held_place_rotation(
        np.array([0.25, 0.05, 0.10]),
        np.array([0.25, 0.05, 0.15]),
        np.zeros(6),
    )
    assert found is not None
    assert found["yaw_deg"] == -30.0
    assert found["tilt_deg"] == 0.0
    assert np.allclose(found["R"], only_feasible, atol=1e-9)
    events = [c.args[0] for c in rt.trace.record_event.call_args_list]
    assert "place_orientation_search" in events


def test_search_prefers_larger_limit_margin():
    """Two feasible candidates with identical joint travel: the one with
    the larger joint-limit margin wins (place_margin_weight)."""
    R_level = np.eye(3)
    kin = MagicMock()
    kin.fk.return_value = make_transform(R_level, np.array([0.25, 0.0, 0.30]))
    q_near_limit = np.array([2.9, 0.0, 0.0, 0.0, 0.0, 0.0])   # margin 0.1
    q_centered = np.array([2.05, 2.05, 0.0, 0.0, 0.0, 0.0])   # margin 0.95
    assert np.linalg.norm(q_near_limit) == pytest.approx(
        np.linalg.norm(q_centered), rel=1e-3
    )
    by_yaw = {15.0: q_near_limit, -15.0: q_centered}

    def ik(T_target, q_init, retries=0, **kwargs):
        R = np.asarray(T_target, dtype=float)[:3, :3]
        for yaw, q in by_yaw.items():
            if np.allclose(R, _rot_z(np.radians(yaw)) @ R_level, atol=1e-9):
                return _IkResult(success=True, q=q)
        return _IkResult(success=False)

    kin.ik.side_effect = ik
    kin.joint_limits = (np.full(6, -3.0), np.full(6, 3.0))
    rt = _stub_runtime(kin)
    rt._held_R_level = R_level.copy()

    found = rt._search_held_place_rotation(
        np.array([0.25, 0.05, 0.10]),
        np.array([0.25, 0.05, 0.15]),
        np.zeros(6),
    )
    assert found is not None
    assert found["yaw_deg"] == -15.0  # larger margin, identical travel


def test_place_at_raises_and_keeps_held_when_all_candidates_fail():
    """No candidate solvable: SkillError mentions the tilt budget, and the
    held state must survive the failure (a silent clear would make the
    next retry think the hand is empty)."""
    kin = MagicMock()
    kin.fk.return_value = np.eye(4)
    kin.ik.return_value = _IkResult(success=False)
    rt = _stub_runtime(kin)
    rt.held_object = "banana"
    rt._held_R_level = np.eye(3)

    with pytest.raises(SkillError, match="object kept level"):
        rt.skill_place_at(0.20, -0.15)
    assert rt.held_object == "banana"
    assert rt._held_R_level is not None


def test_place_orientation_search_trace_counts_candidates():
    """The search trace event reports the scan so the dashboard/narration
    can explain WHY a yaw was chosen (or why place failed)."""
    R_level = _rot_z(0.4)
    kin = MagicMock()
    kin.fk.return_value = make_transform(R_level, np.zeros(3))
    kin.ik.return_value = _IkResult(success=True)  # everything feasible
    rt = _stub_runtime(kin)
    rt._held_R_level = R_level.copy()

    found = rt._search_held_place_rotation(
        np.array([0.25, 0.05, 0.10]),
        np.array([0.25, 0.05, 0.15]),
        np.zeros(6),
    )
    assert found is not None
    assert found["yaw_deg"] == 0.0 and found["tilt_deg"] == 0.0
    event = next(
        c for c in rt.trace.record_event.call_args_list
        if c.args[:1] == ("place_orientation_search",)
    )
    payload = event.args[1]
    expected = len(YAWS) * (1 + (len(TILTS) - 1) * len(TILT_AXES))
    assert payload["n_candidates"] == expected
    assert payload["level_reference"] == "grasp_close"
    assert payload["chosen_yaw_deg"] == 0.0


def test_fallback_level_reference_when_none_captured():
    """A restart while holding left no captured reference: the CURRENT
    orientation becomes the reference and the trace says so (identity
    candidate then equals the old frozen-rotation behavior)."""
    R_now = _rot_z(0.9) @ _rot_x(0.1)
    kin = MagicMock()
    kin.fk.return_value = make_transform(R_now, np.zeros(3))
    kin.ik.return_value = _IkResult(success=True)
    rt = _stub_runtime(kin)
    assert rt._held_R_level is None

    found = rt._search_held_place_rotation(
        np.array([0.25, 0.05, 0.10]),
        np.array([0.25, 0.05, 0.15]),
        np.zeros(6),
    )
    assert found is not None
    assert np.allclose(found["R"], R_now, atol=1e-12)  # identity won
    event = next(
        c for c in rt.trace.record_event.call_args_list
        if c.args[:1] == ("place_orientation_search",)
    )
    assert event.args[1]["level_reference"] == "fallback_current_fk"


# ── held-state bookkeeping ────────────────────────────────────────────


def test_forget_held_object_clears_level_reference_too():
    """Every held-state clear site goes through _forget_held_object; a
    stale level reference must never outlive the hold it belongs to."""
    rt = _stub_runtime(MagicMock())
    rt.held_object = "banana"
    rt._held_det_label = "banana"
    rt._held_color = "yellow"
    rt._held_support_offset_m = 0.03
    rt._held_R_level = np.eye(3)
    rt._held_long_tool = np.array([1.0, 0.0, 0.0])
    rt._forget_held_object()
    assert rt.held_object is None
    assert rt._held_det_label is None
    assert rt._held_color is None
    assert rt._held_support_offset_m is None
    assert rt._held_R_level is None
    assert rt._held_long_tool is None


def test_adopt_unknown_held_captures_current_fk_as_level_reference():
    """Adopted holds (jaws jammed on an unregistered object) get the
    current orientation as their level reference."""
    R_now = _rot_z(0.7) @ _rot_x(0.1)
    kin = MagicMock()
    kin.fk.return_value = make_transform(R_now, np.zeros(3))
    rt = _stub_runtime(kin)
    rt._gripper_width_frac = MagicMock(return_value=0.5)

    rt._adopt_unknown_held()

    assert rt.held_object == "object"
    assert np.allclose(rt._held_R_level, R_now)
    assert rt._held_long_tool is None  # no fix -> no long-axis constraint
    capture = next(
        c for c in rt.trace.record_event.call_args_list
        if c.args[:1] == ("held_level_capture",)
    )
    assert capture.args[1]["source"] == "adopted_current_fk"


# ── compact held carry ───────────────────────────────────────────────


def test_held_carry_lifts_then_executes_vetted_compact_solution():
    kin = MagicMock()
    T_now = make_transform(np.eye(3), np.array([0.40, 0.12, 0.05]))
    kin.fk.return_value = T_now
    rt = _stub_runtime(kin)
    rt.cfg.arm._data["held_carry_z_m"] = 0.16
    rt.cfg.arm._data["held_carry_xyz_candidates_m"] = [[0.28, 0.02, 0.20]]
    rt.arm.move_cartesian.return_value = True
    rt.arm.move_joints.return_value = True
    q_compact = np.array([0.1, 0.2, 0.3, 0.0, 0.1, -0.2])
    rt._search_held_place_rotation = MagicMock(return_value={
        "low": _IkResult(q=q_compact), "yaw_deg": 30.0,
        "tilt_deg": 5.0, "tilt_axis_deg": 90.0, "score": 0.4,
    })

    result = rt._move_held_to_carry(duration_s=2.0)

    assert result["motion"] == "compact_carry"
    assert result["tcp_xyz_m"] == [0.28, 0.02, 0.2]
    assert result["segments"] == 2
    assert rt.arm.move_cartesian.call_count == 0
    assert rt._search_held_place_rotation.call_count == 3
    lift_target = np.asarray(
        rt._search_held_place_rotation.call_args_list[0].args[0]
    )
    assert lift_target.tolist() == pytest.approx([0.40, 0.12, 0.16])
    assert rt.arm.move_joints.call_count == 3
    rt.arm.move_joints.assert_any_call(q_compact, duration_s=2.0)


def test_held_carry_keeps_vertical_lift_when_compact_pose_unreachable():
    kin = MagicMock()
    kin.fk.return_value = make_transform(
        np.eye(3), np.array([0.40, 0.12, 0.05])
    )
    rt = _stub_runtime(kin)
    rt.cfg.arm._data["held_carry_z_m"] = 0.16
    rt.cfg.arm._data["held_carry_xyz_candidates_m"] = [[0.28, 0.02, 0.20]]
    rt.arm.move_cartesian.return_value = True
    rt._search_held_place_rotation = MagicMock(return_value=None)

    result = rt._move_held_to_carry(duration_s=2.0)

    assert result == {"motion": "lifted", "tcp_z_m": 0.16, "compact": False}
    rt.arm.move_joints.assert_not_called()


# ── real O6 kinematics ────────────────────────────────────────────────


@needs_pin
@pytest.mark.skipif(not O6_URDF.exists(), reason="O6 URDF not on disk")
def test_search_finds_candidate_with_real_o6_kinematics(demo_cfg):
    """Property test on the real O6 URDF at a measured-hard destination:
    several frozen-orientation yaws have no IK solution there while
    yaw-rotated candidates do. Property assertions only -- retries > 0
    uses random restarts, so the exact winner is nondeterministic."""
    import pinocchio as pin
    from wrc_demo.safety.harness import Kinematics

    kin = Kinematics(str(O6_URDF), "o6_grasp_tcp", 6)
    # rpy(0, pi/2, 0) reference at (0.25, 0.05, 0.10): measured 2026-09-21,
    # yaws 15/30/45/60/90/120/180 have no solution from the home seed --
    # exactly the frozen-rotation place failure this change fixes.
    R_level = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)
    rt = _stub_runtime(kin, cfg=demo_cfg)
    rt._held_R_level = R_level.copy()
    # This property test exercises endpoint IK feasibility. Its synthetic
    # home q is not a physically consistent held-object pose, so bypass the
    # separate interpolation/tilt vet covered by the stub tests above.
    rt._vet_place_segment = MagicMock(return_value=None)

    t0 = time.monotonic()
    found = rt._search_held_place_rotation(
        np.array([0.25, 0.05, 0.10]),
        np.array([0.25, 0.05, 0.15]),
        O6_HOME_Q.copy(),
    )
    elapsed = time.monotonic() - t0

    assert found is not None, "a yaw/tilt candidate must be feasible here"
    u_tool = R_level.T @ Z_HAT
    up = found["R"] @ u_tool
    tilt_deg = float(np.degrees(np.arccos(np.clip(up @ Z_HAT, -1.0, 1.0))))
    assert tilt_deg <= 12.0 + 1e-6
    # The chosen release pose actually solves: FK(hover/release) lands on
    # the commanded targets.
    assert np.allclose(kin.fk(found["low"].q)[:3, 3], [0.25, 0.05, 0.10], atol=0.01)
    assert np.allclose(kin.fk(found["pre"].q)[:3, 3], [0.25, 0.05, 0.15], atol=0.01)
    # Full 14x5 scan stays interactive: measured ~7-30 ms per IK solve.
    assert elapsed < 15.0
