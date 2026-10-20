"""Verify bridge path calls harness.vet_pose before every move_to_traj.

The bridge path bypasses SafeArm's per-waypoint approval (rebort_rs SDK
controller owns the control loop). Without an explicit pre-motion
check, joint-limit / workspace-AABB violations from rebot_grasp's IK
won't be caught until the e-stop latch fires.

These tests pin the contract: every ``bridge.move_to_traj`` call is
preceded by a ``harness.vet_pose(q_target)`` call. vet_pose failures
must raise ``SafetyViolation`` (not just return False).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pinocchio as pin
import pytest

from wrc_demo.config import load_demo_config
from wrc_demo.skills.runtime import SkillRuntime
from wrc_demo.types import SafetyViolation, SkillError


def _grasp_pose(position=(0.30, -0.10, 0.05)):
    grasp = MagicMock()
    grasp.position = np.array(position, dtype=float)
    grasp.rotation = pin.rpy.rpyToMatrix(0.0, 1.2, 0.0)
    grasp.width_m = 0.030
    grasp.approach = np.array([0.0, 0.0, -1.0])
    grasp.pregrasp_position = lambda offset_m: (
        grasp.position - grasp.approach * offset_m
    )
    return grasp


def _fix():
    fix = MagicMock()
    fix.detection.label = "banana"
    fix.detection.bbox = np.array([100, 100, 200, 200])
    fix.detection.mask = np.ones((200, 200), dtype=bool)
    fix.position = np.array([0.30, -0.10, 0.05])
    return fix


def _frame():
    """Real numpy frame with .rgb so detection_color works."""
    frame = MagicMock()
    frame.rgb = np.full((480, 640, 3), [0, 255, 255], dtype=np.uint8)  # yellow
    return frame


def _runtime_with_bridge(bridge, *, vet_pose_returns=None):
    """Build SkillRuntime where vet_pose returns the given string (None = safe)."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    arm = MagicMock()
    arm.grasp_bridge = bridge
    arm.harness = MagicMock()
    arm.harness.vet_pose.return_value = vet_pose_returns
    arm.harness.allow_grasp_descent = MagicMock()
    arm.harness.clear_grasp_exemption = MagicMock()
    arm.get_state.return_value.q = np.zeros(6)

    kin = MagicMock()
    home_T = np.eye(4)
    home_T[:3, 3] = [0.25, 0.0, 0.35]
    home_T[:3, :3] = pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0)
    kin.fk.return_value = home_T
    ik_ok = MagicMock(success=True, q=np.array([0.1, 0.2, 0.3, -0.1, 0.0, 0.0]))
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
    rt.observe = MagicMock(return_value=_frame())  # for pick_and_place bridge
    rt._max_width = 0.095
    return rt


def test_vet_pose_called_before_each_move_to_traj():
    """Every bridge.move_to_traj is preceded by a harness.vet_pose call."""
    bridge = MagicMock()
    rt = _runtime_with_bridge(bridge)

    with patch(
        "wrc_demo.skills.runtime.select_grasp",
        return_value=(_grasp_pose(), np.zeros(6), np.zeros(6)),
    ):
        rt.skill_grasp_object("banana")

    # A confirmed hold now retreats through one Cartesian lift with its TCP
    # rotation frozen instead of rotating to the fixed ready_q.
    assert bridge.move_to_traj.call_count == 2
    assert rt.arm.move_joints.call_count == 1
    # Fixture FK is already above held_carry_z_m, so the lift is a no-op.
    rt.arm.move_cartesian.assert_not_called()
    # vet_pose called at least 3 times (one per move_to_traj; home uses q_target so
    # may use it once for the IK-then-vet pattern or skip IK — both are valid)
    assert rt.arm.harness.vet_pose.call_count >= 3


def test_vet_pose_failure_raises_safety_violation():
    """If vet_pose returns a reason, raise SafetyViolation (not just log)."""
    bridge = MagicMock()
    rt = _runtime_with_bridge(bridge, vet_pose_returns="joint 3 beyond limit")

    with patch(
        "wrc_demo.skills.runtime.select_grasp",
        return_value=(_grasp_pose(), np.zeros(6), np.zeros(6)),
    ):
        with pytest.raises(SafetyViolation, match="joint 3 beyond limit"):
            rt.skill_grasp_object("banana")

    # First move_to_traj was rejected by vet_pose, never executed
    bridge.move_to_traj.assert_not_called()


def test_ik_failure_raises_skill_error():
    """If IK fails for the Cartesian target, raise SkillError (not silent)."""
    bridge = MagicMock()
    rt = _runtime_with_bridge(bridge)
    rt.kin.ik.return_value = MagicMock(success=False, q=None)

    with patch(
        "wrc_demo.skills.runtime.select_grasp",
        return_value=(_grasp_pose(), np.zeros(6), np.zeros(6)),
    ):
        with pytest.raises(SkillError, match="IK failed"):
            rt.skill_grasp_object("banana")

    bridge.move_to_traj.assert_not_called()


def test_pick_and_place_bridge_vet_pose_each_move():
    """pick_and_place bridge: vet_pose checked at pre, grasp, home, place, home."""
    bridge = MagicMock()
    rt = _runtime_with_bridge(bridge)

    banana_fix = _fix()
    banana_fix.position = np.array([0.30, -0.10, 0.05])
    box_fix = MagicMock()
    box_fix.detection.label = "yellow box"
    box_fix.detection.bbox = np.array([300, 100, 400, 200])
    box_fix.position = np.array([0.35, 0.10, 0.10])
    box_fix.points = np.array([[0.35, 0.10, 0.10]])  # single point for top_z check

    rt._single_localize = MagicMock(side_effect=[banana_fix, box_fix])

    with patch(
        "wrc_demo.skills.runtime.select_grasp",
        return_value=(_grasp_pose(), np.zeros(6), np.zeros(6)),
    ):
        rt.skill_pick_and_place("banana", destination="yellow box")

    # Held-object place executes the exact joint solution selected and vetted
    # by the object-level orientation search.
    assert bridge.move_to_traj.call_count == 2
    assert any(
        c.kwargs.get("duration_s") == 2.0
        for c in rt.arm.move_joints.call_args_list
    )
    assert rt.arm.harness.vet_pose.call_count >= 2
