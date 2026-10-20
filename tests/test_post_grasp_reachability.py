"""Post-grasp reachability regressions (logs 20260922_124006 / _124241).

Two confirmed defects from the 2026-09-22 live runs, both reproduced
offline against the real O6 URDF + harness before fixing:

1. **Post-grasp freeze (run 124006, graspgenx backend).** After a low
   tabletop grasp the TCP legitimately sits below the table-clearance
   floor (banana grasp target z=0.013, floor = table_z + 0.02). The
   held-carry/place orientation search then statically vetted
   ``q_a = q_now`` -- a pose the arm already occupies -- with no
   exemption and no escape rules, so all 52 IK-solvable lift candidates
   were rejected at the first check → ``held_carry_unreachable`` → the
   arm froze holding the object until the operator dropped it. Fix:
   ``_vet_place_segment(vet_start=False)`` for the measured-pose leg
   (mirrors ``_make_hook``'s first-waypoint skip).

2. **No-executable-grasp at lateral positions (run 124241, obb
   backend).** The geometric planners emit one closing-axis yaw per
   candidate. For the banana at y≈+0.23 the wrist-aligned IK branch
   hits the joint4 elbow limit and the surviving branch needs
   |q6|≈2.93, past the ``max_wrist_delta_rad`` soft cap (one rejection
   was by 0.003 rad). Fix: ``max_wrist_delta_rad: 3.05`` in demo.yaml
   plus ``_yaw_shimmed`` fallback candidates (a -15° yaw shim solves at
   |q6|≈2.55; a power grasp of produce tolerates ±30° of jaw yaw).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from wrc_demo.config import load_demo_config
from wrc_demo.skills.runtime import SkillRuntime, _yaw_shimmed
from wrc_demo.safety.harness import Kinematics, SafetyHarness, SafetyLimits
from wrc_demo.types import Grasp, make_transform

from conftest import REPO, needs_pin

O6_URDF = (
    REPO / "assets/urdf/rebot_b601_linkerhand_o6/rebot_b601_linkerhand_o6.urdf"
)

# Measured joints at the banana grasp settle (trace 124006 line 11,
# bridge_move_verify/grasp actual_q): TCP [0.420, 0.204, 0.0175], i.e.
# 2.5 mm BELOW the table-clearance floor — exactly the trap of defect 1.
Q_BANANA_GRASP = np.array([
    -0.6837929487223394, 2.4030544757843018, 2.0352704524993896,
    -1.1110193729400635, -0.8364285230636597, 0.4790005087852478,
])
LIFT_TARGET = np.array([0.4223, 0.2049, 0.16])  # held_carry_z_m


def _o6_harness() -> tuple[Kinematics, SafetyHarness]:
    kin = Kinematics(str(O6_URDF), ee_frame="o6_grasp_tcp", arm_joint_count=6)
    limits = SafetyLimits(
        workspace_min=np.array([0.10, -0.15, -0.01]),
        workspace_max=np.array([0.50, 0.30, 0.55]),
        table_z=0.0, table_clearance=0.02,
    )
    return kin, SafetyHarness(limits, kin)


def _runtime(kin, harness) -> SkillRuntime:
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    # Keep the real-IK candidate count small: this test cares about the
    # vet mechanics, not the full 126-candidate sweep (the sweep was
    # verified offline: 51 IK-OK, 0 vet-OK before the fix, 48 after).
    cfg._data["grasp"].update({
        "place_yaw_search_deg": [0, -30],
        "place_tilt_search_deg": [0, -10],
        "place_tilt_axis_deg": [0, 90],
        "place_ik_retries": 1,
    })
    arm = MagicMock()
    arm.harness = harness
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


# ── defect 1: lift vetoed at the pose the arm already occupies ────────


@needs_pin
def test_low_grasp_pose_does_not_veto_the_lift():
    """Exact repro of run 124006: from the below-floor banana grasp pose,
    the lift search must find a vetted candidate instead of rejecting
    every IK solution at q_a."""
    kin, harness = _o6_harness()
    rt = _runtime(kin, harness)
    R_level = np.asarray(kin.fk(Q_BANANA_GRASP), dtype=float)[:3, :3].copy()
    rt._held_R_level = R_level

    # Document the trap: the occupied pose fails the STATIC vet (that is
    # why vet_start=False is required for the measured-pose leg).
    assert harness.vet_pose(Q_BANANA_GRASP) is not None

    np.random.seed(20260922)  # solve_ik_with_retry restarts are random
    found = rt._search_held_place_rotation(
        LIFT_TARGET, LIFT_TARGET, Q_BANANA_GRASP, vet_from_current=True,
    )
    assert found is not None, (
        "lift from a low grasp must be reachable — if this fails, the "
        "vet_start skip in _vet_place_segment regressed"
    )
    events = {
        c.args[0]: c.args[1]
        for c in rt.trace.record_event.call_args_list
        if c.args and c.args[0] == "place_orientation_search"
    }
    assert events, "search must trace place_orientation_search"
    payload = events["place_orientation_search"]
    assert payload["n_vet_ok"] >= 1
    # Observability: the event now carries a reject-reason histogram.
    assert isinstance(payload["reject_reasons"], dict)


def test_vet_start_flag_skips_only_the_start_pose_check():
    """vet_start=False must skip the static vet of q_a but still vet the
    sampled segment — the flag gates exactly one call."""
    kin = MagicMock()
    kin.fk.return_value = make_transform(np.eye(3), np.zeros(3))
    arm = MagicMock()
    calls = {"n": 0}

    def vet_pose(q, **kwargs):
        calls["n"] += 1
        return "start reason" if calls["n"] == 1 else None

    arm.harness.vet_pose.side_effect = vet_pose
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    rt = SkillRuntime(
        camera=MagicMock(), depth_provider=MagicMock(), detector=MagicMock(),
        extrinsics=MagicMock(), kin=kin, safe_arm=arm, memory=MagicMock(),
        beliefs=MagicMock(), trace=MagicMock(), cfg=cfg,
    )
    q_a, q_b = np.zeros(6), np.ones(6)
    assert rt._vet_place_segment(q_a, q_b, vet_start=True) == (
        "approach unsafe: start reason"
    )
    assert rt._vet_place_segment(q_a, q_b, vet_start=False) is None


# ── defect 2: yaw-shim fallback candidates ────────────────────────────


def _topdown_grasp(jaw_yaw_deg: float = 30.0) -> Grasp:
    yaw = np.radians(jaw_yaw_deg)
    jaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    approach = np.array([0.0, 0.0, -1.0])
    R = np.column_stack([approach, jaw, np.cross(approach, jaw)])
    return Grasp(
        position=np.array([0.40, 0.22, 0.013]),
        rotation=R,
        width_m=0.05,
        approach=approach,
        quality=0.9,
        label="banana",
    )


def test_yaw_shim_rotates_jaw_axis_only():
    """The shim rotates the closing-axis yaw about the approach axis,
    leaves approach/position alone, dings quality, and does not mutate
    the parent."""
    parent = _topdown_grasp()
    R_parent = np.asarray(parent.rotation, dtype=float).copy()
    shim = _yaw_shimmed(parent, -15.0)
    R = np.asarray(shim.rotation, dtype=float)

    assert np.allclose(R[:, 0], R_parent[:, 0], atol=1e-12)  # approach kept
    cos_angle = abs(float(R[:, 1] @ R_parent[:, 1]))
    assert cos_angle == pytest.approx(np.cos(np.radians(15.0)), abs=1e-9)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)       # still a rotation
    assert np.allclose(shim.position, parent.position)
    assert shim.quality == pytest.approx(0.9 * (1.0 - 0.05 * 0.5))
    assert np.allclose(parent.rotation, R_parent)           # parent untouched
    # Shim stays inside the elongated-axis gate (alignment >= 0.75 means
    # cos(41 deg); ±30 deg shims give cos(30 deg) = 0.866).
    shim30 = _yaw_shimmed(parent, 30.0)
    assert abs(float(np.asarray(shim30.rotation)[:, 1] @ R_parent[:, 1])) >= 0.75


def test_yaw_shims_sort_after_their_parent():
    """Quality penalty puts shims immediately after the exact-yaw parent
    in select_grasp's ranked order, so they are only IK-solved when the
    parent was rejected (no cost in the common case)."""
    parent = _topdown_grasp()
    shims = [_yaw_shimmed(parent, d) for d in (15.0, -15.0, 30.0, -30.0)]
    ranked = sorted([*shims, parent], key=lambda g: -g.quality)
    assert ranked[0] is parent
    assert all(s.quality < parent.quality for s in shims)
