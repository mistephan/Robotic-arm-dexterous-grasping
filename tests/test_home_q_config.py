"""Regression test: home_q in configs/arms/*.yaml must satisfy joint
limits AND place TCP at the documented (0.25, 0, 0.35) pose.

Caught the 2026-08-10 real-rig failure where
``configs/arms/rebot_rs.yaml::home_q`` violated joint limits (joints 2 &
3 were positive instead of negative) and FK'd to (0.314, 0, 0.071)
instead of (0.25, 0, 0.35). Every ``vet_pose`` call rejected moves
seeded from home with the broken config, so the grasp path was
permanently blocked.

These tests are run as part of the live Phase 6 preflight — a wrong
``home_q`` is silent at import time (no syntax / type error) and only
surfaces on the rig, so an explicit config check is worth keeping.
"""

from pathlib import Path

import numpy as np
import pytest

from wrc_demo.safety.harness import Kinematics, SafetyHarness, SafetyLimits


REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"


def _make_kin_and_harness():
    kin = Kinematics(str(URDF), ee_frame="gripper_end")
    limits = SafetyLimits(
        workspace_min=np.array([-1, -1, -0.5]),
        workspace_max=np.array([1, 1, 1]),
        table_z=0.0,
        table_clearance=0.02,
        joint_margin=0.02,
        max_joint_vel=1.2,
    )
    return kin, SafetyHarness(limits, kin)


def _load_yaml_home_q(path: Path) -> list[float]:
    """Pull ``home_q: [...]`` out of the YAML without importing yaml."""
    text = path.read_text()
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("home_q:"):
            # `home_q: [0.0, -0.1684, ...]`
            bracket = stripped.split(":", 1)[1].strip()
            return [float(x.strip().rstrip(",")) for x in bracket.strip("[]").split(",") if x.strip()]
    raise ValueError(f"home_q not found in {path}")


@pytest.mark.parametrize("yaml_path", [
    REPO / "configs/arms/rebot_rs.yaml",
])
def test_home_q_fk_matches_documented_pose(yaml_path):
    """FK(home_q) must land TCP at (0.25, 0, 0.35) with identity rotation.

    This is the position the YAML comment claims. If FK disagrees,
    every vet_pose check is checking against a different baseline and
    the IK basin mismatch from CLAUDE.md re-appears.
    """
    kin, _ = _make_kin_and_harness()
    home_q = np.array(_load_yaml_home_q(yaml_path))
    T = kin.fk(home_q)
    xyz = T[:3, 3]
    assert np.allclose(xyz, [0.25, 0.0, 0.35], atol=1e-3), (
        f"FK(home_q) = {xyz}, expected [0.25, 0, 0.35]. "
        f"YAML comment is wrong or q signs are flipped (asset vs local "
        f"frame); see CLAUDE.md 'Joint mirror' section. home_q = {home_q.tolist()}"
    )


@pytest.mark.parametrize("yaml_path", [
    REPO / "configs/arms/rebot_rs.yaml",
])
def test_home_q_within_joint_limits(yaml_path):
    """home_q must be inside the URDF joint limits (without margin).

    Real-rig failure mode: vet_pose rejected home_q[1] = +0.168 because
    joint 2's range is [-3.14, 0] (negative-only). The arm booted,
    move_home succeeded (no vet on joint-only motion), but every
    subsequent grasp_object call was rejected by vet_pose before the
    arm ever moved.
    """
    kin, _ = _make_kin_and_harness()
    home_q = np.array(_load_yaml_home_q(yaml_path))
    lo, hi = kin.joint_limits
    for i, (q, l, h) in enumerate(zip(home_q, lo, hi)):
        assert l <= q <= h, (
            f"home_q joint {i+1} = {q:.4f} outside URDF limit [{l:.4f}, {h:.4f}]. "
            f"home_q = {home_q.tolist()}"
        )


@pytest.mark.parametrize("yaml_path", [
    REPO / "configs/arms/rebot_rs.yaml",
])
def test_home_q_passes_vet_pose(yaml_path):
    """vet_pose(home_q) must return None.

    The end-to-end contract: after move_home, vet_pose on the home pose
    itself is a no-op. If it returns a reason, the safety harness is
    rejecting home as invalid — which means EVERY motion seeded from
    home will also be rejected (or have to clear it on its own).
    """
    kin, harness = _make_kin_and_harness()
    home_q = np.array(_load_yaml_home_q(yaml_path))
    reason = harness.vet_pose(home_q)
    assert reason is None, (
        f"vet_pose(home_q) returned {reason!r}; home_q should be a valid pose. "
        f"home_q = {home_q.tolist()}"
    )
