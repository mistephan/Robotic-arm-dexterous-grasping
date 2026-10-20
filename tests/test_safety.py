import numpy as np
import pytest

from wrc_demo.safety.harness import SafetyHarness, SafetyLimits
from wrc_demo.types import SafetyViolation


def limits(**kw):
    base = dict(
        workspace_min=np.array([0.10, -0.30, -0.01]),
        workspace_max=np.array([0.50, 0.30, 0.55]),
        table_z=0.0,
        table_clearance=0.02,
        max_joint_vel=1.0,
        watchdog_s=1e9,
    )
    base.update(kw)
    return SafetyLimits(**base)


class FakeKin:
    """Kinematics stub: TCP = q[:3], 6 joint origins along z of TCP.

    Matches the surface ``SafetyHarness`` and the wrc_demo Kinematics
    wrapper expose to skills: ``n_joints``, ``joint_limits``, ``fk``,
    ``link_positions``.
    """

    n_joints = 6
    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        # 6 link origins: 5 intermediate (all at z=0.2, safely above
        # the table) + 1 TCP at q[:3].
        z = max(q[2], 0.1)
        return np.array([
            [0, 0, 0.2],
            [0, 0, 0.2],
            [0, 0, 0.2],
            [0, 0, 0.2],
            [0, 0, 0.2],
            [q[0], q[1], z],
        ])


def test_velocity_cap():
    """Per-waypoint approve returns False on violation; SDK hook reads it."""
    h = SafetyHarness(limits(), FakeKin())
    # Start inside workspace so workspace check doesn't trip first.
    q = np.array([0.30, 0.0, 0.30, 0, 0, 0])
    # 0.01 rad step in 0.02s = 0.5 rad/s, under the 1.0 cap → ok
    assert h.approve(q, q + np.array([0.01, 0, 0, 0, 0, 0]), dt=0.02)
    # 0.05 rad step in 0.02s = 2.5 rad/s, over the cap → reject
    assert not h.approve(q, q + np.array([0.05, 0, 0, 0, 0, 0]), dt=0.02)
    assert "vel" in h.violations[-1].lower()


def test_estop_latches():
    h = SafetyHarness(limits(), FakeKin())
    h.estop("test")
    assert not h.approve(np.zeros(6), np.zeros(6), 0.02)
    h.reset_estop()
    # Use a pose inside workspace so the harness can approve it.
    inside = np.array([0.30, 0.0, 0.30, 0, 0, 0])
    assert h.approve(inside, inside, 0.02)


def test_watchdog():
    """begin_motion still raises (single-shot, runs on the calling thread)."""
    # Watchdog = 0 → begin_motion raises immediately
    h = SafetyHarness(limits(watchdog_s=0.0), FakeKin())
    with pytest.raises(SafetyViolation, match="watchdog"):
        h.begin_motion()
    # Watchdog = 60s, fresh heartbeat → approve ok (use inside-workspace pose)
    h2 = SafetyHarness(limits(watchdog_s=60.0), FakeKin())
    h2.heartbeat()
    inside = np.array([0.30, 0.0, 0.30, 0, 0, 0])
    assert h2.approve(inside, inside, 0.02)


def test_joint_limits():
    h = SafetyHarness(limits(joint_margin=0.0), FakeKin())
    bad = np.array([3.5, 0.2, 0.2, 0, 0, 0])
    assert not h.approve(bad, bad, 0.02)
    assert "out of limits" in h.violations[-1]


def test_workspace_aabb():
    h = SafetyHarness(limits(), FakeKin())
    inside = np.array([0.3, 0.0, 0.2, 0, 0, 0])
    assert h.approve(inside, inside, 0.02)
    outside = np.array([0.9, 0.0, 0.2, 0, 0, 0])
    assert not h.approve(inside, outside, dt=1e9)  # huge dt bypasses velocity
    assert "workspace" in h.violations[-1]


def test_table_clearance_and_grasp_exemption():
    h = SafetyHarness(limits(), FakeKin())
    above = np.array([0.3, 0.0, 0.20, 0, 0, 0])
    low = np.array([0.3, 0.0, 0.01, 0, 0, 0])
    # Descending below the clearance floor from above is rejected.
    assert not h.approve(above, low, dt=1e9)
    assert "table" in h.violations[-1]
    # ...unless the exemption cylinder over (0.3, 0) is open.
    h.allow_grasp_descent(np.array([0.3, 0.0]), radius=0.15, z_min=0.0)
    assert h.approve(above, low, dt=1e9)
    # Sideways below the floor, outside the cylinder: rejected.
    # Cylinder at (0.3, 0) radius 0.15 → x ∈ [0.15, 0.45]. 0.46 is outside.
    low_off = np.array([0.46, 0.0, 0.01, 0, 0, 0])
    assert not h.approve(low, low_off, dt=1e9)
    h.clear_grasp_exemption()
    assert not h.approve(above, low, dt=1e9)


def test_keep_out_zone():
    h = SafetyHarness(
        limits(keep_out=[(np.array([0.2, -0.1, 0.0]), np.array([0.3, 0.1, 0.5]))]),
        FakeKin(),
    )
    inside_ko = np.array([0.25, 0.0, 0.3, 0, 0, 0])
    assert not h.approve(inside_ko, inside_ko, dt=1e9)
    assert "keep-out" in h.violations[-1]