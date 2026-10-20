"""Tests for RebotGraspBridge pass-through to rebot_grasp motion + gripper APIs.

The bridge is the seam between wrc_demo's skill runtime and rebot_grasp's
controller + GraspDriver. These tests cover Slice 2 of the alignment work:
instantiation + direct delegation (no harness integration yet; that's B1).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from wrc_demo.skills.rebot_grasp_bridge import RebotGraspBridge


@pytest.fixture
def fake_controller() -> MagicMock:
    """Mock for reBotArm_control_py.controllers.RebotArmEndPose."""
    ctrl = MagicMock()
    ctrl.move_to_traj.return_value = True
    return ctrl


@pytest.fixture
def fake_driver() -> MagicMock:
    """Mock for rebot_grasp.drivers.robot.grasp_driver.GraspDriver."""
    drv = MagicMock()
    drv.grasp.return_value = True
    drv.open_gripper.return_value = None
    drv.release_gripper.return_value = None
    drv.MAX_DISTANCE_M = 0.09
    return drv


@pytest.fixture
def bridge(fake_controller, fake_driver) -> RebotGraspBridge:
    return RebotGraspBridge(fake_controller, fake_driver)


def test_bridge_instantiable(bridge):
    """Bridge wraps controller + driver without error."""
    assert bridge is not None


def test_move_to_traj_passes_through_to_controller(bridge, fake_controller):
    """move_to_traj forwards x,y,z,r,p,y,duration to controller and returns its bool."""
    ok = bridge.move_to_traj(0.25, 0.0, 0.35, 0.0, 1.2, 0.0, duration=2.0)
    assert ok is True
    fake_controller.move_to_traj.assert_called_once_with(
        0.25, 0.0, 0.35, 0.0, 1.2, 0.0, duration=2.0,
    )


def test_move_to_traj_returns_false_when_ik_fails(bridge, fake_controller):
    """If controller.move_to_traj returns False, bridge returns False (no IK)."""
    fake_controller.move_to_traj.return_value = False
    ok = bridge.move_to_traj(0.25, 0.0, 0.35, 0.0, 1.2, 0.0, duration=2.0)
    assert ok is False


def test_move_to_traj_waits_for_measured_joint_settle(fake_driver):
    class Controller:
        _n = 2

        def __init__(self):
            self._q_target = np.array([1.0, -0.5])
            self.samples = iter([
                np.array([0.8, -0.4]),
                np.array([0.99, -0.49]),
                np.array([1.0, -0.5]),
            ])
            self.rebotarm = self

        def move_to_traj(self, *args, **kwargs):
            return True

        def wait_motion_done(self, timeout):
            return True

        def get_state(self):
            return next(self.samples), np.zeros(2), np.zeros(2)

    bridge = RebotGraspBridge(
        Controller(), fake_driver, settle_tol_rad=0.02,
        settle_timeout_ms=3000, settle_samples=2,
    )
    assert bridge.move_to_traj(0, 0, 0, 0, 0, 0, duration=0.01) is True
    assert bridge.last_motion_telemetry()["status"] == "settled"


def test_move_to_traj_fails_when_feedback_never_settles(fake_driver):
    class Controller:
        _n = 1
        _q_target = np.array([1.0])

        def __init__(self):
            self.rebotarm = self

        def move_to_traj(self, *args, **kwargs):
            return True

        def wait_motion_done(self, timeout):
            return True

        def get_state(self):
            return np.array([0.0]), np.zeros(1), np.zeros(1)

    bridge = RebotGraspBridge(
        Controller(), fake_driver, settle_tol_rad=0.02,
        settle_timeout_ms=20, settle_samples=2,
    )
    assert bridge.move_to_traj(0, 0, 0, 0, 0, 0, duration=0.01) is False
    assert bridge.last_motion_telemetry()["status"] == "settle_timeout"


def test_open_gripper_calls_driver_without_timeout(bridge, fake_driver):
    """Default open passes no timeout arg (GraspDriver uses its own default)."""
    bridge.open_gripper()
    fake_driver.open_gripper.assert_called_once_with()


def test_open_gripper_with_distance_forwards(bridge, fake_driver):
    """Adaptive pregrasp opening reaches the native B601 driver."""
    bridge.open_gripper(distance_m=0.072)
    fake_driver.open_gripper.assert_called_once_with(distance_m=0.072)


def test_open_gripper_with_timeout_forwards(bridge, fake_driver):
    """timeout kwarg is forwarded to driver (set.py's release pattern uses 0.5)."""
    bridge.open_gripper(timeout=0.5)
    fake_driver.open_gripper.assert_called_once_with(timeout=0.5)


def test_grasp_returns_driver_bool(bridge, fake_driver):
    """bridge.grasp() returns the driver's bool (True = object held)."""
    fake_driver.grasp.return_value = True
    assert bridge.grasp() is True

    fake_driver.grasp.return_value = False
    assert bridge.grasp() is False


def test_grasp_with_force_overrides_default(bridge, fake_driver):
    """bridge.grasp(force=1.0) forwards force to driver (matches v5 effort=1.0)."""
    bridge.grasp(force=1.0)
    fake_driver.grasp.assert_called_once_with(force=1.0)


def test_grasp_forwards_minimum_safe_width(bridge, fake_driver):
    bridge.grasp(force=0.35, min_width_m=0.024)
    fake_driver.grasp.assert_called_once_with(
        force=0.35, min_width_m=0.024,
    )


def test_grasp_forwards_separate_motion_force(bridge, fake_driver):
    bridge.grasp(force=0.35, min_width_m=0.024, close_force=0.70)
    fake_driver.grasp.assert_called_once_with(
        force=0.35, min_width_m=0.024, close_force=0.70,
    )


def test_last_grasp_telemetry_is_forwarded(bridge, fake_driver):
    fake_driver.last_grasp_telemetry.return_value = {"stop_reason": "width_limit"}
    assert bridge.last_grasp_telemetry() == {"stop_reason": "width_limit"}


def test_release_gripper_calls_driver(bridge, fake_driver):
    """release_gripper delegates to driver (synchronous open + closed rest)."""
    bridge.release_gripper()
    fake_driver.release_gripper.assert_called_once_with()


def test_release_with_async_close_opens_then_closes_after_delay(
    bridge, fake_driver,
):
    """set.py place pattern: open immediately, close after 0.8s in background."""
    thread = bridge.release_with_async_close(timeout=0.5, delay_s=0.05)
    # Open should have happened synchronously
    fake_driver.open_gripper.assert_called_once_with(timeout=0.5)
    # Close target not yet set (still within delay window)
    assert not thread.is_alive() or not _has_set_close_target(fake_driver)
    # Wait for the thread to complete
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "async close thread did not finish"
    # Verify close target was set after the delay
    assert _has_set_close_target(fake_driver), (
        "WrcGripper.set_position_target(0.0) was not called after delay"
    )


def test_release_with_async_close_degrades_if_set_position_target_renamed(
    fake_controller, fake_driver,
):
    """If WrcGripper renames set_position_target, the bridge must not crash.

    The release succeeds (open), but the delayed close is a no-op. The
    caller still gets a thread to join so the API contract holds; the
    fall-back behavior is the gripper stays open until manually closed.

    (Per ADR-0005 the method is now public on WrcGripper; this test
    remains as defensive coverage in case a future refactor renames it.)
    """
    # Simulate rename: delete the public method
    del fake_driver.set_position_target

    bridge = RebotGraspBridge(fake_controller, fake_driver)
    thread = bridge.release_with_async_close(timeout=0.5, delay_s=0.01)

    # Open still works synchronously
    fake_driver.open_gripper.assert_called_once_with(timeout=0.5)
    # Thread completes without raising
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "thread should complete even if method renamed"


def _has_set_close_target(fake_gripper: MagicMock) -> bool:
    """True if any set_position_target call was made with 0.0 (closed rest).

    As of ADR-0005 the method is public (``set_position_target`` was
    renamed from the upstream ``_set_position_target``).
    """
    for call in fake_gripper.set_position_target.call_args_list:
        args, _ = call
        if args and args[0] == 0.0:
            return True
    return False


def test_release_with_async_close_returns_joinable_thread(bridge):
    """Caller can join the returned thread to wait for the close to complete."""
    thread = bridge.release_with_async_close(timeout=0.5, delay_s=0.01)
    thread.join(timeout=2.0)
    assert not thread.is_alive()


# ──────────────────────── B1: harness integration ────────────────────────


from wrc_demo.types import SafetyViolation  # noqa: E402


class _FakeHarness:
    """Minimal stand-in for SafetyHarness — records lifecycle calls and
    can be flipped into estop state to exercise the bridge's failure path.

    Mirrors the surface the bridge touches: ``begin_motion`` / ``end_motion`` /
    ``heartbeat`` / ``estopped``. Raises ``SafetyViolation`` on estop, matching
    the real harness contract.
    """

    def __init__(self, estopped: bool = False) -> None:
        self.estopped_flag = estopped
        self.begin_motion_calls = 0
        self.end_motion_calls = 0
        self.heartbeat_calls = 0

    def begin_motion(self) -> None:
        self.begin_motion_calls += 1
        if self.estopped_flag:
            raise SafetyViolation("e-stopped")

    def end_motion(self) -> None:
        self.end_motion_calls += 1

    def heartbeat(self) -> None:
        self.heartbeat_calls += 1


def test_bridge_without_harness_skips_lifecycle(fake_controller, fake_driver):
    """Mock mode: harness=None is valid; bridge must not call any harness API."""
    bridge = RebotGraspBridge(fake_controller, fake_driver, harness=None)
    bridge.move_to_traj(0.25, 0.0, 0.35, 0.0, 1.2, 0.0, duration=2.0)
    bridge.open_gripper()
    bridge.grasp(force=1.0)
    # no exception = pass; nothing else to assert


def test_move_to_traj_wraps_in_begin_end_motion(fake_controller, fake_driver):
    """With harness, move_to_traj calls begin_motion + end_motion."""
    harness = _FakeHarness()
    bridge = RebotGraspBridge(fake_controller, fake_driver, harness=harness)
    bridge.move_to_traj(0.25, 0.0, 0.35, 0.0, 1.2, 0.0, duration=2.0)
    assert harness.begin_motion_calls == 1
    assert harness.end_motion_calls == 1


def test_move_to_traj_propagates_safety_violation(fake_controller, fake_driver):
    """If harness.begin_motion raises SafetyViolation (e-stop), bridge re-raises."""
    harness = _FakeHarness(estopped=True)
    bridge = RebotGraspBridge(fake_controller, fake_driver, harness=harness)
    with pytest.raises(SafetyViolation):
        bridge.move_to_traj(0.25, 0.0, 0.35, 0.0, 1.2, 0.0, duration=2.0)
    # end_motion should NOT be called when begin_motion fails
    assert harness.end_motion_calls == 0
    # controller.move_to_traj must NOT be called when e-stopped
    fake_controller.move_to_traj.assert_not_called()


def test_gripper_methods_call_heartbeat_only(fake_controller, fake_driver):
    """open_gripper + grasp call heartbeat (watchdog) but NOT begin/end_motion."""
    harness = _FakeHarness()
    bridge = RebotGraspBridge(fake_controller, fake_driver, harness=harness)
    bridge.open_gripper()
    bridge.grasp(force=1.0)
    bridge.release_gripper()
    assert harness.heartbeat_calls == 3
    assert harness.begin_motion_calls == 0
    assert harness.end_motion_calls == 0
