"""Unit tests for WrcGripper.start() lifecycle.

Catches the 2026-08-10 real-rig bug where ``start()`` failed to start
the SDK control loop: ``self._arm.start_control_loop(...)`` and
``self._controller._running = True`` were missing, so the very next
gripper call raised ``RuntimeError: not started``.

These tests pin the four things ``start()`` must do:
1. arm group ``mode_mit`` + ``enable`` (or ``mode_pos_vel`` + ``enable``)
2. gripper group ``mode_mit`` + ``enable``
3. ``self._arm.start_control_loop(self._loop_cb)`` ← the bug
4. ``self._controller._running = True`` ← the bug
"""

from unittest.mock import MagicMock

import pytest

from wrc_demo.control.gripper import WrcGripper


def _make_arm_and_controller(*, control_mode: str = "mit"):
    """Build mocks that look like a real rebot_rs arm + RebotArmEndPose.

    The arm has groups ``arm`` and ``gripper``, each a MagicMock with
    the methods WrcGripper calls. The controller exposes
    ``_arm_control_mode``, ``_has_gripper``, ``_running``, and
    ``_loop_cb`` — the surface WrcGripper reads / mutates.
    """
    arm_group = MagicMock()
    arm_group.num_joints = 6
    arm_group._mit_kp = 50.0
    arm_group._mit_kd = 1.0

    gripper_group = MagicMock()
    gripper_group._jcfgs = [MagicMock(name="gripper_finger")]

    arm = MagicMock()
    arm.groups = {"arm": arm_group, "gripper": gripper_group}
    arm.has_gripper = True

    controller = MagicMock()
    controller._arm_control_mode = control_mode
    controller._has_gripper = True
    controller._running = False
    controller._loop_cb = MagicMock(name="controller._loop_cb")

    return arm, controller


def test_start_sets_running_and_starts_control_loop():
    """WrcGripper.start() must start the SDK control loop."""
    arm, controller = _make_arm_and_controller()
    gripper = WrcGripper(arm, controller)
    gripper.start()

    # The bug fix: this call was missing.
    arm.start_control_loop.assert_called_once()
    # The callback passed must be the WrcGripper wrapper, NOT the raw
    # controller._loop_cb — otherwise gripper_tick never runs. (We can't
    # use ``is`` because bound methods are created on demand per access.)
    args, _ = arm.start_control_loop.call_args
    assert args[0] is not controller._loop_cb, (
        "start_control_loop must receive WrcGripper's wrapper, not the "
        "controller's raw _loop_cb (else gripper_tick never runs)"
    )
    assert args[0].__self__ is gripper, (
        "callback must be a bound method of THIS WrcGripper instance"
    )
    # The bug fix: this assignment was missing.
    assert controller._running is True, (
        "controller._running must be True after start(); "
        "otherwise _ensure_running() raises 'not started' on the next call"
    )


def test_start_enables_arm_group_with_correct_mode():
    """Arm group must be put into MIT mode + enabled before the loop starts."""
    arm, controller = _make_arm_and_controller(control_mode="mit")
    gripper = WrcGripper(arm, controller)
    gripper.start()

    arm.groups["arm"].mode_mit.assert_called_once_with(
        kp=arm.groups["arm"]._mit_kp,
        kd=arm.groups["arm"]._mit_kd,
    )
    arm.groups["arm"].enable.assert_called_once()


def test_start_enables_gripper_group():
    """Gripper group must be in MIT mode + enabled for the state machine."""
    arm, controller = _make_arm_and_controller()
    gripper = WrcGripper(arm, controller)
    gripper.start()

    arm.groups["gripper"].mode_mit.assert_called_once()
    arm.groups["gripper"].enable.assert_called_once()


def test_start_marks_gripper_as_owned_by_driver():
    """controller._has_gripper = False so controller doesn't fight us."""
    arm, controller = _make_arm_and_controller()
    gripper = WrcGripper(arm, controller)
    gripper.start()

    assert controller._has_gripper is False, (
        "WrcGripper must take ownership of the gripper so the controller "
        "doesn't drive it with its own open/close methods"
    )


def test_start_skips_when_already_running():
    """Idempotent: if controller already running, start() is a no-op."""
    arm, controller = _make_arm_and_controller()
    controller._running = True  # simulate already-running state
    gripper = WrcGripper(arm, controller)
    gripper.start()

    arm.start_control_loop.assert_not_called()


def test_ensure_running_passes_after_start():
    """The end-to-end contract: after start(), _ensure_running() doesn't raise.

    This is the regression test for the 2026-08-10 real-rig bug.
    """
    arm, controller = _make_arm_and_controller()
    gripper = WrcGripper(arm, controller)
    gripper.start()

    # _ensure_running should NOT raise now.
    gripper._ensure_running()  # no exception = pass


def test_closing_width_limit_is_safe_stop_not_contact_success():
    arm, controller = _make_arm_and_controller()
    gripper = WrcGripper(arm, controller)
    min_pos = 0.5 * gripper._angle_open
    gripper._last_gripper_state = (min_pos, 0.2, 0.0)
    gripper._state = gripper._STATE_CLOSING
    gripper._start_pos = gripper._angle_open
    gripper._min_grasp_pos = min_pos
    gripper._hold_torque = 0.35 * gripper._close_sign
    gripper._grasp_result = None

    gripper.gripper_tick()

    assert gripper._state == gripper._STATE_HOLDING
    assert gripper._grasp_result is False
    assert gripper._grasp_stop_reason == "width_limit"
    assert gripper._target_pos == pytest.approx(min_pos)


def test_closing_uses_material_force_not_fixed_max_torque():
    arm, controller = _make_arm_and_controller()
    gripper = WrcGripper(arm, controller)
    gripper._last_gripper_state = (0.5 * gripper._angle_open, 1.0, 0.0)
    gripper._state = gripper._STATE_CLOSING
    gripper._start_pos = gripper._angle_open
    gripper._min_grasp_pos = None
    gripper._active_close_torque = 0.35 * gripper._close_sign

    gripper.gripper_tick()

    tau = arm.groups["gripper"].send_mit.call_args.kwargs["tau"][0]
    assert tau == pytest.approx(0.35 * gripper._close_sign)


def test_grasp_separates_motion_and_hold_torque():
    arm, controller = _make_arm_and_controller()
    controller._running = True
    gripper = WrcGripper(arm, controller)
    gripper._wait_gripper_state = MagicMock(side_effect=[
        (4.8, 0.0, 0.0), (2.5, 0.0, 0.0),
    ])

    def finish(_predicate, _timeout):
        assert gripper._active_close_torque == pytest.approx(
            0.70 * gripper._close_sign
        )
        assert gripper._hold_torque == pytest.approx(
            0.35 * gripper._close_sign
        )
        gripper._grasp_result = True
        gripper._grasp_stop_reason = "contact_stall"
        return True

    gripper._wait_until = finish
    assert gripper.grasp(force=0.35, close_force=0.70) is True
    telemetry = gripper.last_grasp_telemetry()
    assert telemetry["close_torque"] == pytest.approx(
        0.70 * gripper._close_sign
    )
    assert telemetry["hold_torque"] == pytest.approx(
        0.35 * gripper._close_sign
    )
