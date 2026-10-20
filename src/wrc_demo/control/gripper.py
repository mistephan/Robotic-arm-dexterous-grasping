"""wrc_demo native gripper driver.

Ported from ``rebot_grasp/drivers/robot/grasp_driver.py`` so wrc_demo
has no runtime dependency on the rebot_grasp repo (see ADR-0005).

Behavioral parity with the upstream ``GraspDriver``:

- State machine: IDLE → POSITION → CLOSING → HOLDING (or back to POSITION
  if hard-stop is hit).
- ``open_gripper(distance_m, timeout)``: position-controlled open.
- ``grasp(force, timeout)``: torque-controlled close with stall
  detection (``|vel| < stall_vel``) and hard-stop detection.
  Returns ``True`` on contact, ``False`` on hard-stop / timeout.
- ``release_gripper(timeout)``: open + return to closed rest.

Differences from upstream ``GraspDriver``:

- Hardcoded to ``rebot_rs`` (counterclockwise=False). wrc_demo only
  supports the B601-RS; the dm-arm branch is dropped.
- ``selected_arm_config()`` / ``find_rebot_repo_root()`` /
  ``ensure_rebot_sdk_in_syspath()`` are removed; wrc_demo uses its
  vendored SDK directly via ``from reBotArm_control_py.kinematics``.
- ``set_position_target`` is **public** (not ``_set_position_target``)
  because the bridge module is the only caller and we own both sides
  of the API now.
- ``get_tcp_pose`` is dropped — wrc_demo computes TCP via its own
  ``Kinematics`` class.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

# Vendored SDK is on sys.path because wrc_demo/__init__.py injects
# wrc_demo/sdk/ at import time (see that file for the rationale).
from reBotArm_control_py.kinematics import (  # noqa: F401  (kept for parity with upstream)
    compute_fk,
    load_robot_model,
    pad_q_for_model,
)


# ── Constants (match upstream's rs defaults) ──────────────────────────────

GRIPPER_MAX_DISTANCE_M = 0.09

# Gripper MIT gains and limits for the reBot RS. These mirror
# rebot_grasp/config/default.yaml::gripper.rs and the upstream driver's
# rs branch. Tuned on real hardware; do NOT change without re-validating
# against the v5 baseline (ADR-0002 documents the hold-torque override).
_RS_ANGLE_OPEN = 5.0
_RS_COUNTERCLOCKWISE = False
_RS_TAU_MAX = 1.5
_RS_CLOSE_TORQUE = 1.5
_RS_DEFAULT_FORCE = 0.30

# State-machine constants.
# 2026-08-11: relaxed from 0.05 -> 0.12 rad/s. The MIT-mode ``vel`` reading
# on a thin/soft object (a 30 mm 3D-printed banana) oscillates at ~0.07-0.10
# rad/s even when the motor is clearly stalled against the object surface:
# kd_close=0.5 under-damps the position loop, and the encoder's velocity
# estimate has ~5 Hz noise. 0.05 caused the grasp() to walk all the way to
# hard-stop on every banana-grasp and report empty-grasp on an actually-
# held object. 0.12 still rejects true air-grasp (free-fall closing
# crosses 0.5 rad/s for ~50 ms before kd kicks in).
_STALL_VEL = 0.12              # |vel| < this ⇒ contact (rad/s)
_STARTUP_DIST = 0.30           # minimum travel before stall counts
_KP_MOVE = 5.0                 # position-control proportional gain
_KD_MOVE = 1.0                 # position-control derivative gain
_KD_CLOSE = 0.5                # closing-state derivative gain
_ARRIVE_TOL = 0.12             # |pos - target| < this ⇒ position reached


class WrcGripper:
    """wrc_demo native gripper driver for the rebot_rs B601.

    Wraps the SDK's gripper motor group + a small state machine that
    implements MIT-mode position / torque control plus stall detection.

    Lifecycle: ``WrcGripper(arm, controller)`` → ``start()`` (called by
    :func:`wrc_demo.control.make_arm` after the arm group is connected
    and the control loop is ready) → ``open_gripper / grasp /
    release_gripper`` for normal operation → ``disconnect()`` at demo
    exit (TODO: not yet wired into the demo's shutdown path; demo
    currently relies on the controller's ``end()``).
    """

    MAX_DISTANCE_M = GRIPPER_MAX_DISTANCE_M
    _STATE_IDLE = "idle"
    _STATE_POSITION = "position"
    _STATE_CLOSING = "closing"
    _STATE_HOLDING = "holding"

    def __init__(
        self,
        arm: Any,
        controller: Any,
        gripper_config: Optional[dict] = None,
    ) -> None:
        self._arm = arm
        self._controller = controller
        self._arm_group = arm.groups.get("arm")
        self._gripper_group = arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        if self._gripper_group is None or not arm.has_gripper:
            raise ValueError("Hardware config missing groups.gripper")
        gripper_jcfgs = getattr(self._gripper_group, "_jcfgs", [])
        if not gripper_jcfgs:
            raise ValueError("groups.gripper has no joints")
        self._gripper_name = gripper_jcfgs[0].name
        self._gripper_motor: Any = None

        # Merge defaults with optional caller-provided gripper_config.
        # wrc_demo callers usually pass None and use the per-call
        # ``force`` override (see ADR-0002).
        cfg = {
            "angle_open": _RS_ANGLE_OPEN,
            "counterclockwise": _RS_COUNTERCLOCKWISE,
            "tau_max": _RS_TAU_MAX,
            "close_torque": _RS_CLOSE_TORQUE,
            "default_force": _RS_DEFAULT_FORCE,
            **(gripper_config or {}),
        }
        motion_sign = 1.0 if bool(cfg.get("counterclockwise")) else -1.0
        self._angle_open = -motion_sign * abs(float(cfg["angle_open"]))
        self._tau_max = abs(float(cfg["tau_max"]))
        self._open_sign = 1.0 if self._angle_open >= 0.0 else -1.0
        self._close_sign = motion_sign
        self._close_torque = self._close_sign * abs(float(cfg["close_torque"]))
        self._active_close_torque = self._close_torque
        self._default_force = self._close_sign * abs(float(cfg["default_force"]))
        self._open_soft_limit = 0.98 * self._angle_open
        self._open_lo = min(self._open_soft_limit, 0.0)
        self._open_hi = max(self._open_soft_limit, 0.0)
        self._hard_stop_angle = self._open_sign * 0.05

        # State machine bookkeeping.
        self._state_lock = threading.Lock()
        self._state = self._STATE_IDLE
        self._target_pos = 0.0
        self._start_pos = 0.0
        self._contact_pos = 0.0
        self._hold_torque = self._default_force
        self._position_reached = True
        self._grasp_result: Optional[bool] = None
        self._last_gripper_state: Optional[tuple[float, float, float]] = None
        self._min_grasp_pos: Optional[float] = None
        self._grasp_stop_reason: Optional[str] = None
        self._last_grasp_telemetry: dict[str, Any] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the gripper state machine + SDK control loop.

        Called by :func:`wrc_demo.control.make_arm` after the arm is
        connected and the gripper group is wired up. We start the SDK
        control loop here (NOT ``sdk_ctrl.start()``, which would raise
        ``RuntimeError: control loop already running`` on the second
        ``start_control_loop`` call) — our :meth:`_loop_cb` wraps the
        SDK's controller callback and adds :meth:`gripper_tick`, so a
        single control loop drives both arm and gripper at 500 Hz.

        Lifecycle: if the controller is already running (e.g. a
        previous demo's loop is still alive — ``RuntimeError: control
        loop already running`` is the failure mode the user sees), we
        early-return. This matches upstream ``GraspDriver.start``.
        """
        if getattr(self._controller, "_running", False):
            return  # already running; SDK control loop drives our tick

        # Tell the controller we own the gripper so it doesn't try to
        # drive it with its own open/close methods.
        self._controller._has_gripper = False
        self._arm.connect()
        self._gripper_motor = self._gripper_group._mm[self._gripper_name]

        # Arm group setup. Upstream ``RebotArmEndPose.start()`` does
        # this; we duplicate here because we don't call that — its
        # ``start_control_loop`` would race with ours.
        if self._arm_group:
            if self._controller._arm_control_mode == "mit":
                self._arm_group.mode_mit(
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                )
            else:
                self._arm_group.mode_pos_vel()
            self._arm_group.enable()

        self._gripper_group.mode_mit()
        self._gripper_group.enable()
        self._prime_arm_target()
        self._prime_gripper_state()

        # Critical: start the SDK control loop with our wrapper callback
        # (controller._loop_cb + gripper_tick). Without this, ``_running``
        # stays False and ``_ensure_running()`` raises "not started" on
        # the first gripper operation. Real-rig caught this bug on
        # 2026-08-10 — the demo launched, the bridge tried to open the
        # gripper, and RuntimeError propagated up through 7 retries.
        self._arm.start_control_loop(self._loop_cb)
        self._controller._running = True

    def _loop_cb(self, r: Any, dt: float) -> None:
        """Wraps the SDK control loop callback with gripper_tick."""
        self._controller._loop_cb(r, dt)
        self.gripper_tick(dt)

    def _ensure_running(self) -> None:
        if not getattr(self._controller, "_running", False):
            raise RuntimeError(
                "WrcGripper is not started; call gripper.start() first"
            )

    # ── Motor primitives ──────────────────────────────────────────────

    def _send_gripper_mit(
        self,
        pos: float,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        pos_cmd = float(np.clip(pos, self._open_lo, self._open_hi))
        tau_cmd = float(np.clip(tau, -self._tau_max, self._tau_max))
        self._gripper_group.send_mit(
            np.array([pos_cmd], dtype=np.float64),
            vel=np.array([vel], dtype=np.float64),
            kp=np.array([kp], dtype=np.float64),
            kd=np.array([kd], dtype=np.float64),
            tau=np.array([tau_cmd], dtype=np.float64),
        )

    def _prime_arm_target(self) -> None:
        # Sync controller's q_target to current joint positions so the
        # first trajectory doesn't jerk.
        n = self._arm_group.num_joints
        q_now = self._arm.get_state()[0][:n]
        self._controller._q_target[:] = q_now
        self._controller._qd_target[:] = 0.0

    def _prime_gripper_state(self, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._gripper_group._request_feedback()
            state = self._read_gripper_state_cached()
            if state is not None:
                with self._state_lock:
                    self._target_pos = state[0]
                    self._contact_pos = state[0]
                    self._state = self._STATE_IDLE
                    self._position_reached = True
                    self._grasp_result = None
                return
            time.sleep(0.02)

    def _read_gripper_state_cached(
        self,
    ) -> Optional[tuple[float, float, float]]:
        if self._gripper_motor is None:
            return self._last_gripper_state
        st = self._gripper_motor.get_state()
        if st is None:
            return self._last_gripper_state
        self._last_gripper_state = (float(st.pos), float(st.vel), float(st.torq))
        return self._last_gripper_state

    def _wait_gripper_state(
        self, timeout: float = 1.0,
    ) -> tuple[float, float, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._read_gripper_state_cached()
            if state is not None:
                return state
            time.sleep(0.02)
        raise RuntimeError("Gripper feedback is not ready")

    def get_gripper_state(self) -> tuple[float, float, float]:
        """Return ``(position, velocity, torque)`` of the gripper motor.

        Raises ``RuntimeError`` if no feedback is available yet (motor
        not started or in fault state).
        """
        state = self._read_gripper_state_cached()
        if state is None:
            raise RuntimeError("Gripper feedback is not ready")
        return state

    # ── Public state machine entry points ─────────────────────────────

    def set_position_target(self, target: float) -> None:
        """Switch to POSITION state with the given target.

        Public (vs upstream's ``_set_position_target``) because the
        ``RebotGraspBridge`` is the only caller and we now own both
        sides of the contract.
        """
        with self._state_lock:
            self._target_pos = float(target)
            self._state = self._STATE_POSITION
            self._position_reached = False
            self._grasp_result = None

    def _wait_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def _position_done(self) -> bool:
        with self._state_lock:
            return self._position_reached

    def _grasp_done(self) -> bool:
        with self._state_lock:
            return self._grasp_result is not None

    def gripper_tick(self, dt: float = 0.0) -> None:
        """One iteration of the gripper state machine.

        Called from the SDK control loop (every ~2 ms at 500 Hz). Reads
        the current motor state, advances the state machine, and sends
        the next MIT command. Behaviour matches upstream ``GraspDriver``
        so the same physical tuning constants (kp_move, kd_close,
        stall_vel) apply.
        """
        del dt  # unused; state machine is event-driven
        pos_vel_torq = self._read_gripper_state_cached()
        with self._state_lock:
            state = self._state
            target = self._target_pos
            command: Optional[tuple[float, float, float, float, float]] = None

            if state == self._STATE_POSITION:
                command = (target, 0.0, _KP_MOVE, _KD_MOVE, 0.0)
                if pos_vel_torq is not None and abs(pos_vel_torq[0] - target) < _ARRIVE_TOL:
                    self._position_reached = True

            elif state == self._STATE_CLOSING:
                command = (0.0, 0.0, 0.0, _KD_CLOSE, self._active_close_torque)
                if pos_vel_torq is not None:
                    pos, vel, _ = pos_vel_torq
                    self._contact_pos = pos
                    moved = abs(pos - self._start_pos) >= _STARTUP_DIST
                    reached_width_limit = (
                        self._min_grasp_pos is not None
                        and self._open_sign * pos
                        <= self._open_sign * self._min_grasp_pos
                    )
                    at_hard_stop = (
                        self._open_sign * pos
                        < self._open_sign * self._hard_stop_angle
                    )
                    if moved and reached_width_limit:
                        # Safety fallback: never squeeze past the minimum jaw
                        # opening derived from the observed object width.
                        self._target_pos = self._min_grasp_pos
                        self._state = self._STATE_HOLDING
                        self._grasp_stop_reason = "width_limit"
                        # Reaching a software width floor proves only that
                        # squeezing stopped safely; it does NOT prove contact.
                        self._grasp_result = False
                        command = (
                            self._target_pos, 0.0, _KP_MOVE, _KD_MOVE,
                            self._hold_torque,
                        )
                    elif moved and at_hard_stop:
                        # Empty grasp — went all the way to hard stop.
                        self._target_pos = 0.0
                        self._state = self._STATE_POSITION
                        self._position_reached = False
                        self._grasp_stop_reason = "hard_stop"
                        self._grasp_result = False
                        command = (0.0, 0.0, _KP_MOVE, _KD_MOVE, 0.0)
                    elif moved and abs(vel) < _STALL_VEL:
                        # Object contact — stall detected.
                        self._target_pos = pos
                        self._state = self._STATE_HOLDING
                        self._grasp_stop_reason = "contact_stall"
                        self._grasp_result = True
                        command = (pos, 0.0, _KP_MOVE, _KD_MOVE, self._hold_torque)

            elif state == self._STATE_HOLDING:
                command = (
                    self._target_pos, 0.0, _KP_MOVE, _KD_MOVE, self._hold_torque,
                )

        if command is not None:
            pos, vel, kp, kd, tau = command
            self._send_gripper_mit(pos, vel=vel, kp=kp, kd=kd, tau=tau)

    # ── High-level operations ─────────────────────────────────────────

    def open_gripper(
        self,
        distance_m: float = GRIPPER_MAX_DISTANCE_M,
        timeout: float = 3.0,
    ) -> None:
        """Open to ``distance_m`` (clipped to ``MAX_DISTANCE_M``).

        Blocks up to ``timeout`` seconds waiting for the gripper to
        settle. Raises ``RuntimeError`` if the driver isn't started.
        """
        self._ensure_running()
        d = float(np.clip(distance_m, 0.0, self.MAX_DISTANCE_M))
        raw_target = (d / self.MAX_DISTANCE_M) * self._angle_open
        target = float(np.clip(raw_target, self._open_lo, self._open_hi))
        self.set_position_target(target)
        if not self._wait_until(self._position_done, timeout):
            actual = self._read_gripper_state_cached()
            detail = (
                "feedback unavailable"
                if actual is None
                else (
                    f"actual={actual[0]:.3f} rad, velocity={actual[1]:.3f} rad/s, "
                    f"torque={actual[2]:.3f}"
                )
            )
            raise RuntimeError(
                f"Gripper did not reach open target {target:.3f} rad "
                f"within {timeout:.1f}s ({detail})"
            )

    def close_gripper(self, timeout: float = 3.0) -> None:
        """Close to the zero-position rest without grasp detection.

        This is the deterministic counterpart of :meth:`open_gripper` for
        operator bring-up commands. Object-aware grasping continues to use
        :meth:`grasp`, which applies torque and contact detection.
        """
        self._ensure_running()
        self.set_position_target(0.0)
        if not self._wait_until(self._position_done, timeout):
            raise RuntimeError(
                f"Gripper did not reach closed target within {timeout:.1f}s"
            )

    def grasp(
        self,
        force: Optional[float] = None,
        timeout: float = 5.0,
        min_width_m: Optional[float] = None,
        close_force: Optional[float] = None,
    ) -> bool:
        """Close with force control and detect contact.

        ``force`` is the hold-torque magnitude (clipped to
        ``[0.05, tau_max]``); ``None`` uses the default from
        ``gripper_config``. The upstream ``set.py`` defaults to 0.30;
        wrc_demo overrides to 1.0 per ADR-0002.

        Returns:
            ``True`` only when contact stall was detected. Reaching the
            configured minimum width is a safe stop, not proof of contact,
            and therefore returns ``False`` like hard-stop or timeout.
        """
        self._ensure_running()
        start_pos, start_vel, start_torque = self._wait_gripper_state()
        min_width = (
            None if min_width_m is None
            else float(np.clip(min_width_m, 0.0, self.MAX_DISTANCE_M))
        )
        min_grasp_pos = (
            None if min_width is None
            else float(np.clip(
                min_width * self._angle_open * (1.0 / self.MAX_DISTANCE_M),
                self._open_lo, self._open_hi,
            ))
        )
        hold_torque = self._close_sign * float(
            np.clip(
                abs(force if force is not None else self._default_force),
                0.05, self._tau_max,
            )
        )
        # Static friction in the RS rack-pinion needs more torque to start
        # closing from a wide-open pose than soft produce needs once held.
        # Keep these two physical jobs separate: close_force moves the jaws;
        # force remains the material-aware contact/holding torque.
        close_torque = self._close_sign * float(
            np.clip(
                abs(close_force if close_force is not None else hold_torque),
                0.05, self._tau_max,
            )
        )
        with self._state_lock:
            self._start_pos = start_pos
            self._contact_pos = start_pos
            self._target_pos = 0.0
            self._min_grasp_pos = min_grasp_pos
            self._grasp_stop_reason = None
            # Travel torque overcomes mechanism friction; hold torque remains
            # material-aware after contact. The width floor still prevents
            # squeezing past the geometry-derived safe opening.
            self._active_close_torque = close_torque
            self._hold_torque = hold_torque
            self._state = self._STATE_CLOSING
            self._position_reached = False
            self._grasp_result = None

        if not self._wait_until(self._grasp_done, timeout):
            with self._state_lock:
                if self._grasp_result is None:
                    # Timed out without stall / hard-stop → assume
                    # empty grasp so the caller doesn't loop forever.
                    self._target_pos = self._contact_pos
                    self._state = self._STATE_POSITION
                    self._position_reached = False
                    self._grasp_stop_reason = "timeout"
                    self._grasp_result = False

        with self._state_lock:
            result = bool(self._grasp_result)
            reason = self._grasp_stop_reason or "unknown"
            target_pos = float(self._target_pos)
        end_pos, end_vel, end_torque = self._wait_gripper_state()
        self._last_grasp_telemetry = {
            "success": result,
            "stop_reason": reason,
            "requested_min_width_m": min_width,
            "start": {"position": start_pos, "velocity": start_vel, "torque": start_torque},
            "end": {"position": end_pos, "velocity": end_vel, "torque": end_torque},
            "target_position": target_pos,
            "close_torque": self._active_close_torque,
            "hold_torque": hold_torque,
        }
        return result

    def last_grasp_telemetry(self) -> dict[str, Any]:
        """Return a copy of the most recent grasp stop/result telemetry."""
        return dict(self._last_grasp_telemetry)

    def release_gripper(self, timeout: float = 4.0) -> None:
        """Open + return to closed rest (idle position 0)."""
        self._ensure_running()
        self.open_gripper(timeout=min(2.0, timeout))
        self.set_position_target(0.0)
        self._wait_until(self._position_done, timeout)
