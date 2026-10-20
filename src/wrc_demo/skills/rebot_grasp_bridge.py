"""Bridge from wrc_demo skills to rebot_grasp motion + gripper APIs.

Maps wrc_demo's high-level skill calls (``skill_grasp_object``,
``skill_pick_and_place``) to rebot_grasp's ``RebotArmEndPose.move_to_traj``
and ``GraspDriver`` primitives so wrc_demo's trajectory matches
``rebot_grasp/scripts/main.py`` and ``set.py`` behavior.

Harness integration (pre/post ``harness.approve`` around ``move_to_traj``)
is layered on top in task B1; this slice is pure pass-through.

Reference (set.py lines 80-119 + 138-153):

    print("[Grasp] Open gripper")
    grasp_driver.open_gripper()

    print("[Grasp] Move to pregrasp")
    controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0)

    print("[Grasp] Move to grasp")
    controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5)

    print("[Grasp] Closing")
    ok = grasp_driver.grasp()

    print("[Grasp] Return ready")
    _move_ready(controller, ready_cfg)

For ``place``:

    grasp_driver.open_gripper(timeout=0.5)
    # background thread sleeps 0.8s then closes to idle (d=0)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from reBotArm_control_py.controllers import RebotArmEndPose
    from rebot_grasp.drivers.robot.grasp_driver import GraspDriver
    from ..safety.harness import SafetyHarness


class RebotGraspBridge:
    """Thin facade exposing rebot_grasp API to wrc_demo skills.

    The bridge is constructed once at startup (in :func:`wrc_demo.control.make_arm`
    for the ``rebot_rs`` backend) and stashed on the wrapped ``SafeArm``. The
    skill runtime reads it as ``self.bridge``; when ``None`` (mock or non-RS
    backend) skills fall back to the legacy path.

    All methods are synchronous; ``release_with_async_close`` is the only one
    that schedules background work, and it returns the thread for the caller
    to ``.join()`` after the next ``move_to_traj``.

    Optional ``harness`` (:class:`wrc_demo.safety.harness.SafetyHarness`)
    integration adds:

    - ``move_to_traj`` is wrapped in ``begin_motion()`` / ``end_motion()``,
      so e-stop and the perception watchdog are checked at the boundary.
      Per-waypoint validation is delegated to the SDK controller (rebot_grasp
      doesn't expose joint-angle hooks; ``approve(q_prev, q_next)`` is only
      invoked from the wrc_demo SafeArm path).
    - Gripper methods call ``heartbeat()`` to refresh the watchdog.
    - ``SafetyViolation`` raised by ``begin_motion`` propagates; the underlying
      controller is not called.
    """

    def __init__(
        self,
        controller: "RebotArmEndPose",
        grasp_driver: "GraspDriver",
        harness: Optional["SafetyHarness"] = None,
        settle_tol_rad: float = 0.02,
        settle_timeout_ms: int = 3000,
        settle_samples: int = 3,
    ) -> None:
        self._controller = controller
        self._grasp_driver = grasp_driver
        self._harness = harness
        self._settle_tol_rad = max(float(settle_tol_rad), 0.0)
        self._settle_timeout_s = max(float(settle_timeout_ms) / 1000.0, 0.0)
        self._settle_samples = max(int(settle_samples), 1)
        self._last_motion_telemetry: dict = {}

    # ──────────────────────────── motion ────────────────────────────

    def move_to_traj(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        duration: float,
    ) -> bool:
        """IK + min-jerk trajectory through ``RebotArmEndPose.move_to_traj``.

        Blocks until the motion thread completes (or times out), matching
        ``rebot_grasp/scripts/main.py``'s ``_wait_motion`` pattern.
        Without this wait, consecutive calls (pregrasp → grasp) would
        override each other — the arm never reaches pregrasp before being
        redirected to the grasp target, causing table collisions and
        empty grasps.

        Returns ``True`` if motion succeeded, ``False`` if IK failed or
        the motion timed out. The caller (``skill_grasp_object`` /
        ``skill_pick_and_place``) raises :class:`SkillError` on
        ``False`` for pregrasp and grasp; place ignores ``False`` (no
        second chance to reposition a held object).

        With ``harness`` set, calls ``begin_motion()`` first (e-stop +
        watchdog check) and ``end_motion()`` after; ``SafetyViolation`` from
        ``begin_motion`` propagates without invoking the controller.
        """
        if self._harness is not None:
            self._harness.begin_motion()
        try:
            ok = bool(
                self._controller.move_to_traj(
                    x, y, z, roll, pitch, yaw, duration=duration,
                )
            )
            if ok:
                ok = self._wait_motion(duration)
        finally:
            if self._harness is not None:
                self._harness.end_motion()
        return ok

    def last_motion_telemetry(self) -> dict:
        return dict(self._last_motion_telemetry)

    def _wait_motion(self, duration: float, extra: float = 0.6) -> bool:
        """Block until trajectory emission and physical settling finish.

        The SDK completion flag only means the final target was published;
        measured joints may still be catching up.
        """
        self._last_motion_telemetry = {"status": "waiting_for_trajectory"}
        wait_fn = getattr(self._controller, "wait_motion_done", None)
        if callable(wait_fn):
            if not bool(wait_fn(timeout=duration + extra + 2.0)):
                self._last_motion_telemetry = {"status": "trajectory_timeout"}
                return False
        else:
            thread = getattr(self._controller, "_send_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=duration + extra + 2.0)
                if thread.is_alive():
                    self._last_motion_telemetry = {"status": "trajectory_timeout"}
                    return False
            else:
                time.sleep(duration + extra)
        return self._wait_feedback_settled()

    def _wait_feedback_settled(self) -> bool:
        """Require several live joint samples near the final SDK target."""
        q_target = getattr(self._controller, "_q_target", None)
        rebotarm = getattr(self._controller, "rebotarm", None)
        get_state = getattr(rebotarm, "get_state", None)
        if not isinstance(q_target, np.ndarray) or not callable(get_state):
            self._last_motion_telemetry = {"status": "feedback_unavailable"}
            return True

        target = np.asarray(q_target, dtype=float).reshape(-1).copy()
        n = int(getattr(self._controller, "_n", target.size))
        target = target[:n]
        t0 = time.monotonic()
        deadline = t0 + self._settle_timeout_s
        last_q = None
        last_err = None
        stable = 0
        while time.monotonic() <= deadline:
            try:
                q_now = np.asarray(get_state()[0], dtype=float).reshape(-1)[:n]
            except Exception as exc:
                self._last_motion_telemetry = {
                    "status": "feedback_error", "error": str(exc),
                }
                return False
            if q_now.size == target.size and np.all(np.isfinite(q_now)):
                last_q = q_now.copy()
                err = float(np.max(np.abs(q_now - target)))
                last_err = err
                stable = stable + 1 if err <= self._settle_tol_rad else 0
                if stable >= self._settle_samples:
                    self._last_motion_telemetry = {
                        "status": "settled",
                        "target_q": target.tolist(),
                        "actual_q": q_now.tolist(),
                        "max_joint_error_rad": err,
                        "wait_ms": round((time.monotonic() - t0) * 1000.0, 1),
                    }
                    return True
            time.sleep(0.05)
        self._last_motion_telemetry = {
            "status": "settle_timeout",
            "target_q": target.tolist(),
            "actual_q": last_q.tolist() if last_q is not None else None,
            "max_joint_error_rad": last_err,
            "wait_ms": round((time.monotonic() - t0) * 1000.0, 1),
            "timeout_ms": round(self._settle_timeout_s * 1000.0, 1),
        }
        return False

    # ──────────────────────────── gripper ───────────────────────────

    def open_gripper(
        self,
        timeout: Optional[float] = None,
        distance_m: Optional[float] = None,
    ) -> None:
        """Open to ``distance_m`` or ``GraspDriver.MAX_DISTANCE_M``.

        ``timeout`` is forwarded only when supplied, so :class:`GraspDriver`'s
        own default (``3.0 s``) is used otherwise. ``set.py`` calls with
        ``timeout=0.5`` on place release; pregrasp uses the default.

        With ``harness`` set, refreshes the perception watchdog first.
        """
        if self._harness is not None:
            self._harness.heartbeat()
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if distance_m is not None:
            kwargs["distance_m"] = distance_m
        self._grasp_driver.open_gripper(**kwargs)

    def close_gripper(self, timeout: Optional[float] = None) -> None:
        """Close to the gripper's zero-position rest and wait for arrival."""
        if self._harness is not None:
            self._harness.heartbeat()
        if timeout is not None:
            self._grasp_driver.close_gripper(timeout=timeout)
        else:
            self._grasp_driver.close_gripper()

    def grasp(
        self,
        force: Optional[float] = None,
        min_width_m: Optional[float] = None,
        close_force: Optional[float] = None,
    ) -> bool:
        """Two-force close with force control.

        Stops on object contact (``|vel| < stall_vel``) or hard stop; returns
        ``True`` if a held object was detected. ``force`` is the hold-torque
        magnitude. wrc_demo passes the material profile (for example 0.35
        for slippery produce) as the hold torque; ``close_force`` may be
        higher to overcome rack-pinion static friction before contact.

        With ``harness`` set, refreshes the perception watchdog first.
        """
        if self._harness is not None:
            self._harness.heartbeat()
        kwargs = {}
        if force is not None:
            kwargs["force"] = force
        if min_width_m is not None:
            kwargs["min_width_m"] = min_width_m
        if close_force is not None:
            kwargs["close_force"] = close_force
        return bool(self._grasp_driver.grasp(**kwargs))

    def last_grasp_telemetry(self) -> dict:
        getter = getattr(self._grasp_driver, "last_grasp_telemetry", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter())
        except Exception:
            return {}

    def release_gripper(self) -> None:
        """Open + return to closed rest synchronously (used at demo exit).

        With ``harness`` set, refreshes the perception watchdog first.
        """
        if self._harness is not None:
            self._harness.heartbeat()
        self._grasp_driver.release_gripper()

    def release_with_async_close(
        self,
        timeout: float = 0.5,
        delay_s: float = 0.8,
    ) -> threading.Thread:
        """Open immediately, close to idle ``delay_s`` later in background.

        Mirrors ``set.py`` lines 138-149: the falling banana leaves the
        fingers before the close starts, so it doesn't snag. The caller
        must ``.join()`` the returned thread after the next ``move_to_traj``
        so the close completes before the next skill's grasp.

        Note: as of ADR-0005 ``set_position_target`` is a **public**
        method on ``WrcGripper`` (was ``_set_position_target`` on the
        upstream ``GraspDriver``). The try/except remains so a future
        rename degrades silently — but it's defensive only, since we
        own both sides of the API now.
        """
        self._grasp_driver.open_gripper(timeout=timeout)

        driver = self._grasp_driver

        def _close_after_delay():
            import time
            time.sleep(delay_s)
            try:
                driver.set_position_target(0.0)
            except AttributeError:
                # Defensive: rename-safe. The gripper stays at the
                # last commanded position (post-open); the next grasp
                # or open call will reset the state machine.
                pass

        thread = threading.Thread(target=_close_after_delay, daemon=True)
        thread.start()
        return thread
