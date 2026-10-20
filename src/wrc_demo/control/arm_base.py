"""Arm interface. All motion is streamed as min-jerk joint waypoints so the
safety harness can vet every intermediate configuration, and completion is
based on feedback (mock: kinematic state; real: RobStride mechPos param
reads), never on sleep(duration) like the baseline.
"""

from __future__ import annotations

import abc
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState


def min_jerk(s: float) -> float:
    """Min-jerk time scaling on s in [0, 1]."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ArmBase(abc.ABC):
    """A 6-DOF arm + 1 gripper motor behind a uniform, feedback-based API."""

    n_joints = 6
    #: settle tolerance (rad). Real arms hold with pure PD (no gravity
    #: feedforward), so steady-state droop under payload needs headroom;
    #: backends override from config.
    settle_tol = 0.03

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def get_state(self) -> RobotState: ...

    @abc.abstractmethod
    def send_joint_target(self, q: np.ndarray) -> None:
        """Stream one joint-space setpoint (already safety-approved)."""

    @abc.abstractmethod
    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        """Command gripper motor position with an effort/torque scale 0..1."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Best-effort immediate halt (software e-stop)."""

    # ── shared streaming executor ────────────────────────────────────────

    def stream_to(
        self,
        q_target: np.ndarray,
        duration_s: float,
        rate_hz: float = 50.0,
        approve=None,
        settle_tol: float | None = None,
        settle_timeout_s: float = 6.0,
    ) -> bool:
        """Min-jerk interpolate current->target, vetting each waypoint.

        `approve(q_prev, q_next, dt)` raises SafetyViolation to abort.
        Returns True when the arm settles within `settle_tol` rad of target.
        """
        if settle_tol is None:
            settle_tol = self.settle_tol
        q_start = self.get_state().q.copy()
        q_target = np.asarray(q_target, dtype=float).reshape(-1)
        steps = max(2, int(duration_s * rate_hz))
        dt = duration_s / steps
        q_prev = q_start
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            s = min_jerk(i / steps)
            q_i = q_start + (q_target - q_start) * s
            if approve is not None:
                approve(q_prev, q_i, dt)
            self.send_joint_target(q_i)
            q_prev = q_i
            # Deadline-based pacing: per-step overhead (approve FK + CAN
            # sends) must not stretch the total duration, or the commanded
            # velocity profile silently slows and drifts.
            sleep_s = t0 + i * dt - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        return self.wait_settled(q_target, settle_tol, settle_timeout_s)

    def wait_settled(self, q_target: np.ndarray, tol: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            err = np.abs(self.get_state().q - q_target).max()
            if err < tol:
                return True
            time.sleep(0.05)
        return False


def make_arm(cfg: Cfg, kinematics=None) -> ArmBase:
    kind = cfg.type
    if kind == "mock":
        from .mock_arm import MockArm

        return MockArm(cfg, kinematics)
    if kind == "rebot_rs":
        from .rebot_rs_arm import RebotRSArm

        return RebotRSArm(cfg)
    raise ValueError(f"unknown arm type {kind!r} (mock|rebot_rs)")
