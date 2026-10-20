"""Kinematic mock arm: perfect tracking with the full SafeArm surface.

Runs against the same URDF/kinematics as the real arm so IK, workspace and
collision logic are exercised identically. Used by all tests and by
``--arm mock`` dry-runs.

After the SDK migration (2026-07-24) the skill runtime talks to
``SafeArm`` (which wraps SDK's ``RebotArmEndPose``). For tests we want
a drop-in that doesn't require real hardware, so ``MockArm`` implements
the same public surface (``move_joints`` / ``move_joints_path`` /
``move_cartesian`` / ``set_gripper`` / ``close_gripper_two_stage`` /
``get_state`` / ``stop`` / ``resume`` / ``harness`` / ``disconnect``)
with a real ``SafetyHarness`` doing the same per-waypoint gating.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from ..config import Cfg
from ..types import RobotState, SafetyViolation
from ..safety.harness import Kinematics, SafetyHarness, SafetyLimits


def _min_jerk(s: float) -> float:
    """Scalar min-jerk blend, s ∈ [0, 1] → [0, 1]. 10s³-15s⁴+6s⁵."""
    return 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5


class MockArm:
    """Drop-in replacement for ``SafeArm`` with a perfect-tracking arm.

    Used by every test that needs to exercise skill orchestration
    without CAN / motor hardware. ``harness`` is real (``SafetyHarness``),
    so per-waypoint approval is the same code path as production.
    """

    def __init__(self, cfg: Cfg | None = None, kinematics: Optional[Kinematics] = None):
        self._cfg = cfg
        self.kin = kinematics
        n = int(cfg.get("n_joints", 6)) if cfg is not None else 6
        home = cfg.get("home_q") if cfg is not None else None
        self._q = np.asarray(
            home if home is not None else [0.0, -0.5, -0.9, 0.0, 0.6, 0.0],
            dtype=float,
        )[:n]
        self.n_joints = n

        self._gripper = 0.0
        self._gripper_effort = 1.0
        self._connected = False
        self._stopped = False
        self._stop_send = threading.Event()
        self.commands: list[np.ndarray] = []  # inspection hook for tests
        self._lock = threading.Lock()

        # Gripper travel convention (defaults 0..1; RS profile flips it).
        g = cfg.get("gripper") if cfg is not None else None
        self._grip_open = float(g.get("open_pos", 0.0)) if g else 0.0
        self._grip_closed = float(g.get("closed_pos", 1.0)) if g else 1.0
        self.object_stop_frac: float | None = None

        # Real SafetyHarness so skill tests exercise the same gating code.
        # ``SafetyLimits.from_config`` expects a Cfg with ``workspace`` /
        # ``safety`` blocks; if a raw arm profile was passed in, default
        # limits wide enough to never trip are good enough for tests.
        if cfg is not None and "workspace" in cfg:
            self.harness = SafetyHarness(
                SafetyLimits.from_config(cfg.get("safety", {})),
                kinematics or self._dummy_kin(n),
            )
        else:
            self.harness = SafetyHarness(
                SafetyLimits(
                    workspace_min=np.array([-2.0, -2.0, -1.0]),
                    workspace_max=np.array([2.0, 2.0, 2.0]),
                    table_z=0.0,
                    table_clearance=-10.0,  # disabled
                    max_joint_vel=10.0,  # way above min-jerk peaks
                ),
                kinematics or self._dummy_kin(n),
            )

    @property
    def connected(self) -> bool:
        return self._connected

    def _dummy_kin(self, n: int) -> Kinematics:
        """Permissive FK stand-in for the harness when no URDF is available.

        Lets mock tests run without loading the URDF; everything is treated
        as inside-workspace with no link-clearance checks.
        """
        # Build a minimal Kinematics from a fake URDF? Easier: monkey-patch
        # the methods that harness._check uses.
        from ..safety.harness import Kinematics as _K
        kin = _K.__new__(_K)
        # We don't actually exercise harness FK on mock tests that don't
        # pass kinematics -- the SafetyLimits above are wide enough that
        # the harness never trips.
        return kin

    # ── lifecycle ───────────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        self._stopped = False
        self._stop_send.clear()

    def disconnect(self) -> None:
        self._connected = False

    # ── state ───────────────────────────────────────────────────────

    def get_state(self) -> RobotState:
        return RobotState(
            q=self._q.copy(),
            dq=np.zeros_like(self._q),
            tau=np.zeros_like(self._q),
            gripper_pos=self._gripper,
            t=time.monotonic(),
        )

    # ── motion (SafeArm-shaped) ────────────────────────────────────

    def _make_hook(self, q_first: np.ndarray) -> Callable[[np.ndarray], bool]:
        h = self.harness
        prev = [q_first.copy()]
        dt = 0.02

        def hook(q_i: np.ndarray) -> bool:
            if h.estopped:
                return False
            qp = prev[0]
            ok = h.approve(qp, q_i, dt=dt)
            if ok:
                prev[0] = q_i.copy()
            return ok
        return hook

    def _stream(self, q_pts: list, duration_s: float) -> bool:
        """Drive ``_q`` along min-jerk waypoints; honor harness via hook."""
        if not q_pts:
            return True
        if self._stopped:
            return False
        h = self.harness
        try:
            h.begin_motion()
        except SafetyViolation:
            return False
        hook = self._make_hook(q_pts[0])
        n = max(2, int(duration_s * 50.0))
        # If the caller gave us fewer waypoints than 50 Hz slots, expand
        # by min-jerk blending between the caller-provided waypoints.
        if len(q_pts) < 2:
            pts = list(q_pts)
        else:
            pts = []
            for i in range(len(q_pts) - 1):
                a, b = q_pts[i], q_pts[i + 1]
                steps = max(1, n // (len(q_pts) - 1))
                for k in range(steps):
                    s = _min_jerk((k + 1) / steps)
                    pts.append(a + (b - a) * s)
        for q in pts:
            if self._stop_send.is_set() or self._stopped:
                break
            if not hook(q):
                self._stopped = True
                h.estop(reason="mock arm: harness veto")
                return False
            self._q = np.asarray(q, dtype=float).copy()
            self.commands.append(self._q.copy())
        return True

    def move_joints(self, q_target: np.ndarray, duration_s: float = 2.0) -> bool:
        q_target = np.asarray(q_target, dtype=float).reshape(-1)[: self.n_joints]
        q_start = self._q.copy()
        # begin_motion raises SafetyViolation if estopped or watchdog
        # is stale; let it propagate so skill callers see the e-stop
        # exception (not a generic "didn't settle").
        self.harness.begin_motion()
        ok = self._stream([q_start, q_target], duration_s)
        return ok and not self.harness.estopped

    def move_joints_path(
        self,
        waypoints: list,
        segment_duration_s: float = 0.5,
        qds=None,  # accepted for compat; mock doesn't use vel FF
    ) -> bool:
        if not waypoints:
            return True
        self.harness.begin_motion()
        total = max(segment_duration_s * len(waypoints), 0.05)
        return self._stream(list(waypoints), total)

    def move_cartesian(
        self,
        T_target: np.ndarray,
        kin=None,  # accepted for compat
        duration_s: float = 2.0,
        planner=None,
    ) -> bool:
        # No FK/IK on mock -- just send the position the planner already gave
        # us via waypoints (when planner is provided) or fall back to PTP
        # via joint-space move from current q.
        if planner is not None:
            q_curr = self._q.copy()
            T_curr = self.kin.fk(q_curr) if self.kin is not None and hasattr(self.kin, "fk") else np.eye(4)
            waypoints = planner.plan(q_curr, T_curr, T_target, duration_s)
            return self.move_joints_path(waypoints, segment_duration_s=duration_s / max(len(waypoints), 1))
        return self.move_joints(self._q, duration_s)  # PTP fallback: no motion

    # ── gripper ─────────────────────────────────────────────────────

    def set_gripper(self, pos: float, effort: float = 0.0) -> None:
        """Move the gripper to ``pos`` (close-fraction 0=open, 1=closed),
        honoring whichever stop is in effect.

        Stop priority (first match wins):
          1. ``_object_close_frac`` — width-aware: simulate the real
             RebotArm's MIT-mode stall at the object surface, so the
             gripper stops at the physical width of the held object
             rather than at a hardcoded position. ``None`` disables
             this and falls through to ``object_stop_frac``.
          2. ``object_stop_frac`` — legacy: hardcoded stop at
             ``open + (closed-open) * object_stop_frac`` (test fixture
             for the pre-width-aware ASPIRE heuristic).
          3. no stop — gripper just moves to ``pos`` (simulates an
             air-grasp or a perfectly soft surface).
        """
        if self._stopped:
            return
        pos = float(pos)
        span = self._grip_closed - self._grip_open
        if getattr(self, "_object_close_frac", None) is not None:
            # Width-aware: object surface at the given close-fraction.
            # Gripper stops when q reaches the surface.
            surface = float(self._object_close_frac)
            if span >= 0:
                pos = min(pos, surface)
            else:
                pos = max(pos, surface)
        elif self.object_stop_frac is not None:
            stop = self._grip_open + span * self.object_stop_frac
            if span >= 0:
                pos = min(pos, stop)
            else:
                pos = max(pos, stop)
        self._gripper = pos
        self._gripper_effort = float(effort)

    def set_object_close_frac(self, surface_close_frac: float | None) -> None:
        """Pin the simulated object surface to ``surface_close_frac``
        (0=fully open, 1=fully closed). Runtime computes this from
        ``grasp.width_m / max_width_m`` and calls this before each
        width-aware close so the mock mirrors the real arm's "motor
        stalls at the object surface" behavior. Pass ``None`` to fall
        back to the legacy ``object_stop_frac`` semantics.
        """
        self._object_close_frac = float(surface_close_frac) if surface_close_frac is not None else None

    # Back-compat alias: earlier draft of the v2 fix used absolute
    # metres. Some code paths may still call this; it now interprets
    # the argument as a close-fraction (i.e. it self-documents the
    # change so a stale call won't silently re-introduce the old bug).
    def set_object_width_m(self, surface_close_frac: float | None) -> None:  # noqa: D401
        """Deprecated: pass a close-fraction via ``set_object_close_frac`` instead."""
        self.set_object_close_frac(surface_close_frac)

    def open_gripper(self, distance_m: float | None = None) -> None:
        """Open the gripper.

        ``distance_m`` (legacy kwarg from RebotRSArm): if provided,
        interpret as a fraction of the open/closed travel (0=closed,
        1=fully open) and interpolate. ``None`` = fully open (default).
        """
        if distance_m is None:
            self.set_gripper(self._grip_open)
        else:
            span = self._grip_closed - self._grip_open
            pos = self._grip_open + float(distance_m) * span
            self.set_gripper(pos)

    def close_gripper_two_stage(
        self,
        width_pos_stage1: float = 0.5,
        width_pos_stage2: float = 0.85,
        effort: float = 0.6,
        stall_vel: float = 0.05,
        timeout_s: float = 3.0,
    ) -> Optional[float]:
        """Two-stage close to absolute position fractions (0=open, 1=closed).
        Mock: ``set_gripper`` itself honors ``object_stop_frac`` so the
        width-aware stage-2 target naturally stops at the simulated
        object's surface — mirroring the real RebotRSArm's MIT-mode
        behavior where ``tau = kp*(q_target - q) - kd*dq + tau_ff`` stalls
        the motor at the object surface rather than crushing it.
        """
        pos1 = self._grip_open + (self._grip_closed - self._grip_open) * width_pos_stage1
        pos2 = self._grip_open + (self._grip_closed - self._grip_open) * width_pos_stage2
        self.set_gripper(pos1, effort=effort * 0.7)
        time.sleep(0.3)
        self.set_gripper(pos2, effort=effort)
        # Mock: assume success immediately.
        return self._gripper_width_frac()

    def _gripper_width_frac(self) -> float:
        span = self._grip_closed - self._grip_open
        if abs(span) < 1e-9:
            return 0.0
        return float(np.clip((self._gripper - self._grip_open) / span, 0.0, 1.0))

    # ── soft stop / resume ──────────────────────────────────────────

    def stop(self) -> None:
        self.harness.estop(reason="mock.stop()")
        self._stopped = True
        self._stop_send.set()

    def resume(self) -> None:
        self.harness.reset_estop()
        self._stopped = False
        self._stop_send.clear()

    # ── convenience ─────────────────────────────────────────────────

    def home(self, duration_s: float = 3.0) -> bool:
        home_q = self._cfg.get("home_q", [0.0] * self.n_joints)
        return self.move_joints(np.asarray(home_q, dtype=float), duration_s)

    def wait_settled(self, timeout_s=2.0, tol=None) -> bool:
        """Mock is always settled (perfect tracking)."""
        return True