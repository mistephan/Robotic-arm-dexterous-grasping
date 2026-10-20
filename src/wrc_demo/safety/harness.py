"""Safety harness + safe arm wrapper.

Modules provided:

- ``Kinematics``: FK + ``link_positions`` wrapper around the SDK Pinocchio
  model. Only used by the harness (skill IK calls go to SDK directly).
- ``SafetyLimits``: workspace / joint / velocity / table-clearance /
  watchdog / keep-out dataclass.
- ``SafetyHarness``: per-waypoint fail-closed approval logic (8 checks).
- ``SafeArm``: thin wrapper around ``reBotArm_control_py``'s
  ``RebotArmEndPose`` that injects the harness via the SDK's
  ``waypoint_hook``. The SDK owns the 500 Hz MIT control loop, gravity
  feed-forward, sign conversion, and IK / CLIK. wrc_demo owns the
  safety gate, two-stage gripper close, and motion bookkeeping.

History: this file used to also carry ``ArmBase`` (an ABC), an
``ArmBase.stream_to`` with 50 Hz min-jerk + per-waypoint ``approve``,
and a 600-line ``RebotRSArm`` wrapper. After the SDK migration (see
``sdk/README.md`` + ``docs/MIGRATION_PLAN.md``), all of that is gone —
the SDK provides ``move_joints_traj`` + ``waypoint_hook`` natively.
"""
from __future__ import annotations

import threading
import time
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Vendored SDK is on sys.path via wrc_demo.__init__.
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import compute_fk, load_robot_model

from ..types import RobotState, SafetyViolation


# ──────────────────────────────────────────────────────────────────────
# Kinematics
# ──────────────────────────────────────────────────────────────────────

class Kinematics:
    """FK + link-origins wrapper around the SDK's Pinocchio model.

    Only the harness uses this — every other consumer (skills, calib,
    grasping) calls SDK's IK / FK directly. Keeping the harness on a
    wrc_demo-local wrapper means the safety gate never depends on the
    SDK's vendored copy behaving identically across re-syncs.

    The RS URDF describes 8 joints (6 arm + 2 gripper fingers).
    ``n_joints`` here returns the **controlled arm joint count** (6),
    not the URDF total — the harness only vets arm motion. Gripper
    fingers are driven separately by SDK and bypass the harness.

    Coordinate convention: this returns positions in the **asset
    frame**, identical to the SDK's FK. ``home_q`` in
    ``configs/arms/*.yaml`` is already in this frame; no
    ``joint_signs`` multiplication is applied (the old wrc_demo control
    layer did that and it was wrong — see ``sdk/README.md``).
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "gripper_end",
        arm_joint_count: int = 6,
    ):
        self._model = load_robot_model(urdf_path)
        self._data = self._model.createData()
        self._ee_frame_id = self._model.getFrameId(ee_frame)
        self._n_arm = int(arm_joint_count)

        # Pull joint limits via SDK's helper (vendor-patched to use idx_q
        # so 2-finger gripper URDFs don't crash — see sdk/README.md).
        # Take only the arm portion (first arm_joint_count).
        from reBotArm_control_py.kinematics.robot_model import get_joint_limits
        limits = get_joint_limits(self._model)[: self._n_arm]
        self._lo = np.array([lo for lo, _ in limits], dtype=np.float64)
        self._hi = np.array([hi for _, hi in limits], dtype=np.float64)

    @property
    def n_joints(self) -> int:
        """Number of arm joints the harness vets (excludes gripper)."""
        return self._n_arm

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lo.copy(), self._hi.copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """4×4 SE3 of the end-effector in the asset frame."""
        q = self._pad(q)
        _, _, T = compute_fk(self._model, q)
        return T

    def ik(
        self,
        T_target: np.ndarray,
        q_init: np.ndarray,
        retries: int = 0,
        limit_margin: float = 0.025,
    ):
        """DLS IK in the asset frame.

        Thin wrapper around SDK's ``solve_ik`` (+ ``solve_ik_with_retry``
        when ``retries > 0``). Returns SDK's ``IKResult`` which has
        ``.q``, ``.success``, ``.error``, ``.iterations``.

        Solver tuning (the SDK's default ``tolerance=1e-4`` traps the
        DLS local solver at err≈1.6 mm on the top-down grasps our
        calibration produces, because home_q is in the elbow-up basin
        and the top-down minimum is in the elbow-down basin — DLS won't
        cross the singularity between them):

        - ``tolerance = 5e-3`` (5 mm, ample for the gripper's jaw width
          and the safety harness' 5 cm workspace AABB slack)
        - ``max_iter  = 2000`` (vs default 1000; convergence is slow near
          the basin boundary)
        - ``retries   = 0`` → 8 random restarts (``solve_ik_with_retry``),
          which jump out of the elbow-up basin

        ``limit_margin`` (default 0.025 rad) post-clamps the solver's
        ``q`` to ``[URDF_lower + margin, URDF_upper - margin]`` BEFORE
        returning. The SDK's internal ``_clamp_config`` only knows the
        raw URDF bounds -- without this post-clamp, a converged
        solution can land *exactly* on the URDF edge and the safety
        harness's own 0.020 rad ``joint_margin`` check rejects it
        as "joint N out of limits", producing a downstream
        ``SafetyViolation: pregrasp target unsafe`` even though the
        IK itself returned ``success=True``. This was the
        "past review defect" called out in CLAUDE.md.

        Invariant (CLAUDE.md §"Contracts & invariants"):
        ``Kinematics.ik limit_margin (0.025) > harness joint_margin
        (0.02)`` -- always strictly greater. Tune one side, tune both.
        """
        from reBotArm_control_py.kinematics.inverse_kinematics import (
            IKParams,
            pos_rot_to_se3,
            solve_ik,
            solve_ik_with_retry,
        )
        q_init = self._pad(np.asarray(q_init, dtype=np.float64))[: self._n_arm]
        T_target = np.asarray(T_target, dtype=np.float64)
        T_se3 = pos_rot_to_se3(
            T_target[:3, 3], rot=T_target[:3, :3],
        )
        params = IKParams(max_iter=2000, tolerance=5e-3, step_size=0.5, damping=1e-6)
        if retries > 0:
            result = solve_ik_with_retry(
                self._model, self._data, self._ee_frame_id,
                T_se3, q_init, params, max_retries=retries,
                controlled_joints=self._n_arm,
            )
        else:
            result = solve_ik(
                self._model, self._data, self._ee_frame_id,
                T_se3, q_init, params,
                controlled_joints=self._n_arm,
            )
        # Post-clamp to honour the limit_margin invariant. The SDK
        # solver can land q at the URDF edge; the harness's own 0.020
        # joint_margin then rejects the motion as "joint N out of
        # limits" even though the IK succeeded. See CLAUDE.md §"Past
        # review defect" -- 2026-08-15 root-caused by user log
        # ``SafetyViolation: pregrasp target unsafe: joint4 out of
        # limits (-1.560 ∉ [-1.550, 1.550])``.
        if limit_margin > 0.0:
            result.q[: self._n_arm] = np.clip(
                result.q[: self._n_arm],
                self._lo + limit_margin,
                self._hi - limit_margin,
            )

        # The safety-margin clamp can move a converged solution. Recompute
        # every diagnostic against the exact q returned, then re-apply the
        # unchanged SDK tolerance so success and error cannot disagree.
        import pinocchio as pin

        actual_T = self.fk(result.q)
        actual = pin.SE3(actual_T[:3, :3], actual_T[:3, 3])
        result.error = float(
            np.linalg.norm(pin.log6(actual.inverse() * T_se3).vector)
        )
        result.position_error_m = float(
            np.linalg.norm(actual_T[:3, 3] - T_target[:3, 3])
        )
        result.rotation_error_rad = float(
            np.linalg.norm(
                pin.log3(actual_T[:3, :3].T @ T_target[:3, :3])
            )
        )
        result.success = bool(result.error < params.tolerance)
        return result

    def link_positions(self, q: np.ndarray) -> np.ndarray:
        """``(n_arm_joints, 3)`` world-space link origins (skip universe + gripper).

        Used by the harness to detect the elbow scooping below the table
        — the TCP clearance check alone misses it because the TCP can be
        safely above the table while link7 is already buried in it.
        """
        import pinocchio as pin
        q = self._pad(q)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        # joints[1..n_arm] = arm joints; joints[n_arm+1..njoints-1] = gripper
        # (we don't expose gripper finger origins to the harness).
        return np.array(
            [self._data.oMi[i].translation.copy() for i in range(1, 1 + self._n_arm)]
        )

    def _pad(self, q: np.ndarray) -> np.ndarray:
        """Pad short q vectors with zeros so Pinocchio gets ``model.nq``.

        Pinocchio FK requires the full configuration vector (URDF has 8
        entries: 6 arm + 2 gripper fingers). The harness only reasons
        about arm joints, but Pinocchio still needs the full vector.
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] == self._model.nq:
            return q
        if q.shape[0] < self._model.nq:
            return np.concatenate([q, np.zeros(self._model.nq - q.shape[0])])
        return q[: self._model.nq]


# ──────────────────────────────────────────────────────────────────────
# SafetyLimits
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SafetyLimits:
    """Harness configuration (workspace, joint margin, velocity, etc.)."""

    workspace_min: np.ndarray  # (3,) base frame, meters
    workspace_max: np.ndarray
    table_z: float = 0.0
    table_clearance: float = 0.02
    max_joint_vel: float = 1.2  # rad/s
    joint_margin: float = 0.02  # rad inside URDF limits
    watchdog_s: float = 5.0  # halt if perception heartbeat older than this
    keep_out: list = field(default_factory=list)  # list of (min(3,), max(3,))

    @classmethod
    def from_config(cls, cfg) -> "SafetyLimits":
        ws = cfg.workspace
        return cls(
            workspace_min=np.asarray(ws.min, dtype=float),
            workspace_max=np.asarray(ws.max, dtype=float),
            table_z=float(cfg.get("table_z", 0.0)),
            table_clearance=float(cfg.get("table_clearance", 0.02)),
            max_joint_vel=float(cfg.get("max_joint_vel", 1.2)),
            joint_margin=float(cfg.get("joint_margin", 0.02)),
            watchdog_s=float(cfg.get("watchdog_s", 5.0)),
            keep_out=[
                (np.asarray(k["min"], dtype=float), np.asarray(k["max"], dtype=float))
                for k in cfg.get("keep_out", [])
            ],
        )


# ──────────────────────────────────────────────────────────────────────
# SafetyHarness
# ──────────────────────────────────────────────────────────────────────

class SafetyHarness:
    """Per-waypoint fail-closed approval logic.

    Checks, in order (cheapest first):

    1. e-stop latch (raised → motion aborts)
    2. perception watchdog (one-shot at ``begin_motion``; not per waypoint)
    3. joint limits with margin (with ESCAPE RULE for joints already at /
       outside the margin — they may move strictly back toward validity)
    4. per-joint velocity cap
    5. workspace AABB on the TCP (FK) — with ESCAPE RULE
    6. table-plane clearance for TCP and intermediate joint origins
       (with a grasp-exemption cylinder so the tool may descend onto
       the active target)
    7. optional keep-out AABBs (e.g. the camera tripod)

    The arm SDK enforces none of this. ``approve()`` raising
    ``SafetyViolation`` aborts the streamed motion mid-flight.
    """

    def __init__(self, limits: SafetyLimits, kinematics: Kinematics):
        self.limits = limits
        self.kin = kinematics
        self._estopped = False
        self._grasp_exempt: Optional[tuple[np.ndarray, float, float]] = None
        self._last_heartbeat = time.monotonic()
        self._motion_active = False
        self.violations: list[str] = []
        self._lock = threading.Lock()

    # ── state management ──────────────────────────────────────────────

    def heartbeat(self) -> None:
        """Refresh the perception watchdog timer (call from WorldWatcher / observe)."""
        with self._lock:
            self._last_heartbeat = time.monotonic()

    def estop(self, reason: str = "external") -> None:
        with self._lock:
            self._estopped = True
            self._motion_active = False
            self.violations.append(reason)

    def reset_estop(self) -> None:
        with self._lock:
            self._estopped = False
            self.violations.clear()

    @property
    def estopped(self) -> bool:
        with self._lock:
            return self._estopped

    def allow_grasp_descent(
        self, center_xy, radius: float, z_min: float
    ) -> None:
        with self._lock:
            self._grasp_exempt = (
                np.asarray(center_xy, dtype=np.float64),
                float(radius),
                float(z_min),
            )

    def clear_grasp_exemption(self) -> None:
        with self._lock:
            self._grasp_exempt = None

    def begin_motion(self) -> None:
        with self._lock:
            if self._estopped:
                raise SafetyViolation("e-stopped")
            age = time.monotonic() - self._last_heartbeat
            if age > self.limits.watchdog_s:
                raise SafetyViolation(
                    f"watchdog: no perception heartbeat for {age:.1f}s "
                    f"(limit {self.limits.watchdog_s}s)"
                )
            self._motion_active = True

    def end_motion(self) -> None:
        with self._lock:
            self._motion_active = False

    # ── per-waypoint approval ─────────────────────────────────────────

    def approve(self, q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> bool:
        """Return True if moving from q_prev to q_next over dt is safe.

        The SDK's waypoint_hook calls this for every waypoint. If this
        returns False, the SDK aborts the trajectory immediately and
        clears ``_moving`` so ``SafeArm.wait_motion_done`` unblocks.

        Watchdog semantics: ``begin_motion`` raises on stale perception,
        and that gate is *only* checked at motion start — not per
        waypoint (a legitimate grasp sequence intentionally runs for
        seconds without re-observing). When called *outside* an active
        motion, we re-check so a stray ``approve`` after a long pause
        can't bypass the watchdog.
        """
        with self._lock:
            if self._estopped:
                return False
            if not self._motion_active:
                age = time.monotonic() - self._last_heartbeat
                if age > self.limits.watchdog_s:
                    self.violations.append(
                        f"watchdog: no perception heartbeat for {age:.1f}s "
                        f"(limit {self.limits.watchdog_s}s)"
                    )
                    return False
            q_prev = np.asarray(q_prev, dtype=np.float64).reshape(-1)
            q_next = np.asarray(q_next, dtype=np.float64).reshape(-1)
            reason = self._check(q_prev, q_next, dt)
            if reason:
                self.violations.append(reason)
                return False
            return True

    def vet_pose(
        self,
        q: np.ndarray,
        exempt_xy=None,
        exempt_radius: float = 0.0,
        exempt_z_min=None,
    ) -> Optional[str]:
        """Static pre-motion check (no escape rules, used before hooks).

        Returns ``None`` when the pose passes, or a short reason string
        (e.g. ``"workspace"``, ``"joint 2 out of limits (…)"``) when it
        doesn't. Callers (``select_grasp``, ``safe_arm._finalize_motion``
        etc.) use the reason to surface useful diagnostics to the agent.

        ``approve`` already covers this for streamed motion; this exists
        for pre-motion vet calls (e.g. from ``select_grasp`` before any
        stream starts).
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        # Static check: same as approve but without velocity + without
        # escape rules and using a fake q_prev=q_next so dq=0.
        return self._check(q, q, dt=1.0, exempt_xy=exempt_xy,
                           exempt_radius=exempt_radius,
                           exempt_z_min=exempt_z_min)

    # ── internals ─────────────────────────────────────────────────────

    def _check(
        self,
        q_prev: np.ndarray,
        q_next: np.ndarray,
        dt: float,
        exempt_xy=None,
        exempt_radius: float = 0.0,
        exempt_z_min=None,
    ) -> Optional[str]:
        limits = self.limits
        n = self.kin.n_joints
        q_prev = q_prev[:n]
        q_next = q_next[:n]

        # joint limits + margin
        lo, hi = self.kin.joint_limits
        margin = limits.joint_margin
        for i in range(n):
            l, h = lo[i] + margin, hi[i] - margin
            v = q_next[i]
            if v < l or v > h:
                # ESCAPE RULE: only if the joint was already out of the
                # margin AND the new value is *strictly closer* to the
                # allowed range (i.e. actual motion toward validity). A
                # static pose at an invalid value still trips the check.
                if v < l and q_prev[i] <= l + 1e-6 and v > q_prev[i]:
                    continue
                if v > h and q_prev[i] >= h - 1e-6 and v < q_prev[i]:
                    continue
                return f"joint{i+1} out of limits ({v:.3f} ∉ [{l:.3f}, {h:.3f}])"

        # per-joint velocity
        if dt > 0:
            dq = np.abs(q_next - q_prev) / dt
            over = np.where(dq > limits.max_joint_vel)[0]
            if over.size:
                i = int(over[0])
                return (
                    f"joint{i+1} vel {dq[i]:.2f} rad/s "
                    f"> cap {limits.max_joint_vel:.2f}"
                )

        # FK on prev + next (also used by the escape rules below)
        T = self.kin.fk(q_next)
        tcp = T[:3, 3]
        T0 = self.kin.fk(q_prev)
        tcp0 = T0[:3, 3]

        # workspace AABB on TCP
        if (
            np.any(tcp < limits.workspace_min - 1e-6)
            or np.any(tcp > limits.workspace_max + 1e-6)
        ):
            # ESCAPE RULE: already outside → moving strictly toward AABB
            # ok. A static pose at an invalid workspace location still
            # trips the check (no motion = no escape).
            tcp0_outside = (
                np.any(tcp0 < limits.workspace_min - 1e-6)
                or np.any(tcp0 > limits.workspace_max + 1e-6)
            )
            if tcp0_outside:
                # Every axis that was *outside* in tcp0 must now be
                # strictly closer to the AABB — i.e. actual inward
                # motion. A static pose at an invalid workspace point
                # has zero inward motion on every axis, so it trips
                # the check.
                inward = True
                for ax in range(3):
                    was_outside_low = tcp0[ax] < limits.workspace_min[ax] - 1e-6
                    was_outside_high = tcp0[ax] > limits.workspace_max[ax] + 1e-6
                    if was_outside_low and tcp[ax] <= tcp0[ax]:
                        inward = False  # not moving toward min
                    elif was_outside_high and tcp[ax] >= tcp0[ax]:
                        inward = False  # not moving toward max
                if not inward:
                    return (
                        f"TCP {tcp} outside workspace "
                        f"[{limits.workspace_min}, {limits.workspace_max}]"
                    )
            else:
                return (
                    f"TCP {tcp} outside workspace "
                    f"[{limits.workspace_min}, {limits.workspace_max}]"
                )

        # table clearance
        floor = limits.table_z + limits.table_clearance
        if tcp[2] < floor - 1e-6:
            if self._in_exempt(tcp, exempt_xy, exempt_radius, exempt_z_min):
                pass
            elif self._in_grasp(tcp):
                pass
            else:
                # ESCAPE RULE: if the TCP was already below the table, only
                # strict *ascent* is allowed (lateral motion at the same
                # z is not — that just slides the link through the surface).
                if tcp[2] > tcp0[2] + 1e-9:
                    pass
                else:
                    return (
                        f"TCP z={tcp[2]:.3f} below table "
                        f"({limits.table_z:.3f}+{limits.table_clearance:.3f})"
                    )

        # coarse link check (every joint origin except TCP must clear table)
        origins = self.kin.link_positions(q_next)
        if origins.shape[0] >= 1:
            # last origin = TCP; earlier origins are intermediate links
            for i, p in enumerate(origins[:-1]):
                if p[2] < limits.table_z + 0.01:
                    return (
                        f"link{i+1} origin z={p[2]:.3f} below table "
                        f"(elbow scooping guard)"
                    )

        # keep-out
        for (mn, mx) in limits.keep_out:
            if np.all(tcp >= mn - 1e-6) and np.all(tcp <= mx + 1e-6):
                return f"TCP {tcp} inside keep-out [{mn}, {mx}]"

        return None

    def _in_grasp(self, tcp: np.ndarray) -> bool:
        """True if TCP is inside the active grasp-exemption cylinder.

        Reads ``self._grasp_exempt`` WITHOUT taking ``self._lock`` —
        callers must already hold it (this is always the case: the only
        caller is ``_check``, which itself runs inside ``approve`` /
        ``vet_pose`` ``with self._lock:``).
        """
        ex = self._grasp_exempt
        if ex is None:
            return False
        center, radius, z_min = ex
        return self._in_cylinder(tcp, center, radius, z_min)

    def _in_exempt(
        self,
        tcp: np.ndarray,
        exempt_xy,
        exempt_radius: float,
        exempt_z_min,
    ) -> bool:
        if exempt_xy is None or exempt_radius < 0:
            return False
        z_min = exempt_z_min if exempt_z_min is not None else -np.inf
        return self._in_cylinder(
            tcp, np.asarray(exempt_xy, dtype=float), exempt_radius, z_min
        )

    @staticmethod
    def _in_cylinder(p, center_xy, radius, z_min) -> bool:
        if p[2] < z_min:
            return False
        return float(np.linalg.norm(p[:2] - center_xy)) <= radius


# ──────────────────────────────────────────────────────────────────────
# SafeArm
# ──────────────────────────────────────────────────────────────────────

class SafeArm:
    """wrC_demo's only motion interface for skills.

    Composes:
    - SDK ``RebotArmEndPose`` (500 Hz loop, gravity FF, sign conversion,
      IK / CLIK — all owned by SDK)
    - ``SafetyHarness`` (per-waypoint approval — owned by wrc_demo)
    - ``Kinematics`` (FK for the harness — owned by wrc_demo)

    The waypoint_hook parameter on the SDK's trajectory methods is the
    integration seam: ``_make_hook`` closes over the harness so every
    waypoint is vetted before it reaches ``_q_target``.

    Threading note: the hook runs in the SDK's daemon ``_send_loop``
    thread. Exceptions raised there would be silent, so the hook
    catches ``SafetyViolation``, stashes it on ``_last_violation``,
    latches the harness e-stop, and returns False. ``move_joints`` /
    ``move_joints_path`` re-raise the stashed exception after
    ``wait_motion_done`` returns so skill callers see the same
    exception shape they always did.
    """

    def __init__(
        self,
        sdk_arm: RebotArm,
        sdk_ctrl: RebotArmEndPose,
        harness: SafetyHarness,
        kinematics: Kinematics,
        cfg,
    ):
        self._arm = sdk_arm
        self._ctrl = sdk_ctrl
        self._h = harness
        self._kin = kinematics
        self._cfg = cfg
        # SDK exposes _n (controlled arm joints = 6). Use it; ignore gripper.
        self._n = int(getattr(sdk_ctrl, "_n", 6))
        self._waypoint_hz = 50.0
        self._dt = 1.0 / self._waypoint_hz
        # cfg is the arm profile (full cfg.arm), so gripper keys live on cfg.
        gripper = cfg.get("gripper", {}) if hasattr(cfg, "get") else {}
        self._grip_open = float(gripper.get("open_pos", 0.0))
        self._grip_closed = float(gripper.get("closed_pos", 1.0))
        self._max_width = float(gripper.get("max_width_m", 0.09))
        self._effort_default = float(gripper.get("effort_default", 0.6))
        self._last_violation: Optional[SafetyViolation] = None

    # ── properties ──────────────────────────────────────────────────

    @property
    def n_joints(self) -> int:
        return self._n

    @property
    def harness(self) -> SafetyHarness:
        return self._h

    # ── motion ──────────────────────────────────────────────────────

    def move_joints(self, q_target: np.ndarray, duration_s: float = 2.0) -> bool:
        """Min-jerk + 50 Hz waypoint approval + SDK streaming.

        Auto-stretches duration so the min-jerk peak velocity
        (1.875 * dq / T) stays under 0.9 × the harness velocity cap
        (harness is still the backstop).
        """
        if not getattr(self._ctrl, "_running", False):
            return False
        q_target = self._pad(np.asarray(q_target, dtype=np.float64))[: self._n]
        q_start = self._arm.get_positions()[: self._n]
        dq_max = float(np.max(np.abs(q_target - q_start)))
        needed = 1.875 * dq_max / (0.9 * self._h.limits.max_joint_vel)
        duration_s = max(duration_s, needed)
        waypoints = self._min_jerk(q_start, q_target, duration_s)

        self._last_violation = None
        self._h.begin_motion()
        try:
            if not self._ctrl.move_joints_traj(
                waypoints, duration_s, waypoint_hook=self._make_hook(q_start)
            ):
                return False
            ok = self._ctrl.wait_motion_done(timeout=duration_s + 5.0)
            return self._finalize_motion(ok)
        except SafetyViolation as e:
            # begin_motion watchdog/e-stop leaked through (rare race)
            self._last_violation = e
            self._h.estop(reason=str(e))
            raise
        finally:
            self._h.end_motion()

    def move_joints_path(
        self,
        waypoints: list,
        segment_duration_s: float = 0.5,
        qds=None,  # accepted for compat; SDK does its own vel FF
    ) -> bool:
        """Stream a pre-computed joint path, each segment gated.

        Used by ``skill_move_to_pose`` (CartesianPlanner output): every
        waypoint was already IK-solved; we just stream + vet.
        """
        if not getattr(self._ctrl, "_running", False):
            return False
        if not waypoints:
            return False
        pts = [self._pad(np.asarray(w, dtype=np.float64))[: self._n] for w in waypoints]
        total = max(segment_duration_s * len(pts), 0.05)

        self._last_violation = None
        self._h.begin_motion()
        try:
            if not self._ctrl.move_joints_traj(
                pts, total, waypoint_hook=self._make_hook(pts[0])
            ):
                return False
            ok = self._ctrl.wait_motion_done(timeout=total + 5.0)
            return self._finalize_motion(ok)
        finally:
            self._h.end_motion()

    def move_cartesian(
        self,
        T_target: np.ndarray,
        kin=None,  # accepted for compat; SDK has its own IK
        duration_s: float = 2.0,
        planner=None,
    ) -> bool:
        """Cartesian move: planner → waypoints → move_joints_path,
        else SDK.move_to_traj (which does IK + SE(3) geodesic + CLIK).
        """
        T_target = np.asarray(T_target, dtype=np.float64)
        if planner is not None:
            q_curr = self._arm.get_positions()[: self._n]
            T_curr = self._kin.fk(q_curr)
            waypoints = planner.plan(q_curr, T_curr, T_target, duration_s)
            return self.move_joints_path(
                waypoints, segment_duration_s=max(duration_s / max(len(waypoints), 1), 0.01)
            )
        # Default: let SDK's CLIK do it. Convert T_target → xyz+rpy.
        x, y, z = float(T_target[0, 3]), float(T_target[1, 3]), float(T_target[2, 3])
        roll, pitch, yaw = self._se3_to_rpy(T_target)

        self._last_violation = None
        self._h.begin_motion()
        try:
            if not self._ctrl.move_to_traj(
                x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                duration=duration_s, waypoint_hook=self._make_hook(None),
            ):
                return False
            ok = self._ctrl.wait_motion_done(timeout=duration_s + 5.0)
            return self._finalize_motion(ok)
        finally:
            self._h.end_motion()

    # ── gripper ─────────────────────────────────────────────────────

    def set_gripper(self, pos: float, effort: float = 0.0) -> None:
        """Set gripper position + per-cycle tau (forward-feed)."""
        self._ctrl.set_gripper_target(pos, effort=effort)

    def open_gripper(self, distance_m: float | None = None,
                     effort: float = 0.0) -> None:
        """Open the gripper.

        ``distance_m`` is the target jaw opening in meters. ``None`` means
        "open to max width" (the SDK's default behaviour, kept for backward
        compatibility with callers that don't specify a width).

        Skill callers (e.g. ``runtime.py:skill_grasp_object``) pass a
        concrete ``distance_m`` derived from the object's grasp width; the
        baseline's LazyArm API contract is preserved here so the skill code
        doesn't need a per-call ``set_gripper(open_pos)`` workaround.

        Mapping: ``distance_m / max_width_m`` fraction of the gripper's
        joint travel (open_pos -> closed_pos) is forwarded to
        ``set_gripper_target``. Grippers without a configured ``open_pos`` /
        ``max_width_m`` silently fall back to the SDK default.
        """
        if not getattr(self._arm, "has_gripper", False):
            return
        if distance_m is None:
            self._ctrl.open_gripper()
            return
        span = abs(self._grip_open - self._grip_closed)
        if span <= 0.0 or self._max_width <= 0.0:
            self._ctrl.open_gripper()
            return
        frac = float(np.clip(distance_m / self._max_width, 0.0, 1.0))
        pos = self._grip_closed + frac * span
        self._ctrl.set_gripper_target(pos, effort=effort)

    def close_gripper_two_stage(
        self,
        width_pos_stage1: float = 0.5,
        width_pos_stage2: float = 0.85,
        effort: float = 0.6,
        stall_vel: float = 0.05,
        timeout_s: float = 3.0,
    ) -> Optional[float]:
        """Two-stage close to ABSOLUTE position fractions, with stall
        detection. SDK doesn't have this; wrc_demo owns it.

        ``width_pos_stage1/2`` are fractions of full close travel in
        [0..1] (0 = open, 1 = closed). They are typically computed
        width-aware by ``grasping.force.compute_close_target`` so the
        stage 2 seat lands at ``grasp.width_m * profile.object_close_frac``
        instead of a fixed 0.85 (= 90% of the rebot_rs travel, ~8 mm
        opening — that crushed 3D-printed bananas).

        Returns the final gripper width fraction (0=closed, 1=open) or
        None on stall timeout. Polls ``get_velocities()`` for the gripper
        joint; two consecutive samples below ``stall_vel`` → object
        grasped.
        """
        if not getattr(self._arm, "has_gripper", False):
            return None

        pos1 = self._grip_open + (self._grip_closed - self._grip_open) * width_pos_stage1
        pos2 = self._grip_open + (self._grip_closed - self._grip_open) * width_pos_stage2

        # stage 1: scout
        self.set_gripper(pos1, effort=effort * 0.7)
        time.sleep(0.3)
        # stage 2: seat
        self.set_gripper(pos2, effort=effort)

        deadline = time.monotonic() + timeout_s
        slow_samples = 0
        while time.monotonic() < deadline:
            vel = float(self._arm.get_velocities()[-1])
            if abs(vel) < stall_vel:
                slow_samples += 1
                if slow_samples >= 2:
                    return self._gripper_width_frac()
            else:
                slow_samples = 0
            time.sleep(0.02)
        return None

    # ── state ───────────────────────────────────────────────────────

    def get_state(self) -> RobotState:
        q, qd, tau = self._arm.get_state()
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        gripper_valid = bool(
            getattr(self._arm, "has_gripper", False) and q.size > self._n
        )
        return RobotState(
            q=q[: self._n],
            dq=qd[: self._n],
            tau=tau[: self._n],
            gripper_pos=float(q[self._n]) if gripper_valid else 0.0,
            gripper_valid=gripper_valid,
            t=time.monotonic(),
        )

    # ── soft stop / resume ──────────────────────────────────────────

    def stop(self) -> None:
        """Soft stop: pause the SDK's send thread, latch harness e-stop.

        Crucially does NOT call ``RebotArm.estop()`` (which is
        ``disable_all`` → torque off → free-fall under gravity).
        """
        self._h.estop(reason="safe_arm.stop()")
        try:
            self._ctrl._stop_send.set()
        except Exception:
            pass

    def resume(self) -> None:
        """Reverse of ``stop()``: clear e-stop + re-enable trajectory streaming."""
        self._h.reset_estop()
        try:
            self._ctrl._stop_send.clear()
        except Exception:
            pass

    def _close_end_effector(self) -> None:
        end_effector = getattr(self, "end_effector", None)
        if end_effector is not None:
            try:
                end_effector.close()
            except Exception:
                pass

    def disconnect(self) -> None:
        """Stop the SDK control loop + disconnect the bus.

        Idempotent. Safe to call from ``shutdown_runtime`` whether the
        arm is still streaming or already idle.
        """
        try:
            self._ctrl.end()
        except Exception:
            pass
        try:
            self._arm.disconnect()
        except Exception:
            pass
        self._close_end_effector()

    def disconnect_gently(self, duration_s: float = 3.0) -> None:
        """Gradually remove MIT support before disabling the RS motors.

        ``RebotArm.disconnect()`` disables all joints immediately, producing
        a sharp torque step and possible free-fall.  For a normal application
        shutdown, stop the 500 Hz owner loop, hold its last target, and fade
        position stiffness, damping, and gravity feed-forward to zero with a
        smoothstep envelope.  This is deliberately *not* used by emergency
        stop, whose priority is stopping motion immediately.

        The arm is unpowered at the end and can still settle under gravity;
        the operator must keep it supported and the workspace clear.
        """
        ctrl = self._ctrl
        raw_arm = self._arm
        arm_group = getattr(ctrl, "_arm_group", None)
        if not getattr(ctrl, "_running", False) or arm_group is None:
            self.disconnect()
            return

        try:
            # Normal shutdown first returns to the SDK's mechanical zero
            # using its minimum-jerk safe_home trajectory. This is distinct
            # from the configured workspace home_q used by `go home`.
            print(
                "[wrc-demo] returning arm to mechanical zero before shutdown...",
                file=sys.stderr,
            )
            ctrl.safe_home()
            raw_arm.stop_control_loop()
            n = int(getattr(ctrl, "_n", len(ctrl._q_target)))
            q_hold = np.asarray(ctrl._q_target[:n], dtype=float).copy()
            kp0 = np.asarray(arm_group._mit_kp, dtype=float).copy()
            kd0 = np.asarray(arm_group._mit_kd, dtype=float).copy()

            tau_g = np.zeros(n, dtype=float)
            if bool(getattr(ctrl, "_use_gravity_ff", False)):
                from reBotArm_control_py.dynamics import compute_generalized_gravity
                from reBotArm_control_py.kinematics.robot_model import pad_q_for_model

                q_model = pad_q_for_model(ctrl._model, q_hold, n)
                tau_g = np.asarray(
                    compute_generalized_gravity(ctrl._model, q_model, ctrl._data)[:n],
                    dtype=float,
                )
                gravity_scale = np.asarray(
                    getattr(ctrl, "_gravity_ff_scale", np.ones(n)), dtype=float,
                )[:n]
                tau_g *= gravity_scale

            duration_s = max(float(duration_s), 0.1)
            steps = max(2, int(duration_s * 50.0))
            print(
                f"[wrc-demo] gently releasing motor support over {duration_s:.1f}s; "
                "support the arm...",
                file=sys.stderr,
            )
            for i in range(steps):
                s = (i + 1) / steps
                scale = 1.0 - (3.0 * s * s - 2.0 * s * s * s)
                arm_group.send_mit(
                    q_hold,
                    vel=np.zeros(n),
                    kp=kp0 * scale,
                    kd=kd0 * scale,
                    tau=tau_g * scale,
                )
                time.sleep(duration_s / steps)

            gripper_group = getattr(ctrl, "_gripper_group", None)
            if gripper_group is not None and getattr(raw_arm, "has_gripper", False):
                gripper_group.send_mit(
                    np.array([float(getattr(ctrl, "_gripper_target", 0.0))]),
                    kp=np.zeros(gripper_group.num_joints),
                    kd=np.zeros(gripper_group.num_joints),
                    tau=np.zeros(gripper_group.num_joints),
                )
        except Exception as exc:
            print(
                f"[wrc-demo] gentle release failed ({exc!r}); disconnecting now",
                file=sys.stderr,
            )
        finally:
            self._close_end_effector()
            # Use the raw disconnect directly.  ctrl.end() calls safe_home(),
            # which would introduce an unexpected motion during shutdown.
            try:
                raw_arm.disconnect()
            except Exception:
                pass
            ctrl._running = False
            print("[wrc-demo] motors disabled and arm disconnected", file=sys.stderr)

    # ── convenience ─────────────────────────────────────────────────

    def home(self, duration_s: float = 3.0) -> bool:
        home_q = self._cfg.get("home_q", [0.0] * self._n)
        return self.move_joints(np.asarray(home_q, dtype=np.float64), duration_s)

    # ── internals ───────────────────────────────────────────────────

    def _make_hook(self, q_first: Optional[np.ndarray]):
        """Build the per-waypoint safety hook for the SDK's _send_loop.

        ``q_first`` is the very first waypoint (the current arm pose — the
        trajectory starts from wherever the arm is). The first hook call
        represents a no-motion transition (q_prev == q_next == start) and
        we skip the per-waypoint harness check for it: the current pose
        is known to be safe (the arm is in it), and the static-pose check
        would otherwise veto a perfectly valid trajectory that just
        happens to start near a limit (e.g. joint2 at 0.004 rad when
        the URDF lower limit is 0).
        """
        h = self._h
        sa = self
        dt = sa._dt
        prev = [None if q_first is None else q_first.copy()]
        first_seen = [False]

        def hook(q_i: np.ndarray) -> bool:
            if h.estopped:
                return False
            if not first_seen[0]:
                # First waypoint = current pose. Record and approve
                # without running the per-waypoint static-pose check
                # (see docstring above).
                prev[0] = q_i.copy()
                first_seen[0] = True
                return True
            qp = prev[0]
            if not h.approve(qp, q_i, dt=dt):
                # Harness vetoed: latch estop, stash the reason so the
                # main-thread caller can re-raise after wait_motion_done.
                reason = h.violations[-1] if h.violations else "harness veto"
                sa._last_violation = SafetyViolation(reason)
                h.estop(reason=reason)
                return False
            prev[0] = q_i.copy()
            return True
        return hook

    def _finalize_motion(self, ok: bool) -> bool:
        """After wait_motion_done: re-raise any stashed violation."""
        v = self._last_violation
        self._last_violation = None
        if v is not None:
            raise v
        return ok

    def _min_jerk(self, q_start: np.ndarray, q_end: np.ndarray, duration: float) -> list:
        n = max(2, int(duration * self._waypoint_hz))
        t = np.linspace(0.0, duration, n)
        s = t / duration
        blend = 10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5
        return [q_start + (q_end - q_start) * b for b in blend]

    def _pad(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] >= self._n:
            return q[: self._n]
        return np.concatenate([q, np.zeros(self._n - q.shape[0])])

    def _gripper_width_frac(self) -> float:
        """0=closed, 1=open from current gripper joint angle."""
        q, _, _ = self._arm.get_state()
        if not getattr(self._arm, "has_gripper", False) or len(q) <= self._n:
            return 0.0
        span = self._grip_closed - self._grip_open
        if abs(span) < 1e-9:
            return 0.0
        closed_frac = (float(q[self._n]) - self._grip_open) / span
        return float(np.clip(1.0 - closed_frac, 0.0, 1.0))

    @staticmethod
    def _se3_to_rpy(T: np.ndarray) -> tuple[float, float, float]:
        import pinocchio as pin
        rpy = pin.rpy.matrixToRpy(T[:3, :3])
        return float(rpy[0]), float(rpy[1]), float(rpy[2])
