"""SE(3) geodesic trajectory planning with CLIK joint tracking.

Wraps ``reBotArm_control_py.trajectory`` to produce safety-vettable
joint-space waypoint lists from Cartesian start/end poses.  The output
is consumed by :class:`SafeArm.move_joints_path` which streams each
segment through min-jerk interpolation with per-waypoint harness
approval — the planner never talks to motors directly.

Design notes
------------
* The model (``kin.model``) is already loaded (with the local frame
  convention baked in at load time), so the planner speaks the same
  convention as FK/IK. We deliberately don't import pinocchio here —
  the harness's :class:`Kinematics` is loaded via the SDK and exposes
  ``model``/``data`` but no ``_pin`` module handle, so we pass
  ``4×4 ndarrays`` straight through to the trajectory sampler's
  ``_se3_interpolate`` (which converts them internally) instead of
  wrapping them in ``pin.SE3``.
* Callers always pass and receive plain numpy arrays.
* CLIK tracking tolerates a configurable fraction of unconverged
  points — near-singular configurations (wrist limits on the B601-RS)
  occasionally fail to converge but the surrounding waypoints are
  still valid.  If the failure ratio exceeds the budget the whole
  trajectory is rejected and the caller falls back to PTP.
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    from reBotArm_control_py.trajectory import (
        plan_cartesian_geodesic_trajectory,
        track_trajectory,
        TrajPlanParams,
        TrajProfile,
        IKParams,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from ..control.kinematics import Kinematics

_PROFILES = {
    "linear": "LINEAR",
    "min_jerk": "MIN_JERK",
    "trapezoid": "TRAPEZOID",
}


class CartesianPlanner:
    """SE(3) geodesic + CLIK trajectory planner bound to a Kinematics model."""

    def __init__(
        self,
        kin: Kinematics,
        dt: float = 0.02,
        profile: str = "min_jerk",
        null_gain: float = 0.1,
    ) -> None:
        if not _AVAILABLE:
            raise RuntimeError(
                "reBotArm_control_py.trajectory is not importable; "
                "install reBotArm_control_py or set sdk_path"
            )
        self.kin = kin
        self._dt = dt
        if profile not in _PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {list(_PROFILES)}")
        self._profile = TrajProfile[_PROFILES[profile]]
        self._null_gain = null_gain

    # ── SE(3) conversion helpers ─────────────────────────────────────────
    #
    # Identity passthroughs: the vendored SDK's ``_se3_interpolate``
    # (in ``trajectory.sampler``) already promotes ``4×4 ndarray``
    # inputs to ``pin.SE3`` internally, so we don't need any pinocchio
    # wrapper here. We deliberately don't import pinocchio in this
    # module because the harness's :class:`Kinematics` is loaded via
    # the SDK and exposes ``model``/``data``/``fid`` but no ``_pin``
    # module handle.

    def _to_se3(self, T: np.ndarray) -> np.ndarray:
        return np.asarray(T, dtype=float)

    def _from_se3(self, T: np.ndarray) -> np.ndarray:
        return np.asarray(T, dtype=float)

    # ── core planning ────────────────────────────────────────────────────

    def plan(
        self,
        q_start: np.ndarray,
        T_start: np.ndarray,
        T_end: np.ndarray,
        duration: float = 2.0,
        max_fail_ratio: float = 0.05,  # 2026-08-13: tighter default; partial-IK-failure trajectories must not silently move the arm short of the target
    ) -> List[np.ndarray]:
        """Plan an SE(3) geodesic trajectory and return joint waypoints.

        Parameters
        ----------
        q_start : (n,) array
            Starting joint configuration.
        T_start, T_end : (4, 4) arrays
            Start and goal TCP poses in the base frame.
        duration : float
            Total trajectory time in seconds (must be > 0).
        max_fail_ratio : float
            Maximum fraction of CLIK-unconverged waypoints before the
            whole trajectory is rejected.

        Returns
        -------
        list of (n,) arrays
            Joint-space waypoints, one per CLIK sample.

        Raises
        ------
        RuntimeError
            If CLIK tracking fails entirely or exceeds *max_fail_ratio*.
        """
        if duration <= 0.0:
            raise ValueError("duration must be > 0")

        params = TrajPlanParams(dt=self._dt, profile=self._profile)
        cart_result = plan_cartesian_geodesic_trajectory(
            self._to_se3(T_start), self._to_se3(T_end), duration, params,
        )

        ik_params = IKParams(
            max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8,
        )
        # Pad q to model.nq if the arm has passive joints (e.g. finger
        # prismatic joints on the RS model: nq=8, n_controlled=6).
        # safety.harness.Kinematics 暴露 _model/_n_arm; control.kinematics.Kinematics 暴露 model/n
        # 兼容两种 Kinematics（safety.harness 是当前用的，control.kinematics 是已废弃）
        _model = getattr(self.kin, '_model', getattr(self.kin, 'model', None))
        _n_arm = getattr(self.kin, '_n_arm', getattr(self.kin, 'n', None))
        if _model is None or _n_arm is None:
            raise RuntimeError("Kinematics missing required attrs (model/_model and n/_n_arm)")
        q_init = np.zeros(_model.nq)
        q_init[:_n_arm] = np.asarray(q_start, dtype=float)[:_n_arm]
        joint_traj = track_trajectory(
            _model,
            getattr(self.kin, '_ee_frame_id', getattr(self.kin, 'fid', None)),
            cart_result.trajectory,
            q_init,
            ik_params,
            null_gain=self._null_gain,
        )

        total = len(joint_traj)
        if total == 0:
            raise RuntimeError("CLIK tracker produced no waypoints")

        failed = sum(1 for pt in joint_traj if not pt.ik_success)
        ratio = failed / total
        if ratio > max_fail_ratio:
            raise RuntimeError(
                f"CLIK tracking failed on {failed}/{total} waypoints "
                f"({ratio:.0%} > {max_fail_ratio:.0%}); trajectory rejected"
            )

        return [pt.q[:_n_arm].copy() for pt in joint_traj]

    def plan_linear(
        self,
        q_start: np.ndarray,
        direction: np.ndarray,
        distance_m: float,
        duration: float = 1.0,
    ) -> List[np.ndarray]:
        """Plan a short Cartesian translation from the current pose.

        Parameters
        ----------
        q_start : (n,) array
            Current joint configuration (used for FK to get T_start).
        direction : (3,) array
            Unit direction vector in the base frame.
        distance_m : float
            Travel distance in metres.
        duration : float
            Trajectory time in seconds.
        """
        T_start = self.kin.fk(q_start)
        T_end = T_start.copy()
        T_end[:3, 3] += np.asarray(direction, dtype=float) * distance_m
        return self.plan(q_start, T_start, T_end, duration)

    def plan_to_pose(
        self,
        q_start: np.ndarray,
        T_target: np.ndarray,
        duration: float = 2.0,
    ) -> List[np.ndarray]:
        """Convenience: plan from *q_start* to an absolute target pose."""
        T_start = self.kin.fk(q_start)
        return self.plan(q_start, T_start, T_target, duration)
