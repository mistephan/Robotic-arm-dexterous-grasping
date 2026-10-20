"""FK/IK on the arm model via Pinocchio.

Self-contained on purpose: reBotArm_control_py's kinematics module reads its
own global config/rebotarm.yaml (ignoring the hw_yaml passed to RebotArm), so
using it with the RS arm silently loads the DM model. Here the model path
comes from the arm profile explicitly. Only .urdf paths are supported (USD
support was dropped together with the Isaac Sim backend on 2026-08-28).

Both shipped RS assets are authored in the MIRRORED joint convention
(q_asset = -q_local); the SDK, the real arm and every q constant in this
repo use the local one. Arm profiles carry `joint_signs: [-1, ...]` and the
flip is baked into the model at load time, so everything downstream (FK, IK,
limits, safety) speaks local convention.

The RS model has nq=8 (6 revolute + 2 passive prismatic finger joints); we
command the first 6 and zero-pad the rest, same convention as the SDK.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IKResult:
    q: np.ndarray  # (n_controlled,) solution
    success: bool
    error: float
    iterations: int
    position_error_m: float = np.inf
    rotation_error_rad: float = np.inf


class Kinematics:
    def __init__(
        self,
        model_path: str,
        ee_frame: str,
        n_controlled: int = 6,
        joint_signs: list[int] | None = None,
    ):
        import pinocchio as pin  # heavy import, keep local

        self._pin = pin
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"robot model not found: {model_path}")
        if path.suffix in (".usd", ".usda"):
            raise ValueError(
                f"USD model assets are no longer supported (path: {model_path}); "
                f"use the shipped URDF under assets/urdf/ instead."
            )
        xml = path.read_text()
        if joint_signs and any(int(s) == -1 for s in joint_signs):
            # Bake joint_signs into the URDF XML: a -1 entry negates that
            # joint's axis and mirrors its limits, which is an exact
            # reparameterization of the same physical joint. Inlined here
            # (formerly usd_model.apply_joint_signs) because the only caller
            # is this loader and the dependency is now gone with the USD
            # backend removal (2026-08-28).
            root = ET.fromstring(xml)
            movable = [j for j in root.iter("joint")
                       if j.get("type") in ("revolute", "prismatic", "continuous")]
            for sign, joint in zip(joint_signs, movable):
                if int(sign) != -1:
                    continue
                axis = joint.find("axis")
                if axis is not None:
                    axis.set("xyz", " ".join(f"{-float(v):.9g}"
                                             for v in axis.get("xyz").split()))
                limit = joint.find("limit")
                if limit is not None:
                    lo, hi = float(limit.get("lower", 0)), float(limit.get("upper", 0))
                    limit.set("lower", f"{-hi:.9g}")
                    limit.set("upper", f"{-lo:.9g}")
            xml = ET.tostring(root, encoding="unicode")
        self.model = pin.buildModelFromXML(xml)
        self.data = self.model.createData()
        self.ee_frame = ee_frame
        self.fid = self.model.getFrameId(ee_frame)
        if self.fid >= len(self.model.frames.tolist()):
            raise ValueError(f"frame {ee_frame!r} not in model")
        self.n = n_controlled
        self.nq = self.model.nq
        lo = np.asarray(self.model.lowerPositionLimit[: self.n])
        hi = np.asarray(self.model.upperPositionLimit[: self.n])
        self.joint_limits = (lo, hi)

    def _pad(self, q: np.ndarray) -> np.ndarray:
        qf = np.zeros(self.nq)
        qf[: self.n] = np.asarray(q, dtype=float).reshape(-1)[: self.n]
        return qf

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(n,) joints -> 4x4 T_tcp2base."""
        pin = self._pin
        qf = self._pad(q)
        pin.forwardKinematics(self.model, self.data, qf)
        pin.updateFramePlacements(self.model, self.data)
        return np.array(self.data.oMf[self.fid].homogeneous)

    def clamp(self, q: np.ndarray, margin: float = 0.0) -> np.ndarray:
        lo, hi = self.joint_limits
        return np.clip(q, lo + margin, hi - margin)

    def ik(
        self,
        T_target: np.ndarray,
        q_init: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-4,
        damping: float = 1e-6,
        step: float = 0.5,
        retries: int = 4,
        seed: int = 0,
        limit_margin: float = 0.025,
    ) -> IKResult:
        """Damped least-squares IK with random restarts inside joint limits.

        `limit_margin` keeps solutions strictly inside the URDF limits so the
        safety harness (which enforces its own margin) never rejects a pose
        the planner called reachable.
        """
        pin = self._pin
        target = pin.SE3(np.asarray(T_target[:3, :3]), np.asarray(T_target[:3, 3]))
        rng = np.random.default_rng(seed)
        lo, hi = self.joint_limits
        best = IKResult(q=np.asarray(q_init, dtype=float)[: self.n].copy(),
                        success=False, error=np.inf, iterations=0)
        starts = [np.asarray(q_init, dtype=float)[: self.n].copy()]
        starts += [rng.uniform(lo + limit_margin, hi - limit_margin) for _ in range(retries)]
        for q0 in starts:
            res = self._ik_once(target, q0, max_iter, tol, damping, step, limit_margin)
            if res.error < best.error:
                best = res
            if res.success:
                return res
        return best

    def _ik_once(self, target, q0, max_iter, tol, damping, step, margin=0.0) -> IKResult:
        pin = self._pin
        q = self.clamp(q0.copy(), margin)
        err_norm = np.inf
        for it in range(max_iter):
            qf = self._pad(q)
            pin.forwardKinematics(self.model, self.data, qf)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.fid]
            err = pin.log6(oMf.actInv(target)).vector
            err_norm = float(np.linalg.norm(err))
            position_error_m, rotation_error_rad = self._split_pose_error(oMf, target)
            if err_norm < tol:
                return IKResult(
                    q=q.copy(), success=True, error=err_norm, iterations=it,
                    position_error_m=position_error_m,
                    rotation_error_rad=rotation_error_rad,
                )
            J = pin.computeFrameJacobian(
                self.model, self.data, qf, self.fid, pin.ReferenceFrame.LOCAL
            )[:, : self.n]
            lam = damping * max(1.0, err_norm * 10.0)
            JJt = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            q = self.clamp(q + step * dq, margin)

        # q was updated after the last error measurement. Recompute FK so
        # every diagnostic field describes the exact q returned to callers.
        qf = self._pad(q)
        pin.forwardKinematics(self.model, self.data, qf)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.fid]
        err_norm = float(np.linalg.norm(pin.log6(oMf.actInv(target)).vector))
        position_error_m, rotation_error_rad = self._split_pose_error(oMf, target)
        return IKResult(
            q=q.copy(), success=False, error=err_norm, iterations=max_iter,
            position_error_m=position_error_m,
            rotation_error_rad=rotation_error_rad,
        )

    def _split_pose_error(self, current, target) -> tuple[float, float]:
        position_error_m = float(
            np.linalg.norm(
                np.asarray(current.translation) - np.asarray(target.translation)
            )
        )
        rotation_error_rad = float(
            np.linalg.norm(
                self._pin.log3(
                    np.asarray(current.rotation).T @ np.asarray(target.rotation)
                )
            )
        )
        return position_error_m, rotation_error_rad

    def link_positions(self, q: np.ndarray) -> np.ndarray:
        """Base-frame positions of every joint frame -> (n_joints, 3).

        Used by the safety harness as coarse collision proxy spheres.
        """
        pin = self._pin
        qf = self._pad(q)
        pin.forwardKinematics(self.model, self.data, qf)
        return np.array([np.asarray(self.data.oMi[j].translation)
                         for j in range(1, self.model.njoints)])
