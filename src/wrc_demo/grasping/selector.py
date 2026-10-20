"""Grasp selection: filter candidates by physics, then by reachability.

A grasp is only offered to the executor if BOTH the grasp pose and its
pregrasp pose solve under IK inside the workspace -- the baseline's "IK walk"
idea, kept, but applied to every candidate up front so the agent gets an
honest 'nothing reachable' instead of a mid-motion abort.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Grasp, SkillError, make_transform


def select_grasp(
    grasps: list[Grasp],
    kin,
    q_current: np.ndarray,
    max_width_m: float = 0.09,
    pregrasp_offset_m: float = 0.12,
    validate=None,
    ik_retries: int = 3,
    rank_key=None,
    prefilter=None,
    debug_callback=None,
    parallel_jaw_symmetry: bool = False,
    wrist_motion_weight: float = 2.0,
    max_wrist_delta_rad: float | None = None,
    deadline: float | None = None,
) -> tuple[Grasp, np.ndarray, np.ndarray]:
    """Return the best executable candidate and its two IK solutions.

    ``debug_callback(event)`` is an optional observation-only hook. Events
    contain the candidate object, stable rank/index, the exact q_current seed,
    IK results and rejection reason. Callback failures are deliberately ignored
    so diagnostics can never change robot planning behaviour.
    """
    reasons: list[str] = []

    def check_deadline() -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise SkillError("grasp planning deadline exceeded")

    check_deadline()
    ranked = sorted(
        grasps,
        key=(
            (lambda g: (rank_key(g), -g.quality))
            if rank_key is not None else (lambda g: -g.quality)
        ),
    )

    def solve_ik(target, seed):
        """Pass retry policy when supported, while keeping simple IK stubs valid."""
        try:
            return kin.ik(target, seed, retries=ik_retries)
        except TypeError as exc:
            # Some lightweight kinematics implementations expose the older
            # ik(target, seed) contract. Do not hide errors inside a solver.
            if "retries" not in str(exc):
                raise
            return kin.ik(target, seed)

    def emit(candidate, index, stage, *, pre=None, grasp=None, reason=None):
        if debug_callback is None:
            return
        event = {
            "candidate": candidate,
            "candidate_index": index,
            "candidate_count": len(ranked),
            "stage": stage,
            "q_current": np.asarray(q_current, dtype=float).copy(),
            "max_width_m": float(max_width_m),
            "pregrasp": pre,
            "grasp": grasp,
            "reject_reason": reason,
        }
        try:
            debug_callback(event)
        except Exception:
            pass

    def motion_cost(q_pre, q_grasp):
        q0 = np.asarray(q_current, dtype=float)
        qp = np.asarray(q_pre, dtype=float)
        qg = np.asarray(q_grasp, dtype=float)
        weights = np.ones_like(qp)
        if len(weights):
            weights[-1] = max(float(wrist_motion_weight), 1.0)
        d_pre = np.arctan2(np.sin(qp - q0), np.cos(qp - q0))
        d_grasp = np.arctan2(np.sin(qg - qp), np.cos(qg - qp))
        return float(np.linalg.norm(weights * d_pre) + 0.25 * np.linalg.norm(d_grasp))

    for index, g in enumerate(ranked, start=1):
        check_deadline()
        emit(g, index, "candidate")
        if prefilter is not None:
            reason = prefilter(g)
            if reason:
                reasons.append(str(reason))
                emit(g, index, "rejected", reason=str(reason))
                continue
        if g.width_m > max_width_m:
            reason = (
                f"{g.label}: required width {g.width_m * 1000:.0f}mm > gripper "
                f"max {max_width_m * 1000:.0f}mm (consider push or regrasp)"
            )
            reasons.append(reason)
            emit(g, index, "rejected", reason=reason)
            continue
        rotations = [(np.asarray(g.rotation, dtype=float), False)]
        if parallel_jaw_symmetry:
            flipped = np.asarray(g.rotation, dtype=float).copy()
            flipped[:, 1:] *= -1.0  # 180 degrees about approach/tool-x
            rotations.append((flipped, True))

        solved = []
        variant_reasons = []
        last_pre = last_grasp = None
        for rotation, flipped in rotations:
            check_deadline()
            T_grasp = make_transform(rotation, g.position)
            T_pre = make_transform(rotation, g.pregrasp_position(pregrasp_offset_m))
            pre = solve_ik(T_pre, q_current)
            check_deadline()
            last_pre = pre
            if not pre.success:
                variant_reasons.append(f"pregrasp IK failed (err {pre.error:.4f})")
                continue
            if max_wrist_delta_rad is not None and len(pre.q):
                wrist_delta = abs(float(np.arctan2(
                    np.sin(pre.q[-1] - q_current[-1]),
                    np.cos(pre.q[-1] - q_current[-1]),
                )))
                if wrist_delta > float(max_wrist_delta_rad):
                    variant_reasons.append(
                        "pregrasp wrist travel "
                        f"{wrist_delta:.3f}rad exceeds "
                        f"{float(max_wrist_delta_rad):.3f}rad"
                    )
                    continue
            grasp = solve_ik(T_grasp, pre.q)
            check_deadline()
            last_grasp = grasp
            if not grasp.success:
                variant_reasons.append(f"grasp IK failed (err {grasp.error:.4f})")
                continue
            if validate is not None:
                reason = validate(g, pre.q, grasp.q)
                if reason:
                    variant_reasons.append(f"{g.label}: {reason}")
                    continue
            solved.append((motion_cost(pre.q, grasp.q), rotation, flipped, pre, grasp))

        if not solved:
            reason = "; ".join(variant_reasons[:2]) or "no IK solution"
            reasons.append(reason)
            emit(g, index, "rejected", pre=last_pre, grasp=last_grasp, reason=reason)
            continue
        _, chosen_rotation, flipped, pre, grasp = min(solved, key=lambda item: item[0])
        g.rotation = chosen_rotation.copy()
        g.approach = chosen_rotation[:, 0].copy()
        g._symmetry_flipped = bool(flipped)
        emit(g, index, "pregrasp_ok", pre=pre)
        emit(g, index, "selected", pre=pre, grasp=grasp)
        return g, pre.q, grasp.q
    if debug_callback is not None:
        try:
            debug_callback({
                "stage": "exhausted",
                "reject_reason": "; ".join(reasons[:4]),
            })
        except Exception:
            pass
    raise SkillError("no executable grasp: " + "; ".join(reasons[:4]))
