"""Frame-convention conversions for hand-eye calibration.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

wrc_demo uses Pinocchio FK with ``ee_frame='gripper_end'`` and
``joint_signs=[-1]*6``. The legacy cuRobo demos used a different
``end_link`` and an unreversed local convention; the round-trip helpers
here let the calibration tools compare/convert between the two. They
are pure numpy + small helpers (no arm SDK import) so they can be unit
tested in isolation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def is_se3(T: np.ndarray, tol: float = 1e-3) -> bool:
    """Sanity-check that ``T`` is on SE(3): last row (0,0,0,1) and R orthogonal."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=tol):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    return True


def round_trip_check(
    T_ab: np.ndarray,
    T_bc: np.ndarray,
    *,
    atol_rot_deg: float = 0.5,
    atol_trans_mm: float = 0.5,
) -> Tuple[bool, float, float]:
    """Compose two transforms and report rotation/translation drift.

    Useful as a one-line sanity check for an FK round-trip test
    (T_local_curobo -> T_wrc_demo). Returns ``(ok, rot_err_deg, trans_err_mm)``.
    The drift is measured against the identity matrix; pass the expected
    identity to verify a true round trip, or a constant S to verify a
    rigid offset is reproduced.
    """
    T = np.asarray(T_ab, dtype=float) @ np.asarray(T_bc, dtype=float)
    # Rotation drift via angle of the relative rotation.
    R_rel = T[:3, :3]
    cos_a = (np.trace(R_rel) - 1.0) / 2.0
    cos_a = max(-1.0, min(1.0, cos_a))
    rot_deg = float(np.degrees(np.arccos(cos_a)))
    trans_mm = float(np.linalg.norm(T[:3, 3])) * 1000.0
    return (
        rot_deg <= atol_rot_deg and trans_mm <= atol_trans_mm,
        rot_deg,
        trans_mm,
    )


def curobo_to_wrc_demo_marker_offset(
    T_gripper2base_curobo: np.ndarray,
    T_gripper2base_wrc_demo: np.ndarray,
) -> np.ndarray:
    """Compute the constant SE(3) offset ``S = T_w @ inv(T_c)`` such that
    ``T_w = S @ T_c`` for any same joint configuration.

    The two FK conventions on the same hardware differ only by a rigid
    offset (``S``): an end_link vs gripper_end convention shift and/or a
    joint-sign flip baked into the model. If FK outputs don't satisfy this
    property, the offsets are NOT a rigid body transform and the loader
    must refuse to bridge them.
    """
    T_c = np.asarray(T_gripper2base_curobo, dtype=float)
    T_w = np.asarray(T_gripper2base_wrc_demo, dtype=float)
    if not (is_se3(T_c) and is_se3(T_w)):
        raise ValueError("FK outputs must be SE(3) to compute a constant offset")
    inv_Tc = np.linalg.inv(T_c)
    return T_w @ inv_Tc
