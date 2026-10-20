"""Regression tests for defects found in the v3 adversarial review pass
(focused on the Orbbec hand-eye bring-up).

Each test corresponds to one confirmed defect with the root cause
recorded in its docstring. Convention from tests/test_review_regressions.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────
# v3 defect 1: ArUco corner order vs PnP object points
# ─────────────────────────────────────────────────────────────────────────
#
# Root cause: ``wrc_demo.calibration.aruco_session.ArUcoSession.detect``
# used the wrong 3D model points for cv2.solvePnP. cv2.aruco's marker
# frame is +x right, +y down (matching image space), +z toward the
# camera. The correct obj_pts in TL/TR/BR/BL order are:
#     TL: (-half, -half, 0)
#     TR: ( half, -half, 0)
#     BR: ( half,  half, 0)
#     BL: (-half,  half, 0)
# The original code used the y-up "printed marker" convention which is
# inverted on the y-axis. PnP then matched the wrong 3D point to each
# detected corner, producing a T_marker2cam that was off by a 180°
# rotation around x for axis-aligned markers. Effect on the hand-eye
# solve: the constant axis flip got absorbed into the joint fit (Y
# rotation absorbed the X flip), so RMSE looked fine but T_cam2base
# was rotated 180° around x.
#
# Companion fix: switched the PnP solver flag from
# SOLVEPNP_IPPE_SQUARE (returned tvec ≈ 0 for our warped synthetic
# images) to SOLVEPNP_SQPNP which is reliable for coplanar square
# markers.


def test_aruco_T_marker2cam_recovers_identity_for_axis_aligned_marker():
    """For a fronto-parallel axis-aligned marker pasted at a known
    image location, the recovered T_marker2cam rotation must be R = I
    (cv2.aruco's marker frame is +x right, +y down, +z INTO the board;
    with the camera looking toward +z, all three marker axes align with
    the camera axes).

    With the v3 corner-order bug, the recovered R was diag(1, -1, -1)
    -- a 180° rotation around x -- because the 3D model points were
    matched to the wrong detected corners.
    """
    from wrc_demo.calibration import build_aruco_session, is_se3

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    side_px = 200
    pad = 30
    marker_bmp = cv2.aruco.generateImageMarker(aruco_dict, 0, side_px)
    padded = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, dtype=np.uint8)
    padded[pad:pad + side_px, pad:pad + side_px] = marker_bmp

    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    img[100:360, 200:460] = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    R_expected = np.eye(3)

    session = build_aruco_session("4x4_50")
    det = session.detect(img, K, None, marker_size_m=0.10)
    assert det is not None
    assert is_se3(det.T_marker2cam)

    R_rec = det.T_marker2cam[:3, :3]
    R_rel = R_rec.T @ R_expected
    cos_a = max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0))
    rot_err_deg = float(np.degrees(np.arccos(cos_a)))
    # v3 defect would produce a 180° rotation around x (R = diag(1,-1,-1)).
    assert rot_err_deg < 2.0, (
        f"v3 defect: rotation off by {rot_err_deg:.2f}° "
        f"(expected ~180° if corner order is wrong)"
    )


# ─────────────────────────────────────────────────────────────────────────
# v3 defect 2: SOLVEPNP_IPPE_SQUARE returned tvec ≈ 0 on warped images
# ─────────────────────────────────────────────────────────────────────────
#
# Root cause: ``cv2.SOLVEPNP_IPPE_SQUARE`` is documented for 4-point
# coplanar square targets but its "first" pose estimate can collapse
# to the origin for synthetic images whose detected corners are not
# strictly axis-aligned. Switching to ``cv2.SOLVEPNP_SQPNP`` recovers
# a sensible pose. The fix lives in
# ``wrc_demo.calibration.aruco_session.ArUcoSession.detect``.


def test_aruco_T_marker2cam_translation_finite_and_nonzero():
    """The recovered translation must be a finite, non-degenerate
    3-vector (not the ~0 collapse SOLVEPNP_IPPE_SQUARE produced).
    """
    from wrc_demo.calibration import build_aruco_session

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    side_px = 200
    pad = 30
    marker_bmp = cv2.aruco.generateImageMarker(aruco_dict, 0, side_px)
    padded = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, dtype=np.uint8)
    padded[pad:pad + side_px, pad:pad + side_px] = marker_bmp
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    img[100:360, 200:460] = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    session = build_aruco_session("4x4_50")
    det = session.detect(img, K, None, marker_size_m=0.10)
    assert det is not None
    t = det.T_marker2cam[:3, 3]
    # The marker is in front of the camera (>0 z).
    assert t[2] > 0.1, f"v3 defect: degenerate tvec={t} (SOLVEPNP_IPPE_SQUARE collapse)"
    assert np.isfinite(t).all()


# ─────────────────────────────────────────────────────────────────────────
# v3 defect 3: Extrinsics.from_config silent-identity fallback for a
# missing extrinsics source. Pinned as a documentation test rather than
# a code change (the fix lives in apps/demo.py:build_runtime which
# inspects the profile and forces fuse=False when the source is absent;
# see PLAN.md §ADR-5 and the requires_calibration guard).
# ─────────────────────────────────────────────────────────────────────────


def test_extrinsics_loader_does_not_silently_swallow_missing_source():
    """A profile that names an extrinsics source path that doesn't exist
    must not yield an identity Extrinsics object marked as the named
    source -- the caller's requires_calibration guard depends on
    ``source == 'inline'`` to detect the gap.
    """
    from wrc_demo.config import Cfg
    from wrc_demo.perception.grounding import Extrinsics

    cfg = Cfg({
        "mode": "eye_to_hand",
        "source": "data/calibration/does_not_exist.json",
        # No inline T block.
    })
    e = Extrinsics.from_config(cfg)
    # Must be flagged 'inline' so the caller's missing-extrinsics
    # branch fires (see apps/demo.py:build_runtime).
    assert e.source == "inline"
    # Default identity matrix.
    assert np.allclose(e.T, np.eye(4))
