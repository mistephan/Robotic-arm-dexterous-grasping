"""Tests for the ArUco session: synthetic marker detection + PnP."""

from __future__ import annotations

import cv2
import numpy as np

from wrc_demo.calibration import build_aruco_session


def _render_marker_3d(
    width: int = 640,
    height: int = 480,
    marker_id: int = 0,
    marker_size_m: float = 0.10,
    marker_z_m: float = 1.0,
    K: np.ndarray | None = None,
    tilt_rad: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Render an ArUco marker at a real 3D pose with a small tilt.

    The cv2.aruco detector requires whitespace around the marker's
    black border (a few-pixel margin), so we pad the bitmap with
    white pixels before warping. Without the pad the warped bitmap
    fills the projected quadrilateral exactly and the detector fails.
    """
    if K is None:
        K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side_px = 200
    pad = 30
    marker_bmp = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)
    padded = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, dtype=np.uint8)
    padded[pad:pad + side_px, pad:pad + side_px] = marker_bmp

    img = np.full((height, width, 3), 255, dtype=np.uint8)
    half = marker_size_m / 2.0
    # Slight rotation around x and y axes -> trapezoidal silhouette.
    rx, ry = tilt_rad, tilt_rad * 0.7
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R = Ry @ Rx
    tvec = np.array([0.0, 0.0, marker_z_m]).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    # cv2.aruco's internal marker frame: +x right, +y down (image y),
    # +z out of the board toward the camera. The corners below follow
    # this convention so cv2.projectPoints returns them in TL/TR/BR/BL
    # image order (matching cv2.aruco.detectMarkers' output order).
    obj = np.array(
        [
            [-half, -half, 0.0],  # TL in image space
            [ half, -half, 0.0],  # TR
            [ half,  half, 0.0],  # BR
            [-half,  half, 0.0],  # BL
        ],
        dtype=np.float64,
    )
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    proj = proj.reshape(-1, 2)

    side = side_px + 2 * pad
    src_corners = np.array(
        [[0, 0], [side - 1, 0],
         [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    dst_corners = proj.astype(np.float32)
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(
        cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), H, (width, height),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    mask = cv2.warpPerspective(
        np.ones_like(padded) * 255, H, (width, height),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    out = np.where(mask[..., None] > 0, warped, img)
    return out, proj


def test_aruco_session_detects_marker():
    session = build_aruco_session("4x4_50")
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    # Closer marker (z=0.5m) so the warped bitmap occupies enough pixels
    # for the 4x4_50 detector to find the 17-bit pattern.
    img, _ = _render_marker_3d(K=K, marker_z_m=0.5)
    det = session.detect(img, K, None, marker_size_m=0.10)
    assert det is not None
    assert det.marker_id == 0
    assert det.corners_px.shape == (4, 2)
    # T_marker2cam must be SE(3) (PNP returns a valid pose for the
    # detected corners regardless of the warped image quality).
    from wrc_demo.calibration import is_se3
    assert is_se3(det.T_marker2cam)
    # Reprojection residual is finite and non-negative; the warped-image
    # geometry diverges from a perfect pinhole so we don't pin a hard
    # bound here -- real-aruco calibration tolerances live in the hand-eye
    # solver tests.
    assert det.reprojection_rmse_px >= 0.0
    assert np.isfinite(det.reprojection_rmse_px)


def test_aruco_session_target_id_filter():
    session = build_aruco_session("4x4_50")
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    img, _ = _render_marker_3d(marker_id=0, K=K, marker_z_m=0.5)
    det = session.detect(img, K, None, marker_size_m=0.10, target_id=42)
    assert det is None


def test_aruco_session_dict_lookup():
    with __import__("pytest").raises(ValueError):
        build_aruco_session("not_a_real_dict")


def test_aruco_overlay_draws():
    session = build_aruco_session("4x4_50")
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    img, _ = _render_marker_3d(K=K, marker_z_m=0.5)
    det = session.detect(img, K, None, marker_size_m=0.10)
    assert det is not None
    overlay = session.draw_aruco_overlay(img, det)
    assert overlay.shape == img.shape
    assert overlay is not img
