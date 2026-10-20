"""ArUco detection + PnP for hand-eye calibration.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

Thin wrapper over ``cv2.aruco`` that returns the 4x4 marker-to-camera
matrix used as ``M_i`` in the hand-eye equation. The legacy cuRobo
``04b_top_calibration`` script does this with a state machine attached;
here we keep the math pure (a function, not a state machine) so the
test suite can exercise it without a CLI / camera / arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


_ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16H5,
}


@dataclass(frozen=True)
class ArUcoDetection:
    """One successful ArUco detection in a single frame."""

    marker_id: int
    corners_px: np.ndarray  # (4, 2) image-space corners, ordered BL/BR/TR/TL
    T_marker2cam: np.ndarray  # (4, 4) SE(3) marker->camera transform
    reprojection_rmse_px: float


def _resolve_dict(name: str) -> int:
    if name not in _ARUCO_DICTS:
        raise ValueError(
            f"unknown aruco dict {name!r}; valid: {sorted(_ARUCO_DICTS)}"
        )
    return _ARUCO_DICTS[name]


def build_aruco_session(dict_name: str = "4x4_50") -> "ArUcoSession":
    return ArUcoSession(dict_name=dict_name)


class ArUcoSession:
    """cv2.aruco wrapper that yields ``ArUcoDetection`` instances."""

    def __init__(self, dict_name: str = "4x4_50") -> None:
        self._dict_name = dict_name
        self._dict_id = _resolve_dict(dict_name)
        self._dictionary = cv2.aruco.getPredefinedDictionary(self._dict_id)
        params = cv2.aruco.DetectorParameters()
        # Corner refinement helps when the marker is far or motion-blurred.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._params = params
        try:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
            self._new_api = True
        except AttributeError:
            self._detector = None
            self._new_api = False

    @property
    def dict_name(self) -> str:
        return self._dict_name

    def _detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                bgr, self._dictionary, parameters=self._params
            )
        if ids is None or len(ids) == 0:
            return None
        return corners, ids, None

    def detect(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray | None,
        marker_size_m: float,
        target_id: int | None = None,
    ) -> ArUcoDetection | None:
        """Run ArUco detection + PnP; return the first (or matching) detection.

        ``K`` is 3x3 fx,fy,cx,cy. ``D`` may be None for an undistorted
        stream. ``marker_size_m`` is the physical marker side length; the
        convention here is the marker lies in the z=0 plane with the
        origin at the marker center.
        """
        out = self._detect(bgr)
        if out is None:
            return None
        corners_list, ids, _ = out
        pick_idx = 0
        if target_id is not None:
            for i, mid in enumerate(ids.flatten().tolist()):
                if int(mid) == int(target_id):
                    pick_idx = i
                    break
            else:
                return None
        corners = np.asarray(corners_list[pick_idx]).reshape(4, 2).astype(np.float64)
        mid = int(ids.flatten()[pick_idx])
        half = float(marker_size_m) / 2.0
        # cv2.aruco.detectMarkers returns corners in TL/TR/BR/BL order
        # (image-space). The 3D model points must mirror the same order
        # in the marker frame cv2.aruco uses internally: +x right, +y
        # down (matching image), +z out of the board toward the camera.
        obj_pts = np.array(
            [
                [-half, -half, 0.0],  # top-left of the marker in image space
                [ half, -half, 0.0],  # top-right
                [ half,  half, 0.0],  # bottom-right
                [-half,  half, 0.0],  # bottom-left
            ],
            dtype=np.float64,
        )
        if D is None or np.asarray(D).size == 0:
            D = np.zeros((5,), dtype=np.float64)
        # SOLVEPNP_SQPNP is reliable for coplanar square markers; we tried
        # SOLVEPNP_IPPE_SQUARE first because it claims to be a Coplanar
        # + 4-points special case but its "first" solution came back as
        # tvec ≈ 0 on real rigs (PLAN.md §Q3 review note: v3 defect;
        # fixed by switching to SQPNP).
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, corners, np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).reshape(-1),
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(tvec).reshape(3)
        # Reprojection RMSE
        proj, _ = cv2.projectPoints(
            obj_pts, rvec, tvec,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).reshape(-1),
        )
        proj = proj.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - corners) ** 2, axis=1))))
        return ArUcoDetection(
            marker_id=mid,
            corners_px=corners,
            T_marker2cam=T,
            reprojection_rmse_px=rmse,
        )

    def draw_aruco_overlay(self, bgr: np.ndarray, det: ArUcoDetection) -> np.ndarray:
        """Render a thin overlay so the operator can verify the detection."""
        out = bgr.copy()
        pts = det.corners_px.astype(int)
        for i in range(4):
            cv2.line(out, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"id={det.marker_id} rmse={det.reprojection_rmse_px:.2f}px",
            (int(pts[0][0]), int(pts[0][1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        return out
