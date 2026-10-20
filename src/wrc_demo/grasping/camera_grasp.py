"""Camera-frame grasp planning — borrowed from rebot_grasp's algorithm.

Pipeline (mirrors ``rebot_grasp/utils/ordinary_grasp.py::estimate_grasp``
and ``rebot_grasp/utils/transforms.py::transform_grasp_pose_to_base``):

  2D mask  ──minAreaRect──>  rect (4 corners) + short-edge direction
     │
     ▼
  depth quantile (median by default) ──backproject──>  position_cam (3D, camera frame)
     │
     ▼
  approach_cam = -normalize(position_cam)         # toward camera origin
  open_axis_cam  = backproject(short_dir) − proj_onto(approach_cam)
  grip_axis_cam  = cross(open_axis, approach)      # right-handed
     │
     ▼
  vision_R = [grip_axis, open_axis, approach]     # in camera frame
  tcp_R    = grasp_axes_to_rebot_tcp_rotation(…)  # X=tool-forward, Y=open
     │
     ▼
  T_grasp_base = T_cam2base @ T_grasp_cam
  T_grasp_base = canonicalize_parallel_gripper_tcp_rotation(T_grasp_base)
     │
     ▼
  T_grasp_base[:, 3] -= insertion_depth * R[:, 0]  # push TCP into the object
                                                   # by the gripper's tip offset

Returns a list of ``Grasp`` instances whose ``approach`` is the tool-forward
direction in the **base** frame, so the existing ``Grasp.pregrasp_position``
(``position − approach * offset``) and the IK seed-from-home flow in
``select_grasp`` work without modification.

Why this replaces the analytic OBB planner:

  * ``approach = -normalize(position_cam)`` aligns the TCP dive-in direction
    with the camera's view ray. For the B601's 29° tilted top camera, this
    means the gripper dives along the actual line of sight — not along a
    hard-coded base-frame vector. The OBB planner had to fake this with
    ``approach_pitch_rad: 1.2`` and was off-axis whenever the camera wasn't
    exactly above the workspace.
  * ``insertion_depth_m: 0.015`` shifts the grasp 1.5 cm INTO the object,
    so the parallel-jaw fingers (which extend past the TCP) actually wrap
    around the mass — the OBB planner had the TCP exactly on the surface,
    which the rebot_grasp team confirmed causes ``air grasp`` failures on
    shallow boxes.
  * 2D OBB on the segmentation mask is much more stable than backprojecting
    4000 depth points into a 3D OBB in base frame, because the mask lives
    entirely in pixel space (no depth noise), and the depth used is a single
    quantile statistic — not 4000 noisy samples.

The OBB planner (``plan_grasps_from_fix``) stays as the fallback for the
``Detection.mask is None`` / ``frame.depth_m is None`` cases (mock camera,
mono-only cameras, hardware bring-up before segmentation weights land).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

from ..types import Grasp

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration constants (mirroring rebot_grasp config/default.yaml)
# ──────────────────────────────────────────────────────────────────────

#: Push the grasp TCP this far INTO the object along the tool-x axis so the
#: gripper fingers (which extend ~1.5 cm past the TCP) wrap the mass
#: instead of pinching the surface.
DEFAULT_INSERTION_DEPTH_M = 0.015

#: Default depth sampling quantile (median) — robust to depth holes / tails.
DEFAULT_DEPTH_QUANTILE = 0.50

#: Pad added to the measured short-edge jaw span before comparing against
#: the gripper's max opening. 1.5 cm matches rebot_grasp's selector.
DEFAULT_WIDTH_PAD_M = 0.015


# ──────────────────────────────────────────────────────────────────────
# Low-level math helpers (copied from rebot_grasp with minimal adaptation)
# ──────────────────────────────────────────────────────────────────────


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    """Unit-normalise; return None for near-zero inputs."""
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return (v / n).astype(np.float64)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation onto SO(3) via SVD."""
    R = np.asarray(R, dtype=np.float64)
    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix has non-finite entries")
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float64)


def _backproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    """Pixel (u, v) at metric depth z_m → camera-frame 3D point.

    Camera convention: +x right, +y down, +z forward (Orbbec / OpenCV).
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float64)


def _pixel_vec_to_3d(vec_uv: np.ndarray, z_m: float, K: np.ndarray) -> np.ndarray:
    """A 2D pixel displacement → a 3D camera-frame displacement at depth z_m.

    Used to convert the OBB's short-edge direction into a 3D direction.
    """
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    return np.array(
        [float(vec_uv[0]) * z_m / fx, float(vec_uv[1]) * z_m / fy, 0.0],
        dtype=np.float64,
    )


def obb_short_axis_base(
    corners_uv: np.ndarray,
    K: np.ndarray,
    T_cam2base: np.ndarray,
    depth_m: float,
) -> np.ndarray | None:
    """Return a detection OBB's short-edge direction in the base XY plane.

    This does not require a full camera-grasp candidate, so elongated-object
    direction gating remains active when mask/depth grasp generation fails.
    """
    corners = np.asarray(corners_uv, dtype=np.float64).reshape(-1, 2)
    if len(corners) < 4 or not np.isfinite(depth_m) or depth_m <= 0.0:
        return None
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    valid = np.flatnonzero(lengths > 1e-6)
    if not len(valid):
        return None
    edge = edges[int(valid[np.argmin(lengths[valid])])]
    axis_cam = _pixel_vec_to_3d(edge, float(depth_m), np.asarray(K, dtype=float))
    axis_base = np.asarray(T_cam2base, dtype=float)[:3, :3] @ axis_cam
    axis_base[2] = 0.0
    norm = float(np.linalg.norm(axis_base))
    return axis_base / norm if norm > 1e-9 else None


# ──────────────────────────────────────────────────────────────────────
# Rotation conversions (lifted from rebot_grasp/utils/transforms.py)
# ──────────────────────────────────────────────────────────────────────


_ROT_X_PI = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Intrinsic ZYX Euler angles (matches rebot_grasp's convention)."""
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = float(np.arctan2(-R[1, 2], R[1, 1]))
        ry = float(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def _canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """Pick the ``R`` vs ``R @ Rx(π)`` branch with smaller |roll|.

    For a symmetric parallel gripper, a 180° twist around tool-x is grasp-
    equivalent. Choosing the smaller-roll branch stabilises IK seeding and
    gives a cleaner RPY for inspection.

    2026-08-14: when |pitch - π/2| < 0.10 (~5.7°), we're in the gimbal-
    lock zone of ZYX Euler. Two fixes here:

    1. Zero R[:, 1, 2] (force open axis horizontal) so pitch is exactly π/2.
       This makes the np.arctan2 branch produce the canonical RPY
       (90°, 90°, 0°) for top-down grasps instead of the numerically-
       arbitrary (-72.5°, 89.6°, -162.5°) you'd get from sub-percent
       floating-point tilt.
    2. Skip the R @ Rx(π) flip — both branches are grasp-equivalent in
       the gimbal-lock zone and the "smaller |roll|" heuristic just picks
       an arbitrary one.
    """
    R = _nearest_rotation_matrix(R)
    pitch = abs(float(_rotation_matrix_to_euler_zyx(R)[1]))
    if abs(pitch - np.pi / 2) < 0.10:
        # Gimbal-lock zone: snap to exact top-down (or bottom-up). Force
        # tool-X = ±z (pure vertical dive), tool-Y horizontal, then
        # recompute tool-Z. The change vs the planner's original choice
        # is sub-percent tilt, well inside the gripper's jaw-width
        # tolerance. The RPY for an exact top-down grasp becomes the
        # canonical (90°, 90°, 0°) instead of (-72.5°, 89.6°, -162.5°).
        R = R.copy()
        if R[2, 0] < 0:  # diving into workspace (top-down)
            R[:, 0] = [0.0, 0.0, -1.0]
        else:
            R[:, 0] = [0.0, 0.0, 1.0]
        # Keep the horizontal projection of original open axis (preserves yaw).
        h = R[:, 1].copy()
        h[2] = 0.0
        n = float(np.linalg.norm(h))
        if n > 1e-9:
            R[:, 1] = h / n
        else:
            R[:, 1] = [1.0, 0.0, 0.0]
        R[:, 2] = np.cross(R[:, 0], R[:, 1])
        return R
    alt = R @ _ROT_X_PI
    roll = abs(float(_rotation_matrix_to_euler_zyx(R)[0]))
    alt_roll = abs(float(_rotation_matrix_to_euler_zyx(alt)[0]))
    return alt if alt_roll < roll else R


def _grasp_axes_to_rebot_tcp_rotation(
    grip_axis: np.ndarray,
    open_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """Map vision-frame ``[grip, open, approach]`` axes to reBotArm TCP.

    reBotArm TCP convention (matches the SDK's asset frame):
      * X = tool-forward (the direction the TCP moves in)
      * Y = gripper-opening direction (jaws separate along this)
      * Z = right-handed completion (cross of X and Y)

    Vision convention (from the OBB):
      * X = grip axis (jaws close along this)
      * Y = open axis (jaws separate along this)
      * Z = approach axis (the camera's plane normal — points UP at the
            object since the camera looks DOWN at it)

    So ``tcp_x = -approach`` flips the up-pointing plane normal to a
    down-pointing tool-forward, ``tcp_y`` projects open onto the plane
    perpendicular to tcp_x, and tcp_z is the right-handed completion.
    """
    grip = np.asarray(grip_axis, dtype=np.float64)
    open_vec = np.asarray(open_axis, dtype=np.float64)
    approach = np.asarray(approach_axis, dtype=np.float64)

    g = _normalize(grip)
    o = _normalize(open_vec)
    a = _normalize(approach)
    if g is None or o is None or a is None:
        raise ValueError("grasp_axes_to_rebot_tcp_rotation: degenerate input")

    tcp_x = -a  # plane normal points up (toward camera); tool points down
    tcp_y = o - float(np.dot(o, tcp_x)) * tcp_x
    tcp_y = _normalize(tcp_y)
    if tcp_y is None:
        tcp_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tcp_z = _normalize(np.cross(tcp_x, tcp_y))
    if tcp_z is None:
        tcp_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Keep tcp_z aligned with the grip axis after the approach flip.
    if float(np.dot(tcp_z, g)) < 0.0:
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    R = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


# ──────────────────────────────────────────────────────────────────────
# Grasp transforms (mirrors rebot_grasp's _make_grasp_base_transform +
# _offset_along_tool_x + transform_grasp_pose_to_base)
# ──────────────────────────────────────────────────────────────────────


def _make_grasp_base_transform(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
) -> np.ndarray:
    T_cam = np.eye(4, dtype=np.float64)
    T_cam[:3, :3] = np.asarray(tcp_rotation_cam, dtype=np.float64)
    T_cam[:3, 3] = np.asarray(position_cam, dtype=np.float64)
    T_base = np.asarray(T_cam2base, dtype=np.float64) @ T_cam
    T_base[:3, :3] = _canonicalize_parallel_gripper_tcp_rotation(T_base[:3, :3])
    return T_base


def _offset_along_tool_x(T: np.ndarray, offset_m: float) -> np.ndarray:
    """Translate ``T`` by ``-offset_m * tool_x`` (i.e. ALONG tool-x with
    the sign convention used by rebot_grasp: ``T_new[3] = T[3] - R[:,0]*offset``).

    Used for both insertion_depth (negative offset → into object) and
    pregrasp (positive offset → retreat along tool-x).
    """
    T_off = T.copy()
    T_off[:3, 3] = T[:3, 3] - T[:3, 0] * float(offset_m)
    return T_off


# ──────────────────────────────────────────────────────────────────────
# Min-area-rect + short-edge extraction (from rebot_grasp/ordinary_grasp.py)
# ──────────────────────────────────────────────────────────────────────


def _rect_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """``cv2.minAreaRect`` of the largest contour of a binary mask.

    Returns the four box corners as ``(4, 2)`` float32, ordered around
    the perimeter (cv2.boxPoints convention).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    rect = cv2.minAreaRect(contour.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def mask_obb_corners(mask: np.ndarray) -> np.ndarray | None:
    """Public, read-only mask-to-OBB helper for runtime direction gating."""
    return _rect_from_mask(np.asarray(mask))


def _short_edge(rect_points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the shortest of the four rect edges and its length in pixels."""
    best_vec = rect_points[1] - rect_points[0]
    best_len = float(np.linalg.norm(best_vec))
    for i in range(4):
        vec = rect_points[(i + 1) % 4] - rect_points[i]
        length = float(np.linalg.norm(vec))
        if length < best_len:
            best_vec = vec
            best_len = length
    return best_vec.astype(np.float32), best_len


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def plan_grasp_from_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cam2base: np.ndarray,
    *,
    insertion_depth_m: float = DEFAULT_INSERTION_DEPTH_M,
    max_width_m: float = 0.09,
    width_pad_m: float = DEFAULT_WIDTH_PAD_M,
    table_z: float = 0.0,
    object_min_height_m: float = 0.003,
    depth_quantile: float = DEFAULT_DEPTH_QUANTILE,
    label: str = "",
    confidence: float = 1.0,
    debug_metadata=None,
    include_wrist_flip: bool = False,
) -> list[Grasp]:
    """Estimate a base-frame parallel-jaw grasp from a 2D segmentation mask.

    Mirrors rebot_grasp's ``estimate_grasp`` + ``transform_grasp_pose_to_base``
    flow: minAreaRect on the mask → backproject center to camera frame →
    derive approach/open/grip axes in the camera frame → convert to
    reBotArm TCP convention → transform to base → apply insertion depth.

    Returns ``[Grasp]`` on success, ``[]`` on any precondition failure
    (no mask, no depth, no contour, no K, no extrinsics) — caller is
    expected to fall back to ``plan_grasps_from_fix`` when this is empty.

    Args:
        mask: ``(H, W)`` bool — segmentation mask of the object in the rgb
            frame. Must be at the same resolution as ``depth_m``.
        depth_m: ``(H, W)`` float32 — metric depth aligned to the rgb
            frame, 0 = invalid. Only ``z > 0`` pixels are sampled.
        K: ``(3, 3)`` — camera intrinsics matrix for the rgb frame.
        T_cam2base: ``(4, 4)`` — static or live camera-to-base transform.
        insertion_depth_m: how far to push the TCP into the object along
            tool-x. 0.015 (1.5 cm) matches the rebot_rs gripper's tip
            offset below the TCP frame.
        max_width_m: gripper max jaw opening (for feasibility tag).
        width_pad_m: extra clearance added to the measured jaw span
            (so the jaws don't scrape on insert).
        depth_quantile: which depth statistic to use inside the mask;
            0.5 = median (default, robust to outliers and depth holes).
        label: object label, threaded onto the returned Grasp.
        confidence: detection confidence, threaded into Grasp.quality.

    Returns:
        A single-element list (most cases produce one best grasp) or an
        empty list on failure. The Grasp's ``approach`` is the tool-forward
        direction in the base frame (so the existing ``pregrasp_position``
        helper and the IK-from-home flow work unchanged).
    """
    # ── preconditions ────────────────────────────────────────────────
    if mask is None or depth_m is None or K is None or T_cam2base is None:
        return []
    mask = np.asarray(mask, dtype=bool)
    depth_m = np.asarray(depth_m)
    K = np.asarray(K, dtype=np.float64)
    T_cam2base = np.asarray(T_cam2base, dtype=np.float64)
    if mask.size == 0 or mask.sum() < 5:
        return []
    if mask.shape != depth_m.shape:
        # resize mask to depth shape; rebot_grasp uses cv2.resize INTER_NEAREST
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth_m.shape[1], depth_m.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # ── 0.5. mask depth cleaning (drop gripper-finger pixels) ───────
    # The new 30 mm longer gripper fingers can land inside the YOLO
    # mask (YOLO sees them as part of the object). The fingers sit
    # closer to the camera than the object surface, so they show up
    # as a depth tail in the mask. We drop mask pixels whose depth is
    # more than ``finger_drop_m`` closer than the mask median — that
    # keeps the object's true surface and discards the gripper
    # fingers. Without this, ``minAreaRect`` lands on a center that
    # is pulled toward the gripper and the arm misses the object.
    #
    # The threshold defaults to 30 mm (the new gripper delta) but is
    # configurable via WRCSIM_FINGER_DROP_M so the same code can be
    # tuned for a different gripper without redeploying.
    finger_drop_m = float(os.environ.get("WRCSIM_FINGER_DROP_M", "0.030"))
    _ys_all, _xs_all = np.nonzero(mask)
    if len(_ys_all) > 5 and finger_drop_m > 0:
        _ds_all = depth_m[_ys_all, _xs_all]
        _valid = _ds_all > 0
        if int(_valid.sum()) > 5:
            _med = float(np.median(_ds_all[_valid]))
            # Keep pixels at the object surface or beyond (away from camera).
            # Drop pixels closer than median - finger_drop_m (the gripper).
            _keep_mask_pixels = _ds_all >= (_med - finger_drop_m)
            if int(_keep_mask_pixels.sum()) >= 5:
                _ys_keep = _ys_all[_valid & _keep_mask_pixels]
                _xs_keep = _xs_all[_valid & _keep_mask_pixels]
                mask_clean = np.zeros_like(mask)
                mask_clean[_ys_keep, _xs_keep] = True
                mask = mask_clean

    # ── 1. minAreaRect on the mask ──────────────────────────────────
    box = _rect_from_mask(mask)
    if box is None:
        return []
    short_vec_px, short_len_px = _short_edge(box)
    short_dir_px = _normalize(short_vec_px)
    if short_dir_px is None:
        return []
    center_px = box.mean(axis=0)

    # ── 2. depth sampling ───────────────────────────────────────────
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    zs = depth_m[ys, xs]
    zs = zs[zs > 0]
    if len(zs) == 0:
        return []
    z_m = float(np.quantile(zs, depth_quantile))

    # ── 3. backproject to camera frame ───────────────────────────────
    position_cam = _backproject(float(center_px[0]), float(center_px[1]), z_m, K)

    # ── 4. approach = -normalize(position_cam) ───────────────────────
    # vector from object center to camera origin → the camera's view
    # direction at this pixel, pointing into the lens
    approach_cam = _normalize(-position_cam)
    if approach_cam is None:
        approach_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # ── 5. open axis from the short edge, projected perpendicular ───
    open_axis_cam = _pixel_vec_to_3d(short_dir_px, z_m, K)
    open_axis_cam = open_axis_cam - float(np.dot(open_axis_cam, approach_cam)) * approach_cam
    open_axis_cam = _normalize(open_axis_cam)
    if open_axis_cam is None:
        return []
    # rebot_grasp canonicalises open_axis[0] > 0 for downstream consistency
    if open_axis_cam[0] < 0:
        open_axis_cam = -open_axis_cam

    # ── 6. grip axis = cross(open, approach) ────────────────────────
    grip_axis_cam = _normalize(np.cross(open_axis_cam, approach_cam))
    if grip_axis_cam is None:
        return []
    # re-orthogonalise open against grip
    open_axis_cam = _normalize(np.cross(approach_cam, grip_axis_cam))
    if open_axis_cam is None:
        return []

    # ── 7. compose vision-frame rotation [grip, open, approach] ──────
    vision_R = np.column_stack([grip_axis_cam, open_axis_cam, approach_cam])

    # ── 8. convert to reBotArm TCP convention ───────────────────────
    try:
        tcp_R = _grasp_axes_to_rebot_tcp_rotation(
            vision_R[:, 0], vision_R[:, 1], vision_R[:, 2]
        )
    except ValueError:
        return []

    # ── 9. transform to base frame ──────────────────────────────────
    T_grasp_base = _make_grasp_base_transform(position_cam, tcp_R, T_cam2base)

    # ── 10. apply insertion depth (push TCP INTO object) ────────────
    # insertion_depth_m is positive in rebot_grasp's config; the offset
    # helper subtracts, so pass -insertion_depth to move INTO the object.
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -float(insertion_depth_m))

    # ── 11. compute jaw span from foreground 3D points ──────────────
    # Low-confidence masks can include a large patch of table. Lift mask
    # pixels into the base frame and remove table points before measuring.
    pixel_jaw_width_m = float(
        np.linalg.norm(_pixel_vec_to_3d(short_dir_px * short_len_px, z_m, K))
    )
    jaw_width_m = pixel_jaw_width_m

    grasp_R = T_grasp_base[:3, :3]
    grasp_pos = T_grasp_base[:3, 3]

    valid = depth_m[ys, xs] > 0
    if int(valid.sum()) >= 20:
        px = xs[valid].astype(np.float64)
        py = ys[valid].astype(np.float64)
        pz = depth_m[ys[valid], xs[valid]].astype(np.float64)
        points_cam = np.column_stack(
            [
                (px - float(K[0, 2])) * pz / float(K[0, 0]),
                (py - float(K[1, 2])) * pz / float(K[1, 1]),
                pz,
            ]
        )
        points_base = (
            points_cam @ T_cam2base[:3, :3].T + T_cam2base[:3, 3]
        )
        foreground = points_base[
            points_base[:, 2] > float(table_z) + float(object_min_height_m)
        ]
        if len(foreground) >= 20:
            open_axis_base = grasp_R[:, 1]
            jaw_coords = foreground @ open_axis_base
            jaw_width_m = float(
                np.quantile(jaw_coords, 0.99) - np.quantile(jaw_coords, 0.01)
            )
            object_center = np.median(foreground, axis=0)
            grasp_pos = object_center + grasp_R[:, 0] * float(insertion_depth_m)

    required_m = jaw_width_m + float(width_pad_m)
    feasible = required_m <= float(max_width_m)

    grasp = Grasp(
        position=grasp_pos,
        rotation=grasp_R,
        width_m=float(required_m),
        approach=grasp_R[:, 0].copy(),
        quality=(1.0 if feasible else 0.2) * float(confidence),
        label=str(label),
    )
    grasps = [grasp]
    if include_wrist_flip:
        grasps.append(
            Grasp(
                position=np.asarray(grasp.position, dtype=float).copy(),
                rotation=np.asarray(grasp.rotation, dtype=float) @ _ROT_X_PI,
                width_m=float(grasp.width_m),
                approach=np.asarray(grasp.approach, dtype=float).copy(),
                quality=float(grasp.quality) * 0.99,
                label=str(grasp.label),
            )
        )
    if debug_metadata is not None:
        camera_xyz = (
            np.asarray(grasp_pos, dtype=float) - T_cam2base[:3, 3]
        ) @ T_cam2base[:3, :3]
        for wrist_variant, candidate in enumerate(grasps):
            try:
                debug_metadata(
                    candidate,
                    {
                        "source": "camera_grasp",
                        "confidence": float(confidence),
                        "camera_xyz": camera_xyz,
                        "wrist_variant": wrist_variant,
                    },
                )
            except Exception:
                pass
    return grasps
