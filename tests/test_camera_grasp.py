"""Unit tests for the camera-frame grasp planner.

Mirrors the contract tests in ``rebot_grasp``'s geometry, adapted to the
wrc_demo ``Grasp`` dataclass. The planner must:

* Return ``[]`` on any precondition failure (no mask / no depth / no K /
  no extrinsics / no contour) so the OBB fallback can take over.
* Produce a top-down approach (``approach ≈ [0, 0, -1]`` in base) for a
  perfectly top-mounted camera, regardless of the mask's yaw.
* Push the TCP INTO the object by ``insertion_depth_m`` along the tool-x
  axis (the gripper fingers live ~1.5 cm below the TCP frame).
* Follow the camera's view direction when the camera is tilted, instead
  of a fixed base-frame vector (this is the whole point of replacing the
  OBB planner on the B601 rig, which has a 29° tilted top camera).
* Compute the jaw span from the OBB's short edge in metric units
  (px → m via K and the depth quantile).
* Produce a right-handed rotation matrix (``det(R) == +1``).
* Honour ``pregrasp_position(offset) = position − approach * offset``
  so the existing ``select_grasp`` IK seed-from-home flow is unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wrc_demo.grasping.camera_grasp import (
    DEFAULT_INSERTION_DEPTH_M,
    _ROT_X_PI,
    _grasp_axes_to_rebot_tcp_rotation,
    _normalize,
    mask_obb_corners,
    obb_short_axis_base,
    plan_grasp_from_mask,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures: synthetic camera + scene
# ──────────────────────────────────────────────────────────────────────


def _top_camera_K(w: int = 640, h: int = 480, fx: float = 600.0) -> np.ndarray:
    """Pinhole intrinsics for a synthetic top-mounted camera at the
    principal-point centre, focal length ``fx``."""
    return np.array(
        [
            [fx, 0.0, w / 2.0],
            [0.0, fx, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_obb_short_axis_base_does_not_require_camera_grasp_candidate():
    corners = np.array([[100, 100], [300, 100], [300, 140], [100, 140]])
    axis = obb_short_axis_base(corners, _top_camera_K(), np.eye(4), 0.5)
    assert axis is not None
    np.testing.assert_allclose(np.abs(axis), [0.0, 1.0, 0.0], atol=1e-9)


def test_mask_obb_corners_supplies_obb_when_detector_has_only_mask():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:120, 40:260] = 1
    corners = mask_obb_corners(mask)
    assert corners is not None and corners.shape == (4, 2)
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    assert edges.max() / edges.min() > 5.0


def _top_camera_T(height_m: float = 1.0) -> np.ndarray:
    """A camera 1 m directly above the base origin, looking straight down.

    Camera convention: +x right, +y down, +z forward.
    Base convention: +x away from robot, +y left, +z up.

    So camera +y (down) ↔ base -y (base left when looking down), and
    camera +z (forward) ↔ base -z (down). Rotation maps cam frame to base
    frame as diag(+1, -1, -1), with translation putting the camera
    origin at base height ``height_m``.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = np.array([0.0, 0.0, height_m], dtype=np.float64)
    return T


def _tilted_camera_T(height_m: float = 1.0, tilt_rad: float = math.radians(20.0)) -> np.ndarray:
    """Top-mounted camera tilted around base-x by ``tilt_rad`` (positive
    = camera looks toward +y base / "back" of the rig).

    Models the B601 rig's roughly 25-30° tilt without burning a real
    hand-eye calibration into the test.
    """
    T = np.eye(4, dtype=np.float64)
    # Build the rotation in base frame: start with top-down orientation
    # (cam +z down = base -z), then tilt around base-x.
    Rx_base = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt_rad), -math.sin(tilt_rad)],
            [0.0, math.sin(tilt_rad), math.cos(tilt_rad)],
        ],
        dtype=np.float64,
    )
    # base_to_cam rotation (top-down convention): diag(1, -1, -1)
    base_to_cam = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    # Tilt the base-to-cam rotation around base-x, then invert to get
    # cam-to-base. (composing rotations in the same frame first, then
    # inverting is the safe form for an off-axis camera.)
    cam_to_base = (Rx_base @ base_to_cam).T
    T[:3, :3] = cam_to_base
    T[:3, 3] = np.array([0.0, 0.0, height_m], dtype=np.float64)
    return T


def _rect_mask(
    cx: float,
    cy: float,
    w_px: int,
    h_px: int,
    H: int = 480,
    W: int = 640,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    """Render an axis-aligned (or yawed) filled rectangle as a bool mask."""
    mask = np.zeros((H, W), dtype=bool)
    half_w, half_h = w_px / 2.0, h_px / 2.0
    cos, sin = math.cos(yaw_rad), math.sin(yaw_rad)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    rx = cos * dx + sin * dy
    ry = -sin * dx + cos * dy
    mask[(np.abs(rx) <= half_w) & (np.abs(ry) <= half_h)] = True
    return mask


# ──────────────────────────────────────────────────────────────────────
# Precondition failures → []
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw",
    [
        {"mask": None},
        {"depth_m": None},
        {"K": None},
        {"T_cam2base": None},
    ],
)
def test_missing_inputs_return_empty(kw):
    base = dict(
        mask=_rect_mask(320, 240, 100, 50),
        depth_m=np.full((480, 640), 0.5, dtype=np.float32),
        K=_top_camera_K(),
        T_cam2base=_top_camera_T(),
    )
    base.update(kw)
    assert plan_grasp_from_mask(**base) == []


def test_empty_mask_returns_empty():
    mask = np.zeros((480, 640), dtype=bool)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)
    out = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m,
        K=_top_camera_K(), T_cam2base=_top_camera_T(),
    )
    assert out == []


def test_all_zero_depth_returns_empty():
    mask = _rect_mask(320, 240, 100, 50)
    depth_m = np.zeros((480, 640), dtype=np.float32)
    assert plan_grasp_from_mask(
        mask=mask, depth_m=depth_m,
        K=_top_camera_K(), T_cam2base=_top_camera_T(),
    ) == []


# ──────────────────────────────────────────────────────────────────────
# Geometry: top-mounted camera + horizontal rect → top-down grasp
# ──────────────────────────────────────────────────────────────────────


def test_top_camera_top_down_approach():
    """For a perfectly top-mounted camera, the tool-forward direction in
    base frame should be straight down (≈ [0, 0, -1]) regardless of the
    rect's yaw -- the camera sees straight down, so the gripper dives
    straight down."""
    K = _top_camera_K()
    T = _top_camera_T(height_m=1.0)
    mask = _rect_mask(320, 240, 100, 50)  # horizontal rect, no yaw
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)

    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
    )
    assert len(grasps) == 1
    g = grasps[0]
    np.testing.assert_allclose(g.approach, [0.0, 0.0, -1.0], atol=1e-6)
    # position should be at the mask centre (320, 240), z = 0.5 in cam,
    # which maps to z = 0.5 in base AFTER insertion_depth.
    np.testing.assert_allclose(g.position[2], 0.5 - DEFAULT_INSERTION_DEPTH_M, atol=1e-6)
    # x, y near zero (mask at principal point)
    np.testing.assert_allclose(g.position[:2], [0.0, 0.0], atol=1e-3)
    # right-handed rotation
    assert np.isclose(np.linalg.det(g.rotation), 1.0, atol=1e-6)


def test_top_camera_approach_independent_of_mask_yaw():
    """Rotation of the mask in image space must not change the approach
    direction -- the camera sees straight down, so the dive-in vector is
    always -z in base frame."""
    K = _top_camera_K()
    T = _top_camera_T(height_m=1.0)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)
    for yaw in [0.0, 0.3, -0.7, 1.2]:
        mask = _rect_mask(320, 240, 100, 50, yaw_rad=yaw)
        grasps = plan_grasp_from_mask(
            mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
        )
        assert len(grasps) == 1
        np.testing.assert_allclose(
            grasps[0].approach, [0.0, 0.0, -1.0], atol=1e-6,
            err_msg=f"yaw={yaw} broke approach",
        )


# ──────────────────────────────────────────────────────────────────────
# Insertion depth: TCP pushed INTO the object along tool-x
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("insertion_m", [0.0, 0.01, 0.015, 0.03])
def test_insertion_depth_pushes_tcp_into_object(insertion_m):
    """TCP must be offset from the surface along -tool-x by exactly the
    configured ``insertion_depth_m``. For a top-down grasp, -tool-x = +z
    in base, so position.z should DECREASE by the insertion amount
    relative to the back-projected centre."""
    K = _top_camera_K()
    T = _top_camera_T(height_m=1.0)
    mask = _rect_mask(320, 240, 100, 50)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)
    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
        insertion_depth_m=insertion_m,
    )
    g = grasps[0]
    expected_z = 0.5 - insertion_m
    np.testing.assert_allclose(g.position[2], expected_z, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Tilted camera: approach follows the camera's view ray
# ──────────────────────────────────────────────────────────────────────


def test_tilted_camera_approach_along_line_of_sight():
    """For a tilted camera, the approach direction in base frame must
    match the line from the camera origin to the object centre -- this
    is the WHOLE POINT of the camera-frame planner over the OBB
    planner's hard-coded base-frame vector."""
    K = _top_camera_K()
    tilt = math.radians(25.0)
    T = _tilted_camera_T(height_m=1.0, tilt_rad=tilt)
    mask = _rect_mask(320, 240, 100, 50)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)

    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
    )
    g = grasps[0]

    # Expected approach in base frame: unit vector from camera origin to
    # object centre (in base frame), pointing "into" the object.
    cam_in_base = T[:3, 3]
    grasp_in_base = g.position + g.approach * DEFAULT_INSERTION_DEPTH_M
    expected = grasp_in_base - cam_in_base
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(g.approach, expected, atol=1e-6)


def test_optional_wrist_flip_keeps_geometry_and_changes_only_roll():
    K = _top_camera_K()
    T = _top_camera_T(height_m=1.0)
    mask = _rect_mask(320, 240, 100, 40)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)
    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
        include_wrist_flip=True,
    )
    assert len(grasps) == 2
    np.testing.assert_allclose(grasps[0].position, grasps[1].position)
    np.testing.assert_allclose(grasps[0].approach, grasps[1].approach)
    assert grasps[0].width_m == grasps[1].width_m
    np.testing.assert_allclose(
        grasps[1].rotation, grasps[0].rotation @ _ROT_X_PI, atol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────
# Jaw width from short edge
# ──────────────────────────────────────────────────────────────────────


def test_jaw_width_from_short_edge():
    """The reported width should equal ``short_edge_px * z / fy + pad``,
    not ``long_edge * z / fx + pad`` -- we want the SHORT edge because
    that's the axis the jaws grip ALONG (the jaws open perpendicular to
    this)."""
    K = _top_camera_K(fx=600.0)
    T = _top_camera_T(height_m=1.0)
    short_px, long_px = 50, 100  # short along y, long along x
    mask = _rect_mask(320, 240, long_px, short_px)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)

    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
    )
    g = grasps[0]
    # expected jaw width ≈ short_edge_m + pad
    expected_short_m = short_px * 0.5 / 600.0  # 50 * 0.5 / 600 ≈ 0.0417 m
    np.testing.assert_allclose(
        g.width_m, expected_short_m + 0.015, atol=2e-3,  # ±2mm pad slop
    )


def test_table_contamination_does_not_inflate_short_edge_width():
    """A broad table mask must not turn a 20 mm object into 110 mm."""
    K = _top_camera_K(fx=600.0)
    T = _top_camera_T(height_m=1.0)
    mask = _rect_mask(320, 240, 180, 120)
    depth_m = np.ones((480, 640), dtype=np.float32)
    # Package top: approximately 20 mm wide, base z=20 mm.
    depth_m[234:246, 270:370] = 0.98

    grasp = plan_grasp_from_mask(
        mask=mask,
        depth_m=depth_m,
        K=K,
        T_cam2base=T,
        table_z=0.0,
        insertion_depth_m=0.0,
    )[0]

    assert 0.025 < grasp.width_m < 0.045
    assert grasp.position[2] == pytest.approx(0.02, abs=0.003)


def test_width_pad_and_max_width_quality():
    """A width that exceeds ``max_width_m`` should still be reported but
    with quality demoted (feasibility flag for the selector)."""
    K = _top_camera_K()
    T = _top_camera_T(height_m=1.0)
    mask = _rect_mask(320, 240, 200, 200)  # huge object → way wider than 90 mm
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)
    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T, max_width_m=0.05,
    )
    g = grasps[0]
    assert g.width_m > 0.05
    # Infeasible ⇒ quality = 0.2 * confidence (per plan_grasps_from_fix convention)
    assert g.quality < 0.5


# ──────────────────────────────────────────────────────────────────────
# Integration with the existing Grasp dataclass contract
# ──────────────────────────────────────────────────────────────────────


def test_pregrasp_position_follows_approach():
    """The Grasp's pregrasp_position must equal position − approach *
    offset so the existing select_grasp IK seeding (which uses this
    helper) is unchanged."""
    K = _top_camera_K()
    T = _top_camera_T(height_m=1.0)
    mask = _rect_mask(320, 240, 100, 50)
    depth_m = np.full((480, 640), 0.5, dtype=np.float32)
    grasps = plan_grasp_from_mask(
        mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
    )
    g = grasps[0]
    for offset in [0.05, 0.08, 0.12]:
        pre = g.pregrasp_position(offset)
        np.testing.assert_allclose(pre, g.position - g.approach * offset, atol=1e-9)


def test_rotation_is_right_handed_so3():
    """Repeated random scenes must all produce right-handed rotations."""
    rng = np.random.default_rng(0)
    K = _top_camera_K()
    for trial in range(20):
        tilt = rng.uniform(-0.4, 0.4)  # up to ~23° tilt
        T = _tilted_camera_T(height_m=1.0, tilt_rad=tilt)
        mask = _rect_mask(
            cx=rng.uniform(200, 440), cy=rng.uniform(180, 300),
            w_px=rng.integers(60, 200), h_px=rng.integers(30, 100),
            yaw_rad=rng.uniform(-1.0, 1.0),
        )
        depth_m = np.full((480, 640), rng.uniform(0.3, 0.8), dtype=np.float32)
        grasps = plan_grasp_from_mask(
            mask=mask, depth_m=depth_m, K=K, T_cam2base=T,
        )
        if not grasps:
            continue  # valid outcome for some degenerate inputs
        R = grasps[0].rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────


def test_normalize_handles_zero_vector():
    assert _normalize(np.zeros(3)) is None
    out = _normalize(np.array([3.0, 0.0, 4.0]))
    np.testing.assert_allclose(out, [0.6, 0.0, 0.8])


def test_grasp_axes_to_rebot_tcp_rotation_right_handed():
    """Random vision-frame grasp axes must produce a right-handed
    reBotArm TCP rotation (det = +1)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        approach = rng.normal(size=3)
        approach /= np.linalg.norm(approach)
        # pick a non-degenerate open axis perpendicular to approach
        open_vec = rng.normal(size=3)
        open_vec -= open_vec @ approach * approach
        open_vec /= np.linalg.norm(open_vec)
        grip = np.cross(open_vec, approach)
        R = _grasp_axes_to_rebot_tcp_rotation(grip, open_vec, approach)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        # Column 0 (tcp_x) must equal -approach by construction
        np.testing.assert_allclose(R[:, 0], -approach, atol=1e-9)


def test_grasp_axes_to_rebot_tcp_rotation_keeps_tcp_z_aligned_with_grip():
    """After the approach-axis flip, ``tcp_z`` must align with the grip
    axis (jaws close along the grip axis when projected to the plane
    perpendicular to approach)."""
    # approach pointing up (+z in cam frame), grip along +x, open along +y
    approach = np.array([0.0, 0.0, 1.0])
    grip = np.array([1.0, 0.0, 0.0])
    open_vec = np.array([0.0, 1.0, 0.0])
    R = _grasp_axes_to_rebot_tcp_rotation(grip, open_vec, approach)
    # tcp_x = -approach = (0, 0, -1)
    np.testing.assert_allclose(R[:, 0], [0.0, 0.0, -1.0], atol=1e-9)
    # tcp_z should be ±grip; sign chosen so dot(tcp_z, grip) > 0
    assert np.dot(R[:, 2], grip) > 0
