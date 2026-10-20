import numpy as np

from wrc_demo.grasping.force import select_profile
from wrc_demo.grasping.obb_grasp import (
    _approach_rotation,
    _yaw_rotation,
    plan_grasps_from_fix,
)
from wrc_demo.perception.grounding import oriented_bbox
from wrc_demo.types import Detection, ObjectFix


def make_fix(center=(0.3, 0.0), size=(0.10, 0.04, 0.06), yaw=0.0, n=400):
    """Synthesize a box point cloud sitting on the table (z in [0, h])."""
    rng = np.random.default_rng(0)
    l, w, h = size
    pts = rng.uniform([-l / 2, -w / 2, 0], [l / 2, w / 2, h], size=(n, 3))
    R = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    pts = pts @ R.T + np.array([center[0], center[1], 0.0])
    c, extents, axes = oriented_bbox(pts)
    det = Detection(label="box", conf=0.9, bbox=np.array([0, 0, 10, 10], dtype=np.float32))
    return ObjectFix(label="box", position=c, points=pts, detection=det, extent=extents, axes=axes)


def test_grasp_across_short_axis():
    # Production tilt must preserve the selected OBB short axis as the
    # physical jaw-opening axis.
    fix = make_fix(size=(0.10, 0.04, 0.06), yaw=0.0)
    grasps = plan_grasps_from_fix(
        fix, table_z=0.0, max_width_m=0.09, approach_pitch_rad=0.2,
    )
    assert grasps
    g = grasps[0]
    # Opening axis should be along the box's short side (y here).
    open_axis = g.rotation[:, 1]
    assert abs(open_axis[1]) > 0.95
    assert g.width_m < 0.07  # 4 cm + pad
    assert 0.0 < g.position[2] < 0.06
    assert g.approach[2] < -0.97


def test_elongated_object_keeps_long_axis_for_width_filter():
    fix = make_fix(size=(0.17, 0.02, 0.03), yaw=0.4)
    grasps = plan_grasps_from_fix(
        fix, table_z=0.0, max_width_m=0.095, width_pad_m=0.0,
    )
    assert len(grasps) == 4
    widths = [grasp.width_m for grasp in grasps]
    assert all(0.015 < width < 0.03 for width in widths[:2])
    assert all(0.15 < width < 0.18 for width in widths[2:])
    side0 = grasps[0].approach[:2] / np.linalg.norm(grasps[0].approach[:2])
    side1 = grasps[1].approach[:2] / np.linalg.norm(grasps[1].approach[:2])
    assert np.dot(side0, side1) < -0.99


def test_near_topdown_tilt_is_continuous_and_mostly_vertical():
    fix = make_fix(size=(0.17, 0.03, 0.04), yaw=0.3)
    grasp = plan_grasps_from_fix(
        fix, table_z=0.0, approach_pitch_rad=0.2,
    )[0]
    assert grasp.approach[2] < -0.97
    assert np.isclose(np.linalg.norm(grasp.approach), 1.0)


def test_tilted_rotation_keeps_yaw_as_opening_axis():
    for yaw in [0.0, 0.4, np.pi / 2, -1.1]:
        R = _approach_rotation(0.2, yaw)
        expected_open = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        assert abs(float(R[:, 1] @ expected_open)) > 0.999
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_grasp_yaw_follows_object():
    # The selected OBB axis remains the opening axis at the production tilt.
    yaw = 0.6
    fix = make_fix(size=(0.12, 0.03, 0.05), yaw=yaw)
    grasps = plan_grasps_from_fix(
        fix, table_z=0.0, approach_pitch_rad=0.2,
    )
    open_axis = grasps[0].rotation[:, 1]
    got = np.arctan2(open_axis[1], open_axis[0])
    # Opening across the short side = perpendicular to the box's long axis.
    want = yaw + np.pi / 2
    diff = (got - want + np.pi / 2) % np.pi - np.pi / 2
    assert abs(diff) < 0.15


def test_too_wide_object_flagged():
    fix = make_fix(size=(0.15, 0.13, 0.05))
    grasps = plan_grasps_from_fix(fix, table_z=0.0, max_width_m=0.09)
    assert all(g.quality < 0.5 for g in grasps)  # feasible=False downweights


def test_grasp_z_clamped_above_table():
    fix = make_fix(size=(0.06, 0.03, 0.008))  # very flat object
    grasps = plan_grasps_from_fix(fix, table_z=0.0, min_grasp_z_above_table=0.005)
    assert grasps[0].position[2] >= 0.005 - 1e-9


def test_yaw_rotation_orthonormal():
    for yaw in [0.0, 0.5, -1.2, 3.0]:
        R = _yaw_rotation(yaw)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[:, 0], [0, 0, -1])  # approach down


def test_force_profiles():
    assert select_profile("wine glass").name == "fragile"
    assert select_profile("plush toy").name == "soft"
    assert select_profile("soda can").name == "slippery"
    assert select_profile("lime").name == "slippery"
    assert select_profile("lime", material_hint="rigid").name == "slippery"
    assert select_profile("passion fruit").name == "slippery"
    assert select_profile("box").name == "rigid"
    assert select_profile("box", material_hint="fragile").name == "fragile"
    assert select_profile("box", material_hint="very fragile item").name == "fragile"
