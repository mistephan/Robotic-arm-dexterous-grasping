import numpy as np

from wrc_demo.grasping.graspgenx_backend import (
    GraspGenXPlanner,
    estimate_local_elongated_geometry,
)
from wrc_demo.types import Detection, ObjectFix


class _FakeClient:
    def __init__(self, poses, scores, branch_tags=None):
        self.response = {
            "grasps": poses, "confidences": scores,
            "branch_tags": (
                list(branch_tags) if branch_tags is not None
                else ["obb"] * len(scores)
            ),
        }

    def request(self, payload):
        return self.response


def _curved_banana_cloud():
    longitudinal = np.linspace(-0.075, 0.075, 90)
    across = np.linspace(-0.016, 0.016, 13)
    points = []
    for s in longitudinal:
        center = np.array([0.35 + s, 0.08 + 3.5 * s * s])
        tangent = np.array([1.0, 7.0 * s])
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        for w in across:
            xy = center + w * normal
            z = 0.028 * np.sqrt(max(1.0 - (w / 0.017) ** 2, 0.0))
            points.append([xy[0], xy[1], z])
    return np.asarray(points)


def test_local_curve_geometry_tracks_banana_tangent_and_section_width():
    points = _curved_banana_cloud()
    left = estimate_local_elongated_geometry(points, [0.30, 0.089])
    right = estimate_local_elongated_geometry(points, [0.40, 0.089])

    assert left is not None and right is not None
    # Eigenvector sign is arbitrary; x*y preserves the unoriented line slope.
    assert np.prod(left["tangent_xy"]) < -0.10
    assert np.prod(right["tangent_xy"]) > 0.10
    assert 0.028 <= left["object_width_m"] <= 0.040
    assert 0.028 <= right["object_width_m"] <= 0.040
    assert left["centerline_offset_m"] < 0.012
    off_center = estimate_local_elongated_geometry(points, [0.30, 0.13])
    assert off_center is not None
    assert off_center["centerline_offset_m"] > left["centerline_offset_m"]


def test_client_enforces_topk_and_preserves_over_limit_width():
    count = 289
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], count, axis=0)
    poses[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
    scores = np.linspace(0.0, 1.0, count, dtype=np.float32)

    planner = GraspGenXPlanner.__new__(GraspGenXPlanner)
    planner._client = _FakeClient(poses, scores)
    planner.gripper = "test"
    planner.tip_offset_m = 0.0
    planner.insertion_depth_m = 0.0
    planner.produce_insertion_depth_m = 0.03
    planner.produce_target_below_cloud_top_m = 0.02
    planner.produce_center_from_xy = True
    planner.produce_center_extra_depth_m = 0.005
    planner.produce_min_tcp_above_table_m = 0.01
    planner.isotropic_aspect_ratio = 1.4
    planner.isotropic_max_yaws_per_branch_tilt = 2
    planner.isotropic_preferred_jaw_axis_xy = np.array([0.0, 1.0])
    planner.num_grasps = count
    planner.topk = 32
    planner.min_score = 0.0
    planner.moe_num_yaws = 18
    planner.moe_tilt_degrees = [0.0, -12.0, 12.0, -20.0, 20.0]
    planner.max_per_tilt_bin = 32
    planner.sweep_params = None
    planner.last_latency_s = None

    x = np.linspace(-0.06, 0.06, 101)
    points = np.column_stack([x, np.zeros_like(x), np.full_like(x, 0.02)])
    detection = Detection(
        label="wide part", conf=0.9,
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
    )
    fix = ObjectFix(
        label="wide part", position=points.mean(axis=0), points=points,
        detection=detection, extent=np.array([0.12, 0.0, 0.0]), axes=np.eye(3),
    )

    grasps = planner.plan(fix, max_width_m=0.095)

    assert len(grasps) == 32
    assert np.allclose(grasps[0]._ggx_raw_origin, [0.0, 0.0, 0.0])
    assert grasps[0]._ggx_tip_offset_m == 0.0
    assert grasps[0]._ggx_insertion_depth_m == 0.0
    assert grasps[0]._ggx_branch == "obb"
    assert grasps[0]._ggx_tilt_deg == 0.0
    assert all(g.width_m > 0.095 for g in grasps)
    assert [g.quality for g in grasps] == sorted(
        (g.quality for g in grasps), reverse=True
    )


def test_full_link_to_tcp_offset_is_rotated_by_returned_base_pose():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.25, 0.1, 0.12]
    planner = GraspGenXPlanner({
        "graspgenx": {
            "tcp_offset_xyz_m": [0.08, 0.0, -0.05],
            "tcp_rotation_matrix": [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            "insertion_depth_m": 0.0,
            "candidate_xy_margin_m": 0.1,
            "candidate_max_above_cloud_m": 0.2,
        }
    })
    planner._client = _FakeClient(pose[None], np.array([0.9], dtype=np.float32))
    points = np.column_stack([
        np.linspace(0.28, 0.32, 100),
        np.full(100, 0.1),
        np.full(100, 0.04),
    ])
    fix = ObjectFix(
        label="part", position=np.array([0.3, 0.1, 0.04]), points=points,
        detection=Detection(
            label="part", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.04, 0.0, 0.0]), axes=np.eye(3),
    )

    grasp = planner.plan(fix)[0]

    np.testing.assert_allclose(grasp.position, [0.33, 0.1, 0.17], atol=1e-8)
    np.testing.assert_allclose(
        grasp.rotation,
        pose[:3, :3] @ np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]),
        atol=1e-8,
    )
    assert grasp._ggx_tcp_offset_xyz_m == [0.08, 0.0, -0.05]
    assert np.isclose(grasp._ggx_tip_offset_m, np.hypot(0.08, 0.05))


def test_base_to_jaw_center_then_capped_object_insertion():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.3, 0.1, 0.075]
    planner = GraspGenXPlanner.__new__(GraspGenXPlanner)
    planner._client = _FakeClient(pose[None], np.array([0.9], dtype=np.float32))
    planner.gripper = "rs"
    planner.tip_offset_m = 0.0225
    planner.insertion_depth_m = 0.015
    planner.produce_insertion_depth_m = 0.03
    planner.produce_target_below_cloud_top_m = 0.02
    planner.produce_center_from_xy = True
    planner.produce_center_extra_depth_m = 0.005
    planner.produce_min_tcp_above_table_m = 0.01
    planner.isotropic_aspect_ratio = 1.4
    planner.isotropic_max_yaws_per_branch_tilt = 2
    planner.isotropic_preferred_jaw_axis_xy = np.array([0.0, 1.0])
    planner.num_grasps = 1
    planner.topk = 1
    planner.min_score = 0.0
    planner.moe_num_yaws = 18
    planner.moe_tilt_degrees = [0.0]
    planner.max_per_tilt_bin = 8
    planner.sweep_params = None
    planner.last_latency_s = None
    points = np.column_stack([
        np.linspace(0.28, 0.32, 100),
        np.full(100, 0.1),
        np.full(100, 0.04),
    ])
    fix = ObjectFix(
        label="lime", position=np.array([0.3, 0.1, 0.04]), points=points,
        detection=Detection(
            label="lime", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.04, 0.0, 0.0]), axes=np.eye(3),
    )
    grasp = planner.plan(fix, max_width_m=0.095)[0]
    # Produce depth seeks cloud_top-20 mm but respects its 30 mm cap.
    np.testing.assert_allclose(grasp.position[2], 0.0225, atol=1e-6)
    assert grasp._ggx_insertion_shortfall_m < 0.003
    assert grasp._ggx_insertion_depth_m == 0.03



def test_tcp_standoff_keeps_palm_outside_produce_internal_grasp_center():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.30, 0.10, 0.10]
    planner = GraspGenXPlanner({
        "graspgenx": {
            "topk": 1,
            "tcp_standoff_m": 0.030,
            "tip_offset_m": 0.0,
            "produce_insertion_depth_m": 0.030,
            "produce_target_below_cloud_top_m": 0.020,
            "produce_center_from_xy": False,
            "candidate_max_above_cloud_m": 0.030,
        }
    })
    planner._client = _FakeClient(pose[None], np.array([0.9], dtype=np.float32))
    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    points = np.column_stack([
        0.30 + 0.02 * np.cos(theta),
        0.10 + 0.02 * np.sin(theta),
        np.full_like(theta, 0.06),
    ])
    fix = ObjectFix(
        label="lime", position=points.mean(axis=0), points=points,
        detection=Detection(
            label="lime", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.04, 0.04, 0.06]), axes=np.eye(3),
    )

    grasp = planner.plan(fix, max_width_m=0.095)[0]

    np.testing.assert_allclose(grasp.position[2], 0.07, atol=1e-6)
    assert grasp._ggx_tcp_standoff_m == 0.030
    assert grasp._ggx_insertion_depth_m == 0.030
    assert grasp._ggx_insertion_shortfall_m < 0.003


def test_rejects_high_candidate_that_cannot_reach_produce_depth_within_cap():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.3, 0.1, 0.14]
    planner = GraspGenXPlanner({
        "graspgenx": {
            "topk": 8,
            "tip_offset_m": 0.035,
            "produce_insertion_depth_m": 0.030,
            "produce_target_below_cloud_top_m": 0.020,
            "produce_max_insertion_shortfall_m": 0.003,
            "max_per_tilt_bin": 8,
        }
    })
    planner._client = _FakeClient(pose[None], np.array([0.9], dtype=np.float32))
    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    points = np.column_stack([
        0.30 + 0.02 * np.cos(theta),
        0.10 + 0.02 * np.sin(theta),
        np.full_like(theta, 0.06),
    ])
    fix = ObjectFix(
        label="banana", position=points.mean(axis=0), points=points,
        detection=Detection(
            label="banana", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.04, 0.04, 0.01]), axes=np.eye(3),
    )

    assert planner.plan(fix, max_width_m=0.095) == []


def test_retracts_converted_produce_tcp_from_below_table_to_target_depth():
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.44, 0.035, 0.124]
    planner = GraspGenXPlanner({
        "graspgenx": {
            "topk": 1,
            "tcp_offset_xyz_m": [0.0, 0.0, 0.135817],
            "produce_target_below_cloud_top_m": 0.020,
            "produce_center_from_xy": False,
            "produce_max_retraction_m": 0.060,
            "produce_min_tcp_above_table_m": 0.010,
            "candidate_xy_margin_m": 0.050,
        }
    })
    planner._client = _FakeClient(pose[None], np.array([0.9], dtype=np.float32))
    points = np.column_stack([
        np.linspace(0.37, 0.51, 120),
        np.full(120, 0.035),
        np.full(120, 0.041),
    ])
    fix = ObjectFix(
        label="banana", position=np.array([0.44, 0.035, 0.026]), points=points,
        detection=Detection(
            label="banana", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.14, 0.05, 0.04]), axes=np.eye(3),
    )

    grasp = planner.plan(fix, max_width_m=0.11)[0]

    # Raw conversion is 124 - 135.817 = -11.817 mm. The signed correction
    # retracts along -approach to cloud_top - 20 mm instead of emitting a
    # below-table TCP.
    np.testing.assert_allclose(grasp.position[2], 0.021, atol=1e-6)
    assert grasp._ggx_depth_correction_m < 0.0
    assert grasp.position[2] >= 0.010


def test_isotropic_cloud_keeps_only_wrist_neutral_yaws():
    angles = np.deg2rad([0, 30, 60, 90, 120, 150])
    poses = []
    for angle in angles:
        opening = np.array([np.cos(angle), np.sin(angle), 0.0])
        approach = np.array([0.0, 0.0, -1.0])
        second = np.cross(approach, opening)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.column_stack([opening, second, approach])
        poses.append(pose)
    planner = GraspGenXPlanner.__new__(GraspGenXPlanner)
    planner._client = _FakeClient(
        np.asarray(poses), np.linspace(0.1, 0.9, len(poses), dtype=np.float32)
    )
    planner.gripper = "rs"
    planner.tip_offset_m = 0.0
    planner.insertion_depth_m = 0.0
    planner.produce_insertion_depth_m = 0.03
    planner.produce_target_below_cloud_top_m = 0.02
    planner.produce_center_from_xy = True
    planner.produce_center_extra_depth_m = 0.005
    planner.produce_min_tcp_above_table_m = 0.01
    planner.isotropic_aspect_ratio = 1.4
    planner.isotropic_max_yaws_per_branch_tilt = 2
    planner.isotropic_preferred_jaw_axis_xy = np.array([0.0, 1.0])
    planner.num_grasps = len(poses)
    planner.topk = 32
    planner.min_score = 0.0
    planner.moe_num_yaws = 18
    planner.moe_tilt_degrees = [0.0]
    planner.max_per_tilt_bin = 32
    planner.sweep_params = None
    planner.last_latency_s = None
    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    points = np.column_stack([
        0.3 + 0.02 * np.cos(theta),
        0.1 + 0.02 * np.sin(theta),
        np.full_like(theta, 0.04),
    ])
    fix = ObjectFix(
        label="lime", position=points.mean(axis=0), points=points,
        detection=Detection(
            label="lime", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.04, 0.04, 0.01]), axes=np.eye(3),
    )
    grasps = planner.plan(fix, max_width_m=0.095)
    assert len(grasps) == 2
    assert all(g._ggx_isotropic for g in grasps)
    assert grasps[0]._ggx_target_depth_m > 0.024
    assert len(grasps[0]._ggx_horizontal_spans_m) == 2
    assert abs(float(grasps[0].rotation[:, 1][:2] @ np.array([0.0, 1.0]))) > 0.99


def test_filters_off_object_and_near_zero_diffusion_candidates():
    rotation = np.diag([1.0, -1.0, -1.0])
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    poses[:, :3, :3] = rotation
    poses[0, :3, 3] = [0.40, 0.20, 0.14]  # high-score but off-object diff
    poses[1, :3, 3] = [0.30, 0.10, 0.06]  # valid OBB
    poses[2, :3, 3] = [0.30, 0.10, 0.06]  # on-object but zero-score diff
    planner = GraspGenXPlanner({
        "graspgenx": {
            "topk": 10,
            "diff_min_score": 0.01,
            "candidate_xy_margin_m": 0.025,
            "candidate_max_above_cloud_m": 0.030,
            "insertion_depth_m": 0.015,
            "produce_insertion_depth_m": 0.030,
        }
    })
    planner._client = _FakeClient(
        poses, np.array([0.9, 0.2, 1e-7], dtype=np.float32),
        branch_tags=["diff", "obb", "diff"],
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    points = np.column_stack([
        0.30 + 0.02 * np.cos(theta),
        0.10 + 0.02 * np.sin(theta),
        np.full_like(theta, 0.04),
    ])
    fix = ObjectFix(
        label="lime", position=points.mean(axis=0), points=points,
        detection=Detection(
            label="lime", conf=0.9,
            bbox=np.array([0, 0, 10, 10], dtype=np.float32),
        ),
        extent=np.array([0.04, 0.04, 0.01]), axes=np.eye(3),
    )

    grasps = planner.plan(fix, max_width_m=0.095)

    assert len(grasps) == 1
    assert grasps[0]._ggx_branch == "obb"
    np.testing.assert_allclose(grasps[0]._ggx_raw_origin, [0.30, 0.10, 0.06])
