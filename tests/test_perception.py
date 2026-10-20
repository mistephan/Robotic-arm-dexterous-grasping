import numpy as np

from wrc_demo.perception.depth_provider import DepthProvider
from wrc_demo.perception.detector import MockDetector
from wrc_demo.perception.grounding import (
    Extrinsics,
    localize_object,
    mask_to_points_cam,
    oriented_bbox,
)
from wrc_demo.perception.mock_camera import MockCamera, synthetic_tabletop
from wrc_demo.config import Cfg


T_CAM2BASE = np.array(
    [
        [0.0, -1.0, 0.0, 0.28],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.00],
    ]
)


def test_synthetic_frame_geometry():
    f = synthetic_tabletop()
    assert f.rgb.shape == (480, 640, 3)
    assert f.depth_m.shape == (480, 640)
    # Box region is nearer than the table.
    assert f.depth_m[230, 320] < f.depth_m[100, 100]


def test_mock_detector_finds_red_box():
    f = synthetic_tabletop()
    dets = MockDetector().detect(f)
    assert len(dets) == 1
    d = dets[0]
    x0, y0, x1, y1 = d.bbox
    assert 270 <= x0 <= 290 and 350 <= x1 <= 370
    assert d.mask is not None and d.mask.sum() > 1000


def test_mask_backprojection_scale():
    f = synthetic_tabletop()
    d = MockDetector().detect(f)[0]
    pts = mask_to_points_cam(f, d.mask)
    assert pts.shape[0] > 100
    # Box top sits at 0.55 m; spread in x should be ~7 cm (80 px at fx=600).
    assert abs(pts[:, 2].mean() - 0.55) < 0.01
    assert 0.05 < np.ptp(pts[:, 0]) < 0.09


def test_localize_object_in_base_frame():
    f = synthetic_tabletop()
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)
    fix = localize_object(f, "red cube", MockDetector(), extr)
    # Camera above (0.28, 0); the box center is near the image center.
    assert abs(fix.position[0] - 0.29) < 0.03
    assert abs(fix.position[1]) < 0.03
    assert abs(fix.position[2] - 0.05) < 0.01  # top surface at 5 cm


def test_localize_fails_readably():
    f = synthetic_tabletop()
    extr = Extrinsics(mode="eye_to_hand", T=T_CAM2BASE)
    det = MockDetector(detections=[])
    try:
        localize_object(f, "unicorn", det, extr)
        raised = False
    except Exception as e:
        raised = True
        assert "unicorn" in str(e)
    assert raised


def test_oriented_bbox_of_box_points(rng):
    pts = rng.uniform([-0.05, -0.02, 0.0], [0.05, 0.02, 0.03], size=(500, 3))
    center, extents, axes = oriented_bbox(pts)
    assert np.allclose(center, [0, 0, 0.015], atol=0.01)
    assert extents[0] > extents[1] > 0


def test_plane_depth_fallback():
    cfg = Cfg({"table_plane_cam": [0.0, 0.0, -1.0, 0.6]})
    dp = DepthProvider(cfg)
    f = synthetic_tabletop()
    f.depth_m = None
    f.depth_source = "none"
    f = dp.ensure_depth(f)
    assert f.depth_source == "plane"
    # Fronto-parallel plane: z-depth is 0.6 m at every pixel (depth maps
    # store z-depth, not ray length).
    assert abs(f.depth_m[240, 320] - 0.6) < 1e-6
    assert abs(f.depth_m[0, 0] - 0.6) < 1e-6


def test_plane_fit_recovers_plane():
    f = synthetic_tabletop()
    plane = DepthProvider.fit_table_plane(f)
    n, d = plane[:3], plane[3]
    assert n[2] < -0.9  # normal points at the camera
    # Table at z=0.6: n.p + d = 0 -> d ~ 0.6 for n=(0,0,-1).
    assert abs(d - 0.6) < 0.03


def test_mock_camera_replay(tmp_path):
    f = synthetic_tabletop()
    np.savez_compressed(tmp_path / "frame_0000.npz", rgb=f.rgb, depth_m=f.depth_m, K=f.K)
    cam = MockCamera(Cfg({"dataset": str(tmp_path)}))
    cam.open()
    g = cam.get_frame()
    assert np.array_equal(g.rgb, f.rgb)
    assert g.frame_id == 1
    cam.close()
