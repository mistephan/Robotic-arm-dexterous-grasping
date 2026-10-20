"""Tests for the calibration dataset dataclass + JSON I/O."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wrc_demo.calibration import (
    CalibrationResult,
    CalibrationSample,
    CamerasBinding,
    CameraBinding,
    load_calibration,
    save_calibration,
)


def _sample(i: int) -> CalibrationSample:
    T = np.eye(4)
    T[:3, 3] = [0.1 * i, 0.0, 0.0]
    T[:3, :3] = np.array(
        [
            [np.cos(i * 0.1), -np.sin(i * 0.1), 0.0],
            [np.sin(i * 0.1), np.cos(i * 0.1), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return CalibrationSample(
        pose_label=f"p{i}",
        T_gripper2base=T,
        T_marker2cam=T,
        aruco_reprojection_px=0.1,
        inlier=True,
    )


def _result() -> CalibrationResult:
    return CalibrationResult(
        schema_version=1,
        frame_convention="wrc_demo",
        frame_convention_note="test",
        created_at="2026-07-20T12:00:00Z",
        method="joint_se3_lm_huber",
        mode="eye_to_hand",
        camera_serial="AY3Z331006L",
        camera_index_when_bound=0,
        marker_size_m=0.10,
        aruco_dict="4x4_50",
        aruco_marker_id=0,
        n_samples=4,
        n_inliers=4,
        n_outliers=0,
        outlier_indices=(),
        T_cam2base=np.eye(4).tolist(),
        T_marker2gripper=np.eye(4).tolist(),
        metrics={"translation_rmse_mm": 1.0, "rotation_rmse_deg": 0.1},
        samples_for_repro=tuple(_sample(i) for i in range(4)),
    )


def test_save_load_roundtrip(tmp_path: Path):
    res = _result()
    p = tmp_path / "hand_eye_top"
    json_path, npz_path = save_calibration(res, p, K=np.eye(3), D=np.zeros(5))
    assert json_path.exists()
    assert npz_path.exists()
    loaded = load_calibration(json_path)
    assert loaded.schema_version == res.schema_version
    assert loaded.frame_convention == res.frame_convention
    assert loaded.camera_serial == res.camera_serial
    assert loaded.n_samples == res.n_samples
    assert np.allclose(loaded.T_cam2base, res.T_cam2base)
    assert np.allclose(loaded.T_marker2gripper, res.T_marker2gripper)
    assert loaded.metrics["translation_rmse_mm"] == 1.0
    # K/D sidecar
    arr = np.load(npz_path)
    assert "K" in arr.files
    assert np.allclose(arr["K"], np.eye(3))


def test_calibration_result_to_json_includes_samples():
    res = _result()
    d = res.to_json()
    assert "samples_for_repro" in d
    assert len(d["samples_for_repro"]) == 4
    assert d["samples_for_repro"][0]["pose_label"] == "p0"


def test_cameras_binding_roundtrip(tmp_path: Path):
    p = tmp_path / "cameras.json"
    binding = CamerasBinding(
        top=CameraBinding(serial="T0", index_when_bound=0),
        wrist=CameraBinding(serial="W1", index_when_bound=1),
        updated_at="2026-07-20T12:00:00Z",
    )
    p.write_text(json.dumps(binding.to_json()))
    loaded = CamerasBinding.from_json(json.loads(p.read_text()))
    assert loaded.top is not None and loaded.top.serial == "T0"
    assert loaded.wrist is not None and loaded.wrist.serial == "W1"


def test_load_missing_file_raises(tmp_path: Path):
    with __import__("pytest").raises(FileNotFoundError):
        load_calibration(tmp_path / "nope.json")


# ─────────────────────────────────────────────────────────────────────────
# Regression: ``calib_top_orbbec._verify`` math must compute marker
# origin in base frame as ``(X @ M)[:3, 3]`` (camera path), not the
# pre-fix ``(G @ Y @ inv(M) @ p_cam)[:3]`` (which was approximately
# ``X[:3, 3]``, the camera origin -- a constant, not the marker).
# See HANDOFF §11 issue 14.
# ─────────────────────────────────────────────────────────────────────────


def _make_verify_fixture(tmp_path, marker_centroid=np.array([0.30, -0.05, 0.28])):
    """Build a CalibrationResult whose inlier samples are all consistent
    with hand-eye equation G @ Y = X @ M, and whose averaged marker
    base-frame position (camera-path) equals ``marker_centroid`` to
    sub-mm precision."""
    from wrc_demo.calibration import (
        CalibrationResult, CalibrationSample, save_calibration,
    )

    # Pick X (camera in base) and Y (marker in gripper) with hand-eye
    # consistent marker position == marker_centroid.
    # Sample design: every sample has G_i == same value (gripper doesn't
    # move in the test) and M_i computed from X, Y so that
    # X @ M_i has translation == marker_centroid.
    X = np.eye(4)
    X[:3, 3] = [0.10, 0.05, 0.85]  # camera 85 cm above + 10cm fwd + 5cm left
    Y = np.eye(4)
    # Choose Y so that (G @ Y)[:3, 3] == marker_centroid. Set G = I.
    G = np.eye(4)
    # (G @ Y)[:3, 3] = Y[:3, 3] = marker_centroid
    Y_no_offset = np.eye(4)
    Y_no_offset[:3, 3] = marker_centroid - G[:3, 3]

    # For each sample, solve for M_i from G_i @ Y = X @ M_i:
    #   M_i = inv(X) @ G_i @ Y
    samples = []
    # Use 5 samples that share the same G (so X @ M_i is constant)
    for i in range(5):
        G_i = G.copy()
        M_i = np.linalg.inv(X) @ G_i @ Y_no_offset
        samples.append(
            CalibrationSample(
                pose_label=f"test_{i}",
                T_gripper2base=G_i,
                T_marker2cam=M_i,
                aruco_reprojection_px=0.5,
                inlier=True,
            )
        )

    res = CalibrationResult(
        schema_version=1,
        frame_convention="wrc_demo",
        frame_convention_note="test fixture",
        created_at="1970-01-01T00:00:00Z",
        method="test_fixture",
        mode="eye_to_hand",
        camera_serial="AY3Z331006L",
        camera_index_when_bound=0,
        marker_size_m=0.10,
        aruco_dict="4x4_50",
        aruco_marker_id=0,
        n_samples=len(samples),
        n_inliers=len(samples),
        n_outliers=0,
        outlier_indices=(),
        T_cam2base=X,
        T_marker2gripper=Y_no_offset,
        metrics={"translation_rmse_mm": 0.0},
        samples_for_repro=tuple(samples),
    )
    p = tmp_path / "cal.json"
    save_calibration(res, p)
    return res, p
