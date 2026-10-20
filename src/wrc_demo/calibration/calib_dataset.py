"""Calibration sample / result dataclasses + JSON I/O.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

The calibration product is two SE(3) matrices:

- X = T_cam2base   (eye-to-hand: camera is fixed in the world)
- Y = T_marker2gripper  (board mounted on the gripper, may be off-origin)

The hand-eye equation is ``G_i Y = X M_i`` for every sample i, where
``G_i = T_gripper2base`` comes from FK and ``M_i = T_marker2cam`` comes
from ArUco PnP. See PLAN.md §13 for the derivation.

The dataclasses are intentionally frozen so the solver/test code can
treat them as values. ``CalibrationResult.metrics`` follows the
PLAN.md §12.3 schema so reviewers can diff calibration runs in git.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
FRAME_CONVENTION = "wrc_demo"


def _mat(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class CalibrationSample:
    """One (FK, PnP) observation captured during calibration.

    `T_gripper2base` is computed by ``wrc_demo.control.kinematics.Kinematics.fk``.
    `T_marker2cam` is computed by ``cv2.aruco`` PnP against the SDK K matrix.
    Both are 4x4 SE(3) arrays.
    """

    pose_label: str
    T_gripper2base: np.ndarray
    T_marker2cam: np.ndarray
    aruco_reprojection_px: float = 0.0
    inlier: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "pose_label": self.pose_label,
            "T_gripper2base": np.asarray(self.T_gripper2base).tolist(),
            "T_marker2cam": np.asarray(self.T_marker2cam).tolist(),
            "aruco_reprojection_px": float(self.aruco_reprojection_px),
            "inlier": bool(self.inlier),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CalibrationSample":
        return cls(
            pose_label=str(d["pose_label"]),
            T_gripper2base=_mat(d["T_gripper2base"]),
            T_marker2cam=_mat(d["T_marker2cam"]),
            aruco_reprojection_px=float(d.get("aruco_reprojection_px", 0.0)),
            inlier=bool(d.get("inlier", True)),
        )


@dataclass(frozen=True)
class CameraBinding:
    """One camera's serial binding recorded after on-rig enumeration."""

    serial: str
    index_when_bound: int = -1


@dataclass(frozen=True)
class CamerasBinding:
    """The full binding (top + wrist). Persists to ``data/calibration/cameras.json``."""

    top: CameraBinding | None = None
    wrist: CameraBinding | None = None
    updated_at: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": self.updated_at,
        }
        if self.top is not None:
            out["top"] = {
                "serial": self.top.serial,
                "index_when_bound": self.top.index_when_bound,
            }
        if self.wrist is not None:
            out["wrist"] = {
                "serial": self.wrist.serial,
                "index_when_bound": self.wrist.index_when_bound,
            }
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CamerasBinding":
        return cls(
            top=CameraBinding(**d["top"]) if d.get("top") else None,
            wrist=CameraBinding(**d["wrist"]) if d.get("wrist") else None,
            updated_at=str(d.get("updated_at", "")),
        )


@dataclass(frozen=True)
class CalibrationResult:
    """Solver output + provenance + metrics. Persists to JSON + sidecar NPZ.

    The sidecar NPZ (``<base>.npz``) carries the 3x3 intrinsics matrix and
    distortion coefficients (binary, no place in JSON). All scalar/metric
    fields live in the JSON for diff/audit.
    """

    schema_version: int
    frame_convention: str
    frame_convention_note: str
    created_at: str
    method: str
    mode: str  # "eye_to_hand" | "eye_in_hand"
    camera_serial: str
    camera_index_when_bound: int
    marker_size_m: float
    aruco_dict: str
    aruco_marker_id: int
    n_samples: int
    n_inliers: int
    n_outliers: int
    outlier_indices: tuple[int, ...]
    T_cam2base: np.ndarray
    T_marker2gripper: np.ndarray
    metrics: dict[str, float]
    samples_for_repro: tuple[CalibrationSample, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "frame_convention": self.frame_convention,
            "frame_convention_note": self.frame_convention_note,
            "created_at": self.created_at,
            "method": self.method,
            "mode": self.mode,
            "camera_serial": self.camera_serial,
            "camera_index_when_bound": int(self.camera_index_when_bound),
            "marker_size_m": float(self.marker_size_m),
            "aruco_dict": self.aruco_dict,
            "aruco_marker_id": int(self.aruco_marker_id),
            "n_samples": int(self.n_samples),
            "n_inliers": int(self.n_inliers),
            "n_outliers": int(self.n_outliers),
            "outlier_indices": [int(i) for i in self.outlier_indices],
            "T_cam2base": np.asarray(self.T_cam2base).tolist(),
            "T_marker2gripper": np.asarray(self.T_marker2gripper).tolist(),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "samples_for_repro": [s.to_json() for s in self.samples_for_repro],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CalibrationResult":
        return cls(
            schema_version=int(d["schema_version"]),
            frame_convention=str(d["frame_convention"]),
            frame_convention_note=str(d.get("frame_convention_note", "")),
            created_at=str(d["created_at"]),
            method=str(d["method"]),
            mode=str(d["mode"]),
            camera_serial=str(d["camera_serial"]),
            camera_index_when_bound=int(d.get("camera_index_when_bound", -1)),
            marker_size_m=float(d["marker_size_m"]),
            aruco_dict=str(d["aruco_dict"]),
            aruco_marker_id=int(d["aruco_marker_id"]),
            n_samples=int(d["n_samples"]),
            n_inliers=int(d["n_inliers"]),
            n_outliers=int(d["n_outliers"]),
            outlier_indices=tuple(int(i) for i in d.get("outlier_indices", ())),
            T_cam2base=_mat(d["T_cam2base"]),
            T_marker2gripper=_mat(d["T_marker2gripper"]),
            metrics={k: float(v) for k, v in d.get("metrics", {}).items()},
            samples_for_repro=tuple(
                CalibrationSample.from_json(s)
                for s in d.get("samples_for_repro", [])
            ),
        )


def save_calibration(
    result: CalibrationResult,
    base_path: str | Path,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Persist a calibration result to ``<base>.json`` (+ ``<base>.npz`` for K/D).

    Returns the (json_path, npz_path) pair actually written. Fail-closed:
    overwrites only when called explicitly; the caller decides the path.
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json") if base.suffix != ".json" else base
    npz_path = base.with_suffix(".npz")
    with open(json_path, "w") as f:
        json.dump(result.to_json(), f, indent=2, sort_keys=True)
    arrays: dict[str, np.ndarray] = {}
    if K is not None:
        arrays["K"] = np.asarray(K, dtype=np.float64)
    if D is not None:
        arrays["D"] = np.asarray(D, dtype=np.float64)
    if arrays:
        np.savez(npz_path, **arrays)
    else:
        # Always materialize an empty NPZ so downstream loaders don't have to
        # branch on "exists but empty". Useful for diff in CI.
        np.savez(npz_path)
    return json_path, npz_path


def load_calibration(path: str | Path) -> CalibrationResult:
    """Load a calibration result from its JSON side.

    Raises ``FileNotFoundError`` if the JSON side is missing -- NPZ-only
    legacy cuRobo files should go through ``Extrinsics.from_config`` with
    the cuRobo loader branch.
    """
    p = Path(path)
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        return CalibrationResult.from_json(json.load(f))
