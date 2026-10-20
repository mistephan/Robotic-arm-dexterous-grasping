"""Core datatypes shared across perception, memory, control and the agent.

Conventions (deviations from the reBot-DevArm-Grasp baseline, on purpose):
- Depth is float32 METERS aligned to color, never raw sensor units. The L515
  reports a 0.25 mm depth scale while D4xx report 1 mm; keeping metric depth in
  the Frame kills that whole class of bug.
- All 3D poses handed between modules are in the robot BASE frame unless the
  name says otherwise (`*_cam` for camera frame).
- Rotations are 3x3 matrices; transforms are 4x4 row-major numpy arrays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """One camera observation. `depth_m` may be None for RGB-only cameras."""

    rgb: np.ndarray  # (H, W, 3) uint8, BGR (OpenCV convention)
    depth_m: np.ndarray | None  # (H, W) float32 meters, aligned to rgb, 0 = invalid
    K: np.ndarray  # (3, 3) float64 intrinsics of the rgb image
    t: float = field(default_factory=time.monotonic)
    frame_id: int = 0
    depth_source: str = "none"  # "sensor" | "plane" | "mono" | "none"
    T_base_cam: np.ndarray | None = None  # (4,4) per-frame cam->base for
    # eye-in-hand cameras (extrinsics move with the arm); None = use the
    # camera profile's static extrinsics

    @property
    def has_depth(self) -> bool:
        return self.depth_m is not None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h


@dataclass
class Detection:
    """One detected object instance in a Frame."""

    label: str
    conf: float
    bbox: np.ndarray  # (4,) float32 xyxy in pixels
    mask: np.ndarray | None = None  # (H, W) bool, full-frame
    obb: np.ndarray | None = None  # (4, 2) float32 oriented box corners, pixels

    def center_px(self) -> tuple[float, float]:
        if self.obb is not None:
            c = self.obb.mean(axis=0)
            return float(c[0]), float(c[1])
        x0, y0, x1, y1 = self.bbox
        return float((x0 + x1) / 2), float((y0 + y1) / 2)


@dataclass
class ObjectFix:
    """A 3D localization of one object in the robot base frame."""

    label: str
    position: np.ndarray  # (3,) base frame, meters (OBB center of mask points)
    points: np.ndarray  # (N, 3) base-frame points backing the fix
    detection: Detection
    extent: np.ndarray  # (3,) OBB extents (sorted descending), meters
    axes: np.ndarray  # (3, 3) OBB axes as columns, base frame
    t: float = field(default_factory=time.monotonic)


@dataclass
class Grasp:
    """A parallel-jaw grasp target in the robot base frame."""

    position: np.ndarray  # (3,) TCP position at grasp, meters
    rotation: np.ndarray  # (3, 3) TCP rotation at grasp
    width_m: float  # required jaw opening
    approach: np.ndarray  # (3,) unit vector, direction the tool travels in
    quality: float = 0.0
    label: str = ""

    def pregrasp_position(self, offset_m: float) -> np.ndarray:
        return self.position - self.approach * offset_m


@dataclass
class RobotState:
    q: np.ndarray  # (n,) joint positions, rad
    dq: np.ndarray | None = None  # (n,) joint velocities, rad/s
    tau: np.ndarray | None = None  # (n,) joint torques, Nm
    gripper_pos: float = 0.0  # gripper motor position, rad
    gripper_valid: bool = True  # False when the gripper read failed (unknown)
    t: float = field(default_factory=time.monotonic)


class SkillError(RuntimeError):
    """A skill failed in a way the agent should reason about (not a crash)."""


class SafetyViolation(RuntimeError):
    """A motion command was rejected by the safety harness."""


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    pts = np.asarray(points, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]
