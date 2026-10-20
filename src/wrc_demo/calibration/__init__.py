"""Hand-eye calibration utilities for the wrc_demo top-camera bring-up.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

This module is the single source of truth for the top-camera hand-eye
calibration product (X = T_cam2base, Y = T_marker2gripper). The CLI lives
in ``wrc_demo.scripts.calib_top_orbbec``; nothing here imports
``pyorbbecsdk`` or any arm SDK, so the package is importable in the test
env (mock stack) without hardware.
"""

from __future__ import annotations

from .calib_dataset import (
    CalibrationResult,
    CalibrationSample,
    CameraBinding,
    CamerasBinding,
    SCHEMA_VERSION,
    load_calibration,
    save_calibration,
)
from .frame_convention import (
    curobo_to_wrc_demo_marker_offset,
    is_se3,
    round_trip_check,
)
from .handeye_solver import (
    JointSolveResult,
    joint_solve,
    joint_solve_restart,
    se3_log,
    se3_log_residual,
    so3_exp,
    so3_log,
)
from .aruco_session import ArUcoDetection, ArUcoSession, build_aruco_session

__all__ = [
    "SCHEMA_VERSION",
    "CalibrationSample",
    "CalibrationResult",
    "CameraBinding",
    "CamerasBinding",
    "ArUcoSession",
    "ArUcoDetection",
    "build_aruco_session",
    "joint_solve",
    "joint_solve_restart",
    "JointSolveResult",
    "se3_log",
    "se3_log_residual",
    "so3_exp",
    "so3_log",
    "curobo_to_wrc_demo_marker_offset",
    "is_se3",
    "round_trip_check",
    "save_calibration",
    "load_calibration",
]
