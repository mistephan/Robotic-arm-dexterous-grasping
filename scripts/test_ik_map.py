import json
from pathlib import Path

import numpy as np
import yaml

from wrc_demo.control.kinematics import Kinematics
from wrc_demo.grasping.camera_grasp import (
    _grasp_axes_to_rebot_tcp_rotation,
    _canonicalize_parallel_gripper_tcp_rotation,
)

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------
# 1. Load arm config
# ------------------------------------------------------------
arm_cfg = yaml.safe_load(
    (ROOT / "configs/arms/rebot_rs.yaml").read_text()
)

urdf = str(
    ROOT
    / "sdk/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
)

kin = Kinematics(
    model_path=urdf,
    ee_frame=arm_cfg["ee_frame"],
    n_controlled=arm_cfg["n_joints"],
)

home_q = np.array(arm_cfg["home_q"], dtype=float)

# ------------------------------------------------------------
# 2. Load calibrated camera pose
# ------------------------------------------------------------
calib = json.loads(
    (ROOT / "data/calibration/hand_eye_top.json").read_text()
)

T_cam2base = np.array(calib["T_cam2base"], dtype=float)
camera_pos = T_cam2base[:3, 3]

print("camera position in base:", np.round(camera_pos, 4))
print("home FK:", np.round(kin.fk(home_q)[:3, 3], 4))

# ------------------------------------------------------------
# 3. 5x5 workspace grid
#
# 可按你桌面实际工作区修改
# ------------------------------------------------------------
xs = np.linspace(0.20, 0.50, 5)
ys = np.linspace(-0.25, 0.25, 5)

# TCP 抓取中心高度，不是桌面高度
z = 0.06

# ------------------------------------------------------------
# 4. Four in-plane object orientations
#
# 模拟 segmentation mask 短边方向：
# 0°, 45°, 90°, 135°
# ------------------------------------------------------------
yaw_degs = [0, 45, 90, 135]


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("degenerate vector")
    return v / n


def grasp_pose_at(p_base, yaw_deg):
    """
    Approximate camera_grasp.py geometry:

      object -> camera       = approach
      object in-plane angle  = open-axis orientation
      tcp_x                  = -approach
    """
    p_base = np.asarray(p_base, dtype=float)

    # --------------------------------------------------------
    # approach: object -> camera
    # --------------------------------------------------------
    approach = normalize(camera_pos - p_base)

    # --------------------------------------------------------
    # nominal object short-edge direction in BASE XY plane
    #
    # yaw=0   => +X
    # yaw=90  => +Y
    # --------------------------------------------------------
    a = np.deg2rad(yaw_deg)
    open_axis = np.array(
        [np.cos(a), np.sin(a), 0.0],
        dtype=float,
    )

    # Project open axis onto plane perpendicular to approach
    open_axis = (
        open_axis
        - np.dot(open_axis, approach) * approach
    )

    # Rare degeneracy fallback
    if np.linalg.norm(open_axis) < 1e-8:
        open_axis = np.array([0.0, 1.0, 0.0])
        open_axis -= np.dot(open_axis, approach) * approach

    open_axis = normalize(open_axis)

    # Match camera_grasp geometry
    grip_axis = normalize(
        np.cross(open_axis, approach)
    )

    # Re-orthogonalize
    open_axis = normalize(
        np.cross(approach, grip_axis)
    )

    tcp_R = _grasp_axes_to_rebot_tcp_rotation(
        grip_axis,
        open_axis,
        approach,
    )

    tcp_R = _canonicalize_parallel_gripper_tcp_rotation(
        tcp_R
    )

    T = np.eye(4)
    T[:3, :3] = tcp_R
    T[:3, 3] = p_base

    return T


# ------------------------------------------------------------
# 5. Run 5x5x4 IK test
# ------------------------------------------------------------
results = {}

print()
print("=== IK workspace map: 5x5 x 4 orientations ===")
print("Each cell = number of successful orientations / 4")
print()

for y in ys:
    row = []

    for x in xs:
        successes = 0
        per_angle = {}

        for yaw_deg in yaw_degs:
            T = grasp_pose_at(
                [x, y, z],
                yaw_deg,
            )

            res = kin.ik(
                T_target=T,
                q_init=home_q,
                retries=8,
                seed=0,
            )

            per_angle[yaw_deg] = res

            if res.success:
                successes += 1

        results[(float(x), float(y))] = per_angle

        pct = int(round(100 * successes / len(yaw_degs)))
        row.append(f"{pct:3d}%")

    print(
        f"y={y:+.3f}:  "
        + " | ".join(row)
    )

# ------------------------------------------------------------
# 6. Detailed report
# ------------------------------------------------------------
print()
print("=== Detailed results ===")

for y in ys:
    for x in xs:
        per_angle = results[(float(x), float(y))]

        ok_angles = []
        fail_angles = []

        for yaw_deg in yaw_degs:
            res = per_angle[yaw_deg]

            if res.success:
                ok_angles.append(yaw_deg)
            else:
                fail_angles.append(
                    (
                        yaw_deg,
                        res.error,
                    )
                )

        print()
        print(
            f"Point x={x:.3f}, y={y:+.3f}, z={z:.3f}"
        )

        print(
            "  success:",
            f"{len(ok_angles)}/4",
            f"({len(ok_angles) * 25}%)",
        )

        print(
            "  OK yaw:",
            ok_angles if ok_angles else "none",
        )

        if fail_angles:
            print(
                "  FAIL:",
                ", ".join(
                    f"{yaw}deg(err={err:.4f})"
                    for yaw, err in fail_angles
                ),
            )

# ------------------------------------------------------------
# 7. Print successful joint solutions
# ------------------------------------------------------------
print()
print("=== Successful IK solutions ===")

for (x, y), per_angle in results.items():
    for yaw_deg, res in per_angle.items():
        if not res.success:
            continue

        print(
            f"x={x:.3f} "
            f"y={y:+.3f} "
            f"z={z:.3f} "
            f"yaw={yaw_deg:3d} "
            f"iter={res.iterations:3d} "
            f"err={res.error:.6f}"
        )
        print(
            "  q =",
            np.round(res.q, 3),
        )