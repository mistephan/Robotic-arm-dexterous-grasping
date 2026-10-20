"""master_arm.py 单元测试。

覆盖 spec §6.3 测试用例 5-7 + Q-α Q-θ Q-ι 等推断。

跑：pytest tests/test_master_arm.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

from wrc_demo.skills.master_arm import (
    GRIPPER_RATIO,
    JOINT_ANGLE_LIMITS,
    PiPERMateArm,
    master_to_slave,
)


# ── Q-α: joint1, joint4, joint6 方向取反 ─────────────────────────────────
def test_joint1_sign_flipped():
    """joint1 输入 +90° → 输出 -90° (rad)。"""
    r = master_to_slave(_full_dict(joint1=90))
    assert np.rad2deg(r[0]) == pytest.approx(-90.0, abs=0.01)


def test_joint2_not_flipped():
    """joint2 输入 +90° → 输出 +90° (不取反)。"""
    r = master_to_slave(_full_dict(joint2=90))
    assert np.rad2deg(r[1]) == pytest.approx(90.0, abs=0.01)


def test_joint3_not_flipped():
    """joint3 输入 +90° → 输出 +90°。"""
    r = master_to_slave(_full_dict(joint3=90))
    assert np.rad2deg(r[2]) == pytest.approx(90.0, abs=0.01)


def test_joint4_sign_flipped():
    """joint4 输入 +60° → 输出 -60°。"""
    r = master_to_slave(_full_dict(joint4=60))
    assert np.rad2deg(r[3]) == pytest.approx(-60.0, abs=0.01)


def test_joint5_not_flipped():
    """joint5 输入 +45° → 输出 +45°。"""
    r = master_to_slave(_full_dict(joint5=45))
    assert np.rad2deg(r[4]) == pytest.approx(45.0, abs=0.01)


def test_joint6_sign_flipped():
    """joint6 输入 +100° → 输出 -100°。"""
    r = master_to_slave(_full_dict(joint6=100))
    assert np.rad2deg(r[5]) == pytest.approx(-100.0, abs=0.01)


# ── Q-ι: 关节角度限制（按 URDF）──────────────────────────────────────────
def test_joint1_clipped_upper():
    """joint1 输入 +200° → 取反为 -200° → 限位到 -160° (URDF 下限)。"""
    r = master_to_slave(_full_dict(joint1=200))
    assert np.rad2deg(r[0]) == pytest.approx(-160.0, abs=0.01)


def test_joint1_clipped_lower():
    """joint1 输入 -200° → 取反为 +200° → 限位到 +160° (URDF 上限)。"""
    r = master_to_slave(_full_dict(joint1=-200))
    assert np.rad2deg(r[0]) == pytest.approx(160.0, abs=0.01)


def test_joint2_clipped_lower():
    """joint2 是 0~180° (URDF)，输入 -10° → 限位到 0°。"""
    r = master_to_slave(_full_dict(joint2=-10))
    assert np.rad2deg(r[1]) == pytest.approx(0.0, abs=0.01)


def test_joint3_clipped_lower():
    """joint3 是 0~180° (URDF)，输入 -10° → 限位到 0°。"""
    r = master_to_slave(_full_dict(joint3=-10))
    assert np.rad2deg(r[2]) == pytest.approx(0.0, abs=0.01)


def test_joint4_clipped():
    """joint4 是 ±90° (URDF)，输入 100° → 限位到 90°。"""
    r = master_to_slave(_full_dict(joint4=100))
    assert np.rad2deg(r[3]) == pytest.approx(-90.0, abs=0.01)  # 取反后 -90


# ── gripper 处理 (GRIPPER_RATIO=6 + > -14 归零化) ─────────────────────────
def test_gripper_zero_input():
    """gripper 输入 0° → 输出 0°。"""
    r = master_to_slave(_full_dict(gripper=0))
    assert np.rad2deg(r[6]) == pytest.approx(0.0, abs=0.01)


def test_gripper_small_input_normalized_to_zero():
    """gripper 输入 1° → -1*6=-6 > -14 → 归零化到 0°。"""
    r = master_to_slave(_full_dict(gripper=1))
    assert np.rad2deg(r[6]) == pytest.approx(0.0, abs=0.01)


def test_gripper_medium_input_kept():
    """gripper 输入 5° → -5*6=-30 < -14 → 保持 -30°。"""
    r = master_to_slave(_full_dict(gripper=5))
    assert np.rad2deg(r[6]) == pytest.approx(-30.0, abs=0.01)


def test_gripper_ratio():
    """GRIPPER_RATIO 常量 = 6（demo 同款）。"""
    assert GRIPPER_RATIO == 6


# ── 单位转换 (deg → rad) ──────────────────────────────────────────────────
def test_output_in_radians():
    """master_to_slave 输出是弧度（不是度）。"""
    r = master_to_slave(_full_dict(joint1=180))
    # -180° (取反 + 限位) 转 rad 应在 [-3.14, -2.79] 区间
    assert -np.pi < r[0] < -2.79


def test_array_shape():
    """master_to_slave 返回 7 floats（6 关节 + gripper）。"""
    r = master_to_slave(_full_dict())
    assert r.shape == (7,)
    assert r.dtype == np.float64


# ── 默认值 (缺字段 → 0) ──────────────────────────────────────────────────
def test_missing_joint_defaults_zero():
    """缺字段默认 0° → 输出 0°。"""
    r = master_to_slave({"joint1": 0})  # 只给 joint1
    assert r[0] == pytest.approx(0.0, abs=0.01)
    assert r[1] == pytest.approx(0.0, abs=0.01)  # joint2 缺 → 0


def test_empty_dict():
    """空字典 → 全部 0。"""
    r = master_to_slave({})
    assert np.allclose(r, 0.0)


# ── Constants ──────────────────────────────────────────────────────────────
def test_joint_angle_limits_urdf_compliant():
    """JOINT_ANGLE_LIMITS 必须符合 reBot URDF 限制（pinocchio 模型）。"""
    # 从 URDF 加载模型验证
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))
    from reBotArm_control_py.kinematics import load_robot_model
    model = load_robot_model()
    # joint1 应该是 ±2.800 rad ≈ ±160°
    for i in range(1, 7):
        joint = model.joints[i]
        lo_rad = model.lowerPositionLimit[joint.idx_q]
        hi_rad = model.upperPositionLimit[joint.idx_q]
        lo_deg = np.rad2deg(lo_rad)
        hi_deg = np.rad2deg(hi_rad)
        # 我们的 JOINT_ANGLE_LIMITS 应该与 URDF 一致（允许 1° 容差）
        lim = JOINT_ANGLE_LIMITS[model.names[i]]
        assert abs(lim[0] - lo_deg) < 1.0, f"{model.names[i]} lower: {lim[0]} vs URDF {lo_deg:.1f}"
        assert abs(lim[1] - hi_deg) < 1.0, f"{model.names[i]} upper: {lim[1]} vs URDF {hi_deg:.1f}"


# ── PiPERMateArm 类基础测试 ─────────────────────────────────────────────────
def test_pipermate_class_exists():
    """PiPERMateArm 类存在，含 open/close/get_joint_states。"""
    arm = PiPERMateArm(port="/dev/ttyUSB0", gripper_exist=True)
    assert hasattr(arm, "open")
    assert hasattr(arm, "close")
    assert hasattr(arm, "get_joint_states")
    assert arm.gripper_exist is True
    assert len(arm.joint_names) == 7  # 6 关节 + gripper
    assert arm.joint_names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    assert arm.servo_ids == [0, 1, 2, 3, 4, 5, 6]


def test_pipermate_class_no_gripper():
    """gripper_exist=False → 6 个关节（无 gripper）。"""
    arm = PiPERMateArm(gripper_exist=False)
    assert arm.gripper_exist is False
    assert len(arm.joint_names) == 6
    assert "gripper" not in arm.joint_names


def test_pipermate_get_joint_states_requires_open():
    """未 open 就调 get_joint_states → RuntimeError。"""
    arm = PiPERMateArm()
    with pytest.raises(RuntimeError, match="not opened"):
        arm.get_joint_states()


# ── Helper ────────────────────────────────────────────────────────────────
def _full_dict(joint1=0, joint2=0, joint3=0, joint4=0, joint5=0, joint6=0, gripper=0):
    """构造完整的 master_q_deg 字典。"""
    return {
        "joint1": joint1, "joint2": joint2, "joint3": joint3,
        "joint4": joint4, "joint5": joint5, "joint6": joint6,
        "gripper": gripper,
    }