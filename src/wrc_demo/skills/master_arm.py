"""PiPER Mate 主臂 SDK 封装 + 主→从关节映射。

参考项目 (Welt-liu/rebot-b601-102-record-demo):
    Python_SDK/pipermate_sdk.py    - PiPER Mate SDK 包装
    Python_SDK/rebot_so102_record.py - 主从臂 loop 模板

API:
    PiPERMateArm(port, gripper_exist).open() → 用 get_joint_states() 读舵机
    master_to_slave(master_q_deg) → np.ndarray 7 floats (rad)

Spec: docs/specs/teach_record_replay.md §5.2
"""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from fashionstar_uart_sdk.uart_pocket_handler import PortHandler
except ImportError as e:
    raise RuntimeError(
        "fashionstar-uart-sdk not installed; install it into the Python "
        "interpreter that runs wrc_demo, e.g.:\n"
        "    uv pip install fashionstar-uart-sdk --python $(which python)\n"
        "    conda run -n <env> pip install fashionstar-uart-sdk\n"
        "See README for the recommended environment setup."
    ) from e


# ── 关节角度限制（reBot URDF 真实限制，硬编码，不做校准）────────────────
# 修正 (2026-8-14): demo JOINT_ANGLE_LIMITS 内部不一致（README 说 joint2=0~180
# 但代码 -180~0）。reBot URDF 的关节限制（来自 sdk/urdf/00-arm-rs_asm-v3）：
#   joint1: ±2.800 rad ≈ ±160°
#   joint2: 0 ~ 3.140 rad = 0 ~ 180°  (水平臂，向上)
#   joint3: 0 ~ 3.140 rad = 0 ~ 180°  (肘关节)
#   joint4: ±1.570 rad = ±90°
#   joint5: ±1.570 rad = ±90°
#   joint6: ±3.140 rad = ±180°
#   gripper: 0 ~ 50 mm (机械行程)
JOINT_ANGLE_LIMITS: Dict[str, tuple] = {
    "joint1": (-160.0, 160.0),
    "joint2": (0.0, 180.0),
    "joint3": (0.0, 180.0),
    "joint4": (-90.0, 90.0),
    "joint5": (-90.0, 90.0),
    "joint6": (-180.0, 180.0),
    "gripper": (-320.0, 0.0),
}
GRIPPER_RATIO = 6  # reBot 夹爪放缩比（demo 同款）


class PiPERMateArm:
    """PiPER Mate 主臂 SDK 封装（FashionStar UART 协议）。"""

    def __init__(self, port: str = "/dev/ttyUSB0", gripper_exist: bool = True,
                 baudrate: int = 1_000_000):
        self.port = port
        self.baudrate = baudrate
        self.gripper_exist = gripper_exist
        self._handler: PortHandler | None = None
        if gripper_exist:
            self.joint_names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
            self.servo_ids = list(range(7))
        else:
            self.joint_names = [f"joint{i}" for i in range(1, 7)]
            self.servo_ids = list(range(6))

    def open(self) -> None:
        """打开串口、解锁舵机、复位多圈计数。"""
        self._handler = PortHandler(self.port, self.baudrate)
        self._handler.openPort()
        # 解锁所有舵机（"unlocked", 900 ms）让舵机能自由转动
        for sid in self.servo_ids:
            try:
                self._handler.write["Stop_On_Control_Mode"](sid, "unlocked", 900)
            except Exception as e:
                print(f"[PiPERMate] 舵机 {sid} 解锁失败: {e}")
        # 复位多圈计数（消除累计误差）
        try:
            self._handler.reset_multi_turn_angle(0xFF)
        except Exception as e:
            print(f"[PiPERMate] reset_multi_turn_angle 失败: {e}")

    def close(self) -> None:
        """关闭串口。"""
        if self._handler is not None:
            try:
                self._handler.closePort()
            except Exception:
                pass
            self._handler = None

    def get_joint_states(self) -> Dict[str, float]:
        """读所有舵机当前角度（degrees）。

        Returns:
            {"joint1": 12.3, "joint2": -45.6, ..., "gripper": 0.0}

        Raises:
            RuntimeError: 如果串口未打开
            Exception: 底层 SDK 异常（serial.SerialException, OSError）
        """
        if self._handler is None:
            raise RuntimeError("PiPERMateArm not opened; call open() first")
        servos_id_map = {name: sid for name, sid in zip(self.joint_names, self.servo_ids)}
        monitor_data = self._handler.sync_read["Monitor"](servos_id_map)
        return {
            name: float(monitor_data[name].current_position)
            for name in self.joint_names
        }


def master_to_slave(master_q_deg: Dict[str, float]) -> np.ndarray:
    """PiPER Mate (度) → reBot 关节角度 (radians)。

    硬编码映射（用户确认不做校准，参考 demo `servoangle2jointstate`）:
    - joint1, joint4, joint6 方向取反
    - 应用 JOINT_ANGLE_LIMITS 限幅
    - gripper 取反 × GRIPPER_RATIO=6, 归一化（> -14 视为 0）

    Returns:
        np.ndarray 7 floats (joint1..joint6 + gripper), 单位 radians
    """
    slave_deg: Dict[str, float] = {}
    for i in range(1, 7):
        name = f"joint{i}"
        angle = float(master_q_deg.get(name, 0.0))
        if i in (1, 4, 6):  # joint1, joint4, joint6 取反
            angle = -angle
        lim = JOINT_ANGLE_LIMITS[name]
        angle = max(lim[0], min(angle, lim[1]))
        slave_deg[name] = angle

    # gripper 单独处理：取反 + 放缩 + 归一化
    grip_deg = float(master_q_deg.get("gripper", 0.0))
    grip_scaled = -grip_deg * GRIPPER_RATIO
    if grip_scaled > -14:
        grip_scaled = 0.0
    grip_lim = JOINT_ANGLE_LIMITS["gripper"]
    grip_scaled = max(grip_lim[0], min(grip_scaled, grip_lim[1]))
    slave_deg["gripper"] = grip_scaled

    return np.deg2rad(
        np.array([slave_deg[f"joint{i}"] for i in range(1, 7)] +
                 [slave_deg["gripper"]], dtype=np.float64)
    )