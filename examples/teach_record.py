#!/usr/bin/env python3
"""reBotArm 示教录制 demo (v2 主从臂遥操)。

启动 → master.open() + slave.connect()/mode_pos_vel() → pynput 后台监听 →
30Hz 主从臂循环 → 按 q 录当前 slot / 按 Esc 取消（不保存）。

Usage:
    python teach_record.py [--slot N] [--dry-run]

Spec: docs/specs/teach_record_replay.md §4.2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# vendored SDK 注入（demo 不在 wrc_demo 包内，wrc_demo/__init__.py 不会被触发）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from reBotArm_control_py.actuator import RebotArm
from wrc_demo.skills import teach as teach_mod
from wrc_demo.skills.master_arm import PiPERMateArm


RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "data" / "recordings"


def main() -> int:
    parser = argparse.ArgumentParser(description="reBotArm 示教录制 demo (主从臂遥操)")
    parser.add_argument("--slot", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="槽位 1-5 (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连硬件, 模拟 1 秒录制验证逻辑")
    args = parser.parse_args()

    slot_path = RECORDINGS_DIR / f"slot{args.slot}.json"

    # 覆盖询问
    if slot_path.exists():
        size = slot_path.stat().st_size
        print(f"[teach_record] slot{args.slot}.json 已存在 ({size} bytes, schema v2)")
        print(f"覆盖？ [y/N] ", end="", flush=True)
        try:
            answer = sys.stdin.readline().strip().lower()
        except EOFError:
            print("\n[teach_record] 无 stdin 输入，取消")
            return 0
        if answer != "y":
            print("[teach_record] 取消")
            return 0

    try:
        if args.dry_run:
            result = teach_mod.dry_run_record(args.slot, RECORDINGS_DIR)
            return 0 if result.get("ok") else 1
        return _record_real(args.slot, RECORDINGS_DIR)
    except Exception as e:
        print(f"[teach_record] 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _record_real(slot: int, recordings_dir: Path) -> int:
    """实机录制：master.open() + slave.connect() + teach_mod.record_real()。"""
    print(f"[teach_record] 启动 slot={slot}")

    # 1. 主臂
    master = PiPERMateArm(port="/dev/ttyUSB0", gripper_exist=True)
    print("[teach_record] 主臂 PiPER Mate: opening /dev/ttyUSB0 ...")
    master.open()
    print("[teach_record] 主臂 ready")

    # 2. 从臂
    slave = RebotArm()
    slave.connect()
    slave.arm.mode_pos_vel()  # POS_VEL 模式（demo 同款）
    if slave.has_gripper:
        slave.gripper.mode_pos_vel()
    slave.disable_all()
    time.sleep(0.1)
    slave.enable_all()
    print(f"[teach_record] 从臂 ready (POS_VEL, {slave.arm.num_joints} joints)")

    try:
        result = teach_mod.record_real(master, slave, slot, recordings_dir)
        if result.get("ok"):
            print(f"[teach_record] {result['frame_count']} 帧, {result['duration_s']:.2f}s")
        return 0 if result.get("ok") else 1
    finally:
        try:
            slave.stop_control_loop()
        except Exception:
            pass
        slave.disconnect()
        master.close()


if __name__ == "__main__":
    sys.exit(main())