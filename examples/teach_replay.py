#!/usr/bin/env python3
"""reBotArm 示教复播 demo (v2 主从臂遥操)。

启动 → 选槽位 → 读 slot{N}.json + checksum 验证 → RebotArmEndPose.move_joints_traj
→ safe_home() 回电机机械零点。

Usage:
    python teach_replay.py [--slot N] [--dry-run]

Spec: docs/specs/teach_record_replay.md §4.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# vendored SDK 注入（demo 不在 wrc_demo 包内，wrc_demo/__init__.py 不会被触发）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from reBotArm_control_py.actuator import RebotArm
from wrc_demo.skills import teach as teach_mod


RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "data" / "recordings"


def main() -> int:
    parser = argparse.ArgumentParser(description="reBotArm 示教复播 demo (主从臂遥操)")
    parser.add_argument("--slot", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="槽位 1-5 (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连硬件, 只验证 JSON 读 + checksum")
    args = parser.parse_args()

    try:
        if args.dry_run:
            result = teach_mod.dry_run_replay(args.slot, RECORDINGS_DIR)
            return 0 if result.get("ok") else 1
        return _replay_real(args.slot, RECORDINGS_DIR)
    except Exception as e:
        print(f"[teach_replay] 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def _replay_real(slot: int, recordings_dir: Path) -> int:
    """实机复播: slave.connect() + teach_mod.replay_real() (内部用 RebotArmEndPose + safe_home)。"""
    print(f"[teach_replay] 启动 slot={slot}")

    slave = RebotArm()
    slave.connect()
    slave.arm.mode_pos_vel()
    if slave.has_gripper:
        slave.gripper.mode_pos_vel()

    try:
        result = teach_mod.replay_real(slave, slot, recordings_dir)
        if result.get("ok"):
            print(f"[teach_replay] {result['frame_count']} 帧, "
                  f"{result['duration_s']:.2f}s, replayed_to_home={result.get('replayed_to_home')}")
        return 0 if result.get("ok") else 1
    finally:
        try:
            slave.stop_control_loop()
        except Exception:
            pass
        slave.disconnect()


if __name__ == "__main__":
    sys.exit(main())