"""reBotArm 示教录制-复播业务层（v2: 主从臂遥操方案）。

主从臂方案 (spec §4-5):
    PiPER Mate 主臂（操作员操控）→ master_to_slave() 关节映射 →
    reBot Arm B601-RS 从臂（POS_VEL 跟随）→ 30Hz 记录主+从关节角度 →
    slot{N}.json (schema v2) → 复播插值 → safe_home 回零。

API:
    save_slot(slot, frames, start_master_q, start_slave_q, recordings_dir) -> Path
    load_slot(slot, recordings_dir) -> dict  (含 checksum + schema v2 验证)
    compute_checksum(frames) -> str

    dry_run_record(slot, recordings_dir) -> dict    # 不连硬件
    dry_run_replay(slot, recordings_dir) -> dict    # 不连硬件

Spec: docs/specs/teach_record_replay.md §5.2-§5.7
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# ── 物理常量（参考 demo TeleopLooper）────────────────────────────────────
NUM_SLOTS = 5
SAMPLE_HZ = 30.0       # 录制采样率（demo 同款）
SAMPLE_INTERVAL = 1.0 / SAMPLE_HZ
PLAY_LOOP = True       # 默认循环播放
END_HOLD_TIME = 0.15   # 录制结束时 hold 最后一帧
LOOP_BLEND_TIME = 0.30 # 循环播放时混合时间
RECORD_FILTER_ALPHA = 0.35  # 指数移动平均平滑
MIN_RECORD_INTERVAL = 0.01  # 秒，最短帧间隔
MIN_JOINT_CHANGE = 0.003    # rad，关节变化阈值
TRANSITION_TIME = 0.6   # 复播前过渡到 slot 起点


# ── I/O ────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = 2  # v2: master_q + slave_q (替代 v1 单存 joints)


def compute_checksum(frames: List[dict]) -> str:
    """SHA256(序列化后的 frames 字节)，防 silent 数据损坏。"""
    payload = json.dumps(frames, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_slot(
    slot: int,
    frames: List[dict],
    start_master_q: np.ndarray,
    start_slave_q: np.ndarray,
    recordings_dir: Path,
) -> Path:
    """保存录制数据到 slot{N}.json (spec §10 v2 schema)。"""
    duration_s = round(frames[-1]["t"] - frames[0]["t"], 3) if frames else 0.0
    payload = {
        "schema_version": SCHEMA_VERSION,
        "slot": slot,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frame_count": len(frames),
        "duration_s": duration_s,
        "sample_rate_hz": int(SAMPLE_HZ),
        "start_pose": {
            "master_q": [round(float(x), 6) for x in start_master_q],
            "slave_q": [round(float(x), 6) for x in start_slave_q],
        },
        "checksum_sha256": compute_checksum(frames),
        "frames": frames,
    }
    recordings_dir.mkdir(parents=True, exist_ok=True)
    path = recordings_dir / f"slot{slot}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_slot(slot: int, recordings_dir: Path) -> dict:
    """读 slot{N}.json + 验证 checksum + 验证 schema v2。"""
    path = recordings_dir / f"slot{slot}.json"
    if not path.exists():
        raise FileNotFoundError(f"slot{slot}.json 不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"slot{slot}.json schema_version={data.get('schema_version')} "
            f"!= {SCHEMA_VERSION} (expected v2)"
        )

    recorded = data.get("checksum_sha256")
    computed = compute_checksum(data["frames"])
    if recorded != computed:
        raise ValueError(
            f"slot{slot}.json checksum 不匹配:\:\n"
            f"  recorded: {recorded}\n  computed: {computed}"
        )
    return data


# ── 数学工具（参考 demo）──────────────────────────────────────────────────
def interpolate_joint_states(frame_a: dict, frame_b: dict, t_now: float) -> np.ndarray:
    """smoothstep 插值（3α² - 2α³），返回 7 floats rad。"""
    t0 = frame_a["t"]
    t1 = frame_b["t"]
    sq0 = np.asarray(frame_a["slave_q"], dtype=np.float64)
    sq1 = np.asarray(frame_b["slave_q"], dtype=np.float64)

    if abs(t1 - t0) < 1e-9:
        return sq0.copy()

    alpha = max(0.0, min(1.0, (t_now - t0) / (t1 - t0)))
    alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep

    return sq0 + alpha * (sq1 - sq0)


def blend_joint_states(sq_from: np.ndarray, sq_to: np.ndarray, alpha: float) -> np.ndarray:
    """smoothstep blend (transition 用)。"""
    alpha = max(0.0, min(1.0, alpha))
    alpha = alpha * alpha * (3 - 2 * alpha)
    return sq_from + alpha * (sq_to - sq_from)


def filter_joint_states(
    new_js: np.ndarray, last_js: Optional[np.ndarray], alpha: float = RECORD_FILTER_ALPHA,
) -> np.ndarray:
    """指数移动平均（demo 同款 alpha=0.35）。"""
    if last_js is None:
        return new_js.copy()
    return last_js + alpha * (new_js - last_js)


def joint_states_changed_enough(
    js1: Optional[np.ndarray], js2: Optional[np.ndarray], threshold: float = MIN_JOINT_CHANGE,
) -> bool:
    """关节变化超过阈值才记（demo 同款 0.003 rad）。"""
    if js1 is None or js2 is None:
        return True
    return bool(np.any(np.abs(js1 - js2) > threshold))


# ── 键盘监听状态机 ──────────────────────────────────────────────────────────
class KeyboardState:
    """pynput 后台监听状态机 (Q-α: demo 风格 q/w/e/r/t + 1-5)。"""
    ENTER = False
    ESC = False
    SLOT_KEYS: Dict[str, int] = {
        "q": 0, "w": 1, "e": 2, "r": 3, "t": 4,  # 录制 slot 1-5
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,  # 播放 slot 1-5
    }

    @classmethod
    def reset(cls) -> None:
        cls.ENTER = False
        cls.ESC = False

    @classmethod
    def on_press(cls, key) -> bool | None:
        """pynput on_press 回调。Returns False 停止监听。"""
        from pynput import keyboard
        try:
            if key == keyboard.Key.esc:
                cls.ESC = True
                return False
            if not hasattr(key, "char") or key.char is None:
                return None
            ch = key.char.lower()
            if ch in cls.SLOT_KEYS:
                cls.ENTER = True
                return False  # 第一次按键触发后停止监听（简化版）
        except AttributeError:
            pass
        return None


# ── Dry Run（测试用）──────────────────────────────────────────────────────
def dry_run_record(slot: int, recordings_dir: Path) -> dict:
    """DRY RUN 模式: 不连硬件, 模拟 30Hz × 1秒 = 30 帧。"""
    print(f"[teach_record] DRY RUN slot={slot}")
    n_frames = 30
    start_master = np.zeros(7)
    start_slave = np.zeros(7)

    frames: List[dict] = []
    for i in range(n_frames):
        t = i / SAMPLE_HZ
        # 模拟主臂慢慢往右转（30°/秒）
        m_deg = np.array([30 * t, 20 * t, -15 * t, 10 * t, 5 * t, 0.0, 0.0])
        m_rad = np.deg2rad(m_deg)               # master_q 存 radians（统一单位，spec §10）
        s_rad = np.deg2rad(np.array([-30 * t, 20 * t, -15 * t, -10 * t, 5 * t, 0.0, 0.0]))  # joint1/4/6 取反
        frames.append({
            "t": round(t, 3),
            "master_q": [round(float(x), 6) for x in m_rad],
            "slave_q": [round(float(x), 6) for x in s_rad],
        })

    path = save_slot(slot, frames, start_master, start_slave, recordings_dir)
    duration_s = round(frames[-1]["t"] - frames[0]["t"], 3)
    print(f"[teach_record] DRY RUN saved: {path}")
    print(f"[teach_record] DRY RUN frames={len(frames)}, duration={duration_s:.2f}s")
    return {
        "ok": True,
        "slot": slot,
        "frame_count": len(frames),
        "duration_s": duration_s,
        "path": str(path),
    }


def dry_run_replay(slot: int, recordings_dir: Path) -> dict:
    """DRY RUN 模式: 不连硬件, 验证 JSON 读 + checksum + interpolation。"""
    print(f"[teach_replay] DRY RUN slot={slot}")
    try:
        data = load_slot(slot, recordings_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[teach_replay] 错误: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    frames = data["frames"]
    duration = data["duration_s"]
    print(f"[teach_replay] DRY RUN: {len(frames)} 帧, duration={duration:.2f}s")
    print(f"[teach_replay] DRY RUN: slave_q[0]={_fmt_q(frames[0]['slave_q'])}")
    print(f"[teach_replay] DRY RUN: slave_q[-1]={_fmt_q(frames[-1]['slave_q'])}")
    # 测试 interpolation
    if len(frames) >= 2:
        mid = (frames[0]["t"] + frames[-1]["t"]) / 2.0
        interp = interpolate_joint_states(frames[0], frames[-1], mid)
        print(f"[teach_replay] DRY RUN: interpolation @ t={mid:.2f}: {_fmt_q(interp)}")
    return {
        "ok": True,
        "slot": slot,
        "frame_count": len(frames),
        "duration_s": duration,
    }


def _fmt_q(arr) -> list:
    """格式化 joint_q 数组（去除 numpy float64 包装）。"""
    return [round(float(x), 3) for x in arr]


# ── 实机录制（T3.3，最小可用版本）────────────────────────────────────────
def record_real(
    master,
    slave,
    slot: int,
    recordings_dir: Path,
    keyboard_listener_factory: Optional[Callable] = None,
) -> dict:
    """实机主从臂录制（Q-α: pynput q 触发当前 slot 录/停 + Esc 退出）。

    Args:
        master: PiPERMateArm (已 open）
        slave: RebotArm (已 connect + mode_pos_vel + enable)
        slot: 1-5，q 按键触发当前 slot 录/停
        recordings_dir: slot json 目录
        keyboard_listener_factory: 测试注入（默认 None 用 pynput）

    Returns:
        {"ok": True, "slot": int, "frame_count": int, "duration_s": float}
        或 {"ok": False, "error": str}
    """
    if not isinstance(slot, int) or slot < 1 or slot > NUM_SLOTS:
        return {"ok": False, "error": f"slot must be int 1-{NUM_SLOTS}, got {slot!r}"}

    # 启 pynput 监听
    KeyboardState.reset()
    if keyboard_listener_factory is None:
        from pynput import keyboard as _kb
        listener = _kb.Listener(suppress=False, on_press=KeyboardState.on_press)
    else:
        listener = keyboard_listener_factory(KeyboardState)
    listener.start()

    try:
        return _teleop_record(master, slave, slot, recordings_dir)
    finally:
        listener.stop()


def _teleop_record(master, slave, slot: int, recordings_dir: Path) -> dict:
    """主从臂循环 + 30Hz 录制（直到 q 按键停止 或 Esc 退出）。"""
    frames: List[dict] = []
    recording = False
    record_start_time: Optional[float] = None
    last_recorded_q: Optional[np.ndarray] = None
    last_recorded_t: float = 0.0

    print(f"[teach_record] slot={slot}: 按 q 开始/停止录制, Esc 退出 (不保存)")

    try:
        while True:
            loop_start = time.monotonic()

            # 1. 读主臂 + 映射从臂
            master_q_deg = master.get_joint_states()
            master_q_rad = np.deg2rad(
                np.array([master_q_deg[n] for n in master.joint_names], dtype=np.float64)
            )
            slave_q_rad = master_to_slave(master_q_deg)

            # 2. 发从臂（POS_VEL 模式）
            slave.arm.send_pos_vel(slave_q_rad[:6], vlim=15.0)
            if slave.has_gripper:
                slave.gripper.send_pos_vel(slave_q_rad[6:])

            # 3. 记录
            if recording:
                t_rel = time.monotonic() - record_start_time
                if t_rel - last_recorded_t >= MIN_RECORD_INTERVAL:
                    filtered = filter_joint_states(slave_q_rad, last_recorded_q)
                    if joint_states_changed_enough(filtered, last_recorded_q):
                        frames.append({
                            "t": round(t_rel, 3),
                            "master_q": [round(float(x), 6) for x in master_q_rad],
                            "slave_q": [round(float(x), 6) for x in filtered],
                        })
                        last_recorded_q = filtered.copy()
                        last_recorded_t = t_rel

            # 4. 键盘事件
            if KeyboardState.ENTER:
                KeyboardState.ENTER = False
                if not recording:
                    recording = True
                    record_start_time = time.monotonic()
                    last_recorded_t = 0.0
                    last_recorded_q = None
                    print(f"[teach_record] 开始录制 slot {slot}")
                else:
                    print(f"[teach_record] 停止录制 slot {slot}, frames={len(frames)}")
                    break
            if KeyboardState.ESC:
                KeyboardState.ESC = False
                if frames:
                    print(f"[teach_record] Esc 退出 (丢弃 {len(frames)} 帧)")
                else:
                    print(f"[teach_record] Esc 退出 (无录制数据)")
                return {"ok": False, "error": "cancelled by Esc", "cancelled": True}

            # 5. sleep 到 30Hz
            elapsed = time.monotonic() - loop_start
            sleep_time = SAMPLE_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print(f"[teach_record] Ctrl+C, frames={len(frames)}")

    # 6. 保存
    if not frames:
        return {"ok": False, "error": "no frames captured"}
    if recording and frames[0]["t"] > 0:
        # End hold: 在最后一帧后加 hold_frame
        last_frame = dict(frames[-1])
        last_frame["t"] = round(last_frame["t"] + END_HOLD_TIME, 3)
        frames.append(last_frame)

    start_master = np.asarray(frames[0]["master_q"], dtype=np.float64)
    start_slave = np.asarray(frames[0]["slave_q"], dtype=np.float64)
    path = save_slot(slot, frames, start_master, start_slave, recordings_dir)
    duration_s = round(frames[-1]["t"] - frames[0]["t"], 3)
    print(f"[teach_record] 保存: {path}")
    return {
        "ok": True,
        "slot": slot,
        "frame_count": len(frames),
        "duration_s": duration_s,
        "path": str(path),
    }


# ── 实机复播（T3.4，最小可用版本）────────────────────────────────────────
def replay_real(slave, slot: int, recordings_dir: Path) -> dict:
    """实机复播（spec §5.7）: 读 slot → interpolation → send_pos_vel → safe_home。"""
    if not isinstance(slot, int) or slot < 1 or slot > NUM_SLOTS:
        return {"ok": False, "error": f"slot must be int 1-{NUM_SLOTS}, got {slot!r}"}

    try:
        data = load_slot(slot, recordings_dir)
    except (FileNotFoundError, ValueError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    frames = data["frames"]
    duration = data["duration_s"]
    n_arm = slave.arm.num_joints
    print(f"[teach_replay] slot={slot}: {len(frames)} 帧, duration={duration:.2f}s")

    q_waypoints = [
        np.asarray(f["slave_q"][:n_arm], dtype=np.float64) for f in frames
    ]
    play_duration = max(duration, 1.0)

    # 用 RebotArmEndPose 播放（wrc_demo/__init__.py 已注入 vendored SDK 路径）
    from reBotArm_control_py.controllers import RebotArmEndPose
    ctrl = RebotArmEndPose(slave, arm_control_mode="posvel", use_gravity_ff=False)
    try:
        ctrl.start()
        if not ctrl.move_joints_traj(q_waypoints, duration=play_duration):
            return {"ok": False, "error": "move_joints_traj failed to start"}
        if not ctrl.wait_motion_done(timeout=play_duration * 2.0):
            return {"ok": False, "error": "playback timeout"}
        ctrl.safe_home()  # 回电机机械零点
        return {
            "ok": True,
            "slot": slot,
            "frame_count": len(frames),
            "duration_s": duration,
            "replayed_to_home": True,
        }
    finally:
        ctrl.end()