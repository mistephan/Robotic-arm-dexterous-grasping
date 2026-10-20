# Spec: 示教录制-复播 (主从臂遥操方案)

> **项目根**: `/home/seeed/wrc_demo/`
> **硬件**: reBot Arm B601-RS（**从臂**, RobStride电机, CAN `can_arm @ 1Mbps`）+ PiPER Mate（**主臂**, FashionStar舵机, USB 串口 `/dev/ttyUSB0`）
> **环境**: `conda activate DM` → `/home/seeed/miniforge3/envs/DM/bin/python`
> **SDK**: vendored `reBotArm_control_py`（从臂，不可修改）+ `fashionstar-uart-sdk`（主臂，需 pip install）
> **创建**: 2026-08-14
> **版本**: 2（**架构变更**：从 schema_version=1 的"重力补偿漂浮"升级为"主从臂遥操"，**不兼容旧数据**）
> **状态**: Phase 1 (Specify)，待用户审阅

---

## 1. Objective

### 1.1 用户故事

```
as a wrc_demo operator at a booth
I want to say "record an action" / "replay the action" to the LLM
so that the PiPER Mate master arm teleoperates the reBot B601-RS slave arm,
records the trajectory to slot{1..5}.json, and replays it later.
```

### 1.2 验收标准

- LLM 调用 `skill_teach_record(slot=1)` → 主臂驱动从臂跟随，操作员按 `q` 录 slot 1，按 `w` 录 slot 2...
- LLM 调用 `skill_teach_replay(slot=1)` → 复播，0.6s 平滑过渡到 slot 起点，播放完 → safe_home 回电机机械零点
- 5 槽位机制，按键切换
- pynput 全局键盘监听（suppress=False）
- JSON schema_version=2（含 master_q + slave_q）
- SHA256 校验防 silent 数据损坏
- safe_home() 回电机机械零点（np.zeros(6)）

### 1.3 Non-goals（明确不做）

- ❌ 不修改 vendored SDK (`wrc_demo/sdk/reBotArm_control_py/`) — upstream 同步冲突
- ❌ 不做"录制期间叠加语音注释"
- ❌ 不做"槽位间平滑过渡"以外的复播模式（demo 的 loop blend 仅在 `PLAY_LOOP=True` 时启用）
- ❌ **不做主从臂关节校准**（demo 硬编码映射 joint1/4/6 取反 + 角度限制，用户已确认另想办法）
- ❌ 不复用 `apps/record.py`（相机 frame 录制，与关节录制无关）
- ❌ **不清理 `_MOTION_SKILLS` 集合里的 13 个 dead code**（保留向后兼容）

---

## 2. Tech Stack

| 组件 | 版本/路径 | 备注 |
|------|----------|------|
| Python | 3.10+ | wrc_demo pyproject.toml `requires-python = ">=3.10"` |
| **主臂**：PiPER Mate | `/dev/ttyUSB0` USB 串口 | FashionStar SDK (`fashionstar-uart-sdk`) |
| **从臂**：reBot Arm B601-RS | RobStride 电机 + SocketCAN (`can_arm @ 1Mbps`) | vendored SDK |
| 关节模型 | `home_q = [0, 0.1684, 0.6226, -0.4543, 0, 0]` | asset frame, 无 `joint_signs` 乘子 |
| 电机库 | `motorbridge >= 0.4.9` | `wrc_demo/pyproject.toml` `arm` extra |
| 主臂 SDK | `fashionstar-uart-sdk` | **需 pip install**（wrc_demo 当前未装） |
| 串口 | `pyserial >= 3.5` | ✅ 已装 |
| 动力学 | `pinocchio >= 3.0` | ✅ 已装（从臂 FK 用） |
| 键盘 | `pynput 1.8.x` | ✅ 已装到 conda DM |
| numpy | >= 1.26 | |

---

## 3. Commands

### 3.1 装包（用户跑）

```bash
conda activate DM
# 必须装到 conda DM env 内
uv pip install fashionstar-uart-sdk --python /home/seeed/miniforge3/envs/DM/bin/python
# 验证
/home/seeed/miniforge3/envs/DM/bin/python -c "from fashionstar_uart_sdk.uart_pocket_handler import PortHandler; print('fashionstar-uart-sdk OK')"
```

### 3.2 硬件前置

```bash
# CAN bus (从臂)
sudo ip link set can_arm down 2>/dev/null
sudo ip link set can_arm type can bitrate 1000000
sudo ip link set can_arm up

# 主臂 USB (PiPER Mate)
ls /dev/ttyUSB*  # 期望: /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB0  # 权限
```

### 3.3 启动 demo

```bash
conda activate DM
# 录制: 进入主从臂跟随模式, 按 q/w/e/r/t 录 slot1-5
/home/seeed/miniforge3/envs/DM/bin/python /home/seeed/wrc_demo/examples/teach_record.py --slot 1
# 复播: 选槽位播放
/home/seeed/miniforge3/envs/DM/bin/python /home/seeed/wrc_demo/examples/teach_replay.py --slot 1

# Dry-run (不连硬件, 模拟数据流)
/home/seeed/miniforge3/envs/DM/bin/python /home/seeed/wrc_demo/examples/teach_record.py --slot 1 --dry-run
/home/seeed/miniforge3/envs/DM/bin/python /home/seeed/wrc_demo/examples/teach_replay.py --slot 1 --dry-run
```

### 3.4 完整测试

```bash
conda activate DM
cd /home/seeed/wrc_demo
/home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/test_teach_logic.py tests/test_teach_skills.py -v
/home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/ -q  # 完整（已有 28 个 pre-existing failures，不应增加）
```

---

## 4. Project Structure

### 4.1 阶段 1 新增 / 重写

```
wrc_demo/
├── examples/                                       ← 已有
│   ├── teach_record.py                             ← **重写**：主从臂遥操 demo
│   └── teach_replay.py                             ← **重写**：复播 demo
└── data/recordings/                                ← 已有（保留）
    ├── .gitkeep
    ├── slot1.json                                  ← **删除**（schema_version=1 不兼容）
    ├── slot2.json                                  ← **删除**
    ├── slot3.json                                  ← **删除**
    └── slot4.json                                  ← **删除**
```

### 4.2 阶段 2 新增 / 修改

```
wrc_demo/
├── docs/specs/teach_record_replay.md               ← **重写** (本文件, v2)
├── src/wrc_demo/skills/
│   ├── master_arm.py                               ← **新增**：PiPER Mate 封装 + 主→从映射
│   ├── teach.py                                    ← **重写**：业务层（主从臂循环）
│   └── runtime.py                                  ← **修改**：skill_teach_* + TOOL_SPECS description
└── tests/
    ├── test_teach_logic.py                         ← **重写**：业务层单测
    └── test_teach_skills.py                        ← **重写**：skill 集成测试
```

---

## 5. Code Style

### 5.1 命名约定（沿用 wrc_demo + CLAUDE.md）

| 类型 | 约定 | 例子 |
|------|------|------|
| 函数 | `snake_case` | `master_to_slave`, `floating_callback`（旧，已删除）|
| 类 | `PascalCase` | `PiPERMateArm`, `SlaveArm`, `TeleopLooper` |
| 常量 | `UPPER_SNAKE_CASE` | `NUM_SLOTS=5`, `RECORD_HZ=30`, `_MOTION_SKILLS` |
| skill 函数 | `skill_<name>` | `skill_teach_record`, `skill_teach_replay` |
| 帧 | `{t, master_q, slave_q, ...}` | 主从臂各 6 关节+gripper |

### 5.2 主→从关节映射（硬编码，参考 demo `servoangle2jointstate`）

```python
def master_to_slave(master_q_deg: dict) -> np.ndarray:
    """PiPER Mate → reBot 关节映射。

    硬编码（用户确认不做校准）:
    - joint1, joint4, joint6 方向取反
    - 应用 joint 角度限制（demo `JOINT_ANGLE_LIMITS`）
    - 关节顺序: joint1..joint6 + gripper
    """
    slave_q_deg = {}
    for i in range(1, 7):
        name = f"joint{i}"
        angle = master_q_deg[name]
        if i in (1, 4, 6):  # joint1, joint4, joint6 取反
            angle = -angle
        lim = JOINT_ANGLE_LIMITS[name]
        angle = max(lim[0], min(angle, lim[1]))
        slave_q_deg[name] = np.deg2rad(angle)

    # gripper 单独处理 (单位: rad, GRIPPER_RATIO=6, angle > -14 视为 0)
    grip = -master_q_deg["gripper"] * GRIPPER_RATIO
    if grip > -14:
        grip = 0
    slave_q_deg["gripper"] = np.deg2rad(max(JOINT_ANGLE_LIMITS["gripper"][0],
                                          min(grip, JOINT_ANGLE_LIMITS["gripper"][1])))
    return np.array([slave_q_deg[f"joint{i}"] for i in range(1, 7)] +
                    [slave_q_deg["gripper"]])
```

### 5.3 主循环（参考 demo `TeleopLooper.run()`）

```python
def teleop_loop(master, slave, recordings_dir, slot: int, mode: str):
    """30Hz 主从臂循环。

    mode ∈ {"follow", "record", "transition", "playback"}
    """
    UPDATE_HZ = 30
    interval = 1.0 / UPDATE_HZ
    while running:
        loop_start = time.monotonic()

        if mode in ("follow", "record"):
            master_q = master.get_joint_states()       # dict
            slave_q = master_to_slave(master_q)        # np.ndarray 7 floats
            slave.arm.send_pos_vel(slave_q[:6], vlim=15)  # POS_VEL 模式
            slave.gripper.send_pos_vel(slave_q[6:])

            if mode == "record":
                update_recording(master_q, slave_q, recordings_dir, slot)

        elif mode == "playback":
            interpolated = interpolate_playback(recordings_dir, slot, t_now)
            slave.arm.send_pos_vel(interpolated[:6], vlim=15)
            slave.gripper.send_pos_vel(interpolated[6:])

        elapsed = time.monotonic() - loop_start
        time.sleep(max(0, interval - elapsed))
```

### 5.4 键盘监听（参考 demo `TeleopLooper.on_press`）

```python
KEY_MAP = {
    "q": ("record_toggle", 0),  # slot 1
    "w": ("record_toggle", 1),  # slot 2
    "e": ("record_toggle", 2),  # slot 3
    "r": ("record_toggle", 3),  # slot 4
    "t": ("record_toggle", 4),  # slot 5
    "1": ("playback", 0),        # slot 1
    "2": ("playback", 1),        # slot 2
    "3": ("playback", 2),        # slot 3
    "4": ("playback", 3),        # slot 4
    "5": ("playback", 4),        # slot 5
    "s": ("stop_all", None),
    "c": ("clear_last", None),
    "a": ("clear_all", None),
    "f": ("follow_mode", None),
}
# Esc: 退出 (Key.esc)
```

**注意**：之前规划用 Enter 保存，**改为** demo 的 `q/w/e/r/t` 触发（toggle 录制/停止）+ `1/2/3/4/5` 播放。

### 5.5 录制过滤 + 阈值（参考 demo）

```python
RECORD_FILTER_ALPHA = 0.35       # 指数移动平均平滑
MIN_RECORD_INTERVAL = 0.01       # 秒，最短帧间隔
MIN_JOINT_CHANGE = 0.003         # rad，关节变化阈值（主臂或从臂）
END_HOLD_TIME = 0.15             # 录制结束时 hold 最后一帧
LOOP_BLEND_TIME = 0.30           # 循环播放时混合时间
TRANSITION_TIME = 0.6             # 复播前平滑过渡到 slot 起点
PLAY_LOOP = True                 # 默认循环播放
```

### 5.6 关节映射常量

```python
JOINT_ANGLE_LIMITS = {
    "joint1": (-150.0, 150.0),
    "joint2": (-180.0, 0.0),
    "joint3": (-170.0, 0.0),
    "joint4": (-80.0, 80.0),
    "joint5": (-70.0, 70.0),
    "joint6": (-120.0, 120.0),
    "gripper": (-320.0, 0.0),
}
GRIPPER_RATIO = 6  # reBot 夹爪放缩比
```

### 5.7 Skill 函数约定（必须返回 dict）

```python
def skill_teach_record(self, slot: int) -> dict:
    """进入主从臂跟随模式, 按 q/w/e/r/t 录 slot1-5。
    
    Long-running: 阻塞直到用户按 Esc 或 LLM 调 task_done。
    """
    if not isinstance(slot, int) or slot < 1 or slot > 5:
        return {"ok": False, "error": f"slot must be int 1-5, got {slot!r}"}

    # ... master + slave setup + 主从臂循环 + pynput ...
    return {
        "ok": True,
        "slot": slot,
        "frame_count": ...,
        "duration_s": ...,
    }


def skill_teach_replay(self, slot: int) -> dict:
    """复播 slot{N}.json, safe_home 回电机机械零点。"""
    return {
        "ok": True,
        "slot": slot,
        "replayed_to_home": True,
    }
```

---

## 6. Testing Strategy

### 6.1 测试层级

| 层级 | 范围 | 工具 |
|------|------|------|
| **Unit** | `teach.py` 纯函数（save/load/checksum/filter_recording/interpolate_joint_states/master_to_slave） | pytest |
| **Unit** | `master_arm.py::master_to_slave` 映射（硬编码，取反 + 限幅） | pytest |
| **Integration** | `examples/teach_record.py --dry-run`：mock master + slave，验证数据流 | pytest + monkeypatch |
| **Integration** | `examples/teach_replay.py --dry-run`：mock slave，验证播放 + safe_home | pytest + monkeypatch |
| **Skill** | `skill_teach_record` / `skill_teach_replay` 返回 dict 格式 | pytest + mock SafeArm |
| **Hardware** | 实机：主从臂跟随 + q/w/e/r/t 录制 + 1-5 播放 + 回零 | 手动 |

### 6.2 覆盖率目标

- `teach.py` 业务层：≥ 80% 行
- `master_arm.py::master_to_slave`：100% 行（硬编码简单）
- skill 入口（`skill_teach_*`）：100% 行

### 6.3 关键测试用例

1. `test_save_load_slot_roundtrip`: 保存后读取数据完全一致
2. `test_checksum_validation`: 数据被破坏 → checksum 不匹配 → `load_slot()` 抛 `ValueError`
3. `test_schema_version_2`: JSON 含 `schema_version: 2`
4. `test_frames_have_master_q_and_slave_q`: 每帧含 `master_q` (7 floats) + `slave_q` (7 floats)
5. `test_master_to_slave_joint_signs`: joint1, joint4, joint6 取反
6. `test_master_to_slave_angle_limits`: 角度超限被 clip
7. `test_master_to_slave_gripper_ratio`: gripper × GRIPPER_RATIO=6
8. `test_record_filter_alpha`: 指数移动平均 (alpha=0.35)
9. `test_record_joint_change_threshold`: 关节变化 < 0.003 rad 不记帧
10. `test_interpolate_smoothstep`: 复播插值用 smoothstep (3α²-2α³)
11. `test_transition_blend`: 0.6s 平滑过渡
12. `test_end_hold_and_loop_blend`: END_HOLD_TIME + LOOP_BLEND_TIME
13. `test_skill_teach_record_returns_dict_format`: 返回 dict
14. `test_skill_teach_record_rejects_invalid_slot`: slot=0/6/"abc" 都拒
15. `test_skill_teach_replay_returns_replayed_to_home`: 返回 `replayed_to_home=True`
16. `test_motion_skills_contains_teach_*`: `_MOTION_SKILLS` 集合含 teach_record/teach_replay
17. `test_tool_specs_total_6`: `TOOL_SPECS` 含 6 条 (4 原生 + 2 teach)
18. `test_keyboard_suppress_false`: `pynput.keyboard.Listener(suppress=False)`

---

## 7. Boundaries

### Always

- ✅ 时间戳约定: in-process `time.monotonic()`, 持久化 `time.time()` (CLAUDE.md §Contracts)
- ✅ 关节单位: radians, 关节 frame: asset frame 无 `joint_signs` 乘子
- ✅ 主臂输入: 度 (degrees) → 转换为 rad
- ✅ slot 文件路径: `wrc_demo/data/recordings/slot{N}.json`
- ✅ `pynput.keyboard.Listener(suppress=False)`
- ✅ SHA256 校验（防 silent 数据损坏）
- ✅ 命名: skill 函数必须 `skill_<name>`
- ✅ TOOL_SPECS 每条 schema 必有对应 `skill_<name>` 方法
- ✅ 从臂用 POS_VEL 模式（`vlim=15`）
- ✅ 录制过滤: `RECORD_FILTER_ALPHA=0.35`, `MIN_JOINT_CHANGE=0.003` rad
- ✅ End hold 0.15s, Loop blend 0.30s, Transition 0.6s

### Ask First

- ⚠️ 新增 conda DM 包（`fashionstar-uart-sdk`）
- ⚠️ 修改 `wrc_demo/sdk/reBotArm_control_py/` 任何文件（vendored snapshot 同步覆盖）
- ⚠️ 修改 `wrc_demo/.gitignore`
- ⚠️ 清理 `_MOTION_SKILLS` 集合里的 13 个 dead code
- ⚠️ 修改 SafeArm / SafetyHarness / Kinematics 行为
- ⚠️ 修改 `home_q` 或 `RebotArmEndPose.safe_home()` 默认参数
- ⚠️ 修改 `data/recordings/` slot 命名约定
- ⚠️ 添加/修改 master_to_slave 映射规则

### Never

- ❌ 修改 vendored SDK（`wrc_demo/sdk/reBotArm_control_py/`）任何文件
- ❌ 复用 `wrc_demo/apps/record.py`（相机 frame 录制，与关节录制无关）
- ❌ 在 skill 命名上违反 `skill_<name>` 约定（CLAUDE.md §Architecture 强制）
- ❌ 让 skill 返回非 dict 类型
- ❌ 在 skill 抛普通 `Exception`（CLAUDE.md: 必须用 `SkillError` / `SafetyViolation`）
- ❌ **不要主从臂校准**（用户明确要求不做；用 demo 硬编码映射）
- ❌ 让 pynput `suppress=True`（拦截 LLM 输入）
- ❌ 修改 `_MOTION_SKILLS` 里 13 个 dead code（CLAUDE.md "never remove failing tests"）

---

## 8. Success Criteria

### 8.1 阶段 1 DoD

- [ ] **`fashionstar-uart-sdk` 装到 conda DM**（用户跑 `pip install`）
- [ ] 旧的 `data/recordings/slot{1..4}.json`（schema_version=1）**删除**
- [ ] `examples/teach_record.py` 重写完成，支持 `--dry-run` + 主从臂循环 + pynput
- [ ] `examples/teach_replay.py` 重写完成，支持 `--dry-run` + 复播 + safe_home
- [ ] dry-run 模式跑通，不连硬件

### 8.2 阶段 2 DoD

- [ ] `src/wrc_demo/skills/master_arm.py` 含 `PiPERMateArm` 类 + `master_to_slave` 函数
- [ ] `src/wrc_demo/skills/teach.py` 重写：业务层含 save/load/checksum/filter/interpolate/transition/end_hold/loop_blend/master_to_slave
- [ ] `skill_teach_record(slot)` 和 `skill_teach_replay(slot)` 在 `runtime.py` 实现
- [ ] `_MOTION_SKILLS` 含 `teach_record`、`teach_replay`（不变）
- [ ] `TOOL_SPECS` 6 条（4 原生 + 2 teach），teach_* schema 更新说明主从臂
- [ ] `tests/test_teach_logic.py` 重写，≥ 80% 行覆盖
- [ ] `tests/test_teach_skills.py` 重写，100% skill 入口覆盖

### 8.3 项目级 DoD

- [ ] 实机：主从臂跟随可工作（操作员推 PiPER Mate → reBot 跟）
- [ ] 实机：按 `q` 录 slot 1，按 `w` 录 slot 2... → JSON 生成
- [ ] 实机：按 `1` 播 slot 1（带 0.6s 过渡）→ 0.5s 播放完 → safe_home 回零
- [ ] 现有 wrc_demo 测试不破（pre-existing 28 failures 不应增加）
- [ ] 阶段 1 + 2 完成，spec §8 所有验收点通过

---

## 9. Open Questions

| Q | 选项 | 我的倾向 | 何时答 |
|---|------|---------|--------|
| Q-α | 键盘用 `q/w/e/r/t` 录制 + `1-5` 播放（demo）？已确认 ✅ | (A) demo 风格 | — |
| Q-β | 采样率 30Hz（demo）？已确认 ✅ | (A) 30Hz | — |
| Q-γ | 0.6s 平滑过渡到 slot 起点（demo）？已确认 ✅ | (A) 保留 | — |
| Q-δ | safe_home 回电机机械零点（`np.zeros(6)`）？已确认 ✅ | (A) 保留 | — |
| Q-ε | 旧的 `slot{1..4}.json`（schema_version=1）删除还是保留？ | (A) **删除**（schema 升级不兼容） | 阶段 1 开始前 |
| Q-ζ | demo 文件`--dry-run`模式保留？ | (A) **保留**（CI 集成测试需要） | — |
| Q-η | TOOL_SPECS 中 teach_* 的 `description` 是否更新成"主从臂遥操"措辞？ | (A) **是**（LLM 看 description 决策） | 阶段 2 实施 |
| Q-θ | 复播时 POS_VEL 的 vlim 用 demo 的 15 还是 SDK yaml 默认？ | (A) **15**（demo 同款，柔和） | — |
| Q-ι | `MASTER_ANGLE_LIMITS` 是否采用 demo 的硬编码值？ | (A) **是**（与 PiPER Mate SDK 一致） | — |
| Q-κ | 是否需要 LLM reflex 规则映射 `record slot N` → `skill_teach_record(N)`？ | (B) **否**（用户说做 skill，不走 reflex） | — |

---

## 10. Data Format (Recording JSON Schema, v2)

```json
{
  "schema_version": 2,
  "slot": 1,
  "created_at": "2026-08-14T15:32:11Z",
  "frame_count": 1280,
  "duration_s": 42.6,
  "sample_rate_hz": 30,
  "start_pose": {
    "master_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "slave_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "checksum_sha256": "abc123...",
  "frames": [
    {
      "t": 0.0,
      "master_q": [0.1, 0.05, -0.05, 0.02, 0.01, 0.0, 0.0],
      "slave_q": [-0.1, 0.05, -0.05, -0.02, 0.01, 0.0, 0.0],
      "ee_pose": [[1,0,0,0.3],[0,1,0,0],[0,0,1,0.2],[0,0,0,1]],
      "ee_position": [0.3, 0.0, 0.2],
      "ee_rpy_deg": [0.0, 0.0, 0.0]
    }
  ]
}
```

**关键字段说明（v2 vs v1）**：

| 字段 | v1（重力补偿） | v2（主从臂） |
|------|----------------|--------------|
| `schema_version` | 1 | **2**（不兼容）|
| `frames[i].joints` | `[j1..j6 + gripper]` (rad) | ❌ **删除**（不再单存从臂）|
| `frames[i].master_q` | ❌ 无 | ✅ **新增**（7 floats, 主臂角度 rad）|
| `frames[i].slave_q` | ❌ 无 | ✅ **新增**（7 floats, 从臂角度 rad）|
| `frames[i].ee_pose` | ✅ | ✅（FK 解算从臂，可选）|
| `frames[i].ee_position` | ✅ | ✅ |
| `frames[i].ee_rpy_deg` | ✅ | ✅ |
| `start_pose` | `{joints, ee_pose}` | ✅ 改为 `{master_q, slave_q}` |
| `sample_rate_hz` | 50 | ✅ 改为 **30** |

**关键约定**：
- `master_q` / `slave_q` 单位：radians（从臂），主臂原始输入是 degrees（内部转换）
- 主→从映射硬编码（demo `servoangle2jointstate`）：joint1/4/6 取反 + 角度限幅 + gripper × GRIPPER_RATIO
- `ee_pose` 是从臂末端的 SE(3)（FK 解算），用于可视化或碰撞检查
- `checksum_sha256` = SHA256(序列化后的 frames 字节)

---

## 11. References

- `wrc_demo/sdk/README.md` §"Vendored reBotArm SDK"
- `wrc_demo/sdk/reBotArm_control_py/controllers/rebotarm_endpose_controller.py:335` (`move_joints_traj`)
- `wrc_demo/sdk/reBotArm_control_py/controllers/rebotarm_endpose_controller.py:183` (`safe_home`)
- `wrc_demo/sdk/reBotArm_control_py/actuator/rebotarm.py:705` (`start_control_loop`)
- `wrc_demo/sdk/reBotArm_control_py/actuator/rebotarm.py:331` (`send_mit`)
- `wrc_demo/sdk/reBotArm_control_py/actuator/rebotarm.py:364` (`send_pos_vel`)
- `wrc_demo/sdk/reBotArm_control_py/dynamics/inverse_dynamics.py:89` (`compute_generalized_gravity`)
- `wrc_demo/sdk/reBotArm_control_py/config/rebotarm_rs.yaml`
- `wrc_demo/CLAUDE.md` §"Motion safety path"
- `wrc_demo/CLAUDE.md` §"Recent fixes (2026-07-24)"
- `wrc_demo/CLAUDE.md` §"Gotchas"（LazyArm 物化）
- 参考项目: <https://github.com/Welt-liu/rebot-b601-102-record-demo>
  - `Python_SDK/rebot_so102_record.py`（主从臂 loop 模板）
  - `Python_SDK/pipermate_sdk.py`（PiPER Mate SDK 封装）

---

## 12. 返工清单（之前 schema_version=1 的工作）

按 `incremental-implementation` "**Rule 2: Keep It Compilable**"——必须明确**哪些之前的工作要返工**：

| 文件 | 之前状态 | 现在处理 |
|------|---------|---------|
| `examples/teach_record.py` | 写了 195 行（重力补偿版） | **重写**（主从臂版，约 250 行） |
| `examples/teach_replay.py` | 写了 110 行 | **重写**（调用 RebotArmEndPose 同前，但数据格式 v2）|
| `src/wrc_demo/skills/teach.py` | 写了 360 行（含 floating_callback, home_to_zero） | **重写**（删除漂浮逻辑，加主从臂循环 + master_to_slave） |
| `src/wrc_demo/skills/runtime.py` | 已加 `skill_teach_record` / `skill_teach_replay` 方法 + `_MOTION_SKILLS` + `TOOL_SPECS` | **修改 TOOL_SPECS description**（说明主从臂），方法实现**重写** |
| `tests/test_teach_logic.py` | 14 个测试（覆盖 save_slot / checksum / floating_callback） | **重写**（删除 floating_callback 测试，加 master_to_slave 测试） |
| `tests/test_teach_skills.py` | 17 个测试 | **重写**（skill 内部用了新流程，但测试结构类似） |
| `data/recordings/slot{1..4}.json` | 已生成（schema_version=1） | **删除**（schema 升级不兼容） |
| `data/recordings/slot5.json`（如果存在） | 同上 | 同上 |

**总返工量**：约 7 个文件重写/修改 + 4 个数据文件删除。

**保留不变**：
- `wrc_demo/CLAUDE.md` 中 v2 段（"Recent additions (2026-08-14) — Teach-Record/Replay skills"）需更新措辞
- `tasks/plan.md` + `tasks/todo.md` 之前的版本**完全失效**，必须重写
- 桌面 `示教录制-复播开发规划.md` 之前的版本**完全失效**（用户已删除）

---

## 13. Phase 1 Verification (spec-driven-development skill)

- [x] **The spec covers all six core areas**: Objective ✓ / Tech Stack ✓ / Commands ✓ / Project Structure ✓ / Code Style ✓ / Testing Strategy ✓
- [x] **Boundaries (Always/Ask First/Never) are defined**: 12 Always / 9 Ask First / 9 Never
- [x] **Success criteria are specific and testable**: 阶段 1 (5 项) + 阶段 2 (8 项) + 项目级 (5 项)
- [x] **The spec is saved to a file in the repository**: `wrc_demo/docs/specs/teach_record_replay.md`
- [ ] **The human has reviewed and approved the spec** ← **等你审阅**

**Phase 1 完成，等待人工审阅。重大架构变更后进入 Phase 2 (Plan) → Phase 3 (Tasks) → Phase 4 (Implement 重写)。**