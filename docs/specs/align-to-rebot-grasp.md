# Spec: Align wrc_demo grasp/place control to rebot_grasp set.py / main.py

**Status:** Approved by user 2026-08-10 ("其他没有问题")
**Date:** 2026-08-10
**Parent intent:** [`docs/intent/align-to-rebot-grasp.md`](../intent/align-to-rebot-grasp.md)

---

## Assumptions I'm making

1. **Refactor, not wrapper.** wrc_demo 的 skill_grasp_object / skill_pick_and_place 内部逻辑被改写为调用 rebot_grasp 的 API，而不是起 subprocess 跑 set.py/main.py 或 import 整个 `G` 流程。
2. **rebot_grasp 加入 PYTHONPATH.** 当前 demo 命令 `PYTHONPATH=/home/seeed/wrc_demo/src` 需要扩展为同时包含 `/home/seeed/rebot_grasp`。
3. **真机 + MockArm 都要支持.** rebot_rs 路径用真 RebotArmEndPose + GraspDriver；test 路径 MockArm 必须扩展出 `move_to_traj(x,y,z,roll,pitch,yaw,duration)` + `grasp()` / `open_gripper(timeout=)` 等新接口。
4. **SafetyHarness 边界保留.** 每次 `move_to_traj` 调用前过 `harness.approve(q_next)`；gripper `grasp()` / `open_gripper()` 过 `harness.heartbeat()`；harness 自身的 7 项检查不动。
5. **box z = 0.08 hardcoded.** 跟 set.py `box_center_base = ... z=0.08` 完全一致；不引入 `release_clearance_m` 参数，不做 z_cap。
6. **depth_quantile = 0.50.** 来自 `rebot_grasp/config/default.yaml::grasp_pipeline.grasp.depth_quantile`；set.py 脚本里的 `0.75` 是 `.get()` fallback，因为 config 实际总是有值所以实际用 0.50。
7. **pregrasp_offset_m = 0.08 m.** 来自 `rebot_grasp/config/default.yaml::grasp_pipeline.grasp.pregrasp_offset_m`，从 `_make_grasp_base_transform` 计算 pre6d。
7a. **insertion_depth_m = 0.015 m.** 来自 rebot_grasp config；TCP 在物体表面往 approach 方向推 1.5cm（区别于 wrc_demo v5 的 0.0）。
8. **home 归位 = wrc_demo `home_q`.** 不是 set.py 的 ready_pose；FK 出 TCP ≈ (0.25, 0.10, 0.35)，腕部姿态跟 ready_pose 不同。
9. **测试不强制保留 215 passed.** skill 端到端测试（`tests/test_streaming_fastpath.py::test_pick_and_place_red_object_end_to_end` 等）如果断言 wrc_demo v5 内部行为，需要改为 mock rebot_grasp 的接口并验证调用参数。
10. **不引入新依赖.** rebot_grasp 代码已在 `/home/seeed/rebot_grasp/`，motorbridge / pinocchio / pyorbbecsdk 已经在 conda `DM` env 里。
11. **WorldWatcher / belief fusion / reflex parser / LLM orchestrator 不动.** 这一层跟机械臂控制正交。
12. **只动 `pick up X` 和 `put X into Y` 两条命令.** 其它 skill（`move_home`, `place_at`, `drop`）如果共享被改的底层函数，连带改；不新增 skill。

**Correct me now or I'll proceed with these.**

---

## Objective

让 wrc_demo 真机 demo 在跑 "pick up the banana" 和 "put the banana into box" 时，机械臂产生的关节轨迹 / TCP 路径 / 夹爪状态序列跟 `rebot_grasp/scripts/main.py` / `set.py` 在同一物体上产生的对齐到误差 ≤ set.py 当天 (2026-08-07) 录像 ± 5 mm / ± 0.05 rad 级别（关节空间允许有 IK basin 选择差异）。

### Acceptance criteria

- **AC1**: `pick up the banana` 命令成功完成，arm 状态变化序列为：
  ```
  ready → open_gripper → move_to_traj(pre6d, dur=2.0) → move_to_traj(grasp6d, dur=1.5)
       → grasp_driver.grasp() (单段 close) → move_to_traj(home_q, dur=3.0)
  ```
- **AC2**: `put the banana into box` 命令成功完成，arm 状态变化序列为：
  ```
  (同帧 detect banana + box)
  → ready → open_gripper → move_to_traj(banana_pre6d, dur=2.0) → move_to_traj(banana_grasp6d, dur=1.5)
       → grasp_driver.grasp() → move_to_traj(home_q, dur=3.0)
       → move_to_traj(box_xy, z=0.08, rpy=(0,1.2,0), dur=2.0)
       → grasp_driver.open_gripper(timeout=0.5) [主线程]
       + 0.8s 后异步 close 回 idle [后台线程]
       → move_to_traj(home_q, dur=3.0)
       → join 异步 close 线程
  ```
- **AC3**: 视觉抓取链路调用 rebot_grasp 函数，可被 mock 拦截：
  - `transform_grasp_pose_to_base(position, tcp_rotation, T_cam2base, pregrasp_offset_m=0.08)`
  - `estimate_grasps(results, depth_mm, K, depth_quantile=0.75)`
  - `compose_cam_to_base_transform(tcp_pose, T_hand_eye, mode, cfg)`
- **AC4**: 测试 `tests/test_align_to_rebot_grasp.py`（新增）覆盖：
  - skill_grasp_object 调用 move_to_traj 两次（dur=2.0, dur=1.5）和 grasp_driver.grasp() 各一次
  - skill_pick_and_place 一次 detect 拿到 banana + box 两个目标
  - place 阶段 move_to_traj 使用 rpy=(0,1.2,0)
  - place 完成后 arm 回 home_q
- **AC5**: 真机 demo 命令与 set.py 当天录像的 xy 错位 < 1cm（继承自 v5 的 T_cam2base 对齐），z 错位 < 1cm，gripper 闭合后 banana 不掉。

---

## Tech Stack

| Layer | 现状 | 变化 |
|---|---|---|
| 机械臂控制 | `motorbridge 0.4.8` + `RebotArmEndPose` + `SafeArm` wrapper | wrc_demo 原生 `WrcGripper`（port rebot_grasp `GraspDriver`，见 ADR-0005）+ 直接调用 `RebotArmEndPose.move_to_traj` |
| IK | pinocchio SDK DLS IK | 不变 |
| 视觉抓取 | wrc_demo 自己写的 `plan_grasp_from_mask` + `Extrinsics.from_config` | **保留**（CLAUDE.md 注 `plan_grasp_from_mask` 已 mirror rebot_grasp 算法；替换风险高） |
| 标定 | `data/calibration/hand_eye_top.json` | 不变（T_cam2base 已跟 set.py NPZ 对齐） |
| LLM | minimax (supports_vision=False) | 不变 |
| Python | 3.11, conda env `DM` | 不变 |
| 测试 | pytest + MockArm + FakeOrbbecSDK | 扩展 MockArm 加 `move_to_traj` / `grasp` / `open_gripper` 接口；新增 `wrc_arm_mocks` fixture |

---

## Commands

### Demo (真机)
```bash
sudo ip link set can_arm down 2>/dev/null
sudo ip link set can_arm type can bitrate 1000000
sudo ip link set can_arm up
sudo env HOME=/home/seeed YOLO_OFFLINE=True \
  PYTHONPATH=/home/seeed/wrc_demo/src \
  WRC_RUN_DIR=/home/seeed/wrc_demo/runs/live_$(date +%Y%m%d_%H%M%S) \
  WRC_STREAM=1 WRC_VIEW=1 \
  /home/seeed/miniforge3/envs/DM/bin/python -u -m wrc_demo.apps.demo \
  --cameras orbbec_overhead --arm rebot_rs --llm minimax --interactive
```

No external PYTHONPATH edits are required: ``wrc_demo/__init__.py``
injects ``wrc_demo/sdk/`` into ``sys.path`` so the vendored
``reBotArm_control_py`` SDK is importable, and ``WrcGripper`` (the
wrc_demo-native port of rebot_grasp's ``GraspDriver``) lives at
``wrc_demo.control.gripper`` — no rebot_grasp runtime dependency
(ADR-0005).

### 测试
```bash
cd /home/seeed/wrc_demo && \
  PYTHONPATH=/home/seeed/wrc_demo/src:/home/seeed/rebot_grasp \
  /home/seeed/miniforge3/envs/DM/bin/python -m pytest tests/ -q --no-header
```

### 单跑新对齐测试
```bash
cd /home/seeed/wrc_demo && \
  PYTHONPATH=/home/seeed/wrc_demo/src:/home/seeed/rebot_grasp \
  /home/seeed/miniforge3/envs/DM/bin/python -m pytest \
    tests/test_align_to_rebot_grasp.py -v --no-header
```

---

## Project Structure

```
wrc_demo/
├── src/wrc_demo/
│   ├── apps/
│   │   └── demo.py                          # 不变（入口）
│   ├── skills/
│   │   ├── runtime.py                       # 【改】skill_grasp_object / skill_pick_and_place 重写
│   │   ├── rebot_grasp_bridge.py            # 【新】wrap RebotArmEndPose.move_to_traj + WrcGripper
│   │   └── __init__.py                      # 不变
│   ├── grasping/                            # (CLAUDE.md: 视觉抓取模块在这,不是 skills/)
│   │   ├── camera_grasp.py                  # 【保留】plan_grasp_from_mask (已 mirror rebot_grasp)
│   │   ├── obb_grasp.py                     # 【保留】fallback, 不动
│   │   ├── force.py                         # 【保留】select_profile 仍用,v5 不变
│   │   └── selector.py                      # 【保留】IK 选择
│   ├── perception/
│   │   ├── grounding.py                     # 【保留】_localize 仍走 wrc_demo pipeline
│   │   ├── orbbec_camera.py                 # 不变
│   │   └── detector.py                      # 不变
│   ├── control/
│   │   ├── __init__.py                      # 【改】make_arm 为 rebot_rs 实例化 WrcGripper + bridge
│   │   ├── gripper.py                       # 【新】WrcGripper (wrc_demo 原生 port of GraspDriver)
│   │   ├── isaac_arm.py                     # 不变
│   │   └── mock_arm.py                      # 不变
│   ├── safety/
│   │   └── harness.py                       # 不变（边界）
│   └── ...
├── tests/
│   ├── conftest.py                          # 【改】wrc_arm_mocks fixture 取代 rebot_grasp_mock
│   ├── test_align_to_rebot_grasp.py         # 【新】AC4 覆盖
│   ├── test_bridge.py / test_bridge_safety.py # 【新】bridge 单元测试
│   ├── test_make_arm_rebot_rs.py            # 【新】rebot_rs 接线测试
│   ├── test_streaming_fastpath.py           # 【保留】原有测试
│   └── ...
└── docs/
    ├── intent/align-to-rebot-grasp.md       # 已存
    └── specs/align-to-rebot-grasp.md        # 本文档
```

**注意 (ADR-0005):**
- wrc_demo 不再依赖 rebot_grasp runtime。`WrcGripper` 是 wrc_demo 原生 port，代码位置 `src/wrc_demo/control/gripper.py`
- `reBotArm_control_py` 是 vendored SDK（在 `wrc_demo/sdk/`），`wrc_demo/__init__.py` 已经把 `wrc_demo/sdk/` 加到 `sys.path`
- 不需要 PYTHONPATH rebot_grasp hack（之前 `tests/conftest.py` 里的 namespace package hack 已删）
- 删除了 `tests/test_rebot_grasp_importable.py` 和 `tests/test_rebot_grasp_mock_fixture.py`（已无意义）

---

## Code Style

保持 wrc_demo 现状（type hints + dataclass + loguru），新增的 `rebot_grasp_bridge.py` 用同样的风格。调用 rebot_grasp 的函数时**显式 import 模块路径**（避免命名空间冲突）：

```python
# src/wrc_demo/skills/rebot_grasp_bridge.py
"""Bridge to rebot_grasp motion + gripper APIs.

Maps wrc_demo's higher-level skill calls (skill_grasp_object, skill_pick_and_place)
to rebot_grasp's RebotArmEndPose.move_to_traj + GraspDriver primitives.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from reBotArm_control_py.controllers import RebotArmEndPose
from rebot_grasp.drivers.robot.grasp_driver import GraspDriver
from rebot_grasp.utils.camera_utils import compose_cam_to_base_transform
from rebot_grasp.utils.ordinary_grasp import GraspPose, estimate_grasps, select_best_grasp
from rebot_grasp.utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    transform_grasp_pose_to_base,
)

if TYPE_CHECKING:
    from wrc_demo.control.rebot_rs import RebotRsArm


class RebotGraspBridge:
    """Thin facade exposing rebot_grasp API to wrc_demo skills."""

    def __init__(self, controller: RebotArmEndPose, grasp_driver: GraspDriver) -> None:
        self.controller = controller
        self.grasp_driver = grasp_driver

    def move_to_traj(self, x: float, y: float, z: float,
                     roll: float, pitch: float, yaw: float,
                     duration: float) -> bool:
        """Mirror set.py / main.py: controller.move_to_traj(...) returns IK success."""
        return bool(self.controller.move_to_traj(x, y, z, roll, pitch, yaw, duration=duration))

    def open_gripper(self, timeout: float | None = None) -> None:
        if timeout is not None:
            self.grasp_driver.open_gripper(timeout=timeout)
        else:
            self.grasp_driver.open_gripper()

    def grasp(self) -> bool:
        return bool(self.grasp_driver.grasp())

    def release_gripper(self) -> None:
        self.grasp_driver.release_gripper()
```

调用方保持 wrc_demo 的 dataclass 风格：

```python
# src/wrc_demo/skills/runtime.py (skill_grasp_object 重写后)
def skill_grasp_object(self, label: str) -> dict[str, Any]:
    bridge: RebotGraspBridge = self._bridge
    fix = self._localize(label)
    grasp6d, pre6d = bridge.transform_to_base(fix.grasp, pregrasp_offset_m=0.08)
    bridge.open_gripper()
    if not bridge.move_to_traj(*pre6d, duration=2.0):
        raise SkillError("pregrasp IK failed")
    if not bridge.move_to_traj(*grasp6d, duration=1.5):
        raise SkillError("grasp IK failed")
    ok = bridge.grasp()
    if not ok:
        return {"ok": False, "error": "empty grasp"}
    bridge.move_to_traj(*self._home6d, duration=3.0)
    self.held_object = label
    return {"ok": True, "held": label}
```

---

## Testing Strategy

| 层次 | 工具 | 覆盖 |
|---|---|---|
| **Unit** | pytest | `transform_grasp_pose_to_base` 参数化测试；`select_profile` keyword→profile 映射 |
| **Bridge unit** | pytest | `RebotGraspBridge.move_to_traj` 透传 controller；`grasp()` 透传 grasp_driver |
| **Skill integration** | pytest + MockArm + rebot_grasp_mock | skill_grasp_object 完整 6 步调用序列断言 |
| **Pick-and-place integration** | pytest + MockArm | skill_pick_and_place 同帧 detect 两目标；grasp→home→place→home 序列 |
| **Real robot** | 手动 | 真机 demo 录像 vs set.py 当天录像目视对照 + 关节记录 diff |

### Mock 策略

`tests/conftest.py` 新增 fixture：

```python
@pytest.fixture
def rebot_grasp_mock(monkeypatch):
    """Mock rebot_grasp modules so tests don't need real hardware."""
    fake_controller = MagicMock()
    fake_controller.move_to_traj.return_value = True
    fake_driver = MagicMock()
    fake_driver.grasp.return_value = True
    fake_driver.open_gripper.return_value = None

    monkeypatch.setattr(
        "wrc_demo.skills.rebot_grasp_bridge.RebotArmEndPose",
        lambda *a, **kw: fake_controller,
    )
    monkeypatch.setattr(
        "wrc_demo.skills.rebot_grasp_bridge.GraspDriver",
        lambda *a, **kw: fake_driver,
    )
    return fake_controller, fake_driver
```

### Coverage expectations

- `test_align_to_rebot_grasp.py` 必须覆盖 AC1, AC2, AC3, AC4 所有 case
- 原 215 passed 中：纯逻辑测试（reflex / grounding / safety）保留并通过
- skill 端到端测试（test_streaming_fastpath.py 等）需要改写以适配新 mock

---

## Boundaries

### Always do
- 每次 `move_to_traj` 前过 `harness.approve(q_next)`；超 workspace / 接近 joint limit 直接 abort
- 每次 `grasp()` 前确认 `held_object is None`，避免覆盖状态
- `home_q` 归位后调用 `harness.clear_grasp_exemption()` 重置安全状态
- PR 引用本 spec 的 AC 编号（如 "implements AC2"）
- 测试覆盖率掉到 < 80% 不合并

### Ask first
- 改 SafetyHarness 的 7 项检查逻辑
- 改 `home_q` 值
- 改 `box z=0.08` hardcoded（提 ADR 说明为什么）
- 给 GraspDriver 传新的 gripper config
- 改 `depth_quantile` / `pregrasp_offset_m`（目前对齐 rebot_grasp 默认值；偏离要明确说明）

### Never do
- 把 `move_to_traj` 换成 `controller.move_joints` 或裸 SDK 调用 — 失去 IK + 时序对齐
- 在 wrc_demo 内手写 `close_gripper_two_stage` 或其它自创 gripper 控制 — 必须走 `GraspDriver`
- 删 `transform_grasp_pose_to_base` 等 rebot_grasp 函数的安全检查（如 `canonicalize_parallel_gripper_tcp_rotation` 的右手系修正）
- 跳过 SafetyHarness 直接调 SDK
- 把 rebot_grasp 当成 subprocess 跑（违反 assumption 1）

---

## Success criteria (summary)

1. `pick up the banana` 在真机 demo 成功，轨迹与 set.py `G` 键录像对齐（目视 + 关节 log）
2. `put the banana into box` 在真机 demo 成功，box z=0.08 落点准确
3. `tests/test_align_to_rebot_grasp.py` 全部通过
4. 既有纯逻辑测试（reflex / grounding / safety）保留并通过
5. skill 端到端测试改写后通过（如果改写无法保留，则标记 xfail 并在文档中说明原因）
6. 真机 demo 命令可成功启动（无 import 错误，无 PYTHONPATH 问题）

---

## Open Questions

1. **Motion duration 参数化?** set.py 用 `duration=2.0/1.5/3.0` hardcoded。是否需要在 wrc_demo 的 config 里允许覆盖？还是保持 hardcoded 以严格对齐？
   **建议**: 保持 hardcoded，理由是 set.py 也是 hardcoded，对齐优先。

2. **GraspDriver 实例化.** rebot_grasp 的 GraspDriver 需要 `rebotarm` + `controller` + `gripper_config`。wrc_demo 当前 SafeArm 不暴露这些。要新建 `wrc_demo/control/rebot_rs.py::RebotRsArm` 同时持有 `controller + grasp_driver` 还是分开？
   **建议**: 同一对象持有两个属性，bridge 直接拿。

3. **WorldWatcher 是否需要适配?** `_localize('box')` 之前走 belief cache，hit 时跳过 YOLO。新流程要求 "同帧 detect banana + box"，是否需要在 reflex 层面禁用 cache miss 时分两次调用？
   **建议**: 在 `skill_pick_and_place` 入口强制 `frame = observe()` 一次，然后只在这一个 frame 上同时跑 banana + box 的 detect。

4. **box z=0.08 是否要在 release 前做 sanity check?** box 顶面比 0.08 高（如 box 堆叠）时硬编码 z 会撞到 box 顶。
   **建议**: 第一版保持 hardcoded 跟 set.py 一致；后续真机验证时如果撞了再改。

5. **GraspDriver.grasp() 的 close 行为 vs v5 width-aware.** 已读 GraspDriver 源码确认：
   - `grasp()` 是单段 close：motor MIT 模式以 `close_torque=1.5` 闭合，检测到 stall (|vel| < 0.05 rad/s) 或 hard_stop 判定物体 / 空抓
   - 默认 `default_force=0.30`（hold torque），wrappers can override `grasp(force=1.0)`
   - v5 的 `_close_two_stage(pos2=0.9053, effort=1.0)` 是 width-aware，30mm 香蕉压到 9mm
   - GraspDriver.grasp() 是 force-aware，停在接触点，hold torque 维持
   - **决策**: 第一版用 `grasp_driver.grasp(force=1.0)`（接近 v5 effort），30mm 香蕉可能压到更紧。如果真机验证发现 banana 表面被压扁或夹爪过紧，降到 0.5 或保留 v5 流程。

6. **ready_pose 用 config 还是 wrc_demo v5? → 已决策 2026-08-10**
   - **决策**: 全部用 **wrc_demo `home_q = [0, 0.1684, 0.6226, -0.4543, 0, 0]` (joint space, FK ≈ (0.25, 0, 0.35))** 作为所有"初始 / 中间 / 归位"位姿。
   - 不使用 rebot_grasp config 默认 `(0.3, 0, 0.3, rpy=0, 0.7, 0)`（set.py 当天跑的值）。
   - 不使用 wrc_demo v5 的 `ready_pose = (0.25, 0, 0.35, rpy=0, 1.2, 0)`（wrist 标准化目标）。
   - 注：归位时通过 `controller.move_to_traj(...)` 走 IK 到 home_q 的 FK 位置，腕部姿态由 IK 解自然产生（不一定 = set.py pitch=0.7）。place 阶段 rpy=(0,1.2,0) 保持 set.py 的 place pose 不变。

7. **hand_eye_compensation 差异.** rebot_grasp config `hand_eye_compensation_m.z = -0.01`，wrc_demo JSON baked `-0.02`。是否需要在 wrc_demo JSON 里把 z 改成 -0.01 来对齐 rebot_grasp 当前配置？
   **建议**: 保持 wrc_demo JSON 不动（handover 已确认 xy < 1mm 是用 -0.02 标定的），但需要在 ADR 里记录这个 1cm 的差异来源。
