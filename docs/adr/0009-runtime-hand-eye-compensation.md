# ADR-0009: Runtime hand-eye manual compensation parameter

**Status:** Accepted 2026-08-26 (v2: per-camera only; v1 originally had demo.yaml global + per-camera override, dropped the global after considering multi-camera future)
**Deciders:** wrc_demo maintainer
**Related:** [ADR-0003 - hand-eye compensation delta](./0003-hand-eye-compensation-delta.md)
**Mirrors:** rebot_grasp cfg.calibration.hand_eye_compensation_m

## Context

configs/cameras/*.yaml 的 extrinsics: block 现在只读三类源（new JSON / cuRobo NPZ / legacy NPZ / inline T），calibration baked 完之后没有运行时手动补偿的钩子。现场校准跑完后，如果发现物体定位系统性偏差几 mm 到几 cm，重新跑 calib_top_orbbec 又太重——想要一个轻量级的运行时微调旋钮。

rebot_grasp 的 default.yaml 早就有这个旋钮（calibration.hand_eye_compensation_m: {x, y, z}，默认 [0, 0, -0.01]），通过 utils/camera_utils.py::compose_cam_to_base_transform 应用：
- ETH: T_cam2base = T_compensation @ T_hand_eye
- EIH: T_cam2base = T_compensation @ T_tcp2base @ T_hand_eye

## Decision

**Per-camera only.** 每个相机 profile 在 `extrinsics:` block 下声明自己的 `hand_eye_compensation_m`：

```yaml
# configs/cameras/orbbec_overhead.yaml
extrinsics:
  source: "${repo}/data/calibration/hand_eye_top.json"
  mode: eye_to_hand
  hand_eye_compensation_m:    # meters; default [0, 0, 0] = identity
    x: 0.00
    y: 0.00
    z: 0.00
```

**为什么不做 demo.yaml 全局 fallback**：

v1 最初设计是 demo.yaml 顶部放全局默认 + per-camera override（"camera wins"）。在考虑多相机未来时被否决——demo.yaml 顶部的全局值会变成 **silent default**：新加相机 profile 时如果不显式声明，就会从 demo.yaml 静默继承一个非零值，调试时非常容易踩坑。每台相机必须显式声明自己的补偿值（即使是 `[0, 0, 0]`），让配置自描述、无隐式魔法。

**实现位置**：
- src/wrc_demo/perception/grounding.py::Extrinsics 读 `cfg.hand_eye_compensation_m`，左乘到 `cam_to_base()`。
- src/wrc_demo/config.py::load_demo_config() **不做 fallback merge**——per-camera 唯一。

## 与 baked compensation 的关系

**先后顺序**：
1. JSON / NPZ loader 应用 baked 矩阵（ADR-0003：hand_eye_top.json 里 baked -0.02）
2. cam_to_base() 应用 `_T_compensation`（来自 camera profile `extrinsics.hand_eye_compensation_m`）

**不会双重补偿**——默认值 `[0, 0, 0]`（identity）。用户给 z=-0.01 是**额外**的 -1 cm，不是叠加在 baked -2 cm 上变成 -3 cm。

## Consequences

### Positive
- **多相机安全**：每台相机显式声明自己值，加新相机时不会从全局静默继承
- **配置自描述**：打开一个 camera profile 就知道它的补偿值
- **跟 rebot_grasp 语义完全对齐**：单相机场景下行为等价
- **默认 identity**：现有配置行为完全不变，向后兼容

### Negative
- **多相机同向偏移要重复写**：如果三台相机都要 z=-0.01，得在三个 profile 各写一遍（v1 全局版可以一行搞定）。可接受——加相机本来就是低频操作
- **不是校准替代品**：大偏差（>1 cm）应该重跑 calib_top_orbbec
- **不验证方向**：写错符号会立刻表现成物体位置偏移，要靠人工 observe 发现

## Verification

- tests/test_extrinsics_loader.py: 4 个测试（default zero / ETH shift / 与 rebot_grasp 语义对齐 / EIH+FK）
- tests/test_hand_eye_compensation_per_camera.py: 2 个测试（per-camera 值被加载到 cfg.camera.extrinsics；不写就 None）
- 现场：在 orbbec_overhead.yaml 的 `extrinsics.hand_eye_compensation_m` 写 z=-0.005，观察物体定位整体偏移 +5 mm
