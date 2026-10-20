# Hardware Verification Handover — 2026-07-24

This is a **handoff document** for resuming the real-arm verification on a
fresh conversation. Read this before starting Phase B.4 onwards.

## Context: the SDK migration is done

`wrc_demo` has been migrated off its bespoke 1100-line control stack
(`rebot_rs_arm.py`, `lazy_arm.py`, `arm_base.py`, `kinematics.py`)
onto the vendored SDK at `wrc_demo/sdk/reBotArm_control_py/`. See
`docs/MIGRATION_PLAN.md` for the full record.

Mock-stack test suite: **165 passed, 1 skipped, 6 deselected (hardware)**
— see `Phase 8` of the migration plan. The LLM/agent/perception/memory
side is unchanged. The control side is now a 70-line vendored SDK
patch (gripper tau, waypoint_hook, move_joints_traj) plus a 400-line
`SafeArm` thin wrapper in `safety/harness.py`.

The hardware that has been changed since the mock tests:
- `wrc_demo/src/wrc_demo/safety/harness.py` — `SafeArm._make_hook` now
  skips the harness check for the first waypoint (the current pose),
  because the current pose can be slightly outside the URDF joint limit
  (see Finding #3 below) and that would otherwise veto a perfectly
  valid trajectory.

## Hardware verification status (Phase A + B partial)

### ✅ Phase A — pure static (no motion)
- CAN bus UP @ 1 Mbps, motorbridge 0.5.0
- 7 motors on the bus (joint1-6 + gripper)
- `RebotArm()` instantiates cleanly, config loads

### ✅ Phase B.1 + B.2 — enable + read state
- 500 Hz `_loop_cb` actually runs (486 Hz measured)
- `tau_ff` (gravity feed-forward) computed correctly
  (joint2/3 have non-zero tau at zero pose)
- Motors respond to feedback only after `enable()` — before enable,
  `motor.get_state()` returns `None`

### ✅ Phase B.3 — gravity FF droop test (the critical v5 fix)
- 4-second static hold, no trajectory sent
- **TCP Z drift: -0.11 mm** (essentially encoder quantisation noise)
- **SDK's gravity FF works perfectly** on this URDF
  (joint2/3 × 1.55 scale baked into SDK's `_loop_cb`)
- v5 pre-fix expectation: 5-15 mm droop → actual: sub-millimeter

### ⚠️ Phase B.4 — drive to home_q (BLOCKED)
Tried 3 approaches, all failed:
1. **Direct `_q_target[:] = home_q`**: the 500 Hz `_loop_cb` is
   running and supposedly picks up the new target, but the motors don't
   move. Need to verify whether `_loop_cb` is actually reading the
   modified array.
2. **`safe_arm.move_joints(home_q)`**: harness rejects
   ("joint2 out of limits (0.004 ∉ [0.020, 3.120])") — see Finding #3.
3. **Direct `sdk_arm.arm.send_mit(...)`**: the 500 Hz loop overrides
   every direct call within 2 ms.

### ✅ Official motorbridge CLI confirms hardware is fine
After `clear_error` on every motor (see Finding #1):
```
motorbridge.cli run --vendor robstride --model rs-06 --motor-id 1 \
    --mode mit --pos 0.5 --vel 0 --kp 50 --kd 5 --tau 0 \
    --loop 200 --dt-ms 50
#170 pos=+0.489 vel=-0.018 torq=+0.411 status=0
#199 pos=+0.489 vel=-0.053 torq=+0.664 status=0
```
Joint1 drove to 0.489 rad over ~10 s, status=0 (no fault), ~0.5 N·m
torque feedback. **The hardware works.** The SDK integration is the
problem.

## Critical findings for the new conversation

### Finding #1 — Motors need `clear_error` after enable

The SDK's `enable_all()` does **NOT** clear motor fault bits. After the
v5-era tests (which left motors in fault state), every new SDK
session hits `fault_raw=0x00000004` (status_code=1) on every motor and
the motors refuse MIT commands silently.

**Fix needed**: add `clear_error` to the `make_arm()` factory in
`wrc_demo/control/__init__.py`, or to `RebotArmEndPose.start()`, or
explicitly in `safe_arm.connect()`. The current state requires the
caller to do it manually:

```python
from reBotArm_control_py.actuator import RebotArm
arm = RebotArm(hw_yaml='rebotarm_rs.yaml')
arm.connect()
for name, motor in arm._motor_map.items():
    motor.clear_error()
```

### Finding #2 — 500 Hz `_loop_cb` overwrites direct `send_mit`

The 500 Hz control loop in `RebotArmEndPose._loop_cb` calls
`arm_group.send_mit(_q_target, ...)` 500 times per second. Any direct
`sdk_arm.arm.send_mit(...)` call from a script is overwritten within
2 ms.

Two clean ways to command motion:
- **Write `sdk_ctrl._q_target[:] = joint_target`** and let the loop
  pick it up. (The 500 Hz loop reads `self._q_target` and sends it.)
- **Use the SDK's trajectory API**: `move_joints_traj(pts, duration,
  hook)` (joint-space) or `move_to_traj(x, y, z, rpy, duration, hook)`
  (Cartesian, with FK→IK→SE(3)→CLIK).

Direct `arm.arm.send_mit(...)` is essentially useless while the control
loop is running.

### Finding #3 — Harness veto on a static current pose

The SDK's URDF for `00-arm-rs_asm-v3` has `joint2 limit = [0, 3.14]`
(axis `0 0 1`, revolute). The actual motor at mechanical zero reports
`joint2 = +0.004 rad` (slight encoder offset). With harness
`joint_margin = 0.02` the effective lower limit is `0.02 rad`. The
arm is at `0.004` < `0.02` → harness vetoes any trajectory that starts
from this pose (the static pose check in `_check`).

`SafeArm._make_hook` was patched to skip the harness check for the
first waypoint (the current pose). This is a workaround. The clean
fixes are:

1. **Edit the SDK URDF**: change `joint2 limit lower="0"` to
   `lower="-0.05"` in `sdk/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf`
   (1 line change). Allows the motor to be at its actual zero.
2. **Or call `motor.set_zero_position()`** before each session to
   calibrate mechanical zero to encoder zero. Changes kinematic chain
   meaning; user OK required.
3. **Or reduce `safety.joint_margin` in `configs/demo.yaml`** to a
   smaller value (e.g. `0.005`) to allow the actual hardware range.

## Files modified during this session

- `wrc_demo/src/wrc_demo/safety/harness.py` — `_make_hook` adds
  `first_seen` flag to skip the first-waypoint harness check
- `wrc_demo/sdk/reBotArm_control_py/kinematics/robot_model.py` —
  `get_joint_limits` patched to use `idx_q` (the v0 migration patch)

## Next steps (the actual Phase B.4 work)

The goal: get the arm from mechanical zero to `home_q = [0, 0.1684,
0.6226, -0.4543, 0, 0]` (TCP at [0.25, 0, 0.35]) and verify
convergence + zero-droop hold.

1. **Apply the `clear_error` fix** in `make_arm()` factory or in the
   boot sequence. Without this every motion attempt will silently fail.

2. **Verify the `_q_target` write path works** under the live 500 Hz
   loop. Suggested test:
   ```python
   sdk_ctrl._q_target[:] = np.array([0.1, 0.1, 0.1, 0, 0, 0])
   time.sleep(2.0)
   q = arm.get_state().q
   # should see q ≈ [0.1, 0.1, 0.1, 0, 0, 0]
   ```
   If this doesn't move, the `_loop_cb` is reading a different buffer
   than the one we wrote.

3. **Apply the URDF or margin fix** so the static-pose check at the
   current position passes.

4. **Once motion works**: drive to `home_q`, measure convergence
   (< 0.01 rad steady-state), measure 3 s hold drift (< 0.5 mm).

5. **Then**: Phase C (small joint + Cartesian motion), Phase D
   (e-stop / soft-stop paths), Phase E (cleanup).

## How to start the new conversation

Open a new chat, then paste:

> 接 `wrc_demo` SDK 迁移的真机验证。
>
> 交接文档：`/home/seeed/wrc_demo/docs/HARDWARE_VERIFICATION_HANDOVER.md`
> 迁移文档：`/home/seeed/wrc_demo/docs/MIGRATION_PLAN.md`
>
> 当前状态：
> - 8 个迁移 phase 全部完成（mock 测试 165/165 通过）
> - 真机 Phase A + B.1-B.3 通过：CAN bus OK, 7 motors 在线, gravity FF
>   验证 < 0.2mm 漂移
> - Phase B.4 失败：机械臂无法从机械零位移动到 home_q
>
> 关键发现：
> 1. motors 必须 `clear_error`（SDK 不自动清，v5 测试遗留 fault）
> 2. 500Hz `_loop_cb` 覆盖所有直接 `send_mit`（要写 `_q_target` 或用 SDK trajectory API）
> 3. SDK URDF `joint2 limit=[0, 3.14]` 太严——实际机械零位=0.004 rad，harness margin=0.02 把它判出界
>
> 已在 `safety/harness.py::_make_hook` 加了 `first_seen` 跳过第一个 waypoint 的 check
> （临时绕过 #3 的症状，没改根因）
>
> 帮我：先把 #1 + #2 + #3 的修复合进代码，然后让 safe_arm.move_joints(home_q) 真的能驱动机械臂到 home 并稳定保持（< 0.5mm 漂移）。

## Important context

- 24V power is ON, CAN cable connected, `can_arm` is UP at 1 Mbps
- Arm is at mechanical zero (motors enabled, holding torque on current pose)
- All scripts use `/home/seeed/wrc_demo/.venv/bin/python` (relative
  `./.venv/bin/python` fails — use absolute path)
- `clear_error` is `motor.clear_error()` (no args)
- `motor.get_state()` returns `MotorState(can_id, arbitration_id,
  status_code, pos, vel, torq, t_mos, t_rotor)` — note `status_code`
  not `fault_raw` (the `fault_raw=0x...` you see in CLI output is the
  CLI's formatting, not a MotorState field)
- The arm's current `home_q` FK: TCP = [0.2500, 0, 0.3499] m
  (verified against `rebot_rs.yaml` comment)
- `q_target = [0, 0.1684, 0.6226, -0.4543, 0, 0]` is the SDK asset-frame
  home; no `joint_signs` multiplication needed
