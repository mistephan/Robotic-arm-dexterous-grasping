# Vendored reBotArm SDK

This directory contains an in-tree copy of
[reBotArm_control_py](https://github.com/Seeed-Projects/reBotArm_control_py)
that ships with `wrc_demo` so the project is self-contained — no external
absolute paths, no PYTHONPATH edits at the shell, no stale `/home/spark`
or `/home/johnny` references in CI.

## Layout

```
sdk/
├── reBotArm_control_py/    # Python package (vendored from upstream)
├── config/                 # SDK hardware YAMLs (rebotarm_rs.yaml, etc.)
├── urdf/                   # URDF + meshes for the reBot DevArm
└── __init__.py             # sys.path injector
```

## Usage

```python
import wrc_demo.sdk  # noqa — this injects sdk/ into sys.path

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import (
    load_robot_model, compute_fk, joint_to_pose,
    get_joint_count, get_joint_limits,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (
    solve_ik, solve_ik_with_retry, IKParams, pos_rot_to_se3,
)
```

## Joint / coordinate convention

The URDF here is **the asset frame** — the same one the SDK's
`RebotArmEndPose._loop_cb` uses for `send_mit` after the gravity
feed-forward. `home_q` values in `configs/arms/*.yaml` are already in
this frame; **do not multiply by `joint_signs`** (the old wrc_demo
control layer used to do that and it was wrong).

The 8-DoF URDF has 6 arm joints + 2 gripper fingers; SDK groups use
the first 6 as `arm` and the last 1-2 as `gripper` (see
`config/rebotarm_rs.yaml`).

## Local patches

The following upstream bugs have been fixed in this vendor copy. Keep
the comments and update them if you re-sync from upstream:

- `reBotArm_control_py/kinematics/robot_model.py::get_joint_limits` —
  upstream indexes `model.lowerPositionLimit` by `jid` (joint index) but
  the array is sized by `idx_q` (configuration index). On URDFs with 2
  gripper fingers this raises `IndexError`. Patched to use
  `model.joints[jid].idx_q` (2026-07-24).

The following upstream API extensions were added by wrc_demo
(2026-07-24, see `docs/MIGRATION_PLAN.md`):

- `RebotArmEndPose.set_gripper_target(pos, effort=0.0)` — `effort` is
  clamped to `[0, 1.5]` and threaded into the gripper's MIT `tau` slot
  by `_loop_cb`. Upstream `send_mit` call to the gripper ignored tau;
  this makes material-aware close force (rigid/fragile/slippery/...)
  actually reach the motor.
- `RebotArmEndPose.move_to_traj(..., waypoint_hook=None)` — optional
  callable invoked at every waypoint BEFORE writing `_q_target`. If it
  returns False the trajectory aborts and `_moving` clears. This is
  how `SafetyHarness.approve` gets to veto individual SDK waypoints
  without re-implementing the 500Hz control loop.
- `RebotArmEndPose.move_joints_traj(q_waypoints, duration,
  waypoint_hook=None)` — new joint-space trajectory method. Same
  send-thread / hook semantics as `move_to_traj`, but skips the
  IK + SE(3) geodesic + CLIK pipeline (callers already know the joint
  path). Used by `SafeArm.move_joints` so `skill_move_home` and the
  grasp descent can drive in joint space without FK↔Euler round-trip.
- `RebotArmEndPose.wait_motion_done(timeout=30.0)` — convenience
  blocking helper that returns when `_moving` clears (or on timeout).
  Pairs naturally with `waypoint_hook` rejection: a hook returning
  False clears `_moving` synchronously inside `_send_loop`.

## Resyncing from upstream

```bash
# (not yet automated) — the plan is a 3-way merge:
#   1. copy upstream's reBotArm_control_py/ over ours
#   2. copy upstream's config/ + urdf/ over ours
#   3. re-apply the local patches above
```

Until that's automated, **treat this as a snapshot**; if upstream
publishes a hot-fix, port it manually with `diff` + `git apply`.

## What this directory is NOT

- Not the source of truth — upstream `Seeed-Projects/reBotArm_control_py`
  is. Open PRs there first.
- Not a runtime installable package — we don't `pip install sdk/`.
  It's only on `sys.path` for `import reBotArm_control_py.xxx`.
- Not independent of `wrc_demo` — the SDK was selected specifically
  because it covers 500Hz loop + gravity FF + sign conversion that
  `wrc_demo` would otherwise need to re-implement (see
  `HANDOFF_V5_CONTROL_FIX.md` for the history that led to vendoring).