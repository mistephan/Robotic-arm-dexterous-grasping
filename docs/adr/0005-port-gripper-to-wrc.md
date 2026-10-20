# ADR-0005: Port `GraspDriver` to `WrcGripper`; wrc_demo has zero runtime dep on rebot_grasp

**Status:** Accepted 2026-08-10
**Deciders:** wrc_demo maintainer
**Supersedes:** the implicit "rebot_grasp is a runtime dep" assumption in
[docs/specs/align-to-rebot-grasp.md §Tech Stack v1](../specs/align-to-rebot-grasp.md)

## Context and Problem Statement

The bridge path of `skill_grasp_object` / `skill_pick_and_place`
originally imported two runtime symbols from outside wrc_demo:

- `rebot_grasp.drivers.robot.grasp_driver.GraspDriver` — high-level
  gripper state machine (open / grasp with stall detection / release).
- `reBotArm_control_py.controllers.RebotArmEndPose` — vendored motor
  control SDK (IK + min-jerk trajectory via CAN).

`reBotArm_control_py` is already vendored inside the wrc_demo checkout
at `wrc_demo/sdk/reBotArm_control_py/`, and `wrc_demo/__init__.py`
injects that directory into `sys.path` at import time — so the SDK
import is self-contained.

`GraspDriver`, however, lives at
`/home/seeed/rebot_grasp/drivers/robot/grasp_driver.py` in a SEPARATE
checkout. `make_arm(rebot_rs)` did
`from rebot_grasp.drivers.robot.grasp_driver import GraspDriver`, and
the rebot_grasp parent directory was forced onto `sys.path` via
`tests/conftest.py` (namespace-package workaround). The demo
launch command required the operator to set
`PYTHONPATH=/home/seeed/wrc_demo/src:/home/seeed`.

The maintainer explicitly rejected this design: **wrc_demo must be
self-contained**; rebot_grasp is a reference implementation, not a
runtime dependency. The PYTHONPATH edit on every launch was the
visible symptom.

## Decision

Port `GraspDriver` into wrc_demo as `WrcGripper` at
`src/wrc_demo/control/gripper.py`. Drop all `rebot_grasp.*` imports
and the `tests/conftest.py` namespace-package hack. Replace
`tests/test_rebot_grasp_importable.py` and
`tests/test_rebot_grasp_mock_fixture.py` with a single preflight test
for `wrc_demo.control.gripper.WrcGripper`.

## Implementation

`src/wrc_demo/control/gripper.py` is a 1:1 port of the upstream
`GraspDriver` class minus the rebot_grasp-specific helper functions
(`find_rebot_repo_root`, `ensure_rebot_sdk_in_syspath`,
`selected_arm_config`, etc.) — those were read-time configuration
lookups against the rebot_grasp repo, which wrc_demo does not need.
wrc_demo only supports the reBot RS hardware; the upstream
`defaults["rs"]` branch is hardcoded as module-level constants
(`_RS_ANGLE_OPEN = 5.0`, `_RS_COUNTERCLOCKWISE = False`,
`_RS_TAU_MAX = 1.5`, `_RS_CLOSE_TORQUE = 1.5`,
`_RS_DEFAULT_FORCE = 0.30`). These mirror
`rebot_grasp/config/default.yaml::gripper.rs`.

Behavioral parity is preserved exactly:

| Aspect | Upstream `GraspDriver` | wrc_demo `WrcGripper` |
|---|---|---|
| State machine | IDLE → POSITION → CLOSING → HOLDING | same |
| `open_gripper(d, t)` | position control to `d / MAX * angle_open` | same |
| `grasp(force, t)` | torque control with stall / hard-stop detection | same |
| `release_gripper(t)` | open + return to position 0 | same |
| Stall velocity | `0.05 rad/s` | `_STALL_VEL = 0.05` |
| Hard-stop angle | `_open_sign * 0.05` | same |
| MIT gains | `kp_move=5.0, kd_move=1.0, kd_close=0.5` | same |
| Hold-torque override | `grasp(force=...)` clips to `[0.05, tau_max]` | same |

The two intentional API differences:

1. **`set_position_target` is public** (was `_set_position_target`
   private upstream). wrc_demo owns both sides of the contract now —
   the bridge in `src/wrc_demo/skills/rebot_grasp_bridge.py` is the
   only caller, and there's no reason to hide the method behind an
   underscore. The `try/except AttributeError` wrapper in the
   bridge's `release_with_async_close` remains as defensive coverage
   against future renames (vs the upstream private API).

2. **`get_tcp_pose` is dropped.** wrc_demo computes TCP via its own
   `Kinematics` class (`src/wrc_demo/safety/harness.py::Kinematics`),
   which is the source of truth for FK; carrying a parallel FK in
   the gripper driver would be duplication.

The vendored `reBotArm_control_py` SDK is unchanged. wrc_demo
continues to import `from reBotArm_control_py.controllers import
RebotArmEndPose` and `from reBotArm_control_py.actuator import RebotArm`
— these are inside wrc_demo's own tree at `wrc_demo/sdk/`.

## Consequences

### Positive
- **No external PYTHONPATH.** The demo launch command is the standard
  wrc_demo form: `PYTHONPATH=/home/seeed/wrc_demo/src python -m
  wrc_demo.apps.demo ...`.
- **No namespace-package hack.** `tests/conftest.py` no longer
  inserts `/home/seeed` into `sys.path`; that line was rebot_grasp
  runtime-leakage. wrc_demo is now a normal Python package install.
- **No silent dependency.** Anyone reading
  `src/wrc_demo/control/__init__.py` sees `from
  wrc_demo.control.gripper import WrcGripper` — fully in-tree.
- **Same behavior, easier ops.** The motion + gripper behavior
  matches the rebot_grasp recording from 2026-08-07 (xy < 1 mm) because
  the tuning constants are copied verbatim.

### Negative
- **Will not auto-track rebot_grasp gripper tuning.** If rebot_grasp
  tunes `_STALL_VEL` or `kp_move` upstream, wrc_demo will not pick
  it up. Mitigation: ADR-0005 documents the values here; any change
  requires a deliberate edit + a commit message linking to the
  rebot_grasp PR.
- **`WrcGripper` is ~360 lines of MIT control + state-machine code
  that wrc_demo now owns.** Future maintenance (Python 3.13, new
  motor firmware, etc.) is on us. Mitigated by being a small,
  self-contained class with no external deps.

### Neutral
- **`tests/test_rebot_grasp_importable.py` and
  `tests/test_rebot_grasp_mock_fixture.py` are deleted.** Their
  purpose was to confirm rebot_grasp was importable from tests; with
  rebot_grasp out of the dep graph, that question is moot. A new
  `test_wrc_demo_native_gripper_module` preflight covers the
  replacement.

## Verification

- `tests/test_bridge.py` — 15 tests pinning the bridge's pass-through
  + harness integration.
- `tests/test_bridge_safety.py` — 4 tests pinning
  `vet_pose` pre-motion checks.
- `tests/test_align_to_rebot_grasp.py` — 15 tests pinning the
  bridge path of `skill_grasp_object` / `skill_pick_and_place`.
- `tests/test_make_arm_rebot_rs.py` — 3 tests pinning the rebot_rs
  arm wiring (now uses `WrcGripper` instead of
  `rebot_grasp_driver_for_arm`).
- `tests/test_phase6_preflight.py` — `test_wrc_demo_native_gripper_module`
  replaces the old rebot_grasp importability check.

Total: **272 passed, 4 skipped, 1 xfailed, 7 xpassed, 0 failed** (vs.
281 before this refactor; net -9 tests because the rebot_grasp-only
importability / mock-fixture tests were deleted without replacement).

## Cross-references

- ADR-0001 (home_q rendezvous): unchanged; the bridge still uses
  `home_q` FK for the post-grasp and post-place return.
- ADR-0002 (grasp force=1.0): unchanged; `bridge.grasp(force=1.0)`
  is forwarded to `WrcGripper.grasp(force=1.0)` — same call, same
  override semantics.
- ADR-0003 (hand-eye compensation delta): unchanged; calibration
  JSON is read by `wrc_demo.perception.grounding`, never via
  rebot_grasp.
- ADR-0004 (drop gripper_config): obsoleted. ADR-0004 said "drop the
  silent-dropped gripper_config". The new `WrcGripper.__init__` takes
  an optional `gripper_config` that does override (no silent-drop),
  but wrc_demo callers pass `None` and use the per-call
  `force=1.0` override — same end-state as ADR-0004 prescribed.
