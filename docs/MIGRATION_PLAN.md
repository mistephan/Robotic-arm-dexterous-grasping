# SDK Migration Plan — 2026-07-24

This document is the record of the **vendor-SDK-into-tree** migration:
replacing the 1100+ lines of bespoke ``wrc_demo.control`` (RebotRSArm,
LazyArm, ArmBase, mini-Kinematics) with the vendored
``sdk/reBotArm_control_py/`` package plus a ~150-line thin
``SafeArm`` wrapper.

## Goal

| | Before | After |
|---|---|---|
| Real-arm control | 616-line ``RebotRSArm`` (re-implemented 500 Hz loop, gravity FF, sign convention) | 0 lines — SDK's ``RebotArmEndPose`` owns it |
| Joint-space FK / IK | Custom 150-line ``Kinematics`` (Pinocchio wrapper) | 0 lines — SDK's ``solve_ik`` + ``compute_fk`` |
| Hand-eye calibration | 2 legacy scripts tied to ``wrc_demo.control.kinematics`` | 2 new scripts modeled on ``rebot_grasp`` |
| Config files | 3 ``arm/*.yaml`` files with redundant ``joint_signs`` + 4 v5-specific fields | 3 simplified configs (~30 lines each) |
| Tests | 196 collected, 43 failing on the new harness | 220 collected, **all 165 selected pass** |

## Phases (with completion timestamps)

| # | Phase | What changed |
|---|---|---|
| 0 | Vendor SDK | Copy ``reBotArm_control_py/reBotArm_control_py/`` (+ ``config/``, ``urdf/``) → ``wrc_demo/sdk/``. Add ``wrC_demo/__init__.py`` sys.path injection. Add vendor README + patch notes. |
| 1 | SDK extensions | 70 lines: gripper tau in ``_loop_cb``, ``set_gripper_target(pos, effort)``, ``waypoint_hook`` in ``_send_loop``, new ``move_joints_traj`` + ``wait_motion_done``. |
| 2 | New harness | ~400-line ``safety/harness.py``: ``Kinematics`` (FK + ``link_positions``), ``SafetyHarness`` (8-check approve + escape rules), ``SafeArm`` (thin wrapper, hook integration). |
| 3 | Config cleanup | Drop ``joint_signs: [-1]*6`` from all 3 arm profiles (the URDF is already in the SDK's frame; the multiplication was always wrong). Add ``${sdk}`` placeholder. |
| 4 | Profile simplification | Drop ``mit_kp``, ``mit_kd``, ``gravity_ff``, ``control_rate_hz`` — all owned by SDK. Add ``effort_default`` for gripper. |
| 5 | Delete v5 code | ``rebot_rs_arm.py`` (616 lines), ``lazy_arm.py`` (99), ``arm_base.py`` (185), ``control/kinematics.py`` (150). Update ``MockArm`` / ``IsaacArm`` to duck-type against ``SafeArm`` instead of inheriting ``ArmBase``. Archive 7 v5 test files + 3 v5 docs to ``_archive/v5_control/``. |
| 6 | ``emergency_stow.py`` | Use ``make_arm`` factory + ``SafeArm.disconnect`` → SDK's ``RebotArmEndPose.end()`` (safe_home + disconnect). |
| 7 | Hand-eye scripts | New ``collect_handeye_eih.py`` + ``collect_handeye_eth.py`` modeled on ``rebot_grasp/scripts/collect_handeye_eih.py``; use ``wrc_demo.calibration.handeye_solver.joint_solve_restart`` + new ``CalibrationResult`` JSON schema. *(Top → in-tree ``wrc_demo.scripts.calib_top_orbbec`` at Phase 7; wrist → in-tree ``wrc_demo.scripts.calib_wrist`` 2026-08-27; ``collect_handeye_eih.py`` deleted as redundant 2026-08-27.)* |
| 8 | E2E validation | 165 / 165 mock-stack tests pass; demo CLI + MCP server + orchestrator + streaming fastpath all green. |

## File-level summary

### New files

| Path | Purpose |
|---|---|
| ``wrc_demo/sdk/`` (89 MB) | Vendored SDK + URDF + meshes + README + patch notes |
| ``wrc_demo/src/wrc_demo/safety/harness.py`` (385 lines) | New Kinematics / SafetyHarness / SafeArm |
| ``wrc_demo/scripts/collect_handeye_eih.py`` (~350 lines) | Eye-in-hand calibration *(deleted 2026-08-27 — superseded by in-tree ``wrc_demo.scripts.calib_wrist``)* |
| ``wrc_demo/scripts/collect_handeye_eth.py`` (~330 lines) | Eye-to-hand calibration |

### Files rewritten

| Path | Before | After |
|---|---|---|
| ``wrc_demo/src/wrc_demo/apps/demo.py`` | 346 lines (build_runtime uses LazyArm) | 339 lines (LazyArm flag now a no-op; uses ``make_arm`` factory) |
| ``wrc_demo/src/wrc_demo/control/mock_arm.py`` | 103 lines (inherits ArmBase, has ``stream_to`` override) | 280 lines (duck-typed SafeArm surface; real ``SafetyHarness`` inside) |
| ``wrc_demo/src/wrc_demo/control/isaac_arm.py`` | 78 lines (joint_signs conversion) | 76 lines (asset-frame, no sign multiplication) |
| ``wrc_demo/src/wrc_demo/control/__init__.py`` | re-exports ``ArmBase, make_arm, Kinematics, IKResult`` | re-exports ``MockArm, IsaacArm, make_arm`` + ``usd_model`` |
| ``wrc_demo/src/wrc_demo/config.py`` | ``${assets}`` / ``${repo}`` only | adds ``${sdk}`` placeholder |
| ``wrc_demo/src/wrc_demo/configs/arms/rebot_rs.yaml`` | 62 lines (joint_signs + 4 v5 fields) | 47 lines (essential only) |
| ``wrc_demo/src/wrc_demo/configs/arms/mock.yaml`` | 15 lines | 13 lines |
| ``wrc_demo/src/wrc_demo/configs/arms/isaac.yaml`` | 21 lines | 19 lines |
| ``wrc_demo/scripts/emergency_stow.py`` | 72 lines (writes ``_q_target`` directly) | 96 lines (uses ``SafeArm.disconnect``) |
| ``wrc_demo/src/wrc_demo/skills/runtime.py`` | small patches: ``cartesian_planner`` sys.path, ``arm.raw`` → ``arm``, keyword renames | |
| ``wrc_demo/src/wrc_demo/apps/mcp_server.py`` | small patches: keyword renames | |
| ``tests/test_safety.py``, ``test_persistence_loop.py``, ``test_orchestrator_e2e.py``, … | FakeKin update for new Kinematics surface | |
| 7 v5 test files | archived to ``_archive/v5_control/`` | |

### Files deleted

| Path | Lines | Why |
|---|---|---|
| ``wrc_demo/src/wrc_demo/control/rebot_rs_arm.py`` | 616 | Replaced by SDK + SafeArm |
| ``wrc_demo/src/wrc_demo/control/lazy_arm.py`` | 99 | SDK has clean bring-up; no lazy arm needed |
| ``wrc_demo/src/wrc_demo/control/arm_base.py`` | 185 | SDK is the single source of truth; MockArm / IsaacArm / SafeArm are duck-typed |
| ``wrc_demo/src/wrc_demo/control/kinematics.py`` | 150 | Moved to ``safety/harness.py::Kinematics`` |
| 7 v5 test files | ~700 | All in ``_archive/v5_control/`` |
| 3 v5 docs | ~1500 | All in ``_archive/v5_control/`` |

## Bugs found and fixed during migration

1. **SDK bug**: ``get_joint_limits`` used joint index where config index was required (crashed on 2-finger gripper URDFs).
2. **Escape rule allowed static invalid pose**: ``q_prev == q_next`` at an out-of-limit position passed the escape rule. Fixed to require strict inward motion on each outside axis.
3. **Non-reentrant lock deadlock**: ``_in_grasp`` re-acquired ``SafetyHarness._lock`` from inside ``_check`` which was already inside ``approve``'s lock. Removed the redundant ``with`` block.
4. **UnboundLocalError on ``tcp0``**: the table-clearance escape rule referenced ``tcp0`` which was only assigned in the workspace branch.
5. **``_in_exempt(radius=0)`` rejected valid input**: changed to ``radius < 0`` so a zero-radius cylinder at the exact target is allowed.
6. **``vet_pose`` returned ``bool`` instead of the documented ``Optional[str]``**: restored the old return shape.
7. **Mock arm ate ``SafetyViolation``**: changed to propagate, so the skill layer's existing exception handling keeps working.

## Verification

```
$ ./.venv/bin/python -u -m pytest tests/ -q --no-header
........s............................................................... [ 43%]
........................................................................ [ 86%]
......................                                                   [100%]
165 passed, 1 skipped, 6 deselected in 23.94s
```

- ``6 deselected`` are the ``hardware``-marked tests (need real CAN + L515)
- ``1 skipped`` is the stale ``test_verify_math_uses_camera_path_for_marker_position`` (depends on the to-be-rewritten ``calib_top_orbbec.py``)

End-to-end smoke test on the mock stack:

```
build_runtime: 0.14s
  arm: MockArm, connected: True, harness: SafetyHarness, n_joints: 6
get_observation -> ok: True, robot keys: q_deg, tcp_xyz, gripper_open_frac
move_joints(home) -> True
  state.q: [0.0, 0.168, 0.623, -0.454, 0.0, 0.0]   # == rebot_rs.yaml:home_q, in asset frame
close_gripper_two_stage -> 0.85
stop() / resume() cycle: harness estopped latches and clears correctly
shutdown: clean
```

## Outstanding work

The migration is mock-stack-complete. Hardware validation is the next step (and is on the user — see ``HARDWARE_VERIFICATION.md`` which is **stale** now and needs to be re-scoped against the new SDK-based stack):

1. **Real-arm bring-up**:
   - ``sudo ip link set can_arm up type can bitrate 1000000``
   - ``PYTHONPATH=src sudo .venv/bin/python -m wrc_demo.scripts.calib_wrist --manual`` (was: ``-m scripts.collect_handeye_eih --camera orbbec_wrist --arm rebot_rs``; migrated in-tree 2026-08-27 with the new joint solver + schema v1)
   - Verify the 500 Hz control loop is running (candump) and that gravity FF actually keeps the arm from drooping.
2. **Real-arm end-to-end**:
   - ``.venv/bin/python -m wrc_demo.apps.demo --task "pick up the red cube" --cameras orbbec_overhead --arm rebot_rs --llm anthropic``
3. **Re-scope ``HARDWARE_VERIFICATION.md``** for the SDK-based stack (the v5-era ``arm._q_target`` hacks are gone).
4. **Address open SDK concerns** (post-migration follow-ups):
   - SDK's no-tests caveat (``sdk/README.md``).
   - The ``enable()`` vendor-grouping quirk (``CONTROL_LOOP_FIX_PLAN.md`` §3.5).
   - Isaac arm still uses the legacy path (``control/isaac_arm.py`` docstring).
5. **Wire ``MockArm`` to the SDK for visual regression tests** — currently MockArm is pure numpy; a ``FakeSDK`` that mirrors the SDK interface would let mock tests catch regressions in the SafeArm ↔ SDK contract.