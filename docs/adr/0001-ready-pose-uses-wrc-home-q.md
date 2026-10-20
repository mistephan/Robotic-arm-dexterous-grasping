# ADR-0001: Use wrc_demo home_q as bridge-path rendezvous pose

**Status:** Accepted 2026-08-10
**Deciders:** wrc_demo maintainer
**Spec:** [docs/specs/align-to-rebot-grasp.md §6 resolved](../specs/align-to-rebot-grasp.md)

## Context and Problem Statement

The bridge path of `skill_grasp_object` / `skill_pick_and_place` needs a
"rendezvous" pose between motion segments — a stable, repeatable pose
the arm returns to between (a) the grasp and the place, and (b) the
place and the end of the skill.

Two candidates exist:

- **(A)** `set.py` / `main.py` use `ready_pose = (0.25, 0, 0.35, rpy=0,
  1.2, 0)` in Cartesian space.
- **(B)** wrc_demo v5 uses `home_q = [0, 0.1684, 0.6226, -0.4543, 0, 0]`
  in joint space; its FK gives `TCP ≈ (0.25, 0, 0.35)` but the wrist
  orientation is whatever the IK naturally converges to.

The wrist orientation matters because `skill_place_at` requires
`rpy=(0, 1.2, 0)` for the place motion to IK-solve on the wrc_demo
B601-RS wrist. If the rendezvous pose has a different orientation,
the place leg must include a wrist reorientation in addition to the
translation — that costs both time and trajectory quality.

## Decision

Use **wrc_demo `home_q`** (joint space) as the bridge-path rendezvous.
FK to Cartesian via `self.kin.fk(home_q)` and pass to
`bridge.move_to_traj`; the controller's IK decides the actual
interpolated path.

## Consequences

### Positive
- **Single source of truth** for "home" across the wrc_demo codebase
  (`home_q` is also `safe_arm.move_home`, the bootstrap-pose, and the
  arm-profile default).
- **No wrc_demo ready-pose duplication** — the Cartesian
  `(0.25, 0, 0.35, rpy=0, 1.2, 0)` is no longer needed by the bridge
  path.
- **User confirmed ("wrc_demo home 就可以, 不影响什么")** that the wrist
  reorientation between `home_q` FK and `place rpy=(0, 1.2, 0)` is
  acceptable.

### Negative
- **Trajectory differs from set.py** at the rendezvous moment: set.py
  calls `_move_ready(ready_pose=…)` which lands at `rpy=(0, 1.2, 0)`;
  we land at whatever FK(home_q) produces (joint-dependent wrist
  orientation). Joint-space trajectory log comparison will show this
  difference.
- **Per-waypoint approval assumes `q_current` is reachable** from
  `home_q` FK. If rebot_grasp's IK converges to a different basin than
  the bridge's `select_grasp`, the wrist reorientation leg may traverse
  a sub-table region the harness would otherwise catch. Mitigated by
  `vet_pose(q_target)` pre-check (review fix #1).

### Neutral
- **Place leg rpy stays `(0, 1.2, 0)`** — set.py parity. The wrist
  reorientation happens during the home→place transition, not during
  place descent itself.

## Alternatives Considered

- **(A)** Use `set.py` ready_pose `(0.25, 0, 0.35, rpy=0, 1.2, 0)`.
  Pros: exact set.py trajectory. Cons: duplicates wrc_demo's
  ready-pose Cartesian config; user explicitly preferred wrc_demo's.
- **(C)** Use `cfg.arm.ready_pose` if set, fall back to `home_q`.
  Pros: configurability. Cons: two code paths; user said "wrc_demo
  的 home 就可以" implying a single value.

## Implementation

`SkillRuntime._home_q_to_6d()` in `src/wrc_demo/skills/runtime.py`:

```python
home_q = self.cfg.arm.get("home_q")
if home_q is None:
    # defensive fallback only — wrc_demo arm profiles always set home_q
    r = self.cfg.arm.get("ready_pose", {}) or {}
    return (r["x"], r["y"], r["z"], r["roll"], r["pitch"], r["yaw"])
q = np.asarray(home_q, dtype=float)
T = self.kin.fk(q)
roll, pitch, yaw = pin.rpy.matrixToRpy(T[:3, :3])
return (float(T[0,3]), float(T[1,3]), float(T[2,3]),
        float(roll), float(pitch), float(yaw))
```

Both `_skill_grasp_object_bridge` and `_skill_pick_and_place_bridge`
call `_safe_bridge_move(home6d, duration=3.0, q_target=home_q,
where="home")` so `vet_pose` validates `home_q` directly without a
redundant IK round-trip.

## Verification

- `tests/test_align_to_rebot_grasp.py` — 15 tests asserting the
  rendezvous is called between grasp and place (bridge path) and
  after release (pick_and_place).
- `tests/test_bridge_safety.py` — 4 tests asserting `vet_pose` runs
  for every `bridge.move_to_traj` including the rendezvous.
- Phase 6 real-rig: visual confirmation that `place rpy=(0, 1.2, 0)` is
  reached from `home_q` FK without violating workspace AABB.
