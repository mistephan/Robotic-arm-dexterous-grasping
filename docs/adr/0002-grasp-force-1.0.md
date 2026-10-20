# ADR-0002: Override GraspDriver.grasp() force to 1.0

**Status:** Accepted 2026-08-10
**Deciders:** wrc_demo maintainer
**Spec:** [docs/specs/align-to-rebot-grasp.md §Open Q5 resolved](../specs/align-to-rebot-grasp.md)

## Context and Problem Statement

`rebot_grasp.GraspDriver.grasp()` takes a `force` parameter (hold-torque
magnitude in MIT τ units). rebot_grasp's own `set.py` does not pass
`force`, so it inherits the default from `rebot_grasp/config/default.yaml`:

```yaml
gripper:
  rs:
    default_force: 0.30
```

`force=0.30` was tuned by rebot_grasp for general object handling —
it is conservative on torque to avoid crushing deformable objects.

wrc_demo v5 used `SafeArm.close_gripper_two_stage(effort=1.00)` for
slippery objects like the 3D-printed banana, which exercises the
full MIT τ clip cap (`τ_max = 1.5` from the rebot_grasp config).
Real-rig observations: with `effort=1.00`, the banana stays gripped
through carry and release; with `effort=0.30`, it slips out within
~2 s of lifting (the printer's surface is too smooth for the lower
hold-torque).

The bridge path replaces `_close_two_stage` with `GraspDriver.grasp()`.
If we keep the default, slippery objects slip out; if we override to
`force=1.0`, we approach v5's effective grip strength.

## Decision

Call `self.bridge.grasp(force=1.0)` from both bridge methods,
overriding rebot_grasp's default `0.30` on every grasp.

## Consequences

### Positive
- **v5 grip strength preserved** for slippery objects without keeping
  the legacy `_close_two_stage` code path.
- **set.py trajectory timing preserved** — the override is a single
  argument, not an extra motion.

### Negative
- **Banana surface damage risk.** `force=1.0` (close to `τ_max=1.5`)
  may visibly dent the 3D-printed surface or stall the motor on a
  stiffer object.
- **Not adaptive.** The legacy v5 path selected `effort` based on
  `select_profile(label, material)` (slippery / rigid / etc.). The
  bridge path always uses 1.0 regardless of object. A future task could
  re-introduce per-object profile lookup.
- **Debug-time trap.** The "magic number 1.0" inside the bridge call
  is not visible from the config or from the `cfg.arm` profile; future
  maintainers must read the bridge source to know why.

### Mitigations
- If banana damage is observed in Phase 6, lower to `force=0.5`.
- Tunable via `cfg.grasp.brasp_force` in a follow-up if a second
  material type appears (deferred — no observed need).

## Alternatives Considered

- **(A)** Use rebot_grasp default `0.30`. Rejected: banana slips out,
  defeats the whole alignment purpose.
- **(B)** Make `force` configurable via `cfg.grasp.brasp_force` (default
  1.0). Deferred: no second-object need; adds a config knob without
  users.
- **(C)** Re-add `_close_two_stage` alongside the bridge for slippery
  objects only. Rejected: doubles the maintenance surface and
  re-introduces the legacy code path we just removed from bridge
  flow.
- **(D)** Read `cfg.arm.gripper.effort_default` (currently 0.6 per
  `configs/arms/rebot_rs.yaml`) and pass that. Rejected: 0.6 is also
  too low for slippery objects; we'd need to either bump the default
  or carry the override anyway.

## Implementation

`src/wrc_demo/skills/runtime.py`:

```python
# _skill_grasp_object_bridge:
ok = self.bridge.grasp(force=1.0)

# _skill_pick_and_place_bridge:
ok = self.bridge.grasp(force=1.0)
```

`src/wrc_demo/skills/rebot_grasp_bridge.py`:

```python
def grasp(self, force: Optional[float] = None) -> bool:
    if force is not None:
        return bool(self._grasp_driver.grasp(force=force))
    return bool(self._grasp_driver.grasp())
```

## Verification

- `tests/test_bridge.py::test_grasp_with_force_overrides_default` —
  asserts `grasp(force=1.0)` is forwarded to the underlying driver.
- `tests/test_align_to_rebot_grasp.py` — 13 tests asserting the bridge
  path calls `bridge.grasp(force=1.0)` exactly once per pick.
- Phase 6 real-rig: banana holds through lift + carry + release.
  Observe surface for dent / stall; tune if needed.
