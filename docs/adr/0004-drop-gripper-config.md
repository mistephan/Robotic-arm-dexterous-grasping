# ADR-0004: Drop `gripper_config` from `GraspDriver.__init__` call

**Status:** Accepted 2026-08-10
**Deciders:** wrc_demo maintainer
**Spec:** [docs/specs/align-to-rebot-grasp.md §"Out of scope" + review fix #5](../specs/align-to-rebot-grasp.md)

## Context and Problem Statement

`wrc_demo/control/__init__.py::make_arm` was originally written to
forward `cfg.arm.get("gripper", {})` to `GraspDriver(sdk_arm, sdk_ctrl,
gripper_config=gripper_cfg)`. Review (2026-08-10 code review pass)
flagged this as silently broken:

- `GraspDriver` expects `gripper_config[arm_type]` — a dict keyed by
  `"dm"` or `"rs"` with keys like `angle_open`, `counterclockwise`,
  `close_torque`, `default_force`.
- wrc_demo arm profile's `gripper` dict uses a different shape:
  `{open_pos, closed_pos, max_width_m, effort_default}`.

`GraspDriver.__init__` does:

```python
gcfg = {**defaults, **((gripper_config or {}).get(selected.arm_type) or {})}
```

When we pass `cfg.arm.gripper = {open_pos: ..., ...}`, the
`.get(selected.arm_type)` returns `None` (no `dm` or `rs` key), so
`gcfg` becomes `defaults` — rebot_grasp's arm-type defaults from
`rebot_grasp/config/default.yaml`. **wrc_demo's gripper settings
are silently ignored.**

This affects `safe_arm._grip_open / _grip_closed / _max_width` which
are loaded by SafeArm from `cfg.arm.gripper` directly — so SafeArm's
view of gripper geometry is correct, but GraspDriver's is the
rebot_grasp default. If a future task reads gripper geometry from
GraspDriver, it would get wrong values.

## Decision

Pass `gripper_config=None` to `GraspDriver(...)`. Document the
shape mismatch in the helper docstring.

## Rationale

- **Don't lie to the caller.** Forwarding `gripper_cfg` and silently
  dropping it is worse than not passing it — the silent-drop form
  hides the shape mismatch from anyone reading the code.
- **Defaults are correct for rs.** rebot_grasp's `default.yaml`
  `gripper.rs = {angle_open: 5.0, counterclockwise: false, tau_max: 1.5,
  close_torque: 1.5, default_force: 0.30}` is the right starting
  point for the B601-RS. We override `default_force=0.30` per-call
  via `grasp(force=1.0)` (ADR-0002), so the per-call override is the
  single knob.
- **If a future task needs `close_torque` override**, it should be
  done via rebot_grasp's expected format in a separate fix — not by
  fixing the silent-drop now (which would require a config-schema
  bridge between wrc_demo and rebot_grasp).

## Consequences

### Positive
- **No silent behavior.** The helper explicitly says "we use rebot_grasp
  defaults" via docstring.
- **Per-call `force=1.0` override remains the v5-compat knob.**
- **Simpler to reason about.** One config source (rebot_grasp
  defaults) plus one explicit override (per-call force).

### Negative
- **`cfg.arm.gripper.close_torque` setting is ignored on the rebot_rs
  path.** If a future task wants to tune close torque, it must edit
  `rebot_grasp/config/default.yaml::gripper.rs.close_torque` or add
  the rebot_grasp-shape config schema to wrc_demo.
- **MockArm / IsaacArm still use `cfg.arm.gripper`.** The default
  fixture fallback path is unaffected; only the rebot_rs bridge path
  loses the config-driven gripper knob.

### Neutral
- `safe_arm._grip_open / _grip_closed / _max_width` are loaded from
  `cfg.arm.gripper` by SafeArm itself, not via GraspDriver. They
  remain correct on the rebot_rs path.

## Implementation

`src/wrc_demo/control/__init__.py::rebot_grasp_driver_for_arm`:

```python
def rebot_grasp_driver_for_arm(sdk_arm, sdk_ctrl, gripper_cfg):
    """Construct a rebot_grasp ``GraspDriver`` for the rebot_rs backend.

    ``gripper_cfg`` is intentionally **not** forwarded to ``GraspDriver``.
    rebot_grasp expects ``gripper_config[arm_type]`` (a dict keyed by
    ``"dm"`` / ``"rs"`` with keys like ``angle_open``, ``close_torque``,
    ``default_force``), but the wrc_demo arm-profile gripper dict uses
    ``open_pos`` / ``closed_pos`` / ``max_width_m`` — a different shape
    that GraspDriver would silently drop on the floor. We pass
    ``gripper_config=None`` so GraspDriver uses its arm-type defaults
    from ``rebot_grasp/config/default.yaml`` (rs: close_torque=1.5,
    default_force=0.30). The v5-style high-effort grasp is restored by
    a per-call ``force=1.0`` override in the bridge path.

    Reviewed in ADR-0004.
    """
    from rebot_grasp.drivers.robot.grasp_driver import GraspDriver

    return GraspDriver(sdk_arm, sdk_ctrl, gripper_config=None)
```

## Verification

- `tests/test_make_arm_rebot_rs.py::test_rebot_rs_arm_has_grasp_bridge_attribute`
  — asserts the rebot_rs path constructs a bridge (no regression from
  dropping the kwarg).
- `tests/test_make_arm_rebot_rs.py::test_rebot_rs_grasp_driver_start_called_not_ctrl_start`
  — asserts `grasp_driver.start()` runs (GraspDriver with `gripper_config=None`
  defaults to its own arm-type config and starts normally).

## Cross-references

- ADR-0002 (grasp force=1.0): the per-call override that compensates
  for not passing `close_torque` / `default_force` via config.
- ADR-0003 (hand-eye compensation delta): orthogonal.
