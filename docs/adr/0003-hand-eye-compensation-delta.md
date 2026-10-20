# ADR-0003: Hand-eye compensation delta (wrc_demo -0.02 vs rebot_grasp -0.01)

**Status:** Accepted 2026-08-10
**Deciders:** wrc_demo maintainer
**Spec:** [docs/specs/align-to-rebot-grasp.md §Open Q7 resolved](../specs/align-to-rebot-grasp.md)

## Context and Problem Statement

`rebot_grasp/config/default.yaml` defines:

```yaml
calibration:
  hand_eye_compensation_m:
    x: 0.00
    y: 0.00
    z: -0.01
```

`wrc_demo/data/calibration/hand_eye_top.json` was baked with
`z=-0.02` (per the v5 handover note: "T_cam2base 跟 set.py 当天 (2026-08-07)
NPZ 同源; baked -0.02").

The delta is **1 cm along the camera-z axis** between rebot_grasp's
current default and wrc_demo's working calibration.

In `rebot_grasp/scripts/set.py::_cam_to_base`, the compensation is
applied via `compose_cam_to_base_transform(tcp_pose, T_hand_eye,
mode, cfg)`:

```python
return compose_cam_to_base_transform(
    grasp_driver.get_tcp_pose(),
    T_hand_eye, hand_eye_mode, cfg,
)
```

If we ran the same `cfg` file through wrc_demo's bridge path,
`compose_cam_to_base_transform` would apply `-0.01` to z, undoing the
`-0.02` baked into `hand_eye_top.json`. Net effect: 1 cm offset in
the camera-z direction.

## Decision

**Keep wrc_demo's `hand_eye_top.json` as the source of truth.** Do
**not** re-import the rebot_grasp NPZ; do **not** overwrite the `-0.02`
compensation. Document the 1 cm delta as an accepted divergence.

## Rationale

- The handover note says `xy 错位从 88mm 降到 0.2mm` after baking `-0.02`.
  That accuracy is the v5 baseline the user has signed off on; rewriting
  the calibration would invalidate the existing rig-verified trajectory
  comparison.
- rebot_grasp's `-0.01` is the upstream default; it may or may not
  match set.py's actual 2026-08-07 recording. We don't have a record of
  which `cfg` file set.py ran with on that day. Picking `-0.02` is the
  safe choice (preserves xy < 1mm).
- The bridge path in wrc_demo does **not** call
  `compose_cam_to_base_transform` — it uses
  `wr_demo.perception.grounding.localize_object` which reads
  `hand_eye_top.json` directly. So the rebot_grasp config compensation
  is **not** applied on the wrc_demo path. The `-0.01` in rebot_grasp's
  config would only matter if we imported `compose_cam_to_base_transform`
  from rebot_grasp.utils.camera_utils (we don't, per
  docs/specs/align-to-rebot-grasp.md §"Out of scope").

## Consequences

### Positive
- **xy < 1 mm accuracy preserved** (verified by v5 handover).
- **No code change required** — the bridge does not touch rebot_grasp's
  compensation.

### Negative
- **Future porting risk.** If a future task adds `compose_cam_to_base_transform`
  to the bridge path (e.g. for a different camera), the import will
  silently apply `-0.01` and shift the work by 1 cm unless the caller
  is aware. Mitigated by this ADR + the comment at the future call
  site.
- **Calibration drift.** If the rebot_grasp team ever updates the
  `-0.01` default, wrc_demo's `-0.02` baked value doesn't follow.
  Re-tuning would require a new hand-eye calibration session.

### Neutral
- The 1 cm delta is smaller than the bridge's `release_clearance_m`
  equivalent (we use hardcoded `0.08`, rebot_grasp `release_clearance_m`
  not consulted). So the delta is invisible at the box level.

## Implementation

No code change required. The wrc_demo `Extrinsics` class reads
`hand_eye_top.json` and applies the baked `-0.02` at load time:

```python
# src/wrc_demo/perception/grounding.py
T_compensation = np.eye(4)
T_compensation[2, 3] = hand_eye_compensation_m["z"]  # -0.02 baked
```

This is unchanged from v5. The ADR exists for future-maintainer
clarity only.

## Verification

- `tests/test_extrinsics_loader.py` — asserts `hand_eye_top.json`
  loads with `z=-0.02` baked; if a future edit overwrites it, this
  test fails loudly.
- Phase 6 real-rig: place position vs set.py recording should remain
  within 1 cm z (inherited v5 accuracy). If the delta widens, this
  ADR + the JSON need re-investigation.

## Cross-references

- ADR-0001 (home_q): orthogonal — different question.
- ADR-0002 (grasp force): orthogonal.
- ADR-0004 (gripper_cfg): orthogonal.
- `docs/intent/align-to-rebot-grasp.md` — original intent notes the
  xy < 1 mm after baking `-0.02` but did not formally accept the
  1 cm delta from rebot_grasp's upstream default. This ADR closes
  that gap.
