# WRC Dashboard grasp and IK diagnostics

Status: implemented
Scope: observability only; no motion, safety, grasp ranking, or IK acceptance rule changes.

## Goal

A failed grasp must expose enough runtime data to distinguish:

- invalid grasp width;
- unreachable position;
- unreachable orientation;
- pregrasp-only failure;
- final-grasp failure;
- safety validation rejection;
- failure after an executable IK solution was selected.

The Dashboard is a display layer. It must never recompute IK.

## Runtime data flow

```text
camera/OBB/learned planner
  -> selector debug callback
  -> SkillRuntime atomic snapshot
  -> demo._runtime_state()
  -> StreamServer GET /state
  -> browser rendering
```

## State contract

`GET /state` contains a JSON-safe `grasp_debug` object:

```json
{
  "status": "planning | executable | succeeded | failed",
  "stage": "candidate | pregrasp_ok | rejected | selected",
  "target": "green candy package",
  "source": "camera_grasp | analytic_obb | graspgenx",
  "confidence": 0.87,
  "width_m": 0.021,
  "max_width_m": 0.095,
  "camera_xyz": [0.126, -0.034, 0.612],
  "base_xyz": [0.372, 0.181, 0.058],
  "approach": [-0.312, -0.218, -0.925],
  "q_current": [0.0, 0.168, 0.623, -0.454, 0.0, 0.0],
  "candidate_index": 1,
  "candidate_count": 3,
  "reject_reason": null,
  "execution_error": null,
  "updated_at": 1789090000.0,
  "pregrasp": {
    "status": "OK | FAIL | --",
    "position_error_m": 0.0004,
    "rotation_error_rad": 0.002,
    "total_error": 0.0021,
    "iterations": 17,
    "q": [0.0, 0.2, 0.6, -0.4, 0.0, 0.0]
  },
  "grasp": {
    "status": "OK | FAIL | --"
  }
}
```

Unavailable numeric values are JSON `null`, never NaN or Infinity.

## Coordinate semantics

- `camera_xyz`: final returned grasp TCP position transformed into camera frame.
- `base_xyz`: final returned grasp TCP position in robot base frame.
- `approach`: final base-frame unit approach vector.
- Camera coordinates are available for camera-mask grasps. Analytic or learned
  candidates may report `null` when no truthful camera-frame metadata exists.
- `q_current`: the exact seed supplied to `select_grasp()`. It may be the
  configured home pose by design and must not be relabelled as live arm state.

## IK error semantics

`total_error` remains the existing norm of Pinocchio SE(3) `log6`. It mixes
translation and rotation components and must not be displayed as millimetres.

For the exact configuration returned in `IKResult.q`:

- `position_error_m = ||current.translation - target.translation||`
- `rotation_error_rad = ||log3(current.rotation.T @ target.rotation)||`

The Robot RS runtime uses the SDK wrapper in `safety/harness.py`. It must
recompute FK after its safety-margin clamp, update all three errors against
the exact returned `q`, and re-apply the unchanged SDK tolerance to `success`.

## Candidate identity and lifecycle

Candidates are sorted once by the existing quality rule. Camera planning may
provide a tool-X 180-degree wrist equivalent. OBB provides both horizontal
axes, narrowest first, and two opposite approach directions per axis. Major-axis
candidates are retained, but any `width_m > max_width_m` candidate is rejected
before IK. The snapshot carries the stable one-based index and total count.

- On task start: replace stale state with `planning` and empty IK stages.
- Before solving a candidate: publish its geometry and identity.
- Pregrasp failure: `pregrasp=FAIL`, `grasp=--`.
- Grasp failure: `pregrasp=OK`, `grasp=FAIL`.
- Safety validation rejection: retain both IK results and publish the reason.
- Candidate selected: status becomes `executable`.
- All candidates rejected: retain the last rejected candidate and status
  becomes `failed`.
- Skill completion: status becomes `succeeded` or `failed`; failures after
  selection are stored in `execution_error`.

A complete JSON-safe dictionary is replaced under a lock. Readers receive a
deep copy, preventing mixed fields from concurrent planner and HTTP threads.
Up to 32 rejected or selected candidates are retained in `candidates`; the
Dashboard shows the latest eight and the complete snapshot is persisted as a
`grasp_debug` event in `trace.jsonl`.

## Dashboard rules

- Width is shown in millimetres and warned when `width_m > max_width_m`.
- Position error is shown in millimetres.
- Rotation error is shown in degrees.
- The error row displays the final grasp result when present, otherwise the
  pregrasp result.
- OK/succeeded is green, FAIL/failed is red, width overflow is amber.
- Reject and execution errors are displayed separately from IK numbers.

## Acceptance checks

1. Existing selection order and returned candidate are unchanged when no
   callback is supplied.
2. Callback exceptions cannot affect planning.
3. Width rejection is visible even though no IK call occurs.
4. Pregrasp and final-grasp failures show distinct stage values.
5. `json.dumps(grasp_debug, allow_nan=False)` succeeds.
6. `_INDEX_HTML.format(tiles="")` succeeds.
7. Dashboard does not invoke kinematics or safety code.
8. Existing camera, grasp, IK, and persistence tests remain the regression
   baseline; unrelated pre-existing failures are reported separately.
