# Architecture

## Design position

Three reference systems informed this design:

- **ASPIRE** (NVIDIA GEAR, 2026): a coding agent programs a robot through a
  *curated primitive API* whose every call records multimodal evidence
  (their ablation: the trace-exposing execution engine alone lifted success
  14% → 62%). We replicate the API surface (`get_observation`, `localize`,
  `plan_grasp`-equivalent, `solve_ik`, motion + gripper primitives), the
  per-call trace format (`trace.jsonl` + before/after keyframes), the
  debugging heuristics (gripper-width-after-close as the grasp-success
  signal), and the markdown skill library (store implemented in
  `skills/library.py`; the load-into-context loop is still on the ROADMAP).
- **Agentic-VLA** (ICML 2026): LLM sub-goal decomposition, a zero-shot VLM
  "exploration critic" that emits ONE spatial suggestion per consultation
  (we reuse their prompt near-verbatim), and embedding-indexed experience
  memory. Their loop is training-time; we repurposed the components into an
  inference-time execute→verify→recover loop.
- **Claude plays robotics** (Anthropic, 2026): control-interface level
  dominates model choice — LLMs commanding high-level primitives massively
  outperform LLMs near the metal; one LLM turn costs 2–15 s, so anything
  routine must not wait on the model; only the newest frames matter; and
  structured text state beats extra image context. This drove the
  livestreaming redesign below.

## Livestreaming + the reflex fast path (2026-07-18)

The demo is now an always-on system rather than a per-task pipeline,
modelled on how humans act: perception never stops, routine commands are
reflexes, and deliberation is reserved for the novel.

```
N cameras ──CameraStream (thread each, latest-frame slot, drop-stale)
   │            │
   │            ├── WorldWatcher (thread, ~3 Hz): detector + HSV color tag
   │            │     └─> BeliefStore (thread-safe): label+color+3D+freshness
   │            └── StreamServer (MJPEG dashboard): camera grid + narration
   │                  feed (episodic events) + object table   -> browser
   │
chat command ("pick and place pink object")
   ├─ tier 1 REFLEX   template grammar -> skill calls          (~µs)
   ├─ tier 2 HABIT    experience memory (hashed-BoW cosine)    (~ms)
   └─ tier 3 LLM      the original orchestrator loop           (2-15 s/turn)
        all tiers execute through the same safety-gated SkillRuntime
```

Key mechanisms:

- **Latest-slot streaming, never queues.** A consumer always gets the
  newest frame; nothing falls behind the sensor (`perception/stream.py`).
- **Warm world model.** The WorldWatcher fuses every rig camera with depth
  + extrinsics into the BeliefStore continuously, so command resolution is
  a dictionary lookup, not an observe→detect round trip. It pauses during
  arm motion (the held object must not be re-fused mid-air) and heartbeats
  the safety watchdog (`perception/world.py`).
- **Color without CLIP.** Detections are color-named from median mask HSV
  (`perception/colors.py`), stored on beliefs, and matched against color
  words in queries — "pink object" works on a closed-set COCO detector.
  When the ultralytics CLIP fork lands, open-vocab prompts slot in and the
  HSV tag becomes redundant metadata. The VLM can still be consulted (tier
  3 / advisor), but never on the hot path.
- **One-call pick-and-place.** `skill_pick_and_place` = resolve → grasp →
  place (named object or drop zone) → home, with stage timings in the
  result; both grasp and place stages keep retrying on fresh perception
  (re-home, re-scan, re-plan) until success or the persistence budget runs
  out (`grasp.persist_seconds: 120` / `grasp.max_pick_attempts: 8` in
  configs/demo.yaml). Hermes executes the whole command as a single MCP
  tool call; compute overhead is <100 ms and total time is arm motion time
  (`skills/runtime.py`).
- **LazyArm.** The MCP gateway pre-warms cameras/detector/world model at
  startup, but motors are not touched until the first motion command
  (`control/lazy_arm.py`) — starting a chat server must not power a robot.
- **Social skills.** wave / point_at / handover / sort_by_color /
  describe_scene / count_objects / move_relative give booth visitors a
  vocabulary beyond pick-and-place; all deterministic, all harness-gated.

We deliberately did NOT build a VLA-policy-in-the-loop baseline first: the
deterministic skill stack is debuggable, safety-gateable, and runs offline.
The LingBot-VLA-style websocket policy server slots in later as an
alternative *executor* behind the same skill API (see ROADMAP).

## Module map

```
src/wrc_demo/
├── types.py            Frame / Detection / ObjectFix / Grasp / RobotState
├── config.py           YAML profiles (cameras/, arms/, llm/) merged into one Cfg
 ├── perception/
│   ├── camera_base.py  CameraBase ABC + factory; Frames carry METRIC depth
│   ├── realsense_camera.py   D4xx + L515 (local fork's pyrealsense2)
│   ├── opencv_camera.py      RGB-only UVC
│   ├── orbbec_camera.py      Orbbec Gemini 2 (pyorbbecsdk, lazy import)
│   ├── mock_camera.py        synthetic tabletop + npz replay (wrc-record)
│   ├── depth_provider.py     sensor → mono plugin → table-plane ray-cast
│   ├── detector.py           open-vocab YOLOE/YOLO-World + MockDetector
│   ├── grounding.py          Extrinsics + localize_object (color/near-aware);
│   │                          multi-schema loader (new JSON / cuRobo NPZ / legacy NPZ)
│   ├── vlm_ground.py         second-chance VLM grounder (detector-miss path)
│   ├── colors.py             mask HSV → color name; color-query parsing
│   ├── stream.py             CameraStream (pump thread) + CameraRig (N cams)
│   └── world.py              WorldWatcher: always-on detection → beliefs
 ├── calibration/      top-camera hand-eye bring-up (see PLAN.md §4)
 │   ├── calib_dataset.py     CalibrationSample / Result dataclasses + JSON I/O
 │   ├── handeye_solver.py     joint SE(3) solve (LM + Huber + 3*MAD outliers)
 │   ├── aruco_session.py      cv2.aruco wrapper (detect + PnP + reprojection)
 │   └── frame_convention.py   SE(3) helpers + cuRobo/wrc_demo FK offset
├── memory/
│   ├── turboquant.py   TurboQuant-style rotation+scalar quantizer (numpy)
│   ├── vector_index.py asymmetric top-k cosine over 4-bit codes
│   ├── episodic.py     10-15 s ring: events, thumbnails (embeddings opt-in)
│   ├── beliefs.py      object permanence (visible/remembered states)
│   └── grasp_memory.py persisted per-object grasp-outcome prior
│                       (re-ranks candidates + nudges grasp z)
├── control/
│   ├── kinematics.py   Pinocchio FK/IK on the RS URDF (assets/), DLS+restarts
│   ├── arm_base.py     min-jerk streaming, feedback-based settling + make_arm
│   ├── mock_arm.py     kinematic sim + gripper object-stop emulation
│   ├── lazy_arm.py     defer motor bring-up until the first motion command
│   └── rebot_rs_arm.py real RS arm: motorbridge CAN, mechPos param reads,
│                       stall-aware two-stage gripper close
├── safety/harness.py   fail-closed gate for every waypoint + SafeArm wrapper
├── grasping/
│   ├── obb_grasp.py    base-frame OBB grasps (short-axis yaw, height frac)
│   ├── graspgenx_backend.py  learned 6-DoF grasps via ZMQ (default backend,
│   │                   silent OBB fallback when the server is down)
│   ├── selector.py     width + IK walk + harness pre-vet callback
│   │                   (dense sampling along the descent segment)
│   └── force.py        material → two-stage close profiles
├── agent/
│   ├── llm.py          OpenAI-compat (cloud + local Qwen) / Anthropic / Mock
│   ├── prompts.py      system persona, decomposition, advisor (paper prompt)
│   ├── advisor.py      failure-triggered VLM consultation
│   ├── reflex.py       tier-1 command grammar + tier-2 experience memory
│   ├── trace.py        ASPIRE trace logger
│   └── orchestrator.py reflex-first agent loop + TaskReport
├── skills/
│   ├── runtime.py      the curated tool surface + JSON schemas (21 specs:
│   │                   20 skills incl. pick_and_place + social skills,
│   │                   plus the loop-terminator task_done)
│   └── library.py      learned-skill markdown store (loading loop: ROADMAP)
└── apps/               demo.py CLI (build_runtime = the composition root),
                        record.py capture, viewer.py (wrc-view live RGB+D),
                        live_view.py (RigViewer window + draw helpers),
                        stream_server.py (MJPEG dashboard + narration +
                        POST /task), mcp_server.py (Hermes/Claude Code
                        front-end: skill runtime over MCP stdio,
                        perception pre-warm, lazy arm)
```

## Key decisions

**Metric depth in the Frame.** The baseline passed uint16 depth in sensor
units; the L515 (0.25 mm/unit) silently breaks any code written against
D4xx's 1 mm/unit. Converting at the camera boundary makes every downstream
consumer unit-safe.

**Grasps planned in the base frame, not the camera frame.** The baseline
derived approach direction from the camera ray, so grasp quality depended on
camera mounting. Here mask points are lifted to 3D, transformed to base, and
the OBB there gives yaw + width + height — the camera pose only affects
visibility, not grasp geometry. That is what "camera-agnostic" means
operationally.

**Learned grasps by default, analytic always available.** Since 2026-07-18
`grasp.backend: graspgenx` (configs/demo.yaml) sends the object's base-frame
cloud to a GraspGen-X ZMQ server (`scripts/serve_graspgenx.sh`, :5556,
~1.2 s for 100 samples on the GB10; reBot jaws passed as a swept volume for
cross-embodiment conditioning, `tip_offset_m` 0.098 empirical) and prepends
the ranked 6-DoF grasps to the analytic OBB candidates, which are always
computed. Any server error degrades silently to OBB — the booth must never
stall on a dead model server. The full pipeline:

```
localize ─▶ ObjectFix (base-frame OBB)
   ├─▶ GraspGen-X candidates (ZMQ, learned 6-DoF) ──┐ prepended; any server
   └─▶ OBB candidates (analytic, always computed) ──┤ error → OBB only
                                                    ▼
   grasp-outcome memory re-rank + z-nudge (~/.wrc_demo/grasp_memory.json)
                                                    ▼
   select_grasp: jaw-width filter ▸ IK (pregrasp, then grasp seeded from
   it) ▸ harness pre-vet (pregrasp WITHOUT the exemption cylinder, then 7
   samples along q_pre→q_grasp with the exemption floored at table_z−0.06)
                                                    ▼
   re-home (forces elbow-up IK branch) ▸ pregrasp ▸ exempted descent ▸
   two-stage stall-aware close ▸ lift ▸ air-grasp check (jaw width fraction)
```

**Own kinematics wrapper.** reBotArm_control_py's kinematics silently loads
the URDF named in its *global* config file, ignoring the hardware YAML you
pass (DM URDF loaded for the RS arm = wrong tool frame). We load the RS URDF
shipped in `assets/` explicitly. IK: damped least squares in the LOCAL frame
with joint-limit clamping and random restarts (matches the SDK's math, minus
the config hazard).

**Feedback, not sleep.** The baseline's motions were `sleep(duration + 0.6)`.
On the RS motors, motorbridge's `get_state()` never decodes the type-0x18
report frames, so live positions come from `mechPos` (0x7019) param reads —
verified on this rig. Settling is `max|q - q_target| < tol` with a timeout.

**Fail-closed safety.** The SDK enforces nothing outside its IK. Our harness
gates every waypoint of every streamed motion; grasp descents happen inside
an explicit exemption cylinder around the target so "don't touch the table"
and "grasp the object on the table" coexist. Auto-scaled durations keep
planned min-jerk peaks under the velocity cap; the harness remains the
backstop.

**Memory is structured first, embeddings second.** The agent's recall tools
work on labels/time/positions (BeliefStore) — deterministic and testable.
The TurboQuant index (dim-agnostic, 4-bit codes, asymmetric search) has one
live consumer today: the tier-2 ExperienceMemory (hashed bag-of-words habit
recall). Episodic-memory embeddings are opt-in via `embed_dim` and stay OFF
in the shipped demo (`recall_similar` returns `[]`) until a visual embedder
is added (ROADMAP). A third store, GraspOutcomeMemory
(`memory/grasp_memory.py`), persists dimensionless per-object grasp features
across sessions and re-ranks planner candidates on the grasp hot path. The
quantizer is pure numpy so the demo carries no exotic dependency (pip
`turbovec` can slot in behind the same interface).

**Gripper force ≈ commanded stiffness.** The RS gripper is one MIT-mode
motor; commanded kp bounds stall torque, so material profiles scale effort
and close depth, and stall detection (mechVel) doubles as the grasp-success
signal. Current-loop force calibration is an onsite task (see rebot_rs.yaml
warnings).

## Verification status (2026-07-20, this rig)

- 139 unit/integration tests collected, 137 selected by default (`pytest
  -q`; 2 hardware-marked), including a full mock-stack grasp-and-place e2e
  that exercises config → perception → beliefs → grasp planning → IK →
  safety-gated streaming → gripper verification → memory → trace files.
- Two adversarial multi-lens review passes, all critical/major findings
  fixed: 2026-07-16 (4 reviewers × skeptic verification, 34 agents, 29
  confirmed defects) and 2026-07-18 on the livestreaming redesign (45-agent
  workflow, 33 confirmed findings). Regression-pinned in
  `tests/test_review_regressions.py`: inverted spatial hints, baseline
  hand-eye npz loading, watchdog tripping mid-grasp, below-clearance
  recovery deadlock, OpenAI/Anthropic tool-protocol violations, unbounded
  image-context growth (air-grasp detection is pinned in
  `tests/test_orchestrator_e2e.py`); v2 findings in
  `tests/test_review_regressions_v2.py`. Other confirmed v1 fixes
  (IK-vs-harness margin mismatch, stale-CAN-feedback masking, free-fall on
  soft stop) are real-arm behaviors enforced in code but not
  regression-pinned.
- Live L515 streaming through the full camera stack (hardware-marked test).
- Live RobStride mechPos param reads for all 7 motors over can_arm (read-only).
- Live YOLO inference (CUDA, GB10) on L515 frames with metric depth lookup.
- Isaac Sim backend removed 2026-08-28 (the upstream bridge was never
  productionized; no replacement in tree).
- GraspGen-X backend integrated with first-light verification in sim;
  tip-offset / sweep-volume calibration still open (see ROADMAP).
- NOT yet exercised: real-arm motion (needs onsite gripper re-verification
  and hand-eye calibration), local Qwen serving (scripts provided; note the
  `local_qwen.yaml` was deleted 2026-08-27 -- see CLAUDE.md.)
- Orbbec Gemini 2 backend (`perception/orbbec_camera.py`, lazy `pyorbbecsdk`
  import) + hand-eye calibration pipeline (`calibration/` module +
  `scripts/calib_top_orbbec.py` CLI) shipped but NOT yet exercised against
  real hardware. The CLI collects calibration samples using FK from
  `wrc_demo.control.kinematics.Kinematics` (Pinocchio on the RS URDF with
  `joint_signs=[-1]*6` and `ee_frame='gripper_end'`) and ArUco PnP against
  the SDK intrinsics, then runs a joint SE(3) LM fit with Huber loss and
  3*MAD outlier rejection. Calibration product is
  `data/calibration/hand_eye_top.json` (+ `.npz` with K, D); binding is
  `data/calibration/cameras.json`. The build_runtime path's
  `requires_calibration` guard fails closed (fuse=False) when the source
  file is absent — see `apps/demo.py:build_runtime` and the
  `orbbec_overhead` / `orbbec_wrist` profiles in `configs/cameras/`.

## Known limitations (accepted for the baseline)

- BeliefStore merges same-label observations within 8 cm (EMA) — two
  identical objects closer than that can blur into one belief.
- `RebotRSArm.disconnect()` goes through the SDK's disable_all: park the arm
  (move_home) before shutting down or it will fall under gravity.
- Grip force is a stiffness proxy (MIT kp scaling + stall detection), not a
  calibrated force loop; current-based calibration is an onsite task.
- The advisor/decompose prompts are single-frame; no video context yet.
- The MCP server is single-threaded: a long `pick_and_place` blocks
  `emergency_stop` and every other tool until it returns (SIGINT e-stop
  works; a bypass stop channel is on the ROADMAP).
