# Roadmap

## Near term (before the demo)

- **Persistence-loop review leftovers (2026-07-18, adversarial review run;
  fixed same-day: e-stop break, fail-fast on never-seen objects, place
  release/ascent desync, frozen belief epoch, descent-path vetting,
  place-stage re-home).** Still open, in priority order:
  1. MCP server is single-threaded: a 150 s pick_and_place blocks
     emergency_stop and every other tool -- needs a stop channel that
     bypasses the request loop (the web runner has the same gap; SIGINT
     estop works and the skill loops now honor it).
  2. Exception between gripper close and held_object assignment leaves a
     physically held object logically unheld (reconcile only clears the
     opposite desync); consider a provisional held marker before close.
  3. Budget can multiply across tiers: fast-path burns persist_seconds,
     then a real-LLM tier can call pick_and_place again. Cap per task.
  4. handover / sort_by_color still single-attempt (inconsistent with
     pick_and_place persistence).
  5. _reconcile_held mistakes a legitimately-held VERY thin object
     (<4% jaw span ~ 3.6 mm) for a slip; booth objects are chunky.
  6. Fail fast when every grasp candidate exceeds jaw width (currently
     retries perception on an object-property error).
  7. Test-coverage gaps flagged: place-stage loop, deadline expiry,
     epoch fallback, z-clamp, exemption z_min through _in_cylinder,
     McpClient timeout is dead code.

- **Newton upstream issue (2026-07-19).** Manipulation contacts are broken
  at the PARSER level on the 6.0 develop build: identical failure under
  mjcwarp default, `use_mujoco_contacts=true` and XPBD (constant +3.7 cm
  float even box-vs-box, fingers pass through objects, boot NaN, "Triangle
  pair buffer overflowed"). Repro = `scripts/physics_probe.py --engine
  newton`. File against isaac-sim/IsaacSim with the probe reports in
  /tmp/probe_newton_*.json. Demo manipulation stays on PhysX (full battery
  green) until fixed.
- **Wrist cam follow-ups.** Validate the eye-in-hand extrinsics during a
  real grasp (reproject wrist depth of the target object against the
  physics-truth pose mid-descent); consider serving the wrist stream a
  narration highlight ("what the gripper sees") on the dashboard; on the
  real rig map `isaac_wrist.yaml` to the physical D435i + hand-eye calib.
- **Sim perception flakiness** (separate campaign): YOLOE misses the YCB
  banana on some boots and label-flickers the soup can (bottle/toy);
  belief 3D positions themselves verified ±3 mm against physics truth.

- **Persistent spatial memory (Johnny's idea, 2026-07-18).** BeliefStore
  already gives in-session object permanence (visible→remembered, EMA
  fusion, grasp-from-memory fallback); add save/load (JSON with wall-clock
  timestamps, loaded as "remembered") so the world model survives restarts,
  plus action↔object consolidation on top of ExperienceMemory. "Even if I
  don't see it, I roughly know where it is — like a human."
- **Straight-up spawn on the local tuned Isaac asset.** Blocked: drive
  travel from q=0 sweeps the props; joint-state authoring and tensor
  teleports NaN the solver on this asset (custom fixed-joint stack).
  Upstream asset gets it properly via Seeed-Projects/reBot-Isaacsim#9.
  Investigate the -plus asset's root joint / articulation root config.

- **GraspGen-X backend (integrated 2026-07-18, first-light verified).**
  `grasp.backend: graspgenx` sends the fix's base-frame object cloud to the
  GraspGen-X ZMQ server (`scripts/serve_graspgenx.sh`, own venv
  `~/Projects/demo/.graspgenx`, checkpoints in `GraspGenX/ext/`) and gets
  ranked 6-DoF grasps back (~1.2 s for 100 samples on the GB10); OBB stays
  as automatic fallback and additional IK candidates. TODO: (1) calibrate
  `tip_offset_m` in Isaac Sim (gripper-base -> reBot jaw center), (2) refine
  the URDF-derived reBot sweep-volume params in `configs/demo.yaml` with an
  Isaac Sim measurement (franka_panda remains only the no-sweep fallback),
  (3) use `infer_scene_pc` for collision-aware grasps in clutter.

- **Onsite bring-up checklist**
  1. `sudo ip link set can_arm up type can bitrate 1000000`; kill any
     motorbridge-gateway/Studio.
  2. Re-verify RS gripper travel + stall torque; update
     `configs/arms/rebot_rs.yaml` (open/closed rad, kp, max_width_m).
  3. Hand-eye calibration: run the baseline repo's `collect_handeye_eih.py`
     (eye-in-hand) or measure the static mount, point the camera profile's
     `extrinsics` at it.
  4. Fit the table plane once (`DepthProvider.fit_table_plane`) and set
     `table_z` / workspace AABB for the physical setup.
  5. `pytest -m hardware` (read-only), then first motions with
     `--arm rebot_rs` at low `max_joint_vel`.
- **Local LLM**: the `local_qwen` profile was deleted (2026-08-27) because it was never wired — its model name had no matching serve script. Production path is `local_qwen3_vl` (Qwen3-VL-2B-Instruct AWQ-4bit, served by `scripts/start_cosmos3_server.sh`). If a 30B-A3B Q4_K_M path is needed later, write a fresh `serve_qwen_llamacpp.sh` defaulting to that GGUF.
- **Visual embedder for memory**: plug a CLIP/SigLIP image encoder into
  `EpisodicMemory(embed_dim=...)` + crops per detection, enabling
  "the thing that looked like X" recall through the TurboQuant index.

## Mid term

- **VLA policy backend**: LingBot-VLA-v2 exposes a websocket policy server
  (msgpack-numpy, `infer(obs) -> action chunks`). Add a `VLAExecutor` behind
  the skill API so `grasp_object` can be served either by the deterministic
  OBB pipeline or by a language-conditioned policy; the agent layer stays
  unchanged. Spark caveats: flash-attn must build for aarch64+Blackwell or
  the two hardcoded attention impls patched to SDPA.
- **GraspNet-class 6-DoF grasps** — ✅ superseded by the GraspGen-X backend
  (integrated 2026-07-18, see near term). The baseline's graspnet path
  (vendored sdk + checkpoint-rs.tar + THC-era CUDA patches) is no longer
  worth pursuing; remaining learned-grasp work (tip-offset calibration,
  reBot sweep params, collision-aware `infer_scene_pc`) is tracked in the
  near-term GraspGen-X item.
- **Skill-library growth loop** (ASPIRE): after each failed→repaired run,
  distill the fix into `skills_library/*.md` (schema already implemented);
  load `relevant(task)` entries into the agent context.

## Long term: sim2real with NuRec / Isaac

- **Isaac Sim bridge (scaffolded 2026-07-18; live on PhysX since
  2026-07-19).** `scripts/isaac_bridge.py` (runs inside Isaac Sim's Python)
  serves RGB-D frames + articulation control over newline-JSON TCP;
  `--cameras isaac --arm isaac` runs the identical demo against the sim.
  Protocol + client backends are covered by fake-server tests, and the live
  path has run scripted picks (`dashboard_runner.py`, `night_runner.sh`)
  with the `physics_probe.py` battery green on PhysX — joint order/signs,
  gripper fraction mapping and camera extrinsics are wired in
  `configs/{arms,cameras}/isaac*.yaml`. Still open: wrist-cam extrinsics
  validation during a real grasp (near term), Newton engine (blocked
  upstream, near term), and the straight-up-spawn asset issue.
  The Downloads/isaac-companion-v1-franka pack's `isaac-sim-remote` skill
  (TCP Python-exec extension) is a good live-debugging companion for this.
- `reBot-Isaacsim` + `sim2real-rebot-devarm` already provide USD assets, a
  real→sim UDP mirror, and an HTTP control daemon for this arm.
- **NuRec (neural reconstruction)**: reconstruct the actual demo tabletop
  into a photoreal digital twin; rehearse perception + grasping against the
  twin (domain gap ≈ 0 for the camera), then replay on the real rig. The
  ASPIRE sim2real recipe applies directly: skills discovered in sim transfer
  as *in-context guidance* for the real-robot agent, not as weights.
- **Online adaptation** (Agentic-VLA proper): once a VLA executor exists,
  their GRPO + reward-synthesis loop is the path to improving it from demo
  logs; the trace format already captures per-primitive evidence needed for
  progress rewards.
