"""The curated skill API the agent programs against (ASPIRE Appendix E.2
adapted to this rig). Every call is safety-gated, trace-logged with
before/after keyframes, and mirrored into episodic memory.

Skills return plain dicts with an `ok` flag; SkillError and SafetyViolation
become {"ok": false, "error": ...} so the agent can reason about failures
instead of crashing the loop.
"""

from __future__ import annotations

import copy
import re
import threading
import time

import cv2
import numpy as np

from ..agent.trace import TraceLogger
from ..config import Cfg
from ..grasping import plan_grasps_from_fix, select_grasp, select_profile
from ..grasping.camera_grasp import (
    DEFAULT_INSERTION_DEPTH_M, mask_obb_corners, obb_short_axis_base,
    plan_grasp_from_mask,
)
from ..memory import BeliefStore, EpisodicMemory
from ..perception.colors import detection_color, parse_color_query
from ..perception.object_names import canonical_destination, canonical_object_name
from ..perception.grounding import (
    Extrinsics,
    localize_object,
    mask_to_points_cam,
    oriented_bbox,
)
from ..types import Detection, Frame, SafetyViolation, SkillError, make_transform, transform_points

#: skills that move the arm: the WorldWatcher is held while they run so the
#: held/handled object is not re-fused at a bogus mid-air position.
#:
#: v5 §P1.4 fix: ``move_to_pose`` (cartesian waypoint trajectory) and
#: ``move_to_joint_zero`` (raw ``arm.move_joints(zeros, ...)``) both move
#: the arm but were missing from this set. LLM calls to either would
#: skip the watcher pause, causing mid-air held objects to be
#: re-fused at bogus positions.
_MOTION_SKILLS = {
    "grasp_object", "place_at", "place_on_object", "push_object",
    "open_gripper", "close_gripper", "move_home", "pick_and_place",
    "point_at", "wave", "handover", "sort_by_color", "move_relative",
    "throw",
    "hand_action",
    "move_to_pose",
    "move_to_joint_zero",
    "teach_record",
    "teach_replay",
}


def _jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _strict_grasp_topdown_rotation(rotation: np.ndarray) -> np.ndarray:
    """Keep the calibrated three-finger closing yaw and make approach -Z."""
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    down = np.array([0.0, 0.0, -1.0])
    opening = R[:, 1].copy()
    opening[2] = 0.0
    if float(np.linalg.norm(opening)) < 1e-9:
        # A vertical opening axis has no usable yaw.  Recover it from the
        # candidate's third axis before falling back to base +X.
        opening = R[:, 2].copy()
        opening[2] = 0.0
    if float(np.linalg.norm(opening)) < 1e-9:
        opening = np.array([1.0, 0.0, 0.0])
    opening /= float(np.linalg.norm(opening))
    return np.column_stack([down, opening, np.cross(down, opening)])


def _apply_strict_grasp_topdown(grasp, metadata: dict) -> None:
    """Convert candidate orientation and its depth target as one operation."""
    original = np.asarray(grasp.approach, dtype=float).copy()
    grasp.rotation = _strict_grasp_topdown_rotation(grasp.rotation)
    grasp.approach = grasp.rotation[:, 0].copy()
    target_z = metadata.get("grasp_target_z")
    if target_z is not None:
        position = np.asarray(grasp.position, dtype=float).copy()
        metadata["pre_topdown_position_z"] = float(position[2])
        metadata["strict_topdown_z_correction_m"] = (
            float(target_z) - float(position[2])
        )
        position[2] = float(target_z)
        grasp.position = position
    metadata["original_approach"] = original.tolist()
    metadata["tilt_deg"] = 0.0
    metadata["direction_rank"] = 0
    metadata["strict_grasp_topdown"] = True


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_horizontal_axis(angle_rad: float, axis_azimuth_rad: float) -> np.ndarray:
    """Rotate about a world-horizontal axis selected by its XY azimuth."""
    axis = np.array([
        np.cos(axis_azimuth_rad), np.sin(axis_azimuth_rad), 0.0,
    ])
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


def _bump(counts: dict, key: str, err: float | None = None) -> None:
    """Reject-reason histogram bump; the last IK error rides along for the
    trace event (bucket by reason type, not by float value)."""
    counts[key] = counts.get(key, 0) + 1
    if err is not None:
        counts[f"{key}: last err"] = round(float(err), 3)


def _yaw_shimmed(grasp, deg: float):
    """Copy of ``grasp`` with its jaw/closing yaw rotated ``deg`` about the
    approach axis (tool X) and quality dinged so it sorts right after its
    parent.

    Why this exists (2026-09-22, run 124241): at lateral positions
    (TCP y ≈ +0.23 m) the wrist-aligned IK branch for the camera-planner
    yaw runs into the joint4 elbow limit, and the surviving branch needs a
    wrist roll beyond ``max_wrist_delta_rad`` — every exact-yaw candidate
    is rejected even though a −15° yaw shim solves cleanly at half the
    wrist travel. A power grasp of produce tolerates ±30° of jaw yaw, so
    shims trade a little grasp yaw fidelity for executability. They only
    cost IK time when every exact-yaw candidate has already failed
    (``select_grasp`` returns on the first executable candidate).
    """
    import copy as _copy

    g2 = _copy.copy(grasp)
    R = np.asarray(grasp.rotation, dtype=float)
    g2.rotation = R @ _rot_x(np.radians(float(deg)))
    g2.approach = np.asarray(grasp.approach, dtype=float).copy()
    g2.quality = float(grasp.quality) * (1.0 - 0.05 * abs(float(deg)) / 30.0)
    return g2


def _place_rotation_candidates(
    R_level, yaw_degs, tilt_degs, max_tilt_rad, long_tool=None,
    tilt_axis_degs=(0.0, 90.0),
):
    """Object-level place orientations: yaw about world Z is free, the
    held object's up axis stays within ``max_tilt_rad`` of world +Z.

    Candidates are ``R = Rtilt(world XY axis) @ Rz(yaw) @ R_level`` where
    yaw and tilt direction are independent. ``R_level``
    is the TCP rotation captured at grasp close (the object still rests
    level on the table then). With the object up axis ``u = R_level.T @ z``
    the up axis under a candidate is ``Rtilt @ z``: its tilt off +Z equals
    ``|tilt|`` for every yaw. The cosine check below is an invariant
    guard, not what makes it true. The identity candidate
    (yaw=0, tilt=0) must stay
    FIRST in the list: callers keep the first candidate on score ties,
    which pins unit-test stubs to the carried rotation.

    ``long_tool`` (the held object's long axis in the TOOL frame, for
    elongated fruit) additionally requires ``|(R @ long_tool).z| <
    sin(max_tilt_rad)`` so a banana never ends up nose-down.
    """
    R_level = np.asarray(R_level, dtype=float).reshape(3, 3)
    u_tool = R_level.T @ np.array([0.0, 0.0, 1.0])
    cos_max = float(np.cos(max_tilt_rad))
    sin_max = float(np.sin(max_tilt_rad))
    candidates = []
    for yaw_deg in yaw_degs:
        Rz = _rot_z(np.radians(float(yaw_deg)))
        for tilt_deg in tilt_degs:
            # Zero tilt has no axis; emit it once to avoid duplicate IK work.
            axes = (0.0,) if abs(float(tilt_deg)) < 1e-12 else tilt_axis_degs
            for tilt_axis_deg in axes:
                Rt = _rot_horizontal_axis(
                    np.radians(float(tilt_deg)),
                    np.radians(float(tilt_axis_deg)),
                )
                R = Rt @ Rz @ R_level
                up = R @ u_tool
                if float(up[2]) < cos_max - 1e-9:
                    continue
                if long_tool is not None and abs(
                    float((R @ np.asarray(long_tool, dtype=float))[2])
                ) >= sin_max:
                    continue
                candidates.append((
                    R, float(yaw_deg), float(tilt_deg), float(tilt_axis_deg),
                ))
    return candidates


def _is_mock(obj: object) -> bool:
    """True if ``obj`` is a ``unittest.mock.Mock`` / ``MagicMock`` instance.

    Duck-typed via the private ``_mock_methods`` attribute that all Mock
    subclasses carry. Used by ``_safe_bridge_move`` to skip its post-move
    FK verify in unit tests where the bridge and arm are Mocks and don't
    simulate actual motion -- the FK comparison would always report the
    arm parked at the initial state and spuriously fail every call.

    Production code paths instantiate ``RebotGraspBridge`` / ``SafeArm``
    / ``MockArm``, none of which carry ``_mock_methods``, so the verify
    runs normally there.
    """
    return hasattr(obj, "_mock_methods")


class SkillRuntime:
    def __init__(
        self,
        camera,
        depth_provider,
        detector,
        extrinsics: Extrinsics,
        kin,
        safe_arm,
        memory: EpisodicMemory,
        beliefs: BeliefStore,
        trace: TraceLogger,
        cfg,
    ):
        self.camera = camera
        self.depth = depth_provider
        self.detector = detector
        self.extrinsics = extrinsics
        self.kin = kin
        self.arm = safe_arm
        self.memory = memory
        self.beliefs = beliefs
        self.trace = trace
        self.cfg = cfg
        self.last_frame: Frame | None = None
        self.held_object: str | None = None
        self._held_det_label: str | None = None
        self._held_color: str | None = None
        # Successful-grasp TCP height above its supporting surface.
        self._held_support_offset_m: float | None = None
        # Half extents of the held cloud along base X/Y, captured before
        # removing it from the world model. Used for collision-free relative
        # placement distances instead of a fixed centre-to-centre offset.
        self._held_xy_half_extent_m: np.ndarray | None = None
        # Level reference for held-object place: the TCP rotation at grasp
        # close (object still resting level on the table). The object up
        # axis in the TOOL frame is R_level.T @ z and stays fixed while
        # held; place orientations are constrained via it, never by
        # freezing wrist joints.
        self._held_R_level: np.ndarray | None = None
        # Held object's long axis in the TOOL frame (elongated fruit only).
        self._held_long_tool: np.ndarray | None = None
        # Tactile confirmation is not perfectly continuous while an object
        # settles or the CAN pressure read races a motion. A verified grasp
        # therefore gets a grace window and must fail several independent
        # checks before software is allowed to forget it.
        self._held_verified_t: float | None = None
        self._held_no_contact_checks: int = 0
        #: optional WorldWatcher (set by the app wiring); paused during motion
        self.watcher = None
        #: optional rebot_grasp motion/gripper bridge (set by make_arm for
        #: ``rebot_rs``; None for mock). Skill methods read this to
        #: decide between the new set.py-aligned flow and the legacy
        #: SafeArm.move_joints pipeline.
        self.bridge = getattr(safe_arm, "grasp_bridge", None)
        #: natural-language task currently executing (dashboard narration)
        self.current_task: str | None = None
        #: monotonic time the current top-level MOTION skill started; while
        #: the arm moves the WorldWatcher is paused, so belief ages measured
        #: from "now" are artificially inflated -- staleness checks measure
        #: from this epoch instead.
        self._motion_t0: float | None = None
        # Absolute budget shared by pick_and_place, remote planning and the
        # per-candidate IK walk.
        self._active_pick_deadline: float | None = None
        #: last time _reobserve completed a fresh scan DURING the current
        #: motion skill: belief-fallback staleness must advance with it, or
        #: a belief that was fresh at task start stays "fresh" through 8
        #: retries even after every re-scan failed to see the object.
        self._last_reobserve_t: float | None = None
        self._graspgenx = None  # lazy GraspGenXPlanner (grasp.backend)
        self._grounder = None  # lazy VLMGrounder (cfg "grounder", 2nd filter)
        # Dashboard-only grasp/IK diagnostics. Writers replace a complete,
        # JSON-safe snapshot under the lock; the UI never runs IK itself.
        self._grasp_debug_lock = threading.Lock()
        self._grasp_debug: dict = {}
        self._grasp_debug_history: list[dict] = []
        self._grasp_debug_serial = 0
        self._grasp_call_serial = 0
        self._grasp_candidate_meta: dict[int, dict] = {}
        self._failed_grasp_signatures: list[dict] = []
        # Fake-RL grasp memory (RPent/Harness-VLA pattern): learns which grasp
        # geometry works per object profile from past attempts (wins AND
        # failures), persisted so accuracy improves across sessions.
        from ..memory.grasp_memory import GraspOutcomeMemory
        from pathlib import Path as _Path
        _gm_path = cfg.grasp.get("memory_path", "~/.wrc_demo/grasp_memory.json")
        self.grasp_memory = GraspOutcomeMemory(
            path=_Path(str(_gm_path)).expanduser())
        g = cfg.arm.gripper
        self._grip_open = float(g.get("open_pos", 0.0))
        self._grip_closed = float(g.get("closed_pos", 1.0))
        self._max_width = float(g.get("max_width_m", 0.09))
        self._default_classes = list(cfg.get("detect_classes", ["cup", "bottle", "box", "fruit", "toy"]))
        self._cartesian_planner = None  # lazy CartesianPlanner (planning module)

    # ── plumbing ─────────────────────────────────────────────────────────

    @property
    def cartesian_planner(self):
        """Lazy CartesianPlanner; None if wrc_demo.planning unavailable.

        The vendored SDK is already on sys.path (wrc_demo.__init__),
        so we just import the planner directly.
        """
        if self._cartesian_planner is None:
            try:
                from ..planning.cartesian_planner import CartesianPlanner
                self._cartesian_planner = CartesianPlanner(self.kin)
            except (RuntimeError, ImportError):
                return None
        return self._cartesian_planner

    def grasp_debug_snapshot(self) -> dict:
        """Thread-safe JSON-ready snapshot consumed by GET /state."""
        with self._grasp_debug_lock:
            state = copy.deepcopy(self._grasp_debug)
            state["candidates"] = copy.deepcopy(self._grasp_debug_history)
            return state

    def _begin_grasp_debug(self, target: str | None) -> None:
        with self._grasp_debug_lock:
            self._grasp_call_serial = getattr(self, "_grasp_call_serial", 0) + 1
            self._grasp_candidate_meta = {}
            self._grasp_debug_history = []
            self._grasp_debug_serial = 0
            self._grasp_debug = {
                "status": "planning",
                "target": target or "--",
                "updated_at": time.time(),
                "candidate_index": 0,
                "candidate_count": 0,
                "candidate_sequence": 0,
                "grasp_call_index": self._grasp_call_serial,
                "pick_attempt": 1,
                "pick_attempt_max": 1,
                "pregrasp": {"status": "--"},
                "grasp": {"status": "--"},
            }

    def _set_grasp_debug_attempt(self, attempt: int, maximum: int) -> None:
        with self._grasp_debug_lock:
            self._grasp_debug["pick_attempt"] = int(attempt)
            self._grasp_debug["pick_attempt_max"] = int(maximum)
            self._grasp_debug["updated_at"] = time.time()

    def _remember_grasp_metadata(self, grasp, metadata: dict) -> None:
        with self._grasp_debug_lock:
            self._grasp_candidate_meta.setdefault(id(grasp), dict(metadata))


    def _serialize_ik(self, result) -> dict:
        if result is None:
            return {"status": "--"}

        def finite(value):
            value = float(value)
            return value if np.isfinite(value) else None

        return {
            "status": "OK" if bool(result.success) else "FAIL",
            "position_error_m": finite(
                getattr(result, "position_error_m", np.inf)
            ),
            "rotation_error_rad": finite(
                getattr(result, "rotation_error_rad", np.inf)
            ),
            "total_error": finite(result.error),
            "iterations": int(result.iterations),
            "q": np.asarray(result.q, dtype=float).round(6).tolist(),
        }

    def _on_grasp_debug(self, event: dict) -> None:
        """Convert selector events into one atomic Dashboard snapshot."""
        if event.get("stage") == "exhausted":
            with self._grasp_debug_lock:
                state = copy.deepcopy(self._grasp_debug)
                state.update({
                    "status": "failed",
                    "reject_reason": event.get("reject_reason"),
                    "updated_at": time.time(),
                })
                self._grasp_debug = state
            return

        grasp = event["candidate"]
        with self._grasp_debug_lock:
            metadata = dict(self._grasp_candidate_meta.get(id(grasp), {}))
            old_target = self._grasp_debug.get("target")
            pick_attempt = int(self._grasp_debug.get("pick_attempt", 1))
            pick_attempt_max = int(self._grasp_debug.get("pick_attempt_max", 1))
            grasp_call_index = int(self._grasp_debug.get("grasp_call_index", 1))
        confidence = metadata.get("confidence")
        if confidence is None:
            confidence = float(getattr(grasp, "quality", 0.0))
        camera_xyz = metadata.get("camera_xyz")
        state = {
            "status": (
                "executable" if event.get("stage") == "selected" else "planning"
            ),
            "stage": event.get("stage"),
            "target": str(getattr(grasp, "label", "") or old_target or "--"),
            "source": metadata.get("source", "analytic_obb"),
            "variant": metadata.get("wrist_variant"),
            "branch": metadata.get("branch"),
            "tilt_deg": metadata.get("tilt_deg"),
            "grasp_score": metadata.get("grasp_score"),
            "confidence": float(confidence),
            "width_m": float(grasp.width_m),
            "max_width_m": float(event["max_width_m"]),
            "camera_xyz": (
                np.asarray(camera_xyz, dtype=float).round(6).tolist()
                if camera_xyz is not None else None
            ),
            "base_xyz": np.asarray(grasp.position, dtype=float).round(6).tolist(),
            "raw_origin_xyz": metadata.get("raw_origin_xyz"),
            "tip_offset_m": metadata.get("tip_offset_m"),
            "tcp_standoff_m": metadata.get("tcp_standoff_m"),
            "insertion_depth_m": metadata.get("insertion_depth_m"),
            "depth_correction_m": metadata.get("depth_correction_m"),
            "grasp_target_z": metadata.get("grasp_target_z"),
            "cloud_top_z": metadata.get("cloud_top_z"),
            "target_depth_m": metadata.get("target_depth_m"),
            "horizontal_spans_m": metadata.get("horizontal_spans_m"),
            "horizontal_aspect": metadata.get("horizontal_aspect"),
            "elongated_short_axis_hint": metadata.get(
                "elongated_short_axis_hint"
            ),
            "elongated_axis_hint_source": metadata.get(
                "elongated_axis_hint_source"
            ),
            "elongated_axis_alignment": metadata.get(
                "elongated_axis_alignment"
            ),
            "local_geometry": metadata.get("local_geometry"),
            "isotropic": metadata.get("isotropic"),
            "symmetry_flipped": bool(getattr(grasp, "_symmetry_flipped", False)),
            "approach": np.asarray(grasp.approach, dtype=float).round(6).tolist(),
            "jaw_opening_axis": (
                np.asarray(grasp.rotation, dtype=float)[:, 1]
                .round(6).tolist()
            ),
            "q_current": np.asarray(event["q_current"], dtype=float).round(6).tolist(),
            "candidate_index": int(event["candidate_index"]),
            "candidate_count": int(event["candidate_count"]),
            "pick_attempt": pick_attempt,
            "pick_attempt_max": pick_attempt_max,
            "grasp_call_index": grasp_call_index,
            "reject_reason": event.get("reject_reason"),
            "pregrasp": self._serialize_ik(event.get("pregrasp")),
            "grasp": self._serialize_ik(event.get("grasp")),
            "updated_at": time.time(),
        }
        with self._grasp_debug_lock:
            if event.get("stage") in {"rejected", "selected"}:
                self._grasp_debug_serial += 1
            state["candidate_sequence"] = self._grasp_debug_serial
            self._grasp_debug = state
            if event.get("stage") in {"rejected", "selected"}:
                history_entry = copy.deepcopy(state)
                history_entry["sequence"] = self._grasp_debug_serial
                self._grasp_debug_history.append(history_entry)
                self._grasp_debug_history = self._grasp_debug_history[-32:]

    def _finish_grasp_debug(self, ok: bool, error: str | None = None) -> None:
        with self._grasp_debug_lock:
            if not self._grasp_debug:
                return
            state = copy.deepcopy(self._grasp_debug)
            state["status"] = "succeeded" if ok else "failed"
            if error:
                state["execution_error"] = error
            state["updated_at"] = time.time()
            self._grasp_debug = state

    def _show_status(self, text: str) -> None:
        """Push agent state to EVERY live view (all rig cameras show the
        same agent, not just the primary). self.camera goes first: it may
        be a FrameHub WRAPPING the primary stream."""
        streams = [self.camera]
        rig = getattr(self, "rig", None)
        if rig is not None:
            streams += [s for s in getattr(rig, "streams", []) if s is not self.camera]
        for s in streams:
            if hasattr(s, "set_overlay"):
                s.set_overlay(status=text)

    def _show_detections(self, dets) -> None:
        if hasattr(self.camera, "set_overlay"):
            self.camera.set_overlay(detections=dets)

    def execute(self, name: str, args: dict) -> dict:
        """Dispatch one skill call with tracing. Never raises."""
        args = dict(args)
        if name in {"grasp_object", "pick_and_place"}:
            label = str(args.get("object") or args.get("label") or "")
            requested = args.get("material")
            effective = select_profile(label, requested).name
            if requested is not None and effective != str(requested).lower():
                args["material"] = effective
                self.trace.record_event(
                    "material_profile_override",
                    {
                        "label": label,
                        "requested": requested,
                        "effective": effective,
                        "reason": "hardware safety profile takes precedence",
                    },
                )
        fn = getattr(self, f"skill_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown skill {name!r}"}
        self._show_status(f"{name}({_short(args)})")
        if name in {"grasp_object", "pick_and_place"}:
            self._begin_grasp_debug(
                str(args.get("object") or args.get("label") or "--")
            )
        if name in _MOTION_SKILLS:
            self._refresh_keyframe_frame(name, "before")
        before = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_before"
        )
        t0 = time.monotonic()
        try:
            import contextlib

            hold = (
                self.watcher.paused()
                if (self.watcher is not None and name in _MOTION_SKILLS)
                else contextlib.nullcontext()
            )
            owns_epoch = name in _MOTION_SKILLS and self._motion_t0 is None
            if owns_epoch:
                self._motion_t0 = time.monotonic()
            with hold:
                try:
                    result = fn(**args)
                finally:
                    if name == "pick_and_place":
                        self._active_pick_deadline = None
                    if owns_epoch:
                        self._motion_t0 = None
                        self._last_reobserve_t = None
                    # Sync the held-object ignore BEFORE fusion resumes, or
                    # one tick could register the object dangling mid-air.
                    if self.watcher is not None:
                        self.watcher.ignore_label(
                            self._held_det_label if self.held_object else None
                        )
            if "ok" not in result:
                result["ok"] = True
        except (SkillError, SafetyViolation) as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except TypeError as e:
            result = {"ok": False, "error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # camera dropouts, CAN loss, ... : the agent
            # loop must survive and report, not crash mid-task.
            result = {
                "ok": False,
                "error": f"unexpected {type(e).__name__}: {e}",
                "hint": "hardware/runtime fault; consider task_done(success=false)",
            }
            record_exception = getattr(self.trace, "record_exception", None)
            if callable(record_exception):
                record_exception(f"skill {name}({_short(args)})", e)
        if name in {"grasp_object", "pick_and_place"}:
            self._finish_grasp_debug(
                bool(result.get("ok")),
                str(result.get("error")) if not result.get("ok") else None,
            )
            self.trace.record_event(
                "grasp_debug", self.grasp_debug_snapshot()
            )
        dur = (time.monotonic() - t0) * 1000
        if name in _MOTION_SKILLS:
            self._refresh_keyframe_frame(name, "after")
        after = self.trace.save_keyframe(
            self.last_frame.rgb if self.last_frame is not None else None, f"{name}_after"
        )
        self.trace.record(name, args, result, dur, before, after)
        err = str(result.get("error", "failed"))
        self._show_status(f"{name} -> " + ("ok" if result["ok"] else err[:60]))
        self.memory.add(
            "action" if result["ok"] else "outcome",
            f"{name}({_short(args)}) -> " + ("ok" if result["ok"] else err[:120]),
        )
        return result

    def _refresh_keyframe_frame(self, skill: str, phase: str) -> None:
        """Best-effort fresh capture so motion keyframes are not stale twins."""
        try:
            try:
                frame = self.camera.get_frame(timeout_s=1.5)
            except TypeError:
                frame = self.camera.get_frame()
            if frame is None:
                return
            self.last_frame = self.depth.ensure_depth(frame)
            self.trace.record_event(
                "keyframe_refresh",
                {"skill": skill, "phase": phase,
                 "frame_id": getattr(frame, "frame_id", None)},
            )
        except Exception as exc:
            self.trace.record_event(
                "keyframe_refresh_failed",
                {"skill": skill, "phase": phase, "error": str(exc)[:200]},
            )

    def observe(self) -> Frame:
        frame = self.camera.get_frame()
        frame = self.depth.ensure_depth(frame)
        self.last_frame = frame
        self.arm.harness.heartbeat()
        return frame

    def _refresh_motion_watchdog(self, allow_recent_s: float = 0.0) -> None:
        """Require a live (or explicitly recent) camera frame."""
        frame = None
        latest_fn = getattr(self.camera, "latest", None)
        if allow_recent_s > 0.0 and callable(latest_fn):
            candidate = latest_fn()
            if candidate is not None and time.monotonic() - candidate.t <= allow_recent_s:
                frame = candidate
        if frame is None:
            frame = self.camera.get_frame()
        if frame is None:
            raise SkillError("camera produced no fresh frame before motion")
        self.arm.harness.heartbeat()
        trace = getattr(self, "trace", None)
        if trace is not None:
            trace.record_event(
                "motion_watchdog_refresh",
                {"frame_id": getattr(frame, "frame_id", None)},
            )

    def frame_jpeg(self) -> bytes | None:
        if self.last_frame is None:
            return None
        return _jpeg(self.last_frame.rgb)

    def _update_beliefs_from_frame(self, frame: Frame, dets, T=None) -> list[dict]:
        summaries = []
        if not frame.has_depth:
            return [
                {"label": d.label, "conf": round(d.conf, 2), "position": None}
                for d in dets
            ]
        if T is None:
            T = (frame.T_base_cam if frame.T_base_cam is not None
                 else self.extrinsics.cam_to_base())
        for d in dets:
            mask = d.mask
            if mask is None:
                h, w = frame.rgb.shape[:2]
                mask = np.zeros((h, w), dtype=bool)
                x0, y0, x1, y1 = d.bbox.astype(int)
                mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < 10:
                continue
            pts_base = transform_points(T, pts_cam)
            center, extents, _ = oriented_bbox(pts_base)
            self.beliefs.update(
                d.label, center, d.conf, extent=extents,
                top_z=float(pts_base[:, 2].max()), t=frame.t,
                points=pts_base if d.mask is not None else None,
            )
            summaries.append(
                {
                    "label": d.label,
                    "conf": round(d.conf, 2),
                    "position": [round(float(x), 3) for x in center],
                }
            )
        return summaries

    def _tcp(self) -> np.ndarray:
        return self.kin.fk(self.arm.get_state().q)[:3, 3]

    def _reconcile_held(self) -> None:
        """Conservatively reconcile software held state with tactile data.

        A single all-zero pressure read is not proof of release: contact can
        momentarily disappear while fruit settles, and an immediate recovery
        move must never open the hand on that basis. A recently verified hold
        gets a grace period; afterwards several independent no-contact checks
        are required before the held state is cleared.
        """
        if not self.held_object:
            return
        hand = getattr(self.arm, "end_effector", None)
        tactile_holding = getattr(hand, "holding_contact", None)
        if callable(tactile_holding):
            try:
                contact = tactile_holding()
                self.trace.record_event("held_object_tactile_check", {
                    "held_object": self.held_object,
                    **contact,
                })
                if bool(contact.get("contacted")):
                    self._held_verified_t = time.monotonic()
                    self._held_no_contact_checks = 0
                    return
                cfg = getattr(self, "cfg", None)
                gcfg = getattr(cfg, "grasp", None) if cfg is not None else None
                grace_s = float(
                    gcfg.get("held_contact_grace_s", 8.0)
                    if gcfg is not None else 8.0
                )
                needed = max(int(
                    gcfg.get("held_contact_loss_checks", 3)
                    if gcfg is not None else 3
                ), 1)
                verified_t = getattr(self, "_held_verified_t", None)
                age_s = (
                    time.monotonic() - verified_t
                    if verified_t is not None else float("inf")
                )
                if age_s < grace_s:
                    self.trace.record_event("held_contact_loss_ignored", {
                        "held_object": self.held_object,
                        "reason": "verified-hold grace period",
                        "age_s": round(age_s, 3),
                        "grace_s": grace_s,
                    })
                    return
                self._held_no_contact_checks = (
                    getattr(self, "_held_no_contact_checks", 0) + 1
                )
                if self._held_no_contact_checks < needed:
                    self.trace.record_event("held_contact_loss_pending", {
                        "held_object": self.held_object,
                        "consecutive_checks": self._held_no_contact_checks,
                        "required_checks": needed,
                    })
                    return
                if not bool(contact.get("contacted")):
                    if contact.get("commanded_open") is False:
                        # A fruit can settle deeper into the hand and leave the
                        # fingertip taxel arrays while the closed hand still
                        # provides form closure. Do not accumulate this into a
                        # false release; only an explicit/open hand may clear it.
                        self._held_no_contact_checks = 0
                        self.trace.record_event("held_contact_migrated", {
                            "held_object": self.held_object,
                            "reason": (
                                "zero fingertip tactile while hand remains "
                                "commanded closed"
                            ),
                        })
                        return
                    released = self.held_object
                    self.memory.add(
                        "outcome",
                        f"O6 tactile no longer detects {released!r}; "
                        "clearing the held state",
                    )
                    self._forget_held_object()
                return
            except Exception as exc:
                record_exception = getattr(self.trace, "record_exception", None)
                if callable(record_exception):
                    record_exception("held-object tactile verification", exc)
                return
        wf = self._gripper_width_frac()
        if wf is None:  # unknown feedback: keep the cautious assumption
            return
        if wf < float(self.cfg.grasp.get("air_grasp_frac", 0.04)):
            self.memory.add(
                "outcome",
                f"I no longer feel {self.held_object!r} in the gripper "
                "(it must have slipped); clearing the held state",
            )
            self._forget_held_object()

    def _capture_held_reference(self, fix=None, R_level=None,
                                source: str = "grasp_close") -> None:
        """Record the held object's level reference for place.

        ``R_level`` is the TCP rotation at grasp close -- the one moment
        the object is known level (still resting on the table), so its up
        axis expressed in the TOOL frame is ``R_level.T @ z_hat`` and
        stays fixed for as long as it is held rigidly. Every later place
        orientation R is constrained through it:
        ``arccos((R @ u_tool).z) <= max_object_tilt_deg``.

        Call immediately after the grip is confirmed and BEFORE any wrist
        reorientation: the carry lift preserves the rotation, but the
        legacy ready-pose standardization does not (it may be passed in
        via ``R_level`` when the caller captured it earlier).

        ``fix`` (the localized ObjectFix) optionally supplies the OBB
        long axis; it is stored in the tool frame so it survives
        arbitrary later TCP orientations. Only sufficiently elongated
        objects get the long-axis constraint (bananas yes, limes no).
        """
        if R_level is None:
            R_level = np.asarray(
                self.kin.fk(self.arm.get_state().q), dtype=float
            )[:3, :3]
        R_level = np.asarray(R_level, dtype=float).reshape(3, 3).copy()
        self._held_R_level = R_level
        if source == "grasp_close":
            self._held_verified_t = time.monotonic()
            self._held_no_contact_checks = 0
        self._held_long_tool = None
        aspect_min = float(
            self.cfg.grasp.get("place_long_axis_aspect_min", 1.6)
        )
        axes = getattr(fix, "axes", None)
        extent = getattr(fix, "extent", None)
        if axes is not None and extent is not None and len(extent) >= 2:
            axes = np.asarray(axes, dtype=float)
            if (
                float(extent[0]) / max(float(extent[1]), 1e-6) >= aspect_min
                and axes.shape == (3, 3)
            ):
                long_world = axes[:, 0]
                norm = float(np.linalg.norm(long_world))
                if norm > 1e-9:
                    self._held_long_tool = R_level.T @ (long_world / norm)
        self.trace.record_event("held_level_capture", {
            "source": source,
            "up_axis_tool": (R_level.T @ np.array([0.0, 0.0, 1.0])).round(4).tolist(),
            "long_axis_tool": (
                self._held_long_tool.round(4).tolist()
                if self._held_long_tool is not None else None
            ),
        })

    def _capture_held_xy_extent(self, fix) -> None:
        points = np.asarray(getattr(fix, "points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            self._held_xy_half_extent_m = None
            return
        lo, hi = np.percentile(points[:, :2], [5.0, 95.0], axis=0)
        self._held_xy_half_extent_m = np.maximum((hi - lo) * 0.5, 0.0)

    def _forget_held_object(self) -> None:
        """Clear ALL held-object state: identity, geometry, level reference."""
        self.held_object = None
        self._held_det_label = None
        self._held_color = None
        self._held_support_offset_m = None
        self._held_xy_half_extent_m = None
        self._held_R_level = None
        self._held_long_tool = None
        self._held_verified_t = None
        self._held_no_contact_checks = 0

    def _gripper_width_frac(self) -> float | None:
        """0 = fully closed, 1 = fully open (from motor angle, linear map).
        None when the gripper position is unknown (feedback failure)."""
        state = self.arm.get_state()
        if not getattr(state, "gripper_valid", True):
            return None
        span = self._grip_closed - self._grip_open
        if abs(span) < 1e-9:
            return 0.0
        closed_frac = (state.gripper_pos - self._grip_open) / span  # 1 at closed
        return float(np.clip(1.0 - closed_frac, 0.0, 1.0))

    # ── language -> world resolution ─────────────────────────────────────

    def _resolve_query(self, query: str) -> dict:
        """Turn a user phrase ("pink object", "red mug", "bottle") into
        detector inputs, using the live belief store when it already knows
        the answer (the WorldWatcher keeps it warm)."""
        color, noun = parse_color_query(query)
        belief = self.beliefs.find(query)
        near = belief.position.copy() if belief is not None else None
        prompts = vocab = prefer = None
        if noun:
            q = query.strip().lower()
            prompts = [noun] if noun == q else [q, noun]
        elif belief is not None and (color is None or belief.color == color):
            # Warm path: the world model already knows the answer. When the
            # class is in the default vocabulary, detect with the WHOLE
            # vocabulary (no set_classes churn -- YOLOE re-embeds text on
            # every class change) and prefer that label among candidates.
            if belief.label in self._default_classes:
                vocab, prefer = self._default_classes, belief.label
            else:
                prompts = [belief.label]
        else:
            # No belief, or an UNTAGGED belief matched a color query only by
            # fallback -- don't trust it blindly: scan the vocabulary and let
            # the color filter decide (near still helps ranking).
            vocab = self._default_classes
        return {
            "prompts": prompts, "vocab": vocab, "color": color,
            "near_xyz": near, "belief": belief, "prefer_label": prefer,
        }

    def _grasp_rank_key(self, grasp) -> tuple[int, int, int, int, int, int, int]:
        """Rank backend/branch, avoid repeated geometry, then minimize yaw."""
        with self._grasp_debug_lock:
            metadata = self._grasp_candidate_meta.get(id(grasp), {})
            source = metadata.get("source", "unknown")
            branch = metadata.get("branch", "unknown")
        configured = list(self.cfg.grasp.get(
            "source_priority", ["graspgenx", "camera_grasp", "analytic_obb"]
        ))
        order = {name: index for index, name in enumerate(configured)}
        approach = np.asarray(grasp.approach, dtype=float)
        norm = float(np.linalg.norm(approach))
        down_cos = (-float(approach[2]) / norm) if norm > 1e-9 else -1.0
        max_tilt = float(self.cfg.grasp.get(
            "prefer_topdown_max_tilt_rad", np.pi / 4.0
        ))
        topdown_bucket = 0 if down_cos >= float(np.cos(max_tilt)) else 1
        branch_priority = list(self.cfg.grasp.get(
            "graspgenx_branch_priority", ["diff", "obb"]
        ))
        branch_order = {name: index for index, name in enumerate(branch_priority)}
        branch_rank = (
            int(branch_order.get(branch, len(branch_order)))
            if source == "graspgenx" else 0
        )
        retry_penalty = 0
        if bool(metadata.get("isotropic", False)):
            base_xyz = np.asarray(grasp.position, dtype=float)
            tilt_deg = float(metadata.get("tilt_deg") or 0.0)
            for failed in getattr(self, "_failed_grasp_signatures", []):
                failed_xyz = np.asarray(failed.get("base_xyz") or [], dtype=float)
                if (
                    failed.get("source") == source
                    and failed.get("branch") == branch
                    and failed_xyz.shape == (3,)
                    and abs(float(failed_xyz[2] - base_xyz[2])) < 0.004
                    and abs(float((failed.get("tilt_deg") or 0.0) - tilt_deg)) < 3.0
                ):
                    retry_penalty = 1
                    break
            preferred = np.asarray(self.cfg.grasp.get(
                "graspgenx", {}
            ).get("isotropic_preferred_jaw_axis_xy", [0.0, 1.0]), dtype=float)
            preferred /= max(float(np.linalg.norm(preferred)), 1e-9)
            opening_xy = np.asarray(grasp.rotation, dtype=float)[:2, 1]
            opening_xy /= max(float(np.linalg.norm(opening_xy)), 1e-9)
            yaw_penalty = int(round((1.0 - abs(float(opening_xy @ preferred))) * 1000))
        else:
            yaw_penalty = 0
        score_band = max(float(self.cfg.grasp.get(
            "graspgenx_branch_score_band", 0.10
        )), 1e-6)
        quality_bucket = (
            int(np.floor((1.0 - np.clip(float(grasp.quality), 0.0, 1.0)) / score_band))
            if source == "graspgenx" else 0
        )
        direction_rank = int(metadata.get("direction_rank", 0))
        return (
            direction_rank, int(order.get(source, len(order))), retry_penalty,
            topdown_bucket, quality_bucket, branch_rank, yaw_penalty,
        )

    def _grasp_prefilter(self, grasp) -> str | None:
        """Return a geometry rejection recorded while assembling candidates."""
        with self._grasp_debug_lock:
            return self._grasp_candidate_meta.get(id(grasp), {}).get(
                "geometry_reject"
            )

    def _grasp_candidate_safety_reason(self, grasp, q_pre, q_grasp) -> str | None:
        """Pre-vet a solved grasp before the bridge moves toward it."""
        harness = self.arm.harness
        reason = harness.vet_pose(q_pre)
        if reason:
            return f"pregrasp unsafe: {reason}"
        exempt_r = float(self.cfg.grasp.get("exempt_radius_m", 0.07))
        z_min = float(harness.limits.table_z) - 0.06
        q_pre = np.asarray(q_pre, dtype=float)
        q_grasp = np.asarray(q_grasp, dtype=float)
        for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
            q = q_pre + s * (q_grasp - q_pre)
            reason = harness.vet_pose(
                q, exempt_xy=grasp.position[:2],
                exempt_radius=exempt_r, exempt_z_min=z_min,
            )
            if reason:
                return f"descent unsafe: {reason}"
        return None

    def _vet_place_segment(
        self, q_a, q_b, *, exempt_xy=None, exempt_radius: float = 0.0,
        exempt_z_min=None, vet_start: bool = True,
    ) -> str | None:
        """Pre-vet a solved place segment: the approach pose plain, then
        samples along the joint interpolation q_a -> q_b with the same
        grasp-descent exemption the executed motion will open (mirrors
        _grasp_candidate_safety_reason, which vets grasps the same way).

        ``vet_start=False`` when ``q_a`` is the MEASURED current pose: the
        arm already occupies it, so the static vet of q_a carries no
        safety content and — being escape-rule-free — would veto poses
        that were legitimately reached under a grasp exemption. Observed
        2026-09-22 (run 124006): after a low banana grasp the TCP sits at
        z≈0.018, just below the table-clearance floor, and every one of
        the 52 IK-solvable lift candidates died at the q_a check
        ("TCP z below table") → held_carry_unreachable → arm frozen
        holding the object. Mirrors _make_hook's first-waypoint skip."""
        harness = self.arm.harness
        max_tilt_rad = np.radians(float(
            self.cfg.grasp.get("max_object_tilt_deg", 12.0)
        ))
        up_tool = None
        if self._held_R_level is not None:
            up_tool = self._held_R_level.T @ np.array([0.0, 0.0, 1.0])

        def _object_reason(q):
            if up_tool is None:
                return None
            R = np.asarray(self.kin.fk(q), dtype=float)[:3, :3]
            up = R @ up_tool
            tilt = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
            if tilt > max_tilt_rad + 1e-6:
                return f"object tilt {np.degrees(tilt):.1f} deg exceeds limit"
            if self._held_long_tool is not None:
                long_z = abs(float((R @ self._held_long_tool)[2]))
                if long_z >= np.sin(max_tilt_rad) + 1e-6:
                    return "held long axis leaves the table plane"
            return None

        reason = (harness.vet_pose(q_a) or _object_reason(q_a)) if vet_start else None
        if reason:
            return f"approach unsafe: {reason}"
        q_a = np.asarray(q_a, dtype=float)
        q_b = np.asarray(q_b, dtype=float)
        for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
            q = q_a + s * (q_b - q_a)
            reason = harness.vet_pose(
                q, exempt_xy=exempt_xy,
                exempt_radius=exempt_radius, exempt_z_min=exempt_z_min,
            ) or _object_reason(q)
            if reason:
                return f"descent unsafe: {reason}"
        return None

    def _search_held_place_rotation(
        self, target, hover, q_now, *,
        exempt_radius: float = 0.0, exempt_z_min=None,
        vet_from_current: bool = False,
        search_profile: str = "full",
    ):
        """Task-space place-orientation search for a held object.

        The constraint is on the OBJECT, not the wrist: keep the held
        object's up axis (captured at grasp close, see
        _capture_held_reference) within ``max_object_tilt_deg`` of world
        +Z, while yaw about world Z and small tilts are SEARCHED for IK
        feasibility. This replaces the old "freeze the carried TCP
        rotation" policy, which starved IK of exactly that freedom and
        made place fail on reachable destinations (a top-down wrist
        orientation forced at a lateral XY drives the wrist to its
        limits; a yaw-rotated one solves fine).

        Per candidate rotation: IK the hover pose, IK the release pose
        from that hover solution, vet the executed segment with the
        harness, then keep the candidate with the lowest weighted joint
        travel minus a joint-limit-margin bonus. The identity candidate
        (carried orientation, yaw=0/tilt=0) is generated first and wins
        score ties, so behavior degrades to the carried rotation when
        everything is equal.

        ``vet_from_current=True`` (bridge single-segment place): the
        executed motion is one carry->drop segment that runs above table
        clearance, so vet ``q_now -> q_release`` plain with NO exemption
        -- the hover pose that motion never visits is not vetted.

        Returns the best candidate as ``{"R", "pre", "low", "yaw_deg",
        "tilt_deg", "score"}`` (``pre``/``low`` are the hover/release IK
        results), or None when no candidate is both IK-solvable and safe.
        """
        gcfg = self.cfg.grasp
        max_tilt_deg = float(gcfg.get("max_object_tilt_deg", 12.0))
        yaw_degs = gcfg.get("place_yaw_search_deg", [
            0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 180,
        ])
        tilt_degs = gcfg.get("place_tilt_search_deg", [0, 5, -5, 10, -10])
        tilt_axis_degs = gcfg.get("place_tilt_axis_deg", [0, 90])
        if search_profile in {"carry", "local_place"}:
            prefix = "carry" if search_profile == "carry" else "local_place"
            yaw_degs = gcfg.get(f"{prefix}_yaw_search_deg", [0, 15, -15])
            tilt_degs = gcfg.get(f"{prefix}_tilt_search_deg", [0, 5, -5, 10, -10])
            tilt_axis_degs = gcfg.get(f"{prefix}_tilt_axis_deg", [0, 90])
        margin_weight = float(gcfg.get("place_margin_weight", 0.5))
        retries = max(int(gcfg.get("place_ik_retries", 1)), 0)
        wrist_weight = max(float(gcfg.get("wrist_motion_weight", 2.0)), 1.0)

        q_now = np.asarray(q_now, dtype=float)
        fallback = self._held_R_level is None
        R_level = (
            np.asarray(self.kin.fk(q_now), dtype=float)[:3, :3].copy()
            if fallback else self._held_R_level
        )
        candidates = _place_rotation_candidates(
            R_level, yaw_degs, tilt_degs, np.radians(max_tilt_deg),
            long_tool=self._held_long_tool, tilt_axis_degs=tilt_axis_degs,
        )

        # Joint limits for the margin bonus; MagicMock kin stubs in unit
        # tests convert to garbage arrays -- validate the shape so the
        # bonus just degrades to 0 there.
        limits = getattr(self.kin, "joint_limits", None)
        q_lo = q_hi = None
        try:
            lo = np.asarray(limits[0], dtype=float)
            hi = np.asarray(limits[1], dtype=float)
            if lo.ndim == 1 and lo.size and lo.shape == hi.shape:
                q_lo, q_hi = lo, hi
        except Exception:
            pass

        def _margin_bonus(q) -> float:
            if q_lo is None:
                return 0.0
            return float(np.min(np.minimum(q - q_lo, q_hi - q)))

        def _joint_travel(q_hover, q_release) -> float:
            # selector.motion_cost semantics: wrapped-angle distance, the
            # last (wrist) joint weighted, second leg at 0.25x.
            d1 = np.arctan2(np.sin(q_hover - q_now), np.cos(q_hover - q_now))
            d2 = np.arctan2(
                np.sin(q_release - q_hover), np.cos(q_release - q_hover)
            )
            weights = np.ones_like(q_hover)
            weights[-1] = wrist_weight
            return float(
                np.linalg.norm(weights * d1) + 0.25 * np.linalg.norm(d2)
            )

        def _q_array(x):
            # Unit-test stubs return Mock IK results whose .q cannot
            # convert; treat those as unscoreable (score 0, vet skipped)
            # so the identity-first ordering keeps the stub behavior
            # identical to the old frozen-rotation code path.
            try:
                q = np.asarray(x, dtype=float)
            except Exception:
                return None
            return q if q.ndim == 1 and q.size else None

        best = None
        n_ik_ok = 0
        n_vet_ok = 0
        reject_counts: dict[str, int] = {}
        for R, yaw_deg, tilt_deg, tilt_axis_deg in candidates:
            pre = self.kin.ik(
                make_transform(R, hover), q_now, retries=retries,
            )
            if not pre.success:
                _bump(reject_counts, "hover IK failed", getattr(pre, "error", None))
                continue
            low = self.kin.ik(
                make_transform(R, target), pre.q, retries=retries,
            )
            if not low.success:
                _bump(reject_counts, "release IK failed", getattr(low, "error", None))
                continue
            n_ik_ok += 1
            q_hover = _q_array(pre.q)
            q_release = _q_array(low.q)
            if q_hover is None or q_release is None:
                n_vet_ok += 1
                if best is None:
                    best = {
                        "R": R, "pre": pre, "low": low,
                        "yaw_deg": yaw_deg, "tilt_deg": tilt_deg,
                        "tilt_axis_deg": tilt_axis_deg,
                        "score": 0.0,
                    }
                continue
            if vet_from_current:
                reason = self._vet_place_segment(q_now, q_release, vet_start=False)
            else:
                # q_now is the measured pose the arm already occupies —
                # never statically vet it (see _vet_place_segment).
                reason = self._vet_place_segment(q_now, q_hover, vet_start=False)
                if reason is None:
                    reason = self._vet_place_segment(
                        q_hover, q_release,
                        exempt_xy=np.asarray(target, dtype=float)[:2],
                        exempt_radius=exempt_radius, exempt_z_min=exempt_z_min,
                    )
            if reason:
                # Vet reasons embed per-candidate floats (TCP coords etc.);
                # truncate so the histogram buckets by failure TYPE.
                _bump(reject_counts, reason[:60])
                continue
            n_vet_ok += 1
            score = (
                _joint_travel(q_hover, q_release)
                - margin_weight * min(
                    _margin_bonus(q_hover), _margin_bonus(q_release)
                )
            )
            if best is None or score < best["score"]:
                best = {
                    "R": R, "pre": pre, "low": low,
                    "yaw_deg": yaw_deg, "tilt_deg": tilt_deg,
                    "tilt_axis_deg": tilt_axis_deg,
                    "score": score,
                }
        self.trace.record_event("place_orientation_search", {
            "search_profile": search_profile,
            "target": np.asarray(target, dtype=float).round(3).tolist(),
            "max_object_tilt_deg": max_tilt_deg,
            "level_reference": (
                "fallback_current_fk" if fallback else "grasp_close"
            ),
            "n_candidates": len(candidates),
            "n_ik_ok": n_ik_ok,
            "n_vet_ok": n_vet_ok,
            "reject_reasons": dict(
                sorted(reject_counts.items(), key=lambda kv: -kv[1])[:6]
            ),
            "chosen_yaw_deg": best["yaw_deg"] if best else None,
            "chosen_tilt_deg": best["tilt_deg"] if best else None,
            "chosen_tilt_axis_deg": best["tilt_axis_deg"] if best else None,
            "best_score": round(best["score"], 3) if best else None,
        })
        return best

    def _search_held_target_rotation(self, target, hover, q_now, **kwargs):
        """Search common nearby place orientations, then expand if needed."""
        found = self._search_held_place_rotation(
            target, hover, q_now, search_profile="local_place", **kwargs,
        )
        if found is not None:
            return found
        self.trace.record_event("place_search_expanded", {
            "target": np.asarray(target, dtype=float).round(4).tolist(),
            "reason": "local place orientation search had no safe solution",
        })
        return self._search_held_place_rotation(
            target, hover, q_now, search_profile="full", **kwargs,
        )

    def _search_held_carry_rotation(self, target, q_now):
        """Fast local carry search with exhaustive fallback."""
        found = self._search_held_place_rotation(
            target, target, q_now,
            vet_from_current=True, search_profile="carry",
        )
        if found is not None:
            return found
        self.trace.record_event("held_carry_search_expanded", {
            "target": np.asarray(target, dtype=float).round(4).tolist(),
            "reason": "local carry orientation search had no safe solution",
        })
        return self._search_held_place_rotation(
            target, target, q_now,
            vet_from_current=True, search_profile="full",
        )

    def _effective_graspgenx_config(self):
        """Return end-effector-specific planner config and disable reason."""
        gcfg = self.cfg.grasp
        ee_cfg = self.cfg.arm.get("end_effector", Cfg({}))
        if str(ee_cfg.get("type", "parallel_gripper")) != "linkerhand_o6":
            return gcfg, None
        override = ee_cfg.get("graspgenx", Cfg({}))
        if not bool(override.get("calibrated", False)):
            return gcfg, "O6 GraspGenX sweep/tip offset is not calibrated"
        merged = gcfg.as_dict()
        gx = dict(merged.get("graspgenx", {}))
        gx.update({k: v for k, v in override.as_dict().items() if k != "calibrated"})
        gx.setdefault("tcp_standoff_m", float(ee_cfg.get("tcp_standoff_m", 0.0)))
        merged["graspgenx"] = gx
        return Cfg(merged), None


    def _plan_grasps(self, fix, label: str | None = None) -> list:
        """Grasp candidates: camera-frame planner primary, OBB fallback.

        Pipeline order (each step is skipped if its preconditions fail,
        the next layer picks up):

        1. **camera-frame planner** (rebot_grasp-style) — runs when the
           detection has a real ``mask`` and the last frame has depth.
           2D OBB on the mask → backproject → approach toward the camera
           origin → ``insertion_depth_m`` shift into the object. Best
           when the camera is tilted: the gripper dives along the actual
           line of sight, not a hard-coded base-frame vector.

        2. **learned GraspGen-X** — only when ``backend: graspgenx`` and
           the server is up; concatenated before the analytic fallback.

        3. **analytic OBB planner** (``plan_grasps_from_fix``) — always
           available; serves as the IK fallback when the camera-frame
           planner produces nothing (mock stack, mono camera, etc.).

        Fake-RL layer: past-attempt memory re-ranks and applies a learned
        z-nudge, just like before.
        """
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        ee_cfg = self.cfg.arm.get("end_effector", Cfg({}))
        direction_cfg = ee_cfg.get("grasp_directions", Cfg({}))
        is_o6 = str(ee_cfg.get("type", "parallel_gripper")) == "linkerhand_o6"
        tcp_standoff_m = (
            max(float(ee_cfg.get("tcp_standoff_m", 0.0)), 0.0) if is_o6 else 0.0
        )
        side_eligible = (
            is_o6
            and float(fix.position[2]) - table_z
            >= float(direction_cfg.get("side_min_center_height_m", 0.07))
        )
        base_tilt = float(gcfg.get(
            "approach_tilt_from_vertical_rad",
            gcfg.get("approach_pitch_rad", 1.2),
        ))
        direction_tilts = (
            [float(v) for v in direction_cfg.get(
                "elevated_tilts_from_vertical_rad", [1.2, 0.7, base_tilt]
            )]
            if side_eligible
            else [float(v) for v in direction_cfg.get(
                "low_tilts_from_vertical_rad", [base_tilt]
            )]
        )

        # 1. camera-frame grasp planner (rebot_grasp-style, primary)
        cam_grasps: list = []
        det = getattr(fix, "detection", None)
        last_frame = getattr(self, "last_frame", None)
        if (
            det is not None
            and getattr(det, "mask", None) is not None
            and last_frame is not None
            and getattr(last_frame, "has_depth", False)
            and getattr(last_frame, "depth_m", None) is not None
        ):
            try:
                T_cam2base = self.extrinsics.cam_to_base()

                def remember_camera_direction(candidate, metadata):
                    approach = np.asarray(candidate.approach, dtype=float)
                    tilt = float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))
                    metadata = dict(metadata)
                    metadata.update({
                        "tilt_deg": float(np.degrees(tilt)),
                        "direction_rank": int(np.argmin(
                            np.abs(np.asarray(direction_tilts) - tilt)
                        )),
                        "tcp_standoff_m": tcp_standoff_m,
                    })
                    self._remember_grasp_metadata(candidate, metadata)

                cam_grasps = plan_grasp_from_mask(
                    mask=det.mask,
                    depth_m=last_frame.depth_m,
                    K=last_frame.K,
                    T_cam2base=T_cam2base,
                    insertion_depth_m=float(
                        gcfg.get("insertion_depth_m", DEFAULT_INSERTION_DEPTH_M)
                    ),
                    max_width_m=self._max_width,
                    table_z=float(self.cfg.safety.get("table_z", 0.0)),
                    label=label or fix.label,
                    confidence=float(getattr(det, "conf", 1.0)),
                    debug_metadata=remember_camera_direction,
                    include_wrist_flip=True,
                )
                for candidate in cam_grasps:
                    candidate.position = (
                        np.asarray(candidate.position, dtype=float)
                        - np.asarray(candidate.approach, dtype=float) * tcp_standoff_m
                    )
                if cam_grasps:
                    self.memory.add(
                        "note",
                        f"camera_grasp: {len(cam_grasps)} candidate(s) "
                        f"(insertion_depth={float(gcfg.get('insertion_depth_m', DEFAULT_INSERTION_DEPTH_M))*1000:.0f}mm)",
                    )
            except Exception as e:
                self.memory.add(
                    "note", f"camera_grasp failed ({str(e)[:90]}); OBB fallback"
                )

        # 2. analytic OBB planner (fallback / supplement)
        obb = []
        detector_confidence = float(getattr(det, "conf", 0.0)) if det is not None else 0.0
        for direction_rank, tilt in enumerate(direction_tilts):
            directional = plan_grasps_from_fix(
                fix,
                table_z=table_z,
                max_width_m=self._max_width,
                depth_fraction=float(gcfg.get("depth_fraction", 0.5)),
                approach_pitch_rad=tilt,
            )
            for candidate in directional:
                candidate.position = (
                    np.asarray(candidate.position, dtype=float)
                    - np.asarray(candidate.approach, dtype=float) * tcp_standoff_m
                )
                self._remember_grasp_metadata(
                    candidate,
                    {
                        "source": "analytic_obb",
                        "confidence": detector_confidence,
                        "tilt_deg": float(np.degrees(tilt)),
                        "direction_rank": direction_rank,
                        "tcp_standoff_m": tcp_standoff_m,
                    },
                )
            obb.extend(directional)

        # 3. learned GraspGen-X (optional, with end-effector geometry)
        planner_cfg, graspgenx_disabled = self._effective_graspgenx_config()
        backend = str(gcfg.get("backend", "obb"))
        ggx_cfg = planner_cfg.get("graspgenx", Cfg({}))
        ggx_endpoint = f"{ggx_cfg.get('host', 'unknown')}:{ggx_cfg.get('port', 'unknown')}"
        if backend != "graspgenx" or graspgenx_disabled:
            reason = graspgenx_disabled or f"backend selected {backend!r}"
            if graspgenx_disabled:
                self.memory.add("note", f"GraspGen-X disabled: {graspgenx_disabled}; using geometric planners")
            trace = getattr(self, "trace", None)
            if trace is not None:
                trace.record_event("graspgenx_status", {
                    "status": "disabled",
                    "reason": reason,
                    "backend": backend,
                    "endpoint": ggx_endpoint,
                    "fallback": ["camera_grasp", "analytic_obb"],
                })
            grasps = cam_grasps + obb
        else:
            ggx_started = time.monotonic()
            try:
                if self._graspgenx is None:
                    from ..grasping.graspgenx_backend import GraspGenXPlanner

                    self._graspgenx = GraspGenXPlanner(planner_cfg)
                learned = self._graspgenx.plan(
                    fix, max_width_m=self._max_width,
                    table_z=float(self.cfg.safety.get("table_z", 0.0)),
                    deadline=self._active_pick_deadline,
                )
                if not learned:
                    raise RuntimeError("no GraspGen-X candidates survived score/spatial gates")
                # Remote inference plus local IK can consume most of the
                # watchdog window. Confirm the camera can still produce a fresh
                # frame before any candidate is allowed to move.
                self._refresh_motion_watchdog()
                for candidate in learned:
                    raw_origin = getattr(candidate, "_ggx_raw_origin", None)
                    self._remember_grasp_metadata(
                        candidate,
                        {
                            "source": "graspgenx",
                            "confidence": detector_confidence,
                            "raw_origin_xyz": (
                                np.asarray(raw_origin, dtype=float).tolist()
                                if raw_origin is not None else None
                            ),
                            "tip_offset_m": float(getattr(
                                candidate, "_ggx_tip_offset_m", 0.0
                            )),
                            "tcp_standoff_m": float(getattr(
                                candidate, "_ggx_tcp_standoff_m", 0.0
                            )),
                            "insertion_depth_m": float(getattr(
                                candidate, "_ggx_insertion_depth_m", 0.0
                            )),
                            "depth_correction_m": float(getattr(
                                candidate, "_ggx_depth_correction_m", 0.0
                            )),
                            "grasp_target_z": getattr(
                                candidate, "_ggx_target_z", None
                            ),
                            "cloud_top_z": getattr(
                                candidate, "_ggx_cloud_top_z", None
                            ),
                            "target_depth_m": getattr(
                                candidate, "_ggx_target_depth_m", None
                            ),
                            "horizontal_spans_m": getattr(
                                candidate, "_ggx_horizontal_spans_m", None
                            ),
                            "horizontal_aspect": float(getattr(
                                candidate, "_ggx_horizontal_aspect", 0.0
                            )),
                            "isotropic": bool(getattr(
                                candidate, "_ggx_isotropic", False
                            )),
                            "local_geometry": getattr(
                                candidate, "_ggx_local_geometry", None
                            ),
                            "branch": getattr(candidate, "_ggx_branch", "unknown"),
                            "tilt_deg": float(getattr(candidate, "_ggx_tilt_deg", 0.0)),
                            "grasp_score": float(candidate.quality),
                        },
                    )
                self.memory.add(
                    "note",
                    f"graspgenx: {len(learned)} grasps in {self._graspgenx.last_latency_s}s "
                    f"(top {learned[0].quality:.2f})",
                )
                trace = getattr(self, "trace", None)
                if trace is not None:
                    trace.record_event("graspgenx_status", {
                        "status": "ok",
                        "endpoint": ggx_endpoint,
                        "candidate_count": len(learned),
                        "top_score": float(learned[0].quality),
                        "server_latency_s": float(self._graspgenx.last_latency_s),
                        "total_elapsed_s": round(time.monotonic() - ggx_started, 4),
                        "candidate_diagnostics": getattr(
                            self._graspgenx, "last_diagnostics", {}
                        ),
                    })
                grasps = learned + cam_grasps + obb
            except Exception as e:
                self.memory.add(
                    "note", f"graspgenx unavailable ({str(e)[:90]}); OBB fallback"
                )
                trace = getattr(self, "trace", None)
                if trace is not None:
                    trace.record_event("graspgenx_status", {
                        "status": "fallback",
                        "endpoint": ggx_endpoint,
                        "exception_type": type(e).__name__,
                        "error": str(e),
                        "elapsed_s": round(time.monotonic() - ggx_started, 4),
                        "fallback": ["camera_grasp", "analytic_obb"],
                        "camera_candidate_count": len(cam_grasps),
                        "analytic_candidate_count": len(obb),
                        "candidate_diagnostics": getattr(
                            self._graspgenx, "last_diagnostics", {}
                        ) if self._graspgenx is not None else {},
                    })
                grasps = cam_grasps + obb

        # O6 tabletop mode: every backend may propose a different tilt, but
        # execution makes the calibrated three-finger approach vertical and
        # preserves only the actual thumb <-> index/middle closing yaw.  This
        # does not equate the hand_base->TCP position vector with approach.
        strict_grasp_topdown = bool(
            direction_cfg.get("strict_grasp_topdown", False)
        )
        if is_o6 and strict_grasp_topdown:
            produce = any(
                word in str(label or fix.label).lower()
                for word in ("banana", "apple", "orange", "lime", "mango", "pear", "fruit")
            )
            cloud_top_z = float(np.quantile(np.asarray(fix.points)[:, 2], 0.98))
            target_depth = float(gcfg.get(
                "graspgenx", {}
            ).get("produce_target_below_cloud_top_m", 0.020))
            common_target_z = max(
                cloud_top_z - target_depth,
                float(self.arm.harness.limits.table_z)
                + float(gcfg.get("min_grasp_z_m", 0.008)),
            )
            for grasp in grasps:
                with self._grasp_debug_lock:
                    metadata = self._grasp_candidate_meta.setdefault(id(grasp), {})
                    # GraspGenX positions were depth-corrected along their
                    # original (possibly oblique) approach. Once that pose is
                    # converted to strict vertical approach, retain x/y but
                    # put the TCP at the already computed produce target z.
                    # Otherwise a near-horizontal candidate can remain several
                    # centimetres above the object after its rotation changes.
                    _apply_strict_grasp_topdown(grasp, metadata)
                    # Camera/analytic fallbacks previously kept their shallow
                    # surface TCP (~60 mm for this banana), while GGX used the
                    # calibrated closure depth (~46 mm). Normalize every O6
                    # tabletop-produce backend to the same physical depth.
                    if produce:
                        p = np.asarray(grasp.position, dtype=float).copy()
                        p[2] = common_target_z
                        grasp.position = p
                        metadata["grasp_target_z"] = common_target_z
                        metadata["z_normalized_for_o6"] = True

        # ---- wrist-feasibility yaw shims ---------------------------------
        # The geometric planners emit ONE closing-axis yaw per candidate. At
        # lateral table positions that exact yaw can be IK-infeasible while
        # a +-15..30 deg relaxation solves cleanly (see _yaw_shimmed).
        # Shims inherit the parent's metadata (minus any geometry reject)
        # and sort immediately after it, so they are only IK-solved when
        # every exact-yaw candidate has already been rejected.
        shim_degs = [
            float(v) for v in gcfg.get(
                "yaw_shim_degrees", [15.0, -15.0, 30.0, -30.0]
            )
            if abs(float(v)) > 1e-6
        ]
        if shim_degs:
            shims = []
            for grasp in grasps:
                shims.append(grasp)
                with self._grasp_debug_lock:
                    parent_meta = dict(
                        self._grasp_candidate_meta.get(id(grasp), {})
                    )
                if parent_meta.get("source") not in (
                    "camera_grasp", "analytic_obb",
                ):
                    continue  # learned candidates already span yaw; no prefilter rejects
                if parent_meta.get("geometry_reject"):
                    continue  # don't rescue candidates the elongated gate killed
                for deg in shim_degs:
                    shim = _yaw_shimmed(grasp, deg)
                    shims.append(shim)
                    meta = dict(parent_meta)
                    meta["yaw_shim_deg"] = deg
                    meta.pop("geometry_reject", None)
                    self._remember_grasp_metadata(shim, meta)
            grasps = shims

        # ---- fake-RL memory prior: re-rank + z-nudge -----------------------
        lbl = label or getattr(fix, "label", None) or "object"
        try:
            prior = self.grasp_memory.prior(lbl, fix)
            if prior:
                grasps = self.grasp_memory.rerank(grasps, lbl, fix)
                dz = float(prior["nudges"].get("grasp_z_delta", 0.0))
                if abs(dz) > 1e-4:
                    for g in grasps:
                        p = np.asarray(g.position, dtype=float).copy()
                        p[2] = p[2] + dz
                        g.position = p
                self._last_grasp_z_nudge = dz
                self.memory.add(
                    "note",
                    f"grasp-memory prior for {prior['profile']}: "
                    f"{prior['wins']}W/{prior['losses']}L sr={prior['success_rate']:.0%}"
                    + (f", z_nudge={dz:+.3f}" if abs(dz) > 1e-4 else "")
                    + (f", avoid={prior['top_fail']}" if prior.get("top_fail") else ""),
                )
            else:
                self._last_grasp_z_nudge = 0.0
        except Exception as e:
            self._last_grasp_z_nudge = 0.0
            self.memory.add("note", f"grasp-memory prior skipped ({str(e)[:60]})")
        for grasp in grasps:
            self._remember_grasp_metadata(
                grasp,
                {"source": "unknown", "confidence": detector_confidence},
            )

        # A global 3-D OBB is unreliable for a curved/elongated object: its
        # PCA axis can make the banana length look like an 80 mm jaw span.
        # Use the detection OBB as a direct short-axis hint and reject
        # candidates whose actual closing axis is inconsistent with it. A
        # camera-grasp candidate is preferred as the hint, but is not required.
        # This is a direction gate, not a score penalty, so an IK-friendly
        # long-axis candidate cannot silently win after the short-axis pose fails.
        elongated = False
        corners = getattr(det, "obb", None) if det is not None else None
        if corners is None and det is not None and getattr(det, "mask", None) is not None:
            # Segmentation models normally populate result.masks, not
            # result.obb. Recompute the oriented rectangle from the mask so
            # elongated-axis gating is available on the normal production
            # detector path as well.
            corners = mask_obb_corners(det.mask)
        if corners is not None:
            corners = np.asarray(corners, dtype=float).reshape(-1, 2)
            if len(corners) >= 4:
                edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
                positive = edges[edges > 1e-6]
                elongated = bool(len(positive) and positive.max() / positive.min() >= 2.0)
        if not elongated and det is not None:
            x0, y0, x1, y1 = np.asarray(det.bbox, dtype=float)
            dims = np.array([abs(x1 - x0), abs(y1 - y0)])
            elongated = bool(dims.min() > 1e-6 and dims.max() / dims.min() >= 2.0)
        hint = None
        hint_source = None
        if elongated and cam_grasps:
            hint = np.asarray(cam_grasps[0].rotation, dtype=float)[:, 1].copy()
            hint_source = "camera_grasp"
        elif elongated and corners is not None and last_frame is not None:
            # Do not disable the short-axis gate merely because camera_grasp
            # produced no candidate. The detector OBB plus localized depth is
            # sufficient to express its short edge in base coordinates.
            try:
                T_cam2base = self.extrinsics.cam_to_base()
                T_base2cam = np.linalg.inv(T_cam2base)
                p_base = np.append(np.asarray(fix.position, dtype=float), 1.0)
                depth_m = float((T_base2cam @ p_base)[2])
                hint = obb_short_axis_base(
                    corners, last_frame.K, T_cam2base, depth_m,
                )
                hint_source = "detection_obb"
            except Exception as e:
                self.memory.add(
                    "note", f"elongated-axis hint unavailable ({str(e)[:70]})"
                )
        if elongated and hint is not None:
            hint = np.asarray(hint, dtype=float)
            hint[2] = 0.0
            hint_norm = float(np.linalg.norm(hint))
            if hint_norm > 1e-6:
                hint /= hint_norm
                for grasp in grasps:
                    local_geometry = getattr(grasp, "_ggx_local_geometry", None)
                    candidate_hint = hint
                    candidate_hint_source = hint_source
                    if local_geometry is not None:
                        local_axis = np.asarray(
                            local_geometry.get("short_axis_xy", []), dtype=float
                        )
                        if local_axis.shape == (2,) and np.linalg.norm(local_axis) > 1e-6:
                            candidate_hint = np.array(
                                [local_axis[0], local_axis[1], 0.0], dtype=float
                            )
                            candidate_hint_source = "local_curve"
                    axis = np.asarray(grasp.rotation, dtype=float)[:, 1].copy()
                    axis[2] = 0.0
                    axis_norm = float(np.linalg.norm(axis))
                    alignment = (
                        abs(float(axis @ candidate_hint)) / axis_norm
                        if axis_norm > 1e-6 else 0.0
                    )
                    with self._grasp_debug_lock:
                        metadata = self._grasp_candidate_meta.setdefault(id(grasp), {})
                        metadata.update({
                            "elongated_short_axis_hint": candidate_hint.tolist(),
                            "elongated_axis_hint_source": candidate_hint_source,
                            "elongated_axis_alignment": alignment,
                            "local_geometry": local_geometry,
                        })
                        centerline_offset = float(
                            (local_geometry or {}).get("centerline_offset_m", 0.0)
                        )
                        if centerline_offset > float(gcfg.get(
                            "elongated_max_centerline_offset_m", 0.012
                        )):
                            metadata["geometry_reject"] = (
                                "elongated object: candidate is too far from "
                                f"the local centerline ({centerline_offset:.3f} m)"
                            )
                        if alignment < float(gcfg.get(
                            "elongated_axis_min_alignment", 0.75
                        )):
                            metadata.setdefault("geometry_reject", (
                                "elongated object: actual closing axis disagrees "
                                f"with {candidate_hint_source} short axis"
                            ))
        return grasps

    def _localize(self, query: str, spatial_hint: str | None = None):
        """Fresh frame + color/proximity-aware 3D fix for a user phrase.

        A single detector pass at low confidence flickers frame to frame, so
        a miss is retried on a couple of FRESH frames before giving up on
        detection. After that it falls back to a fresh belief: the
        WorldWatcher fuses at ~3 Hz with temporal stability, and an object
        the world model has seen recently is still perfectly actionable
        (that is the whole point of keeping the model warm). Staleness is
        measured against the moment the current motion skill STARTED, not
        "now" -- the watcher is paused while the arm moves, so beliefs age
        artificially during exactly the retries that need them."""
        r = self._resolve_query(query)
        frame = None
        last_err: SkillError | None = None
        tries = int(self.cfg.get("perception_loop", {}).get("localize_frames", 3))
        for _ in range(max(tries, 1)):
            if self.arm.harness.estopped:
                raise SafetyViolation("e-stopped")
            frame = self.observe()
            try:
                fix = localize_object(
                    frame, query, self.detector, self.extrinsics,
                    prompts=r["prompts"], spatial_hint=spatial_hint,
                    color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                    prefer_label=r["prefer_label"],
                )
                return frame, fix
            except SkillError as e:
                last_err = e
        # Second filter: THE OTHER CAMERAS. An object 40 px small (or
        # occluded) in the primary view may be plainly visible from the side
        # or wrist camera; each pass costs one detector call.
        from ..perception.grounding import Extrinsics as _Ext

        for cam in self._other_cams():
            try:
                cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                if not cframe.has_depth:
                    continue
                ext = (_Ext(T=cframe.T_base_cam)
                       if cframe.T_base_cam is not None else cam.extrinsics)
                fix = localize_object(
                    cframe, query, self.detector, ext,
                    prompts=r["prompts"], spatial_hint=spatial_hint,
                    color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                    prefer_label=r["prefer_label"],
                )
                self.memory.add(
                    "note",
                    f"localize {query!r}: primary camera missed it; found "
                    f"through {getattr(cam.stream, 'name', 'another camera')}",
                )
                return cframe, fix
            except Exception:
                continue  # a miss or camera hiccup: try the next view

        belief = r["belief"] or self.beliefs.find(query)
        max_age = float(self.cfg.get("perception_loop", {}).get(
            "belief_fallback_age_s", 3.0))
        # Staleness reference: the last fresh re-scan during this motion (a
        # re-scan that FAILED to see the object must age the belief), else
        # the motion start (the watcher is paused while the arm moves), else
        # now.
        if self._last_reobserve_t is not None:
            ref_t = self._last_reobserve_t
        elif self._motion_t0 is not None:
            ref_t = self._motion_t0
        else:
            ref_t = time.monotonic()
        mem_ok = (
            belief is not None
            and (ref_t - belief.last_seen_t) <= max_age
            # Grasping from memory demands an EXACT color match: the belief
            # store's neighbor tolerance (pink~red) is for conversation, not
            # for choosing what the jaws close on.
            and (r["color"] is None or belief.color == r["color"])
        )
        if mem_ok:
            self.memory.add(
                "note",
                f"localize {query!r}: instant detection missed; using the "
                f"world-model fix ({belief.label}, "
                f"{time.monotonic() - belief.last_seen_t:.1f}s old)",
            )
            return frame, self._fix_from_belief(query, belief)
        # Last filter: VLM grounding. YOLOE's text embeddings miss what a
        # full VLM reads easily; one slow call only ever runs on this
        # failure path.
        fix = self._vlm_ground_fix(frame, query)
        if fix is not None:
            return frame, fix
        raise last_err

    def _vlm_ground_fix(self, frame, query: str):
        """Ask the VLM (2nd perception filter) for the object's bbox and
        lift it to a 3D fix -- trying EVERY camera's view (the primary may
        see the object at 40 px while the side camera fills the frame with
        it). Fail-soft: any error returns None so the caller reports the
        original detector failure."""
        gcfg = self.cfg.get("grounder", None)
        if not gcfg:
            return None
        try:
            if self._grounder is None:
                from ..perception.vlm_ground import VLMGrounder

                self._grounder = VLMGrounder(
                    base_url=str(gcfg["base_url"]),
                    model=str(gcfg.get("model", "")),
                    timeout_s=float(gcfg.get("timeout_s", 45.0)),
                )
            # Views to try: (frame, cam->base T). Primary first, then the
            # other fusing cameras with their own extrinsics.
            views = []
            if frame is not None and frame.has_depth:
                views.append((frame, frame.T_base_cam if frame.T_base_cam
                              is not None else self.extrinsics.cam_to_base()))
            for cam in self._other_cams():
                try:
                    cframe = cam.depth.ensure_depth(cam.stream.get_frame())
                except Exception:
                    continue
                if not cframe.has_depth:
                    continue
                views.append((cframe, cframe.T_base_cam if cframe.T_base_cam
                              is not None else cam.extrinsics.cam_to_base()))
            det = gframe = T = None
            for cframe, cT in views:
                det = self._grounder.ground(cframe, query)
                if det is not None:
                    gframe, T = cframe, cT
                    break
            if det is None:
                return None
            h, w = gframe.rgb.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = det.bbox.astype(int)
            mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(gframe, mask)
            if pts_cam.shape[0] < 30:
                return None
            pts = transform_points(T, pts_cam)
            # A bbox rectangle sweeps in table pixels; shave the table plane
            # or the OBB extent balloons (the GGX bbox-slab lesson).
            table_z = float(self.cfg.safety.get("table_z", 0.0))
            above = pts[pts[:, 2] > table_z + 0.005]
            if above.shape[0] >= 30:
                pts = above
            center, extents, axes = oriented_bbox(pts)
            from ..types import ObjectFix

            self.memory.add(
                "note",
                f"YOLOE missed {query!r}; the VLM (2nd filter) found it at "
                f"{[round(float(v), 2) for v in center]}",
            )
            return ObjectFix(
                label=query, position=center, points=pts,
                detection=det, extent=extents, axes=axes,
            )
        except Exception as e:
            self.memory.add("note", f"VLM grounding unavailable: {str(e)[:80]}")
            return None

    def _other_cams(self) -> list:
        """Non-primary fusing cameras from the watcher wiring: each entry
        has .stream, .depth and .extrinsics (an object may be visible from
        the side camera while the primary sees it at 5 px)."""
        cams = getattr(self.watcher, "_cams", None) if self.watcher else None
        if not cams:
            return []
        return [c for c in cams[1:] if getattr(c, "fuse", True)]

    def _reobserve(self, frames: int = 2) -> None:
        """Refresh beliefs with fresh detector passes over EVERY fusing
        camera while the WorldWatcher is paused (motion skills hold it):
        fusion by hand, exactly what the 3 Hz loop would do. The held
        object (if any) is excluded -- fusing it dangling mid-air would
        corrupt its belief."""
        held = self._held_det_label if self.held_object else None
        for _ in range(max(int(frames), 1)):
            try:
                frame = self.observe()
                dets = self.detector.detect(frame, classes=self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                self._show_detections(dets)
                self._update_beliefs_from_frame(frame, dets)
                self._last_reobserve_t = time.monotonic()
            except Exception:
                return  # best effort: a camera hiccup must not kill the retry
        for cam in self._other_cams():
            try:
                frame = cam.depth.ensure_depth(cam.stream.get_frame())
                if not frame.has_depth:
                    continue
                dets = self.detector.detect(frame, classes=self._default_classes)
                if held is not None:
                    dets = [d for d in dets if d.label != held]
                T = (frame.T_base_cam if frame.T_base_cam is not None
                     else cam.extrinsics.cam_to_base())
                self._update_beliefs_from_frame(frame, dets, T=T)
            except Exception:
                continue

    @staticmethod
    def _fix_from_belief(query: str, belief) -> "ObjectFix":
        """Synthesize an ObjectFix from a belief.

        Preferred: the belief's remembered REAL point cloud (true shape ->
        true jaw width; a banana reads 35mm, not the 78mm of its bbox slab),
        re-centered on the fused position. Fallback: an axis-aligned box at
        the remembered extent (dense enough for both the OBB planner and the
        GraspGen-X backend)."""
        from ..types import ObjectFix

        center = np.asarray(belief.position, dtype=float).reshape(3)
        pts_mem = getattr(belief, "points", None)
        if pts_mem is not None and len(pts_mem) >= 50:
            pts = np.asarray(pts_mem, dtype=float)
            obb_center, extents, axes = oriented_bbox(pts)
            pts = pts + (center - obb_center)
            det = Detection(
                label=belief.label, conf=float(belief.conf),
                bbox=np.zeros(4, dtype=np.float32),
            )
            return ObjectFix(
                label=query, position=center.copy(), points=pts,
                detection=det, extent=extents, axes=axes,
            )
        ext = (np.sort(np.abs(np.asarray(belief.extent, dtype=float)))[::-1]
               if belief.extent is not None else np.array([0.05, 0.05, 0.05]))
        half = np.clip(ext[:3] / 2.0, 0.01, 0.2)
        top_z = float(belief.top_z) if belief.top_z is not None else float(center[2] + half[2])
        bottom_z = top_z - 2 * half[2]
        rng = np.random.default_rng(0)
        pts = []
        for ax in range(3):  # sample the 6 box faces
            for sign in (-1.0, 1.0):
                p = (rng.random((60, 3)) - 0.5) * 2 * half
                p[:, ax] = sign * half[ax]
                pts.append(p)
        pts = np.concatenate(pts)
        pts[:, 2] = np.clip(pts[:, 2] + (top_z + bottom_z) / 2, bottom_z, top_z)
        pts[:, :2] += center[:2]
        det = Detection(
            label=belief.label, conf=float(belief.conf),
            bbox=np.zeros(4, dtype=np.float32),
        )
        return ObjectFix(
            label=query,
            position=np.array([center[0], center[1], (top_z + bottom_z) / 2]),
            points=pts,
            detection=det,
            extent=np.array([2 * half[0], 2 * half[1], 2 * half[2]]),
            axes=np.eye(3),
        )

    def _record_grasp_localization(self, frame, fix, query: str) -> None:
        """Persist the exact perception fix used to plan this grasp attempt."""
        position = np.asarray(fix.position, dtype=float)
        points = np.asarray(fix.points, dtype=float)
        extent = np.asarray(fix.extent, dtype=float)
        if (
            position.shape != (3,)
            or points.ndim != 2 or points.shape[1] < 3 or not len(points)
            or extent.size < 3
        ):
            return
        bbox = np.asarray(fix.detection.bbox, dtype=float).reshape(4)
        with self._grasp_debug_lock:
            debug = dict(getattr(self, "_grasp_debug", {}))
        payload = {
            "timestamp": time.time(),
            "grasp_call_index": int(debug.get("grasp_call_index", 0)),
            "pick_attempt": int(debug.get("pick_attempt", 1)),
            "frame_id": int(getattr(frame, "frame_id", 0)),
            "frame_monotonic_s": float(getattr(frame, "t", 0.0)),
            "query": query,
            "detected_as": fix.detection.label,
            "source": (
                "world_model"
                if np.allclose(bbox, 0.0)
                else "fresh_detection"
            ),
            "confidence": float(fix.detection.conf),
            "base_x_m": float(position[0]),
            "base_y_m": float(position[1]),
            "base_z_m": float(position[2]),
            "base_xyz_m": position.tolist(),
            "cloud_top_z_m": float(points[:, 2].max()),
            "cloud_bottom_z_m": float(points[:, 2].min()),
            "n_points": int(points.shape[0]),
            "extent_0_m": float(extent[0]),
            "extent_1_m": float(extent[1]),
            "extent_2_m": float(extent[2]),
            "extent_m": extent.tolist(),
            "bbox_x0_px": float(bbox[0]),
            "bbox_y0_px": float(bbox[1]),
            "bbox_x1_px": float(bbox[2]),
            "bbox_y1_px": float(bbox[3]),
            "bbox_xyxy_px": bbox.tolist(),
        }
        recorder = getattr(self.trace, "record_grasp_localization", None)
        if callable(recorder):
            recorder(payload)
        else:
            self.trace.record_event("grasp_object_localization", payload)

    # ── skills ───────────────────────────────────────────────────────────

    def skill_get_observation(self) -> dict:
        frame = self.observe()
        # Keep the always-on watcher at the detector's fast default size, but
        # spend more pixels on an explicit LOOK.  Small tabletop packages at
        # the edge of the 1280x720 overhead frame disappear when YOLOE scales
        # the image to its default 640 input.
        look_imgsz = int(self.cfg.detector.get("look_imgsz", 0) or 0)
        dets = self.detector.detect(
            frame,
            classes=self._default_classes,
            imgsz=look_imgsz or None,
        )
        self._show_detections(dets)
        objects = self._update_beliefs_from_frame(frame, dets)
        self.memory.add(
            "observation",
            f"saw {[o['label'] for o in objects]} (depth: {frame.depth_source})",
            rgb=frame.rgb,
        )
        # A pure LOOK must not power the motors: a SafeArm that has not
        # been started yet reports standby instead of being poked awake.
        if getattr(self.arm, "connected", True):
            state = self.arm.get_state()
            tcp = self.kin.fk(state.q)[:3, 3]
            wf = self._gripper_width_frac()
            robot = {
                "q_deg": [round(float(np.degrees(x)), 1) for x in state.q],
                "tcp_xyz": [round(float(x), 3) for x in tcp],
                "gripper_open_frac": round(wf, 2) if wf is not None else "unknown",
                "holding": self.held_object,
            }
        else:
            robot = {"status": "standby (motors unpowered until the first motion)",
                     "holding": self.held_object}
        return {
            "objects_visible": objects,
            "objects_remembered": self.beliefs.summary(),
            "robot": robot,
            "depth_source": frame.depth_source,
        }

    def skill_list_objects(self) -> dict:
        return {
            "objects": self.beliefs.summary(),
            "holding": self.held_object,
            "note": "state=remembered means not currently visible; position is last known",
        }

    def skill_recall_memory(self, query: str) -> dict:
        """Recall durable object belief plus matching recent event text."""
        belief = self.beliefs.find(query)
        recent = [
            {"kind": ev.kind, "text": ev.text}
            for ev in self.memory.events()
            if query.lower() in ev.text.lower()
        ][-5:]
        if belief is None and not recent:
            return {"ok": False, "query": query, "error": "no matching memory"}
        result = {"query": query, "recent_events": recent}
        if belief is not None:
            result["object_memory"] = {
                "label": belief.label,
                "last_position": np.asarray(belief.position, dtype=float).round(4).tolist(),
                "confidence": round(float(belief.conf), 3),
                "color": belief.color,
                "state": belief.state(time.monotonic()),
            }
        return result

    def skill_localize_object(self, label: str, spatial_hint: str | None = None) -> dict:
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        color = detection_color(frame.rgb, fix.detection)
        # Beliefs live under DETECTOR labels (what the watcher re-fuses);
        # recording the user's query words would create ghost duplicates.
        self.beliefs.update(
            fix.detection.label or fix.label, fix.position, fix.detection.conf,
            extent=fix.extent, top_z=float(fix.points[:, 2].max()), t=frame.t,
            color=color,
        )
        return {
            "label": label,
            "detected_as": fix.detection.label,
            "color": color,
            "position": [round(float(x), 3) for x in fix.position],
            "extent_m": [round(float(x), 3) for x in fix.extent],
            "n_points": int(fix.points.shape[0]),
        }

    def skill_grasp_object(
        self,
        label: str,
        material: str | None = None,
        spatial_hint: str | None = None,
    ) -> dict:
        self._reconcile_held()
        if self.held_object:
            raise SkillError(f"already holding {self.held_object!r}; place it first")

        # Bridge path: when a rebot_grasp bridge is wired (rebot_rs backend),
        # mirror set.py / main.py ``G`` flow exactly — open → pre(2.0) →
        # grasp(1.5) → width-limited material grip → home(3.0). The legacy v5 path
        # below stays for mock runs and for tests that don't set
        # ``self.bridge``.
        if self.bridge is not None:
            try:
                result = self._skill_grasp_object_bridge(
                    label, spatial_hint, material
                )
                if not result.get("ok", True) and not self.arm.harness.estopped:
                    try:
                        self.skill_move_home(park_gripper=False)
                    except (SkillError, SafetyViolation, RuntimeError) as home_error:
                        self.memory.add(
                            "warning",
                            f"grasp failed and return home also failed: {home_error}",
                        )
                return result
            except (SkillError, SafetyViolation, RuntimeError):
                if not self.arm.harness.estopped:
                    try:
                        self.skill_move_home(park_gripper=False)
                    except (SkillError, SafetyViolation, RuntimeError) as home_error:
                        self.memory.add(
                            "warning",
                            f"grasp failed and return home also failed: {home_error}",
                        )
                raise

        gcfg = self.cfg.grasp
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        self._record_grasp_localization(frame, fix, label)
        profile = select_profile(fix.detection.label or label, material)
        grasps = self._plan_grasps(fix, label=label)

        # Learned grasp z can overshoot below the table by a few mm (the
        # tip-offset conversion is empirical): a millimeter under the
        # workspace floor must shallow the grasp slightly, not abort it.
        lim = self.arm.harness.limits
        z_floor = max(
            float(lim.workspace_min[2]) + 0.002,
            lim.table_z + float(gcfg.get("min_grasp_z_m", 0.008)),
        )
        for g in grasps:
            dz = z_floor - float(g.position[2])
            if 0.0 < dz <= 0.03:  # bigger misses are garbage; let vetting drop them
                g.position = np.asarray(g.position, dtype=float).copy()
                g.position[2] = z_floor

        # Pre-vet every candidate against the harness geometry (with the
        # exemption cylinder the descent will open) so a doomed candidate
        # loses the ranking up front instead of aborting mid-motion.
        harness = self.arm.harness
        exempt_r = float(gcfg.get("exempt_radius_m", 0.07))

        def _vet(g, q_pre, q_grasp):
            # The approach leg executes BEFORE allow_grasp_descent opens the
            # cylinder: vet the pregrasp with NO exemption, or a candidate
            # that needed one is guaranteed to abort at the end of the
            # approach. The descent leg holds the exemption, and min-jerk is
            # a straight segment in joint space, so sampling the q_pre ->
            # q_grasp segment vets the actual executed path, not just its
            # endpoints (near-horizontal approaches can dip below the floor
            # OUTSIDE the cylinder mid-descent).
            reason = harness.vet_pose(q_pre)
            if reason:
                return f"pregrasp unsafe: {reason}"
            # Exemption floor a bit BELOW the table: the reBot RS gripper's
            # finger/wrist links extend ~5 cm below the TCP, so a top-down
            # grasp of a low object legitimately dips a link to z ~ -0.018
            # (below the table plane) while the fingers straddle the object.
            # Flooring the exemption at table_z left that link 18 mm outside
            # the cylinder -> false "link would hit the table" abort. The
            # narrow XY radius still confines this to directly over the
            # target, so allowing a small sub-table dip there is safe.
            z_min = float(harness.limits.table_z) - 0.06
            # Sample the q_pre -> q_grasp segment densely: min-jerk streaming
            # generates many intermediate waypoints, and a coarse 4-sample vet
            # can miss a mid-segment configuration where the elbow dips below
            # the table (IK multi-solution: the interpolated path can bow down
            # even when both endpoints are safe).
            for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
                q = q_pre + s * (np.asarray(q_grasp) - np.asarray(q_pre))
                reason = harness.vet_pose(
                    q, exempt_xy=g.position[:2],
                    exempt_radius=exempt_r, exempt_z_min=z_min,
                )
                if reason:
                    return f"descent unsafe: {reason}"
            return None

        state = self.arm.get_state()
        # Seed grasp IK from HOME (elbow-up), not the live pose: seeding from
        # an arbitrary current configuration can converge to an elbow-down IK
        # branch whose approach path dips a link under the table.
        _home = self.cfg.arm.get("home_q")
        _seed = np.asarray(_home, dtype=float) if _home is not None else state.q
        try:
            grasp, q_pre, q_grasp = select_grasp(
                grasps,
                self.kin,
                _seed,
                max_width_m=self._max_width,
                pregrasp_offset_m=float(gcfg.get("pregrasp_offset_m", 0.12)),
                validate=_vet,
                rank_key=self._grasp_rank_key,
                prefilter=self._grasp_prefilter,
                debug_callback=self._on_grasp_debug,
                parallel_jaw_symmetry=bool(
                    self.cfg.grasp.get("parallel_jaw_symmetry", True)
                ),
                wrist_motion_weight=float(
                    self.cfg.grasp.get("wrist_motion_weight", 2.0)
                ),
                max_wrist_delta_rad=float(
                    self.cfg.grasp.get("max_wrist_delta_rad", 2.8)
                ),
                deadline=self._active_pick_deadline,
            )
        except (SkillError, SafetyViolation) as e:
            # No candidate survived IK + harness vetting. Log it against the
            # object profile so the fake-RL memory learns to bias future
            # candidates (raise z, prefer top-down) for this object.
            try:
                self.grasp_memory.record(
                    label, fix, grasps[0] if grasps else None,
                    success=False, reason=str(e),
                    z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
            except Exception:
                pass
            raise

        # 1. open, go to pregrasp (normal speed). Re-home first so the
        # pregrasp IK seeds from a known elbow-up posture: seeding from an
        # arbitrary current pose can converge to an elbow/wrist-down IK
        # solution whose approach path dips a link below the table near the
        # base (observed: link7 at xy~(0.05,0.04) z=-0.023, far from the
        # target so no grasp-exemption cylinder can cover it).
        # Adaptive opening: open slightly wider than the object width
        grasp_width_m = float(grasp.width_m)
        # The planner width already contains jaw padding. Keep any extra
        # clearance explicit: the B601 finger guards are wider than the jaw
        # tips and a full-open sweep can strike nearby fixtures.
        margin_m = float(gcfg.get("pregrasp_open_margin_m", 0.0))
        open_distance_m = min(grasp_width_m + margin_m, self._max_width)
        if hasattr(self.arm, 'open_gripper'):
            self.arm.open_gripper(distance_m=open_distance_m)
        else:
            self.arm.set_gripper(self._grip_open, effort=0.8)
        _home = self.cfg.arm.get("home_q")
        if _home is not None:
            try:
                self.arm.move_joints(np.asarray(_home, dtype=float),
                                     duration_s=1.5)
            except Exception:
                pass  # best-effort re-home; pregrasp move is the real gate
        if not self.arm.move_joints(q_pre, duration_s=float(gcfg.get("move_duration_s", 2.5))):
            raise SkillError("did not settle at pregrasp pose")

        # 2. descend inside the exemption cylinder (slow)
        self.arm.harness.allow_grasp_descent(
            grasp.position[:2],
            radius=float(gcfg.get("exempt_radius_m", 0.07)),
            z_min=float(self.arm.harness.limits.table_z) - 0.06,
        )
        try:
            descend_dur = float(gcfg.get("descend_duration_s", 2.0))
            if not self.arm.move_cartesian(
                self.kin.fk(q_grasp), self.kin,
                duration_s=descend_dur,
                planner=self.cartesian_planner,
            ):
                raise SkillError("did not settle at grasp pose")

            # 3. close with the material profile (width-aware two-stage,
            # stall-aware). grasp.width_m = jaw_span + width_pad; the close
            # target is width-aware (default 75% of the detected width) so
            # 3D-printed bananas aren't crushed.
            self._close_two_stage(
                profile,
                grasp_width_m=float(grasp.width_m),
                max_width_m=self._max_width,
            )
            # Capture the level reference NOW: the object still rests on
            # the table, so world +Z seen from the tool IS its up axis.
            # The ready-pose standardization below rotates the wrist and
            # would tilt a held object -- it must never overwrite this.
            R_close = np.asarray(
                self.kin.fk(self.arm.get_state().q), dtype=float
            )[:3, :3].copy()

            # 4. lift back to pregrasp (speed scaled by profile)
            lift_dur = float(gcfg.get("descend_duration_s", 2.0)) / max(profile.lift_speed_scale, 0.2)
            self.arm.move_cartesian(
                self.kin.fk(q_pre), self.kin,
                duration_s=lift_dur,
                planner=self.cartesian_planner,
            )
            # Do not standardize to a fixed ready wrist pose here.  q_pre has
            # the same TCP rotation as the grasp and therefore lifts without
            # changing the fruit's gravity-relative attitude.  Any later yaw
            # or small tilt is selected by the object-constrained place
            # search after tactile/width verification confirms the hold.
        finally:
            self.arm.harness.clear_grasp_exemption()

        # 5. verify: ASPIRE heuristic on jaw travel after close. An object in
        # the jaws stalls them ABOVE the commanded close; if they reached the
        # commanded stage-2 fraction even though the object should be much
        # wider, nothing resisted: air grasp. Unknown feedback -> report the
        # grasp as unverified rather than dropping a possibly-held object.
        width_after_lift = self._gripper_width_frac()
        verified = width_after_lift is not None
        if verified:
            # Recompute the width-aware close target (same formula
            # _close_two_stage used) so the air-grasp check compares
            # apples to apples.
            from ..grasping.force import compute_close_target
            _, pos2_close_frac = compute_close_target(
                profile, float(grasp.width_m), self._max_width
            )
            commanded_open = max(1.0 - pos2_close_frac, 0.0)
            expected_open = min(grasp.width_m / self._max_width, 1.0)
            air_grasp = width_after_lift < float(gcfg.get("air_grasp_frac", 0.04)) or (
                width_after_lift <= commanded_open + 0.03
                and expected_open >= commanded_open + 0.07
            )
            if air_grasp:
                self.memory.add("outcome", f"grasp {label!r} FAILED: jaws closed on air")
                try:
                    self.grasp_memory.record(
                        label, fix, grasp, success=False, reason="air grasp",
                        z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
                except Exception:
                    pass
                return {
                    "ok": False,
                    "error": "air grasp: gripper closed fully, object not held",
                    "suggestion": "re-localize the object or try the alternate yaw",
                }
        self.held_object = label
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        self._held_support_offset_m = max(float(grasp.position[2]) - table_z, 0.0)
        self._capture_held_xy_extent(fix)
        # Level reference from the moment of close (see R_close above).
        self._capture_held_reference(fix, R_level=R_close)
        self.beliefs.mark_removed(self._held_det_label or label, near=fix.position)
        try:
            self.grasp_memory.record(
                label, fix, grasp, success=True,
                z_nudge_applied=getattr(self, "_last_grasp_z_nudge", 0.0))
        except Exception:
            pass
        self.memory.add(
            "action",
            f"grasped {label!r} (profile {profile.name}, "
            + (f"jaw at {width_after_lift:.2f})" if verified else "grip UNVERIFIED)"),
        )
        return {
            "held": label,
            "grip_profile": profile.name,
            "grip_verified": verified,
            "gripper_open_frac": round(width_after_lift, 2) if verified else None,
            "grasp_width_m": round(grasp.width_m, 3),
        }

    # ────────────────────── rebot_grasp bridge path ──────────────────────
    #
    # The methods below are reached only when ``self.bridge`` is set
    # (``rebot_rs`` backend). They mirror the motion timing of
    # ``rebot_grasp/scripts/main.py::execute_grasp`` /
    # ``set.py::_execute_place_sequence`` so the recorded trajectory can be
    # compared 1:1 against the set.py recording from 2026-08-07.
    # Differences from the legacy path above:
    #
    #   * material-aware width and torque protection is added around the
    #     set.py-aligned motion sequence; learned z-nudge/pre-vetting remain off
    #   * no wrist standardization to ready pose — set.py just calls
    #     ``_move_ready(ready_cfg)`` at the end of grasp (which IS the home
    #     rendezvous in our spec)
    #   * no air-grasp verification — set.py trusts ``grasp_driver.grasp()``'s
    #     stall-detection return value
    #   * close and hold torque come from the selected material profile
    #
    # The bridge itself wraps ``harness.begin_motion()`` / ``end_motion()``
    # so e-stop and watchdog are checked at the boundary of every move.

    def _skill_grasp_object_bridge(
        self,
        label: str,
        spatial_hint: str | None = None,
        material: str | None = None,
    ) -> dict:
        """set.py-aligned ``G`` flow: open → pre(2.0) → grasp(1.5) → grasp → home(3.0)."""
        frame, fix = self._localize(label, spatial_hint=spatial_hint)
        self._record_grasp_localization(frame, fix, label)
        grasps = self._plan_grasps(fix, label=label)
        if not grasps:
            raise SkillError(f"no grasp candidates for {label!r}")

        # Seed IK from home_q (elbow-up basin) — matches the legacy comment
        # in the v5 path; rebot_grasp's main.py doesn't seed explicitly
        # because it always moves to ready first, which has the same effect.
        home_q = self.cfg.arm.get("home_q")
        seed = (
            np.asarray(home_q, dtype=float)
            if home_q is not None
            else self.arm.get_state().q
        )
        grasp, q_pre, q_grasp = select_grasp(
            grasps,
            self.kin,
            seed,
            max_width_m=self._max_width,
            pregrasp_offset_m=float(
                self.cfg.grasp.get("pregrasp_offset_m", 0.08)
            ),
            validate=self._grasp_candidate_safety_reason,
            rank_key=self._grasp_rank_key,
            prefilter=self._grasp_prefilter,
            debug_callback=self._on_grasp_debug,
            parallel_jaw_symmetry=bool(
                self.cfg.grasp.get("parallel_jaw_symmetry", True)
            ),
            wrist_motion_weight=float(
                self.cfg.grasp.get("wrist_motion_weight", 2.0)
            ),
            max_wrist_delta_rad=float(
                self.cfg.grasp.get("max_wrist_delta_rad", 2.8)
            ),
            deadline=self._active_pick_deadline,
        )
        # Candidate IK may itself take seconds. Require a fresh sensor frame
        # immediately before starting the motion watchdog.
        self._refresh_motion_watchdog()

        pre6d, grasp6d = self._grasp_to_6d(grasp)
        home_q_arr = np.asarray(home_q, dtype=float) if home_q is not None else None

        # 0. Move to ready_pose first (matches main.py _move_ready).
        # This puts the wrist at pitch≈0.7 so subsequent IK is seeded
        # from the standard top-down-grasp configuration, producing
        # smooth trajectories.  Without this the arm starts from home_q
        # (pitch≈0°) and the IK must rotate the wrist ~40° during the
        # approach, giving awkward trajectories.
        ready6d = self._ready_pose_6d()
        ready_dur = self._ready_duration()
        if not self._safe_bridge_move(ready6d, duration=ready_dur, where="ready"):
            raise SkillError("ready motion did not reach its commanded TCP")

        # 1. Open gripper (no timeout → GraspDriver default 3.0 s)
        self.bridge.open_gripper()

        # 2. Pregrasp (set.py uses duration=2.0) — IK + vet_pose first
        if not self._safe_bridge_move(
            pre6d, duration=2.0, q_target=q_pre, where="pregrasp"
        ):
            raise SkillError("pregrasp motion did not reach its commanded TCP")

        # 3. Grasp approach (set.py uses duration=1.5) — IK + vet_pose first
        #
        # Open a grasp-exemption cylinder centered on the grasp XY for
        # the descent motion. Without this, vet_pose rejects every TCP
        # position below table_z + table_clearance (= 0.020 m for the
        # RS), and small/thin objects like a 30 mm banana whose TCP
        # would otherwise dip below the table always fail.
        # Mirrors the legacy v5 path's ``harness.allow_grasp_descent(...)
        # ... clear_grasp_exemption()`` bracket.
        try:
            grasp_xy = (float(grasp.position[0]), float(grasp.position[1]))
            grasp_z = float(grasp.position[2])
            self.arm.harness.allow_grasp_descent(
                center_xy=grasp_xy,
                radius=float(self.cfg.grasp.get("exempt_radius_m", 0.07)),
                z_min=float(self.arm.harness.limits.table_z) - 0.06,
            )
            if not self._safe_bridge_move(
                grasp6d,
                duration=float(self.cfg.arm.get("grasp_duration_s", 1.5)),
                q_target=q_grasp,
                where="grasp",
            ):
                raise SkillError("grasp motion did not reach its commanded TCP")

            # 4. Contact close with a geometry-derived safety floor.
            # grasp.width_m includes the 15 mm approach clearance; remove it
            # before applying the material compression ratio. If stall
            # detection misses a deformable object, WrcGripper stops at this
            # minimum opening instead of crushing it against the hard stop.
            profile = select_profile(fix.detection.label or label, material)
            width_pad_m = float(self.cfg.grasp.get("width_pad_m", 0.015))
            object_span_m = max(float(grasp.width_m) - width_pad_m, 0.005)
            min_hold_width_m = object_span_m * profile.object_close_frac
            close_force = max(
                profile.effort,
                float(self.cfg.grasp.get("gripper_close_motion_effort", 0.70)),
            )
            ok = self.bridge.grasp(
                force=profile.effort, min_width_m=min_hold_width_m,
                close_force=close_force,
            )
            telemetry = self.bridge.last_grasp_telemetry()
            telemetry.update({
                "label": label,
                "profile": profile.name,
                "required_jaw_width_m": float(grasp.width_m),
                "estimated_object_span_m": object_span_m,
                "minimum_hold_width_m": min_hold_width_m,
                "commanded_force": profile.effort,
            })
            self.trace.record_event("gripper_grasp", telemetry)
            if not ok:
                return {
                    "ok": False,
                    "error": "empty grasp: gripper did not detect contact "
                    f"({telemetry.get('stop_reason', 'unknown')})",
                    "gripper_telemetry": telemetry,
                }

        finally:
            # Always clear the exemption so the next motion (place or
            # move_home) gets full table-clearance enforcement.
            try:
                self.arm.harness.clear_grasp_exemption()
            except Exception:
                pass

        # The tactile close has confirmed the object. Latch this state BEFORE
        # any retreat: a fixed ready_q can rotate the wrist while software
        # still thinks the hand is empty and drop the object.
        self.held_object = label
        self._held_det_label = fix.detection.label
        self._held_color = detection_color(frame.rgb, fix.detection)
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        self._held_support_offset_m = max(
            float(grasp.position[2]) - table_z, 0.0
        )
        self._capture_held_xy_extent(fix)
        # The arm has not moved since the tactile close confirmed the
        # object: FK now is the level reference for every later place.
        self._capture_held_reference(fix)
        self.beliefs.mark_removed(
            self._held_det_label or label, near=fix.position
        )

        # 5. Held-object retreat: preserve the complete current TCP rotation
        # (top-down approach plus current closing-axis yaw/roll) and translate
        # upward only. Never send a fixed ready_q while an object is held.
        try:
            self._move_held_to_carry(
                duration_s=ready_dur,
                allow_compact=not bool(
                    getattr(self, "_pick_and_place_in_progress", False)
                ),
            )
        except (SkillError, SafetyViolation, RuntimeError) as e:
            self.memory.add(
                "note",
                f"held-object carry failed ({e}); "
                "grasp still succeeded",
            )

        return {
            "ok": True,
            "held": label,
            "grip_profile": profile.name,
            "grip_verified": telemetry.get("stop_reason") in {
                "contact_stall", "tactile_contact", "tactile_contact_timeout",
            },
            "grasp_width_m": round(float(grasp.width_m), 3),
            "object_span_m": round(object_span_m, 3),
            "minimum_hold_width_m": round(min_hold_width_m, 3),
            "force": profile.effort,
            "gripper_stop_reason": telemetry.get("stop_reason"),
        }

    def _safe_bridge_move(
        self,
        target_6d: tuple[float, ...],
        duration: float,
        *,
        q_target: np.ndarray | None = None,
        where: str = "",
    ) -> bool:
        """Wrap ``bridge.move_to_traj`` with pre-motion ``harness.vet_pose`` check.

        The bridge path bypasses ``SafeArm``'s per-waypoint approval
        (rebort_rs SDK controller owns the control loop). Without an
        explicit pre-motion vet, joint-limit / workspace-AABB violations
        from rebot_grasp's IK won't be caught until the e-stop latch.

        Strategy:
        - If ``q_target`` is supplied (e.g. home_q is already a known joint
          config), call vet_pose directly with it.
        - Otherwise IK the Cartesian target via ``self.kin.ik``, raising
          ``SkillError`` on IK failure, then vet_pose the resulting q.
        - vet_pose failure raises ``SafetyViolation`` (not just a return False);
          the bridge.move_to_traj is never called in that case.

        Per-call IK adds ~10 ms × N moves overhead. 5 moves per skill is
        negligible vs the 7 s total grasp+place runtime.
        """
        import pinocchio as pin

        if q_target is None and where.startswith("ready"):
            configured_ready_q = self.cfg.arm.get("ready_q")
            if configured_ready_q is not None:
                q_target = np.asarray(configured_ready_q, dtype=float)

        if q_target is None:
            x, y, z, roll, pitch, yaw = target_6d
            T = make_transform(
                pin.rpy.rpyToMatrix(roll, pitch, yaw),
                np.asarray([x, y, z], dtype=float),
            )
            ik = self.kin.ik(T, self.arm.get_state().q)
            if not ik.success:
                raise SkillError(
                    f"{where} IK failed for target "
                    f"({x:.3f}, {y:.3f}, {z:.3f}, "
                    f"rpy={roll:.3f}, {pitch:.3f}, {yaw:.3f})"
                )
            q_target = ik.q

        if self.arm.harness is not None:
            reason = self.arm.harness.vet_pose(q_target)
            if reason:
                raise SafetyViolation(
                    f"{where} target unsafe: {reason} (q={q_target.round(3).tolist()})"
                )

        if where == "ready":
            q_now = np.asarray(self.arm.get_state().q, dtype=float).reshape(-1)
            ready_tol = float(self.cfg.arm.get("home_noop_tol_rad", 0.02))
            if (
                q_now.size >= len(q_target)
                and float(np.max(np.abs(q_now[:len(q_target)] - q_target)))
                <= ready_tol
            ):
                self.trace.record_event("bridge_motion_skipped", {
                    "where": where,
                    "reason": "already at ready target",
                    "tolerance_rad": ready_tol,
                })
                return True

        # Reuse the successful, vetted IK solution above instead of asking the
        # SDK to solve the same Cartesian target again from another local IK
        # basin. That duplicate solve can return False before emitting any
        # trajectory even though WRC already found a valid solution.
        # SafeArm.move_joints supplies min-jerk interpolation, per-waypoint
        # safety approval, motion completion, and feedback settling.
        # Unit-test bridge mocks keep the legacy call so bridge contracts can
        # still be tested independently of physical arm motion.
        use_vetted_joint_target = (
            where.startswith("ready")
            or (not _is_mock(self.bridge) and not _is_mock(self.arm))
        )
        if use_vetted_joint_target:
            ok = bool(self.arm.move_joints(
                np.asarray(q_target, dtype=float), duration_s=duration
            ))
            self.trace.record_event("bridge_motion_settle", {
                "where": where, "ok": ok, "mode": "vetted_joint_target",
                "target_q": np.asarray(q_target, dtype=float).tolist(),
            })
        else:
            ok = self.bridge.move_to_traj(*target_6d, duration=duration)
            motion_telemetry_fn = getattr(self.bridge, "last_motion_telemetry", None)
            if callable(motion_telemetry_fn):
                motion_telemetry = motion_telemetry_fn()
                motion_telemetry.update({"where": where, "ok": bool(ok)})
                self.trace.record_event("bridge_motion_settle", motion_telemetry)
        if ok and not _is_mock(self.bridge) and not _is_mock(self.arm):
            # 2026-08-13: post-move FK verify. ``bridge.move_to_traj`` can
            # return True without actually moving (SDK silent fail --
            # dashboard shows arm parked at the prior pose while the skill
            # reports success). Re-FK and compare actual TCP to the
            # commanded Cartesian target. >2 cm error means the move did
            # not really happen; surface it as a False return so the
            # caller's return-value check (or try/except) can react.
            #
            # Skip when either ``self.bridge`` or ``self.arm`` is a Mock
            # (unit-test fixture): the mocked bridge returns True without
            # driving the mock arm, so the FK comparison always shows the
            # arm parked at the initial state and would spuriously fail
            # every test. Production paths use a real ``RebotGraspBridge``
            # + ``SafeArm`` / ``MockArm`` and run the verify normally.
            try:
                final_q = self.arm.get_state().q
                final_tcp = self.kin.fk(final_q)
                T_target = np.asarray(target_6d[:3], dtype=float)
                err_m = float(np.linalg.norm(final_tcp[:3, 3] - T_target))
                trace = getattr(self, "trace", None)
                if trace is not None:
                    trace.record_event(
                        "bridge_move_verify",
                        {
                            "where": where,
                            "commanded_tcp_xyz": T_target.tolist(),
                            "actual_tcp_xyz": final_tcp[:3, 3].tolist(),
                            "position_error_m": err_m,
                            "actual_q": np.asarray(final_q, dtype=float).tolist(),
                            "duration_s": float(duration),
                        },
                    )
                if err_m > 0.02:
                    self.memory.add(
                        "note",
                        f"{where} post-move FK verify FAILED: TCP off by "
                        f"{err_m*1000:.0f} mm; "
                        f"target={target_6d[:3]} actual="
                        f"{final_tcp[:3, 3].round(3).tolist()} (SDK silent fail)",
                    )
                    return False
            except Exception as e:
                self.memory.add(
                    "note",
                    f"{where} post-move FK verify raised {type(e).__name__}: {e}",
                )
        return ok

    def _grasp_to_6d(self, grasp) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Convert a ``Grasp`` into ``(pre6d, grasp6d)`` for ``bridge.move_to_traj``.

        Both 6-tuples are ``(x, y, z, roll, pitch, yaw)``. The rotation is
        the same at pregrasp and grasp — only the position differs by
        ``approach * pregrasp_offset_m``. Uses pinocchio's intrinsic XYZ
        rpy convention, matching rebot_grasp's
        ``rotation_matrix_to_euler_zyx``-derived usage in set.py.
        """
        import pinocchio as pin

        roll, pitch, yaw = pin.rpy.matrixToRpy(grasp.rotation)
        g_pos = grasp.position
        pre_pos = grasp.pregrasp_position(
            float(self.cfg.grasp.get("pregrasp_offset_m", 0.08))
        )
        grasp6d = (
            float(g_pos[0]), float(g_pos[1]), float(g_pos[2]),
            float(roll), float(pitch), float(yaw),
        )
        pre6d = (
            float(pre_pos[0]), float(pre_pos[1]), float(pre_pos[2]),
            float(roll), float(pitch), float(yaw),
        )
        return pre6d, grasp6d

    def _home_q_to_6d(self) -> tuple[float, ...]:
        """FK wrc_demo's ``home_q`` to ``(x, y, z, roll, pitch, yaw)``.

        Falls back to ``cfg.arm.ready_pose`` if ``home_q`` is missing —
        both come from the same arm profile YAML so this is defensive
        only.
        """
        import pinocchio as pin

        home_q = self.cfg.arm.get("home_q")
        if home_q is None:
            r = self.cfg.arm.get("ready_pose", {}) or {}
            return (
                float(r.get("x", 0.3)),
                float(r.get("y", 0.0)),
                float(r.get("z", 0.3)),
                float(r.get("roll", 0.0)),
                float(r.get("pitch", 0.7)),
                float(r.get("yaw", 0.0)),
            )
        q = np.asarray(home_q, dtype=float)
        T = self.kin.fk(q)
        roll, pitch, yaw = pin.rpy.matrixToRpy(T[:3, :3])
        return (
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(roll), float(pitch), float(yaw),
        )

    def _ready_pose_6d(self) -> tuple[float, ...]:
        """Return the arm-profile ready_pose as ``(x, y, z, r, p, y)``.

        Matches ``rebot_grasp/scripts/main.py::_move_ready`` exactly:
        the ready pose puts the wrist at ``pitch≈0.7`` (≈40°), which is
        the standard top-down-grasp configuration.  IK seeded from
        ready produces smooth, natural trajectories; IK seeded from
        ``home_q`` (pitch≈0°) forces the solver to rotate the wrist
        ~40° *during* the approach, giving awkward trajectories.
        """
        ready_q = self.cfg.arm.get("ready_q")
        if ready_q is not None:
            import pinocchio as pin

            T = self.kin.fk(np.asarray(ready_q, dtype=float))
            roll, pitch, yaw = pin.rpy.matrixToRpy(T[:3, :3])
            return (
                float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
                float(roll), float(pitch), float(yaw),
            )
        r = self.cfg.arm.get("ready_pose", {}) or {}
        return (
            float(r.get("x", 0.3)),
            float(r.get("y", 0.0)),
            float(r.get("z", 0.3)),
            float(r.get("roll", 0.0)),
            float(r.get("pitch", 0.7)),
            float(r.get("yaw", 0.0)),
        )

    def _ready_duration(self) -> float:
        r = self.cfg.arm.get("ready_pose", {}) or {}
        return float(r.get("duration", 3.0))

    def _localize_destination(self, frame, destination: str):
        """Localize a container using same-frame detector-vocabulary aliases."""
        labels = [destination]
        if destination == "wicker basket":
            labels.extend(["basket", "box"])
        for label in labels:
            fix = self._single_localize(frame, label)
            if fix is not None:
                if label != destination:
                    self.trace.record_event("destination_alias_matched", {
                        "requested": destination,
                        "matched_query": label,
                        "detected_as": fix.detection.label,
                    })
                return fix
        return None

    def _skill_pick_and_place_bridge(
        self, object_label: str, destination: str,
        already_held: str | None = None, material: str | None = None,
    ) -> dict:
        """set.py-aligned pick-and-place.

        Mirrors ``set.py::_execute_place_sequence`` end to end:

            [Step1] same-frame detect banana + box
            [Step2] open → pregrasp(2.0) → grasp(1.5) → width-limited grip
            [Step3] lift to ready position  (we use wrc_demo home_q FK)
            [Step4] move to destination at configured release z, object
            kept level (max_object_tilt_deg) with searched yaw/tilt
            [Step4b] open_gripper(timeout=0.5) + 0.8 s async close
            [Step5] return to ready  (wrc_demo home_q FK)

        The destination label is resolved via ``_single_localize`` on the
        SAME frame as the object — set.py's YOLO pass produces both
        detections in one inference. This sidesteps the WorldWatcher
        belief-fusion cache (the 3 Hz watcher might serve a stale box
        position from seconds ago).

        Destination release z is configurable via ``bridge_release_height_m``.
        It defaults to the original set.py 0.08 m, while the real wicker-basket
        experiment raises it to 0.14 m so the 50 mm-long fingers clear the 75 mm rim.
        """
        import pinocchio as pin

        # ── [Step1] detect destination (box) ──────────────────────────────
        # When already_held, the arm already has the object — skip pick
        # phase entirely and go straight to place (matches set.py Step4).
        ready6d = self._ready_pose_6d()
        ready_dur = self._ready_duration()

        if already_held:
            # Object is already in the gripper; only need to find the box.
            frame = self.observe()
            box_fix = self._localize_destination(frame, destination)
            if box_fix is None:
                raise SkillError(
                    f"no {destination!r} detected in current frame "
                    "(bridge path requires live box detection)"
                )
            self.memory.add(
                "note",
                f"already holding {already_held!r} -- skipping pick, "
                f"going straight to place on {destination!r}",
            )
        else:
            # ── Full pick phase (same-frame detect + grasp) ─────────────
            frame = self.observe()
            banana_fix = self._single_localize(frame, object_label)
            if banana_fix is None:
                raise SkillError(f"no {object_label!r} detected in current frame")
            box_fix = self._localize_destination(frame, destination)
            if box_fix is None:
                raise SkillError(
                    f"no {destination!r} detected in current frame "
                    "(bridge path requires live box detection)"
                )

            # ── [Step2] pick the object (use the same grasp pipeline as
            # _skill_grasp_object_bridge, but inline so the held-detection
            # bookkeeping lines up with the place stage below).
            grasps = self._plan_grasps(banana_fix, label=object_label)
            if not grasps:
                raise SkillError(f"no grasp candidates for {object_label!r}")
            home_q = self.cfg.arm.get("home_q")
            seed = (
                np.asarray(home_q, dtype=float)
                if home_q is not None
                else self.arm.get_state().q
            )
            grasp, _q_pre, _q_grasp = select_grasp(
                grasps,
                self.kin,
                seed,
                max_width_m=self._max_width,
                pregrasp_offset_m=float(
                    self.cfg.grasp.get("pregrasp_offset_m", 0.08)
                ),
                validate=self._grasp_candidate_safety_reason,
                rank_key=self._grasp_rank_key,
                prefilter=self._grasp_prefilter,
                debug_callback=self._on_grasp_debug,
                parallel_jaw_symmetry=bool(
                    self.cfg.grasp.get("parallel_jaw_symmetry", True)
                ),
                wrist_motion_weight=float(
                    self.cfg.grasp.get("wrist_motion_weight", 2.0)
                ),
                deadline=self._active_pick_deadline,
            )
            # Refresh after candidate IK, not merely before remote planning.
            self._refresh_motion_watchdog()
            pre6d, grasp6d = self._grasp_to_6d(grasp)

            # Move to ready_pose first (matches set.py _move_ready).
            if not self._safe_bridge_move(ready6d, duration=ready_dur, where="ready"):
                raise SkillError(
                    f"ready move failed (target="
                    f"{[round(ready6d[0],3), round(ready6d[1],3), round(ready6d[2],3)]}, "
                    f"q_target unsafe or SDK silent fail)"
                )

            bridge_margin_m = float(
                self.cfg.grasp.get("pregrasp_open_margin_m", 0.0)
            )
            self.bridge.open_gripper(
                distance_m=min(
                    float(grasp.width_m) + bridge_margin_m, self._max_width
                )
            )
            if not self._safe_bridge_move(pre6d, duration=2.0, where="pregrasp"):
                raise SkillError(
                    f"pregrasp move failed (target="
                    f"{[round(pre6d[0],3), round(pre6d[1],3), round(pre6d[2],3)]}, "
                    f"q_target unsafe or SDK silent fail)"
                )

            # Open a grasp-exemption cylinder centered on the grasp XY for
            # the descent motion (matches _skill_grasp_object_bridge).
            # Without this, vet_pose rejects TCP z below table_z + clearance,
            # which small/thin objects (banana) always trigger.
            try:
                grasp_xy = (float(grasp.position[0]), float(grasp.position[1]))
                self.arm.harness.allow_grasp_descent(
                    center_xy=grasp_xy,
                    radius=float(self.cfg.grasp.get("exempt_radius_m", 0.07)),
                    z_min=float(self.arm.harness.limits.table_z) - 0.06,
                )
                if not self._safe_bridge_move(
                    grasp6d,
                    duration=float(self.cfg.arm.get("grasp_duration_s", 1.5)),
                    where="grasp",
                ):
                    raise SkillError(
                        f"grasp move failed (target="
                        f"{[round(grasp6d[0],3), round(grasp6d[1],3), round(grasp6d[2],3)]}, "
                        f"q_target unsafe or SDK silent fail)"
                    )
                profile = select_profile(
                    banana_fix.detection.label or object_label, material,
                )
                width_pad_m = float(self.cfg.grasp.get("width_pad_m", 0.015))
                object_span_m = max(float(grasp.width_m) - width_pad_m, 0.005)
                min_hold_width_m = object_span_m * profile.object_close_frac
                close_force = max(
                    profile.effort,
                    float(self.cfg.grasp.get("gripper_close_motion_effort", 0.70)),
                )
                ok = self.bridge.grasp(
                    force=profile.effort, min_width_m=min_hold_width_m,
                    close_force=close_force,
                )
                telemetry = self.bridge.last_grasp_telemetry()
                telemetry.update({
                    "label": object_label,
                    "profile": profile.name,
                    "required_jaw_width_m": float(grasp.width_m),
                    "estimated_object_span_m": object_span_m,
                    "minimum_hold_width_m": min_hold_width_m,
                    "commanded_force": profile.effort,
                })
                self.trace.record_event("gripper_grasp", telemetry)
                if not ok:
                    return {
                        "ok": False,
                        "error": "empty grasp: gripper did not detect contact "
                        f"({telemetry.get('stop_reason', 'unknown')})",
                        "gripper_telemetry": telemetry,
                    }

                # ── [Step3] lift to ready_pose (matches set.py _move_ready) ────
                self.held_object = object_label
                self._held_det_label = banana_fix.detection.label
                self._held_color = detection_color(frame.rgb, banana_fix.detection)
                table_z = float(self.cfg.safety.get("table_z", 0.0))
                self._held_support_offset_m = max(
                    float(grasp.position[2]) - table_z, 0.0
                )
                self._capture_held_xy_extent(banana_fix)
                self._capture_held_reference(banana_fix)
            finally:
                try:
                    self.arm.harness.clear_grasp_exemption()
                except Exception:
                    pass

        # ── [Step3] orientation-preserving carry lift (non-fatal) ───────
        try:
            self._move_held_to_carry(
                duration_s=ready_dur, allow_compact=False,
            )
        except (SkillError, SafetyViolation) as e:
            self.memory.add(
                "note",
                f"held carry lift failed ({e}); proceeding to place anyway",
            )

        # ── [Step4] place above the box (configured z, object-level
        # tilt constraint with searched yaw/tilt -- same search as
        # skill_place_at) ─────────────────────────────────────────────
        box_pos = box_fix.position
        release_z = float(
            self.cfg.grasp.get("bridge_release_height_m", 0.08)
        )
        z_cap = float(self.cfg.grasp.get("topdown_z_max", 0.15)) - 0.005
        release_z = min(release_z, z_cap)
        drop = (
            float(box_pos[0]),
            float(box_pos[1]),
            release_z,
        )
        # Destination sanity check: warn if its observed top is above release
        # height (the banana would collide with the box wall going down).
        # Phase-6 follow-up: refuse + ask for a different box.
        if (
            box_fix.points is not None
            and len(box_fix.points) > 0
        ):
            box_top_z = float(np.asarray(box_fix.points)[:, 2].max())
            if box_top_z > drop[2]:
                self.memory.add(
                    "warning",
                    f"box top_z={box_top_z:.3f} m exceeds configured release_z="
                    f"{drop[2]} m; banana may collide with box wall",
                )

        placed_ok = False
        try:
            # Object-level constraint, same as skill_place_at: search
            # yaw/tilt around the captured level reference instead of
            # freezing the carried wrist rotation. The single-segment
            # carry->drop motion runs above table clearance, so the vet
            # is the plain q_now -> q_release segment (no exemption).
            q_now = np.asarray(self.arm.get_state().q, dtype=float)
            found = self._search_held_target_rotation(
                drop, drop, q_now, vet_from_current=True,
            )
            if found is None:
                raise SkillError(
                    f"place pose unreachable (drop="
                    f"{[round(drop[0],3), round(drop[1],3), round(drop[2],3)]}; "
                    f"object kept level, no yaw/tilt candidate was both "
                    f"IK-solvable and safe)"
                )
            if not self.arm.move_joints(found["low"].q, duration_s=2.0):
                raise SkillError(
                    f"place move failed (drop="
                    f"{[round(drop[0],3), round(drop[1],3), round(drop[2],3)]}, "
                    f"q_target unsafe or SDK silent fail)"
                )

            # ── [Step4b] release and keep jaws open while retracting ─
            # The old 0.8 s async re-close could clamp a small or sticky
            # object again before it cleared the fingers. open_gripper()
            # blocks until position feedback reaches the open target.
            self.bridge.open_gripper(
                timeout=float(self.cfg.grasp.get("gripper_open_timeout_s", 6.0))
            )
            time.sleep(float(self.cfg.grasp.get("release_settle_s", 1.0)))
            release_width = self._gripper_width_frac()
            if release_width is not None and release_width < 0.8:
                raise SkillError(
                    f"release failed: gripper only {release_width:.0%} open"
                )
            placed_ok = True

            # ── [Step5] retract with jaws open, close only when clear ───
            try:
                self._safe_bridge_move(
                    ready6d, duration=ready_dur,
                    where="ready_after_place",
                )
            except (SkillError, SafetyViolation) as e:
                self.memory.add(
                    "note",
                    f"ready_after_place move failed ({e}); object already released",
                )
            else:
                # Stay open at the clear ready pose. Closing here only to
                # reopen at the next pick adds latency and makes the hand
                # appear to re-grasp after every successful placement.
                self.trace.record_event("gripper_left_open_after_place", {
                    "reason": "ready pose is clear; next pick starts open",
                })
        except Exception:
            # Before release, a motion/planning failure means the fruit is
            # still presumed to be in the hand.  Preserve held state and its
            # level reference; clearing it here can make a retry grasp while
            # already loaded. Tactile reconciliation on the next command may
            # clear the state if the fruit actually slipped.
            if not placed_ok and self.held_object:
                self._reconcile_held()
            raise

        # ── reconcile held + beliefs before returning ───────────────────
        self.beliefs.update(
            self._held_det_label or object_label,
            np.array(drop, dtype=float),
            conf=0.8,
            color=self._held_color,
        )
        placed = self.held_object
        self._forget_held_object()

        return {
            "ok": True,
            "picked": placed,
            "placed_at": [drop[0], drop[1], drop[2]],
            "destination": destination,
            "grip_profile": "bridge",
        }

    def _single_localize(self, frame, query: str):
        """Detect + 3D fix on the supplied frame only (no retry / no cache).

        Bypasses :meth:`_localize`'s multi-frame retry and WorldWatcher
        fallback. Used by :meth:`_skill_pick_and_place_bridge` to detect
        the object AND destination on the same frame in one YOLO pass.

        Returns ``None`` on failure (caller decides how to surface).
        """
        r = self._resolve_query(query)
        try:
            return localize_object(
                frame, query, self.detector, self.extrinsics,
                prompts=r["prompts"], spatial_hint=None,
                color=r["color"], near_xyz=r["near_xyz"], vocab=r["vocab"],
                prefer_label=r["prefer_label"],
            )
        except SkillError:
            return None

    def _close_two_stage(
        self,
        profile,
        grasp_width_m: float,
        max_width_m: float,
    ) -> None:
        """Width-aware two-stage gripper close (v2, 2026-08-10).

        Stage 1 (scout): half-travel — fast, gets the jaws into contact range.
        Stage 2 (seat): width-aware — gripper target = grasp.width_m *
        ``profile.object_close_frac`` (default 0.75 per user directive),
        clipped to ``[profile.min_close_frac, profile.max_close_frac]`` as a
        safety bound.

        The MIT-mode ``tau = kp*(q_target - q) - kd*dq + tau_ff`` means the
        motor only pushes until ``q`` reaches the target. If the object is
        wider than the target, the motor stalls at ``q`` ≈ object surface
        and the surface limits the force, not the position target. The
        previous fixed-frac close (90% travel = ~8 mm opening) pushed past
        the object with high effort and crushed a 30 mm 3D-printed banana.

        Goes through ``self.arm.close_gripper_two_stage`` — SafeArm
        delegates to the backend (RebotRSArm uses stall detection;
        MockArm uses the simple two-stage default). The cfg-level
        gripper travel is owned by the arm, not duplicated in
        SkillRuntime.
        """
        from ..grasping.force import compute_close_target
        pos1_frac, pos2_frac = compute_close_target(profile, grasp_width_m, max_width_m)
        # Pin the simulated object surface (width-aware MIT stall).
        # The MockArm honors this so it stops the gripper at the object
        # surface the way the real RebotArm's MIT mode does
        # (``tau = kp*(q_target - q) - kd*dq + tau_ff`` stalls the motor
        # at the surface, not the position target). The real RebotArm
        # has no equivalent setter — this is a mock-only hook to keep
        # the e2e test meaningful. Skip when the caller has explicitly
        # disabled stop simulation (``object_stop_frac is None``) so
        # air-grasp tests still see a no-stop trajectory.
        if (
            hasattr(self.arm, "set_object_close_frac")
            and getattr(self.arm, "object_stop_frac", "<UNSET>") is not None
            and max_width_m > 0
        ):
            surface_close_frac = 1.0 - float(grasp_width_m) / float(max_width_m)
            self.arm.set_object_close_frac(surface_close_frac)
        self.arm.close_gripper_two_stage(
            width_pos_stage1=pos1_frac,
            width_pos_stage2=pos2_frac,
            effort=profile.effort,
        )

    def _adopt_unknown_held(self) -> None:
        """'save it in the box' must work even when the held object was
        never registered (a crashed task, or a human placed something in
        the jaws): a mid-open gripper stall means SOMETHING is in there --
        adopt it as 'object' so place skills can act on it. (Wrist-camera
        visual confirmation is the ROADMAP upgrade.)"""
        if self.held_object:
            return
        wf = self._gripper_width_frac()
        air = float(self.cfg.grasp.get("air_grasp_frac", 0.04))
        if wf is not None and air < wf < 0.9:
            self.held_object = "object"
            self._held_det_label = None
            self.memory.add(
                "note",
                "the jaws are holding something unregistered; "
                "treating it as 'object'",
            )
            # No recorded grasp exists: treat the CURRENT orientation as
            # the level reference, so the identity place candidate equals
            # the old frozen-rotation behavior for adopted holds.
            self._capture_held_reference(source="adopted_current_fk")

    def skill_place_at(self, x: float, y: float, z: float | None = None) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        gcfg = self.cfg.grasp
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        if z is not None:
            release_z = float(z)
            height_source = "explicit"
        elif self._held_support_offset_m is not None:
            support_offset = float(np.clip(
                self._held_support_offset_m,
                float(gcfg.get("place_support_offset_min_m", 0.015)),
                float(gcfg.get("place_support_offset_max_m", 0.060)),
            ))
            release_z = (
                table_z + support_offset
                + float(gcfg.get("place_release_clearance_m", 0.008))
            )
            height_source = "grasp_geometry"
        else:
            release_z = table_z + float(gcfg.get("release_height_m", 0.05))
            height_source = "fixed_fallback"
        self.trace.record_event("place_height_plan", {
            "surface_z": table_z,
            "release_z": release_z,
            "height_source": height_source,
            "held_support_offset_m": self._held_support_offset_m,
        })
        # Strict top-down poses only solve below ~0.15 m on this wrist: a
        # tall destination (the bin walls) must become "release from the
        # ceiling and let it drop", not an unreachable-pose failure.
        z_cap = float(gcfg.get("topdown_z_max", 0.15)) - 0.005
        if release_z > z_cap:
            self.memory.add(
                "note",
                f"place height {release_z:.2f} m is above the wrist's "
                f"top-down ceiling; releasing from {z_cap:.2f} m instead",
            )
            release_z = z_cap
        target = np.array([x, y, release_z])

        q_now = self.arm.get_state().q
        hover = target + np.array([0.0, 0.0, float(gcfg.get("pregrasp_offset_m", 0.12))])
        hover[2] = min(hover[2], z_cap)  # same wrist ceiling as the release
        # Object-level place constraint: keep the held object's up axis
        # (captured at grasp close) within max_object_tilt_deg of world
        # +Z. Yaw about world Z and small tilts are SEARCHED for IK
        # feasibility -- freezing the carried wrist rotation starved IK
        # of exactly that freedom and made place fail on reachable
        # destinations. The vet exemption mirrors the descent exemption
        # opened below (radius 0.07 around the target).
        found = self._search_held_target_rotation(
            target, hover, q_now,
            exempt_radius=0.07, exempt_z_min=release_z - 0.02,
        )
        if found is None:
            raise SkillError(
                f"place pose unreachable at {target.round(3).tolist()} "
                f"(hover {hover.round(3).tolist()}; object kept level "
                f"within {float(gcfg.get('max_object_tilt_deg', 12.0)):.0f} deg, "
                f"no yaw/tilt candidate was both IK-solvable and safe)"
            )
        pre, low = found["pre"], found["low"]

        if not self.arm.move_joints(
            pre.q, duration_s=float(gcfg.get("move_duration_s", 2.5))
        ):
            raise SkillError("did not settle above the place target")
        self.arm.harness.allow_grasp_descent(target[:2], radius=0.07, z_min=release_z - 0.02)
        try:
            descend_dur = float(gcfg.get("descend_duration_s", 2.0))
            if not self.arm.move_joints(low.q, duration_s=descend_dur):
                raise SkillError("did not settle at place pose")
            # 2026-08-13: verify TCP actually reached the target.
            # cartesian_planner's waypoint IK may partially fail; if so the
            # arm stops short of the target but move_cartesian still returns
            # True. Don't trust the boolean — re-FK and check the error.
            final_q = self.arm.get_state().q
            final_tcp = self.kin.fk(final_q)
            place_err = float(np.linalg.norm(final_tcp[:3, 3] - target))
            if place_err > 0.02:  # 2 cm tolerance
                self.memory.add(
                    "note",
                    f"place TCP error {place_err*1000:.0f} mm exceeds 20 mm tolerance "
                    f"(cartesian_planner may have partial-IK-failure); aborting release",
                )
                raise SkillError(
                    f"place failed: TCP off target by {place_err*1000:.0f} mm "
                    f"(target={[round(float(v),3) for v in target]})"
                )
            if self.bridge is not None:
                # On RS hardware the bridge owns the gripper control loop;
                # SafeArm.set_gripper only updates an unused controller target.
                self.bridge.open_gripper(
                    timeout=float(self.cfg.grasp.get("gripper_open_timeout_s", 6.0))
                )
            else:
                self.arm.set_gripper(self._grip_open, effort=0.6)
            time.sleep(float(self.cfg.grasp.get("release_settle_s", 1.0)))
            release_width = self._gripper_width_frac()
            if release_width is not None and release_width < 0.8:
                raise SkillError(
                    f"release failed: gripper only {release_width:.0%} open"
                )
            # From here the object IS placed: reconcile the held state BEFORE
            # the ascent, or an ascent abort leaves held_object latched and a
            # retry descends onto the object we just released.
            placed = self.held_object
            self.beliefs.update(
                # Re-register under the DETECTOR label so the watcher's next
                # fusion merges here instead of creating a query-string ghost.
                self._held_det_label or placed, target, 0.8, color=self._held_color
            )
            self._forget_held_object()
            self.memory.add("action", f"placed {placed!r} at {target.round(3).tolist()}")
            try:
                self.arm.move_cartesian(
                    self.kin.fk(pre.q), self.kin,
                    duration_s=float(gcfg.get("descend_duration_s", 2.0)),
                    planner=self.cartesian_planner,
                )
            except (SkillError, SafetyViolation) as e:
                # The place already happened; report success and let the
                # caller's move_home park the arm.
                self.memory.add("note", f"placed, but the ascent aborted: {e}")
        finally:
            self.arm.harness.clear_grasp_exemption()
        return {"placed": placed, "at": [round(float(v), 3) for v in target]}

    def skill_place_on_object(self, label: str) -> dict:
        self._adopt_unknown_held()
        if not self.held_object:
            raise SkillError("not holding anything")
        belief = self.beliefs.find(label)
        if belief is None:
            frame, fix = self._localize(label)
            self.beliefs.update(
                fix.detection.label or label, fix.position, fix.detection.conf,
                extent=fix.extent, top_z=float(fix.points[:, 2].max()),
                color=detection_color(frame.rgb, fix.detection),
            )
            belief = self.beliefs.find(label)
        # Use the actually observed highest point of the target, never OBB
        # extents (those are eigenvalue-ordered, not axis-aligned).
        top = belief.top_z if belief.top_z is not None else float(belief.position[2])
        if self._held_support_offset_m is not None:
            support_offset = float(np.clip(
                self._held_support_offset_m,
                float(self.cfg.grasp.get("place_support_offset_min_m", 0.015)),
                float(self.cfg.grasp.get("place_support_offset_max_m", 0.060)),
            ))
            drop = (
                top + support_offset
                + float(self.cfg.grasp.get("place_release_clearance_m", 0.008))
            )
        else:
            drop = top + float(self.cfg.grasp.get("release_clearance_m", 0.06))
        return self.skill_place_at(
            float(belief.position[0]), float(belief.position[1]), drop
        )

    @staticmethod
    def _parse_relative_destination(destination: str | None) -> tuple[str, str] | None:
        """Parse a relation; base +Y is left and base -Y is right."""
        if not destination:
            return None
        text = " ".join(str(destination).strip().split())
        match = re.fullmatch(
            r"(?:to\s+the\s+)?(left|right)(?:\s+side)?\s+of\s+(?:the\s+)?(.+)",
            text, flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), canonical_object_name(match.group(2).strip())
        match = re.fullmatch(r"(.+?)(左边|左侧|右边|右侧)", text)
        if match:
            side = "left" if match.group(2).startswith("左") else "right"
            return side, canonical_object_name(match.group(1).strip())
        return None

    def _relative_place_xy(self, reference: str, side: str) -> tuple[float, float]:
        """Re-localize reference and offset along the robot base Y axis."""
        _frame, fix = self._localize(reference)
        reference_source = "fresh_detection"
        reference_position = np.asarray(fix.position, dtype=float).copy()
        reference_points = np.asarray(getattr(fix, "points", []), dtype=float)
        min_offset = float(self.cfg.grasp.get("relative_place_offset_m", 0.08))
        clearance = float(self.cfg.grasp.get("relative_place_clearance_m", 0.025))
        ref_points = reference_points
        if ref_points.ndim == 2 and ref_points.shape[0] >= 3:
            ref_lo, ref_hi = np.percentile(ref_points[:, 1], [5.0, 95.0])
            reference_half_y = max(float(ref_hi - ref_lo) * 0.5, 0.0)
        else:
            reference_half_y = 0.0
        held_half_y = (
            float(self._held_xy_half_extent_m[1])
            if self._held_xy_half_extent_m is not None
            else 0.0
        )
        geometric_offset = reference_half_y + held_half_y + clearance
        offset = max(min_offset, geometric_offset)
        max_offset = float(self.cfg.grasp.get("relative_place_max_offset_m", 0.16))
        offset = min(offset, max_offset)
        target = reference_position.copy()
        target[1] += offset if side == "left" else -offset
        limits = self.arm.harness.limits
        workspace_min = np.asarray(limits.workspace_min, dtype=float)
        workspace_max = np.asarray(limits.workspace_max, dtype=float)
        if np.any(target[:2] < workspace_min[:2]) or np.any(target[:2] > workspace_max[:2]):
            raise SkillError(
                f"relative place target {target[:2].round(3).tolist()} is outside "
                f"workspace XY [{workspace_min[:2].tolist()}, {workspace_max[:2].tolist()}]"
            )
        self.trace.record_event("relative_place_plan", {
            "frame": "robot_base", "convention": "+Y=left, -Y=right",
            "reference": reference,
            "reference_source": reference_source,
            "reference_xyz": reference_position.round(4).tolist(),
            "side": side, "offset_m": offset,
            "minimum_offset_m": min_offset,
            "clearance_m": clearance,
            "reference_half_y_m": round(reference_half_y, 4),
            "held_half_y_m": round(held_half_y, 4),
            "geometric_offset_m": round(geometric_offset, 4),
            "target_xy": target[:2].round(4).tolist(),
        })
        return float(target[0]), float(target[1])



    def skill_pick_and_place(
        self,
        object: str,
        destination: str | None = None,
        material: str | None = None,
    ) -> dict:
        """One-call pick-and-place that PERSISTS: resolve (color queries
        work) -> grasp -> place on the destination object or the configured
        drop zone -> home. Both stages keep retrying with fresh perception
        (re-home, re-scan, re-plan -- GraspGen-X sampling is stochastic, so
        every retry is a genuinely new grasp) until they succeed or the
        persistence budget (grasp.persist_seconds / grasp.max_pick_attempts)
        runs out. No LLM in the loop; this is the reflex the web chat calls
        for "pick and place pink object"."""
        object = canonical_object_name(object) or object
        destination = canonical_destination(destination)
        self._reconcile_held()
        already_held = None
        if self.held_object:
            hq, oq = self.held_object.lower(), str(object).lower()
            if hq in oq or oq in hq:
                # The jaws already hold what was asked for (a previous task
                # grasped it and died before placing): skip straight to the
                # place stage instead of refusing.
                already_held = self.held_object
                self.memory.add(
                    "note",
                    f"already holding {already_held!r} -- going straight to the place stage",
                )
            else:
                raise SkillError(
                    f"already holding {self.held_object!r}; place it first"
                )
        t0 = time.monotonic()
        timings: dict[str, float] = {}
        self._failed_grasp_signatures = []
        gcfg = self.cfg.grasp
        max_attempts = max(int(gcfg.get("max_pick_attempts", 8)), 1)
        deadline = t0 + float(gcfg.get("persist_seconds", 120.0))
        self._active_pick_deadline = deadline
        relative_destination = self._parse_relative_destination(destination)

        # Bridge path: when rebot_grasp bridge is wired AND a destination
        # label is given (so the place target is detectable, not a fixed
        # drop_zone coordinate), mirror set.py's flow:
        #   same-frame detect → grasp → carry → place(object kept level,
        #   yaw/tilt searched within max_object_tilt_deg, configured z)
        #   → release_with_async_close → home
        # Without a destination we fall back to the legacy flow so
        # ``pick_and_place`` with no destination still uses the drop_zone.
        if self.bridge is not None and destination is not None and relative_destination is None:
            try:
                result = self._skill_pick_and_place_bridge(
                    object, destination, already_held=already_held, material=material,
                )
            except (SkillError, SafetyViolation, RuntimeError):
                if not self.arm.harness.estopped:
                    try:
                        self.skill_move_home(park_gripper=False)
                    except (SkillError, SafetyViolation, RuntimeError) as home_error:
                        self.memory.add(
                            "warning",
                            f"pick_and_place failed and return home also failed: {home_error}",
                        )
                raise
            if not result.get("ok", True) and not self.arm.harness.estopped:
                try:
                    self.skill_move_home(park_gripper=False)
                except (SkillError, SafetyViolation, RuntimeError) as home_error:
                    self.memory.add(
                        "warning",
                        f"pick_and_place failed and return home also failed: {home_error}",
                    )
            return result

        grasp = (
            {"held": already_held, "grip_verified": None, "grip_profile": None}
            if already_held
            else None
        )
        placed = None
        last_err = "unknown"
        place_err = "unknown"
        attempt = 0
        p_attempt = 0
        tp = t0
        while placed is None:
            # ── grasp stage (skipped when re-entering after a mid-carry slip
            # left something verified in the jaws -- cannot happen today, but
            # the guard keeps the loop honest) ─────────────────────────────
            while grasp is None and attempt < max_attempts:
                attempt += 1
                if attempt == 1:
                    self.memory.add(
                        "note",
                        f"picking up {object!r} (attempt 1/{max_attempts}; "
                        "I will keep trying until it works or the budget runs out)",
                    )
                if attempt > 1:
                    if time.monotonic() > deadline:
                        attempt -= 1  # this attempt never ran
                        break
                    self.memory.add(
                        "note",
                        f"not giving up: attempt {attempt}/{max_attempts} on "
                        f"{object!r} -- re-homing, re-scanning, planning a fresh grasp",
                    )
                    try:  # clear the camera view so the re-scan actually sees
                        self.skill_move_home(park_gripper=False)
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                self._set_grasp_debug_attempt(attempt, max_attempts)
                tg = time.monotonic()
                try:
                    self._pick_and_place_in_progress = True
                    try:
                        res = self.skill_grasp_object(object, material=material)
                    finally:
                        self._pick_and_place_in_progress = False
                except (SkillError, SafetyViolation) as e:
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                timings[f"grasp_attempt{attempt}_s"] = round(time.monotonic() - tg, 2)
                if res.get("ok", True) and res.get("held"):
                    grasp = res
                    break
                last_err = str(res.get("error", "grasp failed"))
                failed_debug = self.grasp_debug_snapshot()
                execution_fault = any(token in last_err.lower() for token in (
                    "did not reach", "post-move", "sdk", "watchdog",
                    "e-stop", "e-stopped", "deadline exceeded",
                ))
                if failed_debug.get("base_xyz") and not execution_fault:
                    self._failed_grasp_signatures.append(failed_debug)
                self.memory.add("outcome", f"pick attempt {attempt} failed: {last_err[:100]}")
                if self.arm.harness.estopped:
                    last_err += " (e-stop latched; not retrying)"
                    break
                # Re-planning can help a bad grasp candidate, but it cannot
                # repair a controller/feedback/watchdog failure. Retrying in
                # that case only reruns GraspGenX and expensive IK until the
                # global deadline expires.
                if execution_fault:
                    last_err += " (execution fault; not retrying grasp planning)"
                    break
                # Fail fast on errors persistence cannot cure: an object the
                # world model has NEVER seen after full re-scans is a typo or
                # simply not on the table -- burning 2 minutes on it reads as
                # a hang at a live booth.
                if (
                    attempt >= 2
                    and "no detections" in last_err
                    and self.beliefs.find(object) is None
                ):
                    last_err += " (never seen after re-scans; giving up early)"
                    break
            if grasp is None:
                # Never leave the arm hanging mid-pose over the table after a
                # failed attempt -- park it (best effort, nothing is held).
                try:
                    self.skill_move_home(park_gripper=False)
                except (SkillError, SafetyViolation, RuntimeError):
                    pass
                return {
                    "ok": False, "stage": "grasp",
                    "error": f"grasp failed after {attempt} attempts "
                             f"({round(time.monotonic() - t0, 1)}s): {last_err}",
                    "suggestion": "check world_state / camera_snapshot; the object may be unreachable or mis-detected",
                }

            # ── place stage ────────────────────────────────────────────────
            tp = time.monotonic()
            while p_attempt < max_attempts:
                p_attempt += 1
                if p_attempt > 1:
                    if time.monotonic() > deadline:
                        p_attempt -= 1
                        break
                    self.memory.add(
                        "note",
                        f"still holding {object!r}: place attempt "
                        f"{p_attempt}/{max_attempts} -- re-locating the target",
                    )
                    try:  # carry it home first: clears the view of the arm
                        self.skill_move_home()  # + held object for the re-scan
                    except (SkillError, SafetyViolation):
                        pass
                    self._reobserve()
                try:
                    if relative_destination:
                        side, reference = relative_destination
                        x, y = self._relative_place_xy(reference, side)
                        res = self.skill_place_at(x, y)
                    elif destination:
                        res = self.skill_place_on_object(destination)
                    else:
                        dz = gcfg.get("drop_zone", [0.30, -0.20])
                        res = self.skill_place_at(float(dz[0]), float(dz[1]))
                    if not res.get("ok", True):
                        raise SkillError(str(res.get("error", "place failed")))
                    placed = res
                    break
                except (SkillError, SafetyViolation) as e:
                    place_err = f"{type(e).__name__}: {e}"
                    self.memory.add(
                        "outcome", f"place attempt {p_attempt} failed: {place_err[:100]}"
                    )
                    if self.arm.harness.estopped:
                        place_err += " (e-stop latched; not retrying)"
                        break
                    if not self.held_object:
                        break  # slipped mid-carry: placing again is pointless
            if placed is None:
                can_regrasp = (
                    not self.held_object
                    and not self.arm.harness.estopped
                    and time.monotonic() < deadline
                    and attempt < max_attempts
                )
                if not can_regrasp:
                    return {
                        "ok": False, "stage": "place",
                        "error": f"place failed after {p_attempt} attempts: {place_err}",
                        "note": (
                            f"still holding {object!r}; try place_at with explicit coordinates"
                            if self.held_object
                            else f"{object!r} slipped while carrying; grasp budget exhausted"
                        ),
                        "grip_verified": grasp.get("grip_verified"),
                    }
                self.memory.add(
                    "note",
                    f"{object!r} slipped while I was carrying it -- starting over",
                )
                grasp = None  # back to the grasp stage within the same budget
        timings["place_s"] = round(time.monotonic() - tp, 2)

        if bool(self.cfg.grasp.get("home_after_place", True)):
            try:  # clear the camera view for the next command; best effort
                self.skill_move_home()
            except (SkillError, SafetyViolation):
                pass
        total = round(time.monotonic() - t0, 2)
        self.memory.add(
            "action",
            f"pick_and_place {object!r} -> {destination or 'drop zone'} in {total}s",
        )
        return {
            "picked": object,
            "placed_at": placed.get("at"),
            "destination": destination or "drop zone",
            "grip_profile": grasp.get("grip_profile"),
            "grip_verified": grasp.get("grip_verified"),
            "grasp_attempts": attempt,
            "place_attempts": p_attempt,
            "duration_s": total,
            "timings": timings,
        }

    # ── social / audience skills (deterministic, harness-gated) ─────────

















    def skill_hand_action(self, action: str) -> dict:
        """Run a named dexterous-hand primitive at the current arm pose."""
        hand = getattr(self.arm, "end_effector", None)
        if hand is None:
            raise SkillError("hand_action requires the linkerhand_o6 arm profile")
        actions = {
            "open": hand.open,
            "close": hand.close_hand,
            "power_grasp": hand.power_grasp,
            "pinch": hand.pinch,
        }
        if action not in actions:
            raise SkillError(f"unknown hand action {action!r}")
        result = actions[action]()
        # Opening the O6 hand physically releases the held object. Keep the
        # runtime state in sync so the next grasp is not rejected as if the
        # old object were still in the fingers.
        if action == "open" and self.held_object:
            released = self.held_object
            self.memory.add("action", f"released {released!r}")
            self._forget_held_object()
        return result


    def skill_open_gripper(self) -> dict:
        # On the RS rig WrcGripper owns the gripper motor's 500 Hz command
        # loop. Writing RebotArmEndPose._gripper_target via SafeArm bypasses
        # that owner (the controller intentionally has _has_gripper=False),
        # so the old code reported success without moving the jaws.
        if self.bridge is not None:
            self.bridge.open_gripper()
        else:
            self.arm.set_gripper(self._grip_open, effort=0.8)
        if self.held_object:
            self.memory.add("action", f"released {self.held_object!r}")
            self._forget_held_object()
        wf = self._gripper_width_frac()
        return {
            "gripper": "open",
            "open_frac": round(wf, 2) if wf is not None else "unknown",
        }

    def skill_close_gripper(self) -> dict:
        if self.bridge is not None:
            self.bridge.close_gripper()
        else:
            self.arm.set_gripper(self._grip_closed, effort=0.0)
        wf = self._gripper_width_frac()
        return {"gripper": "closed", "open_frac": round(wf, 2) if wf is not None else "unknown"}

    def _move_held_to_carry(
        self, duration_s: float = 2.0, *, allow_compact: bool = True,
    ) -> dict:
        """Lift safely, then retract to an object-level compact carry pose.

        The first leg searches a gravity-constrained lift endpoint, with the old
        fixed-rotation Cartesian lift retained only as a fallback. Once clear
        of the table, configured compact XYZ targets are reached through
        bounded XY segments; every segment is re-solved from measured joints with the same gravity-relative object
        constraint as place: world-Z yaw is free, tilt is bounded, and the
        exact vetted IK solution is executed.  If no compact target is
        feasible, the safe vertical-lift pose is retained.
        """
        q_now = np.asarray(self.arm.get_state().q, dtype=float)
        T_now = np.asarray(self.kin.fk(q_now), dtype=float)
        T_target = T_now.copy()
        table_z = float(self.cfg.safety.get("table_z", 0.0))
        carry_z = float(self.cfg.arm.get("held_carry_z_m", table_z + 0.14))
        T_target[2, 3] = max(float(T_now[2, 3]), carry_z)
        lifted = not np.allclose(T_target, T_now, atol=1e-5)
        lift_method = "skipped"
        if lifted:
            # Search a gravity-constrained endpoint first. The old strict
            # fixed-rotation Cartesian CLIK lift can be singular even when a
            # nearby yaw/tilt-preserving joint path is safe and reachable.
            found_lift = self._search_held_carry_rotation(
                T_target[:3, 3], q_now,
            )
            if found_lift is not None:
                if self.arm.move_joints(
                    found_lift["low"].q, duration_s=duration_s
                ):
                    lift_method = "gravity_constrained_joint"
                else:
                    found_lift = None
            if found_lift is None:
                try:
                    settled = self.arm.move_cartesian(
                        T_target, self.kin, duration_s=duration_s,
                        planner=self.cartesian_planner,
                    )
                except RuntimeError as exc:
                    settled = False
                    self.trace.record_exception("held carry CLIK fallback", exc)
                if settled:
                    lift_method = "fixed_rotation_clik"
                else:
                    self.trace.record_event("held_carry_unreachable", {
                        "from_xyz": T_now[:3, 3].tolist(),
                        "target_xyz": T_target[:3, 3].tolist(),
                        "action": "hold_position",
                        "harness_violations": list(
                            self.arm.harness.violations[-3:]
                        ),
                    })
                    return {
                        "motion": "held_position",
                        "carry_unreachable": True,
                        "tcp_xyz_m": T_now[:3, 3].round(4).tolist(),
                        "compact": False,
                    }
        self.trace.record_event("held_carry_move", {
            "from_xyz": T_now[:3, 3].tolist(),
            "to_xyz": T_target[:3, 3].tolist(),
            "rotation_preserved": lift_method == "fixed_rotation_clik",
            "method": lift_method,
        })

        compact_targets = (
            self.cfg.arm.get("held_carry_xyz_candidates_m", [])
            if allow_compact else []
        )
        if not allow_compact:
            self.trace.record_event("held_compact_carry_skipped", {
                "reason": "pick_and_place destination already known",
                "lift_xyz": T_target[:3, 3].round(4).tolist(),
            })
        if compact_targets:
            max_xy_step = max(float(
                self.cfg.arm.get("held_carry_max_xy_step_m", 0.10)
            ), 0.02)
            for candidate_index, xyz in enumerate(compact_targets):
                target = np.asarray(xyz, dtype=float).reshape(3).copy()
                # A compact carry target must never undo the clearance lift.
                target[2] = max(float(target[2]), float(T_target[2, 3]))
                q_stage = np.asarray(self.arm.get_state().q, dtype=float)
                start_xyz = np.asarray(self.kin.fk(q_stage), dtype=float)[:3, 3]
                xy_distance = float(np.linalg.norm(target[:2] - start_xyz[:2]))
                segment_count = max(int(np.ceil(xy_distance / max_xy_step)), 1)
                waypoints = [
                    start_xyz + (i / segment_count) * (target - start_xyz)
                    for i in range(1, segment_count + 1)
                ]
                route_ok = True
                chosen = None
                for segment_index, waypoint in enumerate(waypoints, start=1):
                    found = self._search_held_carry_rotation(
                        waypoint, q_stage,
                    )
                    if found is None:
                        route_ok = False
                        self.trace.record_event("held_carry_segment_failed", {
                            "candidate_index": candidate_index,
                            "segment_index": segment_index,
                            "segment_count": segment_count,
                            "target_xyz": waypoint.round(4).tolist(),
                            "reason": "no reachable safe orientation",
                        })
                        break
                    segment_duration = max(
                        0.8, float(duration_s) / float(segment_count)
                    )
                    if not self.arm.move_joints(
                        found["low"].q, duration_s=segment_duration
                    ):
                        route_ok = False
                        self.trace.record_event("held_carry_segment_failed", {
                            "candidate_index": candidate_index,
                            "segment_index": segment_index,
                            "segment_count": segment_count,
                            "target_xyz": waypoint.round(4).tolist(),
                            "reason": "motion did not settle",
                        })
                        break
                    chosen = found
                    q_stage = np.asarray(self.arm.get_state().q, dtype=float)
                    self.trace.record_event("held_carry_segment_move", {
                        "candidate_index": candidate_index,
                        "segment_index": segment_index,
                        "segment_count": segment_count,
                        "target_xyz": waypoint.round(4).tolist(),
                        "yaw_deg": found["yaw_deg"],
                        "tilt_deg": found["tilt_deg"],
                        "score": found["score"],
                    })
                if not route_ok or chosen is None:
                    continue
                self.trace.record_event("held_compact_carry_move", {
                    "target_xyz": target.tolist(),
                    "segments": segment_count,
                    "yaw_deg": chosen["yaw_deg"],
                    "tilt_deg": chosen["tilt_deg"],
                    "tilt_axis_deg": chosen["tilt_axis_deg"],
                    "score": chosen["score"],
                })
                return {
                    "motion": "compact_carry",
                    "tcp_xyz_m": target.round(4).tolist(),
                    "segments": segment_count,
                    "yaw_deg": chosen["yaw_deg"],
                    "tilt_deg": chosen["tilt_deg"],
                }
            self.trace.record_event("held_compact_carry_fallback", {
                "reason": "no configured segmented carry route was reachable and safe",
                "lift_xyz": T_target[:3, 3].tolist(),
            })

        return {
            "motion": "lifted" if lifted else "skipped",
            "tcp_z_m": float(T_target[2, 3]),
            "compact": False,
        }



    def skill_move_home(self, *, park_gripper: bool = True) -> dict:
        self._reconcile_held()
        if self.held_object:
            self._refresh_motion_watchdog(allow_recent_s=1.0)
            carry = self._move_held_to_carry(
                duration_s=float(self.cfg.arm.get("held_carry_duration_s", 2.0))
            )
            return {
                "at": ("held_position" if carry.get("carry_unreachable") else "held_carry"),
                "gripper": "holding",
                "held": self.held_object,
                **carry,
            }
        home = np.asarray(self.cfg.arm.get("home_q", [0.0, -0.5, -0.9, 0.0, 0.6, 0.0]), dtype=float)
        # The first command may arrive before WorldWatcher has emitted its
        # first heartbeat. Refresh from a live camera frame so this reflex
        # does not fail over to the much slower LLM path.
        self._refresh_motion_watchdog(allow_recent_s=1.0)
        q_now = np.asarray(self.arm.get_state().q, dtype=float).reshape(-1)
        home_tol = float(self.cfg.arm.get("home_noop_tol_rad", 0.02))
        already_home = (
            q_now.size >= home.size
            and float(np.max(np.abs(q_now[:home.size] - home))) <= home_tol
        )
        if not already_home and not self.arm.move_joints(home, duration_s=3.0):
            raise SkillError("did not settle at home")
        # Park empty jaws closed so their external guards cannot sweep into
        # nearby fixtures at the beginning of the next task. Never force an
        # object-aware grasp to zero: a failed place may return home while
        # still holding a soft object.
        gripper = "holding" if self.held_object else "unchanged"
        if park_gripper and not self.held_object:
            end_effector = getattr(self.arm, "end_effector", None)
            if end_effector is not None:
                end_effector.open()
                gripper = "open"
            elif self.bridge is not None:
                self.bridge.close_gripper(
                    timeout=float(self.cfg.grasp.get("gripper_close_timeout_s", 4.0))
                )
                gripper = "closed"
            else:
                self.arm.set_gripper(self._grip_closed, effort=0.0)
                gripper = "closed"
        return {
            "at": "home",
            "gripper": gripper,
            "motion": "skipped" if already_home else "moved",
        }

    def skill_wave(self) -> dict:
        """Wave hello: oscillate forearm at home position.

        Stays at home_q (TCP at [0.25, 0, 0.35] per FK), oscillates joint 3
        (forearm roll) by ±0.4 rad for 3 cycles (~3 s total motion). All
        intermediate poses stay within the safety workspace AABB and URDF
        joint limits (joint 3 swings between -0.85 and -0.05 rad; URDF
        limit is ±π - 0.02 rad).

        Added 2026-09-03 to fill the dead-reference in _MOTION_SKILLS
        (and reflex.py's "wave" regex, which had been dispatching to
        skill_wave since 2026-08-13 but found no implementation).

        Returns:
            {"waved": True, "n_cycles": 3, "amplitude_rad": 0.4}
        """
        home = np.asarray(
            self.cfg.arm.get("home_q", [0.0, 0.1684, 0.6226, -0.4543, 0.0, 0.0]),
            dtype=float,
        )
        amplitude = 0.4  # rad on joint 3 (forearm roll)
        n_cycles = 3
        half_period_s = 0.5

        # Park at home first (idempotent — SafeArm.move_joints is a no-op
        # if already at target; otherwise moves us into the safe starting pose).
        if not self.arm.move_joints(home, duration_s=2.0):
            raise SkillError("could not reach home before waving")

        for i in range(n_cycles * 2):
            q = home.copy()
            q[3] = home[3] + (amplitude if i % 2 == 0 else -amplitude)
            if not self.arm.move_joints(q, duration_s=half_period_s):
                raise SkillError(
                    f"wave trajectory failed at step {i + 1}/{n_cycles * 2}"
                )

        if not self.arm.move_joints(home, duration_s=2.0):
            raise SkillError("could not return home after waving")

        return {
            "waved": True,
            "n_cycles": n_cycles,
            "amplitude_rad": amplitude,
        }

    def skill_teach_record(self, slot: int) -> dict:
        """Enter master-slave teleoperation, capture trajectory.

        Long-running skill: blocks waiting for user to press q (start/stop
        recording) or Esc (cancel). Added to _MOTION_SKILLS so WorldWatcher
        pauses during motion. Returns dict with frame_count/duration_s.

        Bypasses SafeArm (uses self.arm.raw → vendored RebotArm) because teach
        needs an external-event-triggered control loop, which SafeArm lacks.
        Master arm is PiPER Mate via fashionstar-uart-sdk.
        """
        from pathlib import Path
        from . import teach as _teach_mod
        from .master_arm import PiPERMateArm

        if not isinstance(slot, int) or slot < 1 or slot > _teach_mod.NUM_SLOTS:
            return {
                "ok": False,
                "error": f"slot must be int 1-{_teach_mod.NUM_SLOTS}, got {slot!r}",
            }

        try:
            raw_arm = self.arm.raw          # LazyArm materializes RebotRSArm
            slave = getattr(raw_arm, "_arm", None)
        except Exception as e:
            return {"ok": False, "error": f"arm not available: {type(e).__name__}: {e}"}

        if slave is None:
            return {"ok": False, "error": "SDK arm not connected"}

        master = PiPERMateArm(port="/dev/ttyUSB0", gripper_exist=True)
        try:
            master.open()
        except Exception as e:
            return {"ok": False, "error": f"master arm open failed: {type(e).__name__}: {e}"}

        recordings_dir = Path(__file__).resolve().parents[2] / "data" / "recordings"
        try:
            return _teach_mod.record_real(master, slave, slot, recordings_dir)
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}
        finally:
            try:
                master.close()
            except Exception:
                pass

    def skill_teach_replay(self, slot: int) -> dict:
        """Replay recorded trajectory; return arm to motor mechanical zero.

        Returns dict with replayed_to_home=True on success. Uses
        RebotArmEndPose.move_joints_traj + safe_home internally via teach.py.
        """
        from pathlib import Path
        from . import teach as _teach_mod

        if not isinstance(slot, int) or slot < 1 or slot > _teach_mod.NUM_SLOTS:
            return {
                "ok": False,
                "error": f"slot must be int 1-{_teach_mod.NUM_SLOTS}, got {slot!r}",
            }

        try:
            raw_arm = self.arm.raw
            slave = getattr(raw_arm, "_arm", None)
        except Exception as e:
            return {"ok": False, "error": f"arm not available: {type(e).__name__}: {e}"}

        if slave is None:
            return {"ok": False, "error": "SDK arm not connected"}

        recordings_dir = Path(__file__).resolve().parents[2] / "data" / "recordings"
        try:
            return _teach_mod.replay_real(slave, slot, recordings_dir)
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}





    def skill_task_done(self, success: bool, summary: str) -> dict:
        if isinstance(success, str):  # schema-lax backends send "false"
            success = success.strip().lower() in ("true", "yes", "1")
        return {"ok": True, "task_complete": True, "success": bool(success), "summary": summary}


def _short(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


TOOL_SPECS: list[dict] = [
    {
        "name": "list_objects",
        "description": "List known objects (visible + remembered) with positions. Call first.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "recall_memory",
        "description": "Recall a remembered objects last known position and matching events.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "pick_and_place",
        "description": "Complete pick-and-place: detect+grasp+place atomically. For 'put X in Y', 'pick up X then place Y', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string", "description": "what to pick, e.g. 'banana', 'red cube'"},
                "destination": {"type": "string", "description": "Named destination ('box') or base-frame relation ('left of lime', 'right of lime'); +Y is left and -Y is right. Omit for default drop zone."},
                "material": {"type": "string", "enum": ["rigid", "fragile", "soft", "deformable", "slippery", "heavy"]},
            },
            "required": ["object"],
        },
    },
    {
        "name": "hand_action",
        "description": "Operate the LinkerHand O6 at its current arm pose: open, close to the calibrated fist pose, tactile power grasp, or thumb-index pinch.",
        "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close", "power_grasp", "pinch"]}}, "required": ["action"]},
    },
    {
        "name": "move_home",
        "description": "Park the arm at home pose. Use to reset between tasks or after failures.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "wave",
        "description": (
            "Wave hello: oscillate the forearm at home position to greet. "
            "Use for 'wave hello' / 'say hi' / '打招呼' / '挥手'. No parameters."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "task_done",
        "description": "Declare task finished with success bool and one-paragraph summary. ALWAYS call last.",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["success", "summary"],
        },
    },
    {
        "name": "teach_record",
        "description": (
            "WHAT: Enter master-slave teleoperation mode. PiPER Mate master arm "
            "(operator-controlled via /dev/ttyUSB0) drives the reBot B601-RS slave "
            "arm in POS_VEL mode. Capture joint positions at 30Hz. User presses q "
            "to start/stop recording into slot N (Enter also works as fallback). "
            "Long-running (waits for q or Esc). "
            "WHEN: User asks to 'record an action' / 'save a motion' / 'teach mode' / "
            "'record slot N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "which slot to save the recording into",
                },
            },
            "required": ["slot"],
        },
    },
    {
        "name": "teach_replay",
        "description": (
            "WHAT: Read the saved slot{N}.json and execute the recorded trajectory "
            "via move_joints_traj. After playback, return the arm to motor mechanical "
            "zero (np.zeros(6)) via safe_home. "
            "WHEN: User asks to 'replay the action' / 'play back' / 'repeat it' / "
            "'replay slot N'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "which slot to replay",
                },
            },
            "required": ["slot"],
        },
    },
]
