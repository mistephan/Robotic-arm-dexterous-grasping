"""CLI: hand-eye calibration for the Orbbec top camera.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

Architecture ported from rebot_grasp/scripts/calibrate_top_camera.py:
- Clean single-loop state machine for auto pose traversal (``tick_auto``
  pattern) replaces the previous per-pose nested loop -- ~500 LOC
  collapses to ~150 LOC.
- Optional ``--gravity-comp`` flag enables MIT-mode gravity compensation
  so the operator can hand-guide the arm during manual capture (uses
  vendored ``RebotArm`` directly, matching the pattern in
  ``skills/teach.py`` -- see CLAUDE.md "Motion safety path").
- Reuses wrc_demo's ``OrbbecCamera``, ``Kinematics``, ``joint_solve_restart``
  + ``save_calibration`` (no rebot_grasp dependency).

Sub-commands:

- ``--list``: enumerate Orbbec devices (no frames captured).
- ``--bind <role>``: write the role -> serial binding into
  ``data/calibration/cameras.json`` (also updates profile YAMLs).
- ``--verify --npz <file> --known-points <csv>``: cross-check a saved
  calibration against operator-measured table points.
- ``--dry-run``: synthesize samples and exercise solve/persist pipeline
  end-to-end without hardware (smoke test).
- (default) collect a calibration dataset. ``--manual`` waits for ENTER
  after each pose (operator moves arm by hand); ``--gravity-comp`` adds
  MIT-mode gravity compensation so the arm floats during manual capture.

Output: ``data/calibration/hand_eye_top.json`` (schema v1) + ``.npz``
sidecar -- loadable by ``scripts/flip_hand_eye.py`` and
``scripts/verify_relative_motion.py``.

Run via ``python -m wrc_demo.scripts.calib_top_orbbec ...`` (editable
install means no PYTHONPATH gymnastics needed).
"""

from __future__ import annotations

import argparse
import atexit
import json
import queue
import signal as _signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from ..calibration import (
    ArUcoSession,
    CamerasBinding,
    CameraBinding,
    CalibrationResult,
    CalibrationSample,
    SCHEMA_VERSION,
    joint_solve_restart,
    load_calibration,
    save_calibration,
)
from ..config import PACKAGE_ROOT
from ..safety.harness import Kinematics
from ..perception.orbbec_camera import OrbbecCamera, list_orbbec_devices


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Stable directory layout for calibration artefacts (PLAN.md §4)
CALIB_DIR = PACKAGE_ROOT / "data" / "calibration"
ARM_PROFILE_DEFAULT = "rebot_rs"
URDF_DEFAULT = (
    PACKAGE_ROOT
    / "assets" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
)

# Headless / SSH env fallback: cv2.namedWindow always raises. We detect
# once at import and fall back to per-frame disk snapshots under
# /tmp/calib_frames/ so the operator can still verify the pipeline.
SNAP_DIR = Path("/tmp/calib_frames")
SNAP_EVERY = 10


def _detect_gui_capable() -> bool:
    """True if cv2.namedWindow works (i.e. we have a real display server)."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


# ──────────────────────────────────────────────────────────────────────
# Preset calibration poses (from rebot_grasp/scripts/calibrate_top_camera.py)
# Cartesian pose sequence around the workspace center (0.30, -0.05, 0.28)
# in the arm's base frame. Pitch is kept small so the gripper stays near
# horizontal -- the ArUco marker (Solution B in 04b: pasted on the
# gripper top with normal +Z) stays face-on to the top camera.
# Source: rebot_grasp/scripts/calibrate_top_camera.py:46-91, copied
# verbatim (rebot_grasp notes the same source: wrc_demo calib_top_orbbec.py,
# so both files have always held identical pose data -- this rewrite
# adopts rebot_grasp's provenance comment for clarity).
# ──────────────────────────────────────────────────────────────────────
TOP_CALIB_POSES: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    # 1. Roll sweep at base (0.30, -0.05, 0.28)
    ((0.30, -0.05, 0.28), ( 0.00, 0.00, -0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.00,  0.50)),
    # 2. Yaw sweep
    ((0.30, -0.05, 0.28), (-0.60, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), (-0.30, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.30, 0.00,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.60, 0.00,  0.00)),
    # 3. Pitch lift + roll combos
    ((0.30, -0.05, 0.28), ( 0.00, 0.20,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.20,  0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.20, -0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.20, -1.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.40,  0.00)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.40,  0.50)),
    ((0.30, -0.05, 0.28), ( 0.00, 0.40, -0.50)),
    # 4. X / Y / Z translation sweep around the center
    ((0.25, -0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.25, -0.05, 0.28), ( 0.00, 0.20,  0.50)),
    ((0.35, -0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.35, -0.05, 0.28), ( 0.00, 0.20, -0.50)),
    ((0.30, -0.10, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.10, 0.28), ( 0.20, 0.20,  0.50)),
    ((0.30, -0.05, 0.30), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.32), ( 0.00, 0.00,  0.00)),
    ((0.30,  0.00, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30,  0.00, 0.28), ( 0.00, 0.20,  0.50)),
    ((0.30,  0.05, 0.28), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.22), ( 0.00, 0.20,  0.50)),
    ((0.30, -0.05, 0.35), ( 0.00, 0.00,  0.00)),
    ((0.30, -0.05, 0.35), ( 0.00, 0.20, -0.50)),
    # 5. Combined rotations at four workspace corners
    ((0.25, -0.05, 0.25), ( 0.30, 0.20,  0.50)),
    ((0.25,  0.00, 0.25), (-0.30, 0.20, -0.50)),
    ((0.35, -0.05, 0.25), (-0.30, 0.20,  0.50)),
    ((0.35,  0.00, 0.25), ( 0.30, 0.20, -0.50)),
    # 6. Extreme combined angles (probe IK feasibility)
    ((0.30, -0.05, 0.28), (-0.40, 0.20, -1.00)),
    ((0.30, -0.05, 0.28), (-0.30, 0.30, -1.00)),
    ((0.30, -0.05, 0.28), ( 0.60, 0.20,  0.30)),
    ((0.30, -0.05, 0.28), (-0.60, 0.20, -0.30)),
    ((0.30, -0.05, 0.28), ( 0.50, 0.30,  0.50)),
    ((0.30, -0.05, 0.28), (-0.50, 0.30, -0.50)),
]

# Auto-mode timing constants (match rebot_grasp; calibration needs
# settle time before PnP reprojection is reliable).
AUTO_MOVE_DURATION_S = 3.0
AUTO_SETTLE_EXTRA_S = 0.6
AUTO_MARKER_TIMEOUT_S = 2.5
AUTO_MARKER_STABLE_FRAMES = 4
MIN_CALIB_SAMPLES = 5
# Calibration fixture mounted on the original two-finger gripper.
# Motor 7 position that firmly holds the PCB.
CALIB_GRIPPER_HOLD_POS = 0.05

# First run only: restrict auto calibration to the first 3 poses.
# Set to None after fixture geometry / visibility is verified.
AUTO_POSE_LIMIT: int | None = None


# ──────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────


def _pose_to_T(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    """xyz (m) + rpy (rad) -> 4x4 SE(3) pose. Mirrors ``pose_to_T`` in
    cuRobo's calibration module so the pose sequence can be copied
    verbatim across the two codebases.

    Rotation order: Rz @ Ry @ Rx (ZYX extrinsic / XYZ intrinsic -- the
    convention rebot_grasp and cuRobo 04b both use).
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyz
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx
    return T

def _gripper_to_marker_T() -> np.ndarray:
    """Fixed rigid transform from gripper_end to the ArUco marker.

    Actual calibration fixture geometry:

      - The PCB/ArUco mounting orientation requires joint6 ~= +pi/2
        relative to the original marker-on-gripper arrangement.
      - FK shows joint6 +pi/2 corresponds to approximately Rx(-pi/2)
        in the gripper_end frame.
      - Therefore the fixed marker mounting rotation is Rx(+pi/2),
        whose inverse supplies the required Rx(-pi/2) TCP correction.
      - Measured ArUco centre offset from gripper_end:
            x = -25 mm
            y =   0 mm
            z = +73 mm

    Returns
    -------
    T_GM : 4x4 ndarray
        gripper_end -> ArUco marker-centre transform.
    """
    T = np.eye(4, dtype=np.float64)

    a = np.pi / 2.0
    ca, sa = np.cos(a), np.sin(a)

    # Rx(+90 deg)
    T[:3, :3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  ca, -sa],
        [0.0,  sa,  ca],
    ], dtype=np.float64)

    # Marker centre expressed in gripper_end coordinates.
    T[:3, 3] = np.array([
        -0.025,
         0.000,
         0.073,
    ], dtype=np.float64)

    return T

def _make_kinematics(arm_profile: str = ARM_PROFILE_DEFAULT) -> Kinematics:
    """Construct the wrc_demo FK on the RS URDF (mirrored joint convention)."""
    if not URDF_DEFAULT.exists():
        raise FileNotFoundError(
            f"RS URDF not found at {URDF_DEFAULT}; run from the wrc_demo repo root"
        )
    return Kinematics(
        urdf_path=str(URDF_DEFAULT),
        ee_frame="gripper_end",
        arm_joint_count=6,
    )


# ──────────────────────────────────────────────────────────────────────
# Manual mode: gravity compensation controller (ported from rebot_grasp)
#
# Uses the vendored ``reBotArm_control_py`` SDK directly (bypasses
# SafeArm) to drive MIT-mode + gravity feed-forward at the SDK's native
# control rate (~500 Hz). This is the same pattern as
# ``skills/teach.py`` (see CLAUDE.md "Motion safety path" exception
# clause) -- external-event-triggered control loops bypass SafeArm's
# trajectory streaming because SafeArm only exposes ``move_joints``.
#
# The operator can hand-guide the arm while the controller compensates
# for gravity + tracks the current pose. End-effector velocity locking
# (V_THRESH / W_THRESH) latches the target when motion is detected, so
# the operator feels the arm "freeze" when they stop moving and re-engage
# as they push it again.
# ──────────────────────────────────────────────────────────────────────


def _ensure_sdk_on_path(sdk_path: str) -> None:
    """Insert the vendored SDK root into ``sys.path`` so the
    ``reBotArm_control_py`` package becomes importable.

    Idempotent; mirrors the pattern in ``control/rebot_rs_arm.py``.
    """
    if sdk_path and sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)


class GravityCompController:
    """MIT mode + gravity FF + end-effector velocity locking.

    Ported from rebot_grasp/scripts/calibrate_top_camera.py::GravityCompController
    with vendored SDK paths swapped in. KP/KD/V_THRESH/W_THRESH match
    rebot_grasp; the 500 Hz control rate matches the SDK's
    ``arm.start_control_loop(rate=...)`` default.
    """

    KP = 8.0
    KD = 1.5
    V_THRESH = 0.04   # End-effector linear velocity threshold (m/s)
    W_THRESH = 0.08   # End-effector angular velocity threshold (rad/s)

    def __init__(self, arm, *, sdk_path: str | None = None,
                 urdf_path: str | None = None) -> None:
        # Lazy imports -- the SDK pulls in pinocchio + the rebot arm
        # drivers, which is heavy and unavailable in the dry-run env.
        _ensure_sdk_on_path(sdk_path or str(PACKAGE_ROOT / "sdk"))
        from reBotArm_control_py.dynamics.inverse_dynamics import (  # type: ignore
            compute_generalized_gravity,
        )
        from reBotArm_control_py.kinematics.robot_model import (  # type: ignore
            load_robot_model,
            get_end_effector_frame_id,
            pad_q_for_model,
        )
        import pinocchio as pin  # type: ignore

        self._compute_gravity = compute_generalized_gravity
        self._pad_q_for_model = pad_q_for_model
        self._pin = pin

        self._arm = arm
        if not arm.has_gripper:
            raise ValueError(
                "Hardware config is missing groups.gripper. "
                "Enable the gripper group in the selected hardware YAML under "
                "sdk/config."
            )
        # load_robot_model() defaults to the SDK-shipped URDF; we pass
        # the wrC URDF explicitly so the gravity comp uses the same model
        # as the rest of the calibration pipeline (joint_signs baked in).
        urdf = urdf_path or str(URDF_DEFAULT)
        if not Path(urdf).exists():
            raise FileNotFoundError(f"URDF not found at {urdf}")
        self._model = load_robot_model(urdf)
        self._data = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n: Optional[int] = None
        self._q_target: Optional[np.ndarray] = None
        self._integral: Optional[np.ndarray] = None
        self._gc_running = threading.Event()
        self._io_lock = threading.RLock()

    def start(self) -> None:
        """Connect (idempotent if already connected), enable, set MIT
        mode, and start the gravity-compensation control loop."""
        self._arm.connect()
        print("[GravityComp] Connected", file=sys.stderr)

        self._arm.arm.mode_mit(
            kp=np.full(self._arm.arm.num_joints, self.KP),
            kd=np.full(self._arm.arm.num_joints, self.KD),
        )
        self._arm.gripper.mode_mit()
        self._arm.enable_all()
        print("[GravityComp] Enabled", file=sys.stderr)

        n = self._arm.arm.num_joints
        self._n = n
        self._wait_state_valid()
        with self._io_lock:
            q0 = self._arm.get_state()[0][:n]
        self._q_target = q0.copy()
        self._integral = np.zeros(n)

        print(
            f"[GravityComp] MIT mode, kp={self.KP} kd={self.KD}. "
            "Move the arm by hand.", file=sys.stderr,
        )

        self._gc_running.set()
        self._arm.start_control_loop(self._worker, rate=self._arm.rate)

    def _wait_state_valid(self, timeout: float = 2.0) -> None:
        """Wait until all motor feedback is available."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._io_lock:
                self._arm.get_state()
                ok = all(
                    m.get_state() is not None
                    for m in self._arm._motor_map.values()
                )
            if ok:
                return
            time.sleep(0.02)
        raise RuntimeError("[GravityComp] Arm feedback is not ready")

    def read_q(self, n: int | None = None) -> Optional[np.ndarray]:
        """Read the current arm joint state. Returns ``None`` on failure.

        Used by the gravity-comp collection loop to feed FK after a
        stable marker detection.
        """
        if self._n is None:
            return None
        try:
            with self._io_lock:
                q_all, _, _ = self._arm.get_state()
            return np.asarray(q_all[: self._n], dtype=float)
        except Exception as e:
            print(f"[GravityComp] read_q failed: {e}", file=sys.stderr)
            return None

    def safe_home(self) -> None:
        """Stop gravity compensation, drive to home with a fresh SDK
        controller, then disconnect. Mirrors rebot_grasp's
        ``GravityCompController.safe_home`` -- uses ``RebotArmEndPose``
        (vendored SDK) directly because the arm is in raw MIT mode and
        cannot accept ``SafeArm.home()``.
        """
        self._gc_running.clear()
        try:
            print("[GravityComp] Homing...", file=sys.stderr)
            self._arm.stop_control_loop()
            n = self._n or self._arm.arm.num_joints
            with self._io_lock:
                q_now = self._arm.get_state()[0][:n]
                g_now = (
                    self._arm.gripper.get_positions()
                    if self._arm.has_gripper else np.array([])
                )

            _ensure_sdk_on_path(str(PACKAGE_ROOT / "sdk"))
            from reBotArm_control_py.controllers import (  # type: ignore
                RebotArmEndPose,
            )
            ctrl = RebotArmEndPose(
                self._arm,
                arm_control_mode="mit",
                use_gravity_ff=True,
            )
            ctrl.set_gripper_target(float(g_now[0]) if g_now.size else 0.0)
            ctrl._q_target[:] = q_now
            ctrl.start()
            ctrl.safe_home()
            self._arm.stop_control_loop()
        except Exception as e:
            print(f"[GravityComp] Homing failed: {e}", file=sys.stderr)
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] Disconnected", file=sys.stderr)

    def _worker(self, r, dt: float) -> None:
        if not self._gc_running.is_set():
            return

        pin = self._pin
        model, data, ee_id = self._model, self._data, self._ee_id
        n = self._n
        KP, KD = self.KP, self.KD

        try:
            with self._io_lock:
                q_all, qd_all, _ = r.get_state()
            q = q_all[:n]
            qd = qd_all[:n]
            q_model = self._pad_q_for_model(model, q, n)
            tau_g = self._compute_gravity(model=model, q=q_model)[:n]

            q_err = self._q_target - q
            self._integral += q_err
            np.clip(self._integral, -0.5, 0.5, out=self._integral)

            qd_model = np.zeros(model.nv)
            qd_model[: min(model.nv, n)] = qd[: min(model.nv, n)]
            pin.computeJointJacobians(model, data, q_model)
            pin.updateFramePlacements(model, data)
            J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd_model

            if (np.linalg.norm(v[:3]) > self.V_THRESH or
                    np.linalg.norm(v[3:]) > self.W_THRESH):
                self._q_target = q.copy()
                self._integral *= 0.9

            with self._io_lock:
                r.arm.send_mit(
                    pos=self._q_target,
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral,
                )
                r.gripper.send_mit(r.gripper.get_positions())
        except Exception:
            # Silent: the worker fires at 500 Hz; transient state-read
            # glitches (motor not ready yet) shouldn't bring the loop down.
            pass


# ──────────────────────────────────────────────────────────────────────
# Cleanup (wrc_demo contract; rebot_grasp has no equivalent)
# ──────────────────────────────────────────────────────────────────────


def _soft_release_and_disconnect(
    safe_arm,
    *,
    duration_s: float = 2.0,
    label: str = "calib_top",
) -> None:
    """Ramp MIT support smoothly to zero, then disable and disconnect.

    This runs only after the operator has removed the calibration PCB and
    confirmed that the arm is physically supported.  The normal SDK
    ``disconnect()`` disables every motor immediately; ramping position
    stiffness, damping, and gravity feed-forward first avoids an abrupt
    torque step at the home pose.

    The implementation intentionally stays local to the calibration CLI:
    it uses the already-running SDK controller and does not change the
    runtime shutdown semantics used by the grasping demo.
    """
    disconnect_fn = getattr(safe_arm, "disconnect", None)
    ctrl = getattr(safe_arm, "_ctrl", None)
    raw_arm = getattr(safe_arm, "_arm", None)
    arm_group = getattr(ctrl, "_arm_group", None)

    if ctrl is None or raw_arm is None or arm_group is None:
        print(
            f"[{label}] soft-release internals unavailable; "
            "using normal disconnect",
            file=sys.stderr,
        )
        if callable(disconnect_fn):
            disconnect_fn()
        return

    try:
        # Stop the 500 Hz controller before sending the ramp ourselves,
        # otherwise its full-gain commands would race our reduced-gain ones.
        raw_arm.stop_control_loop()
        # The RS SDK's generic streamed feedback is not reliable on this
        # hardware; use the controller's last commanded home target.  This
        # is the same target its 500 Hz loop was holding immediately before
        # we stopped that loop.
        n = int(getattr(ctrl, "_n", len(ctrl._q_target)))
        q_hold = np.asarray(ctrl._q_target[:n], dtype=float).copy()
        kp0 = np.asarray(arm_group._mit_kp, dtype=float).copy()
        kd0 = np.asarray(arm_group._mit_kd, dtype=float).copy()

        tau_g = np.zeros(n, dtype=float)
        if bool(getattr(ctrl, "_use_gravity_ff", False)):
            from reBotArm_control_py.dynamics import compute_generalized_gravity  # type: ignore
            from reBotArm_control_py.kinematics.robot_model import pad_q_for_model  # type: ignore

            q_model = pad_q_for_model(ctrl._model, q_hold, n)
            tau_g = np.asarray(
                compute_generalized_gravity(ctrl._model, q_model, ctrl._data)[:n],
                dtype=float,
            )
            gravity_scale = np.asarray(
                getattr(ctrl, "_gravity_ff_scale", np.ones(n)), dtype=float,
            )[:n]
            tau_g *= gravity_scale

        steps = max(2, int(max(duration_s, 0.1) * 50.0))
        print(
            f"[{label}] gently releasing motor support over "
            f"{duration_s:.1f}s ...",
            file=sys.stderr,
        )
        for i in range(steps):
            s = (i + 1) / steps
            # Smoothstep has zero slope at both ends, avoiding a jerk when
            # the ramp begins and immediately before the final disable.
            scale = 1.0 - (3.0 * s * s - 2.0 * s * s * s)
            arm_group.send_mit(
                q_hold,
                vel=np.zeros(n),
                kp=kp0 * scale,
                kd=kd0 * scale,
                tau=tau_g * scale,
            )
            time.sleep(duration_s / steps)

        # Release the gripper only after the PCB has been removed.
        gripper_group = getattr(ctrl, "_gripper_group", None)
        if gripper_group is not None and getattr(raw_arm, "has_gripper", False):
            gripper_group.send_mit(
                np.array([float(getattr(ctrl, "_gripper_target", 0.0))]),
                kp=np.zeros(gripper_group.num_joints),
                kd=np.zeros(gripper_group.num_joints),
                tau=np.zeros(gripper_group.num_joints),
            )

        raw_arm.disable_all()
        raw_arm.disconnect()
        ctrl._running = False
        print(f"[{label}] motors disabled and arm disconnected", file=sys.stderr)
    except Exception as e:
        print(
            f"[{label}] soft release failed ({e!r}); "
            "falling back to normal disconnect",
            file=sys.stderr,
        )
        if callable(disconnect_fn):
            try:
                disconnect_fn()
            except Exception as disconnect_error:
                print(
                    f"[{label}] disconnect raised: {disconnect_error!r}",
                    file=sys.stderr,
                )


def _make_arm_cleanup(safe_arm, *, label: str = "calib_top") -> Callable[[], None]:
    if safe_arm is None:
        return lambda: None

    done = [False]
    # The operator-visible power-on/home position is the SDK joint-zero
    # pose.  SafeArm.home() is a different, configured workspace pose
    # (home_q in configs/arms/rebot_rs.yaml), so use the SDK controller's
    # minimum-jerk safe_home() here.  Keep SafeArm.home() only as a fallback
    # if the SDK zero-return path is unavailable or raises.
    ctrl = getattr(safe_arm, "_ctrl", None)
    zero_home_fn = getattr(ctrl, "safe_home", None)
    fallback_home_fn = getattr(safe_arm, "home", None)

    def cleanup() -> None:
        if done[0]:
            return
        done[0] = True

        home_ok = False
        at_joint_zero = False
        if callable(zero_home_fn):
            try:
                print(f"[{label}] returning to joint-zero home ...", file=sys.stderr)
                zero_home_fn()
                home_ok = True
                at_joint_zero = True
                print(
                    f"[{label}] joint-zero home complete; "
                    "motors remain enabled",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"[{label}] joint-zero return raised: {e!r}; "
                    "trying configured safe home",
                    file=sys.stderr,
                )

        if not home_ok and callable(fallback_home_fn):
            try:
                ok = fallback_home_fn(duration_s=3.0)
                home_ok = ok is not False
                if home_ok:
                    print(
                        f"[{label}] configured safe home complete; "
                        "motors remain enabled",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[{label}] configured safe home refused; "
                        "holding the current pose with motors enabled",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(
                    f"[{label}] configured safe home raised: {e!r}; "
                    "holding the current pose with motors enabled",
                    file=sys.stderr,
                )

        location = (
            "joint-zero home"
            if at_joint_zero
            else "configured safe home"
            if home_ok
            else "the current pose"
        )
        print(
            f"[{label}] Arm is holding at {location}. Remove the PCB, "
            "physically support the arm, then press ENTER to gently "
            "release motor torque.",
            file=sys.stderr,
        )
        try:
            input()
        except EOFError:
            print(
                f"[{label}] stdin closed; proceeding with torque release",
                file=sys.stderr,
            )

        _soft_release_and_disconnect(safe_arm, label=label)

    return cleanup

def _register_process_cleanup(cleanup: Callable[[], None]) -> None:
    """Register ``cleanup`` as an atexit hook + SIGTERM handler so any
    abnormal exit (Ctrl+C in bash, ``kill``, unhandled exception,
    ``sys.exit``) parks the arm to home and disconnects instead of
    leaving it at the last collection pose with motors enabled.
    """
    atexit.register(cleanup)
    try:
        prev = _signal.getsignal(_signal.SIGTERM)
    except (ValueError, OSError):
        return  # not in main thread

    def _on_term(sig, frame):
        cleanup()
        if callable(prev) and prev not in (_signal.SIG_DFL, _signal.SIG_IGN):
            try:
                prev(sig, frame)
            except Exception:
                pass
        raise SystemExit(128 + sig)

    try:
        _signal.signal(_signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        pass


# ──────────────────────────────────────────────────────────────────────
# Camera / serial helpers
# ──────────────────────────────────────────────────────────────────────


def _print_devices() -> int:
    try:
        devices = list_orbbec_devices()
    except Exception as e:
        print(f"[calib] enumerate failed: {e}", file=sys.stderr)
        return 2
    if not devices:
        print("[calib] no Orbbec devices found")
        return 1
    for d in devices:
        print(
            f"  [{d.get('index', '?')}] {d.get('name', '?')} "
            f"serial={d.get('serial', '?')} "
            f"pid={d.get('pid', '?')} vid={d.get('vid', '?')}"
        )
    return 0


def _bind_role(role: str, serial: str | None = None, index: int | None = None) -> int:
    """Persist role -> serial in ``cameras.json``. If ``serial`` is omitted
    we use the first enumerated device whose index matches ``--index``.
    """
    role = role.lower()
    if role not in {"top", "wrist"}:
        print(f"[calib] bind: role must be 'top' or 'wrist', got {role!r}", file=sys.stderr)
        return 2
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    p = CALIB_DIR / "cameras.json"
    existing = (
        CamerasBinding.from_json(json.loads(p.read_text())) if p.exists() else CamerasBinding()
    )
    if serial is None:
        devices = list_orbbec_devices()
        if not devices:
            print("[calib] bind: no Orbbec devices to bind", file=sys.stderr)
            return 1
        idx = index if index is not None else 0
        if idx < 0 or idx >= len(devices):
            print(f"[calib] bind: index {idx} out of range (0..{len(devices) - 1})", file=sys.stderr)
            return 1
        serial = str(devices[idx].get("serial", ""))
        index = int(devices[idx].get("index", idx))
    binding = CameraBinding(serial=serial, index_when_bound=index if index is not None else -1)
    if role == "top":
        updated = CamerasBinding(
            top=binding,
            wrist=existing.wrist,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    else:
        updated = CamerasBinding(
            top=existing.top,
            wrist=binding,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    p.write_text(json.dumps(updated.to_json(), indent=2, sort_keys=True))
    print(f"[calib] bound {role} -> serial={serial} (index {binding.index_when_bound}); wrote {p}")
    # Best-effort: also patch the matching profile YAML so the rig boots
    # straight into the right device without manual editing.
    profile_map = {"top": "orbbec_overhead", "wrist": "orbbec_wrist"}
    prof = PACKAGE_ROOT / "configs" / "cameras" / f"{profile_map[role]}.yaml"
    if prof.exists():
        text = prof.read_text()
        if "\nserial:" in text:
            lines = text.splitlines()
            for k, line in enumerate(lines):
                if line.startswith("serial:"):
                    lines[k] = f'serial: "{serial}"'
                    break
            else:
                lines.append(f'serial: "{serial}"')
            prof.write_text("\n".join(lines) + "\n")
            print(f"[calib] patched {prof.name}: serial={serial!r}")
    return 0


def _open_orbbec(serial: str, width: int, height: int, fps: int) -> OrbbecCamera:
    from ..config import Cfg
    from ..perception.camera_base import CameraError

    cfg = Cfg({
        "type": "orbbec",
        "serial": serial,
        "width": width,
        "height": height,
        "fps": fps,
        "min_depth_m": 0.25,
        "max_depth_m": 2.5,
    })
    cam = OrbbecCamera(cfg)
    try:
        cam.open()
    except CameraError as e:
        print(f"[calib] camera open failed: {e}", file=sys.stderr)
        raise
    return cam


def _get_serial_from_binding(role: str) -> str | None:
    p = CALIB_DIR / "cameras.json"
    if not p.exists():
        return None
    b = CamerasBinding.from_json(json.loads(p.read_text()))
    return getattr(b, role).serial if getattr(b, role) else None


def _wait_for_user(prompt: str) -> None:
    print(prompt, flush=True)
    try:
        input("press ENTER when ready...")
    except EOFError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Frame grab / ArUco detection helpers
# ──────────────────────────────────────────────────────────────────────


def _grab_color(cam) -> Optional[np.ndarray]:
    """Read one frame; return the BGR array or None on failure.

    Returns BGR (``Frame.rgb`` is BGR despite the name -- see CLAUDE.md
    contract).
    """
    try:
        f = cam.get_frame()
    except Exception:
        return None
    return f.rgb


def _stable_detect(
    cam, aruco, K, D, marker_size_m,
    *, settle_time: float, marker_timeout: float, stable_frames: int,
    target_id: int = 0,
) -> tuple[object | None, list[object]]:
    """Grab frames until the marker has been stable for N consecutive
    detections or ``settle_time + marker_timeout`` elapses.

    Returns ``(best_detection, all_detections)``.
    """
    best = None
    stable_count = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_time + marker_timeout:
        try:
            frame = cam.get_frame()
        except Exception:
            time.sleep(0.1)
            continue
        det = aruco.detect(
            frame.rgb, K, D, marker_size_m, target_id=target_id,
        )
        if det is None:
            stable_count = 0
            time.sleep(0.05)
            continue
        stable_count += 1
        best = det
        if stable_count >= stable_frames:
            return det, []
    return best, []


# ──────────────────────────────────────────────────────────────────────
# OSD / HUD (rebot_grasp style: clean status text + sample progress bar)
# ──────────────────────────────────────────────────────────────────────


def _draw_osd(
    vis: np.ndarray,
    *,
    pose_label: str,
    n_samples: int,
    target_n: int,
    state: str,
    marker_id: Optional[int] = None,
    marker_z: Optional[float] = None,
    mode: str = "AUTO",
    capture_flash: bool = False,
) -> None:
    """Draw the calibrator HUD on top of `vis` (in place).

    Style mirrors rebot_grasp/scripts/calibrate_top_camera.py: status
    text drawn directly on the frame (no opaque top panel) with color
    coding by marker-detection state, plus a sample-count progress bar
    and a mode label in the bottom-right corner.
    """
    h, w = vis.shape[:2]

    if state.lower().startswith("settling"):
        status_color = (0, 200, 255)
        status_text = (
            f"Settling - ArUco detection paused  "
            f"samples:{n_samples}/{target_n}"
        )
    elif marker_id is not None and marker_z is not None:
        status_color = (80, 220, 80)
        status_text = f"[ID={marker_id}] z={marker_z:.3f}m  samples:{n_samples}/{target_n}"
    else:
        status_color = (80, 80, 220)
        status_text = f"No marker  samples:{n_samples}/{target_n}"
    cv2.putText(
        vis, status_text, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA,
    )

    cv2.putText(
        vis, f"{pose_label[:60]}  --  {state[:40]}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1, cv2.LINE_AA,
    )

    bar_max = max(target_n, 1)
    filled = min(n_samples, target_n) * (400 // bar_max)
    cv2.rectangle(vis, (10, 70), (10 + filled, 82), (0, 200, 100), -1)
    cv2.rectangle(vis, (10, 70), (410, 82), (160, 160, 160), 1)
    cv2.putText(
        vis, f"{n_samples}/{target_n}", (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    cv2.putText(
        vis, mode, (max(w - 200, 10), h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1, cv2.LINE_AA,
    )

    cv2.putText(
        vis, "press 'c' or 'q' to finish early, ESC to abort",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA,
    )

    if capture_flash:
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)


# ──────────────────────────────────────────────────────────────────────
# CAN bus + arm connection (wrc_demo helpers; rebot_grasp uses different
# APIs and we bridge to wrC's make_arm() factory).
# ──────────────────────────────────────────────────────────────────────


def _diagnose_can_bus() -> str | None:
    """Return a human-readable diagnostic string if the CAN bus looks
    unhealthy (ERROR-PASSIVE / BUS-OFF / bitrate wrong / no device),
    else ``None``.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["ip", "-d", "link", "show", "can_arm"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception:
        return None
    text = (out.stdout or "") + (out.stderr or "")
    if "state ERROR-PASSIVE" in text:
        return (
            "can_arm is ERROR-PASSIVE — the CAN controller stopped active "
            "transmit because it sees errors with no ACK (no device on "
            "the bus). Check: (a) arm 24V power, (b) CAN cable firmly "
            "seated at both ends, (c) `sudo ip link set can_arm down && "
            "sudo ip link set can_arm up type can bitrate 1000000`."
        )
    if "state BUS-OFF" in text:
        return (
            "can_arm is BUS-OFF — too many errors. Same fixes as "
            "ERROR-PASSIVE plus power-cycle the arm."
        )
    if "state DOWN" in text or "state UNKNOWN" in text:
        return (
            "can_arm is DOWN. Run `sudo ip link set can_arm up type can "
            "bitrate 1000000`."
        )
    return None


def _connect_arm_with_safety(arm_cfg_path: str):
    """Construct ``(kinematics, safe_arm, arm_raw)`` from a wrc_demo arm
    profile. Returns ``None`` if the arm can't be connected (so the
    caller can fall back to manual mode).

    ``make_arm`` already calls connect() + clear_error() +
    start_control_loop internally; we verify with a quick get_state() read
    and propagate any failure as a clean error string.
    """
    from ..config import load_demo_config, load_profile
    from ..control import make_arm

    arm_prof = load_profile("arms", arm_cfg_path)
    # load_demo_config returns a full cfg (camera + arm + llm + safety);
    # make_arm needs the whole cfg, not just the arm profile.
    demo_cfg = load_demo_config(camera="mock", arm=arm_cfg_path, llm="mock")
    kin = Kinematics(
        urdf_path=str(demo_cfg.arm.urdf),
        ee_frame=str(demo_cfg.arm.get("ee_frame", "gripper_end")),
        arm_joint_count=int(demo_cfg.arm.get("n_joints", 6)),
    )
    safe_arm = make_arm(demo_cfg, kinematics=kin)

    try:
        _ = safe_arm.get_state()
    except Exception as e:
        try:
            safe_arm.disconnect()
        except Exception:
            pass
        return None, f"post-connect state read failed: {e}"
    return (kin, safe_arm, safe_arm._arm), None


# ──────────────────────────────────────────────────────────────────────
# Auto-mode state machine (ported from rebot_grasp's tick_auto pattern)
# ──────────────────────────────────────────────────────────────────────


def _collect_loop_auto(
    safe_arm, kin, aruco, cam, K, D,
    *, n_target: int, marker_size_m: float, aruco_marker_id: int,
    settle_time: float,
    marker_timeout: float, stable_frames: int,
    samples: list[CalibrationSample], trace,
    win_name: str, show_window: bool,
) -> tuple[list[CalibrationSample], str]:
    """Auto-mode: idle -> moving -> settling -> searching -> capture -> idle.

    Faithful port of rebot_grasp/scripts/calibrate_top_camera.py's
    ``tick_auto`` state machine. Differences vs. rebot_grasp:
      - IK via ``Kinematics.ik`` (Pinocchio on RS URDF, joint_signs baked)
      - Motion via ``safe_arm.move_joints`` (SafetyHarness + velocity cap)
      - Per-pose IK/move failures are skipped + logged, not retried
    """
    finish_reason = "all target poses done"

    poses = (
        TOP_CALIB_POSES[:AUTO_POSE_LIMIT]
        if AUTO_POSE_LIMIT is not None
        else TOP_CALIB_POSES
    )

    target_n = len(poses)
    snap_counter = [0]

    def _stop_requested() -> tuple[bool, str]:
        key = cv2.waitKey(30) & 0xFF if show_window else 0
        if key in (ord('q'), ord('Q'), ord('c'), ord('C'), 27):
            return True, "user finish (key)"
        if show_window:
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    return True, "user finish (window closed)"
            except cv2.error:
                return True, "user finish (window gone)"
        return False, ""

    def _record_sample(best, pose_idx: int, xyz, rpy, q_target: np.ndarray) -> None:
        # Use safe_arm.get_state() (truncates to n_arm joints); falls back
        # to the last commanded q if state read fails (very rare).
        try:
            st = safe_arm.get_state()
            q = np.asarray(st.q, dtype=float)[: safe_arm.n_joints]
        except Exception:
            q = q_target
        G = kin.fk(q)
        sample = CalibrationSample(
            pose_label=f"auto_{pose_idx + 1:02d}",
            T_gripper2base=G,
            T_marker2cam=best.T_marker2cam,
            aruco_reprojection_px=float(best.reprojection_rmse_px),
            inlier=True,
        )
        samples.append(sample)
        trace.write(json.dumps({
            "event": "sample_recorded", "i": pose_idx, "ts": time.time(),
            "q": q.tolist(),
            "pose_label": f"xyz={xyz} rpy={rpy}",
            "aruco_rmse_px": float(best.reprojection_rmse_px),
        }) + "\n")
        print(
            f"[calib] captured sample {len(samples)}: "
            f"marker_id={best.marker_id}; "
            f"xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) "
            f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f}); "
            f"aruco rmse={best.reprojection_rmse_px:.2f}px"
        )

    def _draw_with_hud(
        color: np.ndarray, det, pose_label: str, n: int, state: str,
        *, capture_flash: bool = False,
    ) -> np.ndarray:
        overlay = (
            aruco.draw_aruco_overlay(color, det)
            if det is not None else color
        )
        _draw_osd(
            overlay,
            pose_label=pose_label,
            n_samples=n,
            target_n=target_n,
            state=state,
            marker_id=(
                getattr(det, "marker_id", None) if det is not None else None
            ),
            marker_z=(det.T_marker2cam[2, 3]
                      if det is not None and hasattr(det, "T_marker2cam") else None),
            mode="AUTO",
            capture_flash=capture_flash,
        )
        if show_window:
            cv2.imshow(win_name, overlay)
        else:
            # Headless fallback: every detection plus 1-of-every-SNAP_EVERY
            # frames while settling/searching. Settling is not a failed
            # detection: detection is intentionally paused in that phase.
            snap_counter[0] += 1
            if det is not None or snap_counter[0] % SNAP_EVERY == 0:
                try:
                    label = (
                        "det"
                        if det is not None
                        else "settling"
                        if state.lower().startswith("settling")
                        else "nodet"
                    )
                    out = overlay if det is not None else color
                    cv2.imwrite(
                        str(SNAP_DIR / f"f{snap_counter[0]:05d}_{label}.png"),
                        out,
                    )
                except Exception:
                    pass
        return overlay

    last_q: list[Optional[np.ndarray]] = [None]

    try:
        for idx, (xyz, rpy) in enumerate(poses):
            pose_label = (
                f"xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) "
                f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f})"
            )
            print(f"\n[calib] auto pose {idx + 1}/{target_n}: {pose_label}")

            # IK. Seed from previous solution when available.
            # TOP_CALIB_POSES was designed with the original assumption that
            # the ArUco centre approximately coincides with gripper_end.
            # Treat it as the desired MARKER pose, then compensate our actual
            # PCB/ArUco fixture to obtain the required gripper_end pose.
            T_marker_target = _pose_to_T(xyz, rpy)
            T_target = T_marker_target @ np.linalg.inv(_gripper_to_marker_T())
            q_init = (last_q[0] if last_q[0] is not None
                      else np.zeros(safe_arm.n_joints, dtype=float))
            ik = kin.ik(T_target, q_init, retries=2)
            if not ik.success:
                print(
                    f"[calib] pose {idx + 1}: IK failed (error={ik.error:.4f}); "
                    "skipping", file=sys.stderr,
                )
                trace.write(json.dumps({
                    "event": "pose_skipped", "i": idx, "ts": time.time(),
                    "reason": "ik_failed",
                }) + "\n")
                continue
            q_target = np.asarray(ik.q, dtype=float)[: safe_arm.n_joints]
            last_q[0] = q_target

            # Refresh safety harness's perception watchdog. Calibration
            # is itself the only perception consumer here; the watchdog
            # was last touched at connect() -- too long ago.
            if hasattr(safe_arm, "harness") and hasattr(safe_arm.harness, "heartbeat"):
                safe_arm.harness.heartbeat()

            # Defensive pre-check: IK's ``limit_margin`` (0.025) should
            # keep us clear of the harness's ``joint_margin`` (0.020), but
            # empirical runs at workspace edges (e.g. xyz=(0.25, ...))
            # have produced q with joint2 = 0.019 < URDF lower + margin.
            # Reject such poses locally so we never call move_joints with
            # a guaranteed-to-violate target -- the harness would raise
            # ``SafetyViolation`` and crash the whole loop, losing all
            # earlier samples. Wrapping move_joints in try/except (below)
            # is the catch-all; this pre-check is the optimisation.
            harness = getattr(safe_arm, "harness", None)
            kin_for_limits = getattr(harness, "kin", None) if harness is not None else None
            joint_limits = getattr(kin_for_limits, "joint_limits", None)
            if joint_limits is not None:
                margin = float(getattr(harness.limits, "joint_margin", 0.020))
                lo, hi = joint_limits
                lo = np.asarray(lo, dtype=float)[: safe_arm.n_joints] + margin
                hi = np.asarray(hi, dtype=float)[: safe_arm.n_joints] - margin
                viol = (q_target < lo) | (q_target > hi)
                if viol.any():
                    bad = np.where(viol)[0].tolist()
                    print(
                        f"[calib] pose {idx + 1}: IK solution violates "
                        f"harness margin at joints {[i + 1 for i in bad]} "
                        f"(q={q_target.round(3).tolist()}); skipping",
                        file=sys.stderr,
                    )
                    trace.write(json.dumps({
                        "event": "pose_skipped", "i": idx, "ts": time.time(),
                        "reason": "ik_at_margin",
                        "violated_joints": [i + 1 for i in bad],
                    }) + "\n")
                    continue

            try:
                ok = safe_arm.move_joints(q_target, duration_s=2.0)
            except Exception as e:  # SafetyViolation + any hardware fault
                # The harness's begin_motion / motion check raises on
                # geometric violation (e.g. joint2 = 0.019 < 0.020).
                # Skip this pose gracefully -- the rest of the loop
                # should keep going so we still get >= 4 samples to fit.
                # If we crash here, we lose every earlier sample.
                print(
                    f"[calib] pose {idx + 1}: safe_arm.move_joints raised "
                    f"({type(e).__name__}: {e}); skipping", file=sys.stderr,
                )
                trace.write(json.dumps({
                    "event": "pose_skipped", "i": idx, "ts": time.time(),
                    "reason": "harness_raised",
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e)[:200],
                }) + "\n")
                continue
            if not ok:
                print(
                    f"[calib] pose {idx + 1}: safe_arm.move_joints refused; "
                    "skipping", file=sys.stderr,
                )
                trace.write(json.dumps({
                    "event": "pose_skipped", "i": idx, "ts": time.time(),
                    "reason": "harness_refused",
                }) + "\n")
                continue

            # Wait for arm to settle (best-effort; if backend lacks this,
            # just sleep the configured settle_time).
            arm = getattr(safe_arm, "_arm", safe_arm)
            settled = True
            if hasattr(arm, "wait_settled"):
                settled = bool(arm.wait_settled(q_target, tol=0.05, timeout_s=5.0))
                if not settled:
                    print(
                        f"[calib] pose {idx + 1}: arm didn't settle in 5s; "
                        "continuing with detection anyway", file=sys.stderr,
                    )

            # State machine: settling -> searching -> (capture | skip).
            t_settle_until = time.monotonic() + settle_time
            t_timeout = t_settle_until + marker_timeout
            stable_count = 0
            best = None
            wrong_marker_ids: set[int] = set()
            phase = "settling"
            while time.monotonic() < t_timeout:
                stopped, reason = _stop_requested()
                if stopped:
                    finish_reason = reason
                    return samples, finish_reason

                color = _grab_color(cam)
                if color is None:
                    time.sleep(0.05)
                    continue

                if hasattr(safe_arm, "harness") and hasattr(safe_arm.harness, "heartbeat"):
                    safe_arm.harness.heartbeat()

                if phase == "settling" and time.monotonic() >= t_settle_until:
                    phase = "searching"

                det = (
                    aruco.detect(
                        color,
                        K,
                        D,
                        marker_size_m,
                        target_id=aruco_marker_id,
                    )
                    if phase == "searching"
                    else None
                )
                # A target-id miss used to be reported as the same generic
                # ``no marker`` result as a frame with no ArUco at all.  Probe
                # once without filtering so the timeout message can tell the
                # operator whether another pattern was falsely decoded.
                if phase == "searching" and det is None:
                    observed = aruco.detect(
                        color, K, D, marker_size_m, target_id=None,
                    )
                    if (observed is not None
                            and observed.marker_id != aruco_marker_id):
                        wrong_marker_ids.add(int(observed.marker_id))

                # Update stability before rendering so the HUD reflects
                # the current frame instead of lagging one frame behind.
                if phase == "searching":
                    if det is None:
                        stable_count = 0
                        best = None
                    else:
                        stable_count += 1
                        best = det
                _draw_with_hud(
                    color, det,
                    pose_label=f"{idx + 1}/{target_n} {pose_label}",
                    n=len(samples),
                    state=(f"settling" if phase == "settling"
                           else f"searching ({stable_count}/{stable_frames})"),
                )

                if det is None:
                    time.sleep(0.05)
                    continue
                if stable_count >= stable_frames:
                    break

            if best is None:
                if wrong_marker_ids:
                    detail = (
                        f"target marker id={aruco_marker_id} not detected; "
                        f"saw other id(s)={sorted(wrong_marker_ids)}"
                    )
                    reason = "wrong_marker_id"
                else:
                    detail = (
                        f"target marker id={aruco_marker_id} not detected "
                        "(no ArUco marker found)"
                    )
                    reason = "no_marker"
                print(f"[calib] pose {idx + 1}: {detail}; skipping", file=sys.stderr)
                trace.write(json.dumps({
                    "event": "sample_skipped", "i": idx, "ts": time.time(),
                    "reason": reason,
                    "expected_marker_id": aruco_marker_id,
                    "observed_marker_ids": sorted(wrong_marker_ids),
                }) + "\n")
                continue

            # Capture-flash HUD: one extra frame with green border so the
            # operator sees when a sample was recorded. Skip if no frame
            # is available (camera hiccup) -- no point drawing the flash
            # on a blank canvas.
            color = _grab_color(cam)
            if color is not None:
                vis = color.copy()
                rendered = _draw_with_hud(
                    vis, best,
                    pose_label=f"{idx + 1}/{target_n} {pose_label}",
                    n=len(samples) + 1,
                    state=f"captured ({stable_count}/{stable_frames})",
                    capture_flash=True,
                )
                if show_window:
                    cv2.waitKey(120)
                else:
                    try:
                        cv2.imwrite(
                            str(SNAP_DIR / f"capture_{len(samples):02d}.png"),
                            rendered,
                        )
                    except Exception:
                        pass

            _record_sample(best, idx, xyz, rpy, q_target)

            stopped, reason = _stop_requested()
            if stopped:
                finish_reason = reason
                return samples, finish_reason

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C"
    return samples, finish_reason


# ──────────────────────────────────────────────────────────────────────
# Manual mode (no auto motion; operator moves arm + presses ENTER)
# ──────────────────────────────────────────────────────────────────────


def _read_arm_q(serial: str | None = None) -> Optional[np.ndarray]:
    """Read the current arm joint state from the rebot_rs arm.

    Lazy import so the script is runnable on the dry-run path (no arm).
    Returns ``None`` if the arm isn't connected; the caller decides how
    to handle that (skip vs synthetic).

    Tests ``monkeypatch.setattr(cli, "_read_arm_q", ...)`` this -- the
    public symbol must stay importable from ``calib_top_orbbec``.
    """
    try:
        from ..control.rebot_rs_arm import RebotRSArm  # type: ignore
    except Exception:
        return None
    try:
        arm = RebotRSArm.__new__(RebotRSArm)  # bypass __init__ state checks
        # Minimal state to read .get_state(): see rebot_rs_arm.__init__.
        # If the arm can't be probed, just return None (dry-run path).
        from ..safety.harness import SafetyLimits
        arm.connect()
        st = arm.get_state()
        arm.disconnect()
        return np.asarray(st.q[:6], dtype=float)
    except Exception as e:
        print(f"[calib] arm read failed: {e}", file=sys.stderr)
        return None


def _collect_loop_manual(
    kin, aruco, cam, K, D,
    *, n_target: int, marker_size_m: float, aruco_marker_id: int,
    settle_time: float, marker_timeout: float, stable_frames: int,
    samples: list[CalibrationSample], trace,
    arm_q_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
) -> tuple[list[CalibrationSample], str]:
    """Manual mode: operator moves arm + presses ENTER; we wait for a
    stable ArUco detection and record one sample per pose.

    ``arm_q_provider`` is an optional callable returning the current
    joint vector. If None (or returns None on failure), we fall back to
    ``_read_arm_q()`` and ultimately to a synthetic seed (dry-run-style).
    """
    finish_reason = "operator finished"
    for i in range(n_target):
        print(f"\n[calib] === sample {i + 1}/{n_target} ===")
        _wait_for_user(
            "Move the arm to a new pose, hold still. The detector will "
            f"look for stable ArUco readings for {stable_frames} frames within "
            f"{settle_time + marker_timeout:.1f}s."
        )
        det, _ = _stable_detect(
            cam, aruco, K, D, marker_size_m,
            settle_time=settle_time,
            marker_timeout=marker_timeout,
            stable_frames=stable_frames,
            target_id=aruco_marker_id,
        )
        if det is None:
            print(
                f"[calib] sample {i + 1}: no ArUco detected within timeout, "
                "skipping (operator can re-run with --n N)",
                file=sys.stderr,
            )
            trace.write(json.dumps({
                "event": "sample_skipped", "i": i,
                "ts": time.time(), "reason": "no_marker",
            }) + "\n")
            continue
        if arm_q_provider is not None:
            try:
                q = arm_q_provider()
            except Exception:
                q = None
        else:
            q = _read_arm_q(cam.serial if cam is not None else None)
        if q is None:
            print(
                "[calib] cannot read arm state (no rebot_rs connection); "
                "saving synthetic FK from a deterministic seed for "
                "dry-run only", file=sys.stderr,
            )
            q = np.zeros(6)
            q[0] = 0.30 + 0.05 * i
        G = kin.fk(q)
        sample = CalibrationSample(
            pose_label=f"manual_{i + 1}",
            T_gripper2base=G,
            T_marker2cam=det.T_marker2cam,
            aruco_reprojection_px=float(det.reprojection_rmse_px),
            inlier=True,
        )
        samples.append(sample)
        trace.write(json.dumps({
            "event": "sample_recorded", "i": i, "ts": time.time(),
            "q": q.tolist(),
            "aruco_rmse_px": float(det.reprojection_rmse_px),
        }) + "\n")
        print(
            f"[calib] recorded sample {len(samples)}: aruco rmse="
            f"{det.reprojection_rmse_px:.2f}px, q[:3]={q[:3].round(3).tolist()}"
        )
    return samples, finish_reason


def _collect_loop_gravity_comp(
    gc_ctrl, kin, aruco, cam, K, D,
    *, n_target: int, marker_size_m: float, aruco_marker_id: int,
    settle_time: float, marker_timeout: float, stable_frames: int,
    samples: list[CalibrationSample], trace,
) -> tuple[list[CalibrationSample], str]:
    """Manual mode with MIT gravity-compensation (Phase 3).

    Same operator flow as ``_collect_loop_manual`` (ENTER to capture,
    c/q to finish early) but q is read from the ``GravityCompController``
    which keeps the arm floating under MIT-mode gravity FF.
    """
    finish_reason = "operator finished"
    for i in range(n_target):
        print(f"\n[calib] === sample {i + 1}/{n_target} (gravity-comp) ===")
        _wait_for_user(
            "Move the arm to a new pose (it should float under gravity FF). "
            "Hold still. The detector will look for stable ArUco readings "
            f"for {stable_frames} frames within "
            f"{settle_time + marker_timeout:.1f}s."
        )
        det, _ = _stable_detect(
            cam, aruco, K, D, marker_size_m,
            settle_time=settle_time,
            marker_timeout=marker_timeout,
            stable_frames=stable_frames,
            target_id=aruco_marker_id,
        )
        if det is None:
            print(
                f"[calib] sample {i + 1}: no ArUco detected within timeout, "
                "skipping", file=sys.stderr,
            )
            trace.write(json.dumps({
                "event": "sample_skipped", "i": i,
                "ts": time.time(), "reason": "no_marker",
            }) + "\n")
            continue
        q = gc_ctrl.read_q()
        if q is None:
            print(
                "[calib] cannot read arm q from GravityCompController; "
                "skipping sample", file=sys.stderr,
            )
            trace.write(json.dumps({
                "event": "sample_skipped", "i": i,
                "ts": time.time(), "reason": "no_arm_q",
            }) + "\n")
            continue
        G = kin.fk(q)
        sample = CalibrationSample(
            pose_label=f"gc_{i + 1}",
            T_gripper2base=G,
            T_marker2cam=det.T_marker2cam,
            aruco_reprojection_px=float(det.reprojection_rmse_px),
            inlier=True,
        )
        samples.append(sample)
        trace.write(json.dumps({
            "event": "sample_recorded", "i": i, "ts": time.time(),
            "q": q.tolist(),
            "mode": "gravity_comp",
            "aruco_rmse_px": float(det.reprojection_rmse_px),
        }) + "\n")
        print(
            f"[calib] recorded sample {len(samples)} (gravity-comp): "
            f"aruco rmse={det.reprojection_rmse_px:.2f}px, "
            f"q[:3]={q[:3].round(3).tolist()}"
        )
    return samples, finish_reason


def _connect_gc_controller(arm_profile: str):
    """Construct a ``GravityCompController`` from a wrc_demo arm profile.

    Returns ``(gc_ctrl, None)`` on success, or ``(None, error_str)`` if
    the SDK / hardware / arm profile aren't usable. Caller should fall
    back to plain ``--manual`` on error.
    """
    from ..config import load_profile

    arm_prof = load_profile("arms", arm_profile)
    sdk_path = arm_prof.get("sdk_path") or str(PACKAGE_ROOT / "sdk")
    urdf_path = (
        arm_prof.get("urdf")
        or str(URDF_DEFAULT)
    )
    hw_yaml = arm_prof.get("hw_yaml", "rebotarm_rs.yaml")

    try:
        _ensure_sdk_on_path(sdk_path)
        from reBotArm_control_py.actuator import RebotArm  # type: ignore

        arm = RebotArm(hw_yaml)
        gc_ctrl = GravityCompController(
            arm, sdk_path=sdk_path, urdf_path=urdf_path,
        )
        return gc_ctrl, None
    except Exception as e:
        return None, f"gravity-comp init failed: {e}"


# ──────────────────────────────────────────────────────────────────────
# Public entry point: signature preserved (tests import this).
# ──────────────────────────────────────────────────────────────────────


def _collect_samples(
    n_target: int,
    marker_size_m: float,
    aruco_dict: str,
    settle_time: float,
    marker_timeout: float,
    stable_frames: int,
    serial: str | None,
    width: int,
    height: int,
    fps: int,
    out_json: Path,
    trace_log: Path,
    arm_profile: str = ARM_PROFILE_DEFAULT,
    manual: bool = False,
    show_window: bool = True,
    gravity_comp: bool = False,
    aruco_marker_id: int = 0,
) -> tuple[list[CalibrationSample], str]:
    """Collect ``n_target`` calibration samples.

    Mode:
      - ``manual=False, gravity_comp=False`` (default auto): arm moves
        itself through ``TOP_CALIB_POSES`` (auto state machine); the
        OpenCV window shows live detection + OSD. Falls back to manual
        when the arm can't be reached.
      - ``manual=True, gravity_comp=False``: operator presses ENTER
        after each pose; arm stays where the operator left it. Useful
        for offline / dry-run / debugging.
      - ``gravity_comp=True``: same as manual, but the arm is driven in
        MIT mode with gravity feed-forward via ``GravityCompController``,
        so the operator can hand-guide the arm without it falling. q is
        read directly from the vendored SDK. Requires the rebot_rs arm
        on can_arm; falls back to plain ``manual`` when the SDK is
        unreachable.

    Returns ``(samples, finish_reason)`` so the caller knows whether the
    loop ended because the operator stopped early vs. all poses done.
    """
    kin = _make_kinematics()
    aruco = ArUcoSession(dict_name=aruco_dict)
    cam = _open_orbbec(serial or "", width, height, fps)
    K = cam.K
    D = cam.D
    print(
        f"[calib] camera open: serial={cam.serial} K={K.tolist() if K is not None else None}",
        file=sys.stderr,
    )
    samples: list[CalibrationSample] = []
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    finish_reason = "session end"
    cleanup: Callable[[], None] = lambda: None

    WIN = "Hand-Eye Calibration -- TOP"
    if not show_window:
        try:
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    if show_window:
        try:
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            print(
                "[calib] cv2.namedWindow unavailable (headless opencv). "
                "Frames will be saved to /tmp/calib_frames/ for offline "
                "inspection; one out of every 10 frames is saved.",
                file=sys.stderr,
            )
            show_window = False
            try:
                SNAP_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # Arm connection (auto mode only). Manual mode still tries to read q
    # for FK, but if that fails (no arm) we fall back to a synthetic seed.
    safe_arm = None
    if not manual:
        can_diag = _diagnose_can_bus()
        if can_diag:
            print(f"[calib] CAN bus diagnostic: {can_diag}", file=sys.stderr)
        result, err = _connect_arm_with_safety(arm_profile)
        if err:
            print(
                f"[calib] arm unavailable ({err}); falling back to manual mode",
                file=sys.stderr,
            )
            manual = True
        else:
            (kin, safe_arm, _arm_raw) = result

            # Hold the calibration PCB throughout the complete calibration run.
            # RebotRSArm has its own 10 ms gripper tracking loop, so setting this
            # once is sufficient while joints 1-6 execute their trajectories.
            arm_backend = getattr(safe_arm, "_arm", None)
            if arm_backend is not None and hasattr(arm_backend, "set_gripper"):
                arm_backend.set_gripper(CALIB_GRIPPER_HOLD_POS)
                time.sleep(1.0)

                try:
                    st = safe_arm.get_state()
                    actual_gripper = getattr(st, "gripper_pos", None)
                except Exception:
                    actual_gripper = None

                print(
                    f"[calib] calibration-board gripper hold: "
                    f"target={CALIB_GRIPPER_HOLD_POS:.3f} rad, "
                    f"actual={actual_gripper}"
                )

            cleanup = _make_arm_cleanup(safe_arm, label="calib_top")
            _register_process_cleanup(cleanup)

            # Move to the first calibration pose before starting the sequence.
            # This is not the joint-zero home pose.
            xyz0, rpy0 = TOP_CALIB_POSES[0]
            # As above, TOP_CALIB_POSES describes the desired marker pose.
            T_marker_initial = _pose_to_T(xyz0, rpy0)
            T_initial = T_marker_initial @ np.linalg.inv(_gripper_to_marker_T())
            q_init = np.zeros(int(safe_arm.n_joints), dtype=float)
            initial_ik = kin.ik(T_initial, q_init, retries=2)
            if not initial_ik.success:
                from ..config import load_profile
                arm_prof = load_profile("arms", arm_profile)
                q_init = np.asarray(arm_prof.get("home_q", [0.0] * 6), dtype=float)
                initial_ik = kin.ik(T_initial, q_init, retries=4)
            if not initial_ik.success:
                print(
                    "[calib] IK failed for initial calibration pose "
                    f"xyz={xyz0} rpy={rpy0}; falling back to manual.",
                    file=sys.stderr,
                )
                safe_arm = None
                manual = True
            else:
                initial_q = np.asarray(initial_ik.q, dtype=float)[: safe_arm.n_joints]
                print(
                    f"[calib] initial calibration pose: xyz={xyz0} rpy={rpy0} -> "
                    f"q={initial_q.round(3).tolist()}"
                )
                if hasattr(safe_arm, "harness") and hasattr(safe_arm.harness, "heartbeat"):
                    safe_arm.harness.heartbeat()
                ok = safe_arm.move_joints(initial_q, duration_s=3.0)
                if not ok:
                    estopped = bool(getattr(safe_arm.harness, "estopped", False))
                    if estopped:
                        print(
                            "[calib] safe_arm estopped during initial pose move "
                            "(harness rejected the pose). Falling back to manual.",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            "[calib] safe_arm.move_joints(initial pose) returned False; "
                            "likely CAN bus failure during enable_all / "
                            "mode_mit (check `ip -d link show can_arm` and the "
                            "arm 24V supply). Falling back to manual.",
                            file=sys.stderr,
                        )
                    safe_arm = None
                    manual = True
                else:
                    time.sleep(2.0)

    with trace_log.open("a") as trace:
        trace.write(json.dumps({
            "event": "session_start",
            "ts": time.time(),
            "marker_size_m": marker_size_m,
            "aruco_dict": aruco_dict,
            "n_target": n_target,
            "manual": manual,
            "gravity_comp": gravity_comp,
            "arm_profile": arm_profile,
        }) + "\n")

        if gravity_comp:
            # Stand up the gravity-comp controller (raw RebotArm, MIT mode).
            # On any failure (SDK missing, hardware unreachable), fall back
            # to plain --manual so the operator still gets a working session.
            gc_ctrl, gc_err = _connect_gc_controller(arm_profile)
            if gc_err:
                print(
                    f"[calib] gravity-comp unavailable ({gc_err}); "
                    "falling back to plain --manual", file=sys.stderr,
                )
                manual = True
            else:
                try:
                    gc_ctrl.start()
                except Exception as e:
                    print(
                        f"[calib] gravity-comp start failed ({e}); "
                        "falling back to plain --manual", file=sys.stderr,
                    )
                    manual = True

        if manual:
            if gravity_comp and not gc_err:
                samples, finish_reason = _collect_loop_gravity_comp(
                    gc_ctrl, kin, aruco, cam, K, D,
                    n_target=n_target, marker_size_m=marker_size_m,
                    aruco_marker_id=aruco_marker_id,
                    settle_time=settle_time, marker_timeout=marker_timeout,
                    stable_frames=stable_frames,
                    samples=samples, trace=trace,
                )
            else:
                samples, finish_reason = _collect_loop_manual(
                    kin, aruco, cam, K, D,
                    n_target=n_target, marker_size_m=marker_size_m,
                    aruco_marker_id=aruco_marker_id,
                    settle_time=settle_time, marker_timeout=marker_timeout,
                    stable_frames=stable_frames, samples=samples, trace=trace,
                    arm_q_provider=(lambda: np.asarray(safe_arm.get_state().q[:6], dtype=float)
                                    if safe_arm is not None else None),
                )
        else:
            samples, finish_reason = _collect_loop_auto(
                safe_arm, kin, aruco, cam, K, D,
                n_target=n_target, marker_size_m=marker_size_m,
                aruco_marker_id=aruco_marker_id,
                settle_time=settle_time, marker_timeout=marker_timeout,
                stable_frames=stable_frames, samples=samples, trace=trace,
                win_name=WIN, show_window=show_window,
            )

        trace.write(json.dumps({
            "event": "session_end",
            "ts": time.time(),
            "n_collected": len(samples),
            "finish_reason": finish_reason,
        }) + "\n")

    if show_window and not manual:
        try:
            cv2.destroyWindow(WIN)
        except cv2.error:
            pass
    cam.close()
    # gc mode cleanup is the controller's safe_home (drives via SDK,
    # not via SafeArm). safe_arm cleanup is the atexit-registered one.
    if gravity_comp and 'gc_ctrl' in locals() and gc_ctrl is not None:
        try:
            gc_ctrl.safe_home()
        except Exception as e:
            print(f"[calib] gravity-comp safe_home raised: {e}", file=sys.stderr)
    else:
        cleanup()
    return samples, finish_reason


# ──────────────────────────────────────────────────────────────────────
# Solve + persist (unchanged from prior implementation; only ONE copy)
# ──────────────────────────────────────────────────────────────────────


def _save_raw_samples(samples: list[CalibrationSample], out_json: Path) -> Path:
    """Persist the complete solver input before optimization starts.

    A missing numerical dependency or optimizer failure must never force the
    operator to repeat a full hardware traversal.  This sidecar contains the
    exact FK and PnP transforms needed to retry the solve offline.
    """
    raw_path = out_json.with_suffix(".samples.npz")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        T_gripper2base=np.stack(
            [np.asarray(s.T_gripper2base, dtype=float) for s in samples]
        ),
        T_marker2cam=np.stack(
            [np.asarray(s.T_marker2cam, dtype=float) for s in samples]
        ),
        aruco_reprojection_px=np.asarray(
            [s.aruco_reprojection_px for s in samples], dtype=float
        ),
        pose_label=np.asarray([s.pose_label for s in samples]),
    )
    print(f"[calib] saved raw solver samples to {raw_path}")
    return raw_path


def _synthesize_samples(
    n: int,
    seed: int = 0,
) -> tuple[list[CalibrationSample], np.ndarray, np.ndarray]:
    """Generate synthetic (FK, PnP) samples with a known ground-truth
    ``X = T_cam2base`` and ``Y = T_marker2gripper`` so the operator can
    sanity-check the entire calibration pipeline without hardware.
    """
    from ..calibration.handeye_solver import so3_exp

    rng = np.random.default_rng(seed)
    X_true = np.eye(4)
    X_true[:3, 3] = [0.30, -0.10, 0.85]
    ang_y, ang_z = 0.5, 0.25
    Ry = np.array(
        [[np.cos(ang_y), 0.0, np.sin(ang_y)],
         [0.0, 1.0, 0.0],
         [-np.sin(ang_y), 0.0, np.cos(ang_y)]]
    )
    Rz = np.array(
        [[np.cos(ang_z), -np.sin(ang_z), 0.0],
         [np.sin(ang_z),  np.cos(ang_z), 0.0],
         [0.0, 0.0, 1.0]]
    )
    X_true[:3, :3] = Rz @ Ry

    Y_true = np.eye(4)
    Y_true[:3, 3] = [0.005, 0.0, 0.04]
    Y_true[:3, :3] = so3_exp(rng.normal(scale=0.05, size=3))

    samples: list[CalibrationSample] = []
    for i in range(n):
        ang_z_g = 0.3 * i
        ang_y_g = 0.2 + 0.1 * i
        Rz_g = np.array(
            [[np.cos(ang_z_g), -np.sin(ang_z_g), 0.0],
             [np.sin(ang_z_g),  np.cos(ang_z_g), 0.0],
             [0.0, 0.0, 1.0]]
        )
        Ry_g = np.array(
            [[np.cos(ang_y_g), 0.0, np.sin(ang_y_g)],
             [0.0, 1.0, 0.0],
             [-np.sin(ang_y_g), 0.0, np.cos(ang_y_g)]]
        )
        G = np.eye(4)
        G[:3, :3] = Ry_g @ Rz_g
        G[:3, 3] = [0.28 + 0.02 * i, 0.03 * np.cos(i), 0.15 + 0.04 * i]
        # G Y = X M  =>  M = inv(X) G Y
        M = np.linalg.inv(X_true) @ G @ Y_true
        samples.append(CalibrationSample(
            pose_label=f"dryrun_{i + 1}",
            T_gripper2base=G,
            T_marker2cam=M,
            aruco_reprojection_px=0.1,
            inlier=True,
        ))
    return samples, X_true, Y_true


def _solve_and_persist(
    samples: list[CalibrationSample],
    out_json: Path,
    marker_size_m: float,
    aruco_dict: str,
    aruco_marker_id: int,
    serial: str | None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    X_true: np.ndarray | None = None,
    Y_true: np.ndarray | None = None,
) -> CalibrationResult:
    if len(samples) < 4:
        raise RuntimeError(
            f"only {len(samples)} samples collected; need >= 4 for joint_solve"
        )
    _save_raw_samples(samples, out_json)
    print(f"[calib] running joint_solve on {len(samples)} samples...")
    fit = joint_solve_restart(samples, n_restart=4, seed=0)
    print(
        f"[calib] fit: translation rmse={fit.metrics.get('translation_rmse_mm', float('nan')):.2f}mm "
        f"max={fit.metrics.get('translation_max_err_mm', float('nan')):.2f}mm "
        f"inliers={len(fit.inliers)} outliers={len(fit.outliers)}"
    )
    inlier_set = set(fit.inliers)
    flagged = tuple(
        CalibrationSample(
            pose_label=s.pose_label,
            T_gripper2base=s.T_gripper2base,
            T_marker2cam=s.T_marker2cam,
            aruco_reprojection_px=s.aruco_reprojection_px,
            inlier=(i in inlier_set),
        )
        for i, s in enumerate(samples)
    )
    result = CalibrationResult(
        schema_version=SCHEMA_VERSION,
        frame_convention="wrc_demo",
        frame_convention_note=(
            "base +x away robot, +y left; TCP = gripper_end; "
            "joint_signs = [-1]*6"
        ),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        method="joint_se3_lm_huber",
        mode="eye_to_hand",
        camera_serial=serial or "",
        camera_index_when_bound=-1,
        marker_size_m=marker_size_m,
        aruco_dict=aruco_dict,
        aruco_marker_id=aruco_marker_id,
        n_samples=len(samples),
        n_inliers=len(fit.inliers),
        n_outliers=len(fit.outliers),
        outlier_indices=tuple(int(i) for i in fit.outliers),
        T_cam2base=np.asarray(fit.T_cam2base, dtype=float).tolist(),
        T_marker2gripper=np.asarray(fit.T_marker2gripper, dtype=float).tolist(),
        metrics={k: float(v) for k, v in fit.metrics.items()},
        samples_for_repro=flagged,
    )
    if X_true is not None and Y_true is not None:
        x_err = float(np.linalg.norm(fit.T_cam2base - X_true))
        y_err = float(np.linalg.norm(fit.T_marker2gripper - Y_true))
        result = CalibrationResult(
            **{**result.__dict__, "metrics": {
                **result.metrics,
                "dryrun_X_frob_err": x_err,
                "dryrun_Y_frob_err": y_err,
            }},
        )
    K = np.eye(3)
    D = np.zeros(5)
    json_path, npz_path = save_calibration(result, out_json, K=K, D=D)
    print(f"[calib] wrote {json_path}")
    print(f"[calib] wrote {npz_path}")
    return result


def _verify(npz_path: Path, known_points: Sequence[tuple[float, float, float]]) -> int:
    if not npz_path.exists():
        print(f"[calib] verify: {npz_path} does not exist", file=sys.stderr)
        return 2
    res = load_calibration(npz_path)
    X = np.asarray(res.T_cam2base, dtype=float)
    Y = np.asarray(res.T_marker2gripper, dtype=float)
    if not res.samples_for_repro:
        print("[calib] verify: no samples_for_repro in the saved result; "
              "re-run calibration to populate them", file=sys.stderr)
        return 1
    errs_mm = []
    print(
        f"[calib] verify against {len(known_points)} known point(s); "
        f"prediction is the centroid of inlier sample marker positions "
        f"({len(res.samples_for_repro)} samples on disk)"
    )
    for kp in known_points:
        kp_arr = np.asarray(kp, dtype=float).reshape(3)
        cam_pts = []
        for s in res.samples_for_repro:
            if not s.inlier:
                continue
            M = np.asarray(s.T_marker2cam, dtype=float)
            cam_pts.append((X @ M)[:3, 3])
        if not cam_pts:
            continue
        avg = np.mean(cam_pts, axis=0)
        err = np.linalg.norm(avg - kp_arr) * 1000.0
        errs_mm.append(err)
        print(
            f"  known=({kp_arr[0]:.3f},{kp_arr[1]:.3f},{kp_arr[2]:.3f}) "
            f"predicted=({avg[0]:.3f},{avg[1]:.3f},{avg[2]:.3f}) err={err:.1f}mm"
        )
    if not errs_mm:
        print("[calib] verify: no inlier samples available", file=sys.stderr)
        return 1
    rmse = float(np.sqrt(np.mean(np.asarray(errs_mm) ** 2)))
    mx = float(np.max(errs_mm))
    print(f"[calib] verify RMSE = {rmse:.2f}mm, max = {mx:.2f}mm")
    return 0 if rmse <= 10.0 else 1


# ──────────────────────────────────────────────────────────────────────
# Stub for Phase 3: --gravity-comp manual mode with MIT gravity FF.
# The full implementation is added in Phase 3 (T3.1+T3.2); for now the
# flag is accepted but --gravity-comp falls back to the simplified
# ``_collect_loop_manual`` so the CLI surface stays stable during
# incremental rollout.
# ──────────────────────────────────────────────────────────────────────


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    """Background thread that mirrors stdin to ``line_queue`` without
    blocking the main loop. Rebot_grasp pattern.

    Returns a daemon thread (caller doesn't need to ``.join()``).
    """
    def _loop():
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true",
                   help="enumerate Orbbec devices and exit")
    p.add_argument("--bind", choices=["top", "wrist"], default=None,
                   help="bind role -> serial, write data/calibration/cameras.json")
    p.add_argument("--serial", default=None,
                   help="Orbbec serial to bind/use (skips enumeration)")
    p.add_argument("--index", type=int, default=None,
                   help="device index to pick when enumerating")
    p.add_argument("--arm-profile", default=ARM_PROFILE_DEFAULT)
    p.add_argument("--marker-size", type=float, default=0.10,
                   help="ArUco marker side length (meters)")
    p.add_argument("--aruco-dict", default="4x4_50",
                   help="ArUco dictionary (4x4_50 | 5x5_50 | 6x6_50 | APRILTAG_16h5)")
    p.add_argument("--aruco-marker-id", type=int, default=0)
    p.add_argument("--settle-time", type=float, default=1.5,
                   help="seconds to wait for the arm to settle before grab")
    p.add_argument("--marker-timeout", type=float, default=4.0,
                   help="max seconds to wait for a stable marker detection")
    p.add_argument("--stable-frames", type=int, default=4)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n", type=int, default=15,
                   help="target number of calibration samples")
    p.add_argument("--out", default=str(CALIB_DIR / "hand_eye_top.json"))
    p.add_argument("--verify", action="store_true",
                   help="verify a saved calibration against known points")
    p.add_argument("--npz", default=None,
                   help="path to a saved calibration JSON (for --verify)")
    p.add_argument("--known-points", default=None,
                   help="semicolon-separated base-frame points 'x,y,z;x,y,z'")
    p.add_argument("--dry-run", action="store_true",
                   help="synthesize samples (skip camera + arm) and exercise the "
                        "solve/persist path end-to-end; used to verify the "
                        "pipeline without hardware")
    p.add_argument("--dry-run-out", default=str(CALIB_DIR / "hand_eye_dryrun.json"),
                   help="output JSON path for --dry-run (does NOT clobber --out)")
    p.add_argument("--manual", action="store_true",
                   help="disable auto-mode arm motion: wait for ENTER after "
                        "each pose (operator moves arm by hand)")
    p.add_argument("--gravity-comp", action="store_true",
                   help="enable MIT-mode gravity compensation in manual mode "
                        "so the arm floats (operator can hand-guide without "
                        "it falling). Falls back to --manual if the SDK / "
                        "arm aren't reachable.")
    p.add_argument(
                    "--no-window",
                    action="store_true",
                    help="run auto calibration headless without OpenCV GUI; "
                        "save debug frames under /tmp/calib_frames/",
                )
    args = p.parse_args(argv)

    if args.list:
        return _print_devices()

    if args.bind is not None:
        return _bind_role(args.bind, serial=args.serial, index=args.index)

    if args.verify:
        if not args.npz:
            print("--verify requires --npz", file=sys.stderr)
            return 2
        if not args.known_points:
            print("--verify requires --known-points 'x,y,z;...'", file=sys.stderr)
            return 2
        try:
            kps = [
                tuple(float(t) for t in pp.split(","))
                for pp in args.known_points.split(";")
            ]
        except ValueError as e:
            print(f"--known-points parse error: {e}", file=sys.stderr)
            return 2
        return _verify(Path(args.npz), kps)

    if args.dry_run:
        out_p = Path(args.dry_run_out)
        if out_p.resolve() == Path(args.out).resolve():
            print(
                f"[calib] --dry-run-out must differ from --out ({args.out})",
                file=sys.stderr,
            )
            return 2
        samples, X_true, Y_true = _synthesize_samples(args.n, seed=0)
        result = _solve_and_persist(
            samples,
            out_json=out_p,
            marker_size_m=args.marker_size,
            aruco_dict=args.aruco_dict,
            aruco_marker_id=args.aruco_marker_id,
            serial="DRYRUN",
            X_true=X_true, Y_true=Y_true,
        )
        x_err = float(np.linalg.norm(np.asarray(result.T_cam2base) - X_true))
        y_err = float(np.linalg.norm(np.asarray(result.T_marker2gripper) - Y_true))
        per_sample_mm = float(result.metrics.get("translation_rmse_mm", float("nan")))
        print(
            f"[calib] dry-run: per-sample rmse = {per_sample_mm:.4f}mm; "
            f"||X - X_true||_F = {x_err:.4f}, ||Y - Y_true||_F = {y_err:.4f}; "
            f"wrote {out_p}"
        )
        return 0 if per_sample_mm < 1.0 else 1

    # Default: collect samples.
    if args.gravity_comp:
        # gc mode implies manual capture (operator presses ENTER per
        # pose, but the arm is gravity-compensated so it floats).
        args.manual = True

    serial = args.serial or _get_serial_from_binding("top") or ""
    samples, finish_reason = _collect_samples(
        n_target=args.n,
        marker_size_m=args.marker_size,
        aruco_dict=args.aruco_dict,
        settle_time=args.settle_time,
        marker_timeout=args.marker_timeout,
        stable_frames=args.stable_frames,
        serial=serial or None,
        width=args.width,
        height=args.height,
        fps=args.fps,
        out_json=Path(args.out),
        trace_log=PACKAGE_ROOT / "runs" / f"calibration_{int(time.time())}.jsonl",
        arm_profile=args.arm_profile,
        manual=args.manual,
        show_window=(not args.manual) and (not args.no_window),
        gravity_comp=args.gravity_comp,
        aruco_marker_id=args.aruco_marker_id,
    )
    print(f"[calib] session ended: {finish_reason}")
    if len(samples) < 4:
        print(f"[calib] only {len(samples)} samples; need >=4 to fit. "
              "Re-run with --n for another pass.", file=sys.stderr)
        return 1
    _solve_and_persist(
        samples,
        out_json=Path(args.out),
        marker_size_m=args.marker_size,
        aruco_dict=args.aruco_dict,
        aruco_marker_id=args.aruco_marker_id,
        serial=serial or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
