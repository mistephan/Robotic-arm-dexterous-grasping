"""Real reBot DevArm B601 (RobStride build) backend.

Transport rides on reBotArm_control_py's RebotArm/JointGroup (motorbridge ->
SocketCAN can_arm @ 1 Mbps), but with the rig-verified fixes this SDK lacks:

- Joint positions are read with RobStride param reads of mechPos (0x7019),
  because RS motors stream compact type-0x18 report frames that motorbridge
  get_state() never decodes; robstride_set_active_report(True) yields a
  single type-2 frame and then freezes. Param reads return the exact f32.
- The hardware YAML is forced to the RS profile; kinematics come from
  wrc_demo.control.Kinematics on the RS URDF (this repo's assets), never
  from the SDK's global config (which silently loads the DM URDF).
- Gripper close is a two-stage, effort-scaled MIT command with stall
  detection via mechVel (0x701A) instead of the SDK's broken open/close.

Operational notes (verified on this rig 2026-07):
- bring the bus up first: sudo ip link set can_arm up type can bitrate 1000000
- do NOT run motorbridge-gateway / MotorBridge Studio at the same time; both
  use host id 0xFD and fight over reporting mode.
- gripper open/close angles below were characterized on the DM build; RS
  travel and stall torque MUST be re-verified onsite before first grasp.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from ..config import Cfg
from ..types import RobotState
from .arm_base import ArmBase

MECH_POS = 0x7019
MECH_VEL = 0x701A


class RebotRSArm(ArmBase):
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        sdk_path = cfg.get("sdk_path")
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        self._arm = None
        self._lock = threading.Lock()
        self._stopped = False
        self._last_q = np.zeros(self.n_joints)
        self._mit_kp = None
        self._mit_kd = None
        self.settle_tol = float(cfg.get("settle_tol", 0.05))
        g = cfg.get("gripper", Cfg({}))
        self._grip_open = float(g.get("open_pos", -6.8))
        self._grip_closed = float(g.get("closed_pos", 0.0))
        self._grip_kp = float(g.get("kp", 6.0))
        self._grip_kd = float(g.get("kd", 0.4))
        
        # Gripper control loop state
        self._gripper_target = self._grip_closed  # target position for smooth tracking
        self._gripper_loop_running = False
        self._gripper_loop_thread = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        try:
            from reBotArm_control_py.actuator import RebotArm
        except ImportError as e:
            raise RuntimeError(
                "reBotArm_control_py not importable; set arm.sdk_path in the "
                "arm profile (and install motorbridge in this venv)"
            ) from e
        hw_yaml = self._cfg.get("hw_yaml", "rebotarm_rs.yaml")
        arm = RebotArm(hw_yaml)
        arm.connect()
        grp = arm.arm
        kp = self._cfg.get("mit_kp")
        kd = self._cfg.get("mit_kd")
        self._mit_kp = np.asarray(kp, dtype=float) if kp is not None else grp._mit_kp.copy()
        self._mit_kd = np.asarray(kd, dtype=float) if kd is not None else grp._mit_kd.copy()
        # SDK-canonical order: set modes on ALL groups first, then enable.
        # JointGroup.enable() is controller-wide (enable_all), so the gripper
        # motor powers up together with the arm and must already be in MIT.
        grp.mode_mit()
        if arm.has_gripper:
            arm.gripper.mode_mit()
        grp.enable()
        self._arm = arm
        self._stopped = False
        self._read_failures = 0
        self._last_q = self._read_positions()
        self._last_cmd_q: np.ndarray | None = None
        
        # Start gripper control loop
        self._gripper_loop_running = True
        self._gripper_loop_thread = threading.Thread(
            target=self._gripper_control_loop, daemon=True
        )
        self._gripper_loop_thread.start()

    def disconnect(self) -> None:
        self._gripper_loop_running = False
        if self._gripper_loop_thread is not None:
            self._gripper_loop_thread.join(timeout=1.0)
        if self._arm is not None:
            try:
                self._arm.disconnect()
            finally:
                self._arm = None

    # ── feedback (param reads, the only reliable RS path) ────────────────

    def _motors(self):
        grp = self._arm.arm
        return [grp._mm[name] for name in grp.joint_names]

    def _read_positions(self) -> np.ndarray:
        q = np.zeros(self.n_joints)
        with self._lock:
            for i, m in enumerate(self._motors()[: self.n_joints]):
                q[i] = m.robstride_get_param_f32(MECH_POS)
        return q

    #: consecutive feedback failures tolerated before motion must abort
    MAX_READ_FAILURES = 3

    def get_state(self) -> RobotState:
        try:
            q = self._read_positions()
            self._last_q = q
            self._read_failures = 0
        except Exception as e:
            # One transient CAN hiccup: serve last known. Repeated failures
            # mean the bus is unhealthy -- fail loudly so motion aborts
            # instead of planning from phantom positions.
            self._read_failures += 1
            if self._read_failures >= self.MAX_READ_FAILURES:
                raise RuntimeError(
                    f"CAN feedback lost ({self._read_failures} consecutive "
                    f"mechPos read failures): {e}"
                ) from e
            q = self._last_q
        gp = self._gripper_pos()
        return RobotState(
            q=q.copy(),
            gripper_pos=0.0 if gp is None else gp,
            gripper_valid=gp is not None,
            t=time.monotonic(),
        )

    def _gripper_pos(self) -> float | None:
        """Gripper motor angle, or None when the read fails (callers must
        treat None as 'unknown', never as a position)."""
        if self._arm is None or not self._arm.has_gripper:
            return None
        try:
            with self._lock:
                m = self._arm.gripper._mm[self._arm.gripper.joint_names[0]]
                return float(m.robstride_get_param_f32(MECH_POS))
        except Exception:
            return None

    # ── commands ─────────────────────────────────────────────────────────

    def send_joint_target(self, q: np.ndarray) -> None:
        if self._arm is None:
            raise RuntimeError("arm not connected")
        if self._stopped:
            return
        q = np.asarray(q, dtype=float).reshape(-1)[: self.n_joints]
        with self._lock:
            self._arm.arm.send_mit(q, kp=self._mit_kp, kd=self._mit_kd)
        self._last_cmd_q = q.copy()

    def set_gripper(self, pos: float, effort: float = 1.0) -> None:
        """Set gripper target position for smooth tracking by control loop."""
        if self._arm is None or not self._arm.has_gripper or self._stopped:
            return
        self._gripper_target = float(pos)
    
    def _gripper_control_loop(self) -> None:
        """Continuously send MIT commands to track gripper_target (10ms period)."""
        while self._gripper_loop_running:
            if self._arm is not None and self._arm.has_gripper and not self._stopped:
                try:
                    with self._lock:
                        self._arm.gripper.send_mit(
                            np.array([self._gripper_target]),
                            kp=np.array([self._grip_kp]),
                            kd=np.array([self._grip_kd])
                        )
                except Exception:
                    pass
            time.sleep(0.01)  # 10ms control loop

    def close_gripper_two_stage(
        self,
        width_frac_stage1: float = 0.5,
        width_frac_stage2: float = 0.85,
        effort: float = 0.6,
        stall_vel: float = 0.05,
        timeout_s: float = 3.0,
    ) -> float | None:
        """ASPIRE-style two-stage close; returns the final gripper position
        (rad) or None if feedback was unavailable.

        A stall (jaws stopped short of target) is the grasp-contact signal.
        To avoid mistaking motor spin-up for a stall: skip a grace period
        after each command and require TWO consecutive slow samples with some
        minimum travel. A stage-1 stall does NOT end the close -- stage 2
        still runs at full effort to seat the grip.
        """
        span = self._grip_closed - self._grip_open
        start_pos = self._gripper_pos()
        for frac, eff in ((width_frac_stage1, effort * 0.7), (width_frac_stage2, effort)):
            target = self._grip_open + span * frac
            self.set_gripper(target, effort=eff)
            time.sleep(0.3)  # spin-up grace period
            slow_samples = 0
            deadline = time.monotonic() + timeout_s / 2
            while time.monotonic() < deadline:
                time.sleep(0.15)
                pos = self._gripper_pos()
                vel = self._gripper_vel()
                if pos is None:
                    continue
                if abs(pos - target) < 0.1:
                    break  # reached this stage's target
                traveled = start_pos is not None and abs(pos - start_pos) > 0.05
                if vel is not None and abs(vel) < stall_vel and traveled:
                    slow_samples += 1
                    if slow_samples >= 2:
                        break  # stalled on the object: proceed to seat stage
                else:
                    slow_samples = 0
        return self._gripper_pos()

    def _gripper_vel(self) -> float | None:
        try:
            with self._lock:
                m = self._arm.gripper._mm[self._arm.gripper.joint_names[0]]
                return float(m.robstride_get_param_f32(MECH_VEL))
        except Exception:
            return None

    def open_gripper(self, distance_m: float | None = None) -> None:
        """Open gripper, optionally to a specific distance (adaptive opening).
        
        Args:
            distance_m: Target opening distance in meters (0.0 to max_width_m).
                       If None, opens fully to _grip_open.
        """
        if distance_m is not None:
            # Calculate target position based on distance
            max_width = float(self._cfg.get("gripper", {}).get("max_width_m", 0.09))
            distance_m = float(np.clip(distance_m, 0.0, max_width))
            # Linear interpolation: 0m -> _grip_closed, max_width -> _grip_open
            span = self._grip_open - self._grip_closed
            target = self._grip_closed + span * (distance_m / max_width)
            self.set_gripper(target, effort=0.8)
        else:
            self.set_gripper(self._grip_open, effort=0.8)

    def stop(self) -> None:
        """Soft stop: freeze at the last commanded pose and refuse new
        commands. Does NOT torque-off (RebotArm.estop == disable_all would
        make a loaded arm free-fall); use hard_estop() for that."""
        self._stopped = True
        if self._arm is not None and self._last_cmd_q is not None:
            try:
                with self._lock:
                    self._arm.arm.send_mit(
                        self._last_cmd_q, kp=self._mit_kp, kd=self._mit_kd
                    )
            except Exception:
                pass

    def hard_estop(self) -> None:
        """Torque-off every motor. The arm WILL fall under gravity."""
        self._stopped = True
        if self._arm is not None:
            try:
                with self._lock:
                    self._arm.estop()
            except Exception:
                pass
