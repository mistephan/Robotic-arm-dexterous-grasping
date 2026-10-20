"""wrc_demo control backends + the ``make_arm`` factory.

After the SDK migration (2026-07-24) this package contains:

- ``MockArm`` — perfect-tracking test double; the harness is real so
  per-waypoint approval exercises the same code as production.
- ``make_arm(cfg, kinematics)`` — single factory used by ``demo.py`` /
  ``mcp_server.py``: returns the right arm type (MockArm or
  a ``SafeArm`` wrapping the vendored SDK).

Real hardware (``rebot_rs``) is now driven by the vendored SDK at
``sdk/reBotArm_control_py/`` and wrapped by ``SafeArm`` in
``wrc_demo.safety.harness``. There is no ``ArmBase`` ABC here anymore
— the SDK is the single source of truth for the motion surface, and
``MockArm`` / ``SafeArm`` are duck-typed to share it.

Note (2026-08-28): the Isaac Sim backend and its TCP bridge have been
removed — the upstream project was never productionized and we no
longer ship it. ``make_arm`` only accepts ``mock`` and ``rebot_rs``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Union

import numpy as np

from ..config import Cfg, SDK_DIR
from ..safety.harness import Kinematics, SafeArm, SafetyHarness, SafetyLimits
from .mock_arm import MockArm

# Union of every "arm the skill runtime can hold" type. Both expose the
# same public surface (move_joints / move_cartesian / set_gripper /
# close_gripper_two_stage / stop / resume / disconnect / harness).
Arm = Union[MockArm, SafeArm]

__all__ = [
    "Arm",
    "MockArm",
    "make_arm",
]


def make_arm(cfg: Cfg, kinematics: Kinematics | None = None) -> Arm:
    """Build the arm requested by ``cfg.arm['type']``.

    For ``rebot_rs`` this assembles the full real-arm stack:

        RebotArm(hw_yaml) → connect → RebotArmEndPose(arm, mit)
            → WrcGripper.start() (extends the SDK control loop with
              gripper_tick)
            → SafeArm(arm, ctrl, harness, kin, arm_cfg)
            → RebotGraspBridge(ctrl, gripper, harness) attached as
              ``safe_arm.grasp_bridge``

    For ``mock`` it returns a ``MockArm`` (perfect tracking, real
    harness — exercises every gating code path used in production).
    """
    arm_type = str(cfg.arm.get("type", "mock"))

    if arm_type == "mock":
        arm = MockArm(cfg.arm, kinematics=kinematics)
        arm.connect()
        arm.grasp_bridge = None
        return arm

        arm.connect()
        arm.grasp_bridge = None
        return arm

    if arm_type == "rebot_rs":
        ee_cfg = cfg.arm.get("end_effector", Cfg({}))
        ee_type = str(ee_cfg.get("type", "parallel_gripper"))
        if ee_type == "linkerhand_o6" and not (
            bool(ee_cfg.get("tcp_calibrated", False))
            and bool(ee_cfg.get("poses_calibrated", False))
        ):
            raise RuntimeError(
                "LinkerHand O6 profile is locked: calibrate the flange-to-TCP "
                "transform and hand poses, then set tcp_calibrated and "
                "poses_calibrated true"
            )
        # Real hardware: vendored SDK + WrcGripper + SafeArm wrapping.
        # wrc_demo ``__init__.py`` already inserts ``wrc_demo/sdk/`` into
        # ``sys.path``, so ``reBotArm_control_py`` is importable here
        # without any PYTHONPATH edit from the operator.
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose

        from ..skills.rebot_grasp_bridge import RebotGraspBridge
        from .gripper import WrcGripper

        hw_yaml_name = str(cfg.arm.get("hw_yaml", "rebotarm_rs.yaml"))
        # SDK's RebotArm resolves hw_yaml relative to sdk/config/.
        sdk_cfg_path = SDK_DIR / "config" / hw_yaml_name
        if not sdk_cfg_path.exists():
            raise FileNotFoundError(
                f"SDK hardware config not found: {sdk_cfg_path} "
                f"(wrc_demo/sdk/config/{hw_yaml_name})"
            )
        sdk_arm = RebotArm(hw_yaml=hw_yaml_name)
        sdk_arm.connect()
        # Clear stale fault bits on every motor.  The SDK enable()
        # does NOT call clear_error; motors left in fault state from a
        # previous session silently reject MIT commands.
        #
        # Some RobStride firmwares ACKing the clear_error with a small
        # delay right after power-up or right after the previous session
        # bus-off'd.  Retry up to 3 times with a 100 ms gap so transient
        # comm_type=4 ack timeouts don't kill the boot.
        from motorbridge.errors import CallError as _CE
        for _name, _motor in sdk_arm._motor_map.items():
            for _attempt in range(3):
                try:
                    _motor.clear_error()
                    break
                except _CE as _e:
                    if "ack timeout" not in str(_e).lower() or _attempt == 2:
                        raise
                    import time as _t; _t.sleep(0.1)
        sdk_ctrl = RebotArmEndPose(
            sdk_arm,
            arm_control_mode="mit",
            use_gravity_ff=True,
            gravity_ff_scale=np.asarray(
                cfg.arm.get(
                    "gravity_ff_scale",
                    [1.0] * int(cfg.arm.get("n_joints", 6)),
                ), dtype=float,
            ),
        )
        harness = SafetyHarness(
            SafetyLimits.from_config(cfg.safety), kinematics=kinematics
        )
        end_effector = None
        if ee_type == "linkerhand_o6":
            from ..end_effectors import LinkerHandGraspAdapter, LinkerHandO6, O6Config
            allowed = set(O6Config.__dataclass_fields__)
            options = {k: v for k, v in ee_cfg.as_dict().items() if k in allowed}
            end_effector = LinkerHandO6(
                O6Config(**options),
                abort_requested=lambda: harness.estopped,
            )
            try:
                end_effector.connect()
                baseline = end_effector.calibrate_tactile_baseline()
                print(
                    f"[wrc-demo] LinkerHand O6 ready on "
                    f"{end_effector.cfg.can_interface}; tactile baseline "
                    f"peak={baseline['peak']}",
                    file=sys.stderr,
                )
            except Exception as exc:
                end_effector.close()
                try:
                    sdk_arm.disconnect()
                except Exception:
                    pass
                raise RuntimeError(
                    "LinkerHand O6 startup preflight failed; verify can_hand "
                    "and ensure the open hand is not touching anything: "
                    f"{exc}"
                ) from exc
            gripper = LinkerHandGraspAdapter(end_effector)
            # No WrcGripper wrapper exists to start the arm controller.
            sdk_ctrl.start()
        else:
            # WrcGripper starts and extends the SDK arm control loop.
            gripper = WrcGripper(sdk_arm, sdk_ctrl)
            gripper.start()

        safe_arm = SafeArm(sdk_arm, sdk_ctrl, harness, kinematics, cfg.arm)
        safe_arm.end_effector = end_effector
        safe_arm.grasp_bridge = RebotGraspBridge(
            sdk_ctrl,
            gripper,
            harness=harness,
            settle_tol_rad=float(cfg.arm.get("bridge_settle_tol_rad", 0.02)),
            settle_timeout_ms=int(cfg.arm.get("bridge_settle_timeout_ms", 3000)),
            settle_samples=int(cfg.arm.get("bridge_settle_samples", 3)),
        )
        return safe_arm

    raise ValueError(
        f"unknown arm type {arm_type!r}; "
        f"expected one of: mock, rebot_rs"
    )
