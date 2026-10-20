"""Emergency stow: drive the arm to mechanical zeros, then power off.

Bypasses the regular demo loop (no perception, no LLM, no harness).
For use after a normal shutdown has left the arm in an unstable pose
that physically free-falls on disconnect.

Usage (sudo because motorbridge / SocketCAN needs root):

    # Either run as a module from the repo root:
    sudo python -m scripts.emergency_stow

    # Or paste through sudo with the venv's python:
    printf '0000\\n' | sudo -S -E env PYTHONNOUSERSITE=1 \\
        "$(command -v python)" \\
        -m scripts.emergency_stow

Why this works without the old ``RebotRSArm`` wrapper:
    The vendored SDK's ``RebotArmEndPose.end()`` calls
    ``safe_home()`` (min-jerk to zeros + settle) and then
    ``RebotArm.disconnect()`` (stop 500 Hz loop + ``disable_all``).
    That's exactly what emergency stow needs. ``SafeArm.disconnect``
    forwards to it.

Before the SDK migration this script wrote ``arm._q_target`` directly
under ``arm._lock`` to avoid a dual-writer race with the SDK's own
500 Hz loop (``HANDOFF_V5_CONTROL_FIX.md`` P0-1.f). With the SDK as
the single actuator owner, that hack is gone — we just call the public
``disconnect`` API.
"""
from __future__ import annotations

import numpy as np

from wrc_demo.config import load_demo_config
from wrc_demo.control import make_arm
from wrc_demo.safety.harness import Kinematics


def main() -> None:
    cfg = load_demo_config(
        camera="orbbec_overhead", arm="rebot_rs", llm="mock"
    )

    # The same URDF the runtime uses — pinocchio model must match the
    # one SafeArm was wired with at build_runtime() time, otherwise FK
    # and joint-limit lookups disagree with what the SDK loop is doing.
    kin = Kinematics(
        urdf_path=str(cfg.arm.urdf),
        ee_frame=str(cfg.arm.get("ee_frame", "gripper_end")),
        arm_joint_count=int(cfg.arm.get("n_joints", 6)),
    )

    # make_arm wires RebotArm + RebotArmEndPose + SafetyHarness +
    # SafeArm. The arm is connected + the SDK 500 Hz loop is started
    # by the factory; we don't have to do any of that by hand.
    arm = make_arm(cfg, kinematics=kin)
    assert hasattr(arm, "move_joints"), (
        f"emergency_stow needs the real SafeArm (got {type(arm).__name__}); "
        f"don't pass --arm mock"
    )

    try:
        # Read current state for the operator log.
        state = arm.get_state()
        q_now = state.q
        print(
            f"[stow] current joints: "
            f"{[round(float(qi), 3) for qi in q_now]}"
        )

        zeros = np.zeros(arm.n_joints, dtype=float)
        # ``move_joints`` auto-stretches duration for the min-jerk
        # velocity cap, so 4.0 s is plenty for an arm that's already
        # near the home pose.
        ok = arm.move_joints(zeros, duration_s=4.0)
        print(f"[stow] move_joints(zeros): {ok}")

        # Verify the arm actually settled at zeros.
        q_after = arm.get_state().q
        max_err = float(np.max(np.abs(q_after)))
        print(
            f"[stow] post-park joints: "
            f"{[round(float(qi), 3) for qi in q_after]}, "
            f"max|err|={max_err:.4f} rad"
        )

        # SafeArm.disconnect → RebotArmEndPose.end → safe_home() (idempotent
        # here, zeros already there) + RebotArm.disconnect (stops the
        # 500 Hz loop and calls disable_all → torque off).
        arm.disconnect()
        print("[stow] arm disconnected, torque off.")
    except Exception as e:
        print(f"[stow] FAILED: {type(e).__name__}: {e}")
        # Even on failure, try to power off so we don't leave the
        # arm holding torque under CAN bus error.
        try:
            arm.disconnect()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()