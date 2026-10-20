"""CLI: eye-in-hand calibration for the wrist camera.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

Mirrors ``calib_top_orbbec.py`` but for eye-in-hand (wrist-mounted) cameras:
25 preset Cartesian poses covering the wrist link's working volume; both
``--auto`` (drives through preset poses) and ``--manual`` (operator hand-
guides the arm with gravity comp, presses ENTER to capture) modes.

Solver is the same ``joint_solve_restart`` used by ``calib_top_orbbec.py``
(joint SE(3) LM with Huber loss + multi-restart). The general solve gives
``X = T_cam2base`` and ``Y = T_marker2gripper``; we derive
``T_cam2gripper`` by inverting per-sample (see ``_derive_T_cam2gripper``).

Output: ``data/calibration/hand_eye_wrist.json`` (schema v1) + ``.npz``
sidecar. Schema v1 is what ``wrc_demo.perception.grounding.Extrinsics``
loads for ``eye_in_hand`` mode (``T_cam2gripper`` key).

Run via ``python -m wrc_demo.scripts.calib_wrist ...`` (editable install
means no PYTHONPATH gymnastics needed).

History: this script was originally at ``scripts/calib_wrist.py`` and used
the OpenCV-5-compatible ``Park & Martin`` solver with an old JSON schema
(no ``schema_version`` / ``camera_serial`` / ``frame_convention`` keys).
Migrated in-tree 2026-08-27 to match ``calib_top_orbbec.py``'s pattern;
``scripts/calib_wrist.py`` was deleted in the same pass.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..calibration import (
    ArUcoSession,
    CalibrationResult,
    CalibrationSample,
    CamerasBinding,
    CameraBinding,
    SCHEMA_VERSION,
    joint_solve_restart,
    save_calibration,
)
from ..config import PACKAGE_ROOT
from ..perception.orbbec_camera import OrbbecCamera
from ..safety.harness import Kinematics
from ._calib_cleanup import safe_park_and_disconnect


# Stable layout (matches calib_top_orbbec.py)
CALIB_DIR = PACKAGE_ROOT / "data" / "calibration"
URDF_DEFAULT = (
    PACKAGE_ROOT
    / "assets" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
)

# Headless / SSH fallback
SNAP_DIR = Path("/tmp/calib_frames")
SNAP_EVERY = 10


def _detect_gui_capable() -> bool:
    """True if cv2.namedWindow works (real display server is attached)."""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


# 25 preset Cartesian poses (x, y, z, roll, pitch, yaw) in meters/radians.
# Tuned for the reBot DevArm wrist link (short reach, x bounded 0.20-0.40 m).
CALIB_POSES = [
    (0.28, 0.00, 0.30, 0.00, 0.50, 0.00),
    (0.28, 0.00, 0.30, 0.00, 0.70, 0.00),
    (0.28, 0.00, 0.30, 0.00, 0.90, 0.00),
    (0.28, -0.15, 0.30, 0.00, 0.60, 0.00),
    (0.28, 0.15, 0.30, 0.00, 0.60, 0.00),
    (0.28, -0.20, 0.32, 0.00, 0.50, 0.00),
    (0.28, 0.20, 0.32, 0.00, 0.50, 0.00),
    (0.35, 0.00, 0.28, 0.00, 0.60, 0.00),
    (0.22, 0.00, 0.35, 0.00, 0.70, 0.00),
    (0.38, 0.00, 0.25, 0.00, 0.80, 0.00),
    (0.28, 0.00, 0.30, 0.00, 0.60, -0.30),
    (0.28, 0.00, 0.30, 0.00, 0.60, 0.30),
    (0.28, 0.00, 0.30, 0.00, 0.60, -0.60),
    (0.28, 0.00, 0.30, 0.00, 0.60, 0.60),
    (0.28, 0.00, 0.30, -0.40, 0.60, 0.00),
    (0.28, 0.00, 0.30, 0.40, 0.60, 0.00),
    (0.28, 0.00, 0.40, 0.00, 0.50, 0.00),
    (0.28, 0.10, 0.40, 0.00, 0.50, 0.30),
    (0.28, -0.10, 0.40, 0.00, 0.50, -0.30),
    (0.28, 0.00, 0.22, 0.00, 0.80, 0.00),
    (0.28, 0.08, 0.22, 0.00, 0.80, 0.20),
    (0.28, -0.08, 0.22, 0.00, 0.80, -0.20),
    (0.32, 0.15, 0.26, 0.30, 0.70, 0.40),
    (0.32, -0.15, 0.26, -0.30, 0.70, -0.40),
    (0.24, 0.12, 0.35, 0.20, 0.55, 0.25),
]

MIN_CALIB_SAMPLES = 5
AUTO_MOVE_DURATION_S = 3.0
AUTO_SETTLE_EXTRA_S = 1.5  # wait an extra 1.5s after move completes


def _pose_to_T(xyz, rpy):
    """Convert (x, y, z, roll, pitch, yaw) to a 4x4 homogeneous matrix."""
    import pinocchio as pin
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = xyz
    return T


def _grab_frame(camera):
    """Grab a frame, return color BGR or None."""
    try:
        f = camera._grab()
        return None if f is None else f.rgb
    except Exception:
        return None


def _grab_q(robot, kin):
    """Read current joint positions and compute FK.

    Returns (q, T_gripper2base) or (None, None) on failure.
    """
    try:
        q = robot.get_state()[0][:6]
        return q, kin.fk(q)
    except Exception:
        return None, None


def _save_snapshot(img, det):
    global _snap_counter
    _snap_counter += 1
    if det is not None or _snap_counter % SNAP_EVERY == 0:
        try:
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            label = "det" if det is not None else "nodet"
            cv2.imwrite(str(SNAP_DIR / f"f{_snap_counter:05d}_{label}.png"), img)
        except Exception:
            pass


_snap_counter = 0


def collect_auto(camera, robot, arm_ctrl, kin, aruco, K, D, marker_size_m,
                 show_window, win_name):
    """Auto mode: iterate through preset poses, capture ArUco + FK."""
    samples = []
    last_q = np.zeros(6)

    if show_window:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            show_window = False

    if not show_window:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Headless: frames saved to {SNAP_DIR}/")

    for idx, pose in enumerate(CALIB_POSES):
        xyz = pose[:3]
        rpy = pose[3:]
        pose_label = (
            f"xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}) "
            f"rpy=({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f})"
        )
        print(f"\nPose {idx+1}/{len(CALIB_POSES)}: {pose_label}")

        # IK seed from the last successful joint config
        ik = kin.ik(_pose_to_T(xyz, rpy), last_q, retries=16)
        if not ik.success:
            print(f"  IK failed (err={ik.error:.4f}), skipping")
            continue
        q_target = ik.q[:6]
        last_q = q_target

        success = arm_ctrl.move_to_traj(
            xyz[0], xyz[1], xyz[2],
            rpy[0], rpy[1], rpy[2],
            duration=AUTO_MOVE_DURATION_S,
        )
        if not success:
            print("  move_to_traj failed, skipping")
            continue

        # Settle
        time.sleep(AUTO_MOVE_DURATION_S + AUTO_SETTLE_EXTRA_S)

        # Search for marker: 3 stable detections within 3s
        stable_count = 0
        best_det = None
        t_timeout = time.monotonic() + 3.0
        while time.monotonic() < t_timeout:
            if show_window:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), ord('c'), ord('C'), 27):
                    return samples

            color = _grab_frame(camera)
            if color is None:
                time.sleep(0.05)
                continue

            det = aruco.detect(color, K, D, marker_size_m,
                               marker_z_toward_camera=True)
            overlay = aruco.draw_aruco_overlay(color, det) if det else color.copy()

            cv2.putText(overlay, f"Pose {idx+1}/{len(CALIB_POSES)}  Samples: {len(samples)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            state_str = f"searching ({stable_count}/3)" if det is not None else "waiting ArUco"
            cv2.putText(overlay, state_str, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if show_window:
                cv2.imshow(win_name, overlay)
            else:
                _save_snapshot(overlay, det)

            if det is None:
                stable_count = 0
                best_det = None
                time.sleep(0.05)
                continue

            stable_count += 1
            best_det = det
            if stable_count >= 3:
                break

        if best_det is None:
            print("  No marker within 3s, skipping")
            continue

        q, T_gripper2base = _grab_q(robot, kin)
        if T_gripper2base is None:
            print("  FK read failed, skipping")
            continue

        samples.append(CalibrationSample(
            pose_label=f"s{idx+1}",
            T_gripper2base=T_gripper2base,
            T_marker2cam=best_det.T_marker2cam,
            aruco_reprojection_px=best_det.reprojection_rmse_px,
            inlier=True,
        ))
        print(f"  Captured sample {len(samples)}: "
              f"marker {best_det.marker_id}, "
              f"rmse={best_det.reprojection_rmse_px:.2f}px")

        if show_window:
            cv2.imshow(win_name, overlay)
            cv2.waitKey(120)
        else:
            try:
                cv2.imwrite(str(SNAP_DIR / f"capture_{len(samples):02d}.png"), overlay)
            except Exception:
                pass

    if show_window:
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass

    return samples


def collect_manual(camera, robot, arm_ctrl, kin, aruco, K, D, marker_size_m,
                   show_window, win_name):
    """Manual mode: user moves arm by hand, ENTER to capture, q to finish."""
    samples = []

    if show_window:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        except cv2.error:
            show_window = False

    if not show_window:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Headless: frames saved to {SNAP_DIR}/")

    print("\n=== Manual Mode ===")
    print("Move arm by hand. Press Enter to capture, 'q' to finish.\n")

    while True:
        color = _grab_frame(camera)
        if color is None:
            time.sleep(0.05)
            continue

        det = aruco.detect(color, K, D, marker_size_m,
                           marker_z_toward_camera=True)
        overlay = aruco.draw_aruco_overlay(color, det) if det else color.copy()
        cv2.putText(overlay, f"Samples: {len(samples)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if show_window:
            cv2.imshow(win_name, overlay)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key not in (13, 10):
                continue
        else:
            _save_snapshot(overlay, det)
            if det is not None:
                print(f"  Marker {det.marker_id} visible, "
                      f"z={det.T_marker2cam[2,3]*1000:.0f}mm, "
                      f"rmse={det.reprojection_rmse_px:.2f}px")
            else:
                print("  No marker visible")
            line = input("  Enter=capture, q=quit: ").strip().lower()
            if line in ('q', 'quit'):
                break
            if line != "":
                continue

        if det is None:
            print("  No marker, skipping")
            continue

        q, T_gripper2base = _grab_q(robot, kin)
        if T_gripper2base is None:
            print("  FK read failed, skipping")
            continue

        samples.append(CalibrationSample(
            pose_label=f"manual{len(samples)+1}",
            T_gripper2base=T_gripper2base,
            T_marker2cam=det.T_marker2cam,
            aruco_reprojection_px=det.reprojection_rmse_px,
            inlier=True,
        ))
        print(f"  Captured sample {len(samples)}")

    if show_window:
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass

    return samples


def _derive_T_cam2gripper(samples, joint_result) -> np.ndarray:
    """joint_solve_restart returns T_cam2base (general AX=XB); for eye-in-hand
    we want T_cam2gripper. Derive by averaging across inlier samples::

        T_gripper2cam[i] = inv(T_cam2base) @ T_gripper2base[i]
        T_cam2gripper = inv(T_gripper2cam[i])    (any i; mean for noise)
    """
    X = np.asarray(joint_result.T_cam2base, dtype=np.float64)
    inlier_idx = (list(joint_result.inliers)
                  if joint_result.inliers else list(range(len(samples))))
    Ts = []
    for i in inlier_idx:
        G = samples[i].T_gripper2base
        T_gripper2cam_i = np.linalg.inv(X) @ G
        Ts.append(np.linalg.inv(T_gripper2cam_i))
    T = np.mean(Ts, axis=0)
    # Renormalize the rotation block (mean-of-rotations isn't a rotation).
    try:
        from scipy.spatial.transform import Rotation as _R
        T[:3, :3] = _R.from_matrix(T[:3, :3]).mean().as_matrix()
    except Exception:
        T = Ts[0]  # scipy not available: use first sample
    return T


def save_result(samples, joint_result, camera_serial, marker_size_m,
                aruco_dict, marker_id, output_path):
    """Wrap joint_result into the canonical CalibrationResult schema v1
    and persist to ``output_path``."""
    T_cam2gripper = _derive_T_cam2gripper(samples, joint_result)
    t = T_cam2gripper[:3, 3]
    inlier_idx = (list(joint_result.inliers)
                  if joint_result.inliers else list(range(len(samples))))
    print(
        f"[Result] T_cam2gripper translation: "
        f"x={t[0]:+.4f} y={t[1]:+.4f} z={t[2]:+.4f} m"
    )
    print(f"[Result] inliers: {len(inlier_idx)}/{len(samples)}")

    cal_result = CalibrationResult(
        schema_version=SCHEMA_VERSION,
        frame_convention="wrc_demo",
        frame_convention_note="T_cam2gripper derived from joint_solve_restart via per-sample inversion",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        method="eye_in_hand_joint_solve",
        mode="eye_in_hand",
        camera_serial=camera_serial,
        camera_index_when_bound=0,
        marker_size_m=marker_size_m,
        aruco_dict=aruco_dict,
        aruco_marker_id=marker_id,
        n_samples=len(samples),
        n_inliers=len(inlier_idx),
        n_outliers=len(samples) - len(inlier_idx),
        outlier_indices=tuple(i for i in range(len(samples))
                              if i not in inlier_idx),
        T_cam2gripper=T_cam2gripper.tolist(),
        residual_metrics={
            "rmse_translation_m": joint_result.metrics.get("rmse_translation_m", 0.0),
            "rmse_rotation_rad": joint_result.metrics.get("rmse_rotation_rad", 0.0),
        },
        samples=[s.to_json() for s in samples],
        camera_binding=CamerasBinding(
            cameras=[CameraBinding(profile="orbbec_wrist", index=0,
                                   serial=camera_serial)],
            active_index=0,
        ).to_json(),
    )
    save_calibration(cal_result, output_path)
    print(f"[Result] saved to {output_path}")

    # Sidecar .npz (matches the layout pre-migration had, but with full data).
    npz_path = output_path.with_suffix(".npz")
    np.savez(
        str(npz_path),
        T_cam2gripper=T_cam2gripper,
        T_cam2base=np.asarray(joint_result.T_cam2base, dtype=np.float64),
        T_marker2gripper=np.asarray(joint_result.T_marker2gripper, dtype=np.float64),
        mode=np.array(["eye_in_hand"]),
        n_samples=np.array([len(samples)]),
        method=np.array(["joint_solve_restart"]),
    )
    print(f"[Result] sidecar saved to {npz_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Wrist camera eye-in-hand calibration")
    p.add_argument("--manual", action="store_true",
                   help="Manual mode (operator hand-guides the arm; "
                        "requires RebotArm MIT-mode + gravity comp)")
    p.add_argument("--marker-size", type=float, default=0.05)
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--dict", default="4x4_50",
                   help="ArUco dict name (default: 4x4_50)")
    p.add_argument("--output-dir", default=str(CALIB_DIR / "hand_eye_wrist.json"))
    # output-dir is allowed to be a file path (the .json); we derive the npz
    # sidecar from it.
    p.add_argument("--camera-serial", default="AY3794300W4")
    args = p.parse_args()

    show_window = _detect_gui_capable()
    if not show_window:
        print("[calib] opencv-python-headless: frames -> /tmp/calib_frames/")

    print("Initializing camera...")
    from ..config import Cfg  # late import to keep top-level clean
    camera = OrbbecCamera(Cfg({"serial": args.camera_serial}))
    camera.open()

    print("Initializing arm...")
    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose
    robot = RebotArm()
    robot.connect()
    arm_ctrl = RebotArmEndPose(robot, arm_control_mode="mit",
                               use_gravity_ff=not args.manual)
    if args.manual:
        # Manual mode requires MIT-mode gravity comp so the arm floats under
        # the operator's hand. Auto mode uses position control instead.
        arm_ctrl.start(mode="mit", gravity_ff=True)
    else:
        arm_ctrl.start()

    kin = Kinematics(
        model_path=str(URDF_DEFAULT),
        ee_frame='gripper_end',
        n_controlled=6,
        joint_signs=[-1]*6,
    )

    aruco = ArUcoSession(dict_name=args.dict)

    K = camera.K
    D = camera.D
    if K is None:
        print("Waiting for camera intrinsics...")
        time.sleep(2.0)
        _grab_frame(camera)
        K = camera.K
        D = camera.D
    if K is None:
        print("ERROR: Could not get camera intrinsics!")
        return 1
    print(f"Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    win_name = "Wrist Calibration"

    samples: list[CalibrationSample] = []
    try:
        if args.manual:
            samples = collect_manual(
                camera, robot, arm_ctrl, kin, aruco,
                K, D, args.marker_size, show_window, win_name,
            )
        else:
            samples = collect_auto(
                camera, robot, arm_ctrl, kin, aruco,
                K, D, args.marker_size, show_window, win_name,
            )

        if len(samples) < MIN_CALIB_SAMPLES:
            print(f"\nError: only {len(samples)} samples "
                  f"(need >= {MIN_CALIB_SAMPLES})")
            return 1

        print(f"\n[Solve] running joint_solve_restart on {len(samples)} samples...")
        joint_result = joint_solve_restart(samples)
        out_path = Path(args.output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_result(samples, joint_result,
                    camera_serial=args.camera_serial,
                    marker_size_m=args.marker_size,
                    aruco_dict=args.dict,
                    marker_id=args.marker_id,
                    output_path=out_path)

    except KeyboardInterrupt:
        print("\n[Ctrl+C]")

    finally:
        def _safe_home_zero() -> bool | None:
            if arm_ctrl is None:
                return None
            arm_ctrl.safe_home()
            return True

        def _do_disconnect() -> None:
            if robot is None:
                return
            try:
                arm_ctrl.end()
            except Exception:
                pass
            robot.disconnect()

        safe_park_and_disconnect(
            park_fn=_safe_home_zero,
            disconnect_fn=_do_disconnect,
            label="calib_wrist",
        )

        try:
            camera.close()
        except Exception:
            pass
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
