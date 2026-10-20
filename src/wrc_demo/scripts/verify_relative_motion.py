"""Relative-motion hand-eye verification (no base-frame origin needed).

The standard ``calib_top_orbbec --verify`` flow needs the operator to know
each marker placement's (x, y, z) in the robot base frame, which requires
either a fixture or careful measurement from the arm base. That's hard
on-site.

This script does the indirect check the operator actually wants:

  1. Place the marker at position 1, press Enter.
  2. The top camera grabs frames until the marker is stable, then
     reports the marker origin in base frame:
        pos_1 = (T_cam2base @ T_marker2cam)[:3, 3]
  3. Move the marker to position 2 (e.g. 10 cm to the right), press Enter.
  4. Script reports pos_2 and the camera-predicted offset
        Δ_camera = pos_2 - pos_1
  5. Repeat for ``--n`` positions.
  6. Operator measures the same offsets with a ruler and compares to
     Δ_camera. If every Δ_camera matches its ruler-measured Δ to within
     a few mm, the calibration is geometrically consistent (rotation,
     scale, base-frame alignment all check out at relative level).

Absolute base-frame accuracy isn't directly tested here, but for
grasping that's usually fine: the arm always moves *relative* to its
current pose.

Usage:
    $PY -m wrc_demo.scripts.verify_relative_motion \
        --npz data/calibration/hand_eye_top.json \
        --n 3 \
        --marker-size 0.10 --aruco-dict 4x4_50 --aruco-marker-id 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..calibration import CalibrationResult, load_calibration  # type: ignore
from ..calibration.aruco_session import ArUcoSession, ArUcoDetection  # type: ignore
from ..perception.camera_base import CameraError  # type: ignore
from ..perception.orbbec_camera import OrbbecCamera, list_orbbec_devices  # type: ignore
from .calib_top_orbbec import (  # type: ignore
    CALIB_DIR,
    _get_serial_from_binding,
)


# Reuse the calibration script's camera-open + ArUco setup so the
# operator doesn't have to re-derive the binding.
def _open_top_camera(width: int = 1280, height: int = 720, fps: int = 30) -> OrbbecCamera:
    serial = _get_serial_from_binding("top")
    if not serial:
        raise CameraError(
            "no top serial binding; run `python -m wrc_demo.scripts."
            "calib_top_orbbec --bind top --index <i>` first"
        )
    from ..config import Cfg
    cam = OrbbecCamera(
        Cfg({
            "type": "orbbec",
            "serial": serial,
            "width": width,
            "height": height,
            "fps": fps,
            "min_depth_m": 0.25,
            "max_depth_m": 2.5,
        })
    )
    cam.open()
    return cam


def _grab_stable_marker(
    cam: OrbbecCamera,
    aruco: ArUcoSession,
    K: np.ndarray,
    D: Optional[np.ndarray],
    marker_size_m: float,
    stable_frames: int = 4,
    settle_time: float = 1.0,
    marker_timeout: float = 4.0,
    aruco_marker_id: Optional[int] = None,
) -> Optional[ArUcoDetection]:
    """Same wait-for-stable-frame loop as calib_top_orbbec._grab_color
    + _wait_for_stable_marker, inlined here to keep the script self-contained."""
    best: Optional[ArUcoDetection] = None
    stable_count = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_time + marker_timeout:
        try:
            frame = cam.get_frame()
        except Exception:
            time.sleep(0.05)
            continue
        det = aruco.detect(frame.rgb, K, D, marker_size_m, target_id=aruco_marker_id)
        if det is None:
            stable_count = 0
            time.sleep(0.05)
            continue
        stable_count += 1
        best = det
        if stable_count >= stable_frames:
            return det
    return best


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", required=True, help="saved hand_eye_top.json")
    p.add_argument("--n", type=int, default=3, help="number of positions to capture (>=2)")
    p.add_argument("--marker-size", type=float, default=0.10)
    p.add_argument("--aruco-dict", default="4x4_50")
    p.add_argument("--aruco-marker-id", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--settle-time", type=float, default=1.0)
    p.add_argument("--marker-timeout", type=float, default=4.0)
    p.add_argument("--stable-frames", type=int, default=4)
    args = p.parse_args(argv)

    if args.n < 2:
        print("--n must be >= 2 to compute a relative offset", file=sys.stderr)
        return 2

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"[verify-relative] {npz_path} does not exist", file=sys.stderr)
        return 2

    res: CalibrationResult = load_calibration(npz_path)
    X = np.asarray(res.T_cam2base, dtype=float)
    print(f"[verify-relative] loaded {npz_path}")
    print(f"[verify-relative] T_cam2base translation = {X[:3, 3].tolist()} (m)")
    print(f"[verify-relative] n_positions = {args.n}")

    cam = _open_top_camera(args.width, args.height, args.fps)
    try:
        K = cam.K
        D = cam.D
        if K is None:
            print("[verify-relative] camera opened but intrinsics K is None; abort", file=sys.stderr)
            return 1
        aruco = ArUcoSession(dict_name=args.aruco_dict)

        positions_m: list[np.ndarray] = []
        for i in range(args.n):
            try:
                input(
                    f"\n[verify-relative] Place marker at position {i + 1}/{args.n}. "
                    "Press Enter when ready..."
                )
            except EOFError:
                return 1
            det = _grab_stable_marker(
                cam, aruco, K, D,
                marker_size_m=args.marker_size,
                stable_frames=args.stable_frames,
                settle_time=args.settle_time,
                marker_timeout=args.marker_timeout,
                aruco_marker_id=args.aruco_marker_id,
            )
            if det is None:
                print(
                    f"[verify-relative] marker not detected at position {i + 1}; "
                    "abort (move marker into the top camera FOV and retry)",
                    file=sys.stderr,
                )
                return 1
            pos_base = (X @ np.asarray(det.T_marker2cam, dtype=float))[:3, 3]
            positions_m.append(pos_base)
            print(
                f"  position {i + 1}: base-frame "
                f"(x={pos_base[0]:+.3f}, y={pos_base[1]:+.3f}, z={pos_base[2]:+.3f}) m"
            )
            print(
                f"    aruco reproj = {det.reprojection_rmse_px:.2f}px, "
                f"marker_id = {det.marker_id}"
            )

        print("\n[verify-relative] camera-predicted relative offsets:")
        print("  format: Δ(i, i-1) = (dx, dy, dz) in mm")
        max_err_mm = 0.0
        measured_errs: list[float] = []
        for i in range(1, len(positions_m)):
            delta = (positions_m[i] - positions_m[i - 1]) * 1000.0
            print(
                f"  Δ({i + 1},{i}): "
                f"({delta[0]:+6.1f}, {delta[1]:+6.1f}, {delta[2]:+6.1f}) mm"
            )

        print("\n[verify-relative] now measure each Δ with a ruler and compare.")
        print(
            "  for each i, type the ruler-measured (dx, dy, dz) in mm when prompted.\n"
            "  leave any component blank to skip it (Enter)."
        )
        rmse_terms: list[float] = []
        for i in range(1, len(positions_m)):
            try:
                raw = input(
                    f"  ruler Δ({i + 1},{i}) dx,dy,dz (mm) [or Enter to skip]: "
                ).strip()
            except EOFError:
                raw = ""
            if not raw:
                continue
            # Accept space- or comma-separated values. Strip parens too
            # so "(-1.5, 0, 0)" or "-1.5 0 0" both work.
            cleaned = raw.replace("(", " ").replace(")", " ").replace(",", " ")
            try:
                parts = [float(t) for t in cleaned.split()]
            except ValueError:
                print(f"    could not parse {raw!r}; skip", file=sys.stderr)
                continue
            if len(parts) != 3:
                print(f"    need 3 numbers; skip", file=sys.stderr)
                continue
            measured = np.asarray(parts, dtype=float)
            camera_pred = (positions_m[i] - positions_m[i - 1]) * 1000.0
            diff = camera_pred - measured
            norm = float(np.linalg.norm(diff))
            measured_errs.append(norm)
            print(
                f"    camera = ({camera_pred[0]:+.1f},{camera_pred[1]:+.1f},{camera_pred[2]:+.1f}) mm, "
                f"ruler = ({measured[0]:+.1f},{measured[1]:+.1f},{measured[2]:+.1f}) mm, "
                f"|Δ| = {norm:.2f} mm"
            )
            rmse_terms.append(norm * norm)

        if rmse_terms:
            rmse = float(np.sqrt(np.mean(rmse_terms)))
            mx = max(rmse_terms) ** 0.5
            print(f"\n[verify-relative] offset RMSE = {rmse:.2f} mm, max = {mx:.2f} mm")
            print(
                "[verify-relative] relative-motion check "
                f"{'PASS (≤10mm)' if rmse <= 10.0 else 'FAIL (>10mm)'}."
            )
            return 0 if rmse <= 10.0 else 1
        print("\n[verify-relative] no ruler measurements entered; "
              "just compare the printed Δ rows to your ruler by eye.")
        return 0
    finally:
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
