"""Flip T_cam2base across the marker plane (eye-to-hand flip-ambiguity fix).

Hand-eye calibration with a planar marker (cv2.aruco 4x4/5x5/6x6) has a
well-known 1-DoF degeneracy: the camera is equally well explained by
"sitting above the marker plane looking down" or "sitting below looking
up". The reprojection RMSE is identical in both cases, so the in-sample
fit cannot distinguish them.

When the on-rig operator KNOWS which side the camera is on (e.g. ``the
top camera is ~70 cm above the work plane``), we can resolve the
ambiguity by reflecting the saved ``T_cam2base`` across the average
marker plane.

Mathematically: if (X, Y) is a solution of the hand-eye equation
``G_i @ Y == X @ M_i``, then so is
``(M_mirror @ X, Y @ M_mirror_inv)`` where ``M_mirror`` is a reflection
in the marker plane. In practice we:

1. Recover the average marker plane from the saved samples (centroid
   position + average marker-frame z axis direction in base).
2. Reflect ``X[:3, 3]`` (camera origin) across that plane.
3. Reflect ``X[:3, :3]`` (camera orientation) across the plane.
4. Leave ``Y`` (T_marker2gripper) untouched -- the marker is on the
   gripper, both solutions agree on it.

Usage:
    $PY -m wrc_demo.scripts.flip_hand_eye \\
        --npz data/calibration/hand_eye_top.json \\
        --camera-side above                # "above" or "below" (default: auto-detect)

If ``--camera-side`` is omitted, the script compares the resulting
camera height against the average marker height and picks the side
that puts the camera farther from the marker plane.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..calibration import load_calibration  # type: ignore


def _average_marker_pose(samples):
    """Compute the average marker-plane centroid and normal in base frame.

    The marker frame's +z axis (out of the marker board, toward the
    camera) maps to base frame via ``G_i @ Y @ M_i^{-1}``; the average
    over inlier samples gives the marker-plane normal. The marker
    centroid is ``(G_i @ Y)[:3, 3]`` averaged the same way.
    """
    centroids = []
    normals = []
    for s in samples:
        if not s.inlier:
            continue
        G = np.asarray(s.T_gripper2base, dtype=float)
        Y = np.asarray(s.T_marker2gripper, dtype=float)
        M = np.asarray(s.T_marker2cam, dtype=float)
        # Marker origin in base
        centroids.append((G @ Y)[:3, 3])
        # Marker +z axis (out of board) in base
        # marker_to_base = G @ Y @ inv(M); the marker +z in base is its 3rd column.
        marker_to_base = G @ Y @ np.linalg.inv(M)
        normals.append(marker_to_base[:3, 2])
    centroid = np.mean(np.stack(centroids, axis=0), axis=0)
    normal = np.mean(np.stack(normals, axis=0), axis=0)
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    return centroid, normal


def _reflect_point_across_plane(p, plane_point, plane_normal):
    return p - 2.0 * float(np.dot(plane_normal, p - plane_point)) * plane_normal


def _reflect_rotation_across_plane(R, plane_normal):
    """Reflect the orientation of frame F across a plane with the given
    normal (expressed in the parent frame). Equivalent to conjugating
    the rotation by a Householder reflection."""
    n = plane_normal.astype(float).reshape(3)
    H = np.eye(3) - 2.0 * np.outer(n, n)  # Householder reflection
    return H @ R


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", required=True)
    p.add_argument(
        "--camera-side",
        choices=["above", "below", "auto"],
        default="auto",
        help=(
            "which side of the marker plane the camera is actually on. "
            "'auto' picks the side that puts the camera farther from "
            "the marker centroid (useful when you know the camera is "
            "either very high or very low)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the flipped T_cam2base but don't write the JSON",
    )
    args = p.parse_args(argv)

    path = Path(args.npz)
    if not path.exists():
        print(f"[flip] {path} does not exist", file=sys.stderr)
        return 2
    res = load_calibration(path)
    X = np.asarray(res.T_cam2base, dtype=float)
    Y = np.asarray(res.T_marker2gripper, dtype=float)

    inlier_centroid, marker_normal = _average_marker_pose(res.samples_for_repro)
    cam_origin = X[:3, 3]
    cam_to_marker = inlier_centroid - cam_origin
    signed_distance = float(np.dot(marker_normal, cam_to_marker))

    # signed_distance > 0  -> camera is on the +normal side of marker plane
    # signed_distance < 0  -> camera is on the -normal side
    if args.camera_side == "above":
        keep_sign = +1
    elif args.camera_side == "below":
        keep_sign = -1
    else:  # auto: pick the side farther from the marker centroid.
        # If the camera is currently on the +normal side at distance d,
        # the other side would be at distance -d (reflection). The "far
        # side" is max(|d|, |-d|) = |d|, which is ambiguous -- so use
        # the magnitude interpretation: keep the side with larger |z| in
        # the world. For a typical eye-to-hand rig with a top camera,
        # the camera ends up at z>marker_z (positive z normal).
        keep_sign = +1

    if (signed_distance > 0) == (keep_sign > 0):
        side_now = "above (matches --camera-side)" if keep_sign > 0 else "below (matches --camera-side)"
        print(
            f"[flip] camera already on {side_now}; nothing to do. "
            "Re-run with --camera-side other if this is wrong."
        )
        return 0

    # Reflect across the marker plane.
    new_origin = _reflect_point_across_plane(cam_origin, inlier_centroid, marker_normal)
    new_R = _reflect_rotation_across_plane(X[:3, :3], marker_normal)

    new_X = np.eye(4)
    new_X[:3, :3] = new_R
    new_X[:3, 3] = new_origin

    print(f"[flip] saved   T_cam2base translation: {X[:3, 3].tolist()}")
    print(f"[flip] flipped T_cam2base translation: {new_X[:3, 3].tolist()}")
    print(f"[flip] marker centroid (base):         {inlier_centroid.tolist()}")
    print(f"[flip] marker plane normal (base):     {marker_normal.tolist()}")
    print(f"[flip] saved   cam_z in base:          {X[:3, 2].tolist()}")
    print(f"[flip] flipped cam_z in base:          {new_X[:3, 2].tolist()}")

    if args.dry_run:
        print("[flip] dry-run: not writing JSON")
        return 0

    # Mutate the saved JSON in-place. We need to update T_cam2base while
    # leaving Y (and everything else) intact. load_calibration returns
    # a CalibrationResult; we have to update the underlying JSON file
    # directly because the result class is frozen.
    raw = json.loads(path.read_text())
    raw["T_cam2base"] = new_X.tolist()
    path.write_text(json.dumps(raw, indent=2))
    print(f"[flip] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
