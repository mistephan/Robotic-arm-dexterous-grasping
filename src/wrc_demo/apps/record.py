"""Record frames from a live camera into a mock-camera dataset.

    wrc-record --camera l515 --out datasets/table_scene --frames 30 --hz 2

The .npz files replay through MockCamera (`dataset:` key in a mock camera
profile), so perception changes can be tested against real captured scenes
without the camera attached.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..config import load_profile
from ..perception.camera_base import make_camera


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="record camera frames for replay")
    p.add_argument("--camera", default="l515")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--hz", type=float, default=2.0)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam_cfg = load_profile("cameras", args.camera)
    cam = make_camera(cam_cfg)
    cam.open()
    cam.warm_up(10)
    try:
        for i in range(args.frames):
            f = cam.get_frame()
            payload = {"rgb": f.rgb, "K": f.K}
            if f.depth_m is not None:
                payload["depth_m"] = f.depth_m
            np.savez_compressed(out / f"frame_{i:04d}.npz", **payload)
            print(f"saved frame_{i:04d}.npz (depth: {f.depth_source})")
            time.sleep(1.0 / args.hz)
    finally:
        cam.close()
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
