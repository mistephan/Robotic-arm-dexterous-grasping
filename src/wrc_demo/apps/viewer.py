"""Live perception viewer: what the demo's camera stack actually sees.

    wrc-view --camera l515                       # RGB + depth colormap
    wrc-view --camera l515 --detect yolo11n.pt   # + detection overlays

Left pane: RGB with detection boxes/labels (and center-pixel depth readout).
Right pane: depth colormap (JET, near=red) with invalid pixels black.
Keys: q/ESC quit, s save a PNG snapshot next to the cwd, d toggle detection.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from ..config import load_profile
from ..perception.camera_base import make_camera
from .live_view import depth_colormap, draw_detections, draw_hud


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="live camera/perception viewer")
    p.add_argument("--camera", default="l515")
    p.add_argument("--detect", default=None, help="YOLO weights path to overlay detections")
    p.add_argument("--classes", default=None,
                   help="comma-separated open-vocab classes (needs YOLOE/world model)")
    p.add_argument("--scale", type=float, default=0.75, help="display scale")
    args = p.parse_args(argv)

    detector = None
    if args.detect:
        from ..perception.detector import OpenVocabDetector

        classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
        detector = OpenVocabDetector(args.detect, conf=0.35)
        if classes:
            detector.set_classes(classes)

    cam = make_camera(load_profile("cameras", args.camera))
    cam.open()
    cam.warm_up(5)
    win = f"wrc-demo :: {args.camera}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    detect_on = detector is not None
    fps, t_prev = 0.0, time.monotonic()
    print(f"[wrc-view] streaming {args.camera}; q/ESC to quit", file=sys.stderr)
    try:
        while True:
            frame = cam.get_frame()
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            rgb = frame.rgb.copy()
            dets = []
            if detect_on and detector is not None:
                try:
                    dets = detector.detect(frame)
                except Exception as e:
                    detect_on = False
                    print(f"[wrc-view] detection disabled: {e}", file=sys.stderr)
            draw_detections(rgb, dets)

            h, w = rgb.shape[:2]
            cx, cy = w // 2, h // 2
            center_txt = "no depth"
            if frame.has_depth:
                d = float(frame.depth_m[cy, cx])
                center_txt = f"center {d:.2f} m" if d > 0 else "center invalid"
            cv2.drawMarker(rgb, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            draw_hud(rgb, [
                f"{args.camera}  {w}x{h}  {fps:4.1f} fps",
                f"depth: {frame.depth_source}  |  {center_txt}",
                f"objects: {len(dets)}" + ("" if detect_on else "  (detection off)"),
            ])

            if frame.has_depth:
                panel = np.hstack([rgb, depth_colormap(frame.depth_m)])
            else:
                panel = rgb
            if args.scale != 1.0:
                panel = cv2.resize(panel, None, fx=args.scale, fy=args.scale)
            cv2.imshow(win, panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                name = time.strftime("wrc_view_%H%M%S.png")
                cv2.imwrite(name, panel)
                print(f"[wrc-view] saved {name}", file=sys.stderr)
            if key == ord("d"):
                detect_on = not detect_on and detector is not None
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
