"""Mock camera: replays recorded frames or serves a synthetic tabletop scene.

Two modes:
- `dataset`: replay .npz frame files recorded with `wrc-record` (loops).
- `synthetic` (default): render a flat table at a configurable depth with a
  colored box on it -- enough for the full pipeline (detection via
  MockDetector, depth sampling, grasp planning) to run end-to-end in tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


def synthetic_tabletop(
    width: int = 640,
    height: int = 480,
    table_depth_m: float = 0.6,
    box_px: tuple[int, int, int, int] = (280, 200, 360, 260),
    box_height_m: float = 0.05,
) -> Frame:
    """A camera looking straight down at a table with one red box."""
    rgb = np.full((height, width, 3), 190, dtype=np.uint8)  # light gray table
    x0, y0, x1, y1 = box_px
    rgb[y0:y1, x0:x1] = (40, 40, 200)  # red box (BGR)
    depth = np.full((height, width), table_depth_m, dtype=np.float32)
    depth[y0:y1, x0:x1] = table_depth_m - box_height_m
    fx = 600.0
    K = np.array(
        [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64
    )
    return Frame(rgb=rgb, depth_m=depth, K=K, depth_source="sensor")


class MockCamera(CameraBase):
    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        self._cfg = cfg
        self._files: list[Path] = []
        self._idx = 0
        self._opened = False

    @property
    def has_depth(self) -> bool:
        return True

    def open(self) -> None:
        if self._cfg is not None:
            dataset = self._cfg.get("dataset")
            if dataset:
                self._files = sorted(Path(dataset).glob("frame_*.npz"))
                if not self._files:
                    raise CameraError(f"no frame_*.npz files in {dataset}")
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def _grab(self) -> Frame:
        if not self._opened:
            raise CameraError("camera not opened")
        if self._files:
            data = np.load(self._files[self._idx % len(self._files)])
            self._idx += 1
            depth = data["depth_m"].astype(np.float32) if "depth_m" in data else None
            return Frame(
                rgb=data["rgb"],
                depth_m=depth,
                K=data["K"].astype(np.float64),
                depth_source="sensor" if depth is not None else "none",
            )
        kw = {}
        if self._cfg is not None:
            kw = {
                "width": int(self._cfg.get("width", 640)),
                "height": int(self._cfg.get("height", 480)),
                "table_depth_m": float(self._cfg.get("table_depth_m", 0.6)),
            }
        return synthetic_tabletop(**kw)
