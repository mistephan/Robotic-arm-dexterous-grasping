"""Depth strategy chain: make every camera produce usable metric depth.

Order of preference per frame:
1. sensor depth (stereo/ToF cameras) - already on the Frame;
2. a pluggable mono-depth estimator (e.g. Depth Anything V2 metric), if
   configured and importable;
3. table-plane geometry: with a known table plane in the camera frame,
   depth at any pixel = ray/plane intersection. Only valid for points ON the
   table, which is exactly what tabletop grasping needs (mask pixels of an
   object sitting on the table).

The chain is explicit in `Frame.depth_source` so downstream code and the
agent can reason about depth quality.
"""

from __future__ import annotations

import numpy as np

from ..config import Cfg
from ..types import Frame


class DepthProvider:
    def __init__(self, cfg: Cfg):
        self._cfg = cfg
        self._mono = None
        mono_cfg = cfg.get("mono_depth")
        if mono_cfg and mono_cfg.get("enabled", False):
            self._mono = _load_mono_backend(mono_cfg)
        plane = cfg.get("table_plane_cam")
        # Plane in camera frame as (nx, ny, nz, d): n . p + d = 0, |n| = 1.
        self._plane = np.asarray(plane, dtype=float) if plane is not None else None

    def ensure_depth(self, frame: Frame) -> Frame:
        """Return a frame that has depth, filling it in if necessary."""
        if frame.has_depth and frame.depth_source == "sensor":
            return frame
        if self._mono is not None:
            frame.depth_m = self._mono(frame.rgb).astype(np.float32)
            frame.depth_source = "mono"
            return frame
        if self._plane is not None:
            frame.depth_m = self._plane_depth(frame)
            frame.depth_source = "plane"
            return frame
        return frame  # stays depth-less; grounding will refuse to localize

    def _plane_depth(self, frame: Frame) -> np.ndarray:
        h, w = frame.rgb.shape[:2]
        K = frame.K
        n = self._plane[:3]
        d = self._plane[3]
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        rays = np.stack(
            [(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, dtype=float)],
            axis=-1,
        )
        denom = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = -d / denom
        depth = np.where((denom < -1e-6) | (denom > 1e-6), tvals, 0.0)
        depth[depth < 0] = 0.0
        return depth.astype(np.float32)

    @staticmethod
    def fit_table_plane(frame: Frame, sample_step: int = 8) -> np.ndarray:
        """Least-squares plane fit over sensor depth -> (nx, ny, nz, d) cam frame.

        Call once during setup with an empty table in view; store the result
        in the camera profile as `table_plane_cam` for RGB-only sessions.
        """
        if not frame.has_depth:
            raise ValueError("plane fit needs a frame with sensor depth")
        h, w = frame.depth_m.shape
        K = frame.K
        us, vs = np.meshgrid(np.arange(0, w, sample_step), np.arange(0, h, sample_step))
        zs = frame.depth_m[vs, us]
        good = zs > 0
        us, vs, zs = us[good], vs[good], zs[good]
        xs = (us - K[0, 2]) / K[0, 0] * zs
        ys = (vs - K[1, 2]) / K[1, 1] * zs
        pts = np.stack([xs, ys, zs], axis=-1)
        centroid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
        n = vt[2]
        if n[2] > 0:  # normal should point toward the camera (-z side)
            n = -n
        d = -float(n @ centroid)
        return np.array([n[0], n[1], n[2], d])


def _load_mono_backend(cfg: Cfg):
    """Load an optional mono-depth callable rgb(BGR uint8) -> depth_m float32.

    Kept behind a string entry point so heavy deps stay optional:
        mono_depth: {enabled: true, entry_point: "my_pkg.mono:predict_depth"}
    """
    import importlib

    entry = cfg.get("entry_point")
    if not entry:
        raise ValueError("mono_depth.enabled requires mono_depth.entry_point")
    mod_name, _, attr = entry.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    return fn
