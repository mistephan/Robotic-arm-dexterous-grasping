"""RealSense backend (D4xx and L515) via pyrealsense2.

On this rig pyrealsense2 comes from the local librealsense L515 fork
(l515-on-development, v2.58.x) installed into the .demo venv. Depth is
hardware-aligned to color and converted to float32 meters using the device
depth scale (L515: 0.25 mm/unit, D4xx: 1 mm/unit).
"""

from __future__ import annotations

import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


class RealSenseCamera(CameraBase):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg = cfg
        self._pipe = None
        self._align = None
        self._depth_scale = 0.001
        self._K = None

    @property
    def has_depth(self) -> bool:
        return True

    def open(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        pipe = rs.pipeline()
        config = rs.config()
        serial = self._cfg.get("serial")
        if serial:
            config.enable_device(str(serial))
        w = int(self._cfg.get("width", 1280))
        h = int(self._cfg.get("height", 720))
        fps = int(self._cfg.get("fps", 30))
        # Let librealsense pick native depth resolution; we align to color.
        config.enable_stream(rs.stream.depth, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            profile = pipe.start(config)
        except RuntimeError:
            # Fall back to fully automatic stream selection (any resolution).
            config = rs.config()
            if serial:
                config.enable_device(str(serial))
            config.enable_stream(rs.stream.depth, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, rs.format.bgr8, fps)
            profile = pipe.start(config)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._align = rs.align(rs.stream.color)
        self._pipe = pipe

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None

    def _grab(self) -> Frame:
        if self._pipe is None:
            raise CameraError("camera not opened")
        try:
            frames = self._pipe.wait_for_frames(5000)
            frames = self._align.process(frames)
            depth_f = frames.get_depth_frame()
            color_f = frames.get_color_frame()
            if not depth_f or not color_f:
                raise CameraError("incomplete frameset")
        except RuntimeError as e:
            raise CameraError(str(e)) from e

        color = np.asanyarray(color_f.get_data()).copy()
        depth = np.asanyarray(depth_f.get_data()).astype(np.float32)
        depth *= self._depth_scale  # -> meters, 0 stays invalid

        min_d = float(self._cfg.get("min_depth_m", 0.1))
        max_d = float(self._cfg.get("max_depth_m", 3.0))
        depth[(depth < min_d) | (depth > max_d)] = 0.0

        if self._K is None:
            intr = color_f.profile.as_video_stream_profile().get_intrinsics()
            self._K = np.array(
                [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                dtype=np.float64,
            )
        return Frame(rgb=color, depth_m=depth, K=self._K, depth_source="sensor")
