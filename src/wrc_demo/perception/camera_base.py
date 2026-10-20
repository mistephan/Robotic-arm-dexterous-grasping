"""Camera abstraction: every backend yields metric-depth Frames.

A backend only has to produce (bgr, depth_m aligned to color, K). Cameras
without depth return depth_m=None and the DepthProvider chain fills it in
(mono-depth plugin or table-plane geometry), keeping the rest of the stack
depth-source agnostic.
"""

from __future__ import annotations

import abc
import time

from ..config import Cfg
from ..types import Frame


class CameraError(RuntimeError):
    pass


class CameraBase(abc.ABC):
    """Lifecycle + frame source. Frames use float32 meters for depth."""

    #: consecutive failures before get_frame raises
    MAX_CONSECUTIVE_FAILURES = 30

    def __init__(self) -> None:
        self._failures = 0
        self._frame_id = 0

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def _grab(self) -> Frame:
        """Fetch one frame or raise CameraError."""

    @property
    @abc.abstractmethod
    def has_depth(self) -> bool: ...

    def get_frame(self) -> Frame:
        try:
            frame = self._grab()
        except CameraError:
            self._failures += 1
            if self._failures >= self.MAX_CONSECUTIVE_FAILURES:
                raise
            time.sleep(0.02)
            return self.get_frame()
        self._failures = 0
        self._frame_id += 1
        frame.frame_id = self._frame_id
        return frame

    def warm_up(self, n: int = 10) -> None:
        for _ in range(n):
            self.get_frame()

    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_camera(cfg: Cfg) -> CameraBase:
    """Instantiate a camera backend from a camera profile config."""
    kind = cfg.type
    if kind == "realsense":
        from .realsense_camera import RealSenseCamera

        return RealSenseCamera(cfg)
    if kind == "uvc":
        from .opencv_camera import OpenCVCamera

        return OpenCVCamera(cfg)
    if kind == "mock":
        from .mock_camera import MockCamera

        return MockCamera(cfg)
    if kind == "orbbec":
        from .orbbec_camera import OrbbecCamera

        return OrbbecCamera(cfg)
    raise ValueError(
        f"unknown camera type {kind!r}; supported: realsense | uvc | mock | orbbec"
    )
