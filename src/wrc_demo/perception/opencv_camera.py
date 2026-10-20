"""RGB-only UVC webcam backend.

No depth: the frame carries depth_m=None and DepthProvider fills it in
(table-plane geometry or a mono-depth plugin). Intrinsics must be supplied in
the profile (or a rough default is synthesized from the FOV).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


class OpenCVCamera(CameraBase):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg = cfg
        self._cap = None
        self._K = None

    @property
    def has_depth(self) -> bool:
        return False

    def open(self) -> None:
        index = int(self._cfg.get("device_index", 0))
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise CameraError(f"cannot open /dev/video{index}")
        w = int(self._cfg.get("width", 1280))
        h = int(self._cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap = cap

        k = self._cfg.get("intrinsics")
        if k is not None:
            self._K = np.array(k, dtype=np.float64).reshape(3, 3)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _grab(self) -> Frame:
        if self._cap is None:
            raise CameraError("camera not opened")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise CameraError("frame read failed")
        if self._K is None:
            # Synthesize plausible intrinsics from horizontal FOV (default 70 deg).
            h, w = bgr.shape[:2]
            fov = math.radians(float(self._cfg.get("fov_deg", 70.0)))
            fx = w / (2 * math.tan(fov / 2))
            self._K = np.array(
                [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64
            )
        return Frame(rgb=bgr, depth_m=None, K=self._K, depth_source="none")
