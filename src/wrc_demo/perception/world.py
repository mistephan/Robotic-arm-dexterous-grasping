"""WorldWatcher: the always-on perception loop behind the livestream.

A background thread round-robins over every rig camera that can lift pixels
to 3D (depth + extrinsics), runs the detector on the newest frame, names each
detection's color, and fuses everything into the shared BeliefStore. The
result: when a command like "pick and place pink object" arrives, the world
model is already warm -- resolution against beliefs is instantaneous instead
of waiting on an observe->detect round trip.

It also:
- pushes detection overlays to each stream (the MJPEG viewers show them);
- heartbeats the safety harness on EVERY processed frame (fresh perception
  = motion allowed) -- an empty table must not starve the watchdog, and the
  loop keeps ticking during arm motion precisely because multi-move skills
  outlast `watchdog_s` (adversarial review finding, 2026-07-18);
- while "paused" (arm in motion) it keeps detecting and heartbeating but
  SKIPS belief fusion, so the held object is not re-registered at a bogus
  mid-air position (ignore_label additionally covers it by detector label).
"""

from __future__ import annotations

import contextlib
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np

from ..types import Frame
from .colors import detection_color
from .grounding import Extrinsics, mask_to_points_cam, oriented_bbox
from ..types import transform_points


class LockedDetector:
    """Serialize detector access across threads (YOLO predict is not
    re-entrant); shared by the watcher and the skill runtime."""

    def __init__(self, detector):
        self._detector = detector
        self._lock = threading.Lock()

    def detect(self, frame, classes=None, imgsz=None):
        with self._lock:
            return self._detector.detect(frame, classes=classes, imgsz=imgsz)

    def set_classes(self, classes) -> None:
        with self._lock:
            self._detector.set_classes(classes)


@dataclass
class WatchedCamera:
    stream: object  # CameraStream
    depth: object  # DepthProvider
    extrinsics: Extrinsics
    last_seq: int = -1
    fuse: bool = True  # False: overlay/heartbeat only (no 3D fusion), e.g.
    # an audience camera with placeholder extrinsics
    last_error: str | None = None


class WorldWatcher:
    def __init__(
        self,
        cameras: list[WatchedCamera],
        detector,
        beliefs,
        classes: list[str],
        rate_hz: float = 3.0,
        harness=None,
    ):
        self._cams = cameras
        self._detector = detector
        self._beliefs = beliefs
        self._classes = list(classes)
        self._period = 1.0 / max(rate_hz, 0.1)
        self._harness = harness
        self._stop = False
        self._pause_count = 0
        self._pause_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ignore: set[str] = set()
        self.ticks = 0
        self.last_dets: dict[str, list] = {}
        self.last_update_t: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="world-watcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @contextlib.contextmanager
    def paused(self):
        """Hold BELIEF FUSION while the arm moves (re-entrant). Detection,
        overlays and the watchdog heartbeat keep running: motion skills
        depend on the heartbeat staying alive longer than watchdog_s."""
        with self._pause_lock:
            self._pause_count += 1
        try:
            yield
        finally:
            with self._pause_lock:
                self._pause_count -= 1

    @property
    def is_paused(self) -> bool:
        return self._pause_count > 0

    def ignore_label(self, label: str | None) -> None:
        """Skip fusing this label (e.g. the object currently in the jaws)."""
        self._ignore = {label} if label else set()

    # ── the loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop:
            t0 = time.monotonic()
            for cam in self._cams:
                if self._stop:
                    break
                try:
                    self._tick(cam)
                    cam.last_error = None
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if msg != cam.last_error:  # log state changes, not 3 Hz spam
                        warnings.warn(
                            f"watcher {cam.stream.name}: {msg}", RuntimeWarning
                        )
                    cam.last_error = msg
            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                time.sleep(self._period - elapsed)

    def _tick(self, cam: WatchedCamera) -> None:
        frame = cam.stream.latest()
        if frame is None:
            return
        # A stream serves the same frame until the camera produces a new one;
        # skip re-detection on frames we already processed.
        if frame.frame_id == cam.last_seq:
            return
        cam.last_seq = frame.frame_id
        frame = cam.depth.ensure_depth(frame)
        dets = self._detector.detect(frame, classes=self._classes)
        cam.stream.set_overlay(detections=dets)
        self.last_dets[cam.stream.name] = dets
        self.ticks += 1
        # Perception is alive: a processed frame is a heartbeat even with an
        # empty table or the arm mid-motion (motion skills outlast the
        # watchdog window and rely on this).
        if self._harness is not None:
            self._harness.heartbeat()
        if not frame.has_depth or not cam.fuse or self.is_paused:
            return
        # Eye-in-hand cameras carry their extrinsics IN the frame (the
        # camera rides the arm); static cameras use the profile matrix.
        T = (frame.T_base_cam if frame.T_base_cam is not None
             else cam.extrinsics.cam_to_base())
        fused = 0
        for d in dets:
            if d.label in self._ignore:
                continue
            mask = d.mask
            if mask is None:
                h, w = frame.rgb.shape[:2]
                mask = np.zeros((h, w), dtype=bool)
                x0, y0, x1, y1 = d.bbox.astype(int)
                mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < 10:
                continue
            pts_base = transform_points(T, pts_cam)
            center, extents, _ = oriented_bbox(pts_base)
            if float(extents[0]) > 0.35:
                continue  # tables/walls detected as "box": not a manipuland
            self._beliefs.update(
                d.label, center, d.conf, extent=extents,
                top_z=float(pts_base[:, 2].max()), t=frame.t,
                color=detection_color(frame.rgb, d),
                points=pts_base if d.mask is not None else None,
            )
            fused += 1
        if fused:
            self.last_update_t = time.monotonic()

    def stats(self) -> dict:
        return {
            "ticks": self.ticks,
            "paused": self.is_paused,
            "cameras": [c.stream.name for c in self._cams],
            "last_update_s_ago": (
                round(time.monotonic() - self.last_update_t, 1)
                if self.last_update_t is not None else None
            ),
        }
