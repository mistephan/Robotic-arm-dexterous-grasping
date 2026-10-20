"""Continuous multi-camera streaming: the livestream backbone.

CameraStream wraps one CameraBase in a background pump thread that always
holds the newest frame. It is a latest-slot, not a queue: stale frames are
dropped so no consumer ever falls behind the sensor. CameraRig manages N
named streams; the first one ("primary") drives manipulation, the rest are
context/audience views -- any of them with extrinsics + depth also feeds the
shared belief store through the WorldWatcher.

Rationale (Anthropic robotics post + Agentic-VLA): the agent only ever needs
the newest frames, and perception must already be warm when a command
arrives, so streams start at process startup and never stop between tasks.

CameraStream keeps the FrameHub surface (get_frame fresh-after-call
semantics, set_overlay, warm_up) so it drops into SkillRuntime unchanged.
"""

from __future__ import annotations

import threading
import time
import warnings

from ..types import Frame


class CameraStream:
    """One camera + one pump thread + the latest frame."""

    def __init__(self, camera, name: str = "cam", rate_hz: float = 30.0):
        self.name = name
        self._camera = camera
        self._period = 1.0 / max(rate_hz, 1.0)
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._latest_seq = 0
        self._overlay_dets: list = []
        self._overlay_status = "starting..."
        self._stop = False
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._last_error: str | None = None

    # ── CameraBase-compatible surface ────────────────────────────────────

    @property
    def has_depth(self) -> bool:
        return self._camera.has_depth

    def open(self) -> None:
        self._camera.open()
        self._stop = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"stream-{self.name}"
        )
        self._thread.start()

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._camera.close()

    def warm_up(self, n: int = 10) -> None:
        deadline = time.monotonic() + 10.0
        with self._cond:
            while self._latest_seq < n and time.monotonic() < deadline:
                self._cond.wait(timeout=0.2)

    def get_frame(self, timeout_s: float = 3.0) -> Frame:
        """Return a frame captured AFTER this call (fresh observation)."""
        with self._cond:
            want = self._latest_seq + 1
            deadline = time.monotonic() + timeout_s
            while self._latest_seq < want:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._latest is not None:
                        return self._latest  # degrade: latest known frame
                    raise RuntimeError(f"camera stream {self.name!r} produced no frames")
                self._cond.wait(timeout=remaining)
            return self._latest

    # ── streaming consumers (viewers, watcher) ───────────────────────────

    def latest(self) -> Frame | None:
        """Newest frame without waiting (None before the first grab)."""
        with self._cond:
            return self._latest

    def wait_newer(self, seq: int, timeout_s: float = 1.0) -> tuple[Frame | None, int]:
        """Block until a frame newer than `seq` exists -> (frame, its seq)."""
        with self._cond:
            deadline = time.monotonic() + timeout_s
            while self._latest_seq <= seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, seq
                self._cond.wait(timeout=remaining)
            return self._latest, self._latest_seq

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ── annotations from the runtime/watcher ────────────────────────────

    def set_overlay(self, detections=None, status: str | None = None) -> None:
        with self._cond:
            if detections is not None:
                self._overlay_dets = list(detections)
            if status is not None:
                self._overlay_status = status

    def overlay(self) -> tuple[list, str]:
        with self._cond:
            return list(self._overlay_dets), self._overlay_status

    # ── pump loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        t_prev = time.monotonic()
        while not self._stop:
            t0 = time.monotonic()
            try:
                frame = self._camera.get_frame()
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                warnings.warn(
                    f"stream {self.name}: grab failed: {e}", RuntimeWarning
                )
                time.sleep(0.2)
                continue
            now = time.monotonic()
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            with self._cond:
                self._latest = frame
                self._latest_seq += 1
                self._cond.notify_all()
            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                time.sleep(self._period - elapsed)


class CameraRig:
    """N named CameraStreams. The first stream is the manipulation camera."""

    def __init__(self, streams: list[CameraStream]):
        if not streams:
            raise ValueError("CameraRig needs at least one camera stream")
        names = [s.name for s in streams]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate camera names in rig: {names}")
        self.streams: dict[str, CameraStream] = {s.name: s for s in streams}
        self._order = names

    @property
    def primary(self) -> CameraStream:
        return self.streams[self._order[0]]

    @property
    def names(self) -> list[str]:
        return list(self._order)

    def __iter__(self):
        return (self.streams[n] for n in self._order)

    def __len__(self) -> int:
        return len(self._order)

    def get(self, name: str | None) -> CameraStream:
        if name is None:
            return self.primary
        if name not in self.streams:
            raise KeyError(f"no camera {name!r}; available: {self._order}")
        return self.streams[name]

    def open(self) -> None:
        opened = []
        try:
            for s in self:
                s.open()
                opened.append(s)
        except Exception:
            for s in opened:
                try:
                    s.close()
                except Exception:
                    pass
            raise

    def close(self) -> None:
        for s in self:
            try:
                s.close()
            except Exception as e:
                warnings.warn(f"rig close {s.name}: {e}", RuntimeWarning)

    def warm_up(self, n: int = 5) -> None:
        for s in self:
            s.warm_up(n)

    def stats(self) -> dict:
        out = {}
        for s in self:
            f = s.latest()
            out[s.name] = {
                "fps": round(s.fps, 1),
                "frame_id": getattr(f, "frame_id", 0) if f is not None else 0,
                "depth": s.has_depth,
                "error": s.last_error,
            }
        return out
