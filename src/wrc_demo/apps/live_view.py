"""FrameHub: continuous camera streaming + always-on visualization window.

Wraps any CameraBase. A background thread grabs frames at camera rate and
(optionally) renders them in an OpenCV window with detection overlays and a
status line, while the skill runtime keeps its normal `get_frame()` calls --
served fresh from the same stream, so the camera is never double-opened.

The runtime feeds annotations back via `set_overlay(...)` (duck-typed: it
only calls it when the camera object has the method), so the window shows
what the agent currently sees and does. All GUI calls stay inside the hub
thread; if the display is unavailable the hub silently degrades to a
headless frame pump.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from ..types import Frame


def depth_colormap(depth_m: np.ndarray, max_m: float = 2.0) -> np.ndarray:
    d = np.clip(depth_m, 0, max_m) / max_m
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth_m <= 0] = 0
    return img


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    y = 26
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (60, 255, 60), 1, cv2.LINE_AA)
        y += 26


def draw_detections(img: np.ndarray, dets) -> None:
    for d in dets:
        x0, y0, x1, y1 = d.bbox.astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(img, f"{d.label} {d.conf:.2f}", (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        if d.mask is not None and d.mask.shape == img.shape[:2]:
            overlay = img.copy()
            overlay[d.mask] = (0, 200, 255)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)


def draw_grasp_pose(
    img: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
    grasp_debug: dict, axis_length_m: float = 0.06,
) -> bool:
    """Project current grasp axes: red approach, green jaw, blue orthogonal."""
    p = grasp_debug.get("base_xyz")
    approach = grasp_debug.get("approach")
    jaw = grasp_debug.get("jaw_opening_axis")
    if p is None or approach is None or jaw is None:
        return False
    p = np.asarray(p, dtype=float).reshape(3)
    x_axis = np.asarray(approach, dtype=float).reshape(3)
    y_axis = np.asarray(jaw, dtype=float).reshape(3)
    for axis in (x_axis, y_axis):
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return False
        axis /= n
    z_axis = np.cross(x_axis, y_axis)
    zn = float(np.linalg.norm(z_axis))
    if zn < 1e-9:
        return False
    z_axis /= zn
    T_cam_base = np.linalg.inv(np.asarray(T_base_cam, dtype=float).reshape(4, 4))

    def project(points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        cam = pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
        if np.any(cam[:, 2] <= 1e-5):
            return None
        uvw = cam @ np.asarray(K, dtype=float).reshape(3, 3).T
        return np.rint(uvw[:, :2] / uvw[:, 2:3]).astype(int)

    uv = project(np.vstack([
        p, p + axis_length_m * x_axis, p + axis_length_m * y_axis,
        p + axis_length_m * z_axis,
    ]))
    if uv is None:
        return False
    origin = tuple(uv[0])
    for end, color, label in zip(
        uv[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), ("X", "Y", "Z")
    ):
        cv2.arrowedLine(img, origin, tuple(end), color, 3, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(img, label, tuple(end), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
    cv2.circle(img, origin, 5, (255, 0, 255), -1, cv2.LINE_AA)
    raw = grasp_debug.get("raw_origin_xyz")
    if raw is not None:
        raw_uv = project([raw])
        if raw_uv is not None:
            raw_pt = tuple(raw_uv[0])
            cv2.circle(img, raw_pt, 7, (255, 255, 255), 2, cv2.LINE_AA)
            if raw_pt != origin:
                cv2.line(img, raw_pt, origin, (255, 0, 255), 2, cv2.LINE_AA)
    offset_mm = float(grasp_debug.get("tip_offset_m") or 0.0) * 1000.0
    candidate_index = grasp_debug.get("candidate_index", "?")
    source = grasp_debug.get("source", "?")
    caption = f"#{candidate_index} {source} offset={offset_mm:.1f}mm"
    cv2.putText(img, caption, (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, caption, (origin[0] + 8, origin[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1, cv2.LINE_AA)
    return True


class RigViewer:
    """cv2 window over a CameraRig: all streams side by side, annotated.

    Render-only -- frame pumping lives in each CameraStream. Degrades to a
    silent no-op when no display is available, exactly like FrameHub."""

    def __init__(self, rig, title: str = "wrc-demo :: live", scale: float = 0.7,
                 rate_hz: float = 20.0, tile_h: int = 480):
        self._rig = rig
        self._title = title
        self._scale = scale
        self._period = 1.0 / rate_hz
        self._tile_h = tile_h
        self._stop = False
        self._thread: threading.Thread | None = None
        self._gui_ok = True

    def start(self) -> None:
        import os

        if not os.environ.get("DISPLAY") or os.environ.get("WRC_VIEW") == "0":
            self._gui_ok = False
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rig-viewer")
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _render_tile(self, stream) -> np.ndarray | None:
        frame = stream.latest()
        if frame is None:
            return None
        dets, status = stream.overlay()
        img = frame.rgb.copy()
        draw_detections(img, dets)
        h, w = img.shape[:2]
        draw_hud(img, [
            f"{stream.name}  {w}x{h}  {stream.fps:4.1f} fps  depth: {frame.depth_source}",
            f"agent: {status}",
        ])
        if h != self._tile_h:
            img = cv2.resize(img, (int(w * self._tile_h / h), self._tile_h))
        return img

    def _loop(self) -> None:
        window_up = False
        while not self._stop:
            t0 = time.monotonic()
            tiles = [t for s in self._rig if (t := self._render_tile(s)) is not None]
            if tiles and self._gui_ok:
                try:
                    panel = np.hstack(tiles) if len(tiles) > 1 else tiles[0]
                    if self._scale != 1.0:
                        panel = cv2.resize(panel, None, fx=self._scale, fy=self._scale)
                    if not window_up:
                        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
                        window_up = True
                    cv2.imshow(self._title, panel)
                    cv2.waitKey(1)
                except Exception as e:
                    self._gui_ok = False
                    print(f"[rig-viewer] viewer disabled: {e}", file=sys.stderr)
                    return
            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                time.sleep(self._period - elapsed)
        if window_up:
            try:
                cv2.destroyWindow(self._title)
                cv2.waitKey(1)
            except Exception:
                pass


class FrameHub:
    """Camera wrapper: background pump + optional live window.

    Superseded by perception.stream.CameraStream + RigViewer for the rig
    path; kept for single-camera embedding and back-compat."""

    def __init__(self, camera, title: str = "wrc-demo", show: bool = True,
                 scale: float = 0.7, rate_hz: float = 30.0):
        self._camera = camera
        self._title = title
        self._show = show
        self._scale = scale
        self._period = 1.0 / rate_hz
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._latest_seq = 0
        self._overlay_dets: list = []
        self._overlay_status = "starting..."
        self._stop = False
        self._thread: threading.Thread | None = None
        self._gui_ok = show

    # ── CameraBase-compatible surface ────────────────────────────────────

    @property
    def has_depth(self) -> bool:
        return self._camera.has_depth

    def open(self) -> None:
        self._camera.open()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="framehub")
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
                    raise RuntimeError("frame hub produced no frames")
                self._cond.wait(timeout=remaining)
            return self._latest

    # ── annotations from the runtime ─────────────────────────────────────

    def set_overlay(self, detections=None, status: str | None = None) -> None:
        with self._cond:
            if detections is not None:
                self._overlay_dets = list(detections)
            if status is not None:
                self._overlay_status = status

    # ── pump/render loop (owns ALL GUI calls) ────────────────────────────

    def _loop(self) -> None:
        fps, t_prev = 0.0, time.monotonic()
        window_up = False
        while not self._stop:
            t0 = time.monotonic()
            try:
                frame = self._camera.get_frame()
            except Exception as e:
                print(f"[framehub] frame grab failed: {e}", file=sys.stderr)
                time.sleep(0.2)
                continue
            with self._cond:
                self._latest = frame
                self._latest_seq += 1
                dets = list(self._overlay_dets)
                status = self._overlay_status
                self._cond.notify_all()

            if self._gui_ok:
                try:
                    now = time.monotonic()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                    t_prev = now
                    rgb = frame.rgb.copy()
                    draw_detections(rgb, dets)
                    h, w = rgb.shape[:2]
                    draw_hud(rgb, [
                        f"{w}x{h}  {fps:4.1f} fps  depth: {frame.depth_source}",
                        f"agent: {status}",
                    ])
                    panel = (
                        np.hstack([rgb, depth_colormap(frame.depth_m)])
                        if frame.has_depth else rgb
                    )
                    if self._scale != 1.0:
                        panel = cv2.resize(panel, None, fx=self._scale, fy=self._scale)
                    if not window_up:
                        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
                        window_up = True
                    cv2.imshow(self._title, panel)
                    cv2.waitKey(1)
                except Exception as e:
                    self._gui_ok = False  # headless / display gone: keep pumping
                    print(f"[framehub] viewer disabled: {e}", file=sys.stderr)

            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                time.sleep(self._period - elapsed)
        if window_up:
            try:
                cv2.destroyWindow(self._title)
                cv2.waitKey(1)
            except Exception:
                pass
