"""Orbbec Gemini 2 backend (single-camera pyorbbecsdk wrapper).

Schema version: 1
Frame convention: wrc_demo (base frame +x away, +y left, gripper_end TCP)

A thin layer over ``pyorbbecsdk``: configures a pipeline for the
requested serial, requests hardware-aligned color + depth streams, and
yields ``Frame(rgb=BGR, depth_m=float32 meters, K=3x3)`` like every
other backend. No FK, no detection, no ArUco.

The SDK is heavy and only available on the demo rig's venv -- the
import is local to ``open()`` so the mock / CI stack stays clean.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ..config import Cfg
from ..types import Frame
from .camera_base import CameraBase, CameraError


def list_orbbec_devices() -> list[dict[str, Any]]:
    """Enumerate Orbbec devices without opening any pipeline.

    Returns a list of dicts with at least ``serial`` and ``name`` keys.
    Raises ``CameraError`` if ``pyorbbecsdk`` isn't installed.
    """
    try:
        import pyorbbecsdk as ob  # type: ignore
    except ImportError as e:
        raise CameraError(
            "pyorbbecsdk not installed; install with "
            "`uv pip install pyorbbecsdk` into the demo venv"
        ) from e

    ctx = ob.Context()
    devices = ctx.query_devices()
    out: list[dict[str, Any]] = []
    if devices is None:
        return out
    n = devices.get_count() if hasattr(devices, "get_count") else len(devices)
    for i in range(n):
        try:
            dev = devices.get_device_by_index(i)
        except Exception:
            try:
                dev = devices[i]
            except Exception:
                continue
        info: dict[str, Any] = {"index": i}
        try:
            di = dev.get_device_info()
            info["name"] = di.get_name()
            info["serial"] = di.get_serial_number()
            info["pid"] = di.get_pid()
            info["vid"] = di.get_vid()
        except Exception:
            pass
        out.append(info)
    return out


class OrbbecCamera(CameraBase):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self._cfg = cfg
        self._serial = (cfg.get("serial") or "") if cfg is not None else ""
        if self._serial == "":
            self._serial = None
        else:
            self._serial = str(self._serial)
        self._pipeline: Any = None
        self._align: Any = None
        self._ob: Any = None
        self._K: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._depth_scale: float = 0.001

    @property
    def has_depth(self) -> bool:
        return True

    @property
    def serial(self) -> str | None:
        return self._serial

    def open(self) -> None:
        try:
            import pyorbbecsdk as ob  # type: ignore
        except ImportError as e:
            raise CameraError(
                "pyorbbecsdk not installed; install pyorbbecsdk into the demo "
                "venv (`uv pip install pyorbbecsdk`) or use a different "
                "camera profile"
            ) from e
        self._ob = ob

        w = int(self._cfg.get("width", 1280))
        h = int(self._cfg.get("height", 720))
        fps = int(self._cfg.get("fps", 30))

        # Phase 0: bind the pipeline to the device matching ``self._serial``.
        #
        # pyorbbecsdk v2.x has NO ``Pipeline.set_device_serial_number``; the
        # official SDK example ``examples/beginner/01_hello_camera.py:96``
        # binds to a specific device by enumerating ``Context().query_devices()``
        # then constructing ``Pipeline(device)``. The previous code called
        # the non-existent ``set_device_serial_number`` inside
        # ``try/except Exception: pass``, which silently swallowed the
        # ``AttributeError`` and left ``Pipeline()`` bound to whichever
        # device came up first in USB enumeration. With two Orbbec Gemini
        # 2s on the bench and the binding in ``cameras.json`` pointing to
        # the wrong USB index, this manifested as "I asked for the top
        # camera but the wrist view is showing".
        device: Any = None
        if self._serial:
            try:
                ctx = ob.Context()
                devs = ctx.query_devices()
                n = devs.get_count() if hasattr(devs, "get_count") else len(devs)
                for i in range(n):
                    try:
                        d = devs.get_device_by_index(i)
                        if d.get_device_info().get_serial_number() == self._serial:
                            device = d
                            break
                    except Exception:
                        continue
            except Exception as e:
                raise CameraError(
                    f"Orbbec device enumeration failed while looking for "
                    f"serial={self._serial!r}: {e!r}"
                ) from e
            if device is None:
                raise CameraError(
                    f"Orbbec device with serial={self._serial!r} not found; "
                    "run `python -m wrc_demo.scripts.calib_top_orbbec --list` "
                    "to see available serials, or check the binding in "
                    "data/calibration/cameras.json"
                )

        pipeline = ob.Pipeline(device) if device is not None else ob.Pipeline()

        # Phase 0b: verify the actually-opened device serial matches. Some
        # SDK paths ignore the ``Pipeline(device)`` arg and fall back to the
        # first enumerated device regardless -- we don't want that to be
        # silent. A mismatch here means cameras.json points at the wrong
        # device, the USB topology changed, or there's a stale binding.
        if self._serial:
            try:
                opened = (
                    pipeline.get_device()
                    if hasattr(pipeline, "get_device")
                    else None
                )
                if opened is not None and hasattr(opened, "get_device_info"):
                    opened_serial = opened.get_device_info().get_serial_number()
                    if opened_serial != self._serial:
                        raise CameraError(
                            f"Orbbec serial mismatch: requested {self._serial!r} "
                            f"but pipeline opened {opened_serial!r}; the SDK "
                            "ignored the device selection. Re-run "
                            "`python -m wrc_demo.scripts.calib_top_orbbec "
                            "--bind top --serial <correct-serial>` to fix the "
                            "binding in data/calibration/cameras.json"
                        )
            except CameraError:
                raise
            except Exception:
                pass

        cfg = ob.Config()
        # Attach the requested w/h to the cfg so SDKs (and our test
        # fakes) that look it up before start() return matching profiles.
        try:
            cfg._w, cfg._h, cfg._fps = w, h, fps
        except Exception:
            pass

        # Select an RGB color stream first.  For Gemini 330-series cameras,
        # hardware D2C should use a depth profile explicitly returned as
        # compatible with the selected color profile.
        color_profiles = pipeline.get_stream_profile_list(
            ob.OBSensorType.COLOR_SENSOR
        )

        color_profile = None
        depth_profile = None

        # Prefer RGB @ requested FPS.  Among those, prefer requested resolution;
        # otherwise fall back to another RGB profile supported by HW D2C.
        fallback = None

        for i in range(len(color_profiles)):
            try:
                p = color_profiles[i]

                if not p.is_video_stream_profile():
                    continue

                if p.get_format() != ob.OBFormat.RGB:
                    continue

                if p.get_fps() != fps:
                    continue

                d2c_profiles = pipeline.get_d2c_depth_profile_list(
                    p,
                    ob.OBAlignMode.HW_MODE,
                )

                if len(d2c_profiles) == 0:
                    continue

                # Exact requested color resolution wins.
                if p.get_width() == w and p.get_height() == h:
                    color_profile = p
                    depth_profile = d2c_profiles[0]
                    break

                # Keep the first valid RGB + HW-D2C combination as fallback.
                if fallback is None:
                    fallback = (p, d2c_profiles[0])

            except Exception:
                continue

        if color_profile is None:
            if fallback is None:
                raise CameraError(
                    f"No RGB + HW-D2C compatible stream found at {fps} FPS"
                )

            color_profile, depth_profile = fallback

        cfg.enable_stream(depth_profile)
        cfg.enable_stream(color_profile)

        # Gemini 330-series hardware depth-to-color alignment.
        cfg.set_align_mode(ob.OBAlignMode.HW_MODE)

        try:
            pipeline.enable_frame_sync()
        except Exception:
            pass

        try:
            print(
                "[Orbbec] HW D2C: "
                f"color={color_profile.get_width()}x"
                f"{color_profile.get_height()}@"
                f"{color_profile.get_fps()} "
                f"{color_profile.get_format()}, "
                f"depth={depth_profile.get_width()}x"
                f"{depth_profile.get_height()}@"
                f"{depth_profile.get_fps()} "
                f"{depth_profile.get_format()}"
            )
        except Exception:
            pass

        try:
            pipeline.start(cfg)
            print(f"[Orbbec] depth_scale={self._depth_scale}")
        except Exception as e:
            raise CameraError(f"Orbbec pipeline.start failed: {e}") from e

        # Phase B: read depth scale from the active device.
        self._depth_scale = 0.001
        try:
            device = pipeline.get_device() if hasattr(pipeline, "get_device") else None
            if device is not None and hasattr(device, "get_depth_sensor"):
                depth_sensor = device.get_depth_sensor()
                if depth_sensor is not None and hasattr(
                    depth_sensor, "get_depth_scale"
                ):
                    self._depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            pass

        # Phase C: read color intrinsics from the live stream profile.
        # Re-query after start() because the active profile is what gets
        # intrinsic calc; the candidate picked above may have been
        # downgraded by set_align_mode.
        # Phase C: read color intrinsics from the selected HW-D2C color profile.
        try:
            intr = color_profile.get_intrinsic()
            self._K = np.array(
                [[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
                dtype=np.float64,
            )
            if hasattr(intr, "dcoeff") and intr.dcoeff:
                self._D = np.asarray(intr.dcoeff, dtype=np.float64).reshape(-1)
        except Exception:
            self._K = None

        if self._K is None:
            # Last-ditch: synthetic intrinsics from the requested w/h + 70deg FOV.
            fx = float(w) / (2.0 * np.tan(np.deg2rad(70.0) / 2.0))
            self._K = np.array(
                [[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], dtype=np.float64
            )

        # Depth-aligned-to-color frameset.
        self._align = None
        self._pipeline = pipeline

    @staticmethod
    def _pick_video_profile(
        pipeline,
        sensor_type,
        want_w: int,
        want_h: int,
        want_fps: int,
        prefer=None,
    ):
        """Find the best VideoStreamProfile from
        ``pipeline.get_stream_profile_list`` matching (want_w, want_h, want_fps).

        Falls back through three buckets (exact w/h/fps → same w/h →
        any) and within each bucket **prefers uncompressed RGB/BGR** over
        MJPG/YUYV so ``color.get_data()`` returns ``(h, w, 3)`` bytes
        (no in-band JPEG decode needed). Returns ``None`` if the sensor
        is absent.
        """
        try:
            pl = pipeline.get_stream_profile_list(sensor_type)
        except Exception:
            return None
        n = pl.get_count() if hasattr(pl, "get_count") else len(pl)
        if n == 0:
            return None

        # OBFormat values seen on pyorbbecsdk >= 2.x (Gemini 2):
        # RGB=22, BGR=23, MJPG=5, H264=6, H265=7, NV12=3, NV21=4,
        # YUYV=0, YUY2=1, COMPRESSED=26, Y16=8, ...
        #
        # MJPG (5) is the ONLY format the Gemini 2 exposes at 1280x720 -- the
        # uncompressed RGB/BGR streams top out around 640x480 over USB2. The
        # previous code BLOCKED MJPG because ``np.frombuffer(...).reshape(h,w,3)``
        # on raw JPEG bytes produces garbage; the proper fix is to decode MJPG
        # with ``cv2.imdecode`` (mirrors rebot_grasp/drivers/camera/orbbec_gemini2.py).
        # Without MJPG support the pipeline silently drops to 640x480 RGB --
        # which is what makes wrc_demo's view look "low-res + wrong colors" vs.
        # rebot_grasp's 1280x720 view.
        _PREFERRED_FORMATS = {5, 22, 23}  # MJPG, RGB, BGR (MJPG first)
        _BLOCKED_FORMATS = {0, 1, 2, 3, 4, 6, 7, 26, 27}  # YUYV, YUY2, UYVY, NV12, NV21, H264, H265, COMPRESSED, RVL

        exact: list = []
        same_res: list = []
        any_res: list = []
        for i in range(n):
            try:
                p = pl.get_stream_profile_by_index(i)
            except Exception:
                continue
            if not p.is_video_stream_profile():
                continue
            fmt = p.get_format()
            # Drop compressed/encoded formats outright -- they break the
            # BGR reshape path. Falling back to MJPG here is the bug
            # that left ArUco detection returning None on every pose.
            if fmt in _BLOCKED_FORMATS:
                continue
            pw, ph, pf = p.get_width(), p.get_height(), p.get_fps()
            bucket = (
                exact if (pw, ph, pf) == (want_w, want_h, want_fps)
                else same_res if pw == want_w and ph == want_h
                else any_res
            )
            bucket.append((fmt, p))

        for bucket in (exact, same_res, any_res):
            if not bucket:
                continue
            # Sort: preferred formats first (RGB/BGR), preserve original
            # order otherwise.
            bucket.sort(key=lambda x: (0 if x[0] in _PREFERRED_FORMATS else 1,))
            return bucket[0][1]
        return None

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # ── color format decoding ─────────────────────────────────────────────
    #
    # The Orbbec SDK reports the color stream's pixel format via
    # ``color.get_format()``. Three are observed on Gemini 2:
    #
    #   * OBFormat.MJPG (5)  -- JPEG-compressed bytes; MUST be decoded with
    #                            ``cv2.imdecode`` before reshape, else the raw
    #                            bytes look like garbage. This is the ONLY
    #                            format that exposes 1280x720 over USB2.
    #   * OBFormat.RGB  (22) -- RGB byte order; needs cv2.cvtColor to BGR for
    #                            downstream OpenCV consumers.
    #   * OBFormat.BGR  (23) -- BGR byte order; reshape directly.
    #
    # Anything else (YUYV/NV12/H264/...) is rejected by _pick_video_profile
    # so the active stream is always one of the three above. The fallback
    # here only fires if the SDK reports an unexpected format mid-session.
    #
    # Mirrors rebot_grasp/drivers/camera/orbbec_gemini2.py::get_frame --
    # the two implementations MUST stay in sync, otherwise the wrc_demo
    # view and the rebot_grasp view diverge in color / format.
    _MJPG_FORMAT = 5
    _RGB_FORMAT = 22
    _BGR_FORMAT = 23

    @classmethod
    def _decode_color_frame(cls, payload, w: int, h: int, fmt):
        """Decode a color frame's raw bytes into a contiguous HxWx3 BGR
        uint8 array, regardless of the underlying pixel format.

        * fmt == MJPG  -> ``cv2.imdecode`` (the SDK hands us JPEG bytes;
                          reshape without decoding produces a smeared mess).
        * fmt == RGB   -> reshape + ``cv2.cvtColor(COLOR_RGB2BGR)``.
        * fmt == BGR   -> reshape only.
        * fmt unknown  -> legacy reshape with a warning; ``_pick_video_profile``
                          should have filtered anything outside the 3 handled
                          formats, so reaching this branch indicates a new
                          OBFormat constant the SDK exposes -- add it to the
                          table above.
        """
        if fmt == cls._MJPG_FORMAT:
            raw = np.frombuffer(payload, dtype=np.uint8)
            decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if decoded is None:
                raise CameraError(
                    "cv2.imdecode returned None on MJPG payload -- the "
                    "Orbbec SDK may have handed us a truncated JPEG"
                )
            if decoded.shape[0] != h or decoded.shape[1] != w:
                raise CameraError(
                    f"MJPG decoded shape {decoded.shape[:2][::-1]} != "
                    f"requested {w}x{h}"
                )
            return decoded  # cv2.imdecode returns BGR
        if fmt == cls._RGB_FORMAT:
            raw = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        if fmt == cls._BGR_FORMAT:
            return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3).copy()
        # Unknown format -- try the legacy reshape and surface a clear
        # error if the payload size doesn't match (so we don't ship
        # silently-wrong colors to the detector).
        expected = h * w * 3
        if len(payload) != expected:
            raise CameraError(
                f"Orbbec color frame size mismatch on format={fmt}: got "
                f"{len(payload)} bytes, expected {expected} (h*w*3 = {h}*{w}*3). "
                "Add this format to _decode_color_frame."
            )
        return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3).copy()

    def _grab(self) -> Frame:
        if self._pipeline is None:
            raise CameraError("Orbbec pipeline not started")
        try:
            frames = self._pipeline.wait_for_frames(500)
            if self._align is not None:
                try:
                    frames = self._align.process(frames)
                except Exception:
                    pass
            color = frames.get_color_frame() if frames is not None else None
            depth = frames.get_depth_frame() if frames is not None else None
            if color is None or depth is None:
                raise CameraError("incomplete frameset from Orbbec")
        except CameraError:
            raise
        except Exception as e:  # noqa: BLE001 - SDK raises generic
            raise CameraError(str(e)) from e

        try:
            w = color.get_width()
            h = color.get_height()
            payload = color.get_data()
            fmt = color.get_format() if hasattr(color, "get_format") else None
            bgr = self._decode_color_frame(payload, w, h, fmt)
        except Exception as e:
            raise CameraError(f"failed to decode Orbbec color frame: {e}") from e
        try:
            dw = depth.get_width()
            dh = depth.get_height()
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(dh, dw)
            depth_m = raw.astype(np.float32) * self._depth_scale
            # valid = depth_m[depth_m > 0]
            # if valid.size:
            #     print(
            #         f"[Orbbec] depth raw range="
            #         f"{raw[raw > 0].min()}..{raw.max()}, "
            #         f"depth_m median={np.median(valid):.3f}"
            #     )
        except Exception as e:
            raise CameraError(f"failed to decode Orbbec depth frame: {e}") from e

        min_d = float(self._cfg.get("min_depth_m", 0.25))
        max_d = float(self._cfg.get("max_depth_m", 2.5))
        depth_m[(depth_m < min_d) | (depth_m > max_d)] = 0.0

        return Frame(
            rgb=bgr,
            depth_m=depth_m,
            K=self._K if self._K is not None else np.eye(3),
            depth_source="sensor",
        )

    @property
    def K(self) -> np.ndarray | None:
        return self._K

    @property
    def D(self) -> np.ndarray | None:
        return self._D
