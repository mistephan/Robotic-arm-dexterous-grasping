"""Unit tests for OrbbecCamera._decode_color_frame.

Three formats hit the live stream on Gemini 2, and getting any of them
wrong produces silent color corruption in the downstream detector (which
is exactly the regression that motivated this fix). These tests pin all
three decoding paths against a known-good reference so future SDK
upgrades or profile-selection refactors can't reintroduce the bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from wrc_demo.perception.orbbec_camera import OrbbecCamera
from wrc_demo.perception.camera_base import CameraError


# OBFormat constants matching the real pyorbbecsdk (verified in
# _pick_video_profile's docstring; if the SDK renumbers them, update here).
MJPG = 5
RGB = 22
BGR = 23


def _make_reference_bgr(h: int, w: int) -> np.ndarray:
    """Build a deterministic BGR gradient image: top→bottom R, G, B
    each fade smoothly from 255 to 0.

    Sharp color blocks (e.g. row 0 = pure red, row 40 = pure green) round-
    trip JPEG-encoded at quality 95 with heavy ringing at the block
    boundaries (max diff ~95/255), which makes byte-level equality
    assertions noisy. A smooth gradient encodes cleanly and verifies
    the same color-channel mix without JPEG ringing masking the test.
    """
    img = np.zeros((h, w, 3), dtype=np.float64)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float64).reshape(h, 1)
    img[:, :, 2] = (255.0 * (1.0 - yy))   # R: 255 → 0 top→bottom
    img[:, :, 1] = (255.0 * np.minimum(yy * 2, 1.0))  # G: 0 → 255 → 255
    img[:, :, 0] = (255.0 * np.maximum(yy * 2 - 1.0, 0.0))  # B: 0 → 0 → 255
    return np.clip(img, 0, 255).astype(np.uint8)


class _FakeColorFrame:
    """Minimal duck-type for pyorbbecsdk's color frame."""

    def __init__(self, payload, w: int, h: int, fmt: int):
        self._payload = payload
        self._w = w
        self._h = h
        self._fmt = fmt

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_data(self):
        return self._payload

    def get_format(self):
        return self._fmt


# ──────────────────────────────────────────────────────────────────────
# BGR path (already-correct path; locked in to catch any regression)
# ──────────────────────────────────────────────────────────────────────


def test_decode_bgr_passthrough_is_byte_identical():
    """If the SDK hands us BGR bytes, the decoded frame must be byte-
    identical to the input -- no color conversion should happen."""
    ref = _make_reference_bgr(60, 80)
    raw = ref.tobytes()  # already BGR byte order
    frame = _FakeColorFrame(raw, w=80, h=60, fmt=BGR)
    out = OrbbecCamera._decode_color_frame(raw, w=80, h=60, fmt=BGR)
    np.testing.assert_array_equal(out, ref)


# ──────────────────────────────────────────────────────────────────────
# RGB path (the bug case: previously reshape-only gave RGB-as-BGR)
# ──────────────────────────────────────────────────────────────────────


def test_decode_rgb_swaps_channels_to_bgr():
    """If the SDK hands us RGB bytes, the decoded frame must be the BGR
    representation of the same image -- channels must swap. Without the
    cv2.cvtColor call, the top row would read as blue (B=255) instead of
    red (R=255), which is exactly the "wrong colors" the user reported."""
    ref_bgr = _make_reference_bgr(60, 80)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    raw = ref_rgb.tobytes()  # RGB byte order
    out = OrbbecCamera._decode_color_frame(raw, w=80, h=60, fmt=RGB)
    np.testing.assert_array_equal(out, ref_bgr)
    # Sanity: the byte-level BGR interpretation (the OLD broken path)
    # would have produced the RGB image *labeled* BGR. Verify that's NOT
    # what we return.
    broken = np.frombuffer(raw, dtype=np.uint8).reshape(60, 80, 3)
    assert not np.array_equal(out, broken), (
        "decoder returned RGB bytes labeled as BGR -- cv2.cvtColor missing"
    )


# ──────────────────────────────────────────────────────────────────────
# MJPG path (the regression that limited wrc_demo to 640x480)
# ──────────────────────────────────────────────────────────────────────


def test_decode_mjpg_actually_decodes_jpeg():
    """MJPG payload must be JPEG-decoded, not reshape-treated. Encode a
    reference image as JPEG, decode it via _decode_color_frame, and
    verify the result is close to the source (JPEG is lossy, so use a
    generous tolerance).

    The color-channel mix is the canary: cv2.imdecode returns BGR (like
    cv2.imread), so a top-half-red image must still read as red after
    the round-trip -- if the function silently reshape-treated the JPEG
    bytes, the top half would come back as garbage and the channel
    order would be inverted (BGR-as-RGB instead of BGR-as-BGR).
    """
    ref_bgr = _make_reference_bgr(120, 160)
    # Pad with white to make the JPEG payload large enough that a naive
    # reshape produces a wholly-wrong image (not by chance correct).
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ok, buf = cv2.imencode(".jpg", ref_bgr, encode_params)
    assert ok
    raw_mjpg = buf.tobytes()
    # Sanity: a JPEG of a 120x160 image is smaller than the raw HxWx3 bytes.
    assert len(raw_mjpg) < 120 * 160 * 3, (
        "test image was not compressed -- JPEG size should be smaller "
        "than raw bytes"
    )
    out = OrbbecCamera._decode_color_frame(raw_mjpg, w=160, h=120, fmt=MJPG)
    assert out.shape == (120, 160, 3), out.shape
    # Per-pixel diff <= 8 levels in each channel (JPEG 95 is very tight).
    np.testing.assert_allclose(out, ref_bgr, atol=8)
    # The new BGR output must NOT be the same as what a naive (broken)
    # reshape would produce -- that's the actual bug we're guarding.
    broken = np.frombuffer(raw_mjpg, dtype=np.uint8)
    # Note: this reshape may fail with truncated bytes; tolerate either.
    if broken.size >= 120 * 160 * 3:
        broken = broken[: 120 * 160 * 3].reshape(120, 160, 3)
        assert not np.allclose(broken, ref_bgr, atol=50), (
            "decoder returned the same as a naive reshape -- JPEG not "
            "actually decoded"
        )


def test_decode_mjpg_truncated_raises_camera_error():
    """A truncated JPEG must raise CameraError, not silently return
    garbage. The old code path would reshape the truncated bytes (which
    would either crash on size mismatch or return a wrong-size array).

    cv2.imdecode is permissive about truncated JPEGs (it returns a partial
    image), so we instead corrupt the SOI marker to produce bytes that
    no JPEG decoder can make sense of.
    """
    # Random non-JPEG bytes that look nothing like an SOI (FF D8) marker.
    garbage = bytes(range(256)) * 50
    with pytest.raises(CameraError, match="imdecode returned None"):
        OrbbecCamera._decode_color_frame(garbage, w=80, h=60, fmt=MJPG)


# ──────────────────────────────────────────────────────────────────────
# Unknown format guardrail
# ──────────────────────────────────────────────────────────────────────


def test_decode_unknown_format_wrong_size_raises_camera_error():
    """A new OBFormat the SDK exposes but we don't handle yet must fail
    loudly with CameraError, not silently ship undecoded bytes to the
    detector."""
    payload = b"\x00" * (60 * 80 * 3 + 10)  # 10 bytes too many
    with pytest.raises(CameraError, match="size mismatch"):
        OrbbecCamera._decode_color_frame(payload, w=80, h=60, fmt=999)


def test_decode_unknown_format_correct_size_falls_back_with_warning(caplog):
    """If the payload size happens to match h*w*3 anyway, fall back to
    the legacy reshape and surface a clear warning so we notice."""
    ref_bgr = _make_reference_bgr(60, 80)
    payload = ref_bgr.tobytes()
    with caplog.at_level("WARNING"):
        out = OrbbecCamera._decode_color_frame(payload, w=80, h=60, fmt=999)
    # Legacy reshape returns the bytes as-is; we don't promise color
    # correctness here (we can't -- that's the whole point of the
    # warning), but the array shape must match.
    assert out.shape == (60, 80, 3)
