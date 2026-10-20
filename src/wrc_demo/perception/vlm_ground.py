"""VLM open-vocabulary grounding: the SECOND perception filter.

YOLOE is the fast first filter (~30 ms/frame), but its text embeddings miss
things a full VLM reads easily: renders at small scale (the YCB banana at
~40 px), unusual booth objects, free-form phrases. When the detector misses,
ask a vision LLM for the bounding box directly. One call costs ~1-3 s on the
local Qwen server and only happens on the failure path, so the hot path
stays fast.
"""

from __future__ import annotations

import base64
import json
import re

import cv2
import numpy as np

from ..types import Detection

# Qwen-VL answers grounding in 0-1000-normalized coordinates of the image
# it SAW (the mmproj resizes internally, so absolute pixel coords of the
# original frame come back consistently shifted -- validated empirically:
# asking for pixels of an 848x480 frame put the banana box on a cracker box;
# the same numbers read as 0-1000 landed exactly on the banana). Always ask
# normalized, always rescale.
_PROMPT = (
    "You are a precise visual grounding system for a robot.\n"
    'Locate this object in the image: "{query}"\n'
    "Output ONLY a JSON object, no prose, no code fences, with the bounding\n"
    "box in coordinates normalized to 0-1000 of the full image:\n"
    '{{"found": true, "bbox_2d": [x0, y0, x1, y1]}}\n'
    'If the object is not visible: {{"found": false}}'
)


def parse_bbox(text: str, w: int, h: int) -> np.ndarray | None:
    """Extract a bbox from a VLM reply (0-1000 normalized -> pixels);
    tolerant of fences/prose and thinking blocks. Takes the LAST JSON
    object: reasoning traces may quote example JSON before the answer."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    if not matches:
        return None
    try:
        obj = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    box = obj.get("bbox_2d") or obj.get("bbox")
    if obj.get("found") is False or not isinstance(box, list):
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    x0, x1 = x0 * w / 1000.0, x1 * w / 1000.0
    y0, y1 = y0 * h / 1000.0, y1 * h / 1000.0
    x0, x1 = sorted((max(0.0, x0), min(float(w), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(h), y1)))
    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


class VLMGrounder:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 20.0,
    ):
        from openai import OpenAI  # lazy heavy import

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._model = model

    def ground(self, frame, query: str) -> Detection | None:
        """One grounding call -> Detection with a bbox (no mask), or None."""
        h, w = frame.rgb.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode()
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            max_tokens=768,  # Qwen thinking mode spends tokens before the JSON
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _PROMPT.format(query=query, w=w, h=h)},
                ],
            }],
        )
        text = resp.choices[0].message.content or ""
        bbox = parse_bbox(text, w, h)
        if bbox is None:
            return None
        return Detection(label=query, conf=0.5, bbox=bbox)
