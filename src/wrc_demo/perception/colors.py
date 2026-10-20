"""Dependency-free color naming for detections and color-word queries.

Why this exists: the rig's detector may be closed-set (yolo11n COCO labels
until the ultralytics CLIP fork is installed), so "pink object" is not a
detectable class. Color is recovered *after* detection from the mask pixels
(median HSV -> a small named palette), attached to beliefs, and matched
against color words in the user's query. This keeps "pick and place pink
object" working with any detector backend.
"""

from __future__ import annotations

import numpy as np

# Canonical palette. Hue bands are OpenCV HSV (H in [0, 180)).
COLOR_NAMES = (
    "red", "orange", "yellow", "green", "cyan", "blue",
    "purple", "pink", "brown", "white", "gray", "black",
)

# Query aliases -> canonical color (English + Spanish, demo chat is bilingual).
_ALIASES = {
    "grey": "gray",
    "magenta": "pink",
    "violet": "purple",
    "rosa": "pink", "rosado": "pink", "rosada": "pink",
    "rojo": "red", "roja": "red",
    "naranja": "orange",
    "amarillo": "yellow", "amarilla": "yellow",
    "verde": "green",
    "azul": "blue", "celeste": "cyan",
    "morado": "purple", "morada": "purple", "violeta": "purple",
    "blanco": "white", "blanca": "white",
    "negro": "black", "negra": "black",
    "gris": "gray",
    "marron": "brown", "cafe": "brown",
}

#: nouns that mean "any object" rather than a detectable class
GENERIC_NOUNS = {
    "object", "objects", "thing", "things", "item", "items", "one", "it",
    "objeto", "objetos", "cosa", "cosas",
}

_ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "un", "una"}


def classify_hsv(h: float, s: float, v: float) -> str:
    """Name one OpenCV-HSV color (H 0-180, S/V 0-255)."""
    if v < 46:
        return "black"
    if s < 45:
        return "white" if v > 190 else "gray"
    # Low-saturation bright reds/magentas read as pink to humans.
    if s < 120 and v > 150 and (h <= 10 or h >= 140):
        return "pink"
    if h <= 9 or h >= 170:
        return "red"
    if h <= 22:
        return "brown" if v < 130 else "orange"
    if h <= 33:
        return "yellow"
    if h <= 85:
        return "green"
    if h <= 100:
        return "cyan"
    if h <= 128:
        return "blue"
    if h <= 147:
        return "purple"
    return "pink"  # magenta band 148..169


# Perceptual neighbors: real objects sit on band boundaries (a saturated
# red under warm light reads orange; hot pink reads red). Matching falls
# back to neighbors before giving up, so a boundary misclassification does
# not hard-veto the right object.
_NEIGHBORS = {
    "red": {"pink", "orange"},
    "pink": {"red", "purple"},
    "orange": {"red", "yellow", "brown"},
    "yellow": {"orange"},
    "brown": {"orange", "red"},
    "purple": {"pink", "blue"},
    "blue": {"purple", "cyan"},
    "cyan": {"blue", "green"},
    "green": {"cyan"},
    "white": {"gray"},
    "gray": {"white", "black"},
    "black": {"gray"},
}


def color_matches(want: str, got: str | None, strict: bool = False) -> bool:
    """Does an observed color satisfy a requested one? None = unknown =
    never a hard veto (the caller decides how to rank unknowns)."""
    if got is None:
        return True
    if got == want:
        return True
    return (not strict) and got in _NEIGHBORS.get(want, set())


def center_bbox_mask(shape_hw: tuple[int, int], bbox, frac: float = 0.5):
    """Mask over the central `frac` of a bbox: for detections without a
    segmentation mask, the box edges are mostly background -- naming the
    color from the whole box lets the background veto the object."""
    h, w = shape_hw
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * frac / 2, (y1 - y0) * frac / 2
    m = np.zeros((h, w), dtype=bool)
    m[max(int(cy - hh), 0):min(int(cy + hh) + 1, h),
      max(int(cx - hw), 0):min(int(cx + hw) + 1, w)] = True
    return m


def detection_color(bgr: np.ndarray, det) -> str | None:
    """Color of a Detection: its mask when present, else the bbox center."""
    mask = det.mask
    if mask is None:
        mask = center_bbox_mask(bgr.shape[:2], det.bbox)
    return mask_color(bgr, mask)


def mask_color(bgr: np.ndarray, mask: np.ndarray | None, max_px: int = 4000) -> str | None:
    """Median-HSV color name of the masked pixels (None when unusable)."""
    import cv2

    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    if ys.size > max_px:
        idx = np.random.default_rng(0).choice(ys.size, max_px, replace=False)
        ys, xs = ys[idx], xs[idx]
    px = bgr[ys, xs].reshape(-1, 1, 3)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    h, s, v = np.median(hsv, axis=0)
    # Hue is circular: the median of {2, 178} is meaningless. If hues span the
    # red wrap-around, re-center before taking the median.
    hs = hsv[:, 0]
    if hs.std() > 40 and ((hs < 20) | (hs > 160)).mean() > 0.6:
        h = float(np.median((hs + 90) % 180) - 90) % 180
    return classify_hsv(float(h), float(s), float(v))


def canonical_color(word: str) -> str | None:
    w = word.lower().strip()
    if w in COLOR_NAMES:
        return w
    return _ALIASES.get(w)


def parse_color_query(query: str) -> tuple[str | None, str | None]:
    """Split "pink object" -> ("pink", None); "red mug" -> ("red", "mug");
    "bottle" -> (None, "bottle"). A None noun means "any object of that
    color"; a None color means no color constraint."""
    tokens = [t for t in query.lower().replace(",", " ").split() if t]
    color = None
    rest: list[str] = []
    for t in tokens:
        c = canonical_color(t)
        if c is not None and color is None:
            color = c
        elif t not in _ARTICLES:
            rest.append(t)
    noun = " ".join(t for t in rest if t not in GENERIC_NOUNS).strip() or None
    return color, noun
