"""Canonical object names shared by language and motion runtime paths."""

from __future__ import annotations

import re

ZH_OBJECT_ALIASES = {
    "香蕉": "banana",
    "芒果": "mango",
    "苹果": "red apple",
    "红苹果": "red apple",
    "青柠": "lime",
    "青柠檬": "lime",
    "藤篮": "wicker basket",
    "藤篮子": "wicker basket",
    "篮子": "wicker basket",
    "筐": "wicker basket",
    "框": "wicker basket",
    "果筐": "wicker basket",
    "水果筐": "wicker basket",
    "水果框": "wicker basket",
}


def canonical_object_name(text: str | None) -> str | None:
    """Map a spoken object phrase to the detector vocabulary."""
    if text is None:
        return None
    name = re.sub(r"^(?:这个|那个|一个|把|将)", "", text.strip())
    name = re.sub(r"(?:里面|里边|里|中)$", "", name).strip()
    return ZH_OBJECT_ALIASES.get(name, name or None)


def canonical_destination(text: str | None) -> str | None:
    """Canonicalize a named or Chinese object-relative destination."""
    if text is None:
        return None
    value = text.strip()
    match = re.fullmatch(r"(.+?)(?:的)?(左边|左侧|右边|右侧)", value)
    if match:
        reference = canonical_object_name(match.group(1))
        side = "left" if match.group(2).startswith("左") else "right"
        return f"{side} of {reference}" if reference else None
    return canonical_object_name(value)
