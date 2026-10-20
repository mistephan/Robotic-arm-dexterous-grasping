"""Object-permanence belief store.

Tracks the last known base-frame position of every object the system has
seen, so the agent can act on things that scrolled out of view or got
occluded ("the mug is where it was 8 seconds ago"). Beliefs decay to
`remembered` state after not being re-observed; they are only dropped after
`forget_after_s` (default: never during a demo run).

Beliefs also carry a named color (median mask HSV, see perception.colors) so
color-word queries like "pink object" resolve against the live world model
even when the detector vocabulary has no such class. All methods are
thread-safe: the WorldWatcher fuses observations from N camera streams while
the skill runtime reads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..perception.colors import parse_color_query


@dataclass
class ObjectBelief:
    label: str
    position: np.ndarray  # (3,) base frame
    extent: np.ndarray | None = None  # (3,) OBB extents (desc-sorted, NOT axis-aligned)
    top_z: float | None = None  # highest observed point (base frame) - use this
    # for stacking/placing, never extent[2] (extents are eigenvalue-ordered).
    conf: float = 0.5
    color: str | None = None  # named color (perception.colors palette)
    points: np.ndarray | None = None  # (<=384,3) last REAL-mask cloud, base
    # frame -- lets grasp-from-memory use the object's true shape instead of
    # a box approximation (a bbox-inflated box slab reads as ungraspable).
    last_seen_t: float = field(default_factory=time.monotonic)
    first_seen_t: float = field(default_factory=time.monotonic)
    observations: int = 1

    def state(self, now: float, visible_horizon_s: float = 1.5) -> str:
        return "visible" if (now - self.last_seen_t) <= visible_horizon_s else "remembered"


class BeliefStore:
    def __init__(
        self,
        match_radius_m: float = 0.08,
        pos_alpha: float = 0.4,
        conf_alpha: float = 0.3,
        forget_after_s: float | None = None,
        stale_revive_s: float = 5.0,
        max_per_label: int = 2,
    ):
        self._beliefs: list[ObjectBelief] = []
        self._match_radius = match_radius_m
        self._pos_alpha = pos_alpha
        self._conf_alpha = conf_alpha
        self._forget_after = forget_after_s
        # Anti-accumulation (2026-09-17): one physical object kept showing
        # as N rows because beliefs are never dropped and any observation
        # landing >match_radius away spawns a NEW belief (object moved /
        # two cameras disagreeing / elongated-object jitter).
        self._stale_revive_s = stale_revive_s
        self._max_per_label = max_per_label
        self._lock = threading.RLock()

    def update(
        self,
        label: str,
        position: np.ndarray,
        conf: float,
        extent: np.ndarray | None = None,
        top_z: float | None = None,
        t: float | None = None,
        color: str | None = None,
        points: np.ndarray | None = None,
    ) -> ObjectBelief:
        """Fuse one 3D observation; matches same-label beliefs by proximity.

        `points` should only be passed for REAL segmentation masks (never
        bbox-rectangle fallbacks -- those sweep in table/neighbor pixels and
        poison remembered geometry)."""
        now = time.monotonic() if t is None else t
        position = np.asarray(position, dtype=float).reshape(3)
        if points is not None:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] > 384:
                idx = np.random.default_rng(0).choice(pts.shape[0], 384, replace=False)
                pts = pts[idx]
            points = pts.copy()
        with self._lock:
            best, best_d = None, self._match_radius
            for b in self._beliefs:
                if b.label != label:
                    continue
                d = float(np.linalg.norm(b.position - position))
                if d < best_d:
                    best, best_d = b, d
            if best is None:
                same = [b for b in self._beliefs if b.label == label]
                stale = [b for b in same
                         if now - b.last_seen_t > self._stale_revive_s]
                if same and len(stale) == len(same):
                    # Every belief for this label is stale remembered state
                    # and the object re-appeared FAR from all of them: it
                    # was moved. Revive the stalest belief in place instead
                    # of accumulating a duplicate row in the world model.
                    revived = min(same, key=lambda b: b.last_seen_t)
                    revived.position = position
                    revived.conf = conf
                    if extent is not None:
                        revived.extent = extent
                    if top_z is not None:
                        revived.top_z = top_z
                    if color is not None:
                        revived.color = color
                    if points is not None:
                        revived.points = points
                    revived.last_seen_t = now
                    revived.observations += 1
                    return revived
                fresh = ObjectBelief(
                    label=label, position=position, extent=extent, top_z=top_z,
                    conf=conf, color=color, points=points,
                    last_seen_t=now, first_seen_t=now,
                )
                self._beliefs.append(fresh)
                # Cap same-label accumulation, evicting STALE duplicates
                # only -- live beliefs may be genuine extra instances
                # (three real bottles on the table) and are never evicted.
                same.append(fresh)
                while len(same) > self._max_per_label:
                    if not stale:
                        break
                    victim = min(stale, key=lambda b: b.last_seen_t)
                    self._beliefs = [b for b in self._beliefs
                                     if b is not victim]  # identity, not __eq__
                    same = [b for b in same if b is not victim]
                    stale = [b for b in stale if b is not victim]
                return fresh
            a = self._pos_alpha
            best.position = (1 - a) * best.position + a * position
            best.conf = (1 - self._conf_alpha) * best.conf + self._conf_alpha * conf
            if extent is not None:
                best.extent = extent
            if top_z is not None:
                best.top_z = top_z
            if color is not None:
                best.color = color
            if points is not None:
                best.points = points
            best.last_seen_t = now
            best.observations += 1
            return best

    def mark_removed(self, label: str, near: np.ndarray | None = None) -> bool:
        """Drop a belief after the robot itself moved the object away."""
        with self._lock:
            cands = [b for b in self._beliefs if b.label == label]
            if not cands:  # the label may be a query ("pink object")
                q = self.find(label)
                cands = [q] if q is not None else []
            if near is not None and cands:
                near = np.asarray(near, dtype=float).reshape(3)
                cands.sort(key=lambda b: float(np.linalg.norm(b.position - near)))
            if not cands:
                return False
            # Remove by IDENTITY: list.remove falls back to dataclass __eq__,
            # which compares numpy fields elementwise and raises "truth value
            # of an array is ambiguous" unless the match is the first element.
            victim = cands[0]
            self._beliefs = [b for b in self._beliefs if b is not victim]
            return True

    def all(self, now: float | None = None) -> list[ObjectBelief]:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._forget_after is not None:
                self._beliefs = [
                    b for b in self._beliefs if now - b.last_seen_t <= self._forget_after
                ]
            return list(self._beliefs)

    def find(self, query: str) -> ObjectBelief | None:
        """Resolve a label OR a color query ("pink object", "red mug")."""
        with self._lock:
            matches = [b for b in self._beliefs if b.label == query]
            if not matches:
                color, noun = parse_color_query(query)
                cands = list(self._beliefs)
                if noun:
                    exact = [b for b in cands if b.label.lower() == noun]
                    loose = exact or [
                        b for b in cands
                        if noun in b.label.lower() or b.label.lower() in noun
                    ]
                    cands = loose
                if color:
                    # Prefer the exact palette band, then perceptual
                    # neighbors (red<->pink boundary objects), and only then
                    # untagged beliefs -- never a wrong-colored object.
                    from ..perception.colors import color_matches

                    exact = [b for b in cands if b.color == color]
                    near = [
                        b for b in cands
                        if b.color is not None and b.color != color
                        and color_matches(color, b.color)
                    ]
                    cands = exact or near or [b for b in cands if b.color is None]
                # A bare color word needs the color to actually constrain;
                # a bare noun already did. No constraint at all -> no match.
                if color is None and noun is None:
                    cands = []
                matches = cands
            if not matches:
                return None
            return max(matches, key=lambda b: (b.last_seen_t, b.conf))

    def summary(self, now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        out = []
        for b in sorted(self.all(now), key=lambda b: -b.last_seen_t):
            out.append(
                {
                    "label": b.label,
                    "color": b.color,
                    "position": [round(float(x), 3) for x in b.position],
                    "state": b.state(now),
                    "age_s": round(now - b.last_seen_t, 1),
                    "conf": round(b.conf, 2),
                }
            )
        return out
