"""Short-horizon episodic memory (the 10-15 s window the agent can consult).

A time-pruned ring of events: keyframes (as JPEG thumbnails), detections,
actions and outcomes, plus an optional TurboQuant-compressed embedding per
event for similarity recall. The digest() view is what gets pasted into the
agent's context each turn -- compact, chronological, plain text.

Thread-safe: the skill thread writes while dashboard /state handlers and the
MCP world_state tool read concurrently (deque iteration during popleft
raises RuntimeError without the lock -- reproduced in review 2026-07-18).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .vector_index import QuantizedIndex


@dataclass
class MemoryEvent:
    t: float
    kind: str  # "observation" | "action" | "outcome" | "note"
    text: str  # one-line human/agent readable summary
    data: dict[str, Any] = field(default_factory=dict)
    thumb_jpeg: bytes | None = None


class EpisodicMemory:
    def __init__(
        self,
        horizon_s: float = 15.0,
        max_events: int = 400,
        embed_dim: int | None = None,
        embed_bits: int = 4,
        thumb_width: int = 320,
        clock=time.monotonic,
        display_events: int = 600,
    ):
        self.horizon_s = horizon_s
        self._events: deque[MemoryEvent] = deque(maxlen=max_events)
        self._thumb_width = thumb_width
        self._index = QuantizedIndex(embed_dim, bits=embed_bits) if embed_dim else None
        self._clock = clock
        self._lock = threading.RLock()
        # Display-only narration log: (t, kind, text), never time-pruned.
        # The 15 s horizon above is an AGENT-PROMPT budget (short context,
        # cheap tokens) -- but the dashboard used to read the same pruned
        # store, so narration vanished from the screen after ~15 s. This
        # ring decouples "how long the agent remembers" from "how long the
        # human watches". No thumbnails -- keep it light for long sessions.
        self._history: deque[tuple[float, str, str]] = deque(maxlen=display_events)

    # ── recording ────────────────────────────────────────────────────────

    def add(
        self,
        kind: str,
        text: str,
        data: dict[str, Any] | None = None,
        rgb: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        t: float | None = None,
    ) -> MemoryEvent:
        now = self._clock() if t is None else t
        thumb = None
        if rgb is not None:
            h, w = rgb.shape[:2]
            scale = self._thumb_width / w
            small = cv2.resize(rgb, (self._thumb_width, max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumb = buf.tobytes() if ok else None
        ev = MemoryEvent(t=now, kind=kind, text=text, data=data or {}, thumb_jpeg=thumb)
        with self._lock:
            self._events.append(ev)
            self._history.append((now, kind, text))
            if self._index is not None and embedding is not None:
                self._index.add(embedding, meta=ev)
            self.prune(now)
        return ev

    def prune(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            while self._events and now - self._events[0].t > self.horizon_s:
                self._events.popleft()

    # ── recall ───────────────────────────────────────────────────────────

    def events(self, kinds: tuple[str, ...] | None = None) -> list[MemoryEvent]:
        with self._lock:
            self.prune()
            evs = list(self._events)
        if kinds:
            evs = [e for e in evs if e.kind in kinds]
        return evs

    def recall_similar(self, embedding: np.ndarray, k: int = 3) -> list[MemoryEvent]:
        if self._index is None:
            return []
        with self._lock:
            hits = self._index.search(embedding, k=k)
            live = {id(e) for e in self._events}
        return [meta for _, meta in hits if id(meta) in live]

    def last_frame_jpeg(self) -> bytes | None:
        with self._lock:
            for ev in reversed(self._events):
                if ev.thumb_jpeg is not None:
                    return ev.thumb_jpeg
        return None

    def digest(self, max_lines: int = 20, now: float | None = None) -> str:
        """Chronological plain-text view for the agent prompt."""
        now = self._clock() if now is None else now
        with self._lock:
            self.prune(now)
            recent = list(self._events)[-max_lines:]
        lines = [f"[{now - ev.t:5.1f}s ago] ({ev.kind}) {ev.text}" for ev in recent]
        if not lines:
            return "(memory empty)"
        return "\n".join(lines)

    def display_digest(self, max_lines: int = 40, now: float | None = None) -> str:
        """Like :meth:`digest` but over the never-pruned display history.

        Same ``[  N.Ns ago] (kind) text`` line format so the dashboard's
        renderFeed parser handles both identically. Serves the web
        dashboard / MCP viewers, NOT the agent prompt.
        """
        now = self._clock() if now is None else now
        with self._lock:
            recent = list(self._history)[-max_lines:]
        lines = [f"[{now - t:5.1f}s ago] ({kind}) {text}" for t, kind, text in recent]
        if not lines:
            return "(memory empty)"
        return "\n".join(lines)
