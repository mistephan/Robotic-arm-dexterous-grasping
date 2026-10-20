"""ASPIRE-style multimodal trace: every skill call leaves evidence.

Per run directory:
    trace.jsonl   - one JSON record per skill call (args, duration, result,
                    keyframe paths)
    keyframes/    - JPEG snapshots immediately before/after each call
    summary.txt   - final human-readable outcome

ASPIRE's ablation credits this per-primitive evidence with the single largest
success jump (14% -> 62%); it is also what makes post-hoc debugging of a live
demo possible.
"""

from __future__ import annotations

import csv
import json
import traceback
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TraceLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        self._trace_path = self.run_dir / "trace.jsonl"
        self._step = 0

    def save_keyframe(self, rgb: np.ndarray | None, tag: str) -> str | None:
        if rgb is None:
            return None
        name = f"keyframes/{self._step:04d}_{tag}.jpg"
        cv2.imwrite(str(self.run_dir / name), rgb)
        return name

    def record(
        self,
        skill: str,
        args: dict[str, Any],
        result: dict[str, Any],
        duration_ms: float,
        keyframe_before: str | None = None,
        keyframe_after: str | None = None,
    ) -> None:
        rec = {
            "step": self._step,
            "t": time.time(),
            "skill": skill,
            "args": _jsonable(args),
            "duration_ms": round(duration_ms, 1),
            "result": _jsonable(result),
            "keyframe_before": keyframe_before,
            "keyframe_after": keyframe_after,
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._step += 1

    def finish(self, summary: str) -> None:
        (self.run_dir / "summary.txt").write_text(summary + "\n")

    def record_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Emit a non-skill event to trace.jsonl (e.g. llm_switch, llm_fallback).

        Added 2026-08-11 for ADR-0008 (runtime profile switching + transient
        error fallback). Backward-compatible: no caller needed it before.
        """
        rec = {
            "step": self._step,
            "t": time.time(),
            "event": kind,
            "payload": _jsonable(payload or {}),
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def record_exception(self, context: str, exc: BaseException) -> None:
        """Append a full exception to a file without polluting the REPL."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with open(self.run_dir / "runtime_errors.log", "a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] {context}\n{rendered}\n")

    def record_grasp_localization(self, payload: dict[str, Any]) -> None:
        """Record one grasp-time object fix as JSONL and analysis-ready CSV."""
        self.record_event("grasp_object_localization", payload)
        path = self.run_dir / "grasp_localizations.csv"
        columns = [
            "timestamp", "grasp_call_index", "pick_attempt", "frame_id",
            "query", "detected_as", "source", "confidence",
            "base_x_m", "base_y_m", "base_z_m",
            "cloud_top_z_m", "cloud_bottom_z_m", "n_points",
            "extent_0_m", "extent_1_m", "extent_2_m",
            "bbox_x0_px", "bbox_y0_px", "bbox_x1_px", "bbox_y1_px",
        ]
        row = {name: payload.get(name) for name in columns}
        write_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return np.round(obj, 4).tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj
