"""Grasp outcome memory -- a lightweight "fake RL" over grasp attempts.

Inspired by RPent / Harness-VLA (arXiv 2607.08448): instead of retraining the
grasp model, we *steer* it with memory of past attempts. Every grasp attempt --
success OR failure -- is logged against an OBJECT PROFILE (label + size bucket +
height bucket), together with the grasp geometry that was tried, the outcome,
and (on failure) the diagnosed reason. Over time this yields:

  1. a STRATEGY PRIOR: for a given object profile, which grasp geometry tends
     to succeed (approach verticality, yaw, grasp-z fraction, width) -- used to
     RE-RANK the planner's candidates so the historically-good ones go first.
  2. PARAMETER NUDGES: learned corrections that turned a failure into a success
     (e.g. "cube-small kept tripping the table check -> raise grasp_z by +0.02").

This is deliberately NOT coordinate memory (RPent's key lesson: past coords are
for a DIFFERENT scene). We store DIMENSIONLESS features (approach angle, yaw
relative to the object's OBB, grasp depth fraction, normalized width) and small
scalar nudges, never absolute world positions. The live perception re-derives
this scene's positions every time.

Persisted as JSON so learning survives process restarts. Thread-safe: the
grasp hot path reads under a lock; writes snapshot outside the lock.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def object_profile(label: str, extent) -> str:
    """A coarse, scene-independent key for an object.

    label + size bucket + height bucket. Buckets keep near-identical objects
    (a 5 cm cube here vs a 5.2 cm cube next run) on the SAME memory row while
    keeping a banana and a tall bottle apart. Absolute coords never enter.
    """
    e = np.asarray(extent, dtype=float)
    if e.size < 3 or not np.all(np.isfinite(e)):
        footprint = height = 0.0
    else:
        srt = np.sort(np.abs(e))
        footprint = float(srt[1])   # middle axis ~ horizontal footprint
        height = float(srt[0]) if srt[0] < srt[2] else float(e[2])
        height = abs(float(e[2])) if np.isfinite(e[2]) else height
    fb = round(footprint / 0.03) * 3          # 3 cm buckets -> "3", "6", "9" cm
    hb = round(height / 0.03) * 3
    base = (label or "object").strip().lower().replace(" ", "_")
    return f"{base}|fp{fb}cm|h{hb}cm"


def grasp_features(grasp, fix) -> dict:
    """Dimensionless features of a grasp relative to the object (no coords).

    approach_vert: how top-down (1 = straight down, 0 = horizontal).
    yaw_rel:       jaw-closing yaw relative to the object's principal axis.
    depth_frac:    how far below the object top the grasp sits (0 top .. 1 bottom).
    width_norm:    jaw opening normalized by the object footprint.
    """
    appr = np.asarray(grasp.approach, dtype=float)
    n = np.linalg.norm(appr)
    approach_vert = float(-appr[2] / n) if n > 1e-6 else 0.0  # +1 = pointing down
    # jaw-opening axis is rotation column 1 in wrc convention
    R = np.asarray(grasp.rotation, dtype=float)
    jaw = R[:, 1] if R.shape == (3, 3) else np.array([1.0, 0.0, 0.0])
    yaw = float(math.atan2(jaw[1], jaw[0]))
    # relative to the object's longest horizontal axis, if known
    try:
        axes = np.asarray(fix.axes, dtype=float)
        a0 = axes[:, 0]
        obj_yaw = float(math.atan2(a0[1], a0[0]))
    except Exception:
        obj_yaw = 0.0
    yaw_rel = (yaw - obj_yaw + math.pi) % math.pi   # 0..pi, jaw is symmetric
    try:
        top_z = float(np.asarray(fix.points)[:, 2].max())
        bot_z = float(np.asarray(fix.points)[:, 2].min())
        h = max(top_z - bot_z, 1e-3)
        depth_frac = float(np.clip((top_z - grasp.position[2]) / h, 0.0, 1.0))
    except Exception:
        depth_frac = 0.5
    try:
        e = np.sort(np.abs(np.asarray(fix.extent, dtype=float)))
        fp = float(e[1]) if e.size >= 2 else 0.05
        width_norm = float(grasp.width_m / max(fp, 1e-3))
    except Exception:
        width_norm = 1.0
    return {
        "approach_vert": round(approach_vert, 3),
        "yaw_rel": round(yaw_rel, 3),
        "depth_frac": round(depth_frac, 3),
        "width_norm": round(width_norm, 3),
    }


@dataclass
class _ProfileStats:
    """Aggregated outcomes for one object profile."""
    wins: int = 0
    losses: int = 0
    # exponential-moving mean of the features of WINNING grasps
    win_features: dict = field(default_factory=dict)
    # failure reason -> count, and the nudge that later worked
    fail_reasons: dict = field(default_factory=dict)
    # learned scalar nudges (applied to the planner), EMA toward what worked
    nudges: dict = field(default_factory=lambda: {
        "grasp_z_delta": 0.0,       # meters added to grasp z
        "prefer_vert": 0.0,         # bias toward top-down (0..1)
    })
    last_update: float = 0.0


class GraspOutcomeMemory:
    """Fake-RL memory: learns which grasp geometry works per object profile."""

    def __init__(self, path: Path | None = None, ema: float = 0.3):
        self._path = Path(path) if path else None
        self._ema = float(ema)
        self._lock = threading.Lock()
        self._profiles: dict[str, _ProfileStats] = {}
        if self._path and self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                for k, v in raw.get("profiles", {}).items():
                    st = _ProfileStats()
                    st.__dict__.update(v)
                    self._profiles[k] = st
            except Exception:
                self._profiles = {}

    # ---- read side (grasp hot path) --------------------------------------

    def prior(self, label: str, fix) -> dict | None:
        """Strategy prior for this object profile, or None if unseen.

        Returns the winning-grasp feature centroid + learned nudges + a short
        human-readable summary for the agent narration.
        """
        key = object_profile(label, getattr(fix, "extent", None))
        with self._lock:
            st = self._profiles.get(key)
            if st is None or (st.wins + st.losses) == 0:
                return None
            total = st.wins + st.losses
            return {
                "profile": key,
                "wins": st.wins,
                "losses": st.losses,
                "success_rate": round(st.wins / total, 2),
                "win_features": dict(st.win_features),
                "nudges": dict(st.nudges),
                "top_fail": (max(st.fail_reasons, key=st.fail_reasons.get)
                             if st.fail_reasons else None),
            }

    def rerank(self, grasps: list, label: str, fix) -> list:
        """Re-order planner candidates by similarity to past winners + nudges.

        Non-destructive: returns a new list. Unseen profile -> unchanged order.
        """
        prior = self.prior(label, fix)
        if not prior or not prior["win_features"] or not grasps:
            return grasps
        wf = prior["win_features"]
        prefer_vert = float(prior["nudges"].get("prefer_vert", 0.0))

        def _score(g):
            f = grasp_features(g, fix)
            # negative distance to the winning-feature centroid (closer = better)
            d = 0.0
            for k, w in (("approach_vert", 1.0), ("yaw_rel", 0.5),
                         ("depth_frac", 0.8), ("width_norm", 0.4)):
                if k in wf:
                    d += w * (f[k] - wf[k]) ** 2
            # blend with the model's own quality score (keep it in charge)
            base_q = float(getattr(g, "quality", 0.0))
            vert_bonus = prefer_vert * f["approach_vert"]
            return base_q + 0.5 * (-math.sqrt(d)) + 0.3 * vert_bonus

        return sorted(grasps, key=_score, reverse=True)

    def grasp_z_nudge(self, label: str, fix) -> float:
        prior = self.prior(label, fix)
        return float(prior["nudges"].get("grasp_z_delta", 0.0)) if prior else 0.0

    # ---- write side (after each attempt) ---------------------------------

    def record(self, label: str, fix, grasp, success: bool,
               reason: str = "", z_nudge_applied: float = 0.0) -> None:
        """Log one grasp attempt against the object profile."""
        key = object_profile(label, getattr(fix, "extent", None))
        feats = grasp_features(grasp, fix) if grasp is not None else {}
        snapshot = None
        with self._lock:
            st = self._profiles.setdefault(key, _ProfileStats())
            st.last_update = time.time()
            if success:
                st.wins += 1
                # EMA the winning features toward this success
                for k, v in feats.items():
                    st.win_features[k] = round(
                        self._ema * v + (1 - self._ema) * st.win_features.get(k, v), 3)
                # a success that used a z-nudge reinforces it; otherwise decay
                cur = st.nudges.get("grasp_z_delta", 0.0)
                st.nudges["grasp_z_delta"] = round(
                    0.7 * cur + 0.3 * z_nudge_applied, 4)
                if feats.get("approach_vert", 0) > 0.7:
                    st.nudges["prefer_vert"] = round(
                        min(1.0, 0.8 * st.nudges.get("prefer_vert", 0.0) + 0.2), 3)
            else:
                st.losses += 1
                rk = _reason_key(reason)
                st.fail_reasons[rk] = st.fail_reasons.get(rk, 0) + 1
                # derive a corrective nudge from the diagnosed failure
                if rk in ("link_hits_table", "descent_unsafe", "pregrasp_unsafe"):
                    # grasp/approach dipped too low -> raise the grasp a touch
                    st.nudges["grasp_z_delta"] = round(
                        min(st.nudges.get("grasp_z_delta", 0.0) + 0.008, 0.05), 4)
                    st.nudges["prefer_vert"] = round(
                        min(1.0, st.nudges.get("prefer_vert", 0.0) + 0.1), 3)
                elif rk == "air_grasp":
                    # jaws closed on nothing -> the grasp was too shallow/wide;
                    # bias slightly DEEPER next time
                    st.nudges["grasp_z_delta"] = round(
                        max(st.nudges.get("grasp_z_delta", 0.0) - 0.005, -0.03), 4)
                elif rk in ("width_too_wide", "no_executable_grasp"):
                    st.nudges["prefer_vert"] = round(
                        min(1.0, st.nudges.get("prefer_vert", 0.0) + 0.05), 3)
            snapshot = self._dump_locked()
        if snapshot is not None:
            self._save(snapshot)

    # ---- persistence -----------------------------------------------------

    def _dump_locked(self) -> str:
        return json.dumps(
            {"profiles": {k: v.__dict__ for k, v in self._profiles.items()},
             "saved_at": time.time()},
            indent=1)

    def _save(self, snapshot: str) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(snapshot)
        except OSError:
            pass

    def summary(self) -> str:
        """Human-readable digest for the dashboard / MEMORY.md export."""
        with self._lock:
            if not self._profiles:
                return "grasp memory: empty (no attempts yet)"
            lines = []
            for k, st in sorted(self._profiles.items()):
                total = st.wins + st.losses
                sr = st.wins / total if total else 0.0
                nud = st.nudges
                lines.append(
                    f"- {k}: {st.wins}W/{st.losses}L (sr={sr:.0%}) "
                    f"z_nudge={nud.get('grasp_z_delta', 0):+.3f} "
                    f"vert={nud.get('prefer_vert', 0):.2f}"
                    + (f" top_fail={max(st.fail_reasons, key=st.fail_reasons.get)}"
                       if st.fail_reasons else ""))
            return "grasp memory (fake-RL priors):\n" + "\n".join(lines)

    def export_markdown(self, path) -> None:
        """Write a human/agent-readable MEMORY.md (RPent-style) so the VLM
        agent can read learned grasp wisdom before acting."""
        from pathlib import Path as _P
        p = _P(str(path)).expanduser()
        with self._lock:
            lines = [
                "# wrc_demo Grasp Memory (fake-RL over grasp attempts)",
                "",
                "Learned per-object grasp priors from past attempts (wins AND",
                "failures). Read this BEFORE grasping: it tells you which grasp",
                "geometry works for each object profile and what to avoid.",
                "Profiles are `label|footprint|height` buckets (scene-independent).",
                "",
            ]
            if not self._profiles:
                lines.append("_No attempts recorded yet._")
            for k, st in sorted(self._profiles.items()):
                total = st.wins + st.losses
                sr = st.wins / total if total else 0.0
                lines.append(f"## {k}")
                lines.append(f"- attempts: {total} ({st.wins}W / {st.losses}L, "
                             f"success rate {sr:.0%})")
                if st.win_features:
                    wf = st.win_features
                    lines.append(
                        f"- winning grasp geometry: approach_vert={wf.get('approach_vert', 0):.2f} "
                        f"(1=top-down), depth_frac={wf.get('depth_frac', 0):.2f}, "
                        f"width_norm={wf.get('width_norm', 0):.2f}")
                nud = st.nudges
                if abs(nud.get("grasp_z_delta", 0)) > 1e-4 or nud.get("prefer_vert", 0) > 0:
                    lines.append(
                        f"- learned corrections: grasp_z {nud.get('grasp_z_delta', 0):+.3f} m, "
                        f"prefer_top_down={nud.get('prefer_vert', 0):.2f}")
                if st.fail_reasons:
                    fails = ", ".join(f"{r}×{c}" for r, c in
                                      sorted(st.fail_reasons.items(),
                                             key=lambda x: -x[1]))
                    lines.append(f"- failure modes seen: {fails}")
                lines.append("")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines))
        except OSError:
            pass


def _reason_key(reason: str) -> str:
    """Bucket a free-text failure reason into a stable key."""
    r = (reason or "").lower()
    if "would hit the table" in r or "link" in r and "table" in r:
        return "link_hits_table"
    if "descent unsafe" in r:
        return "descent_unsafe"
    if "pregrasp" in r:
        return "pregrasp_unsafe"
    if "air" in r:
        return "air_grasp"
    if "width" in r or "gripper max" in r:
        return "width_too_wide"
    if "no executable grasp" in r:
        return "no_executable_grasp"
    if "ik" in r:
        return "ik_fail"
    return "other"
