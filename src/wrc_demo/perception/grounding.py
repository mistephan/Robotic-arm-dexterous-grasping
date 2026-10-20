"""Language -> 3D grounding (ASPIRE-style localize_object).

Pipeline: text prompt(s) -> open-vocab detection -> mask depth sampling ->
backprojection to camera-frame points -> base frame via extrinsics -> OBB
center (more robust than the mean for elongated/partially occluded objects).

Extrinsics supports both mounting styles:
- eye_to_hand: static T_cam2base (camera on a tripod/frame looking at the arm)
- eye_in_hand: T_cam2gripper composed with FK at capture time

The loader recognises three on-disk schemas:

- new JSON  (wrcdemo): ``<base>.json`` from
  ``wrc_demo.calibration.calib_dataset.save_calibration`` (preferred).
- cuRobo NPZ: ``T_cam2base`` (+ optional ``T_cam2gripper``); ``camera_serial``
  is required to match the profile, otherwise the loader warns.
- legacy NPZ: ``T_result`` (the old baseline repo's hand_eye.npz shape).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from ..types import Detection, Frame, ObjectFix, SkillError, transform_points


class Extrinsics:
    def __init__(
        self,
        mode: str = "eye_to_hand",
        T: np.ndarray | None = None,
        fk_tcp2base: Callable[[], np.ndarray] | None = None,
        *,
        source: str = "inline",
        frame_convention: str = "wrc_demo",
        schema_version: int | None = None,
        hand_eye_compensation_m: dict | None = None,
    ):
        """`T` is T_cam2base (eye_to_hand) or T_cam2gripper (eye_in_hand).

        ``source`` / ``frame_convention`` / ``schema_version`` are advisory
        metadata exposed so the loader can report the origin of the matrix
        and tests can assert which schema was used.
        """
        if mode not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"bad extrinsics mode {mode!r}")
        self.mode = mode
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=float)
        self._fk = fk_tcp2base
        self.source = source
        self.frame_convention = frame_convention
        self.schema_version = schema_version

        # Build T_compensation from hand_eye_compensation_m (or identity).
        # Mirrors rebot_grasp calibration.hand_eye_compensation_m
        # (utils/camera_utils.py::compose_cam_to_base_transform). See ADR-0009.
        comp = hand_eye_compensation_m or {}
        self._T_compensation = np.eye(4)
        self._T_compensation[:3, 3] = [
            float(comp.get("x", 0.0)),
            float(comp.get("y", 0.0)),
            float(comp.get("z", 0.0)),
        ]

    @classmethod
    def from_config(cls, cfg, fk_tcp2base=None) -> "Extrinsics":
        mode = cfg.get("mode", "eye_to_hand")
        # Schema priority: explicit `source` JSON > `hand_eye_npz` (any) >
        # inline `T` block.
        json_path = cfg.get("source")
        npz_path = cfg.get("hand_eye_npz")
        profile_serial = cfg.get("camera_serial") or cfg.get("serial")
        # ADR-0009: optional per-camera manual compensation on top of whatever
        # the loader produced. Default identity (no compensation).
        comp = cfg.get("hand_eye_compensation_m")

        if json_path:
            jp = Path(str(json_path))
            if jp.exists():
                return cls._load_new_json(jp, mode=mode, fk_tcp2base=fk_tcp2base,
                                          profile_serial=profile_serial,
                                          hand_eye_compensation_m=comp)

        if npz_path and Path(npz_path).exists():
            return cls._load_npz(Path(npz_path), mode=mode, fk_tcp2base=fk_tcp2base,
                                 profile_serial=profile_serial,
                                 hand_eye_compensation_m=comp)

        mat = cfg.get("T")
        T = np.asarray(mat, dtype=float).reshape(4, 4) if mat is not None else None
        return cls(mode=mode, T=T, fk_tcp2base=fk_tcp2base, source="inline",
                   hand_eye_compensation_m=comp)

    @classmethod
    def _load_new_json(cls, jp: Path, *, mode: str, fk_tcp2base, profile_serial,
                          hand_eye_compensation_m=None) -> "Extrinsics":
        with open(jp) as f:
            data = json.load(f)
        schema_version = int(data.get("schema_version", 1))
        fc = str(data.get("frame_convention", "wrc_demo"))
        cam_serial = str(data.get("camera_serial", ""))
        if profile_serial and cam_serial and str(profile_serial) != cam_serial:
            import sys
            print(
                f"[wrc-demo] WARNING: profile serial {profile_serial!r} != "
                f"calibration serial {cam_serial!r} in {jp}; "
                "using the calibration value anyway (override is the "
                "intentional escape hatch after swapping a camera)",
                file=sys.stderr,
            )
        T = np.asarray(data["T_cam2base"], dtype=float)
        T_marker = np.asarray(data["T_marker2gripper"], dtype=float)
        # For eye_in_hand profiles, the JSON's T_cam2base is actually the
        # static cam->gripper matrix (we don't yet produce eye_in_hand JSONs;
        # keep the data dict for future use).
        if mode == "eye_in_hand":
            return cls(
                mode=mode, T=T_marker, fk_tcp2base=fk_tcp2base,
                source=f"new-json:{jp.name}", frame_convention=fc,
                schema_version=schema_version,
                hand_eye_compensation_m=hand_eye_compensation_m,
            )
        return cls(
            mode=mode, T=T, fk_tcp2base=fk_tcp2base,
            source=f"new-json:{jp.name}", frame_convention=fc,
            schema_version=schema_version,
            hand_eye_compensation_m=hand_eye_compensation_m,
        )

    @classmethod
    def _load_npz(cls, npz_path: Path, *, mode: str, fk_tcp2base, profile_serial,
                       hand_eye_compensation_m=None) -> "Extrinsics":
        data = np.load(npz_path, allow_pickle=True)
        keys = set(data.files)
        cam_serial = ""
        if "camera_serial" in keys:
            try:
                cam_serial = str(np.asarray(data["camera_serial"]).ravel()[0])
            except Exception:
                cam_serial = ""
        if profile_serial and cam_serial and str(profile_serial) != cam_serial:
            import sys
            print(
                f"[wrc-demo] WARNING: profile serial {profile_serial!r} != "
                f"npz serial {cam_serial!r} in {npz_path}; "
                "using the npz value anyway",
                file=sys.stderr,
            )
        fc = "curobo"
        schema_version = None
        if "T_cam2base" in keys:
            T = np.asarray(data["T_cam2base"], dtype=float)
            if "T_cam2gripper" in keys:
                fc = "curobo"
            if mode == "eye_in_hand" and "T_cam2gripper" in keys:
                T_use = np.asarray(data["T_cam2gripper"], dtype=float)
            else:
                T_use = T
            return cls(
                mode=mode, T=T_use, fk_tcp2base=fk_tcp2base,
                source=f"curobo-npz:{npz_path.name}", frame_convention=fc,
                schema_version=schema_version,
                hand_eye_compensation_m=hand_eye_compensation_m,
            )
        if "T_result" in keys:
            T = np.asarray(data["T_result"], dtype=float)
            saved_mode = mode
            if "mode" in keys:
                saved_mode = str(np.asarray(data["mode"]).ravel()[0])
            return cls(
                mode=saved_mode, T=T, fk_tcp2base=fk_tcp2base,
                source=f"legacy-npz:{npz_path.name}", frame_convention=fc,
                schema_version=schema_version,
                hand_eye_compensation_m=hand_eye_compensation_m,
            )
        raise ValueError(
            f"{npz_path}: NPZ has neither T_cam2base nor T_result; "
            "cannot infer extrinsics"
        )

    def cam_to_base(self) -> np.ndarray:
        """T_cam2base at this instant (uses live FK for eye-in-hand).

        Composition mirrors rebot_grasp compose_cam_to_base_transform
        (utils/camera_utils.py) plus our _T_compensation from
        hand_eye_compensation_m (ADR-0009):
            ETH: T_cam2base = T_compensation @ T_hand_eye
            EIH: T_cam2base = T_compensation @ T_tcp2base @ T_hand_eye
        T_compensation defaults to identity (no compensation).
        """
        if self.mode == "eye_to_hand":
            return self._T_compensation @ self.T
        if self._fk is None:
            raise SkillError("eye_in_hand extrinsics need a FK callback")
        return self._T_compensation @ self._fk() @ self.T


def mask_to_points_cam(
    frame: Frame,
    mask: np.ndarray,
    max_points: int = 4000,
    depth_band: tuple[float, float] = (0.05, 0.95),
) -> np.ndarray:
    """Backproject mask pixels with valid depth into camera-frame points.

    The inter-quantile depth band drops mixed/flying pixels at mask edges
    (a real problem on the L515 around object silhouettes).
    """
    if not frame.has_depth:
        raise SkillError("no depth available; cannot lift mask to 3D")
    ys, xs = np.nonzero(mask)
    zs = frame.depth_m[ys, xs]
    good = zs > 0
    ys, xs, zs = ys[good], xs[good], zs[good]
    if zs.size == 0:
        return np.empty((0, 3))
    lo, hi = np.quantile(zs, depth_band)
    keep = (zs >= lo) & (zs <= hi)
    ys, xs, zs = ys[keep], xs[keep], zs[keep]
    if zs.size > max_points:
        idx = np.random.default_rng(0).choice(zs.size, max_points, replace=False)
        ys, xs, zs = ys[idx], xs[idx], zs[idx]
    K = frame.K
    pts = np.stack(
        [(xs - K[0, 2]) / K[0, 0] * zs, (ys - K[1, 2]) / K[1, 1] * zs, zs], axis=-1
    )
    return pts


def oriented_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA oriented bounding box -> (center, extents desc-sorted, axes cols)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    proj = centered @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    center = centroid + axes @ ((mins + maxs) / 2)
    extents = maxs - mins
    return center, extents, axes


def _bbox_mask(frame: Frame, det: Detection) -> np.ndarray:
    h, w = frame.rgb.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = det.bbox.astype(int)
    m[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)] = True
    return m


# Base frame: +x points away from the robot, +y points left (matches
# skill_push_object's direction map). Each hint -> (axis, sort_descending):
# "left" = max y, "right" = min y, "front"/"near" = min x, "back"/"far" = max x.
_SPATIAL_AXES = {
    "left": (1, True),
    "right": (1, False),
    "front": (0, False),
    "near": (0, False),
    "back": (0, True),
    "far": (0, True),
}


def localize_object(
    frame: Frame,
    label: str,
    detector,
    extrinsics: Extrinsics,
    prompts: list[str] | None = None,
    min_points: int = 10,
    spatial_hint: str | None = None,
    color: str | None = None,
    near_xyz: np.ndarray | None = None,
    vocab: list[str] | None = None,
    prefer_label: str | None = None,
) -> ObjectFix:
    """Find `label` in the frame and return its base-frame 3D fix.

    Tries an ordered prompt fallback list -- or, with `vocab`, ONE detector
    pass over a whole vocabulary where every detection is a candidate (how
    color queries like "pink object" work on closed-set detectors).
    Candidates are then narrowed by mask color (`color`), a spatial hint word
    ("left", "front", ...), and/or proximity to a remembered position
    (`near_xyz`). Raises SkillError with an agent-readable reason on failure.
    """
    from .colors import color_matches, detection_color

    T_cam2base = extrinsics.cam_to_base()
    if vocab:
        rounds = [(vocab, "vocabulary")]
    else:
        rounds = [([p], p) for p in (prompts or [label])]

    last_reason = f"no detections for {vocab or prompts or [label]}"
    for classes, what in rounds:
        dets = [d for d in detector.detect(frame, classes=classes) if d.conf > 0]
        if not dets:
            continue
        exact: list[ObjectFix] = []   # color matches the palette band
        loose: list[ObjectFix] = []   # neighbor band / unknown color
        for det in sorted(dets, key=lambda d: -d.conf):
            det_color = None
            if color is not None:
                # detection_color uses the mask when present, else the bbox
                # CENTER (whole-box medians let the background veto objects).
                # STRICT here: grabbing the red box when asked for the pink
                # one is a visible error; neighbor tolerance is for verbal
                # recall (beliefs.find), not for choosing a grasp target.
                det_color = detection_color(frame.rgb, det)
                if not color_matches(color, det_color, strict=True):
                    last_reason = f"saw {det.label!r} but it is {det_color}, not {color}"
                    continue
            mask = det.mask if det.mask is not None else _bbox_mask(frame, det)
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < min_points:
                last_reason = (
                    f"detected {what!r} but only {pts_cam.shape[0]} valid depth "
                    f"points (<{min_points}); object may be out of depth range"
                )
                continue
            pts_base = transform_points(T_cam2base, pts_cam)
            center, extents, axes = oriented_bbox(pts_base)
            fix = ObjectFix(
                label=label,
                position=center,
                points=pts_base,
                detection=det,
                extent=extents,
                axes=axes,
            )
            if color is None or det_color == color:
                exact.append(fix)
            else:
                loose.append(fix)
        candidates = exact or loose
        if not candidates:
            continue
        if spatial_hint and spatial_hint in _SPATIAL_AXES and len(candidates) > 1:
            axis, descending = _SPATIAL_AXES[spatial_hint]
            candidates.sort(key=lambda f: f.position[axis], reverse=descending)
        elif near_xyz is not None and len(candidates) > 1:
            ref = np.asarray(near_xyz, dtype=float).reshape(3)
            candidates.sort(key=lambda f: float(np.linalg.norm(f.position - ref)))
        if prefer_label and len(candidates) > 1:
            # Stable partition: same-class detections first, prior ordering
            # (hint/proximity/confidence) preserved within each group.
            candidates.sort(key=lambda f: f.detection.label != prefer_label)
        return candidates[0]
    raise SkillError(f"localize {label!r} failed: {last_reason}")
