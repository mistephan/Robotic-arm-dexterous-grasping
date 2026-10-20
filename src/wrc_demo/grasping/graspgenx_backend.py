"""GraspGen-X grasp backend: learned 6-DoF grasps behind the same Grasp API.

Talks to a GraspGen-X ZMQ server (NVlabs/GraspGenX client-server mode) with a
self-contained msgpack wire client -- the heavy model stack lives in its own
venv/process (~/Projects/demo/.graspgenx), wrc_demo only needs pyzmq +
msgpack-numpy. Launch the server with scripts/serve_graspgenx.sh.

Flow: the segmented object points from an ObjectFix (already in the BASE
frame) go up; ranked (K,4,4) grasp poses come back in the SAME frame
("grasps are returned in the input point cloud's frame"). Poses convert to
wrc_demo's Grasp convention:

    GraspGen-X gripper frame: +Z = approach axis, +X = jaw closing axis,
        origin at the GRIPPER BASE.
    wrc_demo Grasp: rotation columns [x=approach, y=jaw-opening, z=x*y],
        position = the point BETWEEN the jaws (the arm's gripper_end frame).

so R_wrc = [R[:,2], R[:,0], R[:,1]] (even permutation) and the position
uses `tcp_offset_xyz_m` when the returned origin is a robot link frame; the
full 3-D link-to-TCP vector is rotated by the returned pose. Legacy parallel
grippers can still use scalar `tip_offset_m` along the approach axis.

The OBB planner stays as the always-available fallback: any server error
degrades to the analytic path instead of failing the grasp (booth rule).
GraspGen-X's own default planner (GraspMoE) is itself diffusion + OBB
heuristics rescored by a learned discriminator, so this wiring mirrors the
paper's intended usage.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Grasp, ObjectFix


_PRODUCE_WORDS = (
    "banana", "apple", "orange", "lime", "mango", "pear", "fruit",
)


def estimate_local_elongated_geometry(
    points: np.ndarray,
    center_xy: np.ndarray,
    *,
    table_z: float = 0.0,
    height_width_ratio: float = 1.15,
) -> dict | None:
    """Estimate a curved object's local tangent, normal and section width.

    A global PCA describes a banana's chord.  This local PCA uses only a
    neighbourhood around the candidate closure centre, so its minor axis is
    the closing direction at that part of the curve.  Width is measured in a
    short strip along the local tangent and protected by a conservative
    height-derived lower bound for produce resting on the table.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 50 or pts.shape[1] < 3:
        return None
    xy = pts[:, :2]
    global_center = np.median(xy, axis=0)
    centered = xy - global_center
    evals, axes = np.linalg.eigh(np.cov(centered, rowvar=False))
    spans = np.quantile(centered @ axes, 0.95, axis=0) - np.quantile(
        centered @ axes, 0.05, axis=0
    )
    if spans[0] <= 1e-6 or spans[1] / spans[0] < 1.6:
        return None

    query = np.asarray(center_xy, dtype=float).reshape(2)
    distance = np.linalg.norm(xy - query, axis=1)
    radius = float(np.clip(0.35 * spans[1], 0.025, 0.050))
    local = pts[distance <= radius]
    if len(local) < 40:
        take = min(max(80, len(pts) // 8), len(pts))
        local = pts[np.argpartition(distance, take - 1)[:take]]
    local_xy = local[:, :2]
    local_center = np.median(local_xy, axis=0)
    local_cov = np.cov(local_xy - local_center, rowvar=False)
    local_evals, local_axes = np.linalg.eigh(local_cov)
    tangent = local_axes[:, int(np.argmax(local_evals))]
    tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
    normal = np.array([-tangent[1], tangent[0]], dtype=float)

    basis = np.column_stack([tangent, normal])
    coordinates = (xy - query) @ basis
    half_length = float(np.clip(0.65 * spans[0], 0.012, 0.025))
    strip = np.abs(coordinates[:, 0]) <= half_length
    normal_values = coordinates[strip, 1]
    if len(normal_values) < 30:
        normal_values = (local_xy - query) @ normal
    centerline_normal_offset = float(np.median(normal_values))
    section_span = float(
        np.quantile(normal_values, 0.975) - np.quantile(normal_values, 0.025)
    )
    height = max(float(np.quantile(pts[:, 2], 0.98)) - float(table_z), 0.0)
    object_width = max(section_span, float(height_width_ratio) * height)
    return {
        "tangent_xy": tangent.tolist(),
        "short_axis_xy": normal.tolist(),
        "section_width_m": section_span,
        "height_width_m": float(height_width_ratio) * height,
        "object_width_m": object_width,
        "centerline_offset_m": abs(centerline_normal_offset),
        "centerline_xy": (
            query + centerline_normal_offset * normal
        ).tolist(),
        "sample_count": int(len(normal_values)),
    }


class GraspGenXError(RuntimeError):
    pass


class GraspGenXClient:
    """Minimal REQ/REP msgpack wire client (mirrors graspgenx.serving)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, timeout_ms: int = 8000):
        try:
            import msgpack  # noqa: F401
            import msgpack_numpy
            import zmq  # noqa: F401
        except ImportError as e:
            raise GraspGenXError(
                f"graspgenx backend needs pyzmq+msgpack-numpy in this venv ({e})"
            ) from e
        msgpack_numpy.patch()
        self._host, self._port, self._timeout = host, int(port), int(timeout_ms)
        self._sock = None

    def _connect(self):
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self._host}:{self._port}")
        self._sock = sock

    def request(self, payload: dict) -> dict:
        import msgpack
        import zmq

        if self._sock is None:
            self._connect()
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as e:
            self.close()  # REQ socket is wedged after a timeout
            raise GraspGenXError(
                f"graspgenx server at {self._host}:{self._port} timed out "
                f"({self._timeout} ms); is scripts/serve_graspgenx.sh running?"
            ) from e
        resp = msgpack.unpackb(raw, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise GraspGenXError(f"graspgenx server error: {resp['error']}")
        return resp

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class GraspGenXPlanner:
    def __init__(self, cfg):
        g = cfg.get("graspgenx", None)
        get = (lambda k, d: g.get(k, d)) if g is not None else (lambda k, d: d)
        self._client = GraspGenXClient(
            host=str(get("host", "127.0.0.1")),
            port=int(get("port", 5556)),
            timeout_ms=int(get("timeout_ms", 8000)),
        )
        self.gripper = str(get("gripper", "franka_panda"))
        self.tip_offset_m = float(get("tip_offset_m", 0.10))
        self.tcp_standoff_m = max(float(get("tcp_standoff_m", 0.0)), 0.0)
        tcp_offset = get("tcp_offset_xyz_m", None)
        self.tcp_offset_xyz_m = (
            None if tcp_offset is None
            else np.asarray(tcp_offset, dtype=float).reshape(3)
        )
        self.tcp_offset_frame = str(get("tcp_offset_frame", "grasp_frame"))
        grasp_frame_rotation = get("grasp_frame_rotation_matrix", None)
        self.grasp_frame_rotation_matrix = (
            None if grasp_frame_rotation is None
            else np.asarray(grasp_frame_rotation, dtype=float).reshape(3, 3)
        )
        if self.grasp_frame_rotation_matrix is not None:
            should_be_identity = (
                self.grasp_frame_rotation_matrix.T
                @ self.grasp_frame_rotation_matrix
            )
            if not np.allclose(should_be_identity, np.eye(3), atol=1e-4):
                raise ValueError("grasp_frame_rotation_matrix must be orthonormal")
            if np.linalg.det(self.grasp_frame_rotation_matrix) < 0.0:
                raise ValueError("grasp_frame_rotation_matrix must be right-handed")
        if self.tcp_offset_frame == "hand_base_link":
            if self.grasp_frame_rotation_matrix is None:
                raise ValueError(
                    "hand_base_link tcp_offset requires grasp_frame_rotation_matrix"
                )
            # H_R_G columns describe the custom grasp axes in the hand frame;
            # the server pose is W_T_G, so convert H_p_tcp into G coordinates.
            self.tcp_offset_xyz_m = (
                self.grasp_frame_rotation_matrix.T @ self.tcp_offset_xyz_m
            )
        elif self.tcp_offset_frame != "grasp_frame":
            raise ValueError(
                "tcp_offset_frame must be 'grasp_frame' or 'hand_base_link'"
            )
        tcp_rotation = get("tcp_rotation_matrix", None)
        self.tcp_rotation_matrix = (
            None if tcp_rotation is None
            else np.asarray(tcp_rotation, dtype=float).reshape(3, 3)
        )
        self.num_grasps = int(get("num_grasps", 100))
        self.remote_planner = str(get("planner", "graspmoe")).strip().lower()
        if self.remote_planner not in {"diffusion", "graspmoe", "obb"}:
            raise ValueError(
                "graspgenx planner must be 'diffusion', 'graspmoe', or 'obb'"
            )
        self.topk = int(get("topk", 32))
        self.min_score = float(get("min_score", 0.0))
        # Keep the server-wide threshold permissive for useful OBB poses,
        # then reject effectively-zero diffusion scores using branch_tags.
        self.diff_min_score = float(get("diff_min_score", 0.01))
        self.candidate_xy_margin_m = max(
            float(get("candidate_xy_margin_m", 0.025)), 0.0
        )
        self.candidate_max_above_cloud_m = max(
            float(get("candidate_max_above_cloud_m", 0.030)), 0.0
        )
        self.moe_num_yaws = int(get("moe_num_yaws", 18))
        self.moe_tilt_degrees = [
            float(v) for v in get("moe_tilt_degrees", [0.0, -12.0, 12.0, -20.0, 20.0])
        ]
        self.moe_skip_obb_rule = str(get("moe_skip_obb_rule", "auto"))
        self.max_per_tilt_bin = int(get("max_per_tilt_bin", 8))
        self.last_latency_s: float | None = None
        self.last_diagnostics: dict = {}
        # Cross-embodiment mode: a `sweep` block describes OUR gripper by
        # its swept volume (12 numbers) -- no name lookup, no borrowed
        # Franka. The server returns the GRIPPER BASE pose. A full
        # tcp_offset_xyz_m converts a robot link origin to its TCP; legacy
        # parallel grippers use tip_offset_m along +Z. +Z=approach, +X=closing.
        sweep = get("sweep", None)
        self.sweep_params = None
        self.sweep_fingertip_depth_m = None
        if sweep is not None:
            self.sweep_params = {
                "extents_open": [float(v) for v in sweep.get("extents_open")],
                "offset_open": [float(v) for v in sweep.get("offset_open", [0, 0, 0])],
                "extents_mid": [float(v) for v in sweep.get("extents_mid")],
                "offset_mid": [float(v) for v in sweep.get("offset_mid", [0, 0, 0])],
                "gripper_type": int(sweep.get("gripper_type", 0)),
                "fingertip_depth": float(sweep.get("fingertip_depth", 0.0)),
            }
            fingertip_depth = float(sweep.get("fingertip_depth", 0.0))
            self.sweep_fingertip_depth_m = fingertip_depth
            self.tip_offset_m = float(get("tip_offset_m", fingertip_depth))
        self.insertion_depth_m = max(float(get("insertion_depth_m", 0.0)), 0.0)
        self.produce_insertion_depth_m = max(
            float(get("produce_insertion_depth_m", self.insertion_depth_m)), 0.0
        )
        self.produce_target_below_cloud_top_m = max(
            float(get("produce_target_below_cloud_top_m", 0.020)), 0.0
        )
        self.produce_center_from_xy = bool(get("produce_center_from_xy", True))
        self.produce_center_extra_depth_m = max(
            float(get("produce_center_extra_depth_m", 0.005)), 0.0
        )
        self.produce_min_tcp_above_table_m = max(
            float(get("produce_min_tcp_above_table_m", 0.010)), 0.0
        )
        self.produce_max_insertion_shortfall_m = max(
            float(get("produce_max_insertion_shortfall_m", 0.003)), 0.0
        )
        self.produce_max_retraction_m = max(
            float(get("produce_max_retraction_m", 0.060)), 0.0
        )
        self.isotropic_aspect_ratio = max(
            float(get("isotropic_aspect_ratio", 1.40)), 1.0
        )
        self.isotropic_max_yaws_per_branch_tilt = max(
            int(get("isotropic_max_yaws_per_branch_tilt", 2)), 1
        )
        preferred = np.asarray(
            get("isotropic_preferred_jaw_axis_xy", [0.0, 1.0]), dtype=float
        ).reshape(2)
        self.isotropic_preferred_jaw_axis_xy = preferred / max(
            float(np.linalg.norm(preferred)), 1e-9
        )
        self.local_curve_geometry = bool(get("local_curve_geometry", True))
        self.local_width_height_ratio = max(
            float(get("local_width_height_ratio", 1.15)), 0.0
        )
        self.local_width_pad_m = max(float(get("local_width_pad_m", 0.010)), 0.0)

    def plan(
        self, fix: ObjectFix, max_width_m: float = 0.09, table_z: float = 0.0,
        deadline: float | None = None,
    ) -> list[Grasp]:
        """Segmented base-frame object points -> ranked wrc Grasps."""
        pts = np.asarray(fix.points, dtype=np.float32)
        self.last_diagnostics = {
            "input_point_count": int(pts.shape[0]),
            "server_attempts": 0,
            "server_candidate_count": 0,
            "accepted_count": 0,
            "rejected": {},
            "rejected_tilt_bins_deg": {},
        }
        if pts.shape[0] < 50:
            self.last_diagnostics["rejected"]["insufficient_object_points"] = 1
            raise GraspGenXError(f"only {pts.shape[0]} object points (<50)")
        t0 = time.monotonic()
        if self.sweep_params is not None:
            payload = {
                "action": "infer_object",
                "point_cloud": pts,
                "sweep_volume_params": self.sweep_params,
                "num_grasps": self.num_grasps,
                "planner": self.remote_planner,
                "moe_num_yaws": self.moe_num_yaws,
                "moe_tilt_degrees": self.moe_tilt_degrees,
                "moe_skip_obb_rule": self.moe_skip_obb_rule,
            }
        else:
            payload = {
                "action": "infer",
                "point_cloud": pts,
                "gripper_name": self.gripper,
                "num_grasps": self.num_grasps,
                "grasp_threshold": float(self.min_score),
                "topk_num_grasps": self.topk,
            }
        # Diffusion sampling is stochastic: a borderline cloud (e.g. a slab
        # near the jaw limit) can land EVERY sample under the server's
        # internal score threshold on one run and return 200+ on the next.
        # One retry is cheap; a repeat empty means the cloud itself is the
        # problem, so dump it for offline replay.
        poses = scores = None
        branch_tags = []
        for _attempt in range(2):
            self.last_diagnostics["server_attempts"] += 1
            if deadline is not None and time.monotonic() >= deadline:
                raise GraspGenXError("grasp planning deadline exceeded")
            resp = self._client.request(payload)
            poses = np.asarray(resp["grasps"], dtype=np.float32).reshape(-1, 4, 4)
            scores = np.asarray(resp["confidences"], dtype=np.float32).reshape(-1)
            branch_tags = list(resp.get("branch_tags", []))
            if poses.shape[0]:
                break
        self.last_latency_s = round(time.monotonic() - t0, 3)
        self.last_diagnostics["server_candidate_count"] = int(poses.shape[0])
        self.last_diagnostics["server_branch_counts"] = {
            str(tag): int(branch_tags.count(tag)) for tag in set(branch_tags)
        }
        if poses.shape[0] == 0:
            dump = f"/tmp/wrc_ggx_empty_{int(time.time())}.npz"
            try:
                np.savez_compressed(dump, points=pts, label=str(fix.label))
            except OSError:
                dump = "unsaved"
            raise GraspGenXError(
                f"graspgenx returned no grasps twice (cloud dumped: {dump})"
            )

        # Jaw-opening width from the object's span along each grasp's jaw
        # axis (the wire protocol does not carry per-grasp widths; the
        # runtime's air-grasp verification and select_grasp need one).
        # Keep the measured requirement even when it exceeds the gripper
        # limit: select_grasp must be able to reject it, and diagnostics must
        # not misreport an over-width object as exactly max_width_m.
        #
        # topk_num_grasps is only a server hint. Some server versions return
        # every sampled pose, so enforce the configured cap again locally
        # after tabletop filtering (otherwise one pick can run hundreds of
        # unnecessary IK solves).
        grasps: list[Grasp] = []
        # A round tabletop object has no useful yaw. Estimate XY isotropy from
        # the cloud, then order candidates toward the wrist-neutral +Y jaw
        # axis and cap equivalent yaw variants per branch/tilt.
        xy = np.asarray(fix.points[:, :2], dtype=float)
        xy_centered = xy - xy.mean(axis=0)
        xy_evals, xy_axes = np.linalg.eigh(np.cov(xy_centered, rowvar=False))
        xy_evals = np.maximum(xy_evals, 1e-12)
        horizontal_aspect = float(np.sqrt(xy_evals[-1] / xy_evals[0]))
        xy_local = xy_centered @ xy_axes
        horizontal_spans = np.quantile(xy_local, 0.95, axis=0) - np.quantile(
            xy_local, 0.05, axis=0
        )
        horizontal_diameter_m = float(np.mean(horizontal_spans))
        cloud_lo = np.quantile(np.asarray(fix.points, dtype=float), 0.02, axis=0)
        cloud_hi = np.quantile(np.asarray(fix.points, dtype=float), 0.98, axis=0)
        isotropic = horizontal_aspect <= self.isotropic_aspect_ratio
        label_low = str(fix.label or fix.detection.label or "").lower()
        is_produce = any(word in label_low for word in _PRODUCE_WORDS)
        cloud_top_z = float(np.quantile(np.asarray(fix.points)[:, 2], 0.98))
        produce_target_depth_m = self.produce_target_below_cloud_top_m
        if is_produce and isotropic and self.produce_center_from_xy:
            produce_target_depth_m = min(
                self.produce_insertion_depth_m,
                max(
                    produce_target_depth_m,
                    0.5 * horizontal_diameter_m
                    + self.produce_center_extra_depth_m,
                ),
            )

        def order_key(index: int):
            if not isotropic:
                return (0.0, -float(scores[index]))
            axis_xy = np.asarray(poses[index, :2, 0], dtype=float)
            axis_xy /= max(float(np.linalg.norm(axis_xy)), 1e-9)
            alignment = abs(float(axis_xy @ self.isotropic_preferred_jaw_axis_xy))
            return (-alignment, -float(scores[index]))

        order = sorted(range(len(scores)), key=order_key)
        tilt_counts: dict[int, int] = {}
        isotropic_counts: dict[tuple[str, int], int] = {}

        def reject(reason: str, tilt_deg: float | None = None) -> None:
            rejected = self.last_diagnostics["rejected"]
            rejected[reason] = int(rejected.get(reason, 0)) + 1
            if tilt_deg is not None:
                bins = self.last_diagnostics["rejected_tilt_bins_deg"]
                key = str(int(round(tilt_deg / 5.0) * 5))
                reason_bins = bins.setdefault(reason, {})
                reason_bins[key] = int(reason_bins.get(key, 0)) + 1

        for i in order:
            if deadline is not None and time.monotonic() >= deadline:
                raise GraspGenXError("grasp planning deadline exceeded")
            T = poses[i].astype(float)
            R, p = T[:3, :3], T[:3, 3]
            approach = R[:, 2] / np.linalg.norm(R[:, 2])
            # Tabletop sanity: the model sees a floating cloud (no table), so
            # it happily proposes approaches from below. Those are physically
            # unreachable here -- drop them before wasting IK attempts.
            if approach[2] > 0.2:
                reject("approach_from_below", float(np.degrees(np.arccos(
                    np.clip(-approach[2], -1.0, 1.0)
                ))))
                continue
            tilt_deg = float(np.degrees(np.arccos(np.clip(-approach[2], -1.0, 1.0))))
            tilt_bin = int(round(tilt_deg / 5.0))
            if tilt_counts.get(tilt_bin, 0) >= self.max_per_tilt_bin:
                reject("per_tilt_bin_cap", tilt_deg)
                continue
            branch = str(branch_tags[i]) if i < len(branch_tags) else "unknown"
            if branch == "diff" and float(scores[i]) < float(
                getattr(self, "diff_min_score", 0.01)
            ):
                reject("diffusion_score_below_min", tilt_deg)
                continue
            isotropic_key = (branch, tilt_bin)
            if (
                isotropic
                and isotropic_counts.get(isotropic_key, 0)
                >= self.isotropic_max_yaws_per_branch_tilt
            ):
                reject("isotropic_yaw_cap", tilt_deg)
                continue
            open_axis = R[:, 0] / np.linalg.norm(R[:, 0])
            default_tcp_rotation = np.column_stack(
                [approach, open_axis, np.cross(approach, open_axis)]
            )
            tcp_rotation_matrix = getattr(self, "tcp_rotation_matrix", None)
            R_tcp = (
                default_tcp_rotation
                if tcp_rotation_matrix is None
                else R @ tcp_rotation_matrix
            )
            tcp_offset_xyz = getattr(self, "tcp_offset_xyz_m", None)
            sweep_depth = getattr(self, "sweep_fingertip_depth_m", None)
            if self.sweep_params is not None and sweep_depth is not None:
                # Dynamic sweep poses already define the closure anchor by
                # fingertip_depth along +Z. Transform orientation separately;
                # applying hand_base-to-TCP here shifts the anchor twice.
                jaw_center = p + approach * float(sweep_depth)
                effective_tip_offset_m = float(sweep_depth)
            elif tcp_offset_xyz is not None:
                jaw_center = p + R @ tcp_offset_xyz
                effective_tip_offset_m = float(np.linalg.norm(tcp_offset_xyz))
            else:
                jaw_center = p + approach * self.tip_offset_m
                effective_tip_offset_m = float(self.tip_offset_m)
            desired_tcp_center = (
                np.asarray(fix.position, dtype=float)
                - approach * float(getattr(self, "tcp_standoff_m", 0.0))
            )
            toward_center_m = max(
                float(np.dot(desired_tcp_center - jaw_center, approach)),
                0.0,
            )
            target_z = max(
                cloud_top_z - produce_target_depth_m,
                float(table_z) + self.produce_min_tcp_above_table_m,
            )
            target_z -= approach[2] * float(getattr(self, "tcp_standoff_m", 0.0))
            signed_depth_correction_m = 0.0
            if is_produce and approach[2] < -0.2:
                # Positive moves deeper along approach; negative retracts.
                # A long link->TCP offset can place the converted TCP below
                # the object/table even though the returned model origin is
                # valid. Correct both directions to the requested closure
                # depth instead of only pushing high candidates downward.
                signed_depth_correction_m = (
                    jaw_center[2] - target_z
                ) / -float(approach[2])
            insertion_cap_m = (
                self.produce_insertion_depth_m if is_produce
                else self.insertion_depth_m
            )
            if is_produce and approach[2] < -0.2:
                required_insertion_m = signed_depth_correction_m
                insertion_m = float(np.clip(
                    required_insertion_m,
                    -float(getattr(self, "produce_max_retraction_m", 0.060)),
                    insertion_cap_m,
                ))
                insertion_shortfall_m = abs(required_insertion_m - insertion_m)
            else:
                required_insertion_m = toward_center_m
                insertion_m = min(insertion_cap_m, required_insertion_m)
                insertion_shortfall_m = required_insertion_m - insertion_m
            if is_produce and insertion_shortfall_m > float(
                getattr(self, "produce_max_insertion_shortfall_m", 0.003)
            ):
                reject(
                    "retraction_shortfall"
                    if required_insertion_m < 0.0
                    else "insertion_shortfall",
                    tilt_deg,
                )
                continue
            tcp = jaw_center + approach * insertion_m
            # Where the fingers actually close around: with a palm TCP held
            # ``tcp_standoff_m`` OUTSIDE the object (O6), the too-high gate
            # must look at the internal closure centre, not the palm TCP.
            # standoff == 0 (parallel jaws) degrades to the old tcp gate.
            closure_center = tcp + approach * float(
                getattr(self, "tcp_standoff_m", 0.0)
            )
            if is_produce and tcp[2] < (
                float(table_z) + self.produce_min_tcp_above_table_m - 1e-6
            ):
                reject("tcp_below_table_clearance", tilt_deg)
                continue
            xy_margin = float(getattr(self, "candidate_xy_margin_m", np.inf))
            if np.any(tcp[:2] < cloud_lo[:2] - xy_margin) or np.any(
                tcp[:2] > cloud_hi[:2] + xy_margin
            ):
                reject("tcp_outside_cloud_xy_margin", tilt_deg)
                continue
            if closure_center[2] > cloud_hi[2] + float(
                getattr(self, "candidate_max_above_cloud_m", np.inf)
            ):
                reject("tcp_too_high_above_cloud", tilt_deg)
                continue
            local_geometry = None
            if getattr(self, "local_curve_geometry", True) and is_produce and not isotropic:
                local_geometry = estimate_local_elongated_geometry(
                    fix.points,
                    closure_center[:2],
                    table_z=table_z,
                    height_width_ratio=float(getattr(
                        self, "local_width_height_ratio", 1.15
                    )),
                )
            if local_geometry is not None:
                width = float(
                    local_geometry["object_width_m"]
                    + float(getattr(self, "local_width_pad_m", 0.010))
                )
            else:
                span = (fix.points - fix.position) @ open_axis
                width = float(
                    np.quantile(span, 0.95) - np.quantile(span, 0.05) + 0.015
                )
            candidate = Grasp(
                position=tcp,
                rotation=R_tcp,
                width_m=width,
                approach=approach,
                quality=float(scores[i]),
                label=fix.label,
            )
            # Wire-frame geometry is diagnostics only; it never feeds back
            # into planning or motion.
            candidate._ggx_raw_origin = p.copy()
            candidate._ggx_tip_offset_m = effective_tip_offset_m
            candidate._ggx_tcp_offset_xyz_m = (
                None if tcp_offset_xyz is None else tcp_offset_xyz.tolist()
            )
            candidate._ggx_insertion_depth_m = float(insertion_m)
            candidate._ggx_depth_correction_m = float(insertion_m)
            candidate._ggx_required_insertion_depth_m = float(
                required_insertion_m
            )
            candidate._ggx_insertion_shortfall_m = float(insertion_shortfall_m)
            candidate._ggx_target_z = float(target_z) if is_produce else None
            candidate._ggx_tcp_standoff_m = float(
                getattr(self, "tcp_standoff_m", 0.0)
            )
            candidate._ggx_closure_center = closure_center.copy()
            candidate._ggx_cloud_top_z = float(cloud_top_z)
            candidate._ggx_target_depth_m = (
                float(produce_target_depth_m) if is_produce else None
            )
            candidate._ggx_horizontal_spans_m = horizontal_spans.tolist()
            candidate._ggx_horizontal_aspect = horizontal_aspect
            candidate._ggx_isotropic = bool(isotropic)
            candidate._ggx_local_geometry = local_geometry
            candidate._ggx_branch = branch
            candidate._ggx_tilt_deg = tilt_deg
            grasps.append(candidate)
            tilt_counts[tilt_bin] = tilt_counts.get(tilt_bin, 0) + 1
            if isotropic:
                isotropic_counts[isotropic_key] = (
                    isotropic_counts.get(isotropic_key, 0) + 1
                )
            if len(grasps) >= self.topk:
                break
        self.last_diagnostics.update({
            "accepted_count": len(grasps),
            "isotropic": bool(isotropic),
            "horizontal_aspect": horizontal_aspect,
            "is_produce": bool(is_produce),
            "thresholds": {
                "diff_min_score": float(getattr(self, "diff_min_score", 0.01)),
                "max_per_tilt_bin": int(self.max_per_tilt_bin),
                "isotropic_max_yaws_per_branch_tilt": int(
                    self.isotropic_max_yaws_per_branch_tilt
                ),
                "candidate_xy_margin_m": float(getattr(self, "candidate_xy_margin_m", 0.025)),
                "candidate_max_above_cloud_m": float(getattr(
                    self, "candidate_max_above_cloud_m", 0.030
                )),
                "produce_max_insertion_shortfall_m": float(getattr(
                    self, "produce_max_insertion_shortfall_m", 0.003
                )),
                "produce_max_retraction_m": float(getattr(
                    self, "produce_max_retraction_m", 0.060
                )),
            },
        })
        return grasps
