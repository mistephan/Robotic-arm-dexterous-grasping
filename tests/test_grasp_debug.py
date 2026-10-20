import json
import time

import pytest
import threading
from types import SimpleNamespace

import numpy as np

from wrc_demo.control.kinematics import IKResult
from wrc_demo.grasping.selector import select_grasp
from wrc_demo.skills.runtime import (
    SkillRuntime, _apply_strict_grasp_topdown,
    _strict_grasp_topdown_rotation,
)
from wrc_demo.types import Grasp, SkillError


def _grasp(label="part", quality=0.8, width_m=0.02):
    return Grasp(
        position=np.array([0.3, 0.1, 0.04]),
        rotation=np.eye(3),
        width_m=width_m,
        approach=np.array([0.0, 0.0, -1.0]),
        quality=quality,
        label=label,
    )


def test_strict_grasp_topdown_keeps_horizontal_closing_yaw():
    tilted = np.array([
        [0.3, 0.6, 0.0],
        [0.4, 0.8, 0.0],
        [-0.866, 0.2, 1.0],
    ])
    projected = _strict_grasp_topdown_rotation(tilted)
    np.testing.assert_allclose(projected[:, 0], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(projected[:, 1], [0.6, 0.8, 0.0])
    np.testing.assert_allclose(projected.T @ projected, np.eye(3), atol=1e-12)
    assert np.linalg.det(projected) == pytest.approx(1.0)


def test_strict_grasp_topdown_moves_graspgenx_candidate_to_target_depth():
    grasp = _grasp()
    grasp.position[2] = 0.068924
    grasp.approach = np.array([0.8, 0.0, -0.6])
    metadata = {"grasp_target_z": 0.020768}

    _apply_strict_grasp_topdown(grasp, metadata)

    np.testing.assert_allclose(grasp.approach, [0.0, 0.0, -1.0])
    assert grasp.position[2] == pytest.approx(0.020768)
    assert metadata["pre_topdown_position_z"] == pytest.approx(0.068924)
    assert metadata["strict_topdown_z_correction_m"] == pytest.approx(-0.048156)


def test_selector_honors_expired_pick_deadline():
    with pytest.raises(SkillError, match="deadline exceeded"):
        select_grasp(
            [_grasp()], object(), np.zeros(6),
            deadline=time.monotonic() - 0.001,
        )


class _SequenceIK:
    def __init__(self, results):
        self.results = iter(results)

    def ik(self, target, q, retries=0):
        return next(self.results)


def _ik(ok, pos, rot, total):
    return IKResult(
        q=np.arange(6, dtype=float) / 10.0,
        success=ok,
        error=total,
        iterations=7,
        position_error_m=pos,
        rotation_error_rad=rot,
    )


def _bare_runtime():
    runtime = SkillRuntime.__new__(SkillRuntime)
    runtime._grasp_debug_lock = threading.Lock()
    runtime._grasp_debug = {}
    runtime._grasp_debug_history = []
    runtime._grasp_debug_serial = 0
    runtime._grasp_call_serial = 0
    runtime._grasp_candidate_meta = {}
    return runtime


def test_selector_emits_failed_pregrasp_and_exhausted():
    events = []
    grasp = _grasp()
    select_error = _ik(False, 0.032, 0.24, 0.083)

    try:
        select_grasp(
            [grasp], _SequenceIK([select_error]), np.zeros(6),
            debug_callback=events.append,
        )
    except Exception:
        pass

    assert [event["stage"] for event in events] == [
        "candidate", "rejected", "exhausted"
    ]
    assert events[1]["pregrasp"] is select_error
    assert "pregrasp IK failed" in events[-1]["reject_reason"]


def test_selector_honors_source_tier_before_raw_quality():
    low_quality_learned = _grasp("learned", quality=0.2)
    high_quality_obb = _grasp("obb", quality=0.99)
    ok = _ik(True, 0.0, 0.0, 0.0)
    selected, _, _ = select_grasp(
        [high_quality_obb, low_quality_learned],
        _SequenceIK([ok, ok]), np.zeros(6),
        rank_key=lambda g: 0 if g.label == "learned" else 1,
    )
    assert selected is low_quality_learned


def test_selector_prefilter_rejects_long_axis_before_ik():
    long_axis = _grasp("long-axis", quality=0.99)
    short_axis = _grasp("short-axis", quality=0.1)
    ok = _ik(True, 0.0, 0.0, 0.0)
    selected, _, _ = select_grasp(
        [long_axis, short_axis], _SequenceIK([ok, ok]), np.zeros(6),
        prefilter=lambda g: "bad elongated axis" if g is long_axis else None,
    )
    assert selected is short_axis


def test_selector_rejects_near_pi_wrist_flip_and_tries_next_candidate():
    wrist_flip = _grasp("wrist-flip", quality=0.99)
    reachable = _grasp("reachable", quality=0.8)
    bad_pre = _ik(True, 0.0, 0.0, 0.0)
    bad_pre.q = np.array([0.0, 0.2, 0.3, -0.2, 0.1, -3.03])
    good_pre = _ik(True, 0.0, 0.0, 0.0)
    good_pre.q = np.array([0.0, 0.2, 0.3, -0.2, 0.1, 0.6])
    good_grasp = _ik(True, 0.0, 0.0, 0.0)
    good_grasp.q = np.array([0.0, 0.2, 0.4, -0.1, 0.1, 0.6])
    events = []

    selected, _, _ = select_grasp(
        [wrist_flip, reachable],
        _SequenceIK([bad_pre, good_pre, good_grasp]),
        np.zeros(6),
        max_wrist_delta_rad=2.8,
        debug_callback=events.append,
    )

    assert selected is reachable
    assert any(
        "wrist travel" in str(event.get("reject_reason", ""))
        for event in events
    )


def test_runtime_snapshot_is_candidate_consistent_and_strict_json():
    runtime = _bare_runtime()
    runtime._begin_grasp_debug("black screw head")
    runtime._set_grasp_debug_attempt(2, 3)
    grasp = _grasp("black screw head")
    runtime._remember_grasp_metadata(
        grasp,
        {
            "source": "camera_grasp",
            "confidence": 0.73,
            "camera_xyz": np.array([0.1, -0.02, 0.6]),
        },
    )
    pre = _ik(True, 0.0002, 0.001, 0.0012)
    final = _ik(False, 0.004, 0.21, 0.2101)
    runtime._on_grasp_debug({
        "candidate": grasp,
        "candidate_index": 2,
        "candidate_count": 3,
        "stage": "rejected",
        "q_current": np.zeros(6),
        "max_width_m": 0.095,
        "pregrasp": pre,
        "grasp": final,
        "reject_reason": "grasp IK failed",
    })

    state = runtime.grasp_debug_snapshot()
    assert state["candidate_index"] == 2
    assert state["grasp_call_index"] == 1
    assert state["pick_attempt"] == 2
    assert state["pick_attempt_max"] == 3
    assert state["candidate_sequence"] == 1
    assert state["source"] == "camera_grasp"
    assert state["jaw_opening_axis"] == [0.0, 1.0, 0.0]
    assert state["confidence"] == 0.73
    assert state["pregrasp"]["status"] == "OK"
    assert state["grasp"]["status"] == "FAIL"
    assert state["grasp"]["position_error_m"] == 0.004
    assert len(state["candidates"]) == 1
    assert state["candidates"][0]["source"] == "camera_grasp"
    json.dumps(state, allow_nan=False)

    runtime._begin_grasp_debug("banana")
    assert runtime.grasp_debug_snapshot()["grasp_call_index"] == 2


def test_missing_split_errors_become_json_null():
    runtime = _bare_runtime()
    legacy = SimpleNamespace(
        success=False, error=float("inf"), iterations=3, q=np.zeros(6)
    )
    encoded = runtime._serialize_ik(legacy)
    assert encoded["position_error_m"] is None
    assert encoded["rotation_error_rad"] is None
    assert encoded["total_error"] is None
    json.dumps(encoded, allow_nan=False)


def test_runtime_sdk_ik_reports_errors_for_returned_q():
    from wrc_demo.config import load_demo_config
    from wrc_demo.safety.harness import Kinematics

    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    kin = Kinematics(str(cfg.arm.urdf), str(cfg.arm.ee_frame), int(cfg.arm.n_joints))
    q = np.asarray(cfg.arm.home_q, dtype=float)
    target = kin.fk(q)
    result = kin.ik(target, q, retries=0)
    assert result.success
    assert result.position_error_m < 1e-9
    assert result.rotation_error_rad < 1e-9
    assert result.error < 1e-9


def test_dashboard_template_renders_debug_panels():
    from wrc_demo.apps.stream_server import _INDEX_HTML

    html = _INDEX_HTML.format(tiles="")
    assert 'id="g-width"' in html
    assert 'id="g-attempt"' in html
    assert 'id="g-call"' in html
    assert 'id="g-sequence"' in html
    assert 'id="g-jaw-axis"' in html
    assert 'id="ik-pos-error"' in html
    assert "renderGraspDebug(s)" in html
    assert "async function tick() {" in html
    # narration is rendered as structured rows (age / kind badge / text),
    # not a monospace pre-wrap wall; candidates are colour-coded divs.
    assert "renderFeed(s.events)" in html
    assert 'class="ev-k' in html
    assert 'class="cand"' in html


def test_grasp_pose_projects_axes_and_zero_offset_origin():
    from wrc_demo.apps.live_view import draw_grasp_pose

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    debug = {
        "base_xyz": [0.0, 0.0, 1.0],
        "raw_origin_xyz": [0.0, 0.0, 1.0],
        "tip_offset_m": 0.0,
        "approach": [1.0, 0.0, 0.0],
        "jaw_opening_axis": [0.0, 1.0, 0.0],
        "candidate_index": 1,
        "source": "graspgenx",
    }
    assert draw_grasp_pose(img, K, np.eye(4), debug, axis_length_m=0.1)
    assert np.count_nonzero(img) > 0


class _SymmetricIK:
    def ik(self, target, seed, retries=0):
        q = np.zeros(6)
        q[-1] = 2.9 if target[1, 1] > 0.0 else 0.1
        return IKResult(
            q=q, success=True, error=0.0, iterations=1,
            position_error_m=0.0, rotation_error_rad=0.0,
        )


def test_parallel_jaw_symmetry_selects_shorter_wrist_solution():
    grasp = _grasp()
    selected, q_pre, q_grasp = select_grasp(
        [grasp], _SymmetricIK(), np.zeros(6),
        parallel_jaw_symmetry=True, wrist_motion_weight=2.0,
    )
    assert selected is grasp
    assert selected._symmetry_flipped is True
    np.testing.assert_allclose(selected.rotation[:, 1], [0.0, -1.0, 0.0])
    assert q_pre[-1] == 0.1
    assert q_grasp[-1] == 0.1


def test_runtime_prefers_graspgenx_diff_over_internal_obb():
    runtime = _bare_runtime()
    runtime.cfg = SimpleNamespace(grasp={
        "source_priority": ["graspgenx", "camera_grasp", "analytic_obb"],
        "graspgenx_branch_priority": ["diff", "obb"],
        "prefer_topdown_max_tilt_rad": 0.785,
    })
    diff = _grasp("diff")
    obb = _grasp("obb")
    runtime._remember_grasp_metadata(diff, {"source": "graspgenx", "branch": "diff"})
    runtime._remember_grasp_metadata(obb, {"source": "graspgenx", "branch": "obb"})
    assert runtime._grasp_rank_key(diff) < runtime._grasp_rank_key(obb)


def test_runtime_keeps_clearly_higher_score_internal_obb():
    runtime = _bare_runtime()
    runtime.cfg = SimpleNamespace(grasp={
        "source_priority": ["graspgenx", "camera_grasp", "analytic_obb"],
        "graspgenx_branch_priority": ["diff", "obb"],
        "graspgenx_branch_score_band": 0.10,
        "prefer_topdown_max_tilt_rad": 0.785,
    })
    diff = _grasp("diff")
    diff.quality = 0.72
    obb = _grasp("obb")
    obb.quality = 0.91
    runtime._remember_grasp_metadata(diff, {"source": "graspgenx", "branch": "diff"})
    runtime._remember_grasp_metadata(obb, {"source": "graspgenx", "branch": "obb"})
    assert runtime._grasp_rank_key(obb) < runtime._grasp_rank_key(diff)
