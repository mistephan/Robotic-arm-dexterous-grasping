"""Until-success persistence: grasp-candidate safety vetting (vet_pose +
select_grasp validate) and the pick_and_place retry loop that re-observes
between attempts instead of giving up after two."""

import threading
import time

import numpy as np
import pytest

from wrc_demo.config import load_demo_config
from wrc_demo.grasping import select_grasp
from wrc_demo.safety.harness import SafetyHarness, SafetyLimits
from wrc_demo.types import Grasp, SkillError


def limits(**kw):
    base = dict(
        workspace_min=np.array([0.10, -0.30, -0.01]),
        workspace_max=np.array([0.50, 0.30, 0.55]),
        table_z=0.0,
        table_clearance=0.02,
        max_joint_vel=1.0,
        watchdog_s=1e9,
    )
    base.update(kw)
    return SafetyLimits(**base)


class FakeKin:
    """Kinematics stub: TCP = q[:3], joint origins clear of the table."""

    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([[0, 0, 0.2], [0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])


# ── vet_pose: static candidate checks, no motion, no raise ────────────────


def test_vet_pose_accepts_and_rejects():
    h = SafetyHarness(limits(), kinematics=FakeKin())
    assert h.vet_pose(np.array([0.3, 0.0, 0.20, 0, 0, 0])) is None
    assert "workspace" in h.vet_pose(np.array([0.7, 0.0, 0.20, 0, 0, 0]))
    assert "joint" in h.vet_pose(np.array([0.3, 0.0, 0.20, 0, 0, 3.5]))
    # below table clearance without an exemption -> rejected...
    q_low = np.array([0.3, 0.0, 0.005, 0, 0, 0])
    assert "below table" in h.vet_pose(q_low)
    # ...but fine inside the hypothetical descent cylinder
    assert h.vet_pose(q_low, exempt_xy=[0.3, 0.0], exempt_z_min=-0.02) is None
    # nothing raises, and the LIVE exemption state is untouched
    assert h._grasp_exempt is None


def test_vet_pose_keep_out():
    h = SafetyHarness(
        limits(keep_out=[(np.array([0.25, -0.05, 0.1]), np.array([0.35, 0.05, 0.3]))]),
        kinematics=FakeKin(),
    )
    assert "keep-out" in h.vet_pose(np.array([0.3, 0.0, 0.20, 0, 0, 0]))
    assert h.vet_pose(np.array([0.15, 0.2, 0.20, 0, 0, 0])) is None


# ── select_grasp: validate() vetoes candidates IK cannot ─────────────────


class _OkIK:
    def __init__(self, q):
        self.success = True
        self.q = np.asarray(q, dtype=float)
        self.error = 0.0


class StubKin:
    def ik(self, T, q0):
        return _OkIK(np.asarray(q0) + 0.01)


def _grasp(quality, label):
    return Grasp(
        position=np.array([0.3, 0.0, 0.05]),
        rotation=np.eye(3),
        width_m=0.03,
        approach=np.array([0.0, 0.0, -1.0]),
        quality=quality,
        label=label,
    )


def test_select_grasp_validate_vetoes_best_candidate():
    g_best, g_ok = _grasp(0.9, "best"), _grasp(0.5, "fallback")

    def veto_best(g, q_pre, q_grasp):
        return "pregrasp unsafe: elbow would hit the table" if g.label == "best" else None

    g, q_pre, q_grasp = select_grasp(
        [g_best, g_ok], StubKin(), np.zeros(6), validate=veto_best
    )
    assert g.label == "fallback"


def test_select_grasp_all_vetoed_reports_reasons():
    with pytest.raises(SkillError, match="unsafe"):
        select_grasp(
            [_grasp(0.9, "a"), _grasp(0.5, "b")],
            StubKin(),
            np.zeros(6),
            validate=lambda g, qp, qg: "grasp unsafe: TCP outside workspace",
        )


# ── pick_and_place persists until it succeeds ─────────────────────────────


def test_pick_and_place_retries_until_grasp_succeeds(tmp_path):
    """Attempt 1 closes on air; the object 'appears' in the jaws afterwards
    (as if the first try nudged it into a graspable spot). The loop must
    re-observe and land attempt 2 instead of giving up."""
    from wrc_demo.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    arm.object_stop_frac = None  # attempt 1: jaws close fully -> air grasp

    def arm_the_object():
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if any("pick attempt 1 failed" in e.text for e in runtime.memory.events()):
                arm.object_stop_frac = 0.5  # now something is in the jaws
                return
            time.sleep(0.02)

    flipper = threading.Thread(target=arm_the_object, daemon=True)
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red object") is None:
            time.sleep(0.05)
        flipper.start()
        result = runtime.execute("pick_and_place", {"object": "red object"})
        assert result.get("ok"), result
        assert result["grasp_attempts"] == 2
        notes = [e.text for e in runtime.memory.events()]
        assert any("not giving up" in n for n in notes)
    finally:
        flipper.join(timeout=1.0)
        shutdown_runtime(runtime, arm)



def test_pick_and_place_does_not_replan_after_execution_fault(tmp_path):
    from wrc_demo.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["max_pick_attempts"] = 5
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    calls = {"grasp": 0, "home": 0, "observe": 0}

    def fail_motion(*args, **kwargs):
        calls["grasp"] += 1
        raise SkillError("ready motion did not reach its commanded TCP")

    runtime.skill_grasp_object = fail_motion
    runtime.skill_move_home = lambda: calls.__setitem__("home", calls["home"] + 1) or {"at": "home"}
    runtime._reobserve = lambda: calls.__setitem__("observe", calls["observe"] + 1)
    try:
        result = runtime.skill_pick_and_place("banana")
        assert not result.get("ok")
        assert "after 1 attempts" in result["error"]
        assert "execution fault; not retrying" in result["error"]
        assert calls == {"grasp": 1, "home": 1, "observe": 0}
    finally:
        shutdown_runtime(runtime, arm)


def test_pick_and_place_attempt_budget_is_honored(tmp_path):
    from wrc_demo.apps.demo import build_runtime, shutdown_runtime

    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["grasp"]["max_pick_attempts"] = 2
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    arm.object_stop_frac = None  # every close is an air grasp
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red object") is None:
            time.sleep(0.05)
        result = runtime.execute("pick_and_place", {"object": "red object"})
        assert not result.get("ok")
        assert result["stage"] == "grasp"
        assert "after 2 attempts" in result["error"]
        assert runtime.held_object is None
    finally:
        shutdown_runtime(runtime, arm)
