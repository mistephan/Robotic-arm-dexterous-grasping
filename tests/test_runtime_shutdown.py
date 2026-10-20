from types import SimpleNamespace

import numpy as np
import warnings


def test_shutdown_prefers_gentle_disconnect():
    from wrc_demo.apps.demo import shutdown_runtime

    events = []

    class Arm:
        def disconnect_gently(self):
            events.append("gentle")

        def disconnect(self):
            events.append("immediate")

    stopped = SimpleNamespace(stop=lambda: None)
    runtime = SimpleNamespace(
        watcher=stopped,
        stream_server=None,
        viewer=None,
        rig=SimpleNamespace(close=lambda: None),
        camera=None,
    )
    shutdown_runtime(runtime, Arm())
    assert events == ["gentle"]


def test_build_runtime_failure_disables_arm_and_closes_partial_rig(monkeypatch):
    import wrc_demo.apps.demo as demo

    events = []

    class Arm:
        def disconnect_gently(self):
            events.append("disable_arm")

    class Rig:
        def close(self):
            events.append("close_rig")

    def fail_build(*_args, _on_resource, **_kwargs):
        _on_resource("arm", Arm())
        _on_resource("rig", Rig())
        raise ModuleNotFoundError("startup dependency missing")

    monkeypatch.setattr(demo, "_build_runtime_unchecked", fail_build)

    import pytest

    with pytest.raises(ModuleNotFoundError, match="startup dependency missing"):
        demo.build_runtime(object(), object())

    assert events == ["close_rig", "disable_arm"]


def test_build_runtime_failure_uses_full_shutdown_when_runtime_exists(monkeypatch):
    import wrc_demo.apps.demo as demo

    arm = object()
    runtime = object()
    shutdowns = []

    def fail_build(*_args, _on_resource, **_kwargs):
        _on_resource("arm", arm)
        _on_resource("runtime", runtime)
        raise KeyboardInterrupt

    monkeypatch.setattr(demo, "_build_runtime_unchecked", fail_build)
    monkeypatch.setattr(
        demo, "shutdown_runtime",
        lambda got_runtime, got_arm: shutdowns.append((got_runtime, got_arm)),
    )

    import pytest

    with pytest.raises(KeyboardInterrupt):
        demo.build_runtime(object(), object())

    assert shutdowns == [(runtime, arm)]


def test_stream_url_uses_explicit_advertise_host():
    from wrc_demo.apps.stream_server import StreamServer

    server = StreamServer(object(), advertise_host="10.0.130.116", port=8090)
    assert server.url == "http://10.0.130.116:8090/"


def test_close_gripper_is_a_deterministic_reflex():
    from wrc_demo.agent.reflex import parse_command

    plan = parse_command("close the gripper")
    assert plan is not None
    assert plan.intent == "close_gripper"
    assert plan.calls == [("close_gripper", {})]


def test_safe_arm_gentle_disconnect_fades_before_disconnect(monkeypatch):
    from wrc_demo.safety.harness import SafeArm
    import wrc_demo.safety.harness as harness_module

    events = []
    scales = []

    class ArmGroup:
        _mit_kp = np.array([10.0, 20.0])
        _mit_kd = np.array([1.0, 2.0])

        def send_mit(self, _q, *, vel, kp, kd, tau):
            scale = float(kp[0] / self._mit_kp[0])
            scales.append(scale)
            assert np.allclose(kd, self._mit_kd * scale)
            assert np.allclose(tau, 0.0)

    def safe_home():
        events.append("safe_home")

    ctrl = SimpleNamespace(
        _running=True,
        _n=2,
        _q_target=np.array([0.1, 0.2]),
        _arm_group=ArmGroup(),
        _gripper_group=None,
        _use_gravity_ff=False,
        safe_home=safe_home,
    )

    class RawArm:
        has_gripper = False

        def stop_control_loop(self):
            events.append("stop_loop")

        def disconnect(self):
            events.append("disconnect")

    arm = object.__new__(SafeArm)
    arm._ctrl = ctrl
    arm._arm = RawArm()
    monkeypatch.setattr(harness_module.time, "sleep", lambda _seconds: None)

    arm.disconnect_gently(duration_s=0.1)

    assert events == ["safe_home", "stop_loop", "disconnect"]
    assert len(scales) == 5
    assert all(a > b for a, b in zip(scales, scales[1:]))
    assert scales[-1] == 0.0
    assert ctrl._running is False


def test_real_gripper_skills_use_the_gripper_bridge():
    from wrc_demo.skills.runtime import SkillRuntime

    events = []
    runtime = object.__new__(SkillRuntime)
    runtime.bridge = SimpleNamespace(
        open_gripper=lambda: events.append("open"),
        close_gripper=lambda: events.append("close"),
    )
    runtime.arm = SimpleNamespace(
        set_gripper=lambda *_args, **_kwargs: events.append("wrong-owner")
    )
    runtime._grip_open = 5.0
    runtime._grip_closed = 0.0
    runtime.held_object = None
    runtime._gripper_width_frac = lambda: 1.0 if events[-1] == "open" else 0.0

    assert runtime.skill_open_gripper()["gripper"] == "open"
    assert runtime.skill_close_gripper()["gripper"] == "closed"
    assert events == ["open", "close"]


def test_report_prints_observation_details_and_no_false_home_claim(capsys):
    from wrc_demo.apps.demo import _print_report

    report = SimpleNamespace(
        task="look around",
        success=True,
        path="reflex",
        duration_s=1.0,
        steps=1,
        summary="get_observation()",
        tool_log=[{
            "tool": "get_observation",
            "result": {"ok": True, "objects_visible": [{"label": "box"}]},
        }],
    )
    _print_report(report)
    output = capsys.readouterr().out
    assert '"objects_visible"' in output
    assert '"box"' in output
    assert "arm parked at home" not in output


def test_safe_arm_state_includes_gripper_feedback():
    from wrc_demo.safety.harness import SafeArm

    arm = object.__new__(SafeArm)
    arm._n = 2
    arm._arm = SimpleNamespace(
        has_gripper=True,
        get_state=lambda: (
            np.array([0.1, 0.2, 0.75]),
            np.array([0.01, 0.02, 0.03]),
            np.array([1.0, 2.0, 3.0]),
        ),
    )

    state = arm.get_state()
    assert np.allclose(state.q, [0.1, 0.2])
    assert np.allclose(state.dq, [0.01, 0.02])
    assert np.allclose(state.tau, [1.0, 2.0])
    assert state.gripper_pos == 0.75
    assert state.gripper_valid is True


def test_locked_detector_forwards_one_shot_image_size():
    from wrc_demo.perception.world import LockedDetector

    calls = []

    class Detector:
        def detect(self, frame, classes=None, imgsz=None):
            calls.append((frame, classes, imgsz))
            return []

    detector = LockedDetector(Detector())
    assert detector.detect("frame", classes=["package"], imgsz=1280) == []
    assert calls == [("frame", ["package"], 1280)]


def test_python_warnings_are_written_to_run_log(tmp_path, capsys):
    from wrc_demo.apps.demo import _redirect_warnings_to_file

    old_showwarning = warnings.showwarning
    old_filters = warnings.filters[:]
    try:
        warning_path = _redirect_warnings_to_file(tmp_path)
        warnings.simplefilter("always")
        warnings.warn("cuda compatibility diagnostic", UserWarning)
    finally:
        warnings.showwarning = old_showwarning
        warnings.filters[:] = old_filters

    assert capsys.readouterr().err == ""
    content = warning_path.read_text(encoding="utf-8")
    assert "UserWarning: cuda compatibility diagnostic" in content
