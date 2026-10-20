"""Tests for the calibration CLI's state machine (PLAN §11.1).

`wrc_demo.scripts.calib_top_orbbec._collect_samples` runs an outer
loop over `n_target` poses, each iteration waiting for operator input,
then grabbing frames until ArUco detects a stable marker, then reading
the arm pose and computing FK. We dry-run the loop with fakes for
camera, ArUco session, arm reader and user input.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _install_pyrealsense_stub():
    """`OrbbecCamera.open()` does ``import pyorbbecsdk as ob`` lazily;
    we install a no-op stub so the fake camera path doesn't ImportError.
    """
    if "pyorbbecsdk" in sys.modules:
        return
    sdk = types.ModuleType("pyorbbecsdk")
    sdk.Pipeline = object
    sdk.Config = object
    sdk.AlignFilter = object
    sdk.Context = object
    sdk.DeviceInfo = object
    sdk.OB_STREAM_COLOR = 1
    sdk.OB_STREAM_DEPTH = 2
    sdk.OB_FORMAT_BGR = 1
    sdk.OB_FORMAT_RGB = 2
    sdk.OB_FORMAT_Y16 = 3
    sdk.ALIGN_D2C_HW_MODE = 1
    sys.modules["pyorbbecsdk"] = sdk


def _make_fake_camera(width=640, height=480):
    cam = types.SimpleNamespace()
    cam.serial = "FAKE_SERIAL"
    cam.K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    cam.D = np.zeros(5)
    counter = {"n": 0}

    def _grab():
        counter["n"] += 1
        rgb = np.full((height, width, 3), 200, dtype=np.uint8)
        depth = np.full((height, width), 0.6, dtype=np.float32)
        from wrc_demo.types import Frame
        return Frame(rgb=rgb, depth_m=depth, K=cam.K, depth_source="sensor")

    cam.get_frame = _grab
    cam.close = lambda: None
    return cam


def _make_fake_aruco_session(seen_det=None, alternating: bool = True):
    """Returns an ArUcoSession-shaped object whose `detect` always yields
    ``seen_det``.

    By default, ``detect`` returns None on even calls and the
    detection on odd calls -- this lets tests exercise the manual-mode
    stability counter (which must reset on None). Pass
    ``alternating=False`` for auto-mode tests where the loop needs
    consecutive non-None detections.
    """
    if seen_det is None:
        seen_det = types.SimpleNamespace(
            marker_id=0,
            corners_px=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
            T_marker2cam=np.eye(4),
            reprojection_rmse_px=0.5,
        )
    counter = {"n": 0}
    target_ids: list[int | None] = []

    def _detect(bgr, K, D, marker_size_m, target_id=None):
        counter["n"] += 1
        target_ids.append(target_id)
        if alternating and counter["n"] % 2 == 0:
            return None
        return seen_det

    def _draw_overlay(bgr, det):
        return bgr

    s = types.SimpleNamespace()
    s.detect = _detect
    s.draw_aruco_overlay = _draw_overlay
    s.target_ids = target_ids
    return s


def _make_fake_kinematics(n_poses=10):
    """FK that yields a different pose per call (so each sample has a
    unique T_gripper2base)."""
    kin = types.SimpleNamespace()
    counter = {"n": 0}

    def _fk(q):
        counter["n"] += 1
        T = np.eye(4)
        ang = 0.1 * counter["n"]
        T[:3, :3] = np.array(
            [
                [np.cos(ang), -np.sin(ang), 0.0],
                [np.sin(ang),  np.cos(ang), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        T[:3, 3] = [0.28 + 0.01 * counter["n"], 0.0, 0.10]
        return T

    kin.fk = _fk
    return kin


def test_collect_samples_dry_run(tmp_path):
    """Run the state machine with all hardware replaced by fakes and
    verify it collects exactly `n_target` samples with the expected
    trace log entries and out-json content.
    """
    _install_pyrealsense_stub()
    from wrc_demo.scripts import calib_top_orbbec as cli
    from wrc_demo.calibration import CalibrationSample

    fake_cam = _make_fake_camera()
    fake_aruco = _make_fake_aruco_session()
    fake_kin = _make_fake_kinematics()
    trace_log = tmp_path / "trace.jsonl"
    out_json = tmp_path / "out.json"

    with patch.object(cli, "_open_orbbec", return_value=fake_cam), \
         patch.object(cli, "_wait_for_user", return_value=None), \
         patch.object(cli, "_read_arm_q", return_value=np.zeros(6)), \
         patch.object(cli, "_make_kinematics", return_value=fake_kin), \
         patch.object(cli, "ArUcoSession", return_value=fake_aruco):
        samples, finish_reason = cli._collect_samples(
            n_target=3,
            marker_size_m=0.10,
            aruco_dict="4x4_50",
            settle_time=0.1,
            marker_timeout=0.5,
            stable_frames=2,
            serial="FAKE",
            width=640,
            height=480,
            fps=30,
            out_json=out_json,
            trace_log=trace_log,
            manual=True,  # skip auto-mode arm connection
            show_window=False,
        )

    assert len(samples) == 3
    for i, s in enumerate(samples):
        assert s.pose_label == f"manual_{i + 1}"
        assert s.aruco_reprojection_px == 0.5
        assert s.inlier is True
    # Each sample got its own pose (counter incremented per FK call).
    assert np.linalg.norm(samples[0].T_gripper2base[:3, 3]
                          - samples[1].T_gripper2base[:3, 3]) > 1e-6
    assert np.linalg.norm(samples[1].T_gripper2base[:3, 3]
                          - samples[2].T_gripper2base[:3, 3]) > 1e-6
    # Trace log must have a session_start (with manual=True), three
    # sample_recorded events, and a session_end.
    events = [json.loads(line) for line in trace_log.read_text().splitlines()]
    assert events[0]["event"] == "session_start"
    assert events[0]["manual"] is True
    assert events[-1]["event"] == "session_end"
    assert events[-1]["n_collected"] == 3
    assert finish_reason == "operator finished"
    assert fake_aruco.target_ids
    assert set(fake_aruco.target_ids) == {0}


def test_collect_samples_skips_when_no_marker_detected(tmp_path):
    """If the detector never sees the marker, the iteration is recorded
    as a skipped sample and the final count is 0.
    """
    _install_pyrealsense_stub()
    from wrc_demo.scripts import calib_top_orbbec as cli

    # Detect always returns None -> no stable frame ever reached.
    fake_aruco = types.SimpleNamespace()
    fake_aruco.detect = lambda *a, **kw: None
    fake_cam = _make_fake_camera()
    fake_kin = _make_fake_kinematics()
    trace_log = tmp_path / "trace.jsonl"
    out_json = tmp_path / "out.json"

    with patch.object(cli, "_open_orbbec", return_value=fake_cam), \
         patch.object(cli, "_wait_for_user", return_value=None), \
         patch.object(cli, "_read_arm_q", return_value=np.zeros(6)), \
         patch.object(cli, "_make_kinematics", return_value=fake_kin), \
         patch.object(cli, "ArUcoSession", return_value=fake_aruco):
        samples, finish_reason = cli._collect_samples(
            n_target=2,
            marker_size_m=0.10,
            aruco_dict="4x4_50",
            settle_time=0.05,
            marker_timeout=0.1,
            stable_frames=2,
            serial=None,
            width=640,
            height=480,
            fps=30,
            out_json=out_json,
            trace_log=trace_log,
            manual=True,
            show_window=False,
        )

    assert samples == []
    events = [json.loads(line) for line in trace_log.read_text().splitlines()]
    skip_events = [e for e in events if e["event"] == "sample_skipped"]
    assert len(skip_events) == 2
    assert all(e["reason"] == "no_marker" for e in skip_events)


def test_collect_samples_dry_run_arm_unavailable(tmp_path):
    """When the arm can't be read, _read_arm_q returns None and the
    script falls back to a deterministic synthetic q (dry-run path).
    The collected sample must still have a valid T_gripper2base.
    """
    _install_pyrealsense_stub()
    from wrc_demo.scripts import calib_top_orbbec as cli

    fake_cam = _make_fake_camera()
    fake_aruco = _make_fake_aruco_session()
    fake_kin = _make_fake_kinematics()
    trace_log = tmp_path / "trace.jsonl"
    out_json = tmp_path / "out.json"

    with patch.object(cli, "_open_orbbec", return_value=fake_cam), \
         patch.object(cli, "_wait_for_user", return_value=None), \
         patch.object(cli, "_read_arm_q", return_value=None), \
         patch.object(cli, "_make_kinematics", return_value=fake_kin), \
         patch.object(cli, "ArUcoSession", return_value=fake_aruco):
        samples, _ = cli._collect_samples(
            n_target=2,
            marker_size_m=0.10,
            aruco_dict="4x4_50",
            settle_time=0.05,
            marker_timeout=0.2,
            stable_frames=2,
            serial="FAKE",
            width=640,
            height=480,
            fps=30,
            out_json=out_json,
            trace_log=trace_log,
            manual=True,
            show_window=False,
        )
    assert len(samples) == 2
    # Synthetic q seeds q[0]=0.30+0.05*i for sample i; FK returns a
    # different T per call. Just check SE(3) and that the T's differ.
    assert np.linalg.norm(samples[0].T_gripper2base[:3, 3]
                          - samples[1].T_gripper2base[:3, 3]) > 1e-6


def test_diagnose_can_bus_detects_error_passive(monkeypatch):
    """When can_arm reports ERROR-PASSIVE, the diagnostic helper must
    surface a hint pointing the operator at arm power / cable / restart.
    """
    import subprocess

    sample = (
        "5: can_arm: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
        "    link/can\n"
        "    can state ERROR-PASSIVE restart-ms 0\n"
        "    bitrate 1000000 sample-point 0.750\n"
    )

    class _FakeRun:
        def __init__(self, **_kw):
            self.stdout = sample
            self.stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeRun())
    from wrc_demo.scripts import calib_top_orbbec as cli
    diag = cli._diagnose_can_bus()
    assert diag is not None
    assert "ERROR-PASSIVE" in diag
    assert "arm 24V" in diag or "power" in diag
    assert "ip link set can_arm" in diag


def test_diagnose_can_bus_clean_state(monkeypatch):
    """A healthy UP can_arm returns None (no diagnostic)."""
    import subprocess

    sample = (
        "5: can_arm: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n"
        "    link/can\n"
        "    can state OK restart-ms 0\n"
        "    bitrate 1000000\n"
    )

    class _FakeRun:
        stdout = sample
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeRun())
    from wrc_demo.scripts import calib_top_orbbec as cli
    assert cli._diagnose_can_bus() is None


def test_arm_cleanup_returns_to_joint_zero_waits_then_releases(monkeypatch):
    """Normal cleanup must use SDK joint-zero home while enabled, wait for
    ENTER, and only then begin the soft torque release/disconnect sequence.
    """
    from wrc_demo.scripts import calib_top_orbbec as cli

    events: list[str] = []

    class _FakeController:
        def safe_home(self):
            events.append("joint_zero")

    class _FakeSafeArm:
        _ctrl = _FakeController()

        def home(self, duration_s=3.0):
            raise AssertionError("configured home_q must not replace joint-zero home")

    arm = _FakeSafeArm()
    monkeypatch.setattr("builtins.input", lambda: events.append("enter"))
    monkeypatch.setattr(
        cli,
        "_soft_release_and_disconnect",
        lambda target, **_kw: events.append("release"),
    )

    cleanup = cli._make_arm_cleanup(arm)
    cleanup()
    cleanup()  # atexit + explicit cleanup must remain idempotent

    assert events == ["joint_zero", "enter", "release"]


def test_arm_cleanup_falls_back_if_joint_zero_return_raises(monkeypatch):
    """If SDK zero return fails, use configured safe home before waiting."""
    from wrc_demo.scripts import calib_top_orbbec as cli

    events: list[str] = []

    class _FakeController:
        def safe_home(self):
            events.append("joint_zero_failed")
            raise RuntimeError("fake failure")

    class _FakeSafeArm:
        _ctrl = _FakeController()

        def home(self, duration_s=3.0):
            events.append(f"configured_home:{duration_s}")
            return True

    monkeypatch.setattr("builtins.input", lambda: events.append("enter"))
    monkeypatch.setattr(
        cli,
        "_soft_release_and_disconnect",
        lambda target, **_kw: events.append("release"),
    )

    cli._make_arm_cleanup(_FakeSafeArm())()

    assert events == [
        "joint_zero_failed",
        "configured_home:3.0",
        "enter",
        "release",
    ]


def test_soft_release_ramps_support_before_disable(monkeypatch):
    """The gentle release must stop the full-gain loop, send a descending
    gain ramp, and only then disable/disconnect the hardware.
    """
    from wrc_demo.scripts import calib_top_orbbec as cli

    events: list[str] = []
    scales: list[float] = []

    class _ArmGroup:
        _mit_kp = np.array([10.0, 20.0])
        _mit_kd = np.array([1.0, 2.0])

        def send_mit(self, _q, *, vel, kp, kd, tau):
            scales.append(float(kp[0] / self._mit_kp[0]))
            assert np.allclose(kd, self._mit_kd * scales[-1])
            assert np.allclose(tau, 0.0)

    arm_group = _ArmGroup()
    ctrl = types.SimpleNamespace(
        _n=2,
        _q_target=np.array([0.1, 0.2]),
        _arm_group=arm_group,
        _gripper_group=None,
        _use_gravity_ff=False,
        _running=True,
    )

    class _RawArm:
        has_gripper = False

        def stop_control_loop(self):
            events.append("stop_loop")

        def disable_all(self):
            events.append("disable")

        def disconnect(self):
            events.append("disconnect")

    safe_arm = types.SimpleNamespace(_ctrl=ctrl, _arm=_RawArm())
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli._soft_release_and_disconnect(safe_arm, duration_s=0.1)

    assert events == ["stop_loop", "disable", "disconnect"]
    assert len(scales) == 5
    assert all(a > b for a, b in zip(scales, scales[1:]))
    assert scales[-1] == pytest.approx(0.0)
    assert ctrl._running is False


def test_auto_mode_construction(tmp_path):
    """The auto-mode Cartesian pose sequence exists and every pose is
    in the workspace around the rig's reachable center. This is a
    structural test that pins the (xyz, rpy) tuples from cuRobo 04b so
    an accidental edit doesn't break IK feasibility.
    """
    from wrc_demo.scripts import calib_top_orbbec as cli

    poses = cli.TOP_CALIB_POSES
    assert len(poses) >= 5, "auto-mode must cover at least 5 poses"
    for (xyz, rpy) in poses:
        assert len(xyz) == 3 and len(rpy) == 3
        # Reachable workspace AABB around (0.30, -0.05, 0.28).
        assert 0.20 < xyz[0] < 0.40, f"xyz[0] {xyz[0]} out of workspace"
        assert -0.15 < xyz[1] < 0.10, f"xyz[1] {xyz[1]} out of workspace"
        assert 0.15 < xyz[2] < 0.40, f"xyz[2] {xyz[2]} out of workspace"
        # Pitch is bounded so the ArUco marker (Solution B, face-on)
        # stays roughly horizontal to the top camera.
        assert -0.5 < rpy[1] < 0.5, f"pitch {rpy[1]} out of range"


def test_pose_to_T_round_trip():
    """``_pose_to_T`` returns a valid SE(3) with the expected translation
    and an orthogonal rotation matrix.
    """
    from wrc_demo.scripts import calib_top_orbbec as cli

    xyz = (0.30, -0.05, 0.28)
    rpy = (0.1, -0.2, 0.3)
    T = cli._pose_to_T(xyz, rpy)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], xyz)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_auto_mode_loop_moves_arm_and_captures(monkeypatch, tmp_path):
    """Drive the auto-mode loop with all fakes: confirm it iterates the
    Cartesian pose sequence, calls IK + ``safe_arm.move_joints`` per
    pose, waits for stable marker detection, and records samples.
    """
    _install_pyrealsense_stub()
    from wrc_demo.scripts import calib_top_orbbec as cli

    # A stub safe_arm that records move_joints calls and returns True.
    move_calls: list[np.ndarray] = []
    heartbeats: list[int] = []
    def _hb(self):
        heartbeats.append(1)
    class _StubSafeArm:
        class _StubArm:
            def get_state(self):
                from wrc_demo.types import RobotState
                return RobotState(q=np.zeros(6), dq=None, tau=None)
            def wait_settled(self, *_args, **_kw):
                return True
        _arm = _StubArm()
        n_joints = 6
        def __init__(self):
            self.harness = type("H", (), {"estopped": False, "heartbeat": _hb})()
        def move_joints(self, q, duration_s=2.0):
            move_calls.append(np.asarray(q, dtype=float).copy())
            return True

    fake_cam = _make_fake_camera()
    fake_aruco = _make_fake_aruco_session(alternating=False)
    trace_log = tmp_path / "trace.jsonl"
    out_json = tmp_path / "out.json"

    # A stub IK that always succeeds; the resulting q matches the seed.
    class _StubKin:
        def fk(self, q):
            T = np.eye(4)
            T[:3, 3] = [0.30 + 0.01 * float(np.sum(np.abs(q))), 0.0, 0.10]
            return T

        def ik(self, T_target, q_init, retries=2):
            from wrc_demo.control.kinematics import IKResult
            q = np.asarray(q_init, dtype=float).copy()
            q += 0.05  # unique solution per pose so we can tell them apart
            return IKResult(q=q, success=True, error=1e-4, iterations=5)

    safe_arm = _StubSafeArm()
    monkeypatch.setattr(cli, "_connect_arm_with_safety",
                        lambda prof: ((_StubKin(), safe_arm, safe_arm._arm), None))
    monkeypatch.setattr(cli, "_open_orbbec", lambda *a, **kw: fake_cam)
    monkeypatch.setattr(cli, "_make_kinematics", lambda: _StubKin())
    monkeypatch.setattr(cli, "ArUcoSession",
                        lambda *a, **kw: fake_aruco)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "")
    # Limit to 3 Cartesian poses for the test.
    monkeypatch.setattr(cli, "TOP_CALIB_POSES", cli.TOP_CALIB_POSES[:3])

    samples, finish_reason = cli._collect_samples(
        n_target=3,
        marker_size_m=0.10,
        aruco_dict="4x4_50",
        settle_time=0.05,
        marker_timeout=0.4,
        stable_frames=2,
        serial="FAKE",
        width=640,
        height=480,
        fps=30,
        out_json=out_json,
        trace_log=trace_log,
        arm_profile="rebot_rs",
        manual=False,
        show_window=False,
    )

    assert len(samples) >= 1, f"expected >=1 sample, got {len(samples)}"
    assert len(move_calls) >= len(samples), (
        f"move_joints called {len(move_calls)} times for {len(samples)} samples"
    )
    # Home (first pose in TOP_CALIB_POSES) must drive a move_joints call.
    assert len(move_calls) >= 1
    assert move_calls[0].shape == (6,)
    # The auto-loop must heartbeat the safety harness before each motion
    # so the perception watchdog doesn't reject the move.
    assert len(heartbeats) >= len(move_calls), (
        f"heartbeat called {len(heartbeats)} times for {len(move_calls)} moves"
    )


def test_calib_cli_help():
    """The CLI must register all PLAN §3 sub-commands."""
    from wrc_demo.scripts import calib_top_orbbec as cli

    # Invoke --help; argparse exits via SystemExit(0).
    with pytest.raises(SystemExit):
        cli.main(["--help"])


def test_calib_dry_run_end_to_end(tmp_path):
    """`--dry-run` exercises the entire calibration pipeline (synth
    samples -> joint_solve -> JSON save -> load roundtrip) without
    hardware. The per-sample residual must be tiny (sub-mm).
    """
    import numpy as np

    from wrc_demo.scripts import calib_top_orbbec as cli

    out_p = tmp_path / "dryrun.json"
    rc = cli.main([
        "--dry-run", "--n", "12",
        "--dry-run-out", str(out_p),
        "--out", str(tmp_path / "real.json"),  # must differ
    ])
    assert rc == 0, "dry-run must exit 0 when per-sample residual is tiny"
    assert out_p.exists(), "dry-run must persist a JSON result"
    # Roundtrip: load the saved calibration and verify the schema fields.
    from wrc_demo.calibration import load_calibration
    res = load_calibration(out_p)
    assert res.method == "joint_se3_lm_huber"
    assert res.camera_serial == "DRYRUN"
    assert res.n_samples >= 4
    assert res.metrics["dryrun_X_frob_err"] >= 0
    assert res.metrics["dryrun_Y_frob_err"] >= 0
    # The sidecar NPZ contains K, D (identity placeholders).
    npz_p = out_p.with_suffix(".npz")
    assert npz_p.exists()
    arr = np.load(npz_p)
    assert "K" in arr.files and "D" in arr.files
    np.testing.assert_allclose(arr["K"], np.eye(3))


def test_calib_dry_run_rejects_colliding_paths(tmp_path):
    """--dry-run-out must differ from --out so the dry-run can't clobber
    a real calibration file by accident.
    """
    from wrc_demo.scripts import calib_top_orbbec as cli

    same = tmp_path / "calib.json"
    rc = cli.main([
        "--dry-run", "--n", "8",
        "--out", str(same),
        "--dry-run-out", str(same),
    ])
    assert rc == 2
    assert not same.exists()


def test_save_raw_samples_preserves_solver_inputs(tmp_path):
    """A solver failure must still leave the exact FK/PnP input on disk."""
    from wrc_demo.scripts import calib_top_orbbec as cli

    samples, _, _ = cli._synthesize_samples(5, seed=7)
    raw_path = cli._save_raw_samples(samples, tmp_path / "hand_eye.json")

    assert raw_path == tmp_path / "hand_eye.samples.npz"
    saved = np.load(raw_path)
    assert saved["T_gripper2base"].shape == (5, 4, 4)
    assert saved["T_marker2cam"].shape == (5, 4, 4)
    np.testing.assert_allclose(
        saved["T_gripper2base"][0], samples[0].T_gripper2base
    )
    np.testing.assert_allclose(
        saved["T_marker2cam"][0], samples[0].T_marker2cam
    )
