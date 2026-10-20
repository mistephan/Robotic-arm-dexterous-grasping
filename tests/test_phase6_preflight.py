"""Phase 6 pre-flight checks before real robot demo.

Run this on the wrc_demo rig (with rebot_rs + Orbbec Gemini2 connected)
BEFORE the actual demo. Fails fast on any missing dependency / config.

Run with:
    # Conda env (recommended — has every hardware SDK):
    conda activate wrc-demo
    pytest tests/test_phase6_preflight.py -v --no-header

    # OR any Python with the required packages (motorbridge + ultralytics +
    # pyorbbecsdk + pinocchio):
    python -m pytest tests/test_phase6_preflight.py -v --no-header

Hardware checks (can_arm UP, orbbec serial) FAIL on non-rig machines —
that's expected. The tests still tell you which config-side items
are missing.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _ok(cond, msg):
    return cond, msg


# ──────────── Python env ────────────


def test_hardware_packages_present():
    """motorbridge + pyorbbecsdk + ultralytics + pinocchio are required
    for any real-hardware run.

    The conda `wrc-demo` env (from environment.yml) carries all four. Any
    equivalent Python with the same packages installed will also pass.
    """
    missing = []
    for pkg, label in (
        ("motorbridge", "RobStride over SocketCAN"),
        ("pyorbbecsdk", "Orbbec Gemini 2"),
        ("ultralytics", "YOLOE / YOLO-World detector"),
        ("pinocchio", "FK/IK"),
    ):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"{pkg} ({label})")
    assert not missing, (
        f"Missing packages for real hardware: {missing}. "
        "Either activate the conda 'wrc-demo' env (`conda activate wrc-demo`) "
        "or install the four packages with pip."
    )


def test_motorbridge_present():
    """motorbridge drives the rebot_rs CAN bus."""
    import motorbridge  # noqa: F401


def test_ultralytics_8_4():
    """ultralytics 8.4.x for YOLOE-26L-seg."""
    import ultralytics
    assert ultralytics.__version__.startswith("8.4"), ultralytics.__version__


def test_pyorbbecsdk_present():
    """pyorbbecsdk drives the Orbbec Gemini2."""
    import pyorbbecsdk
    assert hasattr(pyorbbecsdk, "Pipeline")


def test_pinocchio_present():
    """pinocchio for IK + rpy conversions."""
    import pinocchio  # noqa: F401


# ──────────── wrc_demo native gripper + bridge ────────────


def test_wrc_demo_native_gripper_module():
    """WrcGripper is wrc_demo's port of rebot_grasp's GraspDriver (ADR-0005)."""
    from wrc_demo.control.gripper import WrcGripper  # noqa: F401


def test_wrc_demo_bridge_module():
    from wrc_demo.skills.rebot_grasp_bridge import RebotGraspBridge  # noqa: F401


# ──────────── Config files ────────────


def test_hand_eye_top_json_exists():
    assert (REPO / "data/calibration/hand_eye_top.json").exists()


def test_rebot_rs_arm_yaml_exists():
    assert (REPO / "configs/arms/rebot_rs.yaml").exists()


def test_orbbec_overhead_camera_yaml_exists():
    assert (REPO / "configs/cameras/orbbec_overhead.yaml").exists()


def test_hand_eye_top_z_in_reasonable_range():
    """The shipped hand_eye_top.json is a *redacted reference* for format
    verification only (camera_serial="REDACTED", samples_for_repro
    stripped; see `anonymization_note` in the JSON). The exact z value
    does NOT correspond to a real camera setup on the user's rig.

    What we DO check: the z translation is in a plausible range for an
    overhead camera (0.3 - 0.8 m above the table). On the real rig the
    -0.02 m compensation is added on top of the calibration height
    (~0.629 m nominal → ~0.609 m); the shipped reference sits slightly
    below that range because it's been redacted to a generic placeholder.

    To verify your own calibration, re-run
    `wrc-calib-top --verify --json <your .json>` and look for RMSE
    < 10 mm.
    """
    data = json.loads((REPO / "data/calibration/hand_eye_top.json").read_text())
    T = data["T_cam2base"]
    z = T[2][3]
    assert 0.3 < z < 0.8, (
        f"T_cam2base z = {z}, expected in (0.3, 0.8) for an overhead camera. "
        f"The shipped JSON is a redacted reference — re-run wrc-calib-top "
        f"to get a real calibration."
    )


def test_hand_eye_top_is_anonymized_reference():
    """The shipped hand_eye_top.json must be safe to publish:
    - camera_serial is "REDACTED" (was the actual Orbbec serial)
    - samples_for_repro is stripped (was 37 calibration samples + outliers)
    - schema_version is 2 with an anonymization_note
    """
    data = json.loads((REPO / "data/calibration/hand_eye_top.json").read_text())
    assert data.get("camera_serial") == "REDACTED", (
        "camera_serial must be REDACTED for open-source release; "
        "real users write their own calibration to hand_eye_top.local.json."
    )
    assert data.get("schema_version") == 2, (
        "schema_version must be 2 (the open-source release schema)."
    )
    assert "anonymization_note" in data, (
        "JSON should carry an anonymization_note explaining the redactions."
    )


def test_camera_serial_matches():
    """camera_serial in hand_eye_top.json must match cameras.json so
    make_arm binds to the same physical camera that the calibration
    was done with.
    """
    he = json.loads((REPO / "data/calibration/hand_eye_top.json").read_text())
    he_serial = he.get("camera_serial")

    cams_path = REPO / "data/calibration/cameras.json"
    if not cams_path.exists():
        pytest.skip("cameras.json not present (fresh clone)")
    cams = json.loads(cams_path.read_text())
    orbbec = next(
        (c for c in cams.get("cameras", []) if "orbbec" in c.get("type", "").lower()),
        None,
    )
    if orbbec is None:
        pytest.skip("no orbbec camera in cameras.json")
    assert orbbec.get("serial") == he_serial, (
        f"camera serial mismatch: hand_eye={he_serial!r}, "
        f"cameras.json={orbbec.get('serial')!r}"
    )


# ──────────── ADRs (Phase 6 deliverables) ────────────


@pytest.mark.parametrize("adr", [
    "0001-ready-pose-uses-wrc-home-q.md",
    "0002-grasp-force-1.0.md",
    "0003-hand-eye-compensation-delta.md",
    "0004-drop-gripper-config.md",
])
def test_adr_exists(adr):
    p = REPO / "docs/adr" / adr
    assert p.exists(), f"ADR missing: {p}"


# ──────────── Hardware (Linux) ────────────
# These FAIL on non-rig machines. They are gated by an environment
# variable so they only run when explicitly enabled (Phase 6
# runbook). Set ``WRC_PHASE6_HARDWARE=1`` on the rig to enable.


def _rig_only():
    """Skip unless the rig-only flag is set.

    The flag is opt-in: pre-flight tests on a dev machine skip the
    hardware probes; on the rig, the operator sets the flag in the
    shell that runs the pre-flight before powering the motors.
    """
    import os
    if os.environ.get("WRC_PHASE6_HARDWARE") != "1":
        pytest.skip(
            "hardware probes disabled (set WRC_PHASE6_HARDWARE=1 on the rig)"
        )


def test_can_arm_link_up():
    """CAN bus must be UP for rebot_rs."""
    _rig_only()
    r = subprocess.run(["ip", "link", "show", "can_arm"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"can_arm not present: {r.stderr}")
    if "state UP" not in r.stdout:
        pytest.fail(f"can_arm not UP: {r.stdout}")


def test_orbbec_serial_present():
    """Orbbec camera with the calibrated serial must be configured."""
    _rig_only()
    cams_path = REPO / "data/calibration/cameras.json"
    if not cams_path.exists():
        pytest.fail("cameras.json missing")
    cams = json.loads(cams_path.read_text())
    orbbec = next(
        (c for c in cams.get("cameras", []) if "orbbec" in c.get("type", "").lower()),
        None,
    )
    if orbbec is None:
        pytest.fail("no orbbec camera in cameras.json")
    assert "AY3Z331006L" in orbbec.get("serial", ""), orbbec.get("serial")
