"""Tests for ADR-0009 per-camera hand_eye_compensation_m loading.

Verifies that camera profiles' `extrinsics.hand_eye_compensation_m` is
loaded onto cfg.camera.extrinsics, and that absence is identity (None),
with no demo.yaml-level fallback (per-camera only).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from wrc_demo.config import load_demo_config


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _write_minimal_config(config_dir: Path) -> None:
    """Bare-minimum config tree so load_demo_config() doesn't crash on
    missing arms/llm profiles. Camera + demo.yaml are written per-test."""
    _write_yaml(config_dir / "demo.yaml", {"perception_loop": {"enabled": True}})
    _write_yaml(config_dir / "arms" / "mock.yaml", {"type": "mock"})
    _write_yaml(config_dir / "llm" / "mock.yaml", {"type": "mock"})


def test_per_camera_compensation_loads(tmp_path: Path):
    """Camera profile's extrinsics.hand_eye_compensation_m is loaded."""
    config_dir = tmp_path / "configs"
    _write_minimal_config(config_dir)

    _write_yaml(config_dir / "cameras" / "cal_cam.yaml", {
        "type": "mock",
        "extrinsics": {
            "mode": "eye_to_hand",
            "hand_eye_compensation_m": {"x": 0.005, "y": 0.0, "z": -0.015},
        },
    })

    cfg = load_demo_config(camera="cal_cam", config_dir=config_dir)
    comp = cfg.camera.extrinsics.hand_eye_compensation_m
    assert comp is not None, "per-camera hand_eye_compensation_m should load"
    assert comp.get("x") == 0.005
    assert comp.get("y") == 0.0
    assert comp.get("z") == -0.015


def test_per_camera_default_is_identity_none(tmp_path: Path):
    """Without per-camera compensation, cfg value is None (identity)."""
    config_dir = tmp_path / "configs"
    _write_minimal_config(config_dir)

    _write_yaml(config_dir / "cameras" / "plain_cam.yaml", {
        "type": "mock",
        "extrinsics": {"mode": "eye_to_hand"},
    })

    cfg = load_demo_config(camera="plain_cam", config_dir=config_dir)
    comp = cfg.camera.extrinsics.get("hand_eye_compensation_m")
    assert comp is None, (
        "no per-camera compensation = None; no demo.yaml fallback by design"
    )


def test_no_demo_yaml_fallback_even_when_camera_missing(tmp_path: Path):
    """demo.yaml hand_eye_compensation_m at top level is NOT injected into
    camera profiles (per-camera only; ADR-0009 v2 design)."""
    config_dir = tmp_path / "configs"
    _write_minimal_config(config_dir)

    _write_yaml(config_dir / "cameras" / "plain_cam.yaml", {
        "type": "mock",
        "extrinsics": {"mode": "eye_to_hand"},
    })
    # demo.yaml HAS a global block but it must NOT be injected
    _write_yaml(config_dir / "demo.yaml", {
        "hand_eye_compensation_m": {"x": 0.0, "y": 0.0, "z": -0.01},
        "perception_loop": {"enabled": True},
    })

    cfg = load_demo_config(camera="plain_cam", config_dir=config_dir)
    comp = cfg.camera.extrinsics.get("hand_eye_compensation_m")
    assert comp is None, (
        "demo.yaml global must NOT inject into cameras (per-camera only)"
    )
