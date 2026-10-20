"""YAML config loading.

The demo is wired from three profile files plus one main config:

    configs/demo.yaml          - task defaults, safety limits, memory horizon
    configs/cameras/<name>.yaml
    configs/arms/<name>.yaml
    configs/llm/<name>.yaml

Profiles are merged into one `DemoConfig` namespace. Everything is plain
dict-backed so new keys never require code changes here.

ADR-0009: each camera profile declares its own `extrinsics.hand_eye_compensation_m`
(meters) under `configs/cameras/<name>.yaml`. No demo.yaml-level fallback --
per-camera is the canonical (and only) level, so adding a new camera requires
explicit compensation choice and avoids silent inheritance.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the wrc_demo repo root
CONFIG_DIR = PACKAGE_ROOT / "configs"
ASSET_DIR = PACKAGE_ROOT / "assets"
SDK_DIR = PACKAGE_ROOT / "sdk"  # vendored reBotArm SDK (wrc_demo/sdk/)


class Cfg:
    """Read-only attribute/dict hybrid over nested config dicts."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError as e:
            raise AttributeError(f"config key missing: {key!r}") from e
        return Cfg(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Cfg(val) if isinstance(val, dict) else val

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Cfg({self._data!r})"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _resolve_paths(data: Any, base: Path) -> Any:
    """Expand `${assets}` / `${repo}` / `${sdk}` placeholders in string values."""
    if isinstance(data, dict):
        return {k: _resolve_paths(v, base) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_paths(v, base) for v in data]
    if isinstance(data, str):
        return (
            data.replace("${assets}", str(ASSET_DIR))
            .replace("${sdk}", str(SDK_DIR))
            .replace("${repo}", str(PACKAGE_ROOT))
        )
    return data


def load_profile(kind: str, name: str, config_dir: Path | None = None) -> Cfg:
    """Load one profile, e.g. load_profile('cameras', 'l515')."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    path = cdir / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (cdir / kind).glob("*.yaml"))
        raise FileNotFoundError(f"no {kind} profile {name!r}; available: {available}")
    return Cfg(_resolve_paths(_load_yaml(path), cdir))


def load_demo_config(
    camera: str = "mock",
    arm: str = "mock",
    llm: str = "mock",
    config_dir: Path | None = None,
    cameras: list[str] | None = None,
) -> Cfg:
    """`cameras` (ordered, first = manipulation camera) supersedes `camera`;
    both populate cfg.camera (primary) and cfg.cameras (all).

    ADR-0009: hand_eye_compensation_m is per-camera only -- set it in the
    camera profile's extrinsics: block. No fallback from demo.yaml.
    """
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    main = _resolve_paths(_load_yaml(cdir / "demo.yaml"), cdir)
    names = [n.strip() for n in (cameras or [camera]) if n and n.strip()]
    cams = []
    for i, name in enumerate(names):
        prof = load_profile("cameras", name, cdir).as_dict()
        prof.setdefault("name", name if names.count(name) == 1 else f"{name}{i}")
        cams.append(prof)
    main["camera"] = cams[0]
    main["cameras"] = cams
    main["arm"] = load_profile("arms", arm, cdir).as_dict()
    main["llm"] = load_profile("llm", llm, cdir).as_dict()
    return Cfg(main)
