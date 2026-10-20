"""Tests for ADR-0008-style llm.profile resolution from configs/demo.yaml.

Verifies that the LLM profile name is sourced from configs/demo.yaml's
top-level ``llm.profile`` field when --llm CLI is not provided. The CLI
flag still wins for ad-hoc overrides.

ADR-0008: profile switching + transient-error fallback. This test
covers the primary (selection) path only — the fallback cascade is a
runtime construct in ``AgentOrchestrator._chat_with_fallback`` and
isn't engaged when ``demo.yaml`` has no ``llm.fallback`` key.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from wrc_demo.apps.demo import _resolve_llm_profile


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_demo_yaml_llm_profile_is_default_when_no_cli(tmp_path, monkeypatch):
    """With no --llm passed, demo.yaml's llm.profile wins."""
    config_dir = tmp_path / "configs"
    (config_dir / "arms").mkdir(parents=True)
    (config_dir / "llm").mkdir(parents=True)
    _write_yaml(config_dir / "demo.yaml", {
        "llm": {"profile": "minimax"},
        "perception_loop": {"enabled": True},
    })
    _write_yaml(config_dir / "arms" / "mock.yaml", {"type": "mock"})
    _write_yaml(config_dir / "llm" / "minimax.yaml", {
        "type": "openai_compat",
        "base_url": "https://api.minimax.chat/v1",
        "model": "MiniMax-M3",
        "env_key": "MINIMAX_API_KEY",
        "supports_vision": False,
        "temperature": 0.2,
    })
    monkeypatch.setattr("wrc_demo.config.CONFIG_DIR", config_dir)

    resolved = _resolve_llm_profile(args_llm=None)
    assert resolved == "minimax"


def test_cli_llm_overrides_demo_yaml(tmp_path, monkeypatch):
    """CLI --llm beats demo.yaml's llm.profile (per ADR-0008 priority)."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_yaml(config_dir / "demo.yaml", {
        "llm": {"profile": "minimax"},
    })
    monkeypatch.setattr("wrc_demo.config.CONFIG_DIR", config_dir)

    resolved = _resolve_llm_profile(args_llm="anthropic")
    assert resolved == "anthropic"


def test_no_demo_yaml_llm_block_falls_back_to_mock(tmp_path, monkeypatch):
    """Without a llm: block in demo.yaml, the resolver returns 'mock'."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_yaml(config_dir / "demo.yaml", {
        "safety": {"workspace": {"min": [0, 0, 0], "max": [1, 1, 1]}},
    })
    monkeypatch.setattr("wrc_demo.config.CONFIG_DIR", config_dir)

    resolved = _resolve_llm_profile(args_llm=None)
    assert resolved == "mock"


def test_demo_yaml_llm_profile_empty_string_falls_back_to_mock(tmp_path, monkeypatch):
    """An empty-string profile in demo.yaml should NOT be used; fall back."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    _write_yaml(config_dir / "demo.yaml", {
        "llm": {"profile": ""},
    })
    monkeypatch.setattr("wrc_demo.config.CONFIG_DIR", config_dir)

    resolved = _resolve_llm_profile(args_llm=None)
    assert resolved == "mock"
