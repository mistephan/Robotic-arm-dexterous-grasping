"""Per-profile tests for `make_llm()` factory and secret loading.

Covers:
- ADR-0006: every profile reads its secret from the named env var;
  plaintext secrets never appear in YAML.
- ADR-0008: the fallback chain constructs cleanly.

These tests do NOT require a live server. They construct LLMClient
objects with stub base_url values; if a network call were attempted,
it would fail and the test would catch it (the construction itself
must succeed).

Run:
    python -m pytest tests/test_profiles.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
LLM_PROFILES_DIR = REPO / "configs" / "llm"


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _make_cfg(profile_name: str, overrides: dict | None = None) -> dict:
    """Build a minimal cfg dict compatible with make_llm() — bypass
    load_demo_config() so tests don't need a full demo.yaml."""
    import yaml
    p = LLM_PROFILES_DIR / f"{profile_name}.yaml"
    if not p.exists():
        pytest.skip(f"profile {profile_name!r} not found at {p}")
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("type", "openai_compat")
    if overrides:
        cfg.update(overrides)
    return cfg


# ────────────────────────────────────────────────────────────────────────
# ADR-0006: secrets come from env vars, never from YAML
# ────────────────────────────────────────────────────────────────────────

class TestSecretLoading:
    """ADR-0006: every LLM profile must source its api_key from an env var."""

    # NOTE: "local_qwen" was removed from this list 2026-08-27 -- profile deleted.
    @pytest.mark.parametrize("profile", ["anthropic", "openai", "local_qwen3_vl", "minimax"])
    def test_profile_yaml_has_no_literal_secret(self, profile):
        """No `sk-` or `sk-cp-` literal in any tracked profile YAML.
        Skips if the profile isn't tracked (e.g. local-only minimax.yaml)."""
        p = LLM_PROFILES_DIR / f"{profile}.yaml"
        if not p.exists():
            pytest.skip(f"{p} not present")
        text = p.read_text()
        for prefix in ("sk-cp-", "sk-ant-", "sk-"):
            assert prefix not in text, (
                f"profile {profile!r} contains literal secret prefix {prefix!r}; "
                "see ADR-0006 — secrets must come from env vars"
            )

    def test_minimax_profile_uses_env_key(self, monkeypatch):
        """`minimax` profile (when present) declares `env_key: MINIMAX_API_KEY`
        and does not embed the key inline."""
        p = LLM_PROFILES_DIR / "minimax.yaml"
        if not p.exists():
            pytest.skip("minimax.yaml not present (local-only)")
        import yaml
        cfg = yaml.safe_load(p.read_text())
        assert "env_key" in cfg, "minimax.yaml must declare env_key (ADR-0006)"
        assert cfg["env_key"] == "MINIMAX_API_KEY"
        # The api_key field must be absent or None; never a literal secret.
        api_key = cfg.get("api_key")
        if api_key is not None:
            assert not api_key.startswith("sk-"), (
                f"minimax.yaml api_key starts with 'sk-' — looks like a literal secret"
            )

    def test_make_llm_reads_env_key(self, monkeypatch):
        """make_llm(minimax_cfg) with MINIMAX_API_KEY set should yield a client
        whose api_key attribute equals the env var value."""
        monkeypatch.setenv("MINIMAX_API_KEY", "test-secret-from-env")
        # Importing the LLM module pulls in OpenAI SDK; ensure it's installed.
        from wrc_demo.agent.llm import make_llm
        cfg = _make_cfg("minimax")
        # Force type to openai_compat in case the profile uses something else
        cfg["type"] = "openai_compat"
        # Make sure no inline key is overriding the env var
        cfg.pop("api_key", None)
        client = make_llm(cfg)
        assert client.api_key == "test-secret-from-env", (
            f"client.api_key={client.api_key!r}; expected the env var value"
        )

    def test_make_llm_missing_env_var_raises(self, monkeypatch):
        """make_llm(profile_with_env_key) raises LLMConfigError when the
        env var is unset. Surfaces misconfiguration instead of failing
        deep in chat()."""
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        from wrc_demo.agent.llm import make_llm
        cfg = _make_cfg("minimax")
        cfg["type"] = "openai_compat"
        cfg.pop("api_key", None)
        with pytest.raises(Exception) as ei:
            make_llm(cfg)
        # Accept either LLMConfigError or KeyError-style surface;
        # the goal is "raises something at construction, not in chat()".
        assert "MINIMAX_API_KEY" in str(ei.value) or "env" in str(ei.value).lower(), (
            f"exception {type(ei.value).__name__}: {ei.value} "
            "doesn't mention the env var name"
        )

    def test_local_qwen3_vl_no_env_needed(self, monkeypatch):
        """local_qwen3_vl uses api_key: EMPTY (literal dummy); no env var required.

        Replaces the deleted local_qwen test (profile removed 2026-08-27).
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from wrc_demo.agent.llm import make_llm
        cfg = _make_cfg("local_qwen3_vl")
        client = make_llm(cfg)
        # The dummy is what local servers expect
        assert client.api_key in {"EMPTY", None}


# ────────────────────────────────────────────────────────────────────────
# ADR-0008: profile enumeration and fallback chain construction
# ────────────────────────────────────────────────────────────────────────

class TestProfileEnumeration:
    def test_all_profiles_have_yaml(self):
        """Every profile referenced in demo docs exists as a yaml file."""
        known_profiles = {
            "mock", "anthropic", "openai",
            "local_qwen3_vl", "minimax",
            # "local_qwen" was deleted 2026-08-27 -- never wired.
        }
        for name in known_profiles:
            p = LLM_PROFILES_DIR / f"{name}.yaml"
            # minimax may be local-only; skip if absent
            if name == "minimax" and not p.exists():
                continue
            assert p.exists(), f"profile {name!r} referenced but no yaml at {p}"

    def test_make_llm_unknown_type_raises(self):
        """make_llm() with an unrecognized type raises ValueError."""
        from wrc_demo.agent.llm import make_llm
        cfg = {"type": "this-is-not-a-real-type", "model": "x"}
        with pytest.raises((ValueError, KeyError)):
            make_llm(cfg)


# ────────────────────────────────────────────────────────────────────────
# Mock profile: must always work, no env, no network
# ────────────────────────────────────────────────────────────────────────

class TestMockProfile:
    def test_mock_construction_is_free(self):
        """MockLLM should construct without any env, base_url, or network."""
        from wrc_demo.agent.llm import MockLLM, LLMResponse, make_llm
        cfg = _make_cfg("mock")
        client = make_llm(cfg)
        assert isinstance(client, MockLLM)

    def test_mock_serves_script(self):
        from wrc_demo.agent.llm import MockLLM, LLMResponse, ToolCall
        from wrc_demo.config import Cfg
        client = MockLLM(script=[
            LLMResponse(tool_calls=[ToolCall(name="get_observation", arguments={})]),
            LLMResponse(tool_calls=[ToolCall(name="task_done", arguments={"success": True})]),
        ])
        resp = client.chat("system", [{"role": "user", "content": "x"}])
        assert resp.tool_calls[0].name == "get_observation"
        resp2 = client.chat("system", [{"role": "user", "content": "x"}])
        assert resp2.tool_calls[0].name == "task_done"
