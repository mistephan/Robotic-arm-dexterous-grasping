"""MCP stdio server protocol tests: a real subprocess, real JSON-RPC frames.

Uses the mock camera/arm profiles (env), so this covers everything Hermes /
Claude Code exercise except physical hardware.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO, has_pinocchio

pytestmark = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")


class McpClient:
    def __init__(self, tmp_run_dir: str, extra_env: dict | None = None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src")
        env["WRC_CAMERA"] = "mock"
        env["WRC_ARM"] = "mock"
        env["WRC_RUN_DIR"] = tmp_run_dir
        env["WRC_STREAM"] = "0"  # no HTTP port binding inside tests
        env.update(extra_env or {})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "wrc_demo.apps.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self._id = 0

    def request(self, method: str, params: dict | None = None, timeout=60):
        self._id += 1
        frame = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, f"server died: {self.proc.stderr.read()[-2000:]}"
        resp = json.loads(line)
        assert resp["id"] == self._id
        return resp

    def notify(self, method: str):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


@pytest.fixture
def client(tmp_path):
    c = McpClient(str(tmp_path / "run"))
    yield c
    c.close()


def _tool_payload(resp):
    result = resp["result"]
    text = next(b["text"] for b in result["content"] if b["type"] == "text")
    return json.loads(text), result.get("isError", False)


def test_initialize_and_tools_list_without_hardware(client):
    resp = client.request("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "0"},
    })
    result = resp["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "wrc-demo"
    assert "tools" in result["capabilities"]
    client.notify("notifications/initialized")

    resp = client.request("tools/list")
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    for expected in (
        "list_objects", "grasp_object", "pick_and_place", "move_home",
        "camera_snapshot", "world_state", "live_view_url",
        "emergency_stop", "reset_stop",
    ):
        assert expected in tools, f"missing tool {expected}"
    assert "task_done" not in tools  # loop-internal, not exposed
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object"
        assert t["description"]


def test_camera_snapshot_returns_image_block(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    resp = client.request("tools/call", {"name": "camera_snapshot", "arguments": {}})
    blocks = resp["result"]["content"]
    img = next(b for b in blocks if b["type"] == "image")
    assert img["mimeType"] == "image/jpeg"
    import base64

    raw = base64.b64decode(img["data"])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic


def test_estop_blocks_motion_until_reset(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.request("tools/call", {"name": "get_observation", "arguments": {}})

    payload, _ = _tool_payload(
        client.request("tools/call", {"name": "emergency_stop", "arguments": {}})
    )
    assert payload["stopped"]
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert is_err and "e-stop" in payload["error"]
    payload, _ = _tool_payload(
        client.request("tools/call", {"name": "reset_stop", "arguments": {}})
    )
    assert payload["stopped"] is False
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "move_home", "arguments": {}})
    )
    assert not is_err, payload


def test_unknown_tool_and_method(client):
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "warp_drive", "arguments": {}})
    )
    assert is_err and "unknown skill" in payload["error"]
    resp = client.request("frobnicate/all")
    assert resp["error"]["code"] == -32601


def test_ping_and_stdout_purity(client):
    """Every stdout line must be a JSON-RPC frame (Hermes chokes otherwise)."""
    r1 = client.request("initialize", {"protocolVersion": "2024-11-05"})
    assert r1["result"]["protocolVersion"] == "2024-11-05"
    r2 = client.request("ping")
    assert r2["result"] == {}
    r3 = client.request("tools/call", {"name": "get_observation", "arguments": {}})
    assert r3["result"]
    # all three lines parsed as JSON already; nothing else was interleaved


def test_hermes_config_upsert():
    sys.path.insert(0, str(REPO / "scripts"))
    from setup_hermes import upsert, yaml_block

    block = yaml_block("l515", "rebot_rs", "/usr/bin/python3")
    # fresh file
    merged = upsert(None, block)
    assert merged.startswith("mcp_servers:")
    assert 'WRC_CAMERAS: "l515"' in merged
    # existing config with another server is preserved
    existing = (
        "model: hermes-4\n"
        "mcp_servers:\n"
        "  agenticros:\n"
        '    command: "node"\n'
        '    args: ["/x/index.js"]\n'
    )
    merged = upsert(existing, block)
    assert "model: hermes-4" in merged
    assert "agenticros:" in merged
    assert "wrc-demo:" in merged
    # idempotent replace: camera changes, no duplicate entries
    block2 = yaml_block("mock", "mock", "/usr/bin/python3")
    merged2 = upsert(merged, block2)
    assert merged2.count("wrc-demo:") == 1
    assert 'WRC_CAMERAS: "mock"' in merged2 and 'WRC_CAMERAS: "l515"' not in merged2


def test_new_livestream_tools_over_jsonrpc(client):
    """world_state / live_view_url / pick_and_place / camera_snapshot(camera=)
    were untested at the protocol level (2026-07-18 review finding)."""
    client.request("initialize", {"protocolVersion": "2025-06-18"})
    client.notify("notifications/initialized")

    tools = {t["name"] for t in client.request("tools/list")["result"]["tools"]}
    for expected in ("pick_and_place", "world_state", "live_view_url"):
        assert expected in tools, f"missing tool {expected}"

    # world_state: instant text, includes objects + cameras, arm connected
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "world_state", "arguments": {}})
    )
    assert not is_err and payload["ok"]
    assert "objects" in payload and "cameras" in payload
    # Mock arm is "connected" (no hardware to protect); the LazyArm
    # "look doesn't power the motors" invariant only applies to real
    # hardware, which the integration suite covers separately.
    assert payload["arm_connected"] is True

    # live_view_url with WRC_STREAM=0: honest error, not a bogus URL
    payload, is_err = _tool_payload(
        client.request("tools/call", {"name": "live_view_url", "arguments": {}})
    )
    assert is_err and "livestream not running" in payload["error"]

    # named-camera snapshot
    resp = client.request("tools/call", {"name": "camera_snapshot",
                                         "arguments": {"camera": "mock"}})
    blocks = resp["result"]["content"]
    assert any(b["type"] == "image" for b in blocks)

    # pick_and_place end-to-end over JSON-RPC (mock jaws close on air ->
    # honest structured failure, never a protocol error)
    payload, is_err = _tool_payload(
        client.request("tools/call",
                       {"name": "pick_and_place", "arguments": {"object": "red object"}},
                       timeout=120)
    )
    assert is_err and payload["stage"] == "grasp"
    assert "grasp failed" in payload["error"]
