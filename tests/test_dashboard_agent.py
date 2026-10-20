"""Agent telemetry on the dashboard: LLM task decomposition + skill-call
timeline.

`AgentOrchestrator` maintains a live `self.progress` dict (reset/finalized
by the `run_task` lifecycle wrapper, fed by `_run_task_impl` /
`_try_fast_path` at each skill execution). `demo._runtime_state` exposes it
as `state["agent"]` when the entry point stashed the orchestrator on
`runtime.agent`, and the dashboard renders it in the `agent-panel`
(milestones + per-step skill records + latest LLM text).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from conftest import needs_pin

from wrc_demo.agent.llm import LLMResponse, MockLLM, ToolCall
from wrc_demo.agent.orchestrator import AgentOrchestrator


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    import math

    demo_cfg._data["grasp"]["approach_pitch_rad"] = math.pi / 2
    demo_cfg._data["grasp"]["memory_path"] = str(tmp_path / "grasp_memory.json")
    from wrc_demo.apps.demo import build_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    yield runtime, arm
    runtime.camera.close()
    arm.disconnect()


@needs_pin
def test_progress_lifecycle_through_llm_task(runtime_and_arm):
    """run_task leaves a done-state snapshot with the decomposition, the
    compacted skill timeline, and JSON-safe values; mid-run the state is
    running with the in-flight skill exposed."""
    runtime, arm = runtime_and_arm

    llm = MockLLM([
        # 1st chat = _decompose (milestones), then one tool per turn.
        LLMResponse(text="1. look at the scene\n2. finish up"),
        LLMResponse(text="thinking about the scene", tool_calls=[
            ToolCall("get_observation", {}),
        ]),
        LLMResponse(text="all good", tool_calls=[
            ToolCall("task_done", {"success": True, "summary": "looked around"}),
        ]),
    ])
    agent = AgentOrchestrator(llm, runtime, advisor=None,
                              decompose=True, max_steps=6)

    mid_run: list[dict] = []
    original_execute = runtime.execute

    def probing_execute(name, args):
        mid_run.append(agent.progress_snapshot())
        return original_execute(name, args)

    runtime.execute = probing_execute
    report = agent.run_task("look around then finish")

    assert report.success
    # Mid-run: state=running and the in-flight tool is exposed.
    assert mid_run, "execute hook should have captured snapshots"
    assert all(s["state"] == "running" for s in mid_run)
    assert mid_run[0]["current"]["tool"] == "get_observation"
    assert mid_run[0]["milestones"] == ["look at the scene", "finish up"]
    assert mid_run[0]["last_llm_text"] == "thinking about the scene"

    # Final snapshot: done, compacted steps, JSON-serializable.
    snap = agent.progress_snapshot()
    assert snap["state"] == "done"
    assert snap["task"] == "look around then finish"
    assert snap["path"] == "llm"
    assert snap["success"] is True
    assert snap["current"] is None
    tools = [s["tool"] for s in snap["steps"]]
    assert tools == ["get_observation", "task_done"]
    assert snap["steps"][0]["ok"] is True
    assert snap["steps"][0]["args"] == ""  # compacted via _short_args
    assert "t_start_monotonic" not in snap
    json.dumps(snap)  # must not raise


def test_progress_lifecycle_without_pinocchio():
    """The lifecycle wrapper works on a stub runtime too (no pinocchio):
    reset -> running -> done even when every call is a mock."""
    runtime = MagicMock()
    runtime.arm.harness.estopped = False  # MagicMock attr is truthy otherwise
    runtime.execute.side_effect = lambda name, args: (
        {"task_complete": True, "ok": True}
        if name == "task_done" else {"ok": True}
    )
    llm = MockLLM([
        LLMResponse(text="1. do the thing"),
        LLMResponse(text="", tool_calls=[ToolCall("task_done", {
            "success": True, "summary": "stub done",
        })]),
    ])
    agent = AgentOrchestrator(llm, runtime, advisor=None,
                              decompose=True, max_steps=3)
    snap_idle = agent.progress_snapshot()
    assert snap_idle["state"] == "idle"

    report = agent.run_task("stub task")
    assert report.success
    snap = agent.progress_snapshot()
    assert snap["state"] == "done"
    assert snap["milestones"] == ["do the thing"]
    assert snap["steps"][-1]["tool"] == "task_done"
    assert snap["steps"][-1]["ok"] is True


def test_runtime_state_exposes_agent():
    """demo._runtime_state includes the agent snapshot when the entry point
    stashed the orchestrator on runtime.agent; omits it otherwise."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wrc_demo.apps.demo import _runtime_state

    class _Agent:
        def progress_snapshot(self):
            return {"state": "idle", "task": None, "steps": []}

    camera = SimpleNamespace()  # no overlay/latest attrs -> branches skipped
    base = SimpleNamespace(
        camera=camera, current_task=None, held_object=None,
        arm=SimpleNamespace(),  # getattr(arm, "connected", True)
        beliefs=MagicMock(summary=lambda: []),
        memory=MagicMock(display_digest=lambda max_lines=40: ""),
        grasp_debug_snapshot=lambda: {}, watcher=None,
    )

    state = _runtime_state(base)
    assert "agent" not in state

    base.agent = _Agent()
    state = _runtime_state(base)
    assert state["agent"] == {"state": "idle", "task": None, "steps": []}


def test_dashboard_template_renders_agent_panel():
    """The agent panel markers survive .format(tiles=...) — mirrors the
    grasp-debug template pins in tests/test_grasp_debug.py."""
    from wrc_demo.apps.stream_server import _INDEX_HTML

    body = _INDEX_HTML.format(tiles='<div class="cam"></div>')
    for marker in (
        'id="agent-panel"', 'id="ag-head"', 'id="ag-milestones"',
        'id="ag-steps"', 'id="ag-llm"', "renderAgent(s)",
    ):
        assert marker in body, f"missing dashboard marker {marker!r}"
    # The panel must stay hidden when the backend has no agent wired.
    assert "s.agent" in body
