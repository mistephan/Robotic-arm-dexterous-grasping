"""End-to-end dry run: mock camera + mock detector + mock arm + scripted LLM.

Exercises the REAL wiring (config loading, perception -> beliefs -> grasp
planning -> IK -> safety-gated motion -> gripper verification -> memory ->
trace files) with only the LLM decisions scripted.
"""

import json

import numpy as np
import pytest

from conftest import needs_pin

from wrc_demo.agent.llm import LLMResponse, MockLLM, ToolCall
from wrc_demo.agent.orchestrator import AgentOrchestrator


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    # Mock arm's home_q is in a different basin than the rebot_rs home_q,
    # so the production `approach_pitch_rad: 1.2` (tilted) pregrasp is
    # unreachable from the mock seed. Pin legacy top-down for the e2e
    # tests so the wiring (perception -> beliefs -> grasp -> IK -> safety
    # -> motion -> memory -> trace) is exercised against the geometry
    # the test was originally written for; the new pitch=1.2 path is
    # covered separately by the real-hardware demo and the live IK check.
    #
    # Also force the grasp-memory path to tmp_path: a stale
    # ``~/.wrc_demo/grasp_memory.json`` from prior sessions can apply a
    # learned z_nudge that pushes the pregrasp into a reach dead-zone
    # (per CLAUDE.md "Persistent learned state outside the run dir").
    import math
    demo_cfg._data["grasp"]["approach_pitch_rad"] = math.pi / 2
    demo_cfg._data["grasp"]["memory_path"] = str(tmp_path / "grasp_memory.json")
    from wrc_demo.apps.demo import build_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    yield runtime, arm
    runtime.camera.close()
    arm.disconnect()


def _script(*calls: ToolCall) -> list[LLMResponse]:
    return [LLMResponse(text="", tool_calls=[c]) for c in calls]


@needs_pin
def test_grasp_and_place_happy_path(runtime_and_arm, tmp_path):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5  # jaws jam halfway: an object is in the gripper

    llm = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("grasp_object", {"label": "red cube", "material": "rigid"}),
            ToolCall("place_at", {"x": 0.20, "y": -0.15}),
            ToolCall("task_done", {"success": True, "summary": "moved the cube"}),
        )
    )
    agent = AgentOrchestrator(llm, runtime, advisor=None, decompose=False, max_steps=10)
    report = agent.run_task("move the red cube to the front-left of the table")

    assert report.success
    assert report.steps == 4
    results = {e["tool"]: e["result"] for e in report.tool_log}
    assert results["get_observation"]["ok"]
    assert any(o["label"] == "red cube" for o in results["get_observation"]["objects_visible"])
    assert results["grasp_object"]["ok"], results["grasp_object"]
    assert results["grasp_object"]["held"] == "red cube"
    assert results["place_at"]["ok"], results["place_at"]
    assert runtime.held_object is None

    # Belief followed the object to the place target.
    b = runtime.beliefs.find("red cube")
    assert b is not None
    assert np.allclose(b.position[:2], [0.20, -0.15], atol=0.02)

    # Trace artifacts exist and parse.
    trace_file = runtime.trace.run_dir / "trace.jsonl"
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    # Non-skill records (place_height_plan, held_level_capture,
    # place_orientation_search, ...) have no "skill" key; filter them out.
    assert [r["skill"] for r in records if "skill" in r] == [
        "get_observation", "grasp_object", "place_at", "task_done",
    ]
    assert (runtime.trace.run_dir / "summary.txt").exists()
    assert list((runtime.trace.run_dir / "keyframes").glob("*.jpg"))

    # Memory recorded the story within the horizon.
    digest = runtime.memory.digest()
    assert "grasp" in digest and "place" in digest


@needs_pin
def test_air_grasp_detected_and_reported(runtime_and_arm):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = None  # nothing stops the jaws: air grasp

    llm = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("grasp_object", {"label": "red cube"}),
            ToolCall("task_done", {"success": False, "summary": "grasp kept missing"}),
        )
    )
    agent = AgentOrchestrator(llm, runtime, advisor=None, decompose=False, max_steps=10)
    report = agent.run_task("pick up the red cube")

    assert not report.success
    grasp_result = next(e["result"] for e in report.tool_log if e["tool"] == "grasp_object")
    assert not grasp_result["ok"]
    assert "air grasp" in grasp_result["error"]
    assert runtime.held_object is None


@needs_pin
def test_advisor_consulted_after_failure(runtime_and_arm):
    runtime, arm = runtime_and_arm
    from wrc_demo.agent.advisor import Advisor

    # Agent LLM: observe -> impossible grasp -> finish.
    agent_llm = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("grasp_object", {"label": "unicorn"}),
            ToolCall("task_done", {"success": False, "summary": "no unicorn found"}),
        )
    )
    advisor_llm = MockLLM([LLMResponse(text="Try approaching from the left side.")])
    agent = AgentOrchestrator(
        agent_llm, runtime, advisor=Advisor(advisor_llm), decompose=False, max_steps=10
    )
    report = agent.run_task("pick up the unicorn")
    assert not report.success
    # Advisor was consulted exactly once and its suggestion reached the agent.
    assert len(advisor_llm.requests) == 1
    joined = str(agent_llm.requests[-1]["messages"])
    assert "Try approaching from the left side." in joined


@needs_pin
def test_memory_survives_between_tasks(runtime_and_arm):
    runtime, arm = runtime_and_arm
    arm.object_stop_frac = 0.5
    llm1 = MockLLM(
        _script(
            ToolCall("get_observation", {}),
            ToolCall("task_done", {"success": True, "summary": "observed"}),
        )
    )
    AgentOrchestrator(llm1, runtime, decompose=False).run_task("look around")
    # Second task: the agent recalls the cube's position without re-observing.
    llm2 = MockLLM(
        _script(
            ToolCall("recall_memory", {"query": "red cube"}),
            ToolCall("task_done", {"success": True, "summary": "recalled"}),
        )
    )
    report = AgentOrchestrator(llm2, runtime, decompose=False).run_task("where was the cube?")
    recall = next(e["result"] for e in report.tool_log if e["tool"] == "recall_memory")
    assert recall["ok"]
    assert "object_memory" in recall
    assert abs(recall["object_memory"]["last_position"][0] - 0.29) < 0.03
