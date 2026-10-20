import numpy as np

from wrc_demo.agent.llm import LLMResponse, MockLLM, OpenAICompatClient, ToolCall
from wrc_demo.agent.orchestrator import AgentOrchestrator
from wrc_demo.skills.library import SkillLibrary
from wrc_demo.skills.runtime import TOOL_SPECS


def test_tool_specs_are_valid_schemas():
    names = set()
    for spec in TOOL_SPECS:
        assert spec["name"] not in names
        names.add(spec["name"])
        assert spec["description"]
        params = spec["parameters"]
        assert params["type"] == "object"
        for req in params.get("required", []):
            assert req in params["properties"]
    # Every spec maps to a runtime method.
    from wrc_demo.skills.runtime import SkillRuntime

    for spec in TOOL_SPECS:
        assert hasattr(SkillRuntime, f"skill_{spec['name']}")


def test_mock_llm_scripting():
    llm = MockLLM([LLMResponse(text="hi"), LLMResponse(tool_calls=[ToolCall("a", {})])])
    r1 = llm.chat("sys", [{"role": "user", "content": "x"}])
    assert r1.text == "hi"
    r2 = llm.chat("sys", [])
    assert r2.tool_calls[0].name == "a"
    r3 = llm.chat("sys", [])
    assert "exhausted" in r3.text
    assert len(llm.requests) == 3


class _StubRuntime:
    """Minimal runtime for orchestrator-only behavior tests."""

    def __init__(self):
        from wrc_demo.memory import EpisodicMemory

        self.memory = EpisodicMemory()
        self.last_frame = None
        self.calls = []

        class _T:
            def finish(self, s):
                self.summary = s

        self.trace = _T()

    def execute(self, name, args):
        self.calls.append((name, args))
        if name == "task_done":
            return {"ok": True, "task_complete": True}
        return {"ok": True}

    def frame_jpeg(self):
        return None


def test_orchestrator_nudges_on_text_only_reply():
    rt = _StubRuntime()
    llm = MockLLM(
        [
            LLMResponse(text="thinking out loud, no tool"),
            LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "ok"})]),
        ]
    )
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=5).run_task("t")
    assert report.success
    # Second request must contain the nudge.
    msgs = llm.requests[1]["messages"]
    assert any("tool call" in str(m.get("content", "")) for m in msgs)


def test_orchestrator_step_budget():
    rt = _StubRuntime()
    llm = MockLLM([LLMResponse(tool_calls=[ToolCall("get_observation", {})]) for _ in range(20)])
    report = AgentOrchestrator(llm, rt, decompose=False, max_steps=3).run_task("t")
    assert not report.success
    assert report.steps == 3


def test_skill_library_roundtrip(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(
        title="Bottle grasp yaw",
        failure="jaws slip off cylindrical bottles",
        when="grasping a bottle or cylinder",
        strategy="Use the OBB major-axis yaw and two-stage close at 50%/70%.",
        origin="wrc bring-up",
    )
    entries = lib.entries()
    assert len(entries) == 1
    hits = lib.relevant("please grasp the bottle on the left")
    assert len(hits) == 1 and "two-stage" in hits[0]
    assert lib.relevant("wave hello") == []


class _ContextLimitCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if len(self.calls) == 1:
            raise RuntimeError(
                "This model's maximum context length is 4096 tokens. "
                "However, you requested 512 output tokens and your prompt "
                "contains at least 3585 input tokens"
            )
        return "ok"


def test_openai_compat_reduces_output_budget_on_context_limit():
    completions = _ContextLimitCompletions()
    client = OpenAICompatClient.__new__(OpenAICompatClient)
    client._client = type("SDK", (), {
        "chat": type("Chat", (), {"completions": completions})(),
    })()

    result = client._call_with_param_fallback({"model": "local", "max_tokens": 512})

    assert result == "ok"
    assert [call["max_tokens"] for call in completions.calls] == [512, 479]
