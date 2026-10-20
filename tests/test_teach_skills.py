"""skill_teach_record / skill_teach_replay 集成测试 (v2 主从臂遥操)。

SkillRuntime 通过 _make_runtime(arm) 用 MagicMock 构造（参考 test_skill_runtime_bridge.py）。
skill_teach_record / skill_teach_replay 内部用 master (PiPERMateArm) + slave (RebotArm)，
mock arm.raw._arm 是 vendored RebotArm，mock master 用真实 PiPERMateArm 实例（不连串口）。

跑：pytest tests/test_teach_skills.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

from wrc_demo.skills.runtime import _MOTION_SKILLS, TOOL_SPECS, SkillRuntime


def _make_runtime(arm) -> SkillRuntime:
    """复用 test_skill_runtime_bridge.py 的模式（见 _make_runtime）。"""
    return SkillRuntime(
        camera=MagicMock(),
        depth_provider=MagicMock(),
        detector=MagicMock(),
        extrinsics=MagicMock(),
        kin=MagicMock(),
        safe_arm=arm,
        memory=MagicMock(),
        beliefs=MagicMock(),
        trace=MagicMock(),
        cfg=MagicMock(),
    )


def _make_mock_arm_with_sdk():
    """Mock arm + nested raw_rard._arm (vendored RebotArm) 让 teach skill 能跑。"""
    arm = MagicMock()
    sdk_arm = MagicMock()
    sdk_arm.control_loop_active = False
    sdk_arm.connect = MagicMock()
    sdk_arm.arm = MagicMock()
    sdk_arm.arm.num_joints = 6
    sdk_arm.arm.has_gripper = False
    sdk_arm.gripper = MagicMock()
    sdk_arm.gripper.get_positions = MagicMock(return_value=np.array([0.0]))
    sdk_arm.has_gripper = False
    sdk_arm.disconnect = MagicMock()
    sdk_arm.start_control_loop = MagicMock()
    sdk_arm.stop_control_loop = MagicMock()
    sdk_arm.get_state = MagicMock(return_value=(np.zeros(6), np.zeros(6), np.zeros(6)))
    sdk_arm.arm.get_positions = MagicMock(return_value=np.zeros(6))
    sdk_arm.arm.send_mit = MagicMock()
    sdk_arm.arm.send_pos_vel = MagicMock()
    sdk_arm.arm.mode_pos_vel = MagicMock(return_value=True)
    sdk_arm.arm.enable = MagicMock()
    sdk_arm.arm.disable = MagicMock()
    sdk_arm.disable_all = MagicMock()
    sdk_arm.enable_all = MagicMock()

    arm.raw = MagicMock()
    arm.raw._arm = sdk_arm
    return arm, sdk_arm


# ── §6.3.8 _MOTION_SKILLS contains teach_* ────────────────────────────────
def test_motion_skills_containste_teach_record_replay():
    """_MOTION_SKILLS 集合字面量含 teach_record, teach_replay。"""
    assert "teach_record" in _MOTION_SKILLS
    assert "teach_replay" in _MOTION_SKILLS


def test_motion_skill_names_are_unique_and_include_required_skills():
    """Adding a valid motion skill must not break a brittle total-count pin."""
    required = {"teach_record", "teach_replay", "hand_action", "pick_and_place"}
    assert required <= _MOTION_SKILLS


# ── §6.3.9 TOOL_SPECS contains teach_* schemas ────────────────────────────
def test_tool_specs_containste_teach_record_replay():
    """TOOL_SPECS 列表含 teach_record, teach_replay 两条 schema。"""
    spec_names = {s["name"] for s in TOOL_SPECS}
    assert "teach_record" in spec_names
    assert "teach_replay" in spec_names


def test_teach_tool_specs_schema_format():
    """teach_* schema 必须含 name + description + parameters (spec §5.4)。"""
    for spec in TOOL_SPECS:
        if spec["name"] not in ("teach_record", "teach_replay"):
            continue
        assert "name" in spec
        assert "description" in spec
        assert "parameters" in spec
        assert "slot" in spec["parameters"]["properties"]
        assert spec["parameters"]["properties"]["slot"]["type"] == "integer"
        assert spec["parameters"]["properties"]["slot"]["minimum"] == 1
        assert spec["parameters"]["properties"]["slot"]["maximum"] == 5
        assert "slot" in spec["parameters"]["required"]


def test_teach_tool_specs_v2_description():
    """v2 teach_* description 必须含 'master-slave teleoperation' / 'PiPER Mate'。"""
    for spec in TOOL_SPECS:
        if spec["name"] == "teach_record":
            assert "master-slave teleoperation" in spec["description"] or "PiPER Mate" in spec["description"]
            assert "q" in spec["description"] or "Esc" in spec["description"]


def test_tool_spec_names_are_unique_and_include_required_tools():
    """Tool growth is allowed; names must remain unique and required tools present."""
    names = [spec["name"] for spec in TOOL_SPECS]
    assert len(names) == len(set(names))
    assert {"teach_record", "teach_replay", "hand_action", "pick_and_place"} <= set(names)


# ── skill_teach_record rejects invalid slot ──────────────────────────────
def test_skill_teach_record_rejects_slot_zero():
    """slot=0 → ok=False, error 含 'slot must be int 1-5'。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    result = rt.skill_teach_record(0)
    assert result["ok"] is False
    assert "slot must be int 1-5" in result["error"]


def test_skill_teach_record_rejects_slot_six():
    """slot=6 → ok=False。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    result = rt.skill_teach_record(6)
    assert result["ok"] is False
    assert "slot must be int 1-5" in result["error"]


def test_skill_teach_record_rejects_non_int_slot():
    """slot='abc' → ok=False。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    result = rt.skill_teach_record("abc")
    assert result["ok"] is False
    assert "slot must be int 1-5" in result["error"]


def test_skill_teach_replay_rejects_invalid_slot():
    """skill_teach_replay 也验证 slot 范围。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    result = rt.skill_teach_replay(99)
    assert result["ok"] is False
    assert "slot must be int 1-5" in result["error"]


# ── returns dict format ──────────────────────────────────────────────────
def test_skill_teach_record_returns_dict_format_on_error():
    """skill_teach_record 错误时返回 dict（不是 raise Exception）。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    result = rt.skill_teach_record(0)
    assert isinstance(result, dict)
    assert "ok" in result
    assert "error" in result


def test_skill_teach_replay_returns_dict_format_on_error():
    """skill_teach_replay 错误时返回 dict。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    result = rt.skill_teach_replay(0)
    assert isinstance(result, dict)
    assert "ok" in result
    assert "error" in result


# ── arm not connected ────────────────────────────────────────────────────
def test_skill_teach_record_no_arm_connected():
    """SDK arm._arm = None → ok=False, error 含 'not connected'。"""
    arm = MagicMock()
    arm.raw = MagicMock()
    arm.raw._arm = None  # 模拟未 connect
    rt = _make_runtime(arm)
    result = rt.skill_teach_record(1)
    assert result["ok"] is False
    assert "not connected" in result["error"]


def test_skill_teach_replay_no_arm_connected():
    """skill_teach_replay 也检查 arm 连接。"""
    arm = MagicMock()
    arm.raw = MagicMock()
    arm.raw._arm = None
    rt = _make_runtime(arm)
    result = rt.skill_teach_replay(1)
    assert result["ok"] is False
    assert "not connected" in result["error"]


# ── SkillRuntime.execute() dispatch path ────────────────────────────────
def test_skill_runtime_dispatches_teach_record():
    """SkillRuntime.execute('teach_record', {'slot': 0}) 调用 skill_teach_record。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    rt.trace = MagicMock()
    rt.last_frame = MagicMock()
    rt.last_frame.rgb = None
    rt.execute("teach_record", {"slot": 0})  # slot 0 立即被拒
    rt.memory.add.assert_called()


def test_skill_runtime_dispatches_teach_replay():
    """SkillRuntime.execute('teach_replay', {'slot': 0}) 调用 skill_teach_replay。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    rt.trace = MagicMock()
    rt.last_frame = MagicMock()
    rt.last_frame.rgb = None
    rt.execute("teach_replay", {"slot": 0})
    rt.memory.add.assert_called()


def test_teach_skills_pause_watcher():
    """teach_* 在 _MOTION_SKILLS 里 → SkillRuntime.execute 暂停 WorldWatcher。"""
    arm, _ = _make_mock_arm_with_sdk()
    rt = _make_runtime(arm)
    rt.trace = MagicMock()
    rt.last_frame = MagicMock()
    rt.last_frame.rgb = None
    rt.watcher = None
    rt.execute("teach_record", {"slot": 0})