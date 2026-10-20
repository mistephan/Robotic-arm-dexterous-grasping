"""teach.py 业务层单元测试 (v2 主从臂遥操)。

覆盖 spec §6.3 测试用例: save/load/checksum/filter/interpolate/transition/master_to_slave。
跑：pytest tests/test_teach_logic.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

from wrc_demo.skills import teach as teach_mod
from wrc_demo.skills.master_arm import master_to_slave


# ── Fixture: 合成 frames (v2 schema) ──────────────────────────────────────
@pytest.fixture
def synthetic_frames():
    """30 个合成 frame 模拟 1 秒录制 (30Hz)。"""
    frames = []
    for i in range(30):
        t = i / 30.0
        m_rad = np.deg2rad(np.array([30 * t, 20 * t, -15 * t, 10 * t, 5 * t, 0.0, 0.0]))
        s_rad = np.deg2rad(np.array([-30 * t, 20 * t, -15 * t, -10 * t, 5 * t, 0.0, 0.0]))
        frames.append({
            "t": round(t, 3),
            "master_q": [round(float(x), 6) for x in m_rad],
            "slave_q": [round(float(x), 6) for x in s_rad],
        })
    return frames


@pytest.fixture
def tmp_recordings_dir(tmp_path):
    return tmp_path / "recordings"


# ── §6.3.1 save_load_slot_roundtrip ───────────────────────────────────────
def test_save_load_slot_roundtrip(tmp_recordings_dir, synthetic_frames):
    """保存后读取数据完全一致。"""
    start_master = np.zeros(7)
    start_slave = np.zeros(7)
    path = teach_mod.save_slot(1, synthetic_frames, start_master, start_slave, tmp_recordings_dir)

    assert path.exists()
    assert path.name == "slot1.json"

    data = teach_mod.load_slot(1, tmp_recordings_dir)
    assert data["schema_version"] == teach_mod.SCHEMA_VERSION == 2
    assert data["slot"] == 1
    assert data["frame_count"] == len(synthetic_frames)
    assert data["sample_rate_hz"] == 30
    assert len(data["frames"]) == len(synthetic_frames)
    # 帧内容
    for orig, loaded in zip(synthetic_frames, data["frames"]):
        assert orig["t"] == loaded["t"]
        assert orig["master_q"] == loaded["master_q"]
        assert orig["slave_q"] == loaded["slave_q"]


# ── §6.3.2 checksum_validation ──────────────────────────────────────────
def test_checksum_validation(tmp_recordings_dir, synthetic_frames):
    """数据被破坏 → checksum 不匹配 → load_slot 抛 ValueError。"""
    teach_mod.save_slot(1, synthetic_frames, np.zeros(7), np.zeros(7), tmp_recordings_dir)
    path = tmp_recordings_dir / "slot1.json"

    # 篡改 frames[0].slave_q[0]
    raw = json.loads(path.read_text())
    raw["frames"][0]["slave_q"][0] = 99.0
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="checksum"):
        teach_mod.load_slot(1, tmp_recordings_dir)


def test_checksum_computation_deterministic(synthetic_frames):
    """compute_checksum 对相同 frames 总是相同值。"""
    c1 = teach_mod.compute_checksum(synthetic_frames)
    c2 = teach_mod.compute_checksum(synthetic_frames)
    assert c1 == c2
    assert len(c1) == 64  # SHA256 hex


def test_load_slot_wrong_schema_version(tmp_recordings_dir, synthetic_frames):
    """schema_version != 2 → ValueError。"""
    teach_mod.save_slot(1, synthetic_frames, np.zeros(7), np.zeros(7), tmp_recordings_dir)
    path = tmp_recordings_dir / "slot1.json"

    # 改 schema_version 为 1
    raw = json.loads(path.read_text())
    raw["schema_version"] = 1
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="schema_version"):
        teach_mod.load_slot(1, tmp_recordings_dir)


# ── §6.3.4 frames_have_master_q_and_slave_q ─────────────────────────────
def test_frames_have_master_q_and_slave_q(tmp_recordings_dir, synthetic_frames):
    """每帧含 master_q (7 floats) + slave_q (7 floats)。"""
    teach_mod.save_slot(1, synthetic_frames, np.zeros(7), np.zeros(7), tmp_recordings_dir)
    data = teach_mod.load_slot(1, tmp_recordings_dir)
    for frame in data["frames"]:
        assert "master_q" in frame
        assert "slave_q" in frame
        assert len(frame["master_q"]) == 7
        assert len(frame["slave_q"]) == 7


def test_frames_no_legacy_joints_field(tmp_recordings_dir, synthetic_frames):
    """v2 不含 v1 单存的 joints 字段。"""
    teach_mod.save_slot(1, synthetic_frames, np.zeros(7), np.zeros(7), tmp_recordings_dir)
    data = teach_mod.load_slot(1, tmp_recordings_dir)
    for frame in data["frames"]:
        assert "joints" not in frame  # v1 单存 joints 已废弃


# ── §6.3.8 record_filter_alpha ──────────────────────────────────────────
def test_record_filter_alpha_first_frame():
    """filter_joint_states 第一次调用直接返回 new_js。"""
    new = np.array([0.1, 0.2, 0.3])
    result = teach_mod.filter_joint_states(new, last_js=None, alpha=0.35)
    assert np.allclose(result, new)


def test_record_filter_alpha_smoothing():
    """filter_joint_states 用 alpha=0.35 平滑 (last + 0.35*(new-last))。"""
    last = np.array([0.0, 0.0, 0.0])
    new = np.array([1.0, 1.0, 1.0])
    alpha = 0.35
    expected = last + alpha * (new - last)
    result = teach_mod.filter_joint_states(new, last, alpha)
    assert np.allclose(result, expected)


# ── §6.3.9 record_joint_change_threshold ─────────────────────────────────
def test_joint_states_changed_enough_first_call():
    """first call (None) → True（强制记第一帧）。"""
    assert teach_mod.joint_states_changed_enough(None, np.zeros(6)) is True
    assert teach_mod.joint_states_changed_enough(np.zeros(6), None) is True


def test_joint_states_changed_above_threshold():
    """关节变化 > 0.003 rad → True。"""
    a = np.zeros(6)
    b = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.004 > 0.003
    assert teach_mod.joint_states_changed_enough(a, b, threshold=0.003) is True


def test_joint_states_changed_below_threshold():
    """关节变化 < 0.003 rad → False（demo 阈值）。"""
    a = np.zeros(6)
    b = np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.002 < 0.003
    assert teach_mod.joint_states_changed_enough(a, b, threshold=0.003) is False


# ── §6.3.10 interpolate_smoothstep ──────────────────────────────────────
def test_interpolate_smoothstep_midpoint():
    """t 在两帧中间 → smoothstep (3α²-2α³) 插值，α=0.5 → 0.5 (linear 同)。"""
    f0 = {"t": 0.0, "slave_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    f1 = {"t": 1.0, "slave_q": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    mid = teach_mod.interpolate_joint_states(f0, f1, 0.5)
    # smoothstep(0.5) = 0.5
    assert np.allclose(mid, [0.5] * 7)


def test_interpolate_smoothstep_smooth_curve():
    """smoothstep 比 linear 在中间更"平"——验证曲线。"""
    f0 = {"t": 0.0, "slave_q": [0.0] * 7}
    f1 = {"t": 1.0, "slave_q": [1.0] * 7}
    # smoothstep 在 t=0.25 应略低于 linear (0.25)，因为曲线 S 形
    t = 0.25
    smooth = teach_mod.interpolate_joint_states(f0, f1, t)[0]
    # smoothstep(0.25) = 3*0.0625 - 2*0.015625 = 0.15625
    assert smooth == pytest.approx(0.15625, abs=0.001)


def test_interpolate_same_time():
    """t1 == t0 → 返回 sq0 副本。"""
    f0 = {"t": 0.5, "slave_q": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]}
    f1 = {"t": 0.5, "slave_q": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    result = teach_mod.interpolate_joint_states(f0, f1, 0.5)
    assert np.allclose(result, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])


# ── §6.3.11 transition_blend ────────────────────────────────────────────
def test_blend_smoothstep():
    """blend_joint_states 用 smoothstep。"""
    a = np.zeros(7)
    b = np.ones(7)
    # alpha=0.5 → 0.5
    assert np.allclose(teach_mod.blend_joint_states(a, b, 0.5), 0.5)


def test_blend_alpha_clamped():
    """alpha 超出 [0,1] 被 clamp 到 [0,1]。"""
    a = np.zeros(7)
    b = np.ones(7)
    # alpha=-0.5 → 0
    assert np.allclose(teach_mod.blend_joint_states(a, b, -0.5), 0.0)
    # alpha=1.5 → 1
    assert np.allclose(teach_mod.blend_joint_states(a, b, 1.5), 1.0)


# ── Constants ─────────────────────────────────────────────────────────────
def test_num_slots_constant():
    """NUM_SLOTS = 5。"""
    assert teach_mod.NUM_SLOTS == 5


def test_sample_hz_constant():
    """SAMPLE_HZ = 30（demo 同款）。"""
    assert teach_mod.SAMPLE_HZ == 30.0


def test_schema_version_constant():
    """SCHEMA_VERSION = 2。"""
    assert teach_mod.SCHEMA_VERSION == 2


def test_recording_constants():
    """录制相关常量 (demo 同款)。"""
    assert teach_mod.RECORD_FILTER_ALPHA == 0.35
    assert teach_mod.MIN_RECORD_INTERVAL == 0.01
    assert teach_mod.MIN_JOINT_CHANGE == 0.003
    assert teach_mod.END_HOLD_TIME == 0.15
    assert teach_mod.LOOP_BLEND_TIME == 0.30
    assert teach_mod.TRANSITION_TIME == 0.6


# ── load_slot missing ────────────────────────────────────────────────────
def test_load_slot_missing_file(tmp_recordings_dir):
    """load_slot 槽位不存在 → FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        teach_mod.load_slot(1, tmp_recordings_dir)


# ── save_slot creates dir ────────────────────────────────────────────────
def test_save_slot_creates_directory(tmp_path, synthetic_frames):
    """save_slot 自动创建 recordings 目录。"""
    target_dir = tmp_path / "new_dir" / "recordings"
    assert not target_dir.exists()
    path = teach_mod.save_slot(1, synthetic_frames, np.zeros(7), np.zeros(7), target_dir)
    assert target_dir.exists()
    assert path.exists()


# ── KeyboardState ─────────────────────────────────────────────────────────
def test_keyboard_state_reset():
    """KeyboardState.reset() 重置所有标志位。"""
    teach_mod.KeyboardState.ENTER = True
    teach_mod.KeyboardState.ESC = True
    teach_mod.KeyboardState.reset()
    assert teach_mod.KeyboardState.ENTER is False
    assert teach_mod.KeyboardState.ESC is False


def test_keyboard_state_slot_keys():
    """SLOT_KEYS 含 q/w/e/r/t (录制) + 1/2/3/4/5 (播放)。"""
    assert "q" in teach_mod.KeyboardState.SLOT_KEYS
    assert "w" in teach_mod.KeyboardState.SLOT_KEYS
    assert "e" in teach_mod.KeyboardState.SLOT_KEYS
    assert "r" in teach_mod.KeyboardState.SLOT_KEYS
    assert "t" in teach_mod.KeyboardState.SLOT_KEYS
    assert "1" in teach_mod.KeyboardState.SLOT_KEYS
    assert "5" in teach_mod.KeyboardState.SLOT_KEYS
    # q=0 (slot 1), t=4 (slot 5), 1=0, 5=4
    assert teach_mod.KeyboardState.SLOT_KEYS["q"] == 0
    assert teach_mod.KeyboardState.SLOT_KEYS["t"] == 4
    assert teach_mod.KeyboardState.SLOT_KEYS["1"] == 0
    assert teach_mod.KeyboardState.SLOT_KEYS["5"] == 4


# ── Dry Run ──────────────────────────────────────────────────────────────
def test_dry_run_record_replay_roundtrip(tmp_recordings_dir):
    """dry_run_record + dry_run_replay 完整往返。"""
    r1 = teach_mod.dry_run_record(1, tmp_recordings_dir)
    assert r1["ok"] is True
    assert r1["frame_count"] == 30
    assert r1["duration_s"] == pytest.approx(0.967, abs=0.01)
    assert r1["path"].endswith("slot1.json")

    r2 = teach_mod.dry_run_replay(1, tmp_recordings_dir)
    assert r2["ok"] is True
    assert r2["frame_count"] == 30


def test_dry_run_replay_missing_slot(tmp_recordings_dir):
    """dry_run_replay 槽位不存在 → ok=False。"""
    r = teach_mod.dry_run_replay(99, tmp_recordings_dir)
    assert r["ok"] is False
    assert "not exist" in r["error"] or "FileNotFound" in r["error"]


# ── master_to_slave 集成 ─────────────────────────────────────────────────
def test_master_to_slave_roundtrip_deg_to_rad():
    """master_to_slave 输入度, 输出弧度。"""
    r = master_to_slave({
        "joint1": 90, "joint2": 0, "joint3": 0,
        "joint4": 0, "joint5": 0, "joint6": 0, "gripper": 0,
    })
    assert r.shape == (7,)
    # joint1 取反 → -90° = -π/2 rad
    assert r[0] == pytest.approx(-np.pi / 2, abs=0.01)
    # 其他为 0
    assert np.allclose(r[1:6], 0.0)
    # gripper 为 0
    assert r[6] == 0.0