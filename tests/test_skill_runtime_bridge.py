"""SkillRuntime exposes the rebot_grasp bridge (or None for legacy paths).

The runtime's ``self.bridge`` is read by skill methods to decide whether
to call the rebot_grasp-aligned motion flow or fall back to SafeArm's
``move_joints``/``set_gripper`` pipeline.
"""

from unittest.mock import MagicMock

from wrc_demo.skills.runtime import SkillRuntime


def _make_runtime(arm):
    """Build a SkillRuntime with minimal mocks for non-arm dependencies."""
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


def test_runtime_bridge_is_none_for_arm_without_bridge():
    """MockArm (no grasp_bridge attr) → runtime.bridge is None."""
    arm = MagicMock(spec=["connected"])  # no grasp_bridge attr
    rt = _make_runtime(arm)
    assert rt.bridge is None


def test_runtime_bridge_is_none_when_arm_bridge_is_none():
    """Arm with explicit grasp_bridge=None → runtime.bridge is None."""
    arm = MagicMock()
    arm.grasp_bridge = None
    rt = _make_runtime(arm)
    assert rt.bridge is None


def test_runtime_bridge_is_set_when_arm_has_bridge():
    """Arm with grasp_bridge → runtime.bridge is the same object."""
    arm = MagicMock()
    bridge = MagicMock()
    arm.grasp_bridge = bridge
    rt = _make_runtime(arm)
    assert rt.bridge is bridge
