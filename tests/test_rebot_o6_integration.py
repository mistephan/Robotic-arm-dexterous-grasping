from pathlib import Path
import xml.etree.ElementTree as ET

from wrc_demo.agent.reflex import parse_command
from wrc_demo.config import load_demo_config
import numpy as np

from wrc_demo.safety.harness import Kinematics

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/urdf/rebot_b601_linkerhand_o6/rebot_b601_linkerhand_o6.urdf"


def test_o6_profile_uses_current_fused_model_and_no_rs_gripper():
    cfg = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    assert cfg.arm.ee_frame == "o6_grasp_tcp"
    assert cfg.arm.end_effector.type == "linkerhand_o6"
    directions = cfg.arm.end_effector.grasp_directions
    assert directions.strict_grasp_topdown is True
    assert directions.elevated_tilts_from_vertical_rad == [0.0]
    assert directions.low_tilts_from_vertical_rad == [0.0]
    assert Path(cfg.arm.urdf) == URDF
    hw = (ROOT / "sdk/config/rebotarm_rs_o6.yaml").read_text()
    assert "joints: [gripper]" not in hw
    assert "name: gripper" not in hw


def test_o6_urdf_is_valid_and_kinematics_uses_six_arm_joints():
    root = ET.parse(URDF).getroot()
    links = {link.get("name") for link in root.findall("link")}
    assert {"adapter", "hand_base_link", "o6_palm_center", "o6_grasp_tcp", "gripper_end"} <= links
    kin = Kinematics(str(URDF), "o6_grasp_tcp", 6)
    assert kin.n_joints == 6
    assert kin._model.nq > 6  # hand joints exist but are zero-padded outside arm IK
    assert kin.fk(np.zeros(6)).shape == (4, 4)

    joint = root.find("./joint[@name='hand_base_to_o6_grasp_tcp']/origin")
    assert joint is not None
    assert np.allclose(
        [float(v) for v in joint.get("xyz").split()],
        [0.0600, 0.0275, 0.1187],
    )


def test_o6_home_keeps_base_and_wrist_neutral():
    cfg = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    q = np.asarray(cfg.arm.home_q, dtype=float)
    assert np.allclose(q[[0, 4, 5]], 0.0)
    ready_q = np.asarray(cfg.arm.ready_q, dtype=float)
    assert np.allclose(ready_q, q)
    assert np.allclose(ready_q[[0, 4, 5]], 0.0)
    kin = Kinematics(str(URDF), "o6_grasp_tcp", 6)
    position = kin.fk(q)[:3, 3]
    assert np.all(np.isfinite(position))


def test_o6_graspgenx_is_disabled_until_hand_geometry_is_calibrated():
    from wrc_demo.skills.runtime import SkillRuntime
    cfg = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    cfg.arm.end_effector.graspgenx._data["calibrated"] = False
    rt = object.__new__(SkillRuntime)
    rt.cfg = cfg
    effective, reason = rt._effective_graspgenx_config()
    assert effective is cfg.grasp or effective.as_dict() == cfg.grasp.as_dict()
    assert "not calibrated" in reason


def test_parallel_and_o6_profiles_use_isolated_graspgenx_geometry():
    parallel = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    o6 = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    parallel_rt = object.__new__(__import__(
        "wrc_demo.skills.runtime", fromlist=["SkillRuntime"]
    ).SkillRuntime)
    parallel_rt.cfg = parallel
    o6_rt = object.__new__(__import__(
        "wrc_demo.skills.runtime", fromlist=["SkillRuntime"]
    ).SkillRuntime)
    o6_rt.cfg = o6
    parallel_cfg, _ = parallel_rt._effective_graspgenx_config()
    o6_cfg, _ = o6_rt._effective_graspgenx_config()
    assert parallel_cfg.graspgenx.sweep.gripper_type == 0
    assert parallel_cfg.graspgenx.tip_offset_m == 0.035
    # gripper_type 2 = revolute_3f (server map: 0=parallel_2f, 1=revolute_2f,
    # 2=revolute_3f). The source TCP is recorded in hand_base_link and the
    # planner converts it into the explicit custom grasp frame.
    assert o6_cfg.graspgenx.sweep.gripper_type == 2
    assert o6_cfg.graspgenx.tcp_offset_xyz_m == [0.0600, 0.0275, 0.1187]
    assert o6_cfg.graspgenx.tcp_offset_frame == "hand_base_link"
    assert o6.arm.end_effector.tcp_standoff_m == 0.0
    assert o6_cfg.graspgenx.tcp_standoff_m == 0.0
    assert o6.arm.end_effector.open_pose[1] == 87
    assert o6.arm.end_effector.power_target[1] == 87
    assert o6.arm.end_effector.pinch_target[1] == 87

    from wrc_demo.grasping.graspgenx_backend import GraspGenXPlanner
    planner = GraspGenXPlanner(o6_cfg)
    # TCP position is independent of the three-finger approach orientation.
    np.testing.assert_allclose(
        planner.tcp_offset_xyz_m,
        [0.04499188, 0.0, 0.12814756],
        atol=5e-5,
    )
    np.testing.assert_allclose(
        planner.tcp_rotation_matrix[:, 0], [0.0, 0.0, 1.0]
    )


def test_o6_graspgenx_uses_end_effector_calibration_overrides():
    from wrc_demo.skills.runtime import SkillRuntime
    cfg = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    original_tip_offset = cfg.grasp.graspgenx.tip_offset_m
    cfg.arm.end_effector.graspgenx._data.update({
        "calibrated": True,
        "tip_offset_m": 0.081,
        "sweep_width_m": 0.095,
    })
    rt = object.__new__(SkillRuntime)
    rt.cfg = cfg
    effective, reason = rt._effective_graspgenx_config()
    assert reason is None
    assert effective.graspgenx.tip_offset_m == 0.081
    assert effective.graspgenx.sweep_width_m == 0.095
    assert cfg.grasp.graspgenx.tip_offset_m == original_tip_offset


def test_english_hand_actions_compile_to_skills():
    expected = {
        "open the hand": "open",
        "close the hand": "close",
        "pinch": "pinch",
    }
    for text, action in expected.items():
        plan = parse_command(text)
        assert plan.calls == [("hand_action", {"action": action})]


def test_chinese_hand_actions_compile_to_skills():
    expected = {
        "张开灵巧手": "open",
        "五指收缩": "close",
        "食指捏合": "pinch",
    }
    for text, action in expected.items():
        plan = parse_command(text)
        assert plan.calls == [("hand_action", {"action": action})]
