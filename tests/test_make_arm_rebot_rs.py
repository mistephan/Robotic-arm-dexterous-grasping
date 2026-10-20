"""Tests for make_arm rebot_rs wiring the RebotGraspBridge.

The rebot_rs backend must (a) instantiate ``WrcGripper`` (wrc_demo's
native port of rebot_grasp's GraspDriver — see ADR-0005) so the bridge
has a real gripper handle, (b) start the SDK control loop once
(WrcGripper.start() cooperates with the loop via ``_loop_cb``
wrapper; calling ``sdk_ctrl.start()`` afterwards would raise
RuntimeError on the second ``start_control_loop``), (c) attach a
``RebotGraspBridge`` to the returned ``SafeArm`` so skill runtime can
read ``arm.grasp_bridge``.

These tests monkeypatch the heavy SDK imports so they run without
real hardware.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from wrc_demo.config import load_demo_config


def test_rebot_rs_arm_has_grasp_bridge_attribute():
    """make_arm for rebot_rs returns an arm with non-None ``grasp_bridge``."""
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")

    fake_arm = MagicMock()
    fake_ctrl = MagicMock()
    fake_ctrl._running = False
    fake_gripper = MagicMock()
    fake_gripper.MAX_DISTANCE_M = 0.09

    with patch("reBotArm_control_py.actuator.RebotArm", return_value=fake_arm), \
         patch(
             "reBotArm_control_py.controllers.RebotArmEndPose",
             return_value=fake_ctrl,
         ), \
         patch(
             "wrc_demo.control.gripper.WrcGripper",
             return_value=fake_gripper,
         ):
        # Lazy import so the patches apply before module import.
        from wrc_demo.control import make_arm

        arm = make_arm(cfg, kinematics=MagicMock())

    assert hasattr(arm, "grasp_bridge"), (
        "SafeArm must have a grasp_bridge attribute set by make_arm"
    )
    assert arm.grasp_bridge is not None


def test_rebot_rs_passes_unity_gravity_ff_scale():
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    fake_arm = MagicMock()
    fake_ctrl = MagicMock()
    fake_gripper = MagicMock()

    with patch("reBotArm_control_py.actuator.RebotArm", return_value=fake_arm), \
         patch(
             "reBotArm_control_py.controllers.RebotArmEndPose",
             return_value=fake_ctrl,
         ) as controller_cls, \
         patch("wrc_demo.control.gripper.WrcGripper", return_value=fake_gripper):
        from wrc_demo.control import make_arm
        make_arm(cfg, kinematics=MagicMock())

    scale = controller_cls.call_args.kwargs["gravity_ff_scale"]
    assert np.array_equal(scale, np.ones(6))

def test_rebot_rs_wrc_gripper_start_called_not_ctrl_start():
    """WrcGripper.start() cooperates with the SDK loop via _loop_cb wrapper.

    The control loop is started exactly once: by the rebot_rs path,
    sdk_ctrl.start() is NOT called (WrcGripper extends the loop, not
    restarts it).
    """
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")

    fake_arm = MagicMock()
    fake_ctrl = MagicMock()
    fake_ctrl._running = False
    fake_gripper = MagicMock()

    with patch("reBotArm_control_py.actuator.RebotArm", return_value=fake_arm), \
         patch(
             "reBotArm_control_py.controllers.RebotArmEndPose",
             return_value=fake_ctrl,
         ), \
         patch(
             "wrc_demo.control.gripper.WrcGripper",
             return_value=fake_gripper,
         ):
        from wrc_demo.control import make_arm

        make_arm(cfg, kinematics=MagicMock())

    fake_gripper.start.assert_called_once()
    fake_ctrl.start.assert_not_called(), (
        "sdk_ctrl.start() must NOT be called when WrcGripper cooperates "
        "with the SDK loop via _loop_cb"
    )


def test_o6_startup_preflights_can_and_tactile_baseline():
    cfg = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    fake_arm = MagicMock()
    fake_arm._motor_map = {}
    fake_ctrl = MagicMock()
    fake_hand = MagicMock()
    fake_hand.cfg.can_interface = "can_hand"
    fake_hand.calibrate_tactile_baseline.return_value = {"samples": 5, "peak": 3}

    with patch("reBotArm_control_py.actuator.RebotArm", return_value=fake_arm), \
         patch("reBotArm_control_py.controllers.RebotArmEndPose", return_value=fake_ctrl), \
         patch("wrc_demo.end_effectors.LinkerHandO6", return_value=fake_hand) as hand_cls:
        from wrc_demo.control import make_arm
        arm = make_arm(cfg, kinematics=MagicMock())

    hand_cls.assert_called_once()
    assert callable(hand_cls.call_args.kwargs["abort_requested"])
    fake_hand.connect.assert_called_once()
    fake_hand.calibrate_tactile_baseline.assert_called_once()
    fake_ctrl.start.assert_called_once()
    assert arm.end_effector is fake_hand


def test_mock_arm_grasp_bridge_is_none():
    """Mock arm has grasp_bridge=None so skills can branch on it uniformly."""
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")

    from wrc_demo.control import make_arm

    arm = make_arm(cfg, kinematics=MagicMock())
    # The MockArm backend doesn't get a bridge; attribute may be absent or None
    assert getattr(arm, "grasp_bridge", None) is None
