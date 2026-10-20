import numpy as np

from conftest import JOINT_SIGNS, URDF, needs_pin


@needs_pin
def test_fk_ik_roundtrip():
    from wrc_demo.control.kinematics import Kinematics

    kin = Kinematics(str(URDF), "gripper_end", joint_signs=JOINT_SIGNS)
    q_ref = np.array([0.3, 1.1, 1.3, 0.2, 0.5, -0.4])
    T = kin.fk(q_ref)
    res = kin.ik(T, q_init=np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0]))
    assert res.success, f"IK failed with error {res.error}"
    T2 = kin.fk(res.q)
    assert np.linalg.norm(T2[:3, 3] - T[:3, 3]) < 1e-3
    assert np.allclose(T2[:3, :3], T[:3, :3], atol=5e-3)


@needs_pin
def test_ik_respects_joint_limits():
    from wrc_demo.control.kinematics import Kinematics

    kin = Kinematics(str(URDF), "gripper_end", joint_signs=JOINT_SIGNS)
    lo, hi = kin.joint_limits
    T = kin.fk(np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0]))
    res = kin.ik(T, q_init=np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0]))
    assert np.all(res.q >= lo - 1e-9) and np.all(res.q <= hi + 1e-9)


@needs_pin
def test_top_down_grasp_pose_reachable():
    from wrc_demo.control.kinematics import Kinematics
    from wrc_demo.grasping.obb_grasp import _yaw_rotation
    from wrc_demo.types import make_transform

    kin = Kinematics(str(URDF), "gripper_end", joint_signs=JOINT_SIGNS)
    R = _yaw_rotation(0.3)
    res = kin.ik(
        make_transform(R, [0.29, 0.0, 0.03]),
        q_init=np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0]),
    )
    assert res.success
    # Tool x must point down at the solution.
    T = kin.fk(res.q)
    assert T[:3, 0] @ np.array([0, 0, -1]) > 0.99


@needs_pin
def test_link_positions_shape():
    from wrc_demo.control.kinematics import Kinematics

    kin = Kinematics(str(URDF), "gripper_end", joint_signs=JOINT_SIGNS)
    links = kin.link_positions(np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0]))
    assert links.shape[1] == 3 and links.shape[0] >= 6
