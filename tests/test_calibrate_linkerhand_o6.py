from argparse import Namespace

import pytest

from tools.calibrate_linkerhand_o6 import o6_config, validate_args
from wrc_demo.config import load_demo_config


def args(**overrides):
    values = dict(samples=10, mode="inspect", action=None,
                  allow_motion=False, object_present=False)
    values.update(overrides)
    return Namespace(**values)


def test_inspect_is_default_safe_mode():
    validate_args(args())


def test_all_actions_require_explicit_motion_unlock():
    with pytest.raises(SystemExit, match="motion is locked"):
        validate_args(args(mode="action", action="open"))


def test_closing_requires_object_present():
    with pytest.raises(SystemExit, match="object"):
        validate_args(args(mode="action", action="pinch", allow_motion=True))


def test_o6_profile_maps_only_driver_fields():
    cfg = load_demo_config(camera="mock", arm="rebot_rs_o6", llm="mock")
    hand_cfg = o6_config(cfg.arm)
    assert hand_cfg.can_interface == "can_hand"
    assert hand_cfg.baseline_samples == 5
    assert not hasattr(hand_cfg, "tcp_calibrated")
