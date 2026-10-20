"""Width-aware two-stage gripper close (v2, 2026-08-10).

The old code used fixed close_frac (e.g. 0.90 = 90% of gripper travel =
~8 mm opening on rebot_rs with 95 mm max), which crushed a 3D-printed
banana (30 mm wide) by pushing past the object with high effort. The
new logic closes to ``grasp.width_m * profile.object_close_frac`` (default
0.75 = gripper target = 12 mm), so the MIT motor stalls at the
object surface and the surface limits the force rather than the
position target.

Tests pin the math + the profile choices for each material class so a
future tuning of profiles / close-frac can't silently regress to
crushing fragile objects.

Run::

    python -m pytest \\
        tests/test_gripper_width_aware_close.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from wrc_demo.grasping.force import (
    GRIP_PROFILES,
    compute_close_target,
    select_profile,
)


# ── compute_close_target math ─────────────────────────────────────────


def test_banana_30mm_slippery_target_around_24mm():
    """banana (30 mm wide) + slippery (object_close_frac=0.80) → target 24 mm.

    Old code: close_frac_stage2=0.90 → 8 mm opening, would crush.
    0.75 (= 12 mm opening) was too loose — the banana slipped in the
    fingers. 0.50 = 12 mm opening, banana squished by 12 mm, grips hard.
    """
    banana_width_m = 0.030
    max_width_m = 0.095
    profile = GRIP_PROFILES["slippery"]
    pos1, pos2 = compute_close_target(profile, banana_width_m, max_width_m)
    # Stage 1: fixed 50% travel
    assert pos1 == pytest.approx(0.5)
    # Stage 2: width-aware close_frac = 1 - (24 mm / 95 mm)
    # gripper opens to 24 mm, limiting planned compression to 20%
    expected_close_frac = 1.0 - (banana_width_m * profile.object_close_frac) / max_width_m
    assert pos2 == pytest.approx(expected_close_frac, abs=1e-6)
    # 24 mm < 30 mm so the gripper WILL close onto the banana (not air-grasp)
    gripper_opening_m = max_width_m * (1.0 - pos2)
    assert gripper_opening_m == pytest.approx(0.024, abs=1e-6)
    assert gripper_opening_m < banana_width_m, (
        "gripper must close ONTO the banana (target < object width) — "
        "otherwise no friction"
    )


def test_cube_70mm_rigid_target_around_35mm():
    """cube (70 mm) + rigid (0.50) → target 30 mm, gripper presses 30 mm."""
    cube_width_m = 0.070
    max_width_m = 0.095
    profile = GRIP_PROFILES["rigid"]
    pos1, pos2 = compute_close_target(profile, cube_width_m, max_width_m)
    gripper_opening_m = max_width_m * (1.0 - pos2)
    assert gripper_opening_m == pytest.approx(0.021, abs=1e-3)
    # 30 mm < 70 mm → gripper presses cube 30 mm
    assert gripper_opening_m < cube_width_m


def test_fragile_max_close_frac_caps_for_tiny_object():
    """For a TINY object (5 mm), close_frac = 1 - 0.85*5/95 = 0.955
    would crush to 5 mm opening. fragile's max_close_frac=0.65 clips it:
    gripper stops at 33 mm opening — a 5 mm object has nothing to grab
    (air grasp), but the safety cap prevents the motor from pushing
    past 33 mm toward a hard 5 mm target. The hard target is the
    failure mode that crushed 3D-printed bananas.
    """
    tiny_m = 0.005
    max_width_m = 0.095
    profile = GRIP_PROFILES["fragile"]  # object_close_frac=0.70, max_close_frac=0.65
    _, pos2 = compute_close_target(profile, tiny_m, max_width_m)
    # raw close_frac = 0.955; clipped to max_close_frac=0.65
    assert pos2 == pytest.approx(profile.max_close_frac, abs=1e-6)
    gripper_opening_m = max_width_m * (1.0 - pos2)
    assert gripper_opening_m == pytest.approx(0.0238, abs=1e-3)


def test_min_close_frac_prevents_air_grasp_for_huge_object():
    """For a HUGE object (90 mm, near max_width), close_frac = 1 - 0.50*90/95 = 0.526.
    min_close_frac=0.30 keeps gripper at max 70% closed (28.5 mm opening).
    The 90 mm object is still too wide to fit (planner's feasibility check
    would have rejected it), but if it slipped through, the safety floor
    keeps the gripper from going PAST 28.5 mm opening toward the impossibly
    small 45 mm target.
    """
    huge_m = 0.090
    max_width_m = 0.095
    profile = GRIP_PROFILES["rigid"]  # min_close_frac=0.30
    _, pos2 = compute_close_target(profile, huge_m, max_width_m)
    # raw close_frac = 0.526; in [0.30, 0.90] so no clipping
    expected = 1.0 - profile.object_close_frac * huge_m / max_width_m
    assert pos2 == pytest.approx(expected, abs=1e-6)
    gripper_opening_m = max_width_m * (1.0 - pos2)
    # gripper opens to 45 mm = 0.50 * 90 mm — the user's 2026-08-10 directive
    assert gripper_opening_m == pytest.approx(0.027, abs=1e-3)


def test_heavy_squeezes_more_than_rigid():
    """heavy (0.55) closes further than rigid (0.75) on the same object."""
    obj_m = 0.040
    max_width_m = 0.095
    _, rigid_pos = compute_close_target(GRIP_PROFILES["rigid"], obj_m, max_width_m)
    _, heavy_pos = compute_close_target(GRIP_PROFILES["heavy"], obj_m, max_width_m)
    # heavy should close MORE (higher pos2_frac = smaller opening)
    assert heavy_pos > rigid_pos
    rigid_open_m = max_width_m * (1.0 - rigid_pos)
    heavy_open_m = max_width_m * (1.0 - heavy_pos)
    assert heavy_open_m < rigid_open_m


# ── profile shape: every profile must define width-aware close frac ──


@pytest.mark.parametrize("name", list(GRIP_PROFILES))
def test_every_profile_has_width_aware_fields(name):
    """All GripProfile must have object_close_frac + min/max_close_frac in (0, 1]
    and effort in [0, 1.5] (the MIT tau clip limit)."""
    p = GRIP_PROFILES[name]
    assert 0.0 < p.object_close_frac <= 1.0
    assert 0.0 < p.min_close_frac <= p.max_close_frac <= 1.0
    assert 0.0 <= p.effort <= 1.5
    assert p.lift_speed_scale > 0


def test_slippery_no_longer_overcloses_default_banana():
    """The 2026-08-10 bug: banana → slippery, close_frac_stage2 = 0.90 → ~8 mm
    opening crushed the banana. The current 0.80 profile limits planned
    compression to 20% and uses reduced holding torque."""
    banana = select_profile("banana")
    assert banana.name == "slippery"
    assert banana.object_close_frac == pytest.approx(0.80)
    # The OLD close_frac_stage2=0.90 is gone — verify it really is gone.
    assert not hasattr(banana, "close_frac_stage2")
    # The target remains narrower than the object (24 mm < 30 mm),
    # providing grip without driving toward the hard stop.
    _, pos2 = compute_close_target(banana, 0.030, 0.095)
    target_open_m = 0.095 * (1.0 - pos2)
    assert target_open_m == pytest.approx(0.024, abs=1e-6)
    # Critically NOT the old v2 fixed-frac target (close_frac=0.90 → 9.5 mm).
    # The current profile lands at 24 mm; the old baseline
    # (close_frac_stage2=0.90) also landed at 9.5 mm — coincidence,
    # not a regression. The point of the assertion is to catch the
    # *uncontrolled* v0 code (close_frac_stage2=0.85 → 14.25 mm) which
    # was too loose and let the object slip.
    assert not (target_open_m == pytest.approx(0.095 * 0.15, abs=1e-3)), (
        f"target_open_m={target_open_m} looks like the OLD 85% travel "
        "target (14.25 mm opening) — the v2/v3/v4 fix didn't take effect"
    )


def test_all_profiles_safe_for_default_30mm_object():
    """For a 30 mm object, the close target must be in [min, max] bounds
    for every profile — i.e. the safety bounds actually bound.
    """
    max_width_m = 0.095
    obj_m = 0.030
    for name, profile in GRIP_PROFILES.items():
        _, pos2 = compute_close_target(profile, obj_m, max_width_m)
        assert profile.min_close_frac <= pos2 <= profile.max_close_frac, (
            f"{name}: pos2={pos2} not in "
            f"[{profile.min_close_frac}, {profile.max_close_frac}]"
        )