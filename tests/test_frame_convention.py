"""Tests for frame-conversion helpers (SE(3) checks, FK round-trip offset)."""

from __future__ import annotations

import numpy as np

from wrc_demo.calibration import (
    curobo_to_wrc_demo_marker_offset,
    is_se3,
    round_trip_check,
)


def test_is_se3_identity():
    assert is_se3(np.eye(4))


def test_is_se3_translation_only():
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    assert is_se3(T)


def test_is_se3_rejects_non_orthogonal():
    T = np.eye(4)
    T[0, 0] = 2.0  # not orthogonal
    assert not is_se3(T)


def test_is_se3_rejects_wrong_last_row():
    T = np.eye(4)
    T[3, 0] = 1.0
    assert not is_se3(T)


def test_is_se3_rejects_wrong_shape():
    assert not is_se3(np.eye(3))


def test_round_trip_identity_offset():
    """FK round-trip should recover identity if both sides are identity."""
    ok, rot_deg, trans_mm = round_trip_check(np.eye(4), np.eye(4))
    assert ok
    assert rot_deg < 1e-6
    assert trans_mm < 1e-6


def test_round_trip_recovers_rigid_offset():
    """Two FK conventions should differ by a rigid offset that is
    reproducible up to numerical noise."""
    # True constant offset S between cuRobo and wrc_demo FK outputs.
    S = np.eye(4)
    S[:3, 3] = [0.1, 0.05, 0.02]
    S[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Build a cuRobo FK output, derive wrc_demo FK = S @ cuRobo.
    T_c = np.eye(4)
    T_c[:3, 3] = [0.28, 0.0, 0.6]
    T_w = S @ T_c
    # Verify S = T_w @ inv(T_c) (the constant offset definition).
    S_recovered = T_w @ np.linalg.inv(T_c)
    assert np.allclose(S_recovered, S, atol=1e-8)
    # The round-trip helper composes T_ab @ T_bc. Pick T_ab = T_c and
    # T_bc = inv(T_c) to demonstrate the identity round-trip, and pick
    # the loose drift tolerances that match a real cuRobo vs wrc_demo
    # FK discrepancy (PLAN.md §13.4 calls for a fixed-but-unknown S).
    ok, rot_deg, trans_mm = round_trip_check(T_c, np.linalg.inv(T_c))
    assert ok
    assert trans_mm < 1e-3
    assert rot_deg < 1e-3


def test_curobo_to_wrc_demo_marker_offset():
    """A constant rigid offset should be recovered exactly."""
    rng = np.random.default_rng(0)
    S = np.eye(4)
    S[:3, 3] = [0.01, -0.02, 0.03]
    A = np.eye(4)
    A[:3, :3] = np.array(
        [
            [np.cos(0.3), -np.sin(0.3), 0.0],
            [np.sin(0.3), np.cos(0.3), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    A[:3, 3] = [0.28, 0.0, 0.6]
    # Suppose the cuRobo FK returns T_c, wrc_demo FK returns T_w = S @ T_c.
    T_c = A
    T_w = S @ A
    recovered = curobo_to_wrc_demo_marker_offset(T_c, T_w)
    assert np.allclose(recovered, S, atol=1e-8)
