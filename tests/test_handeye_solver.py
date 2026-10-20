"""Tests for the joint SE(3) hand-eye solver."""

from __future__ import annotations

import numpy as np
import pytest

from wrc_demo.calibration import (
    CalibrationSample,
    joint_solve,
    joint_solve_restart,
    se3_log,
    se3_log_residual,
    so3_exp,
    so3_log,
)


def _random_rot(rng: np.random.Generator) -> np.ndarray:
    # Uniform on SO(3) via QR
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _make_T(rng: np.random.Generator, t_max: float = 0.5, rot_scale: float = 1.0) -> np.ndarray:
    """Build an SE(3) with bounded rotation magnitude (rot_scale in rad)."""
    T = np.eye(4)
    omega = rng.normal(scale=rot_scale, size=3)
    T[:3, :3] = so3_exp(omega)
    T[:3, 3] = rng.uniform(-t_max, t_max, size=3)
    return T


def _build_dataset(
    rng: np.random.Generator,
    n: int = 15,
    *,
    Y_rot: float = 0.05,
    Y_t: float = 0.04,
    X_rot: float = 0.6,
    X_t: float = 0.4,
    G_rot: float = 0.6,
    G_t: float = 0.3,
):
    """Build a realistic hand-eye dataset.

    The marker is mounted close to the gripper end (small Y rotation/translation),
    the camera is mounted on a tripod well off the gripper (larger X).
    """
    X = _make_T(rng, t_max=X_t, rot_scale=X_rot)
    Y = _make_T(rng, t_max=Y_t, rot_scale=Y_rot)
    samples = []
    for i in range(n):
        G = _make_T(rng, t_max=G_t, rot_scale=G_rot)
        # M_i = inv(X) G Y (from G Y = X M)
        M = np.linalg.inv(X) @ G @ Y
        samples.append(
            CalibrationSample(
                pose_label=f"p{i}",
                T_gripper2base=G,
                T_marker2cam=M,
                aruco_reprojection_px=0.1,
                inlier=True,
            )
        )
    return samples, X, Y


def test_so3_log_exp_roundtrip(rng):
    for _ in range(20):
        R = _random_rot(rng)
        omega = so3_log(R)
        R_back = so3_exp(omega)
        assert np.allclose(R, R_back, atol=1e-8), "so3 log/exp round-trip failed"


def test_se3_log_zero_for_identity():
    assert np.allclose(se3_log(np.eye(4)), np.zeros(6), atol=1e-9)


def test_joint_solve_perfect_data(rng):
    samples, X_true, Y_true = _build_dataset(rng, n=12)
    fit = joint_solve(samples)
    # Translation should match to < 5e-3 m on noiseless data.
    assert np.allclose(fit.T_cam2base[:3, 3], X_true[:3, 3], atol=5e-3), (
        f"X trans err {fit.T_cam2base[:3, 3] - X_true[:3, 3]}"
    )
    assert np.allclose(fit.T_marker2gripper[:3, 3], Y_true[:3, 3], atol=5e-3)
    assert fit.metrics["translation_rmse_mm"] < 1.0


def test_joint_solve_robust_to_noise(rng):
    samples, _, _ = _build_dataset(rng, n=15)
    # Add ~1 mm translation noise and tiny rotation noise to G.
    for s in samples:
        noise_t = rng.normal(scale=1e-3, size=3)
        s_omega = so3_log(s.T_gripper2base[:3, :3]) + rng.normal(scale=0.003, size=3)
        s.T_gripper2base[:3, :3] = so3_exp(s_omega)
        s.T_gripper2base[:3, 3] += noise_t
    fit = joint_solve(samples)
    # The inlier-aggregated SE(3) log residual is the rough proxy; under
    # ~1 mm FK noise the inlier loss stays well below 50 mm.
    assert fit.metrics["translation_rmse_mm"] < 50.0


def test_joint_solve_rejects_outliers(rng):
    samples, X_true, _ = _build_dataset(rng, n=20)
    # Corrupt 4 samples (20%) with big M translation noise (would
    # otherwise drag the fit).
    for s in samples[:4]:
        s.T_marker2cam[:3, 3] += rng.normal(scale=0.20, size=3)
    fit = joint_solve(samples)
    # The fit must converge somewhere close to the true X (translation).
    trans_err = float(np.linalg.norm(fit.T_cam2base[:3, 3] - X_true[:3, 3]))
    assert trans_err < 0.05, f"X recovered off by {trans_err * 1000:.1f} mm"
    # At least the corrupted samples show up as outliers.
    assert sum(1 for i in fit.outliers if i < 4) >= 2


def test_joint_solve_rejects_too_few_samples():
    s = CalibrationSample(pose_label="p0", T_gripper2base=np.eye(4), T_marker2cam=np.eye(4))
    with pytest.raises(ValueError):
        joint_solve([s])


def test_joint_solve_rejects_bad_input():
    bad = CalibrationSample(
        pose_label="p", T_gripper2base=np.eye(4) * 2.0, T_marker2cam=np.eye(4)
    )
    s2 = CalibrationSample(
        pose_label="q", T_gripper2base=np.eye(4), T_marker2cam=np.eye(4)
    )
    s3 = CalibrationSample(
        pose_label="r", T_gripper2base=np.eye(4), T_marker2cam=np.eye(4)
    )
    s4 = CalibrationSample(
        pose_label="s", T_gripper2base=np.eye(4), T_marker2cam=np.eye(4)
    )
    with pytest.raises(ValueError):
        joint_solve([bad, s2, s3, s4])


def test_joint_solve_restart_recovers(rng):
    samples, _, _ = _build_dataset(rng, n=10)
    fit = joint_solve_restart(samples, n_restart=2)
    assert np.isfinite(fit.metrics["translation_rmse_mm"])


def test_joint_solve_restart_reports_underlying_error(monkeypatch, rng):
    """Dependency and optimizer errors must not be hidden behind a generic
    'no fit converged' message.
    """
    import wrc_demo.calibration.handeye_solver as solver

    samples, _, _ = _build_dataset(rng, n=6)

    def _missing_dependency(_samples):
        raise ModuleNotFoundError("No module named 'scipy'")

    monkeypatch.setattr(solver, "joint_solve", _missing_dependency)
    with pytest.raises(RuntimeError, match="ModuleNotFoundError.*scipy"):
        solver.joint_solve_restart(samples, n_restart=2)


def test_handeye_solver_baseline_same_T(rng):
    """PLAN §6.3 #4: joint_solve must agree with cv2.calibrateHandEye
    on the sub-case where the marker coincides with the gripper
    (Y = I). In that case ``X = G_i M_i^{-1}`` for any sample matches
    the eye-to-hand variant's estimate to within sub-mm on noiseless
    synthetic data.
    """
    # Y = I: marker frame == gripper frame. Build diverse gripper
    # poses (varying translation in all 3 axes + 3-axis rotation) so
    # the joint SE(3) fit is not degenerate.
    G = []
    X_true = np.array([
        [0.0, -1.0, 0.0, 0.28],
        [-1.0, 0.0, 0.0, 0.00],
        [0.0, 0.0, -1.0, 0.60],
        [0.0, 0.0, 0.0, 1.00],
    ])
    for i in range(10):
        ang_z = 0.3 * i
        ang_y = 0.2 + 0.1 * i
        Rz = np.array([
            [np.cos(ang_z), -np.sin(ang_z), 0.0],
            [np.sin(ang_z),  np.cos(ang_z), 0.0],
            [0.0, 0.0, 1.0],
        ])
        Ry = np.array([
            [np.cos(ang_y), 0.0, np.sin(ang_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang_y), 0.0, np.cos(ang_y)],
        ])
        T = np.eye(4)
        T[:3, :3] = Ry @ Rz
        T[:3, 3] = [0.28 + 0.02 * i, 0.03 * np.cos(i), 0.15 + 0.04 * i]
        G.append(T)
    M = [np.linalg.inv(X_true) @ G[i] for i in range(len(G))]
    samples = [
        CalibrationSample(pose_label=f"p{i}", T_gripper2base=G[i], T_marker2cam=M[i])
        for i in range(len(G))
    ]
    fit = joint_solve(samples)
    # X must be very close to X_true; Y must collapse to identity.
    assert np.allclose(fit.T_cam2base, X_true, atol=5e-3), (
        f"baseline mismatch: X err={np.linalg.norm(fit.T_cam2base - X_true)}"
    )
    assert np.allclose(fit.T_marker2gripper, np.eye(4), atol=5e-3)
    assert fit.metrics["translation_rmse_mm"] < 1.0


# ─────────────────────────────────────────────────────────────────────────
# Regression: ``joint_solve_restart`` must score every candidate X, Y on
# the REAL samples, never on the "fake" dataset it constructs for the
# k>0 restarts. The previous implementation trusted the fake-data
# ``r.metrics["total_loss"]`` as the score, which always returned
# ~machine-epsilon (because the fake dataset admits a perfect solution
# by construction), so it would pick whatever k>0 restart converged
# last -- even if that solution was a perfect fit for FAKE data but a
# >100 mm garbage fit for the REAL samples. The on-rig calibration in
# July 2026 hit this and shipped a ``hand_eye_top.json`` whose saved
# ``T_cam2base`` differed from the camera's true position by tens of
# centimeters, even though the file reported
# ``translation_rmse_mm = 0.000005 mm``. (HANDOFF §11 issue 15.)
# ─────────────────────────────────────────────────────────────────────────


def test_joint_solve_restart_scores_real_samples(rng):
    """``joint_solve_restart`` must evaluate every candidate solution on
    the real samples -- not the fake dataset used to bootstrap later
    restarts."""
    from wrc_demo.calibration import CalibrationSample
    from wrc_demo.calibration.handeye_solver import (
        joint_solve_restart, joint_solve, se3_log_residual,
    )

    # Build a hand-eye dataset with a known ground-truth (X_true, Y_true).
    # The samples use realistic noise; we just need *enough* structure
    # that the solver can recover (X_true, Y_true) on real data.
    n = 10
    samples = []
    X_true = np.eye(4)
    X_true[:3, 3] = [0.25, -0.10, 0.60]  # camera 60 cm above base origin
    X_true[:3, :3] = so3_exp(np.array([0.2, 0.3, -0.1]))

    Y_true = np.eye(4)
    Y_true[:3, 3] = [0.005, 0.010, 0.040]  # marker offset from gripper
    Y_true[:3, :3] = so3_exp(np.array([0.05, -0.02, 0.03]))

    for i in range(n):
        ang = 0.3 * i
        G = np.eye(4)
        G[:3, 3] = [0.28 + 0.01 * i, 0.02 * np.cos(i), 0.30 + 0.01 * np.sin(i)]
        G[:3, :3] = so3_exp(np.array([0.1 * np.cos(ang), 0.1, 0.1 * np.sin(ang)]))
        # Forward: marker_in_base = G @ Y_true. Camera sees it via X_true.
        # marker_in_cam = inv(X_true) @ G @ Y_true.
        M = np.linalg.inv(X_true) @ G @ Y_true
        # Add tiny detection noise
        M[:3, 3] += rng.normal(scale=1e-4, size=3)
        samples.append(
            CalibrationSample(
                pose_label=f"synth_{i}",
                T_gripper2base=G,
                T_marker2cam=M,
                aruco_reprojection_px=0.5,
                inlier=True,
            )
        )

    fit = joint_solve_restart(samples, n_restart=4, seed=0)

    # The reported ``translation_rmse_mm`` MUST be consistent with the
    # actual per-sample residuals on the REAL samples. Pre-fix bug:
    # reported 0.000005 mm while real residuals were >100 mm.
    X = np.asarray(fit.T_cam2base, dtype=float)
    Y = np.asarray(fit.T_marker2gripper, dtype=float)
    real_norms_mm = [
        float(np.linalg.norm(se3_log_residual(s.T_gripper2base, s.T_marker2cam, X, Y))) * 1000.0
        for s in samples
    ]
    real_rmse_mm = float(np.sqrt(np.mean(np.asarray(real_norms_mm) ** 2)))
    reported_rmse_mm = float(fit.metrics["translation_rmse_mm"])

    # The reported metric must be within an order of magnitude of
    # reality. Pre-fix bug: reported 5e-6 mm while real residuals were
    # >100 mm (8+ orders of magnitude gap).
    assert reported_rmse_mm < 10.0 * max(real_rmse_mm, 1e-6), (
        f"reported translation_rmse_mm={reported_rmse_mm} vs real per-sample "
        f"RMSE={real_rmse_mm} on the actual samples -- the solver is "
        "trusting fake-data metrics. (HANDOFF §11 issue 15)"
    )

    # And the recovered (X, Y) should be close to ground truth (within
    # a few mm given our noise scale). Pre-fix bug returned garbage.
    pos_err_mm = float(np.linalg.norm(X[:3, 3] - X_true[:3, 3])) * 1000.0
    assert pos_err_mm < 30.0, (
        f"recovered X is {pos_err_mm:.1f} mm away from ground truth -- "
        "joint_solve_restart picked a fake-data solution."
    )
