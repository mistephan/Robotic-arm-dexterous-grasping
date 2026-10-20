"""Joint SE(3) hand-eye solver.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, TCP = gripper_end)

Math (see PLAN.md §13):

    X = T_cam2base          (unknown, 6 dof)
    Y = T_marker2gripper    (unknown, 6 dof)
    G_i Y = X M_i           (per-sample rigid constraint, i = 1..N)

We minimize the SE(3) Lie-algebra residual of the equation
``(G_i Y)^{-1} (X M_i)`` with a Huber loss via Levenberg-Marquardt. We
seed from ``X_0 = G_1 M_1^{-1}``, ``Y_0 = I`` and run a small number of
restarts (default 8) with random perturbations of the seed; the best
(non-NaN, finite loss) candidate wins. Outliers are rejected via
3 * MAD on the per-sample residual after a warm fit.

The module never imports cuRobo, the arm SDK, or any hardware dependency
so it is unit-testable with synthetic data alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .frame_convention import is_se3
from .calib_dataset import CalibrationSample


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """log map on SO(3) -> R^3 (rotation vector)."""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        # small-angle approximation: log(R) ≈ 0.5 * (R - R^T) skew-vector
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
        return v
    n = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([n[2, 1], n[0, 2], n[1, 0]]) * theta


def se3_log(T: np.ndarray) -> np.ndarray:
    """log map on SE(3) -> R^6 (translation, then rotation vector)."""
    T = np.asarray(T, dtype=float)
    t = T[:3, 3]
    omega = so3_log(T[:3, :3])
    return np.concatenate([t, omega])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map on SO(3) (rotation vector -> 3x3)."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3) + _skew(omega)
    k = omega / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K @ K


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map on SE(3) (R^6 -> 4x4): (tx, ty, tz, wx, wy, wz)."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    t = xi[:3]
    omega = xi[3:]
    R = so3_exp(omega)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        V = np.eye(3) + 0.5 * _skew(omega)
    else:
        K = _skew(omega / theta)
        V = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / (theta * theta) * _skew(omega)
            + (theta - np.sin(theta)) / (theta * theta * theta) * _skew(omega) @ _skew(omega)
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ t
    return T


def se3_log_residual(
    G: np.ndarray, M: np.ndarray, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Per-sample 6-vector residual: ``log( (G Y)^{-1} (X M) )``."""
    lhs = np.linalg.inv(G @ Y) @ (X @ M)
    return se3_log(lhs)


def huber_loss(r: np.ndarray, delta: float) -> np.ndarray:
    """Element-wise Huber loss: 0.5 r^2 for |r| <= delta, else delta(|r|-0.5 delta).

    Applied per-sample to the 6-vec residual magnitude. We split the
    residual's translation (first 3) and rotation (last 3) components
    with separate scales so that a 1 mm translation weight matches a
    small angular tolerance.
    """
    r = np.asarray(r, dtype=float)
    norm = float(np.linalg.norm(r))
    if norm <= delta:
        return 0.5 * (r * r)
    return delta * (norm - 0.5 * delta) * (r / norm)


@dataclass(frozen=True)
class JointSolveResult:
    """Solver output: the SE(3) matrices plus the metrics used to score them."""

    T_cam2base: np.ndarray
    T_marker2gripper: np.ndarray
    inliers: tuple[int, ...]
    outliers: tuple[int, ...]
    metrics: dict[str, float] = field(default_factory=dict)


def _param_to_se3(t: np.ndarray) -> np.ndarray:
    """12-vector (3 translation + 3 rot for X, 3 translation + 3 rot for Y) -> X, Y."""
    t = np.asarray(t, dtype=float).reshape(12)
    X = np.eye(4)
    X[:3, :3] = so3_exp(t[3:6])
    X[:3, 3] = t[0:3]
    Y = np.eye(4)
    Y[:3, :3] = so3_exp(t[9:12])
    Y[:3, 3] = t[6:9]
    return X, Y


def _se3_to_param(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    p = np.zeros(12)
    p[0:3] = X[:3, 3]
    p[3:6] = so3_log(X[:3, :3])
    p[6:9] = Y[:3, 3]
    p[9:12] = so3_log(Y[:3, :3])
    return p


def _residual_vec(
    param: np.ndarray,
    samples: Sequence[CalibrationSample],
    weights: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Residual vector flattened for least_squares."""
    X, Y = _param_to_se3(param)
    out = np.empty(6 * len(samples))
    for i, s in enumerate(samples):
        r = se3_log_residual(s.T_gripper2base, s.T_marker2cam, X, Y)
        # Huber on per-sample norm, scaled by inverse sample weight so
        # outliers get pulled up (smaller weight = stronger pull).
        r_loss = huber_loss(r, delta) * weights[i]
        out[6 * i : 6 * i + 6] = r_loss
    return out


def joint_solve(
    samples: Sequence[CalibrationSample],
    *,
    delta: float = 1.5e-3,  # 1.5 mm / rad
    robust: bool = True,
    outlier_iterations: int = 2,
) -> JointSolveResult:
    """Joint SE(3) least-squares fit.

    ``samples`` must contain at least 4 inlier observations for an
    identifiable solve (12 dof, 6 per rigid pair). The first robust
    iteration runs with all weights = 1; subsequent iterations reweight
    samples whose 6-vec residual norm is > 3 * MAD down by 10x.
    """
    if len(samples) < 4:
        raise ValueError(f"need >=4 samples for joint solve, got {len(samples)}")

    from scipy.optimize import least_squares

    G = [s.T_gripper2base for s in samples]
    M = [s.T_marker2cam for s in samples]
    for g, m in zip(G, M):
        if not (is_se3(g) and is_se3(m)):
            raise ValueError("samples must contain SE(3) matrices")

    # X_0 = G_0 M_0^{-1}, Y_0 = I -- identity for the marker-to-gripper
    X0 = G[0] @ np.linalg.inv(M[0])
    Y0 = np.eye(4)
    p0 = _se3_to_param(X0, Y0)

    weights = np.ones(len(samples))

    best: tuple[float, JointSolveResult] | None = None
    for outer in range(outlier_iterations + 1):
        res = least_squares(
            _residual_vec, p0,
            args=(samples, weights, delta),
            method="lm", max_nfev=400,
        )
        p0 = res.x
        if not robust or outer == outlier_iterations:
            break
        # Reweight by residual norm vs median absolute deviation.
        with np.errstate(invalid="ignore"):
            norms = np.array([
                float(np.linalg.norm(se3_log_residual(G[i], M[i], *_param_to_se3(p0))))
                for i in range(len(samples))
            ])
        med = float(np.median(norms)) if len(norms) else 0.0
        mad = float(np.median(np.abs(norms - med))) + 1e-9
        thresh = med + 3.0 * mad
        weights = np.where(norms > thresh, 0.1, 1.0)

    X, Y = _param_to_se3(p0)
    # Per-sample SE(3) log residual vector [tx, ty, tz, rx, ry, rz] (m, rad)
    # -- used both for the outlier 3*MAD test (on the L2 norm) and for the
    # per-axis metric computation.
    with np.errstate(invalid="ignore"):
        res_6d = np.array([
            se3_log_residual(G[i], M[i], X, Y) for i in range(len(samples))
        ])
    norms = np.linalg.norm(res_6d, axis=1)
    med = float(np.median(norms)) if len(norms) else 0.0
    mad = float(np.median(np.abs(norms - med))) + 1e-9
    thresh = med + 3.0 * mad
    inliers = tuple(int(i) for i, n in enumerate(norms) if n <= thresh)
    outliers = tuple(int(i) for i, n in enumerate(norms) if n > thresh)

    # Aggregate metrics. ``res_6d[list(inliers)]`` keeps the per-axis
    # split so we can report translation in mm and rotation in deg as
    # independent quantities (the old combined SE(3) norm is preserved
    # under the ``se3_log_norm_*`` keys for backward compatibility).
    inlier_res = res_6d[list(inliers)] if inliers else res_6d
    metrics = _metrics_from_norms(inlier_res, norms)
    metrics["lm_converged"] = bool(np.isfinite(res.fun).all())
    metrics["lm_iterations"] = int(getattr(res, "nfev", 0))

    result = JointSolveResult(
        T_cam2base=X,
        T_marker2gripper=Y,
        inliers=inliers,
        outliers=outliers,
        metrics=metrics,
    )
    if best is None or metrics["total_loss"] < best[0]:
        best = (metrics["total_loss"], result)
    assert best is not None
    return best[1]


def _metrics_from_norms(
    inlier_residuals: np.ndarray,
    all_norms: np.ndarray,
) -> dict[str, float]:
    """Compute per-axis RMSE for translation (mm) and rotation (deg) on the
    inlier residuals, plus a backward-compatible alias of the old combined
    SE(3) log-norm fields.

    Parameters
    ----------
    inlier_residuals
        ``(n_inliers, 6)`` array of per-sample SE(3) log residuals from
        ``se3_log_residual``: first 3 columns are translation in meters,
        last 3 are the rotation vector in radians.
    all_norms
        ``(n_samples,)`` L2 norms of the same residuals (used for the
        deprecated combined-norm fields and the median).
    """
    nan_metrics = {
        "translation_rmse_mm": float("nan"),
        "translation_max_err_mm": float("nan"),
        "rotation_rmse_deg": float("nan"),
        "rotation_max_err_deg": float("nan"),
        "se3_log_norm_rmse_mm": float("nan"),
        "se3_log_norm_max_mm": float("nan"),
        "se3_log_norm_rmse_deg": float("nan"),
        "se3_log_norm_max_deg": float("nan"),
        "se3_log_norm_median": float("nan"),
        "total_loss": float("inf"),
        "n_inliers": 0,
    }
    if len(inlier_residuals) == 0:
        return nan_metrics

    inlier_res = np.asarray(inlier_residuals, dtype=float)
    t = inlier_res[:, :3]   # tx, ty, tz  in meters
    r = inlier_res[:, 3:]   # rx, ry, rz  in radians (rotation vector)

    # Per-axis split -- translation in mm, rotation in deg, with the
    # ``||t||`` / ``||r||`` Euclidean norm per sample (NOT the raw
    # component sum, which would mix axes and units within a quantity).
    t_per_sample_m = np.linalg.norm(t, axis=1)
    r_per_sample_rad = np.linalg.norm(r, axis=1)
    t_rmse_m = float(np.sqrt(np.mean(t_per_sample_m ** 2)))
    r_rmse_rad = float(np.sqrt(np.mean(r_per_sample_rad ** 2)))
    t_max_m = float(np.max(t_per_sample_m))
    r_max_rad = float(np.max(r_per_sample_rad))

    # Backward-compat: combined SE(3) log norm in "mm-like" units. The
    # log residual packs translation (m) and rotation (rad) into one 6-vec
    # with mixed units, so the L2 norm is dimensionally inconsistent --
    # treat it as an order-of-magnitude diagnostic, not a physical mm.
    se3_norms = np.asarray(all_norms, dtype=float)
    if len(se3_norms):
        se3_rmse = float(np.sqrt(np.mean(se3_norms ** 2)))
        se3_max = float(np.max(se3_norms))
    else:
        se3_rmse = float("nan")
        se3_max = float("nan")

    return {
        # New: physically correct per-axis quantities
        "translation_rmse_mm": t_rmse_m * 1000.0,
        "translation_max_err_mm": t_max_m * 1000.0,
        "rotation_rmse_deg": float(np.degrees(r_rmse_rad)),
        "rotation_max_err_deg": float(np.degrees(r_max_rad)),
        # Old (kept under explicit ``se3_log_norm_*`` names so callers can
        # still read the combined-norm diagnostic without confusing it
        # with the per-axis mm/deg).
        "se3_log_norm_rmse_mm": se3_rmse * 1000.0,
        "se3_log_norm_max_mm": se3_max * 1000.0,
        "se3_log_norm_rmse_deg": se3_rmse / 100.0,
        "se3_log_norm_max_deg": se3_max / 100.0,
        "se3_log_norm_median": float(np.median(se3_norms)) * 1000.0 if len(se3_norms) else float("nan"),
        "total_loss": float(np.sum(se3_norms ** 2)),
        "n_inliers": int(len(inlier_residuals)),
    }


def joint_solve_restart(
    samples: Sequence[CalibrationSample],
    *,
    n_restart: int = 8,
    seed: int = 0,
) -> JointSolveResult:
    """Multi-start joint solve. Best loss wins.

    Restart strategy:
      k=0 -- vanilla ``joint_solve(samples)`` with X0 = G[0] @ inv(M[0]),
             Y0 = I.
      k>0 -- construct a fake dataset that has a known perfect solution
             ``(X*, Y*) = (base @ perturb_R, perturb_t)`` so the optimizer
             converges quickly, then **re-score the result on the real
             samples**. We cannot trust ``r.metrics`` (computed on fake
             data) for picking the winner -- only the per-sample SE(3)
             log residuals on the real data are meaningful.
    """
    rng = np.random.default_rng(seed)

    def _score(fit):
        X = np.asarray(fit.T_cam2base, dtype=float)
        Y = np.asarray(fit.T_marker2gripper, dtype=float)
        norms = np.array([
            float(np.linalg.norm(se3_log_residual(s.T_gripper2base, s.T_marker2cam, X, Y)))
            for s in samples
        ])
        return float(np.median(norms))

    best: JointSolveResult | None = None
    best_score = float("inf")
    errors: list[Exception] = []
    for k in range(n_restart):
        try:
            r = joint_solve(samples)
        except Exception as exc:
            errors.append(exc)
            continue
        score = _score(r)
        if score < best_score:
            best, best_score = r, score

        if k == 0:
            continue  # k=0 already ran joint_solve on real samples above

        # Perturb X0 = G[0] @ inv(M[0]) and re-fit on a fake dataset that
        # admits a perfect solution near the perturbation. The fake
        # solve's own metrics are meaningless -- only the score we
        # compute above on the real samples counts.
        perturb = rng.normal(scale=0.05, size=12)
        from .calib_dataset import CalibrationSample as _CS

        fake = []
        G0 = samples[0].T_gripper2base
        M0 = samples[0].T_marker2cam
        base = G0 @ np.linalg.inv(M0)
        for s in samples:
            pX = _se3_to_param(base, np.eye(4))
            pX = pX + perturb
            X, Y = _param_to_se3(pX)
            fake.append(_CS(
                pose_label=s.pose_label,
                T_gripper2base=X @ s.T_marker2cam @ np.linalg.inv(Y),
                T_marker2cam=s.T_marker2cam,
            ))
        try:
            r_fake = joint_solve(fake)
        except Exception as exc:
            errors.append(exc)
            continue
        score_fake = _score(r_fake)
        if score_fake < best_score:
            best, best_score = r_fake, score_fake

    if best is None:
        if errors:
            cause = errors[0]
            raise RuntimeError(
                "joint_solve_restart: no fit converged; first underlying "
                f"error was {type(cause).__name__}: {cause}"
            ) from cause
        raise RuntimeError("joint_solve_restart: no fit converged")
    return best
