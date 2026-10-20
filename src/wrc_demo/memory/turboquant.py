"""TurboQuant-style online vector quantization (pure NumPy).

After the method of "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (arXiv:2504.19874, ICLR 2026): data-oblivious, no codebook
training. Each vector is unit-normalized, hit with a fixed random rotation
(which concentrates every coordinate around ~ N(0, 1/d)), then each coordinate
is quantized independently with a uniform B-bit quantizer over a +-c/sqrt(d)
range. The norm is stored per-vector, so inner products are recovered as
||a|| * ||b|| * <a_hat, b_hat>.

This is intentionally dependency-free: the pip `turbovec` package (a Rust
implementation of the same algorithm) can replace it behind QuantizedIndex if
installed, but the demo must not require it.
"""

from __future__ import annotations

import numpy as np


class TurboQuantizer:
    def __init__(self, dim: int, bits: int = 4, seed: int = 0, clip_sigmas: float = 3.0):
        if not 1 <= bits <= 8:
            raise ValueError("bits must be in [1, 8]")
        self.dim = dim
        self.bits = bits
        self.levels = 2**bits
        # Fixed random rotation: QR of a Gaussian matrix (Haar-ish, orthonormal).
        rng = np.random.default_rng(seed)
        q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
        # Fix QR sign ambiguity for determinism across BLAS implementations.
        q *= np.sign(np.diag(r))
        self._Q = q
        # Post-rotation coordinates of a unit vector have std ~ 1/sqrt(d).
        self._clip = clip_sigmas / np.sqrt(dim)
        self._step = 2 * self._clip / self.levels

    def encode(self, vec: np.ndarray) -> tuple[np.ndarray, float]:
        """-> (codes uint8 (dim,), norm). Zero vectors encode with norm 0."""
        v = np.asarray(vec, dtype=np.float64).reshape(self.dim)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.full(self.dim, self.levels // 2, dtype=np.uint8), 0.0
        u = self._Q @ (v / norm)
        u = np.clip(u, -self._clip, self._clip - 1e-12)
        codes = np.floor((u + self._clip) / self._step).astype(np.uint8)
        return codes, norm

    def decode(self, codes: np.ndarray, norm: float) -> np.ndarray:
        """Reconstruct the (approximate) original vector."""
        u = (codes.astype(np.float64) + 0.5) * self._step - self._clip
        v = self._Q.T @ u
        # Renormalize the direction: quantization shrinks the norm slightly.
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v * norm

    def decode_rotated(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct in rotated space (for asymmetric distance queries)."""
        return (codes.astype(np.float64) + 0.5) * self._step - self._clip

    def rotate(self, vec: np.ndarray) -> tuple[np.ndarray, float]:
        """Rotate + unit-normalize a full-precision query -> (u, norm)."""
        v = np.asarray(vec, dtype=np.float64).reshape(self.dim)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.zeros(self.dim), 0.0
        return self._Q @ (v / norm), norm
