"""Compact vector index over TurboQuant codes.

Asymmetric search: the query stays full-precision (rotated only), the
database lives as B-bit codes -- this recovers most of the recall lost to
quantization at zero extra memory. Storage per vector: dim bytes (4-bit codes
kept unpacked for simplicity) + 8 bytes norm; a 512-d embedding is ~520 B
instead of 2 KB float32, and bits=2 halves that again if needed.

If the `turbovec` package (Rust TurboQuant index) is installed it could be
swapped in here; the pure-NumPy path is the default so the demo has no exotic
dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .turboquant import TurboQuantizer


class QuantizedIndex:
    def __init__(self, dim: int, bits: int = 4, seed: int = 0):
        self._tq = TurboQuantizer(dim, bits=bits, seed=seed)
        self._codes: list[np.ndarray] = []
        self._norms: list[float] = []
        self._meta: list[Any] = []

    def __len__(self) -> int:
        return len(self._codes)

    def add(self, vec: np.ndarray, meta: Any = None) -> int:
        codes, norm = self._tq.encode(vec)
        self._codes.append(codes)
        self._norms.append(norm)
        self._meta.append(meta)
        return len(self._codes) - 1

    def remove_ids(self, ids: set[int]) -> None:
        """Drop entries (used when episodic memory expires events)."""
        keep = [i for i in range(len(self._codes)) if i not in ids]
        self._codes = [self._codes[i] for i in keep]
        self._norms = [self._norms[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[float, Any]]:
        """Top-k by estimated cosine similarity -> [(score, meta), ...]."""
        if not self._codes:
            return []
        q_rot, q_norm = self._tq.rotate(query)
        if q_norm == 0.0:
            return []
        db = np.stack([self._tq.decode_rotated(c) for c in self._codes])
        db_dirs = db / np.maximum(np.linalg.norm(db, axis=1, keepdims=True), 1e-12)
        sims = db_dirs @ q_rot
        order = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), self._meta[i]) for i in order]
