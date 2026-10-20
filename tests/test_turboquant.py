import numpy as np

from wrc_demo.memory.turboquant import TurboQuantizer
from wrc_demo.memory.vector_index import QuantizedIndex


def test_roundtrip_reconstruction(rng):
    dim = 256
    tq = TurboQuantizer(dim, bits=4, seed=7)
    v = rng.standard_normal(dim)
    codes, norm = tq.encode(v)
    v_hat = tq.decode(codes, norm)
    assert codes.dtype == np.uint8
    assert codes.max() < 16
    # Norm is preserved exactly; direction to within quantization error.
    assert np.isclose(np.linalg.norm(v_hat), np.linalg.norm(v))
    cos = v @ v_hat / (np.linalg.norm(v) * np.linalg.norm(v_hat))
    assert cos > 0.95


def test_cosine_preserved_between_pairs(rng):
    dim = 128
    tq = TurboQuantizer(dim, bits=4, seed=1)
    for _ in range(10):
        a = rng.standard_normal(dim)
        b = a + 0.5 * rng.standard_normal(dim)  # correlated pair
        true_cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        a_hat = tq.decode(*tq.encode(a))
        b_hat = tq.decode(*tq.encode(b))
        est_cos = a_hat @ b_hat / (np.linalg.norm(a_hat) * np.linalg.norm(b_hat))
        assert abs(true_cos - est_cos) < 0.1


def test_determinism():
    tq1 = TurboQuantizer(64, bits=4, seed=3)
    tq2 = TurboQuantizer(64, bits=4, seed=3)
    v = np.arange(64, dtype=float)
    c1, n1 = tq1.encode(v)
    c2, n2 = tq2.encode(v)
    assert np.array_equal(c1, c2) and n1 == n2


def test_zero_vector():
    tq = TurboQuantizer(32, bits=4)
    codes, norm = tq.encode(np.zeros(32))
    assert norm == 0.0
    assert np.allclose(tq.decode(codes, norm), 0.0)


def test_index_retrieval_recall(rng):
    dim, n_clusters, per_cluster = 64, 8, 20
    index = QuantizedIndex(dim, bits=4)
    centers = rng.standard_normal((n_clusters, dim)) * 3
    for ci in range(n_clusters):
        for j in range(per_cluster):
            vec = centers[ci] + 0.3 * rng.standard_normal(dim)
            index.add(vec, meta=ci)
    hits = 0
    for ci in range(n_clusters):
        query = centers[ci] + 0.3 * rng.standard_normal(dim)
        results = index.search(query, k=5)
        top_labels = [m for _, m in results]
        if top_labels and top_labels[0] == ci:
            hits += 1
    assert hits >= 7  # near-perfect nearest-cluster retrieval


def test_index_remove(rng):
    index = QuantizedIndex(16, bits=4)
    ids = [index.add(rng.standard_normal(16), meta=f"m{i}") for i in range(5)]
    index.remove_ids({ids[0], ids[1]})
    assert len(index) == 3
