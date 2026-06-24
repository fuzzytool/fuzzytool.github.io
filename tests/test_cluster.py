import numpy as np
import pytest

import fuzzytool as fz
from fuzzytool import cluster
from fuzzytool.datasets import make_blobs


def _matches_centers(found, true, tol=0.6):
    """Every true center has a found center within ``tol`` (order-independent)."""
    true = np.asarray(true, dtype=float)
    for t in true:
        if np.min(np.linalg.norm(found - t, axis=1)) > tol:
            return False
    return True


TRUE = ((0.0, 0.0), (6.0, 6.0), (0.0, 6.0))


def test_fcm_recovers_blob_centers():
    X = make_blobs(centers=TRUE, seed=1)
    res = fz.fuzzy_cmeans(X, c=3, seed=1)
    assert _matches_centers(res.centers, TRUE)
    assert res.labels.shape == (X.shape[0],)


def test_fcm_memberships_sum_to_one():
    X = make_blobs(seed=2)
    res = fz.fuzzy_cmeans(X, c=3, seed=2)
    assert np.allclose(res.u.sum(axis=0), 1.0)


def test_fcm_is_reproducible_with_seed():
    X = make_blobs(seed=3)
    a = fz.fuzzy_cmeans(X, c=3, seed=42)
    b = fz.fuzzy_cmeans(X, c=3, seed=42)
    assert np.allclose(a.centers, b.centers)
    assert np.allclose(a.u, b.u)


def test_gustafson_kessel_recovers_centers():
    X = make_blobs(centers=TRUE, seed=4)
    res = fz.gustafson_kessel(X, c=3, seed=4)
    assert _matches_centers(res.centers, TRUE, tol=0.8)


def test_possibilistic_cmeans_runs_and_is_bounded():
    X = make_blobs(centers=TRUE, seed=5)
    res = fz.possibilistic_cmeans(X, c=3, seed=5)
    assert _matches_centers(res.centers, TRUE, tol=0.9)
    assert np.all((res.u >= 0) & (res.u <= 1))


def test_partition_coefficient_range():
    X = make_blobs(seed=6)
    res = fz.fuzzy_cmeans(X, c=3, seed=6)
    pc = cluster.partition_coefficient(res.u)
    assert 1.0 / 3.0 < pc <= 1.0


def test_xie_beni_positive():
    X = make_blobs(seed=7)
    res = fz.fuzzy_cmeans(X, c=3, seed=7)
    assert cluster.xie_beni(X, res.centers, res.u) > 0


def test_invalid_arguments():
    X = make_blobs(seed=8)
    with pytest.raises(ValueError):
        fz.fuzzy_cmeans(X, c=3, m=1.0)          # m must be > 1
    with pytest.raises(ValueError):
        fz.fuzzy_cmeans(X, c=X.shape[0] + 1)    # c must be <= n_samples
