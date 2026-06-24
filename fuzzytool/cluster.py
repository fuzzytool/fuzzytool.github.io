"""Fuzzy clustering.

Unlike crisp k-means, fuzzy clustering lets each sample belong to several
clusters with graded membership. This module provides three algorithms and a
small set of validity metrics, all on plain NumPy arrays of shape
``(n_samples, n_features)``:

* :func:`fuzzy_cmeans` — Bezdek's FCM (spherical clusters, Euclidean norm).
* :func:`gustafson_kessel` — GK, an adaptive norm per cluster that captures
  ellipsoidal shapes.
* :func:`possibilistic_cmeans` — PCM, which drops the "memberships sum to 1"
  constraint so outliers get low typicality in every cluster.

Each returns a :class:`ClusterResult`. Every algorithm accepts a ``seed`` for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


@dataclass
class ClusterResult:
    """Outcome of a fuzzy clustering run.

    Attributes:
        centers: cluster prototypes, shape ``(c, n_features)``.
        u: membership/typicality matrix, shape ``(c, n_samples)``.
        n_iter: iterations run until convergence.
        objective: final value of the algorithm's objective function.
        labels: hard assignment ``argmax`` over ``u``, shape ``(n_samples,)``.
    """

    centers: np.ndarray
    u: np.ndarray
    n_iter: int
    objective: float
    labels: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.labels = self.u.argmax(axis=0)


def _check(X: np.ndarray, c: int, m: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if not 1 <= c <= X.shape[0]:
        raise ValueError(f"need 1 <= c <= n_samples, got c={c}, n={X.shape[0]}")
    if m <= 1.0:
        raise ValueError(f"fuzziness m must be > 1, got {m}")
    return X


def _init_u(c: int, n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((c, n))
    return u / u.sum(axis=0, keepdims=True)


def _centers(X: np.ndarray, u: np.ndarray, m: float) -> np.ndarray:
    um = u ** m
    return (um @ X) / np.fmax(um.sum(axis=1, keepdims=True), _EPS)


def _dist2(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance, shape ``(c, n)``."""
    return np.sum((centers[:, None, :] - X[None, :, :]) ** 2, axis=2)


def _update_u(d2: np.ndarray, m: float) -> np.ndarray:
    d2 = np.fmax(d2, _EPS)
    inv = d2 ** (-1.0 / (m - 1.0))
    return inv / inv.sum(axis=0, keepdims=True)


def fuzzy_cmeans(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 150,
                 tol: float = 1e-5, seed: int | None = None) -> ClusterResult:
    """Bezdek's fuzzy c-means.

    Args:
        X: data, shape ``(n_samples, n_features)``.
        c: number of clusters.
        m: fuzziness exponent (``> 1``; ``2.0`` is standard).
        max_iter: maximum iterations.
        tol: stop when the membership matrix changes by less than this (max-norm).
        seed: RNG seed for the membership initialization.
    """
    X = _check(X, c, m)
    rng = np.random.default_rng(seed)
    u = _init_u(c, X.shape[0], rng)
    centers = _centers(X, u, m)
    j = np.inf
    it = 0
    while it < max_iter:
        it += 1
        centers = _centers(X, u, m)
        d2 = _dist2(X, centers)
        u_new = _update_u(d2, m)
        j = float((u_new ** m * d2).sum())
        if np.abs(u_new - u).max() < tol:
            u = u_new
            break
        u = u_new
    return ClusterResult(centers, u, it, j)


def gustafson_kessel(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 150,
                     tol: float = 1e-5, seed: int | None = None,
                     reg: float = 1e-10) -> ClusterResult:
    """Gustafson-Kessel clustering (adaptive per-cluster Mahalanobis norm).

    Each cluster learns a covariance-shaped, unit-determinant norm, so GK fits
    ellipsoidal clusters that FCM (fixed spherical norm) cannot.

    Args:
        X: data, shape ``(n_samples, n_features)``.
        c: number of clusters.
        m: fuzziness exponent (``> 1``).
        max_iter: maximum iterations.
        tol: convergence threshold on the membership matrix (max-norm).
        seed: RNG seed for initialization.
        reg: ridge added to each fuzzy covariance for numerical stability.
    """
    X = _check(X, c, m)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    u = _init_u(c, n, rng)
    centers = _centers(X, u, m)
    eye = np.eye(p)
    j = np.inf
    it = 0
    while it < max_iter:
        it += 1
        centers = _centers(X, u, m)
        um = u ** m
        d2 = np.empty((c, n))
        for i in range(c):
            diff = X - centers[i]                                   # (n, p)
            cov = np.einsum("k,kp,kq->pq", um[i], diff, diff) / max(um[i].sum(), _EPS)
            cov = cov + reg * eye
            det = np.linalg.det(cov)
            if det <= 0 or not np.isfinite(det):
                a = eye                                             # fall back to Euclidean
            else:
                a = (det ** (1.0 / p)) * np.linalg.inv(cov)
            d2[i] = np.einsum("kp,pq,kq->k", diff, a, diff)
        u_new = _update_u(d2, m)
        j = float((u_new ** m * d2).sum())
        if np.abs(u_new - u).max() < tol:
            u = u_new
            break
        u = u_new
    return ClusterResult(centers, u, it, j)


def possibilistic_cmeans(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 150,
                         tol: float = 1e-5, seed: int | None = None,
                         k: float = 1.0) -> ClusterResult:
    """Possibilistic c-means (Krishnapuram-Keller).

    PCM drops the probabilistic constraint that memberships sum to 1: each value
    is a *typicality* in ``[0, 1]``, so noise points score low everywhere. It is
    initialized from an FCM run, which also fixes each cluster's scale ``eta``.

    Args:
        X: data, shape ``(n_samples, n_features)``.
        c: number of clusters.
        m: fuzziness exponent (``> 1``).
        max_iter: maximum iterations.
        tol: convergence threshold on the typicality matrix (max-norm).
        seed: RNG seed (passed to the initializing FCM).
        k: scale multiplier for the bandwidth ``eta`` (``1.0`` is standard).
    """
    X = _check(X, c, m)
    init = fuzzy_cmeans(X, c, m=m, max_iter=max_iter, tol=tol, seed=seed)
    centers, u = init.centers, init.u
    d2 = _dist2(X, centers)
    um = u ** m
    eta = k * (um * d2).sum(axis=1) / np.fmax(um.sum(axis=1), _EPS)   # (c,)
    eta = np.fmax(eta, _EPS)
    j = np.inf
    it = 0
    while it < max_iter:
        it += 1
        d2 = _dist2(X, centers)
        t = 1.0 / (1.0 + (d2 / eta[:, None]) ** (1.0 / (m - 1.0)))
        centers_new = _centers(X, t, m)
        j = float((t ** m * d2).sum() + (eta[:, None] * (1.0 - t) ** m).sum())
        if np.abs(centers_new - centers).max() < tol:
            centers = centers_new
            u = t
            break
        centers, u = centers_new, t
    return ClusterResult(centers, u, it, j)


# --- validity metrics ------------------------------------------------------

def partition_coefficient(u: np.ndarray) -> float:
    """Bezdek's partition coefficient in ``(1/c, 1]``; higher = crisper."""
    u = np.asarray(u, dtype=float)
    return float((u ** 2).sum() / u.shape[1])


def partition_entropy(u: np.ndarray) -> float:
    """Partition entropy in ``[0, log c)``; lower = crisper."""
    u = np.asarray(u, dtype=float)
    return float(-(u * np.log(np.fmax(u, _EPS))).sum() / u.shape[1])


def xie_beni(X, centers: np.ndarray, u: np.ndarray, m: float = 2.0) -> float:
    """Xie-Beni index (compactness / separation); lower is better."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    u = np.asarray(u, dtype=float)
    compactness = (u ** m * _dist2(X, centers)).sum()
    cc = _dist2(centers, centers)
    sep = cc[~np.eye(centers.shape[0], dtype=bool)].min()
    return float(compactness / (X.shape[0] * np.fmax(sep, _EPS)))


__all__ = [
    "ClusterResult",
    "fuzzy_cmeans", "gustafson_kessel", "possibilistic_cmeans",
    "partition_coefficient", "partition_entropy", "xie_beni",
]
