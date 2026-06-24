"""Karnik-Mendel type reduction.

Type reduction turns the interval-valued output of an IT2 system into a crisp
interval ``[y_l, y_r]`` (typically defuzzified as its midpoint). The
Karnik-Mendel (KM) algorithm finds each endpoint by locating the *switch point*
where the per-point weight flips between its lower and upper bound.

The single primitive :func:`km_endpoint` is reused everywhere: for an IT2 set's
centroid (points = universe samples, weights = the FOU) and for an IT2 rule base
(points = consequent centroids, weights = rule firing intervals).
"""

from __future__ import annotations

import numpy as np


def _weighted_mean(points: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    if total == 0:
        return float(points.mean())  # nothing fired: neutral fallback
    return float((w * points).sum() / total)


def km_endpoint(points: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                side: str = "l", max_iter: int = 100) -> float:
    """One endpoint of the type-reduced interval via Karnik-Mendel.

    Args:
        points: the y-values (e.g. consequent centroids or universe samples).
        lower: per-point weight lower bounds (same shape as ``points``).
        upper: per-point weight upper bounds (same shape as ``points``).
        side: ``"l"`` for the left endpoint ``y_l``, ``"r"`` for the right ``y_r``.
        max_iter: safeguard on the Karnik-Mendel iteration count.

    Left endpoint uses the *upper* weights to the left of the switch and the
    *lower* weights to its right; the right endpoint mirrors this.
    """
    points = np.asarray(points, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if points.size == 0:
        raise ValueError("no points to type-reduce")
    if side not in ("l", "r"):
        raise ValueError("side must be 'l' or 'r'")

    order = np.argsort(points)
    p, lo, up = points[order], lower[order], upper[order]
    n = p.size
    if n == 1:
        return float(p[0])

    idx = np.arange(n)
    w = (lo + up) / 2.0
    y = _weighted_mean(p, w)
    for _ in range(max_iter):
        k = int(np.searchsorted(p, y, side="right")) - 1
        k = min(max(k, 0), n - 2)
        if side == "l":
            w = np.where(idx <= k, up, lo)
        else:
            w = np.where(idx <= k, lo, up)
        y_new = _weighted_mean(p, w)
        if np.isclose(y_new, y):
            return y_new
        y = y_new
    return y


def karnik_mendel(points, lower, upper) -> tuple[float, float]:
    """Type-reduce to the interval ``(y_l, y_r)`` over a shared set of points."""
    y_l = km_endpoint(points, lower, upper, side="l")
    y_r = km_endpoint(points, lower, upper, side="r")
    return y_l, y_r


def centroid_it2(it2mf, universe) -> tuple[float, float]:
    """Centroid interval ``(c_l, c_r)`` of an IT2 set over ``universe``."""
    x = np.asarray(universe, dtype=float)
    return karnik_mendel(x, it2mf.lower(x), it2mf.upper(x))


__all__ = ["km_endpoint", "karnik_mendel", "centroid_it2"]
