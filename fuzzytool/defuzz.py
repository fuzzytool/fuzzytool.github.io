"""Defuzzification: collapse an aggregated output set into a crisp value.

Each defuzzifier takes the discretized universe ``x`` and the aggregated
membership ``y`` (same shape) and returns a scalar. They are looked up by name
through :func:`get_defuzzifier`, so adding a method = registering a function.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def centroid(x: np.ndarray, y: np.ndarray) -> float:
    """Center of gravity of the area under ``y`` (the most common choice)."""
    total = _trapezoid(y, x)
    if total == 0:
        return float(x[len(x) // 2])  # no rule fired: fall back to mid-universe
    return float(_trapezoid(x * y, x) / total)


def bisector(x: np.ndarray, y: np.ndarray) -> float:
    """Abscissa that splits the area under ``y`` into two equal halves."""
    # Cumulative area via the trapezoidal rule.
    dx = np.diff(x)
    seg = (y[:-1] + y[1:]) / 2.0 * dx
    total = seg.sum()
    if total == 0:
        return float(x[len(x) // 2])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    idx = int(np.searchsorted(cum, total / 2.0))
    return float(x[min(idx, len(x) - 1)])


def mom(x: np.ndarray, y: np.ndarray) -> float:
    """Mean of maxima."""
    peak = y.max()
    if peak == 0:
        return float(x[len(x) // 2])
    return float(x[np.isclose(y, peak)].mean())


def som(x: np.ndarray, y: np.ndarray) -> float:
    """Smallest of maxima."""
    peak = y.max()
    if peak == 0:
        return float(x[len(x) // 2])
    return float(x[np.isclose(y, peak)].min())


def lom(x: np.ndarray, y: np.ndarray) -> float:
    """Largest of maxima."""
    peak = y.max()
    if peak == 0:
        return float(x[len(x) // 2])
    return float(x[np.isclose(y, peak)].max())


_METHODS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "centroid": centroid,
    "bisector": bisector,
    "mom": mom,
    "som": som,
    "lom": lom,
}


def get_defuzzifier(name: str | Callable) -> Callable:
    """Resolve a defuzzifier by name (or pass a callable through unchanged)."""
    if callable(name):
        return name
    try:
        return _METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown defuzzifier {name!r}; options: {sorted(_METHODS)}"
        ) from None


__all__ = ["centroid", "bisector", "mom", "som", "lom", "get_defuzzifier"]
