"""Fuzzy numbers and their arithmetic.

A fuzzy number is a convex, normal fuzzy set on the reals. This module provides
the two most used shapes — triangular (TFN) and trapezoidal (TrFN) — with
standard fuzzy arithmetic (``+ - * /`` and scalar scaling), alpha-cuts, a crisp
(centroid) value, and a vertex distance used by fuzzy MCDM (see
:mod:`fuzzytool.mcdm`).

Multiplication and division use the common positive-support approximation
(operate on the ordered defining points), which is exact for ``+``/``-`` and a
good approximation for ``*``/``/`` when supports are positive.
"""

from __future__ import annotations

from numbers import Real

import numpy as np


class FuzzyNumber:
    """Base class for fuzzy numbers defined by ordered points ``p``."""

    points: tuple[float, ...]

    def __init__(self, *points: float) -> None:
        pts = tuple(float(x) for x in points)
        if list(pts) != sorted(pts):
            raise ValueError(f"fuzzy-number points must be non-decreasing, got {pts}")
        self.points = pts

    # subclasses implement the membership and the crisp value
    def __call__(self, x):  # pragma: no cover - overridden
        raise NotImplementedError

    def centroid(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def alpha_cut(self, alpha: float) -> tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    def _binary(self, other, op, neg=False):
        cls = type(self)
        if isinstance(other, Real):
            other = cls(*([float(other)] * len(self.points)))
        if not isinstance(other, cls):
            return NotImplemented
        b = other.points[::-1] if neg else other.points
        return cls(*[op(a, c) for a, c in zip(self.points, b)])

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, neg=True)

    def __mul__(self, other):
        if isinstance(other, Real):
            k = float(other)
            pts = [k * p for p in self.points]
            return type(self)(*(pts if k >= 0 else pts[::-1]))
        return self._binary(other, lambda a, b: a * b)

    __rmul__ = __mul__
    __radd__ = __add__

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b, neg=True)

    def distance(self, other: FuzzyNumber) -> float:
        """Vertex distance to another fuzzy number of the same shape."""
        if type(self) is not type(other):
            raise TypeError("distance requires fuzzy numbers of the same shape")
        a = np.asarray(self.points)
        b = np.asarray(other.points)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.points == other.points

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.points))


class TriangularFuzzyNumber(FuzzyNumber):
    """Triangular fuzzy number ``(a, b, c)`` with peak at ``b``."""

    def __init__(self, a: float, b: float, c: float) -> None:
        super().__init__(a, b, c)
        self.a, self.b, self.c = self.points

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        left = np.divide(x - self.a, self.b - self.a, out=np.zeros_like(x),
                         where=self.b != self.a)
        right = np.divide(self.c - x, self.c - self.b, out=np.zeros_like(x),
                          where=self.c != self.b)
        left = np.where(self.b == self.a, (x >= self.a).astype(float), left)
        right = np.where(self.c == self.b, (x <= self.c).astype(float), right)
        return np.clip(np.minimum(left, right), 0.0, 1.0)

    def centroid(self) -> float:
        return (self.a + self.b + self.c) / 3.0

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        return (self.a + alpha * (self.b - self.a),
                self.c - alpha * (self.c - self.b))

    def __repr__(self) -> str:
        return f"TFN({self.a}, {self.b}, {self.c})"


class TrapezoidalFuzzyNumber(FuzzyNumber):
    """Trapezoidal fuzzy number ``(a, b, c, d)`` with flat top ``[b, c]``."""

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        super().__init__(a, b, c, d)
        self.a, self.b, self.c, self.d = self.points

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        left = np.divide(x - self.a, self.b - self.a, out=np.ones_like(x),
                         where=self.b != self.a)
        right = np.divide(self.d - x, self.d - self.c, out=np.ones_like(x),
                          where=self.d != self.c)
        return np.clip(np.minimum.reduce([left, np.ones_like(x), right]), 0.0, 1.0)

    def centroid(self) -> float:
        a, b, c, d = self.points
        # Centroid of a trapezoid (handles the triangular degenerate case).
        num = (d ** 2 + c ** 2 + c * d) - (a ** 2 + b ** 2 + a * b)
        den = 3.0 * ((c + d) - (a + b))
        return num / den if den != 0 else (a + b + c + d) / 4.0

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        return (self.a + alpha * (self.b - self.a),
                self.d - alpha * (self.d - self.c))

    def __repr__(self) -> str:
        return f"TrFN({self.a}, {self.b}, {self.c}, {self.d})"


def tfn(a: float, b: float, c: float) -> TriangularFuzzyNumber:
    """Shortcut for :class:`TriangularFuzzyNumber`."""
    return TriangularFuzzyNumber(a, b, c)


def trfn(a: float, b: float, c: float, d: float) -> TrapezoidalFuzzyNumber:
    """Shortcut for :class:`TrapezoidalFuzzyNumber`."""
    return TrapezoidalFuzzyNumber(a, b, c, d)


def rank(numbers: list[FuzzyNumber]) -> list[int]:
    """Indices that sort ``numbers`` from largest to smallest by centroid."""
    return sorted(range(len(numbers)), key=lambda i: numbers[i].centroid(), reverse=True)


__all__ = [
    "FuzzyNumber", "TriangularFuzzyNumber", "TrapezoidalFuzzyNumber",
    "tfn", "trfn", "rank",
]
