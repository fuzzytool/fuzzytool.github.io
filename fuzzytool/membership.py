"""Membership functions.

Every membership function (MF) is a *callable* mapping a crisp value (scalar or
NumPy array) to a membership degree in ``[0, 1]``. The :class:`MembershipFunction`
Protocol is the only contract the rest of the library relies on: the inference
engine never inspects a concrete MF type, so a new shape = a new callable, with
no changes to the core (mirroring the "one variant = one impl" design of the
sibling project ``turboswarm``).

The lowercase factory functions (:func:`tri`, :func:`trap`, :func:`gauss`,
:func:`gbell`, :func:`sigmoid`) are the public API; they return small classes so
the parameters stay introspectable for visualization and serialization.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MembershipFunction(Protocol):
    """A callable ``x -> degree`` with membership degrees in ``[0, 1]``."""

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


class Triangular:
    """Triangular MF with feet at ``a``/``c`` and peak at ``b``."""

    def __init__(self, a: float, b: float, c: float) -> None:
        if not a <= b <= c:
            raise ValueError(f"triangular requires a <= b <= c, got {(a, b, c)}")
        self.a, self.b, self.c = float(a), float(b), float(c)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        left = np.divide(x - self.a, self.b - self.a, out=np.zeros_like(x),
                         where=self.b != self.a)
        right = np.divide(self.c - x, self.c - self.b, out=np.zeros_like(x),
                          where=self.c != self.b)
        # Degenerate edges: a flat shoulder where two points coincide.
        left = np.where(self.b == self.a, (x >= self.a).astype(float), left)
        right = np.where(self.c == self.b, (x <= self.c).astype(float), right)
        return np.clip(np.minimum(left, right), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"tri({self.a}, {self.b}, {self.c})"


class Trapezoidal:
    """Trapezoidal MF; flat top between ``b`` and ``c``."""

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        if not a <= b <= c <= d:
            raise ValueError(f"trapezoid requires a <= b <= c <= d, got {(a, b, c, d)}")
        self.a, self.b, self.c, self.d = map(float, (a, b, c, d))

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        left = np.divide(x - self.a, self.b - self.a, out=np.ones_like(x),
                         where=self.b != self.a)
        right = np.divide(self.d - x, self.d - self.c, out=np.ones_like(x),
                          where=self.d != self.c)
        return np.clip(np.minimum.reduce([left, np.ones_like(x), right]), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"trap({self.a}, {self.b}, {self.c}, {self.d})"


class Gaussian:
    """Gaussian MF centered at ``c`` with spread ``sigma``."""

    def __init__(self, c: float, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError(f"gauss requires sigma > 0, got {sigma}")
        self.c, self.sigma = float(c), float(sigma)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.c) / self.sigma) ** 2)

    def __repr__(self) -> str:
        return f"gauss({self.c}, {self.sigma})"


class GeneralizedBell:
    """Generalized bell MF: ``1 / (1 + |(x - c) / a|^(2b))``."""

    def __init__(self, a: float, b: float, c: float) -> None:
        if a == 0:
            raise ValueError("gbell requires a != 0")
        self.a, self.b, self.c = float(a), float(b), float(c)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.abs((x - self.c) / self.a) ** (2.0 * self.b))

    def __repr__(self) -> str:
        return f"gbell({self.a}, {self.b}, {self.c})"


class Sigmoid:
    """Sigmoidal MF: ``1 / (1 + exp(-a (x - c)))``."""

    def __init__(self, a: float, c: float) -> None:
        self.a, self.c = float(a), float(c)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-self.a * (x - self.c)))

    def __repr__(self) -> str:
        return f"sigmoid({self.a}, {self.c})"


# --- factory shortcuts (public API) ---------------------------------------

def tri(a: float, b: float, c: float) -> Triangular:
    """Triangular MF: feet at ``a``/``c``, peak at ``b``."""
    return Triangular(a, b, c)


def trap(a: float, b: float, c: float, d: float) -> Trapezoidal:
    """Trapezoidal MF: shoulders ``a``/``d``, flat top ``b``..``c``."""
    return Trapezoidal(a, b, c, d)


def gauss(c: float, sigma: float) -> Gaussian:
    """Gaussian MF: center ``c``, spread ``sigma``."""
    return Gaussian(c, sigma)


def gbell(a: float, b: float, c: float) -> GeneralizedBell:
    """Generalized bell MF: width ``a``, slope ``b``, center ``c``."""
    return GeneralizedBell(a, b, c)


def sigmoid(a: float, c: float) -> Sigmoid:
    """Sigmoidal MF: slope ``a``, inflection ``c``."""
    return Sigmoid(a, c)


__all__ = [
    "MembershipFunction",
    "Triangular", "Trapezoidal", "Gaussian", "GeneralizedBell", "Sigmoid",
    "tri", "trap", "gauss", "gbell", "sigmoid",
]
