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
    """Sigmoidal MF: ``1 / (1 + exp(-a (x - c)))``. Monotonic (invertible)."""

    def __init__(self, a: float, c: float) -> None:
        if a == 0:
            raise ValueError("sigmoid requires a != 0 to be invertible")
        self.a, self.c = float(a), float(c)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-self.a * (x - self.c)))

    def inverse(self, y: float) -> float:
        """The ``x`` at which membership equals ``y`` (for Tsukamoto inference)."""
        y = min(max(float(y), 1e-9), 1 - 1e-9)
        return self.c + np.log(y / (1.0 - y)) / self.a

    def __repr__(self) -> str:
        return f"sigmoid({self.a}, {self.c})"


class RampUp:
    """Monotonically increasing ramp: 0 below ``a``, 1 above ``b``."""

    def __init__(self, a: float, b: float) -> None:
        if a >= b:
            raise ValueError(f"ramp_up requires a < b, got {(a, b)}")
        self.a, self.b = float(a), float(b)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - self.a) / (self.b - self.a), 0.0, 1.0)

    def inverse(self, y: float) -> float:
        return self.a + float(y) * (self.b - self.a)

    def __repr__(self) -> str:
        return f"ramp_up({self.a}, {self.b})"


class RampDown:
    """Monotonically decreasing ramp: 1 below ``a``, 0 above ``b``."""

    def __init__(self, a: float, b: float) -> None:
        if a >= b:
            raise ValueError(f"ramp_down requires a < b, got {(a, b)}")
        self.a, self.b = float(a), float(b)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((self.b - x) / (self.b - self.a), 0.0, 1.0)

    def inverse(self, y: float) -> float:
        return self.b - float(y) * (self.b - self.a)

    def __repr__(self) -> str:
        return f"ramp_down({self.a}, {self.b})"


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


def ramp_up(a: float, b: float) -> RampUp:
    """Increasing ramp MF from 0 at ``a`` to 1 at ``b`` (monotonic)."""
    return RampUp(a, b)


def ramp_down(a: float, b: float) -> RampDown:
    """Decreasing ramp MF from 1 at ``a`` to 0 at ``b`` (monotonic)."""
    return RampDown(a, b)


# --- serialization --------------------------------------------------------

# type tag -> (class, ordered attribute names)
_REGISTRY = {
    "tri": (Triangular, ("a", "b", "c")),
    "trap": (Trapezoidal, ("a", "b", "c", "d")),
    "gauss": (Gaussian, ("c", "sigma")),
    "gbell": (GeneralizedBell, ("a", "b", "c")),
    "sigmoid": (Sigmoid, ("a", "c")),
    "ramp_up": (RampUp, ("a", "b")),
    "ramp_down": (RampDown, ("a", "b")),
}
_TYPE_BY_CLASS = {cls: tag for tag, (cls, _) in _REGISTRY.items()}


def to_dict(mf) -> dict:
    """Serialize a built-in membership function to a JSON-ready dict."""
    tag = _TYPE_BY_CLASS.get(type(mf))
    if tag is None:
        raise TypeError(f"cannot serialize membership function {mf!r}")
    attrs = _REGISTRY[tag][1]
    return {"type": tag, "params": [getattr(mf, a) for a in attrs]}


def from_dict(d: dict):
    """Rebuild a membership function from :func:`to_dict` output."""
    try:
        cls = _REGISTRY[d["type"]][0]
    except KeyError:
        raise ValueError(f"unknown membership type {d.get('type')!r}") from None
    return cls(*d["params"])


__all__ = [
    "MembershipFunction",
    "Triangular", "Trapezoidal", "Gaussian", "GeneralizedBell", "Sigmoid",
    "RampUp", "RampDown",
    "tri", "trap", "gauss", "gbell", "sigmoid", "ramp_up", "ramp_down",
    "to_dict", "from_dict",
]
