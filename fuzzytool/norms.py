"""Fuzzy connectives: t-norms (AND) and s-norms / t-conorms (OR).

Connectives are plain vectorized callables ``(a, b) -> result`` operating
elementwise on membership degrees. They are pluggable: the inference engines
look them up by name through :func:`get_tnorm` / :func:`get_snorm`, so adding a
connective means registering one function — the engine never changes.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

Degree = np.ndarray


@runtime_checkable
class Norm(Protocol):
    """A binary connective on membership degrees."""

    def __call__(self, a: Degree, b: Degree) -> Degree: ...


# --- t-norms (fuzzy AND) ---------------------------------------------------

def t_min(a, b):
    """Minimum (Gödel) t-norm."""
    return np.minimum(a, b)


def t_prod(a, b):
    """Product (algebraic) t-norm."""
    return np.asarray(a) * np.asarray(b)


def t_lukasiewicz(a, b):
    """Łukasiewicz t-norm: ``max(0, a + b - 1)``."""
    return np.maximum(0.0, np.asarray(a) + np.asarray(b) - 1.0)


# --- s-norms / t-conorms (fuzzy OR) ---------------------------------------

def s_max(a, b):
    """Maximum (Gödel) s-norm."""
    return np.maximum(a, b)


def s_probor(a, b):
    """Probabilistic OR (algebraic sum): ``a + b - a*b``."""
    a, b = np.asarray(a), np.asarray(b)
    return a + b - a * b


def s_lukasiewicz(a, b):
    """Łukasiewicz s-norm: ``min(1, a + b)``."""
    return np.minimum(1.0, np.asarray(a) + np.asarray(b))


_TNORMS: dict[str, Callable] = {
    "min": t_min,
    "prod": t_prod,
    "product": t_prod,
    "lukasiewicz": t_lukasiewicz,
}

_SNORMS: dict[str, Callable] = {
    "max": s_max,
    "probor": s_probor,
    "sum": s_probor,
    "lukasiewicz": s_lukasiewicz,
}


def get_tnorm(name: str | Callable) -> Callable:
    """Resolve a t-norm by name (or pass a callable through unchanged)."""
    if callable(name):
        return name
    try:
        return _TNORMS[name]
    except KeyError:
        raise ValueError(f"unknown t-norm {name!r}; options: {sorted(_TNORMS)}") from None


def get_snorm(name: str | Callable) -> Callable:
    """Resolve an s-norm by name (or pass a callable through unchanged)."""
    if callable(name):
        return name
    try:
        return _SNORMS[name]
    except KeyError:
        raise ValueError(f"unknown s-norm {name!r}; options: {sorted(_SNORMS)}") from None


def complement(a):
    """Standard fuzzy complement (NOT): ``1 - a``."""
    return 1.0 - np.asarray(a)


__all__ = [
    "Norm",
    "t_min", "t_prod", "t_lukasiewicz",
    "s_max", "s_probor", "s_lukasiewicz",
    "get_tnorm", "get_snorm", "complement",
]
