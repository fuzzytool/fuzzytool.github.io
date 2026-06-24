"""Interval type-2 (IT2) membership functions.

An :class:`IntervalType2MF` carries a lower (LMF) and an upper (UMF) type-1
membership function. Calling it returns the membership *interval*
``(lower, upper)``; the engines also call ``.lower(x)`` / ``.upper(x)`` directly.

The constructors cover the standard ways of building an FOU from a type-1 set:

* :func:`it2` — explicit LMF/UMF;
* :func:`it2_scale` — height uncertainty (LMF is a scaled-down UMF);
* :func:`it2_gauss_uncertain_mean` — a Gaussian with mean in ``[c1, c2]``;
* :func:`it2_gauss_uncertain_std` — a Gaussian with uncertain spread.
"""

from __future__ import annotations

import numpy as np

from ..membership import Gaussian, MembershipFunction


class IntervalType2MF:
    """An IT2 set defined by a lower (LMF) and an upper (UMF) type-1 MF.

    The UMF must dominate the LMF everywhere (``LMF(x) <= UMF(x)``); this is not
    enforced at construction (it would require sampling the universe) but holds
    for every set built through the constructors below.
    """

    def __init__(self, lower: MembershipFunction, upper: MembershipFunction) -> None:
        if not callable(lower) or not callable(upper):
            raise TypeError("lower and upper must be callable membership functions")
        self._lower = lower
        self._upper = upper

    def lower(self, x):
        return np.asarray(self._lower(x), dtype=float)

    def upper(self, x):
        return np.asarray(self._upper(x), dtype=float)

    def __call__(self, x):
        return self.lower(x), self.upper(x)

    def __repr__(self) -> str:
        return f"IT2(lower={self._lower!r}, upper={self._upper!r})"


def it2(lower: MembershipFunction, upper: MembershipFunction) -> IntervalType2MF:
    """An IT2 set from explicit lower and upper type-1 membership functions."""
    return IntervalType2MF(lower, upper)


def it2_scale(mf: MembershipFunction, scale: float) -> IntervalType2MF:
    """Height-uncertainty FOU: UMF is ``mf``, LMF is ``scale * mf`` (0 < scale ≤ 1)."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    lower = lambda x: scale * np.asarray(mf(x), dtype=float)  # noqa: E731
    return IntervalType2MF(lower, mf)


def it2_gauss_uncertain_mean(c1: float, c2: float, sigma: float) -> IntervalType2MF:
    """Gaussian with an *uncertain mean* spanning ``[c1, c2]`` (c1 < c2).

    The UMF is the upper envelope of all Gaussians with mean in ``[c1, c2]``
    (flat-topped at 1 between the means); the LMF is their lower envelope.
    """
    if c1 >= c2:
        raise ValueError(f"need c1 < c2, got ({c1}, {c2})")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    g1, g2 = Gaussian(c1, sigma), Gaussian(c2, sigma)
    mid = (c1 + c2) / 2.0

    def upper(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < c1, g1(x), np.where(x > c2, g2(x), 1.0))

    def lower(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= mid, g2(x), g1(x))

    return IntervalType2MF(lower, upper)


def it2_gauss_uncertain_std(c: float, sigma_lo: float, sigma_hi: float) -> IntervalType2MF:
    """Gaussian with an *uncertain spread*: UMF uses the wider σ, LMF the narrower."""
    if not 0 < sigma_lo < sigma_hi:
        raise ValueError(f"need 0 < sigma_lo < sigma_hi, got ({sigma_lo}, {sigma_hi})")
    return IntervalType2MF(Gaussian(c, sigma_lo), Gaussian(c, sigma_hi))


__all__ = [
    "IntervalType2MF",
    "it2", "it2_scale", "it2_gauss_uncertain_mean", "it2_gauss_uncertain_std",
]
