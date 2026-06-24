"""Interval type-2 (IT2) fuzzy sets, inference, and type reduction.

An IT2 set is bounded by two type-1 membership functions — an upper (UMF) and a
lower (LMF) — whose gap is the *footprint of uncertainty* (FOU). Membership at a
point is therefore an interval ``[LMF(x), UMF(x)]`` rather than a single number.

IT2 rules reuse the very same operator syntax as type-1 rules (``|``, ``&``,
``~``); the engines here propagate membership *intervals* through the antecedent
tree and collapse the result with Karnik-Mendel type reduction.
"""

from .inference import IT2TSK, IT2Mamdani
from .reduction import centroid_it2, karnik_mendel, km_endpoint
from .sets import (
    IntervalType2MF,
    it2,
    it2_gauss_uncertain_mean,
    it2_gauss_uncertain_std,
    it2_scale,
)

__all__ = [
    "IntervalType2MF",
    "it2", "it2_scale", "it2_gauss_uncertain_mean", "it2_gauss_uncertain_std",
    "IT2Mamdani", "IT2TSK",
    "karnik_mendel", "km_endpoint", "centroid_it2",
]
