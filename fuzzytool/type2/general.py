"""General type-2 (GT2) fuzzy sets via the zSlices / alpha-plane representation.

An interval type-2 set treats every point inside its footprint of uncertainty
(FOU) as equally possible. A **general** type-2 set adds a *secondary* membership
that weights those possibilities — the third dimension IT2 throws away.

A practical, exact way to handle that third dimension is the **zSlices** (alpha
-plane) decomposition (Wagner & Hagras): slice the secondary domain at levels
``z`` and, at each level, the GT2 set reduces to an ordinary IT2 set whose FOU is
the ``z``-cut. Inference and type reduction then run the existing IT2 machinery
on each slice and combine the results, weighted by ``z``.

Here a GT2 set is built from an IT2 footprint plus a **triangular secondary**
peaking at the FOU's principal (mid) MF: at ``z -> 0`` a slice is the full FOU,
at ``z = 1`` it collapses to the principal MF.

* :class:`GeneralType2MF` — a stack of IT2 z-slices (also exposes the overall
  FOU via ``lower``/``upper``, so it degrades gracefully to IT2).
* :func:`gt2_from_it2`, :func:`gt2_gauss_uncertain_mean`, :func:`gt2_scale` —
  constructors.
* :class:`GeneralType2Mamdani` — z-weighted stack of IT2 Mamdani inferences.
* :func:`centroid_gt2` — zSlices centroid of a single GT2 set.
"""

from __future__ import annotations

import numpy as np

from ..sets import Variable
from .inference import IT2Mamdani
from .reduction import centroid_it2
from .sets import IntervalType2MF, it2


class GeneralType2MF:
    """A general type-2 set represented as a stack of IT2 ``z``-slices.

    Args:
        slices: IT2 sets, one per ``z`` level, narrowing toward the principal MF
            as ``z`` increases.
        zlevels: the secondary levels (in ``(0, 1]``), ascending, aligned with
            ``slices``.
        footprint: the overall FOU as an IT2 set (used for ``lower``/``upper`` so
            the GT2 set can stand in for its IT2 footprint).
    """

    def __init__(self, slices: list[IntervalType2MF], zlevels: np.ndarray,
                 footprint: IntervalType2MF) -> None:
        self.slices = list(slices)
        self.zlevels = np.asarray(zlevels, dtype=float)
        self._footprint = footprint

    @property
    def n_slices(self) -> int:
        return len(self.slices)

    def slice(self, i: int) -> IntervalType2MF:
        """The IT2 set at ``z``-level index ``i``."""
        return self.slices[i]

    def lower(self, x):
        return self._footprint.lower(x)

    def upper(self, x):
        return self._footprint.upper(x)

    def __call__(self, x):
        return self._footprint(x)

    def __repr__(self) -> str:
        return f"GeneralType2MF(n_slices={self.n_slices})"


def gt2_from_it2(footprint: IntervalType2MF, n_slices: int = 5) -> GeneralType2MF:
    """Build a GT2 set from an IT2 FOU with a triangular secondary MF.

    The secondary membership peaks at the principal MF (the midline of the FOU)
    and falls linearly to 0 at the LMF/UMF. The ``z``-slice at level ``z`` is the
    IT2 set whose FOU is narrowed from the full footprint toward that midline by
    a factor ``(1 - z)``.

    Args:
        footprint: the IT2 set describing the full footprint of uncertainty.
        n_slices: number of ``z`` levels (``z = 1/n .. 1``). More slices = a finer
            secondary, at proportional cost.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    zlevels = np.linspace(1.0 / n_slices, 1.0, n_slices)

    def mid(x):
        return (footprint.lower(x) + footprint.upper(x)) / 2.0

    slices = []
    for z in zlevels:
        # Default-arg binding captures the current z in each closure.
        def lower(x, z=z):
            m = mid(x)
            return m - (1.0 - z) * (m - footprint.lower(x))

        def upper(x, z=z):
            m = mid(x)
            return m + (1.0 - z) * (footprint.upper(x) - m)

        slices.append(it2(lower, upper))
    return GeneralType2MF(slices, zlevels, footprint)


def gt2_gauss_uncertain_mean(c1: float, c2: float, sigma: float,
                             n_slices: int = 5) -> GeneralType2MF:
    """GT2 Gaussian with an uncertain mean in ``[c1, c2]`` (triangular secondary)."""
    from .sets import it2_gauss_uncertain_mean
    return gt2_from_it2(it2_gauss_uncertain_mean(c1, c2, sigma), n_slices)


def gt2_scale(mf, scale: float, n_slices: int = 5) -> GeneralType2MF:
    """GT2 set from a height-uncertainty FOU (UMF ``mf``, LMF ``scale * mf``)."""
    from .sets import it2_scale
    return gt2_from_it2(it2_scale(mf, scale), n_slices)


def centroid_gt2(gt2mf: GeneralType2MF, universe) -> float:
    """zSlices centroid of a GT2 set: ``z``-weighted mean of per-slice IT2 centroids."""
    z = gt2mf.zlevels
    mids = np.array([sum(centroid_it2(gt2mf.slice(i), universe)) / 2.0
                     for i in range(gt2mf.n_slices)])
    return float(np.sum(z * mids) / np.sum(z))


def _antecedent_variables(node, out: dict) -> None:
    if hasattr(node, "variable"):
        out.setdefault(id(node.variable), node.variable)
    if hasattr(node, "left"):
        _antecedent_variables(node.left, out)
        _antecedent_variables(node.right, out)
    if hasattr(node, "operand"):
        _antecedent_variables(node.operand, out)


class GeneralType2Mamdani:
    """General type-2 Mamdani inference via the zSlices decomposition.

    At each ``z`` level the GT2 terms collapse to their IT2 ``z``-slice and the
    existing :class:`~fuzzytool.type2.inference.IT2Mamdani` engine
    (center-of-sets type reduction) produces a crisp output ``y(z)``. The final
    result is the ``z``-weighted average ``Σ z·y(z) / Σ z`` — the centroid of the
    type-reduced general type-2 output.

    Variables may mix GT2 and IT2/type-1 terms; only the GT2 terms are sliced.
    All GT2 terms must share the same ``z`` levels (same ``n_slices``).
    """

    def __init__(self, tnorm="min", snorm="max") -> None:
        self._it2 = IT2Mamdani(tnorm, snorm)

    def rule(self, antecedent, consequent, weight: float = 1.0) -> GeneralType2Mamdani:
        """Add ``IF antecedent THEN output is term`` and return ``self``."""
        self._it2.rule(antecedent, consequent, weight)
        return self

    @property
    def rules(self):
        return self._it2.rules

    def _gt2_variables(self) -> list[Variable]:
        found: dict[int, Variable] = {}
        for r in self._it2.rules:
            _antecedent_variables(r.antecedent, found)
            found.setdefault(id(r.consequent.variable), r.consequent.variable)
        return [v for v in found.values()
                if any(isinstance(t, GeneralType2MF) for t in v.terms.values())]

    def __call__(self, **inputs: float):
        """Run inference. Returns a float for one output, else a dict by name."""
        if not self._it2.rules:
            raise RuntimeError("no rules defined")
        variables = self._gt2_variables()
        if not variables:
            raise RuntimeError("no general type-2 terms found; use IT2Mamdani instead")
        zlevels = next(t.zlevels for v in variables for t in v.terms.values()
                       if isinstance(t, GeneralType2MF))

        z_acc, y_acc = [], []
        for i, z in enumerate(zlevels):
            saved = {id(v): v.terms for v in variables}
            try:
                for v in variables:
                    v.terms = {name: (t.slice(i) if isinstance(t, GeneralType2MF) else t)
                               for name, t in v.terms.items()}
                self._it2._centroids.clear()
                y = self._it2(**inputs)
            finally:
                for v in variables:
                    v.terms = saved[id(v)]
            z_acc.append(float(z))
            y_acc.append(y)

        z = np.asarray(z_acc)
        if isinstance(y_acc[0], dict):
            return {k: float(np.sum(z * np.array([y[k] for y in y_acc])) / np.sum(z))
                    for k in y_acc[0]}
        return float(np.sum(z * np.asarray(y_acc)) / np.sum(z))

    def __repr__(self) -> str:
        return f"GeneralType2Mamdani(rules={len(self._it2.rules)})"


__all__ = [
    "GeneralType2MF", "gt2_from_it2", "gt2_gauss_uncertain_mean", "gt2_scale",
    "centroid_gt2", "GeneralType2Mamdani",
]
