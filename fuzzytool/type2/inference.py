"""Interval type-2 inference engines.

Both engines evaluate each rule's antecedent to a firing *interval*
``[f_low, f_high]`` (via :meth:`Antecedent.eval_interval`) and then collapse the
rule base with Karnik-Mendel type reduction, returning the midpoint of the
type-reduced interval ``[y_l, y_r]``.

* :class:`IT2Mamdani` uses **center-of-sets** type reduction: each consequent
  IT2 set is summarized by its centroid interval (computed once and cached), and
  KM combines those centroids weighted by the firing intervals.
* :class:`IT2TSK` has crisp consequents (numbers, coefficient mappings, or
  callables) and applies KM directly to them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..inference.tsk import _as_consequent_fn
from ..norms import get_snorm, get_tnorm
from ..rules import Rule
from ..sets import Antecedent, Proposition, Variable
from .reduction import centroid_it2, karnik_mendel, km_endpoint


class IT2Mamdani:
    """Interval type-2 Mamdani inference (center-of-sets type reduction).

    Args:
        tnorm: t-norm for AND in antecedents (default ``"min"``).
        snorm: s-norm for OR in antecedents (default ``"max"``).
    """

    def __init__(self, tnorm: str | Callable = "min",
                 snorm: str | Callable = "max") -> None:
        self.tnorm = get_tnorm(tnorm)
        self.snorm = get_snorm(snorm)
        self.rules: list[Rule] = []
        self._outputs: dict[str, Variable] = {}
        self._centroids: dict[tuple[int, str], tuple[float, float]] = {}

    def rule(self, antecedent: Antecedent, consequent: Proposition,
             weight: float = 1.0) -> IT2Mamdani:
        """Add ``IF antecedent THEN output is term`` and return ``self``."""
        if not isinstance(consequent, Proposition):
            raise TypeError("IT2Mamdani consequent must be a `variable[term]` proposition")
        self.rules.append(Rule(antecedent, consequent, weight))
        self._outputs[consequent.variable.name] = consequent.variable
        return self

    def _centroid(self, var: Variable, term: str) -> tuple[float, float]:
        key = (id(var), term)
        if key not in self._centroids:
            self._centroids[key] = centroid_it2(var.terms[term], var.universe)
        return self._centroids[key]

    def __call__(self, **inputs: float):
        """Run inference. Returns a float for one output, else a dict by name."""
        if not self.rules:
            raise RuntimeError("no rules defined")
        out: dict[str, float] = {}
        for name, var in self._outputs.items():
            cl, cr, f_low, f_high = [], [], [], []
            for r in self.rules:
                if r.consequent.variable.name != name:
                    continue
                lo, hi = r.antecedent.eval_interval(inputs, self.tnorm, self.snorm)
                c_l, c_r = self._centroid(var, r.consequent.term)
                cl.append(c_l)
                cr.append(c_r)
                f_low.append(float(lo) * r.weight)
                f_high.append(float(hi) * r.weight)
            y_l = km_endpoint(np.array(cl), np.array(f_low), np.array(f_high), side="l")
            y_r = km_endpoint(np.array(cr), np.array(f_low), np.array(f_high), side="r")
            out[name] = (y_l + y_r) / 2.0
        return next(iter(out.values())) if len(out) == 1 else out

    def __repr__(self) -> str:
        return f"IT2Mamdani(rules={len(self.rules)}, outputs={sorted(self._outputs)})"


class IT2TSK:
    """Interval type-2 Takagi-Sugeno inference (KM over crisp consequents).

    Consequents follow the same convention as the type-1 :class:`~fuzzytool.inference.tsk.TSK`
    engine: a number, a coefficient mapping ``{"const": b0, "x": b1, ...}``, or a
    callable ``f(**inputs) -> float``.
    """

    def __init__(self, tnorm: str | Callable = "min",
                 snorm: str | Callable = "max") -> None:
        self.tnorm = get_tnorm(tnorm)
        self.snorm = get_snorm(snorm)
        self.rules: list[Rule] = []
        self._fns: list[Callable[..., float]] = []

    def rule(self, antecedent: Antecedent, consequent, weight: float = 1.0) -> IT2TSK:
        """Add ``IF antecedent THEN output = consequent`` and return ``self``."""
        self.rules.append(Rule(antecedent, consequent, weight))
        self._fns.append(_as_consequent_fn(consequent))
        return self

    def __call__(self, **inputs: float) -> float:
        """Run inference, returning the midpoint of the type-reduced interval."""
        if not self.rules:
            raise RuntimeError("no rules defined")
        pts, f_low, f_high = [], [], []
        for r, fn in zip(self.rules, self._fns):
            lo, hi = r.antecedent.eval_interval(inputs, self.tnorm, self.snorm)
            pts.append(fn(**inputs))
            f_low.append(float(lo) * r.weight)
            f_high.append(float(hi) * r.weight)
        y_l, y_r = karnik_mendel(pts, f_low, f_high)
        return (y_l + y_r) / 2.0

    def __repr__(self) -> str:
        return f"IT2TSK(rules={len(self.rules)})"


__all__ = ["IT2Mamdani", "IT2TSK"]
