"""Mamdani fuzzy inference.

The engine knows nothing about specific membership functions, connectives, or
defuzzifiers: t-norm, s-norm, implication, aggregation and defuzzification are
all resolved by name (or supplied as callables) and applied uniformly. Adding a
behavior means registering a function in :mod:`fuzzytool.norms` or
:mod:`fuzzytool.defuzz`, never editing this loop.

Pipeline per call: evaluate each rule's antecedent to a firing strength →
shape its consequent term over the output universe via the *implication*
operator → *aggregate* the shaped sets per output variable → *defuzzify*.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..defuzz import get_defuzzifier
from ..norms import get_snorm, get_tnorm
from ..rules import Rule
from ..sets import Antecedent, Proposition, Variable


class Mamdani:
    """A Mamdani inference system.

    Args:
        tnorm: t-norm for AND in antecedents (default ``"min"``).
        snorm: s-norm for OR in antecedents (default ``"max"``).
        implication: how a firing strength shapes its consequent set —
            ``"min"`` (clip) or ``"prod"`` (scale).
        aggregation: s-norm combining shaped sets per output (default ``"max"``).
        defuzz: defuzzification method (default ``"centroid"``).
    """

    def __init__(
        self,
        tnorm: str | Callable = "min",
        snorm: str | Callable = "max",
        implication: str = "min",
        aggregation: str | Callable = "max",
        defuzz: str | Callable = "centroid",
    ) -> None:
        self.tnorm = get_tnorm(tnorm)
        self.snorm = get_snorm(snorm)
        if implication not in ("min", "prod"):
            raise ValueError("implication must be 'min' or 'prod'")
        self.implication = implication
        self.aggregation = get_snorm(aggregation)
        self.defuzz = get_defuzzifier(defuzz)
        self.rules: list[Rule] = []
        self._outputs: dict[str, Variable] = {}

    def rule(self, antecedent: Antecedent, consequent: Proposition,
             weight: float = 1.0) -> Mamdani:
        """Add ``IF antecedent THEN output is term`` and return ``self``."""
        if not isinstance(consequent, Proposition):
            raise TypeError("Mamdani consequent must be a `variable[term]` proposition")
        self.rules.append(Rule(antecedent, consequent, weight))
        self._outputs[consequent.variable.name] = consequent.variable
        return self

    def __call__(self, **inputs: float):
        """Run inference. Returns a float for one output, else a dict by name."""
        if not self.rules:
            raise RuntimeError("no rules defined")
        # Aggregated output membership over each output variable's universe.
        agg = {name: np.zeros_like(var.universe)
               for name, var in self._outputs.items()}

        for r in self.rules:
            firing = float(r.antecedent.eval(inputs, self.tnorm, self.snorm)) * r.weight
            if firing <= 0.0:
                continue
            var = r.consequent.variable
            shape = var.terms[r.consequent.term](var.universe)
            if self.implication == "min":
                shaped = np.minimum(firing, shape)
            else:  # prod
                shaped = firing * shape
            agg[var.name] = self.aggregation(agg[var.name], shaped)

        crisp = {name: self.defuzz(self._outputs[name].universe, y)
                 for name, y in agg.items()}
        return next(iter(crisp.values())) if len(crisp) == 1 else crisp

    def __repr__(self) -> str:
        return f"Mamdani(rules={len(self.rules)}, outputs={sorted(self._outputs)})"


__all__ = ["Mamdani"]
