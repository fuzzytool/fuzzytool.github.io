"""Tsukamoto fuzzy inference.

In a Tsukamoto system every rule's consequent is a *monotonic* membership
function. A rule firing with strength ``w`` produces the crisp value at which
its consequent reaches ``w`` (the inverse of the monotonic MF). The system
output is the firing-weighted average of those crisp values — so, like TSK,
there is no defuzzification step.

Consequents must expose ``inverse(degree) -> value``; the built-in
:func:`~fuzzytool.membership.ramp_up`, :func:`~fuzzytool.membership.ramp_down`
and :func:`~fuzzytool.membership.sigmoid` do.
"""

from __future__ import annotations

from typing import Callable

from ..norms import get_snorm, get_tnorm
from ..rules import Rule
from ..sets import Antecedent


class Tsukamoto:
    """A Tsukamoto inference system (monotonic consequents).

    Args:
        tnorm: t-norm for AND in antecedents (default ``"min"``).
        snorm: s-norm for OR in antecedents (default ``"max"``).
    """

    def __init__(self, tnorm: str | Callable = "min",
                 snorm: str | Callable = "max") -> None:
        self.tnorm = get_tnorm(tnorm)
        self.snorm = get_snorm(snorm)
        self.rules: list[Rule] = []

    def rule(self, antecedent: Antecedent, consequent, weight: float = 1.0) -> Tsukamoto:
        """Add a rule; ``consequent`` is a monotonic MF with an ``inverse``."""
        if not hasattr(consequent, "inverse"):
            raise TypeError("Tsukamoto consequent must expose inverse(degree); "
                            "use ramp_up / ramp_down / sigmoid")
        self.rules.append(Rule(antecedent, consequent, weight))
        return self

    def __call__(self, **inputs: float) -> float:
        """Run inference, returning the firing-weighted average crisp output."""
        if not self.rules:
            raise RuntimeError("no rules defined")
        num = 0.0
        den = 0.0
        for r in self.rules:
            w = float(r.antecedent.eval(inputs, self.tnorm, self.snorm)) * r.weight
            if w <= 0.0:
                continue
            num += w * r.consequent.inverse(w)
            den += w
        if den == 0.0:
            raise ValueError("no rule fired for the given inputs")
        return num / den

    def __repr__(self) -> str:
        return f"Tsukamoto(rules={len(self.rules)})"


__all__ = ["Tsukamoto"]
