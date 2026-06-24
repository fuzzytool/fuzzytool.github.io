"""Takagi-Sugeno-Kang (TSK) fuzzy inference.

Consequents are crisp functions of the inputs rather than fuzzy sets, so there
is no defuzzification: the output is the firing-weighted average of the rule
consequents. A consequent may be

* a number — zero-order (Sugeno constant), e.g. ``5.0``;
* a mapping ``{"const": b0, "x": b1, ...}`` — first-order linear in the inputs;
* any callable ``f(**inputs) -> float`` — arbitrary.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Callable

from ..norms import get_snorm, get_tnorm
from ..rules import Rule
from ..sets import Antecedent


def _as_consequent_fn(consequent) -> Callable[..., float]:
    if isinstance(consequent, Real):
        return lambda **_: float(consequent)
    if isinstance(consequent, Mapping):
        bias = float(consequent.get("const", 0.0))
        coefs = {k: float(v) for k, v in consequent.items() if k != "const"}
        return lambda **xs: bias + sum(c * xs[k] for k, c in coefs.items())
    if callable(consequent):
        return consequent
    raise TypeError("TSK consequent must be a number, a coefficient mapping, "
                    "or a callable")


class TSK:
    """A (zero- or first-order) Takagi-Sugeno inference system.

    Args:
        tnorm: t-norm for AND in antecedents (default ``"min"``).
        snorm: s-norm for OR in antecedents (default ``"max"``).
    """

    def __init__(self, tnorm: str | Callable = "min",
                 snorm: str | Callable = "max") -> None:
        self.tnorm = get_tnorm(tnorm)
        self.snorm = get_snorm(snorm)
        self.rules: list[Rule] = []
        self._fns: list[Callable[..., float]] = []

    def rule(self, antecedent: Antecedent, consequent, weight: float = 1.0) -> TSK:
        """Add ``IF antecedent THEN output = consequent`` and return ``self``."""
        self.rules.append(Rule(antecedent, consequent, weight))
        self._fns.append(_as_consequent_fn(consequent))
        return self

    def __call__(self, **inputs: float) -> float:
        """Run inference, returning the firing-weighted average output."""
        if not self.rules:
            raise RuntimeError("no rules defined")
        num = 0.0
        den = 0.0
        for r, fn in zip(self.rules, self._fns):
            w = float(r.antecedent.eval(inputs, self.tnorm, self.snorm)) * r.weight
            if w <= 0.0:
                continue
            num += w * fn(**inputs)
            den += w
        if den == 0.0:
            raise ValueError("no rule fired for the given inputs")
        return num / den

    def __repr__(self) -> str:
        return f"TSK(rules={len(self.rules)})"


__all__ = ["TSK"]
