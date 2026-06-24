"""Fuzzy sets, linguistic variables, and the rule-antecedent expression tree.

A :class:`Variable` is a linguistic variable: a named universe of discourse plus
a dictionary of *terms* (named membership functions). Indexing a variable with a
term name (``score["good"]``) returns a :class:`Proposition`, the atom of a
rule antecedent. Propositions compose with Python operators:

    score["poor"] | dti["high"]         # OR  (s-norm)
    score["good"] & dti["low"]          # AND (t-norm)
    ~dti["high"]                        # NOT (complement)

The result is a small expression tree that the inference engine evaluates
against crisp inputs to get a firing strength. The same atom doubles as a rule
*consequent* (``premium["high"]``), so there is a single concept to learn.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from . import membership as mf
from .norms import complement

_KINDS = {"triangular": "tri", "tri": "tri", "gauss": "gauss", "gaussian": "gauss"}


def _is_it2(term) -> bool:
    """An IT2 term exposes ``.lower`` and ``.upper`` (duck-typed to avoid an import)."""
    return hasattr(term, "lower") and hasattr(term, "upper")


class Antecedent:
    """Base node of a rule-antecedent expression tree."""

    def __and__(self, other: Antecedent) -> And:
        return And(self, other)

    def __or__(self, other: Antecedent) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def eval(self, inputs: Mapping[str, float], tnorm, snorm) -> np.ndarray:
        """Return the (type-1) firing strength given crisp ``inputs``."""
        raise NotImplementedError

    def eval_interval(self, inputs: Mapping[str, float], tnorm, snorm):
        """Return the firing *interval* ``(lower, upper)`` for IT2 inference.

        Type-1 terms collapse to a degenerate interval ``(d, d)``, so type-1 and
        IT2 terms may be mixed freely in the same antecedent.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize this antecedent subtree to a JSON-ready dict."""
        raise NotImplementedError


class Proposition(Antecedent):
    """An atomic ``variable is term`` clause; also used as a consequent."""

    def __init__(self, variable: Variable, term: str) -> None:
        if term not in variable.terms:
            raise KeyError(
                f"variable {variable.name!r} has no term {term!r}; "
                f"known terms: {sorted(variable.terms)}"
            )
        self.variable = variable
        self.term = term

    def eval(self, inputs, tnorm, snorm) -> np.ndarray:
        try:
            x = inputs[self.variable.name]
        except KeyError:
            raise KeyError(f"missing input for variable {self.variable.name!r}") from None
        term = self.variable.terms[self.term]
        if _is_it2(term):  # collapse an IT2 term to its mean membership
            lo, hi = term.lower(x), term.upper(x)
            return np.asarray((lo + hi) / 2.0, dtype=float)
        return np.asarray(term(x), dtype=float)

    def eval_interval(self, inputs, tnorm, snorm):
        try:
            x = inputs[self.variable.name]
        except KeyError:
            raise KeyError(f"missing input for variable {self.variable.name!r}") from None
        term = self.variable.terms[self.term]
        if _is_it2(term):
            return float(term.lower(x)), float(term.upper(x))
        d = float(np.asarray(term(x), dtype=float))
        return d, d

    def to_dict(self) -> dict:
        return {"op": "is", "var": self.variable.name, "term": self.term}

    def __repr__(self) -> str:
        return f"{self.variable.name} is {self.term}"


class And(Antecedent):
    def __init__(self, left: Antecedent, right: Antecedent) -> None:
        self.left, self.right = left, right

    def eval(self, inputs, tnorm, snorm):
        return tnorm(self.left.eval(inputs, tnorm, snorm),
                     self.right.eval(inputs, tnorm, snorm))

    def eval_interval(self, inputs, tnorm, snorm):
        ll, lu = self.left.eval_interval(inputs, tnorm, snorm)
        rl, ru = self.right.eval_interval(inputs, tnorm, snorm)
        return tnorm(ll, rl), tnorm(lu, ru)

    def to_dict(self) -> dict:
        return {"op": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self) -> str:
        return f"({self.left} and {self.right})"


class Or(Antecedent):
    def __init__(self, left: Antecedent, right: Antecedent) -> None:
        self.left, self.right = left, right

    def eval(self, inputs, tnorm, snorm):
        return snorm(self.left.eval(inputs, tnorm, snorm),
                     self.right.eval(inputs, tnorm, snorm))

    def eval_interval(self, inputs, tnorm, snorm):
        ll, lu = self.left.eval_interval(inputs, tnorm, snorm)
        rl, ru = self.right.eval_interval(inputs, tnorm, snorm)
        return snorm(ll, rl), snorm(lu, ru)

    def to_dict(self) -> dict:
        return {"op": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __repr__(self) -> str:
        return f"({self.left} or {self.right})"


class Not(Antecedent):
    def __init__(self, operand: Antecedent) -> None:
        self.operand = operand

    def eval(self, inputs, tnorm, snorm):
        return complement(self.operand.eval(inputs, tnorm, snorm))

    def eval_interval(self, inputs, tnorm, snorm):
        # The complement is order-reversing, so the bounds swap.
        lo, hi = self.operand.eval_interval(inputs, tnorm, snorm)
        return complement(hi), complement(lo)

    def to_dict(self) -> dict:
        return {"op": "not", "operand": self.operand.to_dict()}

    def __repr__(self) -> str:
        return f"(not {self.operand})"


class Variable:
    """A linguistic variable: a named universe with named fuzzy-set terms.

    Args:
        name: identifier; rule inputs are matched to variables by this name.
        universe: ``(low, high)`` range of the universe of discourse.
        terms: optional list of term names to auto-generate evenly across the
            universe, or a mapping ``{name: MembershipFunction}``.
        kind: shape used by auto-generation (``"triangular"`` or ``"gauss"``).
        resolution: number of samples used to discretize the universe for
            Mamdani defuzzification.
    """

    def __init__(
        self,
        name: str,
        universe: tuple[float, float],
        terms: Iterable[str] | Mapping[str, mf.MembershipFunction] | None = None,
        kind: str = "triangular",
        resolution: int = 501,
    ) -> None:
        self.name = name
        self.low, self.high = float(universe[0]), float(universe[1])
        if self.high <= self.low:
            raise ValueError(f"universe must have low < high, got {universe}")
        self.terms: dict[str, mf.MembershipFunction] = {}
        self.universe = np.linspace(self.low, self.high, resolution)
        if isinstance(terms, Mapping):
            for n, m in terms.items():
                self[n] = m
        elif terms is not None:
            self.auto_terms(list(terms), kind=kind)

    def __getitem__(self, term: str) -> Proposition:
        return Proposition(self, term)

    def __setitem__(self, term: str, membership: mf.MembershipFunction) -> None:
        if not callable(membership):
            raise TypeError("a term must be a callable membership function")
        self.terms[term] = membership

    def auto_terms(self, names: list[str], kind: str = "triangular") -> Variable:
        """Generate evenly-spaced terms named ``names`` across the universe."""
        kind = _KINDS.get(kind)
        if kind is None:
            raise ValueError("kind must be 'triangular' or 'gauss'")
        n = len(names)
        if n < 2:
            raise ValueError("need at least two terms")
        centers = np.linspace(self.low, self.high, n)
        step = (self.high - self.low) / (n - 1)
        for name, c in zip(names, centers):
            if kind == "tri":
                self[name] = mf.tri(c - step, c, c + step)
            else:  # gauss
                self[name] = mf.gauss(c, step / 2.0)
        return self

    def to_dict(self) -> dict:
        """Serialize this variable (universe + terms) to a JSON-ready dict.

        Only built-in membership functions can be serialized (see
        :func:`fuzzytool.membership.to_dict`).
        """
        return {
            "name": self.name,
            "universe": [self.low, self.high],
            "resolution": int(self.universe.size),
            "terms": {name: mf.to_dict(term) for name, term in self.terms.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> Variable:
        """Rebuild a variable from :meth:`to_dict` output."""
        var = cls(d["name"], tuple(d["universe"]), resolution=d.get("resolution", 501))
        for name, term in d["terms"].items():
            var[name] = mf.from_dict(term)
        return var

    def __repr__(self) -> str:
        return (f"Variable({self.name!r}, ({self.low}, {self.high}), "
                f"terms={sorted(self.terms)})")


def antecedent_from_dict(d: dict, variables: Mapping[str, Variable]) -> Antecedent:
    """Rebuild an antecedent tree from :meth:`Antecedent.to_dict` output.

    ``variables`` maps variable names to the :class:`Variable` instances the
    propositions should reference.
    """
    op = d["op"]
    if op == "is":
        return variables[d["var"]][d["term"]]
    if op == "and":
        return And(antecedent_from_dict(d["left"], variables),
                   antecedent_from_dict(d["right"], variables))
    if op == "or":
        return Or(antecedent_from_dict(d["left"], variables),
                  antecedent_from_dict(d["right"], variables))
    if op == "not":
        return Not(antecedent_from_dict(d["operand"], variables))
    raise ValueError(f"unknown antecedent op {op!r}")


__all__ = ["Variable", "Proposition", "Antecedent", "And", "Or", "Not",
           "antecedent_from_dict"]
