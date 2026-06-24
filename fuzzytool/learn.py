"""Learning fuzzy rule bases from data.

:func:`wang_mendel` implements the classic Wang-Mendel (1992) method: it turns
each training sample into one fuzzy rule (picking, per variable, the term with
the highest membership), assigns the rule a degree equal to the product of those
memberships, and resolves antecedent conflicts by keeping the highest-degree
consequent. The result is a ready-to-use :class:`~fuzzytool.inference.Mamdani`.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from .inference import Mamdani
from .sets import Variable


def _best_term(var: Variable, value: float) -> tuple[str, float]:
    """The term of ``var`` with the highest membership at ``value``."""
    best_name, best_mu = None, -1.0
    for name, mf in var.terms.items():
        mu = float(np.asarray(mf(value)))
        if mu > best_mu:
            best_name, best_mu = name, mu
    return best_name, best_mu


def wang_mendel(X: np.ndarray, y: np.ndarray, inputs: list[Variable],
                output: Variable) -> Mamdani:
    """Generate a Mamdani rule base from data with the Wang-Mendel method.

    Args:
        X: inputs, shape ``(n_samples, len(inputs))``.
        y: targets, shape ``(n_samples,)``.
        inputs: the input linguistic variables (each pre-populated with terms);
            column ``j`` of ``X`` maps to ``inputs[j]``.
        output: the output linguistic variable (pre-populated with terms).

    Returns:
        A :class:`~fuzzytool.inference.Mamdani` with one rule per distinct
        antecedent (conflicts resolved by rule degree).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[1] != len(inputs):
        raise ValueError("X must be 2-D with one column per input variable")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have inconsistent lengths")

    # antecedent key (tuple of term names) -> (consequent term, degree)
    best: dict[tuple[str, ...], tuple[str, float]] = {}
    for row, target in zip(X, y):
        terms, degree = [], 1.0
        for var, value in zip(inputs, row):
            name, mu = _best_term(var, value)
            terms.append(name)
            degree *= mu
        out_term, _ = _best_term(output, target)
        key = tuple(terms)
        if key not in best or degree > best[key][1]:
            best[key] = (out_term, degree)

    sys = Mamdani()
    for key, (out_term, _) in best.items():
        antecedent = reduce(lambda acc, jt: acc & jt,
                             (inputs[j][term] for j, term in enumerate(key)))
        sys.rule(antecedent, output[out_term])
    return sys


__all__ = ["wang_mendel"]
