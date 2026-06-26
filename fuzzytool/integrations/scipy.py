"""SciPy integration: tune a fuzzy system's membership functions to data.

Install with ``pip install fuzzytool[scipy]``. :func:`tune` adjusts the
parameters of a system's built-in membership functions so its output fits a
dataset, using :func:`scipy.optimize.least_squares`. It mutates the system in
place and returns SciPy's ``OptimizeResult`` (``.x``, ``.cost``, ``.nfev``,
``.success`` …) — a drop-in way to refine a hand-written Mamdani/TSK system.

Only built-in membership shapes (``tri``, ``trap``, ``gauss``, ``gbell``,
``sigmoid``, ``ramp_up``, ``ramp_down``) are tunable; custom callables are left
untouched. Shape validity is preserved every iteration (breakpoints are kept
ordered, widths kept positive), so the optimizer explores freely without
producing degenerate sets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import membership as mf
from ._util import input_variable_names

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

    from ..inference import TSK, Mamdani

# Per-shape repair applied before rebuilding, so every candidate is a valid MF.
_PENALTY = 1e6


def _sanitize(kind: str, params: list[float]) -> list[float]:
    p = [float(v) for v in params]
    if kind in ("tri", "trap", "ramp_up", "ramp_down"):
        return sorted(p)                       # keep breakpoints ordered
    if kind == "gauss":
        return [p[0], abs(p[1]) or 1e-6]       # sigma > 0
    if kind == "gbell":
        return [abs(p[0]) or 1e-6, abs(p[1]) or 1e-6, p[2]]  # width, slope > 0
    return p


def _require_scipy():
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the scipy integration needs scipy; install with "
            "`pip install fuzzytool[scipy]`"
        ) from exc
    return least_squares


def _tunable_terms(system, tune_outputs: bool):
    """Variables and term names whose membership functions can be tuned."""
    # Collect the Variable instances behind the rules' propositions.
    seen: dict[int, object] = {}

    def walk(node):
        if hasattr(node, "variable"):
            seen.setdefault(id(node.variable), node.variable)
        if hasattr(node, "left"):
            walk(node.left)
            walk(node.right)
        if hasattr(node, "operand"):
            walk(node.operand)

    for rule in system.rules:
        walk(rule.antecedent)
    variables = dict(seen)
    if tune_outputs:
        for out in getattr(system, "_outputs", {}).values():
            variables.setdefault(id(out), out)

    terms = []
    for var in variables.values():
        for term_name, membership in var.terms.items():
            try:
                spec = mf.to_dict(membership)
            except TypeError:
                continue                       # custom callable: skip
            terms.append((var, term_name, spec["type"], list(spec["params"])))
    return terms


def tune(
    system: Mamdani | TSK,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str] | None = None,
    tune_outputs: bool = True,
    **least_squares_kwargs,
) -> OptimizeResult:
    """Fit a system's membership-function parameters to data.

    Any extra keyword arguments are forwarded to
    :func:`scipy.optimize.least_squares` (e.g. ``max_nfev``).

    Args:
        system: a Mamdani or TSK system (mutated in place with the best params).
        X: inputs, shape ``(n_samples, n_inputs)``.
        y: targets, shape ``(n_samples,)``.
        columns: input variable names matching ``X``'s columns, in order.
            Defaults to the variables the rules reference.
        tune_outputs: also tune the output variables' MFs (Mamdani only).

    Returns:
        The SciPy ``OptimizeResult``.
    """
    least_squares = _require_scipy()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    names = list(columns) if columns is not None else input_variable_names(system)
    inputs = {name: X[:, j] for j, name in enumerate(names)}

    terms = _tunable_terms(system, tune_outputs)
    if not terms:
        raise ValueError("the system has no tunable built-in membership functions")

    # Flatten all parameters into one vector; remember each term's slice.
    slices, p0 = [], []
    for _var, _term, _kind, params in terms:
        start = len(p0)
        p0.extend(params)
        slices.append((start, len(params)))
    p0 = np.asarray(p0, dtype=float)

    def apply(p):
        for (var, term_name, kind, _params), (start, size) in zip(terms, slices):
            var[term_name] = mf.from_dict(
                {"type": kind, "params": _sanitize(kind, p[start:start + size])})

    def residual(p):
        apply(p)
        pred = np.asarray(system.predict(**inputs), dtype=float)
        r = pred - y
        return np.where(np.isfinite(r), r, _PENALTY)

    result = least_squares(residual, p0, **least_squares_kwargs)
    apply(result.x)                            # leave the system at the best fit
    return result


__all__ = ["tune"]
