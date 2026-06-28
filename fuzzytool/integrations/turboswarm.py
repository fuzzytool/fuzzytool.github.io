"""turboswarm integration: tune a fuzzy system with Particle Swarm Optimization.

Install with ``pip install fuzzytool[turboswarm]``. :func:`tune` fits the
parameters of a system's built-in membership functions to data with
`turboswarm <https://pypi.org/project/turboswarm/>`_'s PSO — a **gradient-free,
global** optimizer. It is the metaheuristic sibling of
:func:`fuzzytool.integrations.scipy.tune` (local least-squares): slower, but it
escapes poor local optima and needs no derivatives, which suits the rugged,
non-smooth error surfaces fuzzy rule bases often produce.

Each parameter is searched within a box of ``±margin × universe`` around its
current value; shape validity (ordered breakpoints, positive widths) is enforced
every evaluation, so the swarm explores freely. The system is mutated in place
with the best parameters found and the turboswarm ``PsoResult`` is returned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import membership as mf
from ._util import (
    MF_PENALTY,
    input_variable_names,
    sanitize_mf_params,
    tunable_terms,
)

if TYPE_CHECKING:
    from turboswarm import PsoResult

    from ..inference import TSK, Mamdani


def _require_turboswarm():
    try:
        from turboswarm import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the turboswarm integration needs turboswarm; install with "
            "`pip install fuzzytool[turboswarm]`"
        ) from exc
    return minimize


def tune(
    system: Mamdani | TSK,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str] | None = None,
    tune_outputs: bool = True,
    margin: float = 0.5,
    n_particles: int = 30,
    max_iter: int = 100,
    seed: int | None = None,
    **minimize_kwargs,
) -> PsoResult:
    """Fit a system's membership-function parameters to data with PSO.

    Extra keyword arguments are forwarded to :func:`turboswarm.minimize`
    (e.g. ``topology``, ``patience``, ``max_time``).

    Args:
        system: a Mamdani or TSK system (mutated in place with the best params).
        X: inputs, shape ``(n_samples, n_inputs)``.
        y: targets, shape ``(n_samples,)``.
        columns: input variable names matching ``X``'s columns, in order.
            Defaults to the variables the rules reference.
        tune_outputs: also tune the output variables' MFs (Mamdani only).
        margin: search half-width per parameter as a fraction of its variable's
            universe span (``0.5`` searches ±50% of the span around each value).
        n_particles: swarm size.
        max_iter: PSO iterations.
        seed: PSO seed for reproducibility.

    Returns:
        The turboswarm ``PsoResult`` (``.best_position``, ``.best_value``, …).
    """
    minimize = _require_turboswarm()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    names = list(columns) if columns is not None else input_variable_names(system)
    inputs = {name: X[:, j] for j, name in enumerate(names)}

    terms = tunable_terms(system, tune_outputs)
    if not terms:
        raise ValueError("the system has no tunable built-in membership functions")

    # Flatten parameters; each gets a search box of ±(margin * span) around p0.
    slices, p0, bounds = [], [], []
    for var, _term, _kind, params in terms:
        delta = margin * (var.high - var.low)
        start = len(p0)
        for value in params:
            p0.append(value)
            bounds.append((value - delta, value + delta))
        slices.append((start, len(params)))

    def apply(position):
        for (var, term_name, kind, _params), (start, size) in zip(terms, slices):
            var[term_name] = mf.from_dict(
                {"type": kind,
                 "params": sanitize_mf_params(kind, position[start:start + size])})

    def objective(position):
        apply(position)
        pred = np.asarray(system.predict(**inputs), dtype=float)
        r = pred - y
        r = np.where(np.isfinite(r), r, MF_PENALTY)
        return float(np.mean(r * r))           # mean squared error

    result = minimize(objective, bounds=bounds, n_particles=n_particles,
                      max_iter=max_iter, seed=seed, **minimize_kwargs)
    apply(result.best_position)                # leave the system at the best fit
    return result


__all__ = ["tune"]
