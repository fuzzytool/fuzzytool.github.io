"""Parallel execution helpers (Joblib and Dask).

Install with ``pip install fuzzytool[parallel]`` (Joblib) or
``fuzzytool[dask]`` (Dask). These spread embarrassingly-parallel fuzzy workloads
across cores or a Dask cluster:

* :func:`parallel_predict` — chunked batch inference with Joblib.
* :func:`multi_start_cmeans` — run fuzzy c-means from many seeds in parallel and
  keep the best (FCM is sensitive to initialization).
* :func:`dask_predict` — the same chunked inference on a Dask scheduler.

.. note::
   The process backends pickle the system. A hand-written Mamdani/TSK built from
   the built-in shapes pickles fine; a TSK with ``lambda`` consequents does not —
   pass ``backend="threading"`` for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._util import input_variable_names

if TYPE_CHECKING:
    from ..cluster import ClusterResult
    from ..inference import TSK, Mamdani


def _require_joblib():
    try:
        from joblib import Parallel, delayed
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the parallel integration needs joblib; install with "
            "`pip install fuzzytool[parallel]`"
        ) from exc
    return Parallel, delayed


def _concat(results: list):
    """Concatenate chunk results, handling both array and multi-output dicts."""
    results = [r for r in results if r is not None]
    if results and isinstance(results[0], dict):
        return {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    return np.concatenate(results)


def _predict_chunk(system, X_chunk, names):
    return system.predict(**{name: X_chunk[:, j] for j, name in enumerate(names)})


def parallel_predict(
    system: Mamdani | TSK,
    X: np.ndarray,
    columns: list[str] | None = None,
    n_jobs: int = -1,
    n_chunks: int | None = None,
    backend: str = "loky",
) -> np.ndarray | dict:
    """Run batch inference over ``X`` in parallel chunks with Joblib.

    Args:
        system: a Mamdani or TSK system.
        X: inputs, shape ``(n_samples, n_inputs)``.
        columns: input variable names matching ``X``'s columns, in order.
            Defaults to the variables the rules reference.
        n_jobs: Joblib worker count (``-1`` = all cores).
        n_chunks: number of row chunks (defaults to ``4 * n_jobs`` or 8).
        backend: Joblib backend (``"loky"`` processes, ``"threading"`` threads).

    Returns:
        The stacked predictions — an array (single output) or a dict of arrays.
    """
    Parallel, delayed = _require_joblib()
    X = np.asarray(X, dtype=float)
    names = list(columns) if columns is not None else input_variable_names(system)
    if n_chunks is None:
        n_chunks = (4 * n_jobs) if n_jobs and n_jobs > 0 else 8
    n_chunks = max(1, min(n_chunks, X.shape[0]))
    chunks = np.array_split(X, n_chunks)
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_predict_chunk)(system, chunk, names) for chunk in chunks)
    return _concat(results)


def multi_start_cmeans(
    X: np.ndarray,
    c: int,
    n_starts: int = 8,
    n_jobs: int = -1,
    backend: str = "loky",
    **kwargs,
) -> ClusterResult:
    """Run :func:`~fuzzytool.cluster.fuzzy_cmeans` from many seeds; keep the best.

    Fuzzy c-means converges to a local optimum that depends on initialization;
    running several seeds and keeping the lowest-objective result is a standard
    safeguard. The runs are independent, so they parallelize cleanly. Extra
    keyword arguments are forwarded to :func:`~fuzzytool.cluster.fuzzy_cmeans`.

    Args:
        X: data, shape ``(n_samples, n_features)``.
        c: number of clusters.
        n_starts: number of random restarts (seeds ``0 .. n_starts - 1``).
        n_jobs: Joblib worker count (``-1`` = all cores).
        backend: Joblib backend.

    Returns:
        The best :class:`~fuzzytool.cluster.ClusterResult` (minimum objective).
    """
    Parallel, delayed = _require_joblib()
    from ..cluster import fuzzy_cmeans

    runs = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(fuzzy_cmeans)(X, c, seed=s, **kwargs) for s in range(n_starts))
    return min(runs, key=lambda r: r.objective)


def dask_predict(
    system: Mamdani | TSK,
    X: np.ndarray,
    columns: list[str] | None = None,
    n_chunks: int | None = None,
) -> np.ndarray | dict:
    """Run chunked batch inference on a Dask scheduler.

    Mirrors :func:`parallel_predict` but builds a Dask graph (``dask.delayed``)
    and computes it, so it scales to a distributed cluster.

    Args:
        system: a Mamdani or TSK system.
        X: inputs, shape ``(n_samples, n_inputs)``.
        columns: input variable names matching ``X``'s columns.
        n_chunks: number of row chunks (default 8).

    Returns:
        The stacked predictions — an array or a dict of arrays.
    """
    try:
        import dask
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "dask_predict needs dask; install with `pip install fuzzytool[dask]`"
        ) from exc

    X = np.asarray(X, dtype=float)
    names = list(columns) if columns is not None else input_variable_names(system)
    n_chunks = max(1, min(n_chunks or 8, X.shape[0]))
    tasks = [dask.delayed(_predict_chunk)(system, chunk, names)
             for chunk in np.array_split(X, n_chunks)]
    return _concat(list(dask.compute(*tasks)))


__all__ = ["parallel_predict", "multi_start_cmeans", "dask_predict"]
