"""Matplotlib visualization helpers.

Importing this module requires ``matplotlib`` (an optional dependency; install
with ``pip install fuzzytool[viz]``). Visualization is a first-class priority of
the project, mirroring its sibling ``turboswarm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .sets import Variable


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "visualization needs matplotlib; install with `pip install fuzzytool[viz]`"
        ) from exc
    return plt


def plot_variable(variable: Variable, ax=None):
    """Plot every term's membership function over the variable's universe."""
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    x = variable.universe
    for name, m in variable.terms.items():
        ax.plot(x, m(x), label=name)
    ax.set_title(variable.name)
    ax.set_xlabel(variable.name)
    ax.set_ylabel("membership")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_it2_variable(variable: Variable, ax=None):
    """Plot an IT2 variable: each term's lower/upper MF with a shaded FOU.

    Type-1 terms (if any are mixed in) are drawn as a single line.
    """
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    x = variable.universe
    for name, term in variable.terms.items():
        if hasattr(term, "lower") and hasattr(term, "upper"):
            lo, up = term.lower(x), term.upper(x)
            line, = ax.plot(x, up, label=name)
            ax.plot(x, lo, color=line.get_color(), alpha=0.7)
            ax.fill_between(x, lo, up, color=line.get_color(), alpha=0.2)
        else:
            ax.plot(x, term(x), label=name)
    ax.set_title(variable.name)
    ax.set_xlabel(variable.name)
    ax.set_ylabel("membership")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize="small")
    return ax


def control_surface(system, x_var: Variable, y_var: Variable,
                    n: int = 41, ax=None):
    """Plot a system's output as a surface over two input variables.

    ``system`` must be callable as ``system(**{x_var.name: x, y_var.name: y})``
    and return a scalar (single-output Mamdani or TSK).
    """
    plt = _require_mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    xs = np.linspace(x_var.low, x_var.high, n)
    ys = np.linspace(y_var.low, y_var.high, n)
    zz = np.empty((n, n))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            zz[i, j] = system(**{x_var.name: float(xv), y_var.name: float(yv)})
    xx, yy = np.meshgrid(xs, ys)

    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel(x_var.name)
    ax.set_ylabel(y_var.name)
    ax.set_zlabel("output")
    return ax


def plot_clusters(X, result, ax=None):
    """Scatter 2-D data colored by hard label, with cluster centers marked.

    ``result`` is a :class:`~fuzzytool.cluster.ClusterResult`. Point opacity
    encodes the top membership, so fuzzy/boundary points appear fainter.
    """
    plt = _require_mpl()
    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise ValueError("plot_clusters needs 2-D data")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    top = result.u.max(axis=0)
    ax.scatter(X[:, 0], X[:, 1], c=result.labels, cmap="tab10",
               alpha=np.clip(top, 0.25, 1.0), s=25)
    ax.scatter(result.centers[:, 0], result.centers[:, 1],
               marker="X", c="black", s=160, edgecolors="white", zorder=3)
    ax.set_title("fuzzy clusters")
    return ax


__all__ = ["plot_variable", "plot_it2_variable", "control_surface", "plot_clusters"]
