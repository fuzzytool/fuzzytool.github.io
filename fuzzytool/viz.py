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


__all__ = ["plot_variable", "control_surface"]
