"""F-transform — the fuzzy transform of Perfilieva.

The F-transform projects a function onto a *fuzzy partition* of its domain. Over
a uniform partition by triangular basis functions ``A_1..A_n`` that satisfy the
Ruspini condition (they sum to 1 everywhere on ``[a, b]``):

* the **direct** transform reduces the data to ``n`` components, each a
  membership-weighted average of the samples falling under one basis function;
* the **inverse** transform reconstructs an approximation
  ``f̂(x) = Σ_k F_k · A_k(x)``.

With few components the round trip smooths/denoises a signal; with many it
approximates it closely. Pure NumPy.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


class FTransform:
    """A triangular F-transform over a uniform fuzzy partition of ``[a, b]``.

    Args:
        a: left end of the domain.
        b: right end of the domain (``b > a``).
        n_basis: number of basis functions / components (``>= 2``).
    """

    def __init__(self, a: float, b: float, n_basis: int) -> None:
        if b <= a:
            raise ValueError(f"need a < b, got ({a}, {b})")
        if n_basis < 2:
            raise ValueError("need n_basis >= 2")
        self.a, self.b = float(a), float(b)
        self.nodes = np.linspace(a, b, n_basis)            # basis centers
        self.h = self.nodes[1] - self.nodes[0]             # uniform spacing
        self.components_: np.ndarray | None = None

    def basis(self, x) -> np.ndarray:
        """Triangular basis matrix ``A`` of shape ``(len(x), n_basis)``.

        Columns form a partition of unity on ``[a, b]`` (each row sums to 1).
        """
        x = np.asarray(x, dtype=float)
        return np.clip(1.0 - np.abs(x[:, None] - self.nodes[None, :]) / self.h, 0.0, 1.0)

    def direct(self, x, y) -> np.ndarray:
        """Direct F-transform: the ``n_basis`` components from samples ``(x, y)``."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        a = self.basis(x)
        comp = (a.T @ y) / np.fmax(a.sum(axis=0), _EPS)
        self.components_ = comp
        return comp

    def inverse(self, x, components: np.ndarray | None = None) -> np.ndarray:
        """Inverse F-transform: reconstruct values at ``x`` from components."""
        comp = self.components_ if components is None else np.asarray(components, float)
        if comp is None:
            raise RuntimeError("no components; call direct first or pass them in")
        return self.basis(x) @ comp

    def fit(self, x, y) -> FTransform:
        """Compute and store the direct transform of ``(x, y)``; returns ``self``."""
        self.direct(x, y)
        return self

    def smooth(self, x) -> np.ndarray:
        """Convenience: direct-then-inverse at ``x`` (requires a prior fit)."""
        return self.inverse(x)

    def __repr__(self) -> str:
        return f"FTransform([{self.a}, {self.b}], n_basis={len(self.nodes)})"


__all__ = ["FTransform"]
