"""ANFIS — an adaptive network-based fuzzy inference system (Jang, 1993).

ANFIS is a first-order Takagi-Sugeno system whose parameters are learned from
data. With a grid partition, each of the ``p`` inputs carries ``n_mf`` Gaussian
membership functions, giving ``n_mf ** p`` rules; rule ``i`` fires with the
product of its inputs' memberships and emits an affine function of the inputs.

Training uses Jang's **hybrid** scheme, one pass per epoch:

1. with the premise (Gaussian) parameters fixed, the consequent (affine)
   parameters are solved in closed form by least squares — the output is linear
   in them;
2. with the consequents fixed, the premise centers and widths take a
   gradient-descent step on the mean squared error.

Pure NumPy; intended for low-dimensional problems (the rule count grows as
``n_mf ** p``).
"""

from __future__ import annotations

import itertools

import numpy as np

_EPS = 1e-12


class ANFIS:
    """A trainable first-order Sugeno system over a grid partition.

    Args:
        n_inputs: number of input features ``p``.
        n_mf: Gaussian membership functions per input (rules = ``n_mf ** p``).
        learning_rate: step size for the premise gradient updates.
    """

    def __init__(self, n_inputs: int, n_mf: int = 3,
                 learning_rate: float = 0.05) -> None:
        if n_inputs < 1 or n_mf < 2:
            raise ValueError("need n_inputs >= 1 and n_mf >= 2")
        self.p = int(n_inputs)
        self.n_mf = int(n_mf)
        self.lr = float(learning_rate)
        # Rule -> per-input MF index, shape (R, p); R = n_mf ** p.
        self.rule_mf = np.array(list(itertools.product(range(n_mf), repeat=n_inputs)))
        self.R = self.rule_mf.shape[0]
        self.centers_: np.ndarray | None = None     # (p, n_mf)
        self.sigmas_: np.ndarray | None = None       # (p, n_mf)
        self.coeffs_: np.ndarray | None = None        # (R, p + 1)
        self.history_: list[float] = []               # RMSE per epoch

    # --- internals --------------------------------------------------------

    def _init_premise(self, X: np.ndarray) -> None:
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        self.centers_ = np.linspace(lo, hi, self.n_mf).T              # (p, n_mf)
        self.sigmas_ = np.tile((span / (self.n_mf - 1))[:, None], (1, self.n_mf))

    def _firing(self, X: np.ndarray):
        """Return ``(mu, mu_sel, w, wn, W)`` for a batch ``X`` (n, p)."""
        c, s = self.centers_, self.sigmas_
        mu = np.exp(-0.5 * ((X[:, :, None] - c[None]) / s[None]) ** 2)  # (n, p, M)
        j = np.broadcast_to(np.arange(self.p), (self.R, self.p))
        mu_sel = mu[:, j, self.rule_mf]                                 # (n, R, p)
        w = mu_sel.prod(axis=2)                                         # (n, R)
        W = w.sum(axis=1, keepdims=True) + _EPS
        return mu, mu_sel, w, w / W, W

    def _lse_consequents(self, X: np.ndarray, y: np.ndarray, wn: np.ndarray) -> None:
        n = X.shape[0]
        xc = np.hstack([X, np.ones((n, 1))])                           # (n, p+1)
        phi = (wn[:, :, None] * xc[:, None, :]).reshape(n, self.R * (self.p + 1))
        theta, *_ = np.linalg.lstsq(phi, y, rcond=None)
        self.coeffs_ = theta.reshape(self.R, self.p + 1)

    def _outputs(self, X: np.ndarray, wn: np.ndarray):
        xc = np.hstack([X, np.ones((X.shape[0], 1))])
        f = xc @ self.coeffs_.T                                         # (n, R)
        return f, (wn * f).sum(axis=1)                                  # (n, R), (n,)

    # --- public API -------------------------------------------------------

    def fit(self, X, y, epochs: int = 100) -> ANFIS:
        """Train on ``X`` (n, p) and targets ``y`` (n,) for ``epochs`` epochs."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[1] != self.p:
            raise ValueError(f"X must be 2-D with {self.p} columns")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        n = X.shape[0]
        self._init_premise(X)
        self.history_ = []

        for _ in range(epochs):
            mu, mu_sel, w, wn, W = self._firing(X)
            self._lse_consequents(X, y, wn)              # closed-form consequents
            f, y_pred = self._outputs(X, wn)
            err = y_pred - y
            self.history_.append(float(np.sqrt(np.mean(err ** 2))))

            # Backprop the MSE to the premise Gaussians.
            dL_dw = (err[:, None] * (f - y_pred[:, None]) / W) / n      # (n, R)
            ratio = w[:, :, None] / (mu_sel + _EPS)                     # (n, R, p)
            g = dL_dw[:, :, None] * ratio                              # (n, R, p)
            grad_mu = np.zeros_like(mu)                                # (n, p, M)
            for jj in range(self.p):
                kj = self.rule_mf[:, jj]
                for k in range(self.n_mf):
                    mask = kj == k
                    if mask.any():
                        grad_mu[:, jj, k] = g[:, mask, jj].sum(axis=1)

            diff = X[:, :, None] - self.centers_[None]                  # (n, p, M)
            grad_c = (grad_mu * mu * diff / self.sigmas_ ** 2).sum(axis=0)
            grad_s = (grad_mu * mu * diff ** 2 / self.sigmas_ ** 3).sum(axis=0)
            self.centers_ = self.centers_ - self.lr * grad_c
            self.sigmas_ = np.maximum(self.sigmas_ - self.lr * grad_s, 1e-3)

        # Refit consequents to the final premise parameters.
        *_, wn, _ = self._firing(X)
        self._lse_consequents(X, y, wn)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict outputs for ``X`` (n, p). Requires a prior :meth:`fit`."""
        if self.coeffs_ is None:
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.p:
            raise ValueError(f"X must be 2-D with {self.p} columns")
        *_, wn, _ = self._firing(X)
        return self._outputs(X, wn)[1]

    def __repr__(self) -> str:
        return f"ANFIS(n_inputs={self.p}, n_mf={self.n_mf}, rules={self.R})"


__all__ = ["ANFIS"]
