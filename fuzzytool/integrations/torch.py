"""PyTorch integration: a differentiable fuzzy layer.

Install with ``pip install fuzzytool[torch]``. :class:`FuzzyLayer` is a
first-order Takagi-Sugeno system written as a ``torch.nn.Module``: its Gaussian
membership functions and affine consequents are plain ``Parameter``\\ s, so the
whole layer is differentiable and trains end-to-end by autograd — on its own, or
as one block inside a larger network.

This is the gradient-based sibling of :class:`~fuzzytool.anfis.ANFIS` (which
trains with Jang's hybrid least-squares/gradient scheme): same model, but every
parameter is learned by backprop, so it composes with arbitrary upstream layers
and non-standard losses.

The module imports cleanly without torch installed; only constructing a
:class:`FuzzyLayer` requires it.
"""

from __future__ import annotations

import itertools

try:
    import torch
    from torch import nn
except ImportError:  # torch is an optional extra
    nn = None


if nn is not None:

    class FuzzyLayer(nn.Module):
        """A differentiable first-order TSK layer.

        Over a grid partition, each of the ``n_inputs`` inputs gets ``n_mf``
        Gaussian membership functions, giving ``n_mf ** n_inputs`` rules; each
        rule emits an affine function of the inputs. The forward pass returns the
        firing-weighted (normalized) average of the rule consequents.

        Args:
            n_inputs: number of input features.
            n_mf: Gaussian membership functions per input (default ``3``).
            init_range: ``(low, high)`` range the membership centers are spread
                across at initialization (default ``(0.0, 1.0)`` — scale your
                inputs, or pass the data range).

        Shape:
            input ``(batch, n_inputs)`` → output ``(batch, 1)``.
        """

        def __init__(self, n_inputs: int, n_mf: int = 3,
                     init_range: tuple[float, float] = (0.0, 1.0)) -> None:
            super().__init__()
            self.n_inputs = int(n_inputs)
            self.n_mf = int(n_mf)
            low, high = float(init_range[0]), float(init_range[1])

            # Rule grid: every combination of one MF index per input.
            rule_mf = list(itertools.product(range(self.n_mf), repeat=self.n_inputs))
            self.register_buffer("rule_mf", torch.tensor(rule_mf, dtype=torch.long))
            self.n_rules = self.rule_mf.shape[0]

            centers = torch.linspace(low, high, self.n_mf).repeat(self.n_inputs, 1)
            spacing = (high - low) / max(self.n_mf - 1, 1)
            log_sigmas = torch.full((self.n_inputs, self.n_mf),
                                    float(torch.log(torch.tensor(spacing or 1.0))))
            self.centers = nn.Parameter(centers)              # (p, n_mf)
            self.log_sigmas = nn.Parameter(log_sigmas)        # (p, n_mf)
            self.coeffs = nn.Parameter(                        # (R, p + 1)
                torch.zeros(self.n_rules, self.n_inputs + 1))

        def forward(self, x):
            if x.ndim != 2 or x.shape[1] != self.n_inputs:
                raise ValueError(
                    f"expected input of shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
            sigmas = torch.exp(self.log_sigmas)
            # Membership degrees: (batch, p, n_mf)
            z = (x[:, :, None] - self.centers[None, :, :]) / sigmas[None, :, :]
            mu = torch.exp(-0.5 * z * z)
            # Per-rule firing = product over inputs of the selected MF degree.
            j = torch.arange(self.n_inputs).expand(self.n_rules, self.n_inputs)
            mu_sel = mu[:, j, self.rule_mf]                    # (batch, R, p)
            w = mu_sel.prod(dim=2)                             # (batch, R)
            wn = w / (w.sum(dim=1, keepdim=True) + 1e-12)
            # First-order consequents: affine in the inputs.
            xc = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
            f = xc @ self.coeffs.t()                           # (batch, R)
            return (wn * f).sum(dim=1, keepdim=True)           # (batch, 1)

        def fit(self, X, y, epochs: int = 200, lr: float = 0.05):
            """Convenience training loop (Adam + MSE). Returns the RMSE history.

            ``X`` is ``(n, n_inputs)`` and ``y`` is ``(n,)`` or ``(n, 1)``; both
            may be NumPy arrays or tensors.
            """
            X = torch.as_tensor(X, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            history = []
            for _ in range(int(epochs)):
                opt.zero_grad()
                loss = loss_fn(self(X), y)
                loss.backward()
                opt.step()
                history.append(float(loss.detach().sqrt()))
            return history

else:  # pragma: no cover - exercised only without torch installed

    def FuzzyLayer(*args, **kwargs):
        raise ImportError(
            "the PyTorch integration needs torch; install with "
            "`pip install fuzzytool[torch]`")


__all__ = ["FuzzyLayer"]
