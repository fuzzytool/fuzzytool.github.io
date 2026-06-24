# F-transform

The **fuzzy transform** (F-transform, Perfilieva) projects a function onto a
fuzzy partition of its domain. [`FTransform`][fuzzytool.ftransform.FTransform]
uses a uniform triangular partition whose basis functions sum to 1 everywhere
(the Ruspini condition):

- the **direct** transform reduces the data to `n_basis` components, each a
  membership-weighted average of the samples under one basis function;
- the **inverse** transform reconstructs `f̂(x) = Σ_k F_k · A_k(x)`.

Few components → a smoothing/denoising round trip; many components → a close
approximation.

```python
import numpy as np
import fuzzytool as fz

x = np.linspace(0, 2 * np.pi, 400)
noisy = np.sin(x) + np.random.default_rng(1).normal(0, 0.3, size=x.shape)

ft = fz.FTransform(0, 2 * np.pi, n_basis=12).fit(x, noisy)
ft.components_     # the 12 components (a compressed representation)
ft.smooth(x)       # direct-then-inverse: a denoised reconstruction
```

Use `direct(x, y)` / `inverse(x)` directly for fine control, or `fit` + `smooth`
for the common round trip. The number of components trades smoothing against
fidelity — more basis functions track the signal more closely.
