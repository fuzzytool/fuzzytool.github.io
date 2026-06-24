# ANFIS (trainable TSK)

[`ANFIS`][fuzzytool.anfis.ANFIS] is a first-order Takagi-Sugeno system whose
parameters are **learned from data** (Jang, 1993). Over a grid partition, each of
the `p` inputs gets `n_mf` Gaussian membership functions, giving `n_mf ** p`
rules; each rule emits an affine function of the inputs.

Training is Jang's **hybrid** scheme, one pass per epoch:

1. with the premise (Gaussian) parameters fixed, the affine consequents are
   solved in closed form by **least squares** (the output is linear in them);
2. with the consequents fixed, the premise centers and widths take a
   **gradient-descent** step on the MSE.

```python
import numpy as np
import fuzzytool as fz

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

model = fz.ANFIS(n_inputs=1, n_mf=6).fit(x[:, None], y, epochs=100)
model.predict(x[:, None])     # approximates sin(x)
model.history_                # RMSE per epoch
```

`X` is always 2-D `(n_samples, n_features)`. Because the rule count grows as
`n_mf ** p`, ANFIS is meant for low-dimensional problems; keep `n_mf` modest as
`p` grows.
