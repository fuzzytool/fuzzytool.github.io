# Membership functions

A membership function (MF) is any callable mapping a crisp value (scalar or
NumPy array) to a degree in `[0, 1]`. The built-in shapes:

| Factory | Shape | Parameters |
|---|---|---|
| `fz.tri(a, b, c)` | triangular | feet `a`/`c`, peak `b` |
| `fz.trap(a, b, c, d)` | trapezoidal | shoulders `a`/`d`, flat top `b`..`c` |
| `fz.gauss(c, sigma)` | gaussian | center `c`, spread `sigma` |
| `fz.gbell(a, b, c)` | generalized bell | width `a`, slope `b`, center `c` |
| `fz.sigmoid(a, c)` | sigmoidal | slope `a`, inflection `c` |

```python
import numpy as np, fuzzytool as fz

m = fz.gauss(5, 1.5)
m(5)                       # 1.0
m(np.array([3, 5, 7]))     # vectorized
```

## Custom shapes

Any callable works — no registration needed:

```python
def ramp(x):
    return np.clip(np.asarray(x, float) / 10, 0, 1)

var = fz.Variable("x", (0, 10))
var["high"] = ramp
```

This is the core extensibility idea: the inference engine only relies on the
`MembershipFunction` Protocol (`x -> degree`), so a new shape is just a new
callable.
