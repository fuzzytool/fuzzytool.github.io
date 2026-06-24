# Extending fuzzytool

The core principle (shared with the sibling project *turboswarm*): **the
inference loop knows nothing about concrete variants.** Everything that varies
lives behind a small Protocol, so extending the library means adding a callable —
never editing the engine.

## A new membership function

Any callable `x -> degree` (vectorized over NumPy arrays) works directly:

```python
import numpy as np, fuzzytool as fz

def asym_gauss(c, sl, sr):
    def mf(x):
        x = np.asarray(x, float)
        s = np.where(x < c, sl, sr)
        return np.exp(-0.5 * ((x - c) / s) ** 2)
    return mf

v = fz.Variable("v", (0, 10))
v["mid"] = asym_gauss(5, 1.0, 3.0)
```

## A new connective (t-norm / s-norm)

Register a vectorized `(a, b) -> result`:

```python
from fuzzytool import norms

def t_einstein(a, b):
    return (a * b) / (2 - (a + b - a * b))

norms._TNORMS["einstein"] = t_einstein
sys = fz.Mamdani(tnorm="einstein")
```

Or pass the callable straight to the engine: `fz.Mamdani(tnorm=t_einstein)`.

## A new defuzzifier

A callable `(x, y) -> float`:

```python
sys = fz.Mamdani(defuzz=lambda x, y: float((x * y).sum() / y.sum()))
```

## A new inference engine

Implement `__call__(**inputs)` and reuse the antecedent tree
(`antecedent.eval(inputs, tnorm, snorm)`) plus the shared `Rule`. See
`fuzzytool/inference/mamdani.py` and `tsk.py` as templates.
