# Batch inference, serialization & scikit-learn

## Batch (vectorized) inference

Calling a system evaluates one sample. `predict(**arrays)` evaluates many at once
— pass an array per variable and get an array back. Available on `Mamdani` and
`TSK`.

```python
import numpy as np
from fuzzytool import datasets

sys, *_ = datasets.credit_risk()
scores = np.array([520.0, 660.0, 800.0])
dtis   = np.array([42.0, 30.0, 10.0])

sys.predict(score=scores, dti=dtis)   # array of premiums, one per sample
```

The result matches calling the system once per sample, but the firing, implication
and aggregation steps run vectorized.

## Saving and loading systems

Serialize a Mamdani or TSK system to JSON and restore it later. Connectives and
the defuzzifier must be given **by name**, variables must use **built-in**
membership functions, and TSK consequents must be numbers or coefficient mappings.

```python
import fuzzytool as fz

fz.save(sys, "credit_risk.json")
restored = fz.load("credit_risk.json")
```

`fuzzytool.membership.to_dict`/`from_dict` and `Variable.to_dict`/`from_dict`
expose the building blocks if you need finer control.

## scikit-learn compatibility

`ANFIS` follows the estimator protocol (`fit` returns `self`, plus `predict`,
`get_params`, `set_params`), so it drops into a `Pipeline` or `GridSearchCV`
without importing scikit-learn at all:

```python
import fuzzytool as fz

model = fz.ANFIS(n_inputs=2, n_mf=3)
model.get_params()              # {'n_inputs': 2, 'n_mf': 3, 'learning_rate': 0.05}
clone = fz.ANFIS(**model.get_params())
```
