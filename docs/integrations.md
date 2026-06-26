# Integrations

fuzzytool's core is pure NumPy. Optional integrations connect it to the wider
data ecosystem; each lives under `fuzzytool.integrations.*` and pulls in its
third-party dependency **only when imported**, so the base install stays light.

| Integration | Module | Extra |
|---|---|---|
| pandas | `fuzzytool.integrations.pandas` | `pip install fuzzytool[pandas]` |
| scikit-learn | `fuzzytool.integrations.sklearn` | `pip install fuzzytool[sklearn]` |
| PyTorch | `fuzzytool.integrations.torch` | `pip install fuzzytool[torch]` |

More (SciPy tuning, Optuna search, LLM agent tools) are on the roadmap.

## pandas

DataFrame in/out and tabular views of fuzzy objects.

### Batch inference from a DataFrame

`predict_df` matches DataFrame columns to the system's input variables and
returns a `Series` (single output) or `DataFrame` (multiple outputs), aligned to
the original index:

```python
import pandas as pd
from fuzzytool import datasets
from fuzzytool.integrations.pandas import predict_df

sys, *_ = datasets.credit_risk()
df = pd.DataFrame({"score": [520.0, 660.0, 800.0], "dti": [42.0, 30.0, 10.0]})

predict_df(sys, df)
# 0    10.155529
# 1     6.000000
# 2     1.908375
# Name: premium, dtype: float64
```

### Inspect the rule base as a table

```python
from fuzzytool.integrations.pandas import rules_dataframe

rules_dataframe(sys)
#                               antecedent         consequent  weight
# 0         (score is poor or dti is high)    premium is high     1.0
# 1    (score is fair and dti is moderate)  premium is medium     1.0
# 2  (score is good or score is excellent)     premium is low     1.0
```

### Cluster memberships and F-transform components

```python
from fuzzytool.integrations.pandas import memberships_dataframe, components_dataframe

memberships_dataframe(result)   # one row per sample: cluster_0..cluster_k + label
components_dataframe(ftransform) # one row per basis node: node, component
```

## scikit-learn

These objects follow the estimator protocol (`fit`/`predict`/`transform`,
`get_params`/`set_params`, `score`), so they drop into a `Pipeline`,
`GridSearchCV` or `cross_val_score` — but, like
[`ANFIS`](guide/anfis.md), they do **not** import scikit-learn themselves. The
`[sklearn]` extra just installs scikit-learn so you have those tools.

### Fuzzification as a feature transformer

`Fuzzifier` expands each crisp feature into one column per fuzzy term — a soft,
interpretable encoding any downstream model can consume:

```python
import numpy as np
import fuzzytool as fz
from fuzzytool.integrations.sklearn import Fuzzifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

a = fz.Variable("a", (0, 10), terms=["lo", "mid", "hi"])
b = fz.Variable("b", (0, 10), terms=["lo", "mid", "hi"])

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(200, 2))
y = (X[:, 0] + X[:, 1]) / 2

pipe = make_pipeline(Fuzzifier([a, b]), LinearRegression())
cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()   # -> ~1.0
```

`Fuzzifier([a, b]).fit_transform(X)` turns the 2 raw columns into 6 membership
columns (`a[lo], a[mid], a[hi], b[lo], b[mid], b[hi]`), and
`get_feature_names_out()` returns those names.

### Learning a rule base as an estimator

`WangMendelRegressor` learns a Mamdani rule base from data on `fit` and predicts
with it:

```python
from fuzzytool.integrations.sklearn import WangMendelRegressor

out = fz.Variable("out", (0, 10), terms=["lo", "mid", "hi"])
reg = WangMendelRegressor([a, b], out).fit(X, y)
reg.predict(np.array([[8.0, 7.0]]))   # -> ~5.9
reg.score(X, y)                        # R^2 on the training data
```

### Scoring a hand-written system

`FuzzySystemRegressor` wraps an existing single-output Mamdani/TSK system so you
can cross-validate or pipeline it without changing it:

```python
from fuzzytool.integrations.sklearn import FuzzySystemRegressor

reg = FuzzySystemRegressor(sys, columns=["score", "dti"]).fit(X_credit)
reg.predict(X_credit)   # same as sys.predict(score=..., dti=...)
```

## PyTorch

`FuzzyLayer` is a first-order Takagi-Sugeno system written as a
`torch.nn.Module`: its Gaussian membership functions and affine consequents are
plain `Parameter`s, so the whole layer is **differentiable** and trains by
backprop — standalone, or as one block inside a larger network. It is the
gradient-based sibling of [`ANFIS`](guide/anfis.md).

### Train it on its own

A convenience `fit` (Adam + MSE) returns the RMSE history:

```python
import numpy as np
from fuzzytool.integrations.torch import FuzzyLayer

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=(400, 2)).astype("float32")
y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2            # a nonlinear target

layer = FuzzyLayer(n_inputs=2, n_mf=4, init_range=(0, 1))
history = layer.fit(X, y, epochs=300, lr=0.05)
history[0], history[-1]    # -> (~1.07, ~0.008)  RMSE before/after training
```

Scale your inputs (or pass `init_range`) so the membership centers start spread
across the data.

### Compose it into a network

Because it is a regular `nn.Module` with shape `(batch, n_inputs) -> (batch, 1)`,
it drops into any model and trains end-to-end with autograd:

```python
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(8, 2),       # upstream features -> 2 fuzzy inputs
    FuzzyLayer(2, n_mf=3), # interpretable fuzzy reasoning block
)
loss = nn.MSELoss()(net(torch.randn(16, 8)), torch.randn(16, 1))
loss.backward()            # gradients flow through the membership functions
```
