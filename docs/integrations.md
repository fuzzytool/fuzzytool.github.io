# Integrations

fuzzytool's core is pure NumPy. Optional integrations connect it to the wider
data ecosystem; each lives under `fuzzytool.integrations.*` and pulls in its
third-party dependency **only when imported**, so the base install stays light.

| Integration | Module | Extra |
|---|---|---|
| pandas | `fuzzytool.integrations.pandas` | `pip install fuzzytool[pandas]` |
| scikit-learn | `fuzzytool.integrations.sklearn` | `pip install fuzzytool[sklearn]` |
| PyTorch | `fuzzytool.integrations.torch` | `pip install fuzzytool[torch]` |
| SciPy | `fuzzytool.integrations.scipy` | `pip install fuzzytool[scipy]` |
| Optuna | `fuzzytool.integrations.optuna` | `pip install fuzzytool[optuna]` |
| Joblib / Dask | `fuzzytool.integrations.parallel` | `pip install fuzzytool[parallel]` / `[dask]` |
| LLM agents | `fuzzytool.integrations.agents` | `pip install fuzzytool[agents]` |

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

## SciPy

`tune` refines the membership-function parameters of an existing Mamdani/TSK
system to fit data, via `scipy.optimize.least_squares`. It mutates the system in
place and returns SciPy's `OptimizeResult`. Shape validity is preserved every
iteration (breakpoints stay ordered, widths stay positive), so the optimizer
explores freely:

```python
import numpy as np
import fuzzytool as fz
from fuzzytool.integrations.scipy import tune

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(150, 1))
y = np.sin(X[:, 0]) + 0.1 * X[:, 0]

x = fz.Variable("x", (0, 10), terms=["lo", "mid", "hi"])
sys = fz.TSK()
sys.rule(x["lo"], 0.0)
sys.rule(x["mid"], 0.5)
sys.rule(x["hi"], 1.0)

result = tune(sys, X, y, columns=["x"])
result.cost        # SciPy OptimizeResult; the system now fits the data better
```

Only built-in shapes (`tri`, `trap`, `gauss`, `gbell`, `sigmoid`, `ramp_up`,
`ramp_down`) are tuned; custom callables are left untouched. Pass
`tune_outputs=False` to keep a Mamdani's output sets fixed, and forward any
`least_squares` keyword (e.g. `max_nfev=200`).

## Optuna

Fuzzy systems have many discrete/continuous knobs. These helpers turn an Optuna
`trial` into a configured system.

`suggest_inference_spec` samples a Mamdani's connectives, implication and
defuzzifier:

```python
import optuna
from fuzzytool.integrations.optuna import suggest_inference_spec

def objective(trial):
    spec = suggest_inference_spec(trial)   # {tnorm, snorm, implication, defuzz}
    sys = fz.Mamdani(**spec)
    # ... add rules, evaluate on held-out data, return an error ...
    return error
```

`tune_anfis` is a ready-made study that searches an ANFIS's `n_mf` and
`learning_rate` and returns the best, refit model:

```python
from fuzzytool.integrations.optuna import tune_anfis

x = np.linspace(0, 6, 120).reshape(-1, 1)
y = np.sin(x[:, 0])

best, study = tune_anfis(x, y, n_trials=20, seed=0)
study.best_params      # e.g. {'n_mf': 4, 'learning_rate': 0.044}
best.predict(x)        # predictions from the tuned model
```

## Joblib / Dask

Spread embarrassingly-parallel workloads across cores or a Dask cluster.

`parallel_predict` runs batch inference in row chunks with Joblib:

```python
import numpy as np
from fuzzytool import datasets
from fuzzytool.integrations.parallel import parallel_predict, multi_start_cmeans

sys, *_ = datasets.credit_risk()
X = np.column_stack([np.random.uniform(300, 850, 100_000),
                     np.random.uniform(0, 50, 100_000)])

parallel_predict(sys, X, columns=["score", "dti"], n_jobs=-1)   # array of premiums
```

`multi_start_cmeans` runs fuzzy c-means from many seeds in parallel and keeps the
best (FCM is sensitive to initialization):

```python
X = datasets.make_blobs(seed=0)
best = multi_start_cmeans(X, c=3, n_starts=8, n_jobs=-1)
best.objective   # lowest objective across the restarts
```

The process backends pickle the system; built-in Mamdani/TSK systems pickle
fine, but a TSK with `lambda` consequents needs `backend="threading"`. For a
distributed scheduler, `dask_predict(sys, X, columns=...)` builds the same
computation as a Dask graph (`pip install fuzzytool[dask]`).

## LLM agents

A fuzzy system is a great tool for an LLM agent: deterministic, bounded, and
**self-explaining**. `explain` runs a system and reports which rules fired —
no third-party dependency:

```python
from fuzzytool.integrations.agents import explain

explain(sys, score=520, dti=42)
# {'output': 10.16,
#  'fired_rules': [{'rule': 'IF (score is poor or dti is high) THEN premium is high',
#                   'firing': 0.8}]}
```

`inference_tool` wraps the system as a LangChain `StructuredTool` an agent can
call (one float argument per input variable):

```python
from fuzzytool.integrations.agents import inference_tool

tool = inference_tool(sys, columns=["score", "dti"])
tool.invoke({"score": 800, "dti": 10})
# {'premium': 1.91,
#  'fired_rules': [{'rule': 'IF (score is good or score is excellent) THEN premium is low',
#                   'firing': 0.4}]}
```
