# Integrations

fuzzytool's core is pure NumPy. Optional integrations connect it to the wider
data ecosystem; each lives under `fuzzytool.integrations.*` and pulls in its
third-party dependency **only when imported**, so the base install stays light.

| Integration | Module | Extra |
|---|---|---|
| pandas | `fuzzytool.integrations.pandas` | `pip install fuzzytool[pandas]` |
| scikit-learn | `fuzzytool.integrations.sklearn` | `pip install fuzzytool[sklearn]` |

More (SciPy tuning, Optuna search, PyTorch layers, LLM agent tools) are on the
roadmap.

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
