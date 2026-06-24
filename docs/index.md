# fuzzytool

A clean, **extensible fuzzy-logic toolkit** in **pure Python + NumPy**. Design
priorities: a composable API, algorithm comparison, visualization and code
clarity — a modern alternative to the verbose control API of scikit-fuzzy.

```python
import fuzzytool as fz

# Credit-risk premium from a credit score + debt-to-income ratio.
score   = fz.Variable("score", (300, 850), terms=["poor", "fair", "good", "excellent"])
dti     = fz.Variable("dti", (0, 50), terms=["low", "moderate", "high"])
premium = fz.Variable("premium", (0, 12), terms=["low", "medium", "high"])

sys = fz.Mamdani(defuzz="centroid")
sys.rule(score["poor"] | dti["high"], premium["high"])
sys.rule(score["fair"] & dti["moderate"], premium["medium"])
sys.rule(score["good"] | score["excellent"], premium["low"])

sys(score=800, dti=10)   # -> a crisp risk premium
```

## Why fuzzytool

- **Rules read like logic.** `&` is the t-norm (AND), `|` the s-norm (OR), `~`
  the complement (NOT). No `Antecedent`/`Consequent`/`ControlSystem` boilerplate.
- **Everything is pluggable** behind small Protocols — membership functions,
  connectives, defuzzifiers. Adding a variant is adding a callable, never editing
  the inference loop. See [Extending](extending.md).
- **Pure Python + NumPy.** Universal wheel, no compilation.

## Next steps

- [Installation](installation.md)
- [Getting started](getting-started.md)
- [The guide](guide/membership.md)
