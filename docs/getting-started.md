# Getting started

We build a **credit-risk premium** system: a lender turns a borrower's credit
**score** (300–850) and **debt-to-income ratio** (`dti`, 0–50%) into a risk
**premium** (0–12 points) to add on top of its base interest rate.

## 1. Define linguistic variables

A [`Variable`][fuzzytool.sets.Variable] is a named universe plus named *terms*
(fuzzy sets). Generate terms automatically, or set them explicitly.

```python
import fuzzytool as fz

# Explicit membership functions:
score = fz.Variable("score", (300, 850))
score["poor"] = fz.trap(300, 300, 500, 600)
score["fair"] = fz.tri(560, 660, 730)
score["good"] = fz.tri(690, 760, 810)
score["excellent"] = fz.trap(780, 830, 850, 850)

# Or auto-generated, evenly spaced terms:
dti = fz.Variable("dti", (0, 50), terms=["low", "moderate", "high"])
premium = fz.Variable("premium", (0, 12), terms=["low", "medium", "high"])
```

## 2. Write rules with operators

Indexing a variable with a term name gives a *proposition* you combine with
operators: `&` (AND, t-norm), `|` (OR, s-norm), `~` (NOT, complement).

```python
sys = fz.Mamdani(defuzz="centroid")
sys.rule(score["poor"] | dti["high"], premium["high"])
sys.rule(score["fair"] & dti["moderate"], premium["medium"])
sys.rule(score["good"] | score["excellent"], premium["low"])
```

## 3. Run inference

The system is **callable**. Pass crisp inputs by variable name:

```python
sys(score=800, dti=10)    # strong borrower -> low premium
sys(score=520, dti=42)    # weak borrower  -> high premium
```

## 4. Visualize (optional)

```python
import matplotlib.pyplot as plt
from fuzzytool import viz

viz.plot_variable(score)
viz.control_surface(sys, score, dti)
plt.show()
```
