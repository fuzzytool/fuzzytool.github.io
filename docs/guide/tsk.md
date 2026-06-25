# Takagi-Sugeno (TSK) inference

[`TSK`][fuzzytool.inference.tsk.TSK] consequents are crisp functions of the
inputs, so there is **no defuzzification**: the output is the firing-weighted
average of the rule consequents.

A consequent may be:

- a **number** — zero-order (Sugeno constant);
- a **mapping** `{"const": b0, "x": b1, ...}` — first-order linear in the inputs;
- any **callable** `f(**inputs) -> float`.

```python
import fuzzytool as fz

x = fz.Variable("x", (0, 10), terms=["small", "large"])

sys = fz.TSK()
sys.rule(x["small"], 0.0)                       # zero-order
sys.rule(x["large"], {"const": 1.0, "x": 1.0})  # first-order: 1 + x
sys.rule(x["small"], lambda x: x**2)            # arbitrary callable

sys(x=10.0)    # -> 11.0
```

TSK systems are typically faster and differentiable, which makes them the basis
for trainable neuro-fuzzy models (ANFIS, on the roadmap).
