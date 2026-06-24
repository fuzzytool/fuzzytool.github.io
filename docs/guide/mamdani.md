# Mamdani inference

[`Mamdani`][fuzzytool.inference.mamdani.Mamdani] consequents are fuzzy sets
(`output is term`). For each call:

1. Evaluate every rule's antecedent → a firing strength.
2. Shape the consequent set over the output universe via the **implication**
   operator (`min` clips, `prod` scales).
3. **Aggregate** shaped sets per output variable (s-norm, default `max`).
4. **Defuzzify** to a crisp value.

```python
import fuzzytool as fz

sys = fz.Mamdani(
    tnorm="min", snorm="max",     # antecedent connectives
    implication="min",            # "min" (clip) or "prod" (scale)
    aggregation="max",            # combine shaped output sets
    defuzz="centroid",            # see the defuzzification guide
)
```

## Multiple outputs

Add rules whose consequents reference different output variables; calling the
system returns a `dict` keyed by output name (a single output returns a float).

```python
sys.rule(x["hi"], y1["a"])
sys.rule(x["lo"], y2["b"])
sys(x=0.7)   # -> {"y1": ..., "y2": ...}
```
