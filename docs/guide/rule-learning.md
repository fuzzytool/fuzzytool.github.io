# Rule learning

## Learning a rule base from data (Wang-Mendel)

[`wang_mendel`][fuzzytool.learn.wang_mendel] generates a Mamdani rule base from
data: each sample becomes one rule (picking the highest-membership term per
variable), and antecedent conflicts are resolved by rule degree.

```python
import numpy as np
import fuzzytool as fz

X = np.random.default_rng(0).uniform(0, 10, size=(300, 2))
y = (X[:, 0] + X[:, 1]) / 2

a = fz.Variable("a", (0, 10), terms=["lo", "mid", "hi"])
b = fz.Variable("b", (0, 10), terms=["lo", "mid", "hi"])
out = fz.Variable("out", (0, 10), terms=["lo", "mid", "hi"])

sys = fz.wang_mendel(X, y, [a, b], out)   # a ready-to-use Mamdani system
sys(a=8, b=7)
```

The input variables supply the partition; the learned system is a plain
`Mamdani`, so it also supports batch [`predict`](batch-and-io.md) and
serialization.

## Tsukamoto inference

[`Tsukamoto`][fuzzytool.inference.tsukamoto.Tsukamoto] consequents are
**monotonic** membership functions; a rule firing with strength `w` outputs the
value where its consequent reaches `w` (its inverse), and the system returns the
firing-weighted average — no defuzzification.

```python
import fuzzytool as fz

x = fz.Variable("x", (0, 10), terms=["lo", "hi"])
sys = fz.Tsukamoto()
sys.rule(x["lo"], fz.ramp_down(0, 30))   # monotonic consequents only
sys.rule(x["hi"], fz.ramp_up(0, 30))
sys(x=7)
```

Use [`ramp_up`][fuzzytool.membership.ramp_up],
[`ramp_down`][fuzzytool.membership.ramp_down], or
[`sigmoid`][fuzzytool.membership.sigmoid] — each exposes the `inverse` Tsukamoto
needs.
