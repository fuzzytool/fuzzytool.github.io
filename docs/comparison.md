# Comparison with scikit-fuzzy

[scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) is a fuzzy-logic
package for Python. fuzzytool targets the same core use cases with a more
composable API and a broader, extensible scope.

## API ergonomics

The same Mamdani rule base in each library:

=== "fuzzytool"

    ```python
    import fuzzytool as fz

    quality = fz.Variable("quality", (0, 10), terms=["poor", "good"])
    tip     = fz.Variable("tip", (0, 25), terms=["low", "high"])

    sys = fz.Mamdani()
    sys.rule(quality["poor"], tip["low"])
    sys.rule(quality["good"], tip["high"])

    sys(quality=6.5)
    ```

=== "scikit-fuzzy"

    ```python
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    quality = ctrl.Antecedent(np.arange(0, 11, 1), "quality")
    tip = ctrl.Consequent(np.arange(0, 26, 1), "tip")
    quality.automf(2, names=["poor", "good"])
    tip.automf(2, names=["low", "high"])

    r1 = ctrl.Rule(quality["poor"], tip["low"])
    r2 = ctrl.Rule(quality["good"], tip["high"])
    system = ctrl.ControlSystem([r1, r2])
    sim = ctrl.ControlSystemSimulation(system)
    sim.input["quality"] = 6.5
    sim.compute()
    sim.output["tip"]
    ```

fuzzytool builds antecedents with operators (`|`, `&`, `~`), and the system is
just callable — there is no separate `ControlSystem` / `Simulation` step.

## Feature scope

| Capability | fuzzytool | scikit-fuzzy |
|---|:---:|:---:|
| Mamdani inference | ✅ | ✅ |
| Takagi-Sugeno (TSK) | ✅ | ⚠️ manual |
| Operator-based rule syntax | ✅ | ❌ |
| Pluggable t-/s-norms by name | ✅ | partial |
| Defuzzifiers (centroid/bisector/MOM/SOM/LOM) | ✅ | ✅ |
| Interval type-2 sets + Karnik-Mendel | ✅ | ❌ |
| Fuzzy clustering (FCM) | ✅ | ✅ (`cmeans`) |
| Gustafson-Kessel / possibilistic c-means | ✅ | ❌ |
| ANFIS (trainable TSK) | ✅ | ❌ |
| F-transform | ✅ | ❌ |
| Dependencies | NumPy (+ matplotlib for viz) | NumPy/SciPy |

## When to use which

scikit-fuzzy is mature and widely cited; if you only need a classic Mamdani
controller and already use it, it works well. Reach for **fuzzytool** when you
want a cleaner rule syntax, Takagi-Sugeno or interval type-2 systems, the extra
clustering algorithms, ANFIS, or the F-transform — all behind one small,
extensible, NumPy-only API.
