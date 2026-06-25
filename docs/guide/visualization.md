# Visualization

`fuzzytool.viz` needs matplotlib: `pip install fuzzytool[viz]`.

## Membership functions

```python
import matplotlib.pyplot as plt
import fuzzytool as fz
from fuzzytool import viz

premium = fz.Variable("premium", (0, 12), terms=["low", "medium", "high"])
viz.plot_variable(premium)
plt.show()
```

![Membership functions of the premium variable: low, medium and high terms](../images/premium_terms.png)

## Control surface

For a single-output system over two inputs:

```python
from fuzzytool import datasets, viz

sys, score, dti, premium = datasets.credit_risk()
viz.control_surface(sys, score, dti)
plt.show()
```

![Control surface of the credit-risk system: premium as a function of score and dti](../images/credit_control_surface.png)
