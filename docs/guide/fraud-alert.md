# Worked example: fraud-alert score

A complete, runnable Mamdani system that scores card transactions for fraud. A
payment monitor turns how far a charge deviates from the customer's usual spend
(**amount anomaly**, in standard deviations, 0–8) and how many card swipes
happened in the last hour (**velocity**, 0–30) into a **fraud-risk** score
(0–100).

The full script lives in
[`examples/fraud_alert.py`](https://github.com/fuzzytool/fuzzytool/blob/main/examples/fraud_alert.py).

## 1. Define the linguistic variables

```python
import fuzzytool as fz

amount_anomaly = fz.Variable("amount_anomaly", (0, 8))
amount_anomaly["normal"] = fz.trap(0, 0, 1, 2.5)
amount_anomaly["elevated"] = fz.tri(2, 3.5, 5)
amount_anomaly["extreme"] = fz.trap(4.5, 6, 8, 8)

velocity = fz.Variable("velocity", (0, 30))
velocity["low"] = fz.trap(0, 0, 3, 6)
velocity["moderate"] = fz.tri(4, 9, 14)
velocity["burst"] = fz.trap(12, 20, 30, 30)

fraud_risk = fz.Variable("fraud_risk", (0, 100))
fraud_risk["low"] = fz.tri(0, 12, 35)
fraud_risk["medium"] = fz.tri(30, 50, 70)
fraud_risk["high"] = fz.tri(65, 85, 100)
```

## 2. Write the rule base

`&` is AND (t-norm), `|` is OR (s-norm), `~` is NOT (complement):

```python
sys = fz.Mamdani(defuzz="centroid")
sys.rule(amount_anomaly["extreme"] | velocity["burst"], fraud_risk["high"])
sys.rule(amount_anomaly["elevated"] & velocity["moderate"], fraud_risk["medium"])
sys.rule(amount_anomaly["normal"] & velocity["low"], fraud_risk["low"])
```

## 3. Score transactions

```python
sys(amount_anomaly=0.5, velocity=2)    # routine charge   -> ~16/100
sys(amount_anomaly=3.5, velocity=9)    # borderline       -> 50/100
sys(amount_anomaly=6.5, velocity=22)   # likely fraud     -> ~83/100
```

## 4. Visualize (optional)

```python
import matplotlib.pyplot as plt
from fuzzytool import viz

viz.plot_variable(amount_anomaly)
viz.control_surface(sys, amount_anomaly, velocity)
plt.show()
```
