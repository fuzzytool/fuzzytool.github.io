# Defuzzification

A defuzzifier collapses the aggregated output set into a crisp value. Select by
name in a Mamdani system (`fz.Mamdani(defuzz=...)`) or call the functions
directly from [`fuzzytool.defuzz`][fuzzytool.defuzz].

| Name | Method |
|---|---|
| `centroid` | center of gravity (default) |
| `bisector` | splits the area into two equal halves |
| `mom` | mean of maxima |
| `som` | smallest of maxima |
| `lom` | largest of maxima |

```python
from fuzzytool import defuzz
import numpy as np

x = np.linspace(0, 30, 501)
y = np.maximum(0, 1 - np.abs(x - 20) / 10)
defuzz.centroid(x, y)   # ~20
```

A custom defuzzifier is any callable `(x, y) -> float`; pass it directly:

```python
fz.Mamdani(defuzz=lambda x, y: float(x[y.argmax()]))
```
