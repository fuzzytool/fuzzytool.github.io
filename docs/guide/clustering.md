# Fuzzy clustering

Where crisp k-means assigns each point to exactly one cluster, fuzzy clustering
gives each point a graded membership in every cluster. `fuzzytool.cluster` works
on plain NumPy arrays of shape `(n_samples, n_features)` and offers three
algorithms plus validity metrics. Every algorithm takes a `seed` for
reproducibility.

| Function | Norm / idea | Good for |
|---|---|---|
| [`fuzzy_cmeans`][fuzzytool.cluster.fuzzy_cmeans] | Euclidean, memberships sum to 1 | spherical clusters |
| [`gustafson_kessel`][fuzzytool.cluster.gustafson_kessel] | adaptive per-cluster Mahalanobis norm | ellipsoidal clusters |
| [`possibilistic_cmeans`][fuzzytool.cluster.possibilistic_cmeans] | typicalities (no sum-to-1) | noisy data / outliers |

```python
import fuzzytool as fz
from fuzzytool import cluster
from fuzzytool.datasets import make_blobs

X = make_blobs(centers=((0, 0), (6, 6), (0, 6)), seed=0)

res = fz.fuzzy_cmeans(X, c=3, m=2.0, seed=0)
res.centers     # (3, 2) cluster prototypes
res.u           # (3, n) membership matrix (columns sum to 1)
res.labels      # hard assignment, argmax over u
res.n_iter      # iterations to converge
```

## Validity metrics

How many clusters? Compare runs with internal validity indices:

```python
cluster.partition_coefficient(res.u)   # in (1/c, 1]; higher = crisper
cluster.partition_entropy(res.u)       # in [0, log c); lower = crisper
cluster.xie_beni(X, res.centers, res.u)  # compactness/separation; lower = better
```

A common recipe is to sweep `c` and pick the value that maximizes the partition
coefficient (or minimizes Xie-Beni).

## Visualization

```python
import matplotlib.pyplot as plt
from fuzzytool import viz

viz.plot_clusters(X, res)   # 2-D scatter; opacity encodes top membership
plt.show()
```
