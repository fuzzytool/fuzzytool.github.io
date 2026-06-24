# Example notebooks

Run them after installing the notebook extras:

```bash
pip install -e ".[notebooks]"
jupyter lab        # or: jupyter notebook
```

| Notebook | Topic |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Mamdani inference, membership plots, control surface |
| [`02_type2.ipynb`](02_type2.ipynb) | Interval type-2 sets (FOU) and IT2 vs type-1 inference |
| [`03_clustering.ipynb`](03_clustering.ipynb) | Fuzzy c-means, cluster plots, choosing the cluster count |
| [`04_learning.ipynb`](04_learning.ipynb) | ANFIS function approximation and F-transform denoising |

Every notebook is committed already executed, and is re-run end-to-end in CI-like
fashion when regenerated.
