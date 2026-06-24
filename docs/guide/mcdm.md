# Fuzzy numbers & MCDM

## Fuzzy numbers

[`fuzzytool.fuzzynum`][fuzzytool.fuzzynum] provides triangular (TFN) and
trapezoidal (TrFN) fuzzy numbers with arithmetic, alpha-cuts, a crisp centroid,
and a vertex distance.

```python
from fuzzytool.fuzzynum import tfn, trfn, rank

a, b = tfn(1, 2, 3), tfn(2, 3, 4)
a + b              # TFN(3, 5, 7)
a * 2              # TFN(2, 4, 6)
a.centroid()       # 2.0
a.alpha_cut(0.5)   # (1.5, 2.5)
rank([tfn(0,1,2), tfn(5,6,7)])   # [1, 0] (largest first)
```

`+`/`-` are exact; `*`/`/` use the standard positive-support approximation.

## Fuzzy multi-criteria decision making

[`fuzzytool.mcdm`][fuzzytool.mcdm] offers two classic methods.

### Fuzzy TOPSIS (Chen)

Rank alternatives by closeness to the fuzzy positive-ideal solution. The decision
matrix and weights are triangular fuzzy numbers; `benefit[j]` says whether
criterion `j` is maximized.

```python
from fuzzytool.fuzzynum import tfn
from fuzzytool.mcdm import fuzzy_topsis

matrix = [
    [tfn(7, 8, 9), tfn(5, 6, 7)],   # alternative A: ratings per criterion
    [tfn(3, 4, 5), tfn(8, 9, 9)],   # alternative B
]
weights = [tfn(0.4, 0.5, 0.6), tfn(0.3, 0.4, 0.5)]
res = fuzzy_topsis(matrix, weights, benefit=[True, True])
res.closeness   # closeness coefficient per alternative
res.ranking     # alternative indices, best first
```

### Fuzzy AHP (Chang's extent analysis)

Derive crisp criterion weights from a fuzzy pairwise-comparison matrix.

```python
from fuzzytool.fuzzynum import tfn
from fuzzytool.mcdm import fuzzy_ahp

one = tfn(1, 1, 1)
matrix = [
    [one,                  tfn(2, 3, 4)],
    [tfn(1/4, 1/3, 1/2),   one],
]
fuzzy_ahp(matrix)   # normalized weights, e.g. [0.74, 0.26]
```
