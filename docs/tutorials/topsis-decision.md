# Tutorial: choosing between alternatives with Fuzzy TOPSIS

You rarely rate options with exact numbers — ratings like "high return" or
"low fees" are fuzzy. **Fuzzy TOPSIS** ranks alternatives described by triangular
fuzzy numbers (TFNs) by their closeness to the ideal solution. Here we pick an
investment fund across four criteria.

## 1. Frame the decision

Three funds, four criteria. Two criteria are **benefit** (more is better:
*return*, *liquidity*) and two are **cost** (less is better: *risk*, *fees*).
Each cell is a TFN `(low, mid, high)`:

```python
from fuzzytool.fuzzynum import tfn
from fuzzytool.mcdm import fuzzy_topsis

funds = ["Fund A", "Fund B", "Fund C"]
#                 return        risk          fees          liquidity
matrix = [
    [tfn(7, 8, 9), tfn(6, 7, 8), tfn(2, 3, 4), tfn(7, 8, 9)],   # A: high return, higher risk
    [tfn(5, 6, 7), tfn(3, 4, 5), tfn(1, 2, 3), tfn(8, 9, 9)],   # B: steady, cheap, liquid
    [tfn(8, 9, 9), tfn(7, 8, 9), tfn(5, 6, 7), tfn(4, 5, 6)],   # C: aggressive, pricey
]
```

## 2. Weight the criteria

Weights are TFNs too — here *return* matters most, *fees* and *liquidity* least:

```python
weights = [
    tfn(0.3, 0.4, 0.5),   # return
    tfn(0.2, 0.3, 0.4),   # risk
    tfn(0.1, 0.2, 0.3),   # fees
    tfn(0.1, 0.2, 0.3),   # liquidity
]
```

## 3. Rank

`benefit[j]` tells the method whether criterion `j` is maximized:

```python
res = fuzzy_topsis(matrix, weights, benefit=[True, False, False, True])

res.closeness   # -> array([0.202, 0.233, 0.173])  closeness coefficient per fund
res.ranking     # -> [1, 0, 2]                     fund indices, best first
```

## 4. Interpret

`ranking` is `[1, 0, 2]`, so the order is **Fund B → Fund A → Fund C**. Despite
Fund C's top raw return, its high risk *and* fees push it last; Fund B wins by
being steady, cheap and liquid — exactly the trade-off the weights encode.

```python
[funds[i] for i in res.ranking]   # -> ['Fund B', 'Fund A', 'Fund C']
```

## Where to go next

- Derive criterion weights from pairwise comparisons instead of guessing them:
  **Fuzzy AHP** in [Fuzzy numbers & MCDM](../guide/mcdm.md).
- Build the decision matrix from triangular/trapezoidal fuzzy-number arithmetic
  in `fuzzytool.fuzzynum`.
