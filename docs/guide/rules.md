# Variables & rules

## Linguistic variables

```python
import fuzzytool as fz

# Auto-generate evenly-spaced terms ("triangular" default, or "gauss"):
temp = fz.Variable("temp", (0, 40), terms=["cold", "warm", "hot"], kind="gauss")

# Or assign terms explicitly:
temp["freezing"] = fz.trap(-10, -10, 0, 5)
```

## Rules with operators

Indexing a variable yields a *proposition* (`temp["hot"]`). Compose propositions:

| Operator | Meaning | Default |
|---|---|---|
| `&` | AND | t-norm `min` |
| `\|` | OR  | s-norm `max` |
| `~` | NOT | complement `1 - μ` |

```python
fan = fz.Variable("fan", (0, 100), terms=["off", "low", "high"])

sys = fz.Mamdani()
sys.rule(temp["hot"] & ~temp["cold"], fan["high"])
sys.rule(temp["warm"], fan["low"])
```

Each rule may carry a `weight` in `[0, 1]`:

```python
sys.rule(temp["cold"], fan["off"], weight=0.5)
```

## Choosing connectives

Pass them to the engine by name (or supply your own callable):

```python
sys = fz.Mamdani(tnorm="prod", snorm="probor")
```

See [`fuzzytool.norms`][fuzzytool.norms] for the available t-norms and s-norms.
