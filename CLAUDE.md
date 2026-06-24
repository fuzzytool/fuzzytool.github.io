# CLAUDE.md

Guide for working in this repository with Claude Code.

## What this project is

**fuzzytool** — a clean, extensible **fuzzy-logic toolkit** in **pure Python +
NumPy**, published on PyPI with docs on GitHub Pages. Priorities, in order:
1) a composable API, 2) algorithm comparison, 3) visualization, 4) code clarity.
It is the fuzzy-logic sibling of the *turboswarm* PSO library and follows the
same packaging and extensibility philosophy.

## Architecture

```
fuzzytool/
  membership.py     MF Protocol + tri/trap/gauss/gbell/sigmoid
  norms.py          t-norms (AND) / s-norms (OR), resolved by name
  sets.py           FuzzySet, Variable, antecedent tree (& | ~)
  rules.py          Rule dataclass (shared by the engines)
  defuzz.py         centroid / bisector / mom / som / lom
  inference/        mamdani.py, tsk.py
  viz.py            matplotlib: membership plots, control surface
  datasets.py       example systems (credit_risk)
tests/  examples/  docs/
```

### The central design (read before touching the engines)

The inference loop knows **nothing** about concrete variants. Everything that
changes lives behind small Protocols:

- **`MembershipFunction`** (`membership.py`) — a callable `x -> degree`. A new
  shape is a new callable; nothing else changes.
- **`Norm`** (`norms.py`) — t-/s-norms, resolved by name via `get_tnorm` /
  `get_snorm`. You may also pass a callable directly to an engine.
- **defuzzifiers** (`defuzz.py`) — resolved by name via `get_defuzzifier`.

Rules compose with operators on `Proposition` objects: `&` → t-norm (AND),
`|` → s-norm (OR), `~` → complement (NOT). The antecedent tree is evaluated by
`Antecedent.eval(inputs, tnorm, snorm)`; both engines reuse it.

`Mamdani` consequents are propositions (`output[term]`); `TSK` consequents are
numbers, coefficient mappings, or callables.

## Commands (verified)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,docs]"
pytest -q                      # tests + doctests (--doctest-modules)
ruff check fuzzytool tests
python examples/credit_risk.py
mkdocs serve                   # docs at http://127.0.0.1:8000
mkdocs build --strict
```

## Conventions

- **Language:** all code, comments, identifiers and prose in **English**
  (published internationally on GitHub and PyPI).
- **Vectorize:** membership functions accept scalars *and* NumPy arrays.
- **NumPy 2 compat:** `np.trapz` was renamed `np.trapezoid`; `defuzz.py` shims
  both — do not call `np.trapz` directly elsewhere.
- **Tests:** every new MF / connective / defuzzifier / engine needs a test;
  engines need a behavioral check (monotonicity or known output).
- **Docstrings:** Google style (rendered by `mkdocstrings`).

## How to extend (typical tasks)

- **New membership function:** any callable `x -> degree`; add a factory in
  `membership.py` if it is a standard shape, plus a test.
- **New connective:** register a `(a, b) -> result` in `norms._TNORMS` /
  `_SNORMS`, or pass it directly to the engine.
- **New defuzzifier:** add a `(x, y) -> float` to `defuzz._METHODS`.
- **New engine:** implement `__call__(**inputs)`, reuse `Antecedent.eval` and the
  shared `Rule`. Templates: `inference/mamdani.py`, `inference/tsk.py`.

## Status

See `ROADMAP.md`. Phase 1 (core + Mamdani) and Phase 2 (TSK + viz) are done;
type-2, clustering, ANFIS and F-transform are planned.
