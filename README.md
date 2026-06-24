<p align="center">
  <a href="https://pypi.org/project/fuzzytool/"><img src="https://img.shields.io/pypi/v/fuzzytool?logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/fuzzytool/fuzzytool.github.io/actions/workflows/ci.yml"><img src="https://github.com/fuzzytool/fuzzytool.github.io/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://fuzzytool.github.io/"><img src="https://img.shields.io/badge/docs-fuzzytool.github.io-3f51b5" alt="Docs"></a>
  <img src="https://img.shields.io/pypi/pyversions/fuzzytool?logo=python&logoColor=white" alt="Python versions">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
</p>

# fuzzytool

A clean, **extensible fuzzy-logic toolkit** in **pure Python + NumPy**. Its
design priorities are a composable API, algorithm comparison, visualization and
code clarity — a modern alternative to the verbose control API of scikit-fuzzy.

```python
import fuzzytool as fz

# Credit-risk premium: a lender turns a credit score + debt-to-income ratio
# into the risk points it adds on top of its base interest rate.
score   = fz.Variable("score", (300, 850), terms=["poor", "fair", "good", "excellent"])
dti     = fz.Variable("dti", (0, 50), terms=["low", "moderate", "high"])
premium = fz.Variable("premium", (0, 12), terms=["low", "medium", "high"])

sys = fz.Mamdani(defuzz="centroid")
sys.rule(score["poor"] | dti["high"], premium["high"])        # |=OR  &=AND  ~=NOT
sys.rule(score["fair"] & dti["moderate"], premium["medium"])
sys.rule(score["good"] | score["excellent"], premium["low"])

print(sys(score=800, dti=10))    # the system is just callable -> a low premium
```

## The design idea (extensibility)

The inference loop knows **nothing** about any concrete variant. Everything that
changes lives behind small Python **Protocols**:

- **`MembershipFunction`** (`fuzzytool/membership.py`) — a callable `x -> degree`.
  A new shape = a new callable.
- **`Norm`** (`fuzzytool/norms.py`) — t-norms (AND) and s-norms (OR), resolved by
  name. A new connective = one registered function.
- **defuzzifiers** (`fuzzytool/defuzz.py`) — centroid, bisector, MOM/SOM/LOM,
  resolved by name.

Rules read like logic thanks to operator overloading: `&` is the t-norm, `|` the
s-norm, `~` the complement.

## What it includes / roadmap

| Phase | Content | Status |
|------|-----------|--------|
| 1 | Core: membership functions, t-/s-norms, `Variable`, operator rules, **Mamdani** + defuzzification, tipper example, tests | ✅ |
| 2 | **Takagi-Sugeno (TSK)** inference + `viz` (membership plots, control surface) | ✅ (TSK + viz) |
| 3 | **Type-2 / interval type-2** sets (footprint of uncertainty) + Karnik-Mendel type reduction | ⏳ |
| 4 | **Fuzzy clustering**: fuzzy c-means, Gustafson-Kessel, possibilistic | ⏳ |
| 5 | **ANFIS** (trainable TSK) + **F-transform** (direct/inverse) | ⏳ |
| 6 | Notebooks, JOSS `paper.md`, Zenodo DOI, PyPI release | ⏳ |

See [`ROADMAP.md`](ROADMAP.md).

## Install

```bash
pip install fuzzytool            # core (NumPy only)
pip install fuzzytool[viz]       # + matplotlib visualization
```

From source, for development:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,docs]"
pytest -q
python examples/tipper.py
```

## Documentation

A documentation portal (narrative guide + API reference from docstrings) is built
with **MkDocs Material** and published to **GitHub Pages**:
<https://fuzzytool.github.io/>.

```bash
pip install -e ".[docs]"
mkdocs serve        # live portal at http://127.0.0.1:8000
```

## License

MIT. See [`LICENSE`](LICENSE).
