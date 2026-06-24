# Contributing

Thanks for your interest in fuzzytool!

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,docs]"
```

## Checks (run before opening a PR)

```bash
ruff check fuzzytool tests     # lint
pytest -q                      # tests + doctests
mkdocs build --strict          # docs build cleanly
```

## Conventions

- **Language:** all code, comments, identifiers, and prose are in **English**
  (the project is published internationally on GitHub and PyPI).
- **Tests:** every new membership function, connective, defuzzifier, or engine
  needs a test. Inference engines need at least one behavioral test (e.g. a
  monotonicity or known-output check).
- **Docstrings:** Google style (rendered by `mkdocstrings`).
- **Extensibility first:** prefer adding a callable behind an existing Protocol
  over special-casing the inference loop. See [`docs/extending.md`](docs/extending.md).

## Adding a feature

1. Implement it in the relevant module.
2. Export it where appropriate (`__init__.py` for public API).
3. Add tests and a docs entry.
4. Update `CHANGELOG.md` and, if it completes a milestone, `ROADMAP.md`.
