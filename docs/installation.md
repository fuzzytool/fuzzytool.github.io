# Installation

```bash
pip install fuzzytool            # core (NumPy only)
pip install fuzzytool[viz]       # + matplotlib for visualization
```

Requires Python ≥ 3.9.

## From source (development)

```bash
git clone https://github.com/fuzzytool/fuzzytool.github.io
cd fuzzytool.github.io
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,docs]"
pytest -q
```
