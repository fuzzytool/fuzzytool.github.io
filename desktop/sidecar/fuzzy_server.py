"""fuzzytool desktop sidecar.

A long-running process that the Tauri (Rust) backend spawns and talks to over a
line-delimited JSON protocol on stdin/stdout:

    request  (one JSON object per line on stdin):
        {"id": 1, "action": "describe"}
        {"id": 2, "action": "infer",   "inputs": {"score": 700, "dti": 20}}
        {"id": 3, "action": "surface", "x": "score", "y": "dti", "n": 30}
    response (one JSON object per line on stdout):
        {"id": 1, "ok": true,  "result": {...}}
        {"id": 2, "ok": false, "error": "..."}

It exposes the bundled ``datasets.credit_risk`` Mamdani system. The heavy lifting
(inference, fired-rule explanation, control surface) is done by fuzzytool itself;
this file is only a thin JSON shell, so the same library powers the desktop app,
the docs and the tests.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from fuzzytool import datasets
from fuzzytool.integrations.agents import explain

# The demo system: credit score + debt-to-income -> risk premium.
SYS, SCORE, DTI, PREMIUM = datasets.credit_risk()
INPUTS = {SCORE.name: SCORE, DTI.name: DTI}


def _describe() -> dict:
    def var_info(var):
        return {"name": var.name, "low": var.low, "high": var.high,
                "terms": list(var.terms)}
    return {
        "inputs": [var_info(v) for v in INPUTS.values()],
        "output": var_info(PREMIUM),
    }


def _infer(inputs: dict) -> dict:
    clean = {k: float(v) for k, v in inputs.items() if k in INPUTS}
    return explain(SYS, **clean)


def _surface(x_name: str, y_name: str, n: int) -> dict:
    xv, yv = INPUTS[x_name], INPUTS[y_name]
    xs = np.linspace(xv.low, xv.high, n)
    ys = np.linspace(yv.low, yv.high, n)
    z = [[float(SYS(**{x_name: float(x), y_name: float(y)})) for x in xs]
         for y in ys]
    return {"x": xs.tolist(), "y": ys.tolist(), "z": z,
            "x_name": x_name, "y_name": y_name, "output": PREMIUM.name}


def handle(req: dict) -> dict:
    action = req.get("action")
    if action == "describe":
        return _describe()
    if action == "infer":
        return _infer(req.get("inputs", {}))
    if action == "surface":
        return _surface(req.get("x", SCORE.name), req.get("y", DTI.name),
                        int(req.get("n", 30)))
    raise ValueError(f"unknown action {action!r}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            response = {"id": req.get("id"), "ok": True, "result": handle(req)}
        except Exception as exc:  # report errors as data, never crash the sidecar
            response = {"id": locals().get("req", {}).get("id"),
                        "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
