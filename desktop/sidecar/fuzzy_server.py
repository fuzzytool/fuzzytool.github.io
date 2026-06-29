"""fuzzytool desktop sidecar.

A long-running process the Tauri (Rust) backend spawns and talks to over a
line-delimited JSON protocol on stdin/stdout (one JSON object per line each way).

Systems are described by a **spec** (plain JSON) that the sidecar compiles into a
fuzzytool ``Mamdani``; the built-in demos are just predefined specs, and the
desktop editor sends the very same shape, so building your own system, adding
variables/terms and writing rules all go through one path.

    spec = {
      "name": "...", "defuzz": "centroid",
      "variables": [
        {"name": "score", "low": 300, "high": 850, "role": "input",
         "terms": [{"name": "poor", "type": "trap", "params": [300,300,500,600]}, ...]},
        {"name": "premium", "low": 0, "high": 12, "role": "output", "terms": [...]}
      ],
      "rules": [
        {"op": "or",
         "clauses": [{"var": "score", "term": "poor", "negate": false}, ...],
         "consequent": {"var": "premium", "term": "high"}, "weight": 1.0}
      ]
    }

Actions: systems / select / build / get_spec / describe / curves / infer / surface.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import fuzzytool as fz
from fuzzytool.integrations.agents import explain

DEFUZZ = ["centroid", "bisector", "mom", "som", "lom"]
# Membership-function factory + how many parameters each shape takes.
MF = {"tri": (fz.tri, 3), "trap": (fz.trap, 4), "gauss": (fz.gauss, 2)}


# --- compile a spec into a Mamdani system -----------------------------------

def build_from_spec(spec: dict):
    defuzz = spec.get("defuzz", "centroid")
    if defuzz not in DEFUZZ:
        raise ValueError(f"unknown defuzzifier {defuzz!r}")

    variables, inputs, outputs = {}, [], []
    for vs in spec.get("variables", []):
        var = fz.Variable(vs["name"], (float(vs["low"]), float(vs["high"])))
        for ts in vs.get("terms", []):
            kind = ts["type"]
            if kind not in MF:
                raise ValueError(f"unknown membership type {kind!r}")
            factory, n = MF[kind]
            params = [float(p) for p in ts["params"]]
            if len(params) != n:
                raise ValueError(f"{kind} needs {n} params, got {len(params)}")
            var[ts["name"]] = factory(*params)
        if not var.terms:
            raise ValueError(f"variable {var.name!r} has no terms")
        variables[var.name] = var
        (outputs if vs.get("role") == "output" else inputs).append(var)

    if not inputs or not outputs:
        raise ValueError("need at least one input and one output variable")

    sys_ = fz.Mamdani(defuzz=defuzz)
    for rs in spec.get("rules", []):
        clauses = rs.get("clauses", [])
        if not clauses:
            raise ValueError("a rule needs at least one clause")
        nodes = []
        for c in clauses:
            prop = variables[c["var"]][c["term"]]
            nodes.append(~prop if c.get("negate") else prop)
        ante = nodes[0]
        for node in nodes[1:]:
            ante = (ante | node) if rs.get("op", "and") == "or" else (ante & node)
        cons = rs["consequent"]
        sys_.rule(ante, variables[cons["var"]][cons["term"]],
                  float(rs.get("weight", 1.0)))
    if not sys_.rules:
        raise ValueError("the system has no rules")
    return sys_, inputs, outputs[0]


# --- built-in demos (as specs) ----------------------------------------------

def _term(name, kind, *params):
    return {"name": name, "type": kind, "params": list(params)}


DEMOS = {
    "credit_risk": {
        "name": "Credit-risk premium",
        "description": "Credit score + debt-to-income -> risk premium (points).",
        "defuzz": "centroid",
        "variables": [
            {"name": "score", "low": 300, "high": 850, "role": "input", "terms": [
                _term("poor", "trap", 300, 300, 500, 600), _term("fair", "tri", 560, 660, 730),
                _term("good", "tri", 690, 760, 810), _term("excellent", "trap", 780, 830, 850, 850)]},
            {"name": "dti", "low": 0, "high": 50, "role": "input", "terms": [
                _term("low", "trap", 0, 0, 15, 25), _term("moderate", "tri", 20, 30, 40),
                _term("high", "trap", 35, 45, 50, 50)]},
            {"name": "premium", "low": 0, "high": 12, "role": "output", "terms": [
                _term("low", "tri", 0, 1.5, 4), _term("medium", "tri", 3, 6, 9),
                _term("high", "tri", 8, 10.5, 12)]},
        ],
        "rules": [
            {"op": "or", "clauses": [{"var": "score", "term": "poor"}, {"var": "dti", "term": "high"}],
             "consequent": {"var": "premium", "term": "high"}},
            {"op": "and", "clauses": [{"var": "score", "term": "fair"}, {"var": "dti", "term": "moderate"}],
             "consequent": {"var": "premium", "term": "medium"}},
            {"op": "or", "clauses": [{"var": "score", "term": "good"}, {"var": "score", "term": "excellent"}],
             "consequent": {"var": "premium", "term": "low"}},
        ],
    },
    "fraud_alert": {
        "name": "Card-fraud alert",
        "description": "Amount anomaly + transaction velocity -> fraud risk (0-100).",
        "defuzz": "centroid",
        "variables": [
            {"name": "amount_anomaly", "low": 0, "high": 8, "role": "input", "terms": [
                _term("normal", "trap", 0, 0, 1, 2.5), _term("elevated", "tri", 2, 3.5, 5),
                _term("extreme", "trap", 4.5, 6, 8, 8)]},
            {"name": "velocity", "low": 0, "high": 30, "role": "input", "terms": [
                _term("low", "trap", 0, 0, 3, 6), _term("moderate", "tri", 4, 9, 14),
                _term("burst", "trap", 12, 20, 30, 30)]},
            {"name": "fraud_risk", "low": 0, "high": 100, "role": "output", "terms": [
                _term("low", "tri", 0, 12, 35), _term("medium", "tri", 30, 50, 70),
                _term("high", "tri", 65, 85, 100)]},
        ],
        "rules": [
            {"op": "or", "clauses": [{"var": "amount_anomaly", "term": "extreme"}, {"var": "velocity", "term": "burst"}],
             "consequent": {"var": "fraud_risk", "term": "high"}},
            {"op": "and", "clauses": [{"var": "amount_anomaly", "term": "elevated"}, {"var": "velocity", "term": "moderate"}],
             "consequent": {"var": "fraud_risk", "term": "medium"}},
            {"op": "and", "clauses": [{"var": "amount_anomaly", "term": "normal"}, {"var": "velocity", "term": "low"}],
             "consequent": {"var": "fraud_risk", "term": "low"}},
        ],
    },
    "investment_advisor": {
        "name": "Investment-risk advisor",
        "description": "Market volatility + risk tolerance -> % equity allocation.",
        "defuzz": "centroid",
        "variables": [
            {"name": "volatility", "low": 0, "high": 40, "role": "input", "terms": [
                _term("calm", "trap", 0, 0, 8, 15), _term("choppy", "tri", 10, 18, 26),
                _term("turbulent", "trap", 22, 30, 40, 40)]},
            {"name": "risk_tolerance", "low": 0, "high": 10, "role": "input", "terms": [
                _term("cautious", "trap", 0, 0, 2, 4), _term("balanced", "tri", 3, 5, 7),
                _term("aggressive", "trap", 6, 8, 10, 10)]},
            {"name": "equity", "low": 0, "high": 100, "role": "output", "terms": [
                _term("light", "tri", 0, 15, 35), _term("moderate", "tri", 30, 50, 70),
                _term("heavy", "tri", 65, 85, 100)]},
        ],
        "rules": [
            {"op": "or", "clauses": [{"var": "volatility", "term": "turbulent"}, {"var": "risk_tolerance", "term": "cautious"}],
             "consequent": {"var": "equity", "term": "light"}},
            {"op": "and", "clauses": [{"var": "volatility", "term": "choppy"}, {"var": "risk_tolerance", "term": "balanced"}],
             "consequent": {"var": "equity", "term": "moderate"}},
            {"op": "or", "clauses": [{"var": "volatility", "term": "calm"}, {"var": "risk_tolerance", "term": "aggressive"}],
             "consequent": {"var": "equity", "term": "heavy"}},
        ],
    },
}


class Session:
    def __init__(self) -> None:
        self.load_demo("credit_risk")

    def _activate(self, spec: dict) -> None:
        sys_, inputs, output = build_from_spec(spec)   # raises on a bad spec
        self.spec, self.sys, self.inputs, self.output = spec, sys_, inputs, output
        self.by_name = {v.name: v for v in inputs}

    def load_demo(self, demo_id: str) -> None:
        if demo_id not in DEMOS:
            raise ValueError(f"unknown system {demo_id!r}")
        self._activate(json.loads(json.dumps(DEMOS[demo_id])))  # deep copy

    def build(self, spec: dict) -> None:
        self._activate(spec)


SESSION = Session()


# --- actions ----------------------------------------------------------------

def _var_info(var):
    return {"name": var.name, "low": var.low, "high": var.high, "terms": list(var.terms)}


def _describe() -> dict:
    return {
        "name": SESSION.spec.get("name", "system"),
        "defuzz": SESSION.spec.get("defuzz", "centroid"),
        "inputs": [_var_info(v) for v in SESSION.inputs],
        "output": _var_info(SESSION.output),
        "rules": [str(r) for r in SESSION.sys.rules],
    }


def _systems() -> dict:
    return {
        "systems": [{"id": k, "name": v["name"], "description": v["description"]}
                    for k, v in DEMOS.items()],
        "defuzzifiers": DEFUZZ,
        "mf_types": {k: n for k, (_, n) in MF.items()},
    }


def _curves(n: int = 160) -> dict:
    out = {}
    for var in [*SESSION.inputs, SESSION.output]:
        x = np.linspace(var.low, var.high, n)
        out[var.name] = {"x": x.tolist(),
                         "terms": {name: np.asarray(mf(x), dtype=float).tolist()
                                   for name, mf in var.terms.items()}}
    return out


def _infer(inputs: dict) -> dict:
    clean = {k: float(v) for k, v in inputs.items() if k in SESSION.by_name}
    return explain(SESSION.sys, **clean)


def _surface(x_name: str, y_name: str, n: int) -> dict:
    xv, yv = SESSION.by_name[x_name], SESSION.by_name[y_name]
    xs = np.linspace(xv.low, xv.high, n)
    ys = np.linspace(yv.low, yv.high, n)
    z = [[float(SESSION.sys(**{x_name: float(x), y_name: float(y)})) for x in xs]
         for y in ys]
    return {"x": xs.tolist(), "y": ys.tolist(), "z": z,
            "x_name": x_name, "y_name": y_name, "output": SESSION.output.name}


def handle(req: dict) -> dict:
    action = req.get("action")
    if action == "systems":
        return _systems()
    if action == "select":
        SESSION.load_demo(req["system"])
        return _describe()
    if action == "build":
        SESSION.build(req["spec"])
        return _describe()
    if action == "get_spec":
        return SESSION.spec
    if action == "describe":
        return _describe()
    if action == "curves":
        return _curves(int(req.get("n", 160)))
    if action == "infer":
        return _infer(req.get("inputs", {}))
    if action == "surface":
        names = [v.name for v in SESSION.inputs]
        return _surface(req.get("x", names[0]), req.get("y", names[1] if len(names) > 1 else names[0]),
                        int(req.get("n", 40)))
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
