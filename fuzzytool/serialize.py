"""Save and load fuzzy inference systems as JSON.

Supports :class:`~fuzzytool.inference.Mamdani` and
:class:`~fuzzytool.inference.TSK` systems whose connectives/defuzzifier were
given **by name** and whose variables use **built-in** membership functions.
TSK consequents must be numbers or coefficient mappings (callable consequents
cannot be serialized).

    fz.save(system, "system.json")
    system = fz.load("system.json")
"""

from __future__ import annotations

import json
from numbers import Real

from .inference import TSK, Mamdani
from .sets import And, Not, Or, Proposition, Variable, antecedent_from_dict


def _collect_variables(system) -> dict[str, Variable]:
    found: dict[str, Variable] = {}

    def walk(node):
        if isinstance(node, Proposition):
            found[node.variable.name] = node.variable
        elif isinstance(node, (And, Or)):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, Not):
            walk(node.operand)

    for r in system.rules:
        walk(r.antecedent)
    for name, var in getattr(system, "_outputs", {}).items():
        found[name] = var
    return found


def to_dict(system) -> dict:
    """Serialize a Mamdani or TSK ``system`` to a JSON-ready dict."""
    if isinstance(system, Mamdani):
        engine = "mamdani"
    elif isinstance(system, TSK):
        engine = "tsk"
    else:
        raise TypeError(f"cannot serialize {type(system).__name__}")

    variables = {n: v.to_dict() for n, v in _collect_variables(system).items()}
    rules = []
    for r in system.rules:
        entry = {"antecedent": r.antecedent.to_dict(), "weight": r.weight}
        if engine == "mamdani":
            entry["consequent"] = {"var": r.consequent.variable.name,
                                   "term": r.consequent.term}
        else:
            cons = r.consequent
            if not isinstance(cons, (Real, dict)):
                raise TypeError("cannot serialize a callable TSK consequent")
            entry["consequent"] = cons
        rules.append(entry)

    return {"engine": engine, "spec": system._spec, "variables": variables,
            "rules": rules}


def from_dict(d: dict):
    """Rebuild a Mamdani or TSK system from :func:`to_dict` output."""
    variables = {n: Variable.from_dict(vd) for n, vd in d["variables"].items()}
    spec = d.get("spec", {})
    engine = d["engine"]
    system = Mamdani(**spec) if engine == "mamdani" else TSK(**spec)
    for r in d["rules"]:
        antecedent = antecedent_from_dict(r["antecedent"], variables)
        if engine == "mamdani":
            c = r["consequent"]
            consequent = variables[c["var"]][c["term"]]
        else:
            consequent = r["consequent"]
        system.rule(antecedent, consequent, r.get("weight", 1.0))
    return system


def save(system, path: str) -> None:
    """Write ``system`` to ``path`` as JSON."""
    with open(path, "w") as f:
        json.dump(to_dict(system), f, indent=2)


def load(path: str):
    """Read a system back from a JSON file written by :func:`save`."""
    with open(path) as f:
        return from_dict(json.load(f))


__all__ = ["to_dict", "from_dict", "save", "load"]
