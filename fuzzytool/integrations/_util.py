"""Shared helpers for the integration submodules."""

from __future__ import annotations

from .. import membership as mf

# Returned in place of a non-finite residual so tuners avoid degenerate regions.
MF_PENALTY = 1e6


def input_variable_names(system) -> list[str]:
    """Ordered, de-duplicated names of the input variables a system reads.

    Walks every rule's antecedent tree of a :class:`~fuzzytool.inference.Mamdani`
    or :class:`~fuzzytool.inference.TSK` system and collects the variables its
    propositions reference, in first-seen order. This is the natural column
    order for feeding a 2-D array or a DataFrame to the system.
    """
    names: list[str] = []
    seen: set[str] = set()

    def walk(node) -> None:
        if hasattr(node, "variable"):            # Proposition (atom)
            name = node.variable.name
            if name not in seen:
                seen.add(name)
                names.append(name)
        if hasattr(node, "left"):                # And / Or
            walk(node.left)
            walk(node.right)
        if hasattr(node, "operand"):             # Not
            walk(node.operand)

    for rule in getattr(system, "rules", []):
        walk(rule.antecedent)
    return names


def output_name(system, default: str = "output") -> str:
    """Best-effort name for a single-output system (Mamdani tracks it; TSK does not)."""
    outputs = getattr(system, "_outputs", None)
    if outputs:
        return next(iter(outputs))
    return default


def sanitize_mf_params(kind: str, params) -> list[float]:
    """Repair a membership-function parameter slice so it is always a valid shape.

    Keeps breakpoints ordered and widths positive, so a tuner (SciPy least-squares
    or turboswarm PSO) can explore freely without producing degenerate sets.
    """
    p = [float(v) for v in params]
    if kind in ("tri", "trap", "ramp_up", "ramp_down"):
        return sorted(p)                       # keep breakpoints ordered
    if kind == "gauss":
        return [p[0], abs(p[1]) or 1e-6]       # sigma > 0
    if kind == "gbell":
        return [abs(p[0]) or 1e-6, abs(p[1]) or 1e-6, p[2]]  # width, slope > 0
    return p


def tunable_terms(system, tune_outputs: bool = True):
    """Collect ``(variable, term_name, kind, params)`` for every tunable MF.

    Walks the rules' propositions (and, if ``tune_outputs``, the Mamdani output
    variables), keeping only terms backed by a built-in membership shape; custom
    callables are skipped.
    """
    seen: dict[int, object] = {}

    def walk(node):
        if hasattr(node, "variable"):
            seen.setdefault(id(node.variable), node.variable)
        if hasattr(node, "left"):
            walk(node.left)
            walk(node.right)
        if hasattr(node, "operand"):
            walk(node.operand)

    for rule in getattr(system, "rules", []):
        walk(rule.antecedent)
    variables = dict(seen)
    if tune_outputs:
        for out in getattr(system, "_outputs", {}).values():
            variables.setdefault(id(out), out)

    terms = []
    for var in variables.values():
        for term_name, membership in var.terms.items():
            try:
                spec = mf.to_dict(membership)
            except TypeError:
                continue                       # custom callable: skip
            terms.append((var, term_name, spec["type"], list(spec["params"])))
    return terms
