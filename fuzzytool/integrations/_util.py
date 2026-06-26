"""Shared helpers for the integration submodules."""

from __future__ import annotations


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
