"""LLM-agent integration: expose a fuzzy system as an explainable tool.

Install with ``pip install fuzzytool[agents]``. A fuzzy inference system is an
unusually good citizen for tool-using LLMs: it is deterministic, bounded, and —
unlike a black-box model — it can say *why* it produced an answer by reporting
which rules fired and how strongly.

* :func:`explain` — run a system and return its crisp output plus the fired
  rules (no third-party dependency; useful on its own).
* :func:`inference_tool` — wrap a system as a LangChain ``StructuredTool`` an
  agent can call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._util import input_variable_names, output_name

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

    from ..inference import TSK, Mamdani


def explain(system, **inputs) -> dict:
    """Run ``system`` on ``inputs`` and explain the result.

    Returns a dict with the crisp ``output`` (a float, or a dict for a
    multi-output Mamdani) and ``fired_rules``: the rules whose firing strength is
    positive, each ``{"rule", "firing"}``, sorted strongest-first. This is the
    payload an agent (or a human) needs to trust the answer.
    """
    output = system(**inputs)
    fired = []
    for rule in system.rules:
        strength = float(rule.antecedent.eval(inputs, system.tnorm, system.snorm))
        strength *= rule.weight
        if strength > 0.0:
            fired.append({"rule": str(rule), "firing": round(strength, 4)})
    fired.sort(key=lambda r: r["firing"], reverse=True)
    return {"output": output, "fired_rules": fired}


def _require_langchain():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the agents integration needs langchain-core; install with "
            "`pip install fuzzytool[agents]`"
        ) from exc
    return StructuredTool


def inference_tool(system: Mamdani | TSK, columns: list[str] | None = None,
                   name: str = "fuzzy_inference",
                   description: str | None = None) -> StructuredTool:
    """Wrap a fuzzy system as a LangChain ``StructuredTool``.

    The tool takes one float argument per input variable and returns the crisp
    output together with the rules that fired — so the agent can both *use* and
    *explain* the system.

    Args:
        system: a Mamdani or TSK system.
        columns: input variable names (the tool's arguments). Defaults to the
            variables the rules reference.
        name: tool name exposed to the agent.
        description: tool description; a sensible default is generated.

    Returns:
        A ``langchain_core.tools.StructuredTool``.
    """
    StructuredTool = _require_langchain()
    from pydantic import create_model

    names = list(columns) if columns is not None else input_variable_names(system)
    out_name = output_name(system)
    args_model = create_model(
        f"{name}_args", **{n: (float, ...) for n in names})

    def _run(**kwargs) -> dict:
        result = explain(system, **kwargs)
        return {out_name: result["output"], "fired_rules": result["fired_rules"]}

    if description is None:
        description = (
            f"Evaluate a fuzzy inference system. Inputs: {', '.join(names)}. "
            f"Returns the crisp '{out_name}' and which rules fired (with strengths)."
        )
    return StructuredTool.from_function(
        func=_run, name=name, description=description, args_schema=args_model)


__all__ = ["explain", "inference_tool"]
