"""Fuzzy rules shared by the inference engines.

A :class:`Rule` pairs an antecedent expression with a consequent and an optional
weight. The *consequent* is interpreted by the engine: Mamdani expects a
:class:`~fuzzytool.sets.Proposition` (``output is term``); TSK expects a
constant or a linear coefficient vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sets import Antecedent


@dataclass
class Rule:
    """``IF antecedent THEN consequent`` with an optional firing ``weight``."""

    antecedent: Antecedent
    consequent: Any
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"rule weight must be in [0, 1], got {self.weight}")

    def __repr__(self) -> str:
        w = "" if self.weight == 1.0 else f" (w={self.weight})"
        return f"IF {self.antecedent} THEN {self.consequent}{w}"


__all__ = ["Rule"]
