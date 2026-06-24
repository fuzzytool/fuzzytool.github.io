"""Ready-made example systems."""

from __future__ import annotations

from . import membership as mf
from .inference import Mamdani
from .sets import Variable


def credit_risk() -> tuple[Mamdani, Variable, Variable, Variable]:
    """A credit-risk-premium Mamdani system.

    Given a borrower's credit ``score`` (300-850) and ``dti`` (debt-to-income
    ratio, 0-50%), recommend the ``premium`` (risk points, 0-12) a lender should
    add on top of its base interest rate. Returns ``(system, score, dti,
    premium)`` so callers can inspect the variables (e.g. for plotting).

    >>> sys, score, dti, premium = credit_risk()
    >>> safe = sys(score=800, dti=10)      # great score, low leverage
    >>> risky = sys(score=520, dti=42)     # poor score, high leverage
    >>> safe < risky
    True
    """
    score = Variable("score", (300, 850))
    score["poor"] = mf.trap(300, 300, 500, 600)
    score["fair"] = mf.tri(560, 660, 730)
    score["good"] = mf.tri(690, 760, 810)
    score["excellent"] = mf.trap(780, 830, 850, 850)

    dti = Variable("dti", (0, 50))
    dti["low"] = mf.trap(0, 0, 15, 25)
    dti["moderate"] = mf.tri(20, 30, 40)
    dti["high"] = mf.trap(35, 45, 50, 50)

    premium = Variable("premium", (0, 12))
    premium["low"] = mf.tri(0, 1.5, 4)
    premium["medium"] = mf.tri(3, 6, 9)
    premium["high"] = mf.tri(8, 10.5, 12)

    sys = Mamdani()
    sys.rule(score["poor"] | dti["high"], premium["high"])
    sys.rule(score["fair"] & dti["moderate"], premium["medium"])
    sys.rule(score["good"] | score["excellent"], premium["low"])
    return sys, score, dti, premium


__all__ = ["credit_risk"]
