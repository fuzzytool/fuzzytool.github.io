"""Ready-made example systems."""

from __future__ import annotations

from . import membership as mf
from .inference import Mamdani
from .sets import Variable
from .type2 import IT2Mamdani, it2_gauss_uncertain_mean


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


def credit_risk_it2() -> tuple[IT2Mamdani, Variable, Variable, Variable]:
    """An interval type-2 version of :func:`credit_risk`.

    The score and premium terms carry a footprint of uncertainty (uncertain
    Gaussian means), modeling vagueness in how a "good" score or a "low" premium
    is defined. Returns ``(system, score, dti, premium)``.

    >>> sys, score, dti, premium = credit_risk_it2()
    >>> sys(score=800, dti=10) < sys(score=520, dti=42)
    True
    """
    score = Variable("score", (300, 850))
    score["poor"] = it2_gauss_uncertain_mean(420, 480, 70)
    score["fair"] = it2_gauss_uncertain_mean(630, 680, 60)
    score["good"] = it2_gauss_uncertain_mean(740, 780, 50)
    score["excellent"] = it2_gauss_uncertain_mean(810, 840, 40)

    dti = Variable("dti", (0, 50))
    dti["low"] = it2_gauss_uncertain_mean(6, 12, 8)
    dti["moderate"] = it2_gauss_uncertain_mean(27, 33, 7)
    dti["high"] = it2_gauss_uncertain_mean(42, 48, 7)

    premium = Variable("premium", (0, 12))
    premium["low"] = it2_gauss_uncertain_mean(1.5, 2.5, 1.5)
    premium["medium"] = it2_gauss_uncertain_mean(5.5, 6.5, 1.5)
    premium["high"] = it2_gauss_uncertain_mean(9.5, 10.5, 1.5)

    sys = IT2Mamdani()
    sys.rule(score["poor"] | dti["high"], premium["high"])
    sys.rule(score["fair"] & dti["moderate"], premium["medium"])
    sys.rule(score["good"] | score["excellent"], premium["low"])
    return sys, score, dti, premium


__all__ = ["credit_risk", "credit_risk_it2"]
