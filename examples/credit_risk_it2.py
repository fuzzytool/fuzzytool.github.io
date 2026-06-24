"""Interval type-2 credit-risk premium (IT2 Mamdani inference).

Same problem as `credit_risk.py`, but the linguistic terms carry a footprint of
uncertainty (uncertain Gaussian means): there is no single sharp definition of a
"good" score or a "low" premium. Inference type-reduces with Karnik-Mendel.

    python examples/credit_risk_it2.py
"""

import fuzzytool as fz


def main() -> None:
    score = fz.Variable("score", (300, 850))
    score["poor"] = fz.it2_gauss_uncertain_mean(420, 480, 70)
    score["fair"] = fz.it2_gauss_uncertain_mean(630, 680, 60)
    score["good"] = fz.it2_gauss_uncertain_mean(740, 780, 50)
    score["excellent"] = fz.it2_gauss_uncertain_mean(810, 840, 40)

    dti = fz.Variable("dti", (0, 50))
    dti["low"] = fz.it2_gauss_uncertain_mean(6, 12, 8)
    dti["moderate"] = fz.it2_gauss_uncertain_mean(27, 33, 7)
    dti["high"] = fz.it2_gauss_uncertain_mean(42, 48, 7)

    premium = fz.Variable("premium", (0, 12))
    premium["low"] = fz.it2_gauss_uncertain_mean(1.5, 2.5, 1.5)
    premium["medium"] = fz.it2_gauss_uncertain_mean(5.5, 6.5, 1.5)
    premium["high"] = fz.it2_gauss_uncertain_mean(9.5, 10.5, 1.5)

    sys = fz.IT2Mamdani()
    sys.rule(score["poor"] | dti["high"], premium["high"])
    sys.rule(score["fair"] & dti["moderate"], premium["medium"])
    sys.rule(score["good"] | score["excellent"], premium["low"])

    for s, d in [(800, 10), (660, 30), (520, 42)]:
        print(f"score={s}, dti={d}%  ->  IT2 risk premium = {sys(score=s, dti=d):.2f} pts")


if __name__ == "__main__":
    main()
