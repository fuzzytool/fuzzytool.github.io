"""Credit-risk premium (Mamdani inference).

A lender turns a borrower's credit `score` (300-850) and `dti`
(debt-to-income, %) into a risk `premium` (points added to the base rate).

    python examples/credit_risk.py
"""

import fuzzytool as fz


def main() -> None:
    score = fz.Variable("score", (300, 850))
    score["poor"] = fz.trap(300, 300, 500, 600)
    score["fair"] = fz.tri(560, 660, 730)
    score["good"] = fz.tri(690, 760, 810)
    score["excellent"] = fz.trap(780, 830, 850, 850)

    dti = fz.Variable("dti", (0, 50))
    dti["low"] = fz.trap(0, 0, 15, 25)
    dti["moderate"] = fz.tri(20, 30, 40)
    dti["high"] = fz.trap(35, 45, 50, 50)

    premium = fz.Variable("premium", (0, 12))
    premium["low"] = fz.tri(0, 1.5, 4)
    premium["medium"] = fz.tri(3, 6, 9)
    premium["high"] = fz.tri(8, 10.5, 12)

    sys = fz.Mamdani(defuzz="centroid")
    sys.rule(score["poor"] | dti["high"], premium["high"])    # |=OR  &=AND  ~=NOT
    sys.rule(score["fair"] & dti["moderate"], premium["medium"])
    sys.rule(score["good"] | score["excellent"], premium["low"])

    for s, d in [(800, 10), (660, 30), (520, 42)]:
        print(f"score={s}, dti={d}%  ->  risk premium = {sys(score=s, dti=d):.2f} pts")


if __name__ == "__main__":
    main()
