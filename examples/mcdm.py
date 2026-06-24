"""Fuzzy MCDM: rank loan applicants with Fuzzy TOPSIS, weight criteria with AHP.

    python examples/mcdm.py
"""

from fuzzytool.fuzzynum import tfn
from fuzzytool.mcdm import fuzzy_ahp, fuzzy_topsis


def main() -> None:
    # Criteria: repayment ability (benefit), collateral (benefit), risk (cost).
    # Weights from a fuzzy pairwise comparison (AHP).
    one = tfn(1, 1, 1)
    pairwise = [
        [one,                tfn(2, 3, 4),       tfn(1, 2, 3)],
        [tfn(1/4, 1/3, 1/2), one,                tfn(1/3, 1/2, 1)],
        [tfn(1/3, 1/2, 1),   tfn(1, 2, 3),       one],
    ]
    crisp_w = fuzzy_ahp(pairwise)
    print("AHP criterion weights:", crisp_w.round(3))

    # Fuzzy weights for TOPSIS (triangular around the AHP weights).
    weights = [tfn(max(0, w - 0.1), w, min(1, w + 0.1)) for w in crisp_w]

    applicants = ["Ana", "Beto", "Carla"]
    matrix = [
        [tfn(7, 8, 9), tfn(6, 7, 8), tfn(2, 3, 4)],   # Ana
        [tfn(4, 5, 6), tfn(8, 9, 9), tfn(5, 6, 7)],   # Beto
        [tfn(6, 7, 8), tfn(3, 4, 5), tfn(1, 2, 3)],   # Carla
    ]
    res = fuzzy_topsis(matrix, weights, benefit=[True, True, False])

    print("\nFuzzy TOPSIS ranking (best first):")
    for place, i in enumerate(res.ranking, 1):
        print(f"  {place}. {applicants[i]:6s}  CC = {res.closeness[i]:.3f}")


if __name__ == "__main__":
    main()
