"""Fuzzy multi-criteria decision making (MCDM).

* :func:`fuzzy_topsis` — Chen's (2000) fuzzy TOPSIS: rank alternatives by their
  closeness to the fuzzy positive-ideal solution.
* :func:`fuzzy_ahp` — Chang's (1996) extent-analysis AHP: derive crisp criterion
  weights from a triangular-fuzzy pairwise-comparison matrix.

Inputs are triangular fuzzy numbers (:class:`~fuzzytool.fuzzynum.TriangularFuzzyNumber`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fuzzynum import TriangularFuzzyNumber as TFN


@dataclass
class TopsisResult:
    """Result of :func:`fuzzy_topsis`.

    Attributes:
        closeness: closeness coefficient ``CC`` per alternative (higher = better).
        ranking: alternative indices ordered best-to-worst.
        d_plus: distance to the fuzzy positive-ideal solution per alternative.
        d_minus: distance to the fuzzy negative-ideal solution per alternative.
    """

    closeness: np.ndarray
    ranking: list[int]
    d_plus: np.ndarray
    d_minus: np.ndarray


def fuzzy_topsis(matrix: list, weights: list, benefit: list) -> TopsisResult:
    """Rank alternatives with Chen's fuzzy TOPSIS.

    Args:
        matrix: ``m x n`` ratings as ``TriangularFuzzyNumber`` (m alternatives,
            n criteria).
        weights: ``n`` criterion weights as ``TriangularFuzzyNumber``.
        benefit: length-``n`` booleans — ``True`` if a criterion is to be
            maximized (benefit), ``False`` if minimized (cost).

    Returns:
        A :class:`TopsisResult`.
    """
    m = len(matrix)
    n = len(matrix[0])
    if len(weights) != n or len(benefit) != n:
        raise ValueError("weights and benefit must have one entry per criterion")

    # Linear normalization to [0, 1] per criterion (Chen).
    norm = [[None] * n for _ in range(m)]
    for j in range(n):
        col = [matrix[i][j] for i in range(m)]
        if benefit[j]:
            cstar = max(x.c for x in col)
            cstar = cstar if cstar != 0 else 1.0
            for i in range(m):
                a, b, c = matrix[i][j].points
                norm[i][j] = TFN(a / cstar, b / cstar, c / cstar)
        else:
            aminus = min(x.a for x in col)
            for i in range(m):
                a, b, c = matrix[i][j].points
                a = a if a != 0 else 1e-12
                norm[i][j] = TFN(aminus / c, aminus / b, aminus / a)

    # Weighted normalized matrix, then distances to FPIS (1,1,1) and FNIS (0,0,0).
    fpis, fnis = TFN(1, 1, 1), TFN(0, 0, 0)
    d_plus = np.zeros(m)
    d_minus = np.zeros(m)
    for i in range(m):
        for j in range(n):
            v = norm[i][j] * weights[j]
            d_plus[i] += v.distance(fpis)
            d_minus[i] += v.distance(fnis)

    cc = d_minus / (d_plus + d_minus)
    ranking = sorted(range(m), key=lambda i: cc[i], reverse=True)
    return TopsisResult(cc, ranking, d_plus, d_minus)


def fuzzy_ahp(matrix: list) -> np.ndarray:
    """Crisp criterion weights from a fuzzy pairwise matrix (Chang's method).

    Args:
        matrix: ``n x n`` pairwise comparisons as ``TriangularFuzzyNumber``; the
            diagonal is ``(1, 1, 1)`` and ``matrix[j][i]`` is the reciprocal of
            ``matrix[i][j]``.

    Returns:
        A length-``n`` array of normalized, non-negative weights summing to 1.
    """
    n = len(matrix)
    # Row sums (fuzzy synthetic extent numerator) and the total sum.
    row = [TFN(sum(matrix[i][j].a for j in range(n)),
               sum(matrix[i][j].b for j in range(n)),
               sum(matrix[i][j].c for j in range(n))) for i in range(n)]
    total_l = sum(r.a for r in row)
    total_m = sum(r.b for r in row)
    total_u = sum(r.c for r in row)
    # S_i = row_i * (total)^-1  (reciprocal inverts and swaps the bounds).
    S = [TFN(r.a / total_u, r.b / total_m, r.c / total_l) for r in row]

    def degree(a: TFN, b: TFN) -> float:
        # V(a >= b): possibility that fuzzy number a is at least b.
        if a.b >= b.b:
            return 1.0
        if b.a >= a.c:
            return 0.0
        return (b.a - a.c) / ((a.b - a.c) - (b.b - b.a))

    weights = np.array([min(degree(S[i], S[k]) for k in range(n) if k != i)
                        for i in range(n)])
    if weights.sum() == 0:
        weights = np.ones(n)
    return weights / weights.sum()


__all__ = ["fuzzy_topsis", "fuzzy_ahp", "TopsisResult"]
