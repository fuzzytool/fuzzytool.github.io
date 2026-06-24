import numpy as np
import pytest

from fuzzytool.fuzzynum import rank, tfn, trfn
from fuzzytool.mcdm import fuzzy_ahp, fuzzy_topsis

# --- fuzzy numbers ---------------------------------------------------------

def test_tfn_arithmetic():
    a, b = tfn(1, 2, 3), tfn(2, 3, 4)
    assert (a + b).points == (3, 5, 7)
    assert (b - a).points == (-1, 1, 3)        # (2-3, 3-2, 4-1)
    assert (a * 2).points == (2, 4, 6)


def test_tfn_membership_and_centroid():
    t = tfn(0, 5, 10)
    assert t(5) == pytest.approx(1.0)
    assert t(2.5) == pytest.approx(0.5)
    assert t.centroid() == pytest.approx(5.0)


def test_tfn_alpha_cut():
    lo, hi = tfn(0, 5, 10).alpha_cut(0.5)
    assert (lo, hi) == pytest.approx((2.5, 7.5))


def test_trapezoid_centroid_symmetric():
    assert trfn(0, 2, 8, 10).centroid() == pytest.approx(5.0)


def test_rank_orders_by_centroid():
    nums = [tfn(0, 1, 2), tfn(5, 6, 7), tfn(2, 3, 4)]
    assert rank(nums) == [1, 2, 0]


def test_distance_requires_same_shape():
    with pytest.raises(TypeError):
        tfn(1, 2, 3).distance(trfn(1, 2, 3, 4))


# --- fuzzy TOPSIS ----------------------------------------------------------

def test_fuzzy_topsis_prefers_dominant_alternative():
    # Alternative 0 dominates alternative 1 on both benefit criteria.
    matrix = [
        [tfn(7, 8, 9), tfn(7, 8, 9)],
        [tfn(1, 2, 3), tfn(1, 2, 3)],
    ]
    weights = [tfn(0.4, 0.5, 0.6), tfn(0.4, 0.5, 0.6)]
    res = fuzzy_topsis(matrix, weights, benefit=[True, True])
    assert res.ranking[0] == 0
    assert res.closeness[0] > res.closeness[1]
    assert np.all((res.closeness >= 0) & (res.closeness <= 1))


def test_fuzzy_topsis_respects_cost_criterion():
    # On a cost criterion, the smaller rating should win.
    matrix = [[tfn(1, 2, 3)], [tfn(7, 8, 9)]]
    weights = [tfn(0.8, 0.9, 1.0)]
    res = fuzzy_topsis(matrix, weights, benefit=[False])
    assert res.ranking[0] == 0


# --- fuzzy AHP -------------------------------------------------------------

def test_fuzzy_ahp_weights_sum_to_one_and_order():
    one = tfn(1, 1, 1)
    # Criterion 0 strongly preferred over 1 and 2; 1 preferred over 2.
    matrix = [
        [one, tfn(2, 3, 4), tfn(3, 4, 5)],
        [tfn(1 / 4, 1 / 3, 1 / 2), one, tfn(2, 3, 4)],
        [tfn(1 / 5, 1 / 4, 1 / 3), tfn(1 / 4, 1 / 3, 1 / 2), one],
    ]
    w = fuzzy_ahp(matrix)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
