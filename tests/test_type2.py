import numpy as np
import pytest

import fuzzytool as fz
from fuzzytool import datasets
from fuzzytool.type2 import centroid_it2, karnik_mendel


def test_fou_bounds_ordered():
    # The upper MF must dominate the lower MF across the universe.
    s = fz.it2_gauss_uncertain_mean(630, 680, 60)
    x = np.linspace(300, 850, 200)
    assert np.all(s.lower(x) <= s.upper(x) + 1e-12)
    assert np.all((s.lower(x) >= 0) & (s.upper(x) <= 1))


def test_uncertain_mean_flat_top():
    s = fz.it2_gauss_uncertain_mean(4, 6, 1.0)
    # The UMF is 1 across the whole mean interval [4, 6].
    assert s.upper(np.array([4.0, 5.0, 6.0])) == pytest.approx(1.0)
    # The LMF dips in the middle (it is the lower envelope).
    assert float(s.lower(5.0)) < 1.0


def test_it2_scale_is_height_uncertainty():
    s = fz.it2_scale(fz.gauss(0, 1), 0.6)
    assert float(s.upper(0.0)) == pytest.approx(1.0)
    assert float(s.lower(0.0)) == pytest.approx(0.6)


def test_km_collapses_degenerate_interval_to_weighted_mean():
    # When lower == upper the IT2 set is really type-1: KM returns the centroid.
    pts = np.array([0.0, 1.0, 2.0])
    w = np.array([1.0, 1.0, 1.0])
    y_l, y_r = karnik_mendel(pts, w, w)
    assert y_l == pytest.approx(1.0)
    assert y_r == pytest.approx(1.0)


def test_km_endpoints_bracket_the_uncertainty():
    pts = np.array([0.0, 1.0, 2.0])
    lower = np.array([0.2, 0.2, 0.2])
    upper = np.array([1.0, 1.0, 1.0])
    y_l, y_r = karnik_mendel(pts, lower, upper)
    assert 0.0 <= y_l <= y_r <= 2.0


def test_centroid_of_symmetric_set_is_centered():
    s = fz.it2_gauss_uncertain_mean(4.5, 5.5, 1.0)
    c_l, c_r = centroid_it2(s, np.linspace(0, 10, 501))
    assert (c_l + c_r) / 2 == pytest.approx(5.0, abs=0.05)


def test_it2_mamdani_credit_risk_monotone():
    sys, *_ = datasets.credit_risk_it2()
    assert sys(score=520, dti=30) > sys(score=800, dti=30)
    assert sys(score=660, dti=10) < sys(score=660, dti=45)
    assert 0 <= sys(score=800, dti=10) <= 12


def test_it2_tsk_runs():
    x = fz.Variable("x", (0, 10))
    x["small"] = fz.it2_gauss_uncertain_mean(1, 2, 2)
    x["large"] = fz.it2_gauss_uncertain_mean(8, 9, 2)
    sys = fz.IT2TSK()
    sys.rule(x["small"], 0.0)
    sys.rule(x["large"], {"const": 1.0, "x": 1.0})
    assert sys(x=0.0) < sys(x=10.0)


def test_type1_and_it2_terms_mix_in_one_antecedent():
    a = fz.Variable("a", (0, 1), terms=["lo", "hi"])   # type-1
    b = fz.Variable("b", (0, 1))
    b["hi"] = fz.it2_scale(fz.gauss(1, 0.3), 0.5)       # IT2
    lo, hi = (a["hi"] & b["hi"]).eval_interval({"a": 0.8, "b": 1.0},
                                               fz.norms.t_min, fz.norms.s_max)
    assert lo <= hi
