import numpy as np
import pytest

import fuzzytool as fz

# --- Tsukamoto -------------------------------------------------------------

def test_tsukamoto_is_monotone():
    x = fz.Variable("x", (0, 10), terms=["lo", "hi"])
    sys = fz.Tsukamoto()
    sys.rule(x["lo"], fz.ramp_down(0, 30))   # low x  -> output near 0
    sys.rule(x["hi"], fz.ramp_up(0, 30))     # high x -> output near 30
    assert sys(x=0) < sys(x=5) < sys(x=10)
    assert sys(x=0) == pytest.approx(0, abs=1.0)
    assert sys(x=10) == pytest.approx(30, abs=1.0)


def test_tsukamoto_rejects_non_monotonic_consequent():
    x = fz.Variable("x", (0, 10), terms=["lo", "hi"])
    sys = fz.Tsukamoto()
    with pytest.raises(TypeError):
        sys.rule(x["lo"], fz.tri(0, 15, 30))   # triangular has no inverse


def test_ramp_inverse_roundtrip():
    up = fz.ramp_up(0, 30)
    assert up(up.inverse(0.4)) == pytest.approx(0.4)
    down = fz.ramp_down(0, 30)
    assert down(down.inverse(0.7)) == pytest.approx(0.7)


# --- Wang-Mendel -----------------------------------------------------------

def test_wang_mendel_learns_monotone_mapping():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=200)
    y = x                                     # identity target
    xv = fz.Variable("x", (0, 10), terms=["vlo", "lo", "mid", "hi", "vhi"])
    yv = fz.Variable("y", (0, 10), terms=["vlo", "lo", "mid", "hi", "vhi"])
    sys = fz.wang_mendel(x[:, None], y, [xv], yv)
    assert len(sys.rules) >= 1
    assert sys(x=1.0) < sys(x=5.0) < sys(x=9.0)


def test_wang_mendel_two_inputs_resolves_conflicts():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, size=(300, 2))
    y = (X[:, 0] + X[:, 1]) / 2
    a = fz.Variable("a", (0, 10), terms=["lo", "mid", "hi"])
    b = fz.Variable("b", (0, 10), terms=["lo", "mid", "hi"])
    out = fz.Variable("out", (0, 10), terms=["lo", "mid", "hi"])
    sys = fz.wang_mendel(X, y, [a, b], out)
    # At most one rule per distinct antecedent combination (3 x 3 = 9).
    assert len(sys.rules) <= 9
    assert sys(a=1, b=1) < sys(a=9, b=9)


def test_wang_mendel_validates_shape():
    a = fz.Variable("a", (0, 10), terms=["lo", "hi"])
    out = fz.Variable("out", (0, 10), terms=["lo", "hi"])
    with pytest.raises(ValueError):
        fz.wang_mendel(np.zeros((5, 2)), np.zeros(5), [a], out)
