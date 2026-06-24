import numpy as np
import pytest

import fuzzytool as fz


def test_triangular_peaks_and_feet():
    m = fz.tri(0, 5, 10)
    assert m(5) == pytest.approx(1.0)
    assert m(0) == pytest.approx(0.0)
    assert m(10) == pytest.approx(0.0)
    assert m(2.5) == pytest.approx(0.5)


def test_trapezoid_flat_top():
    m = fz.trap(0, 2, 8, 10)
    assert m(5) == pytest.approx(1.0)
    assert m(2) == pytest.approx(1.0)
    assert m(1) == pytest.approx(0.5)


def test_gauss_center_and_symmetry():
    m = fz.gauss(0, 2)
    assert m(0) == pytest.approx(1.0)
    assert m(-3) == pytest.approx(m(3))


def test_membership_is_vectorized_and_bounded():
    m = fz.gbell(2, 4, 0)
    x = np.linspace(-10, 10, 100)
    y = m(x)
    assert y.shape == x.shape
    assert np.all((y >= 0) & (y <= 1))


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        fz.tri(5, 0, 10)
    with pytest.raises(ValueError):
        fz.gauss(0, 0)
