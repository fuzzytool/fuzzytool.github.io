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


def test_sigmoid_is_stable_at_extreme_arguments():
    s = fz.sigmoid(2.0, 5.0)
    with np.errstate(over="raise"):  # overflow in exp would raise here
        y = s(np.array([-1e6, 0.0, 1e6]))
    assert np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1))
    assert y[0] == 0.0 and y[-1] == 1.0
    # Still matches the plain logistic in the numerically safe range.
    x = np.linspace(-20, 30, 500)
    assert np.allclose(s(x), 1.0 / (1.0 + np.exp(-2.0 * (x - 5.0))), atol=1e-12)
    # Scalar input yields a scalar-like (0-d) result.
    assert np.ndim(s(5.0)) == 0
