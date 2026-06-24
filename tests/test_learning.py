import numpy as np
import pytest

import fuzzytool as fz


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --- ANFIS -----------------------------------------------------------------

def test_anfis_fits_1d_nonlinear_function():
    x = np.linspace(0, 2 * np.pi, 120)
    y = np.sin(x)
    model = fz.ANFIS(n_inputs=1, n_mf=6).fit(x[:, None], y, epochs=80)
    pred = model.predict(x[:, None])
    assert _rmse(pred, y) < 0.1


def test_anfis_training_reduces_error():
    x = np.linspace(-3, 3, 100)
    y = np.tanh(x)
    model = fz.ANFIS(n_inputs=1, n_mf=5).fit(x[:, None], y, epochs=60)
    # The hybrid scheme should not diverge: end no worse than the first epoch.
    assert model.history_[-1] <= model.history_[0] + 1e-9
    assert len(model.history_) == 60


def test_anfis_fits_2d_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 2))
    y = X[:, 0] * X[:, 1]
    model = fz.ANFIS(n_inputs=2, n_mf=3).fit(X, y, epochs=40)
    assert _rmse(model.predict(X), y) < 0.1
    assert model.R == 9


def test_anfis_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        fz.ANFIS(n_inputs=1).predict(np.zeros((3, 1)))


def test_anfis_input_validation():
    with pytest.raises(ValueError):
        fz.ANFIS(n_inputs=1, n_mf=1)
    model = fz.ANFIS(n_inputs=2)
    with pytest.raises(ValueError):
        model.fit(np.zeros((5, 3)), np.zeros(5))   # wrong number of columns


# --- F-transform -----------------------------------------------------------

def test_ftransform_partition_of_unity():
    ft = fz.FTransform(0, 10, n_basis=6)
    x = np.linspace(0, 10, 200)
    assert np.allclose(ft.basis(x).sum(axis=1), 1.0)


def test_ftransform_reconstructs_smooth_signal():
    x = np.linspace(0, 2 * np.pi, 300)
    y = np.sin(x)
    ft = fz.FTransform(0, 2 * np.pi, n_basis=25).fit(x, y)
    assert ft.components_.shape == (25,)
    assert _rmse(ft.smooth(x), y) < 0.05


def test_ftransform_denoises():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 2 * np.pi, 400)
    clean = np.sin(x)
    noisy = clean + rng.normal(0, 0.3, size=x.shape)
    ft = fz.FTransform(0, 2 * np.pi, n_basis=12).fit(x, noisy)
    recon = ft.smooth(x)
    # The reconstruction is closer to the clean signal than the noisy input.
    assert _rmse(recon, clean) < _rmse(noisy, clean)


def test_ftransform_more_basis_is_more_accurate():
    x = np.linspace(0, 5, 200)
    y = np.exp(-0.3 * x) * np.cos(2 * x)
    coarse = fz.FTransform(0, 5, 6).fit(x, y).smooth(x)
    fine = fz.FTransform(0, 5, 40).fit(x, y).smooth(x)
    assert _rmse(fine, y) < _rmse(coarse, y)
