import numpy as np
import pytest

import fuzzytool as fz
from fuzzytool import datasets, serialize

# --- batch / vectorized inference ------------------------------------------

def test_mamdani_predict_matches_scalar_calls():
    sys, *_ = datasets.credit_risk()
    scores = np.array([520.0, 660.0, 800.0])
    dtis = np.array([42.0, 30.0, 10.0])
    batch = sys.predict(score=scores, dti=dtis)
    one_by_one = np.array([sys(score=s, dti=d) for s, d in zip(scores, dtis)])
    assert np.allclose(batch, one_by_one)


def test_tsk_predict_matches_scalar_calls():
    x = fz.Variable("x", (0, 10), terms=["small", "large"])
    sys = fz.TSK()
    sys.rule(x["small"], 0.0)
    sys.rule(x["large"], {"const": 1.0, "x": 1.0})
    xs = np.linspace(0, 10, 7)
    batch = sys.predict(x=xs)
    one_by_one = np.array([sys(x=float(v)) for v in xs])
    assert np.allclose(batch, one_by_one)


# --- serialization ----------------------------------------------------------

def test_mamdani_roundtrip(tmp_path):
    sys, *_ = datasets.credit_risk()
    path = tmp_path / "sys.json"
    fz.save(sys, str(path))
    restored = fz.load(str(path))
    for s, d in [(800, 10), (660, 30), (520, 42)]:
        assert restored(score=s, dti=d) == pytest.approx(sys(score=s, dti=d))


def test_tsk_roundtrip_via_dict():
    x = fz.Variable("x", (0, 10), terms=["small", "large"])
    sys = fz.TSK()
    sys.rule(x["small"] | x["large"], {"const": 1.0, "x": 0.5})
    sys.rule(~x["small"], 3.0)
    restored = serialize.from_dict(serialize.to_dict(sys))
    assert restored(x=4.0) == pytest.approx(sys(x=4.0))


def test_serialize_rejects_callable_consequent():
    x = fz.Variable("x", (0, 10), terms=["lo", "hi"])
    sys = fz.TSK()
    sys.rule(x["hi"], lambda x: x ** 2)
    with pytest.raises(TypeError):
        serialize.to_dict(sys)


def test_membership_serialization_roundtrip():
    from fuzzytool import membership as mf
    for m in [fz.tri(0, 1, 2), fz.trap(0, 1, 2, 3), fz.gauss(0, 1),
              fz.gbell(2, 3, 0), fz.sigmoid(1, 0), fz.ramp_up(0, 5)]:
        again = mf.from_dict(mf.to_dict(m))
        assert np.allclose(again(np.linspace(-2, 5, 20)), m(np.linspace(-2, 5, 20)))


# --- sklearn-style estimator interface --------------------------------------

def test_anfis_get_set_params():
    model = fz.ANFIS(n_inputs=2, n_mf=3)
    assert model.get_params() == {"n_inputs": 2, "n_mf": 3, "learning_rate": 0.05}
    model.set_params(n_mf=4)
    assert model.n_mf == 4 and model.R == 4 ** 2


def test_anfis_clone_pattern():
    # The sklearn clone idiom: rebuild from get_params.
    a = fz.ANFIS(n_inputs=1, n_mf=5, learning_rate=0.1)
    b = fz.ANFIS(**a.get_params())
    assert b.get_params() == a.get_params()
