import numpy as np
import pytest

import fuzzytool as fz
from fuzzytool import datasets


def test_credit_risk_premium_responds_to_score_and_dti():
    sys, *_ = datasets.credit_risk()
    # Premium should fall as the score improves (dti held fixed)...
    assert sys(score=520, dti=30) > sys(score=800, dti=30)
    # ...and rise as leverage (dti) increases (score held fixed).
    assert sys(score=660, dti=10) < sys(score=660, dti=45)
    # Output stays within the premium universe.
    assert 0 <= sys(score=800, dti=10) <= 12


def test_operators_build_antecedents():
    a = fz.Variable("a", (0, 1), terms=["lo", "hi"])
    b = fz.Variable("b", (0, 1), terms=["lo", "hi"])
    inputs = {"a": 0.2, "b": 0.9}
    tnorm, snorm = fz.norms.t_min, fz.norms.s_max
    p_and = (a["hi"] & b["hi"]).eval(inputs, tnorm, snorm)
    p_or = (a["hi"] | b["hi"]).eval(inputs, tnorm, snorm)
    p_not = (~a["hi"]).eval(inputs, tnorm, snorm)
    assert p_and <= p_or
    assert p_not == pytest.approx(1.0 - a.terms["hi"](0.2))


def test_unknown_term_raises():
    v = fz.Variable("v", (0, 1), terms=["lo", "hi"])
    with pytest.raises(KeyError):
        v["nope"]


def test_tsk_zero_and_first_order():
    x = fz.Variable("x", (0, 10), terms=["small", "large"])
    sys = fz.TSK()
    sys.rule(x["small"], 0.0)
    sys.rule(x["large"], {"const": 1.0, "x": 1.0})
    # At x=0 only "small" fires meaningfully -> ~0; at x=10 -> ~ 1 + 10 = 11.
    assert sys(x=0.0) == pytest.approx(0.0, abs=1e-6)
    assert sys(x=10.0) == pytest.approx(11.0, abs=1e-6)


def test_centroid_handles_no_rule_fired():
    # An output universe sampled symmetrically falls back to its midpoint.
    x = np.linspace(0, 12, 501)
    assert fz.defuzz.centroid(x, np.zeros_like(x)) == pytest.approx(6.0, abs=0.1)


def test_mamdani_caches_consequent_shapes_and_invalidates_on_replace():
    sys, _, _, premium = datasets.credit_risk()
    sys._shape_cache.clear()

    # Calling twice reuses cached shapes (same array objects, never recomputed).
    sys(score=650, dti=25)
    cached = {k: id(v) for k, v in sys._shape_cache.items()}
    assert cached  # at least one consequent term was shaped
    sys(score=650, dti=25)
    for key, arr_id in cached.items():
        assert id(sys._shape_cache[key]) == arr_id

    # predict shares the cache and agrees with scalar calls.
    pts = [(800, 10), (520, 42), (650, 25)]
    batch = sys.predict(score=np.array([s for s, _ in pts]),
                        dti=np.array([d for _, d in pts]))
    scalar = np.array([sys(score=s, dti=d) for s, d in pts])
    assert np.allclose(batch, scalar)

    # Replacing the consequent term that fires (new MF id) invalidates the
    # cache: for these inputs only the "medium" rule fires.
    before = sys(score=650, dti=25)
    premium["medium"] = fz.tri(7, 9, 11)
    assert sys(score=650, dti=25) != before
