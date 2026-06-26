import numpy as np
import pytest

import fuzzytool as fz
from fuzzytool import datasets

# --- pandas integration -----------------------------------------------------

def test_predict_df_matches_predict():
    pd = pytest.importorskip("pandas")
    from fuzzytool.integrations.pandas import predict_df

    sys, *_ = datasets.credit_risk()
    df = pd.DataFrame({"score": [520.0, 660.0, 800.0], "dti": [42.0, 30.0, 10.0]},
                      index=["a", "b", "c"])
    out = predict_df(sys, df)
    assert isinstance(out, pd.Series)
    assert out.name == "premium"
    assert list(out.index) == ["a", "b", "c"]
    assert np.allclose(out.to_numpy(), sys.predict(score=df["score"].to_numpy(),
                                                   dti=df["dti"].to_numpy()))


def test_predict_df_missing_column_raises():
    pd = pytest.importorskip("pandas")
    from fuzzytool.integrations.pandas import predict_df

    sys, *_ = datasets.credit_risk()
    with pytest.raises(KeyError):
        predict_df(sys, pd.DataFrame({"score": [700.0]}))


def test_rules_dataframe():
    pytest.importorskip("pandas")
    from fuzzytool.integrations.pandas import rules_dataframe

    sys, *_ = datasets.credit_risk()
    df = rules_dataframe(sys)
    assert len(df) == len(sys.rules)
    assert set(df.columns) == {"antecedent", "consequent", "weight"}


def test_memberships_dataframe():
    pytest.importorskip("pandas")
    from fuzzytool.integrations.pandas import memberships_dataframe

    X = datasets.make_blobs(seed=0)
    res = fz.fuzzy_cmeans(X, c=3, seed=0)
    df = memberships_dataframe(res)
    assert df.shape[0] == X.shape[0]
    assert "label" in df.columns
    member_cols = [c for c in df.columns if c.startswith("cluster_")]
    assert len(member_cols) == 3
    # FCM memberships sum to 1 across clusters
    assert np.allclose(df[member_cols].to_numpy().sum(axis=1), 1.0)


def test_components_dataframe():
    pytest.importorskip("pandas")
    from fuzzytool.integrations.pandas import components_dataframe

    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    ft = fz.FTransform(0, 2 * np.pi, n_basis=12).fit(x, y)
    df = components_dataframe(ft)
    assert len(df) == 12
    assert set(df.columns) == {"node", "component"}


# --- scikit-learn integration ----------------------------------------------

def test_fuzzifier_shapes_and_names():
    from fuzzytool.integrations.sklearn import Fuzzifier

    score = fz.Variable("score", (300, 850), terms=["poor", "fair", "good"])
    dti = fz.Variable("dti", (0, 50), terms=["low", "high"])
    X = np.array([[700.0, 20.0], [400.0, 45.0]])
    fz_ = Fuzzifier([score, dti]).fit(X)
    Z = fz_.transform(X)
    assert Z.shape == (2, 5)                       # 3 + 2 terms
    assert (Z >= 0).all() and (Z <= 1).all()
    assert list(fz_.get_feature_names_out()) == [
        "score[poor]", "score[fair]", "score[good]", "dti[low]", "dti[high]"]


def test_wang_mendel_regressor_learns_mean():
    from fuzzytool.integrations.sklearn import WangMendelRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(300, 2))
    y = (X[:, 0] + X[:, 1]) / 2
    a = fz.Variable("a", (0, 10), terms=["lo", "mid", "hi"])
    b = fz.Variable("b", (0, 10), terms=["lo", "mid", "hi"])
    out = fz.Variable("out", (0, 10), terms=["lo", "mid", "hi"])
    reg = WangMendelRegressor([a, b], out).fit(X, y)
    pred = reg.predict(np.array([[8.0, 7.0]]))
    assert 4.0 < pred[0] < 8.0
    assert reg.score(X, y) > 0.5                    # explains most of the variance


def test_fuzzy_system_regressor_matches_predict():
    from fuzzytool.integrations.sklearn import FuzzySystemRegressor

    sys, *_ = datasets.credit_risk()
    X = np.array([[520.0, 42.0], [800.0, 10.0]])
    reg = FuzzySystemRegressor(sys, columns=["score", "dti"]).fit(X)
    pred = reg.predict(X)
    assert np.allclose(pred, sys.predict(score=X[:, 0], dti=X[:, 1]))


def test_estimators_clone_in_sklearn():
    sklearn_base = pytest.importorskip("sklearn.base")
    from fuzzytool.integrations.sklearn import Fuzzifier

    score = fz.Variable("score", (300, 850), terms=["poor", "good"])
    est = Fuzzifier([score])
    cloned = sklearn_base.clone(est)
    assert cloned.get_params().keys() == est.get_params().keys()
