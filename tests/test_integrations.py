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


# --- PyTorch integration ----------------------------------------------------

def test_fuzzy_layer_forward_shape():
    torch = pytest.importorskip("torch")
    from fuzzytool.integrations.torch import FuzzyLayer

    layer = FuzzyLayer(n_inputs=2, n_mf=3)
    out = layer(torch.zeros(5, 2))
    assert out.shape == (5, 1)


def test_fuzzy_layer_trains_and_backprops():
    torch = pytest.importorskip("torch")
    from fuzzytool.integrations.torch import FuzzyLayer

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(300, 2)).astype("float32")
    y = (np.sin(3 * X[:, 0]) + X[:, 1] ** 2).astype("float32")

    layer = FuzzyLayer(n_inputs=2, n_mf=4, init_range=(0, 1))
    hist = layer.fit(X, y, epochs=200, lr=0.05)
    assert hist[-1] < hist[0] * 0.5                 # error at least halved
    # gradients reach the membership-function parameters
    layer(torch.tensor(X[:4])).sum().backward()
    assert layer.centers.grad is not None


def test_fuzzy_layer_rejects_bad_shape():
    torch = pytest.importorskip("torch")
    from fuzzytool.integrations.torch import FuzzyLayer

    layer = FuzzyLayer(n_inputs=2)
    with pytest.raises(ValueError):
        layer(torch.zeros(5, 3))


# --- SciPy integration ------------------------------------------------------

def test_scipy_tune_reduces_error():
    pytest.importorskip("scipy")
    from fuzzytool.integrations.scipy import tune

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(150, 1))
    y = (np.sin(X[:, 0]) + 0.1 * X[:, 0]).ravel()
    x = fz.Variable("x", (0, 10), terms=["lo", "mid", "hi"])
    sys = fz.TSK()
    sys.rule(x["lo"], 0.0)
    sys.rule(x["mid"], 0.5)
    sys.rule(x["hi"], 1.0)

    before = np.sqrt(np.nanmean((sys.predict(x=X[:, 0]) - y) ** 2))
    res = tune(sys, X, y, columns=["x"])
    after = np.sqrt(np.nanmean((sys.predict(x=X[:, 0]) - y) ** 2))
    assert after < before
    assert hasattr(res, "x") and hasattr(res, "cost")


def test_turboswarm_tune_reduces_error():
    pytest.importorskip("turboswarm")
    from fuzzytool.integrations.turboswarm import tune

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(150, 1))
    y = (np.sin(X[:, 0]) + 0.1 * X[:, 0]).ravel()
    x = fz.Variable("x", (0, 10), terms=["lo", "mid", "hi"])
    sys = fz.TSK()
    sys.rule(x["lo"], 0.0)
    sys.rule(x["mid"], 0.5)
    sys.rule(x["hi"], 1.0)

    before = np.sqrt(np.nanmean((sys.predict(x=X[:, 0]) - y) ** 2))
    res = tune(sys, X, y, columns=["x"], n_particles=20, max_iter=40, seed=0)
    after = np.sqrt(np.nanmean((sys.predict(x=X[:, 0]) - y) ** 2))
    assert after < before
    assert hasattr(res, "best_position") and hasattr(res, "best_value")


def test_scipy_tune_keeps_membership_valid():
    pytest.importorskip("scipy")
    from fuzzytool import membership as mf
    from fuzzytool.integrations.scipy import tune

    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, size=(80, 1))
    y = np.cos(X[:, 0])
    x = fz.Variable("x", (0, 10), terms=["lo", "hi"])
    sys = fz.TSK()
    sys.rule(x["lo"], -1.0)
    sys.rule(x["hi"], 1.0)
    tune(sys, X, y, columns=["x"], max_nfev=50)
    # triangular breakpoints remain ordered after tuning
    for term in x.terms.values():
        a, b, c = mf.to_dict(term)["params"]
        assert a <= b <= c


# --- Optuna integration -----------------------------------------------------

def test_suggest_inference_spec():
    optuna = pytest.importorskip("optuna")
    from fuzzytool.integrations.optuna import suggest_inference_spec

    spec = suggest_inference_spec(optuna.create_study().ask())
    assert set(spec) == {"tnorm", "snorm", "implication", "defuzz"}
    sys = fz.Mamdani(**spec)                    # spec is a valid Mamdani config
    assert sys is not None


def test_tune_anfis_returns_fitted_model():
    pytest.importorskip("optuna")
    from fuzzytool.integrations.optuna import tune_anfis

    x = np.linspace(0, 6, 100)[:, None]
    y = np.sin(x[:, 0])
    best, study = tune_anfis(x, y, n_trials=3, epochs=30, seed=0)
    assert best.history_                        # it was trained
    assert np.isfinite(study.best_value)
    assert best.n_mf == study.best_params["n_mf"]


# --- Joblib / Dask integration ----------------------------------------------

def test_parallel_predict_matches_predict():
    pytest.importorskip("joblib")
    from fuzzytool.integrations.parallel import parallel_predict

    sys, *_ = datasets.credit_risk()
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(300, 850, 200), rng.uniform(0, 50, 200)])
    out = parallel_predict(sys, X, columns=["score", "dti"], n_jobs=2,
                           backend="threading")
    assert np.allclose(out, sys.predict(score=X[:, 0], dti=X[:, 1]))


def test_multi_start_cmeans_picks_best():
    pytest.importorskip("joblib")
    from fuzzytool.integrations.parallel import multi_start_cmeans

    X = datasets.make_blobs(seed=0)
    best = multi_start_cmeans(X, c=3, n_starts=4, n_jobs=2, backend="threading")
    single = fz.fuzzy_cmeans(X, c=3, seed=0)
    assert best.objective <= single.objective + 1e-9      # no worse than seed 0
    assert best.centers.shape == (3, X.shape[1])


def test_dask_predict_matches_predict():
    pytest.importorskip("dask")
    from fuzzytool.integrations.parallel import dask_predict

    sys, *_ = datasets.credit_risk()
    X = np.array([[520.0, 42.0], [660.0, 30.0], [800.0, 10.0]])
    out = dask_predict(sys, X, columns=["score", "dti"], n_chunks=2)
    assert np.allclose(out, sys.predict(score=X[:, 0], dti=X[:, 1]))


# --- Agents integration -----------------------------------------------------

def test_explain_reports_fired_rules():
    from fuzzytool.integrations.agents import explain

    sys, *_ = datasets.credit_risk()
    ex = explain(sys, score=520, dti=42)
    assert np.isclose(ex["output"], sys(score=520, dti=42))
    assert ex["fired_rules"]                                  # at least one fired
    firings = [r["firing"] for r in ex["fired_rules"]]
    assert all(f > 0 for f in firings)
    assert firings == sorted(firings, reverse=True)           # strongest first


def test_inference_tool_invocable():
    pytest.importorskip("langchain_core")
    from fuzzytool.integrations.agents import inference_tool

    sys, *_ = datasets.credit_risk()
    tool = inference_tool(sys, columns=["score", "dti"])
    assert tool.name == "fuzzy_inference"
    res = tool.invoke({"score": 800, "dti": 10})
    assert "premium" in res and "fired_rules" in res
    assert np.isclose(res["premium"], sys(score=800, dti=10))
