"""scikit-learn integration: fuzzy systems as estimators and transformers.

These objects follow the scikit-learn estimator protocol (``fit`` returns
``self``, plus ``predict``/``transform``, ``get_params``/``set_params`` and a
``score``/``fit_transform``), so they slot into a ``Pipeline``, ``GridSearchCV``
or ``cross_val_score`` — yet, like :class:`~fuzzytool.anfis.ANFIS`, they do **not
import scikit-learn**. The ``[sklearn]`` extra simply installs scikit-learn so
you have those tools available:

    pip install fuzzytool[sklearn]

* :class:`Fuzzifier` — a transformer turning crisp features into fuzzy
  membership-degree features (one column per term).
* :class:`WangMendelRegressor` — learns a Mamdani rule base from data
  (Wang-Mendel) and predicts with it.
* :class:`FuzzySystemRegressor` — wraps an already-built Mamdani/TSK system so it
  can be scored or pipelined.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

from ._util import input_variable_names

if TYPE_CHECKING:
    from ..inference import TSK, Mamdani
    from ..sets import Variable


class _Estimator:
    """Minimal scikit-learn-compatible base (get_params/set_params by introspection)."""

    def get_params(self, deep: bool = True) -> dict:
        names = [p for p in inspect.signature(self.__init__).parameters if p != "self"]
        return {n: getattr(self, n) for n in names}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _term_degree(membership, x):
    """Membership degree of ``x``; an IT2 term collapses to its interval mean."""
    if hasattr(membership, "lower") and hasattr(membership, "upper"):
        return (np.asarray(membership.lower(x), dtype=float)
                + np.asarray(membership.upper(x), dtype=float)) / 2.0
    return np.asarray(membership(x), dtype=float)


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class Fuzzifier(_Estimator):
    """Expand crisp features into fuzzy membership-degree features.

    Column ``j`` of ``X`` is interpreted through ``variables[j]``: each of that
    variable's terms becomes one output column holding the membership degree.
    The result is a soft, interpretable encoding that downstream linear or tree
    models can consume.

    Args:
        variables: one :class:`~fuzzytool.sets.Variable` per input column, in
            column order.
    """

    def __init__(self, variables: list[Variable]):
        self.variables = variables

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.variables):
            raise ValueError(
                f"X must be 2-D with {len(self.variables)} columns, got {X.shape}"
            )
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = [
            f"{v.name}[{term}]" for v in self.variables for term in v.terms
        ]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.variables):
            raise ValueError(
                f"X must have {len(self.variables)} columns, got {X.shape[1]}"
            )
        columns = [
            _term_degree(membership, X[:, j])
            for j, v in enumerate(self.variables)
            for membership in v.terms.values()
        ]
        return np.column_stack(columns)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


class WangMendelRegressor(_Estimator):
    """A regressor that learns a Mamdani rule base from data (Wang-Mendel).

    Args:
        inputs: input linguistic variables (pre-populated with terms), one per
            column of ``X``.
        output: the output linguistic variable (pre-populated with terms).
    """

    def __init__(self, inputs: list[Variable], output: Variable):
        self.inputs = inputs
        self.output = output

    def fit(self, X, y):
        from ..learn import wang_mendel

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.system_ = wang_mendel(X, y, self.inputs, self.output)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        inputs = {v.name: X[:, j] for j, v in enumerate(self.inputs)}
        return self.system_.predict(**inputs)

    def score(self, X, y):
        return _r2(y, self.predict(X))


class FuzzySystemRegressor(_Estimator):
    """Wrap an already-built single-output fuzzy system as an sklearn regressor.

    ``fit`` does not change the system (it is already defined); it only records
    the input column order. Use this to cross-validate, score, or pipeline a
    hand-written Mamdani/TSK system.

    Args:
        system: a single-output Mamdani or TSK system.
        columns: input variable names matching ``X``'s columns, in order.
            Defaults to the variables the system's rules reference.
    """

    def __init__(self, system: Mamdani | TSK, columns: list[str] | None = None):
        self.system = system
        self.columns = columns

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.input_names_ = (list(self.columns) if self.columns is not None
                             else input_variable_names(self.system))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        inputs = {name: X[:, j] for j, name in enumerate(self.input_names_)}
        out = self.system.predict(**inputs)
        if isinstance(out, dict):
            raise ValueError("FuzzySystemRegressor supports single-output systems only")
        return out

    def score(self, X, y):
        return _r2(y, self.predict(X))


__all__ = ["Fuzzifier", "WangMendelRegressor", "FuzzySystemRegressor"]
