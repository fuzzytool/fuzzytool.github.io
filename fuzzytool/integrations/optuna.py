"""Optuna integration: search fuzzy-system structure and hyperparameters.

Install with ``pip install fuzzytool[optuna]``. Fuzzy systems have plenty of
discrete/continuous knobs — which t-norm, which defuzzifier, how many membership
functions, what learning rate — that are awkward to grid-search by hand. These
helpers turn an Optuna ``trial`` into a configured system, plus a ready-made
:func:`tune_anfis` study.

* :func:`suggest_inference_spec` — sample the connectives/implication/defuzz of a
  Mamdani system from a trial.
* :func:`suggest_anfis` — build an :class:`~fuzzytool.anfis.ANFIS` with a
  trial-suggested ``n_mf`` and ``learning_rate``.
* :func:`tune_anfis` — run a full study that minimizes training RMSE and returns
  the best (refit) model.
"""

from __future__ import annotations

import numpy as np

from ..anfis import ANFIS

# Sensible, well-behaved choices (a subset of the full registries).
_TNORMS = ["min", "prod", "lukasiewicz"]
_SNORMS = ["max", "probor", "lukasiewicz"]
_DEFUZZ = ["centroid", "bisector", "mom", "som", "lom"]


def suggest_inference_spec(trial, prefix: str = "") -> dict:
    """Sample a Mamdani spec ``{tnorm, snorm, implication, defuzz}`` from a trial.

    ``prefix`` namespaces the parameter names so several specs can coexist in one
    study. Pass the result straight to :class:`~fuzzytool.inference.Mamdani`.
    """
    p = prefix
    return {
        "tnorm": trial.suggest_categorical(f"{p}tnorm", _TNORMS),
        "snorm": trial.suggest_categorical(f"{p}snorm", _SNORMS),
        "implication": trial.suggest_categorical(f"{p}implication", ["min", "prod"]),
        "defuzz": trial.suggest_categorical(f"{p}defuzz", _DEFUZZ),
    }


def suggest_anfis(
    trial,
    n_inputs: int,
    n_mf_range: tuple[int, int] = (2, 5),
    lr_range: tuple[float, float] = (1e-3, 0.2),
) -> ANFIS:
    """Build an ANFIS with a trial-suggested ``n_mf`` and ``learning_rate``."""
    n_mf = trial.suggest_int("n_mf", n_mf_range[0], n_mf_range[1])
    lr = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
    return ANFIS(n_inputs=n_inputs, n_mf=n_mf, learning_rate=lr)


def tune_anfis(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    n_mf_range: tuple[int, int] = (2, 5),
    lr_range: tuple[float, float] = (1e-3, 0.2),
    epochs: int = 100,
    seed: int | None = None,
) -> tuple:
    """Tune an ANFIS's ``n_mf``/``learning_rate`` with an Optuna study.

    Args:
        X: inputs, shape ``(n_samples, n_inputs)``.
        y: targets, shape ``(n_samples,)``.
        n_trials: number of Optuna trials.
        n_mf_range: ``(min, max)`` membership functions per input to search.
        lr_range: ``(min, max)`` learning rate to search (log scale).
        epochs: training epochs per trial.
        seed: seed for Optuna's sampler (reproducible search).

    Returns:
        A ``(best_model, study)`` tuple: a fresh ANFIS trained with the best
        hyperparameters, and the Optuna study.
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the optuna integration needs optuna; install with "
            "`pip install fuzzytool[optuna]`"
        ) from exc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_inputs = X.shape[1]

    def objective(trial):
        model = suggest_anfis(trial, n_inputs, n_mf_range, lr_range)
        model.fit(X, y, epochs=epochs)
        return model.history_[-1]              # final training RMSE

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = ANFIS(n_inputs=n_inputs, n_mf=study.best_params["n_mf"],
                 learning_rate=study.best_params["learning_rate"]).fit(X, y, epochs=epochs)
    return best, study


__all__ = ["suggest_inference_spec", "suggest_anfis", "tune_anfis"]
