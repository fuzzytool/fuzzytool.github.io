"""fuzzytool — a clean, extensible fuzzy-logic toolkit in pure Python + NumPy.

Quick start (a credit-risk premium system):

    >>> import fuzzytool as fz
    >>> score = fz.Variable("score", (300, 850), terms=["poor", "fair", "good", "excellent"])
    >>> dti = fz.Variable("dti", (0, 50), terms=["low", "moderate", "high"])
    >>> premium = fz.Variable("premium", (0, 12), terms=["low", "medium", "high"])
    >>> sys = fz.Mamdani()
    >>> _ = sys.rule(score["poor"] | dti["high"], premium["high"])
    >>> _ = sys.rule(score["fair"] & dti["moderate"], premium["medium"])
    >>> _ = sys.rule(score["good"] | score["excellent"], premium["low"])
    >>> sys(score=800, dti=10) < sys(score=520, dti=42)
    True

Everything is pluggable behind small Protocols (membership functions,
t-/s-norms, defuzzifiers): a new variant is a new callable, never a change to
the inference loop.
"""

from . import cluster, datasets, defuzz, membership, norms, type2
from .cluster import fuzzy_cmeans, gustafson_kessel, possibilistic_cmeans
from .inference import TSK, Mamdani
from .membership import gauss, gbell, sigmoid, trap, tri
from .sets import Variable
from .type2 import (
    IT2TSK,
    IT2Mamdani,
    it2,
    it2_gauss_uncertain_mean,
    it2_gauss_uncertain_std,
    it2_scale,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Variable",
    "Mamdani",
    "TSK",
    # membership shortcuts
    "tri", "trap", "gauss", "gbell", "sigmoid",
    # interval type-2
    "IT2Mamdani", "IT2TSK",
    "it2", "it2_scale", "it2_gauss_uncertain_mean", "it2_gauss_uncertain_std",
    # fuzzy clustering
    "fuzzy_cmeans", "gustafson_kessel", "possibilistic_cmeans",
    # submodules
    "membership", "norms", "defuzz", "datasets", "type2", "cluster",
]
