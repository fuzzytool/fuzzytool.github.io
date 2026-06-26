"""pandas integration: DataFrame in/out and tabular views of fuzzy objects.

Install with ``pip install fuzzytool[pandas]``. Nothing here imports pandas at
module load beyond a guarded check, so the rest of the toolkit stays dependency
-free.

* :func:`predict_df` — batch inference straight from a DataFrame.
* :func:`rules_dataframe` — a readable table of a system's rule base.
* :func:`memberships_dataframe` — a clustering result's membership matrix.
* :func:`components_dataframe` — an F-transform's components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._util import input_variable_names, output_name

if TYPE_CHECKING:
    import pandas as pd

    from ..inference import TSK, Mamdani


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the pandas integration needs pandas; install with "
            "`pip install fuzzytool[pandas]`"
        ) from exc
    return pd


def predict_df(
    system: Mamdani | TSK,
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.Series | pd.DataFrame:
    """Run vectorized inference over a DataFrame.

    Args:
        system: a :class:`~fuzzytool.inference.Mamdani` or
            :class:`~fuzzytool.inference.TSK` system.
        df: a DataFrame whose columns include the system's input variables.
        columns: input column names to use, in order. Defaults to the variables
            the system's rules reference (see
            :func:`~fuzzytool.integrations._util.input_variable_names`).

    Returns:
        A ``Series`` (single output, named after the output variable) or a
        ``DataFrame`` (multiple outputs), aligned to ``df``'s index.
    """
    pd = _require_pandas()
    names = list(columns) if columns is not None else input_variable_names(system)
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing input columns: {missing}")
    inputs = {n: df[n].to_numpy(dtype=float) for n in names}
    out = system.predict(**inputs)
    if isinstance(out, dict):
        return pd.DataFrame(out, index=df.index)
    return pd.Series(out, index=df.index, name=output_name(system))


def rules_dataframe(system):
    """Return the system's rule base as a DataFrame (one row per rule).

    Columns: ``antecedent`` (the ``IF`` part as text), ``consequent`` (the
    ``THEN`` part) and ``weight``.
    """
    pd = _require_pandas()
    rows = [
        {
            "antecedent": str(rule.antecedent),
            "consequent": str(rule.consequent),
            "weight": rule.weight,
        }
        for rule in system.rules
    ]
    return pd.DataFrame(rows)


def memberships_dataframe(result, index=None, prefix: str = "cluster"):
    """Return a clustering result's membership matrix as a DataFrame.

    One row per sample, one column ``{prefix}_{k}`` per cluster, plus a ``label``
    column holding the hard (argmax) assignment.
    """
    pd = _require_pandas()
    u = np.asarray(result.u, dtype=float)        # (c, n)
    columns = [f"{prefix}_{k}" for k in range(u.shape[0])]
    df = pd.DataFrame(u.T, columns=columns, index=index)
    df["label"] = np.asarray(result.labels)
    return df


def components_dataframe(ftransform):
    """Return an :class:`~fuzzytool.ftransform.FTransform`'s components as a DataFrame.

    Columns: ``node`` (the basis-function center) and ``component`` (its value).
    """
    pd = _require_pandas()
    if ftransform.components_ is None:
        raise ValueError("the F-transform has no components yet; call fit/direct first")
    return pd.DataFrame(
        {"node": np.asarray(ftransform.nodes, dtype=float),
         "component": np.asarray(ftransform.components_, dtype=float)}
    )


__all__ = ["predict_df", "rules_dataframe", "memberships_dataframe",
           "components_dataframe"]
