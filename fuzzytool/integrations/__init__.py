"""Optional integrations with the wider Python data ecosystem.

Each submodule is self-contained and pulls in a third-party dependency only when
imported, so the core package stays pure NumPy:

* :mod:`fuzzytool.integrations.pandas` — DataFrame in/out and tabular views of
  rule bases, cluster memberships and F-transform components (``[pandas]``).
* :mod:`fuzzytool.integrations.sklearn` — fuzzy systems as scikit-learn
  estimators/transformers, usable in ``Pipeline`` and ``GridSearchCV``
  (``[sklearn]``).

Importing a submodule without its dependency installed raises a clear
``ImportError`` pointing at the right extra.
"""
