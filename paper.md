---
title: "fuzzytool: a clean, extensible fuzzy-logic toolkit in pure Python"
tags:
  - Python
  - fuzzy logic
  - fuzzy inference systems
  - Mamdani
  - Takagi-Sugeno
authors:
  - name: Jose L. Salmeron
    affiliation: 1
affiliations:
  - name: CUNEF Universidad, Madrid, Spain
    index: 1
date: 24 June 2026
bibliography: paper.bib
---

# Summary

`fuzzytool` is a fuzzy-logic library for Python whose goals are a composable
API, easy algorithm comparison, visualization, and code clarity. It provides
membership functions, configurable fuzzy connectives (t-norms and s-norms),
Mamdani and Takagi-Sugeno (TSK) inference, a family of defuzzification methods,
and matplotlib-based visualization, in pure Python on top of NumPy.

Its defining feature is an *extensible-by-protocol* design: the inference
engines depend only on small Python Protocols (a membership function is a
callable `x -> degree`; a connective is a callable `(a, b) -> result`; a
defuzzifier is a callable `(x, y) -> value`). Adding a new variant therefore
means adding a callable, never modifying the inference loop. Rules are written
with operator overloading — `&` for the t-norm, `|` for the s-norm, `~` for the
complement — so a rule base reads like the logic it encodes.

# Statement of need

Fuzzy inference systems are widely used in control, decision support, and
modeling under vagueness [@zadeh1965; @mamdani1975]. In Python, the most common
package is `scikit-fuzzy`, whose control API is verbose (explicit
`Antecedent`/`Consequent`/`ControlSystem` scaffolding) and whose scope is
limited; modern needs such as type-2 fuzzy sets, neuro-fuzzy training, and the
fuzzy transform are not first-class.

`fuzzytool` addresses the ergonomics directly with an operator-based rule syntax
and a uniform, pluggable core, and is structured around a roadmap that adds
interval type-2 systems, fuzzy clustering, ANFIS, and the F-transform on the
same extensible foundation. The pure-Python/NumPy implementation keeps
installation trivial (a universal wheel, no compilation) while remaining fast
enough for research and teaching.

# Functionality

- **Membership functions:** triangular, trapezoidal, Gaussian, generalized
  bell, and sigmoidal, plus arbitrary user callables.
- **Connectives:** minimum, product, and Łukasiewicz t-norms; maximum,
  probabilistic-OR, and Łukasiewicz s-norms.
- **Inference:** Mamdani (with configurable implication and aggregation) and
  zero/first-order Takagi-Sugeno.
- **Defuzzification:** centroid, bisector, and mean/smallest/largest of maxima.
- **Visualization:** membership-function plots and control surfaces.

# Acknowledgements

We thank the open-source scientific Python community.

# References
