# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Interval type-2 (IT2)** support (`fuzzytool.type2`):
  - IT2 membership functions with a footprint of uncertainty: `it2` (explicit
    LMF/UMF), `it2_scale` (height), `it2_gauss_uncertain_mean`,
    `it2_gauss_uncertain_std`.
  - Interval-valued antecedent evaluation (`Antecedent.eval_interval`); type-1
    and IT2 terms can be mixed in one rule.
  - `IT2Mamdani` (center-of-sets type reduction) and `IT2TSK` engines.
  - Karnik-Mendel type reduction: `km_endpoint`, `karnik_mendel`, `centroid_it2`.
  - `viz.plot_it2_variable` (shaded FOU); `datasets.credit_risk_it2`.

## [0.1.0] - 2026-06-24

### Added

- Core membership functions: triangular, trapezoidal, gaussian, generalized
  bell, sigmoid (`fuzzytool.membership`).
- T-norms and s-norms (min/prod/Łukasiewicz, max/probor/Łukasiewicz), resolved
  by name (`fuzzytool.norms`).
- `Variable` (linguistic variable) with auto-generated or explicit terms, and an
  operator-based rule-antecedent expression tree (`&`, `|`, `~`).
- **Mamdani** inference with configurable implication/aggregation and
  defuzzification (centroid, bisector, MOM/SOM/LOM).
- **Takagi-Sugeno (TSK)** inference (zero- and first-order, plus callable
  consequents).
- `fuzzytool.viz`: membership-function plots and 2-input control surfaces.
- `fuzzytool.datasets.tipper`: the classic example system.
- Test suite, MkDocs Material documentation, and CI.
