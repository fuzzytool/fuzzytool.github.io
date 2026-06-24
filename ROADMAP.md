# Roadmap

Phases mirror the structure of the sibling project *turboswarm*: each phase is a
self-contained, testable increment.

## ✅ Phase 1 — Core + Mamdani

- [x] Membership functions (tri, trap, gauss, gbell, sigmoid) behind the
      `MembershipFunction` Protocol.
- [x] T-norms / s-norms resolved by name; standard complement.
- [x] `Variable` + operator-based antecedent tree (`&`, `|`, `~`).
- [x] Mamdani inference + defuzzification (centroid, bisector, MOM/SOM/LOM).
- [x] Tipper example, test suite, packaging, docs, CI.

## ✅ Phase 2 — TSK + visualization

- [x] Takagi-Sugeno inference (zero/first-order + callable consequents).
- [x] `viz`: membership plots, 2-input control surface.
- [ ] Example notebooks (tipper, TSK, comparison).

## ✅ Phase 3 — Type-2 / interval type-2

- [x] IT2 membership functions (footprint of uncertainty): explicit LMF/UMF,
      height scaling, uncertain Gaussian mean, uncertain Gaussian spread.
- [x] Interval antecedent evaluation (type-1 and IT2 terms mix freely).
- [x] IT2 inference: `IT2Mamdani` (center-of-sets) and `IT2TSK`.
- [x] Karnik-Mendel type reduction (`km_endpoint`, `karnik_mendel`,
      `centroid_it2`) + FOU visualization.

## ✅ Phase 4 — Fuzzy clustering

- [x] Fuzzy c-means (FCM).
- [x] Gustafson-Kessel.
- [x] Possibilistic c-means.
- [x] Cluster-quality metrics (partition coefficient, partition entropy,
      Xie-Beni) + 2-D cluster visualization and a synthetic-blob generator.

## ⏳ Phase 5 — ANFIS + F-transform

- [ ] ANFIS: trainable first-order TSK (gradient / least squares).
- [ ] F-transform: direct and inverse, with basis generation.

## ⏳ Phase 6 — Release

- [ ] Example notebooks and a comparison page vs scikit-fuzzy.
- [ ] JOSS `paper.md`, Zenodo DOI.
- [ ] Publish to PyPI; enable GitHub Pages.
