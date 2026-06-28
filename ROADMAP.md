# Roadmap

Phases mirror the structure of the sibling project *turboswarm*: each phase is a
self-contained, testable increment.

## ✅ Phase 1 — Core + Mamdani

- [x] Membership functions (tri, trap, gauss, gbell, sigmoid) behind the
      `MembershipFunction` Protocol.
- [x] T-norms / s-norms resolved by name; standard complement.
- [x] `Variable` + operator-based antecedent tree (`&`, `|`, `~`).
- [x] Mamdani inference + defuzzification (centroid, bisector, MOM/SOM/LOM).
- [x] Fraud-alert example, test suite, packaging, docs, CI.

## ✅ Phase 2 — TSK + visualization

- [x] Takagi-Sugeno inference (zero/first-order + callable consequents).
- [x] `viz`: membership plots, 2-input control surface.
- [ ] Example notebooks (fraud-alert, TSK).

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

## ✅ Phase 5 — ANFIS + F-transform

- [x] ANFIS: trainable first-order TSK with Jang's hybrid learning (least
      squares for consequents + gradient descent for premises).
- [x] F-transform: direct and inverse over a triangular fuzzy partition
      (partition of unity), with smoothing/denoising.

## ✅ v0.2.0 — Decision-making, rule learning, engineering

- [x] Fuzzy numbers (triangular/trapezoidal) + arithmetic; fuzzy MCDM
      (`fuzzy_topsis`, `fuzzy_ahp`).
- [x] Rule learning from data (`wang_mendel`); Tsukamoto inference with
      invertible ramp/sigmoid MFs.
- [x] Vectorized batch inference (`Mamdani.predict` / `TSK.predict`);
      JSON serialization (`save`/`load`); scikit-learn estimator interface on ANFIS.

## 🔌 Ecosystem integrations (`fuzzytool.integrations.*`)

Each pulls its dependency in only on import, behind its own extra.

- [x] **pandas** (`[pandas]`): `predict_df`, `rules_dataframe`,
      `memberships_dataframe`, `components_dataframe`.
- [x] **scikit-learn** (`[sklearn]`): `Fuzzifier` transformer,
      `WangMendelRegressor`, `FuzzySystemRegressor` (ANFIS already compatible).
- [x] **PyTorch** (`[torch]`): `FuzzyLayer`, a differentiable first-order TSK
      `nn.Module` trainable by autograd and composable into a network.
- [x] **SciPy** (`[scipy]`): `tune` MF parameters to data via
      `scipy.optimize.least_squares` (returns the `OptimizeResult`).
- [x] **turboswarm** (`[turboswarm]`): `tune` MF parameters with gradient-free
      Particle Swarm Optimization (the sibling PSO library; returns `PsoResult`).
- [x] **Optuna** (`[optuna]`): `suggest_inference_spec`, `suggest_anfis` and a
      ready-made `tune_anfis` study.
- [x] **Joblib / Dask** (`[parallel]`, `[dask]`): `parallel_predict`,
      `multi_start_cmeans`, `dask_predict`.
- [x] **Agents (LangChain/LangGraph)** (`[agents]`): `explain` plus an
      `inference_tool` LangChain tool that reports which rules fired.

## ✅ General type-2 (zSlices)

- [x] `GeneralType2MF` via the zSlices / alpha-plane representation (a stack of
      IT2 slices; also exposes the overall FOU, so it degrades to IT2).
- [x] Constructors `gt2_from_it2`, `gt2_gauss_uncertain_mean`, `gt2_scale`
      (triangular secondary MF).
- [x] `GeneralType2Mamdani` (z-weighted stack of IT2 Mamdani inferences) and
      `centroid_gt2` zSlices type reduction.

## Ideas for later

- Fuzzy cognitive maps (+ grey FCM).
- More defuzzifiers; non-triangular GT2 secondary MFs.

## ⏳ Phase 6 — Release (v0.1.0)

- [x] Example notebooks (`notebooks/`) and a comparison page vs scikit-fuzzy.
- [x] JOSS `paper.md`; Zenodo metadata (`.zenodo.json`) and a citing/releasing page.
- [x] GitHub Pages live (`https://fuzzytool.github.io`).
- [x] PyPI trusted publisher registered; tagging a version runs `release-pypi.yml`
      (published 0.1.0 → 0.4.0 via OIDC, no stored token).
- [x] Zenodo GitHub integration; each GitHub release is archived with a DOI
      (`10.5281/zenodo.20836712`).
