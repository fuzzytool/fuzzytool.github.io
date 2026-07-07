# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.1] - 2026-07-07

### Fixed

- **`sigmoid` membership function** is now numerically stable at extreme
  arguments: `exp` is only ever applied to non-positive values, so large inputs
  no longer trigger an overflow `RuntimeWarning` (they saturate cleanly to 0/1).
  Output is unchanged in the numerically safe range.

### Changed

- **`Mamdani` caches consequent set shapes** across inference calls. A
  consequent term's membership over its output universe depends only on
  `(term, universe)`, not on the inputs, so it is now memoized (keyed by the
  membership-function identity, which auto-invalidates when a term is replaced).
  Repeated `__call__`/`predict` runs are faster; results are identical.
- Internal cleanup: shared implication logic between `Mamdani.__call__` and
  `Mamdani.predict`, and a single mid-universe fallback helper across the
  defuzzifiers.

## [0.5.0] - 2026-06-28

### Added

- **General type-2 (GT2) fuzzy sets** via the zSlices / alpha-plane
  representation (`fuzzytool.type2.general`):
  - `GeneralType2MF` — a GT2 set as a stack of IT2 z-slices; also exposes the
    overall FOU (`lower`/`upper`), so it can stand in for its IT2 footprint.
  - Constructors `gt2_from_it2`, `gt2_gauss_uncertain_mean`, `gt2_scale`
    (triangular secondary membership peaking at the principal MF).
  - `GeneralType2Mamdani` — inference as a z-weighted stack of IT2 Mamdani
    runs, reusing the Karnik-Mendel machinery.
  - `centroid_gt2` — zSlices type reduction of a single GT2 set.

## [0.4.0] - 2026-06-28

### Added

- **turboswarm integration** (`fuzzytool.integrations.turboswarm`, extra
  `[turboswarm]`): `tune` fits a system's built-in membership-function
  parameters to data with the sibling library's gradient-free, global Particle
  Swarm Optimization — the metaheuristic counterpart of the SciPy least-squares
  tuner. Returns turboswarm's `PsoResult`.

### Changed

- Shared membership-function tuning helpers (`sanitize_mf_params`,
  `tunable_terms`, `MF_PENALTY`) moved to `fuzzytool.integrations._util` and
  reused by both the SciPy and turboswarm tuners.

## [0.3.0] - 2026-06-27

### Added

- **Ecosystem integrations** under `fuzzytool.integrations.*` — each behind its
  own extra and importing its dependency only on use, so the core stays pure
  NumPy:
  - **pandas** (`[pandas]`): `predict_df`, `rules_dataframe`,
    `memberships_dataframe`, `components_dataframe`.
  - **scikit-learn** (`[sklearn]`): `Fuzzifier` transformer (crisp →
    membership-degree features), `WangMendelRegressor`, `FuzzySystemRegressor`.
  - **PyTorch** (`[torch]`): `FuzzyLayer`, a differentiable first-order TSK
    `nn.Module` trainable by autograd and composable into a network.
  - **SciPy** (`[scipy]`): `tune`, fitting a system's membership-function
    parameters to data via `scipy.optimize.least_squares`.
  - **Optuna** (`[optuna]`): `suggest_inference_spec`, `suggest_anfis`, and a
    ready-made `tune_anfis` study.
  - **Joblib / Dask** (`[parallel]`, `[dask]`): `parallel_predict`,
    `multi_start_cmeans`, `dask_predict`.
  - **LLM agents** (`[agents]`): `explain` (crisp output + fired rules) and
    `inference_tool` (a LangChain `StructuredTool`).
- Tutorials section (investment-risk advisor, ANFIS, Fuzzy TOPSIS, clustering)
  and an Integrations guide page; rendered plots and computed results embedded
  throughout the docs.

## [0.2.0] - 2026-06-24

### Added

- **Fuzzy numbers & MCDM**: `fuzzytool.fuzzynum` (triangular/trapezoidal numbers
  with arithmetic, alpha-cuts, centroid, distance, ranking) and `fuzzytool.mcdm`
  (`fuzzy_topsis`, `fuzzy_ahp`).
- **Rule learning**: `wang_mendel` generates a Mamdani rule base from data.
- **Tsukamoto** inference (`fuzzytool.inference.Tsukamoto`) with monotonic
  consequents; added invertible `ramp_up` / `ramp_down` membership functions and
  an `inverse` on `sigmoid`.
- **Batch inference**: `Mamdani.predict` / `TSK.predict` evaluate array-valued
  inputs in a vectorized pass.
- **Serialization**: `fz.save` / `fz.load` (JSON) for Mamdani/TSK systems, plus
  `to_dict`/`from_dict` on membership functions and variables.
- **scikit-learn compatibility**: `ANFIS.get_params` / `set_params`.

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
- `fuzzytool.datasets.credit_risk`: the flagship example system.
- Example notebooks (`notebooks/`): quickstart, interval type-2, clustering, and
  ANFIS/F-transform — committed executed.
- Documentation (MkDocs Material), CI, a comparison page vs scikit-fuzzy, a
  citing/releasing page, and `.zenodo.json` for DOI archival.
- **ANFIS** (`fuzzytool.anfis.ANFIS`): a trainable first-order Sugeno system over
  a grid partition, fit with Jang's hybrid scheme (least-squares consequents +
  gradient-descent premises). `fit` / `predict` / `history_`.
- **F-transform** (`fuzzytool.ftransform.FTransform`): direct and inverse fuzzy
  transform over a triangular partition of unity, with `fit` / `smooth` for
  denoising and compression.
- **Fuzzy clustering** (`fuzzytool.cluster`):
  - `fuzzy_cmeans` (Bezdek FCM), `gustafson_kessel` (adaptive Mahalanobis norm),
    `possibilistic_cmeans` (typicalities); all seeded for reproducibility.
  - Validity metrics: `partition_coefficient`, `partition_entropy`, `xie_beni`.
  - `viz.plot_clusters` and `datasets.make_blobs`.
- **Interval type-2 (IT2)** support (`fuzzytool.type2`):
  - IT2 membership functions with a footprint of uncertainty: `it2` (explicit
    LMF/UMF), `it2_scale` (height), `it2_gauss_uncertain_mean`,
    `it2_gauss_uncertain_std`.
  - Interval-valued antecedent evaluation (`Antecedent.eval_interval`); type-1
    and IT2 terms can be mixed in one rule.
  - `IT2Mamdani` (center-of-sets type reduction) and `IT2TSK` engines.
  - Karnik-Mendel type reduction: `km_endpoint`, `karnik_mendel`, `centroid_it2`.
  - `viz.plot_it2_variable` (shaded FOU); `datasets.credit_risk_it2`.
- Test suite covering every module.
