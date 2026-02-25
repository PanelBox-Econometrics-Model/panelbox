---
title: "Changelog"
description: "PanelBox version history — all releases with key changes, migration notes, and breaking changes."
---

# Changelog

All notable changes to PanelBox are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and PanelBox adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Sections**: Added, Changed, Fixed, Deprecated, Removed, Security, Performance.

---

## [Unreleased]

### Added

#### Interactive Tutorial System

**Status**: Module 1 Complete — Production Ready

A comprehensive, hands-on learning system for panel data econometrics with PanelBox:

- **Module 1: Fundamentals** (3.5–4.5 hours, 4 tutorials)
    - Introduction to Panel Data Structures
    - Model Specification with Formulas
    - Estimation and Results Interpretation
    - Spatial Fundamentals (optional)
- Structured learning paths (linear, spatial, complete)
- Real datasets (Grunfeld, simulated spatial)
- Competency checkpoints and self-assessment

#### Advanced GMM Estimators

- **CUE-GMM** (`panelbox.gmm.ContinuousUpdatedGMM`) — Hansen-Heaton-Yaron (1996) with continuously updated weighting matrix; HAC and cluster-robust options; bootstrap variance
- **Bias-Corrected GMM** (`panelbox.gmm.BiasCorrectedGMM`) — Hahn-Kuersteiner (2002) analytical bias correction with first- and second-order terms
- **Enhanced GMM Diagnostics** (`panelbox.gmm.GMMDiagnostics`) — Hansen J-test with bootstrap p-values, C-statistic, weak instruments detection

#### Panel Selection Models

- **Panel Heckman** (`panelbox.models.selection.PanelHeckman`) — Two-step (Wooldridge 1995) and MLE estimation; Murphy-Topel SE correction; IMR diagnostics

#### Advanced Cointegration Tests

- **Westerlund (2007)** ECM-based tests — Gt, Ga, Pt, Pa with bootstrap critical values
- **Pedroni (1999)** — 7 residual-based statistics (panel and group)
- **Kao (1999)** — DF and ADF statistics

#### Panel Unit Root Tests

- **Hadri (2000)** LM test — Tests H0: all series stationary; heteroskedasticity-robust variant
- **Breitung (2000)** — Bias-corrected pooled estimator with detrending
- **Unified interface** `panel_unit_root_test()` — Run multiple tests simultaneously with comparative summary

#### Specification Tests and Specialized Models

- **Davidson-MacKinnon J-test** — Non-nested model comparison with cluster-robust SEs
- **Encompassing tests** — Cox test, Wald encompassing, likelihood ratio
- **Multinomial Logit** — FE (conditional MLE), RE (GLS), pooled; marginal effects
- **PPML** — Poisson Pseudo-Maximum Likelihood for gravity models; handles zeros

#### Phase 6 — Integration and Documentation (In Progress)

- Namespace integration: 60+ advanced methods accessible via `panelbox.*`
- Comprehensive documentation restructuring (34 pages across 5 deliverables)

---

## [1.0.0] — 2026-02-08

### Summary

**Panel VAR Module — Major Release**

PanelBox v1.0.0 introduces the complete Panel VAR module, establishing PanelBox as the first Python library with full feature parity to R's `pvar` and Stata's `pvar` packages.

**Key metrics**: 150+ tests, 90%+ coverage, validated against R (coefficients within ±1e-6).

### Added

- **PanelVAR** — OLS with fixed effects and GMM with FOD/FD transformations; automatic lag selection (AIC, BIC, HQIC, MBIC, MAIC, MQIC)
- **PanelVARResult** — `.params`, `.std_errors`, `.pvalues`, `.is_stable()`, `.hansen_j`, `.summary()`, `.to_latex()`
- **IRF Module** — Cholesky and Generalized impulse response functions; bootstrap and analytical confidence intervals
- **FEVD Module** — Forecast Error Variance Decomposition (Cholesky and Generalized)
- **Granger Causality** — Pairwise Wald tests, Dumitrescu-Hurlin (2012) panel Granger causality; bootstrap inference; causality network visualization
- **Forecasting** — h-step ahead iterative forecasts; bootstrap and analytical CIs; out-of-sample evaluation (RMSE, MAE, MAPE)
- **PanelVECM** — Panel Vector Error Correction Model for I(1) cointegrated systems; Johansen rank selection; long-run (beta) and short-run (gamma) separation

### Performance

| Benchmark | vs R | vs Stata |
|---|---|---|
| OLS estimation | ~1.5x faster | — |
| GMM estimation | ~1.3x faster | ~1.5x faster |
| IRF Bootstrap | ~1.5x faster | — |
| Memory usage | ~1.5–2x less | — |

### Validation

Validated against R (`plm`, `pvar`, `panelvar`, `urca`):

| Metric | Tolerance | Result |
|---|---|---|
| OLS Coefficients | ± 1e-6 | max diff = 3.2e-7 |
| GMM Coefficients | ± 1e-4 | max diff = 8.4e-5 |
| Hansen J statistic | ± 1e-3 | diff = 0.003 |
| IRFs | ± 1e-6 | max diff = 8.4e-7 |
| FEVD | ± 1e-3 | diff < 0.001 |
| Granger p-values | ± 1e-3 | diff < 0.001 |

---

## [0.8.0] — 2026-02-08

### Summary

**Test Runners and Master Report System**

### Added

- **ValidationTest Runner** — Configurable test runner with `quick`, `basic`, `full` presets; integrates with model `.validate()` methods
- **ComparisonTest Runner** — Multi-model comparison with automatic metric extraction (R², AIC, BIC)
- **Master Report System** — Comprehensive HTML report with experiment overview, model summary grid, navigation to sub-reports
- **PanelExperiment.save_master_report()** — One-liner master report generation

### Fixed

- Fixed `KeyError` when accessing `timestamp` in model metadata (now uses `.get()` with fallback)

---

## [0.7.0] — 2026-02-08

### Summary

**ResidualResult Container and Chart Registration Fix**

### Added

- **ResidualResult** — Diagnostic container with Shapiro-Wilk, Jarque-Bera, Durbin-Watson, Ljung-Box tests; standardized residuals for outlier detection; `summary()`, `save_html()`, `save_json()` methods
- **PanelExperiment.analyze_residuals()** — One-liner residual analysis

### Fixed

- **Chart registration**: Root cause was plotly not installed; all 35 charts now register correctly
- Fixed `jarque_bera()` return value unpacking
- Fixed `ljung_box()` DataFrame indexing (statsmodels return type change)
- HTML reports now include embedded interactive charts (102.9 KB vs 77.5 KB)

---

## [0.6.0] — 2026-02-08

### Summary

**Experiment Pattern and Result Containers**

### Added

- **PanelExperiment** — Unified workflow for fitting, validating, and comparing panel models
- **BaseResult** — Abstract base class for result containers
- **ValidationResult** — Model specification test results with `save_html()` and `save_json()`
- **ComparisonResult** — Multi-model comparison results with automatic best-model identification

---

## [0.5.0] — 2026-02-08

### Summary

**Comprehensive Visualization System — 28+ Interactive Charts**

### Added

- **28+ interactive chart types** using Plotly
- **3 professional themes**: Professional (corporate blue), Academic (journal-ready), Presentation (high-contrast)
- **Validation charts** (5): test overview, p-value distribution, test statistics, comparison heatmap, validation dashboard
- **Residual diagnostics** (7): QQ plot, residual vs. fitted, scale-location, residual vs. leverage, time series, distribution, partial regression
- **Model comparison** (4): coefficient comparison, forest plot, model fit comparison, information criteria
- **Panel-specific** (4): entity effects, time effects, between-within decomposition, panel structure heatmap
- **Econometric test visualizations** (4): ACF/PACF, unit root test, cointegration heatmap, cross-sectional dependence
- **Export system**: HTML, JSON, PNG, SVG, PDF; batch export across formats
- **Report generation**: HTML reports for validation, residuals, comparison, and panel diagnostics
- **Registry/Factory architecture**: `@register_chart` decorator, `ChartFactory.create()`

---

## [1.0.0] — 2026-02-05

### Summary

**Production Release — Complete Panel Data Econometrics Suite**

600+ unit tests, 93% passing, validated against Stata `xtabond2` and R `plm`.

### Added

**Static Panel Models**:

- Pooled OLS, Fixed Effects (Within), Random Effects (GLS), Between Estimator, First Differences
- Hausman test for FE vs. RE specification
- Formula interface (R-style with patsy)

**Dynamic Panel GMM**:

- Difference GMM (Arellano-Bond 1991) — one-step, two-step, iterative
- System GMM (Blundell-Bond 1998) — combined differenced and level equations
- Automatic instrument generation (GMM-style and IV-style)
- Instrument collapse (Roodman 2009)
- Windmeijer (2005) finite-sample SE correction

**Robust Standard Errors (8 types)**:

- HC0–HC3 (White, MacKinnon-White)
- One-way and two-way clustering (Cameron, Gelbach & Miller 2011)
- Driscoll-Kraay (1998)
- Newey-West HAC (1987)
- PCSE (Beck & Katz 1995)

**Bootstrap Inference**:

- 4 methods: Pairs, Wild, Block, Residual
- 3 CI methods: percentile, basic, studentized

**Diagnostic Tests** (50+):

- Heteroskedasticity: White, Breusch-Pagan, Modified Wald
- Serial correlation: Wooldridge, Breusch-Godfrey, Baltagi-Wu
- Cross-sectional dependence: Pesaran CD, Frees, BP-LM
- Unit root: LLC, IPS, Fisher
- Cointegration: Pedroni, Kao, Westerlund

**Robustness Checks**:

- Influence diagnostics (DFBETA, Cook's D, leverage)
- Outlier detection (standardized residuals, IQR)
- Jackknife resampling
- Panel cross-validation

**Report Generation**: HTML, Markdown, and LaTeX exporters with customizable templates.

**Datasets**: `load_grunfeld()`, `load_abdata()`, `list_datasets()`.

**Performance**: Numba JIT compilation for critical paths (up to 348x speedup).

---

## [0.4.0] — 2026-02-05

### Added

**Robust Standard Errors**:

- HC0–HC3 variants with leverage adjustment and caching
- One-way and two-way clustering with finite-sample corrections
- Driscoll-Kraay with 3 kernel options (Bartlett, Parzen, Quadratic Spectral)
- Newey-West HAC with automatic lag selection
- PCSE (Beck & Katz 1995) for contemporaneous cross-sectional correlation
- Integration: Fixed Effects supports 8 covariance types; Random Effects supports 7

---

## [0.3.0] — 2026-01-22

### Added

**Bootstrap Inference**:

- `PanelBootstrap` with 4 methods (pairs, wild, block, residual)
- Performance: ~95–110 iterations/second
- Confidence intervals: percentile, basic, studentized

**Sensitivity Analysis**:

- `SensitivityAnalysis` with leave-one-out entities, leave-one-out periods, and subset sensitivity
- Influential unit detection
- Optional matplotlib visualization

---

## [0.2.0] — 2026-01-21

### Added

**Dynamic Panel GMM**:

- Difference GMM (Arellano-Bond 1991) with one-step, two-step, and iterative estimation
- System GMM (Blundell-Bond 1998) with combined differenced and level equations
- Automatic instrument generation with collapse option
- Windmeijer (2005) finite-sample correction
- Smart instrument selection for unbalanced panels (72% retention vs. 0%)
- Specification tests: Hansen J, Sargan, AR(1), AR(2)

### Fixed

- Arellano-Bond validation now works with unbalanced panels
- Time dummies no longer cause 0% observation retention
- System GMM error handling improved

---

## [0.1.0] — 2025-12

### Added

**Core Framework**:

- `PanelData` — Panel data container with validation
- `FormulaParser` — R-style formula parsing (patsy integration)
- `PanelResults` — Base results class

**Static Models**:

- `PooledOLS`, `FixedEffects`, `RandomEffects`
- `HausmanTest` and `HausmanTestResult`

**Standard Errors**: Homoskedastic, heteroskedasticity-robust, clustered.

**Validation Framework**: Autocorrelation, heteroskedasticity, cross-sectional dependence, and unit root tests.

**Reporting**: HTML (Plotly), static (Matplotlib), LaTeX table export.

---

## Versioning Policy

PanelBox uses [Semantic Versioning](https://semver.org/):

| Component | When incremented |
|---|---|
| **Major** (X.0.0) | Incompatible API changes |
| **Minor** (0.X.0) | New features, backward compatible |
| **Patch** (0.0.X) | Bug fixes, backward compatible |

## Migration Notes

### v0.x → v1.0.0

No breaking changes. All pre-1.0 code continues to work. New features are additive:

```python
# v0.1.0 code still works
from panelbox import FixedEffects, RandomEffects

# v1.0.0 adds Panel VAR
from panelbox.var import PanelVAR, PanelVECM
```

## See Also

- [Contributing Guide](contributing.md) — How to contribute
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
