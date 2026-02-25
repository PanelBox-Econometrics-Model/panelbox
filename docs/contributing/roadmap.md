---
title: "Roadmap"
description: "PanelBox development roadmap — planned features, priorities, and release schedule."
---

# Roadmap

PanelBox aims to be the most comprehensive Python library for panel data econometrics. This roadmap outlines planned features, organized by priority.

!!! info "Current Status"
    PanelBox v1.0.0 is the current stable release with 70+ models, 50+ diagnostic tests, and interactive HTML reports. Documentation Phase 6 (of 6) is in progress.

---

## High Priority

Features actively planned for the next releases.

### Additional Spatial Models

- **SLX (Spatial Lag of X)** — Exogenous spatial interaction model
- **Spatial HAC estimation** — Heteroskedasticity and autocorrelation consistent standard errors with spatial dependence
- **Higher-order spatial weights** — Multi-weight specifications (W1, W2)
- **Spatial panel threshold regression** — Regime switching with spatial dependence

### Bayesian Panel Methods

- **Bayesian Fixed/Random Effects** — MCMC estimation with conjugate priors
- **Bayesian GMM** — Posterior inference for dynamic panels
- **Model averaging** — Bayesian Model Averaging (BMA) for panel data
- **Hierarchical models** — Multi-level panel structures

### Longitudinal Data Methods

- **GEE (Generalized Estimating Equations)** — Population-averaged models
- **Mixed-effects models** — Random slopes and cross-level interactions
- **Survival panel models** — Duration analysis with panel structure
- **Growth curve models** — Latent trajectory modeling

### Additional Bootstrap Methods

- **Moving block bootstrap** — Enhanced temporal dependence handling
- **Sieve bootstrap** — AR(p) residual resampling
- **Parallel bootstrap** — Multiprocessing support for large panels
- **Double bootstrap** — Improved coverage for confidence intervals

---

## Medium Priority

Features planned for future minor releases.

### Panel Threshold Regression

- **Hansen (1999) threshold model** — Endogenous regime detection
- **Multiple thresholds** — Sequential testing for number of regimes
- **Dynamic threshold models** — Threshold effects with lagged dependent variables
- **Bootstrap inference** — Asymptotic refinements for threshold tests

### Functional Coefficient Models

- **Varying coefficient panel models** — Coefficients as smooth functions of covariates
- **Local polynomial estimation** — Bandwidth selection for panel data
- **Partially linear models** — Semi-parametric panel regression

### Machine Learning Integration

- **LASSO panel** — L1-regularized panel models for variable selection
- **Ridge panel** — L2-regularized panel estimation
- **Elastic net panel** — Combined L1/L2 regularization
- **Panel random forests** — Tree-based methods for panel prediction
- **Cross-validated tuning** — Panel-aware hyperparameter selection

### Stochastic Frontier Enhancements

- **Additional inefficiency distributions** — Gamma, truncated Rayleigh
- **Technology heterogeneity** — Latent class SFA
- **Environmental variables** — Inefficiency determinants (Battese-Coelli 1995)
- **Metafrontier analysis** — Cross-group technology comparison

---

## Long-Term Vision

Exploratory features for future major releases.

### GPU Acceleration

- **CuPy / JAX backends** — GPU-accelerated matrix operations for large panels (N > 10,000)
- **Automatic backend selection** — Transparent CPU/GPU switching
- **Batch estimation** — Parallel model fitting across specifications

### Distributed Computing

- **Dask integration** — Out-of-core panel data processing
- **Ray support** — Distributed bootstrap and cross-validation
- **Cloud-native workflows** — AWS/GCP/Azure integration for large-scale analysis

### Interactive Dashboard

- **Streamlit / Panel app** — Point-and-click panel data analysis
- **Model builder** — Visual specification of panel models
- **Real-time diagnostics** — Interactive test interpretation
- **Report designer** — Drag-and-drop report composition

### Cross-Platform Tools

- **Stata `.do` file translator** — Convert Stata panel data scripts to PanelBox Python code
- **R script converter** — Translate R `plm`/`pvar` scripts to PanelBox
- **SPSS syntax reader** — Import SPSS panel specifications

---

## Documentation Roadmap

### Completed

- [x] Phase 1: Documentation infrastructure and MkDocs setup
- [x] Phase 2: Getting Started and User Guide pages
- [x] Phase 3: Diagnostics, Inference, and Visualization guides
- [x] Phase 4: Tutorials and Theory pages

### In Progress

- [ ] Phase 5: API Reference, FAQ, and Benchmarks
- [ ] Phase 6: Contributing, Changelog, Roadmap, and Final Review

### Planned

- [ ] Maintenance: Regular updates as features are added
- [ ] Expansion: Community-contributed tutorials and case studies
- [ ] Translations: Spanish, Portuguese, Chinese documentation

---

## How to Influence the Roadmap

### Feature Requests

Open a [GitHub Issue](https://github.com/PanelBox-Econometrics-Model/panelbox/issues) with the `[Feature]` label. Include:

1. **Use case**: What problem does it solve?
2. **Description**: What should the feature do?
3. **References**: Academic papers, existing implementations (R, Stata)
4. **Priority justification**: Why is this important for panel data analysis?

### Community Voting

React with a thumbs-up on existing feature request issues to signal demand. Features with more community interest are prioritized higher.

### Contributions

The fastest way to get a feature is to implement it yourself! See the [Contributing Guide](contributing.md) for templates and process. We provide mentorship for first-time contributors.

### Sponsorship

For organizations that need specific features on a timeline, we welcome sponsorship discussions. Contact the team via [GitHub Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions).

---

## Release Schedule

### Versioning

PanelBox follows [Semantic Versioning](https://semver.org/):

| Version | Cadence | Content |
|---|---|---|
| **Major** (X.0.0) | As needed | Breaking API changes |
| **Minor** (0.X.0) | Every 2–3 months | New features, backward compatible |
| **Patch** (0.0.X) | As needed | Bug fixes, documentation updates |

### Release Process

1. Feature freeze 1 week before release
2. Release candidate published for testing
3. Final release after validation
4. Changelog and migration notes published

### Support Policy

- **Current major version**: Full support (bug fixes, security patches, new features)
- **Previous major version**: Security patches only for 6 months after new major release
- **Older versions**: Community support only

---

## See Also

- [Contributing Guide](contributing.md) — How to contribute code and documentation
- [Changelog](changelog.md) — Version history
- [Code of Conduct](code-of-conduct.md) — Community standards
- [API Reference](../api/index.md) — Full API documentation
