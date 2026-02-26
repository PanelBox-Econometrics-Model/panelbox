---
title: PanelBox - Panel Data Econometrics for Python
description: Complete Python library for panel data econometrics with 70+ models, 50+ diagnostic tests, and interactive HTML reports
---

# PanelBox

**The complete Python toolkit for panel data econometrics.**

[![CI](https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml/badge.svg)](https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/branch/main/graph/badge.svg)](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox)
[![PyPI](https://img.shields.io/pypi/v/panelbox)](https://pypi.org/project/panelbox/)
[![Python](https://img.shields.io/pypi/pyversions/panelbox)](https://pypi.org/project/panelbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/development%20status-beta-orange)

70+ econometric models | 50+ diagnostic tests | Interactive HTML reports | Google Colab tutorials

---

## Quick Start

=== "Fixed Effects (3 lines)"

    ```python
    from panelbox import FixedEffects
    model = FixedEffects("invest ~ value + capital", data, "firm", "year")
    print(model.fit(cov_type="clustered").summary())
    ```

=== "System GMM (5 lines)"

    ```python
    from panelbox.gmm import SystemGMM
    model = SystemGMM("n ~ L.n + w + k", data, "id", "year",
                       gmm_instruments=["L.n"], iv_instruments=["w", "k"])
    results = model.fit(two_step=True)
    print(results.summary())  # Includes Hansen J, AR(1)/AR(2) tests
    ```

=== "Full Experiment (6 lines)"

    ```python
    from panelbox.experiment import PanelExperiment
    exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")
    exp.fit_all_models(["pooled", "fe", "re"])
    validation = exp.validate_model("fe")  # 15+ automatic tests
    comparison = exp.compare_models(["pooled", "fe", "re"])
    exp.save_master_report("analysis.html")  # Interactive HTML report
    ```

---

## What's Inside

<div class="grid cards" markdown>

-   :material-chart-line: **Static Models**

    ---

    Pooled OLS, Fixed Effects, Random Effects, Between, First Difference

    [:octicons-arrow-right-24: User Guide](user-guide/static-models/index.md)

-   :material-chart-timeline-variant: **Dynamic GMM**

    ---

    Arellano-Bond, Blundell-Bond, CUE-GMM, Bias-Corrected

    [:octicons-arrow-right-24: User Guide](user-guide/gmm/index.md)

-   :material-map-marker-radius: **Spatial Econometrics**

    ---

    SAR, SEM, SDM, Dynamic Spatial, General Nesting Spatial

    [:octicons-arrow-right-24: User Guide](user-guide/spatial/index.md)

-   :material-factory: **Stochastic Frontier**

    ---

    Production/Cost frontiers, Four-Component (unique in Python), TFP

    [:octicons-arrow-right-24: User Guide](user-guide/frontier/index.md)

-   :material-chart-scatter-plot: **Quantile Regression**

    ---

    Pooled, FE, Canay Two-Step, Location-Scale, Dynamic, Treatment Effects

    [:octicons-arrow-right-24: User Guide](user-guide/quantile/index.md)

-   :material-swap-horizontal: **Panel VAR**

    ---

    VAR, VECM, Impulse Response, FEVD, Granger Causality, Forecasting

    [:octicons-arrow-right-24: User Guide](user-guide/var/index.md)

-   :material-toggle-switch: **Discrete Choice**

    ---

    Logit/Probit, FE Logit, RE Probit, Ordered, Multinomial, Dynamic

    [:octicons-arrow-right-24: User Guide](user-guide/discrete/index.md)

-   :material-counter: **Count Data**

    ---

    Poisson, PPML, Negative Binomial, Zero-Inflated

    [:octicons-arrow-right-24: User Guide](user-guide/count/index.md)

-   :material-content-cut: **Censored & Selection**

    ---

    Tobit, Honore, Panel Heckman (Wooldridge 1995)

    [:octicons-arrow-right-24: User Guide](user-guide/censored/index.md)

-   :material-vector-line: **Instrumental Variables**

    ---

    Panel IV/2SLS with first-stage diagnostics

    [:octicons-arrow-right-24: User Guide](user-guide/iv/index.md)

-   :material-shield-check: **Standard Errors**

    ---

    HC0-HC3, Clustered, Driscoll-Kraay, Newey-West, PCSE, Spatial HAC

    [:octicons-arrow-right-24: Inference Guide](inference/index.md)

-   :material-test-tube: **50+ Diagnostic Tests**

    ---

    Specification, serial correlation, heteroskedasticity, unit root, cointegration

    [:octicons-arrow-right-24: Diagnostics Guide](diagnostics/index.md)

-   :material-chart-bar: **Visualization & Reports**

    ---

    35+ Plotly charts, HTML/LaTeX/Markdown reports, Master Reports

    [:octicons-arrow-right-24: Visualization Guide](visualization/charts/index.md)

</div>

---

## PanelBox vs. Stata, R, and linearmodels

| Feature | PanelBox | Stata | R (plm/splm) | linearmodels |
|:--------|:--------:|:-----:|:------------:|:------------:|
| Static Models (FE/RE) | :white_check_mark: | :white_check_mark: xtreg | :white_check_mark: plm | :white_check_mark: |
| Difference GMM | :white_check_mark: | :white_check_mark: xtabond2 | :white_check_mark: pgmm | :white_check_mark: |
| System GMM | :white_check_mark: | :white_check_mark: xtabond2 | :white_check_mark: pgmm | :white_check_mark: |
| CUE-GMM | :white_check_mark: | :x: | :x: | :x: |
| Spatial Models (SAR/SEM/SDM) | :white_check_mark: | :white_check_mark: spxtregress | :white_check_mark: splm | :x: |
| Dynamic Spatial | :white_check_mark: | :x: | :x: | :x: |
| Four-Component SFA | :white_check_mark: | :x: | :x: | :x: |
| Quantile FE (Canay) | :white_check_mark: | :x: | :white_check_mark: quantreg | :x: |
| Panel VAR/VECM | :white_check_mark: | :white_check_mark: pvar | :white_check_mark: panelvar | :x: |
| Interactive HTML Reports | :white_check_mark: | :x: | :x: | :x: |
| Experiment Pattern | :white_check_mark: | :x: | :x: | :x: |
| Google Colab Tutorials | :white_check_mark: 100+ | :x: | :x: | :x: |

---

## Installation

```bash
pip install panelbox
```

With optional extras:

```bash
pip install panelbox[dev]     # Development tools
pip install panelbox[docs]    # Documentation tools
pip install panelbox[test]    # Testing tools
```

See the [Installation Guide](getting-started/installation.md) for detailed instructions.

---

## Explore by Topic

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Install and run your first model in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant: **User Guide**

    ---

    Comprehensive guides for all 13 model families

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-test-tube: **Diagnostics**

    ---

    50+ validation and diagnostic tests

    [:octicons-arrow-right-24: Diagnostics](diagnostics/index.md)

-   :material-notebook: **Tutorials**

    ---

    100+ interactive notebooks with Google Colab

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-code-tags: **API Reference**

    ---

    Complete technical reference for all classes and functions

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-sigma: **Theory**

    ---

    Mathematical foundations and econometric background

    [:octicons-arrow-right-24: Theory](theory/panel-fundamentals.md)

</div>

---

## Library Metrics

| Metric | Value |
|:-------|------:|
| Lines of Code | 127,309 |
| Models | 70+ |
| Tests | 3,986 |
| Coverage | 85-92% |
| Diagnostic Tests | 50+ |
| Interactive Charts | 35+ |
| Tutorial Notebooks | 100+ |

---

## Citation

If you use PanelBox in academic research, please cite:

```bibtex
@software{panelbox2026,
  title = {PanelBox: Panel Data Econometrics for Python},
  author = {PanelBox Development Team},
  year = {2026},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox},
  version = {0.6.1}
}
```

---

## Design Philosophy

PanelBox is built on four principles:

- **Ease of Use** -- R-style formulas and a pandas-friendly API let you go from data to results in three lines of code.
- **Academic Rigor** -- Every estimator follows published econometrics papers and is cross-validated against Stata and R.
- **Performance** -- Numba-optimized critical paths deliver up to 348x speedups on large panels.
- **Publication-Ready Output** -- LaTeX tables, interactive HTML reports, and Plotly visualizations are built in, not bolted on.
