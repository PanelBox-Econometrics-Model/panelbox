---
title: User Guide
description: Comprehensive guides for all panel data econometric models in PanelBox
---

# User Guide

PanelBox provides **70+ econometric models** across 11 families, covering virtually every panel data method used in applied research. Each guide below introduces the model family, lists available estimators, and provides quick-start code examples.

All models follow a consistent API: define a formula, pass your data with entity and time identifiers, and call `.fit()`. Results objects provide `.summary()`, coefficient tables, diagnostic tests, and export to HTML/LaTeX.

<div class="grid cards" markdown>

- :material-chart-line: **[Static Models](static-models/index.md)**

    ---

    Pooled OLS, Fixed Effects, Random Effects, Between, First Difference

- :material-chart-timeline-variant: **[Dynamic Models (GMM)](gmm/index.md)**

    ---

    Arellano-Bond, Blundell-Bond, CUE-GMM, Bias-Corrected GMM

- :material-map-marker-radius: **[Spatial Econometrics](spatial/index.md)**

    ---

    SAR, SEM, SDM, Dynamic Spatial, General Nesting Spatial

- :material-trending-up: **[Stochastic Frontier](frontier/index.md)**

    ---

    SFA, Four-Component (unique in Python), TFP Decomposition

- :material-chart-scatter-plot: **[Quantile Regression](quantile/index.md)**

    ---

    Pooled, Fixed Effects, Canay, Location-Scale, Dynamic, QTE

- :material-vector-polyline: **[Panel VAR](var/index.md)**

    ---

    VAR, VECM, IRF, FEVD, Granger Causality, Forecast

- :material-toggle-switch: **[Discrete Choice](discrete/index.md)**

    ---

    Logit, Probit, FE Logit, RE Probit, Ordered, Multinomial, Conditional

- :material-counter: **[Count Data](count/index.md)**

    ---

    Poisson (Pooled/FE/RE/QML), NegBin, PPML, Zero-Inflated

- :material-content-cut: **[Censored & Selection](censored/index.md)**

    ---

    Tobit (Pooled/RE), Honore Trimmed, Panel Heckman

- :material-swap-horizontal: **[Instrumental Variables](iv/index.md)**

    ---

    Panel IV / 2SLS with first-stage diagnostics

</div>

## Choosing a Model Family

| Your Data | Recommended Family | Guide |
|-----------|-------------------|-------|
| Continuous outcome, no dynamics | [Static Models](static-models/index.md) | Pooled OLS, FE, RE |
| Lagged dependent variable | [Dynamic GMM](gmm/index.md) | Arellano-Bond, System GMM |
| Spatial dependence across units | [Spatial](spatial/index.md) | SAR, SEM, SDM |
| Efficiency / productivity analysis | [Stochastic Frontier](frontier/index.md) | SFA, Four-Component |
| Heterogeneous effects across distribution | [Quantile](quantile/index.md) | FE Quantile, Canay |
| Multiple interdependent outcomes | [Panel VAR](var/index.md) | VAR, VECM, IRF |
| Binary / ordered / multinomial outcome | [Discrete Choice](discrete/index.md) | Logit, Probit, Ordered |
| Count outcome (0, 1, 2, ...) | [Count Data](count/index.md) | Poisson, NegBin, PPML |
| Censored / truncated / selected sample | [Censored & Selection](censored/index.md) | Tobit, Heckman |
| Endogenous regressors | [Instrumental Variables](iv/index.md) | Panel 2SLS |

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="clustered")
print(results.summary())
```

## See Also

- [Getting Started](../getting-started/index.md) -- Installation and first steps
- [Tutorials](../tutorials/index.md) -- Interactive notebooks with Google Colab
- [Inference & Standard Errors](../inference/index.md) -- Choosing the right standard errors
- [Diagnostics & Validation](../diagnostics/index.md) -- Testing model assumptions
