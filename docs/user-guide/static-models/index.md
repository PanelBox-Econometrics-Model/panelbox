---
title: Static Panel Models
description: Guide to static panel data models in PanelBox - Pooled OLS, Fixed Effects, Random Effects, Between, and First Difference estimators.
---

# Static Panel Models

Static panel models are the foundation of panel data econometrics. They exploit the panel structure of data -- repeated observations on the same entities over time -- to control for unobserved heterogeneity, estimate causal effects, and decompose variation into within-entity and between-entity components.

PanelBox provides five static estimators that cover the standard approaches in applied research. Each produces consistent, publication-ready results with flexible standard error options and built-in diagnostic tests.

The choice between estimators depends on assumptions about unobserved individual effects. The Hausman test or Mundlak test guides this decision (see [Specification Tests](../../diagnostics/specification/index.md)).

## Available Models

| Model | Class | Stata Equivalent | When to Use |
|-------|-------|-----------------|-------------|
| Pooled OLS | `PooledOLS` | `reg` | Baseline; assumes no entity heterogeneity |
| Fixed Effects | `FixedEffects` | `xtreg, fe` | Entity effects correlated with regressors |
| Random Effects | `RandomEffects` | `xtreg, re` | Entity effects uncorrelated with regressors |
| Between Estimator | `BetweenEstimator` | `xtreg, be` | Exploiting cross-sectional variation only |
| First Difference | `FirstDifferenceEstimator` | `reg D.y D.x` | Alternative to FE; robust to serial correlation |

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Fixed Effects with entity-clustered standard errors
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit(cov_type="clustered")
print(results.summary())
```

## Model Comparison

```python
from panelbox import PooledOLS, FixedEffects, RandomEffects
from panelbox.experiment import PanelExperiment
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Compare all static models at once
exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")
exp.fit_all_models(["pooled", "fe", "re"])
comparison = exp.compare_models(["pooled", "fe", "re"])
print(comparison.summary())
```

## Key Concepts

### Fixed vs. Random Effects

The central question in static panel modeling is whether unobserved individual effects $\alpha_i$ are correlated with the regressors $X_{it}$:

- **Fixed Effects**: Allows $\text{Corr}(\alpha_i, X_{it}) \neq 0$. Identifies coefficients from within-entity variation only.
- **Random Effects**: Assumes $\text{Corr}(\alpha_i, X_{it}) = 0$. More efficient when assumption holds.

!!! tip "Decision rule"
    Run the Hausman test: if it rejects (p < 0.05), use Fixed Effects. If it does not reject, Random Effects is more efficient.

### Standard Error Options

All static models support the `cov_type` parameter:

| `cov_type` | Description |
|------------|-------------|
| `"nonrobust"` | Classical (homoskedastic) |
| `"robust"` | HC1 heteroskedasticity-robust |
| `"clustered"` | Entity-clustered (recommended default) |
| `"twoway"` | Two-way clustered (entity + time) |
| `"driscoll_kraay"` | Driscoll-Kraay (cross-sectional dependence) |
| `"newey_west"` | Newey-West HAC |

## Detailed Guides

- [Pooled OLS](pooled-ols.md) -- Baseline estimator with no entity effects *(detailed guide coming soon)*
- [Fixed Effects](fixed-effects.md) -- Within estimator with entity demeaning *(detailed guide coming soon)*
- [Random Effects](random-effects.md) -- GLS estimator with quasi-demeaning *(detailed guide coming soon)*
- [Between Estimator](between.md) -- Cross-sectional regression on entity means *(detailed guide coming soon)*
- [First Difference](first-difference.md) -- Differenced estimator for serial correlation *(detailed guide coming soon)*

## Tutorials

See [Static Models Tutorial](../../tutorials/static-models.md) for interactive notebooks with Google Colab.

## API Reference

See [Static Models API](../../api/static-models.md) for complete technical reference.

## References

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
