---
title: Instrumental Variables
description: Guide to Panel IV / 2SLS estimation in PanelBox for addressing endogeneity in panel data models.
---

# Instrumental Variables

Instrumental Variables (IV) estimation addresses the fundamental problem of **endogeneity** -- when a regressor is correlated with the error term due to omitted variables, measurement error, or simultaneity. Even fixed effects cannot solve endogeneity caused by time-varying confounders. IV/2SLS uses external instruments that are correlated with the endogenous regressor but uncorrelated with the error term to produce consistent estimates.

PanelBox provides a Panel IV estimator with first-stage diagnostics, instrument validity tests, and flexible standard error options.

## Available Models

| Model | Class | Method | Key Feature |
|-------|-------|--------|-------------|
| Panel IV | `PanelIV` | 2SLS | Two-Stage Least Squares with panel structure |

## Quick Example

```python
from panelbox.models.iv import PanelIV
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

model = PanelIV(
    "invest ~ value + capital",
    data, "firm", "year",
    endogenous=["value"],
    instruments=["L.value", "L2.value"]
)
results = model.fit(cov_type="clustered")
print(results.summary())
```

## Key Concepts

### The Endogeneity Problem

A regressor $X$ is endogenous when $\text{Cov}(X_{it}, \epsilon_{it}) \neq 0$. Sources include:

| Source | Example | Fix |
|--------|---------|-----|
| Omitted variables | Ability affects both education and wages | IV with external instrument |
| Simultaneity | Price and quantity determined jointly | Supply/demand shifters |
| Measurement error | True $X^*$ measured with noise | IV with alternative measure |
| Reverse causality | Does $X \to Y$ or $Y \to X$? | IV with predetermined variable |

### Instrument Requirements

A valid instrument $Z$ must satisfy two conditions:

1. **Relevance**: $\text{Cov}(Z_{it}, X_{it}) \neq 0$ -- the instrument must be correlated with the endogenous regressor
2. **Exogeneity**: $\text{Cov}(Z_{it}, \epsilon_{it}) = 0$ -- the instrument must be uncorrelated with the error

!!! warning "Weak instruments"
    If the instrument is only weakly correlated with the endogenous regressor (relevance is marginal), IV estimates are biased toward OLS and have very large standard errors. Check the first-stage F-statistic: a value above 10 is the conventional rule of thumb (Stock & Yogo, 2005).

### Two-Stage Least Squares (2SLS)

The 2SLS procedure:

1. **First stage**: Regress endogenous $X$ on instruments $Z$ and exogenous controls. Obtain predicted values $\hat{X}$.
2. **Second stage**: Regress outcome $Y$ on $\hat{X}$ and exogenous controls. Standard errors are adjusted for the two-stage procedure.

### First-Stage Diagnostics

```python
results = model.fit(cov_type="clustered")

# First-stage F-statistic (instrument strength)
print(f"First-stage F: {results.first_stage_f:.2f}")

# Partial R-squared of excluded instruments
print(f"Partial R²: {results.partial_r_squared:.4f}")
```

| Diagnostic | Threshold | Interpretation |
|-----------|-----------|----------------|
| First-stage F | > 10 | Instruments are sufficiently strong |
| Partial R-squared | > 0.10 | Instruments explain meaningful variation |

### Overidentification Test

When you have more instruments than endogenous variables, test whether all instruments are valid:

```python
# Sargan/Hansen J test (if overidentified)
if results.overid_test is not None:
    print(f"Sargan test: p = {results.overid_test.pvalue:.4f}")
    # p > 0.05: instruments are valid (do not reject)
```

### Standard Error Options

Panel IV supports the same `cov_type` options as static models:

```python
results = model.fit(cov_type="clustered")     # Recommended
results = model.fit(cov_type="robust")         # HC robust
results = model.fit(cov_type="driscoll_kraay") # Cross-sectional dependence
```

## When to Use IV vs. GMM

| Feature | Panel IV (2SLS) | GMM (Arellano-Bond) |
|---------|----------------|---------------------|
| Endogenous variables | External instruments available | Uses internal (lagged) instruments |
| Dynamic model | No lagged dependent variable | Designed for lagged $y$ |
| Instrument source | Researcher-specified | Automatically generated from lags |
| When to use | Known external instruments | Dynamic panels, no external instruments |

!!! tip "Rule of thumb"
    Use Panel IV when you have strong external instruments. Use GMM when the endogeneity comes from including a lagged dependent variable and you lack external instruments.

## Detailed Guides

- [Panel 2SLS](panel-iv.md) -- Two-Stage Least Squares estimation *(detailed guide coming soon)*
- [Instrument Selection](diagnostics.md) -- Choosing and testing instruments *(detailed guide coming soon)*

## Tutorials

See [Static Models Tutorial](../../tutorials/static-models.md) for IV examples alongside OLS, FE, and RE models.

## API Reference

See [IV API](../../api/iv.md) for complete technical reference.

## References

- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and Inference for Econometric Models* (pp. 80-108). Cambridge University Press.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
