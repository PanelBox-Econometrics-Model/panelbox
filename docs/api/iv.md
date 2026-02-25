---
title: "IV API"
description: "API reference for panelbox.models.iv — Panel Instrumental Variables / 2SLS"
---

# Instrumental Variables API Reference

!!! info "Module"
    **Import**: `from panelbox.models.iv import PanelIV`
    **Source**: `panelbox/models/iv/`

## Overview

The IV module provides Two-Stage Least Squares (2SLS) estimation for panel data with endogenous regressors. Use when one or more independent variables are correlated with the error term due to omitted variables, measurement error, or simultaneity.

## Classes

### PanelIV

Panel data instrumental variables estimator with support for pooled, fixed effects, and random effects specifications.

#### Constructor

```python
PanelIV(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    model_type: str = "pooled",
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | IV formula with instruments specified using `\|`, e.g. `"y ~ x1 + [x2 ~ z1 + z2]"` |
| `data` | `pd.DataFrame` | *required* | Panel data |
| `entity_col` | `str` | *required* | Entity column |
| `time_col` | `str` | *required* | Time column |
| `model_type` | `str` | `"pooled"` | Model type: `"pooled"`, `"fe"`, `"re"` |
| `weights` | `np.ndarray \| None` | `None` | Observation weights |

#### Methods

##### `.fit()`

```python
def fit(self, cov_type: str = "nonrobust", **cov_kwds) -> PanelResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | `str` | `"nonrobust"` | Covariance type: `"nonrobust"`, `"robust"`, `"clustered"` |

**Returns**: [`PanelResults`](core.md#panelresults) with additional IV diagnostics.

#### Example

```python
from panelbox.models.iv import PanelIV

# Wage equation: education is endogenous, instrumented by parents' education
iv = PanelIV(
    formula="log_wage ~ experience + [education ~ father_educ + mother_educ]",
    data=df,
    entity_col="person_id",
    time_col="year",
    model_type="fe",
)
result = iv.fit(cov_type="clustered")
result.summary()
```

---

## IV Diagnostics

The IV result object provides diagnostic tests accessible through the result attributes:

### First-Stage F-Statistic

Tests instrument relevance. A weak instruments problem exists when F < 10 (Stock-Yogo rule of thumb).

```python
print(f"First-stage F: {result.first_stage_f:.2f}")
```

### Sargan-Hansen Overidentification Test

Tests instrument validity (H0: instruments are exogenous). Only available when the model is overidentified (more instruments than endogenous variables).

```python
print(f"Sargan test: stat={result.sargan_stat:.3f}, p={result.sargan_pvalue:.3f}")
```

### Wu-Hausman Endogeneity Test

Tests whether the suspected endogenous variable is actually endogenous (H0: variable is exogenous). If you cannot reject, OLS is preferred.

```python
print(f"Wu-Hausman: stat={result.wu_hausman_stat:.3f}, p={result.wu_hausman_pvalue:.3f}")
```

---

## When to Use IV

| Problem | Description | Example |
|---------|-------------|---------|
| **Omitted variables** | Unobserved confounders correlated with X | Ability bias in returns to education |
| **Measurement error** | X is measured with error | Self-reported income |
| **Simultaneity** | Y affects X and X affects Y | Price and quantity in supply/demand |

## Instrument Requirements

Good instruments must satisfy:

1. **Relevance**: Correlated with the endogenous variable (testable via first-stage F)
2. **Exogeneity**: Uncorrelated with the error term (partially testable via Sargan-Hansen)

!!! warning "Weak instruments"
    Weak instruments (first-stage F < 10) lead to biased and unreliable 2SLS estimates. In severe cases, 2SLS can be more biased than OLS. Consider using GMM-based estimators or finding stronger instruments.

---

## Complete IV Workflow

```python
from panelbox.models.iv import PanelIV

# Step 1: Estimate IV model
iv = PanelIV(
    formula="log_wage ~ experience + [education ~ father_educ + mother_educ]",
    data=df,
    entity_col="person_id",
    time_col="year",
    model_type="fe",
)
result = iv.fit(cov_type="robust")

# Step 2: Check instrument relevance (first-stage F > 10)
print(f"First-stage F: {result.first_stage_f:.2f}")

# Step 3: Test overidentifying restrictions (p > 0.05 means valid)
print(f"Sargan: p={result.sargan_pvalue:.3f}")

# Step 4: Test endogeneity (p < 0.05 means IV needed)
print(f"Wu-Hausman: p={result.wu_hausman_pvalue:.3f}")

# Step 5: View results
result.summary()
```

## See Also

- [Static Models API](static-models.md) — OLS, FE, RE without instruments
- [GMM API](gmm.md) — GMM for dynamic panels (alternative to IV)
- [Tutorials: Static Models](../tutorials/static-models.md) — When to use IV vs standard models
