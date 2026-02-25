---
title: "Count Models API"
description: "API reference for panelbox.models.count — Poisson, Negative Binomial, PPML, Zero-Inflated"
---

# Count Models API Reference

!!! info "Module"
    **Import**: `from panelbox.models.count import PooledPoisson, NegativeBinomial, PPML, ZeroInflatedPoisson`
    **Source**: `panelbox/models/count/`

## Overview

The count module provides models for non-negative integer-valued dependent variables (event counts, trade flows):

| Model | Description | Reference |
|-------|-------------|-----------|
| `PooledPoisson` | Pooled Poisson with cluster-robust SE | — |
| `PoissonFixedEffects` | Conditional MLE | Hausman, Hall & Griliches (1984) |
| `RandomEffectsPoisson` | RE with Gamma/Normal mixing | — |
| `PoissonQML` | Quasi-Maximum Likelihood | Wooldridge (1999) |
| `NegativeBinomial` | NB2 for overdispersed counts | — |
| `FixedEffectsNegativeBinomial` | FE NB model | Allison & Waterman (2002) |
| `PPML` | Poisson Pseudo-ML for gravity models | Santos Silva & Tenreyro (2006) |
| `ZeroInflatedPoisson` | ZIP for excess zeros | Lambert (1992) |
| `ZeroInflatedNegativeBinomial` | ZINB for excess zeros + overdispersion | — |

## Classes

### PooledPoisson

Pooled Poisson regression with cluster-robust standard errors.

#### Constructor

```python
PooledPoisson(
    endog,
    exog,
    entity_id=None,
    time_id=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray \| pd.Series` | *required* | Count dependent variable |
| `exog` | `np.ndarray \| pd.DataFrame` | *required* | Independent variables |
| `entity_id` | `np.ndarray \| pd.Series \| None` | `None` | Entity identifiers |
| `time_id` | `np.ndarray \| pd.Series \| None` | `None` | Time identifiers |

#### Methods

- `.fit()` — Estimate model
- `.predict(X=None, type="response")` — Predictions (`"response"` or `"linear"`)

#### Example

```python
from panelbox.models.count import PooledPoisson

model = PooledPoisson(
    endog=df["patents"],
    exog=df[["rd_spending", "firm_size", "age"]],
    entity_id=df["firm_id"],
)
result = model.fit()
result.summary()
```

---

### PoissonFixedEffects

Conditional MLE Poisson with entity fixed effects (Hausman, Hall & Griliches 1984). Conditions out entity effects, providing consistent estimates even with many entities.

```python
PoissonFixedEffects(endog, exog, entity_id=None, time_id=None)
```

---

### RandomEffectsPoisson

Random effects Poisson with Gamma or Normal mixing distribution.

```python
RandomEffectsPoisson(endog, exog, entity_id=None, time_id=None)
```

---

### PoissonQML

Quasi-Maximum Likelihood Poisson (Wooldridge 1999). Consistent under misspecification of the conditional variance — only requires correct specification of the conditional mean E[y|X] = exp(X*beta).

```python
PoissonQML(endog, exog, entity_id=None, time_id=None)
```

!!! tip "When to use QMLE"
    PoissonQML is robust to overdispersion and underdispersion. Use it when you want Poisson-like estimates without assuming equidispersion.

---

### NegativeBinomial

NB2 Negative Binomial model for overdispersed count data.

#### Constructor

```python
NegativeBinomial(
    endog: np.ndarray | pd.Series | pd.DataFrame,
    exog: np.ndarray | pd.DataFrame,
    entity_id: np.ndarray | pd.Series | None = None,
    time_id: np.ndarray | pd.Series | None = None,
    weights: np.ndarray | pd.Series | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray \| pd.Series` | *required* | Count dependent variable |
| `exog` | `np.ndarray \| pd.DataFrame` | *required* | Independent variables |
| `entity_id` | `np.ndarray \| pd.Series \| None` | `None` | Entity identifiers |
| `time_id` | `np.ndarray \| pd.Series \| None` | `None` | Time identifiers |
| `weights` | `np.ndarray \| pd.Series \| None` | `None` | Observation weights |

#### Example

```python
from panelbox.models.count import NegativeBinomial

nb = NegativeBinomial(
    endog=df["doctor_visits"],
    exog=df[["age", "income", "chronic_conditions"]],
    entity_id=df["patient_id"],
)
result = nb.fit()
print(f"Overdispersion alpha: {result.alpha:.4f}")
```

---

### FixedEffectsNegativeBinomial

Fixed effects NB model (Allison & Waterman 2002).

```python
FixedEffectsNegativeBinomial(endog, exog, entity_id=None, time_id=None)
```

---

### PPML

Poisson Pseudo-Maximum Likelihood for gravity models (Santos Silva & Tenreyro 2006). Handles zeros in trade data naturally and is robust to heteroskedasticity.

#### Constructor

```python
PPML(endog, exog, entity_id=None, time_id=None)
```

#### Example

```python
from panelbox.models.count import PPML

ppml = PPML(
    endog=df["trade_flow"],
    exog=df[["log_gdp_origin", "log_gdp_dest", "log_distance", "common_border"]],
    entity_id=df["pair_id"],
)
result = ppml.fit()
result.summary()
```

### PPMLResult

Specialized result container for PPML with gravity-model methods.

#### Methods

##### `.elasticity(variable)`

Compute elasticity for a given variable.

```python
elas = result.elasticity("log_gdp_origin")
# Returns: {'elasticity': 0.95, 'se': 0.12, 'ci_lower': 0.72, 'ci_upper': 1.18}
```

##### `.elasticities()`

Compute elasticities for all variables.

```python
elas_df = result.elasticities()  # Returns pd.DataFrame
```

##### `.semi_elasticity(variable)`

Compute semi-elasticity for dummy/categorical variables.

##### `.compare_with_ols(ols_result)`

Compare PPML estimates with log-linear OLS to quantify heteroskedasticity bias.

!!! tip "Why PPML for gravity models?"
    - Handles **zeros** in trade flows (log-linear OLS drops them)
    - Consistent under **heteroskedasticity** (Santos Silva & Tenreyro 2006)
    - Coefficients are **elasticities** when variables are in logs
    - Cluster-robust SE applied automatically

---

### ZeroInflatedPoisson

Zero-Inflated Poisson (ZIP) for data with excess zeros. Models zeros as coming from two processes: a point mass at zero (inflation) and a Poisson count process.

#### Constructor

```python
ZeroInflatedPoisson(
    endog: np.ndarray | pd.Series,
    exog_count: np.ndarray | pd.DataFrame,
    exog_inflate: np.ndarray | pd.DataFrame | None = None,
    exog_count_names: list | None = None,
    exog_inflate_names: list | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `np.ndarray \| pd.Series` | *required* | Count dependent variable |
| `exog_count` | `np.ndarray \| pd.DataFrame` | *required* | Variables for the count equation |
| `exog_inflate` | `np.ndarray \| pd.DataFrame \| None` | `None` | Variables for the inflation equation (defaults to count vars) |
| `exog_count_names` | `list \| None` | `None` | Variable names for count equation |
| `exog_inflate_names` | `list \| None` | `None` | Variable names for inflation equation |

#### Example

```python
from panelbox.models.count import ZeroInflatedPoisson

zip_model = ZeroInflatedPoisson(
    endog=df["doctor_visits"],
    exog_count=df[["age", "income", "chronic"]],
    exog_inflate=df[["age", "insurance"]],
)
result = zip_model.fit()
result.summary()
```

---

### ZeroInflatedNegativeBinomial

Zero-Inflated Negative Binomial for excess zeros with overdispersion.

```python
ZeroInflatedNegativeBinomial(
    endog, exog_count, exog_inflate=None,
    exog_count_names=None, exog_inflate_names=None,
)
```

---

## Result Classes

### ZeroInflatedPoissonResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `count_params` | `pd.Series` | Count equation coefficients |
| `inflate_params` | `pd.Series` | Inflation equation coefficients |
| `loglik` | `float` | Log-likelihood |
| `vuong_stat` | `float` | Vuong test statistic (ZIP vs Poisson) |

### ZeroInflatedNegativeBinomialResult

Same structure with additional overdispersion parameter `alpha`.

## Model Selection Guide

| Data Characteristics | Recommended Model |
|---------------------|-------------------|
| Equidispersed counts | `PooledPoisson` or `PoissonFixedEffects` |
| Overdispersed counts | `NegativeBinomial` |
| Excess zeros | `ZeroInflatedPoisson` |
| Excess zeros + overdispersion | `ZeroInflatedNegativeBinomial` |
| Trade/gravity data | `PPML` |
| Robust to variance misspecification | `PoissonQML` |
| Entity heterogeneity | `PoissonFixedEffects` or `FixedEffectsNegativeBinomial` |

## See Also

- [Discrete Choice API](discrete.md) — Binary and multinomial outcomes
- [Marginal Effects API](marginal-effects.md) — AME for nonlinear models
- [Tutorials: Count](../tutorials/count.md) — Step-by-step count models guide
