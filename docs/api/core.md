---
title: "Core API"
description: "API reference for panelbox.core — PanelData, PanelResults, serialization, and formula parsing"
---

# Core API Reference

!!! info "Module"
    **Import**: `from panelbox.core import PanelData, PanelResults, SerializableMixin`
    **Source**: `panelbox/core/`

## Overview

The core module provides the foundational infrastructure for all PanelBox models:

- **`PanelData`** — Container for panel (entity x time) data
- **`PanelResults`** — Base class for all estimation results
- **`SerializableMixin`** — Save/load models to disk
- **`FormulaParser`** / **`parse_formula`** — Parse R-style formulas
- **`load_model`** — Deserialize a saved model

## Classes

### PanelData

Container for panel data with entity and time structure. Wraps a `pandas.DataFrame` and provides panel-aware operations.

#### Constructor

```python
PanelData(data: pd.DataFrame, entity_col: str, time_col: str)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | DataFrame with panel data |
| `entity_col` | `str` | *required* | Column name identifying entities (firms, countries, individuals) |
| `time_col` | `str` | *required* | Column name identifying time periods (years, quarters) |

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | The underlying DataFrame |
| `entity_col` | `str` | Entity column name |
| `time_col` | `str` | Time column name |
| `entities` | `array` | Unique entity identifiers |
| `time_periods` | `array` | Unique time periods |
| `n_entities` | `int` | Number of entities (N) |
| `n_periods` | `int` | Number of time periods (T) |
| `is_balanced` | `bool` | Whether the panel is balanced (all entities observed in all periods) |

#### Example

```python
import pandas as pd
from panelbox.core import PanelData

df = pd.DataFrame({
    "firm": [1, 1, 2, 2, 3, 3],
    "year": [2020, 2021, 2020, 2021, 2020, 2021],
    "invest": [10.5, 12.3, 8.1, 9.4, 15.2, 16.8],
    "value": [100, 110, 80, 85, 150, 160],
})

panel = PanelData(df, entity_col="firm", time_col="year")
print(f"N={panel.n_entities}, T={panel.n_periods}, Balanced={panel.is_balanced}")
# N=3, T=2, Balanced=True
```

---

### PanelResults

Base result class returned by all estimation methods. Provides a unified interface for accessing coefficients, standard errors, test statistics, and model diagnostics.

#### Constructor

```python
PanelResults(
    params: pd.Series,
    std_errors: pd.Series,
    cov_params: pd.DataFrame,
    resid: np.ndarray,
    fittedvalues: np.ndarray,
    model_info: dict[str, Any],
    data_info: dict[str, Any],
    rsquared_dict: dict[str, float] | None = None,
    model: Any | None = None,
    formula_parser: Any | None = None,
)
```

!!! note
    Users typically do not construct `PanelResults` directly — it is returned by `model.fit()`.

#### Estimation Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | Estimated coefficients |
| `std_errors` | `pd.Series` | Standard errors |
| `tstats` | `pd.Series` | t-statistics (params / std_errors) |
| `pvalues` | `pd.Series` | Two-sided p-values |
| `cov_params` | `pd.DataFrame` | Variance-covariance matrix of coefficients |
| `resid` | `np.ndarray` | Residuals |
| `fittedvalues` | `np.ndarray` | Fitted values |

#### Model Fit Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `rsquared` | `float` | R-squared |
| `rsquared_adj` | `float` | Adjusted R-squared |
| `rsquared_within` | `float` | Within R-squared (FE models) |
| `rsquared_between` | `float` | Between R-squared |
| `rsquared_overall` | `float` | Overall R-squared |
| `fstat` | `float` | F-statistic |
| `nobs` | `int` | Number of observations |
| `n_entities` | `int` | Number of entities |
| `n_periods` | `int` | Number of time periods |
| `df_model` | `int` | Degrees of freedom (model) |
| `df_resid` | `int` | Degrees of freedom (residual) |

#### GMM-Specific Attributes

When results come from GMM estimation (`GMMResults`), additional attributes are available:

| Attribute | Type | Description |
|-----------|------|-------------|
| `hansen_j` | `TestResult` | Hansen J overidentification test |
| `sargan` | `TestResult` | Sargan overidentification test |
| `ar1_test` | `TestResult` | AR(1) serial correlation test |
| `ar2_test` | `TestResult` | AR(2) serial correlation test |
| `diff_hansen` | `TestResult` | Difference-in-Hansen test (System GMM) |
| `n_instruments` | `int` | Number of instruments |

#### Methods

##### `summary()`

Print a formatted summary table with coefficients, standard errors, t-statistics, p-values, and model fit statistics.

```python
result.summary()
```

##### `conf_int(alpha=0.05)`

Compute confidence intervals for all coefficients.

```python
ci = result.conf_int(alpha=0.05)  # 95% confidence intervals
```

**Returns**: `pd.DataFrame` with columns `['lower', 'upper']`.

##### `predict(X=None)`

Generate predictions from the estimated model.

```python
predictions = result.predict()       # In-sample predictions
predictions = result.predict(X_new)  # Out-of-sample predictions
```

##### `to_dict()`

Convert results to a dictionary for serialization.

```python
result_dict = result.to_dict()
```

##### `to_latex()`

Export results as a LaTeX table.

```python
latex_str = result.to_latex()
```

#### Example

```python
from panelbox import FixedEffects, load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
result = model.fit(cov_type="robust")

# Access coefficients
print(result.params)
print(result.pvalues)

# Confidence intervals
print(result.conf_int(alpha=0.01))  # 99% CI

# Model fit
print(f"R-squared = {result.rsquared:.4f}")
print(f"N = {result.nobs}, n = {result.n_entities}")

# Full summary
result.summary()
```

---

### SerializableMixin

Mixin class that adds save/load functionality to models and results.

#### Methods

##### `save(file_path)`

Save the model or result object to disk.

```python
result.save("my_model.pkl")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path where the object will be saved |

##### `load(file_path)` *(classmethod)*

Load a previously saved model or result.

```python
result = PanelResults.load("my_model.pkl")
```

---

### FormulaParser

Parse R-style formula strings into dependent and independent variable names.

#### Constructor

```python
FormulaParser(formula: str)
```

#### Example

```python
from panelbox.core import FormulaParser

parser = FormulaParser("y ~ x1 + x2 + x3")
print(parser.depvar)    # "y"
print(parser.exog_vars) # ["x1", "x2", "x3"]
```

## Functions

### parse_formula

```python
parse_formula(formula: str) -> tuple
```

Convenience function wrapping `FormulaParser`.

```python
from panelbox.core import parse_formula

depvar, exog = parse_formula("invest ~ value + capital")
```

### load_model

```python
load_model(file_path: str) -> Any
```

Load a serialized PanelBox model from disk. This is the recommended way to load saved models when you don't know the exact class.

```python
from panelbox.core import load_model

result = load_model("saved_model.pkl")
print(result.summary())
```

## See Also

- [Static Models API](static-models.md) — Models that return `PanelResults`
- [GMM API](gmm.md) — GMM models that return `GMMResults`
- [Tutorials: Fundamentals](../tutorials/fundamentals.md) — Getting started with PanelData
