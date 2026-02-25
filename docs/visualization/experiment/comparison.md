---
title: "Comparison Reports"
description: "Side-by-side model comparison with metrics, charts, and interactive reports in PanelBox"
---

# Comparison Reports

## Overview

Model comparison is essential for choosing the right specification. PanelBox's `ComparisonResult` computes fit statistics (R$^2$, AIC, BIC, RMSE) across all fitted models, identifies the best model by any criterion, and generates interactive HTML reports with coefficient plots, forest plots, and information criteria charts.

## Running a Comparison

### From PanelExperiment

```python
import panelbox as pb

data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# Fit models
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe')
exp.fit_model('random_effects', name='re')

# Compare all fitted models (default: all)
comp = exp.compare_models()

# Compare a subset
comp = exp.compare_models(model_names=['fe', 're'])
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_names` | `list[str]` or `None` | `None` | Models to compare. `None` compares all fitted models |

### From Model Results Directly

Create a `ComparisonResult` without a `PanelExperiment`:

```python
from panelbox.experiment.results import ComparisonResult

# Manually fitted models
ols = pb.PooledOLS("invest ~ value + capital", data, "firm", "year").fit()
fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re = pb.RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# Create ComparisonResult
comp = ComparisonResult(
    models={"Pooled OLS": ols, "Fixed Effects": fe, "Random Effects": re}
)
```

Or use the factory method:

```python
comp = ComparisonResult.from_experiment(exp, model_names=['ols', 'fe', 're'])
```

## Comparison Metrics

`ComparisonResult` automatically computes the following metrics for each model:

| Metric | Key | Description |
|--------|-----|-------------|
| R$^2$ | `rsquared` | Coefficient of determination |
| Adjusted R$^2$ | `rsquared_adj` | Adjusted for degrees of freedom |
| Within R$^2$ | `r2_within` | FE/RE within R$^2$ |
| Between R$^2$ | `r2_between` | Between R$^2$ |
| Overall R$^2$ | `r2_overall` | Overall R$^2$ |
| AIC | `aic` | Akaike Information Criterion |
| BIC | `bic` | Bayesian Information Criterion |
| RMSE | `rmse` | Root Mean Squared Error |
| F-statistic | `fvalue` | Overall F-test statistic |
| N | `nobs` | Number of observations |

!!! note "AIC/BIC Computation"
    When a model does not provide a log-likelihood directly, PanelBox computes AIC and BIC from the concentrated log-likelihood using the residual sum of squares: $\hat{\ell} = -\frac{n}{2}\left(1 + \log(2\pi) + \log(\hat{\sigma}^2)\right)$.

## Best Model Selection

Use `best_model()` to find the top-performing model by any metric:

```python
# Best by AIC (lower is better -- auto-detected)
best_name = comp.best_model('aic')
print(f"Best by AIC: {best_name}")

# Best by R-squared (higher is better -- auto-detected)
best_name = comp.best_model('rsquared')
print(f"Best by R-squared: {best_name}")

# Override direction explicitly
best_name = comp.best_model('rmse', prefer_lower=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | (required) | Metric key (e.g., `'aic'`, `'rsquared'`, `'bic'`, `'rmse'`) |
| `prefer_lower` | `bool` or `None` | `None` | `True` = lower is better, `False` = higher is better. `None` = auto-detect (AIC, BIC, RMSE, MAE default to lower-is-better) |

Returns the model **name** (`str`) or `None` if the metric is unavailable for all models.

## Comparison Table

Get a DataFrame with all metrics across models:

```python
df = comp.as_dataframe()
print(df[['rsquared', 'aic', 'bic', 'rmse']])
```

Output:

```text
                 rsquared       aic       bic     rmse
ols              0.8124     1423.5    1433.2    84.12
fe               0.7668     1321.8    1331.5    52.77
re               0.7657     1335.2    1344.9    55.48
```

## Output Options

### Text Summary

```python
print(comp.summary())
```

Output:

```text
================================================================================
MODEL COMPARISON SUMMARY
================================================================================

Models Compared: 3
Comparison Date: 2026-02-25 14:30:00

Comparison Metrics:
--------------------------------------------------------------------------------
Model                     R²         R² Adj     AIC          BIC
--------------------------------------------------------------------------------
ols                       0.8124     0.8104     1423.50      1433.20
fe                        0.7668     0.7644     1321.80      1331.50
re                        0.7657     0.7633     1335.20      1344.90
--------------------------------------------------------------------------------

Best Models by Metric:
--------------------------------------------------------------------------------
  * Highest R²: ols
  * Lowest AIC: fe
  * Lowest BIC: fe

================================================================================
```

### Interactive HTML Report

```python
comp.save_html(
    "comparison_report.html",
    test_type="comparison",
    theme="professional",
    title="Model Comparison Report"
)
```

The HTML report includes four interactive charts:

1. **Coefficient comparison** -- side-by-side bar chart of parameter estimates
2. **Forest plot** -- coefficient estimates with confidence intervals
3. **Fit comparison** -- R$^2$ and adjusted R$^2$ across models
4. **Information criteria comparison** -- AIC and BIC bar chart

### JSON Export

```python
comp.save_json("comparison_results.json", indent=2)
```

### Python Dictionary

```python
data = comp.to_dict()
# Keys: 'models', 'comparison_data', 'comparison_metrics', 'charts', 'summary'
```

## Additional Properties

```python
comp.model_names    # list[str] -- names of all models
comp.n_models       # int -- number of models compared
```

## Complete Example

```python
import panelbox as pb

# 1. Set up experiment
data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Fit multiple models
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe')
exp.fit_model('random_effects', name='re')

# 3. Compare all models
comp = exp.compare_models()

# 4. Identify best model
best_aic = comp.best_model('aic')
best_bic = comp.best_model('bic')
best_r2 = comp.best_model('rsquared')

print(f"Best by AIC: {best_aic}")
print(f"Best by BIC: {best_bic}")
print(f"Best by R-squared: {best_r2}")

# 5. Get comparison table
df = comp.as_dataframe()
print(df[['rsquared', 'aic', 'bic']].to_string())

# 6. Generate HTML report
comp.save_html(
    "comparison.html",
    test_type="comparison",
    theme="professional",
    title="Grunfeld Model Comparison"
)

# 7. Archive
comp.save_json("comparison.json")
```

## Comparison with Other Software

| Task | PanelBox | Stata | R |
|------|----------|-------|---|
| Compare models | `exp.compare_models()` | `estimates table` | `modelsummary::msummary()` |
| Best by AIC | `comp.best_model('aic')` | Manual inspection | `AIC(m1, m2, m3)` |
| Comparison table | `comp.as_dataframe()` | `estout` | `modelsummary()` |
| Interactive report | `comp.save_html(...)` | Not built-in | `rmarkdown` (manual) |

## See Also

- [Experiment Overview](index.md) -- Pattern overview and quick start
- [Workflow](fitting.md) -- Fitting and managing models
- [Validation Reports](validation.md) -- Diagnostic testing
- [Residual Analysis](residuals.md) -- Residual diagnostics
- [Master Reports](master-reports.md) -- Combined report generation
