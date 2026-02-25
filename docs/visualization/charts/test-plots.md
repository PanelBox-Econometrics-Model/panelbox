---
title: "Econometric Test Plots"
description: "Visualize diagnostic test results including ACF/PACF, unit root tests, cointegration, and validation dashboards"
---

# Econometric Test Plots

Econometric test plots transform numerical test results into visual summaries. PanelBox provides 9 chart types covering serial correlation, unit roots, cointegration, cross-sectional dependence, and comprehensive validation dashboards.

## Serial Correlation: ACF/PACF Plot

**Registry name**: `acf_pacf_plot`

Two-panel subplot showing autocorrelation (ACF) and partial autocorrelation (PACF) functions with confidence bands and optional Ljung-Box test annotation.

```python
from panelbox.visualization import create_acf_pacf_plot

chart = create_acf_pacf_plot(
    residuals,
    max_lags=20,                  # Default: min(20, T/4)
    confidence_level=0.95,
    show_ljung_box=True,
    theme="academic",
)
chart.figure.show()
```

Using the factory directly:

```python
from panelbox.visualization import ChartFactory

chart = ChartFactory.create(
    "acf_pacf_plot",
    data={
        "residuals": results.resids,
        "max_lags": 20,
        "confidence_level": 0.95,
        "show_ljung_box": True,
    },
    theme="academic",
)
```

**Interpretation**:

- ACF bars within confidence bands: no significant autocorrelation at that lag
- Significant ACF at lag $k$: correlation between $e_t$ and $e_{t-k}$
- Gradual ACF decay + sharp PACF cutoff after lag $p$: AR($p$) process
- Sharp ACF cutoff after lag $q$ + gradual PACF decay: MA($q$) process
- Ljung-Box p-value < 0.05: evidence of serial correlation

## Unit Root Test Plot

**Registry name**: `unit_root_test_plot`

Bar chart comparing test statistics against critical values with color-coded significance. Supports optional time series overlay.

```python
from panelbox.visualization import create_unit_root_test_plot

test_results = {
    "test_names": ["ADF", "Phillips-Perron", "KPSS"],
    "test_stats": [-3.5, -3.8, 0.3],
    "critical_values": {"1%": -3.96, "5%": -3.41, "10%": -3.13},
    "pvalues": [0.008, 0.003, 0.15],
}

chart = create_unit_root_test_plot(
    test_results,
    include_series=False,
    theme="professional",
)
```

**Interpretation**:

- Green bars: test statistic beyond critical value (reject null at that level)
- Red bars: test statistic inside critical value (fail to reject)
- For ADF/PP: rejecting null means stationarity (test stat more negative than critical value)
- For KPSS: rejecting null means non-stationarity (test stat larger than critical value)

## Cointegration Heatmap

**Registry name**: `cointegration_heatmap`

Symmetric heatmap of pairwise cointegration p-values with color-coded significance and optional test statistic annotations.

```python
from panelbox.visualization import create_cointegration_heatmap

results = {
    "variables": ["GDP", "Consumption", "Investment", "Exports"],
    "pvalues": [
        [1.00, 0.02, 0.15, 0.08],
        [0.02, 1.00, 0.01, 0.12],
        [0.15, 0.01, 1.00, 0.03],
        [0.08, 0.12, 0.03, 1.00],
    ],
    "test_name": "Engle-Granger",          # Optional
    "test_stats": None,                     # Optional n x n matrix
}

chart = create_cointegration_heatmap(results, theme="academic")
```

**Interpretation**:

- Green cells (low p-values): strong evidence of cointegration between variables
- Red cells (high p-values): no cointegration detected
- Diagonal is masked (variable cointegrated with itself is trivial)
- Clusters of green: groups of cointegrated variables suggest common trends

## Cross-Sectional Dependence Plot

**Registry name**: `cross_sectional_dependence_plot`

Gauge indicator for the Pesaran CD test statistic with optional entity-level correlation bar chart.

```python
from panelbox.visualization import create_cross_sectional_dependence_plot

cd_results = {
    "cd_statistic": 5.23,
    "pvalue": 0.001,
    "avg_correlation": 0.42,                     # Optional
    "entity_correlations": [0.3, 0.5, 0.6, 0.2], # Optional
}

chart = create_cross_sectional_dependence_plot(cd_results, theme="professional")
```

**Interpretation**:

- CD statistic > 1.96: reject null of cross-sectional independence at 5% level
- High average correlation: strong cross-sectional dependence, consider Driscoll-Kraay SE or spatial models
- Entity-level bars: identify which entities drive the dependence

## Validation Charts

The validation chart suite provides 5 chart types for comprehensive model validation reporting. All are created at once with `create_validation_charts()`.

### Creating Validation Charts

```python
from panelbox.visualization import create_validation_charts

# Prepare validation data
validation_data = {
    "tests": [
        {"name": "Hausman Test", "category": "Specification", "pvalue": 0.003, "statistic": 12.5},
        {"name": "Breusch-Pagan", "category": "Heteroskedasticity", "pvalue": 0.42, "statistic": 2.1},
        {"name": "Wooldridge AR(1)", "category": "Serial Correlation", "pvalue": 0.08, "statistic": 3.8},
        {"name": "Pesaran CD", "category": "Cross-Section", "pvalue": 0.001, "statistic": 5.2},
        {"name": "Jarque-Bera", "category": "Normality", "pvalue": 0.35, "statistic": 2.1},
        {"name": "Ramsey RESET", "category": "Specification", "pvalue": 0.22, "statistic": 1.5},
    ],
}

charts = create_validation_charts(
    validation_data,
    theme="professional",
    alpha=0.05,
)
```

Select specific charts:

```python
charts = create_validation_charts(
    validation_data,
    charts=["test_overview", "pvalue_distribution", "dashboard"],
    alpha=0.05,
)
```

### Test Overview

**Registry name**: `validation_test_overview`

Stacked bar chart showing passed/failed test counts by category.

```python
chart = ChartFactory.create(
    "validation_test_overview",
    data={
        "categories": ["Specification", "Heteroskedasticity", "Normality"],
        "passed": [1, 2, 1],
        "failed": [1, 0, 0],
    },
    theme="professional",
)
```

### P-Value Distribution

**Registry name**: `validation_pvalue_distribution`

Bar chart of individual test p-values with color-coded significance and reference lines at $\alpha$ and $\alpha/10$.

```python
chart = ChartFactory.create(
    "validation_pvalue_distribution",
    data={
        "test_names": ["Hausman", "Breusch-Pagan", "Wooldridge"],
        "pvalues": [0.003, 0.42, 0.08],
        "alpha": 0.05,
        "log_scale": True,         # Log scale for better discrimination
    },
    theme="academic",
)
```

Color coding:

- Green: p-value > $\alpha$ (test passed, no evidence against null)
- Orange: p-value between $\alpha/10$ and $\alpha$ (marginally significant)
- Red: p-value < $\alpha/10$ (strongly significant)

### Test Statistics

**Registry name**: `validation_test_statistics`

Scatter plot of absolute test statistics grouped by category. Marker size scales inversely with p-value.

```python
chart = ChartFactory.create(
    "validation_test_statistics",
    data={
        "test_names": ["Hausman", "Breusch-Pagan", "Wooldridge"],
        "statistics": [12.5, 2.1, 3.8],
        "categories": ["Specification", "Heteroskedasticity", "Serial Correlation"],
        "pvalues": [0.003, 0.42, 0.08],
    },
    theme="professional",
)
```

### Comparison Heatmap

**Registry name**: `validation_comparison_heatmap`

Heatmap comparing test results across multiple models. Useful for model selection.

```python
chart = ChartFactory.create(
    "validation_comparison_heatmap",
    data={
        "models": ["Pooled OLS", "Fixed Effects", "Random Effects"],
        "tests": ["Hausman", "Breusch-Pagan", "Wooldridge"],
        "matrix": [
            [0.003, 0.42, 0.08],
            [None, 0.35, 0.12],
            [0.003, 0.28, 0.06],
        ],
        "alpha": 0.05,
    },
    theme="academic",
)
```

### Validation Dashboard

**Registry name**: `validation_dashboard`

All-in-one 2x2 subplot combining: test overview, p-value distribution, test statistics scatter, and pass rate gauge (with 80% reference threshold).

```python
chart = ChartFactory.create(
    "validation_dashboard",
    data={
        "overview": {"categories": [...], "passed": [...], "failed": [...]},
        "pvalues": {"test_names": [...], "pvalues": [...], "alpha": 0.05},
        "statistics": {"test_names": [...], "statistics": [...], "categories": [...]},
        "summary": {"total_tests": 6, "passed": 4, "failed": 2},
    },
    theme="professional",
)
```

## Data Transformers

The `ValidationDataTransformer` converts `ValidationReport` objects into chart-ready data:

```python
from panelbox.visualization.transformers.validation import ValidationDataTransformer

transformer = ValidationDataTransformer()
chart_data = transformer.transform(validation_report)
# Returns dict with keys: 'tests', 'categories', 'summary', 'model_info'
```

## Complete Example

Full validation workflow with visualization:

```python
import panelbox as pb
from panelbox.visualization import (
    create_validation_charts,
    create_acf_pacf_plot,
    export_charts,
)

# Estimate model
model = pb.FixedEffects(data=data, formula="y ~ x1 + x2 + EntityEffects")
results = model.fit()

# Run validation tests and create charts
validation_data = {
    "tests": [
        {"name": "Hausman", "category": "Specification", "pvalue": 0.003, "statistic": 12.5},
        {"name": "BP Heteroskedasticity", "category": "Heteroskedasticity", "pvalue": 0.42, "statistic": 2.1},
        {"name": "Wooldridge AR(1)", "category": "Serial Correlation", "pvalue": 0.08, "statistic": 3.8},
        {"name": "Pesaran CD", "category": "Cross-Section", "pvalue": 0.15, "statistic": 1.4},
        {"name": "Jarque-Bera", "category": "Normality", "pvalue": 0.35, "statistic": 2.1},
    ],
}

charts = create_validation_charts(validation_data, theme="professional")

# Add ACF/PACF
acf_chart = create_acf_pacf_plot(results.resids, max_lags=15, theme="professional")

# Export all validation charts
export_charts(charts, output_dir="./validation", format="svg", prefix="val_")
acf_chart.save_image("validation/acf_pacf.svg")
```

## Comparison with Other Software

| Chart | PanelBox | Stata | R |
|-------|----------|-------|---|
| ACF/PACF | `create_acf_pacf_plot()` | `ac`, `pac`, `corrgram` | `acf()`, `pacf()`, `forecast::Acf()` |
| Unit root | `create_unit_root_test_plot()` | `dfuller` (table only) | `urca::summary()` (table only) |
| Cointegration heatmap | `create_cointegration_heatmap()` | Manual | Manual |
| CD test | `create_cross_sectional_dependence_plot()` | `xtcsd` (table only) | `plm::pcdtest()` (table only) |
| Validation dashboard | `create_validation_charts()` | Manual | Manual |

## See Also

- [Diagnostics Plots](model-diagnostics.md) -- Residual-level diagnostic charts
- [Comparison Plots](comparison.md) -- Compare test results across models
- [Themes & Customization](themes.md) -- Style your test plots
- [Panel Plots](panel-structure.md) -- Panel structure and effects
