# PanelBox Chart Gallery

Complete reference of all available chart types with code examples.

---

## Residual Diagnostics

### Q-Q Plot

**Chart Type**: `residual_qq_plot`

Assess normality of residuals using quantile-quantile plot

```python
from panelbox.visualization import create_residual_diagnostics

# Generate residuals (normally distributed)
residuals = np.random.randn(100)

# Create diagnostics
diagnostics = create_residual_diagnostics(
    {'residuals': residuals, 'fitted': np.random.randn(100)},
    charts=['qq_plot'],
    theme='academic'
)

diagnostics['qq_plot'].show()
```

---

### Residual vs Fitted

**Chart Type**: `residual_vs_fitted`

Detect heteroskedasticity and non-linear patterns

```python
from panelbox.visualization import create_residual_diagnostics

# Create diagnostics
diagnostics = create_residual_diagnostics(
    results,
    charts=['residual_vs_fitted'],
    theme='professional'
)

diagnostics['residual_vs_fitted'].show()
```

---

### Scale-Location Plot

**Chart Type**: `scale_location`

Check homoskedasticity assumption

```python
diagnostics = create_residual_diagnostics(
    results,
    charts=['scale_location'],
    theme='academic'
)

diagnostics['scale_location'].show()
```

---

### ACF/PACF Plot

**Chart Type**: `acf_pacf_plot`

Detect serial correlation in residuals

```python
from panelbox.visualization import create_acf_pacf_plot

# Generate AR(1) process
residuals = np.zeros(200)
residuals[0] = np.random.randn()
for t in range(1, 200):
    residuals[t] = 0.7 * residuals[t-1] + np.random.randn()

chart = create_acf_pacf_plot(
    residuals,
    max_lags=20,
    confidence_level=0.95,
    show_ljung_box=True,
    theme='academic'
)

chart.show()
```

---

## Validation

### Validation Dashboard

**Chart Type**: `validation_dashboard`

Complete validation overview

```python
from panelbox.visualization import create_validation_charts

charts = create_validation_charts(
    validation_report,
    theme='professional'
)

charts['dashboard'].show()
```

---

## Model Comparison

### Coefficient Comparison

**Chart Type**: `coefficient_comparison`

Compare coefficients across models

```python
from panelbox.visualization import create_comparison_charts

charts = create_comparison_charts(
    [ols_results, fe_results, re_results],
    model_names=['OLS', 'Fixed Effects', 'Random Effects'],
    theme='professional'
)

charts['coefficients'].show()
```

---

## Panel-Specific

### Entity Effects Plot

**Chart Type**: `entity_effects_plot`

Visualize entity-specific fixed effects

```python
from panelbox.visualization import create_entity_effects_plot

# Entity effects data
data = {
    'entity_id': ['Firm A', 'Firm B', 'Firm C', 'Firm D', 'Firm E'],
    'effect': [0.5, -0.3, 0.8, -0.1, 0.2],
    'std_error': [0.15, 0.12, 0.18, 0.10, 0.14]
}

chart = create_entity_effects_plot(
    data,
    theme='professional',
    sort_by='effect'
)

chart.show()
```

---

### Time Effects Plot

**Chart Type**: `time_effects_plot`

Visualize time-period fixed effects

```python
from panelbox.visualization import create_time_effects_plot

# Time effects data
data = {
    'time_id': list(range(2010, 2021)),
    'effect': np.cumsum(np.random.randn(11) * 0.1),
    'std_error': np.random.uniform(0.05, 0.15, 11)
}

chart = create_time_effects_plot(
    data,
    theme='academic',
    show_trend=True
)

chart.show()
```

---

### Between-Within Variation

**Chart Type**: `between_within_plot`

Decompose variance into between and within components

```python
from panelbox.visualization import create_between_within_plot

# Generate panel data
panel_data = pd.DataFrame({
    'entity': np.repeat(range(1, 51), 20),
    'time': np.tile(range(1, 21), 50),
    'capital': np.random.randn(1000),
    'labor': np.random.randn(1000),
    'output': np.random.randn(1000)
})

chart = create_between_within_plot(
    panel_data.set_index(['entity', 'time']),
    variables=['capital', 'labor', 'output'],
    theme='professional',
    style='stacked'
)

chart.show()
```

---

### Panel Structure Plot

**Chart Type**: `panel_structure_plot`

Visualize panel balance and missing data patterns

```python
from panelbox.visualization import create_panel_structure_plot

# Panel data
panel_data = pd.DataFrame({
    'firm': np.repeat(range(1, 21), 30),
    'year': np.tile(range(1, 31), 20),
    'value': np.random.randn(600)
})

chart = create_panel_structure_plot(
    panel_data.set_index(['firm', 'year']),
    theme='professional'
)

chart.show()
```

---

## Econometric Tests

### Unit Root Test Plot

**Chart Type**: `unit_root_test_plot`

Visualize stationarity test results

```python
from panelbox.visualization import create_unit_root_test_plot

results = {
    'test_names': ['ADF', 'PP', 'KPSS', 'DF-GLS'],
    'test_stats': [-3.5, -3.8, 0.3, -2.9],
    'critical_values': {'1%': -3.96, '5%': -3.41, '10%': -3.13},
    'pvalues': [0.008, 0.003, 0.15, 0.04]
}

chart = create_unit_root_test_plot(
    results,
    theme='professional'
)

chart.show()
```

---

### Cointegration Heatmap

**Chart Type**: `cointegration_heatmap`

Visualize pairwise cointegration relationships

```python
from panelbox.visualization import create_cointegration_heatmap

results = {
    'variables': ['GDP', 'Consumption', 'Investment', 'Exports'],
    'pvalues': [
        [1.0, 0.02, 0.15, 0.08],
        [0.02, 1.0, 0.08, 0.12],
        [0.15, 0.08, 1.0, 0.05],
        [0.08, 0.12, 0.05, 1.0]
    ],
    'test_name': 'Engle-Granger'
}

chart = create_cointegration_heatmap(
    results,
    theme='academic'
)

chart.show()
```

---

### Cross-Sectional Dependence

**Chart Type**: `cross_sectional_dependence_plot`

Pesaran CD test visualization

```python
from panelbox.visualization import create_cross_sectional_dependence_plot

results = {
    'cd_statistic': 3.45,
    'pvalue': 0.003,
    'avg_correlation': 0.28,
    'entity_correlations': [0.15, 0.32, 0.45, 0.21, 0.38, 0.29]
}

chart = create_cross_sectional_dependence_plot(
    results,
    theme='professional'
)

chart.show()
```

---

## Distribution

### Histogram

**Chart Type**: `histogram`

Visualize data distribution

```python
from panelbox.visualization import ChartFactory

data = {'values': np.random.randn(500)}

chart = ChartFactory.create(
    'histogram',
    data=data,
    theme='professional'
)

chart.show()
```

---

## Correlation

### Correlation Heatmap

**Chart Type**: `correlation_heatmap`

Visualize correlation matrix

```python
from panelbox.visualization import ChartFactory

# Generate correlation matrix
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
corr_matrix = df.corr()

data = {
    'correlation_matrix': corr_matrix.values.tolist(),
    'variables': corr_matrix.columns.tolist()
}

chart = ChartFactory.create(
    'correlation_heatmap',
    data=data,
    theme='academic'
)

chart.show()
```

---

## Time Series

### Panel Time Series

**Chart Type**: `panel_timeseries`

Visualize time series across entities

```python
from panelbox.visualization import ChartFactory

# Generate panel time series
dates = pd.date_range('2010-01-01', periods=100, freq='M')
entities = ['Entity A', 'Entity B', 'Entity C']

data = {
    'time': dates.tolist() * 3,
    'entity': np.repeat(entities, 100).tolist(),
    'value': np.random.randn(300).cumsum()
}

chart = ChartFactory.create(
    'panel_timeseries',
    data=pd.DataFrame(data),
    theme='professional'
)

chart.show()
```

---

