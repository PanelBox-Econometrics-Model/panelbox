# PanelBox Chart Selection Guide

🎯 **Quick Guide to Choose the Right Chart for Your Analysis**

This guide helps you select the appropriate visualization for your panel data analysis needs.

---

## Interactive Selection

For an interactive selection experience, use the chart selector:

```python
from panelbox.visualization.utils import suggest_chart

# Interactive mode - answer questions to get recommendations
suggest_chart(interactive=True)

# Or search by keywords
charts = suggest_chart(keywords=['residual', 'normality'])
for chart in charts:
    print(chart)
```

---

## Quick Reference by Analysis Goal

### 1. 🔍 Residual Diagnostics

**Goal**: Check model assumptions and identify problems

| What to Check | Recommended Chart | Function |
|---------------|-------------------|----------|
| **Normality** | Q-Q Plot | `create_residual_diagnostics(charts=['qq_plot'])` |
| **Heteroskedasticity** | Residual vs Fitted | `create_residual_diagnostics(charts=['residual_vs_fitted'])` |
| **Variance Stability** | Scale-Location | `create_residual_diagnostics(charts=['scale_location'])` |
| **Influential Observations** | Residual vs Leverage | `create_residual_diagnostics(charts=['residual_vs_leverage'])` |
| **Serial Correlation** | ACF/PACF Plot | `create_acf_pacf_plot(residuals)` |
| **All Diagnostics** | Complete Suite | `create_residual_diagnostics(results)` |

**Example:**
```python
from panelbox.visualization import create_residual_diagnostics

# Get all diagnostic charts
diagnostics = create_residual_diagnostics(
    results,
    theme='academic'
)

# Show specific diagnostic
diagnostics['qq_plot'].show()
```

---

### 2. ✅ Model Validation

**Goal**: Validate model specification and test statistical assumptions

| What to Validate | Recommended Chart | Function |
|------------------|-------------------|----------|
| **Overall Status** | Validation Dashboard | `create_validation_charts()` |
| **Stationarity** | Unit Root Test Plot | `create_unit_root_test_plot(test_results)` |
| **P-value Distribution** | P-Value Distribution | `create_validation_charts()` |
| **Test Comparison** | Comparison Heatmap | `create_validation_charts()` |

**Example:**
```python
from panelbox.visualization import create_validation_charts

# Get validation dashboard
charts = create_validation_charts(
    validation_report,
    theme='professional'
)

charts['dashboard'].show()
```

---

### 3. 📊 Model Comparison

**Goal**: Compare different model specifications

| What to Compare | Recommended Chart | Function |
|-----------------|-------------------|----------|
| **Coefficients** | Coefficient Comparison | `create_comparison_charts()` |
| **Confidence Intervals** | Forest Plot | `create_comparison_charts()` |
| **Model Fit (R², AIC)** | Model Fit Comparison | `create_comparison_charts()` |
| **Information Criteria** | IC Comparison | `create_comparison_charts()` |

**Example:**
```python
from panelbox.visualization import create_comparison_charts

# Compare three models
charts = create_comparison_charts(
    [ols_results, fe_results, re_results],
    model_names=['OLS', 'Fixed Effects', 'Random Effects'],
    theme='professional'
)

charts['coefficients'].show()
```

---

### 4. 🏢 Panel Data Analysis

**Goal**: Analyze panel-specific characteristics

| What to Analyze | Recommended Chart | Function |
|-----------------|-------------------|----------|
| **Entity-Specific Effects** | Entity Effects Plot | `create_entity_effects_plot(results)` |
| **Time-Period Effects** | Time Effects Plot | `create_time_effects_plot(results)` |
| **Between-Within Variation** | Between-Within Plot | `create_between_within_plot(data, variables)` |
| **Panel Structure/Balance** | Panel Structure Plot | `create_panel_structure_plot(data)` |
| **Cross-Sectional Dependence** | CD Plot | `create_cross_sectional_dependence_plot(cd_results)` |

**Example:**
```python
from panelbox.visualization import create_entity_effects_plot

# Visualize entity fixed effects
chart = create_entity_effects_plot(
    fe_results,
    theme='academic',
    sort_by='effect'
)

chart.show()
```

---

### 5. 📈 Econometric Tests

**Goal**: Test econometric properties and relationships

| What to Test | Recommended Chart | Function |
|--------------|-------------------|----------|
| **Serial Correlation** | ACF/PACF Plot | `create_acf_pacf_plot(residuals)` |
| **Unit Roots / Stationarity** | Unit Root Test Plot | `create_unit_root_test_plot(results)` |
| **Cointegration** | Cointegration Heatmap | `create_cointegration_heatmap(results)` |
| **Cross-Sectional Dependence** | CD Plot | `create_cross_sectional_dependence_plot(results)` |

**Example:**
```python
from panelbox.visualization import create_unit_root_test_plot

# Visualize stationarity tests
results = {
    'test_names': ['ADF', 'PP', 'KPSS'],
    'test_stats': [-3.5, -3.8, 0.3],
    'critical_values': {'5%': -3.41},
    'pvalues': [0.008, 0.003, 0.15]
}

chart = create_unit_root_test_plot(results, theme='professional')
chart.show()
```

---

### 6. 🔬 Exploratory Data Analysis

**Goal**: Explore data distributions and relationships

| What to Explore | Recommended Chart | Function |
|-----------------|-------------------|----------|
| **Distribution** | Histogram / KDE | `ChartFactory.create('histogram', data)` |
| **Correlation** | Correlation Heatmap | `ChartFactory.create('correlation_heatmap', data)` |
| **Time Series** | Panel Time Series | `ChartFactory.create('panel_timeseries', data)` |
| **Group Comparison** | Box Plot / Violin | `ChartFactory.create('box_plot', data)` |

---

## Decision Tree

Use this flowchart to find the right chart:

```
START: What is your analysis goal?

├─ Residual Diagnostics
│   ├─ Check normality? → residual_qq_plot
│   ├─ Check heteroskedasticity? → residual_vs_fitted
│   ├─ Check serial correlation? → acf_pacf_plot
│   └─ Check influential points? → residual_vs_leverage
│
├─ Model Validation
│   ├─ Overall validation? → validation_dashboard
│   ├─ Stationarity tests? → unit_root_test_plot
│   └─ Test comparison? → validation_comparison_heatmap
│
├─ Model Comparison
│   ├─ Compare coefficients? → coefficient_comparison / forest_plot
│   ├─ Compare fit? → model_fit_comparison
│   └─ Compare IC? → information_criteria
│
├─ Panel Analysis
│   ├─ Entity effects? → entity_effects_plot
│   ├─ Time effects? → time_effects_plot
│   ├─ Variance decomposition? → between_within_plot
│   └─ Panel structure? → panel_structure_plot
│
├─ Econometric Tests
│   ├─ Serial correlation? → acf_pacf_plot
│   ├─ Unit roots? → unit_root_test_plot
│   ├─ Cointegration? → cointegration_heatmap
│   └─ Cross-sectional dependence? → cross_sectional_dependence_plot
│
└─ Exploratory Analysis
    ├─ Distribution? → histogram / kde / violin_plot
    ├─ Correlation? → correlation_heatmap
    └─ Time series? → panel_timeseries
```

---

## Complete Chart Catalog

For a complete catalog with code examples for all 32+ charts, see:
- [CHART_GALLERY.md](CHART_GALLERY.md) - Markdown reference
- Run: `python examples/gallery_generator.py` - Generate gallery

---

## Tips for Effective Visualization

### 1. Choose the Right Theme

```python
# Professional (corporate blue) - default
theme='professional'

# Academic (journal-ready, grayscale-friendly)
theme='academic'

# Presentation (high-contrast, large fonts)
theme='presentation'

# Custom theme from file
from panelbox.visualization.utils import load_theme
theme = load_theme('my_theme.yaml')
```

### 2. Export for Different Purposes

```python
# Interactive HTML (for sharing, embedding)
chart.to_html('chart.html')

# Static PNG (for papers, presentations)
chart.to_image('chart.png', width=800, height=600)

# JSON (for programmatic manipulation)
chart.to_json('chart.json')

# Multiple formats at once
from panelbox.visualization import export_charts_multiple_formats
export_charts_multiple_formats(
    [chart1, chart2],
    output_dir='output/',
    formats=['html', 'png', 'json']
)
```

### 3. Combine Multiple Charts

```python
# Get all residual diagnostics
diagnostics = create_residual_diagnostics(
    results,
    charts=['qq_plot', 'residual_vs_fitted', 'scale_location'],
    theme='academic'
)

# Show them all
for name, chart in diagnostics.items():
    chart.show()
```

---

## Common Workflows

### Workflow 1: Complete Model Diagnostics

```python
from panelbox.visualization import (
    create_residual_diagnostics,
    create_validation_charts
)

# 1. Check residuals
residual_charts = create_residual_diagnostics(results, theme='academic')
residual_charts['qq_plot'].show()
residual_charts['residual_vs_fitted'].show()

# 2. Validate model
validation_charts = create_validation_charts(validation_report, theme='academic')
validation_charts['dashboard'].show()

# 3. Check serial correlation
from panelbox.visualization import create_acf_pacf_plot
acf_chart = create_acf_pacf_plot(results.resid, max_lags=20)
acf_chart.show()
```

### Workflow 2: Panel Data Exploration

```python
from panelbox.visualization import (
    create_panel_structure_plot,
    create_between_within_plot,
    create_entity_effects_plot
)

# 1. Check panel structure
structure = create_panel_structure_plot(panel_data, theme='professional')
structure.show()

# 2. Variance decomposition
bw_plot = create_between_within_plot(
    panel_data,
    variables=['capital', 'labor', 'output'],
    style='stacked'
)
bw_plot.show()

# 3. Entity effects (after fitting model)
effects = create_entity_effects_plot(fe_results, sort_by='effect')
effects.show()
```

### Workflow 3: Model Comparison

```python
from panelbox.visualization import create_comparison_charts

# Compare multiple model specifications
charts = create_comparison_charts(
    [pooled_results, fe_results, re_results, gmm_results],
    model_names=['Pooled OLS', 'Fixed Effects', 'Random Effects', 'GMM'],
    theme='professional'
)

# Show comparisons
charts['coefficients'].show()
charts['model_fit'].show()
charts['information_criteria'].show()
```

---

## Need More Help?

### Interactive Selection
```bash
# Run interactive chart selector
python -c "from panelbox.visualization.utils import suggest_chart; suggest_chart(interactive=True)"
```

### Search by Keywords
```python
from panelbox.visualization.utils import suggest_chart

# Find charts related to specific terms
charts = suggest_chart(keywords=['residual', 'diagnostic'])
for chart in charts:
    print(f"\n{chart.display_name}")
    print(f"  Use for: {', '.join(chart.use_cases)}")
    print(f"  API: {chart.api_function}")
```

### List by Category
```python
from panelbox.visualization.utils import list_all_charts

# Get all panel-specific charts
panel_charts = list_all_charts(category='Panel-Specific')
for chart in panel_charts:
    print(f"- {chart.display_name}: {chart.description}")
```

---

## Additional Resources

- **Full API Documentation**: See `docs/api/visualization.md`
- **Chart Gallery**: `examples/CHART_GALLERY.md`
- **Tutorial Notebooks**: `examples/jupyter/`
- **Performance Benchmarks**: `python benchmarks/visualization_performance.py`
- **Custom Themes**: `panelbox/visualization/utils/theme_loader.py`

---

**Happy Visualizing! 📊**
