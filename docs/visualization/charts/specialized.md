---
title: "Specialized Plots"
description: "Domain-specific visualizations for SFA, quantile regression, VAR, spatial econometrics, and more"
---

# Specialized Plots

Beyond the core chart categories, PanelBox provides domain-specific visualizations for stochastic frontier analysis, quantile regression, vector autoregression, spatial econometrics, and general-purpose distribution, correlation, and time series charts.

## Stochastic Frontier (SFA) Plots

Located in `panelbox.frontier.visualization`, these charts visualize technical efficiency, frontier surfaces, and component decompositions.

### Efficiency Distribution

Histogram with KDE overlay showing the distribution of estimated technical efficiencies across entities.

```python
from panelbox.frontier.visualization.efficiency_plots import (
    plot_efficiency_distribution,
    plot_efficiency_ranking,
    plot_efficiency_boxplot,
)

# Distribution histogram
fig = plot_efficiency_distribution(
    efficiency_df,
    backend="plotly",       # 'plotly' or 'matplotlib'
    bins=30,
    show_kde=True,
    show_stats=True,        # Annotate mean, median, std
    title="Technical Efficiency Distribution",
)

# Top/bottom entity ranking
fig = plot_efficiency_ranking(
    efficiency_df,
    backend="plotly",
    top_n=10,
    bottom_n=10,
    entity_col="firm_id",
    colorscale="RdYlGn",
)

# Grouped box plots with statistical test
fig = plot_efficiency_boxplot(
    efficiency_df,
    group_var="region",
    backend="plotly",
    test="kruskal",         # 'anova' or 'kruskal'
)
```

### Efficiency Evolution

Time series visualizations tracking efficiency changes over time.

```python
from panelbox.frontier.visualization.evolution_plots import (
    plot_efficiency_timeseries,
    plot_efficiency_spaghetti,
    plot_efficiency_heatmap,
    plot_efficiency_fanchart,
)

# Mean efficiency over time with CI
fig = plot_efficiency_timeseries(
    efficiency_df,
    backend="plotly",
    show_ci=True,
    show_range=True,
    show_median=True,
    events={"2008": "Financial Crisis"},   # Event annotations
)

# Individual entity trajectories
fig = plot_efficiency_spaghetti(
    efficiency_df,
    backend="plotly",
    highlight=["Firm_A", "Firm_B"],   # Highlight specific entities
    alpha=0.2,
    show_mean=True,
)

# Entity x time heatmap
fig = plot_efficiency_heatmap(
    efficiency_df,
    backend="plotly",
    order_by="efficiency",
    colorscale="Viridis",
)

# Fan chart with percentiles
fig = plot_efficiency_fanchart(
    efficiency_df,
    backend="plotly",
    percentiles=[10, 25, 50, 75, 90],
)
```

### Four-Component Model Plots

For GTRE, CSN, and TRE models with persistent and transient inefficiency.

```python
from panelbox.frontier.visualization.four_component_plots import (
    plot_efficiency_distributions,
    plot_efficiency_scatter,
    plot_efficiency_evolution,
    plot_entity_decomposition,
    plot_variance_decomposition,
    plot_comprehensive_summary,
)

# Persistent vs transient vs overall efficiency distributions
fig = plot_efficiency_distributions(result, bins=30)

# Persistent vs transient scatter
fig = plot_efficiency_scatter(result, highlight_entities=["Firm_A"])

# Efficiency time evolution for selected entities
fig = plot_efficiency_evolution(result, n_entities=5)

# Component decomposition by entity
fig = plot_entity_decomposition(result, n_entities=10)

# Variance decomposition pie + bar chart
fig = plot_variance_decomposition(result)

# 3x3 comprehensive dashboard
fig = plot_comprehensive_summary(result, n_evolution=5)
```

### Frontier Surface

2D and 3D frontier visualizations.

```python
from panelbox.frontier.visualization.frontier_plots import (
    plot_frontier_2d,
    plot_frontier_3d,
    plot_frontier_contour,
    plot_frontier_partial,
)

# 2D frontier with efficiency-colored observations
fig = plot_frontier_2d(result, input_var="capital", backend="plotly")

# 3D production surface
fig = plot_frontier_3d(result, input_vars=["capital", "labor"], backend="plotly")

# Contour plot
fig = plot_frontier_contour(result, input_vars=["capital", "labor"], levels=10)

# Partial frontier (fixing other inputs at their means)
fig = plot_frontier_partial(result, input_var="capital", fix_others_at="mean")
```

### SFA Reports

Generate complete reports in multiple formats.

```python
from panelbox.frontier.visualization.reports import (
    to_latex,
    to_html,
    to_markdown,
    compare_models,
    efficiency_table,
)

# LaTeX table for publications
latex = to_latex(result, include_stats=["coef", "se", "pval"], caption="SFA Results")

# HTML report with embedded plots
html = to_html(result, filename="sfa_report.html", include_plots=True, theme="academic")

# Model comparison table
comparison = compare_models(
    {"Half-Normal": result_hn, "Exponential": result_exp},
    output_format="latex",
)

# Ranked efficiency table
eff_table = efficiency_table(result, sort_by="te", top_n=20)
```

## Quantile Regression Plots

Located in `panelbox.visualization.quantile`, these charts visualize how coefficients vary across the conditional distribution.

### Quantile Process Plot

Shows how coefficient estimates change across quantile levels $\tau \in [0.1, 0.9]$ with confidence bands.

```python
from panelbox.visualization.quantile.process_plots import (
    quantile_process_plot,
    qq_plot,
    residual_plot,
)

# Coefficient paths across quantiles
fig, ax = quantile_process_plot(
    quantiles=quantiles,         # Array of tau values
    params=params,               # n_vars x n_quantiles matrix
    std_errors=std_errors,
    alpha=0.05,
)

# Quantile-specific Q-Q plot
fig, ax = qq_plot(residuals)

# Residual scatter plot
fig, ax = residual_plot(residuals)
```

### Advanced Quantile Visualizations

Publication-ready plots with the `QuantileVisualizer` class.

```python
from panelbox.visualization.quantile.advanced_plots import QuantileVisualizer

viz = QuantileVisualizer(style="academic", dpi=300)

# Coefficient paths with uniform confidence bands
fig = viz.coefficient_path(result, var_names=["x1", "x2"], uniform_bands=True)

# Fan chart for prediction intervals
fig = viz.fan_chart(result, X_forecast=X_new, tau_list=[0.1, 0.25, 0.5, 0.75, 0.9])

# Conditional density estimation from quantile functions
fig = viz.conditional_density(result, X_values=X_grid, y_grid=y_grid)

# Spaghetti plot of individual quantile curves
fig = viz.spaghetti_plot(result, highlight_quantiles=[0.1, 0.5, 0.9])

# Save all visualizations at once
viz.save_all(result, output_dir="./quantile_plots", formats=["png", "pdf"])
```

### 3D Surface Plots

Coefficient surfaces showing how effects vary across quantiles and covariate values.

```python
from panelbox.visualization.quantile.surface_plots import SurfacePlotter

plotter = SurfacePlotter(figsize=(12, 8), colormap="viridis")

# 3D surface of beta(tau, X)
fig = plotter.plot_surface(result, var_names=["education"])

# Interactive Plotly 3D surface
fig = plotter.plot_interactive(result, var_names=["education"])

# Coefficient heatmap across quantiles
fig = plotter.coefficient_heatmap(result, var_names=["education", "experience"])
```

### Interactive Quantile Dashboard

Plotly-based interactive exploration.

```python
from panelbox.visualization.quantile.interactive import InteractivePlotter

plotter = InteractivePlotter(theme="plotly_white")

# 2x2 dashboard: paths, heatmap, violin, significance
fig = plotter.coefficient_dashboard(result, var_names=["x1", "x2"])

# Animated coefficient path
fig = plotter.animated_coefficient_path(result, var_idx=0, var_name="education")

# Parallel coordinates across quantiles
fig = plotter.parallel_coordinates(result, n_samples=100)
```

### Publication Themes

Quantile-specific themes for academic journals.

```python
from panelbox.visualization.quantile.themes import PublicationTheme, ColorSchemes

# Apply a journal-specific theme
PublicationTheme.apply("nature")       # Nature, Science, Economics, AEA, IEEE
PublicationTheme.apply("economics", color_palette="colorblind")

# Use as context manager
with PublicationTheme.use("science"):
    fig, ax = quantile_process_plot(quantiles, params, std_errors)

# Get publication-appropriate figure sizes
figsize = PublicationTheme.get_figsize("nature", columns=2)  # Two-column figure

# Color utilities
colors = ColorSchemes.get_qualitative(5, palette="colorblind")
sequential = ColorSchemes.get_sequential(9, cmap_name="viridis")
```

Available themes: `nature`, `science`, `economics`, `presentation`, `poster`, `minimal`, `aea`, `ieee`

Available palettes: `colorblind`, `grayscale`, `nature`, `economics`, `vibrant`

## VAR (Vector Autoregression) Plots

Located in `panelbox.visualization.var_plots`, these functions visualize Panel VAR results.

```python
from panelbox.visualization.var_plots import (
    plot_irf,
    plot_fevd,
    plot_stability,
    plot_instrument_sensitivity,
)

# Impulse Response Functions (grid layout with CI bands)
fig = plot_irf(
    irf_result,
    impulse="gdp",               # Optional: specific impulse variable
    response="inflation",        # Optional: specific response variable
    variables=["gdp", "inflation", "interest_rate"],
    ci=True,
    backend="plotly",            # 'plotly' or 'matplotlib'
    theme="academic",
)

# Forecast Error Variance Decomposition
fig = plot_fevd(
    fevd_result,
    kind="area",                 # 'area' or 'bar'
    variables=["gdp", "inflation"],
    backend="plotly",
    theme="professional",
)

# VAR stability (companion matrix eigenvalues vs unit circle)
fig = plot_stability(eigenvalues, backend="plotly")

# Instrument sensitivity analysis
fig = plot_instrument_sensitivity(sensitivity_results, backend="plotly")
```

## Spatial Econometrics Plots

Located in `panelbox.visualization.spatial_plots`, these functions create spatial maps and diagnostics.

```python
from panelbox.visualization.spatial_plots import (
    create_moran_scatterplot,
    create_lisa_cluster_map,
    plot_morans_i_by_period,
    plot_spatial_weights_structure,
    create_spatial_diagnostics_dashboard,
    plot_spatial_effects,
    plot_direct_vs_indirect,
    plot_effects_comparison,
)

# Moran's I scatter plot
fig = create_moran_scatterplot(
    values=y,
    W=spatial_weights,
    show_regression=True,
    show_quadrants=True,
)

# LISA cluster map (requires GeoDataFrame)
fig = create_lisa_cluster_map(
    lisa_results,
    gdf=geo_dataframe,           # Optional: for choropleth map
    color_map={"HH": "red", "LL": "blue", "HL": "lightcoral", "LH": "lightblue"},
)

# Moran's I over time
fig = plot_morans_i_by_period(
    results,
    show_expected=True,
    show_significance=True,
    alpha=0.05,
)

# Spatial weights matrix heatmap
fig = plot_spatial_weights_structure(W, cmap="Blues", show_colorbar=True)

# Multi-panel spatial diagnostics dashboard
fig = create_spatial_diagnostics_dashboard(spatial_diagnostics)

# Spatial effects decomposition (direct, indirect, total)
fig = plot_spatial_effects(effects_result, show_ci=True)

# Direct vs indirect scatter
fig = plot_direct_vs_indirect(effects_result, show_diagonal=True)

# Multi-model effects comparison
fig = plot_effects_comparison(
    [sar_effects, sdm_effects],
    model_names=["SAR", "SDM"],
    effect_type="total",
)
```

## Distribution Plots

General-purpose distribution charts available through the factory.

```python
from panelbox.visualization import ChartFactory

# Histogram with KDE and normal overlay
chart = ChartFactory.create("distribution_histogram", data={
    "values": data_array,
    "show_kde": True,
    "show_normal": True,
    "bins": "auto",
}, theme="professional")

# Kernel Density Estimation with rug plot
chart = ChartFactory.create("distribution_kde", data={
    "values": data_array,
    "groups": group_labels,    # Optional: grouped KDE
    "show_rug": True,
    "fill": True,
}, theme="academic")

# Violin plot
chart = ChartFactory.create("distribution_violin", data={
    "values": data_array,
    "groups": group_labels,
    "show_box": True,
    "show_points": False,
}, theme="professional")

# Box plot
chart = ChartFactory.create("distribution_boxplot", data={
    "values": data_array,
    "groups": group_labels,
    "show_mean": True,
    "orientation": "v",        # 'v' or 'h'
}, theme="academic")
```

## Correlation Plots

Correlation analysis charts.

```python
# Correlation heatmap
chart = ChartFactory.create("correlation_heatmap", data={
    "correlation_matrix": corr_matrix,
    "variable_names": var_names,
    "show_values": True,
    "mask_diagonal": False,
    "mask_upper": False,
    "threshold": 0.3,          # Optional: only show |r| > threshold
}, theme="academic")

# Pairwise scatter plot grid
chart = ChartFactory.create("correlation_pairwise", data={
    "data": dataframe,
    "variables": ["x1", "x2", "x3", "y"],   # Max 8 variables
    "group": "treatment",                     # Optional: color by group
    "show_diagonal_hist": True,
}, theme="professional")
```

## Time Series Plots

Panel-aware time series visualizations.

```python
# Multi-entity panel time series
chart = ChartFactory.create("timeseries_panel", data={
    "time": time_values,
    "values": y_values,
    "entity_id": entity_ids,
    "variable_name": "GDP Growth",
    "max_entities": 20,
    "show_mean": True,          # Black dashed mean line
}, theme="professional")

# Trend line with moving average
chart = ChartFactory.create("timeseries_trend", data={
    "time": time_values,
    "values": y_values,
    "show_moving_average": True,
    "window": 7,
    "show_trend": True,         # Linear regression trend
}, theme="academic")

# Faceted small multiples
chart = ChartFactory.create("timeseries_faceted", data={
    "time": time_values,
    "values": y_values,
    "entity_id": entity_ids,
    "ncols": 3,                 # Columns in the grid
    "shared_yaxis": True,
}, theme="professional")
```

## Comparison with Other Software

| Category | PanelBox | Stata | R |
|----------|----------|-------|---|
| SFA efficiency | `plot_efficiency_distribution()` | Manual after `frontier` | `frontier::efficiencies()` + custom |
| Quantile process | `quantile_process_plot()` | `qreg` + manual | `quantreg::plot.rqs()` |
| VAR IRF | `plot_irf()` | `irf graph` | `vars::plot(irf(...))` |
| Spatial maps | `create_lisa_cluster_map()` | `spmap` | `spdep`, `tmap` |
| Correlation heatmap | `correlation_heatmap` | Manual | `corrplot::corrplot()` |

## See Also

- [Diagnostics Plots](model-diagnostics.md) -- Core residual diagnostics
- [Panel Plots](panel-structure.md) -- Panel structure visualization
- [Test Plots](test-plots.md) -- Econometric test visualization
- [Comparison Plots](comparison.md) -- Model comparison charts
- [Themes & Customization](themes.md) -- Theme all your charts consistently
