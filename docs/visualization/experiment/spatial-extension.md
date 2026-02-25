---
title: "Spatial Extensions"
description: "Spatial diagnostics, model estimation, and report generation within PanelExperiment"
---

# Spatial Extensions

## Overview

PanelBox extends `PanelExperiment` with spatial econometric capabilities through the `SpatialPanelExperiment` mixin. This extension adds methods for:

- **Spatial diagnostics** -- Moran's I, Local Moran's I (LISA), and LM tests
- **Spatial model estimation** -- SAR, SEM, SDM, and GNS with automatic model selection
- **Spatial effects decomposition** -- direct, indirect (spillover), and total effects
- **Spatial diagnostic reports** -- HTML reports with maps and charts

These methods are automatically available on any `PanelExperiment` instance -- no additional imports are needed.

## Spatial Diagnostics

### Running the Full Diagnostic Battery

```python
import panelbox as pb
import numpy as np

data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# Create a spatial weights matrix (N x N)
N = data["firm"].nunique()
W = np.random.rand(N, N)        # Replace with real spatial weights
np.fill_diagonal(W, 0)
W = W / W.sum(axis=1, keepdims=True)  # Row-normalize

# Run diagnostics
diag = exp.run_spatial_diagnostics(W=W, alpha=0.05, verbose=True)
```

The method on `PanelExperiment` directly (not via the mixin) accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `W` | `np.ndarray` or `SpatialWeights` | (required) | N$\times$N spatial weight matrix |
| `alpha` | `float` | `0.05` | Significance level |
| `model_name` | `str` or `None` | `None` | OLS model name. If `None`, fits pooled OLS internally |
| `verbose` | `bool` | `True` | Print diagnostic summary |

### What is Returned

The diagnostics return a dictionary with four components:

```python
diag = exp.run_spatial_diagnostics(W=W)

# Global spatial autocorrelation
morans = diag['morans_i']
print(f"Moran's I: {morans.statistic:.4f} (p={morans.pvalue:.4f})")

# Local Moran's I (LISA clusters)
lisa = diag['morans_i_local']
print(f"LISA clusters:\n{lisa['cluster_type'].value_counts()}")

# LM tests for model specification
lm = diag['lm_tests']
print(f"LM Lag p-value:   {lm.get('lm_lag', {}).get('pvalue', 'N/A')}")
print(f"LM Error p-value: {lm.get('lm_error', {}).get('pvalue', 'N/A')}")

# Automatic model recommendation
print(f"Recommended model: {diag['recommendation']}")
```

| Key | Content | Description |
|-----|---------|-------------|
| `morans_i` | Test result object | Global Moran's I test for spatial autocorrelation |
| `morans_i_local` | DataFrame | Local Moran's I (LISA) with cluster types |
| `lm_tests` | `dict` | LM Lag, LM Error, Robust LM Lag, Robust LM Error |
| `recommendation` | `str` | Recommended spatial model based on LM decision tree |

### Understanding the Model Recommendation

The recommendation follows the Anselin & Rey (2014) decision tree:

| LM Lag | LM Error | Robust LM Lag | Robust LM Error | Recommendation |
|--------|----------|---------------|-----------------|----------------|
| Significant | Not significant | -- | -- | SAR (Spatial Lag) |
| Not significant | Significant | -- | -- | SEM (Spatial Error) |
| Significant | Significant | Significant | Not significant | SAR |
| Significant | Significant | Not significant | Significant | SEM |
| Significant | Significant | Both significant | Both significant | SDM or GNS |
| Not significant | Not significant | -- | -- | No spatial dependence |

## Estimating Spatial Models

### Automatic Model Selection

After running diagnostics, use `estimate_spatial_model()` with `model_type='auto'` to fit the recommended model:

```python
# Run diagnostics first
diag = exp.run_spatial_diagnostics(W=W)
print(f"Recommendation: {diag['recommendation']}")

# Fit the recommended model
results = exp.estimate_spatial_model(model_type='auto', W=W, name='spatial_auto')
```

### Manual Model Selection

Specify the spatial model type explicitly:

```python
# Spatial Lag (SAR)
sar = exp.estimate_spatial_model(model_type='sar', W=W, name='sar_model')

# Spatial Error (SEM)
sem = exp.estimate_spatial_model(model_type='sem', W=W, name='sem_model')

# Spatial Durbin (SDM)
sdm = exp.estimate_spatial_model(model_type='sdm', W=W, name='sdm_model')
```

**Supported model types:**

| Alias | Model | Description |
|-------|-------|-------------|
| `'sar'`, `'spatial_lag'` | Spatial Lag (SAR) | $y = \rho Wy + X\beta + \epsilon$ |
| `'sem'`, `'spatial_error'` | Spatial Error (SEM) | $y = X\beta + u$, $u = \lambda Wu + \epsilon$ |
| `'sdm'`, `'spatial_durbin'` | Spatial Durbin (SDM) | $y = \rho Wy + X\beta + WX\theta + \epsilon$ |
| `'gns'`, `'general_nesting'` | General Nesting Spatial | Encompasses SAR, SEM, and SDM |
| `'auto'` | Automatic | Uses LM test recommendation |

### Using the SpatialPanelExperiment Mixin

The mixin provides `add_spatial_model()` for more explicit control over spatial model fitting:

```python
from panelbox.core.spatial_weights import SpatialWeights

W_obj = SpatialWeights.from_contiguity(gdf, criterion="queen")

# Add spatial models with effects specification
exp.add_spatial_model("SAR-FE", W=W_obj, model_type="sar", effects="fixed")
exp.add_spatial_model("SDM-FE", W=W_obj, model_type="sdm", effects="fixed")

# Compare spatial and non-spatial models together
comp = exp.compare_models()
```

## Effects Decomposition

For Spatial Durbin (SDM) and General Nesting Spatial (GNS) models, decompose estimated effects into direct, indirect (spillover), and total components:

```python
# Fit an SDM model
exp.add_spatial_model("SDM", W=W_obj, model_type="sdm", effects="fixed")

# Decompose effects
effects = exp.decompose_spatial_effects("SDM")

print("Direct Effects:")
print(effects["direct"])

print("\nIndirect Effects (Spillovers):")
print(effects["indirect"])

print("\nTotal Effects:")
print(effects["total"])
```

!!! note "SDM/GNS Only"
    Effects decomposition is only available for SDM and GNS models. SAR and SEM models do not support this method because their spillover structure is captured differently.

## Comparing Spatial Models

Use `compare_spatial_models()` to compare spatial and non-spatial models together:

```python
# Fit models
exp.fit_model('pooled_ols', name='OLS')
exp.add_spatial_model("SAR", W=W_obj, model_type="sar")
exp.add_spatial_model("SEM", W=W_obj, model_type="sem")
exp.add_spatial_model("SDM", W=W_obj, model_type="sdm")

# Compare all (including non-spatial)
comparison_df = exp.compare_spatial_models(include_non_spatial=True)
print(comparison_df[['Model', 'Type', 'AIC', 'BIC']].to_string())
```

The comparison table includes spatial-specific columns for $\rho$ (spatial lag parameter) and $\lambda$ (spatial error parameter) when applicable.

## Spatial Diagnostic Reports

Generate an HTML report summarizing spatial diagnostics:

```python
path = exp.spatial_diagnostics_report(
    "spatial_diagnostics.html",
    include_maps=False        # Set True with GeoDataFrame for maps
)
```

### With Maps

If you have geometry data in a GeoDataFrame, include choropleth and LISA cluster maps:

```python
import geopandas as gpd

gdf = gpd.read_file("regions.shp")

path = exp.spatial_diagnostics_report(
    "spatial_report.html",
    include_maps=True,
    gdf=gdf
)
```

The report includes:

- Moran's I scatter plot
- LISA cluster map (if `include_maps=True`)
- LM test comparison chart
- Model recommendation summary

## Complete Example

```python
import panelbox as pb
import numpy as np

# 1. Load data and set up experiment
data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Create spatial weights (example: contiguity-based)
N = data["firm"].nunique()
W = np.eye(N, k=1) + np.eye(N, k=-1)   # Simple neighbor matrix
W = W / W.sum(axis=1, keepdims=True)    # Row-normalize

# 3. Fit baseline OLS
exp.fit_model('pooled_ols', name='ols')

# 4. Run spatial diagnostics
diag = exp.run_spatial_diagnostics(W=W, alpha=0.05, verbose=True)
print(f"Moran's I: {diag['morans_i'].statistic:.4f}")
print(f"Recommendation: {diag['recommendation']}")

# 5. Estimate recommended spatial model
spatial_results = exp.estimate_spatial_model(
    model_type='auto',
    W=W,
    name='spatial'
)

# 6. Generate spatial diagnostics report
exp.spatial_diagnostics_report("spatial_diagnostics.html")

# 7. Compare OLS and spatial model
comp = exp.compare_models(model_names=['ols', 'spatial'])
print(comp.summary())
```

## Requirements

Spatial extensions require:

- A spatial weights matrix (`W`): N$\times$N numpy array or `SpatialWeights` object
- For maps: a `GeoDataFrame` with geometry column (requires `geopandas`)
- For `SpatialWeights.from_contiguity()`: `libpysal` or polygon geometries

## Comparison with Other Software

| Task | PanelBox | Stata | R |
|------|----------|-------|---|
| Moran's I | `exp.run_spatial_diagnostics(W)` | `estat moran` | `spdep::moran.test()` |
| LM tests | Included in diagnostics | `spatdiag` | `spdep::lm.LMtests()` |
| SAR estimation | `exp.estimate_spatial_model('sar', W)` | `spxtregress` | `splm::spreml()` |
| Auto model selection | `model_type='auto'` | Manual | Manual |
| Effects decomposition | `exp.decompose_spatial_effects()` | `estat impact` | `spatialreg::impacts()` |

## See Also

- [Experiment Overview](index.md) -- Pattern overview and quick start
- [Workflow](fitting.md) -- Fitting and managing models
- [Spatial Models](../../user-guide/spatial/index.md) -- Spatial model theory and API
- [Spatial Visualization](../charts/specialized.md) -- Choropleth and LISA maps
- [Master Reports](master-reports.md) -- Combined report generation
