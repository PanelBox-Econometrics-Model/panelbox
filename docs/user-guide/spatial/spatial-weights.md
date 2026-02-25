---
title: "Spatial Weight Matrices"
description: "Construction, normalization, and best practices for spatial weight matrices in PanelBox panel data models."
---

# Spatial Weight Matrices

!!! info "Quick Reference"
    **Class:** `panelbox.models.spatial.SpatialWeights`
    **Import:** `from panelbox.models.spatial import SpatialWeights`
    **Stata equivalent:** `spatwmat` / `spmatrix`
    **R equivalent:** `spdep::nb2listw()`

## Overview

A spatial weight matrix $W$ is an $N \times N$ matrix that encodes the spatial relationships between units (regions, firms, individuals) in your panel. Each entry $w_{ij}$ quantifies the "connection" between unit $i$ and unit $j$. These matrices are the foundation of all spatial econometric models — they define the structure through which spillovers, interactions, and spatial dependence propagate.

PanelBox provides the `SpatialWeights` class, which wraps weight matrices with methods for construction, normalization, validation, and visualization. The class supports both dense and sparse representations, making it suitable for datasets ranging from small regional panels to large-scale county-level analyses.

Choosing the right weight matrix is one of the most consequential decisions in spatial econometrics. Different weight structures can lead to substantially different results, so robustness checks across multiple specifications are strongly recommended.

## Quick Example

```python
import numpy as np
from panelbox.models.spatial import SpatialWeights

# Create from a numpy array (3 regions, contiguous pairs)
matrix = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
W = SpatialWeights(matrix=matrix, normalized=False)

# Row-standardize: each row sums to 1
W_std = W.standardize(method='row')

# Compute spatial lag of a variable
y = np.array([10, 20, 15])
Wy = W_std.spatial_lag(y)  # Weighted average of neighbors
print(Wy)  # [20.0, 12.5, 20.0]
```

## When to Use

- Your data has a **geographic structure** (regions, counties, countries) and you need to model spatial dependence
- You have **network data** (trade flows, social connections) and want to define proximity
- You need to test for **spatial autocorrelation** (Moran's I) before choosing a spatial model
- You want to compute **spatial lags** ($Wy$, $WX$) as inputs to spatial models

!!! warning "Key Assumptions"
    - The weight matrix must be **square** ($N \times N$) with $N$ matching the number of cross-sectional units
    - The **diagonal must be zero** (a unit is not its own neighbor)
    - All entries must be **non-negative** ($w_{ij} \geq 0$)
    - For most spatial models, $W$ should be **row-standardized** so each row sums to 1

## Detailed Guide

### Types of Weight Matrices

There are four main approaches to constructing a spatial weight matrix:

#### Contiguity-Based Weights

Contiguity weights define neighbors based on shared boundaries. Two conventions exist:

- **Queen contiguity**: units are neighbors if they share any boundary point (edge or vertex)
- **Rook contiguity**: units are neighbors only if they share an edge

```python
import geopandas as gpd
from panelbox.models.spatial import SpatialWeights

# Load a shapefile with polygon geometries
gdf = gpd.read_file("regions.shp")

# Queen contiguity (default) — most common
W_queen = SpatialWeights.from_contiguity(gdf, criterion='queen')

# Rook contiguity — stricter definition
W_rook = SpatialWeights.from_contiguity(gdf, criterion='rook')
```

!!! tip "When to use contiguity"
    Contiguity is ideal for **administrative regions** (states, counties, municipalities) where shared borders imply economic or social interaction. Queen contiguity is the most common choice.

#### Distance-Based Weights

Distance weights use the physical distance between units to define connections.

```python
import numpy as np
from panelbox.models.spatial import SpatialWeights

# Coordinates: longitude, latitude (or projected x, y)
coords = np.array([
    [-73.9, 40.7],   # New York
    [-87.6, 41.9],   # Chicago
    [-118.2, 34.1],  # Los Angeles
    [-122.4, 37.8],  # San Francisco
])

# Binary: connected if within threshold distance
W_dist = SpatialWeights.from_distance(coords, threshold=2000, binary=True)

# Inverse distance: closer units have stronger connections
W_inv = SpatialWeights.from_distance(coords, threshold=2000, binary=False)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `coords` | Array of coordinates (N x 2) | Required |
| `threshold` | Maximum distance for connectivity | Required |
| `p` | Minkowski distance parameter (2.0 = Euclidean) | `2.0` |
| `binary` | If `True`, W=1 for connected pairs; if `False`, W=1/d | `True` |

#### k-Nearest Neighbors

KNN weights connect each unit to its $k$ closest neighbors. This ensures every unit has the same number of neighbors, which avoids islands.

```python
from panelbox.models.spatial import SpatialWeights

# Each region connected to its 5 nearest neighbors
W_knn = SpatialWeights.from_knn(coords, k=5)
```

!!! note
    KNN weights are **not symmetric** by default — unit $i$ may be among the 5 nearest neighbors of unit $j$, but $j$ may not be among the 5 nearest of $i$. Row-standardization is typically applied after construction.

#### Custom Weight Matrices

For non-geographic relationships (trade flows, economic similarity, social networks), you can construct $W$ directly from a matrix or array.

```python
import numpy as np
from panelbox.models.spatial import SpatialWeights

# Economic distance: inverse of trade volume
trade_flows = np.array([
    [0,  100, 50],
    [80,   0, 30],
    [60,  40,  0]
])
W_trade = SpatialWeights(matrix=trade_flows, normalized=False)
W_trade = W_trade.standardize(method='row')

# From a list of lists
W = SpatialWeights.from_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
```

### Row Standardization

Row standardization divides each entry by the row sum so that each row sums to 1:

$$w_{ij}^{*} = \frac{w_{ij}}{\sum_{j=1}^{N} w_{ij}}$$

After standardization, the spatial lag $Wy_i = \sum_j w_{ij}^{*} y_j$ is a **weighted average** of neighbors' values.

```python
# Standardize in-place (returns a new SpatialWeights object)
W_std = W.standardize(method='row')
```

| Method | Description |
|--------|-------------|
| `'row'` | Row-standardize: each row sums to 1 |
| `'spectral'` | Divide by largest eigenvalue |

!!! warning "When NOT to row-standardize"
    - When edge weights have a meaningful absolute interpretation (e.g., trade volume in dollars)
    - When you want to preserve asymmetry in the relationship
    - Some network models require the original weights

### Properties and Diagnostics

The `SpatialWeights` class provides several useful properties:

```python
W = SpatialWeights(matrix=my_matrix, normalized=False)

# Number of spatial units
print(W.n)  # e.g., 48

# Eigenvalues (computed on demand, cached for reuse)
eigenvals = W.eigenvalues  # Used for log-det computation

# Parameter bounds for spatial models
rho_min, rho_max = W.get_bounds()  # e.g., (-1.52, 1.0)

# Summary statistics for Moran's I computation
print(W.s0)  # Sum of all weights
print(W.s1)  # Trace-based statistic
print(W.s2)  # Row/column sum statistic

# Print a summary of the weight matrix
W.summary()
```

### Handling Islands

Islands are entities with no spatial neighbors (row sum = 0). They can cause problems in spatial models because the spatial lag $Wy_i = 0$ regardless of neighbors' values.

```python
import numpy as np

# Check for islands
W_array = W.matrix if hasattr(W, 'matrix') else W
row_sums = np.asarray(W_array.sum(axis=1)).flatten()
islands = np.where(row_sums == 0)[0]

if len(islands) > 0:
    print(f"Warning: {len(islands)} islands found at indices: {islands}")
```

**Strategies for handling islands:**

1. **Remove them from the analysis** — simplest but loses data
2. **Use KNN weights** — guarantees every unit has at least $k$ neighbors
3. **Increase the distance threshold** — connect more units
4. **Connect to nearest neighbor manually** — ensures connectivity

### Sparse Matrices for Large Datasets

For large $N$ (more than a few hundred units), use sparse representations to save memory and speed up computation.

```python
# Convert to sparse format
W_sparse = W.to_sparse()  # Returns scipy.sparse.csr_matrix

# Convert back to dense
W_dense = W.to_dense()    # Returns numpy.ndarray
```

!!! tip "Performance Guidelines"
    | N (entities) | Recommended | Notes |
    |---|---|---|
    | < 500 | Dense matrix | Fast eigenvalue computation |
    | 500 - 5,000 | Sparse matrix | Use `to_sparse()` |
    | 5,000 - 10,000 | Sparse + LU decomposition | Model uses `sparse_lu` for log-det |
    | > 10,000 | Sparse + Chebyshev | Model uses Chebyshev approximation |

### Visualization

```python
# Plot weight matrix connections (requires geopandas for map overlay)
W.plot(gdf=gdf, figsize=(10, 8), backend='plotly')

# Without geographic data: shows matrix heatmap
W.plot()
```

## Configuration Options

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `matrix` | `np.ndarray` or `csr_matrix` | Required | N x N spatial weight matrix |
| `normalized` | `bool` | `False` | Whether the matrix is already row-normalized |
| `validate` | `bool` | `True` | Validate matrix properties (square, zero diagonal, non-negative) |

### Class Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `from_contiguity(gdf, criterion)` | `gdf`: GeoDataFrame, `criterion`: `'queen'` or `'rook'` | Build from polygon contiguity |
| `from_distance(coords, threshold, p, binary)` | `coords`: Nx2 array, `threshold`: float, `p`: 2.0, `binary`: True | Distance-based weights |
| `from_knn(coords, k)` | `coords`: Nx2 array, `k`: int | k-nearest neighbor weights |
| `from_matrix(array)` | `array`: list or np.ndarray | Build from raw array |

## Best Practices

1. **Always row-standardize** unless you have a specific reason not to. Most spatial econometric theory assumes row-standardized weights.

2. **Test sensitivity to $W$**. Run your spatial model with at least 2-3 different weight specifications (e.g., queen contiguity, KNN with $k$=5, distance threshold). If results change dramatically, your conclusions may not be robust.

3. **Match $W$ to theory**. The weight matrix should reflect the **mechanism** of spatial interaction in your context:
    - Shared borders for policy diffusion → contiguity
    - Physical proximity for commuting/trade → distance
    - Fixed number of peers → KNN
    - Economic linkages → custom trade/input-output matrix

4. **Check sparsity**. Weight matrices are typically sparse (most entries are 0). For large $N$, use `W.to_sparse()` to improve performance.

5. **Report your $W$ specification**. In published work, always describe the weight matrix construction, number of neighbors (mean, min, max), and whether you row-standardized.

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Complete spatial modeling workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Lag (SAR)](spatial-lag.md) — Spatial autoregressive model using $W$
- [Spatial Error (SEM)](spatial-error.md) — Spatial error model using $W$
- [Spatial Durbin (SDM)](spatial-durbin.md) — Includes both $Wy$ and $WX$
- [Choosing a Spatial Model](choosing-model.md) — Decision guide for model selection
- [Spatial Diagnostics](diagnostics.md) — Tests that use $W$ (Moran's I, LM tests)

## References

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic.
2. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
3. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
4. Getis, A. (2009). Spatial weights matrices. *Geographical Analysis*, 41(4), 404-410.
