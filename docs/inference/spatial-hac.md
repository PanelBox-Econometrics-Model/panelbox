---
title: "Spatial HAC Standard Errors"
description: "Conley (1999) spatial heteroskedasticity and autocorrelation consistent standard errors for geographically correlated panel data in PanelBox."
---

# Spatial HAC Standard Errors

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.SpatialHAC`
    **Comparison:** `panelbox.standard_errors.DriscollKraayComparison`
    **Stata equivalent:** `acreg` (Colella et al. 2019)
    **R equivalent:** `conleyreg::conley()`

## Overview

When observations are **geographically located**, nearby entities often have correlated errors --- due to shared local conditions, spatial spillovers, or omitted spatial variables. Standard clustered SE assume correlation within discrete groups, but spatial correlation decays continuously with distance and does not respect administrative boundaries.

The **Conley (1999) spatial HAC estimator** extends the Newey-West framework to the spatial dimension, weighting error cross-products by a kernel function of geographic distance. This provides inference that is robust to:

- Heteroskedasticity
- Spatial autocorrelation (up to a specified cutoff distance)
- Temporal autocorrelation (up to a specified lag)

## When to Use

- **Regional economics**: County-level, state-level, or grid-level data
- **Environmental economics**: Pollution, climate, agriculture with spatial spillovers
- **Real estate**: Property prices with neighborhood effects
- **Development economics**: Village-level data with spatial correlation
- Any panel where entities have **geographic coordinates** and errors are spatially correlated

!!! note "When NOT to use"
    - **No spatial structure**: Use [clustered SE](clustered.md) or [robust SE](robust.md)
    - **Uniform cross-sectional dependence** (no distance decay): Use [Driscoll-Kraay](driscoll-kraay.md)
    - **Very large $N$**: The $O(N^2 T^2)$ computation can be slow for thousands of entities

## Quick Example

```python
from panelbox.standard_errors import SpatialHAC

# Method 1: From geographic coordinates
shac = SpatialHAC.from_coordinates(
    coords=coords,              # (N x 2) array of [latitude, longitude]
    spatial_cutoff=100.0,       # 100 km cutoff
    distance_metric="haversine",
)
V_hac = shac.compute(X, residuals, entity_index, time_index)
se_hac = np.sqrt(np.diag(V_hac))
print(f"Spatial HAC SE: {se_hac}")

# Method 2: From pre-computed distance matrix
import numpy as np
shac = SpatialHAC(
    distance_matrix=dist_mat,   # (N x N) distances in km
    spatial_cutoff=150.0,
    temporal_cutoff=2,          # Also allow 2 lags of temporal correlation
    spatial_kernel="bartlett",
    temporal_kernel="bartlett",
)
V_hac = shac.compute(X, residuals, entity_index, time_index)
```

## Construction Methods

### From Coordinates

Use `SpatialHAC.from_coordinates()` when you have latitude/longitude or other coordinate data:

```python
shac = SpatialHAC.from_coordinates(
    coords=coords,               # (N x 2): [lat, lon] or [x, y]
    spatial_cutoff=100.0,         # Max distance for spatial correlation
    temporal_cutoff=0,            # 0 = no temporal correlation
    distance_metric="haversine",  # For lat/lon in degrees
)
```

**Distance metrics:**

| Metric | Input | Output | Use Case |
|--------|-------|--------|----------|
| `"haversine"` | Lat/lon in degrees | Kilometers | Geographic coordinates |
| `"euclidean"` | Any coordinates | Same units | Projected coordinates or grids |
| `"manhattan"` | Any coordinates | Same units | Grid-based distances |

### From Distance Matrix

Use the constructor directly when you have a pre-computed distance matrix:

```python
shac = SpatialHAC(
    distance_matrix=dist_mat,     # (N x N) symmetric
    spatial_cutoff=100.0,
    temporal_cutoff=0,
    spatial_kernel="bartlett",
    temporal_kernel="bartlett",
)
```

## Mathematical Details

### The Conley (1999) Estimator

The spatial HAC covariance matrix is:

$$
V = (X'X)^{-1} \hat{\Omega} (X'X)^{-1}
$$

where the meat matrix accounts for both spatial and temporal correlation:

$$
\hat{\Omega} = \sum_{i} \sum_{j} K_S(d_{ij}) \cdot K_T(|t_i - t_j|) \cdot \hat{e}_i \hat{e}_j \cdot x_i x_j'
$$

- $K_S(d_{ij})$ is the spatial kernel evaluated at the distance $d_{ij}$ between entities $i$ and $j$
- $K_T(|t_i - t_j|)$ is the temporal kernel evaluated at the time lag
- $\hat{e}_i, \hat{e}_j$ are residuals
- $x_i, x_j$ are regressor vectors

### Spatial Kernels

The spatial kernel determines how correlation decays with distance:

| Kernel | Formula $K(u)$, $u = d/\text{cutoff}$ | Properties |
|--------|----------------------------------------|------------|
| `"bartlett"` | $\max(1 - u, 0)$ | Linear decay, most common |
| `"uniform"` | $\mathbb{1}(u \leq 1)$ | All-or-nothing weighting |
| `"triangular"` | $\max(1 - u, 0)$ | Same as Bartlett |
| `"epanechnikov"` | $0.75(1 - u^2) \cdot \mathbb{1}(u \leq 1)$ | Smooth, optimal for density estimation |

### Temporal Kernels

When `temporal_cutoff > 0`, temporal correlation is also modeled:

| Kernel | Available |
|--------|-----------|
| `"bartlett"` | Yes (default) |
| `"uniform"` | Yes |
| `"parzen"` | Yes |
| `"quadratic_spectral"` | Yes |

### Spatial Cutoff Selection

The **spatial cutoff** defines the maximum distance at which spatial correlation is assumed to exist. Observations beyond this distance receive zero weight.

Guidance for choosing the cutoff:

- **Too small**: Ignores genuine spatial correlation, SEs are too small
- **Too large**: Includes noisy estimates, SEs become imprecise
- **Robustness check**: Report results for multiple cutoffs (e.g., 50, 100, 200 km)

### Small Sample Correction

When `small_sample_correction=True` (the default), PanelBox applies:

$$
V_{\text{corrected}} = \frac{n}{n - k} \cdot V
$$

## Configuration Options

### SpatialHAC Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance_matrix` | `np.ndarray` | --- | Distance matrix $(N \times N)$ |
| `spatial_cutoff` | `float` | --- | Max distance for spatial correlation |
| `temporal_cutoff` | `int` | `0` | Max temporal lag (0 = spatial only) |
| `spatial_kernel` | `str` | `"bartlett"` | `"bartlett"`, `"uniform"`, `"triangular"`, `"epanechnikov"` |
| `temporal_kernel` | `str` | `"bartlett"` | `"bartlett"`, `"uniform"`, `"parzen"`, `"quadratic_spectral"` |

### SpatialHAC.from_coordinates()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coords` | `np.ndarray` | --- | Coordinates $(N \times 2)$ |
| `spatial_cutoff` | `float` | --- | Max distance |
| `temporal_cutoff` | `int` | `0` | Max temporal lag |
| `distance_metric` | `str` | `"haversine"` | `"haversine"`, `"euclidean"`, `"manhattan"` |

### SpatialHAC.compute()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | --- | Design matrix $(NT \times K)$ |
| `residuals` | `np.ndarray` | --- | Residuals $(NT,)$ |
| `entity_index` | `np.ndarray` | --- | Entity identifiers $(NT,)$ |
| `time_index` | `np.ndarray` | --- | Time identifiers $(NT,)$ |
| `small_sample_correction` | `bool` | `True` | Apply $n/(n-k)$ correction |

**Returns:** `np.ndarray` --- Covariance matrix $(K \times K)$

## Comparing with Driscoll-Kraay

```python
from panelbox.standard_errors import SpatialHAC, DriscollKraayComparison, driscoll_kraay
import numpy as np

# Spatial HAC
shac = SpatialHAC.from_coordinates(coords, spatial_cutoff=100.0)
V_shac = shac.compute(X, residuals, entity_index, time_index)
se_shac = np.sqrt(np.diag(V_shac))

# Driscoll-Kraay
dk_result = driscoll_kraay(X, residuals, time_index, max_lags=3)
se_dk = dk_result.std_errors

# Compare
comparison = DriscollKraayComparison.compare(se_shac, se_dk, param_names=["x1", "x2"])
print(comparison)
```

### Spatial HAC vs Driscoll-Kraay

| Feature | Spatial HAC | Driscoll-Kraay |
|---------|-------------|----------------|
| Spatial correlation | Distance-based decay | Uniform (all entities) |
| Requires coordinates | Yes | No |
| Handles distance decay | Yes | No |
| Computational cost | $O(N^2 T^2)$ | $O(NT \cdot T)$ |
| Best for | Regional data with coordinates | Panels with common shocks |

## Comparing with Standard SE Types

```python
comparison = shac.compare_with_standard_errors(X, residuals, entity_index, time_index)

print(f"OLS SE:        {comparison['se_ols']}")
print(f"White SE:      {comparison['se_white']}")
print(f"Spatial HAC:   {comparison['se_hac']}")
print(f"HAC/OLS ratio: {comparison['se_ratio_hac_ols']}")
```

## Common Pitfalls

!!! warning "Pitfall 1: Wrong distance units"
    The `spatial_cutoff` must be in the **same units** as the distance matrix. For `haversine`, distances are in kilometers. For `euclidean`, they are in the same units as the coordinates.

!!! warning "Pitfall 2: Cutoff too large or too small"
    A cutoff that is too small ignores genuine spatial correlation. A cutoff that is too large includes noise. Always perform a sensitivity analysis by varying the cutoff.

!!! warning "Pitfall 3: Unbalanced panels"
    The estimator works with unbalanced panels but warns if $n \neq N \times T$. Severe imbalance can affect performance.

!!! warning "Pitfall 4: Large panels"
    The double loop over all observation pairs makes computation $O(N^2 T^2)$. For panels with thousands of entities, consider approximations or limiting the spatial cutoff to reduce computation.

## See Also

- [Driscoll-Kraay](driscoll-kraay.md) --- When spatial correlation is uniform (no distance decay)
- [Newey-West](newey-west.md) --- Temporal HAC without spatial dimension
- [Clustered](clustered.md) --- When correlation follows discrete group boundaries
- [Comparison](comparison.md) --- Compare Spatial HAC with other SE types
- [Inference Overview](index.md) --- Choosing the right SE type

## References

- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Hsiang, S. M. (2010). Temperatures and cyclones strongly associated with economic production in the Caribbean and Central America. *Proceedings of the National Academy of Sciences*, 107(35), 15367-15372.
- Colella, F., Lalive, R., Sakalli, S. O., & Thoenig, M. (2019). Inference with arbitrary clustering. *IZA Discussion Paper* No. 12584.
- Conley, T. G., & Molinari, F. (2007). Spatial correlation robust inference with errors in location or distance. *Journal of Econometrics*, 140(1), 76-96.
