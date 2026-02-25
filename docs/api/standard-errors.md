---
title: "Standard Errors API"
description: "API reference for panelbox.standard_errors — robust, clustered, HAC, PCSE, and spatial standard errors"
---

# Standard Errors API Reference

!!! info "Module"
    **Import**: `from panelbox.standard_errors import ...`
    **Source**: `panelbox/standard_errors/`

## Overview

PanelBox provides a comprehensive suite of standard error estimators for panel data, addressing heteroskedasticity, autocorrelation, clustering, cross-sectional dependence, and spatial correlation. Each estimator is available both as a **class** (for full control) and as a **convenience function** (for quick usage).

| Estimator | Class | Function | Use Case |
|-----------|-------|----------|----------|
| HC0–HC3 | `RobustStandardErrors` | `robust_covariance()` | Heteroskedasticity |
| Clustered | `ClusteredStandardErrors` | `cluster_by_entity()`, `cluster_by_time()`, `twoway_cluster()` | Within-group correlation |
| Driscoll-Kraay | `DriscollKraayStandardErrors` | `driscoll_kraay()` | Cross-sectional + temporal dependence |
| Newey-West | `NeweyWestStandardErrors` | `newey_west()` | HAC (time series) |
| PCSE | `PanelCorrectedStandardErrors` | `pcse()` | Beck-Katz (T > N panels) |
| Spatial HAC | `SpatialHAC` | — | Spatial + temporal dependence |
| Comparison | `StandardErrorComparison` | — | Compare all SE types side-by-side |

---

## Robust Standard Errors (HC0–HC3)

### `RobustStandardErrors`

White (1980) heteroskedasticity-consistent covariance estimation.

```python
class RobustStandardErrors(X: np.ndarray, resid: np.ndarray)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Design matrix (n x k) |
| `resid` | `np.ndarray` | OLS residuals (n,) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute(method="HC1")` | `RobustCovarianceResult` | Compute covariance with specified HC variant |
| `hc0()` | `RobustCovarianceResult` | White (1980) — no small-sample correction |
| `hc1()` | `RobustCovarianceResult` | Stata default — n/(n-k) correction |
| `hc2()` | `RobustCovarianceResult` | Leverage-adjusted residuals |
| `hc3()` | `RobustCovarianceResult` | Jackknife-like — most conservative |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `leverage` | `np.ndarray` | Hat matrix diagonal |
| `bread` | `np.ndarray` | (X'X)^{-1} matrix |

### `RobustCovarianceResult`

```python
@dataclass
class RobustCovarianceResult:
    cov_matrix: np.ndarray      # Covariance matrix (k x k)
    std_errors: np.ndarray      # Standard errors (k,)
    method: str                 # HC variant used
    n_obs: int                  # Number of observations
    n_params: int               # Number of parameters
    leverage: np.ndarray | None # Hat matrix diagonal (HC2/HC3)
```

### `robust_covariance()`

```python
def robust_covariance(
    X: np.ndarray,
    resid: np.ndarray,
    method: str = "HC1",
) -> RobustCovarianceResult
```

**Example:**

```python
from panelbox.standard_errors import RobustStandardErrors, robust_covariance

# Class-based
rse = RobustStandardErrors(X, residuals)
result = rse.compute(method="HC3")
print(result.std_errors)

# Function-based (equivalent)
result = robust_covariance(X, residuals, method="HC3")
```

---

## Clustered Standard Errors

### `ClusteredStandardErrors`

One-way and two-way cluster-robust covariance estimation (Arellano 1987, Cameron et al. 2011).

```python
class ClusteredStandardErrors(
    X: np.ndarray,
    resid: np.ndarray,
    clusters: np.ndarray | tuple,
    df_correction: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Design matrix (n x k) |
| `resid` | `np.ndarray` | — | Residuals (n,) |
| `clusters` | `np.ndarray \| tuple` | — | Cluster IDs; tuple of two arrays for two-way |
| `df_correction` | `bool` | `True` | Apply finite-sample correction |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute()` | `ClusteredCovarianceResult` | Compute clustered covariance |
| `diagnostic_summary()` | `str` | Summary with cluster counts and sizes |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `bread` | `np.ndarray` | (X'X)^{-1} matrix |
| `n_clusters` | `int \| tuple` | Number of clusters (per dimension) |

### `ClusteredCovarianceResult`

```python
@dataclass
class ClusteredCovarianceResult:
    cov_matrix: np.ndarray    # Covariance matrix (k x k)
    std_errors: np.ndarray    # Standard errors (k,)
    n_clusters: int | tuple   # Number of clusters
    n_obs: int                # Number of observations
    n_params: int             # Number of parameters
    cluster_dims: int         # Clustering dimensions (1 or 2)
    df_correction: bool       # Whether correction was applied
```

### Convenience Functions

```python
def cluster_by_entity(X, resid, entity_ids, df_correction=True) -> ClusteredCovarianceResult
def cluster_by_time(X, resid, time_ids, df_correction=True) -> ClusteredCovarianceResult
def twoway_cluster(X, resid, cluster1, cluster2, df_correction=True) -> ClusteredCovarianceResult
```

**Example:**

```python
from panelbox.standard_errors import cluster_by_entity, twoway_cluster

# One-way clustering by entity
result = cluster_by_entity(X, residuals, entity_ids)

# Two-way clustering by entity and time
result = twoway_cluster(X, residuals, entity_ids, time_ids)
print(f"SE: {result.std_errors}")
print(f"Clusters: {result.n_clusters}")
```

---

## Driscoll-Kraay Standard Errors

### `DriscollKraayStandardErrors`

Driscoll & Kraay (1998) standard errors robust to spatial and temporal dependence. Ideal for macro panels with N large and T moderate.

```python
class DriscollKraayStandardErrors(
    X: np.ndarray,
    resid: np.ndarray,
    time_ids: np.ndarray,
    max_lags: int | None = None,
    kernel: str = "bartlett",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Design matrix |
| `resid` | `np.ndarray` | — | Residuals |
| `time_ids` | `np.ndarray` | — | Time period identifiers |
| `max_lags` | `int \| None` | `None` | Bandwidth (auto-selected if None) |
| `kernel` | `str` | `"bartlett"` | Kernel: `"bartlett"`, `"parzen"`, `"quadratic_spectral"` |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute()` | `DriscollKraayResult` | Compute DK standard errors |
| `diagnostic_summary()` | `str` | Summary with bandwidth and kernel info |

### `DriscollKraayResult`

```python
@dataclass
class DriscollKraayResult:
    cov_matrix: np.ndarray      # Covariance matrix
    std_errors: np.ndarray      # Standard errors
    max_lags: int               # Bandwidth used
    kernel: str                 # Kernel function
    n_obs: int
    n_params: int
    n_periods: int              # Number of time periods
    bandwidth: float | None     # Effective bandwidth
```

### `driscoll_kraay()`

```python
def driscoll_kraay(X, resid, time_ids, max_lags=None, kernel="bartlett") -> DriscollKraayResult
```

---

## Newey-West HAC Standard Errors

### `NeweyWestStandardErrors`

Newey & West (1987) heteroskedasticity and autocorrelation consistent standard errors.

```python
class NeweyWestStandardErrors(
    X: np.ndarray,
    resid: np.ndarray,
    max_lags: int | None = None,
    kernel: str = "bartlett",
    prewhitening: bool = False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Design matrix |
| `resid` | `np.ndarray` | — | Residuals |
| `max_lags` | `int \| None` | `None` | Bandwidth (auto if None) |
| `kernel` | `str` | `"bartlett"` | Kernel function |
| `prewhitening` | `bool` | `False` | Apply Andrews-Monahan prewhitening |

### `NeweyWestResult`

```python
@dataclass
class NeweyWestResult:
    cov_matrix: np.ndarray
    std_errors: np.ndarray
    max_lags: int
    kernel: str
    n_obs: int
    n_params: int
    prewhitening: bool
```

### `newey_west()`

```python
def newey_west(X, resid, max_lags=None, kernel="bartlett", prewhitening=False) -> NeweyWestResult
```

---

## Panel-Corrected Standard Errors (PCSE)

### `PanelCorrectedStandardErrors`

Beck & Katz (1995) panel-corrected standard errors. Best for panels where **T > N** (more time periods than entities).

```python
class PanelCorrectedStandardErrors(
    X: np.ndarray,
    resid: np.ndarray,
    entity_ids: np.ndarray,
    time_ids: np.ndarray,
)
```

### `PCSEResult`

```python
@dataclass
class PCSEResult:
    cov_matrix: np.ndarray
    std_errors: np.ndarray
    sigma_matrix: np.ndarray   # Cross-entity error covariance
    n_obs: int
    n_params: int
    n_entities: int
    n_periods: int
```

### `pcse()`

```python
def pcse(X, resid, entity_ids, time_ids) -> PCSEResult
```

---

## Spatial HAC

### `SpatialHAC`

Conley (1999) spatial HAC standard errors accounting for both spatial and temporal correlation.

```python
class SpatialHAC(
    distance_matrix: np.ndarray,
    spatial_cutoff: float,
    temporal_cutoff: int = 0,
    spatial_kernel: str = "bartlett",
    temporal_kernel: str = "bartlett",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distance_matrix` | `np.ndarray` | — | Pairwise distance matrix |
| `spatial_cutoff` | `float` | — | Maximum distance for spatial correlation |
| `temporal_cutoff` | `int` | `0` | Maximum lag for temporal correlation |
| `spatial_kernel` | `str` | `"bartlett"` | Spatial kernel function |
| `temporal_kernel` | `str` | `"bartlett"` | Temporal kernel function |

**Class Method:**

```python
@classmethod
def from_coordinates(
    cls,
    coords: np.ndarray,
    spatial_cutoff: float,
    temporal_cutoff: int = 0,
    distance_metric: str = "haversine",   # "haversine", "euclidean", "manhattan"
    **kwargs,
) -> SpatialHAC
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compute(X, residuals, entity_index, time_index, small_sample_correction=True)` | `np.ndarray` | Compute spatial HAC covariance matrix |
| `compare_with_standard_errors(X, residuals, entity_index, time_index)` | `dict` | Compare with Driscoll-Kraay SEs |

**Example:**

```python
from panelbox.standard_errors import SpatialHAC

# From geographic coordinates (lat, lon)
shac = SpatialHAC.from_coordinates(
    coords=coords_array,
    spatial_cutoff=500,          # 500 km
    temporal_cutoff=2,           # 2 periods
    distance_metric="haversine",
)
cov_matrix = shac.compute(X, residuals, entity_ids, time_ids)
```

### `DriscollKraayComparison`

```python
class DriscollKraayComparison:
    @staticmethod
    def compare(
        spatial_hac_se: np.ndarray,
        driscoll_kraay_se: np.ndarray,
        param_names: list | None = None,
    ) -> pd.DataFrame
```

---

## Standard Error Comparison

### `StandardErrorComparison`

Compare multiple SE estimators side-by-side for the same model.

```python
class StandardErrorComparison(model_results)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `compare_all(se_types=None, **kwargs)` | `ComparisonResult` | Compare all or specified SE types |
| `compare_pair(se_type1, se_type2, **kwargs)` | `ComparisonResult` | Compare two SE types |
| `plot_comparison(result=None, alpha=0.05, figsize=(12,8))` | `Figure` | Plot SE comparison |
| `summary(result=None)` | — | Print formatted summary |

### `ComparisonResult`

```python
@dataclass
class ComparisonResult:
    se_comparison: pd.DataFrame    # SE values by type
    se_ratios: pd.DataFrame        # Ratios relative to baseline
    t_stats: pd.DataFrame          # t-statistics by SE type
    p_values: pd.DataFrame         # p-values by SE type
    ci_lower: pd.DataFrame         # CI lower bounds
    ci_upper: pd.DataFrame         # CI upper bounds
    significance: pd.DataFrame     # Significance stars
    summary_stats: pd.DataFrame    # Summary statistics
```

**Example:**

```python
from panelbox.standard_errors import StandardErrorComparison

comp = StandardErrorComparison(model_results)
result = comp.compare_all(se_types=["HC1", "clustered", "driscoll_kraay"])
comp.summary(result)
comp.plot_comparison(result)
```

---

## MLE Standard Errors

!!! info "Module"
    **Import**: `from panelbox.standard_errors.mle import ...`

Functions for computing standard errors of maximum likelihood estimators.

### `sandwich_estimator()`

```python
def sandwich_estimator(
    hessian: np.ndarray,
    scores: np.ndarray,
    method: str = "robust",   # "nonrobust" or "robust"
) -> MLECovarianceResult
```

### `cluster_robust_mle()`

```python
def cluster_robust_mle(
    hessian: np.ndarray,
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    df_correction: bool = True,
) -> MLECovarianceResult
```

### `delta_method()`

```python
def delta_method(
    vcov: np.ndarray,
    transform_func: Callable,
    params: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray
```

### `bootstrap_mle()`

```python
def bootstrap_mle(
    estimate_func: Callable,
    y: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 999,
    cluster_ids: np.ndarray | None = None,
    seed: int = 42,
) -> MLECovarianceResult
```

### `MLECovarianceResult`

```python
class MLECovarianceResult:
    cov_matrix: np.ndarray
    std_errors: np.ndarray
    method: str
    n_obs: int
    n_params: int
```

---

## Low-Level Utilities

Building blocks for custom covariance estimators.

```python
from panelbox.standard_errors import (
    compute_bread,       # (X'X)^{-1}
    compute_meat_hc,     # HC meat matrix
    compute_leverage,    # Hat matrix diagonal
    sandwich_covariance, # Bread @ Meat @ Bread
    hc_covariance,       # Full HC covariance in one call
    clustered_covariance,           # One-way clustered covariance
    twoway_clustered_covariance,    # Two-way clustered covariance
)
```

| Function | Signature | Returns |
|----------|-----------|---------|
| `compute_bread(X)` | `np.ndarray → np.ndarray` | (X'X)^{-1} |
| `compute_meat_hc(X, resid, method="HC1", leverage=None)` | → `np.ndarray` | HC meat matrix |
| `compute_leverage(X)` | `np.ndarray → np.ndarray` | Diagonal of hat matrix |
| `sandwich_covariance(bread, meat)` | → `np.ndarray` | Bread @ Meat @ Bread |
| `hc_covariance(X, resid, method="HC1")` | → `np.ndarray` | Full HC covariance matrix |
| `clustered_covariance(X, resid, clusters, df_correction=True)` | → `np.ndarray` | One-way clustered covariance |
| `twoway_clustered_covariance(X, resid, clusters1, clusters2, df_correction=True)` | → `np.ndarray` | Two-way clustered covariance |

---

## See Also

- [Standard Errors Tutorial](../tutorials/standard-errors.md) — practical guide with examples
- [Static Models API](static-models.md) — `cov_type` parameter in model estimation
- [Spatial Models API](spatial.md) — spatial standard errors in spatial models
