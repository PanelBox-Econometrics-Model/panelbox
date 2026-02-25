---
title: "Clustered Standard Errors"
description: "One-way and two-way cluster-robust standard errors for panel data with within-group correlation in PanelBox."
---

# Clustered Standard Errors

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.ClusteredStandardErrors`
    **Convenience:** `cluster_by_entity()`, `cluster_by_time()`, `twoway_cluster()`
    **Model integration:** `model.fit(cov_type="clustered")` or `model.fit(cov_type="twoway")`
    **Stata equivalent:** `vce(cluster id)`, `vce(cluster id1) vce(cluster id2)` (via `reghdfe`)
    **R equivalent:** `plm::vcovHC(cluster="group")`, `multiwayvcov::cluster.vcov()`

## Overview

In panel data, observations within the same entity (firm, individual, country) are typically **correlated** across time. Classical and even heteroskedasticity-robust standard errors ignore this dependence, leading to standard errors that are too small and over-rejection of null hypotheses.

**Clustered standard errors** allow for arbitrary correlation within clusters while assuming independence across clusters. This is the **workhorse method** for panel data inference.

## When to Use

- **One-way clustering by entity**: When errors are correlated within entities across time (most common in micro panels)
- **One-way clustering by time**: When errors are correlated across entities within the same time period (common shocks)
- **Two-way clustering**: When both within-entity and within-time correlation exist (finance panels, macro panels)

!!! note "When NOT to use"
    - If you have **fewer than 20 clusters**, clustered SEs may be unreliable. Consider wild cluster bootstrap.
    - For **spatial correlation** that doesn't follow cluster boundaries, use [Spatial HAC](spatial-hac.md).
    - For **macro panels with $T > N$**, consider [PCSE](pcse.md) instead.

## Quick Example

```python
from panelbox.standard_errors import (
    ClusteredStandardErrors,
    cluster_by_entity,
    cluster_by_time,
    twoway_cluster,
)

# One-way clustering by entity (most common)
result = cluster_by_entity(X, resid, entity_ids)
print(f"SE: {result.std_errors}")
print(f"Clusters: {result.n_clusters}")

# One-way clustering by time
result = cluster_by_time(X, resid, time_ids)

# Two-way clustering (entity + time)
result = twoway_cluster(X, resid, entity_ids, time_ids)
print(f"Clusters: {result.n_clusters}")  # (n_entities, n_times)

# Via model.fit()
from panelbox.models import FixedEffects
model = FixedEffects("y ~ x1 + x2", data, entity="firm", time="year")
results = model.fit(cov_type="clustered")
print(results.summary())
```

## Mathematical Details

### One-Way Clustering

The cluster-robust covariance estimator for $G$ clusters:

$$
V = (X'X)^{-1} \left( \sum_{g=1}^{G} X_g' \hat{u}_g \hat{u}_g' X_g \right) (X'X)^{-1}
$$

where $X_g$ and $\hat{u}_g$ are the design matrix and residuals for cluster $g$. The key insight: the **meat** sums the outer product of the cluster-level score $X_g' \hat{u}_g$, allowing arbitrary correlation within each cluster.

### Two-Way Clustering

Cameron, Gelbach, and Miller (2011) show that two-way clustered variance can be computed as:

$$
V_{\text{2-way}} = V_{\text{entity}} + V_{\text{time}} - V_{\text{intersection}}
$$

where $V_{\text{entity}}$ clusters by entity, $V_{\text{time}}$ clusters by time, and $V_{\text{intersection}}$ clusters by the interaction (entity $\times$ time). The subtraction avoids double-counting.

### Finite-Sample Correction

PanelBox applies the following finite-sample degrees-of-freedom correction (enabled by default):

$$
\text{adjustment} = \frac{G}{G-1} \cdot \frac{N-1}{N-K}
$$

where $G$ is the number of clusters, $N$ is the total number of observations, and $K$ is the number of parameters.

## Configuration Options

### ClusteredStandardErrors Class

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | --- | Design matrix $(n \times k)$ |
| `resid` | `np.ndarray` | --- | Residuals $(n,)$ |
| `clusters` | `np.ndarray` or `tuple` | --- | Cluster IDs: 1D array (one-way) or tuple of two arrays (two-way) |
| `df_correction` | `bool` | `True` | Apply finite-sample correction |

### ClusteredCovarianceResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `cov_matrix` | `np.ndarray` | Cluster-robust covariance matrix $(k \times k)$ |
| `std_errors` | `np.ndarray` | Cluster-robust standard errors $(k,)$ |
| `n_clusters` | `int` or `tuple` | Number of clusters (int for one-way, tuple for two-way) |
| `n_obs` | `int` | Number of observations |
| `n_params` | `int` | Number of parameters |
| `cluster_dims` | `int` | Clustering dimensions (1 or 2) |
| `df_correction` | `bool` | Whether correction was applied |

### Convenience Functions

=== "cluster_by_entity"

    ```python
    result = cluster_by_entity(X, resid, entity_ids, df_correction=True)
    ```

=== "cluster_by_time"

    ```python
    result = cluster_by_time(X, resid, time_ids, df_correction=True)
    ```

=== "twoway_cluster"

    ```python
    result = twoway_cluster(X, resid, entity_ids, time_ids, df_correction=True)
    ```

## Comparing One-Way vs Two-Way Clustering

```python
from panelbox.standard_errors import cluster_by_entity, cluster_by_time, twoway_cluster

# Compare all clustering approaches
result_entity = cluster_by_entity(X, resid, entity_ids)
result_time = cluster_by_time(X, resid, time_ids)
result_twoway = twoway_cluster(X, resid, entity_ids, time_ids)

print("Clustering comparison:")
print(f"  Entity-clustered SE:  {result_entity.std_errors}")
print(f"  Time-clustered SE:    {result_time.std_errors}")
print(f"  Two-way clustered SE: {result_twoway.std_errors}")
print(f"  Entity clusters: {result_entity.n_clusters}")
print(f"  Time clusters:   {result_time.n_clusters}")
print(f"  Two-way clusters: {result_twoway.n_clusters}")
```

## Diagnostics

### Cluster Diagnostic Summary

```python
cse = ClusteredStandardErrors(X, resid, entity_ids)
print(cse.diagnostic_summary())
```

This reports the number of clusters, cluster size distribution (min, max, mean), and warnings if the number of clusters is low.

### How Many Clusters Are Enough?

| Number of Clusters | Reliability | Recommendation |
|-------------------|-------------|----------------|
| $G \geq 50$ | Good | Standard clustered SEs are reliable |
| $20 \leq G < 50$ | Moderate | Use with caution; consider bootstrap |
| $G < 20$ | Poor | Clustered SEs are biased downward |
| $G < 10$ | Unreliable | Use wild cluster bootstrap or T-distribution with $G-1$ df |

## Common Pitfalls

!!! warning "Pitfall 1: Too few clusters"
    With fewer than 20 clusters, standard clustered SEs can severely under-reject. Consider using the wild cluster bootstrap (Cameron, Gelbach, & Miller, 2008) or adjusting critical values.

!!! warning "Pitfall 2: Wrong clustering level"
    Cluster at the level where the treatment or key regressor varies. For a state-level policy, cluster by state --- not by individual. Clustering too finely (e.g., individual level when treatment is at state level) understates standard errors.

!!! warning "Pitfall 3: Unbalanced clusters"
    Very unbalanced cluster sizes can degrade finite-sample performance. The diagnostic summary reports cluster size distribution to help identify this issue.

!!! warning "Pitfall 4: Two-way clustering with few time periods"
    Two-way clustering requires sufficient clusters in **both** dimensions. If $T$ is small (e.g., $T < 20$), the time dimension provides few clusters, and two-way clustering may not improve over entity clustering alone.

## See Also

- [Robust (HC0-HC3)](robust.md) --- When there is no within-cluster correlation
- [Driscoll-Kraay](driscoll-kraay.md) --- When cross-sectional dependence is present
- [PCSE](pcse.md) --- Alternative for macro panels with $T > N$
- [Comparison](comparison.md) --- Compare clustered with other SE types
- [Inference Overview](index.md) --- Decision tree for SE selection

## References

- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414-427.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.
- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies*, 22(1), 435-480.
- Thompson, S. B. (2011). Simple formulas for standard errors that cluster by both firm and time. *Journal of Financial Economics*, 99(1), 1-10.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
