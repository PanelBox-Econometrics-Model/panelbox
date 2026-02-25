---
title: "Inference & Standard Errors"
description: "Choosing the right standard errors for panel data models in PanelBox - from robust HC to spatial HAC estimators."
---

# Inference & Standard Errors

Correct statistical inference in panel data models depends on choosing appropriate standard errors. Classical OLS standard errors assume homoskedastic, independent errors --- assumptions that rarely hold in panel data. PanelBox provides 8 types of standard errors, all built on a unified sandwich estimator framework.

## Why Standard Errors Matter

Standard errors determine hypothesis tests, confidence intervals, and p-values. Using the wrong standard errors leads to:

- **Overly optimistic inference** (too many "significant" results) when errors are correlated but treated as independent
- **Overly conservative inference** (too few "significant" results) when using an unnecessarily complex SE estimator with limited data
- **Invalid confidence intervals** that don't achieve their nominal coverage

!!! warning "The most common mistake"
    Using classical (non-robust) standard errors in panel data almost always understates uncertainty. At minimum, use clustered standard errors by entity.

## The Sandwich Estimator Framework

All robust standard errors in PanelBox follow the **sandwich estimator** structure:

$$
V = \underbrace{(X'X)^{-1}}_{\text{Bread}} \times \underbrace{\hat{\Omega}}_{\text{Meat}} \times \underbrace{(X'X)^{-1}}_{\text{Bread}}
$$

The **bread** $(X'X)^{-1}$ is the same across all types. What changes is the **meat** $\hat{\Omega}$, which captures the assumed error structure:

| SE Type | Meat $\hat{\Omega}$ | Handles |
|---------|---------------------|---------|
| Classical | $\hat{\sigma}^2 X'X$ | Nothing (homoskedastic, independent) |
| [Robust (HC)](robust.md) | $X' \text{diag}(\hat{e}_i^2) X$ | Heteroskedasticity |
| [Clustered](clustered.md) | $\sum_g (X_g' \hat{u}_g)(X_g' \hat{u}_g)'$ | Within-cluster correlation |
| [Driscoll-Kraay](driscoll-kraay.md) | HAC on time-averaged moments | Heteroskedasticity + autocorrelation + cross-sectional dependence |
| [Newey-West](newey-west.md) | $\hat{\Gamma}_0 + \sum_j w_j(\hat{\Gamma}_j + \hat{\Gamma}_j')$ | Heteroskedasticity + autocorrelation |
| [PCSE](pcse.md) | $X'(\hat{\Sigma} \otimes I_T)X$ | Cross-sectional correlation (macro panels) |
| [Spatial HAC](spatial-hac.md) | Distance-weighted cross-products | Spatial + temporal correlation |
| [MLE Sandwich](mle-variance.md) | $\sum s_i s_i'$ (score outer product) | Misspecification in nonlinear models |

## Decision Tree: Choosing the Right Standard Errors

Use the following guide to select the appropriate SE type for your data:

| Data Characteristic | Recommended SE | PanelBox `cov_type` | Page |
|---------------------|----------------|---------------------|------|
| Heteroskedasticity only | Robust HC1 | `"robust"` or `"hc1"` | [Robust](robust.md) |
| Within-entity correlation | Clustered (entity) | `"clustered"` | [Clustered](clustered.md) |
| Entity + time correlation | Two-way clustered | `"twoway"` | [Clustered](clustered.md) |
| Cross-sectional dependence | Driscoll-Kraay | `"driscoll_kraay"` | [Driscoll-Kraay](driscoll-kraay.md) |
| Autocorrelation (time series) | Newey-West | `"newey_west"` | [Newey-West](newey-west.md) |
| Macro panels ($T > N$) | PCSE | `"pcse"` | [PCSE](pcse.md) |
| Spatial correlation | Spatial HAC | via `SpatialHAC` class | [Spatial HAC](spatial-hac.md) |
| Nonlinear models (MLE) | MLE sandwich | `"robust"` | [MLE Variance](mle-variance.md) |

!!! tip "Default recommendation"
    For most panel data applications, **cluster by entity** is a safe default. It allows for arbitrary within-entity correlation across time, which is the most common dependence structure.

## How PanelBox Integrates Standard Errors

### High-Level API: `model.fit(cov_type=...)`

The simplest way to use robust standard errors is through the `cov_type` parameter:

```python
from panelbox.models import FixedEffects

model = FixedEffects("y ~ x1 + x2", data, entity="firm", time="year")

# Classical standard errors (default)
results = model.fit()

# Robust HC1 standard errors
results = model.fit(cov_type="robust")

# Clustered by entity
results = model.fit(cov_type="clustered")

# Driscoll-Kraay with custom lags
results = model.fit(cov_type="driscoll_kraay", max_lags=3)
```

### Low-Level API: Direct Classes and Functions

For more control, use the classes and convenience functions directly:

```python
from panelbox.standard_errors import (
    RobustStandardErrors, robust_covariance,
    ClusteredStandardErrors, cluster_by_entity, twoway_cluster,
    DriscollKraayStandardErrors, driscoll_kraay,
    NeweyWestStandardErrors, newey_west,
    PanelCorrectedStandardErrors, pcse,
    SpatialHAC,
    StandardErrorComparison,
)

# All result objects share a common interface:
result = robust_covariance(X, resid, method="HC1")
result.cov_matrix   # np.ndarray (k x k) - Covariance matrix
result.std_errors   # np.ndarray (k,)    - Standard errors
result.n_obs        # int                - Number of observations
result.n_params     # int                - Number of parameters
```

### Comparing SE Methods

Use `StandardErrorComparison` to systematically compare different SE types:

```python
from panelbox.standard_errors import StandardErrorComparison

comparison = StandardErrorComparison(results)
comp = comparison.compare_all()

# Examine how inference changes across SE types
print(comp.se_comparison)   # SEs by type
print(comp.significance)    # Significance stars

# Visualize differences
comparison.plot_comparison(comp, alpha=0.05)
```

See the [Comparison](comparison.md) page for details.

## Quick Example

```python
import panelbox as pb

# Load panel data
data = pb.datasets.load_grunfeld()

# Fit Fixed Effects model with different SE types
model = pb.FixedEffects("invest ~ value + capital", data, entity="firm", time="year")

# Compare standard errors
results_classical = model.fit()
results_robust = model.fit(cov_type="robust")
results_clustered = model.fit(cov_type="clustered")

# Print comparison
print("Classical SE: ", results_classical.std_errors.values)
print("Robust SE:    ", results_robust.std_errors.values)
print("Clustered SE: ", results_clustered.std_errors.values)
```

## Common Mistakes

!!! danger "Pitfall 1: Too few clusters"
    Clustered standard errors require a sufficient number of clusters (rule of thumb: $G \geq 50$). With fewer clusters, SEs are biased downward. Consider wild cluster bootstrap when $G < 50$.

!!! danger "Pitfall 2: Wrong clustering dimension"
    Cluster at the level where treatment varies. If a policy intervention varies by state, cluster by state --- not by individual.

!!! danger "Pitfall 3: Ignoring cross-sectional dependence"
    Entity-clustered SEs do not account for cross-sectional dependence (common shocks). If entities are affected by common factors (e.g., macroeconomic shocks), use Driscoll-Kraay or two-way clustering.

!!! danger "Pitfall 4: PCSE with micro panels"
    PCSE requires $T > N$. Using PCSE with typical micro panels ($N \gg T$) produces unreliable results. Use clustered SEs instead.

## Software Equivalents

| PanelBox | Stata | R |
|----------|-------|---|
| `cov_type="robust"` | `vce(robust)` | `sandwich::vcovHC()` |
| `cov_type="clustered"` | `vce(cluster id)` | `plm::vcovHC(cluster="group")` |
| `cov_type="driscoll_kraay"` | `xtscc` (Hoechle 2007) | `plm::vcovSCC()` |
| `cov_type="newey_west"` | `newey` | `sandwich::NeweyWest()` |
| `cov_type="pcse"` | `xtpcse` | `pcse::pcse()` |
| `SpatialHAC` | `acreg` (Colella et al.) | `conleyreg::conley()` |

## Learning Path

<div class="grid cards" markdown>

-   :material-shield-check: **Start here**

    ---

    [Robust (HC0-HC3)](robust.md) --- Heteroskedasticity-robust standard errors. The foundation for all other methods.

-   :material-group: **Most common**

    ---

    [Clustered](clustered.md) --- One-way and two-way clustering. The workhorse for panel data applications.

-   :material-waves: **Time dependence**

    ---

    [Driscoll-Kraay](driscoll-kraay.md) and [Newey-West](newey-west.md) --- HAC estimators for autocorrelation and cross-sectional dependence.

-   :material-map-marker: **Spatial data**

    ---

    [Spatial HAC](spatial-hac.md) --- Conley (1999) for geographically correlated errors.

-   :material-earth: **Macro panels**

    ---

    [PCSE](pcse.md) --- Beck & Katz (1995) for time-series cross-section data with $T > N$.

-   :material-function-variant: **Nonlinear models**

    ---

    [MLE Variance](mle-variance.md) --- Sandwich, delta method, and bootstrap for MLE estimators.

-   :material-compare-horizontal: **Compare all**

    ---

    [Comparison](comparison.md) --- Systematically compare SE methods and assess inference sensitivity.

</div>

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- Arellano, M. (1987). Computing robust standard errors for within-groups estimators. *Oxford Bulletin of Economics and Statistics*, 49(4), 431-434.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Beck, N., & Katz, J. N. (1995). What to do (and not to do) with time-series cross-section data. *American Political Science Review*, 89(3), 634-647.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549-560.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1-45.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.
- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies*, 22(1), 435-480.
