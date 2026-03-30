---
title: "Static Models API"
description: "API reference for panelbox.models.static — PooledOLS, FixedEffects, RandomEffects, Between, FirstDifference, MeanGroup, PooledMeanGroup, SUR"
---

# Static Models API Reference

!!! info "Module"
    **Import**: `from panelbox.models.static import PooledOLS, FixedEffects, RandomEffects, BetweenEstimator, FirstDifferenceEstimator, MeanGroupEstimator, PooledMeanGroupEstimator, SUR`
    **Source**: `panelbox/models/static/`

## Overview

Static panel models are the workhorses of panel data econometrics. All estimators share a consistent interface: construct with a formula and data, then call `.fit()` to obtain results.

| Estimator | Description | Use Case |
|-----------|-------------|----------|
| `PooledOLS` | Ordinary least squares ignoring panel structure | Baseline comparison |
| `FixedEffects` | Within estimator eliminating entity-specific intercepts | Time-invariant unobserved heterogeneity |
| `RandomEffects` | GLS with random entity effects | Uncorrelated unobserved effects |
| `BetweenEstimator` | OLS on entity means | Cross-sectional variation |
| `FirstDifferenceEstimator` | OLS on first-differenced data | Alternative to FE for T=2 |
| `MeanGroupEstimator` | Average of entity-specific OLS regressions | Slope heterogeneity (Pesaran & Smith 1995) |
| `PooledMeanGroupEstimator` | ECM with homogeneous long-run coefficients | Long-run homogeneity + short-run heterogeneity (Pesaran, Shin & Smith 1999) |
| `SUR` | Seemingly Unrelated Regressions (Zellner 1962) | Cross-equation correlated errors + different regressors |

## Common Constructor Pattern

All static models share the same constructor signature:

```python
ModelClass(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | R-style formula, e.g. `"y ~ x1 + x2"` |
| `data` | `pd.DataFrame` | *required* | Panel data DataFrame |
| `entity_col` | `str` | *required* | Column identifying entities |
| `time_col` | `str` | *required* | Column identifying time periods |
| `weights` | `np.ndarray \| None` | `None` | Observation weights for WLS |

## Common `.fit()` Method

```python
model.fit(cov_type: str = "nonrobust", **cov_kwds) -> PanelResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cov_type` | `str` | `"nonrobust"` | Covariance estimator type |
| `**cov_kwds` | `dict` | — | Additional keyword arguments for the covariance estimator |

### Available `cov_type` Options

| Value | Description |
|-------|-------------|
| `"nonrobust"` | Classical OLS/GLS standard errors |
| `"robust"` | Heteroskedasticity-robust (HC1) |
| `"hc0"` -- `"hc3"` | White heteroskedasticity-consistent variants |
| `"clustered"` | Cluster-robust by entity (default clustering) |
| `"twoway"` | Two-way clustering by entity and time |
| `"driscoll_kraay"` | Driscoll-Kraay (cross-sectionally robust) |
| `"newey_west"` | Newey-West HAC |
| `"pcse"` | Panel-corrected standard errors (Beck-Katz) |

**Returns**: [`PanelResults`](core.md#panelresults)

---

## Classes

### PooledOLS

Pooled Ordinary Least Squares. Treats all observations as independent, ignoring the panel structure. Useful as a baseline for comparison with panel estimators.

```python
PooledOLS(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

#### Example

```python
from panelbox import PooledOLS, load_grunfeld

data = load_grunfeld()
model = PooledOLS("invest ~ value + capital", data, "firm", "year")
result = model.fit(cov_type="clustered")
result.summary()
```

---

### FixedEffects

Within estimator that eliminates entity-specific (and optionally time-specific) fixed effects by demeaning.

```python
FixedEffects(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    entity_effects: bool = True,
    time_effects: bool = False,
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_effects` | `bool` | `True` | Include entity fixed effects |
| `time_effects` | `bool` | `False` | Include time fixed effects (two-way FE) |

!!! tip "When to use Fixed Effects"
    Use FE when you suspect unobserved entity-level heterogeneity is **correlated** with the regressors. The Hausman test can help decide between FE and RE.

#### Example

```python
from panelbox import FixedEffects, load_grunfeld

data = load_grunfeld()

# One-way entity FE
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
result = fe.fit(cov_type="robust")

# Two-way FE (entity + time)
fe2 = FixedEffects(
    "invest ~ value + capital", data, "firm", "year",
    entity_effects=True, time_effects=True
)
result2 = fe2.fit(cov_type="clustered")
```

---

### RandomEffects

GLS estimator with random entity effects. Uses the Swamy-Arora variance decomposition by default.

```python
RandomEffects(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    variance_estimator: str = "swamy-arora",
    weights: np.ndarray | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variance_estimator` | `str` | `"swamy-arora"` | Method for estimating variance components |

!!! tip "When to use Random Effects"
    Use RE when unobserved heterogeneity is **uncorrelated** with the regressors. RE is more efficient than FE under this assumption. Verify with the Hausman test.

#### Example

```python
from panelbox import RandomEffects, load_grunfeld

data = load_grunfeld()
model = RandomEffects("invest ~ value + capital", data, "firm", "year")
result = model.fit()
result.summary()
```

---

### BetweenEstimator

OLS regression on entity means (cross-sectional variation only). Estimates the relationship using between-entity variation by averaging all observations within each entity.

```python
BetweenEstimator(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

#### Example

```python
from panelbox import BetweenEstimator, load_grunfeld

data = load_grunfeld()
model = BetweenEstimator("invest ~ value + capital", data, "firm", "year")
result = model.fit()
result.summary()
```

---

### FirstDifferenceEstimator

OLS on first-differenced data. Eliminates entity fixed effects by differencing consecutive observations. Equivalent to Fixed Effects when T=2.

```python
FirstDifferenceEstimator(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
)
```

!!! tip "FD vs FE"
    First Difference uses only adjacent-period variation, while FE uses all within-entity variation. FD is more robust to serial correlation in errors but less efficient when errors are not a random walk.

#### Example

```python
from panelbox import FirstDifferenceEstimator, load_grunfeld

data = load_grunfeld()
model = FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year")
result = model.fit(cov_type="robust")
result.summary()
```

---

## Comparison Example

```python
from panelbox import (
    PooledOLS, FixedEffects, RandomEffects,
    BetweenEstimator, FirstDifferenceEstimator,
    load_grunfeld,
)

data = load_grunfeld()
formula = "invest ~ value + capital"

models = {
    "Pooled OLS": PooledOLS(formula, data, "firm", "year"),
    "Fixed Effects": FixedEffects(formula, data, "firm", "year"),
    "Random Effects": RandomEffects(formula, data, "firm", "year"),
    "Between": BetweenEstimator(formula, data, "firm", "year"),
    "First Difference": FirstDifferenceEstimator(formula, data, "firm", "year"),
}

for name, model in models.items():
    result = model.fit()
    print(f"{name:20s}  R-sq={result.rsquared:.4f}  N={result.nobs}")
```

---

### MeanGroupEstimator

Mean Group estimator (Pesaran & Smith, 1995). Estimates entity-specific OLS regressions and averages coefficients across entities. Consistent under slope heterogeneity where FE/RE are inconsistent.

```python
MeanGroupEstimator(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    weights: np.ndarray | None = None,
    min_obs_per_entity: int = 10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | R-style formula, e.g. `"y ~ x1 + x2"` |
| `data` | `pd.DataFrame` | *required* | Panel data DataFrame |
| `entity_col` | `str` | *required* | Column identifying entities |
| `time_col` | `str` | *required* | Column identifying time periods |
| `weights` | `np.ndarray \| None` | `None` | Entity-level weights for weighted MG estimation |
| `min_obs_per_entity` | `int` | `10` | Minimum observations per entity to include |

**Returns**: `MeanGroupResults` (extends `PanelResults`)

#### MeanGroupResults

| Attribute | Type | Description |
|-----------|------|-------------|
| `entity_params` | `dict` | `{entity_id: pd.Series}` of entity-specific coefficients |
| `entity_std_errors` | `dict` | `{entity_id: pd.Series}` of entity-specific standard errors |
| `entity_rsquared` | `dict` | `{entity_id: float}` of entity-specific R-squared |
| `n_entities_used` | `int` | Number of entities used in estimation |
| `entities_excluded` | `list` | Entities excluded due to insufficient observations |
| `swamy_test_result` | `dict` | Swamy (1970) slope homogeneity test (`statistic`, `p_value`, `df`) |

| Method | Description |
|--------|-------------|
| `entity_summary(entity_id)` | Print OLS summary for a single entity |
| `coefficient_table()` | DataFrame of all entity-specific coefficients |
| `plot_coefficient_distribution(variable)` | Boxplot of entity coefficients for a variable |

#### Example

```python
from panelbox.models.static import MeanGroupEstimator

data = ...  # panel DataFrame with columns: country, year, y, x1, x2
mg = MeanGroupEstimator("y ~ x1 + x2", data, "country", "year")
result = mg.fit()

# Average coefficients and Swamy test
result.summary()
print(result.swamy_test_result)

# Entity-level detail
result.coefficient_table()
result.entity_summary("USA")
result.plot_coefficient_distribution("x1")
```

---

### PooledMeanGroupEstimator

Pooled Mean Group estimator (Pesaran, Shin & Smith, 1999). Estimates an error-correction model where long-run coefficients are homogeneous across entities while short-run dynamics are entity-specific.

```python
PooledMeanGroupEstimator(
    formula: str,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    lags: int = 1,
    max_iter: int = 100,
    tol: float = 1e-5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str` | *required* | R-style formula, e.g. `"y ~ x1 + x2"` |
| `data` | `pd.DataFrame` | *required* | Panel data DataFrame |
| `entity_col` | `str` | *required* | Column identifying entities |
| `time_col` | `str` | *required* | Column identifying time periods |
| `lags` | `int` | `1` | Number of lags for short-run dynamics |
| `max_iter` | `int` | `100` | Maximum optimization iterations |
| `tol` | `float` | `1e-5` | Convergence tolerance |

#### Example

```python
from panelbox.models.static import PooledMeanGroupEstimator

pmg = PooledMeanGroupEstimator(
    "y ~ x1 + x2", data, "country", "year", lags=1
)
result = pmg.fit()
result.summary()
```

---

### hausman_mg_pmg

Hausman test comparing MG and PMG estimators. Tests whether the long-run homogeneity restriction imposed by PMG is valid.

```python
from panelbox.models.static import hausman_mg_pmg

test = hausman_mg_pmg(mg_result, pmg_result)
# Returns dict with 'statistic', 'p_value', 'df'
```

| Result | Interpretation |
|--------|---------------|
| Fail to reject $H_0$ | PMG preferred (efficient + consistent) |
| Reject $H_0$ | MG preferred (consistent under heterogeneity) |

---

### SUR

Seemingly Unrelated Regressions (Zellner, 1962). Treats each entity as a separate equation in a system with cross-equation correlated errors. Estimates via Feasible GLS using the Kronecker structure of the covariance matrix.

```python
SUR(
    formula: str | dict,
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    homogeneous: bool = False,
    iterate: bool = False,
    max_iter: int = 100,
    tol: float = 1e-6,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | `str \| dict` | *required* | R-style formula (str) applied to all entities, or dict mapping entity IDs to per-entity formulas |
| `data` | `pd.DataFrame` | *required* | Panel data DataFrame |
| `entity_col` | `str` | *required* | Column identifying entities (each entity = one equation) |
| `time_col` | `str` | *required* | Column identifying time periods |
| `homogeneous` | `bool` | `False` | If True, constrain all entities to share the same coefficients |
| `iterate` | `bool` | `False` | If True, iterate FGLS until convergence (ISUR ≈ MLE) |
| `max_iter` | `int` | `100` | Maximum iterations for iterated SUR |
| `tol` | `float` | `1e-6` | Convergence tolerance (relative change in beta) |

!!! tip "When to use SUR"
    SUR is most useful when entities have **different regressors** and **correlated errors**. When all entities share the same regressors, SUR point estimates equal OLS — no efficiency gain. Use the Breusch-Pagan test in the results to verify that cross-equation correlation is significant.

#### `.fit()` Method

```python
sur.fit() -> SURResults
```

The SUR `.fit()` method does not take `cov_type` — covariances are determined by the GLS structure.

#### SURResults

Extends `PanelResults` with system-level diagnostics.

| Attribute | Type | Description |
|-----------|------|-------------|
| `entity_params` | `dict` | `{entity_id: pd.Series}` per-entity SUR coefficient estimates |
| `entity_std_errors` | `dict` | `{entity_id: pd.Series}` per-entity standard errors |
| `entity_rsquared` | `dict` | `{entity_id: float}` per-entity R-squared |
| `sigma_matrix` | `np.ndarray` | Cross-equation covariance matrix ($N \times N$) |
| `correlation_matrix` | `np.ndarray` | Cross-equation correlation matrix ($N \times N$) |
| `system_rsquared` | `float` | McElroy (1977) system R-squared |
| `n_iterations` | `int` | FGLS iterations performed (0 if not iterated) |
| `converged` | `bool` | Whether iterated SUR converged |
| `ols_params` | `dict` | `{entity_id: pd.Series}` pre-GLS OLS coefficients |
| `efficiency_gain` | `dict` | `{entity_id: pd.Series}` ratio SE(SUR)/SE(OLS) — values < 1 indicate gain |
| `bp_independence_test` | `dict` | Breusch-Pagan test (`statistic`, `pvalue`, `df`) |

| Method | Description |
|--------|-------------|
| `system_summary()` | Print system-level summary with Sigma, BP test, efficiency gains |
| `equation_summary(entity_id)` | Print SUR vs OLS comparison for a single entity |
| `plot_correlation_matrix(ax=None)` | Heatmap of cross-equation correlations |

#### Examples

**String formula (same for all entities):**

```python
import panelbox as pb

data = pb.load_grunfeld()
sur = pb.SUR("invest ~ value + capital", data, "firm", "year")
result = sur.fit()
print(result.system_summary())

# Check if SUR helped
print(f"BP test p-value: {result.bp_independence_test['pvalue']:.4f}")
```

**Dict formula (per-entity specifications):**

```python
sur = pb.SUR(
    formula={
        "General Motors": "invest ~ value + capital",
        "Chrysler": "invest ~ value",
        "General Electric": "invest ~ capital",
    },
    data=data,
    entity_col="firm",
    time_col="year",
)
result = sur.fit()
result.equation_summary("General Motors")
```

**Iterated SUR (MLE-equivalent under normality):**

```python
sur = pb.SUR(
    "invest ~ value + capital", data, "firm", "year",
    iterate=True, max_iter=200, tol=1e-8,
)
result = sur.fit()
print(f"Converged: {result.converged}, iterations: {result.n_iterations}")
```

---

## See Also

- [Core API](core.md) — `PanelResults` attributes and methods
- [Dynamic Models API](dynamic-models.md) — LSDVC bias-corrected estimator for dynamic panels
- [GMM API](gmm.md) — Arellano-Bond, Blundell-Bond GMM estimators
- [IV API](iv.md) — Instrumental Variables extension
- [Standard Errors](standard-errors.md) — All covariance estimator types
- [Tutorials: Static Models](../tutorials/static-models.md) — Step-by-step guide
- [Validation API](validation.md) — Hausman test for FE vs RE
- [Mean Group Theory](../theory/mean-group-estimator.md) — Derivations and asymptotic properties
- [Slope Heterogeneity Guide](../user-guide/slope-heterogeneity.md) — When to use MG vs FE vs RE
- [SUR Theory](../theory/sur-zellner.md) — Zellner's SUR derivation and properties
- [SUR Estimation Guide](../user-guide/sur-estimation.md) — When to use SUR vs FE vs MG
