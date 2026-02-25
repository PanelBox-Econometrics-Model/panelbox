---
title: "Impulse Response Functions"
description: "Guide to computing and interpreting Impulse Response Functions (IRFs) in Panel VAR models with PanelBox."
---

# Impulse Response Functions

!!! info "Quick Reference"
    **Class:** `panelbox.var.irf.IRFResult`
    **Method:** `PanelVARResult.irf()`
    **Import:** `from panelbox.var import PanelVAR` (IRFs accessed via results)
    **Stata equivalent:** `irf create`, `irf graph`
    **R equivalent:** `vars::irf()`

## Overview

Impulse Response Functions (IRFs) trace the dynamic effect of a one-time shock to one variable on all variables in the system over time. They are the primary tool for understanding how shocks propagate through a Panel VAR system.

An IRF answers the question: *If variable $j$ receives a one-standard-deviation shock at time $t=0$, how does variable $i$ respond at horizons $h = 0, 1, 2, \ldots, H$?*

Formally, the IRF at horizon $h$ is defined through the Moving Average (MA) representation of the VAR:

$$
Y_t = \sum_{s=0}^{\infty} \Phi_s \varepsilon_{t-s}
$$

where $\Phi_s$ are the MA coefficient matrices computed recursively from the VAR coefficient matrices:

$$
\Phi_0 = I_K, \quad \Phi_h = \sum_{l=1}^{\min(h, p)} A_l \Phi_{h-l}
$$

PanelBox supports two identification methods: **Cholesky** (orthogonalized) and **Generalized** (Pesaran-Shin), with bootstrap confidence intervals.

## Quick Example

```python
from panelbox.var import PanelVARData, PanelVAR

# Estimate model
var_data = PanelVARData(df, endog_vars=["gdp", "inflation", "rate"],
                         entity_col="country", time_col="year", lags=2)
model = PanelVAR(data=var_data)
results = model.fit(cov_type="clustered")

# Compute IRFs with bootstrap confidence intervals
irf = results.irf(
    periods=20,
    method="cholesky",
    ci_method="bootstrap",
    n_bootstrap=500,
    ci_level=0.95,
)

# Access specific response
response = irf["gdp", "inflation"]  # GDP response to inflation shock

# Plot all IRFs
irf.plot()
```

## When to Use

- **Analyzing shock transmission**: How does a monetary policy shock affect output and inflation?
- **Dynamic multiplier analysis**: What is the cumulative effect of a fiscal shock on GDP?
- **Structural analysis**: Identifying the causal chain of macroeconomic shocks
- **Policy evaluation**: Comparing the dynamic effects of different interventions

!!! warning "Key Assumptions"
    - **Stability**: The VAR must be stable (`results.is_stable() == True`) for IRFs to converge to zero
    - **Identification**: Cholesky IRFs depend on variable ordering; Generalized IRFs are order-invariant but shocks are not orthogonal
    - **Homogeneity**: The same impulse responses apply to all entities (pooled coefficient assumption)

## Detailed Guide

### Identification Methods

The raw VAR residuals are typically correlated across equations. To isolate the effect of a shock to a single variable, the residuals must be orthogonalized.

=== "Cholesky (Recursive)"

    The **Cholesky decomposition** of $\Sigma = PP'$ produces a lower-triangular matrix $P$. Orthogonalized IRFs are:

    $$
    \Phi_0^{\text{orth}} = P, \quad \Phi_h^{\text{orth}} = \sum_{l=1}^{\min(h,p)} A_l \Phi_{h-l}^{\text{orth}}
    $$

    **Ordering matters**: Variables listed first are treated as more "exogenous" -- they respond only to their own shocks contemporaneously, while later variables respond to shocks from earlier variables as well.

    ```python
    # Default ordering (order of endog_vars)
    irf = results.irf(periods=20, method="cholesky")

    # Custom ordering: rate is most exogenous, gdp is most endogenous
    irf = results.irf(
        periods=20,
        method="cholesky",
        order=["rate", "inflation", "gdp"],
    )
    ```

    !!! tip "Choosing Variable Ordering"
        Place variables that respond more slowly (or are set by policy) **first**. Common macroeconomic orderings:

        - `["output", "inflation", "interest_rate"]` -- Output-first (Christiano, Eichenbaum & Evans)
        - `["interest_rate", "inflation", "output"]` -- Policy-first

=== "Generalized (Pesaran-Shin)"

    **Generalized IRFs** (Pesaran & Shin, 1998) are **invariant to variable ordering**. The GIRF for a shock to variable $j$ is:

    $$
    \text{GIRF}_j(h) = \frac{1}{\sqrt{\sigma_{jj}}} \Phi_h \Sigma e_j
    $$

    where $\sigma_{jj}$ is the variance of $\varepsilon_j$ and $e_j$ is the $j$-th unit vector.

    ```python
    irf = results.irf(periods=20, method="generalized")
    ```

    **Trade-off**: Generalized IRFs avoid the arbitrary ordering problem but the shocks are **not orthogonal** -- they reflect the actual correlation structure of the residuals.

### Computing IRFs

```python
irf = results.irf(
    periods=20,                       # Forecast horizons
    method="cholesky",                # "cholesky" or "generalized"
    shock_size="one_std",             # "one_std" or numerical value
    cumulative=False,                 # True for accumulated responses
    order=None,                       # Custom ordering (Cholesky only)
    ci_method="bootstrap",            # None or "bootstrap"
    n_bootstrap=500,                  # Bootstrap replications
    ci_level=0.95,                    # Confidence level
    bootstrap_ci_method="percentile", # "percentile" or "bias_corrected"
    n_jobs=-1,                        # Parallel jobs (-1 = all cores)
    seed=42,                          # Reproducibility
    verbose=True,                     # Show progress bar
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `periods` | `int` | `20` | Number of horizons |
| `method` | `str` | `"cholesky"` | `"cholesky"` or `"generalized"` |
| `shock_size` | `str` or `float` | `"one_std"` | Shock magnitude |
| `cumulative` | `bool` | `False` | Accumulated responses ($\Psi_h = \sum_{s=0}^{h} \Phi_s$) |
| `order` | `list[str]` | `None` | Variable ordering (Cholesky only) |
| `ci_method` | `str` | `None` | `None` or `"bootstrap"` |
| `n_bootstrap` | `int` | `500` | Number of bootstrap replications |
| `ci_level` | `float` | `0.95` | Confidence level for CIs |
| `bootstrap_ci_method` | `str` | `"percentile"` | `"percentile"` or `"bias_corrected"` |
| `n_jobs` | `int` | `-1` | Parallel jobs for bootstrap |
| `seed` | `int` | `None` | Random seed for reproducibility |

### Accessing Results

The `IRFResult` object provides flexible access to the computed IRFs.

```python
# Access specific IRF: response of variable i to shock in variable j
response = irf["gdp", "inflation"]       # np.ndarray of shape (periods+1,)

# Convert to DataFrame
df_all = irf.to_dataframe()                        # All IRFs
df_gdp_shock = irf.to_dataframe(impulse="gdp")     # All responses to GDP shock
df_infl_resp = irf.to_dataframe(response="inflation")  # How inflation responds to all shocks
df_pair = irf.to_dataframe(impulse="gdp", response="inflation")  # Single pair

# Summary at selected horizons
print(irf.summary(horizons=[0, 1, 5, 10, 20]))
```

**IRFResult Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `irf_matrix` | `np.ndarray` | Shape `(periods+1, K, K)`. `irf_matrix[h, i, j]` = response of var $i$ to shock in var $j$ at horizon $h$ |
| `var_names` | `list[str]` | Variable names |
| `periods` | `int` | Number of horizons |
| `method` | `str` | Identification method used |
| `cumulative` | `bool` | Whether cumulative |
| `ci_lower` | `np.ndarray` or `None` | Lower CI bounds `(periods+1, K, K)` |
| `ci_upper` | `np.ndarray` or `None` | Upper CI bounds `(periods+1, K, K)` |
| `ci_level` | `float` or `None` | Confidence level |
| `bootstrap_dist` | `np.ndarray` or `None` | Full bootstrap distribution `(n_bootstrap, periods+1, K, K)` |

### Bootstrap Confidence Intervals

Bootstrap CIs account for both parameter uncertainty and estimation error. The procedure:

1. Resample residuals with replacement
2. Reconstruct data using the estimated VAR coefficients and resampled residuals
3. Re-estimate the VAR on the bootstrap sample
4. Compute IRFs from the bootstrap estimates
5. Repeat $B$ times and compute percentile intervals

```python
# Standard percentile bootstrap
irf = results.irf(
    periods=20,
    method="cholesky",
    ci_method="bootstrap",
    n_bootstrap=1000,
    ci_level=0.95,
    bootstrap_ci_method="percentile",
    seed=42,
)

# Bias-corrected percentile (Hall 1992) -- more accurate in small samples
irf_bc = results.irf(
    periods=20,
    method="cholesky",
    ci_method="bootstrap",
    n_bootstrap=1000,
    bootstrap_ci_method="bias_corrected",
    seed=42,
)
```

### Cumulative IRFs

Cumulative IRFs show the **accumulated** response over time:

$$
\Psi_h = \sum_{s=0}^{h} \Phi_s
$$

Useful when variables are in growth rates (differences) and you want to see the level effect.

```python
irf_cum = results.irf(periods=20, cumulative=True, ci_method="bootstrap")
irf_cum.plot()
```

### Visualization

```python
# Plot all IRFs in a grid
irf.plot(backend="matplotlib")

# Plot responses to a specific shock
irf.plot(impulse="gdp", backend="plotly")

# Plot how a specific variable responds to all shocks
irf.plot(response="inflation")

# Customize
irf.plot(
    impulse="gdp",
    response="inflation",
    ci=True,                    # Show confidence intervals
    backend="plotly",           # Interactive plot
    theme="academic",           # Visual theme
)
```

### Interpretation Guidelines

**Significant effects**: An IRF is statistically significant at horizon $h$ if the confidence interval at $h$ **excludes zero**.

| Pattern | Interpretation |
|---------|----------------|
| Positive, decaying | Positive shock has temporary expansionary effect |
| Negative, decaying | Positive shock has temporary contractionary effect |
| Oscillating, converging | Shock triggers cyclical adjustment |
| Persistent (non-zero at long horizons) | Possible non-stationarity or near-unit root |
| Zero throughout | No dynamic effect -- variables are independent |

!!! warning "Common Pitfalls"
    - **Ordering sensitivity**: Always report which ordering you used for Cholesky IRFs, and check robustness with alternative orderings
    - **Wide CIs**: Very wide confidence intervals suggest insufficient data or a weakly identified system. Consider reducing the number of variables or lags
    - **Unstable system**: If `results.is_stable() == False`, IRFs will diverge -- this is a fundamental problem, not a display issue

## Configuration Options

### Method Comparison

| Feature | Cholesky | Generalized |
|---------|----------|-------------|
| Ordering-dependent | Yes | No |
| Shocks orthogonal | Yes | No |
| Structural interpretation | With correct ordering | Limited |
| Best for | Theory-guided analysis | Exploratory analysis |

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| IRF Analysis | Computing and interpreting IRFs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/02_irf_analysis.ipynb) |

## See Also

- [Panel VAR Estimation](estimation.md) -- Model setup and estimation
- [FEVD](fevd.md) -- Variance decomposition (complementary to IRFs)
- [VECM](vecm.md) -- IRFs from cointegrated systems (permanent effects)
- [Granger Causality](granger.md) -- Statistical causality tests
- [Forecasting](forecasting.md) -- Out-of-sample predictions

## References

- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag.
- Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17-29.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.
- Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. *Review of Economics and Statistics*, 80(2), 218-230.
- Hall, P. (1992). *The Bootstrap and Edgeworth Expansion*. Springer Science & Business Media.
