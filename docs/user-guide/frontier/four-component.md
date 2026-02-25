---
title: "Four-Component SFA"
description: "Decomposing persistent and transient inefficiency with the four-component stochastic frontier model in PanelBox"
---

# Four-Component SFA

!!! info "Quick Reference"
    **Class:** `panelbox.frontier.advanced.FourComponentSFA`
    **Import:** `from panelbox.frontier.advanced import FourComponentSFA`
    **Stata equivalent:** `sfpanel` (user-written extensions)
    **R equivalent:** `sfaR::sfacross()` (partial support)

!!! tip "Unique in Python"
    PanelBox is the **only Python library** that implements the four-component SFA model. This model is available in Stata through user-written extensions and partially in R through the `sfaR` package, but PanelBox provides the most complete Python implementation.

## Overview

Standard SFA models estimate a single inefficiency term $u_{it}$ that combines all sources of underperformance. However, in practice, inefficiency has distinct components with different policy implications:

- **Persistent inefficiency** ($\eta_i$): structural, long-run inefficiency due to organizational design, corporate culture, or institutional constraints. This component is time-invariant and requires fundamental reforms to address.
- **Transient inefficiency** ($u_{it}$): short-run, time-varying inefficiency due to temporary managerial decisions, demand shocks, or operational disruptions. This component can be addressed through day-to-day management improvements.

The four-component model (Kumbhakar, Lien & Hardaker, 2014; Colombi et al., 2014) decomposes the error into four parts, providing a richer picture of firm performance and more targeted policy recommendations.

## Quick Example

```python
from panelbox.frontier.advanced import FourComponentSFA

model = FourComponentSFA(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier_type="production",
)
result = model.fit(verbose=True)

# Efficiency decomposition
te_persistent = result.persistent_efficiency()
te_transient = result.transient_efficiency()
te_overall = result.overall_efficiency()

print(f"Mean persistent efficiency: {te_persistent['persistent_efficiency'].mean():.4f}")
print(f"Mean transient efficiency:  {te_transient['transient_efficiency'].mean():.4f}")
print(f"Mean overall efficiency:    {te_overall['overall_efficiency'].mean():.4f}")
```

## When to Use

- You want to distinguish **structural** (persistent) from **managerial** (transient) inefficiency
- Policy recommendations need to differentiate between **long-run reforms** and **short-run management improvements**
- You need to understand **why** overall efficiency is low -- is it structural or managerial?
- Your panel has sufficient time periods ($T \geq 5$) for reliable within-entity estimation

!!! warning "Key Assumptions"
    - Requires panel data with both entity and time identifiers
    - Persistent inefficiency $\eta_i$ is assumed time-invariant (entity-specific)
    - Transient inefficiency $u_{it}$ is assumed half-normally distributed
    - Random heterogeneity $\mu_i$ is assumed normally distributed
    - Sufficient within-entity variation is needed for identification

## The Four-Component Model

### Model Specification

$$y_{it} = X_{it}'\beta + \mu_i - \eta_i + v_{it} - u_{it}$$

| Component | Symbol | Distribution | Interpretation |
|---|---|---|---|
| Random heterogeneity | $\mu_i$ | $N(0, \sigma_\mu^2)$ | Technology differences, unobserved firm characteristics |
| Persistent inefficiency | $\eta_i$ | $N^+(0, \sigma_\eta^2)$ | Structural, long-run underperformance |
| Random noise | $v_{it}$ | $N(0, \sigma_v^2)$ | Weather, measurement error, luck |
| Transient inefficiency | $u_{it}$ | $N^+(0, \sigma_u^2)$ | Short-run managerial inefficiency |

### Three-Step Estimation

The four-component model is estimated sequentially:

**Step 1: Within (FE) Estimator**

Demean data within entities to estimate $\beta$ and recover entity-level effects $\alpha_i$ and time-varying residuals $\varepsilon_{it}$:

$$y_{it} - \bar{y}_i = (X_{it} - \bar{X}_i)'\beta + (\varepsilon_{it} - \bar{\varepsilon}_i)$$

where $\alpha_i = \mu_i - \eta_i$ and $\varepsilon_{it} = v_{it} - u_{it}$.

**Step 2: Separate Transient Inefficiency**

Apply cross-sectional SFA to the residuals $\hat{\varepsilon}_{it}$ to separate $v_{it}$ and $u_{it}$:

$$\hat{\varepsilon}_{it} = v_{it} - u_{it}$$

Uses half-normal MLE and JLMS estimator for $\hat{u}_{it}$.

**Step 3: Separate Persistent Inefficiency**

Apply cross-sectional SFA to the estimated fixed effects $\hat{\alpha}_i$ to separate $\mu_i$ and $\eta_i$:

$$\hat{\alpha}_i = \mu_i - \eta_i$$

Uses half-normal MLE and JLMS estimator for $\hat{\eta}_i$.

### Efficiency Types

**Persistent efficiency** (structural capability):

$$TE_{p,i} = \exp(-\eta_i) \in (0, 1]$$

**Transient efficiency** (short-run management):

$$TE_{t,it} = \exp(-u_{it}) \in (0, 1]$$

**Overall efficiency** (combined):

$$TE_{o,it} = TE_{p,i} \times TE_{t,it}$$

## Detailed Guide

### Full Estimation Example

```python
from panelbox.frontier.advanced import FourComponentSFA

# Fit the model
model = FourComponentSFA(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier_type="production",
)
result = model.fit(verbose=False)

# Print full summary
result.print_summary()
```

The `print_summary()` method displays:

- Sample information (observations, entities, periods)
- Variance components ($\sigma_v^2, \sigma_u^2, \sigma_\mu^2, \sigma_\eta^2$) and their shares
- Efficiency summary statistics (persistent, transient, overall)

### Variance Components

```python
# Individual variance components
print(f"sigma_v  (noise):              {result.sigma_v:.4f}")
print(f"sigma_u  (transient ineff.):   {result.sigma_u:.4f}")
print(f"sigma_mu (heterogeneity):      {result.sigma_mu:.4f}")
print(f"sigma_eta (persistent ineff.): {result.sigma_eta:.4f}")

# Variance shares
total_var = (result.sigma_v**2 + result.sigma_u**2
             + result.sigma_mu**2 + result.sigma_eta**2)
print(f"\nVariance shares:")
print(f"  Noise:              {100 * result.sigma_v**2 / total_var:.1f}%")
print(f"  Transient ineff.:   {100 * result.sigma_u**2 / total_var:.1f}%")
print(f"  Heterogeneity:      {100 * result.sigma_mu**2 / total_var:.1f}%")
print(f"  Persistent ineff.:  {100 * result.sigma_eta**2 / total_var:.1f}%")
```

### Efficiency Decomposition

```python
# Get all efficiency types
te_p = result.persistent_efficiency()
te_t = result.transient_efficiency()
te_o = result.overall_efficiency()

# Full decomposition table
decomp = result.decomposition()
print(decomp.head())
# Columns: entity, time, mu_i, eta_i, u_it, v_it

# Overall efficiency with both components
print(te_o.head())
# Columns: entity, time, overall_efficiency,
#           persistent_efficiency, transient_efficiency
```

### Bootstrap Confidence Intervals

```python
boot = result.bootstrap(
    n_bootstrap=100,
    confidence_level=0.95,
    random_state=42,
    verbose=True,
)

# Persistent efficiency with CIs
pers_ci = boot.persistent_efficiency_ci()
print(pers_ci.head())
# Columns: entity, persistent_efficiency, ci_lower, ci_upper

# Variance component CIs
boot.print_summary()
```

### Policy Implications

The decomposition directly informs policy:

| Scenario | Persistent TE | Transient TE | Policy Response |
|---|---|---|---|
| High / High | Structurally efficient, well-managed | Maintain current practices |
| High / Low | Good structure, poor daily management | Management training, operational improvements |
| Low / High | Structural problems, good management | Institutional reforms, technology upgrades |
| Low / Low | Both structural and managerial issues | Comprehensive restructuring |

## Configuration Options

### FourComponentSFA Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `DataFrame` | required | Panel DataFrame |
| `depvar` | `str` | required | Dependent variable (in logs) |
| `exog` | `list[str]` | required | Exogenous variable names |
| `entity` | `str` | required | Entity identifier column |
| `time` | `str` | required | Time identifier column |
| `frontier_type` | `str` | `"production"` | `"production"` or `"cost"` |

### fit() Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `verbose` | `bool` | `False` | Print step-by-step estimation progress |

### FourComponentResult Methods

| Method | Returns | Description |
|---|---|---|
| `persistent_efficiency()` | `DataFrame` | $TE_{p,i} = \exp(-\eta_i)$ per entity |
| `transient_efficiency()` | `DataFrame` | $TE_{t,it} = \exp(-u_{it})$ per observation |
| `overall_efficiency()` | `DataFrame` | $TE_{o,it} = TE_p \times TE_t$ with both components |
| `efficiency(estimator="bc")` | `DataFrame` | Overall efficiency (API compatibility) |
| `decomposition()` | `DataFrame` | Full 4-component decomposition table |
| `print_summary()` | `None` | Display formatted results |
| `tfp_decomposition(periods)` | `TFPDecomposition` | TFP growth decomposition |
| `bootstrap(n_bootstrap, ...)` | `BootstrapResult` | Bootstrap confidence intervals |

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| Four-Component SFA | Persistent vs transient efficiency analysis | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/03_four_component_tfp.ipynb) |

## See Also

- [Production and Cost Frontiers](production-cost.md) -- SFA fundamentals
- [Panel SFA Models](panel-sfa.md) -- Classical panel models
- [True Models (TFE/TRE)](true-models.md) -- Two/three-component heterogeneity separation
- [TFP Decomposition](tfp.md) -- Productivity growth analysis using SFA results
- [SFA Diagnostics](diagnostics.md) -- Model validation and testing

## References

- Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014). Technical efficiency in competing panel data models: a study of Norwegian grain farming. *Journal of Productivity Analysis*, 41(2), 321-337.
- Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014). Closed-skew normality in stochastic frontiers with individual effects and long/short-run efficiency. *Journal of Productivity Analysis*, 42, 123-136.
- Kumbhakar, S. C., & Lovell, C. A. K. (2000). *Stochastic Frontier Analysis*. Cambridge University Press.
- Greene, W. H. (2005). Reconsidering heterogeneity in panel data estimators of the stochastic frontier model. *Journal of Econometrics*, 126(2), 269-303.
