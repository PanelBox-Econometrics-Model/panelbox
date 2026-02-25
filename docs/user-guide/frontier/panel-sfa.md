---
title: "Panel SFA Models"
description: "Time-invariant and time-varying panel stochastic frontier models including Pitt-Lee, Battese-Coelli, CSS, Kumbhakar, and Lee-Schmidt"
---

# Panel SFA Models

!!! info "Quick Reference"
    **Class:** `panelbox.frontier.StochasticFrontier`
    **Import:** `from panelbox.frontier import StochasticFrontier`
    **Stata equivalent:** `sfpanel`
    **R equivalent:** `frontier::sfa()`, `plm` + SFA

## Overview

Panel data offers significant advantages for stochastic frontier analysis by exploiting repeated observations over time. Panel SFA models can separate persistent from transient inefficiency, allow inefficiency to change over time, and improve the precision of efficiency estimates by pooling information across periods.

PanelBox implements six panel SFA model types, ranging from the simple time-invariant Pitt-Lee (1981) model to the distribution-free Cornwell-Schmidt-Sickles (1990) approach. All models are accessed through the unified `StochasticFrontier` class by specifying the `model_type` parameter.

The key decision in panel SFA is whether inefficiency varies over time and, if so, how. Time-invariant models are simpler but assume firms cannot improve their efficiency. Time-varying models are more flexible but require additional parameters.

## Quick Example

```python
from panelbox.frontier import StochasticFrontier

# Battese-Coelli (1992) model with time-varying inefficiency
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="bc92",
)
results = model.fit()
print(results.summary())
```

## When to Use

- You have **panel data** (repeated observations on the same entities over time)
- You want to estimate **entity-specific efficiency** with improved precision
- You need to model **time-varying inefficiency** (efficiency improving or deteriorating)
- You want to include **determinants of inefficiency** (BC95 model)
- You prefer a **distribution-free** approach (CSS model)

!!! warning "Key Assumptions"
    - Panel structure requires both `entity` and `time` identifiers
    - Parametric models (Pitt-Lee, BC92, BC95, Kumbhakar, Lee-Schmidt) require a distributional assumption for $u$
    - CSS model does not require distributional assumptions but needs $T \geq 5$
    - BC95 model requires `dist="truncated_normal"` and `inefficiency_vars` to be specified

## Panel Model Types

| Model | `model_type` | Time-Varying | Determinants | Distribution | Reference |
|---|---|---|---|---|---|
| Pitt-Lee | `"pitt_lee"` | No | No | Required | Pitt & Lee (1981) |
| Battese-Coelli 92 | `"bc92"` | Yes (decay) | No | Required | Battese & Coelli (1992) |
| Battese-Coelli 95 | `"bc95"` | Yes | Yes | Truncated normal | Battese & Coelli (1995) |
| CSS | `"css"` | Yes | No | Not required | Cornwell, Schmidt & Sickles (1990) |
| Kumbhakar | `"kumbhakar_1990"` | Yes (logistic) | No | Required | Kumbhakar (1990) |
| Lee-Schmidt | `"lee_schmidt_1993"` | Yes (dummies) | No | Required | Lee & Schmidt (1993) |

## Detailed Guide

### Pitt-Lee (1981): Time-Invariant Inefficiency

The simplest panel SFA model assumes inefficiency is constant over time:

$$y_{it} = X_{it}'\beta + v_{it} - u_i$$

where $u_i \geq 0$ is time-invariant inefficiency for entity $i$.

**When to use:** When the panel is short ($T$ small) and you believe inefficiency is a stable structural characteristic.

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="pitt_lee",
)
result = model.fit()

# Efficiency is constant over time for each entity
eff = result.efficiency(estimator="bc")
print(eff.head())
```

!!! note "Auto-detection"
    If you specify `entity` and `time` without `model_type` or `inefficiency_vars`, PanelBox automatically selects the Pitt-Lee model.

### Battese-Coelli 1992 (BC92): Exponential Decay

BC92 models time-varying inefficiency using an exponential decay function:

$$u_{it} = u_i \cdot \exp\big(-\eta(t - T)\big)$$

where $\eta$ is the decay parameter and $T$ is the last time period.

- $\eta > 0$: efficiency **improves** over time (inefficiency decays)
- $\eta < 0$: efficiency **deteriorates** over time (inefficiency grows)
- $\eta = 0$: efficiency is **constant** (reduces to Pitt-Lee)

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="bc92",
)
result = model.fit()

# Check the eta parameter
eta = result.temporal_params.get("eta", 0)
print(f"Decay parameter eta: {eta:.4f}")

if eta > 0:
    print("Efficiency improves over time")
elif eta < 0:
    print("Efficiency deteriorates over time")
else:
    print("Efficiency is constant over time")

# Time-varying efficiency
eff = result.efficiency(estimator="bc", by_period=True)
print(eff.head(10))
```

!!! tip "Interpretation of $\eta$"
    The BC92 model restricts all entities to share the same temporal pattern. Entities can differ in their initial level of inefficiency $u_i$, but the rate of change $\eta$ is common. This is restrictive but parsimonious.

### Battese-Coelli 1995 (BC95): Inefficiency Determinants

BC95 allows inefficiency to depend on observable characteristics:

$$u_{it} \sim N^+(\mathbf{Z}_{it}'\delta, \sigma_u^2)$$

where $\mathbf{Z}_{it}$ are variables that affect the mean of inefficiency and $\delta$ are coefficients to be estimated.

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="truncated_normal",  # Required for BC95
    model_type="bc95",
    inefficiency_vars=["firm_age", "export_share"],
)
result = model.fit()

# Interpret delta coefficients
# For production frontier:
#   positive delta -> increases mean inefficiency -> decreases efficiency
#   negative delta -> decreases mean inefficiency -> increases efficiency
print(result.params.filter(like="delta_"))

# Marginal effects on inefficiency
me = result.marginal_effects(method="location")
print(me)
```

!!! warning "Distribution Requirement"
    BC95 requires `dist="truncated_normal"`. Using any other distribution with `inefficiency_vars` raises a `ValueError`.

### CSS (1990): Distribution-Free Model

The Cornwell-Schmidt-Sickles model does not require distributional assumptions for inefficiency. Instead, it uses time-varying entity-specific intercepts:

$$y_{it} = \alpha_i(t) + X_{it}'\beta + v_{it}$$

where $\alpha_i(t) = \theta_{i1} + \theta_{i2} \cdot t + \theta_{i3} \cdot t^2$ is a quadratic function of time.

Efficiency is derived from the intercepts: $TE_{it} = \exp(\alpha_{it} - \max_j \alpha_{jt})$.

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    model_type="css",
    css_time_trend="quadratic",  # 'none', 'linear', or 'quadratic'
)
result = model.fit()

# CSS-specific results
css_result = result._css_result

# Efficiency by entity (averaged over time)
eff_entity = css_result.efficiency_by_entity()
print(eff_entity.head())

# Efficiency by period (averaged over entities)
eff_period = css_result.efficiency_by_period()
print(eff_period)
```

**Time trend options:**

| `css_time_trend` | Intercept Form | Flexibility |
|---|---|---|
| `"none"` | $\alpha_i$ (constant) | Entity FE only; no time variation |
| `"linear"` | $\theta_{i1} + \theta_{i2} \cdot t$ | Linear time trend per entity |
| `"quadratic"` | $\theta_{i1} + \theta_{i2} \cdot t + \theta_{i3} \cdot t^2$ | Flexible; default and recommended |

!!! tip "When to Use CSS"
    CSS is ideal when you are concerned about distributional misspecification. It does not impose any parametric form on inefficiency. However, it requires sufficient time periods ($T \geq 5$, preferably $T \geq 10$) and produces less precise estimates than parametric models when the distributional assumption is correct.

### Kumbhakar (1990): Logistic Decay

Kumbhakar's model uses a logistic function for the temporal pattern:

$$u_{it} = u_i \cdot B(t)$$

where $B(t) = [1 + \exp(b \cdot t + c \cdot t^2)]^{-1}$ is a bounded function.

This allows non-monotonic time patterns -- efficiency can first improve then deteriorate (or vice versa).

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="kumbhakar_1990",
)
result = model.fit()

# Temporal parameters
b = result.temporal_params.get("b", 0)
c = result.temporal_params.get("c", 0)
print(f"b = {b:.4f}, c = {c:.4f}")
```

### Lee-Schmidt (1993): Time Dummies

Lee-Schmidt uses time dummies for maximum flexibility:

$$u_{it} = u_i \cdot \delta_t$$

where $\delta_t$ are period-specific loadings (with $\delta_T = 1$ for normalization).

This imposes no parametric structure on the temporal pattern but estimates $T-1$ additional parameters.

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="lee_schmidt_1993",
)
result = model.fit()

# Time loadings
delta_t = result.temporal_params.get("delta_t", [])
print("Time loadings:", delta_t)
```

## Choosing a Panel Model

```text
Start
  |
  v
Is T very small (T < 5)?
  |-- Yes --> Pitt-Lee (time-invariant)
  |-- No
       |
       v
  Do you want distribution-free estimates?
    |-- Yes --> CSS (quadratic time trend)
    |-- No
         |
         v
    Do you have determinants of inefficiency (Z variables)?
      |-- Yes --> BC95 (truncated normal + inefficiency_vars)
      |-- No
           |
           v
      Do you expect monotonic efficiency change?
        |-- Yes --> BC92 (exponential decay)
        |-- No --> Kumbhakar or Lee-Schmidt
```

## Standard Errors

All parametric panel models estimate standard errors from the Hessian matrix at the MLE optimum. The delta method is used for transformed parameters (variance components estimated on the log scale).

## Diagnostics

### Test for Time-Varying Efficiency

```python
# For BC92: test H0: eta = 0 (time-invariant)
constancy_test = result.test_temporal_constancy()
print(f"Test statistic: {constancy_test['test_statistic']:.4f}")
print(f"P-value: {constancy_test['p_value']:.4f}")
print(f"Conclusion: {constancy_test['conclusion']}")
```

### Variance Decomposition

```python
var_decomp = result.variance_decomposition(ci_level=0.95)
print(f"gamma: {var_decomp['gamma']:.4f}")
print(var_decomp['interpretation'])
```

### Efficiency Evolution Plots

```python
# Time series of mean efficiency
fig = result.plot_efficiency_evolution(kind="timeseries", show_ci=True)

# Spaghetti plot (individual entity trajectories)
fig = result.plot_efficiency_evolution(kind="spaghetti", alpha=0.3)

# Heatmap of efficiency by entity and period
fig = result.plot_efficiency_evolution(kind="heatmap")
```

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| Panel SFA Models | Comparing panel model specifications | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/02_panel_sfa.ipynb) |
| BC95 with Determinants | Modeling inefficiency determinants | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/04_determinants_heterogeneity.ipynb) |

## See Also

- [Production and Cost Frontiers](production-cost.md) -- Cross-section SFA fundamentals
- [True Models (TFE/TRE)](true-models.md) -- Separating heterogeneity from inefficiency
- [Four-Component SFA](four-component.md) -- Persistent vs transient inefficiency
- [SFA Diagnostics](diagnostics.md) -- Comprehensive diagnostic tests
- [TFP Decomposition](tfp.md) -- Total Factor Productivity analysis

## References

- Pitt, M. M., & Lee, L. F. (1981). The measurement and sources of technical inefficiency in the Indonesian weaving industry. *Journal of Development Economics*, 9(1), 43-64.
- Battese, G. E., & Coelli, T. J. (1992). Frontier production functions, technical efficiency and panel data: with application to paddy farmers in India. *Journal of Productivity Analysis*, 3(1-2), 153-169.
- Battese, G. E., & Coelli, T. J. (1995). A model for technical inefficiency effects in a stochastic frontier production function for panel data. *Empirical Economics*, 20(2), 325-332.
- Cornwell, C., Schmidt, P., & Sickles, R. C. (1990). Production frontiers with cross-sectional and time-series variation in efficiency levels. *Journal of Econometrics*, 46(1-2), 185-200.
- Kumbhakar, S. C. (1990). Production frontiers, panel data, and time-varying technical inefficiency. *Journal of Econometrics*, 46(1-2), 201-211.
- Lee, Y. H., & Schmidt, P. (1993). A production frontier model with flexible temporal variation in technical efficiency. In *The Measurement of Productive Efficiency: Techniques and Applications*, Oxford University Press.
- Wang, H. J. (2002). Heteroscedasticity and non-monotonic efficiency effects of a stochastic frontier model. *Journal of Productivity Analysis*, 18, 241-253.
