---
title: "True Fixed Effects and True Random Effects"
description: "Greene's True FE and True RE stochastic frontier models that separate heterogeneity from inefficiency in PanelBox"
---

# True Fixed Effects and True Random Effects

!!! info "Quick Reference"
    **Class:** `panelbox.frontier.StochasticFrontier`
    **Import:** `from panelbox.frontier import StochasticFrontier`
    **Model types:** `model_type="tfe"` or `model_type="tre"`
    **Stata equivalent:** `sfpanel, model(tfe)` / `sfpanel, model(tre)`
    **R equivalent:** `sfaR::sfacross()` (partial support)

## Overview

Classical panel SFA models like Pitt-Lee (1981) specify $y_{it} = X_{it}'\beta + v_{it} - u_i$, where all cross-entity variation in the intercept is attributed to inefficiency. This creates a fundamental **confounding problem**: legitimate technological differences between firms (heterogeneity) are mistakenly classified as inefficiency.

Greene (2005) proposed "True" panel stochastic frontier models that explicitly separate entity-specific heterogeneity from time-varying inefficiency. The True Fixed Effects (TFE) model adds entity-specific intercepts $\alpha_i$, while the True Random Effects (TRE) model adds a random heterogeneity component $w_i$. Both models allow $u_{it}$ to vary over time, enabling the study of efficiency dynamics.

These models are essential when the research question requires distinguishing structural differences between entities (technology, geography, regulation) from managerial inefficiency.

## Quick Example

```python
from panelbox.frontier import StochasticFrontier

# True Fixed Effects model
model_tfe = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="tfe",
)
result_tfe = model_tfe.fit()

# True Random Effects model
model_tre = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="tre",
)
result_tre = model_tre.fit()
```

## When to Use

- You suspect that **heterogeneity** (technology differences, geographic advantages) is being confused with **inefficiency**
- You need **time-varying inefficiency** estimates that are not contaminated by entity-level heterogeneity
- Your entities have fundamentally different production technologies or operating environments
- Policy analysis requires distinguishing "firms need better technology" from "firms need better management"

!!! warning "Key Assumptions"
    - **TFE**: Entity effects $\alpha_i$ can be correlated with regressors $X_{it}$; suffers from incidental parameters problem when $T$ is small
    - **TRE**: Random heterogeneity $w_i$ must be independent of regressors $X_{it}$; more efficient under this assumption
    - Both models require balanced or nearly balanced panels for reliable estimation

## The Confounding Problem

### Classical Panel SFA

In the Pitt-Lee model, the intercept is common to all entities:

$$y_{it} = \alpha + X_{it}'\beta + v_{it} - u_i$$

**Everything** that differs systematically across entities ends up in $u_i$ -- including:

- Genuine inefficiency (poor management)
- Technology differences (older vs newer equipment)
- Geographic advantages (proximity to markets)
- Regulatory environment (favorable vs restrictive)

This means efficiency rankings are biased: a firm with inferior technology appears "inefficient" even if it is well-managed given its constraints.

### Greene's Solution

| Component | Classical (Pitt-Lee) | True FE | True RE |
|---|---|---|---|
| Entity heterogeneity | Confounded with $u_i$ | Captured by $\alpha_i$ | Captured by $w_i$ |
| Inefficiency | $u_i$ (time-invariant) | $u_{it}$ (time-varying) | $u_{it}$ (time-varying) |
| Noise | $v_{it}$ | $v_{it}$ | $v_{it}$ |
| Correlation with $X$ | Not applicable | $\alpha_i$ may correlate | $w_i$ must not correlate |

## Detailed Guide

### True Fixed Effects (TFE)

**Model:**

$$y_{it} = \alpha_i + X_{it}'\beta + v_{it} - u_{it}$$

where:

- $\alpha_i$ = entity-specific fixed effect capturing heterogeneity
- $u_{it} \sim N^+(0, \sigma_u^2)$ = time-varying inefficiency (half-normal)
- $v_{it} \sim N(0, \sigma_v^2)$ = noise

**Estimation:** PanelBox uses a concentrated likelihood approach. For each candidate $(\beta, \sigma_v^2, \sigma_u^2)$, the optimal $\alpha_i$ is computed numerically, reducing the optimization from $N + k$ to just $k + 2$ parameters.

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="tfe",
)
result = model.fit()
print(result.summary())
```

#### Incidental Parameters Problem

The TFE model estimates $N$ fixed effects ($\alpha_i$), which creates a bias of order $O(1/T)$ in the variance parameters $\sigma_v^2$ and $\sigma_u^2$. This bias is negligible for large $T$ but can be substantial when $T < 10$.

**Analytical bias correction** (Hahn & Newey, 2004):

```python
from panelbox.frontier.true_models import bias_correct_tfe_analytical

alpha_corrected = bias_correct_tfe_analytical(
    alpha_hat=alpha_estimates,
    T=n_periods,
    sigma_v_sq=result.sigma_v_sq,
    sigma_u_sq=result.sigma_u_sq,
)
```

**Jackknife bias correction** (more accurate but slower):

```python
from panelbox.frontier.true_models import bias_correct_tfe_jackknife

jk_result = bias_correct_tfe_jackknife(
    y, X, entity_id, time_id,
    theta=result.params.values,
    sign=1,
)
alpha_corrected = jk_result["alpha_corrected"]
bias_estimate = jk_result["bias_estimate"]
```

### True Random Effects (TRE)

**Model:**

$$y_{it} = X_{it}'\beta + w_i + v_{it} - u_{it}$$

where:

- $w_i \sim N(0, \sigma_w^2)$ = random heterogeneity (time-invariant)
- $u_{it} \sim N^+(0, \sigma_u^2)$ = time-varying inefficiency
- $v_{it} \sim N(0, \sigma_v^2)$ = noise

The TRE model has a **three-component error structure**. The likelihood requires integrating over $w_i$, which PanelBox performs using Gauss-Hermite quadrature (default, `n_quadrature=32`) or simulated MLE with Halton sequences.

```python
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="tre",
)
result = model.fit()
print(result.summary())
```

#### Variance Decomposition (Three Components)

For TRE models, the variance is decomposed into three shares:

$$\gamma_v + \gamma_u + \gamma_w = 1$$

where:

- $\gamma_v = \sigma_v^2 / \sigma_{\text{total}}^2$ -- proportion due to noise
- $\gamma_u = \sigma_u^2 / \sigma_{\text{total}}^2$ -- proportion due to inefficiency
- $\gamma_w = \sigma_w^2 / \sigma_{\text{total}}^2$ -- proportion due to heterogeneity

```python
var_decomp = result.variance_decomposition(ci_level=0.95)

print(f"Noise share (gamma_v):        {var_decomp['gamma_v']:.4f}")
print(f"Inefficiency share (gamma_u): {var_decomp['gamma_u']:.4f}")
print(f"Heterogeneity share (gamma_w):{var_decomp['gamma_w']:.4f}")
print(f"\n{var_decomp['interpretation']}")
```

### Model Selection: Hausman Test

The Hausman test determines whether TFE or TRE is more appropriate:

- **H0:** TRE is consistent and efficient ($w_i$ independent of $X$)
- **H1:** Only TFE is consistent ($w_i$ correlated with $X$)

```python
from panelbox.frontier.tests import hausman_test_tfe_tre

hausman = hausman_test_tfe_tre(
    params_tfe=result_tfe.params.values,
    params_tre=result_tre.params.values,
    vcov_tfe=result_tfe.vcov,
    vcov_tre=result_tre.vcov,
    param_names=result_tfe.params.index.tolist(),
)

print(f"Hausman statistic: {hausman['statistic']:.4f}")
print(f"P-value: {hausman['pvalue']:.4f}")
print(f"Recommendation: {hausman['conclusion']}")
print(hausman['interpretation'])
```

| Decision | P-value | Interpretation |
|---|---|---|
| Use **TFE** | $p < 0.05$ | Reject H0; heterogeneity correlates with regressors |
| Use **TRE** | $p \geq 0.05$ | Do not reject H0; TRE is more efficient |

### TFE and TRE with BC95 Determinants

Both True models can be combined with Battese-Coelli (1995) inefficiency determinants:

$$u_{it} \sim N^+(\mathbf{Z}_{it}'\delta, \sigma_u^2)$$

The $\delta$ coefficients have cleaner interpretation in True models because heterogeneity is already captured by $\alpha_i$ or $w_i$.

```python
# TFE with inefficiency determinants
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="truncated_normal",
    model_type="tfe",
    inefficiency_vars=["firm_age", "export_share"],
)
result = model.fit()

# Interpretation:
# positive delta -> increases mean inefficiency
# negative delta -> decreases mean inefficiency
print(result.params.filter(like="delta_"))
```

## Practical Guidelines

| Criterion | TFE | TRE |
|---|---|---|
| Correlation $E[w_i \mid X]$ | Allowed | Requires independence |
| Panel length ($T$) | Needs $T \geq 10$ (or bias correction) | Works for any $T$ |
| Efficiency | Less efficient | More efficient (under H0) |
| Incidental parameters | Yes (needs correction) | No |
| Computational cost | Moderate | High (quadrature integration) |
| Variance decomposition | Two components | Three components |

### Recommended Workflow

1. Estimate both TFE and TRE
2. Perform Hausman test
3. If TFE is selected and $T < 10$, apply bias correction
4. If TRE is selected, examine variance decomposition for $\gamma_w$
5. If $\gamma_w$ is very small, heterogeneity may not be important

### Common Pitfalls

1. **Forgetting bias correction for TFE**: Always apply when $T < 10$
2. **Too few quadrature points for TRE**: Use at least `n_quadrature=20`; PanelBox defaults to 32
3. **Interpreting $u_{it}$ as total inefficiency**: In True models, $u_{it}$ is inefficiency *after* controlling for $\alpha_i$ or $w_i$
4. **Using Z variables also in X**: Inefficiency determinants $Z$ should be different from frontier variables $X$

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| True Models | TFE vs TRE estimation and comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/05_testing_comparison.ipynb) |

## See Also

- [Production and Cost Frontiers](production-cost.md) -- SFA fundamentals
- [Panel SFA Models](panel-sfa.md) -- Classical panel models (Pitt-Lee, BC92, BC95)
- [Four-Component SFA](four-component.md) -- Further decomposition of inefficiency
- [SFA Diagnostics](diagnostics.md) -- Diagnostic tests including Hausman test
- [TFP Decomposition](tfp.md) -- Total Factor Productivity analysis

## References

- Greene, W. H. (2005). Reconsidering heterogeneity in panel data estimators of the stochastic frontier model. *Journal of Econometrics*, 126(2), 269-303.
- Greene, W. H. (2005). Fixed and random effects in stochastic frontier models. *Journal of Productivity Analysis*, 23(1), 7-32.
- Hahn, J., & Newey, W. (2004). Jackknife and analytical bias reduction for nonlinear panel models. *Econometrica*, 72(4), 1295-1319.
- Dhaene, G., & Jochmans, K. (2015). Split-panel jackknife estimation of fixed-effect models. *The Review of Economic Studies*, 82(3), 991-1030.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 1251-1271.
- Pitt, M. M., & Lee, L. F. (1981). The measurement and sources of technical inefficiency in the Indonesian weaving industry. *Journal of Development Economics*, 9(1), 43-64.
- Battese, G. E., & Coelli, T. J. (1995). A model for technical inefficiency effects in a stochastic frontier production function for panel data. *Empirical Economics*, 20(2), 325-332.
