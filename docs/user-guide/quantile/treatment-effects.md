---
title: "Quantile Treatment Effects"
description: "Quantile treatment effects estimation for panel data with DiD and Changes-in-Changes methods in PanelBox"
---

# Quantile Treatment Effects

!!! info "Quick Reference"
    **Class:** `panelbox.models.quantile.treatment_effects.QuantileTreatmentEffects`
    **Import:** `from panelbox.models.quantile import QuantileTreatmentEffects`
    **Stata equivalent:** `qte y, treatment(d) quantiles(0.25 0.5 0.75)`
    **R equivalent:** `qte::qte()`, `DRDID::drdid()`

## Overview

Quantile Treatment Effects (QTE) extend traditional Average Treatment Effects (ATE) by estimating how a treatment affects different parts of the outcome distribution. While ATE answers "what is the average effect?", QTE answers "who benefits most and who benefits least?"

For a binary treatment $D \in \{0, 1\}$, the QTE at quantile $\tau$ is:

$$QTE(\tau) = Q_{Y(1)}(\tau) - Q_{Y(0)}(\tau)$$

where $Y(1)$ and $Y(0)$ are potential outcomes under treatment and control.

This reveals crucial heterogeneity that mean-based methods miss. For example, a job training program might:

- Increase wages by 5% at the 10th percentile (helps the lowest earners most)
- Increase wages by 2% at the 90th percentile (modest effect for high earners)
- Show an ATE of 3% that masks this distributional heterogeneity

PanelBox implements four QTE estimation methods, each suited to different identification strategies.

## Quick Example

```python
import numpy as np
from panelbox.models.quantile import QuantileTreatmentEffects

qte = QuantileTreatmentEffects(
    data=df,
    outcome="wage",
    treatment="trained",
    covariates=["experience", "education"],
    entity_col="worker_id",
    time_col="year",
)

results = qte.estimate_qte(
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="standard",
)
```

## When to Use

- **Heterogeneous effects**: treatment effects differ across the outcome distribution
- **Policy targeting**: identify who benefits most from an intervention
- **Inequality analysis**: understand how a policy affects distributional spread
- **Welfare analysis**: evaluate if a program compresses or expands the distribution
- **Program evaluation**: go beyond mean impact to characterize the full effect profile

!!! warning "Key Assumptions"
    Assumptions depend on the chosen method:

    - **Standard QTE**: conditional independence (selection on observables), common support
    - **DiD QTE**: parallel trends in quantiles (pre-treatment trends are common)
    - **CiC**: monotonicity of outcomes in unobservables, time invariance of unobservables distribution
    - **Unconditional QTE**: correct specification of the RIF regression

## Detailed Guide

### Conditional vs Unconditional QTE

Two distinct concepts:

| Type | Definition | Interpretation |
|------|-----------|----------------|
| **Conditional QTE** | Effect at quantiles of $Y\|X$ | Effect for individuals at a given covariate-conditional quantile |
| **Unconditional QTE** | Effect at quantiles of $Y$ | Effect at a given point in the marginal distribution |

The unconditional QTE is often more policy-relevant: it tells us "what happens at the 10th percentile of the wage distribution?" rather than "what happens for the lowest earner conditional on their education level?"

### Method 1: Standard QTE

Estimates QTE by comparing conditional quantile regressions for treated and control groups:

```python
results = qte.estimate_qte(
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="standard",
)

# Access results
for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
    qte_est = results.qte_results[tau]
    print(f"tau={tau:.2f}: QTE = {qte_est['qte']:.4f} "
          f"(SE = {qte_est['se']:.4f})")
```

**When**: cross-sectional setting with good control variables (selection on observables).

### Method 2: Unconditional QTE via RIF Regression

Uses the Recentered Influence Function (Firpo, 2007) to estimate marginal quantile effects:

```python
results_unconditional = qte.estimate_qte(
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="unconditional",
)
```

The RIF approach:

1. Compute the RIF for each quantile: $RIF(y; Q_\tau) = Q_\tau + \frac{\tau - \mathbb{1}\{y \leq Q_\tau\}}{f_Y(Q_\tau)}$
2. Regress RIF on treatment and covariates
3. The treatment coefficient is the unconditional QTE

**When**: policy-relevant marginal effects are needed.

### Method 3: Difference-in-Differences QTE

For panel data with treatment and control groups observed over time:

```python
results_did = qte.estimate_qte(
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="did",
)
```

The DiD QTE extends the parallel trends assumption to quantiles:

$$QTE_{DiD}(\tau) = [Q_{Y,1}^{treat}(\tau) - Q_{Y,0}^{treat}(\tau)] - [Q_{Y,1}^{control}(\tau) - Q_{Y,0}^{control}(\tau)]$$

**When**: panel data with pre/post treatment periods and a control group.

### Method 4: Changes-in-Changes (Athey & Imbens 2006)

A nonlinear generalization of DiD that relaxes parallel trends:

```python
results_cic = qte.estimate_qte(
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
    method="cic",
)
```

The CiC method assumes:

- Outcomes are monotone functions of a scalar unobservable: $Y = h(U, T)$
- The distribution of $U$ is time-invariant within groups
- The production function $h$ satisfies $\partial h / \partial U > 0$

**When**: parallel trends in quantiles is doubtful but monotonicity is reasonable.

### Comparing Methods

```python
import pandas as pd

tau_grid = [0.1, 0.25, 0.5, 0.75, 0.9]

comparison = pd.DataFrame({
    "tau": tau_grid,
    "Standard": [qte.estimate_qte(tau=t, method="standard")
                  .qte_results[t]["qte"] for t in tau_grid],
    "Unconditional": [qte.estimate_qte(tau=t, method="unconditional")
                       .qte_results[t]["qte"] for t in tau_grid],
})
print(comparison)
```

### Interpreting Results

The `QTEResult` object provides:

```python
# QTE at each quantile
results.qte_results[0.5]["qte"]     # point estimate
results.qte_results[0.5]["se"]      # standard error
results.qte_results[0.5]["ci"]      # confidence interval

# Overall heterogeneity
qte_values = [results.qte_results[t]["qte"] for t in tau_grid]
heterogeneity = np.std(qte_values)
print(f"QTE heterogeneity (std): {heterogeneity:.4f}")
```

**Interpretation guide**:

| Pattern | Interpretation |
|---------|---------------|
| QTE constant across $\tau$ | Homogeneous effect (ATE suffices) |
| QTE increasing in $\tau$ | Larger effects at the top of the distribution |
| QTE decreasing in $\tau$ | Larger effects at the bottom (pro-poor) |
| QTE positive at low $\tau$, negative at high $\tau$ | Compresses the distribution |
| QTE negative at low $\tau$, positive at high $\tau$ | Expands the distribution |

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame/PanelData | *required* | Data (panel or cross-sectional) |
| `outcome` | str | *required* | Outcome variable name |
| `treatment` | str | *required* | Binary treatment variable name |
| `covariates` | list | `None` | Control variable names |
| `entity_col` | str | `None` | Entity identifier (for panel data) |
| `time_col` | str | `None` | Time identifier (for panel data) |

### Estimation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float/array | `0.5` | Quantile level(s) in $(0, 1)$ |
| `method` | str | `"standard"` | Method: `"standard"`, `"unconditional"`, `"did"`, `"cic"` |

### Method Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Cross-sectional with controls | `"standard"` |
| Policy-relevant marginal effects | `"unconditional"` |
| Panel with common quantile trends | `"did"` |
| Panel without parallel trends | `"cic"` |
| Need non-crossing guarantee | Use [Location-Scale](location-scale.md) instead |

## Diagnostics

### Testing for Heterogeneous Effects

```python
# Compare QTE across quantiles
tau_grid = [0.1, 0.25, 0.5, 0.75, 0.9]
qte_values = [results.qte_results[t]["qte"] for t in tau_grid]

# If QTE is approximately constant, ATE may suffice
from scipy import stats
# Rough test: are QTE values significantly different from their mean?
qte_mean = np.mean(qte_values)
qte_se = [results.qte_results[t]["se"] for t in tau_grid]
print(f"Mean QTE: {qte_mean:.4f}")
print(f"QTE range: [{min(qte_values):.4f}, {max(qte_values):.4f}]")
```

### Checking for Crossing

```python
from panelbox.models.quantile import QuantileMonotonicity

# If QTE curves cross, apply monotonicity correction
# (relevant when covariates are included)
```

### Sensitivity Analysis

```python
# Try different specifications
specs = [
    ["experience"],
    ["experience", "education"],
    ["experience", "education", "age"],
]

for covs in specs:
    qte_temp = QuantileTreatmentEffects(
        data=df, outcome="wage", treatment="trained",
        covariates=covs, entity_col="id", time_col="year"
    )
    r = qte_temp.estimate_qte(tau=0.5, method="standard")
    print(f"Covariates {covs}: QTE(0.5) = {r.qte_results[0.5]['qte']:.4f}")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| QTE Basics | Standard and unconditional QTE estimation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/09_quantile_treatment_effects.ipynb) |
| QTE with DiD | Panel DiD and Changes-in-Changes methods | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/quantile/notebooks/09_quantile_treatment_effects.ipynb) |

## See Also

- [Pooled Quantile Regression](pooled.md) — standard QR (step 2 of standard QTE)
- [Dynamic Quantile](dynamic.md) — dynamic treatment effects with lags
- [Location-Scale Model](location-scale.md) — non-crossing QTE via LS decomposition
- [Non-Crossing Constraints](monotonicity.md) — fix crossing in QTE curves
- [Diagnostics](diagnostics.md) — testing and visualization

## References

- Firpo, S. (2007). Efficient semiparametric estimation of quantile treatment effects. *Econometrica*, 75(1), 259-276.
- Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. *Econometrica*, 77(3), 953-973.
- Athey, S., & Imbens, G. W. (2006). Identification and inference in nonlinear difference-in-differences models. *Econometrica*, 74(2), 431-497.
- Callaway, B., & Li, T. (2019). Quantile treatment effects in difference in differences models with panel data. *Quantitative Economics*, 10(4), 1579-1618.
- Chernozhukov, V., & Hansen, C. (2005). An IV model of quantile treatment effects. *Econometrica*, 73(1), 245-261.
