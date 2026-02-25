---
title: "Sign Conventions"
description: "Production vs cost frontier sign conventions, common pitfalls, and validation checklist for stochastic frontier analysis in PanelBox"
---

# Sign Conventions

!!! info "Quick Reference"
    **Production frontier:** `frontier="production"` -- $y = X\beta + v - u$
    **Cost frontier:** `frontier="cost"` -- $y = X\beta + v + u$
    **Internal sign:** Production = `+1`, Cost = `-1`

## Overview

The sign convention determines how the inefficiency term $u$ enters the stochastic frontier model. Getting it wrong produces nonsensical efficiency estimates and misleading policy recommendations. This page clarifies the conventions used in PanelBox, explains how to verify your specification, and documents common pitfalls.

PanelBox handles sign conventions internally through the `frontier` parameter. Users should **never** need to set the sign directly -- simply specify `frontier="production"` or `frontier="cost"`, and the library applies the correct sign throughout estimation, efficiency calculation, and diagnostic tests.

## The Sign Convention

### Production Frontier

$$y_{it} = X_{it}'\beta + v_{it} - u_{it}$$

| Element | Description |
|---|---|
| $y_{it}$ | Log output |
| $X_{it}'\beta$ | Maximum feasible output (frontier) |
| $v_{it} \sim N(0, \sigma_v^2)$ | Symmetric random noise |
| $u_{it} \geq 0$ | Technical inefficiency (reduces output) |
| Sign of $u$ | **Negative** (subtracted from frontier) |

**Technical Efficiency:** $TE = \exp(-u) \in (0, 1]$, where 1 = fully efficient.

Firms produce **below** the frontier: inefficiency reduces output.

### Cost Frontier

$$y_{it} = X_{it}'\beta + v_{it} + u_{it}$$

| Element | Description |
|---|---|
| $y_{it}$ | Log cost |
| $X_{it}'\beta$ | Minimum feasible cost (frontier) |
| $v_{it} \sim N(0, \sigma_v^2)$ | Symmetric random noise |
| $u_{it} \geq 0$ | Cost inefficiency (increases cost) |
| Sign of $u$ | **Positive** (added to frontier) |

**Cost Efficiency:** $CE = \exp(-u) \in (0, 1]$, where 1 = minimum cost achieved.

Firms operate **above** the cost frontier: inefficiency increases cost above the minimum.

### Summary Table

| Aspect | Production | Cost |
|---|---|---|
| Model | $y = X\beta + v - u$ | $y = X\beta + v + u$ |
| Frontier | Maximum (output) | Minimum (cost) |
| Sign of $u$ | Negative | Positive |
| Effect of $u$ | Reduces output | Increases cost |
| Efficiency | $TE = e^{-u}$ | $CE = e^{-u}$ |
| Efficiency range | $(0, 1]$ | $(0, 1]$ |
| Skewness of OLS residuals | Negative | Positive |
| Frontier position in plots | Above data points | Below data points |

!!! note "Efficiency Scale"
    PanelBox uses $CE = \exp(-u) \in (0, 1]$ for cost efficiency, which is on the **same scale** as technical efficiency. This is consistent with the R `frontier` package. An alternative convention uses $CE = \exp(u) \in [1, \infty)$. To convert: $CE_{\text{ratio}} = 1 / CE_{\text{panelbox}}$.

## How PanelBox Handles It

```python
from panelbox.frontier import StochasticFrontier

# Production frontier -- PanelBox internally uses sign = +1
model_prod = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",  # sign = +1
    dist="half_normal",
)

# Cost frontier -- PanelBox internally uses sign = -1
model_cost = StochasticFrontier(
    data=df,
    depvar="log_cost",
    exog=["log_output", "log_price_labor"],
    frontier="cost",  # sign = -1
    dist="half_normal",
)
```

The `frontier` parameter controls:

1. The sign in the likelihood function
2. The direction of the skewness check
3. The efficiency calculation formula
4. The frontier position in plots

## Skewness Validation

The composed error $\varepsilon = v - su$ (where $s = 1$ for production, $s = -1$ for cost) should have a specific skewness:

- **Production**: $\varepsilon = v - u$ is **negatively skewed** (left tail from $u > 0$)
- **Cost**: $\varepsilon = v + u$ is **positively skewed** (right tail from $u > 0$)

```python
from panelbox.frontier.tests import skewness_test

# Check skewness before estimation
skew = skewness_test(
    residuals=ols_residuals,
    frontier_type="production",
)

print(f"Skewness: {skew['skewness']:.4f}")
print(f"Expected: {skew['expected_sign']}")
print(f"Correct: {skew['correct_sign']}")

if skew['warning']:
    print(f"WARNING: {skew['warning']}")
```

!!! warning "Wrong Skewness"
    If OLS residuals have the wrong skewness sign, consider:

    1. You specified the wrong frontier type (production vs cost)
    2. Extreme outliers are distorting the distribution
    3. The functional form is misspecified
    4. There is no significant inefficiency in the data

## Efficiency Interpretation

### Production Frontier

```python
result = model_prod.fit()
eff = result.efficiency(estimator="bc")

# TE in (0, 1]
print(f"Mean TE: {eff['efficiency'].mean():.3f}")
# Example: TE = 0.85 means the firm produces 85% of maximum feasible output
# The firm could increase output by (1 - 0.85) / 0.85 = 17.6% without changing inputs
```

### Cost Frontier

```python
result = model_cost.fit()
eff = result.efficiency(estimator="bc")

# CE in (0, 1] (same scale as TE)
print(f"Mean CE: {eff['efficiency'].mean():.3f}")
# Example: CE = 0.80 means the firm could reduce costs by 20%

# Convert to ratio scale if needed: CE_ratio = 1 / CE
eff['ce_ratio'] = 1 / eff['efficiency']
print(f"Mean CE (ratio): {eff['ce_ratio'].mean():.3f}")
# Example: CE_ratio = 1.25 means the firm spends 25% more than the minimum
```

### Determinant Interpretation (BC95)

For models with inefficiency determinants ($u_{it} \sim N^+(\mathbf{Z}'\delta, \sigma_u^2)$), the sign of $\delta$ coefficients depends on the frontier type:

| Frontier | $\delta_j > 0$ | $\delta_j < 0$ |
|---|---|---|
| Production | Increases $u$ (decreases efficiency) | Decreases $u$ (increases efficiency) |
| Cost | Increases $u$ (increases cost above minimum) | Decreases $u$ (brings cost closer to minimum) |

In **both** cases, a positive $\delta$ means the variable **increases** inefficiency.

## Common Pitfalls

### 1. Using `frontier="production"` for Cost Data

```python
# WRONG: cost data with production frontier
model = StochasticFrontier(
    data=df,
    depvar="log_cost",       # Cost variable
    frontier="production",   # WRONG!
    ...
)

# CORRECT: use cost frontier for cost data
model = StochasticFrontier(
    data=df,
    depvar="log_cost",
    frontier="cost",         # Correct
    ...
)
```

**Symptoms**: Very high efficiency (> 0.95), wrong skewness direction, results not economically meaningful.

### 2. Interpreting $u$ as Efficiency

$u$ is **in**efficiency, not efficiency. Higher $u$ means **worse** performance:

- $u = 0$: fully efficient ($TE = 1$)
- $u = 0.1$: slightly inefficient ($TE = 0.905$)
- $u = 0.5$: moderately inefficient ($TE = 0.607$)
- $u = 1.0$: highly inefficient ($TE = 0.368$)

### 3. Comparing Efficiency Across Different Specifications

Efficiency scores from different frontier specifications (different distributions, functional forms, or model types) are **not directly comparable**. Only compare efficiencies from the same model specification, or use rankings rather than levels.

### 4. Ignoring the Skewness Check

Always check skewness of OLS residuals before SFA estimation. If skewness has the wrong sign, the MLE may still converge, but the results are unreliable.

### 5. Wrong Sign on Inefficiency Determinants

For BC95 models, remember that $\delta$ affects the **mean of inefficiency**, not efficiency. A positive $\delta$ increases inefficiency for both production and cost frontiers.

## Validation Checklist

Use this checklist to verify your frontier specification:

```python
import numpy as np
from scipy import stats

# 1. Fit the model
result = model.fit()

# 2. Verify skewness
skewness = stats.skew(result.residuals)
if model.frontier_type.value == "production":
    assert skewness < 0, f"Production: expected negative skewness, got {skewness:.4f}"
else:
    assert skewness > 0, f"Cost: expected positive skewness, got {skewness:.4f}"

# 3. Verify efficiency range
eff = result.efficiency(estimator="bc")
assert np.all(eff['efficiency'] > 0), "All efficiencies must be > 0"
assert np.all(eff['efficiency'] <= 1), "All efficiencies must be <= 1"

# 4. Verify convergence
assert result.converged, "Optimizer did not converge"

# 5. Verify mean efficiency is plausible
mean_eff = eff['efficiency'].mean()
assert 0.3 < mean_eff < 0.99, f"Mean efficiency {mean_eff:.3f} may be implausible"

# 6. Verify gamma is reasonable
gamma = result.gamma
assert 0 < gamma < 1, f"Gamma = {gamma:.4f} is out of range"

print("All validation checks passed!")
```

## Side-by-Side Example

```python
from panelbox.frontier import StochasticFrontier

# --- Production Frontier ---
sf_prod = StochasticFrontier(
    data=prod_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="half_normal",
)
res_prod = sf_prod.fit()

# --- Cost Frontier ---
sf_cost = StochasticFrontier(
    data=cost_df,
    depvar="log_cost",
    exog=["log_output", "log_price_labor", "log_price_capital"],
    frontier="cost",
    dist="half_normal",
)
res_cost = sf_cost.fit()

# Compare
print("Production Frontier:")
print(f"  sigma_u: {res_prod.sigma_u:.4f}")
print(f"  gamma:   {res_prod.gamma:.4f}")
print(f"  Mean TE: {res_prod.mean_efficiency:.4f}")

print("\nCost Frontier:")
print(f"  sigma_u: {res_cost.sigma_u:.4f}")
print(f"  gamma:   {res_cost.gamma:.4f}")
print(f"  Mean CE: {res_cost.mean_efficiency:.4f}")
```

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| Production vs Cost | Sign conventions in practice | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/02_panel_sfa.ipynb) |

## See Also

- [Production and Cost Frontiers](production-cost.md) -- Full SFA fundamentals
- [SFA Diagnostics](diagnostics.md) -- Skewness test and other diagnostics
- [Panel SFA Models](panel-sfa.md) -- Panel model specifications
- [True Models (TFE/TRE)](true-models.md) -- Heterogeneity separation

## References

- Aigner, D., Lovell, C. K., & Schmidt, P. (1977). Formulation and estimation of stochastic frontier production function models. *Journal of Econometrics*, 6(1), 21-37.
- Christensen, L. R., & Greene, W. H. (1976). Economies of scale in U.S. electric power generation. *Journal of Political Economy*, 84(4), 655-676.
- Kumbhakar, S. C., & Lovell, C. A. K. (2000). *Stochastic Frontier Analysis*. Cambridge University Press.
- Coelli, T. J. (1995). Estimators and hypothesis tests for a stochastic frontier function: A Monte Carlo analysis. *Journal of Productivity Analysis*, 6(4), 247-268.
