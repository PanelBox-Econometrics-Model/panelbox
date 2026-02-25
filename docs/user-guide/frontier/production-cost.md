---
title: "Production and Cost Frontiers"
description: "Cross-section and basic panel stochastic frontier analysis with production and cost frontier specifications in PanelBox"
---

# Production and Cost Frontiers

!!! info "Quick Reference"
    **Class:** `panelbox.frontier.StochasticFrontier`
    **Import:** `from panelbox.frontier import StochasticFrontier`
    **Stata equivalent:** `sfcross` / `sfpanel`
    **R equivalent:** `frontier::sfa()`

## Overview

Stochastic Frontier Analysis (SFA), introduced independently by Aigner, Lovell & Schmidt (1977) and Meeusen & van den Broeck (1977), is a parametric method for estimating the maximum achievable output (production frontier) or minimum achievable cost (cost frontier) and measuring the distance of each observation from that frontier.

Unlike ordinary regression, SFA decomposes the error term into two components: symmetric random noise $v$ and one-sided inefficiency $u \geq 0$. This allows researchers to separate genuine stochastic shocks from systematic underperformance, producing observation-specific efficiency scores.

PanelBox implements SFA with four distributional assumptions for the inefficiency term, full maximum likelihood estimation, and three efficiency estimators with confidence intervals. Both production and cost frontier specifications are supported through the unified `StochasticFrontier` class.

## Quick Example

```python
from panelbox.frontier import StochasticFrontier

# Production frontier with half-normal inefficiency
model = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="half_normal",
)
results = model.fit()

# Print results
print(results.summary())

# Get efficiency scores (Battese-Coelli estimator)
eff = results.efficiency(estimator="bc")
print(f"Mean efficiency: {results.mean_efficiency:.4f}")
print(eff.describe())
```

## When to Use

- You want to measure how close firms/units operate relative to a **best-practice frontier**
- You need observation-specific **efficiency scores** rather than just average performance
- Your data exhibits one-sided deviations from the frontier (not just symmetric noise)
- You want to decompose total variation into **noise** (random shocks) and **inefficiency** (systematic underperformance)
- You need to test whether inefficiency is statistically significant vs. OLS

!!! warning "Key Assumptions"
    - The frontier function is correctly specified (e.g., Cobb-Douglas, Translog)
    - The inefficiency term $u$ follows a known distribution (half-normal, exponential, truncated-normal, or gamma)
    - The noise term $v$ is normally distributed: $v \sim N(0, \sigma_v^2)$
    - Inefficiency $u$ is independent of regressors $X$ and noise $v$
    - The dependent variable is in logarithms (for multiplicative frontier)

## Detailed Guide

### The Stochastic Frontier Model

#### Production Frontier

The production frontier measures maximum feasible output given inputs:

$$y_{it} = X_{it}'\beta + v_{it} - u_{it}$$

where:

- $y_{it}$ is log output for unit $i$ at time $t$
- $X_{it}'\beta$ is the deterministic frontier (maximum achievable output)
- $v_{it} \sim N(0, \sigma_v^2)$ is symmetric random noise (weather, measurement error, luck)
- $u_{it} \geq 0$ is technical inefficiency (one-sided, non-negative)

The composed error is $\varepsilon_{it} = v_{it} - u_{it}$. Since $u \geq 0$, the composed error is negatively skewed for production frontiers.

**Technical Efficiency:**

$$TE_i = \frac{\text{observed output}}{\text{frontier output}} = \exp(-u_i) \in (0, 1]$$

A score of $TE = 0.85$ means the firm produces 85% of its maximum feasible output.

#### Cost Frontier

The cost frontier measures minimum feasible cost given outputs and input prices:

$$y_{it} = X_{it}'\beta + v_{it} + u_{it}$$

where:

- $y_{it}$ is log cost
- $X_{it}'\beta$ is the minimum cost frontier
- $u_{it} \geq 0$ increases cost above the minimum (cost inefficiency)

The composed error is $\varepsilon_{it} = v_{it} + u_{it}$. Since $u \geq 0$, the composed error is positively skewed for cost frontiers.

**Cost Efficiency:**

$$CE_i = \exp(-u_i) \in (0, 1]$$

A score of $CE = 0.80$ means the firm could reduce costs by 20% without changing output.

### Distributional Assumptions

PanelBox supports four distributions for the inefficiency term $u$:

| Distribution | API Value | Parameters | Characteristics |
|---|---|---|---|
| Half-Normal | `"half_normal"` | $\sigma_u$ | Simplest; mode at zero; one parameter |
| Exponential | `"exponential"` | $\sigma_u$ | Monotonically decreasing density; one parameter |
| Truncated Normal | `"truncated_normal"` | $\mu, \sigma_u$ | Flexible mode location; two parameters |
| Gamma | `"gamma"` | $P, \theta$ | Most flexible shape; two parameters |

=== "Half-Normal"

    $u \sim N^+(0, \sigma_u^2)$

    The simplest specification. The mode of inefficiency is at zero, implying most firms are relatively efficient with a declining tail of highly inefficient firms. Recommended as a starting point.

    ```python
    model = StochasticFrontier(
        data=df, depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production", dist="half_normal",
    )
    ```

=== "Exponential"

    $u \sim \text{Exp}(\sigma_u)$

    Also has mode at zero with a monotonically decreasing density. Produces slightly different efficiency rankings than half-normal. Useful as a robustness check.

    ```python
    model = StochasticFrontier(
        data=df, depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production", dist="exponential",
    )
    ```

=== "Truncated Normal"

    $u \sim N^+(\mu, \sigma_u^2)$

    Allows the mode of inefficiency to be non-zero. When $\mu > 0$, most firms have moderate inefficiency. Nests half-normal as a special case ($\mu = 0$). Requires truncated-normal for BC95 models with inefficiency determinants.

    ```python
    model = StochasticFrontier(
        data=df, depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production", dist="truncated_normal",
    )
    ```

=== "Gamma"

    $u \sim \text{Gamma}(P, \theta)$

    Most flexible shape with two parameters. Can produce a wide range of density shapes. Computationally more intensive, especially for panel models. Nests exponential as a special case ($P = 1$).

    ```python
    model = StochasticFrontier(
        data=df, depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production", dist="gamma",
    )
    ```

### Efficiency Estimation

After fitting the model, PanelBox provides three methods to estimate observation-specific inefficiency/efficiency:

| Estimator | API Value | Formula | Returns |
|---|---|---|---|
| JLMS | `"jlms"` | $E[u \mid \varepsilon]$ | Point estimate of inefficiency |
| Battese-Coelli | `"bc"` | $E[\exp(-u) \mid \varepsilon]$ | Efficiency score directly |
| Mode | `"mode"` | $M[u \mid \varepsilon]$ | Modal inefficiency estimate |

The **BC estimator** (Battese & Coelli, 1988) is the default and recommended choice because it directly estimates efficiency rather than inefficiency, avoiding Jensen's inequality bias.

```python
# Battese-Coelli estimator (recommended)
eff_bc = results.efficiency(estimator="bc")

# JLMS estimator (Jondrow et al. 1982)
eff_jlms = results.efficiency(estimator="jlms")

# Modal estimator
eff_mode = results.efficiency(estimator="mode")

# Each returns a DataFrame with columns:
# inefficiency, efficiency, ci_lower, ci_upper
print(eff_bc.head())
```

**Confidence intervals** follow the Horrace & Schmidt (1996) method, which accounts for the truncated nature of the conditional distribution.

### Complete Example

```python
from panelbox.frontier import StochasticFrontier

# --- Production frontier ---
sf_prod = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="half_normal",
)
result_prod = sf_prod.fit()

# Summary with diagnostics
print(result_prod.summary())

# Variance components
print(f"sigma_v (noise):        {result_prod.sigma_v:.4f}")
print(f"sigma_u (inefficiency): {result_prod.sigma_u:.4f}")
print(f"lambda (sigma_u/sigma_v): {result_prod.lambda_param:.4f}")
print(f"gamma (share of inefficiency): {result_prod.gamma:.4f}")

# Efficiency scores
eff = result_prod.efficiency(estimator="bc")
print(f"\nMean TE: {result_prod.mean_efficiency:.4f}")
print(eff.describe())

# --- Cost frontier ---
sf_cost = StochasticFrontier(
    data=df,
    depvar="log_cost",
    exog=["log_output", "log_price_labor", "log_price_capital"],
    frontier="cost",
    dist="exponential",
)
result_cost = sf_cost.fit()
print(result_cost.summary())

# Cost efficiency
eff_cost = result_cost.efficiency(estimator="bc")
print(f"Mean CE: {result_cost.mean_efficiency:.4f}")
```

## Configuration Options

### StochasticFrontier Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `DataFrame` | required | DataFrame with all variables |
| `depvar` | `str` | required | Dependent variable name (in logs) |
| `exog` | `list[str]` | required | List of exogenous variable names |
| `entity` | `str` | `None` | Entity identifier (for panel data) |
| `time` | `str` | `None` | Time identifier (for panel data) |
| `frontier` | `str` | `"production"` | `"production"` or `"cost"` |
| `dist` | `str` | `"half_normal"` | `"half_normal"`, `"exponential"`, `"truncated_normal"`, `"gamma"` |
| `inefficiency_vars` | `list[str]` | `None` | Variables affecting mean inefficiency (BC95) |
| `het_vars` | `list[str]` | `None` | Variables affecting variance of inefficiency (Wang 2002) |
| `model_type` | `str` | auto | Panel model type (auto-detected from entity/time) |
| `css_time_trend` | `str` | `None` | CSS time trend: `"none"`, `"linear"`, `"quadratic"` |

### fit() Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"mle"` | Estimation method (only `"mle"` supported) |
| `start_params` | `ndarray` | `None` | Initial parameter values (auto-computed) |
| `optimizer` | `str` | `"L-BFGS-B"` | Optimizer: `"L-BFGS-B"`, `"Newton-CG"`, `"BFGS"` |
| `maxiter` | `int` | `1000` | Maximum iterations |
| `tol` | `float` | `1e-8` | Convergence tolerance |
| `grid_search` | `bool` | `False` | Use grid search for starting values |
| `verbose` | `bool` | `False` | Print optimization progress |

## Result Attributes

| Attribute | Type | Description |
|---|---|---|
| `params` | `Series` | Parameter estimates |
| `se` | `Series` | Standard errors |
| `tvalues` | `Series` | t-statistics |
| `pvalues` | `Series` | p-values |
| `loglik` | `float` | Log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `converged` | `bool` | Convergence flag |
| `sigma_v` | `float` | Noise standard deviation |
| `sigma_u` | `float` | Inefficiency standard deviation |
| `lambda_param` | `float` | $\lambda = \sigma_u / \sigma_v$ |
| `gamma` | `float` | $\gamma = \sigma_u^2 / (\sigma_v^2 + \sigma_u^2)$ |
| `mean_efficiency` | `float` | Mean efficiency (BC estimator) |
| `residuals` | `ndarray` | Frontier residuals $\varepsilon = y - X'\beta$ |

## Diagnostics

### Variance Decomposition

The parameter $\gamma = \sigma_u^2 / (\sigma_v^2 + \sigma_u^2)$ measures the share of total variance due to inefficiency:

```python
var_decomp = results.variance_decomposition(ci_level=0.95, method="delta")

print(f"gamma: {var_decomp['gamma']:.4f}")
print(f"95% CI: [{var_decomp['gamma_ci'][0]:.4f}, {var_decomp['gamma_ci'][1]:.4f}]")
print(f"lambda: {var_decomp['lambda_param']:.4f}")
print(var_decomp['interpretation'])
```

| $\gamma$ Range | Interpretation |
|---|---|
| $\gamma < 0.1$ | Inefficiency negligible; OLS may be adequate |
| $0.1 \leq \gamma \leq 0.3$ | Inefficiency minor but present |
| $0.3 < \gamma < 0.7$ | Both noise and inefficiency important |
| $0.7 \leq \gamma \leq 0.9$ | Inefficiency dominant |
| $\gamma > 0.9$ | Nearly deterministic frontier; check specification |

### Returns to Scale

```python
rts = results.returns_to_scale_test(
    input_vars=["log_labor", "log_capital"],
    alpha=0.05,
)
print(f"RTS = {rts['rts']:.4f} ({rts['conclusion']})")
print(f"p-value: {rts['pvalue']:.4f}")
```

### Distribution Comparison

```python
comparison = results.compare_distributions(
    distributions=["half_normal", "exponential", "truncated_normal"]
)
print(comparison[["Distribution", "AIC", "BIC", "Mean Efficiency"]])
```

### Bootstrap Confidence Intervals

```python
# Bootstrap CIs for parameters
boot_params = results.bootstrap(n_boot=999, seed=42)
print(boot_params[["parameter", "estimate", "ci_lower", "ci_upper"]])

# Bootstrap CIs for efficiency scores
boot_eff = results.bootstrap_efficiency(n_boot=999, seed=42)
print(boot_eff[["te", "ci_lower", "ci_upper"]].describe())
```

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| SFA Fundamentals | Basic SFA estimation and interpretation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/01_introduction_sfa.ipynb) |
| Production vs Cost | Side-by-side comparison of frontier types | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/02_panel_sfa.ipynb) |

## See Also

- [Panel SFA Models](panel-sfa.md) -- Time-varying and time-invariant panel models
- [True Models (TFE/TRE)](true-models.md) -- Separating heterogeneity from inefficiency
- [Four-Component SFA](four-component.md) -- Persistent vs transient inefficiency
- [SFA Diagnostics](diagnostics.md) -- Comprehensive diagnostic tests
- [Sign Conventions](sign-convention.md) -- Production vs cost sign conventions
- [TFP Decomposition](tfp.md) -- Total Factor Productivity analysis

## References

- Aigner, D., Lovell, C. K., & Schmidt, P. (1977). Formulation and estimation of stochastic frontier production function models. *Journal of Econometrics*, 6(1), 21-37.
- Meeusen, W., & van den Broeck, J. (1977). Efficiency estimation from Cobb-Douglas production functions with composed error. *International Economic Review*, 435-444.
- Jondrow, J., Lovell, C. K., Materov, I. S., & Schmidt, P. (1982). On the estimation of technical inefficiency in the stochastic frontier production function model. *Journal of Econometrics*, 19(2-3), 233-238.
- Battese, G. E., & Coelli, T. J. (1988). Prediction of firm-level technical efficiencies with a generalized frontier production function and panel data. *Journal of Econometrics*, 38(3), 387-399.
- Horrace, W. C., & Schmidt, P. (1996). Confidence statements for efficiency estimates from stochastic frontier models. *Journal of Productivity Analysis*, 7(2-3), 257-282.
- Greene, W. H. (1990). A gamma-distributed stochastic frontier model. *Journal of Econometrics*, 46(1-2), 141-163.
- Kumbhakar, S. C., & Lovell, C. A. K. (2000). *Stochastic Frontier Analysis*. Cambridge University Press.
