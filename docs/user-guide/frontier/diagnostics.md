---
title: "SFA Diagnostics"
description: "Diagnostic tests, model validation, and specification testing for stochastic frontier models in PanelBox"
---

# SFA Diagnostics

!!! info "Quick Reference"
    **Module:** `panelbox.frontier.tests`
    **Import:** `from panelbox.frontier.tests import inefficiency_presence_test, skewness_test, vuong_test, lr_test, wald_test`
    **Stata equivalent:** Post-estimation tests
    **R equivalent:** `frontier::efficiencies()`, custom tests

## Overview

After estimating a stochastic frontier model, rigorous diagnostics are essential to validate the specification and assess the reliability of efficiency estimates. PanelBox provides a comprehensive suite of diagnostic tests organized into five categories:

1. **Inefficiency presence** -- Is SFA necessary, or is OLS sufficient?
2. **Distribution selection** -- Which distributional assumption fits best?
3. **Variance decomposition** -- How much variation is due to inefficiency vs noise?
4. **Functional form** -- Are returns to scale constant? Is Translog needed?
5. **Panel model selection** -- TFE vs TRE, time-varying vs time-invariant?

This page documents each test with its statistical foundation, PanelBox API, and interpretation guidelines.

## Diagnostic Checklist

Before reporting SFA results, verify each of these items:

| Step | Test | What to Check |
|---|---|---|
| 1 | Skewness test | OLS residuals have correct skewness sign |
| 2 | Inefficiency presence | LR test rejects $H_0: \sigma_u^2 = 0$ |
| 3 | Distribution comparison | AIC/BIC or Vuong test supports chosen distribution |
| 4 | Variance decomposition | $\gamma$ is in a reasonable range |
| 5 | Mean efficiency | Values are plausible for the industry |
| 6 | Convergence | Optimizer converged successfully |

## Detailed Guide

### 1. Testing for Inefficiency Presence

The fundamental question: **Is SFA necessary, or is OLS sufficient?**

**Hypotheses:**

- $H_0: \sigma_u^2 = 0$ (no inefficiency; OLS is adequate)
- $H_1: \sigma_u^2 > 0$ (inefficiency is present; SFA is needed)

Since $\sigma_u^2$ is constrained to be non-negative (boundary parameter), the standard LR test distribution does not apply. Under $H_0$, the LR statistic follows a **mixed chi-square** distribution (Kodde & Palm, 1986):

- For half-normal or exponential: $LR \sim \frac{1}{2}\chi^2(0) + \frac{1}{2}\chi^2(1)$
- For truncated-normal: $LR \sim \frac{1}{2}\chi^2(1) + \frac{1}{2}\chi^2(2)$

```python
from panelbox.frontier.tests import inefficiency_presence_test

test = inefficiency_presence_test(
    loglik_sfa=result.loglik,
    loglik_ols=ols_loglik,
    residuals_ols=ols_residuals,
    frontier_type="production",
    distribution="half_normal",
)

print(f"LR statistic: {test['lr_statistic']:.4f}")
print(f"P-value (mixed chi-sq): {test['pvalue']:.4f}")
print(f"Conclusion: {test['conclusion']}")
print(f"Skewness: {test['skewness']:.4f}")

if test['skewness_warning']:
    print(f"WARNING: {test['skewness_warning']}")
```

The function also performs a **skewness diagnostic**:

- Production frontier: OLS residuals should be **negatively skewed**
- Cost frontier: OLS residuals should be **positively skewed**

Wrong skewness suggests misspecification (wrong frontier type, outliers, or no inefficiency).

!!! note "Automatic Diagnostics"
    The inefficiency presence test is automatically included in `result.summary(include_diagnostics=True)` when OLS log-likelihood information is available.

### 2. Skewness Test

A quick preliminary diagnostic before fitting SFA:

```python
from panelbox.frontier.tests import skewness_test

skew = skewness_test(
    residuals=ols_residuals,
    frontier_type="production",
)

print(f"Skewness: {skew['skewness']:.4f}")
print(f"Expected sign: {skew['expected_sign']}")
print(f"Correct sign: {skew['correct_sign']}")

if skew['warning']:
    print(f"WARNING: {skew['warning']}")
```

### 3. Distribution Selection

#### Nested Distributions: LR Test

For nested distribution pairs (e.g., half-normal $\subset$ truncated-normal when $\mu = 0$):

```python
from panelbox.frontier.tests import compare_nested_distributions

lr = compare_nested_distributions(
    loglik_restricted=result_halfnormal.loglik,
    loglik_unrestricted=result_truncnormal.loglik,
    dist_restricted="half_normal",
    dist_unrestricted="truncated_normal",
)

print(f"LR statistic: {lr['lr_statistic']:.4f}")
print(f"P-value: {lr['pvalue']:.4f}")
print(f"Preferred: {lr['conclusion']}")
```

Common nested pairs:

| Restricted | Unrestricted | Restriction | df |
|---|---|---|---|
| half-normal | truncated-normal | $\mu = 0$ | 1 (mixed $\chi^2$) |
| exponential | gamma | $P = 1$ | 1 |

#### Non-Nested Distributions: Vuong Test

For non-nested distributions (e.g., half-normal vs exponential):

```python
from panelbox.frontier.tests import vuong_test

vuong = vuong_test(
    loglik1=ll_halfnormal_obs,  # Observation-level log-likelihoods
    loglik2=ll_exponential_obs,
    model1_name="half_normal",
    model2_name="exponential",
)

print(f"Vuong statistic: {vuong['statistic']:.4f}")
print(f"P-value (two-sided): {vuong['pvalue_two_sided']:.4f}")
print(f"Preferred: {vuong['conclusion']}")
```

!!! warning "Vuong Test Requirements"
    The Vuong test requires **observation-level** log-likelihoods (arrays), not just the total log-likelihood value. Both models must be estimated on the same sample, and the sample should be large ($N \gg 30$).

#### Automatic Distribution Comparison

Compare multiple distributions at once using AIC/BIC:

```python
comparison = result.compare_distributions(
    distributions=["half_normal", "exponential", "truncated_normal"]
)
print(comparison[["Distribution", "Log-Likelihood", "AIC", "BIC", "Mean Efficiency"]])
```

### 4. Variance Decomposition

The parameter $\gamma = \sigma_u^2 / (\sigma_v^2 + \sigma_u^2)$ measures the share of total variance attributable to inefficiency:

```python
var_decomp = result.variance_decomposition(ci_level=0.95, method="delta")

print(f"gamma: {var_decomp['gamma']:.4f}")
print(f"95% CI: [{var_decomp['gamma_ci'][0]:.4f}, {var_decomp['gamma_ci'][1]:.4f}]")
print(f"lambda: {var_decomp['lambda_param']:.4f}")
print(f"95% CI: [{var_decomp['lambda_ci'][0]:.4f}, {var_decomp['lambda_ci'][1]:.4f}]")
print(var_decomp['interpretation'])
```

**Interpretation:**

| $\gamma$ Value | Interpretation |
|---|---|
| $\gamma < 0.1$ | Inefficiency negligible. OLS may be adequate. |
| $0.1 \leq \gamma \leq 0.3$ | Inefficiency minor but present. |
| $0.3 < \gamma < 0.7$ | Both noise and inefficiency are important. |
| $0.7 \leq \gamma \leq 0.9$ | Inefficiency is dominant. |
| $\gamma > 0.9$ | Nearly deterministic frontier. Check specification. |

**Confidence interval methods:**

| Method | API | Characteristics |
|---|---|---|
| Delta method | `method="delta"` | Fast, analytical; default |
| Bootstrap | `method="bootstrap"` | Robust, slower |

#### Three-Component Decomposition (TRE)

For True Random Effects models with heterogeneity:

```python
var_decomp = tre_result.variance_decomposition()

print(f"Noise (gamma_v):         {var_decomp['gamma_v']:.4f}")
print(f"Inefficiency (gamma_u):  {var_decomp['gamma_u']:.4f}")
print(f"Heterogeneity (gamma_w): {var_decomp['gamma_w']:.4f}")
# gamma_v + gamma_u + gamma_w = 1
```

### 5. Functional Form Tests

#### Returns to Scale

Test $H_0: RTS = \sum_j \beta_j = 1$ (constant returns to scale):

```python
rts = result.returns_to_scale_test(
    input_vars=["log_labor", "log_capital"],
    alpha=0.05,
)

print(f"RTS = {rts['rts']:.4f} +/- {rts['rts_se']:.4f}")
print(f"Wald statistic: {rts['test_statistic']:.4f}")
print(f"P-value: {rts['pvalue']:.4f}")
print(f"Conclusion: {rts['conclusion']}")  # 'CRS', 'IRS', or 'DRS'
print(rts['interpretation'])
```

#### Elasticities

```python
# Cobb-Douglas: constant elasticities = coefficients
elas = result.elasticities(input_vars=["log_labor", "log_capital"])
print(elas)

# Translog: varying elasticities
elas_tl = result.elasticities(translog=True, translog_vars=["ln_K", "ln_L"])
print(elas_tl.describe())
```

#### Cobb-Douglas vs Translog

```python
comparison = cd_result.compare_functional_form(translog_result, alpha=0.05)

print(f"LR statistic: {comparison['lr_statistic']:.4f}")
print(f"df: {comparison['df']}")
print(f"P-value: {comparison['pvalue']:.4f}")
print(f"Preferred: {comparison['conclusion']}")
```

### 6. Panel Model Selection

#### Hausman Test: TFE vs TRE

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
```

#### LR Test: Nested Models

```python
from panelbox.frontier.tests import lr_test

# Test if time-varying is better than time-invariant
lr = lr_test(
    loglik_restricted=result_pitt_lee.loglik,
    loglik_unrestricted=result_bc92.loglik,
    df_diff=1,  # One additional parameter (eta)
)
print(f"LR = {lr['statistic']:.4f}, p = {lr['pvalue']:.4f}")
```

#### Wald Test

```python
from panelbox.frontier.tests import wald_test
import numpy as np

# Test if beta_1 = beta_2 = 0
R = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
w = wald_test(params=result.params.values, vcov=result.vcov, R=R)
print(f"Wald = {w['statistic']:.4f}, p = {w['pvalue']:.4f}")
```

#### Heterogeneity Significance (TRE)

```python
from panelbox.frontier.tests import heterogeneity_significance_test

het = heterogeneity_significance_test(
    sigma_w_sq=result_tre.sigma_w_sq,
    se_sigma_w_sq=result_tre.se["sigma_w_sq"],
)
print(f"z-statistic: {het['statistic']:.4f}")
print(f"P-value: {het['pvalue']:.4f}")
print(f"Conclusion: {het['conclusion']}")
```

## Complete Diagnostic Workflow

```python
from panelbox.frontier import StochasticFrontier

# 1. Estimate base model
sf = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_capital", "log_labor"],
    frontier="production",
    dist="half_normal",
)
result = sf.fit()

# 2. Full summary with diagnostics
print(result.summary(include_diagnostics=True))

# 3. Compare distributions
dist_comp = result.compare_distributions(
    distributions=["half_normal", "exponential", "truncated_normal"]
)
print("\nDistribution Comparison:")
print(dist_comp)

# 4. Variance decomposition
var_decomp = result.variance_decomposition(method="delta")
print(f"\nGamma: {var_decomp['gamma']:.4f}")
print(var_decomp['interpretation'])

# 5. Returns to scale
rts = result.returns_to_scale_test(input_vars=["log_capital", "log_labor"])
print(f"\nRTS = {rts['rts']:.4f} ({rts['conclusion']})")

# 6. Efficiency estimates
eff = result.efficiency(estimator="bc")
print(f"\nMean efficiency: {eff['efficiency'].mean():.4f}")
```

## Available Test Functions

| Function | Test | H0 |
|---|---|---|
| `inefficiency_presence_test` | Mixed chi-square LR | $\sigma_u^2 = 0$ |
| `skewness_test` | Residual skewness diagnostic | Correct sign for frontier type |
| `lr_test` | Likelihood ratio | Restricted model adequate |
| `wald_test` | Wald test for linear restrictions | $R\theta = r$ |
| `vuong_test` | Non-nested model comparison | Models equally close to truth |
| `compare_nested_distributions` | Nested distribution LR | Simpler distribution adequate |
| `hausman_test_tfe_tre` | Hausman test for TFE vs TRE | TRE is consistent |
| `heterogeneity_significance_test` | $\sigma_w^2 > 0$ test | No heterogeneity in TRE |

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| SFA Diagnostics | Complete diagnostic workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/05_testing_comparison.ipynb) |

## See Also

- [Production and Cost Frontiers](production-cost.md) -- SFA fundamentals
- [Panel SFA Models](panel-sfa.md) -- Panel model specifications
- [True Models (TFE/TRE)](true-models.md) -- Hausman test context
- [Sign Conventions](sign-convention.md) -- Skewness and sign interpretation
- [TFP Decomposition](tfp.md) -- Returns to scale and elasticities

## References

- Kodde, D. A., & Palm, F. C. (1986). Wald criteria for jointly testing equality and inequality restrictions. *Econometrica*, 1243-1248.
- Coelli, T. J. (1995). Estimators and hypothesis tests for a stochastic frontier function: A Monte Carlo analysis. *Journal of Productivity Analysis*, 6(4), 247-268.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 307-333.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 1251-1271.
- Greene, W. H. (2005). Fixed and random effects in stochastic frontier models. *Journal of Productivity Analysis*, 23(1), 7-32.
- Kumbhakar, S. C., & Lovell, C. A. K. (2000). *Stochastic Frontier Analysis*. Cambridge University Press.
