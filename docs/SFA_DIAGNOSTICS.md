# SFA Diagnostic Tests and Model Selection

This guide covers the comprehensive suite of diagnostic tests and model selection tools available in PanelBox for Stochastic Frontier Analysis (SFA).

## Overview

After estimating an SFA model, it's crucial to validate the specification and assess the robustness of efficiency estimates. PanelBox provides several categories of diagnostic tests:

1. **Inefficiency Presence Tests** - Test whether inefficiency is significant
2. **Distribution Selection** - Compare alternative distributional assumptions
3. **Variance Decomposition** - Quantify sources of variation
4. **Functional Form Tests** - Test returns to scale and functional specification
5. **Panel Model Selection** - Choose between fixed and random effects (TFE vs TRE)

---

## 1. Testing for Presence of Inefficiency

### The Fundamental Question: Is SFA Necessary?

Before proceeding with SFA, we should test whether the inefficiency component is statistically significant. If not, OLS regression may be adequate.

### Test Specification

**Null Hypothesis:** σ²_u = 0 (no inefficiency, OLS is sufficient)
**Alternative:** σ²_u > 0 (inefficiency is present, SFA is needed)

Since σ²_u is constrained to be non-negative (boundary parameter), the standard LR test doesn't apply. Instead, we use the **mixed chi-square distribution** (Kodde & Palm 1986):

- For **half-normal** or **exponential**: LR ~ ½χ²(0) + ½χ²(1)
- For **truncated-normal**: LR ~ ½χ²(1) + ½χ²(2)

### Usage

```python
from panelbox.frontier import StochasticFrontier, inefficiency_presence_test

# Estimate SFA model
sf = StochasticFrontier(
    data=df,
    depvar='log_output',
    exog=['log_capital', 'log_labor'],
    frontier='production',
    dist='half_normal'
)
result = sf.fit()

# Test for inefficiency presence
# (Automatically included in result.summary() if OLS loglik is available)
ineff_test = inefficiency_presence_test(
    loglik_sfa=result.loglik,
    loglik_ols=ols_loglik,  # From OLS estimation
    residuals_ols=ols_residuals,
    frontier_type='production',
    distribution='half_normal'
)

print(f"LR statistic: {ineff_test['lr_statistic']:.4f}")
print(f"P-value: {ineff_test['pvalue']:.4f}")
print(f"Conclusion: {ineff_test['conclusion']}")
print(f"Skewness: {ineff_test['skewness']:.4f}")
```

### Skewness Diagnostic

In addition to the LR test, the function performs a **skewness check** on OLS residuals:

- **Production frontier:** Residuals should have **negative skewness** (inefficiency reduces output)
- **Cost frontier:** Residuals should have **positive skewness** (inefficiency increases cost)

Wrong skewness suggests:
- Incorrect frontier type (production vs cost)
- Presence of outliers
- Misspecified distributional assumption

### Interpretation

```python
if ineff_test['pvalue'] < 0.05:
    print("✓ Inefficiency is significant. SFA is appropriate.")
else:
    print("✗ No evidence of inefficiency. OLS may be sufficient.")

if ineff_test['skewness_warning']:
    print(f"⚠ {ineff_test['skewness_warning']}")
```

---

## 2. Distribution Selection

### Comparing Distributional Assumptions

SFA models allow different distributions for the inefficiency term u:
- Half-normal: u ~ N⁺(0, σ²_u)
- Exponential: u ~ Exp(λ)
- Truncated-normal: u ~ N⁺(μ, σ²_u)
- Gamma: u ~ Gamma(P, λ)

The choice affects:
- Efficiency estimates
- Interpretation of parameters
- Model fit

### Automatic Comparison

```python
# Estimate base model
result = sf.fit()

# Compare with other distributions automatically
comparison = result.compare_distributions(
    distributions=['half_normal', 'exponential', 'truncated_normal']
)

print(comparison)
#   Distribution      Log-Likelihood    AIC       BIC    Best AIC  Best BIC  Rank AIC
# 0 truncated_normal      -117.5       243.1     252.4      True      True       1
# 1 half_normal           -119.6       245.3     252.1     False     False       2
# 2 exponential           -120.9       247.8     254.6     False     False       3
```

### Nested Models: LR Test

For **nested distributions** (e.g., half-normal ⊂ truncated-normal when μ = 0):

```python
from panelbox.frontier import compare_nested_distributions

# Estimate both models
result_hn = fit_sfa(..., dist='half_normal')
result_tn = fit_sfa(..., dist='truncated_normal')

# Test H0: μ = 0 (half-normal adequate)
lr_result = compare_nested_distributions(
    loglik_restricted=result_hn.loglik,
    loglik_unrestricted=result_tn.loglik,
    dist_restricted='half_normal',
    dist_unrestricted='truncated_normal'
)

print(f"LR statistic: {lr_result['lr_statistic']:.4f}")
print(f"P-value: {lr_result['pvalue']:.4f}")
print(f"Recommendation: {lr_result['conclusion']}")
```

### Non-Nested Models: Vuong Test

For **non-nested distributions** (e.g., half-normal vs exponential):

```python
from panelbox.frontier import vuong_test

# Get observation-level log-likelihoods
# (requires modification of likelihood function to return individual contributions)
vuong_result = vuong_test(
    loglik1=ll_halfnormal_obs,  # Array of log-likelihoods
    loglik2=ll_exponential_obs,
    model1_name='Half-Normal',
    model2_name='Exponential'
)

print(f"Vuong statistic: {vuong_result['statistic']:.4f}")
print(f"P-value: {vuong_result['pvalue_two_sided']:.4f}")
print(f"Preferred: {vuong_result['conclusion']}")
```

---

## 3. Variance Decomposition

### Understanding Sources of Variation

Total variation in the composite error ε = v - u can be decomposed:

**For standard SFA (two components):**
- γ = σ²_u / (σ²_v + σ²_u) — proportion due to inefficiency
- λ = σ_u / σ_v — ratio of standard deviations

**For True RE (three components):**
- γ_v = σ²_v / (σ²_v + σ²_u + σ²_w) — noise
- γ_u = σ²_u / (σ²_v + σ²_u + σ²_w) — inefficiency
- γ_w = σ²_w / (σ²_v + σ²_u + σ²_w) — heterogeneity

### Usage

```python
# Standard SFA
var_decomp = result.variance_decomposition(ci_level=0.95, method='delta')

print(f"γ (inefficiency share): {var_decomp['gamma']:.4f}")
print(f"95% CI: [{var_decomp['gamma_ci'][0]:.4f}, {var_decomp['gamma_ci'][1]:.4f}]")
print(f"λ (ratio σ_u/σ_v): {var_decomp['lambda_param']:.4f}")
print(f"95% CI: [{var_decomp['lambda_ci'][0]:.4f}, {var_decomp['lambda_ci'][1]:.4f}]")
print(f"\n{var_decomp['interpretation']}")
```

### Confidence Intervals

Two methods are available:

1. **Delta Method** (default): Fast, analytical
2. **Bootstrap**: Robust, but slower

```python
# Delta method (fast)
var_decomp_delta = result.variance_decomposition(method='delta')

# Bootstrap (robust)
var_decomp_boot = result.variance_decomposition(method='bootstrap')
```

### Interpretation

| γ Value | Interpretation |
|---------|----------------|
| γ < 0.1 | Inefficiency negligible. OLS may be adequate. |
| 0.1 ≤ γ ≤ 0.3 | Inefficiency minor but present. |
| 0.3 < γ < 0.7 | Both noise and inefficiency important. |
| 0.7 ≤ γ ≤ 0.9 | Inefficiency dominant. |
| γ > 0.9 | Nearly deterministic frontier. Check specification. |

### True RE Decomposition

```python
# For True Random Effects models
var_decomp_tre = tre_result.variance_decomposition()

print(f"Noise (v):         {100*var_decomp_tre['gamma_v']:.1f}%")
print(f"Inefficiency (u):  {100*var_decomp_tre['gamma_u']:.1f}%")
print(f"Heterogeneity (w): {100*var_decomp_tre['gamma_w']:.1f}%")
# Note: γ_v + γ_u + γ_w = 1
```

---

## 4. Functional Form Tests

### Returns to Scale (RTS) Test

For Cobb-Douglas production function: ln(y) = β₀ + β₁·ln(K) + β₂·ln(L) + v - u

**RTS = β₁ + β₂** (sum of input elasticities)

- RTS > 1: Increasing returns to scale
- RTS = 1: Constant returns to scale
- RTS < 1: Decreasing returns to scale

### Testing CRS

```python
# Test H0: RTS = 1 (constant returns to scale)
rts_test = result.returns_to_scale_test(
    input_vars=['log_capital', 'log_labor'],
    alpha=0.05
)

print(f"RTS estimate: {rts_test['rts']:.4f}")
print(f"Standard error: {rts_test['rts_se']:.4f}")
print(f"Test statistic: {rts_test['test_statistic']:.4f}")
print(f"P-value: {rts_test['pvalue']:.4f}")
print(f"Conclusion: {rts_test['conclusion']}")  # 'CRS', 'IRS', or 'DRS'
print(f"\n{rts_test['interpretation']}")
```

### Elasticities

**Cobb-Douglas:** Elasticities are constant (= coefficients)

```python
# Get elasticities (constant for Cobb-Douglas)
elasticities = result.elasticities(input_vars=['log_capital', 'log_labor'])
print(elasticities)
# log_capital    0.35
# log_labor      0.65
```

**Translog:** Elasticities vary by observation

```python
# For Translog specification
elasticities_df = result.elasticities(
    translog=True,
    translog_vars=['ln_K', 'ln_L']
)

print(elasticities_df.describe())
#           ε_K       ε_L
# mean     0.35      0.65
# std      0.05      0.08
# ...
```

### Efficient Scale

For Translog, find input levels where RTS = 1:

```python
eff_scale = result.efficient_scale(translog_vars=['ln_K', 'ln_L'])

print(f"Efficient scale:")
print(f"  ln(K): {eff_scale['efficient_scale'][0]:.4f}")
print(f"  ln(L): {eff_scale['efficient_scale'][1]:.4f}")
print(f"RTS at efficient scale: {eff_scale['rts_at_efficient']:.4f}")
print(f"Elasticities: {eff_scale['elasticities']}")
print(f"Converged: {eff_scale['converged']}")
```

### Cobb-Douglas vs Translog

Test whether Translog flexibility is statistically justified:

```python
# Estimate both models
cd_result = fit_sfa(data, depvar='ln_y', exog=['ln_K', 'ln_L'], ...)

# Add Translog terms
from panelbox.frontier import add_translog
data_tl = add_translog(data, variables=['ln_K', 'ln_L'])
tl_result = fit_sfa(data_tl, depvar='ln_y', exog=['ln_K', 'ln_L', 'ln_K_sq', ...], ...)

# Compare
comparison = cd_result.compare_functional_form(tl_result, alpha=0.05)

print(f"LR statistic: {comparison['lr_statistic']:.4f}")
print(f"Degrees of freedom: {comparison['df']}")
print(f"P-value: {comparison['pvalue']:.4f}")
print(f"Recommendation: {comparison['conclusion']}")  # 'cobb_douglas' or 'translog'
print(f"\nAIC: CD = {comparison['aic_cd']:.2f}, TL = {comparison['aic_tl']:.2f}")
print(f"BIC: CD = {comparison['bic_cd']:.2f}, TL = {comparison['bic_tl']:.2f}")
print(f"\n{comparison['interpretation']}")
```

---

## 5. Panel Model Selection: TFE vs TRE

### Hausman Test for Panel Models

For panel data, choose between:
- **True Fixed Effects (TFE):** Allows correlation between w_i and X
- **True Random Effects (TRE):** Assumes w_i ⊥ X (more efficient if valid)

```python
from panelbox.frontier import hausman_test_tfe_tre

# Estimate both models
tfe_result = fit_tfe(...)
tre_result = fit_tre(...)

# Hausman test
hausman = hausman_test_tfe_tre(
    params_tfe=tfe_result.params.values,
    params_tre=tre_result.params.values,
    vcov_tfe=tfe_result.cov,
    vcov_tre=tre_result.cov,
    param_names=tfe_result.params.index.tolist()
)

print(f"Hausman statistic: {hausman['statistic']:.4f}")
print(f"Degrees of freedom: {hausman['df']}")
print(f"P-value: {hausman['pvalue']:.4f}")
print(f"Recommendation: {hausman['conclusion']}")  # 'TFE' or 'TRE'
print(f"\n{hausman['interpretation']}")
```

### Interpretation

| P-value | Decision |
|---------|----------|
| p < 0.05 | Reject H0 → Use **TFE** (w_i correlated with X) |
| p ≥ 0.05 | Do not reject → Use **TRE** (more efficient) |

---

## 6. Complete Diagnostic Workflow

### Recommended Workflow

```python
from panelbox.frontier import StochasticFrontier

# 1. Estimate base model
sf = StochasticFrontier(
    data=df,
    depvar='log_output',
    exog=['log_capital', 'log_labor'],
    frontier='production',
    dist='half_normal'
)
result = sf.fit()

# 2. Check summary (includes inefficiency test and variance decomposition)
print(result.summary(include_diagnostics=True))

# 3. Compare distributions
dist_comparison = result.compare_distributions(
    distributions=['half_normal', 'exponential', 'truncated_normal']
)
print("\nDistribution Comparison:")
print(dist_comparison)

# 4. Variance decomposition
var_decomp = result.variance_decomposition(method='delta')
print(f"\nVariance Decomposition:")
print(f"  γ = {var_decomp['gamma']:.4f} [{var_decomp['gamma_ci'][0]:.4f}, {var_decomp['gamma_ci'][1]:.4f}]")
print(f"  {var_decomp['interpretation']}")

# 5. Test returns to scale
rts_test = result.returns_to_scale_test(input_vars=['log_capital', 'log_labor'])
print(f"\nReturns to Scale Test:")
print(f"  RTS = {rts_test['rts']:.4f} (p = {rts_test['pvalue']:.4f})")
print(f"  Conclusion: {rts_test['conclusion']}")

# 6. Get efficiency estimates
efficiency = result.efficiency(estimator='bc')
print(f"\nMean Efficiency: {efficiency['efficiency'].mean():.4f}")
```

---

## References

1. **Kodde, D. A., & Palm, F. C. (1986).** Wald criteria for jointly testing equality and inequality restrictions. *Econometrica*, 1243-1248.

2. **Coelli, T. J. (1995).** Estimators and hypothesis tests for a stochastic frontier function: A Monte Carlo analysis. *Journal of Productivity Analysis*, 6(4), 247-268.

3. **Vuong, Q. H. (1989).** Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 307-333.

4. **Greene, W. H. (2005).** Fixed and random effects in stochastic frontier models. *Journal of Productivity Analysis*, 23(1), 7-32.

5. **Hausman, J. A. (1978).** Specification tests in econometrics. *Econometrica*, 1251-1271.

---

## See Also

- [SFA Model Guide](./guides/sfa_guide.md)
- [True Models Guide](./TRUE_MODELS_GUIDE.md)
- [API Reference](./api/frontier.md)
