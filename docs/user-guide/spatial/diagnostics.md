---
title: "Spatial Diagnostics"
description: "Diagnostic tests for spatial autocorrelation, model selection, and validation in PanelBox spatial panel models."
---

# Spatial Diagnostics

!!! info "Quick Reference"
    **Module:** `panelbox.diagnostics.spatial`
    **Key tests:** Moran's I, LM-lag, LM-error, Robust LM, LR tests
    **Purpose:** Detect spatial dependence, select spatial model, validate estimation

## Overview

Spatial diagnostics serve three purposes in the spatial econometrics workflow:

1. **Detection**: Is there spatial dependence in my data? (Moran's I)
2. **Classification**: What type of spatial dependence? Lag or error? (LM tests)
3. **Validation**: Did my spatial model adequately capture the dependence? (post-estimation tests)

This page covers the complete diagnostic workflow, from initial OLS-based tests to post-estimation validation. The testing strategy follows the classical Anselin (1988) approach, augmented with robust tests and model comparison metrics.

## Diagnostic Workflow

The recommended workflow proceeds in stages:

```text
1. Fit standard panel model (OLS/FE/RE)
          |
2. Test residuals: Moran's I
          |
    Significant? ──No──> Standard panel model is adequate
          |
         Yes
          |
3. Run LM tests: LM-lag and LM-error
          |
    ┌─────┴─────┐
    |           |
  Only LM-lag  Only LM-error  Both significant
  significant  significant         |
    |           |          Run Robust LM tests
    |           |               |
   SAR         SEM       ┌─────┴─────┐
                         |           |
                    Robust LM-lag  Robust LM-error
                    dominates      dominates
                         |           |
                        SAR         SEM
                                     |
                              Both significant → SDM or GNS
```

## Moran's I Test

### Theory

Moran's I tests for spatial autocorrelation in a variable or in model residuals. The test statistic is:

$$I = \frac{n}{S_0} \frac{e'We}{e'e}$$

where:

- $e$ is the vector of residuals (or variable values, centered)
- $W$ is the spatial weight matrix
- $S_0 = \sum_i \sum_j w_{ij}$ is the sum of all weights
- $n$ is the number of observations

Under $H_0$ (no spatial autocorrelation):

$$E[I] = \frac{-1}{n-1}, \quad Z = \frac{I - E[I]}{\sqrt{\text{Var}(I)}} \sim N(0, 1)$$

### Interpretation

| Moran's I Value | Interpretation |
|-----------------|---------------|
| $I > E[I]$ (positive) | **Clustering**: similar values near each other |
| $I \approx E[I]$ | **Random**: no spatial pattern |
| $I < E[I]$ (negative) | **Dispersion**: dissimilar values near each other |

### Code Example

```python
from panelbox import FixedEffects
from panelbox.diagnostics.spatial import MoranIPanelTest

# Step 1: Fit standard panel model
fe_model = FixedEffects("y ~ x1 + x2", data, "region", "year")
fe_results = fe_model.fit()

# Step 2: Test residuals for spatial autocorrelation
moran = MoranIPanelTest(fe_results.resid, W)
result = moran.run()

print(f"Moran's I statistic: {result.statistic:.4f}")
print(f"Expected value:      {result.expected:.4f}")
print(f"Z-score:             {result.z_score:.4f}")
print(f"p-value:             {result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("Significant spatial autocorrelation detected.")
    print("Proceed with LM tests to determine model type.")
else:
    print("No significant spatial autocorrelation.")
    print("Standard panel model is adequate.")
```

!!! tip "Moran's I Scatterplot"
    The Moran's I scatterplot plots the variable against its spatial lag ($Wy$). A positive slope indicates positive spatial autocorrelation (clustering). The four quadrants correspond to High-High, Low-Low, High-Low, and Low-High spatial clusters.

## LM Tests for Spatial Dependence

### Theory

The Lagrange Multiplier (LM) tests are computed from OLS residuals and do not require estimating the spatial model. They test specific forms of spatial dependence:

**LM-lag** ($H_0: \rho = 0$ in SAR model):

$$LM_{lag} = \frac{(e'Wy / \hat{\sigma}^2)^2}{J_\rho}$$

where $J_\rho$ is a function of $W$, $X$, and $\hat{\sigma}^2$.

**LM-error** ($H_0: \lambda = 0$ in SEM model):

$$LM_{error} = \frac{(e'We / \hat{\sigma}^2)^2}{\text{tr}(W'W + W^2)}$$

**Robust LM-lag** (adjusts for potential spatial error):

$$RLM_{lag} = LM_{lag} - \text{correction for } \lambda$$

**Robust LM-error** (adjusts for potential spatial lag):

$$RLM_{error} = LM_{error} - \text{correction for } \rho$$

### Decision Rule

| Test Result | Recommendation |
|---|---|
| LM-lag significant, LM-error not | Use **SAR** |
| LM-error significant, LM-lag not | Use **SEM** |
| Both significant, Robust LM-lag > Robust LM-error | Use **SAR** (or SDM) |
| Both significant, Robust LM-error > Robust LM-lag | Use **SEM** (or SDM) |
| Both robust tests significant | Use **SDM** or **GNS** |
| Neither significant | No spatial model needed |

### Code Example

```python
from panelbox.diagnostics.spatial import (
    LMSpatialLagTest,
    LMSpatialErrorTest,
    RobustLMSpatialLagTest,
    RobustLMSpatialErrorTest,
)

# Compute LM tests from OLS residuals
lm_lag = LMSpatialLagTest(fe_results, W)
lm_error = LMSpatialErrorTest(fe_results, W)
rlm_lag = RobustLMSpatialLagTest(fe_results, W)
rlm_error = RobustLMSpatialErrorTest(fe_results, W)

# Run all tests
results = {
    'LM-lag': lm_lag.run(),
    'LM-error': lm_error.run(),
    'Robust LM-lag': rlm_lag.run(),
    'Robust LM-error': rlm_error.run(),
}

# Print summary
print(f"{'Test':<20} {'Statistic':>10} {'p-value':>10} {'Significant':>12}")
print("-" * 55)
for name, r in results.items():
    sig = "***" if r.pvalue < 0.001 else "**" if r.pvalue < 0.01 else "*" if r.pvalue < 0.05 else ""
    print(f"{name:<20} {r.statistic:>10.4f} {r.pvalue:>10.4f} {sig:>12}")
```

## Model Comparison

### Information Criteria

After fitting multiple spatial models, compare them using AIC and BIC:

```python
from panelbox.models.spatial import SpatialLag, SpatialError, SpatialDurbin

# Fit competing models
sar = SpatialLag("y ~ x1 + x2", data, "region", "year", W=W)
sar_res = sar.fit(effects='fixed', method='qml')

sem = SpatialError("y ~ x1 + x2", data, "region", "year", W=W)
sem_res = sem.fit(effects='fixed', method='gmm')

sdm = SpatialDurbin("y ~ x1 + x2", data, "region", "year", W=W)
sdm_res = sdm.fit(effects='fixed', method='qml')

# Compare
print(f"{'Model':<8} {'Log-lik':>10} {'AIC':>10} {'BIC':>10} {'Pseudo R2':>10}")
print("-" * 50)
for name, res in [('SAR', sar_res), ('SEM', sem_res), ('SDM', sdm_res)]:
    print(f"{name:<8} {res.llf:>10.1f} {res.aic:>10.1f} {res.bic:>10.1f} "
          f"{res.rsquared_pseudo:>10.4f}")
```

**Selection rules:**

- Lower AIC favors better prediction (less penalty for complexity)
- Lower BIC favors parsimony (stronger penalty for additional parameters)
- When AIC and BIC disagree, consider the research goal (prediction vs. explanation)

### Likelihood Ratio Tests

For nested models estimated by ML, use the LR test:

$$LR = 2(\ell_{\text{unrestricted}} - \ell_{\text{restricted}}) \sim \chi^2(q)$$

where $q$ is the number of restrictions.

```python
from scipy import stats

# LR test: SDM vs SAR (restriction: theta = 0)
lr_stat = 2 * (sdm_res.llf - sar_res.llf)
df = 2  # number of theta parameters (x1, x2)
p_value = 1 - stats.chi2.cdf(lr_stat, df)
print(f"SDM vs SAR: LR = {lr_stat:.2f}, df = {df}, p = {p_value:.4f}")

# Or use GNS test_restrictions for formal tests
from panelbox.models.spatial import GeneralNestingSpatial

gns = GeneralNestingSpatial("y ~ x1 + x2", data, "region", "year",
                             W1=W, W2=W, W3=W)
gns_res = gns.fit(effects='fixed', method='ml')

# Test nested models
test = gns.test_restrictions(restrictions={'theta': 0, 'lambda': 0})
print(f"GNS vs SAR: LR = {test['lr_statistic']:.2f}, p = {test['p_value']:.4f}")
```

## Post-Estimation Diagnostics

### Residual Spatial Autocorrelation

After fitting a spatial model, the residuals should be free of spatial autocorrelation. Re-run Moran's I on the spatial model residuals:

```python
# Moran's I on spatial model residuals
moran_post = MoranIPanelTest(sar_res.resid, W)
post_result = moran_post.run()

print(f"Post-estimation Moran's I: {post_result.statistic:.4f}")
print(f"p-value: {post_result.pvalue:.4f}")

if post_result.pvalue > 0.05:
    print("No remaining spatial autocorrelation. Model is adequate.")
else:
    print("Spatial autocorrelation persists. Consider a different specification.")
```

!!! warning
    If residual Moran's I is still significant after fitting a SAR model, the spatial dependence may be more complex. Consider switching to SDM or GNS.

### Goodness of Fit

```python
# Pseudo R-squared
print(f"Pseudo R-squared: {results.rsquared_pseudo:.4f}")

# Predicted vs actual
import numpy as np
y_actual = data["y"].values
y_pred = results.fitted_values
correlation = np.corrcoef(y_actual, y_pred)[0, 1]
print(f"Correlation (predicted vs actual): {correlation:.4f}")
```

### Hansen J-Test (GMM Models)

For models estimated by GMM (SEM, Dynamic Spatial Panel), the Hansen J-test checks instrument validity:

$$J = n \cdot \hat{g}' \hat{W}^{-1} \hat{g} \sim \chi^2(L - K)$$

where $L$ is the number of instruments and $K$ is the number of parameters.

```python
# Hansen J-test is reported in the GMM summary
sem_res = sem.fit(effects='fixed', method='gmm', n_lags=2)
print(sem_res.summary())  # Includes J-test if applicable
```

- **$H_0$: instruments are valid** (orthogonal to errors)
- **Do not reject** ($p > 0.05$): instruments are valid
- **Reject** ($p < 0.05$): instruments may be invalid; reconsider model or instruments

## Diagnostic Checklist

Use this checklist to ensure a thorough spatial analysis:

- [ ] **Moran's I on OLS residuals**: is spatial autocorrelation present?
- [ ] **LM tests**: which type of spatial dependence (lag, error, both)?
- [ ] **Robust LM tests**: which dominates when both LM tests are significant?
- [ ] **Model estimation**: fit the recommended model (SAR, SEM, or SDM)
- [ ] **Post-estimation Moran's I**: is spatial autocorrelation eliminated?
- [ ] **Model comparison**: AIC/BIC across competing specifications
- [ ] **Effect decomposition**: direct, indirect, total effects (for SAR/SDM)
- [ ] **Weight matrix sensitivity**: do results hold with different $W$?
- [ ] **Hansen J-test**: are instruments valid? (for GMM models)

## Troubleshooting

### Model Does Not Converge

1. **Check the weight matrix**: ensure it is properly row-standardized and has no islands
2. **Simplify the model**: start with SAR or SEM before trying SDM or GNS
3. **Adjust optimizer settings**: increase `maxiter`, try different `optim_method`
4. **Check multicollinearity**: VIF > 10 for any covariate may cause problems
5. **Scale variables**: very large or very small values can cause numerical issues

### Spatial Parameter at Boundary ($|\rho| \approx 1$)

If $\rho$ or $\lambda$ is estimated at or near the boundary:

- Check weight matrix normalization
- Consider a different $W$ specification
- May indicate model misspecification
- Try a simpler model

### Residual Autocorrelation Persists

If Moran's I is still significant after fitting a spatial model:

- **SAR residuals significant**: try SDM (add $WX$ terms) or SEM
- **SEM residuals significant**: try SAR or SDM (spatial lag may be needed)
- **SDM residuals significant**: try GNS (add spatial error term) or dynamic spatial

### Estimation Is Slow

| Problem | Solution |
|---------|----------|
| Large $N$ (> 500) | Use sparse weight matrices: `W.to_sparse()` |
| Large $N$ (> 5,000) | Model automatically uses sparse LU for log-det |
| Large $N$ (> 10,000) | Model automatically uses Chebyshev approximation |
| Many covariates | Consider dimensionality reduction |
| Complex model (GNS) | Start with simpler models, use GNS only for model selection |

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Includes full diagnostic workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Weight Matrices](spatial-weights.md) — Weight matrix construction and properties
- [Spatial Lag (SAR)](spatial-lag.md) — When LM-lag is significant
- [Spatial Error (SEM)](spatial-error.md) — When LM-error is significant
- [Spatial Durbin (SDM)](spatial-durbin.md) — When both LM tests are significant
- [General Nesting Spatial (GNS)](gns.md) — LR tests for nested models
- [Direct, Indirect, and Total Effects](spatial-effects.md) — Effect interpretation
- [Choosing a Spatial Model](choosing-model.md) — Decision framework

## References

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic.
2. Anselin, L., Bera, A.K., Florax, R., and Yoon, M.J. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics*, 26(1), 77-104.
3. Moran, P.A.P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1-2), 17-23.
4. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
5. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
