# Advanced GMM Estimators

## Overview

This guide provides theoretical background and practical guidance for advanced Generalized Method of Moments (GMM) estimators implemented in PanelBox.

**Contents:**
- [Continuous Updated GMM (CUE-GMM)](#continuous-updated-gmm)
- [Bias-Corrected GMM](#bias-corrected-gmm)
- [Diagnostic Tests](#diagnostic-tests)
- [Practical Recommendations](#practical-recommendations)

---

## Continuous Updated GMM (CUE-GMM)

### Motivation

Standard two-step GMM uses a fixed weighting matrix $\hat{W}$ computed at first-step estimates:

```
Step 1: β̂₁ = argmin g(β)' I g(β)           (W = I)
Step 2: β̂₂ = argmin g(β)' Ŵ⁻¹ g(β)         (Ŵ = Var(g(β̂₁)))
```

This approach has limitations:
- **Finite-sample bias:** Estimates can be biased in moderate samples
- **Sensitivity to normalization:** Results depend on how moments are scaled
- **Suboptimal efficiency:** Not always the most efficient estimator

### CUE-GMM Solution

CUE-GMM continuously updates the weighting matrix as a function of parameters:

$$
\hat{\beta}^{CUE} = \argmin_\beta Q(\beta) = g(\beta)' W(\beta)^{-1} g(\beta)
$$

where $W(\beta) = \frac{1}{N} \sum_{i=1}^N g_i(\beta) g_i(\beta)'$ is recomputed at each candidate $\beta$.

### Theoretical Properties

**Theorem (Hansen, Heaton, Yaron 1996):**

Under regularity conditions, CUE-GMM is:

1. **Consistent:** $\hat{\beta}^{CUE} \xrightarrow{p} \beta_0$
2. **Asymptotically normal:** $\sqrt{N}(\hat{\beta}^{CUE} - \beta_0) \xrightarrow{d} N(0, V)$
3. **Efficient:** Achieves the GMM efficiency bound
4. **Invariant:** Results don't depend on moment normalization

**Finite-sample advantages:**
- Lower bias than two-step GMM
- Better coverage of confidence intervals
- More reliable in moderate samples (N = 100-500)

### Variance Estimation

For CUE-GMM, the variance-covariance matrix is:

$$
\hat{V} = (\bar{G}' \hat{W}^{-1} \bar{G})^{-1}
$$

where $\bar{G} = \frac{\partial g(\hat{\beta})}{\partial \beta'}$ is the Jacobian of moment conditions.

### When to Use CUE-GMM

✅ **Recommended:**
- Moderate sample sizes (N = 100-1000)
- Concerns about normalization sensitivity
- When finite-sample properties matter

⚠️ **Cautions:**
- Computationally expensive (recomputes W at each iteration)
- Requires good starting values
- May have convergence issues with weak instruments

### Implementation in PanelBox

```python
from panelbox.gmm import ContinuousUpdatedGMM

model = ContinuousUpdatedGMM(
    data=data,
    dep_var='y',
    exog_vars=['x1', 'x2'],
    instruments=['z1', 'z2', 'z3'],
    weighting='hac',      # HAC-robust weighting
    bandwidth='auto'      # Automatic Newey-West bandwidth
)

results = model.fit()
print(results.summary())

# Check overidentification
j_test = model.j_statistic()
if j_test['reject']:
    print("Warning: J-test rejects model specification")
```

---

## Bias-Corrected GMM

### The Bias Problem in Dynamic Panels

For dynamic panel models with fixed effects:

$$
y_{it} = \rho y_{i,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

Standard GMM (Arellano-Bond) has **finite-sample bias**:

$$
E[\hat{\rho}^{GMM} - \rho] \approx \frac{B(\rho)}{N} + O(N^{-2})
$$

**Source of bias:** Correlation between differenced instruments $\Delta y_{i,t-2}$ and differenced errors $\Delta \varepsilon_{it}$ through the fixed effect.

**Magnitude:** For $N = 100$, $T = 10$, bias can be 10-20% of the true parameter!

### Hahn-Kuersteiner Correction

**Idea:** Derive analytical expression for $B(\rho)$ and subtract it:

$$
\hat{\rho}^{BC} = \hat{\rho}^{GMM} - \frac{\hat{B}(\hat{\rho})}{N}
$$

This reduces bias to $O(N^{-2})$ — much smaller!

### Bias Formula (Simplified)

For AR(1) coefficient, the approximate bias is:

$$
B(\rho) \approx -\frac{1 + \rho}{T - 1}
$$

This is the **Nickell bias**. For $\rho = 0.7$, $T = 10$:
- Bias $\approx -\frac{1.7}{9} \approx -0.189$
- True $\rho = 0.7$ → Estimated $\hat{\rho} \approx 0.51$ (downward bias)

**Bias correction** brings estimate back toward 0.70.

### When to Use Bias Correction

✅ **Recommended:**
- Dynamic panels with moderate $N$ and $T$ (both 50-200)
- Lagged dependent variables
- Concerns about bias in policy analysis

⚠️ **Not recommended:**
- Very small $N$ or $T$ (< 30)
- Static panels (no lags)
- Very large $N$ (> 1000) where bias is negligible

### Implementation

```python
from panelbox.gmm import BiasCorrectedGMM

model = BiasCorrectedGMM(
    data=panel_data,
    dep_var='y',
    lags=[1],              # Include y_{t-1}
    exog_vars=['x'],
    bias_order=1,          # First-order correction
    min_n=50,              # Warn if N < 50
    min_t=10               # Warn if T < 10
)

results = model.fit()

# Check magnitude of bias correction
bias_magnitude = model.bias_magnitude()
print(f"Bias correction: {bias_magnitude:.4f}")

# Compare with uncorrected
print(f"Uncorrected: {model.params_uncorrected_}")
print(f"Corrected:   {model.params_}")
```

---

## Diagnostic Tests

### Hansen J-Test for Overidentification

**Null hypothesis:** All moment conditions are valid (model correctly specified)

**Test statistic:**

$$
J = N \times Q(\hat{\beta}) \sim \chi^2(L - K)
$$

where $L$ = # instruments, $K$ = # parameters

**Interpretation:**
- **High p-value (> 0.05):** Don't reject → Model appears valid
- **Low p-value (< 0.05):** Reject → Model misspecified or invalid instruments

**Example:**

```python
from panelbox.gmm.diagnostics import GMMDiagnostics

diagnostics = GMMDiagnostics(model, results)

# Hansen J-test is automatically included
print(diagnostics.summary())
```

### C-Statistic (Difference-in-Sargan)

Tests validity of a **subset** of instruments:

$$
C = J_{restricted} - J_{unrestricted} \sim \chi^2(\text{# tested instruments})
$$

**Use case:** Test if adding extra instruments improves or worsens specification.

**Example:**

```python
# Test if instrument z3 is valid
c_test = diagnostics.c_statistic(subset_indices=[3])

if c_test.pvalue < 0.05:
    print("Instrument z3 may be invalid")
```

### Weak Instruments Test

**Problem:** Weak instruments lead to:
- Biased estimates
- Invalid inference (t-tests wrong)
- Poor finite-sample approximations

**Cragg-Donald F-statistic:**

First-stage regression: $X_{endog} = Z \pi + v$

$$
F = \frac{R^2 / K}{(1 - R^2) / (N - K - 1)}
$$

**Rule of thumb (Stock-Yogo 2005):**
- $F < 10$: **Weak instruments** → Be very concerned
- $F \approx 10-20$: **Moderate** → Use caution
- $F > 20$: **Strong** → Instruments likely adequate

**Example:**

```python
weak_test = diagnostics.weak_instruments_test()

print(f"Cragg-Donald F: {weak_test['cragg_donald_f']:.2f}")
print(f"Status: {weak_test['warning_level']}")
```

---

## Practical Recommendations

### Choosing Between Estimators

| Scenario | Recommended Estimator | Reason |
|----------|----------------------|---------|
| Large N (> 1000), static | Two-step GMM | Computationally efficient |
| Moderate N (100-500), static | CUE-GMM | Better finite-sample properties |
| Dynamic panel, moderate N, T | Bias-Corrected GMM | Reduces Nickell bias |
| Large N (> 1000), dynamic | System GMM | Efficient and consistent |
| Small N (< 50) | Avoid GMM | Use alternative methods (FE, RE) |

### Workflow for Applied Research

1. **Start with diagnostics:**
   ```python
   diagnostics = GMMDiagnostics(model, results)
   print(diagnostics.summary())
   ```

2. **Check weak instruments:**
   - If F < 10, **stop** — find better instruments
   - If F > 20, proceed with confidence

3. **Check overidentification:**
   - If J-test rejects, investigate:
     - Use C-statistic to test subsets
     - Consider alternative moment conditions
     - Check for model misspecification

4. **For dynamic panels:**
   - Always report both standard and bias-corrected estimates
   - Document bias magnitude
   - Use if $N$ and $T$ are moderate

5. **Robustness checks:**
   - Different weighting (HAC vs cluster)
   - Different bandwidth (for HAC)
   - Different lag structures
   - Compare CUE vs two-step

### Common Pitfalls

❌ **Mistake:** Using GMM with very weak instruments (F < 5)
✅ **Solution:** Find stronger instruments or use alternative methods

❌ **Mistake:** Ignoring overidentification test rejection
✅ **Solution:** Investigate with C-statistic, consider model misspecification

❌ **Mistake:** Using bias correction with small samples (N < 30)
✅ **Solution:** Only use bias correction with N ≥ 50, T ≥ 10

❌ **Mistake:** Not checking convergence
✅ **Solution:** Always verify `model.converged_` before interpreting results

---

## Mathematical Details

### GMM Objective Function

The GMM criterion minimizes weighted moment conditions:

$$
Q(\beta) = g_N(\beta)' W_N g_N(\beta)
$$

where:
- $g_N(\beta) = \frac{1}{N} \sum_{i=1}^N g_i(\beta)$ — sample moments
- $W_N$ — weighting matrix (positive definite)

**First-order condition:**

$$
\frac{\partial Q}{\partial \beta} = 2 G' W_N g_N = 0
$$

where $G = \frac{\partial g_N}{\partial \beta'}$

### Optimal Weighting

The **optimal weighting matrix** is:

$$
W_{opt} = \Sigma^{-1} \quad \text{where} \quad \Sigma = E[g_i(\beta_0) g_i(\beta_0)']
$$

This gives the **efficient GMM estimator** with variance:

$$
Var(\hat{\beta}) = (G' \Sigma^{-1} G)^{-1}
$$

### HAC Weighting

For time series or panel data, use Newey-West HAC:

$$
\hat{\Sigma} = \Gamma_0 + \sum_{l=1}^L w(l) (\Gamma_l + \Gamma_l')
$$

where:
- $\Gamma_l = \frac{1}{N} \sum_t g_t g_{t-l}'$ — autocovariances
- $w(l) = 1 - \frac{l}{L+1}$ — Bartlett kernel weights
- $L = \lfloor 4(T/100)^{2/9} \rfloor$ — automatic bandwidth (Newey-West)

---

## References

### Key Papers

1. **Hansen, L.P., Heaton, J., & Yaron, A. (1996)**
   "Finite-Sample Properties of Some Alternative GMM Estimators."
   *Journal of Business & Economic Statistics*, 14(3), 262-280.
   - **CUE-GMM** theory and Monte Carlo evidence

2. **Hahn, J., & Kuersteiner, G. (2002)**
   "Asymptotically Unbiased Inference for a Dynamic Panel Model with Fixed Effects."
   *Econometrica*, 70(4), 1639-1657.
   - **Bias correction** for dynamic panels

3. **Newey, W.K., & West, K.D. (1987)**
   "A Simple, Positive Semi-Definite, HAC Covariance Matrix."
   *Econometrica*, 55(3), 703-708.
   - **HAC variance estimation**

4. **Stock, J.H., & Yogo, M. (2005)**
   "Testing for Weak Instruments in Linear IV Regression."
   - **Weak instruments diagnostics**

### Textbooks

- **Cameron, A.C., & Trivedi, P.K. (2005)**
  *Microeconometrics: Methods and Applications*
  - Comprehensive GMM treatment (Chapters 5-6)

- **Wooldridge, J.M. (2010)**
  *Econometric Analysis of Cross Section and Panel Data* (2nd ed.)
  - Panel GMM with applications (Chapters 8-9)

- **Hall, A.R. (2005)**
  *Generalized Method of Moments*
  - Advanced GMM theory

---

## Appendix: Notation

| Symbol | Meaning |
|--------|---------|
| $\beta$ | Parameter vector (K × 1) |
| $g_i(\beta)$ | Moment function for observation i (L × 1) |
| $g_N(\beta)$ | Sample average moments (L × 1) |
| $W$ | Weighting matrix (L × L) |
| $G$ | Jacobian of moments (L × K) |
| $\Sigma$ | Variance of moments (L × L) |
| $N$ | Sample size (observations or clusters) |
| $T$ | Time periods (for panels) |
| $L$ | Number of moment conditions |
| $K$ | Number of parameters |

**Overidentification:** $L > K$ (more moments than parameters)

**Just-identified:** $L = K$ (exactly identified)

**Underidentified:** $L < K$ (not estimable)

---

## See Also

- [API Reference: ContinuousUpdatedGMM](../api/gmm/cue_gmm.rst)
- [API Reference: BiasCorrectedGMM](../api/gmm/bias_corrected.rst)
- [API Reference: GMMDiagnostics](../api/gmm/diagnostics.rst)
- [Validation Report](../../tests/validation/VALIDATION_GMM.md)
- [Examples](../../examples/gmm/)
