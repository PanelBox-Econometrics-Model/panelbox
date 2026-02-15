# Theory Guide: Panel Selection Models

## Overview

Sample selection models address a fundamental econometric problem: **outcomes are observed only for a non-random subsample**. This creates **selection bias** when the selection process is correlated with the outcome of interest.

**Classic Example:** Wage determination

- We observe wages only for employed individuals
- Employment decisions may correlate with potential wages
- Simply analyzing employed workers yields biased wage estimates

The **Panel Heckman model** (Wooldridge 1995) extends Heckman's (1979) two-step correction to panel data, accounting for individual heterogeneity and correlation over time.

---

## The Selection Problem

### Basic Setup

Consider a panel dataset with N individuals observed over T periods.

**Outcome Equation (of interest):**
```
y_it = X_it'β + α_i + ε_it
```

where:
- `y_it`: outcome of interest (e.g., wage)
- `X_it`: outcome regressors
- `α_i`: individual fixed/random effect
- `ε_it`: idiosyncratic error

**Selection Equation:**
```
d_it = 1[W_it'γ + η_i + v_it > 0]
```

where:
- `d_it = 1` if outcome observed, 0 otherwise
- `W_it`: selection regressors
- `η_i`: individual random effect in selection
- `v_it`: selection shock

**Key Assumption:** Outcome is observed only when `d_it = 1`:
```
y_it^obs = y_it  if d_it = 1
y_it^obs = ∅    if d_it = 0
```

### When Does Selection Bias Occur?

Selection bias arises when:

```
Corr(v_it, ε_it) = ρ ≠ 0
```

**Intuition:**

- **ρ > 0 (Positive selection):** High-outcome individuals more likely to be selected
  - Example: High-wage workers more likely to be employed

- **ρ < 0 (Negative selection):** Low-outcome individuals more likely to be selected
  - Example: Low-skilled workers more likely to participate in training programs

- **ρ = 0 (No selection bias):** Selection independent of outcome
  - OLS on selected sample is unbiased

---

## Panel Heckman Model (Wooldridge 1995)

### Model Specification

**Joint Distribution:**

Assume:
```
(η_i, α_i) ~ N(0, Σ_u)    [Random effects]
(v_it, ε_it) ~ N(0, Σ_v)  [Transitory shocks]
```

with:
```
     ┌         ┐
Σ_v =│  1    ρ │
     │  ρ  σ_ε²│
     └         ┘
```

where:
- Selection error variance normalized to 1
- ρ is the key parameter (selection bias)
- σ_ε² is outcome error variance

**Conditional Distribution:**

Given selection (`d_it = 1`), the outcome expectation is:

```
E[y_it | d_it = 1, X_it, W_it] = X_it'β + ρσ_ε λ_it
```

where:
```
λ_it = φ(W_it'γ) / Φ(W_it'γ)
```

is the **Inverse Mills Ratio (IMR)**.

**Key Insight:** The IMR `λ_it` captures the selection correction. If we include it as a regressor, we can obtain unbiased estimates of β.

---

## Estimation Methods

### 1. Heckman Two-Step Estimator

**Step 1: Estimate Selection Equation**

Estimate γ via **probit** (or random effects probit for panel):

```
max_γ Σ_i Σ_t [d_it log Φ(W_it'γ) + (1-d_it) log(1 - Φ(W_it'γ))]
```

Obtain: `γ̂`

**Step 2: Augmented Outcome Regression**

Compute IMR:
```
λ̂_it = φ(W_it'γ̂) / Φ(W_it'γ̂)  for selected sample (d_it = 1)
```

Estimate outcome equation with IMR:
```
y_it = X_it'β + θ λ̂_it + error_it   (only for d_it = 1)
```

via OLS, where `θ = ρσ_ε`.

**Recover ρ:**
```
ρ̂ = θ̂ / σ̂_ε
```

where `σ̂_ε` is estimated from residuals.

**Standard Errors:**

Naive OLS standard errors are **incorrect** because they ignore estimation error in `γ̂` (Step 1).

**Murphy-Topel Correction** adjusts for this:

```
V̂(β̂, θ̂) = V̂_OLS + correction_term
```

where the correction accounts for uncertainty in `γ̂`.

**Pros:**
- Simple and fast
- Transparent two-stage process
- Robust initial estimator

**Cons:**
- Standard errors require Murphy-Topel correction
- Less efficient than MLE

---

### 2. Full Information Maximum Likelihood (FIML)

**Joint Likelihood:**

For each individual i:

```
L_i(θ) = ∫∫ [∏_t L_it(d_it, y_it | X_it, W_it, α_i, η_i)] φ(α_i, η_i | Σ_u) dα_i dη_i
```

where:

```
L_it(d, y | X, W, α, η) =
  [φ((y - X'β - α)/σ_ε) / σ_ε × Φ((W'γ + η)/σ_v)]^d ×
  [1 - Φ((W'γ + η)/σ_v)]^(1-d)
```

**Parameters to estimate:**
- β: outcome coefficients
- γ: selection coefficients
- σ_ε: outcome error SD
- ρ: correlation
- Σ_u: random effects variance

**Numerical Integration:**

The integral over (α_i, η_i) is computed via **Gauss-Hermite quadrature**:

```
∫∫ f(α, η) φ(α) φ(η) dα dη ≈ Σ_i Σ_j w_i w_j f(α_i, η_j)
```

Typically use m × m grid (e.g., 10×10 = 100 evaluation points).

**Optimization:**

Maximize joint log-likelihood:
```
max_{β,γ,σ,ρ} Σ_i log L_i(θ)
```

Use Newton-Raphson or BFGS with:
- Starting values from two-step
- Parameter transformations to ensure constraints:
  - σ_ε > 0: use log(σ_ε)
  - ρ ∈ (-1, 1): use Fisher z-transform

**Pros:**
- Asymptotically efficient
- Correct standard errors automatically
- Joint estimation of all parameters

**Cons:**
- Computationally expensive
- May have convergence issues
- Requires good starting values

---

## Identification

### Exclusion Restriction

**Critical for identification:** At least one variable in W_it (selection equation) should be excluded from X_it (outcome equation).

**Why?**

Without exclusion restriction, identification relies solely on:
1. Nonlinearity of IMR
2. Distributional assumptions (normality)

This is weak and often fails in practice.

**Good Exclusion Restrictions:**

Variables that affect **selection but not outcome directly**:

| Application | Selection Var | Outcome Var | Exclusion Restriction |
|-------------|---------------|-------------|----------------------|
| Wage determination | Employment | Wage | Number of children, non-labor income |
| Training effects | Training participation | Earnings | Program availability, distance to center |
| Insurance choice | Purchase insurance | Medical costs | State regulations, premium subsidies |

**Testing:**

While there's no formal test for exclusion restriction validity, you can:
1. Test joint significance in outcome equation (should be insignificant)
2. Check economic plausibility
3. Perform sensitivity analysis

---

## Inverse Mills Ratio (IMR)

### Definition

The IMR is the ratio of PDF to CDF of standard normal:

```
λ(z) = φ(z) / Φ(z)
```

**Properties:**

1. **Always positive:** λ(z) > 0 for all z

2. **Decreasing in z:**
   ```
   dλ/dz = -λ(λ + z) < 0
   ```

3. **Asymptotic behavior:**
   ```
   λ(z) → 0      as z → +∞  (high selection probability)
   λ(z) → +∞     as z → -∞  (low selection probability)
   ```

4. **At z = 0:** λ(0) ≈ 0.7979

### Interpretation

In the outcome equation:
```
E[y | selected] = X'β + ρσ_ε λ(W'γ)
```

The term `ρσ_ε λ(W'γ)` is the **selection correction**.

**Magnitude:**

- High λ → Strong selection effect
- Low λ → Weak selection effect

**Sign:**

- ρ > 0 and λ > 0 → Positive correction (selected sample has higher E[ε])
- ρ < 0 and λ > 0 → Negative correction (selected sample has lower E[ε])

### Diagnostics

**High IMR values (λ > 2)** indicate:
- Very low selection probabilities
- Strong selection effects
- Potentially problematic observations

**Example:**
```python
diag = result.imr_diagnostics()
print(f"Mean IMR: {diag['imr_mean']:.3f}")
print(f"High IMR count: {diag['high_imr_count']}")
```

---

## Testing for Selection Bias

### H₀: ρ = 0 (No Selection Bias)

**Two equivalent approaches:**

#### 1. t-test on IMR coefficient

In Step 2:
```
y_it = X_it'β + θ λ̂_it + error
```

Test: `H₀: θ = 0`

Use standard t-test with Murphy-Topel corrected SE.

#### 2. Wald test on ρ

From MLE, directly test:
```
H₀: ρ = 0
```

**Interpretation:**

- **Reject H₀:** Selection bias present → Use Heckman correction
- **Fail to reject:** No significant selection bias → OLS may be adequate

**Example:**
```python
test = result.selection_effect()
print(test['interpretation'])
# Output: "Selection bias detected (ρ ≠ 0, p=0.0012).
#          OLS would be biased. Heckman correction is necessary."
```

---

## Comparison with OLS

### OLS on Selected Sample (Biased)

If we ignore selection and run OLS only on observations with `d_it = 1`:

```
y_it = X_it'β + error_it   (only d_it = 1)
```

**Bias:**

```
E[β̂_OLS | selected sample] = β + bias
```

where:
```
bias = ρσ_ε (X'X)^(-1) X'λ
```

**Direction of bias depends on:**
1. Sign of ρ
2. Correlation between X and λ

### Heckman Corrected (Unbiased)

```
y_it = X_it'β + ρσ_ε λ̂_it + error_it
```

Including IMR removes the bias:
```
E[β̂_Heckman] = β
```

### Empirical Comparison

```python
comparison = result.compare_ols_heckman()
print(f"Max difference: {comparison['max_abs_difference']:.3f}")
print(comparison['interpretation'])
```

**Example Output:**
```
Coefficient Estimates:
  Variable        OLS    Heckman  Difference  % Diff
  --------------------------------------------------------
  Intercept     1.6234   1.5123    0.1111     6.8%
  Experience    0.0245   0.0298   -0.0053   -21.6%
  Education     0.0923   0.0789    0.0134    14.5%

Substantial selection bias detected (max diff: 0.134).
OLS estimates are biased. Heckman correction is necessary.
```

---

## Practical Considerations

### When to Use Heckman Model?

**Use when:**
1. ✓ Outcome observed only for subset of sample
2. ✓ Selection likely correlated with outcome
3. ✓ Valid exclusion restriction available
4. ✓ Selection probabilities vary sufficiently
5. ✓ Normality assumption reasonable

**Don't use when:**
1. ✗ Selection purely random (ρ = 0)
2. ✗ No exclusion restriction
3. ✗ Perfect or near-perfect selection (everyone/no one selected)
4. ✗ Selection mechanism complex (use alternative methods)

### Common Issues

#### 1. Collinearity

If X and W are very similar (weak exclusion restriction):
- IMR highly collinear with X
- Estimates unstable
- Large standard errors

**Solution:** Find better exclusion restriction

#### 2. Extreme Selection Probabilities

If some Φ(W'γ) very close to 0 or 1:
- IMR explodes (λ → ∞)
- Numerical instability

**Solution:**
- Check for outliers
- Trim extreme observations
- Use robust estimation

#### 3. Non-Normality

Heckman model assumes:
```
(v_it, ε_it) ~ Bivariate Normal
```

If violated:
- Estimates inconsistent
- Test results unreliable

**Solutions:**
- Semi-parametric estimators (e.g., Kyriazidou 1997)
- Transformation to normality
- Robustness checks

#### 4. MLE Convergence

MLE may fail to converge if:
- Starting values poor
- Likelihood flat
- Parameters at boundary (|ρ| near 1)

**Solutions:**
- Use two-step as starting values
- Try multiple starting points
- Check parameter bounds
- Simplify model if needed

### Robustness Checks

1. **Sensitivity to exclusion restriction:**
   - Try different exclusion variables
   - Check coefficient stability

2. **Distributional assumptions:**
   - Test normality of residuals
   - Compare with semi-parametric estimators

3. **Specification:**
   - Test functional form (quadratic terms, interactions)
   - Check for heteroskedasticity

4. **Sample selection:**
   - Bootstrap standard errors
   - Jackknife leave-one-out

---

## Extensions

### 1. Kyriazidou (1997) Semi-Parametric Estimator

Avoids distributional assumptions using pairwise differences:

```
β̂ = argmin Σ_i Σ_t<s K_h(W_it - W_is) d_it d_is (Δy_its - ΔX_its'β)²
```

where:
- K_h: kernel function with bandwidth h
- Uses observations with similar W_it, W_is

**Pros:** No normality assumption

**Cons:**
- Computationally intensive
- Bandwidth selection critical
- Less efficient than MLE

### 2. Dynamic Selection Models

Extend to:
```
d_it = f(W_it, d_{i,t-1}, ...)
y_it = g(X_it, y_{i,t-1}, ...)
```

Accounts for state dependence in selection.

### 3. Multiple Selection Rules

Multiple types of selection:
```
d1_it = 1[employed]
d2_it = 1[full-time | employed]
```

Requires nested or sequential selection models.

---

## References

### Key Papers

1. **Heckman, J.J. (1979).** "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153-161.
   - Original two-step correction for cross-section

2. **Wooldridge, J.M. (1995).** "Selection Corrections for Panel Data Models Under Conditional Mean Independence Assumptions." *Journal of Econometrics*, 68(1), 115-132.
   - Panel extension with random effects

3. **Kyriazidou, E. (1997).** "Estimation of a Panel Data Sample Selection Model." *Econometrica*, 65(6), 1335-1364.
   - Semi-parametric approach

4. **Murphy, K.M., & Topel, R.H. (1985).** "Estimation and Inference in Two-Step Econometric Models." *Journal of Business & Economic Statistics*, 3(4), 370-379.
   - Variance correction for two-step estimators

### Textbooks

- **Wooldridge, J.M. (2010).** *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
  - Chapter 19: Sample selection models

- **Cameron, A.C., & Trivedi, P.K. (2005).** *Microeconometrics: Methods and Applications*. Cambridge University Press.
  - Chapter 16: Sample selection

---

## See Also

- [Selection Models API Reference](../api/selection.md)
- [Panel Heckman Tutorial](../../examples/selection/panel_heckman_tutorial.py)
- [Discrete Choice Models](../api/discrete_models.md)
