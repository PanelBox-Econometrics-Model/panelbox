---
title: Panel Selection Models Theory
description: "Theoretical foundations of panel selection models including Heckman correction, inverse Mills ratio, and panel extensions for sample selection bias."
---

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

Consider a panel dataset with $N$ individuals observed over $T$ periods.

**Outcome Equation (of interest):**

$$y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}$$

where:

- $y_{it}$: outcome of interest (e.g., wage)
- $X_{it}$: outcome regressors
- $\alpha_i$: individual fixed/random effect
- $\varepsilon_{it}$: idiosyncratic error

**Selection Equation:**

$$d_{it} = \mathbf{1}[W_{it}'\gamma + \eta_i + v_{it} > 0]$$

where:

- $d_{it} = 1$ if outcome observed, 0 otherwise
- $W_{it}$: selection regressors
- $\eta_i$: individual random effect in selection
- $v_{it}$: selection shock

**Key Assumption:** Outcome is observed only when $d_{it} = 1$:

$$y_{it}^{obs} = \begin{cases} y_{it} & \text{if } d_{it} = 1 \\ \varnothing & \text{if } d_{it} = 0 \end{cases}$$

### When Does Selection Bias Occur?

Selection bias arises when:

$$\text{Corr}(v_{it}, \varepsilon_{it}) = \rho \neq 0$$

**Intuition:**

- **$\rho > 0$ (Positive selection):** High-outcome individuals more likely to be selected
  - Example: High-wage workers more likely to be employed

- **$\rho < 0$ (Negative selection):** Low-outcome individuals more likely to be selected
  - Example: Low-skilled workers more likely to participate in training programs

- **$\rho = 0$ (No selection bias):** Selection independent of outcome
  - OLS on selected sample is unbiased

---

## Panel Heckman Model (Wooldridge 1995)

### Model Specification

**Joint Distribution:**

Assume:

$$(\eta_i, \alpha_i) \sim N(0, \Sigma_u) \quad \text{[Random effects]}$$

$$(\varepsilon_{it}, v_{it}) \sim N(0, \Sigma_v) \quad \text{[Transitory shocks]}$$

with:

$$\Sigma_v = \begin{pmatrix} \sigma_\varepsilon^2 & \rho \\ \rho & 1 \end{pmatrix}$$

where:

- Selection error variance normalized to 1
- $\rho$ is the key parameter (selection bias)
- $\sigma_\varepsilon^2$ is outcome error variance

**Conditional Distribution:**

Given selection ($d_{it} = 1$), the outcome expectation is:

$$E[y_{it} \mid d_{it} = 1, X_{it}, W_{it}] = X_{it}'\beta + \rho\sigma_\varepsilon \lambda_{it}$$

where:

$$\lambda_{it} = \frac{\phi(W_{it}'\gamma)}{\Phi(W_{it}'\gamma)}$$

is the **Inverse Mills Ratio (IMR)**.

**Key Insight:** The IMR $\lambda_{it}$ captures the selection correction. If we include it as a regressor, we can obtain unbiased estimates of $\beta$.

---

## Estimation Methods

### 1. Heckman Two-Step Estimator

**Step 1: Estimate Selection Equation**

Estimate $\gamma$ via **probit** (or random effects probit for panel):

$$\max_\gamma \sum_i \sum_t \left[ d_{it} \log \Phi(W_{it}'\gamma) + (1 - d_{it}) \log(1 - \Phi(W_{it}'\gamma)) \right]$$

Obtain: $\hat{\gamma}$

**Step 2: Augmented Outcome Regression**

Compute IMR:

$$\hat{\lambda}_{it} = \frac{\phi(W_{it}'\hat{\gamma})}{\Phi(W_{it}'\hat{\gamma})} \quad \text{for selected sample } (d_{it} = 1)$$

Estimate outcome equation with IMR:

$$y_{it} = X_{it}'\beta + \theta \hat{\lambda}_{it} + \text{error}_{it} \quad \text{(only for } d_{it} = 1\text{)}$$

via OLS, where $\theta = \rho\sigma_\varepsilon$.

**Recover $\rho$:**

$$\hat{\rho} = \frac{\hat{\theta}}{\hat{\sigma}_\varepsilon}$$

where $\hat{\sigma}_\varepsilon$ is estimated from residuals.

**Standard Errors:**

Naive OLS standard errors are **incorrect** because they ignore estimation error in $\hat{\gamma}$ (Step 1).

**Murphy-Topel Correction** adjusts for this:

$$\hat{V}(\hat{\beta}, \hat{\theta}) = \hat{V}_{OLS} + \text{correction term}$$

where the correction accounts for uncertainty in $\hat{\gamma}$.

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

For each individual $i$:

$$L_i(\theta) = \int\!\!\int \left[\prod_t L_{it}(d_{it}, y_{it} \mid X_{it}, W_{it}, \alpha_i, \eta_i)\right] \phi(\alpha_i, \eta_i \mid \Sigma_u)\, d\alpha_i\, d\eta_i$$

where:

$$L_{it}(d, y \mid X, W, \alpha, \eta) = \left[\frac{\phi\!\left(\frac{y - X'\beta - \alpha}{\sigma_\varepsilon}\right)}{\sigma_\varepsilon} \cdot \Phi\!\left(\frac{W'\gamma + \eta}{\sigma_v}\right)\right]^d \cdot \left[1 - \Phi\!\left(\frac{W'\gamma + \eta}{\sigma_v}\right)\right]^{1-d}$$

**Parameters to estimate:**

- $\beta$: outcome coefficients
- $\gamma$: selection coefficients
- $\sigma_\varepsilon$: outcome error SD
- $\rho$: correlation
- $\Sigma_u$: random effects variance

**Numerical Integration:**

The integral over $(\alpha_i, \eta_i)$ is computed via **Gauss-Hermite quadrature**:

$$\int\!\!\int f(\alpha, \eta)\, \phi(\alpha)\, \phi(\eta)\, d\alpha\, d\eta \approx \sum_i \sum_j w_i w_j f(\alpha_i, \eta_j)$$

Typically use $m \times m$ grid (e.g., $10 \times 10 = 100$ evaluation points).

**Optimization:**

Maximize joint log-likelihood:

$$\max_{\beta,\gamma,\sigma,\rho} \sum_i \log L_i(\theta)$$

Use Newton-Raphson or BFGS with:

- Starting values from two-step
- Parameter transformations to ensure constraints:
  - $\sigma_\varepsilon > 0$: use $\log(\sigma_\varepsilon)$
  - $\rho \in (-1, 1)$: use Fisher z-transform

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

**Critical for identification:** At least one variable in $W_{it}$ (selection equation) should be excluded from $X_{it}$ (outcome equation).

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

$$\lambda(z) = \frac{\phi(z)}{\Phi(z)}$$

**Properties:**

1. **Always positive:** $\lambda(z) > 0$ for all $z$

2. **Decreasing in $z$:**

    $$\frac{d\lambda}{dz} = -\lambda(\lambda + z) < 0$$

3. **Asymptotic behavior:**

    $$\lambda(z) \to 0 \quad \text{as } z \to +\infty \quad \text{(high selection probability)}$$

    $$\lambda(z) \to +\infty \quad \text{as } z \to -\infty \quad \text{(low selection probability)}$$

4. **At $z = 0$:** $\lambda(0) \approx 0.7979$

### Interpretation

In the outcome equation:

$$E[y \mid \text{selected}] = X'\beta + \rho\sigma_\varepsilon \lambda(W'\gamma)$$

The term $\rho\sigma_\varepsilon \lambda(W'\gamma)$ is the **selection correction**.

**Magnitude:**

- High $\lambda$ → Strong selection effect
- Low $\lambda$ → Weak selection effect

**Sign:**

- $\rho > 0$ and $\lambda > 0$ → Positive correction (selected sample has higher $E[\varepsilon]$)
- $\rho < 0$ and $\lambda > 0$ → Negative correction (selected sample has lower $E[\varepsilon]$)

### Diagnostics

**High IMR values ($\lambda > 2$)** indicate:

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

### $H_0$: $\rho = 0$ (No Selection Bias)

**Two equivalent approaches:**

#### 1. t-test on IMR coefficient

In Step 2:

$$y_{it} = X_{it}'\beta + \theta \hat{\lambda}_{it} + \text{error}$$

Test: $H_0\!: \theta = 0$

Use standard t-test with Murphy-Topel corrected SE.

#### 2. Wald test on $\rho$

From MLE, directly test:

$$H_0\!: \rho = 0$$

**Interpretation:**

- **Reject $H_0$:** Selection bias present → Use Heckman correction
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

If we ignore selection and run OLS only on observations with $d_{it} = 1$:

$$y_{it} = X_{it}'\beta + \text{error}_{it} \quad \text{(only } d_{it} = 1\text{)}$$

**Bias:**

$$E[\hat{\beta}_{OLS} \mid \text{selected sample}] = \beta + \text{bias}$$

where:

$$\text{bias} = \rho\sigma_\varepsilon (X'X)^{-1} X'\lambda$$

**Direction of bias depends on:**

1. Sign of $\rho$
2. Correlation between $X$ and $\lambda$

### Heckman Corrected (Unbiased)

$$y_{it} = X_{it}'\beta + \rho\sigma_\varepsilon \hat{\lambda}_{it} + \text{error}_{it}$$

Including IMR removes the bias:

$$E[\hat{\beta}_{Heckman}] = \beta$$

### Empirical Comparison

```python
comparison = result.compare_ols_heckman()
print(f"Max difference: {comparison['max_abs_difference']:.3f}")
print(comparison['interpretation'])
```

**Example Output:**

```text
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

1. Outcome observed only for subset of sample
2. Selection likely correlated with outcome
3. Valid exclusion restriction available
4. Selection probabilities vary sufficiently
5. Normality assumption reasonable

**Don't use when:**

1. Selection purely random ($\rho = 0$)
2. No exclusion restriction
3. Perfect or near-perfect selection (everyone/no one selected)
4. Selection mechanism complex (use alternative methods)

### Common Issues

#### 1. Collinearity

If $X$ and $W$ are very similar (weak exclusion restriction):

- IMR highly collinear with $X$
- Estimates unstable
- Large standard errors

**Solution:** Find better exclusion restriction

#### 2. Extreme Selection Probabilities

If some $\Phi(W'\gamma)$ very close to 0 or 1:

- IMR explodes ($\lambda \to \infty$)
- Numerical instability

**Solution:**

- Check for outliers
- Trim extreme observations
- Use robust estimation

#### 3. Non-Normality

Heckman model assumes:

$$(v_{it}, \varepsilon_{it}) \sim \text{Bivariate Normal}$$

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
- Parameters at boundary ($|\rho|$ near 1)

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

$$\hat{\beta} = \arg\min \sum_i \sum_{t < s} K_h(W_{it} - W_{is})\, d_{it}\, d_{is}\, (\Delta y_{its} - \Delta X_{its}'\beta)^2$$

where:

- $K_h$: kernel function with bandwidth $h$
- Uses observations with similar $W_{it}$, $W_{is}$

**Pros:** No normality assumption

**Cons:**

- Computationally intensive
- Bandwidth selection critical
- Less efficient than MLE

### 2. Dynamic Selection Models

Extend to:

$$d_{it} = f(W_{it}, d_{i,t-1}, \ldots)$$

$$y_{it} = g(X_{it}, y_{i,t-1}, \ldots)$$

Accounts for state dependence in selection.

### 3. Multiple Selection Rules

Multiple types of selection:

$$d_{1,it} = \mathbf{1}[\text{employed}]$$

$$d_{2,it} = \mathbf{1}[\text{full-time} \mid \text{employed}]$$

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

- [Selection Models API Reference](../api/censored.md)
- [Panel Heckman Tutorial](../tutorials/censored.md)
- [Discrete Choice Models](../api/discrete.md)
