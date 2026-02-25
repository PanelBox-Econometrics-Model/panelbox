---
title: "GMM Theory"
description: "Mathematical foundations of Generalized Method of Moments for dynamic panel data models"
---

# GMM Theory --- Dynamic Panel Models

!!! abstract "Key Takeaway"
    In dynamic panels with lagged dependent variables, fixed effects estimation is inconsistent due to Nickell bias. GMM exploits moment conditions from lagged values as instruments to produce consistent estimates. Difference GMM uses lagged levels for differenced equations; System GMM adds lagged differences for level equations, improving efficiency.

## Motivation: The Nickell Bias

Consider a dynamic panel model:

$$
y_{it} = \rho y_{i,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

Applying the within transformation does not eliminate the correlation between the demeaned lagged dependent variable $\ddot{y}_{i,t-1}$ and the demeaned error $\ddot{\varepsilon}_{it}$, because both contain $\bar{\varepsilon}_i$. This produces the **Nickell (1981) bias**:

$$
\text{plim}(\hat{\rho}_{FE} - \rho) \approx -\frac{1 + \rho}{T - 1}
$$

For typical micro panels ($T = 5\text{--}10$), this bias is substantial. With $\rho = 0.7$ and $T = 10$: bias $\approx -0.189$, yielding $\hat{\rho} \approx 0.51$ instead of 0.70.

!!! warning "Nickell Bias Severity"
    The bias is always **downward** for positive $\rho$ and becomes worse as $T$ decreases. It does not vanish as $N \to \infty$ with fixed $T$.

## The GMM Framework

### Moment Conditions

The Generalized Method of Moments approach relies on orthogonality (moment) conditions of the form:

$$
E[Z_i' \Delta \varepsilon_i] = 0
$$

where $Z_i$ is a matrix of instruments and $\Delta \varepsilon_i$ is the vector of first-differenced errors. Under the assumption that $\varepsilon_{it}$ is serially uncorrelated and predetermined regressors are uncorrelated with future errors, lagged values serve as valid instruments.

### GMM Estimator

The GMM estimator minimizes the quadratic form:

$$
\hat{\beta}_{GMM} = \arg\min_\beta \left[\left(\frac{1}{N}\sum_{i=1}^N Z_i' \Delta \varepsilon_i(\beta)\right)' W \left(\frac{1}{N}\sum_{i=1}^N Z_i' \Delta \varepsilon_i(\beta)\right)\right]
$$

For the linear model, the closed-form solution is:

$$
\hat{\beta} = (X'Z W Z'X)^{-1} X'Z W Z'y
$$

where $W$ is a positive definite weighting matrix, $X$ is the regressor matrix (possibly including $y_{t-1}$), $Z$ is the instrument matrix, and $y$ is the dependent variable vector.

### Optimal Weighting

The **efficient GMM** estimator uses the optimal weighting matrix:

$$
W_{opt} = \left(\frac{1}{N}\sum_{i=1}^N Z_i' \hat{\Omega}_i Z_i\right)^{-1}
$$

where $\hat{\Omega}_i$ is a consistent estimate of $E[\Delta \varepsilon_i \Delta \varepsilon_i']$. This yields the estimator with the smallest asymptotic variance among all GMM estimators using instruments $Z$.

## Difference GMM (Arellano-Bond 1991)

### First-Difference Transformation

Taking first differences eliminates the individual effect:

$$
\Delta y_{it} = \rho \Delta y_{i,t-1} + \Delta X_{it}'\beta + \Delta \varepsilon_{it}
$$

But $\Delta y_{i,t-1} = y_{i,t-1} - y_{i,t-2}$ is correlated with $\Delta \varepsilon_{it} = \varepsilon_{it} - \varepsilon_{i,t-1}$ through the common $\varepsilon_{i,t-1}$ term, so OLS on the differenced equation is also inconsistent.

### Instrument Strategy

Under the assumption that $E[\varepsilon_{it} \varepsilon_{is}] = 0$ for $t \neq s$, lagged **levels** of $y$ dated $t-2$ and earlier are valid instruments for the differenced equation:

$$
E[y_{i,t-s} \cdot \Delta \varepsilon_{it}] = 0, \quad s \geq 2
$$

### Instrument Matrix

For entity $i$, the instrument matrix has a block-diagonal structure:

$$
Z_i = \begin{pmatrix}
y_{i1} & 0 & 0 & \cdots \\
0 & y_{i1}, y_{i2} & 0 & \cdots \\
0 & 0 & y_{i1}, y_{i2}, y_{i3} & \cdots \\
\vdots & & & \ddots
\end{pmatrix}
$$

The number of instruments grows quadratically with $T$: $L = T(T-1)/2$ for the dependent variable alone.

### Weak Instrument Problem

When $\rho$ is close to 1 or the variance of $\alpha_i$ is large relative to $\varepsilon_{it}$, lagged levels are **weak instruments** for first differences. This leads to:

- Large finite-sample bias
- Imprecise estimates (wide confidence intervals)
- Poor asymptotic approximation

## System GMM (Blundell-Bond 1998)

### Additional Moment Conditions

System GMM augments the differenced equation with the **level equation**:

$$
y_{it} = \rho y_{i,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

using lagged **differences** as instruments:

$$
E[\Delta y_{i,t-1} \cdot (\alpha_i + \varepsilon_{it})] = 0
$$

This requires an additional **stationarity assumption**: the initial conditions $y_{i1}$ satisfy the mean-stationarity condition $E[\alpha_i \Delta y_{i2}] = 0$.

### Stacked System

The system GMM estimator stacks the differenced and level equations:

$$
\begin{pmatrix} \Delta y \\ y \end{pmatrix} = \begin{pmatrix} \Delta X \\ X \end{pmatrix} \beta + \begin{pmatrix} \Delta \varepsilon \\ \alpha + \varepsilon \end{pmatrix}
$$

with the combined instrument matrix containing lagged levels (for differenced equations) and lagged differences (for level equations).

### Advantages over Difference GMM

- **Stronger instruments** when $\rho$ is close to 1 (persistent series)
- **More efficient** due to additional moment conditions
- **Better finite-sample performance**, especially with small $T$
- Can estimate effects of **time-invariant** variables

## One-Step vs. Two-Step Estimation

### One-Step GMM

Uses a **known** initial weighting matrix, typically:

$$
W_1 = \left(\frac{1}{N}\sum_{i=1}^N Z_i' H Z_i\right)^{-1}
$$

where $H$ is a matrix reflecting the first-difference error structure (e.g., $H_{ts} = 2$ if $t=s$, $-1$ if $|t-s|=1$, $0$ otherwise).

- Produces consistent but generally **inefficient** estimates
- Standard errors from heteroskedasticity-robust variance matrix are reliable

### Two-Step GMM

Re-estimates using the **optimal** weighting matrix from one-step residuals:

$$
W_2 = \left(\frac{1}{N}\sum_{i=1}^N Z_i' \hat{u}_i^{(1)} \hat{u}_i^{(1)'} Z_i\right)^{-1}
$$

where $\hat{u}_i^{(1)}$ are residuals from the one-step estimator.

- **Asymptotically efficient** (achieves the GMM efficiency bound)
- Standard errors tend to be **downward biased** in finite samples

### Windmeijer (2005) Correction

The finite-sample bias in two-step standard errors arises because $W_2$ is estimated. Windmeijer's correction adjusts the variance:

$$
\widehat{\text{Var}}_{WC}(\hat{\beta}_2) = \widehat{\text{Var}}_2 + \text{correction}
$$

The correction accounts for the variability in $\hat{W}_2$ due to its dependence on first-step estimates. This yields standard errors that are much closer to their true sampling distribution.

## CUE-GMM (Continuously Updated Estimator)

Hansen, Heaton, and Yaron (1996) proposed **simultaneously** updating the weighting matrix and parameters:

$$
\hat{\beta}^{CUE} = \arg\min_\beta \; g_N(\beta)' W(\beta)^{-1} g_N(\beta)
$$

where $W(\beta)$ is recomputed at each candidate $\beta$.

### Properties

- **Consistent and asymptotically efficient** (same as two-step)
- **Invariant** to moment condition normalization (unlike two-step)
- **Lower finite-sample bias** than two-step in moderate samples ($N = 100$--$500$)
- **Computationally expensive** --- requires nonlinear optimization
- May have **convergence issues** with weak instruments

## Bias-Corrected GMM

Hahn and Kuersteiner (2002) derived an **analytical bias correction** for the GMM estimator:

$$
\hat{\rho}^{BC} = \hat{\rho}^{GMM} - \frac{\hat{B}(\hat{\rho})}{N}
$$

where $\hat{B}(\hat{\rho})$ is the estimated leading term of the bias, reducing the total bias from $O(N^{-1})$ to $O(N^{-2})$.

This approach is recommended for dynamic panels with moderate $N$ and $T$ (both 50--200), where Nickell bias is non-negligible but GMM instruments are not too weak.

## Diagnostic Tests

### Hansen J-Test (Overidentification)

Tests whether the moment conditions are jointly valid:

$$
J = N \cdot Q(\hat{\beta}) \xrightarrow{d} \chi^2(L - K)
$$

where $L$ is the number of instruments and $K$ is the number of parameters.

| p-value | Interpretation |
|---------|---------------|
| $> 0.05$ | Instruments appear valid (do not reject) |
| $< 0.05$ | Evidence of invalid instruments or misspecification |

!!! warning "Too Many Instruments"
    When the instrument count $L$ is large relative to $N$, the J-test has weak power and tends not to reject even when instruments are invalid. This is the **instrument proliferation** problem.

### Arellano-Bond Serial Correlation Tests

Test for serial correlation in the differenced residuals:

- **AR(1):** Expected to be significant (by construction of first differences)
- **AR(2):** Should be **insignificant** if the original errors $\varepsilon_{it}$ are serially uncorrelated

If AR(2) rejects, the moment conditions $E[y_{i,t-2} \cdot \Delta\varepsilon_{it}] = 0$ are invalid, and deeper lags must be used as instruments.

### Instrument Proliferation

The number of GMM instruments grows quadratically with $T$. Too many instruments can:

- Overfit endogenous variables
- Weaken the Hansen J-test
- Bias results toward OLS/2SLS estimates

**Solutions:**

- **Collapse** the instrument matrix (use one instrument per lag distance rather than per period)
- **Limit lag depth** (e.g., use only lags 2--4 instead of all available)
- Rule of thumb: keep $L < N$

## Practical Implications

1. **Start with one-step GMM** with robust standard errors for initial estimates
2. **Report two-step** with Windmeijer correction for efficient estimates
3. **Always check** AR(2) test --- serial correlation invalidates moment conditions
4. **Monitor instrument count** --- collapse instruments if $L$ approaches $N$
5. **Compare Difference and System GMM** --- if $\hat{\rho}_{DIF}$ is much lower than $\hat{\rho}_{SYS}$, weak instruments may be an issue for Difference GMM
6. **Bounds check:** $\hat{\rho}_{FE} < \hat{\rho}_{true} < \hat{\rho}_{OLS}$ --- GMM estimate should fall between FE and pooled OLS estimates

## Key References

- Arellano, M. & Bond, S. (1991). "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277--297. --- Original Difference GMM.
- Blundell, R. & Bond, S. (1998). "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115--143. --- System GMM with additional moment conditions.
- Roodman, D. (2009). "How to Do xtabond2: An Introduction to Difference and System GMM in Stata." *Stata Journal*, 9(1), 86--136. --- Practical guide to GMM estimation.
- Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics*, 126(1), 25--51. --- Finite-sample correction for two-step standard errors.
- Hansen, L.P., Heaton, J. & Yaron, A. (1996). "Finite-Sample Properties of Some Alternative GMM Estimators." *Journal of Business & Economic Statistics*, 14(3), 262--280. --- CUE-GMM theory.
- Hahn, J. & Kuersteiner, G. (2002). "Asymptotically Unbiased Inference for a Dynamic Panel Model with Fixed Effects When Both N and T Are Large." *Econometrica*, 70(4), 1639--1657. --- Analytical bias correction.
- Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects." *Econometrica*, 49(6), 1417--1426. --- Derivation of the Nickell bias.

## See Also

- [Panel Fundamentals](panel-fundamentals.md) --- the static panel models that GMM extends
- [Cointegration Theory](../diagnostics/cointegration/index.md) --- non-stationary panels where GMM may be needed
- [References](references.md) --- complete bibliography
