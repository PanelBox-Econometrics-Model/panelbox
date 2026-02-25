---
title: "Quantile Regression Theory"
description: "Mathematical foundations of quantile regression and distributional analysis in panel data"
---

# Quantile Regression Theory --- Distributional Analysis

!!! abstract "Key Takeaway"
    While OLS estimates the conditional mean $E[y \mid X]$, quantile regression estimates the conditional quantile function $Q_\tau(y \mid X)$ for any $\tau \in (0,1)$. This reveals how covariates affect the entire distribution of outcomes, not just the average --- capturing heterogeneous effects across the distribution.

## Motivation: Beyond the Mean

Standard regression asks: "What is the average effect of $X$ on $y$?" But many important questions involve the distribution:

- Does education reduce wage **inequality** (compressing quantiles)?
- Do trade policies help the **poorest** firms or only the largest?
- Is the effect of a drug stronger for the **sickest** patients?

Quantile regression answers these questions by estimating separate coefficients at each quantile $\tau$.

## The Quantile Regression Estimator

### Definition (Koenker-Bassett 1978)

The $\tau$-th conditional quantile of $y$ given $X$ is:

$$
Q_\tau(y \mid X) = X'\beta(\tau)
$$

The estimator $\hat{\beta}(\tau)$ minimizes the **check function** objective:

$$
\hat{\beta}(\tau) = \arg\min_\beta \sum_{i=1}^N \rho_\tau(y_i - X_i'\beta)
$$

where the check (or pinball loss) function is:

$$
\rho_\tau(u) = u \cdot [\tau - \mathbf{1}(u < 0)] = \begin{cases} \tau \cdot u & \text{if } u \geq 0 \\ (\tau - 1) \cdot u & \text{if } u < 0 \end{cases}
$$

### Key Properties

- At $\tau = 0.5$, quantile regression is **median regression** (LAD --- Least Absolute Deviations)
- **No distributional assumptions** on the error term --- robust to outliers and non-normality
- Coefficients $\beta(\tau)$ vary across $\tau$, revealing heterogeneous effects
- The objective function is convex but **non-differentiable** at zero --- solved via linear programming

### Interpretation

$\beta_k(\tau)$ represents the marginal effect of $x_k$ on the $\tau$-th quantile of $y$:

$$
\frac{\partial Q_\tau(y \mid X)}{\partial x_k} = \beta_k(\tau)
$$

If $\beta_k(0.1) = 0.02$ and $\beta_k(0.9) = 0.08$, the effect of $x_k$ is four times larger at the top of the distribution than at the bottom.

## Panel Fixed Effects Quantile Regression

### The Challenge

Adding fixed effects $\alpha_i$ to quantile regression creates two problems:

1. **Incidental parameters:** With $N$ fixed effects and finite $T$, estimation of $\beta(\tau)$ is inconsistent
2. **Non-additivity:** Unlike OLS, demeaning does not eliminate $\alpha_i$ because quantiles of demeaned variables are not the same as demeaned quantiles

### Koenker (2004) Penalized Approach

Adds an $L_1$ penalty on the fixed effects:

$$
\min_{\alpha, \beta} \sum_{\tau} \sum_{i,t} \rho_\tau(y_{it} - \alpha_i - X_{it}'\beta(\tau)) + \lambda \sum_i |\alpha_i|
$$

The penalty $\lambda$ shrinks $\alpha_i$ toward zero, mitigating the incidental parameters problem. This estimates **multiple quantiles jointly** while sharing the fixed effects across quantiles.

### Canay (2011) Two-Step Estimator

A simpler approach that works well in practice:

**Step 1:** Estimate fixed effects from the conditional mean model (FE-OLS):

$$
\hat{\alpha}_i = \bar{y}_i - \bar{X}_i'\hat{\beta}_{FE}
$$

**Step 2:** Subtract fixed effects and run standard QR:

$$
\hat{\beta}(\tau) = \arg\min_\beta \sum_{i,t} \rho_\tau(\tilde{y}_{it} - X_{it}'\beta)
$$

where $\tilde{y}_{it} = y_{it} - \hat{\alpha}_i$.

This is consistent as $T \to \infty$ (or $T$ moderately large) but may have bias for small $T$ because $\hat{\alpha}_i$ converges slowly.

### Powell (2022) Non-Additive FE

Quantile regression with **non-additive** fixed effects, allowing the fixed effects themselves to vary across quantiles. This is the most flexible but also the most computationally intensive approach.

## Location-Scale Model (Machado-Santos Silva 2019)

### Model Specification

The location-scale model represents conditional quantiles through two components:

$$
Q_{y_{it}}(\tau \mid X_{it}, \alpha_i, \delta_i) = \underbrace{(\alpha_i + X_{it}'\beta)}_{\text{location}} + \underbrace{(\delta_i + X_{it}'\gamma)}_{\text{scale}} \cdot q(\tau)
$$

where:

- **Location:** $\alpha_i + X_{it}'\beta$ is the conditional mean component
- **Scale:** $\delta_i + X_{it}'\gamma$ determines how the distribution spreads
- $q(\tau)$: quantile function of a reference distribution

### Estimation Procedure

**Step 1 --- Location:** Estimate $\alpha_i$ and $\beta$ via FE-OLS.

**Step 2 --- Scale:** Estimate $\delta_i$ and $\gamma$ from the absolute residuals:

$$
\ln|\hat{\varepsilon}_{it}| = \delta_i + X_{it}'\gamma + \text{error}
$$

**Step 3 --- Quantile coefficients:** For any $\tau$:

$$
\hat{\beta}(\tau) = \hat{\beta} + \hat{\gamma} \cdot q(\tau)
$$

### Reference Distributions

| Distribution | $q(\tau)$ | Properties |
|-------------|-----------|------------|
| **Normal** | $\Phi^{-1}(\tau)$ | Most common default |
| **Logistic** | $\ln[\tau/(1-\tau)]$ | Heavier tails |
| **Student's $t$** | $t_\nu^{-1}(\tau)$ | Adjustable tail weight via $\nu$ |
| **Laplace** | $-\text{sign}(\tau - 0.5)\ln(1 - 2|\tau - 0.5|)$ | Sharper peak |

### Non-Crossing Guarantee

A major advantage of the location-scale approach: quantile curves **cannot cross** by construction, because $q(\tau)$ is strictly monotonic in $\tau$ and the scale function is shared across all quantiles.

### Computational Efficiency

Only **two regressions** (location and scale) are needed regardless of how many quantiles are evaluated. In contrast, standard QR requires a separate optimization for each $\tau$.

## Quantile Treatment Effects (QTE)

### Conditional QTE

The effect of treatment at quantile $\tau$:

$$
\Delta(\tau) = Q_\tau(y^1 \mid X) - Q_\tau(y^0 \mid X)
$$

where $y^1$ and $y^0$ are potential outcomes under treatment and control.

If $\Delta(0.1) > \Delta(0.9)$, the treatment has a larger effect at the bottom of the distribution --- it **compresses** inequality.

### Unconditional QTE

Population-level quantile treatment effect:

$$
\Delta^{UQ}(\tau) = Q_\tau(y^1) - Q_\tau(y^0)
$$

This measures the effect on the **marginal** distribution, not conditional on covariates.

### DID-QTE (Difference-in-Differences at Quantiles)

Extends the standard DID framework to quantiles:

$$
\Delta^{DID}(\tau) = [Q_\tau(y_{post}^{treat}) - Q_\tau(y_{pre}^{treat})] - [Q_\tau(y_{post}^{control}) - Q_\tau(y_{pre}^{control})]
$$

### Changes-in-Changes (Athey-Imbens 2006)

A nonparametric approach that:

- Identifies the entire counterfactual distribution
- Requires monotonicity of the production function
- Allows for heterogeneous treatment effects across the distribution
- Provides distributional treatment effects without functional form assumptions

## Inference

### Analytical Standard Errors

For standard QR, the asymptotic variance involves the **sparsity function** $s(\tau) = 1/f_{y|X}(Q_\tau)$:

$$
\sqrt{N}(\hat{\beta}(\tau) - \beta(\tau)) \xrightarrow{d} N\left(0, \frac{\tau(1-\tau)}{f^2(Q_\tau)} (X'X)^{-1}\right)
$$

Kernel density estimation of $f$ is required, introducing bandwidth sensitivity.

### Bootstrap Methods

| Method | Description | Use Case |
|--------|------------|----------|
| **Pairs bootstrap** | Resample $(y_i, X_i)$ pairs | Cross-section data |
| **Wild bootstrap** | Perturb residuals with random signs | Heteroskedastic errors |
| **Block bootstrap** | Resample contiguous blocks | Time-dependent data |
| **Cluster bootstrap** | Resample entire entities | Panel data (recommended) |

!!! info "Cluster Bootstrap for Panels"
    For panel data, cluster bootstrap (resampling entire entities $i$ with all their time observations) is recommended to preserve the within-entity dependence structure.

## Non-Crossing Quantiles

### The Problem

Standard QR estimated at multiple quantiles may produce **crossing curves**: $Q_{\tau_1}(y \mid X) > Q_{\tau_2}(y \mid X)$ for some $X$ values even though $\tau_1 < \tau_2$. This violates the definition of quantiles.

### Solutions

1. **Location-Scale model:** Non-crossing by construction (recommended)
2. **Rearrangement** (Chernozhukov et al. 2010): Sort estimated quantiles post-estimation
3. **Monotonicity constraints:** Constrained optimization ensuring $\beta(\tau_1) \leq \beta(\tau_2)$ for ordered regressors

## Practical Implications

1. **Report multiple quantiles** --- the common set is $\tau \in \{0.10, 0.25, 0.50, 0.75, 0.90\}$
2. **Test for heterogeneity** --- if $\beta(\tau)$ varies across $\tau$, OLS misses important structure
3. **Use Location-Scale** for panel FE quantile regression --- computationally efficient and non-crossing
4. **Use Canay** when $T$ is moderately large ($T > 10$) --- simple and effective
5. **Bootstrap inference** --- cluster bootstrap is the safest choice for panel data
6. **Visualize** quantile coefficient plots to communicate distributional effects

## Key References

- Koenker, R. & Bassett, G. (1978). "Regression Quantiles." *Econometrica*, 46(1), 33--50. --- Original quantile regression estimator.
- Koenker, R. (2004). "Quantile Regression for Longitudinal Data." *Journal of Multivariate Analysis*, 91(1), 74--89. --- Penalized FE approach.
- Canay, I.A. (2011). "A Simple Approach to Quantile Regression for Panel Data." *The Econometrics Journal*, 14(3), 368--386. --- Two-step FE quantile regression.
- Machado, J.A.F. & Santos Silva, J.M.C. (2019). "Quantiles via Moments." *Journal of Econometrics*, 213(1), 145--173. --- Location-scale model.
- Athey, S. & Imbens, G.W. (2006). "Identification and Inference in Nonlinear Difference-in-Differences Models." *Econometrica*, 74(2), 431--497. --- Changes-in-changes.
- Chernozhukov, V., Fernandez-Val, I. & Melly, B. (2013). "Inference on Counterfactual Distributions." *Econometrica*, 81(6), 2205--2268. --- Distributional treatment effects.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press. --- Comprehensive textbook.

## See Also

- [Panel Fundamentals](panel-fundamentals.md) --- mean-regression panel models
- [Frontier Theory](sfa-theory.md) --- another approach to distributional analysis (efficiency)
- [References](references.md) --- complete bibliography
