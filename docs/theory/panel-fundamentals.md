---
title: "Panel Data Fundamentals"
description: "Mathematical foundations of fixed effects, random effects, and other core panel data estimators"
---

# Panel Data Fundamentals

!!! abstract "Key Takeaway"
    The central question in panel data econometrics is how to handle unobserved individual heterogeneity $\alpha_i$. Whether $\alpha_i$ is correlated with the regressors determines the choice between fixed effects (allows correlation) and random effects (assumes independence). This choice has profound implications for consistency, efficiency, and interpretation.

## Motivation

Cross-sectional data cannot separate the effect of observed variables from unobserved individual characteristics. Panel data solves this by exploiting variation **within** individuals over time, enabling us to control for time-invariant unobserved heterogeneity without measuring it directly.

## The Fundamental Panel Model

The starting point for panel data analysis is the **linear unobserved effects model**:

$$
y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T
$$

where:

- $y_{it}$ is the outcome for entity $i$ at time $t$
- $X_{it}$ is a $K \times 1$ vector of observed time-varying regressors
- $\beta$ is the $K \times 1$ parameter vector of interest
- $\alpha_i$ is the unobserved individual effect (time-invariant)
- $\varepsilon_{it}$ is the idiosyncratic error term

**The key question:** What is the relationship between $\alpha_i$ and $X_{it}$?

| Assumption | Estimator | Efficiency |
|-----------|-----------|------------|
| $\text{Corr}(\alpha_i, X_{it}) \neq 0$ | Fixed Effects | Less efficient but consistent |
| $\text{Corr}(\alpha_i, X_{it}) = 0$ | Random Effects | More efficient if assumption holds |

## Fixed Effects Estimator

### Within Transformation

The fixed effects (FE) estimator eliminates $\alpha_i$ by **demeaning** within each entity. Define entity means $\bar{y}_i = T^{-1}\sum_t y_{it}$ and $\bar{X}_i = T^{-1}\sum_t X_{it}$. The within-transformed model is:

$$
\ddot{y}_{it} = \ddot{X}_{it}'\beta + \ddot{\varepsilon}_{it}
$$

where $\ddot{y}_{it} = y_{it} - \bar{y}_i$ and similarly for other variables. OLS on this equation yields the **within estimator**:

$$
\hat{\beta}_{FE} = \left(\sum_{i=1}^N \sum_{t=1}^T \ddot{X}_{it}\ddot{X}_{it}'\right)^{-1} \sum_{i=1}^N \sum_{t=1}^T \ddot{X}_{it}\ddot{y}_{it}
$$

### Consistency

The FE estimator is consistent under **strict exogeneity**:

$$
E[\varepsilon_{it} \mid X_{i1}, \ldots, X_{iT}, \alpha_i] = 0 \quad \text{for all } t
$$

This allows **arbitrary correlation** between $\alpha_i$ and $X_{it}$, which is the key advantage of FE over RE.

### Frisch-Waugh-Lovell Equivalence

The FE estimator is numerically identical to OLS with entity dummies (LSDV --- Least Squares Dummy Variable), a consequence of the Frisch-Waugh-Lovell theorem. The within transformation is computationally preferred when $N$ is large.

### Limitations

- Cannot estimate effects of **time-invariant** variables (they are absorbed by $\alpha_i$)
- Requires **within variation** in $X_{it}$ for identification
- Less efficient than RE when RE assumptions hold

## Random Effects Estimator

### GLS Approach

Under the **random effects assumption** --- $E[\alpha_i \mid X_{it}] = 0$ --- we can treat $\alpha_i$ as part of a composite error $c_{it} = \alpha_i + \varepsilon_{it}$. The RE estimator uses **feasible GLS** (FGLS):

$$
\hat{\beta}_{RE} = \left(\sum_{i=1}^N \sum_{t=1}^T \tilde{X}_{it}\tilde{X}_{it}'\right)^{-1} \sum_{i=1}^N \sum_{t=1}^T \tilde{X}_{it}\tilde{y}_{it}
$$

where the quasi-demeaned variables are:

$$
\tilde{y}_{it} = y_{it} - \hat{\theta}\bar{y}_i, \quad \tilde{X}_{it} = X_{it} - \hat{\theta}\bar{X}_i
$$

and $\hat{\theta} = 1 - \sqrt{\hat{\sigma}_\varepsilon^2 / (\hat{\sigma}_\varepsilon^2 + T\hat{\sigma}_\alpha^2)}$ is estimated from variance components.

### Properties

- **Consistent** if $E[\alpha_i \mid X_{it}] = 0$ (no correlation between effects and regressors)
- **More efficient** than FE when RE assumptions hold (uses both within and between variation)
- **Can estimate** effects of time-invariant variables
- **Inconsistent** if $\alpha_i$ is correlated with $X_{it}$

## Between Estimator

The **between estimator** (BE) uses only cross-sectional variation by regressing entity means:

$$
\bar{y}_i = \bar{X}_i'\beta + \alpha_i + \bar{\varepsilon}_i
$$

OLS on entity averages yields $\hat{\beta}_{BE}$. This estimator:

- Uses only **between-entity** variation
- Consistent under stronger assumptions (including $E[\alpha_i \mid \bar{X}_i] = 0$)
- Useful for decomposing variation (between vs. within)

## First Difference Estimator

Taking first differences of the panel equation eliminates $\alpha_i$:

$$
\Delta y_{it} = \Delta X_{it}'\beta + \Delta \varepsilon_{it}, \quad t = 2, \ldots, T
$$

### Properties

- Consistent under strict exogeneity (like FE)
- **More robust to serial correlation** than FE when $\varepsilon_{it}$ follows a random walk
- Identical to FE when $T = 2$
- Loses one time period per entity

## Model Selection: The Hausman Test

The Hausman (1978) test compares FE and RE estimators to test whether $\alpha_i$ is correlated with $X_{it}$:

$$
H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})'[\widehat{\text{Var}}(\hat{\beta}_{FE}) - \widehat{\text{Var}}(\hat{\beta}_{RE})]^{-1}(\hat{\beta}_{FE} - \hat{\beta}_{RE}) \sim \chi^2(K)
$$

| Result | Interpretation | Action |
|--------|---------------|--------|
| Reject $H_0$ | $\alpha_i$ correlated with $X_{it}$ | Use Fixed Effects |
| Fail to reject | No evidence of correlation | Random Effects may be appropriate |

!!! warning "Hausman Test Limitations"
    The Hausman test may be unreliable with heteroskedasticity, serial correlation, or cluster-dependent errors. In such cases, use robust versions or the Mundlak (1978) approach.

## Asymptotic Properties

### Large N, Fixed T

This is the standard micro-panel framework:

- **FE:** $\hat{\beta}_{FE}$ is consistent as $N \to \infty$ with $T$ fixed
- **RE:** $\hat{\beta}_{RE}$ is consistent under stronger assumptions
- **Individual effects $\hat{\alpha}_i$:** inconsistent (incidental parameters problem)

### Incidental Parameters Problem

In nonlinear models (logit, probit, Poisson) with fixed effects, the maximum likelihood estimator of $\beta$ is **inconsistent** when $T$ is fixed because the number of nuisance parameters ($N$ fixed effects) grows with $N$.

**Solutions:**

- Conditional likelihood (eliminates $\alpha_i$ --- available for logit)
- Bias correction (analytical or jackknife)
- Random effects with appropriate distributional assumptions

## Discrete Choice Models in Panel Context

### Fixed Effects Logit

For binary outcomes $y_{it} \in \{0, 1\}$:

$$
P(y_{it} = 1 \mid X_{it}, \alpha_i) = \frac{\exp(X_{it}'\beta + \alpha_i)}{1 + \exp(X_{it}'\beta + \alpha_i)}
$$

Chamberlain (1980) showed that conditioning on the sufficient statistic $\sum_t y_{it}$ eliminates $\alpha_i$:

$$
P(y_{i1}, \ldots, y_{iT} \mid X_i, \textstyle\sum_t y_{it}) = \frac{\exp(\sum_t y_{it} X_{it}'\beta)}{\sum_{d \in B_i} \exp(\sum_t d_t X_{it}'\beta)}
$$

where $B_i$ is the set of all binary sequences with the same sum as the observed sequence. This approach:

- Produces consistent estimates of $\beta$ with fixed $T$
- Drops entities with no variation in $y_{it}$ (all 0s or all 1s)
- Cannot estimate effects of time-invariant variables

### Random Effects Probit

$$
P(y_{it} = 1 \mid X_{it}, \alpha_i) = \Phi(X_{it}'\beta + \alpha_i), \quad \alpha_i \sim N(0, \sigma_\alpha^2)
$$

The likelihood requires integrating over $\alpha_i$, computed via **Gauss-Hermite quadrature**:

$$
L_i = \int \prod_t [\Phi(X_{it}'\beta + \alpha)]^{y_{it}}[1 - \Phi(X_{it}'\beta + \alpha)]^{1-y_{it}} \phi(\alpha; 0, \sigma_\alpha^2) \, d\alpha
$$

### Multinomial Logit

For unordered categorical outcomes $y_{it} \in \{1, \ldots, J\}$ with $J > 2$ alternatives, the multinomial logit model is based on **random utility maximization**:

$$
P(y_{it} = j \mid X_{it}) = \frac{\exp(X_{it}'\beta_j)}{\sum_{k=1}^J \exp(X_{it}'\beta_k)}
$$

**Identification** requires normalizing one category as baseline ($\beta_1 = 0$), yielding $(J-1) \times K$ parameters.

**IIA assumption** (Independence of Irrelevant Alternatives): The odds ratio between any two alternatives is independent of the other alternatives available. When IIA is violated, consider nested logit or mixed logit.

**Marginal effects** for continuous variable $x_k$:

$$
\frac{\partial P(y=j)}{\partial x_k} = P(y=j)\left[\beta_{jk} - \sum_{m=1}^J P(y=m)\beta_{mk}\right]
$$

### Ordered Models

For ordered categorical outcomes $y_{it} \in \{1, \ldots, J\}$ with a natural ordering (e.g., satisfaction levels), the ordered logit/probit uses a **latent variable formulation**:

$$
y_{it}^* = X_{it}'\beta + \varepsilon_{it}, \quad y_{it} = j \iff \kappa_{j-1} < y_{it}^* \leq \kappa_j
$$

where $\kappa_0 = -\infty < \kappa_1 < \cdots < \kappa_{J-1} < \kappa_J = +\infty$ are threshold (cutpoint) parameters.

## Selection Models

### The Selection Problem

Outcomes are observed only for a non-random subsample. If selection into the sample is correlated with the outcome, OLS on the selected sample is **biased**.

**Outcome equation:** $y_{it} = X_{it}'\beta + \alpha_i + \varepsilon_{it}$

**Selection equation:** $d_{it} = \mathbf{1}[W_{it}'\gamma + \eta_i + v_{it} > 0]$

Selection bias arises when $\text{Corr}(v_{it}, \varepsilon_{it}) = \rho \neq 0$.

### Heckman Two-Step Correction

**Step 1:** Estimate $\gamma$ via probit on the selection equation, compute the **Inverse Mills Ratio (IMR)**:

$$
\hat{\lambda}_{it} = \frac{\phi(W_{it}'\hat{\gamma})}{\Phi(W_{it}'\hat{\gamma})}
$$

**Step 2:** Augmented outcome regression on the selected sample ($d_{it} = 1$):

$$
y_{it} = X_{it}'\beta + \theta \hat{\lambda}_{it} + \text{error}_{it}
$$

where $\theta = \rho \sigma_\varepsilon$. A significant $\hat{\theta}$ indicates selection bias.

### Identification

A valid **exclusion restriction** is critical: at least one variable in $W_{it}$ should be excluded from $X_{it}$ --- it affects selection but not the outcome directly. Without this, identification relies solely on the nonlinearity of $\lambda$, which is fragile.

### Panel Extension (Wooldridge 1995)

The panel Heckman model accounts for individual random effects in both equations:

$$
(\eta_i, \alpha_i) \sim N(0, \Sigma_u), \quad (v_{it}, \varepsilon_{it}) \sim N(0, \Sigma_v)
$$

Standard errors require **Murphy-Topel correction** for the two-step approach, or joint MLE via Gauss-Hermite quadrature.

## Practical Implications

1. **Always test FE vs. RE** using the Hausman test (or Mundlak approach) before interpreting results
2. **FE is the safe default** when you suspect $\alpha_i$ is correlated with $X_{it}$ (most economic applications)
3. **RE gains efficiency** but at the cost of stronger assumptions
4. **First Difference** is useful when serial correlation in $\varepsilon_{it}$ is a concern
5. **Selection models** require careful identification through exclusion restrictions
6. **Discrete choice** with FE: use conditional logit; with RE: use random effects probit

## Key References

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press. --- Comprehensive textbook covering all fundamental panel methods.
- Baltagi, B.H. (2013). *Econometric Analysis of Panel Data*. Wiley. --- Standard reference for panel data econometrics.
- Arellano, M. (2003). *Panel Data Econometrics*. Oxford University Press. --- Advanced treatment of panel data theory.
- Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data." *Review of Economic Studies*, 47(1), 225--238. --- Conditional likelihood for FE logit.
- Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153--161. --- Original two-step selection correction.
- Wooldridge, J.M. (1995). "Selection Corrections for Panel Data Models Under Conditional Mean Independence Assumptions." *Journal of Econometrics*, 68(1), 115--132. --- Panel extension of Heckman model.
- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*, ed. P. Zarembka. --- Foundation of discrete choice theory.
- Hausman, J.A. (1978). "Specification Tests in Econometrics." *Econometrica*, 46(6), 1251--1271. --- The Hausman specification test.

## See Also

- [GMM Theory](gmm-theory.md) --- when strict exogeneity fails and dynamic models are needed
- [Quantile Theory](quantile-theory.md) --- going beyond the conditional mean
- [Spatial Theory](spatial-theory.md) --- when observations are spatially dependent
- [Frontier Theory](sfa-theory.md) --- efficiency measurement with panel data
- [Cointegration Theory](../diagnostics/cointegration/index.md) --- non-stationary panel data
- [References](references.md) --- complete bibliography
