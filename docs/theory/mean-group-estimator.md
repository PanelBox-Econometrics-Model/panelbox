---
title: "Mean Group Estimator Theory"
description: "Mathematical foundations of the Mean Group (MG) and Pooled Mean Group (PMG) estimators for heterogeneous panel data"
---

# Mean Group Estimator Theory

!!! abstract "Key Takeaway"
    When slope coefficients vary across entities, the standard Fixed Effects estimator produces inconsistent estimates of the average slope. The Mean Group (MG) estimator of Pesaran & Smith (1995) provides consistent estimates by averaging entity-specific regressions. The Pooled Mean Group (PMG) estimator extends this to error-correction models with long-run homogeneity constraints.

## The Problem: Slope Heterogeneity

### Standard Panel Model Assumption

The conventional panel data model assumes **slope homogeneity**:

$$
y_{it} = \alpha_i + X_{it}'\beta + \varepsilon_{it}
$$

where $\beta$ is common across all entities. Fixed Effects and Random Effects estimators are designed for this setting.

### When Slopes Differ

In many applications — cross-country growth regressions, firm-level production functions, household consumption — the slope coefficients genuinely vary across entities:

$$
y_{it} = \alpha_i + X_{it}'\beta_i + \varepsilon_{it}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T
$$

where $\beta_i$ is entity-specific. Pesaran & Smith (1995) showed that applying Fixed Effects to this model yields:

$$
\hat{\beta}_{FE} \xrightarrow{p} \beta + \underbrace{\left(\sum_i \Sigma_{X_i}\right)^{-1} \sum_i \Sigma_{X_i}(\beta_i - \beta)}_{\text{heterogeneity bias}}
$$

where $\Sigma_{X_i} = \text{plim } T^{-1} \ddot{X}_i'\ddot{X}_i$. The FE estimator converges to a **weighted average** of the true $\beta_i$, where the weights depend on the variance of the regressors — not the simple average. This bias persists even as $N, T \to \infty$.

!!! warning "FE Inconsistency"
    Under slope heterogeneity, the FE estimator is **inconsistent** for the mean slope $\bar{\beta} = N^{-1}\sum_i \beta_i$. The bias depends on the correlation between $\beta_i$ and $\Sigma_{X_i}$: entities with larger regressor variance receive disproportionate weight.

## The Mean Group Estimator

### Setup

Assume a random coefficients model:

$$
y_{it} = \alpha_i + X_{it}'\beta_i + \varepsilon_{it}
$$

where:

- $\beta_i = \beta + \eta_i$ with $E[\eta_i] = 0$
- $\varepsilon_{it} \sim \text{iid}(0, \sigma_i^2)$
- $\beta$ is the **mean slope** of interest

### Step 1: Entity-Specific OLS

For each entity $i$, run OLS on the $T_i$ observations:

$$
\hat{\beta}_i = (X_i'X_i)^{-1} X_i' y_i
$$

where $X_i$ includes a constant (entity-specific intercept) and the regressors.

### Step 2: Average Across Entities

The MG estimator is the simple (unweighted) average:

$$
\hat{\beta}_{MG} = \frac{1}{N} \sum_{i=1}^{N} \hat{\beta}_i
$$

### Variance Estimation

The variance of $\hat{\beta}_{MG}$ is estimated non-parametrically using the cross-sectional dispersion of the entity-specific estimates:

$$
\widehat{\text{Var}}(\hat{\beta}_{MG}) = \frac{1}{N(N-1)} \sum_{i=1}^{N} (\hat{\beta}_i - \hat{\beta}_{MG})(\hat{\beta}_i - \hat{\beta}_{MG})'
$$

This is the key advantage of MG: the variance estimator is **robust** to any form of heteroskedasticity and serial correlation in $\varepsilon_{it}$, because it relies only on the cross-sectional variation in the $\hat{\beta}_i$.

!!! note "No Sandwich Needed"
    Unlike robust standard errors for pooled estimators, MG standard errors do not require specifying a covariance structure. The cross-entity variation in $\hat{\beta}_i$ automatically captures all sources of uncertainty, including estimation error within each entity and genuine parameter heterogeneity.

## Asymptotic Properties

### Consistency

Under standard regularity conditions, as $N \to \infty$ with $T$ fixed (or $T \to \infty$):

$$
\hat{\beta}_{MG} \xrightarrow{p} \beta = E[\beta_i]
$$

The MG estimator is consistent for the **population mean** of the slope coefficients.

### Asymptotic Normality

As $N \to \infty$:

$$
\sqrt{N}(\hat{\beta}_{MG} - \beta) \xrightarrow{d} \mathcal{N}(0, \Sigma_\eta + \Sigma_\varepsilon)
$$

where:

- $\Sigma_\eta = \text{Var}(\eta_i) = \text{Var}(\beta_i)$ is the **cross-entity heterogeneity** in slopes
- $\Sigma_\varepsilon$ captures the **estimation uncertainty** within each entity (vanishes as $T \to \infty$)

### Requirements

| Condition | Description |
|-----------|-------------|
| $N \to \infty$ | Required for averaging to work |
| $T$ sufficiently large | Each entity-specific OLS must be well-identified ($T_i > K$) |
| $E[\eta_i] = 0$ | Heterogeneity is centered at the mean |
| Independent entities | Cross-sectional independence (can be relaxed) |

!!! tip "Minimum T per Entity"
    PanelBox requires $T_i \geq$ `min_obs_per_entity` (default: 10) for each entity. Entities with too few observations are excluded with a warning. Small $T_i$ leads to imprecise $\hat{\beta}_i$, inflating the MG variance.

## Comparison: MG vs FE Under Heterogeneity

| Property | Fixed Effects | Mean Group |
|----------|--------------|------------|
| **Target** | Weighted average of $\beta_i$ | Simple average of $\beta_i$ |
| **Consistency** | Inconsistent for $\bar{\beta}$ | Consistent for $\bar{\beta}$ |
| **Efficiency** | More efficient if slopes are homogeneous | Less efficient under homogeneity |
| **Variance** | Requires robust SE specification | Automatically robust |
| **Min $T$** | $T \geq 2$ | $T > K$ per entity |

When slopes are truly homogeneous ($\beta_i = \beta$ for all $i$), the MG estimator is consistent but **inefficient** relative to FE/RE. The Swamy test helps decide which regime applies.

## Swamy (1970) Test for Slope Homogeneity

### Hypothesis

$$
H_0: \beta_1 = \beta_2 = \cdots = \beta_N = \beta \quad \text{(slope homogeneity)}
$$

### Test Statistic

$$
\hat{S} = \sum_{i=1}^{N} (\hat{\beta}_i - \hat{\beta}_{WFE})' \hat{V}_i^{-1} (\hat{\beta}_i - \hat{\beta}_{WFE})
$$

where:

- $\hat{\beta}_{WFE}$ is the weighted Fixed Effects (GLS) estimator
- $\hat{V}_i = \hat{\sigma}_i^2 (X_i'X_i)^{-1}$ is the estimated covariance of $\hat{\beta}_i$

Under $H_0$:

$$
\hat{S} \xrightarrow{d} \chi^2\big((N-1) \cdot K\big)
$$

where $K$ is the number of slope coefficients.

### Interpretation

| Result | Action |
|--------|--------|
| Reject $H_0$ (p-value < 0.05) | Use MG or PMG — slopes are heterogeneous |
| Fail to reject $H_0$ | FE/RE are valid — MG is consistent but inefficient |

!!! warning "Power Considerations"
    With large $N$, the Swamy test has very high power and may reject even economically negligible heterogeneity. Consider the magnitude of the heterogeneity (via `coefficient_table()`) alongside the test result.

## Pooled Mean Group (PMG) Estimator

### Motivation

The MG estimator treats all coefficients as heterogeneous. In many economic applications, theory suggests that **long-run** relationships should be the same across entities (e.g., purchasing power parity, law of one price) while **short-run** dynamics can differ.

### Error Correction Model (ECM)

The PMG estimator is based on an autoregressive distributed lag (ARDL) model reparameterized as an ECM:

$$
\Delta y_{it} = \phi_i (y_{i,t-1} - \theta' X_{i,t-1}) + \delta_i' \Delta X_{it} + \mu_i + \varepsilon_{it}
$$

where:

| Symbol | Meaning | Homogeneous? |
|--------|---------|--------------|
| $\theta$ | Long-run coefficients | **Yes** — constrained equal across entities |
| $\phi_i$ | Error-correction speed of adjustment | No — entity-specific |
| $\delta_i$ | Short-run dynamics | No — entity-specific |
| $\mu_i$ | Entity intercept | No — entity-specific |

### The Long-Run Equilibrium

The term $y_{i,t-1} - \theta' X_{i,t-1}$ measures the deviation from the long-run equilibrium. The coefficient $\phi_i$ captures how fast entity $i$ corrects back to equilibrium:

- $\phi_i < 0$: the system is **stable** (error-correcting)
- $|\phi_i|$ close to 1: fast adjustment
- $|\phi_i|$ close to 0: slow adjustment
- $\phi_i \geq 0$: **no error correction** (model misspecified for this entity)

### Estimation: Concentrated Maximum Likelihood

PMG estimation proceeds by concentrated MLE:

1. **For a given $\theta$**: Run entity-specific OLS regressions to obtain $\hat{\phi}_i(\theta)$, $\hat{\delta}_i(\theta)$, and $\hat{\sigma}_i^2(\theta)$

2. **Concentrated log-likelihood**:
$$
\ell_c(\theta) = -\frac{1}{2} \sum_{i=1}^{N} T_i \ln \hat{\sigma}_i^2(\theta) + \text{const}
$$

3. **Optimize**: Find $\hat{\theta}_{PMG} = \arg\max_\theta \ell_c(\theta)$

4. **Final estimates**: Plug $\hat{\theta}_{PMG}$ back to recover all entity-specific parameters

### PMG Asymptotic Properties

Under the assumption of long-run homogeneity:

$$
\sqrt{N}(\hat{\theta}_{PMG} - \theta) \xrightarrow{d} \mathcal{N}(0, V_\theta)
$$

PMG is **more efficient** than MG when the long-run homogeneity restriction is valid, because it pools information across entities for the long-run parameters.

### MG vs PMG: Hausman Test

To test whether the long-run homogeneity restriction is valid:

$$
H_0: \theta_{MG} = \theta_{PMG} \quad \text{(homogeneity holds)}
$$

$$
H = (\hat{\theta}_{MG} - \hat{\theta}_{PMG})' [\widehat{\text{Var}}(\hat{\theta}_{MG}) - \widehat{\text{Var}}(\hat{\theta}_{PMG})]^{-1} (\hat{\theta}_{MG} - \hat{\theta}_{PMG})
$$

$$
H \xrightarrow{d} \chi^2(K)
$$

| Result | Conclusion |
|--------|-----------|
| Fail to reject | PMG is preferred (efficient + consistent) |
| Reject | MG is preferred (consistent under heterogeneity) |

## Weighted Mean Group

When entities have different sample sizes $T_i$ or different estimation precision, a weighted version can improve efficiency:

$$
\hat{\beta}_{WMG} = \sum_{i=1}^{N} w_i \hat{\beta}_i, \quad \sum_i w_i = 1
$$

Common weighting schemes:

| Weights | Formula | Rationale |
|---------|---------|-----------|
| Equal | $w_i = 1/N$ | Standard MG |
| Precision | $w_i \propto (X_i'X_i)$ | More weight to precisely estimated entities |
| Sample size | $w_i \propto T_i$ | More weight to entities with more observations |

PanelBox supports custom entity-level weights via the `weights` parameter.

## References

1. **Pesaran, M. H. & Smith, R. (1995)**. Estimating long-run relationships from dynamic heterogeneous panels. *Journal of Econometrics*, 68(1), 79-113.

2. **Pesaran, M. H., Shin, Y. & Smith, R. P. (1999)**. Pooled mean group estimation of dynamic heterogeneous panels. *Journal of the American Statistical Association*, 94(446), 621-634.

3. **Swamy, P. A. V. B. (1970)**. Efficient inference in a random coefficient regression model. *Econometrica*, 38(2), 311-323.

4. **Hsiao, C. (2014)**. *Analysis of Panel Data* (3rd ed.). Cambridge University Press. Chapter 6.

## See Also

- [Static Models API](../api/static-models.md) — `MeanGroupEstimator` and `PooledMeanGroupEstimator` reference
- [Slope Heterogeneity Guide](../user-guide/slope-heterogeneity.md) — Practical guide for choosing between MG, PMG, FE, and RE
- [Panel Data Fundamentals](panel-fundamentals.md) — FE and RE theory
