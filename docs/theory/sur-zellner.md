---
title: "SUR Theory — Zellner's Seemingly Unrelated Regressions"
description: "Derivation of the SUR/FGLS estimator, Kronecker product structure, iterated SUR equivalence with MLE, Breusch-Pagan test, and McElroy system R-squared"
---

# Seemingly Unrelated Regressions (SUR)

!!! abstract "Key Takeaway"
    SUR exploits cross-equation error correlations to produce more efficient estimates than equation-by-equation OLS. In panel data, each entity is an equation. When entities have correlated errors **and** different regressors, SUR yields substantial efficiency gains.

## Motivation

Consider a panel with $N$ entities observed over $T$ periods. Standard panel estimators (FE, RE) impose a common coefficient vector $\beta$ across entities. But what if:

- Entities have **different regression specifications** (heterogeneous formulas)?
- Errors across entities are **contemporaneously correlated**?
- You want **entity-specific coefficients** while borrowing strength across equations?

Zellner (1962) showed that if the errors across equations are correlated and the regressors differ, joint estimation via GLS is more efficient than equation-by-equation OLS.

---

## The SUR Model

### Setup

For $N$ entities (equations) and $T$ time periods, the $i$-th equation is:

$$
y_i = X_i \beta_i + \varepsilon_i, \quad i = 1, \ldots, N
$$

where:

- $y_i$ is a $T \times 1$ vector of observations for entity $i$
- $X_i$ is a $T \times k_i$ design matrix (potentially different across entities)
- $\beta_i$ is a $k_i \times 1$ coefficient vector
- $\varepsilon_i$ is a $T \times 1$ error vector

### Stacked System

Stack all equations:

$$
\underbrace{\begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{pmatrix}}_{y}
=
\underbrace{\begin{pmatrix} X_1 & 0 & \cdots & 0 \\ 0 & X_2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & X_N \end{pmatrix}}_{X}
\underbrace{\begin{pmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_N \end{pmatrix}}_{\beta}
+
\underbrace{\begin{pmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_N \end{pmatrix}}_{\varepsilon}
$$

### Error Structure

The key assumption is contemporaneous correlation across equations:

$$
E[\varepsilon_i \varepsilon_j'] = \sigma_{ij} I_T
$$

Collecting all cross-equation covariances into an $N \times N$ matrix $\Sigma = [\sigma_{ij}]$:

$$
E[\varepsilon \varepsilon'] = \Sigma \otimes I_T
$$

where $\otimes$ denotes the Kronecker product. Errors are uncorrelated over time but correlated across entities within the same period.

---

## The GLS Estimator

### Derivation

Given the covariance structure $\Omega = \Sigma \otimes I_T$, the GLS estimator is:

$$
\hat{\beta}_{GLS} = (X' \Omega^{-1} X)^{-1} X' \Omega^{-1} y
$$

Using the Kronecker product property $(\Sigma \otimes I_T)^{-1} = \Sigma^{-1} \otimes I_T$:

$$
\hat{\beta}_{SUR} = \bigl(X'(\Sigma^{-1} \otimes I_T) X\bigr)^{-1} X'(\Sigma^{-1} \otimes I_T) y
$$

### Block Structure

The $(i,j)$ block of the system matrix $A = X'(\Sigma^{-1} \otimes I_T) X$ is:

$$
A_{ij} = \sigma^{ij} X_i' X_j
$$

where $\sigma^{ij}$ is the $(i,j)$ element of $\Sigma^{-1}$. Similarly, the $i$-th block of $X'(\Sigma^{-1} \otimes I_T)y$ is:

$$
b_i = \sum_{j=1}^{N} \sigma^{ij} X_i' y_j
$$

The SUR estimator then solves:

$$
A \hat{\beta} = b
$$

### Covariance of the Estimator

$$
\text{Var}(\hat{\beta}_{SUR}) = \bigl(X'(\Sigma^{-1} \otimes I_T) X\bigr)^{-1}
$$

---

## Efficient Implementation via Kronecker Structure

!!! tip "Implementation Detail"
    PanelBox **never** materializes the full $NT \times NT$ matrix $\Sigma^{-1} \otimes I_T$. Instead, it loops over the $N^2$ blocks, accumulating $\sigma^{ij} X_i' X_j$ into the $\sum k_i \times \sum k_i$ system matrix. Memory scales as $O((\sum k_i)^2)$, not $O(N^2 T^2)$.

For a balanced panel with $N$ entities, $T$ periods, and $k$ regressors per equation:

| Approach | Memory | Time |
|----------|--------|------|
| Naive full $\Omega^{-1}$ | $O(N^2 T^2)$ | $O(N^3 T^3)$ |
| Kronecker block | $O(N^2 k^2)$ | $O(N^2 T k^2)$ |

The block approach is feasible even for large $N$ and $T$ as long as $k$ is modest.

---

## Feasible GLS (FGLS)

In practice, $\Sigma$ is unknown. The **two-step** FGLS procedure is:

**Step 1.** Estimate each equation by OLS:

$$
\hat{\beta}_i^{OLS} = (X_i' X_i)^{-1} X_i' y_i
$$

Compute residuals $\hat{\varepsilon}_i = y_i - X_i \hat{\beta}_i^{OLS}$.

**Step 2.** Estimate $\Sigma$:

$$
\hat{\sigma}_{ij} = \frac{1}{T} \hat{\varepsilon}_i' \hat{\varepsilon}_j
$$

!!! note "Unbalanced Panels"
    For unbalanced panels, $\hat{\sigma}_{ij}$ is computed using the $T_{ij}$ common time periods between entities $i$ and $j$: $\hat{\sigma}_{ij} = \hat{\varepsilon}_i' \hat{\varepsilon}_j / T_{ij}$.

**Step 3.** Apply GLS using $\hat{\Sigma}$ in place of $\Sigma$ to obtain $\hat{\beta}_{FGLS}$.

The FGLS estimator is consistent and asymptotically equivalent to GLS as $T \to \infty$.

---

## Iterated SUR (ISUR) and MLE Equivalence

### ISUR Algorithm

Iterated SUR repeats the FGLS procedure:

1. Obtain $\hat{\beta}^{(0)}$ from FGLS (the two-step estimator).
2. Compute new residuals $\hat{\varepsilon}^{(s)} = y_i - X_i \hat{\beta}_i^{(s)}$.
3. Re-estimate $\hat{\Sigma}^{(s+1)}$ from $\hat{\varepsilon}^{(s)}$.
4. Re-estimate $\hat{\beta}^{(s+1)}$ via GLS with $\hat{\Sigma}^{(s+1)}$.
5. Repeat until $\|\hat{\beta}^{(s+1)} - \hat{\beta}^{(s)}\| / \|\hat{\beta}^{(s)}\| < \text{tol}$.

### MLE Equivalence

!!! warning "Key Result"
    Under normality of errors, the iterated SUR estimator converges to the **Maximum Likelihood Estimator**. This is because ISUR iteratively maximizes the concentrated log-likelihood, which is equivalent to iterating between estimating $\beta$ and $\Sigma$ — the EM algorithm for the multivariate normal regression model.

The concentrated log-likelihood is:

$$
\ell(\beta) = -\frac{NT}{2}\ln(2\pi) - \frac{T}{2}\ln|\hat{\Sigma}(\beta)| - \frac{NT}{2}
$$

At convergence, ISUR satisfies the first-order conditions of MLE.

---

## Breusch-Pagan Test for Independence

The Breusch-Pagan (1980) LM test checks whether the off-diagonal elements of $\Sigma$ are jointly zero:

$$
H_0: \sigma_{ij} = 0 \quad \text{for all } i \neq j
$$

### Test Statistic

$$
\lambda_{BP} = T \sum_{i=2}^{N} \sum_{j=1}^{i-1} r_{ij}^2
$$

where $r_{ij} = \hat{\sigma}_{ij} / \sqrt{\hat{\sigma}_{ii} \hat{\sigma}_{jj}}$ is the sample correlation of OLS residuals between equations $i$ and $j$.

Under $H_0$:

$$
\lambda_{BP} \sim \chi^2\left(\frac{N(N-1)}{2}\right)
$$

### Interpretation

| Result | Meaning | Implication |
|--------|---------|-------------|
| Reject $H_0$ | Cross-equation correlations are significant | SUR is more efficient than OLS |
| Fail to reject $H_0$ | Errors appear independent across equations | OLS is as efficient as SUR; no gain |

!!! tip "When the BP test fails to reject"
    If errors are uncorrelated, SUR reduces to equation-by-equation OLS. The efficiency gain from SUR depends on the magnitude of off-diagonal correlations in $\Sigma$ **and** the difference in regressors across equations.

---

## McElroy System R-squared

McElroy (1977) proposed a system-wide goodness-of-fit measure:

$$
R^2_{sys} = 1 - \frac{\hat{\varepsilon}' (\hat{\Sigma}^{-1} \otimes I_T) \hat{\varepsilon}}{y^{*\prime} (\hat{\Sigma}^{-1} \otimes I_T) y^*}
$$

where $y^* = y - \bar{y}$ is the de-meaned dependent variable vector.

This measure accounts for the cross-equation covariance structure and ranges from 0 to 1. A value of 1 indicates perfect fit across all equations simultaneously.

---

## Degenerate Case: Identical Regressors

!!! warning "No Efficiency Gain"
    When all equations have the **same regressors** ($X_1 = X_2 = \cdots = X_N$), the SUR estimator is **numerically identical** to equation-by-equation OLS. This is Zellner's classic result: the GLS transformation cannot improve on OLS when the design matrices are the same.

Formally, when $X_i = X$ for all $i$:

$$
\hat{\beta}_{SUR} = \hat{\beta}_{OLS}
$$

The standard errors may still differ (GLS SEs account for $\Sigma$), but the point estimates are identical. Efficiency gains from SUR come from **cross-equation correlations** combined with **different regressors**.

---

## Relationship to Other Estimators

### SUR vs Fixed Effects (FE)

| Aspect | SUR | FE |
|--------|-----|-----|
| Coefficients | Entity-specific $\beta_i$ | Common $\beta$ |
| Cross-equation correlation | Exploited via $\Sigma$ | Ignored |
| Heterogeneous formulas | Supported | Not supported |
| Asymptotic regime | $T \to \infty$ (fixed $N$) | $N \to \infty$ or both |

### SUR vs Mean Group (MG)

| Aspect | SUR | MG |
|--------|-----|-----|
| Estimation | Joint GLS system | Separate OLS, then average |
| Cross-equation correlation | Exploited | Ignored |
| Efficiency | More efficient when $\Sigma$ off-diagonal | No cross-equation info |
| Coefficients | Entity-specific | Entity-specific (averaged) |

### SUR vs PCSE

Panel-Corrected Standard Errors (Beck-Katz) estimate a common $\beta$ with Pooled OLS but correct SEs for cross-sectional correlation via $\hat{\Sigma}$. SUR goes further by using $\hat{\Sigma}$ in estimation itself (not just inference), and allows entity-specific coefficients.

### SUR and PanelVAR

The `cov_type='sur'` option in PanelBox's PanelVAR uses the same cross-equation covariance structure. In a VAR context, each equation in the VAR system is a "seemingly unrelated regression" — this is the standard approach for VAR estimation when error terms are correlated across equations but regressors differ.

---

## Asymptotic Properties

Under standard regularity conditions:

1. **Consistency**: $\hat{\beta}_{FGLS} \xrightarrow{p} \beta$ as $T \to \infty$.
2. **Asymptotic normality**: $\sqrt{T}(\hat{\beta}_{FGLS} - \beta) \xrightarrow{d} N(0, V)$ where $V = \text{plim}_{T \to \infty} T \cdot (X'(\Sigma^{-1} \otimes I_T)X)^{-1}$.
3. **Asymptotic efficiency**: Among linear unbiased estimators, SUR achieves the Gauss-Markov lower bound under the assumed error structure.
4. **ISUR convergence**: The iterated estimator converges to MLE under normality; inference is valid without normality for the non-iterated version (sandwich SEs).

!!! note "Finite-Sample Considerations"
    - SUR requires $T > k_i$ for each equation (more periods than regressors).
    - $\hat{\Sigma}$ must be estimable: need common time periods across entities.
    - With many entities and few periods, $\hat{\Sigma}$ is poorly estimated — consider restricting to a subset of entities or using shrinkage.

---

## References

1. Zellner, A. (1962). "An Efficient Method of Estimating Seemingly Unrelated Regressions and Tests for Aggregation Bias." *Journal of the American Statistical Association*, 57(298), 348-368.
2. Zellner, A. & Huang, D. S. (1962). "Further Properties of Efficient Estimators for Seemingly Unrelated Regression Equations." *International Economic Review*, 3(3), 300-313.
3. McElroy, M. B. (1977). "Goodness of Fit for Seemingly Unrelated Regressions." *Journal of Econometrics*, 6(3), 381-387.
4. Breusch, T. S. & Pagan, A. R. (1980). "The Lagrange Multiplier Test and its Applications to Model Specification in Econometrics." *Review of Economic Studies*, 47(1), 239-253.
5. Greene, W. H. (2018). *Econometric Analysis*, 8th ed. Pearson. Chapter 10.
