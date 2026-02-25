---
title: "Spatial Econometrics Theory"
description: "Mathematical foundations of spatial panel data models and geographic dependence"
---

# Spatial Econometrics Theory --- Modeling Geographic Dependence

!!! abstract "Key Takeaway"
    When observations are spatially dependent, standard estimators produce biased or inefficient results. Spatial econometric models explicitly incorporate geographic relationships through a spatial weight matrix $W$, capturing spillovers between entities. The choice between spatial lag (SAR), spatial error (SEM), and spatial Durbin (SDM) models depends on the nature of the dependence.

## Motivation

Tobler's First Law of Geography states: "Everything is related to everything else, but near things are more related than distant things." In economic data, this manifests as:

- Housing prices in neighboring areas are correlated
- Unemployment rates cluster geographically
- Technology adoption spreads through proximity
- Policy decisions of one region affect neighbors

Ignoring spatial dependence leads to:

- **Biased estimates** (if dependence is in the dependent variable)
- **Inefficient estimates** and invalid inference (if dependence is in the errors)
- **Misleading standard errors** and hypothesis tests

## The Spatial Weight Matrix

All spatial models require a **spatial weight matrix** $W$ that encodes the relationships between entities:

$$
W = \begin{pmatrix}
0 & w_{12} & w_{13} & \cdots & w_{1N} \\
w_{21} & 0 & w_{23} & \cdots & w_{2N} \\
\vdots & & \ddots & & \vdots \\
w_{N1} & w_{N2} & \cdots & w_{N,N-1} & 0
\end{pmatrix}
$$

The diagonal is zero by convention (an entity is not its own neighbor).

### Types of Weight Matrices

| Type | Definition | Use Case |
|------|-----------|----------|
| **Queen contiguity** | $w_{ij} = 1$ if $i$ and $j$ share a boundary or vertex | Administrative regions |
| **Rook contiguity** | $w_{ij} = 1$ if $i$ and $j$ share an edge | Regular grids |
| **Distance-based** | $w_{ij} = 1$ if $d_{ij} < \bar{d}$, else 0 | Point data |
| **Inverse distance** | $w_{ij} = d_{ij}^{-\alpha}$ | Gravity-type models |
| **K-nearest neighbors** | $w_{ij} = 1$ if $j$ is among $k$ nearest neighbors of $i$ | Irregular layouts |

### Row Standardization

Most applications use **row-standardized** weights:

$$
w_{ij}^* = \frac{w_{ij}}{\sum_{j=1}^N w_{ij}}
$$

This ensures $Wy_t$ is a weighted average of neighbors' values and the spatial parameter has a natural interpretation on $(-1, 1)$.

## Spatial Lag Model (SAR)

### Specification

$$
y_{it} = \rho \sum_{j=1}^N w_{ij} y_{jt} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

or in matrix form: $y_t = \rho W y_t + X_t \beta + \alpha + \varepsilon_t$.

### Key Features

- The term $Wy$ is **endogenous** --- $y_j$ depends on $y_i$ and vice versa
- OLS is inconsistent; requires **ML** or **IV** estimation
- Captures **global spillovers** with multiplier effects

### Spatial Multiplier

Solving for $y$:

$$
y_t = (I_N - \rho W)^{-1}(X_t\beta + \alpha + \varepsilon_t)
$$

The matrix $(I_N - \rho W)^{-1}$ is the **spatial multiplier**, which propagates shocks through the spatial network. A change in $x_j$ affects not just entity $j$ but all connected entities through higher-order feedback.

### Parameter Interpretation

- $\rho > 0$: Positive spatial dependence (clustering of similar values)
- $\rho < 0$: Negative spatial dependence (dissimilar neighbors)
- $|\rho| < 1$ required for stability (with row-standardized $W$)

## Spatial Error Model (SEM)

### Specification

$$
y_{it} = X_{it}'\beta + \alpha_i + u_{it}
$$
$$
u_{it} = \lambda \sum_{j=1}^N w_{ij} u_{jt} + \varepsilon_{it}
$$

Spatial dependence is in the **error term**, not the dependent variable.

### Key Features

- OLS is **unbiased but inefficient** (standard errors are wrong)
- Reflects omitted spatially correlated variables
- **No multiplier effects** --- a shock to entity $j$'s covariates affects only $j$'s outcome
- $\beta$ coefficients have the standard interpretation: $\partial y_i / \partial x_{ik} = \beta_k$

### When to Use

SEM is appropriate when spatial dependence arises from:

- Common unobserved shocks (weather, macro conditions)
- Measurement error with spatial patterns
- Omitted spatially correlated variables that do not directly cause $y$

## Spatial Durbin Model (SDM)

### Specification

$$
y_{it} = \rho \sum_{j=1}^N w_{ij} y_{jt} + X_{it}'\beta + \sum_{j=1}^N w_{ij} X_{jt}'\theta + \alpha_i + \varepsilon_{it}
$$

The SDM includes spatial lags of both the dependent variable and the covariates.

### Generality

The SDM is a general specification that **nests** both SAR and SEM:

- **SAR:** $\theta = 0$
- **SEM:** $\theta + \rho\beta = 0$ (common factor restriction)

LeSage and Pace (2009) recommend the SDM as the default starting point for spatial analysis, testing restrictions to simpler models afterward.

### Direct and Indirect Effects

In spatial models with endogenous spatial lags, coefficient interpretation requires computing **partial derivatives**:

$$
\frac{\partial y}{\partial x_k'} = (I_N - \rho W)^{-1}(I_N \beta_k + W\theta_k)
$$

| Effect | Definition | Interpretation |
|--------|-----------|----------------|
| **Direct** | Average diagonal of $(I - \rho W)^{-1}(I\beta_k + W\theta_k)$ | Effect of $x_{ik}$ on $y_i$ (includes feedback) |
| **Indirect** | Average off-diagonal row sum | Effect of $x_{jk}$ on $y_i$ for $j \neq i$ (spillover) |
| **Total** | Direct + Indirect | Complete effect including all spatial channels |

!!! info "Coefficient $\neq$ Marginal Effect"
    In SAR and SDM models, the regression coefficient $\beta_k$ is **not** the marginal effect of $x_k$ on $y$. Always compute and report direct, indirect, and total effects.

## General Nesting Spatial Model (GNS)

### Specification

$$
y_{it} = \rho \sum_{j=1}^N w_{ij} y_{jt} + X_{it}'\beta + \sum_{j=1}^N w_{ij} X_{jt}'\theta + u_{it}
$$
$$
u_{it} = \lambda \sum_{j=1}^N w_{ij} u_{jt} + \varepsilon_{it}
$$

The GNS nests all other spatial models ($\rho$, $\theta$, and $\lambda$ all present). It is the most flexible but also the most complex specification.

## Dynamic Spatial Panels

When both spatial and temporal dependence are present:

$$
y_{it} = \tau y_{i,t-1} + \rho \sum_j w_{ij} y_{jt} + \phi \sum_j w_{ij} y_{j,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

This model captures:

- **Temporal persistence** ($\tau$): own past affects current outcome
- **Contemporaneous spatial dependence** ($\rho$): neighbors' current outcomes matter
- **Space-time diffusion** ($\phi$): neighbors' past outcomes matter

## Estimation Methods

### Maximum Likelihood (ML)

The log-likelihood for SAR with panel FE:

$$
\ell = -\frac{NT}{2}\ln(2\pi\sigma^2) + T\ln|I_N - \rho W| - \frac{1}{2\sigma^2}\sum_{i,t}(y_{it} - \rho Wy_{it} - X_{it}'\beta - \alpha_i)^2
$$

The term $\ln|I_N - \rho W|$ (log-determinant) is the main computational challenge. Efficient computation uses eigenvalues of $W$: $\ln|I - \rho W| = \sum_{i=1}^N \ln(1 - \rho \omega_i)$.

- **Best for** small to medium panels ($N < 1000$)
- Efficient under correct specification

### Generalized Method of Moments

For large panels, GMM avoids computing the log-determinant:

- Uses instruments $W^2X, W^3X, \ldots$ for the endogenous $Wy$
- **Best for** large panels ($N > 1000$)
- Less efficient than ML but computationally cheaper

### Quasi-Maximum Likelihood (QML)

- Robust to non-normality of $\varepsilon_{it}$
- Consistent under weaker distributional assumptions
- Requires the same log-determinant computation as ML

## Testing for Spatial Autocorrelation

### Moran's I

The most common global test for spatial autocorrelation:

$$
I = \frac{N}{S_0} \frac{e'We}{e'e}
$$

where $e$ are residuals and $S_0 = \sum_i \sum_j w_{ij}$. The standardized statistic $Z_I = (I - E[I])/\sqrt{\text{Var}[I]}$ is asymptotically $N(0,1)$ under the null of no spatial autocorrelation.

- $I > E[I]$: Positive spatial autocorrelation (clustering)
- $I < E[I]$: Negative spatial autocorrelation (dispersion)

### Local Indicators of Spatial Association (LISA)

Local Moran's $I_i$ identifies **local clusters** and outliers:

$$
I_i = \frac{(y_i - \bar{y})}{\sigma^2} \sum_j w_{ij}(y_j - \bar{y})
$$

| Cluster Type | Meaning |
|-------------|---------|
| High-High (HH) | Hot spots --- high values surrounded by high values |
| Low-Low (LL) | Cold spots --- low values surrounded by low values |
| High-Low (HL) | Spatial outlier --- high value among low neighbors |
| Low-High (LH) | Spatial outlier --- low value among high neighbors |

### LM Tests for Model Selection

Lagrange Multiplier tests on OLS residuals guide model choice:

**LM-Lag test** ($H_0: \rho = 0$):

$$
LM_\rho = \frac{(e'Wy / \hat{\sigma}^2)^2}{T_\rho}
$$

**LM-Error test** ($H_0: \lambda = 0$):

$$
LM_\lambda = \frac{(e'We / \hat{\sigma}^2)^2}{\text{tr}(W'W + W^2)}
$$

**Decision rule:**

1. Only LM-Lag significant: use **SAR**
2. Only LM-Error significant: use **SEM**
3. Both significant: check robust versions
    - Robust LM-Lag significant: **SAR**
    - Robust LM-Error significant: **SEM**
    - Both robust significant: **SDM** or **GNS**

## Model Comparison

| Feature | SAR | SEM | SDM | GNS |
|---------|-----|-----|-----|-----|
| Spatial lag of $y$ | Yes | No | Yes | Yes |
| Spatial lag of $X$ | No | No | Yes | Yes |
| Spatial error | No | Yes | No | Yes |
| Global spillovers | Yes | No | Yes | Yes |
| Parameters | $K+2$ | $K+2$ | $2K+2$ | $2K+3$ |
| Interpretation | Simple | Simple | Complex | Complex |

## Practical Implications

1. **Always test first:** Run Moran's I on OLS residuals before fitting a spatial model
2. **Use LM tests** to guide model selection, but also consider economic theory
3. **SDM as default:** When unsure, start with SDM (nests SAR and SEM)
4. **Report effects:** For SAR and SDM, always compute direct, indirect, and total effects
5. **Sensitivity to $W$:** Test robustness to different weight matrix specifications
6. **Row-standardize $W$** for interpretable spatial parameters

## Key References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers. --- Foundational textbook for spatial econometrics.
- LeSage, J. & Pace, R.K. (2009). *Introduction to Spatial Econometrics*. CRC Press. --- Modern treatment with direct/indirect effects decomposition.
- Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer. --- Comprehensive guide to spatial panel models.
- Lee, L.F. & Yu, J. (2010). "Estimation of Spatial Autoregressive Panel Data Models with Fixed Effects." *Journal of Econometrics*, 154(2), 165--185. --- ML estimation of spatial panel models.
- Anselin, L. (1995). "Local Indicators of Spatial Association --- LISA." *Geographical Analysis*, 27(2), 93--115. --- Local spatial autocorrelation measures.
- Moran, P.A.P. (1950). "Notes on Continuous Stochastic Phenomena." *Biometrika*, 37, 17--23. --- The original Moran's I statistic.

## See Also

- [Panel Fundamentals](panel-fundamentals.md) --- the non-spatial panel models that spatial methods extend
- [GMM Theory](gmm-theory.md) --- GMM as an alternative estimation method for spatial models
- [Cointegration Theory](../diagnostics/cointegration/index.md) --- non-stationarity in spatial panels
- [References](references.md) --- complete bibliography
