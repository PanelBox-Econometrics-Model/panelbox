---
title: "Panel VAR Theory"
description: "Mathematical foundations of Panel Vector Autoregression models, estimation, identification, and impulse response analysis"
---

# Panel VAR Theory --- Dynamic Multivariate Analysis

!!! abstract "Key Takeaway"
    Panel VAR models extend vector autoregression to panel data, allowing researchers to study dynamic interdependencies between multiple variables across entities. GMM-based estimation handles individual heterogeneity, while orthogonalized impulse response functions and variance decompositions reveal causal transmission mechanisms.

## Motivation

Standard panel regression models capture the effect of one variable on another, but many economic systems involve **simultaneous feedback** between multiple variables. For example:

- Investment and profitability: Does investment drive profits, or do profits drive investment?
- Trade and growth: Do trade openness and GDP growth reinforce each other?
- Government spending and output: What are the dynamic fiscal multiplier effects?

Panel VAR models address these questions by:

- Treating all variables as **jointly endogenous**
- Capturing **dynamic interdependencies** through lagged values
- Exploiting the **panel dimension** for more efficient estimation
- Providing tools for **structural analysis** (IRF, FEVD, Granger causality)

## Model Specification

### The Panel VAR(p) Model

For entity $i$ at time $t$, the Panel VAR of order $p$ is:

$$
Y_{it} = \sum_{j=1}^{p} A_j Y_{i,t-j} + \mu_i + \varepsilon_{it}
$$

where:

- $Y_{it}$ is a $K \times 1$ vector of endogenous variables
- $A_j$ are $K \times K$ coefficient matrices for lag $j$
- $\mu_i$ is a $K \times 1$ vector of entity-specific fixed effects
- $\varepsilon_{it}$ is a $K \times 1$ innovation vector with $E[\varepsilon_{it}] = 0$ and $E[\varepsilon_{it}\varepsilon_{it}'] = \Sigma$

### Stationarity Conditions

The system is stable if the eigenvalues of the companion matrix lie inside the unit circle:

$$
\det\left(I_{Kp} - A_1 z - A_2 z^2 - \cdots - A_p z^p\right) \neq 0 \quad \text{for } |z| \leq 1
$$

## Estimation

### Forward Orthogonal Deviations (Helmert Transformation)

To eliminate fixed effects $\mu_i$, PanelBox uses the **Helmert transformation** (forward orthogonal deviations) instead of first-differencing:

$$
\tilde{y}_{it} = \sqrt{\frac{T-t}{T-t+1}} \left( y_{it} - \frac{1}{T-t} \sum_{s=t+1}^{T} y_{is} \right)
$$

This transformation:

- Removes individual effects while preserving orthogonality of errors
- Avoids the serial correlation introduced by first-differencing
- Allows use of lagged levels as valid GMM instruments

### GMM Estimation

The transformed model is estimated by GMM using lagged levels as instruments:

$$
\hat{A} = \left(\tilde{X}' Z W^{-1} Z' \tilde{X}\right)^{-1} \tilde{X}' Z W^{-1} Z' \tilde{Y}
$$

where $Z$ is the instrument matrix constructed from appropriate lags.

## Structural Analysis

### Impulse Response Functions (IRF)

Orthogonalized IRFs use a Cholesky decomposition of the residual covariance matrix $\Sigma = PP'$ to identify structural shocks:

$$
\Theta_h = \Phi_h P
$$

where $\Phi_h$ are the moving average coefficients at horizon $h$, computed recursively:

$$
\Phi_h = \sum_{j=1}^{h} \Phi_{h-j} A_j, \quad \Phi_0 = I_K
$$

!!! warning "Ordering Sensitivity"
    Cholesky-based identification assumes a recursive causal ordering. The first variable in the system is assumed to be contemporaneously exogenous to all others. Results can be sensitive to variable ordering --- always test robustness.

### Forecast Error Variance Decomposition (FEVD)

FEVD measures the proportion of the $h$-step forecast error variance of variable $i$ attributable to shock $j$:

$$
\text{FEVD}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{k=1}^{K} (\Theta_s)_{ik}^2}
$$

### Granger Causality

Variable $j$ Granger-causes variable $i$ if lagged values of $j$ significantly predict $i$, conditional on lagged values of all other variables. This is tested via a Wald test:

$$
H_0: (A_1)_{ij} = (A_2)_{ij} = \cdots = (A_p)_{ij} = 0
$$

## Panel VECM

When variables are cointegrated (non-stationary but with stable long-run relationships), the Panel VECM representation is:

$$
\Delta Y_{it} = \Pi Y_{i,t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta Y_{i,t-j} + \mu_i + \varepsilon_{it}
$$

where:

- $\Pi = \alpha \beta'$ is the long-run impact matrix with rank $r$ (number of cointegrating relationships)
- $\alpha$ is the $K \times r$ matrix of adjustment speeds
- $\beta$ is the $K \times r$ matrix of cointegrating vectors
- $\Gamma_j$ are short-run dynamics matrices

## Confidence Intervals

PanelBox computes confidence bands for IRFs and FEVDs using **Monte Carlo simulation** from the estimated coefficient distribution, providing reliable inference on dynamic responses.

## References

- Abrigo, M. R. M. & Love, I. (2016). Estimation of panel vector autoregression in Stata. *The Stata Journal*, 16(3), 778--804.
- Holtz-Eakin, D., Newey, W. & Rosen, H. S. (1988). Estimating vector autoregressions with panel data. *Econometrica*, 56(6), 1371--1395.
- Love, I. & Zicchino, L. (2006). Financial development and dynamic investment behavior. *The Quarterly Review of Economics and Finance*, 46(2), 190--210.

## See Also

- [VAR User Guide](../user-guide/var/estimation.md) -- Estimation and usage
- [VAR Tutorial](../tutorials/var.md) -- Hands-on notebooks
- [API Reference: VAR](../api/var.md) -- Complete API documentation
- [VECM Models](../user-guide/var/vecm.md) -- Cointegrated VAR
