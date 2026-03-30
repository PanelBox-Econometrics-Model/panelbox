---
title: "Nickell Bias & LSDVC Theory"
description: "Mathematical foundations of the Nickell bias in dynamic panels and the LSDVC bias correction"
---

# Nickell Bias & LSDVC Correction

!!! abstract "Key Takeaway"
    Fixed Effects estimation of dynamic panel models is inconsistent for fixed T due to the Nickell (1981) bias. The LSDVC estimator corrects this analytically using Kiviet (1995) bias formulas, with bootstrap inference following Bruno (2005). LSDVC is particularly useful when N is small to moderate, where GMM asymptotics may not hold.

## The Dynamic Panel Model

Consider the standard dynamic panel data model with individual fixed effects:

$$
y_{it} = \rho \, y_{i,t-1} + X_{it}'\beta + \alpha_i + \varepsilon_{it}
$$

where:

- $y_{it}$ is the dependent variable for entity $i$ at time $t$
- $\rho$ is the autoregressive parameter ($|\rho| < 1$ for stationarity)
- $X_{it}$ is a vector of exogenous regressors
- $\alpha_i$ is the unobserved entity-specific fixed effect
- $\varepsilon_{it} \sim \text{iid}(0, \sigma^2_\varepsilon)$, independent of $\alpha_i$

For $i = 1, \ldots, N$ entities and $t = 1, \ldots, T$ time periods.

## The Nickell (1981) Bias

### Within Transformation

The standard Fixed Effects (within) estimator removes $\alpha_i$ by demeaning:

$$
\ddot{y}_{it} = \rho \, \ddot{y}_{i,t-1} + \ddot{X}_{it}'\beta + \ddot{\varepsilon}_{it}
$$

where $\ddot{z}_{it} = z_{it} - \bar{z}_i$ and $\bar{z}_i = \frac{1}{T}\sum_{t=1}^T z_{it}$.

### Source of the Bias

The demeaned lagged dependent variable $\ddot{y}_{i,t-1}$ is correlated with the demeaned error $\ddot{\varepsilon}_{it}$, because:

$$
\ddot{y}_{i,t-1} = y_{i,t-1} - \bar{y}_i
$$

contains $y_{i,t-1}$, which depends on $\varepsilon_{i,t-1}$, while:

$$
\ddot{\varepsilon}_{it} = \varepsilon_{it} - \bar{\varepsilon}_i = \varepsilon_{it} - \frac{1}{T}\sum_{s=1}^T \varepsilon_{is}
$$

contains $-\frac{1}{T}\varepsilon_{i,t-1}$ through the entity mean $\bar{\varepsilon}_i$. This correlation does **not** vanish as $N \to \infty$ with fixed $T$.

### Bias Formula

Nickell (1981) showed that for the AR(1) model without exogenous regressors:

$$
\text{plim}_{N \to \infty}(\hat{\rho}_{FE} - \rho) \approx -\frac{1 + \rho}{T - 1} \left[1 - \frac{1}{T(1-\rho)} \cdot \frac{1 - \rho^T}{1 - \rho}\right]^{-1}
$$

For large $T$, this simplifies to the well-known approximation:

$$
\text{plim}(\hat{\rho}_{FE} - \rho) \approx -\frac{1 + \rho}{T - 1}
$$

!!! warning "Key Properties of the Nickell Bias"
    - **Always downward** for positive $\rho$: the FE estimator underestimates persistence
    - **Does not vanish** as $N \to \infty$ with fixed $T$
    - **Magnitude depends on T**: with $\rho = 0.7$ and $T = 10$, bias $\approx -0.19$; with $T = 5$, bias $\approx -0.43$
    - **Contaminates $\hat{\beta}$**: bias in $\hat{\rho}$ propagates to all coefficient estimates

### Numerical Examples

| $\rho$ | $T = 5$ | $T = 10$ | $T = 20$ | $T = 50$ |
|--------|---------|----------|----------|----------|
| 0.3 | -0.325 | -0.144 | -0.068 | -0.027 |
| 0.5 | -0.375 | -0.167 | -0.079 | -0.031 |
| 0.7 | -0.425 | -0.189 | -0.089 | -0.035 |
| 0.9 | -0.475 | -0.211 | -0.100 | -0.039 |

The table shows the approximate bias $-\frac{1+\rho}{T-1}$ for different combinations of $\rho$ and $T$.

## The LSDV Estimator

The LSDV (Least Squares Dummy Variable) estimator is algebraically equivalent to FE but explicitly includes entity dummies $D_i$:

$$
y_{it} = \rho \, y_{i,t-1} + X_{it}'\beta + \sum_{i=1}^{N} \alpha_i D_i + \varepsilon_{it}
$$

In matrix notation, define $W = [y_{-1}, X]$ and $D$ as the matrix of entity dummies. The LSDV estimator is:

$$
\hat{\delta}_{LSDV} = (W'A W)^{-1} W'A y
$$

where $A = I_{NT} - D(D'D)^{-1}D'$ is the within-transformation matrix that projects out entity means, and $\delta = (\rho, \beta')'$ is the full parameter vector.

## Kiviet (1995) Bias Correction

### The Idea

Rather than using instruments (as in GMM), Kiviet (1995) derives an **analytical expression** for the bias of $\hat{\delta}_{LSDV}$ and subtracts it:

$$
\hat{\delta}_{LSDVC} = \hat{\delta}_{LSDV} - \widehat{\text{Bias}}(\hat{\delta}_{LSDV})
$$

The bias depends on unknown parameters $(\rho, \beta, \sigma^2)$, which are estimated iteratively.

### Bias Decomposition

The bias of the LSDV estimator can be decomposed as:

$$
E[\hat{\delta}_{LSDV} - \delta] = c_1(\delta, \sigma^2) + c_2(\delta, \sigma^2) + c_3(\delta, \sigma^2) + O(N^{-1}T^{-2})
$$

where $c_1$, $c_2$, $c_3$ are bias terms of increasing order.

### Order 1: $O(T^{-1})$

The leading bias term is:

$$
c_1 = \sigma^2 (W'AW)^{-1} \left[\text{tr}(AF) \cdot q_0 + \text{tr}(AF^2) \cdot q_1\right]
$$

where:

- $F = \rho L + X \cdot g(\beta)$ captures the dynamic structure (with $L$ being the lag operator matrix)
- $q_0, q_1$ are functions of the data matrices
- $\text{tr}(\cdot)$ denotes the matrix trace

More specifically, for the AR(1) model with exogenous regressors:

$$
c_1 = \sigma^2 (W'AW)^{-1} \sum_{j=0}^{T-2} \text{tr}(A \, C^j) \cdot (W'A)_{(\cdot, j+1)}
$$

where $C$ is the companion matrix encoding the AR dynamics.

### Order 2: $O(N^{-1}T^{-1})$

The second-order term corrects for the interaction between the cross-sectional and time-series dimensions:

$$
c_2 = \frac{\sigma^2}{N} (W'AW)^{-1} \left[\text{tr}(B_1) \cdot h_1 + \text{tr}(B_2) \cdot h_2\right]
$$

where $B_1, B_2$ are functions of $A$, the companion matrix, and the data.

### Order 3: $O(N^{-1}T^{-2})$

Bun & Kiviet (2003) derive the third-order term, which provides a further refinement:

$$
c_3 = \frac{\sigma^2}{NT} (W'AW)^{-1} \left[\text{tr}(G_1) \cdot r_1 + \text{tr}(G_2) \cdot r_2 + \text{tr}(G_3) \cdot r_3\right]
$$

!!! info "Practical Guidance on Bias Order"
    - **Order 1** (`bias_order=1`): Simplest; captures the dominant bias from the within transformation. Sufficient when T is moderate.
    - **Order 2** (`bias_order=2`): Recommended default. Adds the $O(N^{-1}T^{-1})$ correction, which matters when both N and T are small.
    - **Order 3** (`bias_order=3`): Bun & Kiviet (2003) refinement. Gains over order 2 are typically small; use when maximum accuracy is needed.

### Iterated Bias Correction

Since the bias formulas depend on the unknown parameters $\delta$ and $\sigma^2$, LSDVC uses an iterative procedure:

1. **Initialize**: Obtain a consistent estimate $\hat{\rho}_0$ from an initial estimator (Anderson-Hsiao, Arellano-Bond, or Blundell-Bond)
2. **LSDV**: Compute $\hat{\delta}_{LSDV}$ and $\hat{\sigma}^2$ from Fixed Effects
3. **Correct**: $\hat{\delta}^{(k+1)} = \hat{\delta}_{LSDV} - \widehat{\text{Bias}}(\hat{\delta}^{(k)}, \hat{\sigma}^2)$
4. **Iterate**: Repeat step 3 until $\|\hat{\delta}^{(k+1)} - \hat{\delta}^{(k)}\| < \text{tol}$

The initial consistent estimator is needed only to start the iteration. The final LSDVC estimates are typically robust to the choice of initial estimator, though convergence may differ.

## Bootstrap Inference

### Why Bootstrap?

The analytical bias correction produces corrected point estimates, but the asymptotic distribution of $\hat{\delta}_{LSDVC}$ is not straightforward. Standard errors from the LSDV variance-covariance matrix are not valid for the corrected estimator. Bruno (2005) proposes **parametric bootstrap** for inference.

### Parametric Bootstrap Procedure

For $b = 1, \ldots, B$ bootstrap replications:

1. **Generate** errors: $\varepsilon^*_{it} \sim N(0, \hat{\sigma}^2)$
2. **Simulate** panel data from the estimated DGP:

    $$
    y^*_{it} = \hat{\rho} \, y^*_{i,t-1} + X_{it}'\hat{\beta} + \hat{\alpha}_i + \varepsilon^*_{it}
    $$

    using the original $X_{it}$ and estimated fixed effects $\hat{\alpha}_i$

3. **Re-estimate** LSDVC on the simulated panel → $\hat{\delta}^*_b$

The bootstrap distribution $\{\hat{\delta}^*_1, \ldots, \hat{\delta}^*_B\}$ provides:

- **Standard errors**: $\text{SE}(\hat{\delta}_k) = \text{sd}(\hat{\delta}^*_{1,k}, \ldots, \hat{\delta}^*_{B,k})$
- **Confidence intervals**: Percentile method using the $\alpha/2$ and $1-\alpha/2$ quantiles
- **z-statistics and p-values**: Using bootstrap standard errors with normal approximation

!!! tip "Number of Bootstrap Replications"
    - 200-500 replications are sufficient for standard errors
    - 1000+ replications recommended for accurate confidence interval endpoints
    - Use `seed` parameter for reproducibility

## Comparison with GMM

Both LSDVC and GMM address the Nickell bias, but they take fundamentally different approaches:

| Feature | LSDVC | Difference GMM | System GMM |
|---------|-------|---------------|------------|
| **Approach** | Analytical bias correction | Moment conditions + instruments | Moment conditions + instruments |
| **Asymptotics** | Fixed N, fixed T | N → ∞, fixed T | N → ∞, fixed T |
| **Small N** | Works well (N < 100) | Unreliable | Unreliable |
| **Instruments** | Not needed (uses initial estimator) | Lagged levels | Lagged levels + differences |
| **Exogenous regressors** | Assumed strictly exogenous | Can handle predetermined/endogenous | Can handle predetermined/endogenous |
| **Inference** | Bootstrap | Asymptotic (Windmeijer-corrected) | Asymptotic (Windmeijer-corrected) |
| **Overidentification test** | Not applicable | Hansen J / Sargan | Hansen J / Sargan |
| **Persistent series** | Works with any $\rho$ | Weak instruments for high $\rho$ | Better for high $\rho$ |

### When to Use Each

- **LSDVC**: Small to moderate N (N < 100), strictly exogenous regressors, preference for simplicity
- **Difference GMM**: Large N, predetermined or endogenous regressors, need for overidentification tests
- **System GMM**: Large N, highly persistent processes, near-unit-root data

!!! warning "Key limitation of LSDVC"
    LSDVC assumes that all regressors in $X_{it}$ are **strictly exogenous** — they cannot be correlated with past, present, or future errors. If you have predetermined or endogenous regressors, use GMM instead.

## Extension to Unbalanced Panels

Bruno (2005) extends the Kiviet bias correction to unbalanced panels. The key modifications are:

- Entity-specific time dimensions $T_i$ replace the common $T$
- The within-transformation matrix $A$ accounts for varying $T_i$
- Bias formulas sum over entity-specific contributions weighted by $T_i$

The PanelBox implementation handles unbalanced panels automatically.

## References

1. **Nickell, S.** (1981). "Biases in Dynamic Models with Fixed Effects." *Econometrica*, 49(6), 1417-1426.

2. **Anderson, T. W., & Hsiao, C.** (1982). "Formulation and Estimation of Dynamic Models Using Panel Data." *Journal of Econometrics*, 18(1), 47-82.

3. **Kiviet, J. F.** (1995). "On Bias, Inconsistency, and Efficiency of Various Estimators in Dynamic Panel Data Models." *Journal of Econometrics*, 68(1), 53-78.

4. **Bun, M. J. G., & Kiviet, J. F.** (2003). "On the Diminishing Returns of Higher-Order Terms in Asymptotic Expansions of Bias." *Economics Letters*, 79(2), 145-152.

5. **Bruno, G. S. F.** (2005). "Approximating the Bias of the LSDV Estimator for Dynamic Unbalanced Panel Data Models." *Economics Letters*, 87(3), 361-366.

6. **Arellano, M., & Bond, S.** (1991). "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297.

7. **Blundell, R., & Bond, S.** (1998). "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115-143.
