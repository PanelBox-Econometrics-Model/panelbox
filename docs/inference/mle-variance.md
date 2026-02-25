---
title: "MLE Variance Estimation"
description: "Sandwich estimator, delta method, and bootstrap standard errors for maximum likelihood models in PanelBox."
---

# MLE Variance Estimation

!!! info "Quick Reference"
    **Module:** `panelbox.standard_errors.mle`
    **Functions:** `sandwich_estimator()`, `cluster_robust_mle()`, `delta_method()`, `bootstrap_mle()`
    **Model integration:** `model.fit(cov_type="robust")` on MLE models
    **Stata equivalent:** `vce(robust)` on `logit`, `probit`, `poisson`
    **R equivalent:** `sandwich::sandwich()`, `car::deltaMethod()`

## Overview

Nonlinear models estimated by **Maximum Likelihood Estimation (MLE)** --- such as Logit, Probit, Poisson, and Tobit --- require specialized variance estimators. PanelBox provides three complementary approaches:

1. **Sandwich estimator** (Huber-White): Robust to distributional misspecification
2. **Delta method**: For standard errors of transformed parameters (marginal effects, odds ratios)
3. **Bootstrap**: Distribution-free inference when analytical SEs are difficult

## When to Use

| Method | Use Case |
|--------|----------|
| Classical (`method="nonrobust"`) | Correctly specified model with known distribution |
| Sandwich (`method="robust"`) | Robust to distributional misspecification |
| Cluster-robust | MLE with panel data (clustered observations) |
| Delta method | Standard errors for nonlinear functions of parameters |
| Bootstrap | Complex models where analytical SEs are unavailable |

## The Sandwich Estimator

### Classical MLE Variance

Under correct specification, the MLE variance is the inverse of the **information matrix**:

$$
V_{\text{classical}} = -H^{-1} = \left[ -\sum_{i=1}^{n} \frac{\partial^2 \ell_i}{\partial \theta \partial \theta'} \right]^{-1}
$$

where $H$ is the Hessian of the log-likelihood evaluated at the MLE.

### Robust (Sandwich) MLE Variance

The sandwich estimator is robust to misspecification of the likelihood:

$$
V_{\text{robust}} = H^{-1} S H^{-1}
$$

where $S$ is the **outer product of scores** (OPG):

$$
S = \sum_{i=1}^{n} s_i s_i', \quad s_i = \frac{\partial \ell_i}{\partial \theta}
$$

This is also known as the Huber-White or QMLE (Quasi-MLE) estimator.

### Quick Example

```python
from panelbox.standard_errors.mle import sandwich_estimator

# Classical (non-robust) MLE standard errors
result = sandwich_estimator(hessian=H, scores=scores, method="nonrobust")
print(f"Classical SE: {result.std_errors}")
print(f"Method: {result.method}")

# Robust sandwich standard errors
result = sandwich_estimator(hessian=H, scores=scores, method="robust")
print(f"Robust SE: {result.std_errors}")
print(f"n_obs: {result.n_obs}, n_params: {result.n_params}")
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hessian` | `np.ndarray` | --- | Hessian at MLE $(k \times k)$, negative definite |
| `scores` | `np.ndarray` | --- | Score vectors per observation $(n \times k)$ |
| `method` | `str` | `"robust"` | `"nonrobust"` (classical) or `"robust"` (sandwich) |

**Returns:** `MLECovarianceResult` with `.cov_matrix`, `.std_errors`, `.method`, `.n_obs`, `.n_params`

## Cluster-Robust MLE

For panel data with MLE models, cluster-robust standard errors aggregate scores within clusters before computing the sandwich:

$$
V_{\text{cluster}} = H^{-1} \left[ \sum_{g=1}^{G} \left( \sum_{i \in g} s_i \right) \left( \sum_{i \in g} s_i \right)' \right] H^{-1}
$$

### Quick Example

```python
from panelbox.standard_errors.mle import cluster_robust_mle

result = cluster_robust_mle(
    hessian=H,
    scores=scores,
    cluster_ids=entity_ids,
    df_correction=True,
)
print(f"Cluster-robust SE: {result.std_errors}")
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hessian` | `np.ndarray` | --- | Hessian at MLE $(k \times k)$ |
| `scores` | `np.ndarray` | --- | Score vectors $(n \times k)$ |
| `cluster_ids` | `np.ndarray` | --- | Cluster identifiers $(n,)$ |
| `df_correction` | `bool` | `True` | Apply $\frac{G}{G-1} \cdot \frac{N-1}{N-K}$ correction |

## The Delta Method

The delta method provides approximate standard errors for **nonlinear transformations** of estimated parameters. If $g(\hat{\theta})$ is a smooth function of the MLE $\hat{\theta}$, then:

$$
\text{Var}[g(\hat{\theta})] \approx J \cdot V(\hat{\theta}) \cdot J'
$$

where $J = \frac{\partial g}{\partial \theta'}$ is the Jacobian matrix, computed numerically via central differences.

### Use Cases

- **Odds ratios**: $g(\beta) = \exp(\beta)$ in logistic regression
- **Marginal effects**: $g(\beta) = \beta \cdot f(X\beta)$ evaluated at means
- **Elasticities**: $g(\beta) = \beta \cdot \bar{X} / \bar{Y}$
- **Predictions at specific values**: $g(\beta) = F(x_0' \beta)$

### Quick Example

```python
from panelbox.standard_errors.mle import delta_method
import numpy as np

# Parameters and covariance from logit model
params = np.array([0.5, 1.0, -0.3])
vcov = np.diag([0.01, 0.02, 0.005])

# Transform to odds ratios: g(beta) = exp(beta)
def odds_ratio(beta):
    return np.exp(beta)

vcov_or = delta_method(vcov, odds_ratio, params)
se_or = np.sqrt(np.diag(vcov_or))

print(f"Odds ratios: {odds_ratio(params)}")
print(f"SE (delta method): {se_or}")
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vcov` | `np.ndarray` | --- | Covariance matrix of parameters $(k \times k)$ |
| `transform_func` | `callable` | --- | Function $g: \mathbb{R}^k \to \mathbb{R}^m$ |
| `params` | `np.ndarray` | --- | Parameter estimates $(k,)$ |
| `epsilon` | `float` | `1e-7` | Step size for numerical Jacobian |

**Returns:** `np.ndarray` --- Covariance matrix of transformed parameters $(m \times m)$

## Bootstrap MLE

When analytical standard errors are difficult to derive or the delta method approximation is poor, **bootstrap** provides a distribution-free alternative:

1. Resample observations (or clusters) with replacement
2. Re-estimate the model on each bootstrap sample
3. Compute the sample covariance of bootstrap estimates

### Quick Example

```python
from panelbox.standard_errors.mle import bootstrap_mle

def estimate_logit(y, X):
    from scipy.optimize import minimize
    def neg_ll(beta):
        eta = X @ beta
        return -np.sum(y * eta - np.log1p(np.exp(eta)))
    result = minimize(neg_ll, np.zeros(X.shape[1]))
    return result.x

# Standard bootstrap
result = bootstrap_mle(
    estimate_func=estimate_logit,
    y=y, X=X,
    n_bootstrap=999,
    seed=42,
)
print(f"Bootstrap SE: {result.std_errors}")

# Cluster bootstrap
result = bootstrap_mle(
    estimate_func=estimate_logit,
    y=y, X=X,
    n_bootstrap=999,
    cluster_ids=entity_ids,
    seed=42,
)
print(f"Cluster bootstrap SE: {result.std_errors}")
```

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimate_func` | `callable` | --- | Function `(y, X) -> params` |
| `y` | `np.ndarray` | --- | Dependent variable $(n,)$ |
| `X` | `np.ndarray` | --- | Design matrix $(n \times k)$ |
| `n_bootstrap` | `int` | `999` | Number of bootstrap replications |
| `cluster_ids` | `np.ndarray` or `None` | `None` | Cluster IDs for cluster bootstrap |
| `seed` | `int` | `42` | Random seed |

**Returns:** `MLECovarianceResult` with `.cov_matrix`, `.std_errors`, `.method="bootstrap"`

### Bootstrap vs Analytical SE

| Feature | Sandwich | Bootstrap |
|---------|----------|-----------|
| Speed | Fast (closed-form) | Slow ($B$ re-estimations) |
| Distributional assumptions | Asymptotic normality | None |
| Small sample | May be imprecise | Better coverage |
| Implementation | Requires Hessian + scores | Requires re-estimation function |
| Cluster-robust | Yes | Yes (cluster bootstrap) |

## MLECovarianceResult

All MLE variance functions return an `MLECovarianceResult` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `cov_matrix` | `np.ndarray` | Covariance matrix $(k \times k)$ |
| `std_errors` | `np.ndarray` | Standard errors $(k,)$ |
| `method` | `str` | `"nonrobust"`, `"robust"`, `"cluster"`, or `"bootstrap"` |
| `n_obs` | `int` | Number of observations |
| `n_params` | `int` | Number of parameters |

## Common Pitfalls

!!! warning "Pitfall 1: Hessian sign convention"
    The `hessian` parameter should be the **actual Hessian** of the log-likelihood (negative definite at a maximum). PanelBox internally negates it: $-H^{-1}$. Do not pre-negate it.

!!! warning "Pitfall 2: Summed vs per-observation scores"
    The `scores` parameter must contain **per-observation** score vectors $(n \times k)$, not the summed gradient. Each row $s_i$ is $\partial \ell_i / \partial \theta$.

!!! warning "Pitfall 3: Delta method with highly nonlinear transformations"
    The delta method is a **first-order** approximation. For highly nonlinear transformations or parameters near boundary values, bootstrap may be more reliable.

!!! warning "Pitfall 4: Too few bootstrap replications"
    Use at least $B = 999$ for reliable standard errors and $B = 9999$ for confidence intervals. PanelBox warns if more than 50% of replications fail.

## See Also

- [Robust (HC0-HC3)](robust.md) --- Robust SE for linear models
- [Clustered](clustered.md) --- Clustered SE for linear models
- [Comparison](comparison.md) --- Compare SE methods
- [Inference Overview](index.md) --- Choosing the right SE type

## References

- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica*, 50(1), 1-25.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
