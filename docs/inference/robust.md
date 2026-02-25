---
title: "Robust Standard Errors (HC0-HC3)"
description: "Heteroskedasticity-consistent standard errors using White's sandwich estimator and finite-sample improvements in PanelBox."
---

# Robust Standard Errors (HC0-HC3)

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.RobustStandardErrors`
    **Convenience:** `panelbox.standard_errors.robust_covariance()`
    **Model integration:** `model.fit(cov_type="robust")` or `model.fit(cov_type="hc1")`
    **Stata equivalent:** `vce(robust)` (= HC1), `vce(hc2)`, `vce(hc3)`
    **R equivalent:** `sandwich::vcovHC(type="HC1")`

## Overview

Classical OLS standard errors assume **homoskedasticity**: $\text{Var}(\varepsilon_i) = \sigma^2$ for all observations. When this assumption fails --- and it almost always does in applied work --- the resulting standard errors, t-statistics, and p-values are invalid.

**Heteroskedasticity-robust** (HC) standard errors, introduced by White (1980), provide valid inference regardless of the error variance structure. PanelBox implements four HC variants (HC0--HC3) with progressively better finite-sample properties.

## When to Use

- Errors have non-constant variance across observations
- You suspect heteroskedasticity but don't know its form
- As a **baseline robustness check** for any linear model
- When you have a cross-sectional dataset or pooled panel

!!! note "When NOT to use"
    Robust HC standard errors do **not** account for within-cluster correlation (use [clustered SE](clustered.md)) or serial correlation (use [Newey-West](newey-west.md) or [Driscoll-Kraay](driscoll-kraay.md)). In panel data, clustering is almost always preferred over simple HC.

## Quick Example

```python
from panelbox.standard_errors import RobustStandardErrors, robust_covariance

# Class-based approach
rse = RobustStandardErrors(X, resid)
result = rse.hc1()

print(f"Method: {result.method}")
print(f"Standard errors: {result.std_errors}")
print(f"n_obs: {result.n_obs}, n_params: {result.n_params}")

# Convenience function (equivalent)
result = robust_covariance(X, resid, method="HC1")
print(result.std_errors)

# Via model.fit()
from panelbox.models import PooledOLS
model = PooledOLS("y ~ x1 + x2", data, entity="firm", time="year")
results = model.fit(cov_type="robust")  # Uses HC1 by default
print(results.summary())
```

## HC Variants

All four variants use the sandwich formula $V = (X'X)^{-1} \hat{\Omega} (X'X)^{-1}$, where the **meat** $\hat{\Omega}$ differs:

| Type | Meat $\hat{\Omega}$ | Correction | Best For | Reference |
|------|---------------------|------------|----------|-----------|
| HC0 | $\sum \hat{e}_i^2 x_i x_i'$ | None | Large samples | White (1980) |
| HC1 | $\frac{n}{n-k} \sum \hat{e}_i^2 x_i x_i'$ | Degrees of freedom | General use (default) | --- |
| HC2 | $\sum \frac{\hat{e}_i^2}{1-h_{ii}} x_i x_i'$ | Leverage-adjusted | Moderate samples | --- |
| HC3 | $\sum \frac{\hat{e}_i^2}{(1-h_{ii})^2} x_i x_i'$ | Aggressive leverage | Small samples | MacKinnon & White (1985) |

### HC0: White's Original Estimator

The original White (1980) estimator. Consistent but can be biased downward in finite samples:

$$
\hat{\Omega}_{HC0} = \sum_{i=1}^{n} \hat{e}_i^2 x_i x_i'
$$

```python
result = rse.hc0()
```

### HC1: Degrees-of-Freedom Correction

Applies a simple $n/(n-k)$ scaling to HC0. This is the **default** in PanelBox and Stata's `vce(robust)`:

$$
\hat{\Omega}_{HC1} = \frac{n}{n-k} \sum_{i=1}^{n} \hat{e}_i^2 x_i x_i'
$$

```python
result = rse.hc1()
# Or equivalently:
result = rse.compute(method="HC1")
```

### HC2: Leverage-Adjusted

Divides each squared residual by $1 - h_{ii}$, where $h_{ii}$ is the **leverage** (hat value) of observation $i$:

$$
\hat{\Omega}_{HC2} = \sum_{i=1}^{n} \frac{\hat{e}_i^2}{1 - h_{ii}} x_i x_i'
$$

```python
result = rse.hc2()
print(f"Leverage values: {result.leverage}")
```

### HC3: MacKinnon-White

The most conservative variant. Squares the leverage correction, providing the best finite-sample performance:

$$
\hat{\Omega}_{HC3} = \sum_{i=1}^{n} \frac{\hat{e}_i^2}{(1 - h_{ii})^2} x_i x_i'
$$

```python
result = rse.hc3()
print(f"Leverage values: {result.leverage}")
```

## Mathematical Details

### Leverage (Hat Values)

The **leverage** $h_{ii}$ is the $i$-th diagonal element of the hat matrix:

$$
H = X(X'X)^{-1}X'
$$

Properties of leverage values:

- $0 \leq h_{ii} \leq 1$
- $\sum_{i=1}^{n} h_{ii} = k$ (number of parameters)
- Average leverage: $\bar{h} = k/n$
- High-leverage threshold: $h_{ii} > 2k/n$ or $3k/n$

Observations with high leverage have disproportionate influence on the fitted regression. HC2 and HC3 upweight these observations, compensating for the fact that their residuals are artificially small.

### Why HC2/HC3 Matter

Under homoskedasticity, $E(\hat{e}_i^2) = \sigma^2(1 - h_{ii})$. This means:

- OLS residuals **underestimate** the true error variance for high-leverage points
- HC0 and HC1 inherit this bias
- HC2 corrects exactly: $\hat{e}_i^2 / (1 - h_{ii})$ is unbiased for $\sigma^2$
- HC3 over-corrects, which gives better finite-sample size in hypothesis testing

## Configuration Options

### RobustStandardErrors Class

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Design matrix $(n \times k)$ |
| `resid` | `np.ndarray` | OLS residuals $(n,)$ |

### RobustCovarianceResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `cov_matrix` | `np.ndarray` | Robust covariance matrix $(k \times k)$ |
| `std_errors` | `np.ndarray` | Robust standard errors $(k,)$ |
| `method` | `str` | HC variant used (`'HC0'`, `'HC1'`, `'HC2'`, `'HC3'`) |
| `n_obs` | `int` | Number of observations |
| `n_params` | `int` | Number of parameters |
| `leverage` | `np.ndarray` or `None` | Leverage values (HC2, HC3 only) |

## Comparing All HC Variants

```python
from panelbox.standard_errors import RobustStandardErrors

rse = RobustStandardErrors(X, resid)

# Compute all four variants
results = {
    "HC0": rse.hc0(),
    "HC1": rse.hc1(),
    "HC2": rse.hc2(),
    "HC3": rse.hc3(),
}

# Compare standard errors
for method, result in results.items():
    print(f"{method}: {result.std_errors}")
```

!!! tip "Which HC to choose?"
    - **HC1** is the safe default for most applications (matches Stata's `robust`)
    - **HC3** is recommended for small samples ($n < 250$) or when leverage varies substantially
    - **HC2** is a middle ground with exact unbiasedness under homoskedasticity
    - **HC0** is mainly of historical interest; prefer HC1 or higher

## Diagnostics

### Detecting Heteroskedasticity

Before applying robust SEs, you can test for heteroskedasticity:

```python
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest
from panelbox.validation.heteroskedasticity.white import WhiteTest

bp_result = BreuschPaganTest(results).run(alpha=0.05)
white_result = WhiteTest(results).run(alpha=0.05)

print(f"Breusch-Pagan: p={bp_result.pvalue:.4f} -> {bp_result.conclusion}")
print(f"White test:    p={white_result.pvalue:.4f} -> {white_result.conclusion}")
```

!!! note
    Even if tests do not reject homoskedasticity, robust SEs are still valid (just slightly less efficient). Many applied researchers use robust SEs by default.

## See Also

- [Clustered Standard Errors](clustered.md) --- For within-entity or within-time correlation
- [Newey-West HAC](newey-west.md) --- For autocorrelation in time series
- [Comparison](comparison.md) --- Compare HC with other SE types
- [Inference Overview](index.md) --- Choosing the right SE type

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325.
- Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*, 54(3), 217-224.
