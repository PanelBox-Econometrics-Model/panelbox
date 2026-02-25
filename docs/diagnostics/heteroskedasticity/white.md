---
title: "White Test"
description: "White's general test for heteroskedasticity with no functional form assumption in panel data using PanelBox."
---

# White Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.heteroskedasticity.white.WhiteTest`
    **H₀:** Homoskedastic errors ($\text{Var}(\varepsilon_i) = \sigma^2$)
    **H₁:** Heteroskedasticity of unknown form
    **Statistic:** LM = $nR^2_{\text{aux}}$ ~ $\chi^2(q)$
    **Stata equivalent:** `estat imtest, white`
    **R equivalent:** `skedastic::white()`

## What It Tests

White's (1980) test is a **model-free test** for heteroskedasticity that does not assume any particular functional form for the variance. It regresses squared residuals on the original regressors, their squares, and their cross-products, then tests if these terms are jointly significant.

This makes it more general than the [Breusch-Pagan test](breusch-pagan.md), which only tests for linear relationships between variance and regressors.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.heteroskedasticity.white import WhiteTest

# Estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run White test with cross-product terms
test = WhiteTest(results)
result = test.run(alpha=0.05, cross_terms=True)

print(f"LM statistic:   {result.statistic:.3f}")
print(f"P-value:        {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
print(result.conclusion)

# Metadata
meta = result.metadata
print(f"R² (auxiliary):    {meta['R2_auxiliary']:.4f}")
print(f"Original regressors: {meta['n_original_regressors']}")
print(f"Auxiliary terms:     {meta['n_auxiliary_terms']}")
print(f"Cross terms:         {meta['includes_cross_terms']}")
```

## Interpretation

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Strong rejection | Strong evidence of heteroskedasticity |
| 0.01 -- 0.05 | Rejection | Heteroskedasticity present; use robust SE |
| 0.05 -- 0.10 | Borderline | Weak evidence; consider robust SE |
| > 0.10 | Fail to reject | No evidence of heteroskedasticity |

!!! tip "Cross Terms Trade-off"
    - **`cross_terms=True`** (default): More comprehensive but uses more degrees of freedom, reducing power with many regressors
    - **`cross_terms=False`**: Only squares; fewer auxiliary terms, higher power, but misses interaction-driven heteroskedasticity

## Mathematical Details

### Auxiliary Regression

Given $k$ regressors $X_1, X_2, \ldots, X_k$ (excluding the constant), the test estimates:

$$\hat{e}_{it}^2 = \alpha_0 + \sum_{j=1}^{k} \gamma_j X_{j,it} + \sum_{j=1}^{k} \delta_j X_{j,it}^2 + \sum_{j<l} \phi_{jl} X_{j,it} X_{l,it} + v_{it}$$

The auxiliary regression includes:

| Component | Count | Purpose |
|-----------|-------|---------|
| Constant | 1 | Intercept |
| Original variables $X_j$ | $k$ | Linear effects |
| Squared terms $X_j^2$ | $k$ | Quadratic variance patterns |
| Cross-products $X_j X_l$ | $\binom{k}{2}$ | Interaction effects (if `cross_terms=True`) |
| **Total** | $1 + 2k + \binom{k}{2}$ | |

### Hypotheses

$$H_0: \gamma = \delta = \phi = 0 \quad \text{(homoskedasticity)}$$

$$H_1: \text{At least one coefficient is non-zero}$$

### LM Statistic

$$LM = n \times R^2_{\text{aux}} \sim \chi^2(q)$$

where $q$ is the number of auxiliary regressors minus 1 (excluding the constant).

### Degrees of Freedom

With $k$ original regressors (excluding constant):

| Setting | $q$ (degrees of freedom) |
|---------|-------------------------|
| `cross_terms=True` | $2k + \binom{k}{2}$ |
| `cross_terms=False` | $2k$ |

For example, with $k = 3$ regressors:

- With cross terms: $q = 6 + 3 = 9$
- Without cross terms: $q = 6$

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |
| `cross_terms` | `bool` | `True` | Include cross-product $X_j \times X_l$ terms |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `R2_auxiliary` | `float` | R-squared from the auxiliary regression |
| `n_obs` | `int` | Number of observations |
| `n_original_regressors` | `int` | Regressors in original model |
| `n_auxiliary_terms` | `int` | Total terms in auxiliary regression |
| `includes_cross_terms` | `bool` | Whether cross-product terms were included |

## Diagnostics

### Comparing With and Without Cross Terms

```python
test = WhiteTest(results)

# With cross terms (full White test)
result_full = test.run(cross_terms=True)
print(f"Full White:     LM={result_full.statistic:.3f}, df={result_full.df}, "
      f"p={result_full.pvalue:.4f}")

# Without cross terms (simplified)
result_simple = test.run(cross_terms=False)
print(f"Simplified:     LM={result_simple.statistic:.3f}, df={result_simple.df}, "
      f"p={result_simple.pvalue:.4f}")
```

!!! example "Reading the Comparison"
    - **Both reject**: Heteroskedasticity is strong and present in multiple forms
    - **Full rejects, simplified does not**: Heteroskedasticity driven by interaction effects between regressors
    - **Simplified rejects, full does not**: Simple variance-regressor relationship exists, but full test loses power from extra parameters
    - **Neither rejects**: No evidence of heteroskedasticity

### Comparing with Breusch-Pagan

```python
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest

bp_result = BreuschPaganTest(results).run()
white_result = WhiteTest(results).run(cross_terms=True)

print(f"Breusch-Pagan: LM={bp_result.statistic:.3f}, p={bp_result.pvalue:.4f} "
      f"(df={bp_result.df})")
print(f"White test:    LM={white_result.statistic:.3f}, p={white_result.pvalue:.4f} "
      f"(df={white_result.df})")
```

The White test uses more degrees of freedom than Breusch-Pagan. If the sample size is small relative to the number of regressors, Breusch-Pagan may be more powerful for detecting linear heteroskedasticity.

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Design matrix required**: The test needs the original design matrix $X$. Raises `ValueError` if unavailable.
    2. **Many regressors**: With $k$ regressors and cross terms, the auxiliary regression has $O(k^2)$ parameters. For models with many regressors ($k > 10$), this severely reduces power. Use `cross_terms=False` or the [Breusch-Pagan test](breusch-pagan.md) instead.
    3. **Constant column handling**: The implementation automatically detects and removes the constant column from $X$ before computing squares and cross-products. The constant is re-added to the auxiliary regression.
    4. **Also a specification test**: The White test is simultaneously a test for heteroskedasticity and a general model misspecification test. A rejection could indicate heteroskedasticity, omitted nonlinear terms, or both.
    5. **R-squared bounds**: The auxiliary $R^2$ is computed as $1 - SSR/SST$ and may occasionally be slightly negative due to numerical precision. The implementation clips it to $[0, 1]$.

## See Also

- [Heteroskedasticity Tests Overview](index.md) -- comparison of all tests
- [Modified Wald Test](modified-wald.md) -- groupwise heteroskedasticity for FE models
- [Breusch-Pagan Test](breusch-pagan.md) -- parametric heteroskedasticity test
- [Robust Standard Errors](../../inference/robust.md) -- HC0--HC3 corrections

## References

- White, H. (1980). "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." *Econometrica*, 48(4), 817-838.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
