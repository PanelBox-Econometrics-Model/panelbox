---
title: "Breusch-Pagan Test"
description: "Breusch-Pagan LM test for heteroskedasticity in panel data regression models using PanelBox."
---

# Breusch-Pagan Test

!!! info "Quick Reference"
    **Class:** `panelbox.validation.heteroskedasticity.breusch_pagan.BreuschPaganTest`
    **H₀:** $\text{Var}(\varepsilon_i) = \sigma^2$ (homoskedasticity)
    **H₁:** $\text{Var}(\varepsilon_i) = h(X_i)$ (variance depends on regressors)
    **Statistic:** LM = $nR^2_{\text{aux}}$ ~ $\chi^2(k)$
    **Stata equivalent:** `estat hettest`
    **R equivalent:** `lmtest::bptest()`

## What It Tests

The Breusch-Pagan (1979) test is a **Lagrange Multiplier (LM) test** for heteroskedasticity that checks whether the variance of the errors depends on the values of the independent variables. It regresses the squared residuals on the original regressors and tests if the resulting coefficients are jointly zero.

Unlike the [White test](white.md), the Breusch-Pagan test assumes a **specific linear functional form** for the variance function, making it a parametric test.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest

# Estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Run Breusch-Pagan test
test = BreuschPaganTest(results)
result = test.run(alpha=0.05)

print(f"LM statistic: {result.statistic:.3f}")
print(f"P-value:      {result.pvalue:.4f}")
print(f"Degrees of freedom: {result.df}")
print(result.conclusion)

# Metadata
print(f"R² (auxiliary): {result.metadata['R2_auxiliary']:.4f}")
print(f"N observations: {result.metadata['n_obs']}")
print(f"N regressors:   {result.metadata['n_regressors']}")
```

## Interpretation

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| < 0.01 | Strong rejection | Strong evidence that variance depends on regressors |
| 0.01 -- 0.05 | Rejection | Variance is not constant; use robust SE |
| 0.05 -- 0.10 | Borderline | Weak evidence; consider robust SE as precaution |
| > 0.10 | Fail to reject | No evidence of heteroskedasticity of the assumed form |

!!! note "Important Caveat"
    Failing to reject does **not** prove homoskedasticity. The Breusch-Pagan test only detects heteroskedasticity of the specific functional form $\sigma^2_i = h(X_i\gamma)$. The errors could still be heteroskedastic in a way not captured by this parametric model. For a more general test, use the [White test](white.md).

## Mathematical Details

### Auxiliary Regression

Given model residuals $\hat{e}_{it}$, the test estimates:

$$\hat{e}_{it}^2 = X_{it}\gamma + v_{it}$$

where $X_{it}$ is the design matrix from the original regression (including the constant).

### Hypotheses

$$H_0: \gamma = 0 \quad \text{(homoskedasticity)}$$

$$H_1: \gamma \neq 0 \quad \text{(variance depends on } X \text{)}$$

### LM Statistic

$$LM = n \times R^2_{\text{aux}} \sim \chi^2(k)$$

where:

- $n$ is the total number of observations
- $R^2_{\text{aux}}$ is the R-squared from the auxiliary regression
- $k$ is the number of regressors excluding the constant (degrees of freedom)

### Intuition

If $\gamma = 0$ (homoskedasticity), the squared residuals should not be systematically related to $X$. A high $R^2_{\text{aux}}$ indicates that $X$ explains a significant portion of the variation in $\hat{e}^2$, suggesting the error variance depends on $X$.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Significance level |

### Result Metadata

| Key | Type | Description |
|-----|------|-------------|
| `R2_auxiliary` | `float` | R-squared from auxiliary regression |
| `n_obs` | `int` | Number of observations |
| `n_regressors` | `int` | Number of regressors in design matrix |

## Diagnostics

### Comparing Breusch-Pagan and White Tests

Running both tests helps assess the nature of heteroskedasticity:

```python
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest
from panelbox.validation.heteroskedasticity.white import WhiteTest

bp_result = BreuschPaganTest(results).run()
white_result = WhiteTest(results).run(cross_terms=True)

print(f"Breusch-Pagan: LM={bp_result.statistic:.3f}, p={bp_result.pvalue:.4f}")
print(f"White test:    LM={white_result.statistic:.3f}, p={white_result.pvalue:.4f}")
```

!!! example "Reading the Comparison"
    - **Both reject**: Heteroskedasticity present, linear form is sufficient to detect it
    - **BP rejects, White does not**: The variance has a simple linear relationship with $X$; White test has lower power due to extra parameters
    - **BP does not reject, White rejects**: Heteroskedasticity exists but in a nonlinear form (squares or cross-products of $X$)
    - **Neither rejects**: No evidence of heteroskedasticity (or both lack power)

## Common Pitfalls

!!! warning "Common Pitfalls"
    1. **Design matrix required**: The test needs the original design matrix $X$. If the model does not store it, a `ValueError` is raised.
    2. **Parametric assumption**: The BP test assumes $\text{Var}(\varepsilon) = h(X\gamma)$. It misses heteroskedasticity that depends on nonlinear functions of $X$ or omitted variables. Use the [White test](white.md) for a more general alternative.
    3. **Constant column detection**: The test automatically detects and adjusts for a constant column in the design matrix. Degrees of freedom are $k - 1$ if a constant is present, $k$ otherwise.
    4. **Non-negative LM**: The LM statistic is clamped to be non-negative ($\geq 0$). In rare numerical cases where $R^2$ is slightly negative, the statistic is set to zero.
    5. **Large samples**: With many observations, even trivially small heteroskedasticity becomes statistically significant. Always pair the test with practical measures of heteroskedasticity severity (e.g., the Modified Wald variance ratio).

## See Also

- [Heteroskedasticity Tests Overview](index.md) -- comparison of all tests
- [Modified Wald Test](modified-wald.md) -- groupwise heteroskedasticity for FE models
- [White Test](white.md) -- model-free heteroskedasticity test
- [Robust Standard Errors](../../inference/robust.md) -- HC0--HC3 corrections

## References

- Breusch, T. S., & Pagan, A. R. (1979). "A simple test for heteroscedasticity and random coefficient variation." *Econometrica*, 47(5), 1287-1294.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- Koenker, R. (1981). "A note on studentizing a test for heteroscedasticity." *Journal of Econometrics*, 17(1), 107-112.
