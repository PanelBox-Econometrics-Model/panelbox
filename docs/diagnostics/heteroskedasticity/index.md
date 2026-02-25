---
title: "Heteroskedasticity Tests"
description: "Testing for non-constant error variance in panel data with Modified Wald, Breusch-Pagan, and White tests in PanelBox."
---

# Heteroskedasticity Tests

## What Is Heteroskedasticity?

Heteroskedasticity occurs when the variance of the error terms is not constant:

$$\text{Var}(\varepsilon_{it}) \neq \sigma^2$$

In panel data, the most common form is **groupwise heteroskedasticity**, where the error variance differs across entities:

$$\sigma_i^2 \neq \sigma_j^2 \quad \text{for some } i \neq j$$

This means some entities (e.g., large firms) may have systematically larger or smaller residual variance than others (e.g., small firms).

!!! warning "Consequences of Ignoring Heteroskedasticity"
    - OLS coefficient estimates remain **consistent** but are **inefficient** (not minimum variance)
    - Classical standard errors are **biased** (can be too small or too large)
    - t-tests, F-tests, and confidence intervals are **unreliable**
    - GLS estimators assume homoskedasticity and become suboptimal

## Available Tests

PanelBox provides three complementary tests for detecting heteroskedasticity:

| Test | H₀ | Type | Best For |
|------|-----|------|----------|
| [Modified Wald](modified-wald.md) | $\sigma_i^2 = \sigma^2$ for all $i$ | Groupwise | FE models |
| [Breusch-Pagan](breusch-pagan.md) | Homoskedasticity | LM (parametric) | General; identifies source |
| [White](white.md) | Homoskedasticity | LM (model-free) | No functional form assumed |

### When to Use Each Test

=== "Fixed Effects Models"

    Use the **Modified Wald test** as your primary diagnostic. It is specifically designed for FE models and tests whether entity-level variances differ.

    ```python
    from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest

    test = ModifiedWaldTest(results)
    result = test.run(alpha=0.05)
    ```

=== "General Models"

    Use the **White test** when you want a model-free test with no assumptions about the form of heteroskedasticity.

    ```python
    from panelbox.validation.heteroskedasticity.white import WhiteTest

    test = WhiteTest(results)
    result = test.run(alpha=0.05, cross_terms=True)
    ```

=== "Identifying Sources"

    Use the **Breusch-Pagan test** when you want to identify which regressors drive the heteroskedasticity.

    ```python
    from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest

    test = BreuschPaganTest(results)
    result = test.run(alpha=0.05)
    ```

## Recommended Workflow

```python
from panelbox import FixedEffects
from panelbox.datasets import load_grunfeld
from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest
from panelbox.validation.heteroskedasticity.white import WhiteTest

# Load data and estimate model
data = load_grunfeld()
fe = FixedEffects(data, "invest", ["value", "capital"], "firm", "year")
results = fe.fit()

# Step 1: Modified Wald test (FE-specific)
mw = ModifiedWaldTest(results)
mw_result = mw.run()
print(f"Modified Wald: chi2={mw_result.statistic:.3f}, p={mw_result.pvalue:.4f}")
print(f"  Variance ratio (max/min): {mw_result.metadata['variance_ratio']:.2f}")

# Step 2: White test (model-free)
white = WhiteTest(results)
w_result = white.run(cross_terms=True)
print(f"White test:    LM={w_result.statistic:.3f}, p={w_result.pvalue:.4f}")

# Step 3: Breusch-Pagan (parametric)
bp = BreuschPaganTest(results)
bp_result = bp.run()
print(f"Breusch-Pagan: LM={bp_result.statistic:.3f}, p={bp_result.pvalue:.4f}")

# Decision
if any(r.reject_null for r in [mw_result, w_result, bp_result]):
    print("\nHeteroskedasticity detected. Use robust standard errors:")
    results_robust = fe.fit(cov_type="robust")
    print(results_robust.summary())
```

## What to Do If Heteroskedasticity Is Detected

### Option 1: Robust Standard Errors (HC0--HC3)

The simplest correction -- adjusts SE without changing coefficient estimates:

```python
results_robust = fe.fit(cov_type="robust")    # Default HC1
results_hc3 = fe.fit(cov_type="hc3")          # HC3 (recommended for small samples)
```

### Option 2: Clustered Standard Errors

Also handles serial correlation within entities:

```python
results_cluster = fe.fit(cov_type="clustered")
```

### Option 3: Variable Transformation

If the variance is proportional to a variable (e.g., firm size):

```python
import numpy as np

# Log transformation to stabilize variance
data["log_invest"] = np.log(data["invest"])
fe_log = FixedEffects(data, "log_invest", ["value", "capital"], "firm", "year")
results_log = fe_log.fit()
```

## Interpreting Results

All heteroskedasticity tests return a `ValidationTestResult`:

```python
result.test_name       # Name of the test
result.statistic       # Test statistic (chi-squared or Wald)
result.pvalue          # p-value
result.df              # Degrees of freedom
result.reject_null     # True if H₀ rejected
result.conclusion      # Human-readable conclusion
result.metadata        # Test-specific details
```

| p-value | Decision | Action |
|---------|----------|--------|
| < 0.01 | Strong rejection | Use robust or clustered SE |
| 0.01 -- 0.05 | Rejection | Use robust SE |
| 0.05 -- 0.10 | Borderline | Consider robust SE as precaution |
| > 0.10 | Fail to reject | Standard SE likely adequate |

!!! tip "Practical Advice"
    Even when the test fails to reject, using robust standard errors is a common practice in applied work. The cost of robustness (slight efficiency loss) is small compared to the cost of incorrect inference from biased SE.

## Software Equivalents

| PanelBox | Stata | R |
|----------|-------|---|
| `ModifiedWaldTest` | `xttest3` | Custom implementation |
| `BreuschPaganTest` | `estat hettest` | `lmtest::bptest()` |
| `WhiteTest` | `estat imtest, white` | `skedastic::white()` |

## See Also

- [Serial Correlation Tests](../serial-correlation/index.md) -- testing for autocorrelation
- [Cross-Sectional Dependence Tests](../cross-sectional/index.md) -- testing for correlation across entities
- [Robust Standard Errors](../../inference/robust.md) -- HC0--HC3 standard errors
- [Clustered Standard Errors](../../inference/clustered.md) -- cluster-robust inference

## References

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson, Chapter 14.
- Breusch, T. S., & Pagan, A. R. (1979). "A simple test for heteroscedasticity and random coefficient variation." *Econometrica*, 47(5), 1287-1294.
- White, H. (1980). "A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity." *Econometrica*, 48(4), 817-838.
- Baum, C. F. (2001). "Residual diagnostics for cross-section time series regression models." *Stata Journal*, 1(1), 101-104.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press, Chapter 10.
