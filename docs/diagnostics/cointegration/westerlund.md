---
title: "Westerlund Test"
description: "Westerlund error correction-based panel cointegration test in PanelBox — four test statistics with bootstrap p-values for cross-sectional dependence correction."
---

# Westerlund Test (ECM-Based Panel Cointegration)

!!! info "Quick Reference"
    **Function:** `panelbox.diagnostics.cointegration.westerlund.westerlund_test()`
    **H₀:** No cointegration ($\alpha_i = 0$ for all $i$, no error correction)
    **H₁:** Cointegration exists ($\alpha_i < 0$ for some/all $i$, error correction active)
    **Stata equivalent:** `xtwest y x1 x2, bootstraps(200)`
    **R equivalent:** Custom implementation (adapted from `urca::ca.jo()`)

## What It Tests

The Westerlund (2007) test is an **error correction model (ECM)-based** cointegration test. Unlike residual-based tests (Pedroni, Kao) that test for unit roots in regression residuals, the Westerlund test directly examines whether the **error correction mechanism** is active -- whether the system adjusts back to equilibrium after deviations.

The test produces **four statistics** ($G_\tau$, $G_\alpha$, $P_\tau$, $P_\alpha$) and supports **bootstrap p-values** to correct for cross-sectional dependence, making it more robust than residual-based alternatives.

## Quick Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.diagnostics.cointegration.westerlund import westerlund_test

data = load_grunfeld()

# Westerlund test with bootstrap
result = westerlund_test(
    data,
    entity_col="firm",
    time_col="year",
    y_var="invest",
    x_vars=["value", "capital"],
    method="all",
    trend="c",
    n_bootstrap=200,
    random_state=42,
)
print(result)
```

Output:

```text
Westerlund (2007) Cointegration Test Results
============================================================
Method: all
Trend: c
Lags: 1
Entities: 10, Time periods: 20
Bootstrap replications: 200

 Test  Statistic  P-value   CV 1%   CV 5%  CV 10%  Reject (5%)
   Gt     -2.456    0.025  -3.124  -2.547  -2.156           **
   Ga     -8.234    0.075  -12.45  -9.876  -8.123            *
   Pt     -4.567    0.005  -4.890  -3.456  -2.987          ***
   Pa    -12.345    0.025  -15.67 -13.456 -11.234           **

H0: No cointegration (alpha_i = 0 for all i)
***, **, * denote rejection at 1%, 5%, 10% level
```

## The Four Statistics

The Westerlund test produces four statistics testing different aspects of error correction:

| Statistic | Type | What It Tests | H₁ |
|:----------|:-----|:-------------|:---|
| $G_\tau$ | Group-mean | Mean group t-ratio of $\alpha_i$ | $\alpha_i < 0$ for some $i$ |
| $G_\alpha$ | Group-mean | Mean group normalized $\alpha_i$ | $\alpha_i < 0$ for some $i$ |
| $P_\tau$ | Panel | Panel t-ratio of pooled $\alpha$ | $\alpha < 0$ (pooled) |
| $P_\alpha$ | Panel | Panel normalized pooled $\alpha$ | $\alpha < 0$ (pooled) |

**Group-mean statistics** ($G_\tau$, $G_\alpha$) allow heterogeneous error correction speeds -- different entities can adjust at different rates. Rejection means at least some entities exhibit cointegration.

**Panel statistics** ($P_\tau$, $P_\alpha$) assume a common error correction speed. Rejection means the panel as a whole exhibits cointegration.

## Interpretation

### Using Bootstrap P-values

```python
# Check rejection at 5% significance
rejections = result.reject_at(alpha=0.05)
for test, rejected in rejections.items():
    print(f"{test}: {'Reject H0' if rejected else 'Fail to reject'}")
```

### Summary Table

```python
# Get formatted summary as DataFrame
summary_df = result.summary()
print(summary_df.to_string(index=False))
```

### Interpreting the Four Statistics Together

| $G$ statistics | $P$ statistics | Interpretation |
|:--------------|:--------------|:---------------|
| Both reject | Both reject | Strong evidence of cointegration for all entities |
| Both reject | Neither rejects | Cointegration in some (not all) entities |
| Neither rejects | Both reject | Pooled cointegration, but not entity-specific |
| Neither rejects | Neither rejects | No evidence of cointegration |

## Mathematical Details

### Error Correction Model

For each entity $i$, the ECM is:

$$\Delta y_{it} = \alpha_i d_t + \alpha_i(y_{i,t-1} - \beta_i' x_{i,t-1}) + \sum_{j=1}^{p} \gamma_{ij} \Delta y_{i,t-j} + \sum_{j=0}^{p} \delta_{ij} \Delta x_{i,t-j} + \varepsilon_{it}$$

where:

- $\alpha_i$ is the **error correction parameter** (speed of adjustment)
- $\beta_i$ is the cointegrating vector
- $d_t$ includes deterministic terms
- $\gamma_{ij}$, $\delta_{ij}$ capture short-run dynamics

### Hypotheses

$$H_0: \alpha_i = 0 \text{ for all } i \quad \text{(no error correction} \to \text{no cointegration)}$$

$$H_1: \alpha_i < 0 \text{ for some/all } i \quad \text{(error correction active} \to \text{cointegration)}$$

### Test Statistics

**Group-mean statistics:**

$$G_\tau = \frac{1}{N} \sum_{i=1}^{N} \frac{\hat{\alpha}_i}{SE(\hat{\alpha}_i)}, \qquad G_\alpha = \frac{1}{N} \sum_{i=1}^{N} \frac{T \hat{\alpha}_i}{\hat{\alpha}_i^{(1)}}$$

**Panel statistics:**

$$P_\tau = \frac{\bar{\alpha}}{SE(\bar{\alpha})}, \qquad P_\alpha = T \bar{\alpha}$$

where $\bar{\alpha}$ is the pooled error correction coefficient.

### Bootstrap P-values

The bootstrap procedure generates data under H₀ ($\alpha_i = 0$, no cointegration) by:

1. Estimating the ECM and collecting residuals
2. Resampling residuals with replacement
3. Generating new $y_{it}$ as a random walk (no error correction)
4. Re-computing test statistics on bootstrap data
5. Comparing observed statistics to bootstrap distribution

Bootstrap p-values are more reliable than asymptotic p-values when cross-sectional dependence is present.

## Configuration Options

```python
westerlund_test(
    data,                   # pd.DataFrame: Panel data in long format
    entity_col,             # str: Entity identifier column
    time_col,               # str: Time identifier column
    y_var,                  # str: Dependent variable name
    x_vars,                 # str or list[str]: Independent variable name(s)
    method='all',           # str: 'all', 'Gt', 'Ga', 'Pt', or 'Pa'
    trend='c',              # str: 'n' (none), 'c' (constant), 'ct' (constant + trend)
    lags='auto',            # int or 'auto': Number of lags ('auto' = AIC selection)
    max_lags=4,             # int: Maximum lags when lags='auto'
    lag_criterion='aic',    # str: 'aic' or 'bic' for lag selection
    n_bootstrap=1000,       # int: Number of bootstrap replications
    random_state=None,      # int or None: Random seed for reproducibility
    use_bootstrap=True,     # bool: Whether to compute bootstrap p-values
)
```

!!! note "Parameter Order"
    Unlike Pedroni and Kao, the Westerlund function takes `entity_col` and `time_col` **before** `y_var` and `x_vars`. The dependent variable parameter is named `y_var` (not `dependent`), and independent variables use `x_vars` (not `independents`).

### Bootstrap Options

=== "With bootstrap (recommended)"

    ```python
    # Bootstrap p-values for robust inference
    result = westerlund_test(
        data, "firm", "year", "invest", ["value"],
        n_bootstrap=500, random_state=42
    )
    ```

=== "Without bootstrap (fast)"

    ```python
    # Asymptotic p-values (faster but less robust to CD)
    result = westerlund_test(
        data, "firm", "year", "invest", ["value"],
        use_bootstrap=False
    )
    ```

### Lag Selection

=== "Automatic (AIC)"

    ```python
    result = westerlund_test(
        data, "firm", "year", "invest", ["value"],
        lags="auto", max_lags=4, lag_criterion="aic"
    )
    ```

=== "Automatic (BIC)"

    ```python
    result = westerlund_test(
        data, "firm", "year", "invest", ["value"],
        lags="auto", lag_criterion="bic"
    )
    ```

=== "Fixed lags"

    ```python
    result = westerlund_test(
        data, "firm", "year", "invest", ["value"],
        lags=2
    )
    ```

### Result Object: `WesterlundResult`

| Attribute | Type | Description |
|:----------|:-----|:-----------|
| `statistic` | `dict[str, float]` | Test statistics (`{Gt, Ga, Pt, Pa}`) |
| `pvalue` | `dict[str, float]` | P-values (bootstrap or asymptotic) |
| `critical_values` | `dict` | Critical values at 1%, 5%, 10% per test |
| `method` | `str` | Method used (`'all'`, `'Gt'`, etc.) |
| `trend` | `str` | Trend specification |
| `lags` | `int` | Number of lags used |
| `n_bootstrap` | `int` | Number of bootstrap replications |
| `n_entities` | `int` | Number of cross-sectional units |
| `n_time` | `int` | Average number of time periods |

**Methods:**

- `result.reject_at(alpha=0.05)` -- Returns rejection decisions for all tests
- `result.reject_at(alpha=0.05, test="Gt")` -- Returns rejection for a specific test
- `result.summary()` -- Returns formatted `pd.DataFrame` summary

## When to Use

**Use Westerlund when:**

- You suspect **cross-sectional dependence** (use bootstrap p-values)
- You want a **more powerful** test than residual-based alternatives
- You are interested in the **error correction mechanism** directly
- You need both group-mean and panel perspectives

**Advantages over Pedroni/Kao:**

- More powerful (tests ECM directly, not residuals indirectly)
- Bootstrap p-values handle cross-sectional dependence
- Automatic lag selection
- Four complementary statistics

**Limitations:**

- **Computationally intensive** with bootstrap (especially for large panels)
- Requires choosing `max_lags` and `n_bootstrap`
- Bootstrap with large $N \times T$ can take several minutes

!!! tip "Performance Tip"
    For exploratory analysis, use `n_bootstrap=200`. For publication-quality results, use `n_bootstrap=1000` or more. Set `random_state` for reproducibility.

## Common Pitfalls

!!! warning "Computation Time"
    With large panels (N > 50, T > 100) and high bootstrap replications (> 1000), the test can take considerable time. Start with `n_bootstrap=200` and increase if needed. The function will warn if computation is expected to be slow.

!!! warning "I(1) Prerequisite"
    All variables must be individually I(1). Run [unit root tests](../unit-root/index.md) before applying cointegration tests.

!!! warning "Lag Specification"
    Too few lags leads to size distortion (over-rejection). Too many lags leads to power loss. Use automatic selection (`lags='auto'`) unless you have a specific reason for fixed lags.

!!! warning "Interpreting Group vs. Panel Statistics"
    $G$ statistics reject when **some** entities are cointegrated. $P$ statistics reject when **all** entities (pooled) are cointegrated. Different rejections have different implications for the panel.

## See Also

- [Cointegration Tests Overview](index.md) -- Comparison of all three tests
- [Pedroni Test](pedroni.md) -- Residual-based heterogeneous test (7 statistics)
- [Kao Test](kao.md) -- Residual-based homogeneous test (simpler)
- [Unit Root Tests](../unit-root/index.md) -- Prerequisite: testing for I(1)

## References

- Westerlund, J. (2007). "Testing for error correction in panel data." *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.
- Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and error correction: representation, estimation, and testing." *Econometrica*, 55(2), 251-276.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapter 12.
