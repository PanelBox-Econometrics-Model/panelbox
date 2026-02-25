---
title: "Outlier Detection"
description: "Detect outliers in panel data using univariate, multivariate, and residual-based methods with leverage diagnostics."
---

# Outlier Detection

!!! info "Quick Reference"
    **Class:** `panelbox.validation.robustness.OutlierDetector`
    **Import:** `from panelbox.validation.robustness import OutlierDetector`
    **Key methods:** `detect_outliers_residuals()`, `detect_outliers_univariate()`, `detect_outliers_multivariate()`
    **Stata equivalent:** `predict, rstudent`
    **R equivalent:** `stats::influence.measures()`

## What It Does

Outliers are observations that deviate markedly from the bulk of the data. In panel data, outliers can arise from data entry errors, structural breaks, or genuinely extreme realizations. The `OutlierDetector` class provides four detection methods covering univariate, multivariate, and regression-based approaches.

## Four Detection Methods

| Method | Approach | Default Threshold | Best For |
|--------|----------|:-----------------:|----------|
| Univariate IQR | $Q_1 - k \cdot IQR$ to $Q_3 + k \cdot IQR$ | $k = 1.5$ | Exploring individual variables |
| Univariate Z-score | $\|z_i\| > t$ where $z_i = (x_i - \bar{x})/s$ | $t = 2.5$ | Normally distributed variables |
| Multivariate | Mahalanobis distance | $3.0$ | Correlated regressors |
| Residual-based | Standardized or studentized residuals | $2.5$ | Regression outliers |

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import OutlierDetector
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Detect outliers via standardized residuals
detector = OutlierDetector(results, verbose=True)
outliers = detector.detect_outliers_residuals(method="standardized", threshold=2.5)

print(f"Outliers detected: {outliers.n_outliers} / {len(outliers.outliers)}")
print(outliers.outlier_table)  # Only flagged rows

# 4-panel diagnostic plot
detector.plot_diagnostics()
```

## Univariate Detection

### IQR Method

Flags observations outside the "fences" defined by the interquartile range:

$$\text{Lower} = Q_1 - k \cdot IQR, \quad \text{Upper} = Q_3 + k \cdot IQR$$

where $IQR = Q_3 - Q_1$ and $k$ is the threshold (default: 1.5 for "outlier", 3.0 for "extreme outlier").

```python
# IQR method on residuals (default when variable=None)
outliers_iqr = detector.detect_outliers_univariate(method="iqr", threshold=1.5)

# IQR method on a specific variable
outliers_var = detector.detect_outliers_univariate(variable="value", method="iqr", threshold=1.5)
```

### Z-Score Method

Flags observations with $|z| > t$ where $z_i = (x_i - \bar{x}) / s$:

```python
outliers_z = detector.detect_outliers_univariate(method="zscore", threshold=3.0)
```

## Multivariate Detection

Uses Mahalanobis distance, which accounts for correlations between variables:

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

```python
outliers_mahal = detector.detect_outliers_multivariate(threshold=3.0)
```

The threshold is scaled by the $\chi^2$ distribution critical value to account for dimensionality.

!!! note "Singular Covariance"
    If the covariance matrix is singular (e.g., multicollinear regressors), PanelBox automatically switches to the pseudo-inverse with a warning.

## Residual-Based Detection

The most common approach for regression diagnostics:

```python
# Standardized residuals: r / sqrt(MSE)
outliers_std = detector.detect_outliers_residuals(method="standardized", threshold=2.5)

# Studentized residuals: r / sqrt(MSE * (1 - h_ii))
outliers_stud = detector.detect_outliers_residuals(method="studentized", threshold=2.5)
```

| Type | Formula | Accounts For |
|------|---------|-------------|
| Standardized | $e_i / \sqrt{MSE}$ | Overall error scale |
| Studentized | $e_i / \sqrt{MSE \cdot (1 - h_{ii})}$ | Overall error scale + leverage |

The studentized residual adjusts for the observation's leverage ($h_{ii}$), making it more reliable for detecting outliers at high-leverage points.

!!! note "Convenience Alias"
    `detector.detect_outliers(method, threshold)` is a shortcut for `detect_outliers_residuals()` with a default threshold of 3.0.

## OutlierResults

All detection methods return an `OutlierResults` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `outliers` | `pd.DataFrame` | Full results with `entity`, `time`, `is_outlier`, `distance` columns |
| `method` | `str` | Method description string |
| `threshold` | `float` | Threshold used |
| `n_outliers` | `int` | Count of flagged observations |
| `outlier_table` | `pd.DataFrame` | Property: only rows where `is_outlier == True` |

## Leverage Points

High-leverage points are observations with unusual predictor values that can pull the regression line toward them. The default threshold is $2K/N$ where $K$ is the number of parameters and $N$ is the sample size.

```python
leverage_df = detector.detect_leverage_points(threshold=None)  # Uses 2K/N
print(leverage_df[leverage_df["is_high_leverage"]])
```

Returns a `pd.DataFrame` with columns: `entity`, `time`, `leverage`, `is_high_leverage`.

!!! note "Approximate Leverage"
    For panel models with entity fixed effects, exact leverage computation requires the full hat matrix. PanelBox provides an approximation based on the Mahalanobis distance of the design matrix.

## Diagnostic Plots

```python
detector.plot_diagnostics()
```

Produces a 4-panel figure:

1. **Residuals vs Fitted** -- Checks for heteroskedasticity and nonlinearity
2. **Normal Q-Q Plot** -- Checks normality of residuals
3. **Scale-Location** -- $\sqrt{|r_i^*|}$ vs fitted values (detects heteroskedasticity)
4. **Residual Histogram** -- Distribution of residuals with normal overlay

## What To Do With Outliers

!!! warning "Never Blindly Remove Outliers"
    Outliers are data, not noise. Removing them without justification biases your estimates.

Recommended workflow:

1. **Investigate**: Look up the flagged observations. Are they data entry errors? Extreme but real events? Structural breaks?
2. **Understand**: If they are real, consider whether the model should accommodate them (e.g., with additional controls or a different functional form).
3. **Compare**: Re-estimate with and without outliers. If results change substantially, report both sets of estimates.
4. **Winsorize**: If you must limit the impact of outliers, winsorize at the 1st/99th percentile rather than dropping observations.
5. **Robust estimation**: Consider robust regression methods that downweight influential observations.

## Comprehensive Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import OutlierDetector
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

detector = OutlierDetector(results, verbose=True)

# Multiple detection methods
residual_outliers = detector.detect_outliers_residuals(method="standardized", threshold=2.5)
mahal_outliers = detector.detect_outliers_multivariate(threshold=3.0)
leverage_points = detector.detect_leverage_points()

print(f"Residual outliers:     {residual_outliers.n_outliers}")
print(f"Multivariate outliers: {mahal_outliers.n_outliers}")
print(f"High-leverage points:  {leverage_points['is_high_leverage'].sum()}")

# Investigate flagged observations
print("\nResidual outlier details:")
print(residual_outliers.outlier_table)

# Diagnostic plots
detector.plot_diagnostics()
```

## Common Pitfalls

!!! warning "Watch Out"

    1. **Threshold choice matters**: A threshold of 2.0 flags ~5% of normally distributed data; 2.5 flags ~1.2%; 3.0 flags ~0.3%. Choose based on context, not convenience.
    2. **Masking effect**: In the presence of multiple outliers, standard methods can fail to detect some outliers because the contaminated mean and variance absorb the outlier signal. Use robust methods (multivariate Mahalanobis) when you suspect clusters of outliers.
    3. **Panel structure**: An observation may be an outlier within one entity but normal across entities. Consider entity-specific outlier detection for heterogeneous panels.
    4. **Confusing outliers with influence**: An outlier (unusual residual) is not necessarily influential (changes coefficients). Use [InfluenceDiagnostics](influence.md) to assess impact on estimates.

## See Also

- [Influence Diagnostics](influence.md) -- Cook's D, DFFITS, DFBETAS for measuring observation impact
- [Sensitivity Analysis](sensitivity.md) -- Parameter stability when entities/periods are removed
- [Robustness Overview](index.md) -- Full robustness toolkit

## References

- Cook, R. D., & Weisberg, S. (1982). *Residuals and Influence in Regression*. Chapman and Hall.
- Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust Regression and Outlier Detection*. John Wiley & Sons.
- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. John Wiley & Sons.
