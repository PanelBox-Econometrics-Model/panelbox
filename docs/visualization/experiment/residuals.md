---
title: "Residual Analysis"
description: "Comprehensive residual diagnostics with interactive plots and statistical tests in PanelBox"
---

# Residual Analysis

## Overview

Residual analysis is the primary tool for checking whether a model's assumptions hold. PanelBox's `ResidualResult` extracts residuals from a fitted model, runs four diagnostic tests (Shapiro-Wilk, Jarque-Bera, Durbin-Watson, Ljung-Box), computes summary statistics, and generates interactive HTML reports with six diagnostic plots.

## Running Residual Analysis

### From PanelExperiment

```python
import panelbox as pb

data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

exp.fit_model('fixed_effects', name='fe')

# Analyze residuals
resid = exp.analyze_residuals('fe')
```

### From Model Results Directly

```python
from panelbox.experiment.results import ResidualResult

fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
results = fe.fit()

resid = ResidualResult.from_model_results(results)
```

## Diagnostic Tests

`ResidualResult` provides four diagnostic tests as properties. Each test is computed on demand and cached:

### Shapiro-Wilk Test (Normality)

```python
stat, pvalue = resid.shapiro_test
print(f"Shapiro-Wilk: W={stat:.4f}, p={pvalue:.4f}")
```

Tests the null hypothesis that residuals are drawn from a normal distribution. Sensitive to sample size -- for large samples, even small deviations from normality will be flagged.

### Jarque-Bera Test (Normality)

```python
stat, pvalue = resid.jarque_bera
print(f"Jarque-Bera: JB={stat:.2f}, p={pvalue:.4f}")
```

Tests normality based on sample skewness and kurtosis. More appropriate for large samples than Shapiro-Wilk.

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

where $S$ is sample skewness and $K$ is sample kurtosis.

### Durbin-Watson Test (Serial Correlation)

```python
dw = resid.durbin_watson
print(f"Durbin-Watson: {dw:.4f}")
```

The DW statistic ranges from 0 to 4:

- $DW \approx 2$: no autocorrelation
- $DW < 2$: positive autocorrelation
- $DW > 2$: negative autocorrelation

### Ljung-Box Test (Serial Correlation)

```python
stat, pvalue = resid.ljung_box
print(f"Ljung-Box: Q={stat:.2f}, p={pvalue:.4f}")
```

Tests the null hypothesis that there is no autocorrelation up to 10 lags. More general than Durbin-Watson, as it detects higher-order serial correlation.

## Interpreting Results

| Test | Null Hypothesis | Good Result | Bad Result |
|------|----------------|-------------|------------|
| Shapiro-Wilk | Residuals are normal | $p > 0.05$ | $p \leq 0.05$ |
| Jarque-Bera | Residuals are normal | $p > 0.05$ | $p \leq 0.05$ |
| Durbin-Watson | No autocorrelation | $\approx 2.0$ (1.5--2.5) | $< 1.5$ or $> 2.5$ |
| Ljung-Box | No autocorrelation | $p > 0.05$ | $p \leq 0.05$ |

!!! warning "Interpreting Normality Tests"
    With large panel datasets ($N \times T > 500$), normality tests almost always reject. This does not necessarily invalidate inference -- the Central Limit Theorem ensures that OLS estimators are asymptotically normal regardless of residual distribution. Focus on Durbin-Watson and Ljung-Box for practical diagnostics.

## Summary Statistics

`ResidualResult` also provides descriptive statistics as properties:

```python
resid.mean       # float -- should be close to 0
resid.std        # float -- residual standard deviation
resid.skewness   # float -- should be close to 0 for normality
resid.kurtosis   # float -- excess kurtosis, should be close to 0
resid.min        # float -- minimum residual
resid.max        # float -- maximum residual
```

## Output Options

### Text Summary

```python
print(resid.summary())
```

Output:

```text
Residual Diagnostic Analysis
==================================================

Summary Statistics:
--------------------------------------------------
Observations:               200
Mean:                     0.0000
Std. Deviation:          49.7594
Min:                   -163.4139
Max:                    173.2238
Skewness:                0.2341
Kurtosis:                0.1234

Diagnostic Tests:
--------------------------------------------------
Shapiro-Wilk (Normality):        W = 0.987, p = 0.078 PASS
Jarque-Bera (Normality):         JB = 3.21, p = 0.201 PASS
Durbin-Watson (Autocorrelation): DW = 1.087 (Positive autocorrelation)
Ljung-Box (Autocorrelation):     Q = 34.56, p = 0.000 FAIL

Interpretation:
--------------------------------------------------
Residuals appear normally distributed
Autocorrelation may be present
Some model assumptions may be violated
```

### Interactive HTML Report

```python
resid.save_html(
    "residual_diagnostics.html",
    test_type="residuals",
    theme="professional",
    title="FE Residual Diagnostics"
)
```

The HTML report includes six interactive diagnostic plots:

1. **Q-Q Plot** -- normality assessment (points should fall on the 45-degree line)
2. **Residuals vs Fitted** -- linearity and homoskedasticity check (should show no pattern)
3. **Scale-Location** -- homoskedasticity check ($\sqrt{|\text{standardized residuals}|}$ vs fitted values)
4. **Residuals vs Leverage** -- influential observations (points beyond Cook's distance contours)
5. **Residual Time Series** -- temporal patterns in residuals
6. **Residual Distribution** -- histogram with kernel density overlay

### JSON Export

```python
resid.save_json("residual_results.json", indent=2)
```

### Python Dictionary

```python
data = resid.to_dict()
# Keys: 'residuals', 'fitted', 'tests', 'summary', ...
```

## Complete Example

```python
import panelbox as pb

# 1. Set up experiment
data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Fit model
exp.fit_model('fixed_effects', name='fe')

# 3. Analyze residuals
resid = exp.analyze_residuals('fe')

# 4. Check normality
shapiro_stat, shapiro_p = resid.shapiro_test
jb_stat, jb_p = resid.jarque_bera
print(f"Shapiro-Wilk: p={shapiro_p:.4f}")
print(f"Jarque-Bera:  p={jb_p:.4f}")

# 5. Check serial correlation
dw = resid.durbin_watson
lb_stat, lb_p = resid.ljung_box
print(f"Durbin-Watson: {dw:.4f}")
print(f"Ljung-Box:     p={lb_p:.4f}")

# 6. Summary statistics
print(f"Mean: {resid.mean:.4f}")
print(f"Std:  {resid.std:.4f}")
print(f"Skew: {resid.skewness:.4f}")
print(f"Kurt: {resid.kurtosis:.4f}")

# 7. Take corrective action
if dw < 1.5 or lb_p < 0.05:
    print("Serial correlation detected -- consider clustered SEs")

# 8. Generate report
resid.save_html(
    "residuals.html",
    test_type="residuals",
    theme="professional"
)

# 9. Archive
resid.save_json("residuals.json")
```

## Comparison with Other Software

| Task | PanelBox | Stata | R |
|------|----------|-------|---|
| Extract residuals | `exp.analyze_residuals('fe')` | `predict res, residuals` | `residuals(model)` |
| Shapiro-Wilk | `resid.shapiro_test` | `swilk res` | `shapiro.test(resid)` |
| Durbin-Watson | `resid.durbin_watson` | `estat dwatson` | `lmtest::dwtest()` |
| Ljung-Box | `resid.ljung_box` | `wntestq res` | `Box.test(resid)` |
| Diagnostic plots | `resid.save_html(...)` | Manual `rvfplot`, `qnorm` | `plot(model)` |

## See Also

- [Experiment Overview](index.md) -- Pattern overview and quick start
- [Workflow](fitting.md) -- Fitting and managing models
- [Validation Reports](validation.md) -- Diagnostic testing
- [Comparison Reports](comparison.md) -- Side-by-side model comparison
- [Master Reports](master-reports.md) -- Combined report generation
- [Residual Diagnostic Charts](../charts/model-diagnostics.md) -- Visualization details
