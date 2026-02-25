---
title: "GMM Diagnostics"
description: "Complete guide to GMM diagnostic tests including Hansen J, Sargan, AR(1)/AR(2), Difference-in-Hansen, instrument ratio, and overfitting checks."
---

# GMM Diagnostics

!!! info "Quick Reference"
    **Results class:** `panelbox.gmm.results.GMMResults`
    **Diagnostics class:** `panelbox.gmm.diagnostics.GMMDiagnostics`
    **Overfit class:** `panelbox.gmm.GMMOverfitDiagnostic`
    **Test result class:** `panelbox.gmm.results.TestResult`

## Overview

Diagnostic tests are **mandatory** for GMM estimation. Unlike OLS or Fixed Effects, GMM results are only valid if the underlying moment conditions hold. This page provides a complete guide to interpreting every diagnostic test available in PanelBox, with decision rules, code examples, and troubleshooting guidance.

GMM estimation without proper diagnostics is meaningless. Always verify:

1. **AR(2) test** -- the most critical test for moment condition validity
2. **Hansen J test** -- overidentifying restrictions (instrument validity)
3. **Instrument ratio** -- overfitting and proliferation
4. **Coefficient bounds** -- plausibility check against OLS and FE

## The Diagnostic Checklist

!!! example "GMM Validation Checklist"
    Before accepting any GMM result, verify all of the following:

    - **AR(2) p-value > 0.10** -- Moment conditions valid
    - **Hansen J: 0.10 < p < 0.25** -- Instruments appear valid
    - **Instrument ratio < 1.0** -- No proliferation
    - **Coefficient between FE and OLS** -- Plausible estimate
    - **AR(1) rejected (p < 0.10)** -- Expected (informational)
    - **Reasonable observation count** -- Not too many dropped

    If any essential test fails, **do not trust the results**.

## AR(2) Test (Arellano-Bond)

### What It Tests

$$H_0: \text{No second-order autocorrelation in differenced errors}$$
$$H_1: \text{Second-order autocorrelation present}$$

### Why It Is Critical

The AR(2) test checks whether the original (level) errors $\varepsilon_{it}$ are serially uncorrelated. First-differencing mechanically creates MA(1) autocorrelation in $\Delta \varepsilon_{it}$, so AR(1) rejection is expected. But if $\varepsilon_{it}$ has true serial correlation, then $\Delta \varepsilon_{it}$ will show AR(2) autocorrelation, **invalidating** the key moment condition:

$$E[y_{i,t-2} \cdot \Delta \varepsilon_{it}] = 0$$

If AR(2) is rejected, the instruments are correlated with the error term, and GMM is **inconsistent**.

### Interpretation

| p-value | Conclusion | Action |
|---------|------------|--------|
| p > 0.10 | Moment conditions valid | Proceed |
| 0.05 < p < 0.10 | Borderline | Proceed with caution |
| p < 0.05 | **REJECTED** -- GMM invalid | Fix specification |

### Code

```python
ar2 = results.ar2_test
print(f"AR(2): z = {ar2.statistic:.3f}, p = {ar2.pvalue:.4f}")

if ar2.pvalue > 0.10:
    print("Moment conditions valid")
elif ar2.pvalue < 0.05:
    print("CRITICAL: Moment conditions rejected -- do not use these results")
```

### If AR(2) Rejects

1. **Add more lags** of the dependent variable:
   ```python
   model = DifferenceGMM(data=data, dep_var="y", lags=[1, 2], ...)
   ```
2. **Check for omitted variables** that might cause serial correlation
3. **Consider different model specification** (functional form, additional controls)
4. If nothing works, GMM may not be appropriate for this data

## AR(1) Test (Arellano-Bond)

### What It Tests

$$H_0: \text{No first-order autocorrelation in differenced errors}$$

### Expected Result: REJECT (p < 0.10)

First-differencing mechanically induces MA(1):

$$\text{Cov}(\Delta \varepsilon_{it}, \Delta \varepsilon_{i,t-1}) = -\text{Var}(\varepsilon_{i,t-1}) < 0$$

**Failing to reject** AR(1) is unusual and warrants investigation.

### Code

```python
ar1 = results.ar1_test
print(f"AR(1): z = {ar1.statistic:.3f}, p = {ar1.pvalue:.4f}")

if ar1.pvalue < 0.10:
    print("Expected: MA(1) structure from differencing")
else:
    print("Unexpected: Investigate data structure")
```

## Hansen J Test (Overidentification)

### What It Tests

$$H_0: \text{All instruments are valid (orthogonal to errors)}$$
$$H_1: \text{At least one instrument is invalid}$$

The test statistic is:

$$J = N \cdot g(\hat{\beta})' \hat{W}^{-1} g(\hat{\beta}) \sim \chi^2(L - K)$$

where $L$ is the number of instruments and $K$ is the number of parameters.

### Interpretation

| p-value | Assessment | Interpretation |
|---------|------------|----------------|
| p < 0.05 | **REJECT** | Instruments invalid, model misspecified |
| 0.05 < p < 0.10 | Warning | Weak evidence against instruments |
| 0.10 < p < 0.25 | **IDEAL** | Instruments appear valid |
| 0.25 < p < 0.50 | Acceptable | No strong evidence against |
| p > 0.50 | **WARNING** | Possible weak instruments or overfitting |

!!! warning "Why High p-Values Can Be Bad"
    When there are too many instruments (ratio > 1.0), the Hansen J test **loses power**. It will almost never reject even with invalid instruments. A p-value near 1.0 combined with a high instrument ratio signals overfitting, not validity.

### Code

```python
hansen = results.hansen_j
print(f"Hansen J: stat = {hansen.statistic:.3f}, p = {hansen.pvalue:.4f}, df = {hansen.df}")

if hansen.pvalue < 0.10:
    print("Instruments rejected -- check specification")
elif 0.10 <= hansen.pvalue <= 0.25:
    print("Instruments appear valid (ideal range)")
elif hansen.pvalue > 0.50:
    print("WARNING: p-value very high -- check for weak instruments or overfitting")
```

## Sargan Test

### What It Is

The Sargan test is the **non-robust** version of the Hansen J test. It is only valid under homoskedasticity.

### When to Use

- Use **Hansen J** when `robust=True` (the default and recommended setting)
- Use **Sargan** only when `robust=False` and homoskedasticity is assumed

```python
sargan = results.sargan
print(f"Sargan: stat = {sargan.statistic:.3f}, p = {sargan.pvalue:.4f}")
```

## Instrument Ratio

### Definition

$$\text{Instrument Ratio} = \frac{L}{N} = \frac{\text{Number of instruments}}{\text{Number of groups}}$$

### Interpretation

| Ratio | Assessment | Recommendation |
|-------|------------|----------------|
| < 0.5 | Good | Proceed with confidence |
| 0.5 -- 1.0 | Acceptable | Monitor other diagnostics |
| 1.0 -- 2.0 | Warning | Use `collapse=True`, reduce `gmm_max_lag` |
| > 2.0 | Problematic | Severe overfitting, results unreliable |

### Code

```python
print(f"Instruments: {results.n_instruments}")
print(f"Groups: {results.n_groups}")
print(f"Ratio: {results.instrument_ratio:.3f}")

if results.instrument_ratio > 1.0:
    print("WARNING: Too many instruments -- use collapse=True")
```

## Difference-in-Hansen Test (System GMM)

### What It Tests

$$H_0: \text{Level instruments are valid (stationarity assumption holds)}$$
$$H_1: \text{Level instruments are invalid}$$

This test compares the Hansen J statistic from the full system with the statistic from the difference-only model:

$$C = J_{system} - J_{difference} \sim \chi^2(q_{level})$$

### When Available

Only for System GMM (`SystemGMM`). Tests the additional assumption required by System GMM: the stationarity of initial conditions.

### Code

```python
if results.diff_hansen is not None:
    dh = results.diff_hansen
    print(f"Diff-in-Hansen: stat = {dh.statistic:.3f}, p = {dh.pvalue:.4f}")

    if dh.pvalue > 0.10:
        print("Level instruments valid -- System GMM appropriate")
    else:
        print("Level instruments REJECTED -- use Difference GMM instead")
```

## Windmeijer Correction

### What It Is

The Windmeijer (2005) correction adjusts two-step standard errors for the estimation error in the weighting matrix. Without this correction, two-step SEs can be **30-50% too small**.

### When Applied

Automatically applied when `two_step=True` and `robust=True` (the default). The results indicate this in the summary:

```python
print(f"Two-step: {results.two_step}")
print(f"Windmeijer corrected: {results.windmeijer_corrected}")
```

!!! tip "Always Use Windmeijer Correction"
    There is no reason to disable it. Set `robust=True` (default) to ensure correction is applied.

## Common Diagnostic Patterns

### Pattern 1: Valid Results

```text
Hansen J: p = 0.183          PASS
AR(2):    p = 0.312          PASS
AR(1):    p = 0.001          EXPECTED
Ratio:    8/140 = 0.057      GOOD
Coefficient: 0.576           Within [FE, OLS] bounds
```

All diagnostics pass. Results are reliable.

### Pattern 2: Instrument Proliferation

```text
Hansen J: p = 0.892          WARNING (too high)
AR(2):    p = 0.421          PASS
Ratio:    187/140 = 1.336    PROBLEMATIC
Coefficient: 0.698           Close to OLS (overfitting)
```

**Fix:** Use `collapse=True` and/or reduce `gmm_max_lag`.

### Pattern 3: Invalid Instruments

```text
Hansen J: p = 0.023          REJECTED
AR(2):    p = 0.156          PASS
Coefficient: 0.892           Outside bounds
```

**Fix:** Treat more variables as endogenous, remove suspect regressors, or change lag structure.

### Pattern 4: Serial Correlation

```text
Hansen J: p = 0.142          PASS
AR(2):    p = 0.018          REJECTED
```

**Fix:** Add more lags (`lags=[1, 2]`), check for omitted variables.

### Pattern 5: Weak Instruments

```text
Hansen J: p = 0.782          WARNING (too high)
AR(2):    p = 0.421          PASS
SE on L1.y: 0.456            Very large
95% CI: [-0.282, 1.506]      Very wide
```

**Fix:** Try System GMM, increase sample size, or check instrument relevance.

## Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| All coefficients zero | No valid observations | `collapse=True`, `time_dummies=False` |
| "Singular matrix" warning | Multicollinearity or insufficient variation | Remove redundant variables |
| Very large SEs | Weak instruments | Try System GMM |
| AR(2) rejected | Serial correlation in levels | Add lags: `lags=[1, 2]` |
| Hansen J rejected | Invalid instruments | Reclassify variables, reduce instruments |
| Hansen J p near 1.0 | Too many instruments | `collapse=True`, reduce `gmm_max_lag` |
| Very few observations retained | Specification too complex | `time_dummies=False`, fewer variables |
| Coefficient outside bounds | Overfitting or misspecification | Check instrument count, simplify model |

## Complete Diagnostic Code

```python
from panelbox.gmm import DifferenceGMM, GMMOverfitDiagnostic
from panelbox.datasets import load_abdata

# Estimate
data = load_abdata()
model = DifferenceGMM(
    data=data, dep_var="n", lags=1,
    id_var="id", time_var="year",
    exog_vars=["w", "k"],
    collapse=True, two_step=True, robust=True,
)
results = model.fit()

# --- Full Diagnostic Report ---
print("=" * 70)
print("GMM DIAGNOSTIC REPORT")
print("=" * 70)

# 1. AR(2) -- CRITICAL
ar2 = results.ar2_test
status = "PASS" if ar2.pvalue > 0.10 else "FAIL"
print(f"\n1. AR(2) test: z={ar2.statistic:.3f}, p={ar2.pvalue:.4f} [{status}]")

# 2. Hansen J
hansen = results.hansen_j
if hansen.pvalue < 0.10:
    status = "FAIL"
elif 0.10 <= hansen.pvalue <= 0.25:
    status = "IDEAL"
elif hansen.pvalue > 0.50:
    status = "WARNING"
else:
    status = "PASS"
print(f"2. Hansen J: stat={hansen.statistic:.3f}, p={hansen.pvalue:.4f} [{status}]")

# 3. Instrument ratio
ratio = results.instrument_ratio
status = "GOOD" if ratio < 1.0 else "WARNING"
print(f"3. Instrument ratio: {results.n_instruments}/{results.n_groups} = {ratio:.3f} [{status}]")

# 4. AR(1)
ar1 = results.ar1_test
status = "EXPECTED" if ar1.pvalue < 0.10 else "UNEXPECTED"
print(f"4. AR(1) test: z={ar1.statistic:.3f}, p={ar1.pvalue:.4f} [{status}]")

# 5. Overfitting diagnostics
diag = GMMOverfitDiagnostic(model, results)
bounds = diag.coefficient_bounds_test()
print(f"5. Bounds: OLS={bounds['ols_coef']:.4f}, GMM={bounds['gmm_coef']:.4f}, FE={bounds['fe_coef']:.4f}")
print(f"   Within bounds: {bounds['within_bounds']} [{bounds['signal']}]")

print("\n" + "=" * 70)
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Complete GMM Guide | End-to-end workflow with diagnostics | [Complete Guide](complete-guide.md) |
| Instruments | Controlling instrument count | [Instruments](instruments.md) |

## See Also

- [Difference GMM](difference-gmm.md) -- Arellano-Bond estimator
- [System GMM](system-gmm.md) -- Blundell-Bond estimator with Diff-in-Hansen
- [Instruments](instruments.md) -- Instrument selection and overfitting
- [Complete Guide](complete-guide.md) -- Applied tutorial with diagnostics

## References

1. Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297.
2. Hansen, L. P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029-1054.
3. Roodman, D. (2009). "How to do xtabond2: An Introduction to Difference and System GMM in Stata." *The Stata Journal*, 9(1), 86-136.
4. Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics*, 126(1), 25-51.
5. Stock, J. H., & Yogo, M. (2005). "Testing for Weak Instruments in Linear IV Regression." In *Identification and Inference for Econometric Models*.
6. Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects." *Econometrica*, 49(6), 1417-1426.
