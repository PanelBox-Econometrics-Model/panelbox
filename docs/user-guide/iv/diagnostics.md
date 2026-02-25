---
title: "IV Diagnostics"
description: "Diagnostic tests for instrumental variables estimation: instrument strength, validity, and endogeneity testing."
---

# IV Diagnostics

!!! info "Quick Reference"
    **Applies to:** `PanelIV` results
    **Import:** `from panelbox.models.iv import PanelIV`
    **Stata equivalent:** `estat firststage`, `estat overid`, `estat endogenous`
    **R equivalent:** `ivreg::summary.ivreg(diagnostics=TRUE)`

## Overview

Instrumental Variables estimation requires careful diagnostic checking. Unlike OLS, where the main concerns are specification and heteroskedasticity, IV estimation can fail silently if instruments are **weak**, **invalid**, or **unnecessary**. Three fundamental questions must be addressed:

1. **Are the instruments strong enough?** (Relevance -- testable)
2. **Are the instruments valid?** (Exogeneity -- partially testable when overidentified)
3. **Is IV actually needed?** (Is the regressor truly endogenous?)

This page covers the diagnostic tools available in PanelBox for answering these questions and provides practical guidance for interpreting results.

## Three Requirements for Valid Instruments

| Requirement | Formal condition | Testable? | Key diagnostic |
|-------------|-----------------|-----------|----------------|
| **Relevance** | $\text{Cov}(Z, X) \neq 0$ | Yes | First-stage F-statistic |
| **Exogeneity** | $\text{Cov}(Z, \varepsilon) = 0$ | Partially (if overidentified) | Hansen J / Sargan test |
| **Exclusion** | $Z$ affects $y$ only through $X$ | No (maintained assumption) | Economic reasoning |

## Weak Instruments Test

### First-Stage F-Statistic

The most important IV diagnostic. The first-stage regression tests whether the instruments have sufficient predictive power for the endogenous variable.

```python
from panelbox.models.iv import PanelIV

model = PanelIV(
    formula="invest ~ capital + value | capital + lag_value + lag2_value",
    data=df,
    entity_col="firm",
    time_col="year",
    model_type="fe",
)
results = model.fit(cov_type="clustered")

# Check first-stage F for each endogenous variable
for var, fs in results.first_stage_results.items():
    f_stat = fs["f_statistic"]
    r2 = fs["rsquared"]
    print(f"Endogenous variable: {var}")
    print(f"  First-stage F-statistic: {f_stat:.2f}")
    print(f"  First-stage R-squared:   {r2:.4f}")
    if f_stat < 10:
        print("  WARNING: Weak instruments (F < 10)")
    else:
        print("  OK: Instruments are sufficiently strong")
```

### Interpreting the F-Statistic

The Stock & Yogo (2005) rule of thumb:

| F-statistic | Interpretation | Action |
|-------------|---------------|--------|
| $F > 10$ | Strong instruments | Proceed with 2SLS |
| $F \in [5, 10]$ | Borderline | Results should be interpreted with caution |
| $F < 5$ | Weak instruments | Do not trust 2SLS; find better instruments |

### Automatic Weak Instruments Warning

PanelBox automatically flags weak instruments:

```python
if results.weak_instruments:
    print("At least one endogenous variable has a weak first stage (F < 10)")
```

### Consequences of Weak Instruments

When instruments are weak:

- 2SLS estimates are **biased toward OLS** (the very estimator you are trying to improve upon)
- Standard errors are unreliable (often understated)
- Confidence intervals have poor coverage
- Test statistics have incorrect size (too many rejections)

### What to Do About Weak Instruments

1. **Find better instruments**: the best solution is stronger instruments
2. **Use more instruments**: additional relevant instruments increase F (but beware overfitting)
3. **Consider LIML**: Limited Information Maximum Likelihood is less biased than 2SLS with weak instruments
4. **Use GMM**: if the endogeneity comes from dynamic structure, Arellano-Bond GMM generates instruments internally
5. **Report reduced-form**: regress $y$ directly on $Z$ to show the instrument has an effect

## First-Stage Diagnostics

### Partial R-Squared

The first-stage $R^2$ measures how much variation in the endogenous variable is explained by all instruments plus exogenous controls:

```python
for var, fs in results.first_stage_results.items():
    print(f"{var}: R² = {fs['rsquared']:.4f}")
```

A high $R^2$ is necessary but not sufficient -- what matters is the **marginal** contribution of the excluded instruments (captured by the F-statistic).

### Multiple Endogenous Variables

With multiple endogenous variables, check each first-stage separately:

```python
# Each endogenous variable has its own first-stage regression
for var in results.endogenous_vars:
    fs = results.first_stage_results[var]
    print(f"\n--- First stage: {var} ---")
    print(f"  F-stat:    {fs['f_statistic']:.2f}")
    print(f"  R-squared: {fs['rsquared']:.4f}")
```

!!! warning "Multiple endogenous variables"
    With multiple endogenous variables, each needs its own strong first stage. A high F for one variable does not help if another has a weak first stage. In this case, Shea's partial $R^2$ (comparing the full and restricted first-stage regressions) provides a more informative diagnostic.

## Overidentification Test

When you have more instruments than endogenous variables (overidentified model), you can test whether all instruments are jointly valid.

### Hansen J / Sargan Test

- $H_0$: All instruments are exogenous ($\text{Cov}(Z_j, \varepsilon) = 0$ for all $j$)
- $H_1$: At least one instrument is endogenous

```python
# The overidentification test is available when:
#   n_excluded_instruments > n_endogenous
n_excluded = results.n_instruments - len(results.exogenous_vars)
n_endog = results.n_endogenous
print(f"Excluded instruments: {n_excluded}")
print(f"Endogenous variables: {n_endog}")

if n_excluded > n_endog:
    print("Model is overidentified -- can test instrument validity")
else:
    print("Model is exactly identified -- cannot test instrument validity")
```

### Interpreting the Test

| Test result | Interpretation | Action |
|-------------|---------------|--------|
| Fail to reject ($p > 0.05$) | Instruments appear valid | Proceed |
| Reject ($p < 0.05$) | At least one instrument may be invalid | Investigate which instruments are problematic |

!!! note "Limitations"
    The overidentification test requires that **at least one** instrument is valid. It tests validity of the "extra" instruments relative to the just-identified case. If all instruments are invalid, the test may not reject. It is a necessary but not sufficient check.

## Endogeneity Test

### Is IV Actually Needed?

Before using IV, test whether the regressor is actually endogenous. If it is not, OLS is more efficient.

### Durbin-Wu-Hausman Test

Compare OLS and 2SLS estimates. Under the null of exogeneity, both are consistent but OLS is more efficient. Under the alternative, only 2SLS is consistent.

The procedure:

1. Estimate the first stage and obtain residuals $\hat{\nu}$
2. Include $\hat{\nu}$ in the outcome equation alongside $X$
3. Test whether the coefficient on $\hat{\nu}$ is zero

If the coefficient on $\hat{\nu}$ is significant, the regressor is endogenous and IV is needed.

### Practical Comparison

```python
# Fit both OLS and IV
from panelbox import FixedEffects

# Standard FE (assumes exogeneity)
ols_model = FixedEffects("invest ~ capital + value", df, "firm", "year")
ols_results = ols_model.fit(cov_type="clustered")

# IV FE (allows endogeneity)
iv_model = PanelIV(
    "invest ~ capital + value | capital + lag_value + lag2_value",
    df, "firm", "year", model_type="fe",
)
iv_results = iv_model.fit(cov_type="clustered")

# Compare coefficients
print("Variable     | OLS      | IV       | Difference")
print("-" * 55)
for var in ols_results.params.index:
    if var in iv_results.params.index:
        ols_coef = ols_results.params[var]
        iv_coef = iv_results.params[var]
        print(f"{var:<12} | {ols_coef:>8.4f} | {iv_coef:>8.4f} | {iv_coef - ols_coef:>8.4f}")
```

If the coefficients are similar, endogeneity may not be a practical concern.

## Diagnostic Checklist

Use this checklist when reporting IV results:

### 1. Instrument Relevance

```python
# First-stage F > 10?
for var, fs in results.first_stage_results.items():
    assert fs["f_statistic"] > 10, f"Weak instruments for {var}"
print("PASS: Instruments are strong")
```

### 2. Instrument Validity (if overidentified)

```python
# Hansen J test not rejected?
n_excluded = results.n_instruments - len(results.exogenous_vars)
if n_excluded > results.n_endogenous:
    # Overidentification test available
    print("Check: overidentification test p-value > 0.05")
else:
    print("Exactly identified: cannot test overidentification")
```

### 3. Endogeneity Justified

```python
# Hausman test rejects OLS?
print("Check: OLS and IV estimates differ significantly")
print("If similar, OLS is more efficient -- use OLS")
```

### 4. Results Plausible

```python
# Coefficients have expected signs and reasonable magnitudes?
print(results.params)
print("Check: signs and magnitudes are economically plausible")
```

### Summary Table

| Step | Test | Threshold | Result |
|------|------|-----------|--------|
| 1 | First-stage F | $> 10$ | Instruments strong? |
| 2 | Hansen J (if overidentified) | $p > 0.05$ | Instruments valid? |
| 3 | Hausman (OLS vs IV) | $p < 0.05$ | Endogeneity confirmed? |
| 4 | Economic plausibility | -- | Signs and magnitudes sensible? |

If all four pass, the IV results are credible. If any fail, investigate further.

## Common Pitfalls

### Too Many Instruments

Using many instruments can:

- Overfit the first stage (high F but biased 2SLS)
- Make the overidentification test lose power
- Bias 2SLS toward OLS in finite samples

**Rule**: keep the number of instruments modest relative to the sample size.

### Instruments That Are "Too Good"

If the first-stage $R^2$ is very close to 1, the instruments may be essentially the same as the endogenous variable. This can happen with lagged values when the variable is highly persistent.

### Testing Exclusion Restrictions

The exclusion restriction -- that instruments affect $y$ only through $X$ -- **cannot be tested statistically**. It must be justified by economic reasoning. The overidentification test is a necessary but not sufficient check (it assumes at least one instrument is valid).

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| IV Diagnostics | Complete diagnostic walkthrough | [Static Models Tutorial](../../tutorials/static-models.md) |

## See Also

- [Panel IV / 2SLS](panel-iv.md) -- Main IV estimation guide
- [GMM (Arellano-Bond)](../gmm/difference-gmm.md) -- Alternative for dynamic panels with internal instruments
- [Static Models](../static-models/index.md) -- OLS, FE, RE when instruments are not needed

## References

- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In D. W. K. Andrews & J. H. Stock (Eds.), *Identification and Inference for Econometric Models* (pp. 80-108). Cambridge University Press.
- Sargan, J. D. (1958). The estimation of economic relationships using instrumental variables. *Econometrica*, 26(3), 393-415.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029-1054.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press. Chapter 4.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapters 5-6.
