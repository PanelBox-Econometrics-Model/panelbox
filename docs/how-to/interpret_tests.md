# How to Interpret Diagnostic Tests

> Complete guide to understanding and interpreting panel data diagnostic tests in PanelBox.

## Overview

Panel data models come with various diagnostic tests to validate assumptions and assess model quality. This guide explains:

- **What** each test does
- **When** to use it
- **How** to interpret results
- **What to do** if tests fail

## Test Categories

PanelBox provides tests in four categories:

1. **Specification Tests**: Choose between models (Hausman, LM tests)
2. **Serial Correlation**: Test for autocorrelation (AR tests, Wooldridge)
3. **Heteroskedasticity**: Test for non-constant variance (Breusch-Pagan)
4. **GMM-Specific**: Validate instruments (Hansen J, Sargan, AR tests)

## Specification Tests

### Hausman Test (FE vs RE)

**Purpose:** Choose between Fixed Effects and Random Effects

**Hypotheses:**
- **H₀**: Random Effects is consistent (no correlation between u_i and X)
- **H₁**: Fixed Effects is preferred (correlation exists)

**Usage:**

```python
from panelbox.validation import HausmanTest

# Estimate both models
fe_results = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year").fit()
re_results = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year").fit()

# Run test
hausman = HausmanTest(fe_results, re_results)
print(hausman)
```

**Output:**
```
================================================================================
                            Hausman Test Results
================================================================================
Test statistic:                  12.456
Degrees of freedom:                   2
P-value:                         0.0020
================================================================================
H0: Random Effects model is consistent
H1: Fixed Effects model is preferred

Decision: Reject H0 (p = 0.0020)
Recommendation: Use Fixed Effects (Random Effects is inconsistent)
================================================================================
```

**Interpretation:**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p < 0.01** | Reject H₀ | Strong evidence RE inconsistent | ✅ **Use Fixed Effects** |
| **0.01 ≤ p < 0.05** | Reject H₀ | Moderate evidence against RE | ✅ **Use Fixed Effects** |
| **0.05 ≤ p < 0.10** | Borderline | Weak evidence against RE | ⚠️ Report both, prefer FE |
| **p ≥ 0.10** | Fail to reject | No evidence against RE | ✅ **Use Random Effects** |

**Common issues:**

❌ **Test fails (negative statistic)**
- Cause: Variance difference matrix not positive definite
- Solution: Use cluster-robust SEs or different variance estimator

❌ **Very small p-value (p < 0.001)**
- Interpretation: Strong rejection, definitely use FE
- Common in practice (RE assumption is restrictive)

### Breusch-Pagan LM Test (Pooled vs RE)

**Purpose:** Test for random effects (is there unobserved heterogeneity?)

**Hypotheses:**
- **H₀**: Var(u_i) = 0 (no random effects, use Pooled OLS)
- **H₁**: Var(u_i) > 0 (random effects exist)

**Usage:**

```python
from panelbox.validation import BreuschPaganLM

re_results = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year").fit()

# Test for random effects
bp_test = BreuschPaganLM(re_results)
print(bp_test)
```

**Output:**
```
Breusch-Pagan LM Test for Random Effects
LM statistic: 45.23
P-value: 0.0000
H0: Var(u_i) = 0 (no panel effect)
Decision: Reject H0 - Random effects model is appropriate
```

**Interpretation:**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p < 0.05** | Reject H₀ | Unobserved heterogeneity exists | ✅ Use RE or FE |
| **p ≥ 0.05** | Fail to reject | No evidence of heterogeneity | ✅ Pooled OLS sufficient |

**Workflow:**

1. **BP test rejects** (p < 0.05) → Heterogeneity exists → Run Hausman test
2. **BP test fails to reject** (p ≥ 0.05) → No heterogeneity → Use Pooled OLS

## Serial Correlation Tests

### Wooldridge Test (First-order Autocorrelation)

**Purpose:** Test for AR(1) serial correlation in idiosyncratic errors

**Hypotheses:**
- **H₀**: No first-order autocorrelation (Cov(ε_it, ε_i,t-1) = 0)
- **H₁**: AR(1) autocorrelation exists

**Usage:**

```python
from panelbox.validation import WooldridgeTest

fe_results = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year").fit()

# Test for serial correlation
wool_test = WooldridgeTest(fe_results)
print(wool_test)
```

**Output:**
```
Wooldridge Test for Serial Correlation
F-statistic: 23.45
P-value: 0.0001
H0: No first-order autocorrelation
Decision: Reject H0 - Serial correlation detected
```

**Interpretation:**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p < 0.05** | Reject H₀ | Serial correlation exists | ⚠️ Use robust SEs or Driscoll-Kraay |
| **p ≥ 0.05** | Fail to reject | No serial correlation | ✅ Standard SEs OK |

**What to do if test rejects:**

```python
# Option 1: Cluster-robust SEs (accounts for serial correlation)
fe_results = fe.fit(cov_type='clustered')

# Option 2: Driscoll-Kraay SEs (HAC for panels)
fe_results = fe.fit(cov_type='driscoll_kraay')

# Option 3: Newey-West SEs
fe_results = fe.fit(cov_type='newey_west', maxlags=2)
```

## Heteroskedasticity Tests

### Breusch-Pagan Test (Heteroskedasticity)

**Purpose:** Test for heteroskedasticity in panel residuals

**Hypotheses:**
- **H₀**: Homoskedasticity (constant variance)
- **H₁**: Heteroskedasticity (variance depends on X)

**Usage:**

```python
from panelbox.validation import BreuschPaganTest

fe_results = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year").fit()

bp_het = BreuschPaganTest(fe_results)
print(bp_het)
```

**Output:**
```
Breusch-Pagan Test for Heteroskedasticity
LM statistic: 67.89
P-value: 0.0000
H0: Homoskedasticity
Decision: Reject H0 - Heteroskedasticity detected
```

**Interpretation:**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p < 0.05** | Reject H₀ | Heteroskedasticity exists | ⚠️ Use robust SEs |
| **p ≥ 0.05** | Fail to reject | Homoskedasticity | ✅ Standard SEs OK |

**What to do if test rejects:**

```python
# Use heteroskedasticity-robust SEs
fe_results = fe.fit(cov_type='HC1')  # or HC2, HC3

# Or cluster-robust (handles both hetero and clustering)
fe_results = fe.fit(cov_type='clustered')
```

## GMM-Specific Tests

### Hansen J Test (Overidentification)

**Purpose:** Test validity of instruments (overidentification test)

**Hypotheses:**
- **H₀**: All instruments are valid (orthogonality conditions hold)
- **H₁**: Some instruments are invalid

**Available in:** DifferenceGMM, SystemGMM results

**Usage:**

```python
gmm_results = pb.DifferenceGMM(...).fit()

# Hansen J test (automatically computed)
print(f"Hansen J statistic: {gmm_results.hansen_j.statistic:.2f}")
print(f"P-value: {gmm_results.hansen_j.pvalue:.3f}")
```

**Interpretation:**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p > 0.25** | Strong support | Instruments appear valid | ✅ Continue with model |
| **0.10 < p ≤ 0.25** | Acceptable | Instruments likely valid | ✅ OK but check robustness |
| **0.05 < p ≤ 0.10** | Borderline | Weak evidence against instruments | ⚠️ Re-examine specification |
| **p ≤ 0.05** | Reject H₀ | Instruments likely invalid | ❌ **Do not use results** |

**Critical thresholds:**
- **p > 0.10**: Generally acceptable in applied work
- **p > 0.25**: Strong evidence of validity
- **p ≈ 1.0**: Warning! May indicate too many instruments (overfitting)

**What to do if test rejects (p < 0.10):**

1. **Use collapse=True** (if not already):
   ```python
   gmm = pb.DifferenceGMM(..., collapse=True)
   ```

2. **Reduce lags used as instruments:**
   ```python
   gmm = pb.DifferenceGMM(..., maxlags=3)  # Limit to 3 lags
   ```

3. **Remove potentially endogenous variables:**
   - Treat fewer variables as exogenous
   - Use different instruments

4. **Check for instrument proliferation:**
   ```python
   print(f"Instrument ratio: {gmm_results.instrument_ratio:.2f}")
   # Should be < 1.0 ideally, < 2.0 acceptable
   ```

### Sargan Test (Alternative to Hansen J)

**Purpose:** Same as Hansen J, but not robust to heteroskedasticity

**When to use:** One-step GMM with homoskedastic errors (rare)

**Typical use:** Hansen J is preferred (robust to heteroskedasticity)

### AR(1) and AR(2) Tests (GMM Serial Correlation)

**Purpose:** Test for autocorrelation in **differenced** errors

**Hypotheses:**
- **AR(1)**: E[Δε_it · Δε_i,t-1] = 0
- **AR(2)**: E[Δε_it · Δε_i,t-2] = 0

**Usage:**

```python
gmm_results = pb.DifferenceGMM(...).fit()

# Automatically computed
print(f"AR(1) p-value: {gmm_results.ar1_test.pvalue:.3f}")
print(f"AR(2) p-value: {gmm_results.ar2_test.pvalue:.3f}")
```

**Interpretation:**

**AR(1) Test:**

| P-value | Decision | Interpretation |
|---------|----------|----------------|
| **p < 0.05** | Reject H₀ | **Expected and OK** (mechanical due to differencing) |
| **p ≥ 0.05** | Fail to reject | Unusual but not necessarily bad |

**AR(2) Test (CRITICAL!):**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p > 0.10** | Fail to reject | No second-order autocorrelation | ✅ **Model is valid** |
| **0.05 < p ≤ 0.10** | Borderline | Weak evidence of AR(2) | ⚠️ Check robustness |
| **p ≤ 0.05** | Reject H₀ | Second-order autocorrelation | ❌ **Model invalid** |

**Why AR(2) matters:**

If AR(2) test rejects (p < 0.05):
- Moment conditions E[y_{t-2} · Δε_t] = 0 are **violated**
- Instruments starting at t-2 are **invalid**
- GMM estimates are **inconsistent**

**What to do if AR(2) rejects:**

1. **Add more lags of dependent variable:**
   ```python
   gmm = pb.DifferenceGMM(..., lags=2)  # Include y_{t-2}
   ```

2. **Use deeper lags as instruments:**
   ```python
   gmm = pb.DifferenceGMM(..., minlags=3)  # Start instruments at t-3
   ```

3. **Check for misspecification:**
   - Missing relevant variables
   - Wrong functional form

### Difference-in-Hansen Test (System GMM)

**Purpose:** Test validity of **additional** instruments in System GMM (level equations)

**Hypotheses:**
- **H₀**: Level instruments are valid
- **H₁**: Level instruments are invalid

**Usage:**

```python
gmm_sys_results = pb.SystemGMM(...).fit()

print(f"Difference-in-Hansen p-value: {gmm_sys_results.diff_hansen.pvalue:.3f}")
```

**Interpretation:**

| P-value | Decision | Interpretation | Action |
|---------|----------|----------------|--------|
| **p > 0.10** | Fail to reject | Level instruments valid | ✅ System GMM OK |
| **p ≤ 0.10** | Reject H₀ | Level instruments invalid | ❌ Use Difference GMM |

**What to do if test rejects:**

- **Fall back to Difference GMM** (doesn't use level instruments)
- Check if initial conditions assumption E[Δy_{i1} · η_i] = 0 is violated

## Complete Testing Workflow

### Static Panel Models (Pooled, FE, RE)

```python
import panelbox as pb
from panelbox.validation import BreuschPaganLM, HausmanTest, WooldridgeTest

# Step 1: Estimate all models
pooled = pb.PooledOLS("y ~ x1 + x2", data, "firm", "year").fit()
fe = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year").fit()
re = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year").fit()

# Step 2: Test for random effects
bp_lm = BreuschPaganLM(re)
print(bp_lm)

if bp_lm.pvalue < 0.05:
    print("✓ Random effects exist (use FE or RE)")

    # Step 3: Hausman test (FE vs RE)
    hausman = HausmanTest(fe, re)
    print(hausman)

    if hausman.pvalue < 0.05:
        print("✓ Use Fixed Effects")
        final_model = fe
    else:
        print("✓ Use Random Effects")
        final_model = re
else:
    print("✓ No random effects (use Pooled OLS)")
    final_model = pooled

# Step 4: Test for serial correlation
wool = WooldridgeTest(final_model)
print(wool)

if wool.pvalue < 0.05:
    print("⚠ Serial correlation detected - use Driscoll-Kraay SEs")
    if isinstance(final_model, pb.FixedEffects):
        final_model = fe.fit(cov_type='driscoll_kraay')

# Step 5: Report final model
print("\n" + "="*80)
print("FINAL MODEL")
print("="*80)
print(final_model.summary())
```

### GMM Models

```python
import panelbox as pb

# Estimate GMM
gmm = pb.DifferenceGMM(
    data=data,
    dep_var='y',
    lags=1,
    exog_vars=['x1', 'x2'],
    id_var='firm',
    time_var='year',
    collapse=True,
    robust=True
)

results = gmm.fit()

# Check diagnostics
print("="*80)
print("GMM DIAGNOSTIC TESTS")
print("="*80)

# 1. Hansen J test
hansen_ok = results.hansen_j.pvalue > 0.10
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f} {'✓' if hansen_ok else '✗'}")

# 2. AR(2) test
ar2_ok = results.ar2_test.pvalue > 0.10
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f} {'✓' if ar2_ok else '✗'}")

# 3. Instrument count
inst_ok = results.instrument_ratio < 2.0
print(f"Instrument ratio: {results.instrument_ratio:.2f} {'✓' if inst_ok else '✗'}")

# Decision
if hansen_ok and ar2_ok and inst_ok:
    print("\n✓✓✓ All tests pass - GMM estimates are valid")
    print(results.summary())
else:
    print("\n✗✗✗ Some tests fail - reconsider specification")
    if not hansen_ok:
        print("  - Hansen J failed: Instruments may be invalid")
    if not ar2_ok:
        print("  - AR(2) failed: Moment conditions violated")
    if not inst_ok:
        print("  - Too many instruments: Use collapse or reduce lags")
```

## Quick Reference Table

| Test | What it tests | Good result | Bad result | Fix |
|------|---------------|-------------|------------|-----|
| **Hausman** | FE vs RE | p ≥ 0.05 (use RE) | p < 0.05 (use FE) | Switch to FE |
| **BP LM** | Random effects exist | p < 0.05 (yes) | p ≥ 0.05 (no) | Use Pooled OLS |
| **Wooldridge** | Serial correlation | p ≥ 0.05 (none) | p < 0.05 (exists) | Robust SEs |
| **BP Hetero** | Heteroskedasticity | p ≥ 0.05 (homo) | p < 0.05 (hetero) | Robust SEs |
| **Hansen J** | GMM instruments | p > 0.10 (valid) | p ≤ 0.10 (invalid) | Collapse, reduce lags |
| **AR(1)** | GMM serial corr | p < 0.05 (expected) | - | - |
| **AR(2)** | GMM serial corr | p > 0.10 (good) | p ≤ 0.10 (bad) | Add lags, respecify |
| **Diff-Hansen** | System GMM levels | p > 0.10 (valid) | p ≤ 0.10 (invalid) | Use Difference GMM |

## Common Questions

**Q: My Hausman test has negative statistic. What does that mean?**

A: The variance difference matrix is not positive definite. This can happen with:
- Small samples
- High correlation between FE and RE estimates
- Numerical precision issues

**Solution:** Use alternative Hausman test formulation or cluster-robust SEs

---

**Q: Hansen J p-value is 1.000. Is that good?**

A: **No!** p ≈ 1.0 usually means **too many instruments** (overfitting). The test loses power.

**Solution:** Reduce instruments with `collapse=True` and check instrument ratio < 2.0

---

**Q: AR(1) test fails to reject (p = 0.45). Is my GMM invalid?**

A: **No.** AR(1) rejection (p < 0.05) is expected but not required. Focus on AR(2).

---

**Q: All my tests reject. What do I do?**

A: This is common! Tests are meant to guide specification:
1. Serial correlation → Use robust SEs
2. Heteroskedasticity → Use robust SEs
3. Hausman rejects → Use FE (not a problem, just a choice)
4. Hansen J / AR(2) reject → **Problem!** Respecify model

---

**Q: Can I ignore test results?**

A: **Depends:**
- Hausman, BP LM, Wooldridge: Guide specification but estimates still valid with robust SEs
- **Hansen J, AR(2)**: **Cannot ignore!** If these fail, GMM estimates are inconsistent

## Next Steps

**Learn more about models:**
1. **[Tutorial 2: Static Models](../tutorials/02_static_models.md)**: FE vs RE
2. **[Tutorial 3: GMM Intro](../tutorials/03_gmm_intro.md)**: GMM diagnostics

**Deep dives:**
1. **[Guide: Fixed vs Random](../guides/fixed_vs_random.md)**: When to use each
2. **[Guide: GMM Explained](../guides/gmm_explained.md)**: How GMM works

**API Reference:**
- [Validation Tests API](../api/validation.md): Complete documentation

---

**Remember:** Tests are tools to validate assumptions, not obstacles. Use them to improve your model specification!
