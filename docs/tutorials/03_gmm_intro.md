# Tutorial 3: Introduction to GMM for Panel Data

> Learn when and how to use Generalized Method of Moments (GMM) for dynamic panel models.

## What You'll Learn

In this tutorial, you will:

- Understand why standard panel models fail with dynamics
- Learn the difference between Difference GMM and System GMM
- Estimate your first GMM model
- Interpret diagnostic tests (Hansen J, AR tests)
- Understand instrument selection and collapse
- Avoid common pitfalls

## Prerequisites

This tutorial assumes you've completed:
- [Tutorial 1: Getting Started](01_getting_started.md)
- [Tutorial 2: Static Panel Models](02_static_models.md)

You should understand:
- Fixed Effects estimation
- Panel data structure
- Basic econometric concepts (endogeneity, instruments)

## The Dynamic Panel Problem

### Why Fixed Effects Fails

Consider a **dynamic panel model** where current investment depends on past investment:

```
invest_it = γ·invest_{i,t-1} + β₁·value_it + β₂·capital_it + α_i + ε_it
```

**Problem:** Using Fixed Effects on this model produces **biased estimates** (Nickell bias).

**Why?** The within transformation creates correlation:

```
(invest_it - invest̄_i) = γ·(invest_{i,t-1} - invest̄_i) + ... + (ε_it - ε̄_i)
```

Even if `ε_it` is uncorrelated with `invest_{i,t-1}`, the demeaned error `(ε_it - ε̄_i)` **is** correlated with the demeaned lag `(invest_{i,t-1} - invest̄_i)` because `invest̄_i` includes all periods including period t.

**Magnitude:** Bias is O(1/T), so severe for short panels (T < 10).

### When Do You Need GMM?

✅ **Use GMM when:**
- Your model includes **lagged dependent variable** (y_{t-1})
- You have **short panels** (small T, large N)
- **Strict exogeneity fails** (E[ε_it | X_i] ≠ 0)
- You suspect **endogenous regressors**

❌ **Don't use GMM when:**
- No dynamics (use FE or RE instead)
- Long panels (T > 20) where FE bias is negligible
- You have weak instruments (persistent series with small T)

## Difference GMM (Arellano-Bond 1991)

### The Idea

**Step 1: First-difference** to eliminate fixed effects α_i:

```
Δinvest_it = γ·Δinvest_{i,t-1} + β₁·Δvalue_it + β₂·Δcapital_it + Δε_it
```

**Problem:** Δinvest_{i,t-1} is still correlated with Δε_it

**Step 2: Use lagged levels as instruments:**

Valid instruments: `invest_{i,t-2}, invest_{i,t-3}, ...` (if E[invest_{i,t-s}·Δε_it] = 0)

### Your First GMM Model

Let's estimate a dynamic investment model:

```python
import panelbox as pb

# Load data
data = pb.load_grunfeld()

# Create lagged investment (GMM does this automatically, but let's see it)
data = data.sort_values(['firm', 'year'])
data['invest_lag'] = data.groupby('firm')['invest'].shift(1)

# Difference GMM
gmm_diff = pb.DifferenceGMM(
    data=data,
    dep_var='invest',           # Dependent variable
    lags=1,                     # Include invest_{t-1}
    exog_vars=['value', 'capital'],  # Exogenous regressors
    id_var='firm',              # Entity identifier
    time_var='year',            # Time identifier
    collapse=True,              # CRITICAL: avoid instrument proliferation
    robust=True                 # Windmeijer-corrected SEs
)

# Fit the model
gmm_results = gmm_diff.fit()
print(gmm_results.summary())
```

**Output (typical):**
```
================================================================================
                     Difference GMM Estimation Results
================================================================================
Dependent Variable:              invest        No. Observations:             170
Model:                    Difference GMM        No. Entities:                  10
Method:                Two-step efficient        No. Instruments:               18
Date:                       2026-02-05          Hansen J statistic:         14.23
Time:                         16:00:00          Hansen J p-value:           0.290
                                                AR(1) p-value:              0.012
                                                AR(2) p-value:              0.356
================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
invest_L1         0.485      0.145      3.345      0.001       0.201       0.769
value             0.092      0.038      2.421      0.015       0.017       0.167
capital           0.198      0.091      2.176      0.030       0.020       0.376
================================================================================
Instruments for level equation: None (differenced model only)
Instruments for diff equation: L2.invest, L3.invest, ..., L2.value, L2.capital
Number of instruments: 18
Instrument ratio (instruments/entities): 1.8
================================================================================
Diagnostic Tests:
  Hansen J test (p=0.290): Overidentifying restrictions valid
  AR(1) test (p=0.012): Expected first-order autocorrelation
  AR(2) test (p=0.356): No second-order autocorrelation ✓
================================================================================
```

### Understanding GMM Output

**Coefficients:**
- `invest_L1 = 0.485`: Strong persistence (past investment predicts current)
- `value = 0.092`: Smaller than FE (0.110) after controlling for dynamics
- `capital = 0.198`: Also smaller than FE (0.310)

**Why coefficients differ from FE:**
- FE was **biased upward** on the lag coefficient
- GMM properly accounts for endogeneity

**Diagnostic Tests (CRITICAL!):**

1. **Hansen J-test (p=0.290)**
   - Tests if instruments are valid (overidentification test)
   - **p > 0.10**: Good (instruments appear valid)
   - **p < 0.10**: Bad (instruments may be invalid)
   - **Interpretation**: p=0.290 → instruments pass validity test ✓

2. **AR(1) test (p=0.012)**
   - Tests for first-order autocorrelation in **differenced** errors
   - **p < 0.05**: Expected and OK (mechanical due to differencing)
   - **Interpretation**: p=0.012 is fine ✓

3. **AR(2) test (p=0.356)**
   - Tests for second-order autocorrelation in **differenced** errors
   - **p > 0.10**: Good (moment conditions are valid)
   - **p < 0.10**: Bad (model is misspecified)
   - **Interpretation**: p=0.356 → no problematic autocorrelation ✓

**Instrument count:**
- 18 instruments for 10 entities
- **Instrument ratio = 1.8** (< 1.0 is ideal, < 2.0 is acceptable)
- Always use `collapse=True` to keep this low!

### Interpreting Results

**Model validity:** ✅
- Hansen J p-value > 0.10 ✓
- AR(2) p-value > 0.10 ✓
- Instrument ratio reasonable (1.8)

**Economic interpretation:**
- **Strong persistence**: 48.5% of last year's investment carries over
- **Value matters**: $1M increase in firm value → $0.092M more investment
- **Capital stock matters**: $1M increase in capital → $0.198M more investment
- **Dynamics are important**: Static FE would have overestimated these effects

## System GMM (Blundell-Bond 1998)

### Why System GMM?

**Problem with Difference GMM:** When series are **persistent** (invest_it highly correlated with invest_{i,t-1}), lagged levels are **weak instruments** for first-differences.

**Solution:** Add **level equations** with lagged differences as instruments.

### The System

**System GMM combines:**

1. **Difference equations** (like Difference GMM):
   - Δinvest_it = γ·Δinvest_{i,t-1} + ...
   - Instruments: Lagged levels

2. **Level equations** (additional):
   - invest_it = γ·invest_{i,t-1} + ...
   - Instruments: Lagged differences

**Extra assumption:** E[Δinvest_{i,1}·η_i] = 0 (stationarity of initial conditions)

**Benefit:** More efficient (smaller standard errors), especially for persistent series.

### Estimation

```python
# System GMM
gmm_sys = pb.SystemGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=True,
    robust=True
)

gmm_sys_results = gmm_sys.fit()
print(gmm_sys_results.summary())
```

**Output (typical):**
```
================================================================================
                       System GMM Estimation Results
================================================================================
Dependent Variable:              invest        No. Observations:             180
Model:                       System GMM        No. Entities:                  10
Method:                Two-step efficient        No. Instruments:               25
Date:                       2026-02-05          Hansen J statistic:         18.45
Time:                         16:05:00          Hansen J p-value:           0.185
                                                AR(1) p-value:              0.009
                                                AR(2) p-value:              0.412
                                                Difference-in-Hansen p:     0.523
================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
invest_L1         0.512      0.098      5.224      0.000       0.320       0.704
value             0.088      0.029      3.034      0.002       0.031       0.145
capital           0.185      0.067      2.761      0.006       0.054       0.316
================================================================================
Instruments for level equation: L1.Δinvest, L1.Δvalue, L1.Δcapital
Instruments for diff equation: L2.invest, L3.invest, ..., L2.value, L2.capital
Number of instruments: 25
Instrument ratio (instruments/entities): 2.5
================================================================================
Diagnostic Tests:
  Hansen J test (p=0.185): Overidentifying restrictions valid
  Difference-in-Hansen (p=0.523): Level instruments valid
  AR(1) test (p=0.009): Expected first-order autocorrelation
  AR(2) test (p=0.412): No second-order autocorrelation ✓
================================================================================
```

### System vs Difference GMM

**Comparison:**

| Aspect | Difference GMM | System GMM |
|--------|----------------|------------|
| **invest_L1** | 0.485 (SE=0.145) | 0.512 (SE=0.098) |
| **Standard errors** | Larger | **Smaller (32% reduction!)** |
| **Instruments** | 18 | 25 |
| **Observations** | 170 (loses first period) | 180 (keeps more data) |
| **Hansen J p-value** | 0.290 | 0.185 |
| **AR(2) p-value** | 0.356 | 0.412 |

**Key insight:** System GMM is more **efficient** (smaller SEs) because it uses more moment conditions.

**When to use System GMM:**
- Series are persistent (ρ > 0.8)
- You want more efficient estimates
- Panel doesn't start at "event time" (e.g., firm entry)
- Stationarity assumption is plausible

**When to use Difference GMM:**
- Series not highly persistent
- Panel starts at event time (initial conditions assumption fails)
- More conservative (fewer assumptions)

## The Critical Importance of Collapse

### The Instrument Proliferation Problem

**Without collapse:**

```python
# BAD: Don't do this!
gmm_bad = pb.DifferenceGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=False,  # ❌ Danger!
    robust=True
)

bad_results = gmm_bad.fit()
print(f"Number of instruments: {bad_results.n_instruments}")  # 87!
print(f"Instrument ratio: {bad_results.instrument_ratio:.1f}")  # 8.7!
```

**Problems with 87 instruments:**
- **Overfitting**: Model fits sample perfectly but doesn't generalize
- **Hansen J test loses power**: Always p ≈ 1.0 (can't detect invalid instruments)
- **Downward bias in SEs**: Confidence intervals too narrow
- **Computational issues**: Matrix inversion unstable

### Always Use Collapse

**Rule of thumb (Roodman 2009):**
- Instrument count should be **< number of entities**
- Instrument ratio < 1.0 is ideal
- **Always set collapse=True** unless you have a specific reason

```python
# GOOD: Always do this
gmm_good = pb.DifferenceGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=True,  # ✅ Essential
    robust=True
)
```

## Complete GMM Workflow

### Step 1: Verify dynamics are needed

```python
# Compare static FE with GMM
fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

# If value/capital coefficients differ much between FE and GMM → dynamics matter
```

### Step 2: Start with Difference GMM

```python
gmm_diff = pb.DifferenceGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=True,
    robust=True
)

gmm_diff_results = gmm_diff.fit()
```

### Step 3: Check diagnostics

```python
# Print results
print(gmm_diff_results.summary())

# Check critical tests
print(f"\nHansen J p-value: {gmm_diff_results.hansen_j.pvalue:.3f}")
print(f"AR(2) p-value: {gmm_diff_results.ar2_test.pvalue:.3f}")
print(f"Instrument ratio: {gmm_diff_results.instrument_ratio:.2f}")

# Decision criteria
hansen_ok = gmm_diff_results.hansen_j.pvalue > 0.10
ar2_ok = gmm_diff_results.ar2_test.pvalue > 0.10
instruments_ok = gmm_diff_results.instrument_ratio < 2.0

if hansen_ok and ar2_ok and instruments_ok:
    print("✓ Model passes all diagnostic tests")
else:
    print("✗ Model fails diagnostics - reconsider specification")
```

### Step 4: Try System GMM (if appropriate)

```python
# Estimate System GMM
gmm_sys = pb.SystemGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=True,
    robust=True
)

gmm_sys_results = gmm_sys.fit()

# Compare standard errors
print("\nStandard Error Comparison:")
print(f"Difference GMM - invest_L1 SE: {gmm_diff_results.std_errors['invest_L1']:.3f}")
print(f"System GMM - invest_L1 SE: {gmm_sys_results.std_errors['invest_L1']:.3f}")
print(f"Efficiency gain: {(1 - gmm_sys_results.std_errors['invest_L1']/gmm_diff_results.std_errors['invest_L1'])*100:.1f}%")
```

### Step 5: Report final model

```python
# Choose model (System GMM if diagnostics pass and more efficient)
final_results = gmm_sys_results

# Export to LaTeX
final_results.to_latex("investment_gmm.tex")

# Create summary table
summary = {
    'Coefficient': final_results.params,
    'Std Error': final_results.std_errors,
    'z-stat': final_results.tstats,
    'p-value': final_results.pvalues
}

import pandas as pd
summary_df = pd.DataFrame(summary)
print("\n" + "="*60)
print("FINAL GMM RESULTS")
print("="*60)
print(summary_df)
```

## Common Pitfalls and Solutions

### ❌ Pitfall 1: Forgetting collapse=True

**Problem:** 87 instruments, Hansen J always passes (p ≈ 1.0)

**Solution:** **Always use collapse=True**

```python
# GOOD
gmm = pb.DifferenceGMM(..., collapse=True)
```

### ❌ Pitfall 2: Ignoring diagnostic tests

**Problem:** Reporting results even when Hansen J or AR(2) fail

**Solution:** Check tests BEFORE interpreting coefficients

```python
# Check tests first
if results.hansen_j.pvalue < 0.10:
    print("⚠ Warning: Instruments may be invalid")
if results.ar2_test.pvalue < 0.10:
    print("⚠ Warning: Moment conditions violated")
```

### ❌ Pitfall 3: Using System GMM inappropriately

**Problem:** Using System GMM when panel starts at "event time" (e.g., firm entry)

**Solution:** Use Difference GMM if initial conditions assumption fails

### ❌ Pitfall 4: Too few time periods

**Problem:** T < 5 gives very weak instruments

**Solution:**
- Need at least T ≥ 5 for Difference GMM
- Consider static FE if T < 5

### ❌ Pitfall 5: Highly persistent series with small T

**Problem:** When ρ > 0.9 and T < 8, lagged levels are weak instruments

**Solution:**
- Use System GMM (more efficient with weak instruments)
- Or consider long-difference IV (not in PanelBox yet)

## Key Takeaways

✅ **You've learned to:**
- Recognize when GMM is needed (dynamics, endogeneity)
- Estimate Difference GMM and System GMM
- Interpret Hansen J and AR(2) diagnostic tests
- Understand the critical importance of collapse=True
- Follow a complete GMM workflow
- Avoid common pitfalls

🔑 **Critical concepts:**
- **Dynamics cause bias in FE** (Nickell bias)
- **Difference GMM**: First-difference + lagged levels as instruments
- **System GMM**: Add level equations for efficiency
- **Always collapse**: Avoid instrument proliferation
- **Diagnostic tests**: Hansen J > 0.10, AR(2) > 0.10

⚠️ **Remember:**
- GMM is powerful but requires careful diagnostic checking
- Don't report results if tests fail
- Instrument count should be < number of entities

## Next Steps

**Deepen your understanding:**

1. **[How-To: Interpret Tests](../how-to/interpret_tests.md)**: Deep dive into Hansen J, AR tests, Sargan

2. **[Guide: GMM Explained](../guides/gmm_explained.md)**: Mathematical details of GMM estimation

3. **[API Reference: GMM](../api/gmm.md)**: Complete documentation of DifferenceGMM and SystemGMM

**Advanced topics:**
- Including predetermined variables (not strictly exogenous)
- Testing for weak instruments
- Difference-in-Hansen test for subset validity
- Bootstrap standard errors

**Practice yourself:**

```python
# Try GMM with your data
data = pd.read_csv('my_dynamic_panel.csv')

# Difference GMM
gmm = pb.DifferenceGMM(
    data=data,
    dep_var='y',
    lags=1,  # or 2 for y_{t-2}
    exog_vars=['x1', 'x2'],
    id_var='entity_id',
    time_var='time',
    collapse=True,  # Always!
    robust=True
)

results = gmm.fit()

# Check diagnostics FIRST
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f}")
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f}")

# Then interpret if tests pass
if results.hansen_j.pvalue > 0.10 and results.ar2_test.pvalue > 0.10:
    print(results.summary())
```

## Further Reading

**Seminal Papers:**
- **Arellano & Bond (1991)**: "Some Tests of Specification for Panel Data", *Review of Economic Studies*, 58(2), 277-297
  - Original Difference GMM paper
- **Blundell & Bond (1998)**: "Initial Conditions and Moment Restrictions", *Journal of Econometrics*, 87(1), 115-143
  - Introduces System GMM
- **Windmeijer (2005)**: "A Finite Sample Correction for GMM", *Journal of Econometrics*, 126(1), 25-51
  - Two-step standard error correction

**Practical Guides:**
- **Roodman (2009)**: "How to do xtabond2", *The Stata Journal*, 9(1), 86-136
  - Best practical guide to GMM (Stata, but concepts apply)
- **Bond (2002)**: "Dynamic Panel Data Models: A Guide to Micro Data Methods", *Portuguese Economic Journal*, 1(2), 141-162
  - Accessible introduction

**Textbooks:**
- **Baltagi (2021)**: *Econometric Analysis of Panel Data*, Chapter 8
- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data*, Chapter 11

---

**Congratulations!** You can now estimate and validate dynamic panel models using GMM. 🎉

You've completed all three core PanelBox tutorials! Explore the how-to guides and API reference for more advanced features.
