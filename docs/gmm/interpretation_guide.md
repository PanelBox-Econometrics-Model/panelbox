# GMM Results Interpretation Guide

**A Complete Guide to Reading and Understanding PanelBox GMM Output**

---

## Table of Contents

1. [Reading the Output](#reading-the-output)
2. [Specification Tests](#specification-tests)
3. [Diagnostic Metrics](#diagnostic-metrics)
4. [Common Patterns](#common-patterns)
5. [Troubleshooting](#troubleshooting)
6. [Practical Examples](#practical-examples)

---

## Reading the Output

### Standard GMM Output Format

```
==============================================================================
                                Difference GMM
==============================================================================
Number of observations:            751
Number of groups:                  140
Number of instruments:               8
Instrument ratio:                0.057
GMM type:           Two-step (Windmeijer)
------------------------------------------------------------------------------
Variable                    Coef.     Std.Err.        z    P>|z|     [95% Conf. Int.]
------------------------------------------------------------------------------
L1.y                     0.5764     0.1245     4.631    0.000 [ 0.3324,  0.8204] ***
w                       -0.4082     0.0892    -4.577    0.000 [-0.5830, -0.2334] ***
k                        0.2341     0.0654     3.579    0.000 [ 0.1059,  0.3623] ***
ys                       0.3892     0.0987     3.943    0.000 [ 0.1957,  0.5827] ***
==============================================================================
Specification Tests:
------------------------------------------------------------------------------
Hansen J-test: statistic=3.891, p-value=0.143, df=4 [Instruments valid]
Sargan test: statistic=4.123, p-value=0.127, df=4 [Instruments valid]
AR(1) test: statistic=-3.214, p-value=0.001 [Expected (MA(1) by construction)]
AR(2) test: statistic=-0.891, p-value=0.373 [Moment conditions valid]
==============================================================================
```

### Understanding Each Component

#### Header Section

**Number of observations: 751**
- Number of observations used in estimation (after differencing and lags)
- Should be reasonably close to input data size
- Large drop may indicate:
  - Unbalanced panel with many missing periods
  - Too many time dummies
  - Insufficient instrument availability

**Number of groups: 140**
- Number of cross-sectional units (individuals, firms, countries)
- Should match your panel structure

**Number of instruments: 8**
- Total columns in instrument matrix Z
- Depends on:
  - Number of time periods
  - Lag depths used
  - `collapse=True` vs `collapse=False`
  - Exogenous vs predetermined vs endogenous variables

**Instrument ratio: 0.057**
- Formula: `n_instruments / n_groups`
- **Critical metric** for instrument proliferation
- See detailed interpretation below

**GMM type: Two-step (Windmeijer)**
- One-step: Uses (Z'Z)^(-1) as weight matrix
- Two-step: Uses optimal weight matrix with Windmeijer (2005) finite-sample correction
- Two-step generally preferred (more efficient)

#### Coefficient Table

**Variable**: Name of regressor
- `L1.y`: First lag of dependent variable (γ coefficient)
- `L2.y`: Second lag (if included)
- Variable names as specified in `exog_vars`
- `year_XXXX`: Time dummies (if `time_dummies=True`)

**Coef.**: Point estimate of coefficient
- Main result - the estimated parameter
- Interpretation depends on variable units and model specification

**Std.Err.**: Standard error of coefficient
- Measure of estimation uncertainty
- Smaller = more precise estimate
- Large SE suggests:
  - Weak instruments
  - High variance in data
  - Small sample size

**z**: Test statistic (Wald test)
- Formula: `Coef. / Std.Err.`
- Tests H0: coefficient = 0
- |z| > 1.96 → significant at 5% level
- |z| > 2.576 → significant at 1% level

**P>|z|**: P-value for two-sided test
- Probability of observing this z-statistic under H0
- p < 0.05 → reject H0 at 5% level (significant)
- p < 0.01 → reject H0 at 1% level (highly significant)

**[95% Conf. Int.]**: 95% confidence interval
- Range: [Coef. - 1.96×Std.Err., Coef. + 1.96×Std.Err.]
- Interpretation: "We are 95% confident the true value lies in this range"
- If interval excludes 0 → coefficient is significant

**Significance stars**:
- `*`: p < 0.05 (significant)
- `**`: p < 0.01 (highly significant)
- `***`: p < 0.001 (very highly significant)

---

## Specification Tests

### Hansen J-test (Overidentification Test)

**What it tests:**
```
H0: All instruments are valid (orthogonal to errors)
H1: At least one instrument is invalid
```

**Interpretation Guideline:**

| p-value Range | Interpretation | Action |
|---------------|----------------|--------|
| p < 0.05 | **REJECT** - Instruments likely invalid | Check model specification |
| p < 0.10 | **WARNING** - Weak evidence against instruments | Investigate further |
| 0.10 < p < 0.25 | **IDEAL** - Instruments appear valid | Proceed with confidence |
| 0.25 < p < 0.50 | **ACCEPTABLE** - No strong evidence against | Proceed cautiously |
| p > 0.50 | **WARNING** - Possible weak instruments | Check instrument relevance |

**Decision Rules:**

```python
if 0.10 < hansen_p < 0.25:
    print("✓ Instruments appear valid")
elif hansen_p < 0.10:
    print("✗ Instruments rejected - try different specification")
    # Actions:
    # - Remove potentially endogenous variables
    # - Use fewer instruments (collapse=True)
    # - Try different lag structure
else:
    print("⚠ Very high p-value - check instrument strength")
    # Actions:
    # - Examine first-stage relevance
    # - Consider more lags as instruments
```

**Why high p-value can be bad:**
- J-test has low power with weak instruments
- Weak instruments → test fails to detect invalidity
- Always combine with other diagnostics

**When test is not available:**
- Exactly identified: n_instruments = n_parameters → df = 0
- Under-identified: n_instruments < n_parameters → impossible
- Output shows: `[N/A (exactly/under-identified)]`

### Sargan Test

**What it is:**
- Non-robust version of Hansen J-test
- Same interpretation as Hansen J

**When to use which:**

```python
if robust == True:
    # Use Hansen J-test (robust to heteroskedasticity)
    primary_test = results.hansen_j
else:
    # Use Sargan test (assumes homoskedasticity)
    primary_test = results.sargan
```

**Recommendation:** Always use `robust=True` → focus on Hansen J.

### AR(1) Test (First-Order Autocorrelation)

**What it tests:**
```
H0: No first-order autocorrelation in differenced errors
H1: First-order autocorrelation present
```

**Expected result:** **REJECT** (p < 0.10)

**Why?**
First-differencing mechanically induces MA(1) autocorrelation:
```
Δε_it = ε_it - ε_{i,t-1}
Δε_{i,t-1} = ε_{i,t-1} - ε_{i,t-2}

Cov(Δε_it, Δε_{i,t-1}) = -Var(ε_{i,t-1}) < 0
```

**Interpretation:**

| p-value | Interpretation |
|---------|----------------|
| p < 0.10 | ✓ **Expected** - Normal MA(1) structure |
| p > 0.10 | ⚠ **Unexpected** - Investigate data structure |

**If NOT rejected:**
- Check if original errors have negative autocorrelation (unusual)
- Verify first-differencing was applied correctly
- Not necessarily a problem, but unexpected

### AR(2) Test (Second-Order Autocorrelation)

**What it tests:**
```
H0: No second-order autocorrelation in differenced errors
H1: Second-order autocorrelation present
```

**Expected result:** **DO NOT REJECT** (p > 0.10)

**Why it's critical:**
If original errors are serially uncorrelated:
```
E[ε_it · ε_{i,t-2}] = 0

Then: E[Δε_it · Δε_{i,t-2}] = 0
```

This validates the moment condition `E[y_{i,t-2} · Δε_it] = 0`.

**Interpretation:**

| p-value | Interpretation | Validity |
|---------|----------------|----------|
| p > 0.10 | ✓ **Moment conditions valid** | GMM consistent |
| 0.05 < p < 0.10 | ⚠ **Weak evidence of problem** | Borderline |
| p < 0.05 | ✗ **Moment conditions invalid** | GMM inconsistent! |

**If rejected (p < 0.05):**

This is **serious** - means GMM estimates are inconsistent.

**Possible causes:**
1. Original errors have autocorrelation
2. Model is misspecified
3. Measurement error in variables
4. Dynamic misspecification (need more lags)

**Actions to take:**
```python
# 1. Try including more lags of y
gmm = DifferenceGMM(..., lags=[1, 2], ...)

# 2. Check for autocorrelation in levels
# Run AR tests on OLS residuals

# 3. Consider different model specification
# Add omitted variables?

# 4. If nothing works, GMM may not be appropriate
```

**Critical rule:** If AR(2) is rejected, do NOT trust GMM results!

---

## Diagnostic Metrics

### Instrument Ratio

**Formula:**
```python
instrument_ratio = n_instruments / n_groups
```

**Rule of Thumb:**

| Ratio | Assessment | Recommendation |
|-------|------------|----------------|
| < 0.5 | ✓ **Good** | Proceed with confidence |
| 0.5 - 1.0 | ⚠ **Acceptable** | Monitor carefully |
| 1.0 - 2.0 | ⚠ **Warning** | Use `collapse=True` |
| > 2.0 | ✗ **Problematic** | Reduce instruments |

**Why it matters:**
- Too many instruments → overfitting
- Biases coefficients toward OLS/FE
- Weakens specification tests (Roodman 2009)

**Solutions for high ratio:**

```python
# 1. Use collapsed instruments (most effective)
gmm = DifferenceGMM(..., collapse=True, ...)

# 2. Reduce time dummies
gmm = DifferenceGMM(..., time_dummies=False, ...)
# Or use trend instead of full dummies

# 3. Limit lag depth (manual instrument specification - advanced)

# 4. Reduce number of variables
```

### Number of Observations

**What to check:**
```python
retention_rate = results.nobs / len(input_data)
```

**Expected rates:**

| Panel Type | Expected Retention | If Much Lower |
|------------|-------------------|---------------|
| Balanced | 70-90% | Check specification |
| Unbalanced | 50-80% | May be normal |
| Very unbalanced | 30-60% | Use simpler spec |

**Reasons for low retention:**
- First-differencing loses first period
- Lags lose additional periods
- Time dummies in unbalanced panels
- Many missing values in instruments

**Action if very low (<30%):**
```python
# 1. Simplify specification
gmm = DifferenceGMM(
    ...,
    time_dummies=False,  # Remove time dummies
    collapse=True,       # Essential
    ...
)

# 2. Check for missing data
print(df[['y', 'x1', 'x2']].isnull().sum())

# 3. Try balanced subset
df_balanced = df.groupby('id').filter(lambda x: len(x) == max_T)
```

---

## Common Patterns

### Pattern 1: Good Results

**Characteristics:**
- ✓ AR(2) p-value > 0.10
- ✓ Hansen J: 0.10 < p < 0.25
- ✓ Instrument ratio < 0.5
- ✓ Coefficient in credible range (between FE and OLS)
- ✓ Reasonable standard errors

**Example:**
```
Hansen J: p = 0.183
AR(2): p = 0.312
Instruments: 8, Groups: 140, Ratio: 0.057
γ̂ = 0.576 (SE: 0.125), Range: [0.464, 0.712]
```

**Interpretation:** Proceed with confidence! Results are reliable.

### Pattern 2: Weak Instruments

**Characteristics:**
- Hansen J: p > 0.50 (very high)
- Very large standard errors
- Wide confidence intervals
- Coefficients may be implausible

**Example:**
```
Hansen J: p = 0.782
AR(2): p = 0.421
γ̂ = 0.612 (SE: 0.456)  # SE very large
95% CI: [-0.282, 1.506]  # Very wide
```

**Diagnosis:** Instruments not strongly correlated with endogenous variables.

**Solutions:**
```python
# 1. Try System GMM (more instruments)
sys_gmm = SystemGMM(...)

# 2. Use more lag depths
# 3. Increase sample size if possible
# 4. Check if GMM is really necessary (compare OLS/FE gap)
```

### Pattern 3: Invalid Instruments

**Characteristics:**
- Hansen J: p < 0.05 (rejected)
- AR(2) may or may not reject
- Coefficients may be outside credible range

**Example:**
```
Hansen J: p = 0.023  # REJECTED
AR(2): p = 0.156
γ̂ = 0.892  # Outside [0.464, 0.712]
```

**Diagnosis:** Instruments correlated with errors.

**Solutions:**
```python
# 1. Remove potentially endogenous variables
exog_vars = ['x1']  # Remove x2 if suspect

# 2. Treat more variables as endogenous
predetermined_vars = ['x2']  # Instead of exog_vars

# 3. Check for model misspecification
# - Missing variables?
# - Wrong functional form?
# - Measurement error?

# 4. Try different lag structure
```

### Pattern 4: Serial Correlation Problem

**Characteristics:**
- AR(2): p < 0.05 (REJECTED)
- Hansen J may pass or fail
- **Critical issue**

**Example:**
```
Hansen J: p = 0.142
AR(2): p = 0.018  # REJECTED - PROBLEM!
```

**Diagnosis:** Moment conditions invalid, GMM inconsistent.

**Solutions:**
```python
# 1. Add more lags of y
gmm = DifferenceGMM(..., lags=[1, 2], ...)

# 2. Check original specification
# - Omitted variables?
# - Need different dynamics?

# 3. Try System GMM
sys_gmm = SystemGMM(...)

# 4. If all fail, GMM may not be appropriate for this data
```

### Pattern 5: Too Many Instruments

**Characteristics:**
- Instrument ratio > 1.0
- Hansen J: p > 0.50 (very high)
- Results close to OLS/FE

**Example:**
```
Instruments: 187, Groups: 140, Ratio: 1.336
Hansen J: p = 0.892
γ̂ = 0.698 (very close to OLS)
```

**Diagnosis:** Instrument proliferation, overfitting.

**Solution:**
```python
# Use collapsed instruments
gmm = DifferenceGMM(..., collapse=True, ...)
```

---

## Troubleshooting

### Problem: "Number of observations: 0"

**Cause:** No observations have sufficient valid instruments.

**Solutions:**
1. Set `collapse=True`
2. Set `time_dummies=False`
3. Simplify specification (fewer variables)
4. Check for missing data

### Problem: "All coefficients are zero"

**Cause:** Estimation failed, no valid observations.

**Same solutions as above.**

### Problem: "Singular matrix" warnings

**Cause:** Perfect multicollinearity or insufficient variation.

**Solutions:**
1. Check for redundant variables
2. Remove time dummies causing perfect collinearity
3. Ensure sufficient variation in data

### Problem: Very large standard errors

**Cause:** Weak instruments.

**Solutions:**
1. Try System GMM
2. Use more lag depths
3. Increase sample size
4. Check if GMM is necessary

### Problem: Results very different from OLS/FE

**Not necessarily a problem!** This is expected if OLS/FE are biased.

**But check:**
1. Coefficient in credible range [FE, OLS]?
2. All diagnostics pass?
3. Results make economic sense?

---

## Practical Examples

### Example 1: Employment Equation

```
Variable        Coef.   Std.Err.    Interpretation
L1.n          0.6860    0.1370     High persistence in employment
w            -0.5210    0.1490     Wages negatively affect employment
k             0.2940    0.0890     Capital positively affects employment

Hansen J: p = 0.163  ✓ Valid
AR(2): p = 0.289     ✓ Valid
Ratio: 0.057         ✓ Good
```

**Interpretation:**
- 68.6% of employment persists from previous year
- 10% wage increase → 5.2% employment decrease
- 10% capital increase → 2.9% employment increase
- All diagnostics pass → results are reliable

### Example 2: Firm Growth

```
Variable        Coef.   Std.Err.    Interpretation
L1.size       0.8520    0.0890     Very persistent size
investment    0.1230    0.0340     Investment promotes growth
age          -0.0120    0.0056     Older firms grow slower

Hansen J: p = 0.201  ✓ Valid
AR(2): p = 0.412     ✓ Valid
Ratio: 0.071         ✓ Good
```

**Interpretation:**
- 85.2% persistence → highly persistent process
- System GMM might be more efficient (high γ)
- All diagnostics excellent

---

## Summary Checklist

Before accepting GMM results, verify:

- [ ] AR(2) p-value > 0.10 (critical!)
- [ ] Hansen J: 0.10 < p < 0.25 (ideal) or at least p > 0.10
- [ ] Instrument ratio < 1.0
- [ ] Coefficient in credible range [FE, OLS]
- [ ] Standard errors reasonable (not huge)
- [ ] Number of observations reasonable
- [ ] Results make economic sense

If all checks pass: **Results are reliable!**

If any fail: Investigate, revise specification, or consider alternatives.

---

**Guide Version:** 1.0
**Last Updated:** January 2026
**Author:** PanelBox Development Team
