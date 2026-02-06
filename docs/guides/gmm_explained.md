# GMM for Panel Data: A Complete Explanation

> Deep dive into the theory, mechanics, and intuition behind Generalized Method of Moments (GMM) estimators for dynamic panel data.

## Overview

This guide provides a comprehensive explanation of GMM for panel data:

- **Why GMM is needed** (the dynamic panel bias problem)
- **Moment conditions** and instrument construction
- **Difference GMM** (Arellano-Bond 1991)
- **System GMM** (Blundell-Bond 1998)
- **Estimation mechanics** (one-step, two-step)
- **Diagnostic tests** and their interpretation
- **When GMM works** and when it fails

## The Dynamic Panel Problem

### The Model

Consider a **dynamic panel model**:

```
y_it = γ y_{i,t-1} + X_it'β + α_i + ε_it
```

**Where:**
- `y_{i,t-1}` = lagged dependent variable (dynamics)
- `X_it` = exogenous regressors
- `α_i` = entity-specific fixed effect
- `ε_it` = idiosyncratic error

**Assumptions:**
- E[α_i] = 0
- E[ε_it] = 0
- E[ε_it | ε_is] = 0 for all t ≠ s (no serial correlation)
- E[X_it ε_is] = 0 for all t, s (strict exogeneity of X)

### Why Fixed Effects Fails

**Naive approach:** Use Fixed Effects (within transformation)

**Problem:** Within transformation creates correlation!

**Proof:**

Within transformation:
```
(y_it - ȳ_i) = γ(y_{i,t-1} - ȳ_i) + (X_it - X̄_i)'β + (ε_it - ε̄_i)
```

**Note:**
- `ȳ_i = (1/T)Σ_t y_it` includes `y_i,t-1`, `y_it`, `y_{i,t+1}`, etc.
- `ε̄_i = (1/T)Σ_t ε_it` includes `ε_it`

**Correlation:**
```
Cov(y_{i,t-1} - ȳ_i, ε_it - ε̄_i) ≠ 0
```

Even though `E[y_{i,t-1} ε_it] = 0`, the demeaning creates dependence!

**Result:** **Nickell bias** (biased and inconsistent estimates)

**Magnitude:** Bias = O(1/T)
- T = 5: Bias ≈ 20%
- T = 10: Bias ≈ 10%
- T = 20: Bias ≈ 5%

**Implication:** Severe for short panels (T < 10)

## The GMM Solution: Moment Conditions

### Key Insight

**Cannot** use levels with FE (demeaning causes correlation)

**Solution:** Use **first-differences** to eliminate α_i

### First-Differencing

**Difference equation:**
```
Δy_it = γ Δy_{i,t-1} + Δ X_it'β + Δε_it

where Δy_it = y_it - y_{i,t-1}
```

**Success:** Fixed effect α_i is gone!

**New problem:** `Δy_{i,t-1}` is **still** correlated with `Δε_it`

**Proof:**
```
Δy_{i,t-1} = y_{i,t-1} - y_{i,t-2}

Δε_it = ε_it - ε_{i,t-1}
```

So `Δy_{i,t-1}` includes `y_{i,t-1}` which depends on `ε_{i,t-1}`, which appears in `Δε_it`!

**Conclusion:** OLS on differenced equation is still biased

### Instruments: The GMM Idea

**Question:** What variables are:
1. Correlated with `Δy_{i,t-1}` (relevant)
2. Uncorrelated with `Δε_it` (exogenous)

**Answer:** **Lagged levels** `y_{i,t-2}, y_{i,t-3}, ...`

**Why this works:**

**Moment condition:**
```
E[y_{i,t-s} Δε_it] = E[y_{i,t-s} (ε_it - ε_{i,t-1})] = 0  for s ≥ 2
```

**Proof (for s = 2):**
```
E[y_{i,t-2} ε_it] = 0  (ε_it is future, y_{i,t-2} is past)
E[y_{i,t-2} ε_{i,t-1}] = 0  (no serial correlation in ε)

→ E[y_{i,t-2} Δε_it] = 0 ✓
```

**This is the foundation of GMM!**

## Difference GMM (Arellano-Bond 1991)

### Moment Conditions

For each time period t, we have moment conditions:

**t = 3:**
```
E[y_{i1} (Δy_{i3} - γ Δy_{i2} - Δ X_{i3}'β)] = 0
```

**t = 4:**
```
E[y_{i1} (Δy_{i4} - γ Δy_{i3} - Δ X_{i4}'β)] = 0
E[y_{i2} (Δy_{i4} - γ Δy_{i3} - Δ X_{i4}'β)] = 0
```

**General (for period t):**
```
E[y_is Δε_it] = 0  for s = 1, ..., t-2
```

**Number of moment conditions:** Grows with T
- t = 3: 1 instrument
- t = 4: 2 instruments
- t = T: T-2 instruments
- **Total:** (T-2)(T-1)/2 instruments (without collapse)

### Instrument Matrix (Without Collapse)

For T = 5:

```
Period | Instruments
-------|------------------
  3    | y_i1
  4    | y_i1, y_i2
  5    | y_i1, y_i2, y_i3
```

**Matrix Z_i:**
```
        [y_i1   0     0    0  ]
Z_i =   [  0  y_i1  y_i2  0  ]
        [  0    0   y_i1 y_i2 y_i3]
```

**Problem:** Instrument count explodes!

### Collapsed Instruments (Roodman 2009)

**Instead of** using all lags separately...

**Use:** One instrument per period (linear combination)

**Collapsed Z_i for T = 5:**
```
        [y_i1         0            0      ]
Z_i =   [  0     (y_i1+y_i2)/2      0      ]
        [  0          0      (y_i1+y_i2+y_i3)/3]
```

**Result:** Number of instruments = T - 2 (linear in T, not quadratic!)

**In PanelBox:** `collapse=True` always does this

### GMM Estimation

**Step 1: Form moment conditions**

```
m_i(θ) = Z_i' Δε_i(θ)

where Δε_i(θ) = Δy_i - γ Δy_i,(-1) - ΔX_i β
```

**Step 2: Minimize objective function**

```
θ̂ = argmin_θ [Σ_i m_i(θ)]' W [Σ_i m_i(θ)]
```

where W is a weighting matrix

**Step 3: Choose W**

**One-step GMM:** W = I (identity matrix)

**Two-step GMM:**
1. Estimate with W = I, get residuals
2. Estimate optimal W = Σ̂^(-1) where Σ̂ = (1/N)Σ_i Z_i' Δε̂_i Δε̂_i' Z_i
3. Re-estimate with optimal W

**Two-step is asymptotically efficient** (smallest variance)

### Windmeijer Correction (2005)

**Problem:** Two-step SEs are **downward biased** in finite samples

**Magnitude:** Bias can be 30-50% (SEs too small!)

**Solution:** Windmeijer finite-sample correction

**In PanelBox:** Automatically applied when `robust=True`

## System GMM (Blundell-Bond 1998)

### Motivation

**Problem with Difference GMM:** When y_it is **persistent** (near unit root), lagged levels are **weak instruments** for first-differences.

**Intuition:**
- If `y_it ≈ y_{i,t-1}` (highly persistent)
- Then `Δy_it ≈ 0` (small variation)
- And `y_{i,t-2}` doesn't predict `Δy_it` well (weak instrument)

**Consequence:** Large standard errors, imprecise estimates

### The System GMM Idea

**Add level equations** to the system with **lagged differences as instruments**

**System:**

1. **Difference equations** (Arellano-Bond):
   ```
   Δy_it = γ Δy_{i,t-1} + ΔX_it'β + Δε_it
   Instruments: y_{i,t-2}, y_{i,t-3}, ... (levels)
   ```

2. **Level equations** (additional):
   ```
   y_it = γ y_{i,t-1} + X_it'β + η_i + ε_it
   Instruments: Δy_{i,t-1}, Δy_{i,t-2}, ... (differences)
   ```

**Key:** Use lags of **differences** as instruments for **levels**

### Additional Moment Conditions

**For level equation:**
```
E[Δy_{i,t-1} (α_i + ε_it)] = 0
```

**Critical assumption (stationarity of initial conditions):**
```
E[Δy_{i,1} α_i] = 0
```

**Interpretation:** Initial deviations from steady-state are uncorrelated with fixed effects

**When this holds:**
- Time-series is stationary
- Panel doesn't start at "event time" (e.g., firm entry)

**When this fails:**
- Panel starts at event (firm entry, policy change)
- Initial period is special

### Efficiency Gain

**Why System GMM is more efficient:**

1. **More moment conditions** (level equations added)
2. **Uses level variation** (not just differences)
3. **Better instruments for persistent series**

**Typical gain:** 20-50% reduction in standard errors

**Trade-off:** Stronger assumption (stationarity of initial conditions)

## Estimation Steps (Detailed)

### Two-Step System GMM

**Step 1: First-step estimation**

1. Form instrument matrix Z_i (difference + level instruments)
2. Set W = I (identity)
3. Compute:
   ```
   θ̂₁ = (Σ_i Z_i' X̃_i)' W (Σ_i Z_i' X̃_i))^(-1) (Σ_i Z_i' X̃_i)' W (Σ_i Z_i' ỹ_i)
   ```
4. Compute residuals: ε̂₁ = ỹ - X̃ θ̂₁

**Step 2: Optimal weighting matrix**

1. Construct Σ̂₁ = (1/N) Σ_i Z_i' ε̂₁ ε̂₁' Z_i
2. Set W = Σ̂₁^(-1)

**Step 3: Second-step estimation**

1. Re-estimate with optimal W:
   ```
   θ̂₂ = (Σ_i Z_i' X̃_i)' W (Σ_i Z_i' X̃_i))^(-1) (Σ_i Z_i' X̃_i)' W (Σ_i Z_i' ỹ_i)
   ```

**Step 4: Windmeijer correction**

1. Compute corrected variance:
   ```
   Var(θ̂₂) = (1/N) D'_N Σ̂_corr D_N
   ```
   with finite-sample adjustment

## Diagnostic Tests

### Hansen J Test

**Purpose:** Test overidentifying restrictions

**Statistic:**
```
J = N · (Σ_i Z_i' ê_i)' Σ̂^(-1) (Σ_i Z_i' ê_i) ~ χ²(q - k)
```

where q = # instruments, k = # parameters

**Interpretation:**

| J statistic | p-value | Interpretation |
|-------------|---------|----------------|
| Small | p > 0.25 | Strong evidence instruments valid |
| Moderate | 0.10 < p < 0.25 | Instruments likely valid |
| Large | p < 0.10 | Instruments may be invalid |

**Problem:** With too many instruments, J → 0 (always p ≈ 1)

**Solution:** **Always use collapse=True**

### AR(1) and AR(2) Tests

**Purpose:** Test for serial correlation in **differenced** errors

**AR(m) statistic:**
```
AR(m) = (Σ_i Δê_it Δê_{i,t-m}) / √Var(Σ_i Δê_it Δê_{i,t-m})
```

**Under H₀:** AR(m) ~ N(0, 1)

**Expected results:**

**AR(1) test:**
- **Typically rejects** (p < 0.05) → Expected!
- Mechanical due to MA(1) structure in Δε_it

**AR(2) test:**
- **Should NOT reject** (p > 0.10) → Critical!
- If rejects → moment conditions E[y_{i,t-2} Δε_it] = 0 invalid

**Intuition:**

If AR(2) rejects:
- Serial correlation in levels: E[ε_it ε_{i,t-2}] ≠ 0
- Then E[y_{i,t-2} Δε_it] ≠ 0 (instruments invalid!)
- GMM estimator is inconsistent

### Difference-in-Hansen Test (System GMM only)

**Purpose:** Test validity of additional level instruments

**Statistic:**
```
Diff-Hansen = J_system - J_difference
```

**Under H₀:** Diff-Hansen ~ χ²(q_level)

**Interpretation:**

| p-value | Interpretation |
|---------|----------------|
| p > 0.10 | Level instruments valid |
| p < 0.10 | Level instruments invalid → Use Difference GMM |

**When to use:** To check if System GMM assumptions hold

## When GMM Works and When It Fails

### GMM Works Well When:

✅ **Short panel** (small T, large N)
- T = 5-10 years, N = 500+ entities
- Asymptotic theory relies on N → ∞

✅ **No serial correlation** in ε_it
- Critical for moment conditions
- AR(2) test should pass

✅ **Moderate persistence** (for Difference GMM)
- 0.3 < γ < 0.8
- Instruments have enough variation

✅ **High persistence** (for System GMM)
- γ > 0.8
- Additional level moments help

✅ **Enough lags available**
- Need t ≥ 3 for Difference GMM
- Preferably T ≥ 5

### GMM Fails When:

❌ **Too many instruments**
- Without collapse: q/N > 1
- Overfitting, Hansen J loses power

❌ **Serial correlation** in levels
- E[ε_it ε_{i,t-s}] ≠ 0 for s > 0
- Invalidates moment conditions
- AR(2) test will reject

❌ **Weak instruments** (Difference GMM)
- Near unit root (γ ≈ 1)
- Very persistent series
- **Solution:** Use System GMM

❌ **Very short panel** (T < 5)
- Too few instruments
- Large bias

❌ **Measurement error**
- Amplified in differences
- Larger bias than levels

❌ **Initial conditions violated** (System GMM)
- Panel starts at event time
- E[Δy_{i1} α_i] ≠ 0
- **Solution:** Use Difference GMM

## Practical Workflow

### Step 1: Choose Difference or System GMM

**Flow:**

```
Is series highly persistent (ρ > 0.8)?
  YES → Try System GMM (more efficient)
  NO → Start with Difference GMM (fewer assumptions)

Does panel start at event time?
  YES → Use Difference GMM
  NO → System GMM OK
```

### Step 2: Start Conservative

```python
gmm = pb.DifferenceGMM(
    data=data,
    dep_var='y',
    lags=1,              # One lag of y
    exog_vars=['x1'],    # Few exog vars
    id_var='id',
    time_var='year',
    collapse=True,       # ALWAYS
    robust=True          # Windmeijer correction
)

results = gmm.fit()
```

### Step 3: Check Diagnostics

```python
# Hansen J test
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f}")
# Want: p > 0.10

# AR(2) test
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f}")
# Want: p > 0.10

# Instrument ratio
print(f"Instrument ratio: {results.instrument_ratio:.2f}")
# Want: < 1.0 (or at most < 2.0)
```

### Step 4: If Tests Fail

**If Hansen J fails (p < 0.10):**
1. Check instrument count (collapse=True?)
2. Remove potentially endogenous X variables
3. Reduce number of lags used

**If AR(2) fails (p < 0.10):**
1. Add more lags of dependent variable (lags=2)
2. Check for misspecification (omitted variables)
3. Consider deeper lags as instruments (minlags=3)

### Step 5: Compare Difference and System

```python
# Estimate both
diff_gmm = pb.DifferenceGMM(..., collapse=True).fit()
sys_gmm = pb.SystemGMM(..., collapse=True).fit()

# Compare
print("\nCoefficient Comparison:")
print(f"Difference GMM: {diff_gmm.params['y_lag1']:.3f} (SE: {diff_gmm.std_errors['y_lag1']:.3f})")
print(f"System GMM: {sys_gmm.params['y_lag1']:.3f} (SE: {sys_gmm.std_errors['y_lag1']:.3f})")

# Efficiency gain
se_reduction = (1 - sys_gmm.std_errors['y_lag1'] / diff_gmm.std_errors['y_lag1']) * 100
print(f"SE reduction: {se_reduction:.1f}%")

# Check System GMM assumptions
print(f"\nDifference-in-Hansen p-value: {sys_gmm.diff_hansen.pvalue:.3f}")
```

## Key Takeaways

🔑 **GMM is needed** when:
- Lagged dependent variable (dynamics)
- Fixed Effects creates Nickell bias
- Short panel (T < 20)

🔑 **Difference GMM:**
- First-difference to eliminate α_i
- Use lagged levels as instruments
- Moment condition: E[y_{i,t-s} Δε_it] = 0

🔑 **System GMM:**
- Add level equations to Difference GMM
- Use lagged differences as instruments for levels
- More efficient for persistent series
- Extra assumption: E[Δy_{i1} α_i] = 0

🔑 **Always collapse:**
- Avoid instrument proliferation
- Keep q/N < 1
- Instrument ratio < 2.0

🔑 **Diagnostic tests are mandatory:**
- Hansen J > 0.10 (instruments valid)
- AR(2) > 0.10 (no serial correlation)
- If tests fail, **do not use results!**

## Next Steps

**Hands-on learning:**

1. **[Tutorial 3: GMM Intro](../tutorials/03_gmm_intro.md)**: Practical GMM estimation

2. **[How-To: Interpret Tests](../how-to/interpret_tests.md)**: Test interpretation details

**Further reading:**

**Foundational papers:**
- **Arellano & Bond (1991)**: "Some Tests of Specification for Panel Data", *Review of Economic Studies*
- **Blundell & Bond (1998)**: "Initial Conditions and Moment Restrictions", *Journal of Econometrics*
- **Windmeijer (2005)**: "A Finite Sample Correction", *Journal of Econometrics*

**Practical guides:**
- **Roodman (2009)**: "How to do xtabond2", *The Stata Journal* (best practical guide)
- **Bond (2002)**: "Dynamic Panel Data Models: A Guide", *Portuguese Economic Journal*

**Textbooks:**
- **Baltagi (2021)**: *Econometric Analysis of Panel Data*, Chapter 8
- **Wooldridge (2010)**: *Cross Section and Panel Data*, Chapter 11
- **Arellano (2003)**: *Panel Data Econometrics* (advanced)

---

**GMM is powerful for dynamic panels, but requires careful diagnostic checking. Always use collapse, always check Hansen J and AR(2), and never report results if tests fail!**
