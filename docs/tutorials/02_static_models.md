# Tutorial 2: Static Panel Models

> Master the core panel data models: Pooled OLS, Fixed Effects, and Random Effects.

## What You'll Learn

In this tutorial, you will:

- Understand when Pooled OLS is insufficient
- Estimate Fixed Effects (FE) models
- Estimate Random Effects (RE) models
- Run and interpret the Hausman test
- Compare models systematically
- Choose the right model for your data

## Prerequisites

This tutorial assumes you've completed:
- [Tutorial 1: Getting Started](01_getting_started.md)

You should be familiar with:
- Loading panel data
- Basic PanelBox syntax
- Interpreting regression output

## The Problem with Pooled OLS

In Tutorial 1, we estimated a Pooled OLS model:

```python
import panelbox as pb

# Load data
data = pb.load_grunfeld()

# Pooled OLS
pooled = pb.PooledOLS("invest ~ value + capital", data, "firm", "year")
pooled_results = pooled.fit(cov_type='robust')
print(pooled_results.summary())
```

**Problem:** Pooled OLS assumes all firms are identical except for their observed characteristics (value, capital). But in reality:

- **Firm 1** might be well-managed (higher investment for same fundamentals)
- **Firm 2** might have conservative culture (lower investment)
- **Firm 3** might face different regulations

These **unobserved firm-specific factors** (α₁, α₂, α₃, ...) cause:

1. **Omitted variable bias** if they're correlated with X
2. **Inefficient estimates** even if uncorrelated

**Solution:** Use panel models that account for this heterogeneity!

## Fixed Effects: Controlling for Unobservables

### The Model

Fixed Effects (FE) adds firm-specific constants (α_i):

```
invest_it = α_i + β₁·value_it + β₂·capital_it + ε_it
```

where:
- `α_i` = firm-specific effect (captures management, culture, etc.)
- `β₁, β₂` = coefficients we want to estimate
- `ε_it` = idiosyncratic error

**Key idea:** Within transformation removes α_i by demeaning within each firm.

### Estimation

```python
# Fixed Effects
fe = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year",
    entity_effects=True,   # Include firm fixed effects (default)
    time_effects=False     # Exclude time fixed effects (can add if needed)
)

fe_results = fe.fit(cov_type='clustered')  # Cluster by firm
print(fe_results.summary())
```

**Output (typical):**
```
================================================================================
                       Fixed Effects Estimation Results
================================================================================
Dependent Variable:              invest        No. Observations:             200
Model:                     Fixed Effects        Df Residuals:                 188
Method:                           Within        Df Model:                       2
Date:                       2026-02-05          R-squared (within):         0.766
Time:                         15:30:00          R-squared (between):        0.892
Cov. Type:                   clustered          R-squared (overall):        0.812
================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
value             0.110      0.012      9.180      0.000       0.086       0.134
capital           0.310      0.052      5.962      0.000       0.208       0.412
================================================================================
F-statistic:                     120.5
Prob (F-statistic):           1.23e-31
Entity fixed effects:              Yes
Time fixed effects:                 No
Number of entities:                 10
Avg obs per entity:                 20
================================================================================
```

### Understanding the Output

**R-squared values:**
- **Within R² = 0.766**: Variation explained within firms over time
- **Between R² = 0.892**: Variation explained between firms
- **Overall R² = 0.812**: Total variation explained

**Coefficients:**
- `value = 0.110`: Within a firm, $1M increase in value → $0.11M more investment
- `capital = 0.310`: Within a firm, $1M increase in capital → $0.31M more investment

**Note:** Coefficients slightly different from Pooled OLS (0.116, 0.231) because we've removed bias from unobserved firm effects.

**Standard errors:**
- Clustered by firm to account for within-firm correlation over time
- More conservative than non-clustered SEs

### When to Use Fixed Effects

✅ **Use FE when:**
- Unobserved heterogeneity exists
- It's likely correlated with regressors
- You don't need to estimate time-invariant effects (gender, country, etc.)
- You have multiple observations per entity (T ≥ 2)

❌ **Don't use FE when:**
- You need to estimate time-invariant variables (they get dropped!)
- You have very few time periods (T = 2 is borderline)
- No unobserved heterogeneity exists

## Random Effects: Efficiency Gains

### The Model

Random Effects (RE) also includes entity-specific effects, but treats them as random draws:

```
invest_it = β₀ + β₁·value_it + β₂·capital_it + u_i + ε_it
```

where:
- `u_i ~ N(0, σ²_u)` = random firm effect
- `ε_it ~ N(0, σ²_ε)` = idiosyncratic error

**Key assumption:** `u_i` is **uncorrelated** with regressors (value, capital)

**Key advantage:** Can estimate time-invariant variables and has smaller standard errors than FE (if assumption holds).

### Estimation

```python
# Random Effects
re = pb.RandomEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year"
)

re_results = re.fit()
print(re_results.summary())
```

**Output (typical):**
```
================================================================================
                      Random Effects Estimation Results
================================================================================
Dependent Variable:              invest        No. Observations:             200
Model:                    Random Effects        Df Residuals:                 197
Method:                  GLS (Swamy-Arora)      Df Model:                       2
Date:                       2026-02-05          R-squared:                  0.812
Time:                         15:32:00
================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -57.834     28.899     -2.001      0.047    -114.566      -1.102
value             0.110      0.011     10.000      0.000       0.088       0.132
capital           0.308      0.049      6.286      0.000       0.212       0.404
================================================================================
Variance Components:
    Var(u_i):                    2341.62
    Var(ε_it):                    457.82
    Theta:                        0.862
================================================================================
```

### Understanding Random Effects Output

**Coefficients:**
- Very similar to FE (0.110 vs 0.110 for value)
- But we can estimate the intercept (unlike FE)

**Variance components:**
- `Var(u_i) = 2341.62`: Variance between firms
- `Var(ε_it) = 457.82`: Variance within firms over time
- `Theta = 0.862`: Quasi-demeaning parameter (θ ≈ 1 means close to FE)

**Standard errors:**
- Slightly smaller than FE (0.011 vs 0.012 for value)
- But only valid if E[u_i | X_it] = 0 holds!

### When to Use Random Effects

✅ **Use RE when:**
- Effects are uncorrelated with X (e.g., random sample from population)
- You need to estimate time-invariant variables
- You want more efficient estimates than FE
- Hausman test doesn't reject RE

❌ **Don't use RE when:**
- Effects are correlated with X (biased and inconsistent!)
- Hausman test rejects RE

## The Hausman Test: FE vs RE

### The Question

**Should I use Fixed Effects or Random Effects?**

The Hausman test answers this by testing:

- **H₀**: Random effects are consistent (E[u_i | X_it] = 0 holds)
- **H₁**: Random effects are inconsistent (use fixed effects)

**Decision rule:**
- **p < 0.05**: Reject H₀ → Use **Fixed Effects**
- **p ≥ 0.05**: Fail to reject → Use **Random Effects**

### Running the Test

```python
# Estimate both models
fe_results = pb.FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re_results = pb.RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# Hausman test
from panelbox.validation import HausmanTest

hausman = HausmanTest(fe_results, re_results)
print(hausman)
```

**Output:**
```
================================================================================
                            Hausman Test Results
================================================================================
Test statistic:                  2.3356
Degrees of freedom:                   2
P-value:                         0.3113
================================================================================
H0: Random Effects model is consistent
H1: Fixed Effects model is preferred

Decision: Fail to reject H0 (p = 0.3113)
Recommendation: Use Random Effects (more efficient)
================================================================================
```

### Interpreting the Result

**p = 0.3113 > 0.05** → Fail to reject H₀

**Interpretation:**
- No evidence that RE is inconsistent
- The difference between FE and RE coefficients is small (not statistically significant)
- RE assumption E[u_i | X_it] = 0 appears to hold

**Conclusion:** Use **Random Effects** (more efficient, smaller standard errors)

### What if Hausman Rejects?

If **p < 0.05**, you'd see:

```
P-value:                         0.0089
Decision: Reject H0 (p = 0.0089)
Recommendation: Use Fixed Effects (Random Effects is inconsistent)
```

**Interpretation:**
- RE coefficients significantly differ from FE
- RE assumption E[u_i | X_it] = 0 is violated
- Firm effects are correlated with regressors

**Conclusion:** Use **Fixed Effects** (consistent even if effects correlated with X)

## Comparing All Three Models

### Side-by-Side Comparison

```python
import pandas as pd

# Estimate all three
pooled_results = pb.PooledOLS("invest ~ value + capital", data, "firm", "year").fit()
fe_results = pb.FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re_results = pb.RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# Extract coefficients
comparison = pd.DataFrame({
    'Pooled OLS': pooled_results.params,
    'Fixed Effects': fe_results.params,
    'Random Effects': re_results.params
})

print(comparison)
```

**Output:**
```
           Pooled OLS  Fixed Effects  Random Effects
Intercept   -42.71400            NaN       -57.83400
value         0.11556        0.11006         0.11009
capital       0.23079        0.31002         0.30815
```

**Key observations:**
1. **Intercept**: Pooled and RE have intercepts; FE does not (absorbed by firm effects)
2. **value coefficient**: Similar across all three (0.116, 0.110, 0.110)
3. **capital coefficient**: Pooled OLS lower (0.231 vs 0.310) → omitted variable bias!

### Standard Error Comparison

```python
se_comparison = pd.DataFrame({
    'Pooled OLS': pooled_results.std_errors,
    'Fixed Effects': fe_results.std_errors,
    'Random Effects': re_results.std_errors
})

print(se_comparison)
```

**Output:**
```
           Pooled OLS  Fixed Effects  Random Effects
Intercept    14.53246            NaN        28.89900
value         0.00586        0.01199         0.01100
capital       0.02797        0.05200         0.04900
```

**Observation:** RE has smaller SEs than FE (more efficient), but larger than Pooled (which ignores heterogeneity).

## Complete Workflow: Model Selection

### Step 1: Estimate Pooled OLS (Baseline)

```python
pooled = pb.PooledOLS("invest ~ value + capital", data, "firm", "year")
pooled_results = pooled.fit()

print(f"Pooled R²: {pooled_results.rsquared:.4f}")
```

### Step 2: Test for Unobserved Heterogeneity

```python
fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

print(f"FE within R²: {fe_results.rsquared_within:.4f}")
print(f"FE overall R²: {fe_results.rsquared_overall:.4f}")

# If FE R² much higher → heterogeneity exists
```

**If FE R² ≈ Pooled R²:** No heterogeneity → Use Pooled OLS

**If FE R² >> Pooled R²:** Heterogeneity exists → Continue to Step 3

### Step 3: Hausman Test (FE vs RE)

```python
re = pb.RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

from panelbox.validation import HausmanTest
hausman = HausmanTest(fe_results, re_results)
print(hausman)
```

**If p ≥ 0.05:** Use **Random Effects** (efficient and consistent)

**If p < 0.05:** Use **Fixed Effects** (consistent, RE is biased)

### Step 4: Report Final Model

```python
# Assume Hausman favored RE
final_results = re_results

print("="*80)
print("FINAL MODEL: Random Effects")
print("="*80)
print(final_results.summary())

# Export to LaTeX
final_results.to_latex("investment_model.tex")
```

## Adding Time Fixed Effects

Sometimes you want to control for **time-varying shocks** affecting all firms (e.g., recession, policy changes):

```python
# Two-way fixed effects (entity + time)
fe_twoway = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year",
    entity_effects=True,   # Firm fixed effects
    time_effects=True      # Year fixed effects
)

fe_twoway_results = fe_twoway.fit(cov_type='twoway')  # Two-way clustering
print(fe_twoway_results.summary())
```

**When to use:**
- Time-varying aggregate shocks exist (business cycles, policy changes)
- You want to control for common trends
- Difference-in-differences designs

## Common Pitfalls

### ❌ Mistake 1: Using Pooled OLS when FE/RE is needed

**Problem:** Omitted variable bias

**Solution:** Always test for heterogeneity by comparing Pooled vs FE R²

### ❌ Mistake 2: Using RE when Hausman rejects

**Problem:** Biased and inconsistent estimates

**Solution:** Trust the Hausman test! If it rejects, use FE

### ❌ Mistake 3: Forgetting to cluster standard errors

**Problem:** Standard errors too small (over-rejection of H₀)

**Solution:** Always use `cov_type='clustered'` for FE

```python
# GOOD
fe_results = fe.fit(cov_type='clustered')

# BAD (SEs likely too small)
fe_results = fe.fit()
```

### ❌ Mistake 4: Expecting time-invariant variables in FE

**Problem:** Variables like firm size, country, gender get dropped

**Solution:** Use RE if you need time-invariant variables, or use Between estimator

### ❌ Mistake 5: Using FE with T = 2

**Problem:** No degrees of freedom for identifying dynamics

**Solution:** Need T ≥ 3 for meaningful FE; consider first-differences if T = 2

## Key Takeaways

✅ **You've learned to:**
- Identify when Pooled OLS is insufficient
- Estimate Fixed Effects and Random Effects models
- Run and interpret the Hausman test
- Follow a systematic model selection workflow
- Avoid common pitfalls

📊 **Model Selection Summary:**
1. **Pooled OLS**: No heterogeneity
2. **Fixed Effects**: Heterogeneity correlated with X
3. **Random Effects**: Heterogeneity uncorrelated with X

🧪 **Decision Tools:**
- Compare R²: Pooled vs FE
- Hausman test: FE vs RE

## Next Steps

**Continue learning:**

1. **[Tutorial 3: GMM Introduction](03_gmm_intro.md)**: Learn when and how to use dynamic panel models

2. **[How-To: Interpret Tests](../how-to/interpret_tests.md)**: Deep dive into diagnostic tests

3. **[Guide: Fixed vs Random Effects](../guides/fixed_vs_random.md)**: Detailed explanation of the differences

**Practice yourself:**

```python
# Try with your own data
import pandas as pd

data = pd.read_csv('my_panel_data.csv')

# Model selection workflow
pooled = pb.PooledOLS("y ~ x1 + x2", data, "entity_id", "time")
pooled_results = pooled.fit()

fe = pb.FixedEffects("y ~ x1 + x2", data, "entity_id", "time")
fe_results = fe.fit(cov_type='clustered')

re = pb.RandomEffects("y ~ x1 + x2", data, "entity_id", "time")
re_results = re.fit()

# Hausman test
from panelbox.validation import HausmanTest
hausman = HausmanTest(fe_results, re_results)
print(hausman)
```

## Further Reading

**Textbooks:**
- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data*, Chapter 10
- **Baltagi (2021)**: *Econometric Analysis of Panel Data*, Chapters 2-3
- **Cameron & Trivedi (2005)**: *Microeconometrics*, Chapter 21

**Papers:**
- **Hausman (1978)**: "Specification Tests in Econometrics", *Econometrica*, 46(6), 1251-1271
- **Mundlak (1978)**: "On the Pooling of Time Series and Cross Section Data", *Econometrica*, 46(1), 69-85

---

**Congratulations!** You now understand the core static panel models and how to choose between them. 🎉

Ready for dynamics and endogeneity? Continue to [Tutorial 3: GMM Introduction](03_gmm_intro.md).
