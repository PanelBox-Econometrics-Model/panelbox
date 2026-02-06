# How to Choose the Right Panel Model

> Decision guide for selecting the appropriate estimator for your data and research question.

## Quick Decision Tree

```
START: Do you have panel data (entities × time)?
│
├─ NO → Use cross-sectional or time-series methods
│
└─ YES → Continue below
    │
    ├─ Q1: Does y_it depend on y_i,t-1 (lagged dependent)?
    │   │
    │   ├─ YES → **Use GMM** (go to GMM Decision Tree)
    │   │
    │   └─ NO → Q2: Is there unobserved heterogeneity?
    │       │
    │       ├─ NO → **Pooled OLS**
    │       │
    │       └─ YES → Q3: Are effects correlated with X?
    │           │
    │           ├─ YES → **Fixed Effects**
    │           │
    │           ├─ NO → **Random Effects**
    │           │
    │           └─ UNSURE → Run **Hausman Test**
    │               ├─ p < 0.05 → **Fixed Effects**
    │               └─ p ≥ 0.05 → **Random Effects**
```

## Detailed Decision Guide

### Step 1: Check for Dynamics

**Question:** Does your dependent variable depend on its past values?

**Examples of dynamic models:**
- Employment depends on past employment
- Investment depends on past investment
- GDP growth depends on past growth

**Decision:**
- **YES** → Use GMM (Difference or System GMM)
- **NO** → Continue to Step 2

**Why GMM?**
Including y_{t-1} as a regressor creates correlation with the error term. Fixed Effects and Random Effects are biased in this case. GMM uses instruments to handle this endogeneity.

### Step 2: Test for Unobserved Heterogeneity

**Question:** Are there entity-specific factors you can't observe but matter?

**Examples:**
- Firm "culture" or management quality
- Individual "ability" or preferences
- Country institutions or geography

**How to test:**
1. Estimate both Pooled OLS and Fixed Effects
2. Compare R-squared values
3. Run F-test for fixed effects

```python
import panelbox as pb

# Pooled OLS
pooled = pb.PooledOLS("y ~ x1 + x2", data, "entity", "time")
pooled_results = pooled.fit()

# Fixed Effects
fe = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
fe_results = fe.fit()

# Compare R-squared
print(f"Pooled R²: {pooled_results.rsquared:.4f}")
print(f"FE R²: {fe_results.rsquared_within:.4f}")

# If FE R² much higher → unobserved heterogeneity exists
```

**Decision:**
- **NO heterogeneity** → **Pooled OLS**
- **YES heterogeneity** → Continue to Step 3

### Step 3: Hausman Test (FE vs RE)

**Question:** Are the entity-specific effects correlated with your regressors?

**Run Hausman test:**

```python
# Estimate both models
fe = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
re = pb.RandomEffects("y ~ x1 + x2", data, "entity", "time")

fe_results = fe.fit()
re_results = re.fit()

# Hausman test
hausman = pb.HausmanTest(fe_results, re_results)
print(hausman)
```

**Interpretation:**

- **p < 0.05** (reject H0) → Use **Fixed Effects**
  - Effects are correlated with X
  - RE is biased and inconsistent

- **p ≥ 0.05** (fail to reject) → Use **Random Effects**
  - No evidence of correlation
  - RE is more efficient (smaller standard errors)

## Model Comparison Table

| Model | Use When | Assumptions | Pros | Cons |
|-------|----------|-------------|------|------|
| **Pooled OLS** | No heterogeneity, no dynamics | - Strict exogeneity<br>- No unobserved effects | - Simple<br>- Efficient | - Biased if heterogeneity<br>- Ignores panel structure |
| **Fixed Effects** | Unobserved heterogeneity correlated with X | - Strict exogeneity<br>- E[α_i X_it] ≠ 0 allowed | - Consistent<br>- Controls unobserved factors | - Can't estimate time-invariant effects<br>- Less efficient than RE |
| **Random Effects** | Unobserved heterogeneity uncorrelated with X | - E[α_i X_it] = 0 | - Can estimate time-invariant<br>- More efficient | - Inconsistent if E[α_i X_it] ≠ 0<br>- Stronger assumptions |
| **Difference GMM** | Dynamics + short T | - Lagged dependent<br>- E[y_{t-s} Δε_t] = 0 | - Handles dynamics<br>- Allows endogeneity | - Weak instruments if persistent<br>- Requires T ≥ 3 |
| **System GMM** | Dynamics + persistent series | - Same as Diff GMM<br>- E[Δy_{i1} η_i] = 0 | - More efficient<br>- Better for persistent | - Extra assumption<br>- More complex |

## GMM Decision Tree

If your model has dynamics (lagged dependent variable):

```
GMM DECISION TREE
│
├─ Q1: How persistent is your series?
│   │
│   ├─ Highly persistent (AR coef > 0.8)
│   │   → **System GMM**
│   │       - Lagged levels are weak instruments
│   │       - Need additional moment conditions
│   │
│   └─ Moderately persistent (AR coef < 0.8)
│       → **Difference GMM**
│           - Sufficient instrument strength
│           - Fewer assumptions
│
├─ Q2: Panel starts at event time? (firm entry, policy change)
│   │
│   ├─ YES → **Difference GMM**
│   │       - Initial conditions assumption violated
│   │
│   └─ NO → Can use either (System GMM more efficient)
│
└─ Q3: How many time periods?
    │
    ├─ T < 5 → **Difficult** (consider static FE)
    ├─ 5 ≤ T < 10 → **Difference or System GMM**
    └─ T ≥ 10 → **Either works well**
```

## Checklist Approach

Use this checklist to narrow down your choice:

### [ ] **Pooled OLS**
- [ ] No unobserved entity-specific effects
- [ ] All entities are homogeneous
- [ ] Errors are i.i.d. across entities and time
- [ ] Just want a baseline/benchmark

### [ ] **Fixed Effects**
- [ ] Unobserved heterogeneity exists
- [ ] Effects likely correlated with regressors
- [ ] Don't need to estimate time-invariant effects
- [ ] Have T ≥ 2 observations per entity
- [ ] Hausman test rejects Random Effects

### [ ] **Random Effects**
- [ ] Unobserved heterogeneity exists
- [ ] Effects uncorrelated with regressors
- [ ] Want to estimate time-invariant effects (e.g., gender, geography)
- [ ] Sample is random draw from population
- [ ] Hausman test supports Random Effects

### [ ] **Difference GMM**
- [ ] Dependent variable depends on its lag
- [ ] Short panel (small T, large N)
- [ ] Series not highly persistent
- [ ] Panel may start at "event time"
- [ ] Strict exogeneity fails

### [ ] **System GMM**
- [ ] Dependent variable depends on its lag
- [ ] Series is highly persistent (ρ > 0.8)
- [ ] Panel is stationary (not starting at event)
- [ ] Want most efficient estimates
- [ ] Can justify initial conditions assumption

## Common Scenarios

### Scenario 1: Firm Investment

**Setup:** Studying firm investment decisions
- **N = 500 firms**, **T = 10 years**
- Variables: Investment, sales, debt, size

**Decision:**
1. Investment likely depends on past investment → Dynamic
2. Firms differ in unmeasured ways (management, culture)
3. These likely correlate with sales/debt

**Choice:** **System GMM** (dynamic + persistent) or **Difference GMM** (conservative)

### Scenario 2: Wage Determination

**Setup:** Worker wages over career
- **N = 5,000 individuals**, **T = 5 years**
- Variables: Wage, experience, education, industry

**Decision:**
1. No strong dynamic component (wage_{t-1} doesn't directly cause wage_t)
2. Individual "ability" is unobserved and correlated with education
3. Want to control for this

**Choice:** **Fixed Effects**

### Scenario 3: Country Growth

**Setup:** Economic growth across countries
- **N = 100 countries**, **T = 40 years**
- Variables: GDP growth, investment, education, institutions

**Decision:**
1. Growth may depend on past growth → Dynamic
2. Institutions are time-invariant and important
3. Want to estimate institutional effects

**Choice:** **Random Effects** if no dynamics, **System GMM** if including lagged growth

## Testing Your Choice

After selecting a model, validate your choice:

### 1. **Specification Tests**

```python
# Run diagnostic tests
results = model.fit()

# Check residuals
results.plot_residuals()  # Should be random

# Heteroskedasticity test
from panelbox.validation import BreuschPaganTest
bp_test = BreuschPaganTest(results)
print(bp_test)  # p > 0.05 is good
```

### 2. **Robustness Checks**

- **Try alternative models** and compare
- **Add/remove variables** to check stability
- **Different subsamples** (time periods, entity groups)
- **Different standard errors** (robust, clustered)

### 3. **For GMM: Diagnostic Tests**

```python
# Hansen J-test (overidentification)
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f}")
# p > 0.10 is good

# AR(2) test (no serial correlation)
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f}")
# p > 0.10 is good

# Instrument ratio
print(f"Instrument ratio: {results.instrument_ratio:.2f}")
# < 1.0 is good
```

## Common Mistakes

❌ **Using Pooled OLS when FE/RE is needed**
- Leads to omitted variable bias
- Standard errors are too small (over-rejection)

❌ **Using Random Effects when Fixed Effects is correct**
- Estimates are biased and inconsistent
- Hausman test will reject

❌ **Using Fixed Effects for dynamics**
- Nickell bias (biased estimates)
- Use GMM instead

❌ **Using too many instruments in GMM**
- Overfitting (instrument proliferation)
- Always use `collapse=True`

❌ **Ignoring diagnostic tests**
- GMM tests tell you if estimates are valid
- Don't skip Hansen J and AR(2) tests!

## Summary Flowchart

```
┌─────────────────────────────────────┐
│  Do you have lagged dependent var?  │
└──────────┬──────────────────────────┘
           │
     ┌─────┴─────┐
     │           │
    YES         NO
     │           │
     v           v
  ┌──────┐   ┌──────────┐
  │ GMM  │   │ Static   │
  │      │   │ Models   │
  └──┬───┘   └────┬─────┘
     │            │
     │            v
     │       ┌────────────┐
     │       │ Hausman    │
     │       │ Test       │
     │       └──┬─────┬───┘
     │          │     │
     │          │     v
     │          │  ┌──────┐
     │          │  │  RE  │
     │          │  └──────┘
     │          v
     │       ┌──────┐
     │       │  FE  │
     │       └──────┘
     v
┌──────────────┐
│ Persistent?  │
└──┬────────┬──┘
   │        │
System    Difference
 GMM        GMM
```

## Further Reading

- **Wooldridge (2010)** Chapter 10-11: FE vs RE, dynamics
- **Baltagi (2021)** Chapter 2-8: All static models
- **Arellano & Bond (1991)**: Difference GMM
- **Blundell & Bond (1998)**: System GMM
- **Roodman (2009)**: Practical guide to GMM

## Need Help?

Still unsure which model to use?

1. **[Static Models Tutorial](../tutorials/02_static_models.md)**: Compare FE, RE, Pooled
2. **[GMM Tutorial](../tutorials/03_gmm_intro.md)**: Learn when GMM is needed
3. **[GitHub Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions)**: Ask the community

---

**Remember:** The right model depends on your data structure, research question, and the assumptions you're willing to make. When in doubt, estimate multiple models and compare!
