# Fixed Effects vs Random Effects: A Deep Dive

> Detailed comparison of Fixed Effects and Random Effects models with theory, intuition, and practical guidance.

## Overview

The choice between Fixed Effects (FE) and Random Effects (RE) is one of the most important decisions in panel data analysis. This guide provides:

- **Mathematical foundations** of both models
- **Intuition** for when each is appropriate
- **Theoretical comparison** of assumptions and properties
- **Practical guidance** for applied work

## The Models

### Fixed Effects (Within Estimator)

**Model:**
```
y_it = α_i + X_it'β + ε_it
```

**Where:**
- `α_i` = entity-specific **fixed effect** (constant for entity i)
- `X_it` = regressors (can vary over i and t)
- `β` = coefficients of interest
- `ε_it` = idiosyncratic error

**Key feature:** `α_i` can be **correlated** with `X_it`

**Estimation:** Within transformation (demeaning)

```
(y_it - ȳ_i) = (X_it - X̄_i)'β + (ε_it - ε̄_i)
```

### Random Effects (GLS Estimator)

**Model:**
```
y_it = β₀ + X_it'β + u_i + ε_it
```

**Where:**
- `u_i ~ N(0, σ²_u)` = entity-specific **random effect**
- `ε_it ~ N(0, σ²_ε)` = idiosyncratic error
- `u_i ⊥ ε_it` (independent)

**Key assumption:** `u_i ⊥ X_it` (uncorrelated)

**Estimation:** Generalized Least Squares (GLS)

```
(y_it - θȳ_i) = β₀(1 - θ) + (X_it - θX̄_i)'β + error
```

where `θ = 1 - √(σ²_ε / (σ²_ε + Tσ²_u))`

## Fundamental Difference

The **critical distinction** is the correlation assumption:

| Model | Assumption | Interpretation |
|-------|------------|----------------|
| **Fixed Effects** | E[α_i \| X_it] ≠ 0 allowed | Effects **correlated** with X |
| **Random Effects** | E[u_i \| X_it] = 0 required | Effects **uncorrelated** with X |

**Example (firm profitability):**

Suppose unobserved `α_i` = "management quality"

**FE allows:** Good managers choose higher investment (correlation)

**RE requires:** Management quality independent of investment (unlikely!)

**Implication:** FE is consistent in both cases; RE only if assumption holds

## Mathematical Details

### Fixed Effects Estimation

**Step 1: Within transformation**

For each variable, subtract entity mean:

```
ỹ_it = y_it - ȳ_i
X̃_it = X_it - X̄_i
```

**Step 2: OLS on demeaned data**

```
β̂_FE = (Σ_i Σ_t X̃_it X̃_it')^(-1) (Σ_i Σ_t X̃_it ỹ_it)
```

**Properties:**
- Consistent even if `α_i` correlated with `X_it`
- Asymptotically normal as N → ∞ (with fixed T)
- Inefficient if `α_i ⊥ X_it` (larger SEs than RE)

**Loss:** Cannot estimate time-invariant variables (they get absorbed)

### Random Effects Estimation

**Step 1: Estimate variance components**

Using ANOVA, Swamy-Arora, or maximum likelihood:

```
σ̂²_ε = (1/N(T-1)) Σ_i Σ_t ε̂²_it  (within residuals)
σ̂²_u = (1/N) Σ_i (ū_i² - σ̂²_ε/T)  (between - within)
```

**Step 2: Compute θ**

```
θ̂ = 1 - √(σ̂²_ε / (σ̂²_ε + Tσ̂²_u))
```

**Step 3: Quasi-demean and estimate**

```
y*_it = y_it - θ̂ȳ_i
X*_it = X_it - θ̂X̄_i

β̂_RE = (Σ_i Σ_t X*_it X*_it')^(-1) (Σ_i Σ_t X*_it y*_it)
```

**Properties:**
- Consistent **only if** `u_i ⊥ X_it`
- More efficient than FE (smaller SEs) when assumption holds
- Can estimate time-invariant variables

**Interpretation of θ:**

- `θ = 0`: No entity effects (σ²_u = 0) → Pooled OLS
- `θ = 1`: All variation is between-entity → Fixed Effects
- `0 < θ < 1`: Partial quasi-demeaning (typical)

## Assumptions Comparison

### Fixed Effects Assumptions

**Strict exogeneity:**
```
E[ε_it | X_i1, ..., X_iT, α_i] = 0
```

For all t and s: errors uncorrelated with **all** X's

**No correlation assumption for α_i:**
- `α_i` can correlate with `X_it` (key advantage!)
- `α_i` captures all time-invariant confounders

**Homoskedasticity (for efficiency, not consistency):**
```
Var(ε_it | X_i, α_i) = σ²_ε
```

**No serial correlation (for standard SEs):**
```
E[ε_it ε_is | X_i, α_i] = 0  for t ≠ s
```

### Random Effects Assumptions

**All FE assumptions plus:**

**Orthogonality of random effect:**
```
E[u_i | X_it] = 0  for all i, t
```

This is **very restrictive!**

**Random effect homoskedasticity:**
```
Var(u_i) = σ²_u  (constant across i)
```

**No correlation between u_i and X_it:**

This is the **key additional assumption** that makes RE stronger than FE.

## When Each Assumption Holds

### FE Orthogonality Holds When:

✅ **Fixed T, no dynamics:**
- No lagged dependent variables
- X's are strictly exogenous

✅ **Example:**
- Wage regression: education, experience (predetermined)
- No feedback from current wage to past education

### FE Orthogonality Fails When:

❌ **Lagged dependent variable:**
```
y_it = γ y_i,t-1 + X_it'β + α_i + ε_it
```
- `y_i,t-1` correlated with `(ε_it - ε̄_i)` → Nickell bias
- Solution: Use GMM

❌ **Feedback effects:**
- Current shock affects future X
- Example: Firm profit shock → affects next year's investment

### RE Orthogonality Holds When:

✅ **Random sampling from population:**
- Entities are random draws
- Example: Survey of individuals from general population

✅ **No selection:**
- Unobserved `u_i` is not related to why entity is in sample

### RE Orthogonality Fails When:

❌ **Omitted variable bias:**
- Any time-invariant factor correlated with X
- Example: Ability correlated with education

❌ **Common in practice:**
- Management quality → investment choices
- Individual preferences → consumption choices
- Institutions → policy choices

## Efficiency Comparison

### When Both Are Consistent (RE assumption holds)

**Variance comparison:**

```
Var(β̂_RE) ≤ Var(β̂_FE)
```

**Why RE is more efficient:**

1. **Uses between-entity variation:**
   - FE only uses within-entity variation (over time)
   - RE uses both within and between

2. **Example:**
   - FE: How does X affect Y within firm i over time?
   - RE: How does X affect Y within **and** across firms?

**Efficiency gain:** Typically 10-40% reduction in standard errors

### When RE Is Inconsistent (assumption fails)

**Bias vs Efficiency trade-off:**

- **RE:** Smaller SEs but **biased** estimates
- **FE:** Larger SEs but **consistent** estimates

**Decision:** Always prefer consistency over efficiency

**Rule:** Use Hausman test to decide

## The Hausman Test

### Purpose

Test whether `E[u_i | X_it] = 0` holds

### Intuition

- **FE is always consistent** (robust to correlation)
- **RE is consistent only if** `u_i ⊥ X_it`

**If both are consistent:** Estimates should be similar

**If RE is inconsistent:** Estimates will differ systematically

### Test Statistic

```
H = (β̂_FE - β̂_RE)' [Var(β̂_FE) - Var(β̂_RE)]^(-1) (β̂_FE - β̂_RE)
```

**Under H₀:** `H ~ χ²(K)` where K = number of coefficients

### Decision Rule

| p-value | Interpretation | Recommendation |
|---------|----------------|----------------|
| p < 0.05 | Reject H₀ | **Use FE** (RE is inconsistent) |
| p ≥ 0.05 | Fail to reject | **Use RE** (more efficient) |

### Example

```python
import panelbox as pb
from panelbox.validation import HausmanTest

fe = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year").fit()
re = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year").fit()

hausman = HausmanTest(fe, re)
print(hausman)
```

**Output:**
```
Hausman Test: χ² = 15.67, p = 0.0004
Decision: Reject H₀ → Use Fixed Effects
```

**Interpretation:** RE assumption violated → FE is preferred

## Practical Guidance

### Prefer Fixed Effects When:

✅ **Applied microeconomics:**
- Firms, individuals, households
- Unobserved heterogeneity likely correlated with X

✅ **Not a random sample:**
- Selection bias
- Specific set of entities (e.g., Fortune 500 firms)

✅ **Time-invariant variables not of interest:**
- Focus is on time-varying effects
- OK to lose constant characteristics

✅ **Conservative approach:**
- FE is robust to correlation
- "Safest" choice

### Prefer Random Effects When:

✅ **Random sample from population:**
- Survey data with random sampling
- Cross-country with representative selection

✅ **Time-invariant variables are key:**
- Gender, race, country fixed characteristics
- Need to estimate their effects

✅ **Hausman test supports RE:**
- p > 0.10
- No evidence of correlation

✅ **Efficiency matters:**
- Small sample, large standard errors
- RE provides tighter confidence intervals

### Mundlak Approach (Hybrid)

**Problem:** Want RE efficiency but worried about correlation

**Solution:** Correlated Random Effects (Mundlak 1978)

**Model:**
```
y_it = β₀ + X_it'β + X̄_i'γ + u_i + ε_it
```

**Include entity means** `X̄_i` as regressors

**Properties:**
- If `γ = 0`: No correlation → Standard RE
- If `γ ≠ 0`: Controls for correlation
- Allows time-invariant variables
- Can test for correlation

**In PanelBox:**

```python
# Create entity means
data['x1_mean'] = data.groupby('firm')['x1'].transform('mean')

# Mundlak model
re_mundlak = pb.RandomEffects(
    "y ~ x1 + x1_mean",
    data, "firm", "year"
).fit()

# Test γ = 0
# If significant → correlation exists
```

## Common Scenarios

### Scenario 1: Wage Determination

**Setup:** Individual wages over time

**Model:** wage_it = education_it + experience_it + ...

**Unobserved:** Ability (α_i)

**Question:** Is ability correlated with education?

**Answer:** Almost certainly YES (able people get more education)

**Conclusion:** **Use Fixed Effects**

### Scenario 2: Country Growth

**Setup:** GDP growth across 100+ countries

**Model:** growth_it = investment_it + institutions_i + ...

**Unobserved:** Geography, culture (u_i)

**Question:** Are institutions time-invariant and of interest?

**Answer:** YES, and likely random sample of countries

**Conclusion:** **Use Random Effects** (can estimate institution effects)

### Scenario 3: Firm Investment

**Setup:** Investment decisions of S&P 500 firms

**Model:** invest_it = cash_flow_it + debt_it + ...

**Unobserved:** Management quality (α_i)

**Question:** Do good managers have different cash flows?

**Answer:** Probably (selection into S&P 500)

**Conclusion:** **Use Fixed Effects** (not a random sample)

### Scenario 4: School Performance

**Setup:** Test scores across schools over time

**Model:** score_it = class_size_it + funding_it + ...

**Unobserved:** School quality, neighborhood (α_i)

**Question:** Does school quality affect class size choice?

**Answer:** Likely (better schools attract more students)

**Conclusion:** **Use Fixed Effects**, or run Hausman test

## Comparison Table

| Feature | Fixed Effects | Random Effects |
|---------|---------------|----------------|
| **Assumption** | E[α_i \| X_it] unrestricted | E[u_i \| X_it] = 0 required |
| **Consistency** | Always (if strict exogeneity) | Only if orthogonality holds |
| **Efficiency** | Less efficient | More efficient (if consistent) |
| **Time-invariant X** | Cannot estimate | Can estimate |
| **Interpretation** | Within-entity effects | Weighted within/between |
| **Typical use** | Micro (firms, individuals) | Macro (countries), surveys |
| **Sample** | Any | Preferably random |
| **Robustness** | Very robust | Sensitive to violations |

## Summary Workflow

```
START: Panel data with entity-specific effects

    ↓

Q1: Do you NEED to estimate time-invariant variables?

    YES → Consider Random Effects (run Hausman test)
    NO → Continue

    ↓

Q2: Is sample a random draw from population?

    YES → Consider Random Effects (run Hausman test)
    NO → Prefer Fixed Effects

    ↓

Q3: Run Hausman Test

    p < 0.05 → Use Fixed Effects
    p ≥ 0.05 → Use Random Effects

    ↓

DECISION MADE
```

## Key Takeaways

🔑 **Core difference:** Correlation assumption
- FE allows correlation between α_i and X_it
- RE requires no correlation

🔑 **Trade-off:** Consistency vs Efficiency
- FE: Consistent but less efficient
- RE: More efficient but only if assumption holds

🔑 **Practical rule:**
- **When in doubt, use Fixed Effects** (safer)
- Only use RE if Hausman test supports it

🔑 **Hausman test is your friend:**
- Let the data decide
- Don't pre-commit to one model

## Next Steps

**Learn more:**

1. **[Tutorial 2: Static Models](../tutorials/02_static_models.md)**: Hands-on FE vs RE

2. **[How-To: Interpret Tests](../how-to/interpret_tests.md)**: Hausman test details

3. **[How-To: Choose Model](../how-to/choose_model.md)**: Decision flowchart

**Advanced topics:**
- Correlated Random Effects (Mundlak, Chamberlain)
- Hausman-Taylor estimator (IV for RE)
- Clustered standard errors for both FE and RE

**Further reading:**

- **Wooldridge (2010)**, Chapter 10: Comprehensive treatment
- **Hausman (1978)**: Original specification test paper
- **Mundlak (1978)**: Correlated random effects
- **Baltagi (2021)**, Chapters 2-3: Detailed comparison

---

**Remember: The choice between FE and RE is fundamentally about whether unobserved effects are correlated with your regressors. When in doubt, FE is the conservative choice.**
