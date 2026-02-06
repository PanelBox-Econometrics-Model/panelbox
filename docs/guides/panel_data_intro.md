# Introduction to Panel Data

> Understanding panel data structure, advantages, and when to use panel methods.

## What is Panel Data?

Panel data (also called **longitudinal data** or **cross-sectional time-series data**) combines two dimensions:

1. **Cross-sectional**: Multiple entities (firms, individuals, countries, etc.)
2. **Time-series**: Each entity observed over multiple time periods

**Example:**

```
   firm  year   sales  profit
0     1  2020   100.0    10.0
1     1  2021   120.0    15.0
2     1  2022   135.0    18.0
3     2  2020    80.0     8.0
4     2  2021    85.0     9.0
5     2  2022    90.0    10.0
```

**Structure:**
- **N = 2 firms** (cross-sectional units)
- **T = 3 years** (time periods)
- **N × T = 6 observations**

## Panel Data vs Other Data Types

### Cross-Sectional Data

**One** observation per entity at **one** point in time:

```
   firm   sales  profit
0     1   120.0    15.0
1     2    85.0     9.0
2     3   150.0    20.0
```

- **N entities, T = 1**
- Cannot study dynamics (changes over time)
- Cannot control for unobserved time-invariant characteristics

### Time-Series Data

**One** entity observed over **multiple** time periods:

```
   year   gdp  inflation
0  2020  20.5       2.1
1  2021  21.2       2.5
2  2022  21.8       3.2
```

- **N = 1 entity, T periods**
- Can study dynamics
- Cannot control for cross-sectional heterogeneity

### Panel Data

**Multiple** entities over **multiple** time periods:

```
   country  year   gdp  inflation
0      USA  2020  20.5       2.1
1      USA  2021  21.2       2.5
2      CAN  2020  1.65       0.7
3      CAN  2021  1.72       1.2
```

- **N entities, T periods**
- Can study both dynamics AND control for heterogeneity
- **Best of both worlds!**

## Types of Panel Data

### Balanced Panel

Every entity observed in **all** time periods.

**Example:**

```
   firm  year  sales
0     1  2020    100
1     1  2021    120  ← Firm 1: all 3 years ✓
2     1  2022    135
3     2  2020     80
4     2  2021     85  ← Firm 2: all 3 years ✓
5     2  2022     90
```

**Characteristics:**
- All N entities have exactly T observations
- Total observations = N × T
- Easier to analyze (some methods require balance)

### Unbalanced Panel

Some entities missing in some periods.

**Example:**

```
   firm  year  sales
0     1  2020    100
1     1  2021    120  ← Firm 1: only 2 years
2     2  2020     80
3     2  2021     85
4     2  2022     90  ← Firm 2: all 3 years
5     3  2021    200
6     3  2022    215  ← Firm 3: only 2 years
```

**Characteristics:**
- Different T_i for different entities
- Total observations < N × T_max
- Common in practice (attrition, entry/exit)
- PanelBox handles unbalanced panels automatically

### Short vs Long Panels

**Short (wide) panel:** Large N, small T
- Example: N = 10,000 individuals, T = 5 years
- Typical in microeconometrics (firm, household data)
- Asymptotic theory: N → ∞, T fixed
- **Use:** Fixed Effects, Random Effects, GMM

**Long (narrow) panel:** Small N, large T
- Example: N = 50 countries, T = 60 years
- Typical in macroeconomics
- Asymptotic theory: T → ∞ (or both N,T → ∞)
- **Use:** Panel cointegration, panel VARs, unit root tests

## Advantages of Panel Data

### 1. Control for Unobserved Heterogeneity

**Problem (cross-sectional):** Omitted variable bias

Example: Estimate effect of education on wages

```
wage_i = β₀ + β₁·education_i + ε_i
```

**Omitted:** "Ability" (unobserved, correlated with education)

**Bias:** β̂₁ overestimates education effect

---

**Solution (panel):** Individual fixed effects

```
wage_it = α_i + β₁·education_it + ε_it
```

Where `α_i` captures time-invariant ability.

**Result:** Consistent estimate of β₁ (controls for ability)

### 2. More Degrees of Freedom

**Cross-sectional:** N = 500 observations

**Panel:** N = 500 firms, T = 10 years → 5,000 observations

**Benefits:**
- More precise estimates (smaller standard errors)
- Can estimate more parameters
- Better power for hypothesis tests

### 3. Study Dynamics

**Panel allows:** Including lagged dependent variables

```
y_it = γ·y_i,t-1 + β·x_it + α_i + ε_it
```

**Examples:**
- Investment depends on past investment (habit formation)
- Current health depends on past health (state dependence)
- GDP growth exhibits persistence

**Cross-sectional data:** Cannot estimate γ

### 4. Identify Effects Better

**With panel data, you can identify:**

- **Within-entity effects:** How does X affect Y **within** the same firm over time?
- **Between-entity effects:** How do differences in X across firms relate to Y?

**Example:** Effect of firm size on productivity

- **Within:** As a firm grows, does productivity increase?
- **Between:** Are larger firms more productive than smaller ones?

These can differ! Panel data lets you distinguish them.

### 5. Reduce Collinearity

**Problem (cross-sectional):** Two variables highly correlated

**Panel:** Variation **within** entities over time may differ from variation **between** entities

**Example:**
- Cross-sectionally: Education and income highly correlated
- Within-person over time: Education changes slowly, income fluctuates

**Result:** Better identification of separate effects

## Disadvantages and Challenges

### 1. Data Collection Costs

**Panel data requires:**
- Tracking same entities over time
- Consistent measurement across periods
- Dealing with attrition (entities leaving sample)

**Cost:** More expensive than single cross-section

### 2. Attrition Bias

**Problem:** Entities drop out non-randomly

**Example:** Less profitable firms exit market

**Bias:** Surviving firms are systematically different

**Solutions:**
- Attrition correction models
- Selection models (Heckman)
- Inverse probability weighting

### 3. Limited Time Variation

**Problem:** Some variables don't change much over time

**Example:** Education, gender, country of birth

**Issue:** Fixed Effects **drops** time-invariant variables

**Trade-off:** Can't estimate effect of time-invariant X with FE

### 4. Short Panels and Bias

**Problem:** With small T, Fixed Effects has bias (Nickell bias)

**Magnitude:** O(1/T), so T < 10 can be problematic

**Solution:** Use GMM estimators instead

### 5. Complex Analysis

**Panel methods are more complex:**
- Need to choose between Pooled OLS, FE, RE, GMM
- More diagnostic tests required
- More assumptions to verify
- Computational burden (large N × T)

## Panel Data Notation

### Standard Notation

**Entity index:** i = 1, 2, ..., N

**Time index:** t = 1, 2, ..., T (or T_i if unbalanced)

**Observation:** y_it (outcome for entity i at time t)

**Regressor:** X_it (can be scalar or vector)

### Common Subscript Conventions

| Symbol | Meaning | Example |
|--------|---------|---------|
| y_it | Observation i at time t | sales_it |
| y_i,t-1 | Observation i at time t-1 (lag) | sales_i,t-1 |
| ȳ_i | Entity i mean over time | ȳ_i = (1/T)Σ_t y_it |
| ȳ_t | Time t mean over entities | ȳ_t = (1/N)Σ_i y_it |
| ȳ | Grand mean | ȳ = (1/NT)Σ_i Σ_t y_it |

### Panel Transformations

**Within (demeaning):**
```
ỹ_it = y_it - ȳ_i
```
Removes entity-specific means (used in Fixed Effects)

**Between (entity means):**
```
ȳ_i = (1/T_i)Σ_t y_it
```
Cross-sectional regression of entity means

**First-difference:**
```
Δy_it = y_it - y_i,t-1
```
Removes time-invariant effects (used in Difference GMM)

## When to Use Panel Methods

### Use Panel Methods When:

✅ You have multiple entities observed over multiple periods

✅ Unobserved entity-specific effects likely exist

✅ These effects may be correlated with regressors

✅ You want to study dynamics or changes over time

✅ You want more precise estimates (more data)

### Use Cross-Sectional Methods When:

❌ Only one time period available

❌ No unobserved heterogeneity

❌ Pooled OLS is sufficient (Breusch-Pagan test confirms)

❌ Time dimension is irrelevant to research question

### Use Time-Series Methods When:

❌ Only one entity

❌ Focus is on aggregate dynamics, forecasting

❌ Interested in unit roots, cointegration for single series

## Common Applications

### Microeconometrics

**Labor economics:**
- Wage determination over worker careers
- Effect of education/training on earnings
- Employment dynamics

**Corporate finance:**
- Firm investment decisions
- Capital structure choices
- Dividend policy

**Industrial organization:**
- Firm productivity evolution
- Market entry/exit
- Price dynamics

### Macroeconomics

**Growth:**
- Determinants of economic growth across countries
- Convergence testing
- Institutions and development

**Trade:**
- Gravity models of bilateral trade
- Effects of trade agreements
- Exchange rate effects

**Public finance:**
- Tax competition between jurisdictions
- Government spending effects
- Fiscal policy effectiveness

### Health Economics

**Individual health:**
- Health production functions
- Effect of insurance on utilization
- Lifestyle choices and outcomes

**Hospital performance:**
- Quality and efficiency over time
- Policy interventions
- Technology adoption

### Development Economics

**Households:**
- Poverty dynamics
- Consumption smoothing
- Migration decisions

**Villages/regions:**
- Development program effects
- Infrastructure impact
- Climate shocks

## Example: Why Panel Data Matters

### Cross-Sectional Analysis (Wrong)

**Data:** 100 firms in 2022

**Model:**
```
profit_i = β₀ + β₁·size_i + ε_i
```

**Result:** β̂₁ = 0.15 (larger firms more profitable)

**Problem:** Omitted variable bias!
- Management quality (α_i) affects both size and profit
- Correlation between α_i and size biases β̂₁ upward

### Panel Analysis (Correct)

**Data:** Same 100 firms, 2015-2022 (T=8 years)

**Model:**
```
profit_it = α_i + β₁·size_it + ε_it
```

**Result:** β̂₁ = 0.08 (smaller effect after controlling for α_i)

**Interpretation:** Within a firm, size increases profit by 0.08 (not 0.15)

**Key insight:** Cross-sectional estimate was biased by 87.5%!

### What Changed?

Fixed Effects controls for **time-invariant** firm characteristics:
- Management quality
- Industry
- Location
- Culture
- Brand value

**Result:** Estimates the **causal effect** of size on profit

## Key Takeaways

✅ **Panel data** = Cross-sectional + Time-series

✅ **Advantages:**
- Control for unobserved heterogeneity
- Study dynamics
- More data (precision)
- Better identification

⚠️ **Challenges:**
- Data collection costs
- Attrition
- Complexity of methods
- Short panel bias

🎯 **When to use:**
- Multiple entities, multiple periods
- Unobserved effects likely correlated with X
- Want to study changes over time

## Next Steps

**Learn the methods:**

1. **[Tutorial 1: Getting Started](../tutorials/01_getting_started.md)**: Your first panel model

2. **[Tutorial 2: Static Models](../tutorials/02_static_models.md)**: Pooled OLS, FE, RE

3. **[How-To: Choose Model](../how-to/choose_model.md)**: Decision guide

**Deep dives:**

1. **[Guide: Fixed vs Random](fixed_vs_random.md)**: Detailed comparison

2. **[Guide: GMM Explained](gmm_explained.md)**: Dynamic panel methods

**Further reading:**

- **Hsiao (2014)**: *Analysis of Panel Data* (3rd ed.)
- **Wooldridge (2010)**: *Econometric Analysis of Cross Section and Panel Data*
- **Baltagi (2021)**: *Econometric Analysis of Panel Data* (6th ed.)

---

**Panel data is powerful because it lets you see both across entities AND over time—giving you two sources of variation to identify effects more credibly.**
