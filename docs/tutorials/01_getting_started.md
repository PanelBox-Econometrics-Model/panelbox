# Getting Started with PanelBox

> Learn the basics of panel data analysis with PanelBox in 15 minutes.

## What You'll Learn

In this tutorial, you will:

- Install PanelBox
- Load and explore panel data
- Estimate your first panel model (Pooled OLS)
- Interpret the results
- Understand the output

## Prerequisites

This tutorial assumes you have:

- **Basic Python knowledge**: Variables, functions, imports
- **Familiarity with pandas**: DataFrames and basic operations
- **Understanding of linear regression**: OLS, coefficients, p-values

No prior panel data experience required!

## Installation

Install PanelBox using pip:

```bash
pip install panelbox
```

**Requirements:**
- Python ≥ 3.9
- NumPy ≥ 1.24.0
- Pandas ≥ 2.0.0

**Verify installation:**

```python
import panelbox as pb
print(f"PanelBox version: {pb.__version__}")
```

## What is Panel Data?

Panel data combines **cross-sectional** and **time-series** dimensions:

- **Cross-sectional**: Multiple entities (firms, countries, individuals)
- **Time-series**: Observed over multiple time periods

**Example:** 10 firms observed over 20 years = 200 observations

Panel data allows us to:
- Control for unobserved heterogeneity
- Study dynamics over time
- Increase statistical power

## Your First Panel Model

### Step 1: Load Data

PanelBox includes the classic Grunfeld dataset:

```python
import panelbox as pb

# Load example data
data = pb.load_grunfeld()

# Display first rows
print(data.head())
```

**Output:**
```
   firm  year   invest     value   capital
0     1  1935   317.60   3078.50    2.80
1     1  1936   391.80   4661.70   52.60
2     1  1937   410.60   5387.10  156.90
3     1  1938   257.70   2792.20  209.20
4     1  1939   330.80   4313.20  203.40
```

**Variables:**
- `firm`: Firm identifier (1-10)
- `year`: Year (1935-1954)
- `invest`: Gross investment (millions of dollars)
- `value`: Market value of the firm
- `capital`: Stock of plant and equipment

### Step 2: Explore the Data

Understanding your panel structure is crucial:

```python
# Panel dimensions
n_firms = data['firm'].nunique()
n_years = data['year'].nunique()
n_obs = len(data)

print(f"Entities (N): {n_firms}")  # 10 firms
print(f"Time periods (T): {n_years}")  # 20 years
print(f"Total observations: {n_obs}")  # 200

# Check if balanced
obs_per_firm = data.groupby('firm').size()
is_balanced = (obs_per_firm == n_years).all()
print(f"Balanced panel: {is_balanced}")  # True
```

**Output:**
```
Entities (N): 10
Time periods (T): 20
Total observations: 200
Balanced panel: True
```

**Visualize relationships:**

```python
import matplotlib.pyplot as plt

# Investment over time for each firm
for firm_id in data['firm'].unique()[:3]:  # First 3 firms
    firm_data = data[data['firm'] == firm_id]
    plt.plot(firm_data['year'], firm_data['invest'],
             label=f'Firm {firm_id}', marker='o')

plt.xlabel('Year')
plt.ylabel('Investment')
plt.title('Investment Over Time')
plt.legend()
plt.show()
```

### Step 3: Estimate Pooled OLS

Let's estimate a simple investment model:

```python
# Estimate Pooled OLS
model = pb.PooledOLS(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year"
)

# Fit the model
results = model.fit(cov_type='robust')

# Display results
print(results.summary())
```

**Output:**
```
================================================================================
                        Pooled OLS Estimation Results
================================================================================
Dependent Variable:              invest        No. Observations:             200
Model:                       Pooled OLS        Df Residuals:                 197
Method:                    Least Squares        Df Model:                       2
Date:                     2026-02-05            R-squared:                  0.812
Time:                     14:30:25              Adj. R-squared:             0.810
Cov. Type:                    robust
================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -42.714     14.532     -2.939      0.004     -71.365     -14.063
value             0.116      0.006     19.721      0.000       0.104       0.127
capital           0.231      0.028      8.251      0.000       0.176       0.286
================================================================================
```

### Step 4: Understand the Results

Let's interpret each part of the output:

**Model Information:**
- **200 observations**: All firm-year combinations
- **R-squared: 0.812**: Model explains 81.2% of investment variation
- **Cov. Type: robust**: Heteroskedasticity-robust standard errors

**Coefficient Interpretation:**

1. **value = 0.116** (p < 0.001)
   - A $1 million increase in firm value → $0.116 million more investment
   - Highly significant (p < 0.001)

2. **capital = 0.231** (p < 0.001)
   - A $1 million increase in capital stock → $0.231 million more investment
   - Also highly significant

3. **Intercept = -42.714**
   - Baseline investment when value and capital are zero
   - Less meaningful interpretation

**Statistical Significance:**

- **t-statistic**: Tests if coefficient ≠ 0
  - |t| > 2 typically indicates significance
- **p-value**: Probability of observing this coefficient by chance
  - p < 0.05: Significant at 5% level
  - p < 0.01: Significant at 1% level

All our coefficients have p < 0.001, showing strong evidence of effects.

**Confidence Intervals:**

The 95% confidence interval for `value` is [0.104, 0.127]:
- We're 95% confident the true effect lies in this range
- Since it doesn't include 0, confirms significance

### Step 5: Access Results Programmatically

Extract specific results for further analysis:

```python
# Coefficients
print("Coefficients:")
print(results.params)

# Standard errors
print("\nStandard Errors:")
print(results.std_errors)

# R-squared
print(f"\nR-squared: {results.rsquared:.4f}")

# Specific coefficient
value_coef = results.params['value']
value_se = results.std_errors['value']
print(f"\nValue coefficient: {value_coef:.4f} (SE: {value_se:.4f})")
```

**Output:**
```
Coefficients:
Intercept   -42.713967
value         0.115562
capital       0.230789
dtype: float64

Standard Errors:
Intercept    14.532461
value         0.005860
capital       0.027971
dtype: float64

R-squared: 0.8119

Value coefficient: 0.1156 (SE: 0.0059)
```

## Key Takeaways

✅ **You've learned to:**
- Load panel data with `pb.load_grunfeld()`
- Explore panel structure (N, T, balanced/unbalanced)
- Estimate a Pooled OLS model
- Interpret regression output
- Access results programmatically

⚠️ **Important caveat:**
Pooled OLS ignores the panel structure! It treats all observations as independent, which may not be realistic. In most applications, you'll want to use:
- **Fixed Effects**: Control for time-invariant firm characteristics
- **Random Effects**: Model entity-specific effects
- **GMM**: Handle dynamics and endogeneity

## Next Steps

**Continue learning:**

1. **[Static Panel Models Tutorial](02_static_models.md)**: Learn Fixed Effects, Random Effects, and when to use each

2. **[How-To: Choose a Model](../how-to/choose_model.md)**: Decision guide for selecting the right estimator

3. **[API Reference](../api/models.md)**: Complete documentation of all models

**Try it yourself:**

```python
# Load your own data
import pandas as pd
data = pd.read_csv('your_panel_data.csv')

# Estimate a model
model = pb.PooledOLS(
    formula="dependent ~ var1 + var2",
    data=data,
    entity_col="entity_id",  # Your entity column
    time_col="time"          # Your time column
)
results = model.fit(cov_type='robust')
print(results.summary())
```

## Common Issues

**Problem:** `KeyError: 'firm'`
- **Solution:** Check that `entity_col` and `time_col` match your column names

**Problem:** `ValueError: Formula parse error`
- **Solution:** Use R-style formulas: `"y ~ x1 + x2"`, not Python-style

**Problem:** Import error
- **Solution:** Reinstall: `pip install --upgrade panelbox`

## Further Reading

- **Textbooks:**
  - Wooldridge (2010): *Econometric Analysis of Cross Section and Panel Data*
  - Baltagi (2021): *Econometric Analysis of Panel Data*

- **Papers:**
  - Grunfeld (1958): Original dataset paper

---

**Congratulations!** You've completed your first panel data analysis with PanelBox. 🎉

Ready to learn more advanced models? Continue to [Tutorial 2: Static Panel Models](02_static_models.md).
