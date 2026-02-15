# Multinomial Logit API Reference

## Overview

The `panelbox.models.discrete.MultinomialLogit` class implements multinomial logit for unordered categorical outcomes with J > 2 alternatives.

## MultinomialLogit

### Class: `MultinomialLogit(endog, exog, n_alternatives=None, base_alternative=0, method='pooled', entity_col=None, time_col=None)`

Multinomial logit model for panel data with unordered categorical outcomes.

**Parameters:**

- `endog` : array-like or pd.Series
  - Dependent variable (categorical: 0, 1, ..., J-1)
- `exog` : array-like or pd.DataFrame
  - Explanatory variables (regressors)
- `n_alternatives` : int, optional
  - Number of choice alternatives. If None, inferred from data
- `base_alternative` : int, default=0
  - Reference/baseline category (normalized to zero)
- `method` : str, default='pooled'
  - Estimation method: 'pooled', 'fixed_effects', or 'random_effects'
- `entity_col` : str, optional
  - Column name for entity/individual identifier (required for FE/RE)
- `time_col` : str, optional
  - Column name for time identifier

**Attributes:**

- `n_alternatives` : int - Number of choice alternatives (J)
- `n_params` : int - Total parameters: (J-1) × K
- `base_alternative` : int - Reference category
- `method` : str - Estimation method used

**Methods:**

### `fit(start_params=None, method='BFGS', maxiter=1000, **kwargs)`

Estimate multinomial logit parameters.

**Parameters:**
- `start_params` : np.ndarray, optional - Starting values
- `method` : str, default='BFGS' - Optimization method
- `maxiter` : int, default=1000 - Maximum iterations

**Returns:**
- `MultinomialLogitResult` - Fitted model results

**Example:**

```python
import numpy as np
import pandas as pd
from panelbox.models.discrete import MultinomialLogit

# Generate data
n = 500
X = np.random.randn(n, 3)
# ... generate categorical outcome y ...

# Estimate pooled multinomial logit
model = MultinomialLogit(
    endog=y,
    exog=X,
    n_alternatives=3,
    base_alternative=0,
    method='pooled'
)

result = model.fit()
print(result.summary())
```

## MultinomialLogitResult

Result class containing estimation results and methods for inference.

**Attributes:**

- `params` : np.ndarray - Estimated parameters (flattened)
- `params_matrix` : np.ndarray - Parameters reshaped to (J-1, K)
- `llf` : float - Log-likelihood value
- `converged` : bool - Convergence status
- `aic` : float - Akaike Information Criterion
- `bic` : float - Bayesian Information Criterion
- `pseudo_r2` : float - McFadden's pseudo R²
- `accuracy` : float - Prediction accuracy
- `confusion_matrix` : np.ndarray - Classification confusion matrix
- `predicted_probs` : np.ndarray - Predicted probabilities (n × J)

**Methods:**

### `predict_proba(exog=None)`

Predict probabilities for all alternatives.

**Parameters:**
- `exog` : array-like, optional - New data for prediction. If None, uses training data

**Returns:**
- `probs` : np.ndarray, shape (n, J) - Predicted probabilities

**Example:**

```python
# In-sample predictions
probs = result.predict_proba()
print(probs[:5])  # First 5 observations

# Out-of-sample predictions
X_new = np.random.randn(10, 3)
probs_new = result.predict_proba(X_new)
```

### `predict(exog=None)`

Predict most likely alternative for each observation.

**Parameters:**
- `exog` : array-like, optional - New data for prediction

**Returns:**
- `choices` : np.ndarray - Predicted categories (0, 1, ..., J-1)

**Example:**

```python
# Predict choices
y_pred = result.predict()

# Prediction accuracy
accuracy = (y_pred == y_true).mean()
print(f"Accuracy: {accuracy:.2%}")
```

### `marginal_effects(at='mean', variable=None)`

Compute marginal effects on choice probabilities.

**Parameters:**
- `at` : str, default='mean'
  - Where to evaluate: 'mean', 'median', or 'overall'
- `variable` : int or str, optional
  - Specific variable index or name. If None, compute for all

**Returns:**
- `me` : np.ndarray, shape (J, K)
  - Marginal effects for each alternative and variable

**Interpretation:**
- Shows how a one-unit change in X affects probability of each outcome
- Probabilities sum to 1 → marginal effects sum to 0
- Can be positive for some alternatives, negative for others

**Example:**

```python
# Average marginal effects at mean values
me = result.marginal_effects(at='mean')
print(me)
# Output shape: (3, 3) for J=3 alternatives, K=3 variables

# Verify they sum to zero
print(me.sum(axis=0))  # Should be approximately [0, 0, 0]

# Marginal effect for specific variable
me_education = result.marginal_effects(variable=0)
print(me_education)  # (3,) array for J=3 alternatives
```

### `summary()`

Generate formatted summary of estimation results.

**Returns:**
- `summary_str` : str - Formatted text summary

**Example:**

```python
print(result.summary())
```

**Output includes:**
- Model information (n_obs, n_alternatives, log-likelihood)
- Goodness-of-fit statistics (AIC, BIC, pseudo R², accuracy)
- Parameter estimates by alternative
- Standard errors and z-statistics
- Confusion matrix

## Model Interpretation

### Coefficients

Coefficients represent effects on **log-odds** relative to baseline:

- $\beta_{jk} > 0$: Variable $k$ increases log-odds of alternative $j$ vs. baseline
- $\beta_{jk} < 0$: Variable $k$ decreases log-odds of alternative $j$ vs. baseline

**Example:** If $\beta_{\text{education, white collar}} = 0.8$:
- 1 more year of education increases log-odds of white collar vs. baseline by 0.8

### Marginal Effects

Marginal effects show impact on **probabilities**:

$$\frac{\partial P(y=j)}{\partial x_k} = P(y=j)\left[\beta_{jk} - \sum_m P(y=m)\beta_{mk}\right]$$

**Key properties:**
1. Sum to zero: $\sum_j \frac{\partial P(y=j)}{\partial x_k} = 0$
2. Magnitude depends on probabilities (nonlinear)
3. Can differ in sign from coefficients!

## Estimation Methods

### Pooled Multinomial Logit

Standard MLE ignoring panel structure:

```python
model_pooled = MultinomialLogit(y, X, n_alternatives=3, method='pooled')
result = model_pooled.fit()
```

**When to use:**
- Large cross-sections
- No time dimension or panels not important
- Fast estimation needed

### Fixed Effects

Controls for individual heterogeneity (Chamberlain 1980):

```python
model_fe = MultinomialLogit(
    y, X,
    n_alternatives=3,
    method='fixed_effects',
    entity_col='id'
)
result_fe = model_fe.fit()
```

**When to use:**
- Individual heterogeneity suspected
- Correlation between individual effects and regressors
- **Warning:** Very slow for J > 4 or T > 10

### Random Effects

Assumes random individual effects:

```python
model_re = MultinomialLogit(
    y, X,
    n_alternatives=3,
    method='random_effects',
    entity_col='id'
)
result_re = model_re.fit()
```

**When to use:**
- Individual effects uncorrelated with regressors
- Middle ground between pooled and FE
- Faster than FE for large J or T

## Examples

### Occupational Choice

```python
import pandas as pd
from panelbox.models.discrete import MultinomialLogit

# Load data
df = pd.read_csv('occupation_data.csv')
# Columns: worker_id, year, occupation, education, experience

# Prepare variables
y = df['occupation'].values  # 0=unemployed, 1=blue collar, 2=white collar
X = df[['education', 'experience']].values

# Estimate model
model = MultinomialLogit(y, X, n_alternatives=3, base_alternative=0)
result = model.fit()

# Results
print(result.summary())

# Marginal effects
me = result.marginal_effects()
print("\nMarginal Effects:")
print(pd.DataFrame(
    me,
    index=['Unemployed', 'Blue Collar', 'White Collar'],
    columns=['Education', 'Experience']
))

# Predictions
probs = result.predict_proba()
choices = result.predict()
```

### Brand Choice

```python
# Brand choice: A, B, C
brands = df['brand'].map({'A': 0, 'B': 1, 'C': 2}).values
features = df[['price', 'quality', 'advertising']].values

model = MultinomialLogit(brands, features, n_alternatives=3, base_alternative=0)
result = model.fit()

# What if price increases?
me_price = result.marginal_effects(variable=0)  # Price is first variable
print("Price increase marginal effects:")
for i, brand in enumerate(['A', 'B', 'C']):
    print(f"  {brand}: {me_price[i]:.4f}")
```

## Diagnostics

### Model Fit

- **Pseudo R²**: McFadden's R² (1 - llf / llf_null)
- **AIC/BIC**: For model comparison
- **Prediction accuracy**: Fraction correctly classified

### IIA Assumption

Multinomial logit assumes **Independence of Irrelevant Alternatives** (IIA):
- Adding or removing an alternative doesn't affect relative odds of other alternatives
- Can be tested with Hausman-McFadden test (not yet implemented)
- If violated, consider nested logit or mixed logit

## Best Practices

1. **Standardize variables** for better convergence
2. **Check for separation** (perfect prediction)
3. **Use marginal effects** for interpretation
4. **Test IIA assumption** when possible
5. **Compare methods** (pooled vs. FE vs. RE) using Hausman test
6. **Visualize** predicted probabilities for key variables

## References

- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior."
- Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data."
- Train, K. (2009). "Discrete Choice Methods with Simulation."

## See Also

- [Tutorial: Multinomial Logit](../tutorials/multinomial_tutorial.ipynb)
- [Discrete Choice Guide](../guides/discrete_choice.md)
- [Binary Choice Models](binary_choice.md)
