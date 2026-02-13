# Discrete Choice and Limited Dependent Variable Models

This module implements discrete choice and limited dependent variable models for panel data, including binary, ordered, and count data models with various treatments of individual heterogeneity.

## Overview

The `panelbox.models.discrete` module provides:

- **Binary Choice Models**: Logit and Probit for binary outcomes
- **Fixed Effects Models**: Conditional MLE for removing individual effects
- **Random Effects Models**: Incorporating unobserved heterogeneity (Phase 2)
- **Ordered Choice Models**: For ordinal outcomes (Phase 3)
- **Count Data Models**: Poisson and Negative Binomial (Phase 4)

## Currently Implemented Models (Phase 1)

### Pooled Logit
Standard logistic regression pooling all observations:
```python
from panelbox.models.discrete import PooledLogit

model = PooledLogit("employed ~ age + education + experience", data, "person_id", "year")
results = model.fit(se_type='cluster')  # Cluster-robust SEs by entity
```

### Pooled Probit
Probit model with normal CDF link function:
```python
from panelbox.models.discrete import PooledProbit

model = PooledProbit("employed ~ age + education", data, "person_id", "year")
results = model.fit()
```

### Fixed Effects Logit
Chamberlain (1980) conditional MLE eliminating fixed effects:
```python
from panelbox.models.discrete import FixedEffectsLogit

model = FixedEffectsLogit("employed ~ experience + married", data, "person_id", "year")
results = model.fit()

# Check which entities were used
print(f"Entities with variation: {model.n_used_entities}")
print(f"Entities dropped: {len(model.dropped_entities)}")
```

## Model Selection Guide

| Model | Use When | Key Features |
|-------|----------|--------------|
| **Pooled Logit/Probit** | No individual effects or effects uncorrelated with X | Simple, fast, cluster-robust SEs |
| **Fixed Effects Logit** | Individual effects correlated with X | Removes fixed effects, drops entities without variation |
| **Random Effects Probit** | Individual effects uncorrelated with X (Phase 2) | More efficient than FE, keeps all entities |
| **Correlated Random Effects** | Want to test correlation assumption (Phase 2) | Nests FE and RE models |

## Key Features

### 1. Optimization Infrastructure

All models use a robust MLE optimization framework:

```python
# Multiple optimization methods
results = model.fit(method='bfgs')     # Default: BFGS
results = model.fit(method='newton')   # Newton-Raphson
results = model.fit(method='trust-constr')  # Trust region

# Multiple starting values to avoid local minima
results = model.fit(n_starts=5)

# Convergence diagnostics
if not results.converged:
    print(f"Warning: Did not converge after {results.n_iter} iterations")
```

### 2. Standard Errors

Multiple types of standard errors for inference:

```python
# Non-robust (assumes homoskedasticity)
results = model.fit(se_type='nonrobust')

# Heteroskedasticity-robust (Sandwich)
results = model.fit(se_type='robust')

# Cluster-robust (default, clusters by entity)
results = model.fit(se_type='cluster')

# Bootstrap standard errors
results = model.fit(se_type='bootstrap', n_bootstrap=999)
```

### 3. Predictions and Diagnostics

```python
# Predictions
linear_pred = results.predict(type='linear')  # X'β
prob_pred = results.predict(type='prob')       # P(y=1|X)
class_pred = results.predict(type='class')     # Binary 0/1

# Classification metrics
metrics = results.classification_metrics()
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")

# Confusion matrix
print(results.classification_table())

# Pseudo-R² measures
print(f"McFadden R²: {results.pseudo_r2('mcfadden'):.3f}")
print(f"Cox-Snell R²: {results.pseudo_r2('cox_snell'):.3f}")

# Goodness of fit
hl_test = results.hosmer_lemeshow_test()
print(f"Hosmer-Lemeshow p-value: {hl_test['p_value']:.3f}")
```

### 4. Information Criteria

```python
print(f"Log-likelihood: {results.llf:.2f}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")
```

## Mathematical Background

### Pooled Logit
$$P(y_{it} = 1 | X_{it}) = \Lambda(X_{it}'\beta) = \frac{exp(X_{it}'\beta)}{1 + exp(X_{it}'\beta)}$$

### Pooled Probit
$$P(y_{it} = 1 | X_{it}) = \Phi(X_{it}'\beta)$$
where $\Phi$ is the standard normal CDF.

### Fixed Effects Logit
Uses conditional MLE (Chamberlain 1980) to eliminate fixed effects $\alpha_i$:
$$P(y_i | \sum_t y_{it}, X_i, \alpha_i) = P(y_i | \sum_t y_{it}, X_i)$$

Only entities with within-variation (0 < $\sum_t y_{it}$ < $T_i$) contribute to the likelihood.

## Examples

### Basic Binary Choice Model

```python
import pandas as pd
from panelbox.models.discrete import PooledLogit

# Load data
data = pd.read_csv("labor_force.csv")

# Fit model
model = PooledLogit("employed ~ age + education + kids", data, "person_id", "year")
results = model.fit()

# View results
print(results.summary())

# Make predictions for new data
new_data = pd.DataFrame({
    'age': [25, 35, 45],
    'education': [12, 16, 18],
    'kids': [0, 2, 1]
})
probabilities = results.predict(new_data, type='prob')
```

### Fixed Effects for Panel Data

```python
from panelbox.models.discrete import FixedEffectsLogit

# Fixed effects removes time-invariant characteristics
model = FixedEffectsLogit("employed ~ experience + married + kids",
                          data, "person_id", "year")
results = model.fit()

# Entities without variation are automatically dropped
print(f"Used {model.n_used_entities} out of {model.n_entities} entities")

# No intercept in FE Logit (absorbed by fixed effects)
print(f"Coefficients: {results.params}")  # No intercept term
```

### Model Comparison

```python
# Compare different models
models = {
    'Pooled Logit': PooledLogit("y ~ x1 + x2 + x3", data, "id", "time"),
    'Pooled Probit': PooledProbit("y ~ x1 + x2 + x3", data, "id", "time"),
    'Fixed Effects': FixedEffectsLogit("y ~ x1 + x2 + x3", data, "id", "time")
}

for name, model in models.items():
    results = model.fit()
    print(f"{name}:")
    print(f"  AIC: {results.aic:.2f}")
    print(f"  BIC: {results.bic:.2f}")
    print(f"  Pseudo-R²: {results.pseudo_r2():.3f}")
```

## Numerical Optimization

The module uses sophisticated numerical methods:

```python
# Numerical gradients when analytical not available
from panelbox.optimization.numerical_grad import approx_gradient, approx_hessian

gradient = approx_gradient(log_likelihood, params, method='central')
hessian = approx_hessian(log_likelihood, params)

# Automatic step size selection
gradient = approx_gradient(log_likelihood, params, epsilon='auto')

# Manual step size for fine control
gradient = approx_gradient(log_likelihood, params, epsilon=1e-5)
```

## Upcoming Features

### Phase 2: Marginal Effects & Random Effects
- Average Marginal Effects (AME)
- Marginal Effects at Means (MEM)
- Random Effects Probit
- Correlated Random Effects (Mundlak)

### Phase 3: Ordered Choice Models
- Ordered Logit/Probit
- Fixed Effects Ordered Logit

### Phase 4: Count Data Models
- Poisson and Negative Binomial
- Fixed Effects Poisson
- Zero-inflated models

### Phase 5: Selection Models
- Heckman selection model
- Panel Tobit

## Performance Considerations

- **Fixed Effects Logit**: Computational complexity increases with panel length T
  - T ≤ 10: Fast enumeration
  - 10 < T ≤ 20: Dynamic programming
  - T > 20: May be slow, consider alternatives

- **Multiple Starting Values**: Use `n_starts>1` for complex models to avoid local optima

- **Bootstrap SEs**: Can be slow for large datasets, consider parallel processing

## Validation

Models are validated against:
- R `pglm` package (panel GLM)
- Stata `xtlogit`, `xtprobit` commands
- Python `statsmodels` (for pooled models)

## References

- Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data." *Review of Economic Studies*, 47(1), 225-238.
- Greene, W.H. (2018). *Econometric Analysis*, 8th Edition. Pearson.
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd Edition. MIT Press.

## Contributing

When adding new models:
1. Inherit from `NonlinearPanelModel`
2. Implement `_log_likelihood()` method
3. Optionally override `_score()` and `_hessian()` for analytical derivatives
4. Add comprehensive tests including R/Stata validation
5. Document mathematical formulation and use cases
