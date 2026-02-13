# Discrete Choice Models API Reference

## Overview

The `panelbox.models.discrete` module provides maximum likelihood estimation for discrete choice models in panel data settings. This includes binary choice models (logit and probit) with various specifications for handling unobserved heterogeneity.

## Base Classes

### `NonlinearPanelModel`

Abstract base class for all nonlinear panel models estimated via Maximum Likelihood Estimation (MLE).

```python
from panelbox.models.discrete.base import NonlinearPanelModel
```

#### Key Features

- Multiple optimization algorithms (BFGS, Newton-Raphson, Trust-Region)
- Multiple starting values to avoid local minima
- Convergence diagnostics (gradient norm, Hessian eigenvalues)
- Numerical gradient and Hessian computation
- Support for constrained optimization

#### Abstract Methods

Subclasses must implement:

- `_log_likelihood(params)`: Compute log-likelihood at parameter values
- `_score(params)` (optional): Analytical gradient
- `_hessian(params)` (optional): Analytical Hessian

#### Main Methods

##### `fit(method='bfgs', **kwargs)`

Fit the model via maximum likelihood estimation.

**Parameters:**
- `method` : {'bfgs', 'newton', 'trust-constr'}, default='bfgs'
  - Optimization algorithm to use
- `start_params` : array-like, optional
  - Starting values for parameters
- `n_starts` : int, default=1
  - Number of random starting values to try
- `maxiter` : int, default=1000
  - Maximum number of iterations
- `verbose` : bool, default=False
  - Print optimization progress

**Returns:**
- `PanelResults` : Fitted model results

## Binary Choice Models

### `PooledLogit`

Pooled Logit model for panel data with cluster-robust standard errors.

```python
from panelbox.models.discrete import PooledLogit

model = PooledLogit(formula, data, entity_col, time_col)
results = model.fit(cov_type='cluster')
```

#### Model Specification

$$P(y_{it} = 1 | X_{it}) = \Lambda(X_{it}'\beta)$$

where $\Lambda(z) = \frac{e^z}{1 + e^z}$ is the logistic CDF.

#### Parameters

- `formula` : str
  - Model formula in R-style syntax (e.g., "y ~ x1 + x2")
- `data` : pd.DataFrame
  - Panel data in long format
- `entity_col` : str
  - Name of column identifying entities
- `time_col` : str
  - Name of column identifying time periods
- `weights` : array-like, optional
  - Observation weights for weighted MLE

#### Methods

##### `fit(cov_type='cluster', **kwargs)`

Fit the Pooled Logit model.

**Parameters:**
- `cov_type` : {'nonrobust', 'robust', 'cluster'}, default='cluster'
  - Type of standard errors:
    - 'nonrobust': Classical (assumes iid)
    - 'robust': Heteroskedasticity-robust (sandwich)
    - 'cluster': Cluster-robust by entity (default)

**Returns:**
- `PanelResults` with additional attributes:
  - `llf`: Log-likelihood value
  - `aic`, `bic`: Information criteria
  - `pseudo_r2_mcfadden`: McFadden's pseudo R-squared
  - `converged`: Convergence flag

##### `predict(type='prob')`

Generate predictions from fitted model.

**Parameters:**
- `type` : {'linear', 'prob', 'class'}, default='prob'
  - Type of prediction:
    - 'linear': Linear predictor $X'\beta$
    - 'prob': Predicted probabilities $P(y=1|X)$
    - 'class': Binary classification (0/1)

#### Results Methods

The fitted results object provides additional diagnostic methods:

##### `pseudo_r2(kind='mcfadden')`

Compute pseudo R-squared measures.

**Parameters:**
- `kind` : {'mcfadden', 'cox_snell', 'nagelkerke'}

##### `classification_metrics(threshold=0.5)`

Compute classification performance metrics.

**Returns:** Dictionary with:
- `accuracy`: Overall accuracy
- `precision`: Precision (positive predictive value)
- `recall`: Recall (sensitivity)
- `f1`: F1-score
- `auc_roc`: Area under ROC curve
- `confusion_matrix`: TP, TN, FP, FN counts

##### `hosmer_lemeshow_test(n_groups=10)`

Hosmer-Lemeshow goodness-of-fit test for panel data.

**Returns:** Dictionary with:
- `statistic`: Chi-squared test statistic
- `p_value`: P-value
- `df`: Degrees of freedom
- `interpretation`: Text interpretation

##### `information_matrix_test()`

Information Matrix Test for model misspecification.

Tests if the information matrix equality holds: $-E[H] = E[ss']$

**Returns:** Dictionary with test statistic, p-value, and interpretation.

##### `link_test()`

Link test for functional form misspecification.

Tests if the squared linear predictor is significant when added to the model.

**Returns:** Dictionary with coefficient, t-statistic, p-value for squared term.

### `PooledProbit`

Pooled Probit model for panel data.

```python
from panelbox.models.discrete import PooledProbit

model = PooledProbit(formula, data, entity_col, time_col)
results = model.fit(cov_type='cluster')
```

#### Model Specification

$$P(y_{it} = 1 | X_{it}) = \Phi(X_{it}'\beta)$$

where $\Phi$ is the standard normal CDF.

#### Interface

The PooledProbit class has the same interface as PooledLogit, with identical methods and parameters. The only difference is the link function (normal vs logistic).

### `FixedEffectsLogit`

Fixed Effects Logit model using Chamberlain's (1980) conditional maximum likelihood.

```python
from panelbox.models.discrete import FixedEffectsLogit

model = FixedEffectsLogit(formula, data, entity_col, time_col)
results = model.fit(method='bfgs')
```

#### Model Specification

The fixed effects logit model includes entity-specific intercepts $\alpha_i$:

$$P(y_{it} = 1 | X_{it}, \alpha_i) = \Lambda(X_{it}'\beta + \alpha_i)$$

The conditional likelihood eliminates $\alpha_i$ by conditioning on $\sum_t y_{it}$:

$$P(y_{i1}, ..., y_{iT} | \sum_t y_{it}, X_i) \propto \exp(\sum_t y_{it} X_{it}'\beta)$$

#### Key Features

- Automatically drops entities without temporal variation in the dependent variable
- Only time-varying covariates are identified
- Computationally intensive for large T (uses enumeration for T ≤ 10)

#### Attributes

- `entities_with_variation` : array
  - Entity IDs that contribute to estimation
- `dropped_entities` : array
  - Entity IDs dropped due to no variation
- `n_used_entities` : int
  - Number of entities used in estimation
- `n_dropped_entities` : int
  - Number of entities dropped

#### Methods

##### `fit(method='bfgs', **kwargs)`

Fit the Fixed Effects Logit via conditional MLE.

**Parameters:**
- `method` : {'bfgs', 'newton'}, default='bfgs'
  - Optimization algorithm

**Returns:**
- `PanelResults` with entity-specific information

## Numerical Optimization

### Gradient and Hessian Approximation

The module provides numerical differentiation utilities:

```python
from panelbox.optimization.numerical_grad import approx_gradient, approx_hessian

# Approximate gradient at point x
grad = approx_gradient(func, x, method='central')

# Approximate Hessian at point x
hess = approx_hessian(func, x, method='central')
```

#### Functions

##### `approx_gradient(func, x, method='central', epsilon='auto')`

Approximate gradient using finite differences.

**Parameters:**
- `func` : callable
  - Function to differentiate
- `x` : array-like
  - Point at which to evaluate gradient
- `method` : {'central', 'forward'}, default='central'
  - Finite difference method
- `epsilon` : float or 'auto'
  - Step size (auto uses optimal step size)

##### `approx_hessian(func, x, method='central', epsilon='auto')`

Approximate Hessian using finite differences.

## Standard Errors

### MLE Standard Errors

The module provides various standard error estimators for MLE models:

```python
from panelbox.standard_errors.mle import cluster_robust_mle

# Compute cluster-robust standard errors
result = cluster_robust_mle(hessian, scores, clusters, df_correction=True)
```

#### Available Estimators

1. **Non-robust (Classical)**
   - $V = -H^{-1}$ where $H$ is the Hessian

2. **Robust (Sandwich)**
   - $V = H^{-1} S H^{-1}$ where $S = \sum_i s_i s_i'$

3. **Cluster-robust**
   - $V = H^{-1} [\sum_i g_i g_i'] H^{-1}$ where $g_i = \sum_t s_{it}$

4. **Bootstrap**
   - Resample entities and re-estimate

## Example Usage

### Basic Binary Choice Model

```python
import panelbox as pb
import pandas as pd

# Load data
data = pd.read_csv('panel_data.csv')

# Pooled Logit with cluster-robust SEs
logit = pb.PooledLogit("y ~ x1 + x2 + x3", data, "id", "year")
logit_results = logit.fit(cov_type='cluster')

# View results
print(logit_results.summary())

# Diagnostics
print(f"Pseudo R²: {logit_results.pseudo_r2('mcfadden'):.4f}")
metrics = logit_results.classification_metrics()
print(f"Accuracy: {metrics['accuracy']:.3f}")

# Goodness of fit test
hl_test = logit_results.hosmer_lemeshow_test()
print(f"H-L test p-value: {hl_test['p_value']:.4f}")

# Fixed Effects Logit
fe_logit = pb.FixedEffectsLogit("y ~ x1 + x2", data, "id", "year")
fe_results = fe_logit.fit()

print(f"Entities used: {fe_logit.n_used_entities}/{data['id'].nunique()}")
print(fe_results.summary())
```

### Model Comparison

```python
# Compare different models
models = {
    'Pooled Logit': pb.PooledLogit(formula, data, "id", "year"),
    'Pooled Probit': pb.PooledProbit(formula, data, "id", "year"),
    'FE Logit': pb.FixedEffectsLogit(formula, data, "id", "year")
}

results = {}
for name, model in models.items():
    results[name] = model.fit()
    print(f"{name}: AIC={results[name].aic:.1f}, BIC={results[name].bic:.1f}")
```

### Prediction

```python
# Get predictions
probabilities = logit_results.predict(type='prob')
linear_pred = logit_results.predict(type='linear')
classifications = logit_results.predict(type='class')

# Predict for new data
import numpy as np
X_new = np.array([[1, 2.5, -1.0, 0.5]])  # Include intercept
prob_new = logit_results.predict(X_new, type='prob')
```

## Performance Considerations

### Computational Complexity

- **Pooled models**: O(NT × k²) where N=entities, T=periods, k=parameters
- **Fixed Effects Logit**: O(N × 2^T) worst case for enumeration
  - Efficient for T ≤ 10
  - Dynamic programming for 10 < T ≤ 20
  - May be slow for T > 20

### Memory Usage

- Pooled models store full N×T dataset
- FE Logit only processes entities with variation
- Bootstrap SEs require storing B bootstrap samples

### Optimization Tips

1. **Use BFGS as default** - robust and efficient
2. **Try multiple starting values** for non-convex likelihoods
3. **Check convergence diagnostics** via `verbose=True`
4. **Use analytical gradients** when available (FE Logit has them)

## References

1. Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data." *Review of Economic Studies*, 47(1), 225-238.

2. Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapters 15-16.

3. Cameron, A.C., & Trivedi, P.K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.

4. Hosmer, D.W., & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley.

5. White, H. (1982). "Maximum Likelihood Estimation of Misspecified Models." *Econometrica*, 50(1), 1-25.
