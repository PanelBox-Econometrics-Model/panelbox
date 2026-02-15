# PanelBox Architecture - Quick Reference Guide

## Core Class Hierarchy

```
NonlinearPanelModel
├── PooledLogit / PooledProbit        (Binary choice, pooled estimation)
├── FixedEffectsLogit                 (Binary, Chamberlain conditional MLE)
├── RandomEffectsProbit               (Binary, with Gauss-Hermite quadrature)
├── OrderedLogit / OrderedProbit      (Ordered choice models)
├── RandomEffectsOrderedLogit         (Ordered, random effects)
├── PooledPoisson / RandomEffectsPoisson (Count data)
├── NegativeBinomial                  (Overdispersed count)
└── CensoredModels                    (Tobit, etc.)
```

## Essential Files for Implementation

### 1. Base Class
- **File**: `panelbox/models/discrete/base.py`
- **Class**: `NonlinearPanelModel`
- **Key Methods**:
  - `fit()` - MLE optimization with convergence checks
  - `_log_likelihood(params)` - MUST override
  - `_score(params)` - Optional (numerical gradient if not provided)
  - `_hessian(params)` - Optional (numerical Hessian if not provided)
  - `_create_results()` - MUST override for results customization

### 2. Discrete Choice Models
- **File**: `panelbox/models/discrete/binary.py`
- **Models**: PooledLogit, PooledProbit, FixedEffectsLogit, RandomEffectsProbit
- **Pattern**: Copy from PooledLogit for standard implementation

### 3. Ordered Choice Models
- **File**: `panelbox/models/discrete/ordered.py`
- **Key Feature**: Cutpoint ordering with κⱼ = κ_{j-1} + exp(γⱼ)

### 4. Count Data Models
- **File**: `panelbox/models/count/poisson.py`, `negbin.py`
- **Pattern**: Score = Σ (y - μ) * X, Hessian = -Σ μ * X*X'

### 5. Censored Models
- **File**: `panelbox/models/censored/tobit.py`, `honore.py`
- **Pattern**: Integrate censoring probability over random effects

## Key Utilities

### Quadrature Integration
```python
from panelbox.optimization.quadrature import gauss_hermite_quadrature

nodes, weights = gauss_hermite_quadrature(12)  # Returns (12,) arrays
# Transform: alpha = sqrt(2) * sigma * node
# Integrate: sum(weights * f(alpha) for node, weight)
```

### Marginal Effects
```python
from panelbox.marginal_effects.delta_method import delta_method_se

se = delta_method_se(me_func, params, vcov)
```

### Data Validation
```python
from panelbox.utils.data import check_panel_data

y, X, entity_id, time_id, weights = check_panel_data(
    y, X, entity_id, time_id, weights
)
```

## Implementation Template

### For a New Binary Model (e.g., PooledLogit variant)

```python
from panelbox.models.discrete.base import NonlinearPanelModel

class MyBinaryModel(NonlinearPanelModel):
    """Documentation."""

    def __init__(self, formula, data, entity_col, time_col, weights=None):
        super().__init__(formula, data, entity_col, time_col, weights)

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Core computation: must return scalar float
        """
        y, X = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        y = y.ravel()

        # Your likelihood computation here
        eta = X @ params
        # ...
        llf = float(sum_of_log_likelihoods)
        return llf

    def _score(self, params: np.ndarray) -> np.ndarray:
        """Optional: analytical gradient"""
        # If not provided, uses numerical gradient
        pass

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """Optional: analytical Hessian"""
        # If not provided, uses numerical Hessian
        pass

    def _create_results(self, params, var_names, y, X):
        """Create and return PanelResults object"""
        from panelbox.core.results import PanelResults

        # Compute fitted values
        fitted_values = ...

        # Compute covariance
        H = self._hessian(params)
        vcov = -np.linalg.inv(H)

        # Create results
        results = PanelResults(
            params=pd.Series(params, index=var_names),
            std_errors=pd.Series(np.sqrt(np.diag(vcov)), index=var_names),
            cov_params=pd.DataFrame(vcov, index=var_names, columns=var_names),
            fittedvalues=fitted_values,
            resid=y - fitted_values,
            model_info={"model_type": "MyModel", "formula": self.formula, ...},
            data_info={"nobs": len(y), "n_entities": self.data.n_entities, ...},
            rsquared_dict={...},
            model=self
        )

        return results
```

### For a Censored Model (e.g., Tobit)

```python
class MyCensoredModel(NonlinearPanelModel):
    """Censored model with Gauss-Hermite quadrature."""

    def __init__(self, formula, data, entity_col, time_col,
                 quadrature_points=12, censoring_point=0.0):
        super().__init__(formula, data, entity_col, time_col)
        self.censoring_point = censoring_point
        self.quadrature_points = quadrature_points

        # Pre-compute quadrature
        from panelbox.optimization.quadrature import gauss_hermite_quadrature
        self._quad_nodes, self._quad_weights = gauss_hermite_quadrature(
            quadrature_points
        )

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Integrate out random effects via quadrature
        """
        beta = params[:-1]
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)

        y, X = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        y = y.ravel()
        entities = self.data.data[self.data.entity_col].values

        llf = 0.0
        for entity in np.unique(entities):
            mask = entities == entity
            y_i = y[mask]
            X_i = X[mask]

            # Quadrature integration
            entity_sum = 0.0
            for node, weight in zip(self._quad_nodes, self._quad_weights):
                alpha = np.sqrt(2) * sigma * node
                # Compute: P(y | X, α; β)
                prob = self._compute_censoring_prob(y_i, X_i, beta, alpha)
                entity_sum += weight * prob

            llf += np.log(entity_sum)

        return float(llf)

    def _compute_censoring_prob(self, y, X, beta, alpha):
        """Compute probability/density accounting for censoring"""
        eta = X @ beta + alpha
        # Handle censoring logic here
        pass
```

## Covariance Matrix Types

### Standard Implementation Pattern

```python
# After fitting, for a binary model:

# 1. Nonrobust (classical)
vcov = -np.linalg.inv(H)  # H = Hessian

# 2. Robust (sandwich)
W = fitted_probs * (1 - fitted_probs)
H = -(X.T * W) @ X
H_inv = np.linalg.inv(H)
scores = (y - fitted_probs)[:, np.newaxis] * X
B = scores.T @ scores
vcov = H_inv @ B @ H_inv

# 3. Cluster-robust
from panelbox.standard_errors.mle import cluster_robust_mle
result = cluster_robust_mle(H, scores, entities, df_correction=True)
vcov = result.cov_matrix
```

## Marginal Effects Pattern

```python
from panelbox.marginal_effects.discrete_me import compute_ame, compute_mem

# In results.marginal_effects() method:
def marginal_effects(self, at='mean'):
    if at == 'overall':
        return compute_ame(self._results)
    elif at == 'mean':
        return compute_mem(self._results)
```

## Testing Checklist

When implementing a new model:

1. **Log-likelihood**: Returns scalar float, matches numerical gradient
2. **Score/Hessian**: Numerical stability, correct signs
3. **Convergence**: Gradient norm < 1e-3, Hessian negative definite
4. **Covariance**: Positive definite, reasonable magnitudes
5. **Predictions**: Match fitted values in-sample
6. **Results**: All attributes populated correctly
7. **Panel structure**: Respects entity/time indices
8. **Edge cases**: Handles singular data, all zeros/ones, etc.

## Common Pitfalls

1. **Log-likelihood not scalar**: Must return `float(llf)`, not array
2. **Wrong Hessian sign**: Should be negative for maximum; return -2nd derivative
3. **Singular Hessian**: Use `np.linalg.pinv()` as fallback
4. **Numerical stability**: Clip probabilities [1e-10, 1-1e-10], use `log1p`, etc.
5. **Quadrature transform**: Remember √2 factor: `alpha = √2 * sigma * node`
6. **Cutpoint ordering**: Use log-differences to ensure inequality constraints

## File Locations (Absolute Paths)

- Base classes: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/base.py`
- Binary models: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py`
- Ordered models: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/ordered.py`
- Count models: `/home/guhaase/projetos/panelbox/panelbox/models/count/poisson.py`
- Censored models: `/home/guhaase/projetos/panelbox/panelbox/models/censored/`
- Quadrature: `/home/guhaase/projetos/panelbox/panelbox/optimization/quadrature.py`
- Marginal effects: `/home/guhaase/projetos/panelbox/panelbox/marginal_effects/`

---

**Generated**: February 14, 2026
**For**: PanelBox Development
