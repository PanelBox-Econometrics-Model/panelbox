# Codebase Exploration Summary: Count Models Implementation

## Executive Summary

This document summarizes the exploration of the PanelBox codebase to understand the patterns and architecture needed for implementing count data models (Poisson, Negative Binomial, etc.).

## What Was Explored

### 1. Directory Structure (panelbox/models/discrete/)
- **base.py**: Core NonlinearPanelModel class - handles MLE optimization
- **binary.py**: 4 implemented binary choice models (2,264 lines)
  - PooledLogit (uses statsmodels wrapper)
  - PooledProbit (analytical link function)
  - FixedEffectsLogit (Chamberlain conditional MLE)
  - RandomEffectsProbit (Gauss-Hermite quadrature)
- **results.py**: Results container (minimal)
- **__init__.py**: Module exports
- **README.md**: Comprehensive documentation of models

### 2. Core Infrastructure Classes

#### NonlinearPanelModel (base.py)
- Abstract base class for all MLE-based discrete models
- Key methods:
  - `fit()`: Main estimation with multiple algorithms (BFGS, Newton, trust-constr)
  - `_log_likelihood()`: Abstract - must be implemented by subclasses
  - `_score()`, `_hessian()`: Use numerical differentiation by default
  - `_check_convergence()`: Validates convergence with diagnostics

#### PanelResults (core/results.py)
- Stores estimation results with comprehensive attributes
- Auto-computes: tvalues, pvalues, degrees of freedom
- Supports custom methods attachment
- Pattern: Always include model_info and data_info dicts

### 3. Marginal Effects Module

**Location**: panelbox/marginal_effects/discrete_me.py
- `MarginalEffectsResult`: Container with properties z_stats, pvalues, methods conf_int(), summary()
- `compute_ame()`: Average Marginal Effects
- `compute_mem()`: Marginal Effects at Means
- `compute_mer()`: Marginal Effects at Representative Values
- Uses delta method for standard errors
- Automatically detects binary variables for discrete differences

### 4. Optimization Utilities

#### Quadrature (optimization/quadrature.py)
- `gauss_hermite_quadrature(n_points)`: Main function, returns nodes and weights
- `integrate_normal()`: Integrate function over normal distribution
- `adaptive_gauss_hermite()`: Auto-select precision
- Used by RandomEffectsProbit for integrating random effects

#### Numerical Gradients (optimization/numerical_grad.py)
- `approx_gradient()`: Central or forward differences
- `approx_hessian()`: Second-order finite differences
- Automatic step size: h = eps^(1/2) for gradients, eps^(1/3) for Hessians
- Enforces Hessian symmetry

### 5. Testing Framework

**Location**: tests/models/discrete/test_re_probit.py
- Fixture-based setup with synthetic data generation
- Tests cover: initialization, fitting, convergence, predictions, edge cases
- Compares with baseline models (Pooled Probit)
- Validates parameter signs and ranges
- Tests for perfect separation, no variation, small SEs cases

## Key Design Patterns

### Pattern 1: Model Class Structure
```
PanelModel (abstract)
  └─ NonlinearPanelModel (abstract, MLE)
      └─ PooledLogit / PooledProbit / etc.
```

### Pattern 2: Log-Likelihood
- MUST return scalar float, not array
- Supports weights if self.weights is not None
- Uses numerically stable computations

### Pattern 3: Results Creation
- Build design matrices via formula_parser
- Compute fitted values and residuals
- Create pd.Series/DataFrame for params, SEs, covariance
- Create model_info and data_info dicts
- Return PanelResults object

### Pattern 4: Standard Errors
Three types supported:
1. **nonrobust**: Classical (assumes iid)
2. **robust**: Heteroskedasticity-robust sandwich
3. **cluster**: Cluster-robust by entity

Formula: Vcov = H^{-1} S H^{-1}
- H: Information matrix (negative Hessian)
- S: Meat matrix (outer product of scores)

### Pattern 5: Covariance Computation
```python
# H is the Hessian, varies by model
# For Logit: H = -Σ Λ(η)[1-Λ(η)] X_i X_i'
# For Poisson: H = -Σ λ_i X_i X_i'

# Scores (residuals weighted by design)
scores = (y - fitted)[:, np.newaxis] * X

# For cluster-robust, use cluster_robust_mle helper
from panelbox.standard_errors.mle import cluster_robust_mle
vcov = cluster_robust_mle(H, scores, entities).cov_matrix
```

## Critical Implementation Requirements

### For Count Models (Poisson, NB, etc.)

1. **_log_likelihood() method**
   - Return scalar float
   - Handle weights
   - Support both Poisson (λ = exp(X'β)) and NB (with shape parameter)

2. **Standard Errors**
   - Key difference: Variance function depends on mean (heteroskedasticity)
   - For Poisson: Var(y|X) = λ
   - For NB: Var(y|X) = λ + α*λ²
   - Scores always: (y - fitted) * X

3. **Predictions**
   - Linear: X'β (on log scale for Poisson)
   - Response: exp(X'β) (expected counts)

4. **Results Object**
   - Use pseudo R² for count models
   - Pseudo R² = 1 - LL_model / LL_null
   - Include AIC, BIC

5. **Edge Cases**
   - Handle zero counts
   - Handle missing values
   - Handle numerical overflow (clip large eta)

## File Locations and Absolute Paths

Key files for reference:
- `/home/guhaase/projetos/panelbox/panelbox/models/discrete/base.py` (370 lines)
- `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (2,264 lines)
- `/home/guhaase/projetos/panelbox/panelbox/marginal_effects/discrete_me.py` (587 lines)
- `/home/guhaase/projetos/panelbox/panelbox/optimization/quadrature.py` (358 lines)
- `/home/guhaase/projetos/panelbox/panelbox/optimization/numerical_grad.py` (314 lines)
- `/home/guhaase/projetos/panelbox/tests/models/discrete/test_re_probit.py` (429 lines)

## Naming Conventions

- **Classes**: CamelCase (PooledLogit, FixedEffectsLogit)
- **Methods**: snake_case (fit, predict, marginal_effects)
- **Private methods**: _snake_case (_log_likelihood, _score, _hessian)
- **Parameters**: β in math, params in code
- **Variables**: descriptive snake_case (fitted_probs, linear_pred, etc.)

## Next Steps for Implementation

1. **Create panelbox/models/discrete/count.py**
   - PooledPoisson
   - PooledNegativeBinomial
   - Follow patterns from binary.py exactly

2. **Add tests**: tests/models/discrete/test_count_*.py
   - Use fixture-based setup
   - Test against synthetic data with known parameters

3. **Update marginal_effects**
   - Add support for count models
   - ME formula: β_k * E[λ]

4. **Update documentation**
   - Add mathematical formulations
   - Add usage examples
   - Update README.md with count models

5. **Integration points**
   - Update `panelbox/__init__.py` to export new classes
   - Update `panelbox/models/discrete/__init__.py`
   - Add count models to `panelbox/models/__init__.py`

## Documentation Generated

Three detailed documents have been created:

1. **CODEBASE_STRUCTURE_ANALYSIS.md** (14 sections)
   - Complete architectural overview
   - Inheritance hierarchy
   - All key classes and methods
   - Design patterns
   - Naming conventions

2. **COUNT_MODELS_IMPLEMENTATION_GUIDE.md** (9 sections)
   - Ready-to-use code examples
   - Exact patterns to follow
   - Testing templates
   - Critical checklist
   - Numerical stability tips

3. **EXPLORATION_SUMMARY.md** (this document)
   - High-level overview
   - What was explored
   - Key findings
   - Next steps

## Conclusion

The PanelBox codebase is well-organized and follows consistent patterns. The NonlinearPanelModel base class provides excellent infrastructure for MLE estimation. Implementing count models will be straightforward by:

1. Inheriting from NonlinearPanelModel
2. Implementing _log_likelihood() for Poisson/NB
3. Following the exact patterns used in PooledLogit/PooledProbit
4. Adding tests following the RandomEffectsProbit test template
5. Integrating with marginal effects module

All required infrastructure (optimization, gradients, covariance computation, quadrature) is already in place.
