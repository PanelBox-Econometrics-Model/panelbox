# PanelBox Codebase Structure Analysis: Count Models Implementation Guide

## 1. OVERALL CODEBASE ORGANIZATION

### Project Structure
```
panelbox/
├── models/
│   ├── discrete/              # Discrete choice and limited dependent variable models
│   │   ├── base.py           # NonlinearPanelModel - abstract base class for MLE
│   │   ├── binary.py         # Binary choice models (Logit, Probit, FE Logit, RE Probit)
│   │   ├── results.py        # Results container (placeholder)
│   │   └── __init__.py       # Module exports
│   ├── static/               # Linear panel models (FE, RE, Pooled OLS)
│   ├── dynamic/              # Dynamic panel models
│   └── iv/                   # IV/2SLS models
├── marginal_effects/         # Marginal effects computation
│   ├── discrete_me.py        # AME, MEM, MER for binary choice
│   └── delta_method.py       # Delta method for standard errors
├── optimization/             # Numerical optimization utilities
│   ├── quadrature.py         # Gauss-Hermite quadrature
│   └── numerical_grad.py     # Gradient/Hessian approximation
├── core/
│   ├── base_model.py        # PanelModel abstract base class
│   ├── results.py           # PanelResults container class
│   └── panel_data.py        # Panel data container
└── diagnostics/             # Diagnostic tests and validation
```

## 2. NONLINEAR PANEL MODEL BASE CLASS

### Location: `panelbox/models/discrete/base.py`

**Key Class: `NonlinearPanelModel`**
- Inherits from: `PanelModel`
- Handles Maximum Likelihood Estimation (MLE) for all discrete models
- Provides core optimization infrastructure

**Abstract Methods to Implement:**
```python
def _log_likelihood(self, params: np.ndarray) -> float:
    """Must return scalar log-likelihood value"""
    pass
```

**Optional Methods to Override:**
```python
def _score(self, params: np.ndarray) -> np.ndarray:
    """Gradient of log-likelihood. Default uses numerical differentiation"""

def _hessian(self, params: np.ndarray) -> np.ndarray:
    """Hessian matrix. Default uses numerical differentiation"""
```

**Key Methods (Inherited):**
- `fit()`: Main estimation method with options for multiple optimization algorithms
  - Supports: BFGS (default), Newton-Raphson, Trust-Region
  - Multiple starting values to avoid local minima
  - Convergence diagnostics
- `_check_convergence()`: Validates convergence with warnings
- `_get_starting_values()`: Generates starting parameter values

**Important Design Patterns:**
1. All subclasses must implement `_log_likelihood()` returning scalar
2. Numerical gradient/Hessian fallback when analytical not available
3. MLE optimization via scipy.optimize.minimize
4. Support for weighted models via `weights` parameter

## 3. EXISTING BINARY CHOICE MODELS

### Location: `panelbox/models/discrete/binary.py`

**Implemented Models:**

#### 1. PooledLogit
- Pools all observations, ignores panel structure
- Log-likelihood: ℓ = Σ[y*log(Λ(η)) + (1-y)*log(1-Λ(η))]
- Uses statsmodels.Logit wrapper
- Supports cluster-robust SEs by entity
- Methods: `fit()`, `predict()`, `marginal_effects()`
- Additional methods:
  - `classification_metrics()`: Accuracy, precision, recall, F1, AUC
  - `hosmer_lemeshow_test()`: Goodness-of-fit
  - `information_matrix_test()`: Model specification
  - `link_test()`: Link function validation
  - `pseudo_r2()`: McFadden, Cox-Snell, Nagelkerke R²

#### 2. PooledProbit
- Similar to PooledLogit but with normal CDF (Φ)
- Log-likelihood: ℓ = Σ[y*log(Φ(η)) + (1-y)*log(1-Φ(η))]
- Similar methods and diagnostics to PooledLogit

#### 3. FixedEffectsLogit
- Chamberlain (1980) conditional MLE
- Eliminates fixed effects using sufficient statistic conditioning
- Only includes entities with temporal variation (0 < Σyᵢₜ < Tᵢ)
- Analytical score and Hessian implemented
- Enumeration-based for T ≤ 20 periods
- Properties: `n_used_entities`, `n_dropped_entities`, `entities_with_variation`

#### 4. RandomEffectsProbit
- Random effects α_i ~ N(0, σ²_α)
- Integrates out random effects using Gauss-Hermite quadrature
- Quadrature nodes/weights: `_quad_nodes`, `_quad_weights`
- Parameterization: [β; log(σ_α)]
- Properties: `rho` (intra-class correlation), `sigma_alpha`
- Starting values from Pooled Probit
- Methods for marginal effects integration

## 4. CORE PATTERNS FOR DISCRETE MODELS

### Pattern 1: Results Creation
All models follow this pattern:
```python
def fit(self, ...):
    # Build design matrices
    y, X = self.formula_parser.build_design_matrices(...)
    var_names = self.formula_parser.get_variable_names(...)

    # Fit model (via parent class or custom)
    results = PanelResults(
        params=pd.Series(params, index=var_names),
        std_errors=pd.Series(std_errors, index=var_names),
        cov_params=pd.DataFrame(vcov, index=var_names, columns=var_names),
        resid=resid,
        fittedvalues=fitted_probs,
        model_info={...},
        data_info={...},
        rsquared_dict={...},
        model=self
    )
    return results
```

### Pattern 2: Standard Errors Computation
Three types implemented in binary models:
1. **Nonrobust**: Classical (assumes iid)
2. **Robust**: Heteroskedasticity-robust (sandwich)
3. **Cluster**: Cluster-robust by entity

```python
# For logit-type models:
# H = -Σ Λ(η)[1-Λ(η)] Xᵢ Xᵢ'
# S = Σ sᵢ sᵢ' where sᵢ = (yᵢ - Λ(ηᵢ)) Xᵢ
# V = H⁻¹ S H⁻¹
```

### Pattern 3: Model Information Dictionary
```python
model_info = {
    "model_type": "PooledLogit",
    "formula": self.formula,
    "cov_type": cov_type,
    "cov_kwds": {},
    "llf": llf,
    "ll_null": ll_null,
}
```

### Pattern 4: Data Information Dictionary
```python
data_info = {
    "nobs": n,
    "n_entities": self.data.n_entities,
    "n_periods": self.data.n_periods,
    "df_model": df_model,
    "df_resid": df_resid,
    "entity_index": entity_values,
    "time_index": time_values,
}
```

## 5. MARGINAL EFFECTS MODULE

### Location: `panelbox/marginal_effects/discrete_me.py`

**Key Classes:**
- `MarginalEffectsResult`: Container for marginal effects with SEs
  - Properties: `z_stats`, `pvalues`
  - Methods: `conf_int()`, `summary()`

**Key Functions:**
1. `compute_ame()`: Average Marginal Effects
   - For continuous: Average of ∂P/∂x across observations
   - For binary: Average discrete difference

2. `compute_mem()`: Marginal Effects at Means
   - Evaluated at X̄ (mean of covariates)

3. `compute_mer()`: Marginal Effects at Representative Values
   - User-specified representative values

**Integration Pattern:**
```python
# For Logit
me_i = β_k * Λ(X'β)[1 - Λ(X'β)]

# For Probit
me_i = β_k * φ(X'β)
```

**Standard Errors:**
- Delta method via `delta_method_se()`
- Numerical gradient computation
- Handles binary variables via discrete difference

## 6. OPTIMIZATION UTILITIES

### Location: `panelbox/optimization/`

#### A. Quadrature (`quadrature.py`)
**Main Function:**
```python
gauss_hermite_quadrature(n_points: int) -> (nodes, weights)
```
- Hermite polynomial roots and weights
- Normalized for standard normal integration
- Used in RandomEffectsProbit for integrating random effects

**Other Functions:**
- `integrate_normal()`: Integrate function over normal dist
- `adaptive_gauss_hermite()`: Auto-select quadrature points
- `integrate_product_normal()`: For panel data products
- `gauss_hermite_2d()`: Bivariate integration
- `GaussHermiteQuadrature`: Class interface for repeated use

#### B. Numerical Gradients (`numerical_grad.py`)
**Functions:**
1. `approx_gradient(func, x, method='central', epsilon='auto')`
   - Central difference: O(h²) accuracy
   - Forward difference: O(h) accuracy
   - Automatic step size: h = √ε_mach * max(1, |x|)

2. `approx_hessian(func, x, method='central', epsilon='auto')`
   - Central difference for diagonal and off-diagonal
   - Step size: h = ε_mach^(1/3) * max(1, |x|)
   - Enforces symmetry

**Usage in Discrete Models:**
- Default for _score() and _hessian() when not overridden
- Used in convergence checking
- Central differences provide O(h²) accuracy

## 7. DESIGN PATTERNS AND CONVENTIONS

### Pattern 1: Linear Predictor
All models compute:
```python
eta = X @ params  # Linear predictor
```

### Pattern 2: Link Functions
- Logit: Λ(z) = 1/(1 + exp(-z)) = exp(z)/(1 + exp(z))
- Probit: Φ(z) = norm.cdf(z)
- Log (for count): exp(z)

### Pattern 3: Log-Likelihood Calculation
Always returns **scalar** float:
```python
# Correct
return float(np.sum(...))

# Wrong
return np.sum(...)  # Returns array
```

### Pattern 4: Weight Handling
```python
if self.weights is not None:
    ll = np.sum(self.weights * (y * eta - log1p(exp(eta))))
else:
    ll = np.sum(y * eta - log1p(exp(eta)))
```

### Pattern 5: Numerical Stability
- Logit: Use `np.log1p()` and `expit()` to avoid overflow
- Probit: Clip probabilities to avoid log(0)
- Always use log-sum-exp tricks when available

### Pattern 6: Formula Parsing
```python
y, X = self.formula_parser.build_design_matrices(
    self.data.data,
    return_type="array"
)
var_names = self.formula_parser.get_variable_names(self.data.data)
```

## 8. KEY CLASSES AND INHERITANCE

```
PanelModel (abstract)
  ↓
NonlinearPanelModel (abstract, MLE infrastructure)
  ↓
├─ PooledLogit
├─ PooledProbit
├─ FixedEffectsLogit
├─ RandomEffectsProbit
└─ [COUNT MODELS GO HERE]
```

## 9. TESTING PATTERNS

### Location: `tests/models/discrete/`

**Test Structure:**
- Fixture-based setup with synthetic data generation
- Use `np.random.seed()` for reproducibility
- Test both functionality and edge cases
- Compare with Pooled models where applicable
- Validation against expected parameter signs

**Key Test Classes:**
- `TestRandomEffectsProbit`: 16+ test methods
- `TestREProbitEdgeCases`: Edge cases
- `TestREProbitStartingValues`: Starting value validation

## 10. MARGINAL EFFECTS AND RESULTS

### Results Object Pattern
```python
results.params              # pd.Series of coefficients
results.std_errors         # pd.Series of standard errors
results.cov_params         # pd.DataFrame covariance matrix
results.tvalues            # t-statistics
results.pvalues            # p-values
results.llf                # Log-likelihood
results.pseudo_r2()        # Pseudo R² method
results.classification_metrics()  # Metrics method
results.predict()          # Prediction method
results.summary()          # Summary table
```

### Add Custom Methods to Results:
```python
def custom_method(...):
    """Custom computation using params, X, y"""
    pass

results.custom_method = custom_method
```

## 11. NUMERICAL CONSIDERATIONS

### Convergence Checks
- Gradient norm: Should be < 1e-3
- Hessian eigenvalues: Should be negative (maximum)
- Condition number: Should be < 1e10
- Warnings issued if issues detected

### Optimization Options
```python
results = model.fit(
    method='bfgs',        # 'bfgs', 'newton', 'trust-constr'
    start_params=None,    # Starting values
    n_starts=1,          # Multiple starts to avoid local minima
    maxiter=1000,        # Maximum iterations
    verbose=False        # Print progress
)
```

## 12. CRITICAL IMPLEMENTATION REQUIREMENTS FOR COUNT MODELS

1. **_log_likelihood() Method**
   - MUST return scalar float
   - Support weights if provided
   - Use numerically stable computations

2. **Score and Hessian**
   - Can use numerical (default) or analytical
   - Must be compatible with scipy.optimize

3. **Results Creation**
   - Follow PanelResults pattern
   - Include model_info and data_info dicts
   - Support summary() method

4. **Standard Errors**
   - Implement cluster-robust version
   - Support at least nonrobust and cluster types

5. **Predictions**
   - Linear predictor (link scale)
   - Response scale (probabilities for binary, counts for Poisson)
   - Can add to results.predict() method

6. **Edge Cases**
   - Handle perfect separation
   - Handle zero counts
   - Handle convergence failures gracefully

## 13. NAMING CONVENTIONS

- Model classes: CamelCase (PooledLogit, FixedEffectsLogit)
- Methods: snake_case (fit, predict, marginal_effects)
- Private methods: _snake_case (_log_likelihood, _score)
- Parameters: β or params in documentation
- Entities: firm_id, person_id, country_id
- Time: year, quarter, time

## 14. KEY FILES AND LOCATIONS

**For Count Models Implementation:**
```
panelbox/models/discrete/
├── base.py                 # Reuse NonlinearPanelModel
├── binary.py              # Patterns to follow
├── count.py               # [NEW FILE FOR COUNT MODELS]
├── __init__.py            # Export new classes
└── README.md              # Document count models
```

**Tests:**
```
tests/models/discrete/
├── test_count_poisson.py         # [NEW]
├── test_count_negative_binomial.py # [NEW]
└── test_re_probit.py            # Reference pattern
```

**Integration Points:**
- `panelbox/__init__.py`: Export new classes
- `panelbox/marginal_effects/discrete_me.py`: Extend for count
- `panelbox/diagnostics/`: Add count-specific tests
