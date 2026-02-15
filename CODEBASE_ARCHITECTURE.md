# PanelBox Codebase Architecture - Comprehensive Exploration

## Executive Summary

PanelBox is a sophisticated Python econometric library for panel data analysis. The codebase features a well-structured hierarchy of model classes, robust estimation frameworks, and modular utilities for handling discrete choice, count data, and censored models.

---

## 1. MODELS ALREADY IMPLEMENTED

### 1.1 Discrete Choice Models (Binary and Ordered)

#### Binary Choice Models
Located in: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py`

**Implemented:**
- **PooledLogit**: Standard logistic regression with cluster-robust standard errors
  - Log-likelihood: ℓ = Σ[y*log(Λ(η)) + (1-y)*log(1-Λ(η))]
  - Supports three covariance types: 'nonrobust', 'robust', 'cluster'
  - Features: Classification metrics, Hosmer-Lemeshow test, Information Matrix Test, Link Test

- **PooledProbit**: Probit regression with normal CDF
  - Log-likelihood: ℓ = Σ[y*log(Φ(η)) + (1-y)*log(1-Φ(η))]
  - Cluster-robust SEs by default
  - Same diagnostics as Logit

- **FixedEffectsLogit**: Chamberlain (1980) conditional MLE
  - Eliminates fixed effects through conditional likelihood
  - Identifies only time-varying variables
  - Conditional likelihood: ℓᵢ = yᵢ'Xᵢβ - log(Σ_{s:Σsₜ=Σyᵢₜ} exp(s'Xᵢβ))
  - Automatically drops entities with no variation
  - Analytical gradient and Hessian implemented

- **RandomEffectsProbit**: RE Probit with Gauss-Hermite quadrature
  - Integrates out random effects: α_i ~ N(0, σ²_α)
  - Quadrature nodes/weights computed once at init
  - Computes intra-class correlation ρ = σ²_α / (1 + σ²_α)

#### Ordered Choice Models
Located in: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/ordered.py`

**Implemented:**
- **OrderedLogit**: Ordered logistic regression with cutpoints
- **OrderedProbit**: Ordered probit with cutpoints
- **RandomEffectsOrderedLogit**: RE variant with quadrature

**Key Features:**
- Cutpoint parameterization ensures ordering: κ₀ < κ₁ < ... < κ_{J-2}
- Uses log-difference transform: κⱼ = κ_{j-1} + exp(γⱼ)
- Marginal effects differ by outcome category
- Sum of MEs across categories = 0 (verified in computation)

### 1.2 Count Data Models

Located in: `/home/guhaase/projetos/panelbox/panelbox/models/count/`

**Implemented:**
- **PooledPoisson**: Standard Poisson regression
  - Log-likelihood: ℓ = Σ[ylog(λ) - λ - log(y!)]
  - Analytical score and Hessian
  - Overdispersion testing

- **PoissonFixedEffects**: Conditional MLE (Hausman et al. 1984)
- **RandomEffectsPoisson**: With Gamma/Normal distribution assumption
- **PoissonQML**: Quasi-Maximum Likelihood (Wooldridge 1999)

- **NegativeBinomial**: For overdispersed count data
  - Includes NB1 and NB2 parameterizations

### 1.3 Censored Models

Located in: `/home/guhaase/projetos/panelbox/panelbox/models/censored/`

**Implemented:**
- **RandomEffectsTobit**: For left/right/interval censored data
  - y* = X'β + α_i + ε_it
  - y = max(c, y*) for left censoring
  - Uses Gauss-Hermite quadrature for integration
  - Quadrature points: 2-50 nodes supported

- **Honore Model**: Semi-parametric censored model

---

## 2. BASE CLASSES AND INTERFACES

### 2.1 Core Hierarchy

```
PanelModel (ABC)
├── NonlinearPanelModel (ABC)
│   ├── PooledLogit / PooledProbit
│   ├── FixedEffectsLogit
│   ├── RandomEffectsProbit
│   ├── OrderedLogit / OrderedProbit
│   ├── PooledPoisson
│   └── ... (other models)
```

### 2.2 PanelModel (Base Class)

**File:** `/home/guhaase/projetos/panelbox/panelbox/models/discrete/base.py`

```python
class PanelModel(ABC):
    def __init__(self, formula, data, entity_col, time_col, weights=None)

    # Required by subclasses:
    @abstractmethod
    def _log_likelihood(self, params: np.ndarray) -> float

    # Optional overrides:
    def _score(self, params) -> np.ndarray        # Gradient
    def _hessian(self, params) -> np.ndarray      # Hessian
    def _get_starting_values(n_params, method) -> np.ndarray
```

**Key Attributes:**
- `formula`: R-style formula string
- `data`: PanelData container with long-format DataFrame
- `entity_col`, `time_col`: Column names for identifiers
- `weights`: Optional observation weights
- `_fitted`: Boolean tracking fitted state
- `_results`: PanelResults object

### 2.3 NonlinearPanelModel (Extended Base Class)

**File:** `/home/guhaase/projetos/panelbox/panelbox/models/discrete/base.py`

**Key Methods:**

```python
class NonlinearPanelModel(PanelModel):
    def fit(
        self,
        method: Literal["bfgs", "newton", "trust-constr"] = "bfgs",
        start_params: Optional[np.ndarray] = None,
        n_starts: int = 1,
        bounds: Optional = None,
        constraints: Optional = None,
        maxiter: int = 1000,
        verbose: bool = False
    ) -> PanelResults

    def _check_convergence(self, result, params, verbose=False)
        # Checks:
        # - Optimization success
        # - Gradient norm (should be < 1e-3)
        # - Hessian negative definiteness (for maximum)
        # - Hessian condition number (should be < 1e10)
```

**Optimization Features:**
- Multiple starting values to avoid local minima
- Three optimization methods (BFGS, Newton, Trust-Region)
- Convergence diagnostics with gradient/Hessian analysis
- Support for constraints and bounds
- Automatic numerical gradient/Hessian if not overridden

### 2.4 PanelResults (Results Container)

**Key Attributes:**
- `params`: Series of estimated parameters
- `std_errors`: Series of standard errors
- `cov_params`: DataFrame of covariance matrix
- `fittedvalues`: Predicted values
- `resid`: Residuals
- `model_info`: Dictionary with model type, formula, covtype, log-likelihood
- `data_info`: Dictionary with observations, entities, periods, DFs
- `rsquared_dict`: R² metrics (varies by model type)
- Model-specific attributes: `llf`, `ll_null`, `aic`, `bic`, `converged`, etc.

**Standard Methods:**
- `summary()`: Formatted results table
- `predict()`: Predictions from estimated model
- `pseudo_r2()`: McFadden, Cox-Snell, or Nagelkerke R²
- Diagnostic methods (Hosmer-Lemeshow, Information Matrix Test, etc.)

---

## 3. MARGINAL EFFECTS CALCULATION

### 3.1 Marginal Effects Infrastructure

**Location:** `/home/guhaase/projetos/panelbox/panelbox/marginal_effects/`

#### Binary Models (discrete_me.py)

**MarginalEffectsResult Container:**
```python
class MarginalEffectsResult:
    marginal_effects: pd.Series      # Point estimates
    std_errors: pd.Series            # Delta method SEs
    me_type: str                     # 'AME', 'MEM', or 'MER'
    at_values: dict                  # Values for MER

    # Properties:
    z_stats: pd.Series               # ME / SE
    pvalues: pd.Series               # Two-sided test p-values

    # Methods:
    conf_int(alpha=0.05) -> pd.DataFrame
    summary(alpha=0.05) -> pd.DataFrame
```

#### Ordered Models (ordered_me.py)

**OrderedMarginalEffectsResult:**
```python
class OrderedMarginalEffectsResult:
    marginal_effects: pd.DataFrame   # Shape: (n_vars, n_categories)
    std_errors: pd.DataFrame
    n_categories: int

    # Key property:
    verify_sum_to_zero(tol=1e-10) -> bool
        # Sums across categories should equal zero
```

#### Count Models (count_me.py)

- Average Marginal Effects (AME)
- Marginal Effects at Means (MEM)
- Elasticity calculations for semi-log models

### 3.2 Delta Method for Standard Errors

**Location:** `/home/guhaase/projetos/panelbox/panelbox/marginal_effects/delta_method.py`

```python
def delta_method_se(
    me_func: Callable,          # Marginal effect function g(β)
    params: np.ndarray,         # β estimates
    vcov: np.ndarray,           # Var-Cov matrix of β
    numerical: bool = True
) -> np.ndarray
```

**Computation:**
- SE(g(β)) ≈ √[∇g(β)' Var(β) ∇g(β)]
- Supports both analytical and numerical gradients
- Used throughout discrete and count models

### 3.3 Formulas by Model Type

**Logit Marginal Effects:**
```
ME = β * Λ(η) * (1 - Λ(η))
AME = (1/N) Σᵢ ME_i
MEM = β * Λ(η̄) * (1 - Λ(η̄))  where η̄ = X̄'β
```

**Probit Marginal Effects:**
```
ME = β * φ(η)
AME = (1/N) Σᵢ β * φ(Xᵢ'β)
MEM = β * φ(X̄'β)
```

**Ordered Model Marginal Effects:**
- P(y=j) marginal effect = [F(κⱼ - η) - F(κⱼ₋₁ - η)] (depends on j)
- Sum across j equals zero (fundamental property)

**Poisson Marginal Effects:**
```
ME = β * λ = β * exp(X'β)
AME = (1/N) Σᵢ β * exp(Xᵢ'β)
```

---

## 4. QUADRATURE UTILITIES

### 4.1 Gauss-Hermite Quadrature

**Location:** `/home/guhaase/projetos/panelbox/panelbox/optimization/quadrature.py`

#### Core Function
```python
def gauss_hermite_quadrature(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (nodes, weights) for integrating ∫ f(x) φ(x) dx

    Supports n_points from 2 to 50
    Uses scipy.special.roots_hermite for polynomial roots
    Normalizes weights: weights / √π for standard normal
    """
    nodes, weights = special.roots_hermite(n_points)
    weights = weights / np.sqrt(np.pi)
    return nodes, weights
```

#### Integration Functions

```python
def integrate_normal(func, n_points=12, mu=0, sigma=1) -> float:
    """
    Integrates: ∫ func(x) φ(x; μ, σ²) dx
    Transforms nodes: x = μ + √2 * σ * ξ
    """

def adaptive_gauss_hermite(func, n_points_list=[8,12,16,20], tol=1e-8):
    """
    Automatic refinement: increases points until convergence
    Returns: (integral, n_points_used)
    """

def gauss_hermite_2d(n_points) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D tensor product for bivariate integration
    Returns: (nodes_2d, weights_2d)
    """
```

#### Class Interface

```python
class GaussHermiteQuadrature:
    def __init__(self, n_points=12, cache=True)

    # Properties:
    @property
    def nodes -> np.ndarray
    @property
    def weights -> np.ndarray

    # Methods:
    def integrate(func, mu=0, sigma=1) -> float
    def integrate_vectorized(func, mu=0, sigma=1) -> float
```

### 4.2 Application in Panel Models

**RandomEffectsProbit:**
- Initialization: `self._quad_nodes, self._quad_weights = gauss_hermite_quadrature(quad_points)`
- Log-likelihood: Sums quadrature contributions over entities
- Transformation: α_i = √2 * σ_α * ξ (Hermite nodes)

**RandomEffectsTobit:**
- Integrates censoring probability over random effects
- Handles left/right/interval censoring

**OrderedModels with RE:**
- Integrates choice probabilities over random effects

---

## 5. DIAGNOSTIC TOOLS

### 5.1 Convergence Diagnostics (NonlinearPanelModel)

**Integrated in fit() method:**

```python
def _check_convergence(result, params, verbose=False):
    # 1. Optimization success flag
    if not result.success:
        warnings.warn(f"Optimization may not converged: {result.message}")

    # 2. Gradient norm
    score = self._score(params)
    grad_norm = np.linalg.norm(score)
    if grad_norm > 1e-3:
        warnings.warn(f"Large gradient norm: {grad_norm:.6f}")

    # 3. Hessian negative definiteness (for maximum)
    H = self._hessian(params)
    eigenvalues = np.linalg.eigvalsh(H)
    if np.any(eigenvalues > 1e-10):
        warnings.warn("Hessian not negative definite")

    # 4. Condition number
    cond = np.linalg.cond(H)
    if cond > 1e10:
        warnings.warn(f"Poorly conditioned Hessian: cond={cond:.2e}")
```

### 5.2 Specification Tests for Binary Models

**Implemented in PooledLogit results:**

#### Hosmer-Lemeshow Test
- Divides observations into deciles by predicted probability
- Tests match between observed and expected frequencies
- Accounts for panel structure (entities may cluster)

```python
def hosmer_lemeshow_test(n_groups=10) -> dict:
    # Returns: statistic, p_value, df, n_groups, interpretation
    # H-L statistic: Σ (O - E)² / (n_g * π̄ * (1 - π̄))
```

#### Information Matrix Test
- Tests if information matrix equality holds: -E[H] = E[s*s']
- Specification test (White 1982)

```python
def information_matrix_test() -> dict:
    # Returns: statistic, p_value, df, interpretation
```

#### Link Test
- Adds squared linear predictor term
- Tests if squared term is significant
- Misspecification of link function leads to significant squared term

```python
def link_test() -> dict:
    # Returns: squared_term_coef, se, t_stat, p_value, interpretation
```

### 5.3 Classification Metrics (Binary Models)

```python
def classification_metrics(threshold=0.5) -> dict:
    # Returns:
    # - accuracy: (TP + TN) / Total
    # - precision: TP / (TP + FP)
    # - recall: TP / (TP + FN)
    # - f1: 2 * (precision * recall) / (precision + recall)
    # - auc_roc: Area under ROC curve
    # - confusion_matrix: {tp, tn, fp, fn}
```

### 5.4 Model Fit Measures

**For All MLE Models:**
- Log-likelihood: `llf`
- Null log-likelihood: `ll_null`
- Pseudo R²: McFadden = 1 - (ℓ/ℓ₀), Cox-Snell, Nagelkerke variants
- AIC: -2ℓ + 2k
- BIC: -2ℓ + k*log(N)

**For Fixed Effects Models:**
- Proportion of entities with variation
- Number of entities dropped
- Effective sample size for estimation

---

## 6. COVARIANCE MATRIX ESTIMATION

### 6.1 Supported Covariance Types

#### Nonrobust (Classical)
- Assumes IID errors
- Hessian-based: Var(β) = -H⁻¹

#### Heteroskedasticity-Robust (Sandwich)
- Huber-White sandwich estimator
- Var(β) = H⁻¹ * B * H⁻¹
- Where B = Σ sᵢ sᵢ' (outer product of scores)

#### Cluster-Robust
- Accounts for within-cluster correlation
- Clusters by entity
- Adjustment factor: (G/(G-1)) * (N/(N-1))
- Where G = number of entities, N = total observations

### 6.2 Implementation Details

**PooledLogit example:**
```python
# Logit score for observation i:
s_i = (y_i - Λ(η_i)) * X_i

# Hessian (information matrix):
W = Λ(η) * (1 - Λ(η))  # Weights
H = -(X.T * diag(W)) @ X

# Cluster-robust:
result = cluster_robust_mle(H, scores, entities, df_correction=True)
vcov = result.cov_matrix
```

---

## 7. DATA STRUCTURES

### 7.1 PanelData Container

**From core.base_model:**
```python
class PanelData:
    data: pd.DataFrame              # Long-format data
    entity_col: str                 # Entity identifier column
    time_col: str                   # Time identifier column
    n_entities: int                 # Number of unique entities
    n_periods: int                  # Number of unique time periods
    n_obs: int                      # Total observations
```

### 7.2 Formula Parsing

```python
class FormulaParser:
    # Parses R-style formulas: "y ~ x1 + x2 + x3"

    def build_design_matrices(data, return_type='array'):
        # Returns: (y, X) as arrays or DataFrames

    def get_variable_names(data) -> list:
        # Returns: ['const', 'x1', 'x2', ...]

    @property
    def has_intercept -> bool
```

---

## 8. ARCHITECTURE PATTERNS

### 8.1 Template Method Pattern (Fit)

All models inherit fit() from NonlinearPanelModel:

```
fit()
├── Build design matrices (formula_parser)
├── Get starting values (_get_starting_values or override)
├── For each random start (n_starts loop):
│   ├── Optimize: minimize(-_log_likelihood)
│   ├── Track best result (highest LL)
├── Check convergence (_check_convergence)
├── Create results (_create_results - implemented by subclass)
└── Return PanelResults
```

### 8.2 Extensibility Points

**Subclasses should override:**
1. `_log_likelihood(params)` - Core computation (required)
2. `_score(params)` - Gradient (optional, uses numerical if not provided)
3. `_hessian(params)` - Hessian (optional, uses numerical if not provided)
4. `_create_results(...)` - Results object customization
5. `marginal_effects()` - Model-specific ME computation
6. `predict()` - Model-specific prediction logic

### 8.3 Standard Error Computation Pipeline

```
Fitted Model
├── Hessian H (analytical or numerical)
├── Scores sᵢ (analytical or numerical)
├── Choose covariance type (nonrobust/robust/cluster)
│   ├── Nonrobust: Var(β) = -H⁻¹
│   ├── Robust: Var(β) = H⁻¹ * (Σ sᵢ sᵢ') * H⁻¹
│   └── Cluster: Var(β) = H⁻¹ * (Σ_c (Σ_i∈c sᵢ)(Σ_i∈c sᵢ)') * H⁻¹
├── Invert Hessian (with pinv fallback)
└── Compute SE = √diag(Var)
```

---

## 9. KEY UTILITY MODULES

### 9.1 Data Utilities (utils/data.py)

```python
def check_panel_data(y, X, entity_id, time_id, weights):
    # Validates panel structure, returns numpy arrays

def get_panel_info(entity_id, time_id) -> dict:
    # Returns: n_entities, n_periods, n_obs, balanced status
```

### 9.2 Statistical Utilities (utils/statistics.py)

```python
def compute_sandwich_covariance(hessian, gradients, entity_id=None):
    # Sandwich estimator with optional clustering

def compute_cluster_robust_covariance(residuals, X, entity_id, vcov_base=None):
    # Entity-level clustering
```

### 9.3 Numerical Gradient (optimization/numerical_grad.py)

```python
def approx_gradient(func, params, method='central'):
    # Central differences: (f(x+h) - f(x-h)) / (2h)

def approx_hessian(func, params, method='central'):
    # Second-order finite differences
```

---

## 10. IMPLEMENTATION GUIDELINES FOR NEW MODELS

### 10.1 For Censored Models

**Must Implement:**
1. `_log_likelihood(params)` with:
   - Censoring point extraction
   - Normal CDF for continuous part
   - Probability mass at censoring point
   - Numerical stability (use `stats.norm.cdf`, clip probabilities)

2. `_create_results()` with:
   - Fitted values (conditional mean or probabilities)
   - Residuals (accounting for censoring)
   - Pseudo R² if applicable
   - Model-specific attributes (censoring info)

**Should Implement:**
1. `marginal_effects()` for changes in conditional expectation
2. `predict()` with options for censored/uncensored predictions

**Quadrature Integration Pattern:**
```python
for entity in entities:
    entity_llf = 0
    for node, weight in zip(quad_nodes, quad_weights):
        # Transform: alpha = √2 * sigma * node
        # Evaluate: censoring probability or density
        # Accumulate: entity_llf += weight * prob_or_density
    total_llf += log(entity_llf)
```

### 10.2 For Ordered Models

**Must Implement:**
1. Cutpoint ordering mechanism
   - Use: κⱼ = κ_{j-1} + exp(γⱼ)
   - Transform: unconstrained → ordered cutpoints

2. Log-likelihood with cutpoints:
   ```python
   P(y=j) = F(κⱼ - η) - F(κⱼ₋₁ - η)
   ℓ = Σᵢ log P(yᵢ)
   ```

3. Marginal effects verification:
   - Sum across categories = 0
   - Implement `verify_sum_to_zero()` check

### 10.3 For Count Models

**Standard Pattern:**
1. Mean function: λ = exp(X'β) for Poisson/NB
2. Variance function: Var(y) = λ for Poisson, λ + αλ² for NB1
3. Score: Σ (y - λ) * X
4. Hessian: -Σ λ * X * X'

---

## 11. FILE ORGANIZATION

```
panelbox/
├── models/
│   ├── base.py                    ← NonlinearPanelModel
│   ├── discrete/
│   │   ├── base.py                ← NonlinearPanelModel
│   │   ├── binary.py              ← Logit, Probit, FE, RE variants
│   │   ├── ordered.py             ← Ordered choice models
│   │   └── results.py
│   ├── count/
│   │   ├── poisson.py             ← Pooled, FE, RE, QML
│   │   └── negbin.py              ← Negative Binomial variants
│   └── censored/
│       ├── tobit.py               ← Tobit models
│       └── honore.py              ← Semi-parametric censored
│
├── marginal_effects/
│   ├── discrete_me.py             ← Binary model ME
│   ├── ordered_me.py              ← Ordered model ME
│   ├── count_me.py                ← Count model ME
│   └── delta_method.py            ← Delta method SE
│
├── optimization/
│   ├── quadrature.py              ← Gauss-Hermite
│   └── numerical_grad.py          ← Finite differences
│
├── diagnostics/
│   └── hausman.py                 ← Hausman test
│
├── standard_errors/
│   ├── cluster_robust.py
│   └── mle.py                     ← cluster_robust_mle
│
└── utils/
    ├── data.py                    ← Panel data validation
    └── statistics.py              ← Covariance computation
```

---

## 12. CONVERGENCE AND OPTIMIZATION NOTES

### 12.1 Starting Values

**Default Strategy:**
1. Try zeros first (often good for logit/probit)
2. If `n_starts > 1`, try random perturbations
3. For RE models, use Pooled model estimates as warmstart

**For Ordered/Censored:**
1. Initialize cutpoints at quantiles of y
2. Initialize β from linear approximation

### 12.2 Optimization Method Selection

| Method | Best For | Requires | Notes |
|--------|----------|----------|-------|
| BFGS | General | Gradient (numerical OK) | Robust, default |
| Newton | Analytical Hessian | Hessian | Faster, needs good Hessian |
| Trust-Constr | Constrained | Bounds/Constraints | Most flexible, slower |

### 12.3 Convergence Criteria

- Gradient norm: < 1e-3 (warning if larger)
- Hessian negative definite: eigenvalues < -1e-10
- Hessian condition: < 1e10 (warning if larger)
- Optimizer success flag: checked and reported

---

## Summary

PanelBox provides:

1. **Hierarchical Model Architecture**: Clear inheritance enabling code reuse and extensibility
2. **Comprehensive MLE Framework**: Optimization, convergence checking, multiple SE types
3. **Quadrature Integration**: For random effects and censored models
4. **Marginal Effects**: With delta method standard errors
5. **Diagnostic Tools**: Specification tests, classification metrics, fit measures
6. **Panel-Specific Features**: Entity/time clustering, unbalanced panels, fixed effects handling

The codebase is ready for implementing new censored and ordered variants by following the established patterns.
