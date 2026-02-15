# True Fixed Effects and True Random Effects Models (Greene 2005)

## Overview

This guide explains Greene's (2005) "True" panel stochastic frontier models that properly separate firm-level heterogeneity from technical inefficiency. These models address a fundamental confounding problem in classical panel SFA approaches.

## The Confounding Problem in Classical Models

### Classical Panel SFA (Pitt-Lee 1981)

Classical models like Pitt-Lee (1981) specify:

```
y_it = X_it'β + v_it - u_i
```

where:
- `u_i` is **time-invariant** inefficiency
- `v_it` is noise

**Problem**: This model confounds:
- **Technological heterogeneity** (legitimate differences between firms)
- **Technical inefficiency** (managerial performance)

If firm A has better technology than firm B, the model attributes this entirely to "inefficiency" of firm B.

### Why This Matters

1. **Efficiency estimates are biased**: Firms with worse technology appear inefficient
2. **Cannot study dynamics**: Time-invariant u_i prevents analyzing efficiency changes
3. **Misleading policy implications**: Recommending "efficiency improvements" for technology gaps

## Greene's Solution: True Models

Greene (2005) proposed two models that separate heterogeneity from inefficiency:

### 1. True Fixed Effects (TFE)

**Model**:
```
y_it = α_i + X_it'β + v_it - u_it
```

where:
- `α_i` = firm-specific fixed effect (heterogeneity)
- `u_it` = time-varying inefficiency (half-normal)
- `v_it` = noise (normal)

**Key Features**:
- Separates α_i (technology/management) from u_it (inefficiency)
- Allows time-varying inefficiency
- α_i can correlate with X (flexible)
- Requires bias correction when T is small

### 2. True Random Effects (TRE)

**Model**:
```
y_it = X_it'β + w_i + v_it - u_it
```

where:
- `w_i ~ N(0, σ²_w)` = random heterogeneity
- `u_it ~ N⁺(0, σ²_u)` = time-varying inefficiency
- `v_it ~ N(0, σ²_v)` = noise

**Key Features**:
- Three-component error structure
- Does not suffer from incidental parameters problem
- Requires w_i independent of X
- More efficient than TFE under independence assumption

## Implementation in PanelBox

### Basic Usage

```python
import numpy as np
from panelbox.frontier.true_models import (
    loglik_true_fixed_effects,
    loglik_true_random_effects
)

# TFE model
theta_tfe = [β, ln(σ²_v), ln(σ²_u)]
loglik = loglik_true_fixed_effects(
    theta_tfe, y, X, entity_id, time_id, sign=1
)

# TRE model
theta_tre = [β, ln(σ²_v), ln(σ²_u), ln(σ²_w)]
loglik = loglik_true_random_effects(
    theta_tre, y, X, entity_id, time_id,
    sign=1, n_quadrature=32, method='gauss-hermite'
)
```

### Estimation Details

#### TFE Estimation

The TFE model uses **concentrated likelihood**:

1. For given (β, σ²_v, σ²_u), maximize over α_i numerically
2. This reduces optimization from N+k to k parameters
3. Much faster than including all α_i in θ

```python
# Get alpha estimates along with likelihood
result = loglik_true_fixed_effects(
    theta, y, X, entity_id, time_id,
    sign=1, return_alpha=True
)

loglik = result['loglik']
alpha_estimates = result['alpha']  # Dict: {firm_id: α_i}
```

#### TRE Estimation

The TRE model requires **numerical integration** over w_i:

**Option 1: Gauss-Hermite Quadrature**
```python
loglik = loglik_true_random_effects(
    theta, y, X, entity_id, time_id,
    sign=1,
    n_quadrature=32,  # More points = more accuracy
    method='gauss-hermite'
)
```

**Option 2: Simulated MLE (Halton sequences)**
```python
loglik = loglik_true_random_effects(
    theta, y, X, entity_id, time_id,
    sign=1,
    n_quadrature=200,  # Used as n_simulations
    method='simulated'
)
```

Recommendations:
- Use Gauss-Hermite with n_quadrature ≥ 20 for accuracy
- Use Simulated MLE for very large N (faster)
- Both methods should give similar results

### Bias Correction for TFE

The **incidental parameters problem** causes bias in α_i when T is small.

**Analytical Correction (Hahn & Newey 2004)**:
```python
from panelbox.frontier.true_models import bias_correct_tfe_analytical

alpha_corrected = bias_correct_tfe_analytical(
    alpha_hat,  # Uncorrected estimates
    T,          # Time periods (scalar or array)
    sigma_v_sq,
    sigma_u_sq
)
```

**Jackknife Correction** (slower but more accurate):
```python
from panelbox.frontier.true_models import bias_correct_tfe_jackknife

result = bias_correct_tfe_jackknife(
    y, X, entity_id, time_id,
    theta, sign=1
)

alpha_corrected = result['alpha_corrected']
bias_estimate = result['bias_estimate']
```

**When to correct**:
- Always apply correction when T < 10
- Analytical correction is fast but approximate
- Jackknife is exact but computationally intensive
- Warning is issued automatically if T < 10

### Model Selection: Hausman Test

Choose between TFE and TRE using the Hausman test:

```python
from panelbox.frontier.tests import hausman_test_tfe_tre

result = hausman_test_tfe_tre(
    params_tfe,   # TFE estimates
    params_tre,   # TRE estimates
    vcov_tfe,     # TFE variance-covariance matrix
    vcov_tre      # TRE variance-covariance matrix
)

print(f"Statistic: {result['statistic']:.4f}")
print(f"P-value: {result['pvalue']:.4f}")
print(f"Recommendation: {result['conclusion']}")  # 'TFE' or 'TRE'
```

**Interpretation**:
- **H0**: TRE is consistent and efficient (w_i ⊥ X)
- **H1**: Only TFE is consistent (w_i correlates with X)

Decision rule:
- If p < 0.05: **Use TFE** (correlation detected)
- If p ≥ 0.05: **Use TRE** (more efficient)

### Variance Decomposition (TRE)

Decompose total variance into three components:

```python
from panelbox.frontier.true_models import variance_decomposition_tre

decomp = variance_decomposition_tre(sigma_v_sq, sigma_u_sq, sigma_w_sq)

print(f"Total variance: {decomp['sigma_total_sq']:.4f}")
print(f"Noise share (γ_v):          {decomp['gamma_v']:.1%}")
print(f"Inefficiency share (γ_u):   {decomp['gamma_u']:.1%}")
print(f"Heterogeneity share (γ_w):  {decomp['gamma_w']:.1%}")
```

**Interpretation**:
- High γ_w: Heterogeneity dominates (technology differences)
- High γ_u: Inefficiency is important (managerial performance)
- High γ_v: Noisy data (random shocks)

## Advanced: True Models with BC95 Determinants

Combine True models with Battese-Coelli (1995) inefficiency determinants:

### TFE + BC95

```python
from panelbox.frontier.true_models import loglik_tfe_bc95

# Model: y_it = α_i + X_it'β + v_it - u_it
# where: u_it ~ N⁺(Z_it'δ, σ²_u)

theta = [β, ln(σ²_v), ln(σ²_u), δ]

loglik = loglik_tfe_bc95(
    theta, y, X, Z, entity_id, time_id, sign=1
)
```

### TRE + BC95

```python
from panelbox.frontier.true_models import loglik_tre_bc95

# Model: y_it = X_it'β + w_i + v_it - u_it
# where: u_it ~ N⁺(Z_it'δ, σ²_u)

theta = [β, ln(σ²_v), ln(σ²_u), ln(σ²_w), δ]

loglik = loglik_tre_bc95(
    theta, y, X, Z, entity_id, time_id,
    sign=1, n_quadrature=32
)
```

**Interpretation of δ**:
- δ_j > 0: Z_j **increases** expected inefficiency
- δ_j < 0: Z_j **decreases** expected inefficiency
- Different from classical BC95: heterogeneity is separately captured

## Practical Guidelines

### When to Use Each Model

| Criterion | TFE | TRE |
|-----------|-----|-----|
| Correlation E[w_i \| X] | Allowed | Requires independence |
| Panel length (T) | Needs T ≥ 10 (or bias correction) | Works for any T |
| Efficiency | Less efficient | More efficient (under H0) |
| Incidental parameters | Yes (needs correction) | No |
| Computational cost | Moderate | High (integration) |

### Recommended Workflow

1. **Estimate both models**:
   ```python
   tfe_result = estimate_tfe(...)
   tre_result = estimate_tre(...)
   ```

2. **Perform Hausman test**:
   ```python
   hausman = hausman_test_tfe_tre(...)
   ```

3. **Choose model based on test**:
   - If TFE: Apply bias correction if T < 10
   - If TRE: Perform variance decomposition

4. **Interpret results**:
   - TFE: α_i shows firm-specific technology levels
   - TRE: γ_w shows importance of heterogeneity

### Common Pitfalls

1. **Forgetting bias correction for TFE**:
   - Always check T
   - Apply correction if T < 10
   - Report both corrected and uncorrected

2. **Too few quadrature points for TRE**:
   - Use at least n_quadrature=20
   - Check sensitivity: does result change with 32 vs 20?

3. **Interpreting u_it as total inefficiency**:
   - In True models, u_it is inefficiency AFTER controlling for α_i/w_i
   - Do not compare u_it to classical models directly

4. **Using Z variables also in X**:
   - Z should be different from X
   - Z affects inefficiency, not frontier

## Computational Considerations

### TFE Performance

- Complexity: O(N × T × k)
- Bottleneck: Optimizing N separate α_i
- Fast for N < 200, T < 20
- For large N: Consider parallel optimization

### TRE Performance

- Complexity: O(N × T × Q) where Q = quadrature points
- Bottleneck: Q-point quadrature per firm
- Recommended: Q=32 for accuracy

**Speed comparison** (N=100, T=10):
- TFE: ~10 seconds
- TRE (Q=20): ~30 seconds
- TRE (Q=32): ~50 seconds
- Simulated (S=100): ~40 seconds

## References

### Primary References

1. **Greene, W. H. (2005)**. "Reconsidering heterogeneity in panel data estimators of the stochastic frontier model." *Journal of Econometrics*, 126(2), 269-303.
   - Original TRE paper
   - Three-component error structure
   - Quadrature integration method

2. **Greene, W. H. (2005)**. "Fixed and random effects in stochastic frontier models." *Journal of Productivity Analysis*, 23(1), 7-32.
   - TFE specification
   - Comparison with classical models
   - Empirical applications

### Bias Correction

3. **Hahn, J., & Newey, W. (2004)**. "Jackknife and analytical bias reduction for nonlinear panel models." *Econometrica*, 72(4), 1295-1319.

4. **Dhaene, G., & Jochmans, K. (2015)**. "Split-panel jackknife estimation of fixed-effect models." *The Review of Economic Studies*, 82(3), 991-1030.

### Classical Models (for comparison)

5. **Pitt, M. M., & Lee, L. F. (1981)**. "The measurement and sources of technical inefficiency in Indonesian weaving industry." *Journal of Development Economics*, 9(1), 43-64.

6. **Battese, G. E., & Coelli, T. J. (1995)**. "A model for technical inefficiency effects in a stochastic frontier production function for panel data." *Empirical Economics*, 20(2), 325-332.

## Examples

See `examples/sfa_true_models.py` for a complete working example with:
- Data generation
- TFE and TRE estimation
- Bias correction
- Hausman test
- Variance decomposition
- Visualizations

## Support

For questions or issues with True models implementation:
- GitHub Issues: https://github.com/panelbox/panelbox/issues
- Documentation: https://panelbox.readthedocs.io
- Example notebooks: `examples/notebooks/true_models_tutorial.ipynb`

---

**Last updated**: February 2025
**PanelBox version**: 0.4.0+
