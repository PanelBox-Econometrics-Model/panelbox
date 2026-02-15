# Selection Models API Reference

## Overview

The `panelbox.models.selection` module provides models for handling sample selection bias in panel data. The main implementation is the **Panel Heckman** model following Wooldridge (1995).

## Panel Heckman Model

### PanelHeckman

```python
from panelbox.models.selection import PanelHeckman

model = PanelHeckman(
    endog,
    exog,
    selection,
    exog_selection,
    entity=None,
    time=None,
    method='two_step'
)
```

**Parameters:**

- `endog` : array-like
  - Outcome variable (observed only if selected)
  - Should contain NaN for non-selected observations

- `exog` : array-like
  - Regressors for outcome equation (X)
  - Shape: (n_obs, k_outcome)

- `selection` : array-like
  - Binary selection indicator (1 if observed, 0 otherwise)
  - Shape: (n_obs,)

- `exog_selection` : array-like
  - Regressors for selection equation (Z)
  - Shape: (n_obs, k_selection)
  - **Should include at least one exclusion restriction** (variable not in `exog`)

- `entity` : array-like, optional
  - Entity/individual identifiers
  - Shape: (n_obs,)

- `time` : array-like, optional
  - Time period identifiers
  - Shape: (n_obs,)

- `method` : str, default='two_step'
  - Estimation method:
    - `'two_step'`: Heckman two-step estimator
    - `'mle'`: Full information maximum likelihood

**Methods:**

#### fit(method=None, **kwargs)

Estimate the Heckman model.

```python
result = model.fit()  # Uses default method
result_mle = model.fit(method='mle')  # Override method
```

**Returns:** `PanelHeckmanResult` object

---

### PanelHeckmanResult

Result object from Panel Heckman estimation.

**Attributes:**

- `params` : np.ndarray
  - Combined parameter vector [β, γ, σ, ρ]

- `outcome_params` : np.ndarray
  - Outcome equation coefficients (β)

- `probit_params` : np.ndarray
  - Selection equation coefficients (γ)

- `sigma` : float
  - Standard deviation of outcome equation errors

- `rho` : float
  - Correlation between selection and outcome errors
  - ρ = Corr(v_it, ε_it)

- `lambda_imr` : np.ndarray
  - Inverse Mills Ratio for each observation

- `method` : str
  - Estimation method used ('two_step' or 'mle')

- `llf` : float (MLE only)
  - Log-likelihood value

- `converged` : bool
  - Whether optimization converged

**Methods:**

#### summary()

Print summary of estimation results.

```python
print(result.summary())
```

**Example Output:**
```
Panel Heckman Selection Model Results
============================================================
Method: TWO_STEP
Total observations: 2500
Selected observations: 1876
Censored observations: 624

Selection Equation (Probit):
----------------------------------------
gamma_0: -2.4532
gamma_1: 0.0487
gamma_2: 0.1456
...

Outcome Equation:
----------------------------------------
beta_0: 1.5234
beta_1: 0.0298
beta_2: 0.0789
...

Selection Parameters:
----------------------------------------
sigma: 0.5123
rho: 0.3876

Note: Positive selection (rho > 0)
Selection bias is present. OLS would be biased.
```

#### predict(exog=None, exog_selection=None, type='unconditional')

Generate predictions.

```python
# Unconditional expectation: E[y*] (latent outcome)
pred_uncond = result.predict(type='unconditional')

# Conditional expectation: E[y|selected] (observed outcome)
pred_cond = result.predict(type='conditional')
```

**Parameters:**

- `exog` : np.ndarray, optional
  - Outcome regressors for prediction
  - If None, uses training data

- `exog_selection` : np.ndarray, optional
  - Selection regressors for prediction
  - If None, uses training data

- `type` : str, default='unconditional'
  - `'unconditional'`: E[y*] without selection correction
  - `'conditional'`: E[y|selected] with selection correction

**Returns:** np.ndarray of predictions

#### selection_effect(alpha=0.05)

Test for selection bias (H₀: ρ = 0).

```python
test = result.selection_effect()
print(test['interpretation'])
print(f"P-value: {test['pvalue']:.4f}")
```

**Returns:** dict with keys:

- `'statistic'`: test statistic
- `'pvalue'`: two-sided p-value
- `'reject'`: bool (reject H₀ at alpha level)
- `'interpretation'`: str describing result

**Interpretation:**

- Reject H₀ → Selection bias present, Heckman correction necessary
- Fail to reject → No significant selection bias, OLS may be adequate

#### imr_diagnostics()

Compute diagnostic statistics for Inverse Mills Ratio.

```python
diag = result.imr_diagnostics()
print(f"Mean IMR: {diag['imr_mean']:.3f}")
print(f"High IMR count: {diag['high_imr_count']}")
```

**Returns:** dict with keys:

- `'imr_mean'`: mean IMR for selected observations
- `'imr_std'`: std dev of IMR
- `'imr_min'`: minimum IMR
- `'imr_max'`: maximum IMR
- `'high_imr_count'`: count of observations with IMR > 2
- `'selection_rate'`: fraction selected
- `'n_selected'`: number selected
- `'n_total'`: total observations

**Notes:**

- High IMR values (> 2) indicate strong selection effects
- Mean IMR indicates average selection correction magnitude

#### compare_ols_heckman()

Compare OLS (biased) vs Heckman (corrected) estimates.

```python
comparison = result.compare_ols_heckman()
print(comparison['interpretation'])
```

**Returns:** dict with keys:

- `'beta_ols'`: OLS coefficients
- `'beta_heckman'`: Heckman coefficients
- `'difference'`: beta_ols - beta_heckman
- `'pct_difference'`: percentage difference
- `'max_abs_difference'`: maximum absolute difference
- `'interpretation'`: str describing results

**Interpretation:**

- Large differences → Substantial selection bias
- Small differences → Minimal selection bias (ρ ≈ 0)

#### plot_imr(figsize=(12, 5))

Create diagnostic plots for Inverse Mills Ratio.

```python
fig = result.plot_imr()
plt.show()
```

**Creates two plots:**

1. Scatter: IMR vs predicted selection probability
2. Histogram: distribution of IMR for selected sample

**Returns:** matplotlib Figure object

---

## Utility Functions

### compute_imr

```python
from panelbox.models.selection import compute_imr

imr = compute_imr(linear_pred, selected=None, clip_bounds=(1e-10, 1-1e-10))
```

Compute Inverse Mills Ratio from linear predictions.

**Formula:**

For selected observations (d=1):
```
λ(z) = φ(z) / Φ(z)
```

For non-selected (d=0):
```
λ(z) = -φ(z) / [1 - Φ(z)]
```

**Parameters:**

- `linear_pred` : np.ndarray
  - Linear prediction from selection equation (W'γ)

- `selected` : np.ndarray, optional
  - Binary selection indicator
  - If None, computes for selected case only

- `clip_bounds` : tuple
  - Bounds for clipping probabilities to avoid division by zero

**Returns:** np.ndarray of IMR values

### imr_derivative

```python
from panelbox.models.selection import imr_derivative

deriv = imr_derivative(linear_pred)
```

Compute derivative of IMR with respect to z.

**Formula:**
```
dλ/dz = -λ(λ + z)
```

Needed for Murphy-Topel variance correction.

### test_selection_effect

```python
from panelbox.models.selection import test_selection_effect

result = test_selection_effect(imr_coefficient, imr_se, alpha=0.05)
```

Test for selection bias based on IMR coefficient.

**Parameters:**

- `imr_coefficient` : float
  - Coefficient on IMR in outcome equation (θ̂ = ρσ_ε)

- `imr_se` : float
  - Standard error of IMR coefficient

- `alpha` : float
  - Significance level

**Returns:** dict with test results

### imr_diagnostics

```python
from panelbox.models.selection import imr_diagnostics

diag = imr_diagnostics(linear_pred, selected)
```

Compute diagnostic statistics for IMR.

**Parameters:**

- `linear_pred` : np.ndarray
  - Linear prediction from selection equation

- `selected` : np.ndarray
  - Binary selection indicator

**Returns:** dict with diagnostic information

---

## Example Usage

### Basic Heckman Two-Step

```python
import numpy as np
from panelbox.models.selection import PanelHeckman

# Prepare data
y = data['wage'].values  # Outcome (with NaN for non-selected)
X = data[['experience', 'education']].values  # Outcome regressors
selection = data['employed'].values  # Selection indicator (0/1)
Z = data[['age', 'education', 'kids']].values  # Selection regressors

# Estimate model
model = PanelHeckman(
    endog=y,
    exog=X,
    selection=selection,
    exog_selection=Z,
    method='two_step'
)

result = model.fit()
print(result.summary())

# Test for selection bias
test = result.selection_effect()
print(test['interpretation'])

# Compare OLS vs Heckman
comparison = result.compare_ols_heckman()
print(comparison['interpretation'])

# Diagnostic plots
fig = result.plot_imr()
```

### MLE Estimation

```python
# Full information maximum likelihood
model_mle = PanelHeckman(
    endog=y,
    exog=X,
    selection=selection,
    exog_selection=Z,
    method='mle'
)

result_mle = model_mle.fit()
print(f"Log-likelihood: {result_mle.llf:.2f}")
print(f"ρ (MLE): {result_mle.rho:.3f}")
```

### Predictions

```python
# Unconditional prediction (latent outcome)
E_y_star = result.predict(type='unconditional')

# Conditional prediction (accounting for selection)
E_y_given_selected = result.predict(type='conditional')

# Out-of-sample prediction
X_new = np.array([[10, 12], [15, 16]])  # New data
Z_new = np.array([[30, 12, 2], [35, 16, 0]])
pred = result.predict(exog=X_new, exog_selection=Z_new, type='conditional')
```

---

## References

1. **Heckman, J.J. (1979).** "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153-161.

2. **Wooldridge, J.M. (1995).** "Selection Corrections for Panel Data Models Under Conditional Mean Independence Assumptions." *Journal of Econometrics*, 68(1), 115-132.

3. **Murphy, K.M., & Topel, R.H. (1985).** "Estimation and Inference in Two-Step Econometric Models." *Journal of Business & Economic Statistics*, 3(4), 370-379.

---

## See Also

- [Selection Models Theory Guide](../theory/selection_models.md)
- [Panel Heckman Tutorial](../../examples/selection/panel_heckman_tutorial.py)
- [Discrete Choice Models](discrete_models.md)
