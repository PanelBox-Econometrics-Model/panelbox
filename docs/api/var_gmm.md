# Panel VAR GMM API

API documentation for Panel Vector Autoregression (VAR) GMM estimation.

---

## Overview

The Panel VAR GMM module implements Generalized Method of Moments estimation for Panel Vector Autoregression models, following the methodology of Holtz-Eakin, Newey & Rosen (1988) and the implementation of Abrigo & Love (2016).

**Key Features:**

- Forward Orthogonal Deviations (FOD) transformation (Arellano-Bover 1995)
- First-Differences (FD) transformation
- One-step and Two-step GMM estimation
- Windmeijer (2005) finite-sample correction for standard errors
- Comprehensive instrument diagnostics
- Hansen J test for overidentification
- AR(1)/AR(2) serial correlation tests
- Instrument proliferation detection

---

## Main Functions

### estimate_panel_var_gmm

::: panelbox.var.gmm.estimate_panel_var_gmm
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Main function for estimating Panel VAR using GMM.

**Parameters:**

- `data` (pd.DataFrame): Panel data with entity, time, and variable columns
- `var_lags` (int): Number of VAR lags (p)
- `value_cols` (List[str]): List of variable column names
- `entity_col` (str): Entity identifier column (default: 'entity')
- `time_col` (str): Time identifier column (default: 'time')
- `transform` (str): Transformation type - 'fod' (default) or 'fd'
- `gmm_step` (str): GMM step - 'one-step' or 'two-step' (default)
- `instrument_type` (str): Instrument type - 'all' (default) or 'collapsed'
- `max_instruments` (Optional[int]): Maximum number of instruments per equation
- `windmeijer_correction` (bool): Apply Windmeijer correction (default: True for two-step)

**Returns:**

- `PanelVARGMMResult`: Result object with coefficients, diagnostics, and methods

**Example:**

```python
import pandas as pd
from panelbox.var.gmm import estimate_panel_var_gmm

# Load your panel data
data = pd.DataFrame({
    'entity': [0, 0, 0, 1, 1, 1, ...],
    'time': [0, 1, 2, 0, 1, 2, ...],
    'y1': [...],
    'y2': [...]
})

# Estimate VAR(1) with GMM
result = estimate_panel_var_gmm(
    data=data,
    var_lags=1,
    value_cols=['y1', 'y2'],
    transform='fod',
    gmm_step='two-step',
    instrument_type='collapsed',
    max_instruments=10
)

# View results
print(result.summary())
```

---

## Transformations

### forward_orthogonal_deviation

::: panelbox.var.transforms.forward_orthogonal_deviation
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Applies Forward Orthogonal Deviations (FOD) transformation to remove fixed effects.

**Theory:**

FOD is preferred over first-differences because:
- Preserves orthogonality of transformed errors
- Loses fewer observations in unbalanced panels
- Allows using all available lags as instruments

**Formula:**

```
y*ᵢₜ = cₜ (yᵢₜ - (1/(Tᵢ-t)) Σₛ₌ₜ₊₁ᵀⁱ yᵢₛ)

where: cₜ = √((Tᵢ - t)/(Tᵢ - t + 1))
```

### first_difference

::: panelbox.var.transforms.first_difference
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Applies First-Differences (FD) transformation to remove fixed effects.

**Formula:**

```
Δyᵢₜ = yᵢₜ - yᵢ,ₜ₋₁
```

---

## Instruments

### build_gmm_instruments

::: panelbox.var.instruments.build_gmm_instruments
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Constructs the instrument matrix for GMM estimation.

**Instrument Types:**

- `'all'`: Uses all available lags as instruments (may lead to proliferation)
- `'collapsed'`: Uses collapsed instruments (Roodman 2009) to reduce dimensionality

**Rule of Thumb (Roodman 2009):**

- Number of instruments ≤ number of entities (N)
- Ratio instruments/parameters ≤ 2-3

---

## Diagnostics

### GMMDiagnostics

::: panelbox.var.diagnostics.GMMDiagnostics
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Comprehensive diagnostics for GMM estimation.

**Available Tests:**

1. **Hansen J Test** - Tests overidentifying restrictions
2. **Sargan Test** - Non-robust alternative to Hansen J
3. **AR(1)/AR(2) Tests** - Serial correlation in transformed residuals
4. **Instrument Diagnostics** - Proliferation detection
5. **Sensitivity Analysis** - Coefficient stability across instrument counts

**Example:**

```python
# Get diagnostics from result
diagnostics = result.diagnostics

# Hansen J test
j_stat, j_pval = diagnostics.hansen_j_test()
print(f"Hansen J: {j_stat:.4f} (p-value: {j_pval:.4f})")

# AR tests
ar1_stat, ar1_pval = diagnostics.ar_test(order=1)
ar2_stat, ar2_pval = diagnostics.ar_test(order=2)

# Instrument diagnostics report
print(diagnostics.instrument_diagnostics_report())

# Sensitivity analysis
sensitivity = diagnostics.instrument_sensitivity_analysis(
    max_instrument_counts=[6, 12, 18, 24]
)
```

---

## Tests

### hansen_j_test

::: panelbox.var.diagnostics.hansen_j_test
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Hansen J test for overidentifying restrictions.

**Interpretation:**

- **H₀**: All instruments are valid (moment conditions satisfied)
- **p < 0.05**: Reject H₀ - instruments may be invalid or model misspecified
- **p > 0.99**: Warning - possible weak instruments (many instruments problem)
- **0.1 < p < 0.9**: Ideal range

**Formula:**

```
J = N · ê'Z (Z'ΩZ)⁻¹ Z'ê ~ χ²(#instruments - #parameters)

where Ω = covariance matrix of moment conditions
```

### ar_test

::: panelbox.var.diagnostics.ar_test
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Tests for serial correlation in transformed residuals.

**Expected Results:**

- **AR(1)**: Should reject (correlation by construction of transformation)
- **AR(2)**: Should NOT reject (if model is well-specified)
- **AR(2) rejects**: Indicates under-specification (too few lags) or endogeneity

**Formula:**

```
z = (Σ ê*ₜ₋ₘ ê*ₜ) / √Var(Σ ê*ₜ₋ₘ ê*ₜ) ~ N(0,1)
```

---

## Comparison Functions

### compare_transforms

::: panelbox.var.diagnostics.compare_transforms
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Compares FOD vs FD transformations on the same data.

**Use Cases:**

- Check robustness of results to transformation choice
- Evaluate observation loss in unbalanced panels
- Compare coefficient stability

**Returns:**

Dictionary with:
- `fod_result`: FOD GMM result
- `fd_result`: FD GMM result
- `n_obs_fod`, `n_obs_fd`: Observation counts
- `n_instruments_fod`, `n_instruments_fd`: Instrument counts
- `coef_diff_max`, `coef_diff_mean`: Coefficient differences
- `interpretation`: Automatic interpretation
- `recommendation`: Action recommendation
- `summary`: Formatted summary string

**Example:**

```python
from panelbox.var.diagnostics import compare_transforms

comparison = compare_transforms(
    data=df,
    var_lags=1,
    value_cols=['y1', 'y2'],
    gmm_step='two-step',
    instrument_type='collapsed',
    max_instruments=10
)

print(comparison['summary'])
```

---

## Result Classes

### PanelVARGMMResult

::: panelbox.var.result.PanelVARGMMResult
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Result object for Panel VAR GMM estimation.

**Key Attributes:**

- `params`: Estimated coefficients (pd.DataFrame)
- `std_errors`: Standard errors (pd.DataFrame)
- `t_stats`: t-statistics (pd.DataFrame)
- `p_values`: p-values (pd.DataFrame)
- `n_obs`: Number of observations
- `n_entities`: Number of entities
- `n_instruments`: Number of instruments (total and per equation)
- `gmm_step`: GMM step used ('one-step' or 'two-step')
- `transform`: Transformation used ('fod' or 'fd')
- `instrument_type`: Instrument type ('all' or 'collapsed')
- `diagnostics`: GMMDiagnostics object

**Key Methods:**

- `summary()`: Comprehensive summary table
- `instrument_diagnostics()`: Instrument proliferation diagnostics
- `compare_one_step_two_step()`: Compare one-step vs two-step GMM
- `irf()`: Impulse response functions (inherited from PanelVARResult)
- `forecast_error_variance_decomposition()`: FEVD (inherited)

**Example:**

```python
# View summary
print(result.summary())

# Check instrument diagnostics
print(result.instrument_diagnostics())

# Compare GMM steps
comparison = result.compare_one_step_two_step()
print(comparison['summary'])

# Impulse response functions
irf_result = result.irf(periods=10, shock_var='y1', response_var='y2')
```

---

## Visualization

### plot_instrument_sensitivity

::: panelbox.visualization.var_plots.plot_instrument_sensitivity
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

Plots coefficient stability across different instrument counts.

**Use Case:**

Detect instrument proliferation by checking if coefficients change significantly with instrument count.

**Expected Behavior:**

- **Valid instruments**: Coefficients stable across instrument counts
- **Invalid/weak instruments**: Coefficients vary significantly

**Example:**

```python
from panelbox.visualization.var_plots import plot_instrument_sensitivity

# Run sensitivity analysis
sensitivity = result.diagnostics.instrument_sensitivity_analysis(
    max_instrument_counts=[6, 10, 15, 20, 25]
)

# Plot results
fig = plot_instrument_sensitivity(
    sensitivity_results=sensitivity,
    title="Instrument Sensitivity Analysis - VAR(1)"
)
fig.show()
```

---

## References

### Foundational Papers

**GMM Theory for Panel VAR:**
- Holtz-Eakin, Newey & Rosen (1988) - "Estimating Vector Autoregressions with Panel Data"
- Arellano & Bover (1995) - "Another Look at the Instrumental Variable Estimation of Error-Components Models"
- Love & Zicchino (2006) - "Financial Development and Dynamic Investment Behavior"

**Implementation and Diagnostics:**
- **Abrigo & Love (2016)** - "Estimation of Panel Vector Autoregression in Stata" (PRIMARY REFERENCE)
- Roodman (2009) - "How to do xtabond2: An Introduction to Difference and System GMM in Stata"
- Windmeijer (2005) - "A Finite Sample Correction for the Variance of Linear Efficient Two-step GMM Estimators"

**Tests:**
- Hansen (1982) - "Large Sample Properties of Generalized Method of Moments Estimators"
- Sargan (1958) - "The Estimation of Economic Relationships using Instrumental Variables"
- Arellano & Bond (1991) - "Some Tests of Specification for Panel Data" (AR tests)

---

## Best Practices

### Choosing Transformation

**Use FOD when:**
- Panel is unbalanced (FOD preserves more observations)
- You want to maximize efficiency in small samples
- Following Abrigo & Love (2016) implementation

**Use FD when:**
- Replicating older papers that use FD
- Panel is balanced (FOD and FD should be similar)
- Comparing with Arellano-Bond estimators

### Choosing GMM Step

**One-step GMM:**
- Faster computation
- More robust to misspecification
- Standard errors may be less efficient

**Two-step GMM:**
- More efficient asymptotically
- Requires Windmeijer correction (automatically applied)
- Preferred in most applications

### Managing Instruments

**General Guidelines:**
1. Start with `instrument_type='collapsed'` and `max_instruments=2*K*p`
2. Check Hansen J p-value (should be 0.1-0.9)
3. Run instrument sensitivity analysis
4. If proliferation detected, reduce `max_instruments`

**Warning Signs:**
- Number of instruments > N (number of entities)
- Hansen J p-value > 0.99
- AR(2) test rejects (consider adding lags)
- Coefficients unstable in sensitivity analysis

---

## Common Workflows

### Basic Estimation

```python
from panelbox.var.gmm import estimate_panel_var_gmm

# Estimate VAR(1) with default settings
result = estimate_panel_var_gmm(
    data=df,
    var_lags=1,
    value_cols=['gdp', 'investment', 'consumption']
)

print(result.summary())
```

### With Instrument Diagnostics

```python
# Estimate with conservative instrument settings
result = estimate_panel_var_gmm(
    data=df,
    var_lags=2,
    value_cols=['y1', 'y2', 'y3'],
    instrument_type='collapsed',
    max_instruments=15
)

# Check diagnostics
print(result.instrument_diagnostics())

# Run sensitivity analysis
sensitivity = result.diagnostics.instrument_sensitivity_analysis()
print(sensitivity['summary'])
```

### Comparing Transformations

```python
from panelbox.var.diagnostics import compare_transforms

# Compare FOD vs FD
comparison = compare_transforms(
    data=df,
    var_lags=1,
    value_cols=['y1', 'y2'],
    gmm_step='two-step',
    instrument_type='collapsed'
)

print(comparison['summary'])

# If difference is small, proceed with FOD (more efficient)
if comparison['coef_diff_pct_max'] < 10:
    print("Transformations agree - using FOD")
    final_result = comparison['fod_result']
```

### Full Diagnostic Workflow

```python
# 1. Estimate model
result = estimate_panel_var_gmm(
    data=df,
    var_lags=1,
    value_cols=['y1', 'y2'],
    instrument_type='collapsed',
    max_instruments=10
)

# 2. Check Hansen J
j_stat, j_pval = result.diagnostics.hansen_j_test()
print(f"Hansen J p-value: {j_pval:.4f}")

# 3. Check AR tests
ar1_stat, ar1_pval = result.diagnostics.ar_test(order=1)
ar2_stat, ar2_pval = result.diagnostics.ar_test(order=2)
print(f"AR(1) p-value: {ar1_pval:.4f} (expect < 0.05)")
print(f"AR(2) p-value: {ar2_pval:.4f} (expect > 0.05)")

# 4. Check instrument proliferation
print(result.instrument_diagnostics())

# 5. Sensitivity analysis
sensitivity = result.diagnostics.instrument_sensitivity_analysis()
if sensitivity['is_stable']:
    print("Coefficients stable - instruments valid")
else:
    print("WARNING: Coefficient instability detected")
```

---

## Troubleshooting

### Hansen J Test Rejects (p < 0.05)

**Possible causes:**
1. Invalid instruments (endogeneity not addressed)
2. Model misspecification (wrong lag order)
3. Omitted variables

**Solutions:**
- Check lag order with information criteria
- Try different instrument sets
- Add additional lags
- Check for structural breaks

### Hansen J p-value Very High (p > 0.99)

**Cause:** Likely instrument proliferation (too many weak instruments)

**Solutions:**
- Use `instrument_type='collapsed'`
- Set `max_instruments` to smaller value
- Run instrument sensitivity analysis

### AR(2) Test Rejects

**Cause:** Model under-specified or remaining endogeneity

**Solutions:**
- Increase `var_lags` (try p+1)
- Check for omitted variables
- Verify transformation is appropriate

### Coefficients Differ Between FOD and FD

**Cause:** Unbalanced panel or small sample

**Expected:** Some difference is normal, especially in unbalanced panels

**Action if large difference:**
- Check panel balance
- Increase sample size if possible
- Report both and discuss sensitivity

---

## See Also

- [Examples: GMM Estimation](../../examples/var/gmm_estimation.py)
- [Examples: Instrument Diagnostics](../../examples/var/instrument_diagnostics.py)
- [Tutorial: Panel VAR GMM](../../tutorials/panel_var_gmm.md)
- [API: Standard GMM](gmm.md)
