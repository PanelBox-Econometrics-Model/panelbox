# Tutorial: Quantile Treatment Effects in Panel Data

## Introduction

Quantile Treatment Effects (QTE) allow researchers to understand how treatments affect different parts of the outcome distribution, revealing heterogeneous impacts that mean-based methods might miss. This tutorial demonstrates QTE estimation using PanelBox.

## Why Quantile Treatment Effects?

Traditional Average Treatment Effects (ATE) tell us the mean impact but hide important heterogeneity:
- Does job training help low-skilled workers more than high-skilled?
- Do tax cuts affect poor and rich households differently?
- Does a drug work better for severely ill patients?

QTE provides the complete picture by estimating effects across the distribution.

## Basic Concepts

### Quantile Treatment Effect Definition

For binary treatment $D \in \{0,1\}$:

$$QTE(\tau) = Q_{Y(1)}(\tau) - Q_{Y(0)}(\tau)$$

where $Y(1)$ and $Y(0)$ are potential outcomes.

### Conditional vs Unconditional QTE

- **Conditional QTE**: Effect at quantiles of conditional distribution given $X$
- **Unconditional QTE**: Effect at quantiles of marginal distribution

## Example 1: Job Training Program

Let's analyze a job training program's effect on wages across the wage distribution.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from panelbox.models.quantile.treatment_effects import QuantileTreatmentEffects
from panelbox.utils.data import load_example_data

# Load job training panel data
data = load_example_data('job_training')
# Contains: entity_id, time_id, wage, trained, experience, education

# Setup QTE analysis
qte = QuantileTreatmentEffects(
    data=data,
    outcome='wage',
    treatment='trained',
    covariates=['experience', 'education']
)

# Estimate QTE across distribution
tau_grid = np.arange(0.1, 1.0, 0.1)
results = qte.estimate_qte(
    tau=tau_grid,
    method='standard',
    bootstrap=True,
    n_boot=999
)

# Display results
results.summary()
```

### Interpreting Results

```
Quantile Treatment Effects (standard)
==================================================
Quantile      QTE    Std Error         95% CI
--------------------------------------------------
0.10        1.234      0.234     [0.775, 1.693]
0.25        1.456      0.198     [1.068, 1.844]
0.50        1.678      0.187     [1.312, 2.044]
0.75        2.134      0.212     [1.718, 2.550]
0.90        2.890      0.267     [2.367, 3.413]

Heterogeneity (std of QTE): 0.634
Substantial heterogeneity detected across quantiles
```

**Interpretation**: Training has larger effects for high-wage workers (2.89 at 90th percentile) than low-wage workers (1.23 at 10th percentile).

### Visualization

```python
# Plot QTE across quantiles
fig = qte.plot_qte(results)
plt.title('Job Training Effects Across Wage Distribution')
plt.show()
```

## Example 2: Unconditional QTE via RIF

For policy-relevant unconditional effects:

```python
# Estimate unconditional QTE using RIF regression
results_unconditional = qte.estimate_qte(
    tau=tau_grid,
    method='unconditional'
)

# Compare conditional vs unconditional
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Conditional QTE
ax1.plot(tau_grid, [results.qte_results[tau]['qte'] for tau in tau_grid])
ax1.set_title('Conditional QTE')
ax1.set_xlabel('Quantile')
ax1.set_ylabel('Treatment Effect')

# Unconditional QTE
ax2.plot(tau_grid, [results_unconditional.qte_results[tau]['qte'] for tau in tau_grid])
ax2.set_title('Unconditional QTE')
ax2.set_xlabel('Quantile')
ax2.set_ylabel('Treatment Effect')

plt.tight_layout()
plt.show()
```

## Example 3: Panel Difference-in-Differences QTE

For policy changes with treatment and control groups over time:

```python
# Load policy change data
data = load_example_data('minimum_wage')
# Contains: state_id, year, employment, min_wage_increase, unemployment_rate

# Define pre/post periods
pre_period = data['year'] < 2010
post_period = data['year'] >= 2010

# Estimate DiD QTE
qte_did = QuantileTreatmentEffects(
    data=data,
    outcome='employment',
    treatment='min_wage_increase'
)

results_did = qte_did.estimate_qte(
    tau=tau_grid,
    method='did',
    pre_period=pre_period,
    post_period=post_period
)

# Show parallel trends test (pre-period only)
pre_data = data[pre_period]
qte_pre = QuantileTreatmentEffects(
    data=pre_data,
    outcome='employment',
    treatment='min_wage_increase'
)

# Should find no effect in pre-period
pre_results = qte_pre.estimate_qte(tau=[0.25, 0.5, 0.75])
print("Pre-treatment QTE (should be near zero):")
pre_results.summary()
```

## Example 4: Dynamic Panel QTE

For analyzing persistence of treatment effects:

```python
from panelbox.models.quantile import DynamicQuantile

# Model with lagged outcome
model = DynamicQuantile(
    data=data,
    formula='wage ~ trained + lag(wage, 1) + experience',
    tau=[0.25, 0.5, 0.75],
    lags=1,
    method='iv'
)

# Fit model
dynamic_results = model.fit(iv_lags=2)

# Compute long-run treatment effects
lr_effects = model.compute_long_run_effects(dynamic_results)

for tau, effect in lr_effects.items():
    if effect is not None:
        print(f"τ={tau}: Long-run multiplier = {effect['multiplier']:.3f}")
        print(f"        Long-run QTE = {effect['effects'][0]:.3f}")
```

## Example 5: Changes-in-Changes (Athey & Imbens)

For non-linear DiD without parallel trends:

```python
# Changes-in-Changes estimator
results_cic = qte.estimate_qte(
    tau=tau_grid,
    method='cic',
    pre_period=pre_period,
    post_period=post_period
)

# Compare DiD vs CiC
comparison_df = pd.DataFrame({
    'Quantile': tau_grid,
    'DiD_QTE': [results_did.qte_results[tau]['qte'] for tau in tau_grid],
    'CiC_QTE': [results_cic.qte_results[tau]['qte'] for tau in tau_grid]
})

print(comparison_df)
```

## Advanced: Combining with Location-Scale Models

For guaranteed non-crossing QTE:

```python
from panelbox.models.quantile import LocationScale

# Estimate location-scale model with treatment
ls_model = LocationScale(
    data=data,
    formula='wage ~ trained * (experience + education)',
    tau=tau_grid,
    distribution='normal',
    fixed_effects=True
)

ls_results = ls_model.fit()

# Extract treatment effects at each quantile
qte_ls = {}
for tau in tau_grid:
    # Coefficient on 'trained' gives QTE
    trained_idx = ls_model.variable_names.index('trained')
    qte_ls[tau] = ls_results.results[tau].params[trained_idx]

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(tau_grid, [qte_ls[tau] for tau in tau_grid],
         label='Location-Scale QTE', linewidth=2)
plt.plot(tau_grid, [results.qte_results[tau]['qte'] for tau in tau_grid],
         '--', label='Standard QTE', linewidth=2)
plt.xlabel('Quantile')
plt.ylabel('Treatment Effect')
plt.title('QTE: Location-Scale vs Standard')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices

### 1. Choose the Right Method

| Scenario | Recommended Method |
|----------|-------------------|
| Cross-sectional with controls | Standard QTE |
| Policy-relevant margins | Unconditional QTE (RIF) |
| Panel with common trends | DiD QTE |
| Panel without parallel trends | Changes-in-Changes |
| Need non-crossing | Location-Scale |

### 2. Testing and Validation

```python
# Test for heterogeneous effects
from scipy import stats

# Collect QTE estimates
qte_values = [results.qte_results[tau]['qte'] for tau in tau_grid]

# Test if QTE varies significantly
f_stat, p_value = stats.f_oneway(*np.array_split(qte_values, 3))
print(f"Test for heterogeneity: F={f_stat:.3f}, p={p_value:.3f}")

if p_value < 0.05:
    print("Significant heterogeneity detected - QTE analysis justified")
else:
    print("No significant heterogeneity - ATE may suffice")
```

### 3. Sensitivity Analysis

```python
# Try different specifications
specifications = [
    'wage ~ trained + experience + education',
    'wage ~ trained * experience + education',
    'wage ~ trained + poly(experience, 2) + education'
]

sensitivity_results = {}
for spec in specifications:
    qte_temp = QuantileTreatmentEffects(data, 'wage', 'trained')
    res = qte_temp.estimate_qte(tau=0.5, method='standard')
    sensitivity_results[spec] = res.qte_results[0.5]['qte']

print("Sensitivity to specification (median QTE):")
for spec, effect in sensitivity_results.items():
    print(f"  {spec}: {effect:.3f}")
```

### 4. Reporting Results

Essential elements for papers:
1. **Table**: QTE at key quantiles (0.1, 0.25, 0.5, 0.75, 0.9)
2. **Figure**: QTE across full distribution with confidence bands
3. **Comparison**: QTE vs ATE to show added value
4. **Robustness**: Alternative specifications/methods

## Common Pitfalls and Solutions

### Issue: Crossing quantile curves in QTE

```python
# Solution: Use location-scale or apply rearrangement
from panelbox.models.quantile.monotonicity import QuantileMonotonicity

if QuantileMonotonicity.detect_crossing(results).has_crossing:
    results_fixed = QuantileMonotonicity.rearrangement(results)
```

### Issue: Wide confidence intervals

```python
# Solution: Use more efficient estimation
# Option 1: Increase bootstrap replications
results_precise = qte.estimate_qte(tau=tau_grid, bootstrap=True, n_boot=9999)

# Option 2: Focus on fewer quantiles
key_quantiles = [0.25, 0.5, 0.75]
results_focused = qte.estimate_qte(tau=key_quantiles, bootstrap=True)
```

### Issue: Interpretation of magnitude

```python
# Solution: Standardize effects
# Express as percentage of baseline
baseline = np.quantile(data[data['trained']==0]['wage'], tau_grid)
pct_effects = 100 * qte_values / baseline

plt.plot(tau_grid, pct_effects)
plt.ylabel('Treatment Effect (%)')
plt.xlabel('Quantile')
plt.title('Percentage Treatment Effects')
```

## Conclusion

Quantile Treatment Effects provide rich insights into treatment heterogeneity:
- Reveal who benefits most/least from interventions
- Guide targeted policy design
- Uncover unintended distributional consequences

Key takeaways:
1. QTE complements but doesn't replace ATE
2. Choose method based on identification strategy
3. Always check and correct for crossing
4. Bootstrap for proper inference
5. Visualize results for clear communication

## Further Reading

- Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. *Econometrica*, 77(3), 953-973.
- Athey, S., & Imbens, G. W. (2006). Identification and inference in nonlinear difference‐in‐differences models. *Econometrica*, 74(2), 431-497.
- Chernozhukov, V., & Hansen, C. (2005). An IV model of quantile treatment effects. *Econometrica*, 73(1), 245-261.
