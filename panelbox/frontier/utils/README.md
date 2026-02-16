# Frontier Utilities Module

Advanced utilities for stochastic frontier analysis (SFA) and productivity measurement.

## Overview

This module provides essential tools for empirical productivity analysis:

1. **TFP Decomposition** - Decompose productivity growth into components
2. **Marginal Effects** - Analyze determinants of inefficiency/efficiency

## Installation

```python
from panelbox.frontier.utils import TFPDecomposition, marginal_effects
```

---

## 1. TFP Decomposition

Decompose Total Factor Productivity (TFP) growth into:
- **Technical Change (ΔTC)**: Frontier shift (innovation)
- **Efficiency Change (ΔTE)**: Catch-up to frontier
- **Scale Efficiency (ΔSE)**: Gains from changing scale

### Quick Start

```python
from panelbox.frontier.utils import TFPDecomposition

# Estimate SFA model (panel data)
result = model.fit()

# Create decomposition object
tfp = TFPDecomposition(result, periods=(2015, 2019))

# Decompose TFP for each firm
decomp = tfp.decompose()
print(decomp[['entity', 'delta_tfp', 'delta_tc', 'delta_te', 'delta_se']])

# Aggregate statistics
agg = tfp.aggregate_decomposition()
print(f"Mean TFP growth: {agg['mean_delta_tfp']:.3f}")
print(f"  From technical change: {agg['pct_from_tc']:.1f}%")
print(f"  From efficiency change: {agg['pct_from_te']:.1f}%")

# Visualize
tfp.plot_decomposition(kind='bar', top_n=20)
tfp.plot_decomposition(kind='scatter')

# Text summary
print(tfp.summary())
```

### Theory

The decomposition follows the Malmquist productivity index approach:

**TFP Growth:**
```
Δ ln(TFP) = Δ ln(Y) - Σ εⱼ · Δ ln(Xⱼ)
```

where εⱼ are output elasticities.

**Components:**

1. **Technical Efficiency Change (ΔTE):**
   ```
   ΔTE = ln(TE_t2) - ln(TE_t1)
   ```
   - Positive: Firm catching up to frontier
   - Negative: Firm falling behind

2. **Technical Change (ΔTC):**
   - Frontier shift over time
   - Reflects industry-wide innovation

3. **Scale Efficiency Change (ΔSE):**
   ```
   ΔSE = (RTS - 1) · Δ(weighted inputs)
   ```
   - Depends on returns to scale (RTS)
   - With DRS (RTS < 1): Expansion decreases SE
   - With IRS (RTS > 1): Expansion increases SE

**Identity:**
```
Δ ln(TFP) = ΔTC + ΔTE + ΔSE
```

### API Reference

#### `TFPDecomposition`

**Constructor:**
```python
TFPDecomposition(result, periods=None)
```

Parameters:
- `result`: SFResult from panel SFA model
- `periods`: Tuple (t1, t2) of periods to compare. If None, uses first and last.

**Methods:**

`decompose()` → DataFrame
- Returns firm-level decomposition with columns:
  - `entity`: Firm identifier
  - `delta_tfp`: Total TFP growth
  - `delta_tc`: Technical change
  - `delta_te`: Efficiency change
  - `delta_se`: Scale effect
  - `verification`: Should be ≈ 0

`aggregate_decomposition()` → dict
- Returns aggregate statistics:
  - `mean_delta_tfp`: Average TFP growth
  - `pct_from_tc`: % from technical change
  - `pct_from_te`: % from efficiency change
  - `pct_from_se`: % from scale effects
  - `std_delta_tfp`: Standard deviation
  - `n_firms`: Number of firms

`plot_decomposition(kind='bar', top_n=20)` → Figure
- Visualizes decomposition
- `kind`: 'bar' (stacked bars) or 'scatter' (TC vs TE)
- `top_n`: Number of firms to show

`summary()` → str
- Formatted text summary

### Interpretation Guide

| Component | Interpretation | Policy Implication |
|-----------|---------------|-------------------|
| ΔTC > 0 | Frontier shifting outward | Support R&D, technology adoption |
| ΔTC < 0 | Frontier contracting | Investigate structural issues |
| ΔTE > 0 | Catch-up to frontier | Management training working |
| ΔTE < 0 | Falling behind | Need best practice diffusion |
| ΔSE > 0 (IRS) | Expansion beneficial | Support firm growth |
| ΔSE < 0 (DRS) | Expansion harmful | Firms may be too large |

### Example: Manufacturing Sector

```python
# Decompose for each year
years = [2015, 2016, 2017, 2018, 2019]
time_series = []

for t1, t2 in zip(years[:-1], years[1:]):
    tfp_t = TFPDecomposition(result, periods=(t1, t2))
    agg_t = tfp_t.aggregate_decomposition()
    time_series.append(agg_t)

# Plot evolution
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(time_series))

# Stacked components
ax.bar(x, [d['mean_delta_tc'] for d in time_series],
       label='Technical Change')
ax.bar(x, [d['mean_delta_te'] for d in time_series],
       bottom=[d['mean_delta_tc'] for d in time_series],
       label='Efficiency Change')

ax.plot(x, [d['mean_delta_tfp'] for d in time_series],
        'ko-', label='Total TFP')

ax.set_xlabel('Period')
ax.set_ylabel('Growth Rate')
ax.legend()
plt.show()
```

---

## 2. Marginal Effects

Analyze how firm characteristics affect inefficiency or efficiency.

Supports:
- **Wang (2002)**: Heteroscedastic inefficiency model
- **Battese & Coelli (1995)**: Inefficiency determinants

### Quick Start

```python
from panelbox.frontier.utils import marginal_effects

# Estimate SFA with inefficiency determinants
model = StochasticFrontier(
    depvar='log_output',
    exog=['log_capital', 'log_labor'],
    inefficiency_vars=['firm_age', 'manager_education'],
    data=df,
)
result = model.fit()

# Compute marginal effects on inefficiency
me = marginal_effects(result, method='mean')
print(me)

#        variable  marginal_effect  std_error  z_stat  p_value
# 0      firm_age           0.0230      0.005    4.60    0.000
# 1  manager_edu          -0.0150      0.007   -2.14    0.032

# Effect on efficiency (instead of inefficiency)
me_eff = marginal_effects(result, method='efficiency')
```

### Theory

**Battese & Coelli (1995) Model:**
```
u_it ~ N⁺(μ_it, σ²_u)
μ_it = z_it'δ
```

Marginal effect on E[u]:
```
∂E[u_it]/∂z_k = δ_k · [1 - mills_it · (mills_it + α_it)]
```

where mills_it is the inverse Mills ratio.

**Wang (2002) Model:**
```
u_i ~ N⁺(μ_i, σ²_u,i)
μ_i = z_i'δ (location)
ln(σ²_u,i) = w_i'γ (scale)
```

Marginal effects on:
- **Mean:** ∂E[u_i]/∂z_k
- **Variance:** ∂Var[u_i]/∂w_k

### API Reference

#### `marginal_effects()`

```python
marginal_effects(result, method='mean', var=None, at_values=None)
```

Parameters:
- `result`: SFResult with inefficiency determinants
- `method`: 'mean', 'efficiency', or 'variance' (Wang only)
- `var`: Specific variable (if None, all variables)
- `at_values`: Values at which to evaluate (default: sample means)

Returns: DataFrame with columns
- `variable`: Name of determinant
- `marginal_effect`: Marginal effect value
- `std_error`: Standard error
- `z_stat`: Z-statistic
- `p_value`: P-value
- `interpretation`: Text interpretation

#### Model-Specific Functions

**BC95:**
```python
marginal_effects_bc95(result, method='mean', var=None, at_values=None)
```

**Wang (2002):**
```python
marginal_effects_wang_2002(result, method='mean', var=None)
```

#### Summary Formatting

```python
from panelbox.frontier.utils import marginal_effects_summary

me = marginal_effects(result)
print(marginal_effects_summary(me))
```

### Interpretation

**Marginal Effect on Inefficiency (u):**
- **Positive:** Variable increases inefficiency (bad)
  - Example: Firm age → more inefficient
- **Negative:** Variable decreases inefficiency (good)
  - Example: Education → less inefficient

**Marginal Effect on Efficiency (TE = exp(-u)):**
- **Positive:** Variable increases efficiency (good)
- **Negative:** Variable decreases efficiency (bad)
- Signs are reversed from inefficiency effects!

### Example: Policy Analysis

```python
# Estimate model with determinants
model = StochasticFrontier(
    depvar='log_output',
    exog=['log_capital', 'log_labor'],
    inefficiency_vars=[
        'firm_age',
        'manager_education',
        'export_share',
        'r_and_d_intensity',
    ],
    data=df,
)
result = model.fit()

# Compute marginal effects
me = marginal_effects(result, method='mean')

# Identify policy levers
significant = me[me['p_value'] < 0.05]
improvements = significant[significant['marginal_effect'] < 0]

print("Variables that significantly reduce inefficiency:")
for _, row in improvements.iterrows():
    print(f"  {row['variable']}: ME = {row['marginal_effect']:.4f}")

# Policy recommendations
if 'manager_education' in improvements['variable'].values:
    print("\n→ Policy: Invest in management training programs")

if 'r_and_d_intensity' in improvements['variable'].values:
    print("→ Policy: R&D subsidies to improve efficiency")
```

---

## References

### TFP Decomposition

1. **Kumbhakar, S. C., & Lovell, C. A. K. (2000).** *Stochastic Frontier Analysis.* Cambridge University Press. Chapter 7: Productivity and its components.

2. **Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994).** "Productivity growth, technical progress, and efficiency change in industrialized countries." *American Economic Review*, 84(1), 66-83.

3. **Orea, L. (2002).** "Parametric decomposition of a generalized Malmquist productivity index." *Journal of Productivity Analysis*, 18(1), 5-22.

4. **Balk, B. M. (2001).** "Scale efficiency and productivity change." *Journal of Productivity Analysis*, 15(3), 159-183.

### Marginal Effects

5. **Wang, H. J., & Schmidt, P. (2002).** "One-step and two-step estimation of the effects of exogenous variables on technical efficiency levels." *Journal of Productivity Analysis*, 18, 129-144.

6. **Battese, G. E., & Coelli, T. J. (1995).** "A model for technical inefficiency effects in a stochastic frontier production function for panel data." *European Journal of Operational Research*, 38, 325-332.

7. **Alvarez, A., Amsler, C., Orea, L., & Schmidt, P. (2006).** "Interpreting and testing the scaling property in models where inefficiency depends on firm characteristics." *Journal of Productivity Analysis*, 25, 201-212.

---

## Testing

Run tests with:
```bash
pytest tests/frontier/utils/
```

Tests cover:
- ✓ Decomposition components sum to total
- ✓ Efficiency change calculation
- ✓ Returns to scale computation
- ✓ Aggregate statistics
- ✓ Visualizations
- ✓ Marginal effects for Wang and BC95
- ✓ Model auto-detection
- ✓ Numerical accuracy

---

## Examples

### Complete Examples

1. **Jupyter Notebook:**
   - `examples/notebooks/tfp_decomposition.ipynb`
   - Full tutorial with synthetic data

2. **Integrated Analysis:**
   - `examples/productivity_analysis.py`
   - Combines TFP + marginal effects
   - Manufacturing sector application

### Running Examples

```bash
# Python script
python examples/productivity_analysis.py

# Jupyter notebook
jupyter notebook examples/notebooks/tfp_decomposition.ipynb
```

---

## Advanced Usage

### Custom Returns to Scale

For translog production functions, RTS varies by output level:

```python
class CustomTFP(TFPDecomposition):
    def _compute_returns_to_scale(self, beta, x_t1, x_t2):
        # For translog: RTS = Σβⱼ + 2·Σβⱼⱼ·xⱼ + Σ_k Σ_j βⱼₖ·xₖ
        # Evaluate at midpoint
        x_mid = (x_t1 + x_t2) / 2

        # Linear terms
        rts = beta[:2].sum()  # β_K + β_L

        # Quadratic terms (if estimated)
        # rts += 2 * beta_KK * x_mid[0] + 2 * beta_LL * x_mid[1]
        # rts += beta_KL * (x_mid[0] + x_mid[1])

        return rts

# Use custom class
tfp = CustomTFP(result, periods=(2015, 2019))
```

### Confidence Intervals for Marginal Effects

```python
from scipy import stats

me = marginal_effects(result)

# 95% confidence intervals
alpha = 0.05
z_crit = stats.norm.ppf(1 - alpha/2)

me['ci_lower'] = me['marginal_effect'] - z_crit * me['std_error']
me['ci_upper'] = me['marginal_effect'] + z_crit * me['std_error']

print(me[['variable', 'marginal_effect', 'ci_lower', 'ci_upper']])
```

### Multi-Period Analysis

```python
# Decompose all consecutive periods
years = range(2010, 2021)
decomp_panel = []

for t1, t2 in zip(years[:-1], years[1:]):
    tfp_t = TFPDecomposition(result, periods=(t1, t2))
    decomp_t = tfp_t.decompose()
    decomp_t['period_start'] = t1
    decomp_t['period_end'] = t2
    decomp_panel.append(decomp_t)

full_panel = pd.concat(decomp_panel, ignore_index=True)

# Analyze trends
import seaborn as sns

sns.lineplot(data=full_panel, x='period_start', y='delta_tfp',
             estimator='mean', ci=95)
plt.title('TFP Growth Over Time')
plt.show()
```

---

## Troubleshooting

### Error: "TFP decomposition requires panel data model"

**Cause:** Model is cross-sectional (no panel structure)

**Solution:** Ensure model has `entity` and `time` identifiers:
```python
model = StochasticFrontier(
    ...,
    entity='firm_id',  # Required!
    time='year',       # Required!
)
```

### Error: "Marginal effects require model with inefficiency determinants"

**Cause:** No inefficiency determinants specified

**Solution:** Add `inefficiency_vars` to model:
```python
model = StochasticFrontier(
    ...,
    inefficiency_vars=['age', 'education'],  # Add this!
)
```

### Warning: Large verification errors in decomposition

**Cause:** Numerical issues or model misspecification

**Solutions:**
1. Check that model converged properly
2. Verify frontier parameters are reasonable
3. Ensure efficiency scores are in (0, 1]
4. Try different optimization algorithm

### Marginal effects seem too large/small

**Cause:** Scale of variables

**Solution:** Standardize determinant variables:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['age', 'education']] = scaler.fit_transform(df[['age', 'education']])
```

Then interpret marginal effects as "effect of 1 std dev increase".

---

## Contributing

Contributions welcome! Areas for extension:
- [ ] Time-varying technical change
- [ ] Confidence intervals for decomposition
- [ ] Bootstrap for marginal effects
- [ ] Distance functions (output vs input)
- [ ] Environmental variables (good/bad outputs)

---

## License

Part of the PanelBox package. See main package license.

---

## Citation

If you use this module in research, please cite:

```bibtex
@software{panelbox,
  title = {PanelBox: Panel Data Analysis Toolkit},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/panelbox}
}
```

And the relevant methodological papers (see References section).
