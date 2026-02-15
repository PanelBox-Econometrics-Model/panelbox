# Panel Selection Models Examples

This directory contains examples and tutorials for panel data selection models.

## Models Available

### Panel Heckman Selection Model

Corrects for sample selection bias using Heckman's (1979) two-step procedure or full maximum likelihood estimation, extended to panel data following Wooldridge (1995).

**When to use:**
- Outcome variable is only observed for a selected subsample
- Selection is potentially endogenous (correlated with outcome errors)
- Example: Wages observed only for labor force participants

**Methods:**
- `two_step`: Heckman two-step estimator (faster, asymptotically less efficient)
- `mle`: Full information maximum likelihood (slower, asymptotically efficient)

## Files

### `panel_heckman_tutorial.py`

Complete tutorial demonstrating:

1. **Data Generation**
   - Realistic DGP: wage determination with labor force participation
   - Known parameters for validation
   - Selection bias (ρ = 0.4)

2. **Estimation**
   - Two-step Heckman
   - Maximum likelihood estimation
   - Comparison of methods

3. **Diagnostics**
   - Selection effect test (H0: ρ = 0)
   - Inverse Mills Ratio diagnostics
   - OLS vs Heckman comparison
   - Parameter recovery validation

4. **Visualization**
   - IMR scatter plot (IMR vs selection probability)
   - IMR histogram
   - Distribution of selection effects

**Run the tutorial:**
```bash
cd /home/guhaase/projetos/panelbox
python examples/selection/panel_heckman_tutorial.py
```

**Output:**
- Estimation results for two-step and MLE
- Diagnostic tests and comparisons
- Plot saved to `heckman_imr_diagnostics.png`

### Output Files

- `heckman_imr_diagnostics.png` - Diagnostic plots for Inverse Mills Ratio

## Quick Start Example

```python
import numpy as np
import pandas as pd
from panelbox.models.selection import PanelHeckman

# Prepare your data
# y: outcome variable (with NaN for non-selected)
# X: outcome equation regressors
# selection: binary (1=observed, 0=censored)
# Z: selection equation regressors (should include exclusion restriction)

# Two-Step Estimation
model = PanelHeckman(
    endog=y,
    exog=X,
    selection=selection,
    exog_selection=Z,
    entity=entity_id,
    time=time_id,
    method="two_step"
)

result = model.fit()
print(result.summary())

# Diagnostics
test = result.selection_effect()
print(test['interpretation'])

comparison = result.compare_ols_heckman()
print(f"Max OLS bias: {comparison['max_abs_difference']:.3f}")

# Visualization
fig = result.plot_imr()
fig.savefig('my_imr_plot.png')
```

## Key Diagnostic Methods

### `selection_effect()`

Tests for presence of selection bias (H0: ρ = 0).

```python
test = result.selection_effect()
print(test['interpretation'])
# Output: "Selection bias detected (ρ ≠ 0, p=0.0000).
#          OLS would be biased. Heckman correction is necessary."
```

### `imr_diagnostics()`

Provides statistics about the Inverse Mills Ratio:

```python
diag = result.imr_diagnostics()
print(f"Mean IMR: {diag['imr_mean']:.3f}")
print(f"High selection observations: {diag['high_imr_count']}")
```

### `compare_ols_heckman()`

Compares OLS (biased) vs Heckman (corrected) estimates:

```python
comp = result.compare_ols_heckman()
print("OLS coefficients:", comp['beta_ols'])
print("Heckman coefficients:", comp['beta_heckman'])
print("Difference:", comp['difference'])
print(comp['interpretation'])
```

### `plot_imr()`

Visualizes the Inverse Mills Ratio:

```python
fig = result.plot_imr()
# Creates 2 plots:
# 1. Scatter: IMR vs predicted selection probability
# 2. Histogram: distribution of IMR
```

## Theoretical Background

### The Selection Problem

When outcome $y_{it}$ is only observed if selection $d_{it} = 1$, and selection is endogenous:

$$
\\begin{align}
d_{it}^* &= W_{it}' \\gamma + \\eta_i + v_{it} \\\\
d_{it} &= 1[d_{it}^* > 0] \\\\
y_{it} &= X_{it}' \\beta + \\alpha_i + \\epsilon_{it} \\quad \\text{if } d_{it} = 1
\\end{align}
$$

If $\\text{Corr}(v_{it}, \\epsilon_{it}) = \\rho \\neq 0$, then OLS on the selected sample is biased.

### The Solution: Heckman Correction

**Two-Step Procedure:**

1. Estimate selection equation (Probit): $\\hat{\\gamma}$
2. Compute Inverse Mills Ratio: $\\lambda_{it} = \\phi(W_{it}'\\hat{\\gamma}) / \\Phi(W_{it}'\\hat{\\gamma})$
3. Add $\\lambda_{it}$ to outcome equation: $y_{it} = X_{it}'\\beta + \\theta \\lambda_{it} + \\epsilon_{it}^*$

where $\\theta = \\rho \\sigma_\\epsilon$ captures the selection effect.

**Maximum Likelihood:**

Jointly estimates all parameters by maximizing:

$$
\\mathcal{L} = \\sum_i \\log \\int \\left[ \\prod_t L_{it}(d_{it}, y_{it} | X_{it}, W_{it}, \\alpha_i, \\eta_i) \\right] \\phi(\\alpha_i, \\eta_i) d\\alpha_i d\\eta_i
$$

## Identification

**Exclusion Restriction:** At least one variable in $W$ (selection) must not appear in $X$ (outcome).

Example:
- Selection: `participate ~ age + education + kids + other_income`
- Outcome: `log_wage ~ experience + education + tenure`
- Exclusion: `kids`, `other_income` affect participation but not wage

**Why needed:** Without exclusion restriction, identification relies solely on functional form (non-linearity of IMR), which is weak.

## References

1. Heckman, J.J. (1979). "Sample Selection Bias as a Specification Error." *Econometrica*, 47(1), 153-161.

2. Wooldridge, J.M. (1995). "Selection Corrections for Panel Data Models Under Conditional Mean Independence Assumptions." *Journal of Econometrics*, 68(1), 115-132.

3. Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 19.

4. Cameron, A.C., & Trivedi, P.K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 16.

## Common Applications

1. **Labor Economics**
   - Wage determination (observed only for workers)
   - Training program effects (participation is selective)

2. **Health Economics**
   - Medical expenditures (observed only for hospital users)
   - Drug effectiveness (conditional on adherence)

3. **Development Economics**
   - Firm productivity (conditional on survival)
   - Microfinance impact (conditional on participation)

4. **Education**
   - College wage premium (conditional on attendance)
   - Teacher effects (observed only for tested students)

## Troubleshooting

### "No convergence" warning (MLE)

- Try different starting values
- Use two-step estimates as starting values (default)
- Increase maxiter
- Check for multicollinearity in regressors

### ρ > 1 or ρ < -1

- This shouldn't happen if properly implemented
- May indicate numerical issues or misspecification
- Check for perfect prediction in selection equation

### Large IMR values (> 5)

- Indicates very strong selection
- May cause numerical instability
- Consider trimming extreme observations

### Murphy-Topel SEs not available

- Currently using simplified SEs
- Full Murphy-Topel correction coming in future update
- Use bootstrap for research-grade SEs

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-repo/panelbox/issues
- Documentation: https://panelbox.readthedocs.io/
- Examples: `examples/selection/`

---

**Last Updated:** February 15, 2026
