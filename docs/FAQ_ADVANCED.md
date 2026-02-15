# FAQ - Advanced Methods
## Frequently Asked Questions for PanelBox Advanced Features

**Last Updated:** 2026-02-15
**Covers:** GMM, Selection Models, Cointegration, Unit Root, Specification Tests, Specialized Models

---

## Table of Contents

1. [GMM Advanced](#1-gmm-advanced)
2. [Panel Heckman Selection Models](#2-panel-heckman-selection-models)
3. [Cointegration Tests](#3-cointegration-tests)
4. [Unit Root Tests](#4-unit-root-tests)
5. [Specification Tests](#5-specification-tests)
6. [PPML (Gravity Models)](#6-ppml-gravity-models)
7. [Multinomial Logit](#7-multinomial-logit)
8. [Common Errors](#8-common-errors)
9. [Performance Tips](#9-performance-tips)

---

## 1. GMM Advanced

### Q1.1: When should I use CUE-GMM instead of two-step GMM?

**A:** Use Continuous Updated Estimator (CUE-GMM) when:

- **You need maximum efficiency** - CUE is asymptotically more efficient than two-step
- **Sample size is moderate to large** (N > 200) - CUE requires more observations
- **Weak instruments** are a concern - CUE is more robust to weak instruments
- **You can afford longer computation time** - CUE is slower (iterative optimization)

**Use two-step GMM when:**
- Quick results needed
- Small sample (N < 200)
- Instruments are strong
- Initial exploration

**Example:**
```python
import panelbox as pb

# Two-step GMM (fast)
gmm2 = pb.DifferenceGMM(data, dep_var='y', lags=[1], exog_vars=['x'])
result2 = gmm2.fit(steps=2)

# CUE-GMM (more efficient but slower)
cue = pb.ContinuousUpdatedGMM(data, dep_var='y', lags=[1], exog_vars=['x'])
result_cue = cue.fit()

# Compare
print(f"Two-step SE: {result2.bse}")
print(f"CUE SE: {result_cue.bse}")  # Usually smaller (more efficient)
```

**References:**
- Hansen, Heaton & Yaron (1996) - CUE efficiency
- Windmeijer (2005) - Finite sample properties

---

### Q1.2: What is Bias-Corrected GMM and when do I need it?

**A:** Bias-Corrected GMM addresses the **O(1/N) bias** in standard GMM estimators when:

- **Dynamic panels with small N** (N < 100 especially)
- **Persistent dependent variable** (autocorrelation > 0.7)
- **Short time periods** (T < 10)

**The correction:**
```python
import panelbox as pb

# Standard GMM (biased in small samples)
gmm = pb.DifferenceGMM(data, dep_var='y', lags=[1])
result_gmm = gmm.fit()

# Bias-Corrected GMM
bc_gmm = pb.BiasCorrectedGMM(data, dep_var='y', lags=[1])
result_bc = bc_gmm.fit(order=2)  # Second-order correction

print(f"Bias magnitude: {result_bc.bias_magnitude()}")
# If > 10% of coefficient, correction is important
```

**Minimum sample sizes:**
- First-order correction: N ≥ 50
- Second-order correction: N ≥ 100

**References:**
- Hahn & Kuersteiner (2002)
- Bun & Windmeijer (2010)

---

### Q1.3: CUE-GMM is not converging. What should I do?

**A:** Troubleshooting steps:

**1. Use two-step starting values:**
```python
cue = pb.ContinuousUpdatedGMM(data, ...)
result = cue.fit(
    starting_values='two_step',  # Better initialization
    maxiter=500,                  # More iterations
    tol=1e-6                      # Loosen tolerance if needed
)
```

**2. Check moment conditions:**
```python
# Are moments valid?
diag = pb.GMMDiagnostics(result)
print(diag.diagnostic_tests())

# Too many moments → overidentification issues
# Rule of thumb: # moments ≤ 2 * # parameters
```

**3. Regularize weighting matrix:**
```python
result = cue.fit(
    weighting_matrix='hac',
    bandwidth='auto',
    regularization=1e-4  # Add small ridge
)
```

**4. Try different optimization algorithm:**
```python
result = cue.fit(
    method='bfgs',      # Instead of default 'newton'
    use_gradient=True   # Analytical gradient
)
```

**If still not converging:**
- Use two-step GMM instead
- Check for multicollinearity
- Verify data quality (no NaN, outliers)
- Consider fewer instruments

---

### Q1.4: How do I choose bandwidth for HAC standard errors?

**A:** Bandwidth choice affects robustness to autocorrelation:

**Automatic selection (recommended):**
```python
result = cue.fit(
    weighting_matrix='hac',
    bandwidth='auto',           # Newey-West automatic
    kernel='bartlett'           # Or 'parzen', 'quadratic-spectral'
)
```

**Manual selection:**
```python
# Rule of thumb: bandwidth = floor(T^(1/4))
import numpy as np
T = len(data['time'].unique())
bw = int(np.floor(T**(1/4)))

result = cue.fit(
    weighting_matrix='hac',
    bandwidth=bw
)
```

**Bandwidth too small:**
- Underestimates SEs
- Tests too liberal

**Bandwidth too large:**
- Overestimates SEs
- Tests too conservative

**Diagnostics:**
```python
# Check if bandwidth matters
result1 = cue.fit(bandwidth=3)
result2 = cue.fit(bandwidth=5)

print(f"SE with bw=3: {result1.bse}")
print(f"SE with bw=5: {result2.bse}")
# If very different → use 'auto'
```

---

## 2. Panel Heckman Selection Models

### Q2.1: When should I use Heckman selection model?

**A:** Use Panel Heckman when:

**1. Sample selection is non-random**
- Wages observed only for workers (not unemployed)
- Firm performance observed only for surviving firms
- Exports observed only for exporting firms

**2. Selection equation is known**
- You can model the selection decision
- Exclusion restriction is available

**3. Selection affects outcome**
```python
# Test for selection bias
import panelbox as pb

heckman = pb.PanelHeckman(
    data=data,
    outcome_formula='wage ~ educ + exper',
    selection_formula='work ~ age + kids + married',
    entity_id='person_id',
    time_id='year'
)

result = heckman.fit(method='two_step')

# Check if selection matters
print(result.selection_effect())
# If p-value < 0.05 → selection bias exists
```

**When NOT to use Heckman:**
- Selection is random (use OLS)
- No exclusion restriction available
- Binary outcome (use probit/logit)

---

### Q2.2: Two-step vs MLE - which should I choose?

**A:** **Comparison:**

| Feature | Two-Step | MLE |
|---------|----------|-----|
| **Speed** | ✅ Fast | ⚠️ Slow |
| **Efficiency** | Good | ✅ Best (asymptotically) |
| **Robustness** | ✅ More robust | Sensitive to misspecification |
| **Convergence** | ✅ Always works | May fail |
| **Sample size** | Works with N < 100 | ✅ Better with N > 200 |
| **Standard errors** | Murphy-Topel correction | ✅ Direct from Hessian |

**Recommendation:**
```python
# Start with two-step
result_2step = heckman.fit(method='two_step')

# If sample is large (N > 200) and two-step shows selection
if result_2step.rho < 1.0:  # Valid correlation
    # Try MLE for efficiency
    result_mle = heckman.fit(
        method='mle',
        starting_values=result_2step.params  # Use two-step as start
    )
```

**If MLE fails to converge:**
- Use two-step results
- Check data quality
- Reduce quadrature points: `quadrature_points=10`

---

### Q2.3: What does ρ (rho) mean and what if ρ > 1?

**A:** **ρ = correlation between selection and outcome errors**

**Interpretation:**
- **ρ = 0:** No selection bias → OLS is fine
- **ρ > 0:** Positive selection (e.g., high-ability workers select into labor force)
- **ρ < 0:** Negative selection
- **|ρ| > 0.5:** Strong selection bias

**Example:**
```python
result = heckman.fit()
print(f"ρ = {result.rho:.3f}")
print(f"SE(ρ) = {result.rho_se:.3f}")

# Test H0: ρ = 0
t_stat = result.rho / result.rho_se
if abs(t_stat) > 1.96:
    print("Significant selection bias")
```

**If ρ > 1 or ρ < -1:**

This indicates **model misspecification**:

**Possible causes:**
1. **Wrong exclusion restriction**
2. **Functional form misspecification**
3. **Omitted variables in selection equation**
4. **Non-linearity not captured**

**Solutions:**
```python
# 1. Check exclusion restriction
#    Variables in selection but not outcome should affect selection
#    Example: 'kids' affects labor force participation but not wage

# 2. Add non-linear terms
heckman = pb.PanelHeckman(
    data=data,
    outcome_formula='wage ~ educ + exper + I(exper**2)',
    selection_formula='work ~ age + I(age**2) + kids',
    ...
)

# 3. Use two-step instead of MLE
result = heckman.fit(method='two_step')  # More robust

# 4. Check for outliers
data_clean = data[(data['wage'] > 0) & (data['wage'] < 1000)]
```

---

### Q2.4: How do I interpret Inverse Mills Ratio (IMR)?

**A:** **IMR measures selection bias correction:**

**Coefficient on IMR (λ):**
```python
result = heckman.fit(method='two_step')
print(result.summary())

# Look for 'lambda' or 'IMR' in output
# λ = coefficient on IMR
```

**Interpretation:**
- **λ = 0:** No selection bias after controlling for observables
- **λ > 0:** Selected sample has higher outcomes than random sample
- **λ < 0:** Selected sample has lower outcomes

**Effect size:**
```python
# Average selection bias
avg_imr = result.imr.mean()
lambda_coef = result.params['lambda']

avg_bias = lambda_coef * avg_imr
print(f"Average selection bias: {avg_bias:.2f}")
# Units are same as dependent variable
```

**Diagnostics:**
```python
# Check if IMR is well-behaved
from panelbox.models.selection import imr_diagnostics

diag = imr_diagnostics(result.selection_probs)
print(diag)

# Warning signs:
# - IMR very large (> 10) → poor exclusion restriction
# - IMR almost constant → weak selection
```

---

### Q2.5: Heckman MLE is very slow. Are there alternatives?

**A:** **Speed optimization:**

**1. Reduce quadrature points:**
```python
# Default: 15 points (slow but accurate)
result = heckman.fit(
    method='mle',
    quadrature_points=10  # Faster, still good
)
# 5 points → very fast but less accurate
# 20 points → very slow but best for irregular distributions
```

**2. Use two-step as starting values:**
```python
# Two-step is fast
result_2step = heckman.fit(method='two_step')

# Use as initialization for MLE
result_mle = heckman.fit(
    method='mle',
    starting_values=result_2step.params,
    quadrature_points=10
)
```

**3. Just use two-step:**
```python
# Two-step with Murphy-Topel SEs is very reliable
result = heckman.fit(method='two_step')
# Efficiency loss is small in most applications
```

**Speed comparison (N=1000, T=5):**
- Two-step: ~5 seconds
- MLE (10 quadrature): ~30 seconds
- MLE (15 quadrature): ~60 seconds

**Recommendation:**
- Exploratory analysis: two-step
- Final results with N > 500: MLE
- Final results with N < 500: two-step is fine

---

## 3. Cointegration Tests

### Q3.1: Which cointegration test should I use: Westerlund, Pedroni, or Kao?

**A:** **Decision tree:**

```
START
│
├─ Do you suspect heterogeneous cointegration?
│  ├─ YES → Pedroni or Westerlund
│  │  ├─ Need error correction interpretation? → Westerlund
│  │  └─ Prefer residual-based tests? → Pedroni
│  │
│  └─ NO (homogeneous) → Kao
│
├─ Do you have cross-sectional dependence?
│  ├─ YES → Westerlund (more robust)
│  └─ NO → Any test works
│
└─ Is T small (< 20)?
   ├─ YES → Westerlund (better power)
   └─ NO → Pedroni or Kao
```

**Detailed comparison:**

| Feature | Westerlund | Pedroni | Kao |
|---------|-----------|---------|-----|
| **Heterogeneity** | ✅ Allows | ✅ Allows | ❌ Assumes homogeneous |
| **Power (small T)** | ✅ High | Medium | Medium |
| **Interpretation** | ✅ ECM-based | Residual-based | Residual-based |
| **Bootstrap** | ✅ Built-in | Manual | Manual |
| **Cross-section dependence** | ✅ Robust | ⚠️ Sensitive | ⚠️ Sensitive |
| **Computation** | ⚠️ Slow | ✅ Fast | ✅ Fast |

**Example:**
```python
import panelbox as pb

# Westerlund (recommended for most cases)
west = pb.westerlund_test(
    data,
    dependent='y',
    covariates=['x1', 'x2'],
    bootstrap=True,
    n_bootstrap=1000
)
print(west.summary())

# Pedroni (if Westerlund too slow)
ped = pb.pedroni_test(
    data,
    dependent='y',
    covariates=['x1', 'x2']
)
print(ped.summary())

# Kao (if you believe in homogeneity)
kao = pb.kao_test(
    data,
    dependent='y',
    covariates=['x1', 'x2']
)
print(kao.summary())
```

---

### Q3.2: Should I use bootstrap or asymptotic critical values?

**A:** **Use bootstrap when:**

- ✅ **Small sample** (N < 50 or T < 30)
- ✅ **Unbalanced panel**
- ✅ **Cross-sectional dependence** suspected
- ✅ **You have computational time** (bootstrap is slow)

**Use asymptotic (tabulated) when:**

- ✅ **Large sample** (N > 100, T > 50)
- ✅ **Balanced panel**
- ✅ **Quick results** needed
- ✅ **First-pass analysis**

**Example:**
```python
# Quick check with asymptotic
result_asymp = pb.westerlund_test(
    data, ...,
    bootstrap=False  # Fast
)

# If borderline significant, confirm with bootstrap
if 0.05 < result_asymp.p_value_Gt < 0.15:
    result_boot = pb.westerlund_test(
        data, ...,
        bootstrap=True,
        n_bootstrap=1999  # More accurate
    )
    print(f"Asymptotic p-value: {result_asymp.p_value_Gt:.3f}")
    print(f"Bootstrap p-value: {result_boot.p_value_Gt:.3f}")
```

**Bootstrap parameters:**
```python
result = pb.westerlund_test(
    data, ...,
    bootstrap=True,
    n_bootstrap=1999,  # Should be odd number for percentile CIs
    seed=42            # For reproducibility
)
```

**Computational time (N=50, T=30):**
- Asymptotic: < 1 second
- Bootstrap (B=1000): ~5-10 minutes
- Bootstrap (B=2000): ~10-20 minutes

---

### Q3.3: All tests reject cointegration. What now?

**A:** If **all tests reject** H0 of cointegration:

**1. First, check if you're testing the right hypothesis:**

Westerlund:
- H0: **No cointegration**
- Reject → **Cointegration EXISTS** ✅

Pedroni/Kao:
- Some tests: H0 = No cointegration
- Check which tests you're using!

**2. If truly no cointegration:**

**Options:**

**A. Difference the data:**
```python
# If no cointegration, use first differences
data['dy'] = data.groupby('entity_id')['y'].diff()
data['dx'] = data.groupby('entity_id')['x'].diff()

# Estimate in differences
from panelbox import FixedEffects
fe = FixedEffects(data, dep_var='dy', exog_vars=['dx'])
result = fe.fit()
```

**B. Use VAR instead:**
```python
# Panel VAR allows for no cointegration
from panelbox.var import PanelVAR

pvar = PanelVAR(data, variables=['y', 'x'], lags=2)
result = pvar.fit()
```

**C. Check for structural breaks:**
```python
# Maybe cointegration exists in subperiods
data_early = data[data['year'] < 2010]
data_late = data[data['year'] >= 2010]

result_early = pb.westerlund_test(data_early, ...)
result_late = pb.westerlund_test(data_late, ...)
```

**D. Reconsider variable selection:**
```python
# Maybe wrong set of variables
# Try different combinations
# Use economic theory to guide
```

---

### Q3.4: How do I choose number of lags for cointegration tests?

**A:** **Lag selection affects test power and size:**

**Automatic selection (recommended):**
```python
result = pb.westerlund_test(
    data, ...,
    lags='auto',      # Automatic AIC/BIC selection
    max_lags=4        # Upper bound
)

print(f"Selected lags: {result.lags_used}")
```

**Manual selection:**
```python
# Rule of thumb: lags = floor(T^(1/3))
import numpy as np
T = len(data['time'].unique())
lags = int(np.floor(T**(1/3)))

result = pb.westerlund_test(data, ..., lags=lags)
```

**Diagnostic check:**
```python
# Try different lag lengths
results = {}
for lag in range(1, 5):
    results[lag] = pb.westerlund_test(data, ..., lags=lag)
    print(f"Lags={lag}: p-value={results[lag].p_value_Gt:.3f}")

# If results very sensitive to lags → data issues
```

**Too few lags:**
- Residual autocorrelation
- Test size distortion

**Too many lags:**
- Loss of power
- Degrees of freedom issues

**Recommendation:**
- Use 'auto' for final analysis
- Try 1-4 lags manually for robustness check

---

## 4. Unit Root Tests

### Q4.1: Hadri vs IPS - which test should I use?

**A:** **Key difference: the null hypothesis!**

| Test | H0 | H1 | When to use |
|------|----|----|-------------|
| **Hadri** | **Stationarity** | Unit root | Believe series is stationary, want to test |
| **IPS** | **Unit root** | Stationarity | Believe series has unit root, want to test |
| **Breitung** | Unit root | Stationarity | Robust to heterogeneous trends |

**Interpretation:**

**Hadri Test:**
```python
hadri = pb.hadri_test(data, variable='y')
if hadri.p_value < 0.05:
    print("Reject stationarity → Unit root present")
else:
    print("Cannot reject stationarity → Series is stationary")
```

**IPS Test:**
```python
ips = pb.IPSTest(data, variable='y')
result = ips.run()
if result.p_value < 0.05:
    print("Reject unit root → Series is stationary")
else:
    print("Cannot reject unit root → Unit root present")
```

**Recommendation:**
```python
# Run BOTH tests for robustness
result = pb.panel_unit_root_test(
    data,
    variable='y',
    tests=['hadri', 'ips', 'breitung']
)
print(result.summary())

# Consensus:
# - Both reject their nulls → Ambiguous
# - Both fail to reject → Ambiguous
# - Hadri fails to reject, IPS rejects → Stationary ✅
# - Hadri rejects, IPS fails to reject → Unit root ✅
```

---

### Q4.2: Hadri and IPS contradict each other. How do I interpret this?

**A:** **Contradiction scenarios:**

**Scenario 1: Hadri rejects, IPS fails to reject**
- **Conclusion:** Unit root present
- **Why:** Hadri rejects stationarity (evidence of unit root)
- **Action:** Difference the data or use cointegration

**Scenario 2: Hadri fails to reject, IPS rejects**
- **Conclusion:** Stationary
- **Why:** IPS rejects unit root (evidence of stationarity)
- **Action:** Use level data

**Scenario 3: Both reject their nulls**
- **Interpretation:** Data is "borderline" - near unit root but not exactly
- **Possible causes:**
  - Strong AR(1) but < 1 (near-unit root)
  - Structural breaks
  - Cross-sectional dependence
  - Heterogeneous dynamics

**Action for Scenario 3:**
```python
# 1. Check Breitung (more robust)
breitung = pb.breitung_test(data, variable='y')
print(f"Breitung p-value: {breitung.p_value}")

# 2. Test with trend
hadri_trend = pb.hadri_test(data, variable='y', trend=True)
ips_trend = pb.IPSTest(data, variable='y', trend=True).run()

# 3. Check for structural breaks
# (Plot the data)
import matplotlib.pyplot as plt
data.groupby('time')['y'].mean().plot()
plt.title('Check for breaks')
plt.show()

# 4. If still ambiguous, estimate in both levels and differences
#    Compare results for robustness
```

**Scenario 4: Both fail to reject their nulls**
- **Interpretation:** Insufficient power or complex dynamics
- **Possible causes:**
  - Small sample size
  - Very heterogeneous dynamics
  - Data quality issues

**Action:**
```python
# Check sample size
N = data['entity_id'].nunique()
T = data.groupby('entity_id').size().mean()
print(f"N={N}, T={T}")

# If N < 30 or T < 20 → may have low power
# Consider using level data and checking results carefully
```

---

### Q4.3: All tests reject unit root, but the series looks non-stationary in plots. Why?

**A:** **Possible explanations:**

**1. Trend-stationary (not difference-stationary):**
```python
# Series has deterministic trend but no unit root
# Solution: Include trend in model

from panelbox import FixedEffects
fe = FixedEffects(
    data,
    dep_var='y',
    exog_vars=['x', 'time']  # Add time trend
)
result = fe.fit()
```

**2. Structural break:**
```python
# Unit root tests may reject if there's a break
# Check for breaks:

import matplotlib.pyplot as plt
data.groupby('time')['y'].mean().plot()
plt.axvline(x=2008, color='r', linestyle='--', label='Financial crisis')
plt.show()

# If break present, test in subsamples
pre_break = data[data['time'] < 2008]
post_break = data[data['time'] >= 2008]

result_pre = pb.panel_unit_root_test(pre_break, variable='y')
result_post = pb.panel_unit_root_test(post_break, variable='y')
```

**3. Cross-sectional dependence:**
```python
# Common shocks can make stationary series look non-stationary
# Test for CD:

from panelbox import PesaranCDTest
cd = PesaranCDTest(data, variable='y')
cd_result = cd.run()

if cd_result.p_value < 0.05:
    print("Cross-sectional dependence detected")
    # Use cross-sectionally demeaned data
    data['y_demeaned'] = data.groupby('time')['y'].transform(
        lambda x: x - x.mean()
    )
    # Re-run unit root tests on y_demeaned
```

**4. Near unit root:**
```python
# AR(1) coefficient very close to 1 (e.g., 0.95)
# Technically stationary but looks like unit root

# Check AR(1) coefficient
from statsmodels.tsa.stattools import acf
for entity in data['entity_id'].unique()[:5]:
    entity_data = data[data['entity_id'] == entity]['y']
    rho = acf(entity_data, nlags=1)[1]
    print(f"Entity {entity}: AR(1) = {rho:.3f}")

# If most ρ > 0.9, consider treating as unit root for practical purposes
```

**5. Data quality issues:**
```python
# Check for:
# - Missing values
# - Outliers
# - Coding errors

print(data['y'].describe())
print(f"Missing: {data['y'].isna().sum()}")

# Plot distributions
data.boxplot(column='y', by='entity_id')
plt.show()
```

---

## 5. Specification Tests

### Q5.1: What is the J-test and when should I use it?

**A:** **Davidson-MacKinnon J-test** tests between **non-nested models**.

**Use when:**
- Two competing theories/specifications
- Models have different functional forms
- Want to test which fits better

**Example:**
```python
import panelbox as pb

# Model 1: Production function (Cobb-Douglas)
from panelbox import FixedEffects
model1 = FixedEffects(data, dep_var='log_output', exog_vars=['log_capital', 'log_labor'])
result1 = model1.fit()

# Model 2: Production function (Translog)
model2 = FixedEffects(
    data,
    dep_var='log_output',
    exog_vars=['log_capital', 'log_labor', 'log_capital_sq', 'log_labor_sq', 'log_capital_labor']
)
result2 = model2.fit()

# J-test
jtest = pb.j_test(result1, result2, cov_type='cluster', cluster='entity_id')
print(jtest.summary())
```

**Interpretation:**
```python
# jtest.forward_test: H0 = Model 1 is true
# jtest.reverse_test: H0 = Model 2 is true

if jtest.forward_test['p_value'] < 0.05:
    print("Reject Model 1 → Model 2 is better")
elif jtest.reverse_test['p_value'] < 0.05:
    print("Reject Model 2 → Model 1 is better")
elif (jtest.forward_test['p_value'] >= 0.05 and
      jtest.reverse_test['p_value'] >= 0.05):
    print("Both models adequate → Use economic theory or other criteria")
else:  # Both rejected
    print("Neither model adequate → Consider alternative specification")
```

**NOT for:**
- Nested models (use Wald test or LR test)
- Different dependent variables
- Different samples

---

## 6. PPML (Gravity Models)

### Q6.1: When should I use PPML instead of OLS on log-transformed data?

**A:** **Use PPML when:**

**1. Zeros in dependent variable**
```python
# Gravity equation with zero trade flows
# OLS(log) loses zero observations
import pandas as pd
print(f"Zeros: {(data['trade'] == 0).sum()}")
print(f"% Zeros: {100 * (data['trade'] == 0).mean():.1f}%")

# If > 5% zeros → PPML is better
```

**2. Heteroskedasticity**
```python
# PPML is consistent under heteroskedasticity
# OLS(log) is biased under heteroskedasticity

# Test for heteroskedasticity
from panelbox import BreuschPaganTest
bp = BreuschPaganTest(ols_result)
bp_result = bp.run()

if bp_result.p_value < 0.05:
    print("Heteroskedasticity detected → Use PPML")
```

**3. Multiplicative functional form is appropriate**
```python
# Gravity equation:
# Trade_ij = exp(β0 + β1*log_distance + β2*log_GDP_i + β3*log_GDP_j)

# This is natural multiplicative form → PPML
```

**Use OLS(log) when:**
- No zeros in dependent variable
- Homoskedasticity assumption reasonable
- Simplicity preferred (OLS is faster)

**Comparison:**
```python
import panelbox as pb
import numpy as np

# OLS on log
data_pos = data[data['trade'] > 0].copy()
data_pos['log_trade'] = np.log(data_pos['trade'])

ols = pb.FixedEffects(
    data_pos,
    dep_var='log_trade',
    exog_vars=['log_distance', 'log_gdp_i', 'log_gdp_j']
)
result_ols = ols.fit()

# PPML
ppml = pb.PPML(
    data,  # Can include zeros!
    dep_var='trade',
    exog_vars=['log_distance', 'log_gdp_i', 'log_gdp_j']
)
result_ppml = ppml.fit()

# Compare
print(f"OLS N = {len(data_pos)}")
print(f"PPML N = {len(data)}")
print(f"\\nOLS distance coef: {result_ols.params['log_distance']:.3f}")
print(f"PPML distance coef: {result_ppml.params['log_distance']:.3f}")
```

**References:**
- Santos Silva & Tenreyro (2006) - PPML for gravity
- Head & Mayer (2014) - Gravity equations

---

### Q6.2: PPML is not converging. What should I do?

**A:** **Troubleshooting:**

**1. Better starting values:**
```python
# Use Poisson FE as starting values
from panelbox import PoissonFixedEffects

poisson = PoissonFixedEffects(data, dep_var='trade', exog_vars=[...])
result_pois = poisson.fit()

# Use as initialization
ppml = pb.PPML(data, ...)
result_ppml = ppml.fit(
    starting_values=result_pois.params,
    maxiter=500
)
```

**2. Scale variables:**
```python
# Large values can cause numerical issues
data['trade_scaled'] = data['trade'] / 1e6  # Millions
data['gdp_scaled'] = data['gdp'] / 1e9      # Billions

ppml = pb.PPML(
    data,
    dep_var='trade_scaled',
    exog_vars=['log_distance', 'gdp_scaled']
)
```

**3. Remove problematic observations:**
```python
# Check for:
# - Extreme outliers in dependent variable
# - Perfect collinearity

# Outliers
data = data[data['trade'] < data['trade'].quantile(0.99)]

# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
# If VIF > 10 → remove collinear variables
```

**4. Use robust algorithm:**
```python
result = ppml.fit(
    method='bfgs',        # Instead of 'newton'
    tol=1e-5,             # Loosen tolerance
    maxiter=1000,
    step_size='auto'      # Adaptive step size
)
```

**5. If still failing:**
```python
# Use Poisson QML instead
from panelbox import PoissonQML

qml = PoissonQML(data, dep_var='trade', exog_vars=[...])
result_qml = qml.fit()
# QML is more robust but slightly less efficient
```

---

### Q6.3: How do I interpret elasticities in PPML?

**A:** **PPML estimates multiplicative effects:**

**For log-transformed variables:**
```python
# If regressor is log(X), coefficient is elasticity directly
result = ppml.fit()

# Example: log_distance coefficient = -0.8
# Interpretation: 1% increase in distance → 0.8% decrease in trade
print(f"Distance elasticity: {result.params['log_distance']:.3f}")
```

**For level variables (dummies):**
```python
# If regressor is dummy (0/1), exponentiate coefficient
# Example: border coefficient = 0.5
import numpy as np
border_effect = np.exp(result.params['border']) - 1
print(f"Border effect: {100 * border_effect:.1f}% increase")
# Sharing a border increases trade by 50%
```

**Computing elasticities from level variables:**
```python
# For level variable X, elasticity at mean:
result_ppml = ppml.fit()

elasticity = result_ppml.elasticity(
    variable='gdp',
    at_means=True
)
print(f"GDP elasticity at mean: {elasticity:.3f}")

# Elasticity at all observations
all_elasticities = result_ppml.elasticities()
print(all_elasticities.describe())
```

**Example interpretation:**
```python
# Gravity equation:
# log(Trade_ij) = β0 + β1*log_distance + β2*log_GDP_i + β3*log_GDP_j + β4*border

result = ppml.fit()

# Elasticity interpretations:
print("=== Elasticities ===")
print(f"Distance: {result.params['log_distance']:.2f}")
# "1% farther → {-β1}% less trade"

print(f"Exporter GDP: {result.params['log_gdp_i']:.2f}")
# "1% higher exporter GDP → {β2}% more trade"

print(f"Importer GDP: {result.params['log_gdp_j']:.2f}")
# "1% higher importer GDP → {β3}% more trade"

border_effect = 100 * (np.exp(result.params['border']) - 1)
print(f"Border effect: {border_effect:.1f}%")
# "Common border → {border_effect}% more trade"
```

---

### Q6.4: PPML vs Poisson Fixed Effects - what's the difference?

**A:** **Key differences:**

| Feature | PPML | Poisson FE |
|---------|------|------------|
| **Distributional assumption** | ❌ None (quasi-ML) | ✅ Assumes Poisson distribution |
| **Standard errors** | ✅ Robust to misspecification | ⚠️ Assumes Poisson variance |
| **Efficiency** | Lower if Poisson true | ✅ Higher if Poisson true |
| **Robustness** | ✅ More robust | Less robust |
| **Use case** | General gravity/count | True count data |

**When to use PPML:**
```python
# Use PPML when:
# - Data is NOT truly Poisson (e.g., trade flows)
# - Overdispersion suspected
# - Want robust inference

ppml = pb.PPML(data, ...)
result = ppml.fit(cov_type='robust')  # Robust SEs
```

**When to use Poisson FE:**
```python
# Use Poisson FE when:
# - Data is truly count (e.g., number of patents)
# - Poisson assumption reasonable
# - Want maximum efficiency

from panelbox import PoissonFixedEffects
pois_fe = PoissonFixedEffects(data, ...)
result = pois_fe.fit()
```

**Test for Poisson:**
```python
# Test for overdispersion
poisson = PoissonFixedEffects(data, ...)
result_pois = poisson.fit()

# Overdispersion test
# If variance > mean → overdispersed → use PPML or Negative Binomial
mean_y = data['y'].mean()
var_y = data['y'].var()
print(f"Mean: {mean_y:.2f}")
print(f"Variance: {var_y:.2f}")
print(f"Variance/Mean: {var_y/mean_y:.2f}")

if var_y/mean_y > 1.5:
    print("Overdispersed → Use PPML or Negative Binomial")
```

**In practice:**
- **Gravity models:** Always use PPML
- **Patent counts:** Start with Poisson FE, test for overdispersion
- **When unsure:** Use PPML (more robust)

---

## 7. Multinomial Logit

### Q7.1: Fixed Effects vs Random Effects for multinomial logit?

**A:** **FE Multinomial Logit:**

**Advantages:**
- ✅ No distributional assumptions on heterogeneity
- ✅ Consistent even if unobserved heterogeneity correlated with covariates
- ✅ Conditional MLE is consistent

**Disadvantages:**
- ⚠️ Computationally intensive (especially for J > 3, T > 10)
- ⚠️ Time-invariant variables drop out
- ⚠️ Incidental parameters problem for small T

**RE Multinomial Logit:**

**Advantages:**
- ✅ Fast computation
- ✅ Can include time-invariant variables
- ✅ More efficient if RE assumption holds

**Disadvantages:**
- ❌ Assumes unobserved heterogeneity uncorrelated with regressors
- ❌ Inconsistent if assumption violated

**Decision rule:**
```python
import panelbox as pb

# If you have time-invariant variables of interest → RE
# Example: gender, race (don't vary over time)

# If worried about endogeneity → FE
# Example: income affects choice, choice affects income

# Practical rule:
J = data['choice'].nunique()
T = data.groupby('entity_id').size().median()

if J > 4 or T > 10:
    print("Use RE (FE too slow)")
    model = pb.MultinomialLogit(data, method='random_effects')
elif 'time_invariant_var_of_interest' in data.columns:
    print("Use RE (to include time-invariant)")
    model = pb.MultinomialLogit(data, method='random_effects')
else:
    print("Use FE (safest)")
    model = pb.MultinomialLogit(data, method='fixed_effects')

result = model.fit()
```

**Hausman test for FE vs RE:**
```python
# Estimate both
re_result = pb.MultinomialLogit(data, method='random_effects').fit()
fe_result = pb.MultinomialLogit(data, method='fixed_effects').fit()

# Hausman test
from panelbox import HausmanTest
hausman = HausmanTest(fe_result, re_result)
h_result = hausman.run()

if h_result.p_value < 0.05:
    print("Reject RE → Use FE")
else:
    print("RE is consistent → Use RE")
```

---

### Q7.2: FE Multinomial is very slow. Are there alternatives?

**A:** **Speed improvements:**

**1. Reduce sample:**
```python
# FE Multinomial has complexity O(J^T * N)
# Limit J (choices) and T (time periods)

# If too slow:
# - Use first T=5 periods only
# - Aggregate similar choices (reduce J)
```

**2. Use simpler model for exploration:**
```python
# Use RE for initial exploration
re = pb.MultinomialLogit(data, method='random_effects')
result_re = re.fit()  # Fast

# Then use FE for final results on subset of specifications
fe = pb.MultinomialLogit(data_subset, method='fixed_effects')
result_fe = fe.fit()  # Slower but robust
```

**3. Parallelization:**
```python
# Enable multiprocessing (if implemented)
fe = pb.MultinomialLogit(data, method='fixed_effects')
result = fe.fit(n_jobs=-1)  # Use all cores
```

**4. Just use RE:**
```python
# If FE prohibitively slow, RE is usually good approximation
# Especially if Hausman test is not significant

re = pb.MultinomialLogit(data, method='random_effects')
result = re.fit(cov_type='cluster', cluster='entity_id')
# Cluster SEs help robustness
```

**Computational time estimates (N=1000):**

| J | T | FE time | RE time |
|---|---|---------|---------|
| 3 | 5 | 30 sec | 5 sec |
| 3 | 10 | 5 min | 10 sec |
| 4 | 5 | 5 min | 8 sec |
| 4 | 10 | 2 hours | 15 sec |
| 5 | 5 | 30 min | 10 sec |

**Recommendation:**
- J ≤ 3 and T ≤ 10 → FE is feasible
- J = 4 or T > 10 → Use RE
- J ≥ 5 → Always use RE

---

### Q7.3: How do I interpret marginal effects in multinomial logit?

**A:** **Marginal effects are choice-specific:**

```python
import panelbox as pb

model = pb.MultinomialLogit(
    data,
    choice_var='mode',  # e.g., car, bus, train
    covariates=['cost', 'time', 'distance']
)
result = model.fit()

# Marginal effects for all choices
me = result.marginal_effects(at_means=True)
print(me)
```

**Output interpretation:**
```
         cost_car  cost_bus  cost_train  time_car  time_bus  time_train
car      -0.012     0.005      0.007     -0.010     0.003      0.007
bus       0.005    -0.015      0.010      0.003    -0.013      0.010
train     0.007     0.010     -0.017      0.007     0.010     -0.017
```

**Reading the table:**
- **Row:** Choice that's being analyzed
- **Column:** Variable being changed
- **Value:** Change in probability

**Example interpretations:**
```python
# cost_car coefficient for car row: -0.012
# "Increasing car cost by $1 reduces Pr(car) by 1.2 percentage points"

# cost_car coefficient for bus row: 0.005
# "Increasing car cost by $1 increases Pr(bus) by 0.5 percentage points"
# (Substitution effect: car becomes less attractive, bus more attractive)
```

**Individual-level marginal effects:**
```python
# Compute for each observation
me_individual = result.marginal_effects(at_means=False)

# Average marginal effects
ame = me_individual.mean(axis=0)
print("Average Marginal Effects:")
print(ame)
```

**Choice-specific marginal effects:**
```python
# Marginal effect of cost on Pr(car)
me_car = result.marginal_effect(
    choice='car',
    variable='cost',
    at_means=True
)
print(f"ME of cost on Pr(car): {me_car:.4f}")
```

**Cross-marginal effects:**
```python
# How does car cost affect probability of choosing train?
cross_me = result.marginal_effect(
    choice='train',    # Outcome
    variable='cost_car',  # Variable (from car alternative)
    at_means=True
)
# Positive → car and train are substitutes
# Negative → car and train are complements (rare)
```

---

## 8. Common Errors

### Error 1: "Weighting matrix is singular"

**Cause:** Too many instruments or perfect collinearity in moment conditions.

**Solution:**
```python
# 1. Reduce number of instruments
gmm = pb.DifferenceGMM(
    data,
    max_lags_instruments=2  # Instead of default (all available)
)

# 2. Add regularization
result = cue.fit(
    regularization=1e-4  # Ridge regularization
)

# 3. Check for collinearity
import numpy as np
moments = gmm._compute_moments(...)
rank = np.linalg.matrix_rank(moments)
print(f"Rank: {rank}, Columns: {moments.shape[1]}")
# If rank < columns → collinearity
```

---

### Error 2: "Heckman MLE did not converge"

**Cause:** MLE is difficult, poor starting values, or misspecification.

**Solution:**
```python
# 1. Use two-step as starting values
heckman = pb.PanelHeckman(...)
result_2step = heckman.fit(method='two_step')

result_mle = heckman.fit(
    method='mle',
    starting_values=result_2step.params  # Better initialization
)

# 2. Reduce quadrature points
result_mle = heckman.fit(
    method='mle',
    quadrature_points=10  # Instead of 15
)

# 3. Just use two-step
result = heckman.fit(method='two_step')  # Usually reliable
```

---

### Error 3: "Bootstrap taking too long"

**Cause:** Too many bootstrap replications or large dataset.

**Solution:**
```python
# 1. Reduce bootstrap replications
result = pb.westerlund_test(
    data,
    bootstrap=True,
    n_bootstrap=499  # Instead of 1999
)
# 499 is usually sufficient

# 2. Use asymptotic critical values instead
result = pb.westerlund_test(
    data,
    bootstrap=False  # Much faster
)

# 3. Parallel bootstrap (if implemented)
result = pb.westerlund_test(
    data,
    bootstrap=True,
    n_bootstrap=999,
    n_jobs=-1  # Use all cores
)
```

---

### Warning 1: "ρ > 1 in Heckman model"

**Cause:** Model misspecification.

**Solution:**
```python
# 1. Check exclusion restriction
# Variable in selection but not outcome must truly affect selection

# 2. Add non-linear terms
heckman = pb.PanelHeckman(
    data,
    outcome_formula='wage ~ educ + exper + I(exper**2)',
    selection_formula='work ~ age + I(age**2) + kids + married',
    ...
)

# 3. Use two-step instead of MLE
result = heckman.fit(method='two_step')  # More robust

# 4. Check data for outliers
```

---

### Warning 2: "Weak instruments detected"

**Cause:** F-statistic < 10 in first-stage regression.

**Solution:**
```python
# Check instrument strength
diag = pb.GMMDiagnostics(result)
print(diag.weak_instruments_test())

# If F < 10:
# 1. Use stronger instruments
# 2. Reduce number of instruments
# 3. Use CUE-GMM (more robust)
cue = pb.ContinuousUpdatedGMM(...)
result = cue.fit()

# 4. Report both two-step and CUE results
# If similar → instruments OK
# If very different → weak instruments issue
```

---

## 9. Performance Tips

### Tip 1: Start with Fast Methods for Exploration

```python
# Exploration phase: Use fast methods
# Two-step GMM
result_2step = pb.DifferenceGMM(...).fit(steps=2)

# RE Multinomial
result_re = pb.MultinomialLogit(..., method='random_effects').fit()

# Asymptotic critical values
result_asymp = pb.westerlund_test(..., bootstrap=False)

# Final analysis: Use robust methods
# CUE-GMM
result_cue = pb.ContinuousUpdatedGMM(...).fit()

# FE Multinomial
result_fe = pb.MultinomialLogit(..., method='fixed_effects').fit()

# Bootstrap critical values
result_boot = pb.westerlund_test(..., bootstrap=True, n_bootstrap=1999)
```

---

### Tip 2: Limit Multinomial FE to J ≤ 4, T ≤ 10

```python
# Check feasibility before running
J = data['choice'].nunique()
T = data.groupby('entity_id').size().median()

print(f"J={J}, T={T}")

if J > 4 or T > 10:
    print("FE Multinomial will be slow. Consider:")
    print("1. Use RE instead")
    print("2. Aggregate choices (reduce J)")
    print("3. Use subsample of time periods")

    # Use RE
    model = pb.MultinomialLogit(data, method='random_effects')
else:
    print("FE Multinomial is feasible")
    model = pb.MultinomialLogit(data, method='fixed_effects')

result = model.fit()
```

---

### Tip 3: Use Tabulated Critical Values if Bootstrap Too Slow

```python
import time

# Try asymptotic first
start = time.time()
result_asymp = pb.westerlund_test(data, ..., bootstrap=False)
asymp_time = time.time() - start

print(f"Asymptotic time: {asymp_time:.2f}s")
print(f"Asymptotic p-value: {result_asymp.p_value_Gt:.3f}")

# Only bootstrap if borderline or small sample
if 0.05 < result_asymp.p_value_Gt < 0.15 or N < 50:
    print("Running bootstrap (may take a while)...")
    start = time.time()
    result_boot = pb.westerlund_test(data, ..., bootstrap=True, n_bootstrap=999)
    boot_time = time.time() - start
    print(f"Bootstrap time: {boot_time:.2f}s")
    print(f"Bootstrap p-value: {result_boot.p_value_Gt:.3f}")
else:
    print("Asymptotic result is clear, skip bootstrap")
```

---

### Tip 4: Heckman MLE — Start with `quadrature_points=10`, Increase if Needed

```python
# Start conservative
result_10 = heckman.fit(method='mle', quadrature_points=10)

# Check convergence
if result_10.converged:
    print("Converged with 10 points")

    # If critical application, verify with more points
    result_15 = heckman.fit(method='mle', quadrature_points=15)

    # Compare
    diff = abs(result_10.params - result_15.params).max()
    if diff < 0.01:
        print("Results stable, use 10 points")
    else:
        print("Results differ, use 15 points")
else:
    # Try two-step instead
    result = heckman.fit(method='two_step')
```

---

### Tip 5: PPML — Good Starting Values from Poisson

```python
from panelbox import PoissonFixedEffects, PPML

# Use Poisson FE for initialization
pois = PoissonFixedEffects(data, dep_var='y', exog_vars=['x1', 'x2'])
result_pois = pois.fit()

# PPML with good starting values
ppml = PPML(data, dep_var='y', exog_vars=['x1', 'x2'])
result_ppml = ppml.fit(
    starting_values=result_pois.params,
    maxiter=500
)

# Converges much faster!
```

---

### Tip 6: Multinomial Marginal Effects Can Be Computed for Subset of Observations

```python
# Don't compute for all observations if not needed
result = multinomial.fit()

# Just at means
me_means = result.marginal_effects(at_means=True)  # Fast

# For subset
representative_sample = data.sample(n=1000, random_state=42)
me_sample = result.marginal_effects(
    data=representative_sample,
    at_means=False
)
ame = me_sample.mean(axis=0)
# Much faster than all observations
```

---

## Additional Resources

### Documentation
- **API Reference:** `/docs/api/`
- **Theory Guides:** `/docs/theory/`
- **Tutorials:** `/docs/tutorials/`
- **Examples:** `/examples/`

### Support
- **GitHub Issues:** https://github.com/panelbox/panelbox/issues
- **Discussions:** https://github.com/panelbox/panelbox/discussions

### References
See individual method documentation for academic references.

---

**Last Updated:** 2026-02-15
**Version:** 1.0.0
**Feedback:** Submit issues or questions on GitHub
