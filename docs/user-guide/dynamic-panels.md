---
title: "Dynamic Panels — LSDVC"
description: "Practical guide to dynamic panel estimation with LSDVC: decision trees, model selection, and diagnostics"
---

# Dynamic Panels — LSDVC Guide

Dynamic panel models include lagged values of the dependent variable as regressors, capturing persistence and adjustment over time. This guide covers the LSDVC (Least Squares Dummy Variable Corrected) estimator — when to use it, how to configure it, and how to interpret results.

!!! tip "New to dynamic panels?"
    If you're unfamiliar with the Nickell bias problem, start with the [theory page](../theory/nickell-bias-lsdvc.md) for the mathematical foundations.

## Decision Tree: Which Estimator?

Use this flowchart to choose between Fixed Effects, LSDVC, and GMM:

```
Is y_{t-1} a regressor?
├── NO → Use static models (FE, RE, etc.)
│         See: Static Models Guide
│
└── YES → Dynamic panel model
    │
    ├── Are all regressors strictly exogenous?
    │   │
    │   ├── YES
    │   │   ├── N < 100? → LSDVC (recommended)
    │   │   ├── N ≥ 100, T ≤ 10? → LSDVC or GMM (both work)
    │   │   └── N large, T small? → GMM is fine
    │   │
    │   └── NO (predetermined/endogenous X)
    │       └── Use GMM (Difference or System)
    │           See: GMM Guide
    │
    └── Is ρ close to 1 (unit root)?
        ├── YES → System GMM or LSDVC with BB initial
        └── NO → Any estimator works
```

## Guide by Panel Dimensions

| N \ T | T ≤ 5 | T = 6-15 | T = 16-30 | T > 30 |
|-------|-------|----------|-----------|--------|
| **N < 30** | LSDVC (only option) | LSDVC | LSDVC | FE bias small; LSDVC optional |
| **N = 30-100** | LSDVC preferred | LSDVC or GMM | LSDVC or GMM | FE may suffice |
| **N = 100-500** | LSDVC or GMM | GMM or LSDVC | GMM or LSDVC | FE may suffice |
| **N > 500** | GMM | GMM | GMM or FE | FE (bias negligible) |

!!! info "Rule of thumb"
    The Nickell bias is approximately $-(1+\rho)/(T-1)$. When $T > 30$, the bias is often small enough to ignore for practical purposes (< 5% of $\rho$).

## Quick Start

```python
from panelbox.models.dynamic import LSDVC
from panelbox import load_grunfeld

data = load_grunfeld()

# Basic LSDVC with defaults
model = LSDVC("invest ~ value + capital", data, "firm", "year")
results = model.fit(n_bootstrap=500, seed=42)
print(results.summary())
```

## Choosing the Initial Estimator

The initial estimator provides a consistent starting value for the iterative bias correction. Three options are available:

### Anderson-Hsiao (`"ah"`)

- **Method**: IV estimation on first-differenced equation using $y_{t-2}$ as instrument
- **Pros**: Fast, minimal assumptions, works with any N
- **Cons**: Least efficient; may give noisy initial estimates
- **Best for**: Small panels, quick exploration

### Arellano-Bond (`"ab"`)

- **Method**: One-step Difference GMM with lagged levels as instruments
- **Pros**: More efficient than AH; well-established
- **Cons**: Requires moderate N; weak instruments if $\rho$ is high
- **Best for**: Moderate to large N with $\rho < 0.8$

### Blundell-Bond (`"bb"`)

- **Method**: One-step System GMM with both lagged levels and differences as instruments
- **Pros**: Most efficient; handles high persistence well
- **Cons**: Stronger assumptions (stationarity of initial conditions)
- **Best for**: Persistent processes ($\rho > 0.8$), near-unit-root data

```python
# Compare all three
for init in ["ah", "ab", "bb"]:
    model = LSDVC("invest ~ value + capital", data, "firm", "year",
                   initial_estimator=init)
    res = model.fit(n_bootstrap=200, seed=42)
    print(f"{init.upper()}: rho={res.params.iloc[0]:.4f}, "
          f"initial_rho={res.initial_rho:.4f}")
```

!!! tip "In practice"
    The final LSDVC estimates are typically robust to the choice of initial estimator. If results differ substantially across initial estimators, this may indicate model misspecification or insufficient data.

## Choosing the Bias Order

| Order | Bias Terms | When to Use |
|-------|-----------|-------------|
| 1 | $O(T^{-1})$ | Quick estimation; T is moderate (T > 15) |
| 2 | $O(T^{-1}) + O(N^{-1}T^{-1})$ | **Recommended default**; both N and T are small |
| 3 | Orders 1+2 + $O(N^{-1}T^{-2})$ | Maximum accuracy; very small T |

```python
# Compare bias orders
for order in [1, 2, 3]:
    model = LSDVC("invest ~ value + capital", data, "firm", "year",
                   bias_order=order)
    res = model.fit(n_bootstrap=0)  # skip bootstrap for speed
    print(f"Order {order}: rho={res.params.iloc[0]:.4f}, "
          f"bias_correction={res.bias_correction.iloc[0]:.4f}")
```

## Interpreting Results

### The Summary Table

```python
results = model.fit(n_bootstrap=500, seed=42)
print(results.summary())
```

The summary includes:

1. **Header**: Model info, initial estimator, bias order, convergence status
2. **Coefficient table**: LSDVC estimates with bootstrap standard errors, z-statistics, p-values, and 95% confidence intervals
3. **Bias correction table**: Side-by-side comparison of LSDV (biased) and LSDVC (corrected) estimates with the magnitude of correction

### Key Things to Check

**Convergence**: The iterative bias correction should converge (`Converged: Yes`). If it doesn't, try:

- A different initial estimator
- Increasing `max_iter`
- Checking for data issues (too few time periods, near-unit-root)

**Bias correction magnitude**: The bias correction table shows how much the estimates changed. For $\hat{\rho}$, the correction should be positive (correcting the downward Nickell bias). Large corrections (> 50% of LSDV estimate) may indicate very short T or near-unit-root behavior.

**Bootstrap confidence intervals**: With `n_bootstrap > 0`, confidence intervals are based on the bootstrap percentile method. Check that they are reasonably symmetric and not too wide.

### Bias Summary

For a focused view of the bias correction:

```python
print(results.bias_summary())
```

### Bootstrap Distribution

Visualize the bootstrap distribution to assess the shape and spread:

```python
fig = results.plot_bootstrap_distribution()  # plots first param (rho)
fig = results.plot_bootstrap_distribution("value")  # specific variable
```

A well-behaved bootstrap distribution should be approximately normal and centered near the point estimate.

## Diagnostics and Robustness

### Sensitivity to Initial Estimator

Run LSDVC with all three initial estimators. If the corrected $\hat{\rho}$ values are similar across AH, AB, and BB, the results are robust:

```python
estimates = {}
for init in ["ah", "ab", "bb"]:
    model = LSDVC("invest ~ value + capital", data, "firm", "year",
                   initial_estimator=init)
    res = model.fit(n_bootstrap=500, seed=42)
    estimates[init] = res.params

import pandas as pd
comparison = pd.DataFrame(estimates)
print(comparison)
```

### Sensitivity to Bias Order

Similarly, compare across bias orders 1-3. Results should be similar (especially orders 2 and 3):

```python
for order in [1, 2, 3]:
    model = LSDVC("invest ~ value + capital", data, "firm", "year",
                   bias_order=order)
    res = model.fit(n_bootstrap=500, seed=42)
    print(f"Order {order}: {res.params.to_dict()}")
```

### Comparing with FE and GMM

A useful diagnostic is to compare LSDVC with (biased) FE and GMM estimates. The expected pattern is:

- $\hat{\rho}_{FE} < \hat{\rho}_{LSDVC} \approx \hat{\rho}_{GMM}$ (FE is downward biased)
- $\hat{\rho}_{OLS} > \hat{\rho}_{LSDVC}$ (OLS is upward biased due to $\alpha_i$)

```python
from panelbox import FixedEffects
from panelbox.gmm import SystemGMM

# FE (biased)
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_res = fe.fit()

# LSDVC (corrected)
lsdvc = LSDVC("invest ~ value + capital", data, "firm", "year")
lsdvc_res = lsdvc.fit(n_bootstrap=500, seed=42)

# System GMM
sgmm = SystemGMM(
    "invest ~ L.invest + value + capital", data, "firm", "year",
    gmm_instruments=["L.invest"],
    iv_instruments=["value", "capital"],
)
gmm_res = sgmm.fit(two_step=True)

print(f"FE:    rho = {fe_res.params.get('L.invest', 'N/A')}")
print(f"LSDVC: rho = {lsdvc_res.params.iloc[0]:.4f}")
print(f"GMM:   rho = {gmm_res.params.get('L.invest', 'N/A')}")
```

## FAQ

### How many bootstrap replications do I need?

For standard errors: 200-500 is usually sufficient. For confidence intervals: 1000+ is recommended for accurate tail probabilities. Use the `seed` parameter for reproducibility.

### Can LSDVC handle unbalanced panels?

Yes. The implementation follows Bruno (2005) which extends the Kiviet bias correction to unbalanced panels. No special configuration is needed.

### Can I use LSDVC with more than one lag?

Currently, only `lags=1` (AR(1)) is supported. For higher-order dynamics, consider using [GMM estimators](gmm/index.md) which support arbitrary lag structures.

### What if the initial estimators give very different results?

This typically indicates one of:

- **Too few observations**: The initial estimators are all imprecise
- **Model misspecification**: The AR(1) assumption may not hold
- **Near-unit-root**: $\rho$ is close to 1; use `initial_estimator="bb"` (System GMM handles this better)

### When should I use GMM instead?

Use GMM when:

- You have **endogenous or predetermined regressors** (LSDVC requires strict exogeneity)
- N is large (N > 100-200) and you want overidentification tests
- You need to test instrument validity (Hansen J test)

### My LSDVC rho is larger than 1. Is that possible?

The bias correction can occasionally push estimates above 1, especially with very short T or noisy data. This usually indicates the model is near-unit-root territory. Consider:

- Using `initial_estimator="bb"` (better for persistent series)
- Checking if a unit root test is more appropriate for your data
- Increasing the number of time periods if possible

## See Also

- [LSDVC API Reference](../api/dynamic-models.md) — Full parameter documentation
- [Nickell Bias & LSDVC Theory](../theory/nickell-bias-lsdvc.md) — Mathematical foundations
- [GMM User Guide](gmm/index.md) — Alternative dynamic panel estimators
- [Static Models Guide](static-models/index.md) — Non-dynamic panel models
