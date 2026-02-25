---
title: Dynamic Models (GMM)
description: Guide to GMM estimators for dynamic panel data in PanelBox - Arellano-Bond, Blundell-Bond, CUE-GMM, and Bias-Corrected GMM.
---

# Dynamic Models (GMM)

Dynamic panel models include lagged dependent variables as regressors, capturing persistence and adjustment dynamics. Standard fixed effects estimation of dynamic panels is inconsistent due to Nickell bias -- the correlation between the lagged dependent variable and the transformed error term. Generalized Method of Moments (GMM) estimators solve this problem by using lagged levels or differences as instruments.

PanelBox provides four GMM estimators covering the main approaches in the literature, with built-in diagnostic tests for instrument validity (Hansen J, Sargan) and serial correlation (AR(1)/AR(2)).

## Why GMM?

In a dynamic panel model $y_{it} = \rho y_{i,t-1} + X_{it}\beta + \alpha_i + \epsilon_{it}$:

- **OLS** is biased upward ($\hat{\rho}$ too high) because $y_{i,t-1}$ is correlated with $\alpha_i$
- **Fixed Effects** is biased downward ($\hat{\rho}$ too low) due to the Nickell bias, especially when $T$ is small
- **GMM** produces consistent estimates by using lagged values as instruments

!!! warning "Rule of thumb"
    GMM is designed for panels with **large N, small T** (many entities, few time periods). With large T, the instrument count can explode; use `collapse=True` to control proliferation.

## Available Models

| Model | Class | Reference | Key Feature |
|-------|-------|-----------|-------------|
| Difference GMM | `DifferenceGMM` | Arellano-Bond (1991) | First-differenced equations with lagged level instruments |
| System GMM | `SystemGMM` | Blundell-Bond (1998) | Adds level equations with lagged difference instruments |
| CUE-GMM | `ContinuousUpdatedGMM` | Hansen et al. (1996) | Continuously updated weight matrix; more efficient |
| Bias-Corrected | `BiasCorrectedGMM` | Hahn-Kuersteiner (2002) | Analytical correction for finite-sample bias |

## Quick Example

```python
from panelbox.gmm import SystemGMM
from panelbox.datasets import load_abdata

data = load_abdata()

model = SystemGMM(
    "n ~ L.n + w + k",
    data, "id", "year",
    gmm_instruments=["L.n"],
    iv_instruments=["w", "k"],
    collapse=True,
    time_dummies=False
)
results = model.fit(two_step=True)
print(results.summary())
```

## Key Concepts

### One-Step vs. Two-Step

| Step | Weight Matrix | Properties |
|------|--------------|------------|
| One-step | Identity or robust | Consistent; less efficient |
| Two-step | Optimal (from step 1 residuals) | More efficient; SEs may be downward biased |

!!! tip "Practical advice"
    Two-step estimates are more efficient, but report Windmeijer-corrected standard errors (applied automatically by PanelBox) to account for the downward bias of two-step SEs.

### Instrument Proliferation

The number of GMM instruments grows quadratically with $T$. Too many instruments:

- Overfit endogenous variables
- Weaken the Hansen J test (test has no power)
- Bias coefficient estimates toward OLS

**Solutions**: Use `collapse=True` or limit the lag depth with `max_lags`.

### Essential Diagnostics

| Test | What It Tests | Good Result |
|------|---------------|-------------|
| Hansen J | Instrument validity (overidentification) | p > 0.10 (do not reject) |
| AR(1) | First-order serial correlation in differences | Reject (expected) |
| AR(2) | Second-order serial correlation in differences | Do not reject |

```python
# Diagnostics are included in the summary
print(results.summary())

# Or access individually
print(f"Hansen J: p = {results.hansen_test.pvalue:.4f}")
print(f"AR(1):    p = {results.ar_tests[1].pvalue:.4f}")
print(f"AR(2):    p = {results.ar_tests[2].pvalue:.4f}")
```

## Detailed Guides

- [Difference GMM](difference-gmm.md) -- Arellano-Bond estimator *(detailed guide coming soon)*
- [System GMM](system-gmm.md) -- Blundell-Bond estimator *(detailed guide coming soon)*
- [CUE-GMM](cue-gmm.md) -- Continuously updated GMM *(detailed guide coming soon)*
- [Bias-Corrected GMM](bias-corrected.md) -- Finite-sample correction *(detailed guide coming soon)*
- [Instruments](instruments.md) -- Choosing and validating instruments *(detailed guide coming soon)*

## Tutorials

See [GMM Tutorial](../../tutorials/gmm.md) for interactive notebooks with Google Colab.

## API Reference

See [GMM API](../../api/gmm.md) for complete technical reference.

## References

- Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *Review of Economic Studies*, 58(2), 277-297.
- Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*, 87(1), 115-143.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. *Stata Journal*, 9(1), 86-136.
- Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. *Journal of Econometrics*, 126(1), 25-51.
