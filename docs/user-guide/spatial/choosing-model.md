---
title: "Choosing a Spatial Model"
description: "Decision guide for selecting the right spatial econometric model for panel data in PanelBox."
---

# Choosing a Spatial Model

!!! info "Quick Reference"
    **Key decision tools:** Moran's I, LM tests, LR tests, information criteria
    **Default recommendation:** Start with SDM (LeSage & Pace, 2009)
    **Formal approach:** Estimate GNS and test restrictions

## Overview

Choosing the right spatial model is one of the most important decisions in spatial econometrics. Different models make different assumptions about how spatial dependence operates, and selecting the wrong model can lead to biased coefficients, incorrect standard errors, or misleading spillover estimates.

There are two complementary approaches to model selection:

1. **Theory-driven**: match the model to the economic mechanism generating spatial dependence
2. **Data-driven**: use statistical tests to let the data guide model choice

In practice, you should use both. Economic theory narrows the set of plausible models, and statistical tests help discriminate among them.

## Decision Framework

### Step 1: Is There Spatial Dependence?

Before fitting any spatial model, test whether spatial dependence exists:

```python
from panelbox import FixedEffects
from panelbox.diagnostics.spatial import MoranIPanelTest

# Fit standard panel model
fe = FixedEffects("y ~ x1 + x2", data, "region", "year")
fe_results = fe.fit()

# Test for spatial autocorrelation
moran = MoranIPanelTest(fe_results.resid, W)
result = moran.run()

if result.pvalue >= 0.05:
    print("No spatial autocorrelation detected.")
    print("Standard panel model is adequate.")
else:
    print(f"Moran's I = {result.statistic:.4f} (p = {result.pvalue:.4f})")
    print("Spatial dependence detected. Proceed to Step 2.")
```

If Moran's I is not significant, there is no statistical evidence for spatial dependence and a standard panel model (FE/RE) is appropriate.

### Step 2: What Type of Spatial Dependence?

Use Lagrange Multiplier tests to classify the spatial dependence:

```python
from panelbox.diagnostics.spatial import (
    LMSpatialLagTest,
    LMSpatialErrorTest,
    RobustLMSpatialLagTest,
    RobustLMSpatialErrorTest,
)

# Run LM tests
lm_lag = LMSpatialLagTest(fe_results, W).run()
lm_error = LMSpatialErrorTest(fe_results, W).run()
rlm_lag = RobustLMSpatialLagTest(fe_results, W).run()
rlm_error = RobustLMSpatialErrorTest(fe_results, W).run()

# Decision logic
if lm_lag.pvalue < 0.05 and lm_error.pvalue >= 0.05:
    print("Only LM-lag significant -> SAR")
elif lm_error.pvalue < 0.05 and lm_lag.pvalue >= 0.05:
    print("Only LM-error significant -> SEM")
elif lm_lag.pvalue < 0.05 and lm_error.pvalue < 0.05:
    # Both significant: check robust versions
    if rlm_lag.pvalue < 0.05 and rlm_error.pvalue >= 0.05:
        print("Robust LM-lag dominates -> SAR")
    elif rlm_error.pvalue < 0.05 and rlm_lag.pvalue >= 0.05:
        print("Robust LM-error dominates -> SEM")
    else:
        print("Both robust tests significant -> SDM or GNS")
```

### Step 3: Do You Need Dynamics?

If your data has both temporal and spatial dependence:

```python
# Check for temporal autocorrelation in the outcome
from panelbox.validation import WooldridgeARTest

ar_test = WooldridgeAutocorrelationTest("y ~ x1 + x2", data, "region", "year")
ar_result = ar_test.run()

if ar_result.pvalue < 0.05:
    print("Temporal autocorrelation detected.")
    print("Consider Dynamic Spatial Panel model.")
```

### Step 4: Formal Model Selection with GNS

For a rigorous model selection procedure, estimate the GNS and test restrictions:

```python
from panelbox.models.spatial import GeneralNestingSpatial

# Fit the full GNS model
gns = GeneralNestingSpatial("y ~ x1 + x2", data, "region", "year",
                             W1=W, W2=W, W3=W)
gns_results = gns.fit(effects='fixed', method='ml')

# Test restrictions to find the most parsimonious adequate model
tests = {
    'SAR':  {'theta': 0, 'lambda': 0},   # Only rho
    'SEM':  {'rho': 0, 'theta': 0},       # Only lambda
    'SDM':  {'lambda': 0},                 # rho + theta
    'SAC':  {'theta': 0},                  # rho + lambda
    'SDEM': {'rho': 0},                    # theta + lambda
    'OLS':  {'rho': 0, 'theta': 0, 'lambda': 0},  # No spatial
}

print(f"{'Model':<8} {'LR stat':>10} {'p-value':>10} {'Conclusion':>15}")
print("-" * 45)
for name, restrictions in tests.items():
    test = gns.test_restrictions(restrictions=restrictions)
    conclusion = "Reject" if test['p_value'] < 0.05 else "Accept"
    print(f"{name:<8} {test['lr_statistic']:>10.2f} {test['p_value']:>10.4f} "
          f"{conclusion:>15}")

# Automatic identification
model_type = gns.identify_model_type(gns_results)
print(f"\nIdentified model: {model_type}")
```

## Theory-Driven Selection

### Match Model to Mechanism

| Mechanism | Model | Examples |
|-----------|-------|---------|
| **Outcome spillovers** | SAR | Trade flows, migration, contagion, policy diffusion |
| **Correlated shocks** | SEM | Weather, regional policy, measurement error |
| **Both outcome and covariate spillovers** | SDM | Housing (neighbor prices + amenities), education (peer effects + background) |
| **Outcome and error dependence** | SAC | Simultaneous competition and shared shocks |
| **Covariate spillovers and correlated shocks** | SDEM | Neighbor characteristics matter but no outcome feedback |
| **Temporal + spatial** | Dynamic | GDP growth, epidemics, technology adoption |
| **Unknown mechanism** | GNS or SDM | Exploratory analysis |

### Scenario-Based Recommendations

#### Regional Economics
- **GDP growth**: Dynamic Spatial Panel (persistence + spillovers)
- **Unemployment**: SAR (labor mobility creates outcome spillovers)
- **Public spending**: SAR or SDM (fiscal competition / yardstick competition)

#### Epidemiology
- **Disease prevalence**: Dynamic Spatial Panel (temporal persistence + geographic contagion)
- **Health outcomes**: SDM (neighbor health infrastructure AND neighbor health outcomes matter)

#### Housing Markets
- **House prices**: SDM (neighbor prices AND neighbor amenities affect value)
- **Housing supply**: SEM (shared regulatory or geographic constraints)

#### Trade and Migration
- **Trade volumes**: SAR (trade begets trade; gravity model with spatial lag)
- **Migration flows**: SAR (network effects in destination choice)

#### Environmental Economics
- **Pollution**: SEM (shared atmospheric conditions)
- **Resource management**: SAR (commons problems with spatial spillovers)

## Model Comparison Table

| Model | Specification | Spatial Parameters | Indirect Effects | Estimation | Complexity |
|-------|--------------|-------------------|-----------------|------------|------------|
| **OLS/FE** | $y = X\beta + \alpha + \varepsilon$ | None | No | OLS | Lowest |
| **SAR** | $y = \rho Wy + X\beta + \varepsilon$ | $\rho$ | Yes | QML/ML | Low |
| **SEM** | $y = X\beta + u$, $u = \lambda Wu + \varepsilon$ | $\lambda$ | No | GMM/ML | Low |
| **SLX** | $y = X\beta + WX\theta + \varepsilon$ | None | $\theta$ | OLS | Low |
| **SDM** | $y = \rho Wy + X\beta + WX\theta + \varepsilon$ | $\rho$, $\theta$ | Yes | QML/ML | Medium |
| **SAC** | $y = \rho Wy + X\beta + u$, $u = \lambda Wu + \varepsilon$ | $\rho$, $\lambda$ | Yes | ML | Medium |
| **SDEM** | $y = X\beta + WX\theta + u$, $u = \lambda Wu + \varepsilon$ | $\lambda$, $\theta$ | $\theta$ | ML | Medium |
| **GNS** | $y = \rho Wy + X\beta + WX\theta + u$, $u = \lambda Wu + \varepsilon$ | $\rho$, $\lambda$, $\theta$ | Yes | ML | Highest |
| **Dynamic** | $y_{it} = \gamma y_{i,t-1} + \rho Wy_{it} + X\beta + \varepsilon$ | $\gamma$, $\rho$ | Yes | GMM | High |

## Practical Recommendations

### Default Strategy

!!! tip "Start with SDM"
    LeSage and Pace (2009) recommend starting with the **Spatial Durbin Model (SDM)** because:

    1. It nests SAR ($\theta = 0$) and SEM ($\theta = -\rho\beta$) as special cases
    2. If the true model is SAR but you estimate SDM, you lose some efficiency but remain consistent
    3. If the true model is SEM but you estimate SAR, you get biased and inconsistent estimates
    4. SDM protects against omitted spatially lagged variable bias

### Sensitivity Analysis

Always test your results' sensitivity to the weight matrix specification:

```python
from panelbox.models.spatial import SpatialLag, SpatialWeights

# Fit with different weight matrices
W_queen = SpatialWeights.from_contiguity(gdf, criterion='queen')
W_knn5 = SpatialWeights.from_knn(coords, k=5)
W_knn10 = SpatialWeights.from_knn(coords, k=10)

results_queen = SpatialLag("y ~ x1 + x2", data, "region", "year",
                            W=W_queen.matrix).fit(effects='fixed')
results_knn5 = SpatialLag("y ~ x1 + x2", data, "region", "year",
                           W=W_knn5.matrix).fit(effects='fixed')
results_knn10 = SpatialLag("y ~ x1 + x2", data, "region", "year",
                            W=W_knn10.matrix).fit(effects='fixed')

# Compare key results
print(f"{'W specification':<20} {'rho':>8} {'beta_x1':>10} {'AIC':>10}")
print("-" * 50)
for name, r in [('Queen', results_queen), ('KNN-5', results_knn5),
                ('KNN-10', results_knn10)]:
    print(f"{name:<20} {r.rho:>8.4f} {r.params['x1']:>10.4f} {r.aic:>10.1f}")
```

If results change substantially across $W$ specifications, be cautious about drawing strong conclusions.

### Complete Model Selection Workflow

```python
import numpy as np
from panelbox import FixedEffects
from panelbox.models.spatial import (
    SpatialLag, SpatialError, SpatialDurbin,
    GeneralNestingSpatial, SpatialWeights
)
from panelbox.diagnostics.spatial import MoranIPanelTest

# ---- Step 1: Baseline FE model ----
fe = FixedEffects("y ~ x1 + x2", data, "region", "year")
fe_results = fe.fit()

# ---- Step 2: Test for spatial dependence ----
moran = MoranIPanelTest(fe_results.resid, W)
moran_result = moran.run()
print(f"Moran's I: {moran_result.statistic:.4f} (p = {moran_result.pvalue:.4f})")

if moran_result.pvalue >= 0.05:
    print("No spatial dependence. Use standard FE model.")
else:
    # ---- Step 3: Fit competing spatial models ----
    sar = SpatialLag("y ~ x1 + x2", data, "region", "year", W=W)
    sar_res = sar.fit(effects='fixed', method='qml')

    sem = SpatialError("y ~ x1 + x2", data, "region", "year", W=W)
    sem_res = sem.fit(effects='fixed', method='gmm')

    sdm = SpatialDurbin("y ~ x1 + x2", data, "region", "year", W=W)
    sdm_res = sdm.fit(effects='fixed', method='qml')

    # ---- Step 4: Compare models ----
    print(f"\n{'Model':<8} {'Log-lik':>10} {'AIC':>10} {'BIC':>10}")
    print("-" * 40)
    for name, res in [('SAR', sar_res), ('SEM', sem_res), ('SDM', sdm_res)]:
        print(f"{name:<8} {res.llf:>10.1f} {res.aic:>10.1f} {res.bic:>10.1f}")

    # ---- Step 5: Post-estimation check ----
    best_model = sdm_res  # Start with SDM
    moran_post = MoranIPanelTest(best_model.resid, W)
    post = moran_post.run()
    print(f"\nPost-SDM Moran's I: {post.statistic:.4f} (p = {post.pvalue:.4f})")

    # ---- Step 6: Effect decomposition ----
    effects = best_model.spillover_effects
    print("\nEffect Decomposition (SDM):")
    print(f"{'Variable':<10} {'Direct':>10} {'Indirect':>10} {'Total':>10}")
    print("-" * 42)
    for var in effects['direct']:
        print(f"{var:<10} {effects['direct'][var]:>10.4f} "
              f"{effects['indirect'][var]:>10.4f} "
              f"{effects['total'][var]:>10.4f}")
```

## Common Pitfalls

### 1. Over-Parameterization
GNS and SDM may have too many parameters for small samples. Check:

- $N > 30$ (cross-sectional units) for simple spatial models
- $N > 50$ for SDM with many covariates
- $N > 100$ for GNS with separate weight matrices

### 2. Weight Matrix Sensitivity
Results can be sensitive to $W$ specification. Always:

- Test at least 2-3 different weight structures
- Report sensitivity analysis in publications
- Justify your choice based on economic theory

### 3. Ignoring Temporal Dynamics
Panel data often has temporal dependence. If the lagged dependent variable is significant:

- Omitting it biases spatial parameter estimates
- Use the Dynamic Spatial Panel model
- Or at minimum, include time dummies

### 4. Mechanical Model Selection
Do not rely purely on statistical tests:

- Economic theory should guide model choice
- Consider interpretability of results
- SDM is safer as a default than pure statistical selection

### 5. Reporting Only the Preferred Model
For publication:

- Report at least SAR, SEM, and SDM side-by-side
- Show diagnostic test results
- Include sensitivity to weight matrix
- Report direct, indirect, and total effects for SAR/SDM

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Full model selection workflow | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Weight Matrices](spatial-weights.md) — Constructing the weight matrix
- [Spatial Lag (SAR)](spatial-lag.md) — For outcome spillovers
- [Spatial Error (SEM)](spatial-error.md) — For correlated shocks
- [Spatial Durbin (SDM)](spatial-durbin.md) — Recommended starting point
- [Dynamic Spatial Panel](dynamic-spatial.md) — When temporal dynamics matter
- [General Nesting Spatial (GNS)](gns.md) — For formal restriction tests
- [Direct, Indirect, and Total Effects](spatial-effects.md) — Interpreting spatial effects
- [Spatial Diagnostics](diagnostics.md) — Full diagnostic test suite

## References

1. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
2. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
3. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic.
4. Lee, L.F. and Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*, 154(2), 165-185.
5. Manski, C.F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531-542.
6. Gibbons, S. and Overman, H.G. (2012). Mostly pointless spatial econometrics? *Journal of Regional Science*, 52(2), 172-191.
