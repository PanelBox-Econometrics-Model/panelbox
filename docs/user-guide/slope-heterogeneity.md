---
title: "Slope Heterogeneity"
description: "When and how to use Mean Group and Pooled Mean Group estimators for heterogeneous panels"
---

# Slope Heterogeneity in Panel Data

!!! abstract "Summary"
    If slope coefficients vary across entities, Fixed Effects and Random Effects are inconsistent for the average effect. This guide helps you decide when to use the Mean Group (MG) or Pooled Mean Group (PMG) estimator, and how to interpret the results.

## When Do Slopes Vary?

Slope heterogeneity is common in:

- **Cross-country regressions**: different institutional structures, development levels
- **Firm-level studies**: industry-specific production technologies
- **Household panels**: heterogeneous preferences and constraints
- **Long-T panels**: with $T > 20$, entity-specific regressions become feasible and slope differences become estimable

**Rule of thumb**: If you have a large, heterogeneous sample and $T > K$ (time periods exceed the number of regressors per entity), slope heterogeneity is worth investigating.

## Decision Tree

```
Is T > K for most entities?
│
├── NO → Cannot estimate entity-specific slopes
│        → Use FE/RE with robust standard errors
│
└── YES → Run the Swamy test for slope homogeneity
          │
          ├── FAIL TO REJECT H₀ → Slopes are homogeneous
          │   → FE/RE are valid and more efficient
          │   → MG is consistent but less efficient
          │
          └── REJECT H₀ → Slopes are heterogeneous
              │
              ├── Is there a long-run equilibrium relationship?
              │   │
              │   ├── YES → Estimate both MG and PMG
              │   │         Run Hausman test (MG vs PMG)
              │   │         │
              │   │         ├── Fail to reject → Use PMG (more efficient)
              │   │         └── Reject → Use MG (robust to heterogeneity)
              │   │
              │   └── NO → Use MG
              │
              └── Use MG estimator
```

## Quick Start

### Mean Group Estimator

```python
from panelbox.models.static import MeanGroupEstimator

model = MeanGroupEstimator(
    formula="y ~ x1 + x2",
    data=panel_df,
    entity_col="country",
    time_col="year",
    min_obs_per_entity=10,
)
result = model.fit()

# Average coefficients across entities
result.summary()

# Swamy test for slope homogeneity
print(result.swamy_test_result)

# Entity-specific coefficients
result.coefficient_table()

# Distribution of a single coefficient
result.plot_coefficient_distribution("x1")
```

### Pooled Mean Group Estimator

```python
from panelbox.models.static import PooledMeanGroupEstimator

pmg = PooledMeanGroupEstimator(
    formula="y ~ x1 + x2",
    data=panel_df,
    entity_col="country",
    time_col="year",
    lags=1,
)
pmg_result = pmg.fit()

# Long-run coefficients (homogeneous)
pmg_result.summary()
```

### Hausman Test: MG vs PMG

```python
from panelbox.models.static import hausman_mg_pmg

mg_result = MeanGroupEstimator(...).fit()
pmg_result = PooledMeanGroupEstimator(...).fit()

test = hausman_mg_pmg(mg_result, pmg_result)
print(f"Hausman stat: {test['statistic']:.3f}, p-value: {test['p_value']:.4f}")
```

## Interpreting Results

### MG Output

The MG estimator reports:

| Output | Meaning |
|--------|---------|
| **Coefficients** | Average of entity-specific OLS coefficients |
| **Std. Errors** | Based on cross-entity dispersion of $\hat{\beta}_i$ |
| **Swamy test** | Tests $H_0$: all $\beta_i$ are equal |
| **N entities used** | Entities meeting the `min_obs_per_entity` threshold |
| **Entities excluded** | Dropped due to insufficient observations or collinearity |

### What to Look For

1. **Large standard errors relative to FE**: Expected — MG is less efficient under homogeneity. If SEs are similar, heterogeneity was already inflating FE standard errors.

2. **Swamy test rejects**: Confirms slope heterogeneity. Check the coefficient table to see *which* variables have heterogeneous effects.

3. **Excluded entities**: Entities with $T_i < K+1$ cannot be estimated. If many are excluded, consider reducing the number of regressors or using FE instead.

4. **Coefficient distribution**: Use `plot_coefficient_distribution()` to visualize how a coefficient varies across entities. A tight distribution suggests homogeneity for that variable; a wide or multimodal distribution suggests heterogeneity.

### PMG-Specific Output

| Output | Meaning |
|--------|---------|
| **Long-run coefficients ($\theta$)** | Homogeneous across entities |
| **Error-correction speed ($\phi_i$)** | Entity-specific; should be negative |
| **Short-run coefficients ($\delta_i$)** | Entity-specific dynamics |
| **Convergence** | Whether the optimizer converged |

**Red flags** for PMG:

- $\phi_i > 0$ for many entities: model may be misspecified
- Non-convergence: try different starting values or increase `max_iter`
- Hausman test rejects: long-run homogeneity assumption is invalid

## Comparison Table

| Feature | FE | RE | MG | PMG |
|---------|----|----|----|----|
| **Slope homogeneity** | Required | Required | Not required | Long-run only |
| **Correlated effects** | Allowed | Not allowed | Allowed | Allowed |
| **Min $T$ per entity** | 2 | 2 | $> K$ | $> K + \text{lags}$ |
| **Efficiency** | High | Highest | Lower | Medium |
| **Long-run estimation** | No | No | No | Yes |
| **Dynamic model** | No | No | No | Yes (ECM) |

## Common Pitfalls

### 1. Small T Bias

Entity-specific OLS with small $T$ is noisy. The MG average is unbiased but the variance is large. Ensure `min_obs_per_entity` is set high enough (default: 10).

### 2. Unbalanced Panels

Entities with different $T_i$ contribute equally to the MG average. If some entities have very few observations, their imprecise estimates inflate the MG variance. Consider using entity-level weights proportional to $T_i$.

### 3. Over-Rejecting Homogeneity

The Swamy test has high power with large $N$. Even tiny, economically irrelevant heterogeneity will be detected. Always examine the **magnitude** of coefficient variation using `coefficient_table()`, not just the p-value.

### 4. PMG Starting Values

PMG uses numerical optimization. Poor starting values can lead to local optima. If results are sensitive to initialization, the long-run relationship may be weak.

## FAQ

**Q: Can I use MG with time fixed effects?**

A: The standard MG runs separate regressions per entity, so time effects would need to be included as regressors (e.g., time dummies) in the formula. This is different from the demeaning approach used in two-way FE.

**Q: How many entities do I need?**

A: MG requires $N \to \infty$ for its asymptotic properties. In practice, $N \geq 20$ is a rough minimum, with $N \geq 30$ preferred. With very few entities, the cross-entity variance estimate is unreliable.

**Q: What if some entities have collinear regressors?**

A: PanelBox automatically excludes entities where the regressor matrix is rank-deficient. Check `result.entities_excluded` to see which entities were dropped and why.

**Q: Should I demean the data before MG estimation?**

A: No. Each entity-specific OLS includes an intercept, which absorbs the entity mean. Demeaning beforehand is unnecessary and would remove the entity-specific intercepts.

**Q: When should I prefer PMG over MG?**

A: When economic theory suggests a common long-run relationship (e.g., convergence models, PPP). PMG is more efficient than MG when the long-run homogeneity restriction holds. Use the Hausman test to verify.

## See Also

- [Mean Group Theory](../theory/mean-group-estimator.md) — Derivations and asymptotic results
- [Static Models API](../api/static-models.md) — `MeanGroupEstimator` and `PooledMeanGroupEstimator` reference
- [FE vs RE (Hausman)](static-models/fe-vs-re.md) — Testing under slope homogeneity
