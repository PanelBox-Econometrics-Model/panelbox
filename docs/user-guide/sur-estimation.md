---
title: "SUR Estimation Guide"
description: "Practical guide to Seemingly Unrelated Regressions in PanelBox — when to use SUR, decision checklist, iterated vs non-iterated, and connection to PanelVAR"
---

# SUR Estimation Guide

!!! abstract "Key Takeaway"
    Use SUR when your panel entities have **correlated errors** and **different regressors**. If both conditions hold, SUR is more efficient than equation-by-equation OLS. If regressors are identical across entities, SUR gives the same point estimates as OLS.

## When to Use SUR

### Decision Checklist

Ask these questions in order:

```
1. Do entities have correlated errors?
   ├── NO  → OLS per entity is fine. SUR won't help.
   └── YES
        2. Do entities have different regressors?
           ├── NO  → SUR = OLS (no efficiency gain). Use FE/RE instead.
           └── YES
                3. Are coefficients entity-specific?
                   ├── YES → SUR is the right model.
                   └── NO (common β)
                        → Consider Pooled OLS + PCSE for common β
                          with cross-sectional correlation correction.
```

### SUR vs Alternatives

| Scenario | Best Model | Why |
|----------|-----------|-----|
| Correlated errors + different regressors | **SUR** | Exploits cross-equation structure |
| Correlated errors + same regressors | FE/RE or OLS+PCSE | SUR = OLS when X identical |
| Uncorrelated errors + different regressors | OLS per entity | No cross-equation info to exploit |
| Common $\beta$ + cross-sectional dependence | Pooled OLS + PCSE | One coefficient vector, robust SEs |
| Slope heterogeneity + large N | Mean Group | MG averages, SUR estimates jointly |
| Dynamic model + endogeneity | GMM | SUR assumes exogenous regressors |

---

## Quick Start

### Basic SUR (same formula, all entities)

```python
import panelbox as pb

data = pb.load_grunfeld()
sur = pb.SUR("invest ~ value + capital", data, "firm", "year")
result = sur.fit()

# System-level summary with Sigma matrix and BP test
print(result.system_summary())

# Individual equation
print(result.equation_summary("General Motors"))
```

### Heterogeneous Formulas (dict)

```python
sur = pb.SUR(
    formula={
        "General Motors": "invest ~ value + capital",
        "Chrysler": "invest ~ value",
        "General Electric": "invest ~ capital",
    },
    data=data,
    entity_col="firm",
    time_col="year",
)
result = sur.fit()
```

Each entity gets its own regression specification. Entities not in the dict are excluded.

### Iterated SUR (ISUR ≈ MLE)

```python
sur = pb.SUR(
    "invest ~ value + capital", data, "firm", "year",
    iterate=True, max_iter=200, tol=1e-8,
)
result = sur.fit()
print(f"Converged: {result.converged} in {result.n_iterations} iterations")
```

---

## Interpreting Results

### System Summary

The `system_summary()` output includes:

| Section | What to Look For |
|---------|-----------------|
| **System R² (McElroy)** | Overall fit across all equations jointly |
| **Sigma matrix** | Cross-equation error covariances — large off-diagonal → SUR gains |
| **Correlation matrix** | Standardized version of Sigma — magnitudes close to ±1 mean strong cross-equation dependence |
| **Breusch-Pagan test** | If $p < 0.05$, reject independence → SUR is beneficial |
| **Per-equation R²** | Fit quality per entity |
| **Efficiency gain** | Ratio SE(SUR)/SE(OLS) — values < 1 mean SUR improved precision |

### Breusch-Pagan Independence Test

```python
bp = result.bp_independence_test
print(f"LM stat: {bp['statistic']:.2f}, p-value: {bp['pvalue']:.4f}")
```

| Result | Interpretation |
|--------|---------------|
| $p < 0.05$ | Errors are correlated across entities — SUR is worthwhile |
| $p \geq 0.05$ | Cannot reject independence — SUR ≈ OLS, no efficiency gain |

### Efficiency Gain Table

```python
# Values < 1 mean SUR is more efficient
for entity, gains in result.efficiency_gain.items():
    print(f"\n{entity}:")
    print(gains)
```

A gain ratio of 0.85 means SUR standard errors are 85% of OLS standard errors for that coefficient — a 15% improvement.

---

## The Degenerate Case

!!! warning "Same Regressors = No Gain"
    When all entities share the **exact same formula and data structure**, SUR point estimates are identical to equation-by-equation OLS (Zellner 1962). The efficiency gain comes from **different regressors** across equations combined with correlated errors.

If your Breusch-Pagan test rejects independence but the efficiency gains are negligible, check whether your regressors are effectively the same across entities.

---

## Iterated SUR vs Non-Iterated

| Feature | Non-iterated (FGLS) | Iterated (ISUR) |
|---------|---------------------|-----------------|
| Steps | 2-step: OLS → GLS | Iterates until convergence |
| Equivalence | — | MLE under normality |
| Efficiency | Asymptotically efficient | Asymptotically efficient |
| Finite-sample | May differ slightly | Closer to MLE in small samples |
| Robustness | Less sensitive to $\hat{\Sigma}$ | More sensitive to specification |
| Use when | Default choice | Want MLE-equivalent estimates |

!!! tip "Practical Advice"
    Start with the non-iterated SUR (`iterate=False`, the default). If results are sensitive to switching to ISUR, this may indicate misspecification. In well-specified models, the two should be very close.

---

## Connection to PanelVAR

PanelBox's PanelVAR module supports `cov_type='sur'`, which uses the same cross-equation covariance estimation:

```python
from panelbox.var import PanelVAR

var = PanelVAR(data, variables=["y1", "y2"], entity_col="firm", time_col="year", lags=2)
result = var.fit(cov_type="sur")
```

In a VAR system, each equation has different regressors (different lags appear on the RHS), so the SUR structure provides genuine efficiency gains. The underlying mechanics are the same:

1. Estimate each VAR equation by OLS
2. Build the cross-equation $\hat{\Sigma}$ from residuals
3. Apply FGLS using Kronecker structure

---

## Common Pitfalls

### 1. Too Many Entities, Too Few Periods

$\hat{\Sigma}$ is an $N \times N$ matrix estimated from $T$ observations. When $N$ is large relative to $T$, $\hat{\Sigma}$ is poorly estimated or singular.

**Rule of thumb**: SUR works best with small $N$ (few entities) and moderate to large $T$.

### 2. Unbalanced Panels

SUR handles unbalanced panels by computing $\hat{\sigma}_{ij}$ from the $T_{ij}$ common time periods between entities $i$ and $j$. If two entities share very few periods, their covariance estimate will be noisy.

### 3. Endogenous Regressors

SUR assumes all regressors are exogenous. If you suspect endogeneity, use GMM or IV estimation instead.

### 4. Expecting Gains with Identical Regressors

If all entities have the same formula and the same observations, SUR cannot improve on OLS. This is a mathematical identity, not a limitation.

---

## Full Example

```python
import panelbox as pb

# Load data
data = pb.load_grunfeld()

# Fit SUR with heterogeneous formulas
sur = pb.SUR(
    formula={
        "General Motors": "invest ~ value + capital",
        "Chrysler": "invest ~ value",
    },
    data=data,
    entity_col="firm",
    time_col="year",
    iterate=True,
)
result = sur.fit()

# System-level diagnostics
print(result.system_summary())

# Correlation heatmap
result.plot_correlation_matrix()

# Check if SUR was worth it
bp = result.bp_independence_test
if bp["pvalue"] < 0.05:
    print("Cross-equation correlation is significant — SUR is beneficial.")
else:
    print("No significant cross-equation correlation — OLS is sufficient.")
```

---

## See Also

- [SUR Theory](../theory/sur-zellner.md) — Mathematical derivation and properties
- [Static Models API](../api/static-models.md#sur) — Full API reference
- [Slope Heterogeneity (MG/PMG)](slope-heterogeneity.md) — Alternative for heterogeneous slopes
- [PCSE Inference](../inference/pcse.md) — Cross-sectional correction for common $\beta$
- [Panel VAR](var/index.md) — VAR estimation with SUR covariance
