---
title: "Direct, Indirect, and Total Effects"
description: "Proper interpretation of spatial spillover effects using partial derivatives in PanelBox spatial models."
---

# Direct, Indirect, and Total Effects

!!! info "Quick Reference"
    **Attribute:** `results.spillover_effects`
    **Returns:** `dict` with keys `'direct'`, `'indirect'`, `'total'`
    **Relevant models:** SAR, SDM, GNS (models with spatially lagged $y$)
    **Not applicable:** SEM (coefficients are directly interpretable)

## Overview

In spatial models that include a spatially lagged dependent variable ($Wy$), the estimated coefficients $\beta$ are **not** the marginal effects of the covariates. This is one of the most common mistakes in applied spatial econometrics.

The reason is the **spatial multiplier**: because $y_i$ depends on $y_j$ (through $Wy$), and $y_j$ in turn depends on $y_i$ (through $Wy$ again), there is a feedback loop. A change in $x_i$ affects $y_i$ directly, which then affects neighbors' $y_j$, which feeds back to $y_i$, and so on. The total effect is larger than $\beta$ because of this spatial feedback.

LeSage and Pace (2009) formalized this through the **partial derivative interpretation**, decomposing effects into three components:

- **Direct effect**: the impact of changing $x_i$ on $y_i$ (own-unit, including feedback)
- **Indirect effect**: the impact of changing $x_i$ on $y_j$ for $j \neq i$ (spillover to others)
- **Total effect**: Direct + Indirect

## The Mathematics

### SAR Model

For the SAR model $y = \rho Wy + X\beta + \varepsilon$, the reduced form is:

$$y = (I - \rho W)^{-1} X\beta + (I - \rho W)^{-1} \varepsilon$$

The partial derivative of $y$ with respect to the $k$-th variable $x_k$ across all units is:

$$\frac{\partial y}{\partial x_k'} = (I - \rho W)^{-1} I_N \beta_k = S_k(\rho)$$

This is an $N \times N$ matrix. The $(i, j)$ element gives the effect of a change in $x_{jk}$ on $y_i$.

- **Direct effect** of $x_k$ = average of the diagonal of $S_k(\rho)$:

$$\text{Direct}_k = \frac{1}{N} \text{tr}\left[(I - \rho W)^{-1}\right] \beta_k$$

- **Total effect** of $x_k$ = average row (or column) sum of $S_k(\rho)$:

$$\text{Total}_k = \frac{1}{N} \mathbf{1}'(I - \rho W)^{-1}\mathbf{1} \cdot \beta_k$$

- **Indirect effect** of $x_k$ = Total - Direct:

$$\text{Indirect}_k = \text{Total}_k - \text{Direct}_k$$

### SDM Model

For the SDM $y = \rho Wy + X\beta + WX\theta + \varepsilon$, the reduced form is:

$$y = (I - \rho W)^{-1}(X\beta + WX\theta) + (I - \rho W)^{-1}\varepsilon$$

The partial derivative matrix for the $k$-th variable is:

$$S_k(\rho) = (I - \rho W)^{-1}(I_N \beta_k + W\theta_k)$$

Now the effect depends on **both** $\beta_k$ and $\theta_k$, as well as $\rho$. This is why reading $\beta$ as "direct effect" and $\theta$ as "indirect effect" is **wrong**.

### SEM Model

For the SEM $y = X\beta + u$, $u = \lambda Wu + \varepsilon$:

$$\frac{\partial y}{\partial x_k'} = I_N \beta_k$$

The matrix is diagonal. There are **no indirect effects**:

- Direct effect = $\beta_k$
- Indirect effect = 0
- Total effect = $\beta_k$

This is a major advantage of the SEM: coefficients are directly interpretable.

## Comparison Table

| Model | Direct Effect | Indirect Effect | Total Effect | Notes |
|-------|--------------|-----------------|--------------|-------|
| **OLS** | $= \beta_k$ | None | $= \beta_k$ | No spatial component |
| **SAR** | $\neq \beta_k$ | Yes | Yes | Both depend on $\rho$ |
| **SEM** | $= \beta_k$ | 0 | $= \beta_k$ | No spillovers |
| **SDM** | $\neq \beta_k$ | Yes | Yes | Depends on $\rho$, $\beta_k$, $\theta_k$ |
| **SLX** | $= \beta_k$ | $= \theta_k$ | $= \beta_k + \theta_k$ | No feedback loop |
| **GNS** | $\neq \beta_k$ | Yes | Yes | Most complex |

!!! warning "Common Mistake"
    In the SDM, $\beta$ is NOT the direct effect and $\theta$ is NOT the indirect effect. The actual effects must be computed from the spatial multiplier matrix $(I - \rho W)^{-1}$.

## Computing Effects in PanelBox

### From SAR Results

```python
from panelbox.models.spatial import SpatialLag

model = SpatialLag("y ~ x1 + x2", data, "region", "year", W=W)
results = model.fit(effects='fixed', method='qml')

# Access pre-computed spillover effects
effects = results.spillover_effects

# Direct effects (including spatial feedback)
print("Direct effects:")
for var, val in effects['direct'].items():
    print(f"  {var}: {val:.4f}")

# Indirect effects (spillovers)
print("\nIndirect effects:")
for var, val in effects['indirect'].items():
    print(f"  {var}: {val:.4f}")

# Total effects
print("\nTotal effects:")
for var, val in effects['total'].items():
    print(f"  {var}: {val:.4f}")
```

### From SDM Results

```python
from panelbox.models.spatial import SpatialDurbin

model = SpatialDurbin("y ~ x1 + x2", data, "region", "year", W=W)
results = model.fit(method='qml', effects='fixed')

# Effects decomposition
effects = results.spillover_effects

# Compare raw coefficients vs proper effects
print("Raw coefficients (NOT marginal effects):")
print(f"  beta_x1  = {results.params['x1']:.4f}")
print(f"  theta_x1 = {results.params['W_x1']:.4f}")
print(f"  rho      = {results.rho:.4f}")

print("\nProper marginal effects:")
print(f"  Direct(x1)   = {effects['direct']['x1']:.4f}")
print(f"  Indirect(x1) = {effects['indirect']['x1']:.4f}")
print(f"  Total(x1)    = {effects['total']['x1']:.4f}")
```

### Interpreting the Results

Consider a regional economics application where $y$ is GDP growth and $x_1$ is infrastructure investment:

```text
Direct effect of x1:   0.45
Indirect effect of x1: 0.18
Total effect of x1:    0.63
```

Interpretation:

- A 1-unit increase in infrastructure investment in region $i$ increases region $i$'s GDP growth by **0.45** (direct effect)
- This investment also raises GDP growth across all other regions by a total of **0.18** on average (indirect effect / spillover)
- The total impact on the economy is **0.63** (0.45 direct + 0.18 spillover)

## The Spatial Multiplier

### For SAR

The spatial multiplier for the mean effect is:

$$\text{Multiplier} = \frac{1}{1 - \rho}$$

For $\rho = 0.3$: multiplier $= 1/(1 - 0.3) = 1.43$

This means the total effect is 43% larger than the direct coefficient $\beta$ due to spatial feedback.

### Multiplier Decomposition

| $\rho$ | Multiplier | Direct/Total Ratio | Indirect Share |
|--------|-----------|-------------------|----------------|
| 0.0 | 1.00 | 100% | 0% |
| 0.1 | 1.11 | ~95% | ~5% |
| 0.3 | 1.43 | ~80% | ~20% |
| 0.5 | 2.00 | ~65% | ~35% |
| 0.7 | 3.33 | ~50% | ~50% |
| 0.9 | 10.00 | ~30% | ~70% |

As $\rho$ increases, a larger share of the total effect comes from spatial spillovers.

## Standard Errors for Effects

Standard errors for direct, indirect, and total effects are typically computed using one of two methods:

1. **Delta method**: analytical approximation using the gradient of the effect with respect to parameters
2. **Simulation**: draw from the asymptotic distribution of $(\hat{\rho}, \hat{\beta}, \hat{\theta})$ and compute effects for each draw

Both methods yield confidence intervals and p-values for each effect component.

## Practical Guidance

### Reporting Effects

When publishing results from SAR or SDM models, always report the **effect decomposition**, not just the raw coefficients. A standard table should include:

```text
Variable    | Direct  | Indirect | Total
------------|---------|----------|-------
x1          |  0.450  |  0.182   | 0.632
            | (0.052) | (0.041)  | (0.067)
x2          |  0.123  |  0.050   | 0.173
            | (0.031) | (0.019)  | (0.038)
```

### Common Pitfalls

1. **Interpreting SDM $\beta$ as direct effect**: The direct effect in SDM depends on $\rho$, $\beta$, and $\theta$ through the spatial multiplier. Always use `results.spillover_effects`.

2. **Ignoring feedback in SAR**: Even in the SAR model, the direct effect is not $\beta$ but $\beta$ times the average diagonal of $(I - \rho W)^{-1}$. The feedback through $W$ makes the direct effect slightly larger than $\beta$.

3. **Comparing effects across models**: Direct effects from different model specifications (SAR vs SDM) are not directly comparable because they are computed differently. Compare total effects if the goal is to assess overall impact.

4. **Applying to SEM**: The SEM has no indirect effects — do not compute or report them. The coefficients are directly interpretable.

## Tutorials

| Tutorial | Description | Links |
|----------|-------------|-------|
| Spatial Econometrics | Effect decomposition examples | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/spatial/notebooks/01_intro_spatial_econometrics.ipynb) |

## See Also

- [Spatial Lag (SAR)](spatial-lag.md) — Model with $\rho Wy$
- [Spatial Error (SEM)](spatial-error.md) — Model with no indirect effects
- [Spatial Durbin (SDM)](spatial-durbin.md) — Most common model needing effect decomposition
- [General Nesting Spatial (GNS)](gns.md) — Full model with all effect channels
- [Choosing a Spatial Model](choosing-model.md) — Selecting the appropriate specification

## References

1. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC. (Chapter 2: Spatial Effects)
2. Elhorst, J.P. (2010). Applied spatial econometrics: raising the bar. *Spatial Economic Analysis*, 5(1), 9-28.
3. LeSage, J. and Pace, R.K. (2014). The biggest myth in spatial econometrics. *Econometrics*, 2(4), 217-249.
4. Pace, R.K. and LeSage, J. (2010). Omitted variable biases of OLS and spatial lag models. In *Progress in Spatial Analysis*, 17-28.
