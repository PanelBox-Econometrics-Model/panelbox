---
title: "TFP Decomposition"
description: "Total Factor Productivity growth decomposition into technical change, efficiency change, and scale effects using PanelBox SFA"
---

# TFP Decomposition

!!! info "Quick Reference"
    **Class:** `panelbox.frontier.utils.decomposition.TFPDecomposition`
    **Import:** `from panelbox.frontier.utils.decomposition import TFPDecomposition`
    **Stata equivalent:** Custom post-estimation
    **R equivalent:** Custom post-estimation

## Overview

Total Factor Productivity (TFP) measures the portion of output growth not explained by input growth. It captures technological progress, efficiency improvements, and scale effects. SFA provides a natural framework for TFP decomposition because it explicitly estimates technical efficiency, allowing researchers to separate frontier shifts (innovation) from catch-up effects (efficiency change).

PanelBox implements the Kumbhakar & Lovell (2000) decomposition framework, which decomposes TFP growth into three components:

$$\Delta \ln TFP = \Delta TC + \Delta TE + \Delta SE$$

where $\Delta TC$ is technical change (frontier shift), $\Delta TE$ is efficiency change (catch-up), and $\Delta SE$ is scale efficiency change.

## Quick Example

```python
from panelbox.frontier import StochasticFrontier
from panelbox.frontier.utils.decomposition import TFPDecomposition

# Fit a panel SFA model
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="bc92",
)
result = model.fit()

# Decompose TFP growth
tfp = TFPDecomposition(result, periods=(2010, 2020))
decomp = tfp.decompose()
print(decomp[["entity", "delta_tfp", "delta_tc", "delta_te", "delta_se"]])
```

## When to Use

- You want to understand the **sources of productivity growth** (or decline)
- You need to distinguish **frontier innovation** from **catch-up** effects
- You want entity-level decomposition showing which firms improved and why
- Policy analysis requires identifying whether growth comes from technology, efficiency, or scale

!!! warning "Key Assumptions"
    - Requires a fitted panel SFA model with entity and time identifiers
    - The decomposition assumes a Cobb-Douglas functional form (coefficients are elasticities)
    - Both comparison periods must exist in the data
    - Entities must appear in both periods for decomposition

## Detailed Guide

### Components of TFP Growth

**1. Technical Change ($\Delta TC$)**: Shift of the production frontier over time. Positive $\Delta TC$ indicates that the best-practice technology has improved (innovation, adoption of new methods).

**2. Technical Efficiency Change ($\Delta TE$)**: Movement of a firm relative to the frontier. Positive $\Delta TE$ means the firm is catching up to best practice.

$$\Delta TE = \ln(TE_{t_2}) - \ln(TE_{t_1})$$

**3. Scale Efficiency Change ($\Delta SE$)**: Gains or losses from changing the scale of operation. Depends on returns to scale (RTS):

- $RTS > 1$ (IRS): Expansion increases scale efficiency
- $RTS = 1$ (CRS): No scale effects ($\Delta SE = 0$)
- $RTS < 1$ (DRS): Expansion decreases scale efficiency

$$\Delta SE = (RTS - 1) \cdot \Delta(\text{weighted inputs})$$

### TFPDecomposition API

```python
from panelbox.frontier.utils.decomposition import TFPDecomposition

# Initialize with result and periods
tfp = TFPDecomposition(result, periods=(2010, 2020))

# Entity-level decomposition
decomp = tfp.decompose()
# Returns DataFrame with columns:
#   entity, delta_tfp, delta_tc, delta_te, delta_se, rts, verification

# Aggregate statistics
agg = tfp.aggregate_decomposition()
# Returns dict with keys:
#   mean_delta_tfp, mean_delta_tc, mean_delta_te, mean_delta_se,
#   pct_from_tc, pct_from_te, pct_from_se, std_delta_tfp, n_firms

# Text summary
print(tfp.summary())

# Visualization
fig = tfp.plot_decomposition(kind="bar", top_n=20)
fig = tfp.plot_decomposition(kind="scatter")
```

### Complete Example

```python
from panelbox.frontier import StochasticFrontier
from panelbox.frontier.utils.decomposition import TFPDecomposition

# Estimate panel SFA model
model = StochasticFrontier(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    dist="half_normal",
    model_type="bc92",
)
result = model.fit()

# Decompose TFP between first and last period
tfp = TFPDecomposition(result, periods=(2010, 2020))

# Entity-level decomposition
decomp = tfp.decompose()
print(decomp.describe())

# Aggregate summary
agg = tfp.aggregate_decomposition()
print(f"\nAverage TFP growth: {agg['mean_delta_tfp']:.4f}")
print(f"  Technical change:   {agg['mean_delta_tc']:.4f} ({agg['pct_from_tc']:.1f}%)")
print(f"  Efficiency change:  {agg['mean_delta_te']:.4f} ({agg['pct_from_te']:.1f}%)")
print(f"  Scale effects:      {agg['mean_delta_se']:.4f} ({agg['pct_from_se']:.1f}%)")

# Print full summary
print(tfp.summary())

# Visualizations
fig_bar = tfp.plot_decomposition(kind="bar", top_n=15)
fig_scatter = tfp.plot_decomposition(kind="scatter")
```

### Four-Component TFP

The `FourComponentSFA` model provides TFP decomposition with separate persistent and transient efficiency contributions:

```python
from panelbox.frontier.advanced import FourComponentSFA

model = FourComponentSFA(
    data=panel_df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="firm_id",
    time="year",
    frontier_type="production",
)
result = model.fit()

# TFP decomposition via FourComponentResult
tfp = result.tfp_decomposition(periods=(2010, 2020))
decomp = tfp.decompose()
print(tfp.summary())
```

## Related Analysis Methods

### Returns to Scale

```python
# Test H0: CRS (sum of elasticities = 1)
rts = result.returns_to_scale_test(
    input_vars=["log_labor", "log_capital"],
    alpha=0.05,
)
print(f"RTS = {rts['rts']:.4f} ({rts['conclusion']})")
```

### Elasticities

```python
# Cobb-Douglas: constant elasticities
elas = result.elasticities(input_vars=["log_labor", "log_capital"])
print(elas)

# Translog: observation-varying elasticities
elas_tl = result.elasticities(translog=True, translog_vars=["ln_K", "ln_L"])
print(elas_tl.describe())
```

### Marginal Effects on Inefficiency

For BC95 models with inefficiency determinants:

```python
# Effect of Z variables on mean inefficiency
me = result.marginal_effects(method="location")
print(me)
```

## Configuration Options

### TFPDecomposition Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `SFResult` | required | Fitted SFA result object |
| `periods` | `tuple[int, int]` | `None` | Comparison periods $(t_1, t_2)$; uses first/last if `None` |

### Key Methods

| Method | Returns | Description |
|---|---|---|
| `decompose()` | `DataFrame` | Entity-level decomposition |
| `aggregate_decomposition()` | `dict` | Mean decomposition with percentages |
| `plot_decomposition(kind, top_n)` | `Figure` | Bar chart or scatter plot |
| `summary()` | `str` | Formatted text summary |

### Aggregate Decomposition Keys

| Key | Description |
|---|---|
| `mean_delta_tfp` | Average TFP growth across firms |
| `mean_delta_tc` | Average technical change |
| `mean_delta_te` | Average efficiency change |
| `mean_delta_se` | Average scale effects |
| `pct_from_tc` | Percentage of TFP from technical change |
| `pct_from_te` | Percentage of TFP from efficiency change |
| `pct_from_se` | Percentage of TFP from scale effects |
| `std_delta_tfp` | Standard deviation of TFP growth |
| `n_firms` | Number of firms in decomposition |

## Tutorials

| Tutorial | Description | Link |
|---|---|---|
| TFP Analysis | Productivity decomposition with SFA | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/frontier/notebooks/03_four_component_tfp.ipynb) |

## See Also

- [Production and Cost Frontiers](production-cost.md) -- SFA fundamentals and elasticities
- [Panel SFA Models](panel-sfa.md) -- Panel models for TFP analysis
- [Four-Component SFA](four-component.md) -- Persistent vs transient efficiency for TFP
- [SFA Diagnostics](diagnostics.md) -- Returns to scale tests

## References

- Kumbhakar, S. C., & Lovell, C. A. K. (2000). *Stochastic Frontier Analysis*. Cambridge University Press. Chapter 7.
- Fare, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994). Productivity growth, technical progress, and efficiency change in industrialized countries. *American Economic Review*, 84(1), 66-83.
- Orea, L. (2002). Parametric decomposition of a generalized Malmquist productivity index. *Journal of Productivity Analysis*, 18(1), 5-22.
- Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014). Technical efficiency in competing panel data models: a study of Norwegian grain farming. *Journal of Productivity Analysis*, 41(2), 321-337.
