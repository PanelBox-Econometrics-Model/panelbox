---
title: Stochastic Frontier Analysis
description: Guide to stochastic frontier models in PanelBox - SFA, Four-Component model, and TFP decomposition for efficiency analysis.
---

# Stochastic Frontier Analysis

Stochastic Frontier Analysis (SFA) estimates how efficiently firms, hospitals, schools, or other decision-making units convert inputs into outputs (production frontier) or minimize costs (cost frontier). Unlike standard regression, SFA decomposes the error term into a symmetric noise component and a one-sided inefficiency component, providing entity-level efficiency scores.

PanelBox provides a comprehensive SFA toolkit, including the **Four-Component model** -- the only Python implementation -- which separates persistent inefficiency from transient inefficiency while controlling for unobserved heterogeneity.

## Why SFA?

Standard regression estimates the **average** relationship between inputs and outputs. SFA estimates the **frontier** (best-practice) relationship and measures how far each entity falls below it:

```text
Production:  y_it = f(X_it; beta) + v_it - u_it
                                    ↑        ↑
                                  noise   inefficiency (u >= 0)
```

The inefficiency term $u_{it} \geq 0$ ensures that entities operate at or below the frontier, and the noise term $v_{it}$ accounts for measurement error and random shocks.

## Available Models

| Model | Class | Inefficiency Structure | Key Feature |
|-------|-------|----------------------|-------------|
| Cross-Section SFA | `StochasticFrontier` | $u_i$ (time-invariant) | Basic efficiency estimation |
| Panel SFA | `StochasticFrontier` | $u_{it}$ (time-varying) | Exploits panel structure |
| True Fixed Effects | `StochasticFrontier` | $u_{it} + \alpha_i$ | Entity effects + inefficiency |
| True Random Effects | `StochasticFrontier` | $u_{it} + \alpha_i$ | Entity effects as random |
| Four-Component | `FourComponentSFA` | $\mu_i + \eta_{it}$ | Persistent + transient inefficiency |

!!! info "Unique in Python"
    The **Four-Component SFA** model (Colombi et al., 2014; Kumbhakar et al., 2014) decomposes the error into four parts: firm heterogeneity ($\alpha_i$), persistent inefficiency ($\mu_i$), transient inefficiency ($\eta_{it}$), and noise ($v_{it}$). PanelBox is the only Python library implementing this model.

## Quick Example

```python
from panelbox.frontier import StochasticFrontier
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

model = StochasticFrontier(
    "invest ~ value + capital",
    data, "firm", "year",
    frontier_type="production",
    dist_type="half_normal"
)
results = model.fit()
print(results.summary())

# Entity-level efficiency scores
print(results.efficiency_scores)
```

## Key Concepts

### Frontier Type

| Type | Inefficiency Direction | Interpretation |
|------|----------------------|----------------|
| Production | $u \geq 0$ subtracted | Output below maximum |
| Cost | $u \geq 0$ added | Cost above minimum |

### Inefficiency Distributions

| Distribution | `dist_type` | Properties |
|-------------|-------------|------------|
| Half-Normal | `"half_normal"` | Mode at zero; most entities near-efficient |
| Exponential | `"exponential"` | Mode at zero; longer tail |
| Truncated Normal | `"truncated_normal"` | Flexible mode location |

### Four-Component Model

```python
from panelbox.frontier import FourComponentSFA

model = FourComponentSFA(
    "invest ~ value + capital",
    data, "firm", "year",
    frontier_type="production"
)
results = model.fit()

# Separate persistent and transient inefficiency
print(results.persistent_efficiency)   # Time-invariant component
print(results.transient_efficiency)    # Time-varying component
print(results.overall_efficiency)      # Combined
```

### TFP Decomposition

Decompose Total Factor Productivity growth into technical change, efficiency change, and scale effects:

```python
tfp = results.tfp_decomposition()
print(tfp.technical_change)    # Frontier shifts over time
print(tfp.efficiency_change)   # Catching up to the frontier
print(tfp.scale_effect)        # Returns to scale
print(tfp.tfp_growth)          # Total TFP growth
```

## Detailed Guides

- [Production Frontier](production-cost.md) -- Production efficiency estimation *(detailed guide coming soon)*
- [Cost Frontier](production-cost.md) -- Cost efficiency estimation *(detailed guide coming soon)*
- [Four-Component Model](four-component.md) -- Persistent vs. transient inefficiency *(detailed guide coming soon)*
- [TFP Decomposition](tfp.md) -- Productivity growth decomposition *(detailed guide coming soon)*

## Tutorials

See [Stochastic Frontier Tutorial](../../tutorials/frontier.md) for interactive notebooks with Google Colab.

## API Reference

See [Frontier API](../../api/frontier.md) for complete technical reference.

## References

- Aigner, D., Lovell, C. A. K., & Schmidt, P. (1977). Formulation and estimation of stochastic frontier production function models. *Journal of Econometrics*, 6(1), 21-37.
- Battese, G. E., & Coelli, T. J. (1992). Frontier production functions, technical efficiency and panel data. *Journal of Productivity Analysis*, 3(1), 153-169.
- Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014). Closed-skew normality in stochastic frontiers with individual effects and long/short-run efficiency. *Journal of Productivity Analysis*, 42(2), 123-136.
- Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014). Technical efficiency in competing panel data models: A study of Norwegian grain farming. *Journal of Productivity Analysis*, 41(2), 321-337.
