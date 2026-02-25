---
title: Count Data Models
description: Guide to panel count data models in PanelBox - Poisson, Negative Binomial, PPML, and Zero-Inflated models for panel data.
---

# Count Data Models

Count data models are designed for non-negative integer outcomes: number of patents filed, hospital admissions, trade flows, crime incidents, etc. Standard linear regression is inappropriate for count data because it can predict negative values and ignores the discrete, non-negative nature of the outcome. Panel count models address these issues while accounting for unobserved entity heterogeneity.

PanelBox provides 11 count data estimators covering Poisson, Negative Binomial, PPML (for trade/gravity models), and Zero-Inflated specifications with pooled, fixed effects, random effects, and quasi-maximum likelihood options.

## Available Models

### Poisson Models

| Model | Class | Estimation | Key Feature |
|-------|-------|-----------|-------------|
| Pooled Poisson | `PooledPoisson` | MLE + Cluster-robust | Baseline count model |
| Poisson FE | `PoissonFixedEffects` | Conditional MLE | Consistent with entity effects |
| Poisson RE | `RandomEffectsPoisson` | MLE | Gamma-distributed RE |
| Poisson QML | `PoissonQML` | Quasi-MLE | Robust to overdispersion |

### Negative Binomial Models

| Model | Class | Estimation | Key Feature |
|-------|-------|-----------|-------------|
| Negative Binomial | `NegativeBinomial` | MLE (NB2) | Handles overdispersion |
| NB Fixed Effects | `FixedEffectsNegativeBinomial` | Conditional MLE | FE + overdispersion |

### PPML (Gravity Models)

| Model | Class | Estimation | Key Feature |
|-------|-------|-----------|-------------|
| PPML | `PPML` | Pseudo-MLE | Trade/gravity; handles zeros |

### Zero-Inflated Models

| Model | Class | Estimation | Key Feature |
|-------|-------|-----------|-------------|
| Zero-Inflated Poisson | `ZeroInflatedPoisson` | EM/MLE | Excess zeros (structural) |
| Zero-Inflated NB | `ZeroInflatedNegativeBinomial` | EM/MLE | Excess zeros + overdispersion |

## Quick Example

```python
from panelbox.models.count import PoissonFixedEffects

model = PoissonFixedEffects(
    "patents ~ rd_spending + employees",
    data, "firm", "year"
)
results = model.fit()
print(results.summary())
```

## Key Concepts

### Overdispersion

The Poisson model assumes $\text{Var}(y) = E(y)$ (equidispersion). In practice, count data often exhibits **overdispersion**: $\text{Var}(y) > E(y)$. Overdispersion does not bias Poisson coefficients but invalidates standard errors.

**Solutions**:

| Approach | Implementation | When to Use |
|----------|---------------|-------------|
| Cluster-robust SEs | `PooledPoisson` with cluster | Mild overdispersion |
| Quasi-MLE | `PoissonQML` | Moderate overdispersion |
| Negative Binomial | `NegativeBinomial` | Strong overdispersion |

### Excess Zeros

When the data contains more zeros than the Poisson or NB distributions predict, a **zero-inflated** model separates the zero-generating process from the count process:

```python
from panelbox.models.count import ZeroInflatedPoisson

model = ZeroInflatedPoisson(
    "patents ~ rd_spending + employees",
    data, "firm", "year",
    inflate_formula="~ small_firm + new_entrant"  # Predicts excess zeros
)
results = model.fit()
```

### PPML for Gravity Models

The Poisson Pseudo-Maximum Likelihood estimator is the standard approach for gravity models in international trade. It handles zeros in trade flows and provides consistent estimates under heteroskedasticity:

```python
from panelbox.models.count import PPML

model = PPML(
    "trade ~ log_gdp_origin + log_gdp_dest + log_distance",
    data, "pair", "year"
)
results = model.fit()
```

!!! tip "PPML advantage"
    Unlike log-linear OLS (`log(trade) ~ ...`), PPML handles zero trade flows naturally and is consistent under heteroskedasticity (Santos Silva & Tenreyro, 2006).

### Poisson Fixed Effects

The Poisson FE estimator uses conditional MLE (Hausman, Hall, and Griliches, 1984), conditioning out the fixed effects to avoid the incidental parameters problem:

```python
from panelbox.models.count import PoissonFixedEffects

model = PoissonFixedEffects("count ~ x1 + x2", data, "id", "year")
results = model.fit()
```

## Detailed Guides

- [Poisson Models](poisson.md) -- Pooled, FE, RE, QML
- [Negative Binomial](negative-binomial.md) -- Overdispersion modeling
- [PPML](ppml.md) -- Gravity models and trade
- [Zero-Inflated Models](zero-inflated.md) -- Excess zeros
- [Marginal Effects](marginal-effects.md) -- Interpreting nonlinear count models

## Tutorials

See [Count Data Tutorial](../../tutorials/count.md) for interactive notebooks with Google Colab.

## API Reference

See [Count Data API](../../api/count.md) for complete technical reference.

## References

- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- Hausman, J. A., Hall, B. H., & Griliches, Z. (1984). Econometric models for count data with an application to the patents-R&D relationship. *Econometrica*, 52(4), 909-938.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. *Review of Economics and Statistics*, 88(4), 641-658.
- Wooldridge, J. M. (1999). Distribution-free estimation of some nonlinear panel data models. *Journal of Econometrics*, 90(1), 77-97.
