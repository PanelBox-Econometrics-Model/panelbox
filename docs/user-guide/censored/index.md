---
title: Censored & Selection Models
description: Guide to censored and sample selection models in PanelBox - Tobit, Honore trimmed estimator, and Panel Heckman for panel data.
---

# Censored & Selection Models

Censored and selection models address two related problems in panel data:

- **Censoring**: The outcome variable is observed only within a limited range. For example, hours worked are censored at zero (we do not observe negative hours), or test scores are capped at 100. Standard linear models ignore the pile-up at the boundary, producing biased estimates.

- **Sample selection**: The outcome is observed only for a non-random subset of the population. For example, wages are observed only for employed individuals. Analyzing only the observed subsample produces selection bias.

PanelBox provides three estimators covering the main approaches for panel data with censoring or selection.

## Available Models

| Model | Class | Reference | When to Use |
|-------|-------|-----------|-------------|
| Pooled Tobit | `PooledTobit` | Tobin (1958) | Censored outcome, no entity effects |
| Random Effects Tobit | `RandomEffectsTobit` | -- | Censored outcome with RE |
| Honore Trimmed | `HonoreTrimmedEstimator` | Honore (1992) | Censored outcome with FE |
| Panel Heckman | `PanelHeckman` | Wooldridge (1995) | Sample selection bias |

## Quick Example

```python
from panelbox.models.censored import PooledTobit

# Hours worked, censored at 0
model = PooledTobit(
    "hours ~ age + education + children",
    data, "id", "year",
    lower=0  # Left-censored at 0
)
results = model.fit()
print(results.summary())
```

## Key Concepts

### Censoring vs. Truncation vs. Selection

| Problem | Definition | Model |
|---------|------------|-------|
| **Censoring** | $y^* = X\beta + \epsilon$; observe $y = \max(0, y^*)$ | Tobit |
| **Truncation** | Only observe cases where $y > 0$ (others missing entirely) | Truncated regression |
| **Selection** | $y$ observed only if selection equation $z > 0$ | Heckman |

### Tobit: Censored Outcomes

The Tobit model handles outcomes that are censored at a known boundary (typically zero):

```python
from panelbox.models.censored import RandomEffectsTobit

model = RandomEffectsTobit(
    "hours ~ age + education + children",
    data, "id", "year",
    lower=0
)
results = model.fit()
```

The Tobit model estimates three types of marginal effects:

| Effect | Interpretation |
|--------|---------------|
| Unconditional | Effect on $E(y)$ including the censored region |
| Conditional | Effect on $E(y \mid y > 0)$ for the uncensored subpopulation |
| Probability | Effect on $P(y > 0)$ |

### Honore Trimmed Estimator: FE Tobit

Standard Tobit with fixed effects suffers from the incidental parameters problem. The Honore (1992) trimmed estimator provides consistent FE estimates by exploiting the panel structure:

```python
from panelbox.models.censored import HonoreTrimmedEstimator

model = HonoreTrimmedEstimator(
    "hours ~ age + education + children",
    data, "id", "year",
    lower=0
)
results = model.fit()
```

!!! info "Requires T >= 2"
    The Honore estimator uses pairwise differences across time periods within each entity. It requires at least 2 time periods per entity and works best with balanced panels.

### Panel Heckman: Sample Selection

When the outcome is observed only for a non-random subsample, the Heckman model corrects for selection bias using a two-equation system:

```python
from panelbox.models.selection import PanelHeckman

model = PanelHeckman(
    outcome_formula="log_wage ~ education + experience",
    selection_formula="employed ~ education + experience + children",
    data=data,
    entity_col="id",
    time_col="year"
)
results = model.fit()
print(results.summary())

# Selection correction term (inverse Mills ratio)
print(f"Lambda: {results.lambda_coef:.4f}")
print(f"Lambda p-value: {results.lambda_pvalue:.4f}")
```

!!! tip "Exclusion restriction"
    For identification, the selection equation should include at least one variable that affects selection but not the outcome (an exclusion restriction). In the wage example, `children` affects employment but may not directly affect wages.

## Detailed Guides

- [Tobit Models](tobit.md) -- Pooled and RE Tobit *(detailed guide coming soon)*
- [Honore Estimator](honore.md) -- FE Tobit with trimmed estimator *(detailed guide coming soon)*
- [Panel Heckman](heckman.md) -- Sample selection correction *(detailed guide coming soon)*

## Tutorials

See [Censored & Selection Tutorial](../../tutorials/censored.md) for interactive notebooks with Google Colab.

## API Reference

See [Censored Models API](../../api/censored.md) for complete technical reference.

## References

- Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica*, 26(1), 24-36.
- Honore, B. E. (1992). Trimmed LAD and least squares estimation of truncated and censored regression models with fixed effects. *Econometrica*, 60(3), 533-565.
- Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153-161.
- Wooldridge, J. M. (1995). Selection corrections for panel data models under conditional mean independence assumptions. *Journal of Econometrics*, 68(1), 115-132.
