---
title: Quantile Regression
description: Guide to panel quantile regression models in PanelBox - Pooled, Fixed Effects, Canay, Location-Scale, Dynamic, and Quantile Treatment Effects.
---

# Quantile Regression

Standard panel models estimate the conditional mean of the outcome variable. Quantile regression goes beyond the mean, estimating the effect of covariates at different points of the outcome distribution -- the 10th percentile, the median, the 90th percentile, etc. This reveals heterogeneous effects: a policy may help low-income households more than high-income ones, or a treatment may reduce extreme outcomes without affecting the median.

PanelBox provides six quantile estimators and two analysis tools, covering pooled, fixed effects, and dynamic specifications.

## Why Quantile Regression?

Mean regression answers: "What happens to the average $y$ when $x$ increases by one unit?"

Quantile regression answers: "What happens to the $\tau$-th quantile of $y$ when $x$ increases by one unit?"

This distinction matters when:

- Effects differ across the distribution (e.g., education premium varies by income level)
- You care about tail behavior (e.g., risk management, poverty analysis)
- The outcome distribution is skewed or has outliers

## Available Models

| Model | Class | Reference | Key Feature |
|-------|-------|-----------|-------------|
| Pooled Quantile | `PooledQuantile` | Koenker & Bassett (1978) | Ignores panel structure |
| Fixed Effects Quantile | `FixedEffectsQuantile` | Koenker (2004) | Penalized FE for quantiles |
| Canay Two-Step | `CanayTwoStep` | Canay (2011) | Two-step debiased FE quantile |
| Location-Scale | `LocationScale` | -- | Separate location and scale effects |
| Dynamic Quantile | `DynamicQuantile` | -- | Lagged dependent variable at quantiles |
| Quantile Treatment Effects | `QuantileTreatmentEffects` | -- | Distributional treatment effects |

## Quick Example

```python
from panelbox.models.quantile import FixedEffectsQuantile
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate at the median (tau = 0.5)
model = FixedEffectsQuantile(
    "invest ~ value + capital",
    data, "firm", "year",
    quantile=0.5
)
results = model.fit()
print(results.summary())
```

### Comparing Across Quantiles

```python
from panelbox.models.quantile import PooledQuantile

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

for tau in quantiles:
    model = PooledQuantile(
        "invest ~ value + capital",
        data, "firm", "year",
        quantile=tau
    )
    results = model.fit()
    print(f"tau={tau}: value={results.params['value']:.4f}")
```

## Key Concepts

### Quantile Estimator Comparison

| Estimator | Handles FE? | Consistency | Best For |
|-----------|-------------|-------------|----------|
| Pooled | No | Yes (no FE) | Baseline, quick analysis |
| FE Quantile (Koenker) | Yes | Yes (large T) | Large T panels |
| Canay Two-Step | Yes | Yes (large N, T) | Standard micro panels |
| Location-Scale | Yes | Yes | Testing for heterogeneous dispersion |
| Dynamic | Yes | Yes (GMM-style) | Persistence at quantiles |

### Quantile Treatment Effects

```python
from panelbox.models.quantile import QuantileTreatmentEffects

qte = QuantileTreatmentEffects(
    "outcome ~ treatment + controls",
    data, "id", "year",
    treatment_var="treatment",
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)
qte_results = qte.fit()
print(qte_results.summary())
```

### Monotonicity and Quantile Crossing

Quantile estimates should be monotonically ordered (the 10th percentile should be below the 90th). PanelBox provides tools to detect and address quantile crossing:

```python
from panelbox.models.quantile import QuantileMonotonicity

mono = QuantileMonotonicity(results_dict)
report = mono.check_crossing()
print(report)
```

## Detailed Guides

- [Pooled Quantile](pooled.md) -- Basic quantile regression *(detailed guide coming soon)*
- [FE Quantile](fixed-effects.md) -- Koenker penalized approach *(detailed guide coming soon)*
- [Canay Two-Step](canay.md) -- Debiased two-step estimator *(detailed guide coming soon)*
- [Location-Scale](location-scale.md) -- Heterogeneous dispersion *(detailed guide coming soon)*
- [Dynamic Quantile](dynamic.md) -- Persistence at quantiles *(detailed guide coming soon)*

## Tutorials

See [Quantile Regression Tutorial](../../tutorials/quantile.md) for interactive notebooks with Google Colab.

## API Reference

See [Quantile API](../../api/quantile.md) for complete technical reference.

## References

- Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2004). Quantile regression for longitudinal data. *Journal of Multivariate Analysis*, 91(1), 74-89.
- Canay, I. A. (2011). A simple approach to quantile regression for panel data. *Econometrics Journal*, 14(3), 368-386.
