---
title: "Jackknife Analysis"
description: "Leave-one-out entity analysis for panel data models: bias estimation, standard errors, and entity influence diagnostics."
---

# Jackknife Analysis

!!! info "Quick Reference"
    **Class:** `panelbox.validation.robustness.PanelJackknife`
    **Import:** `from panelbox.validation.robustness import PanelJackknife`
    **Key method:** `jk.run()` returns `JackknifeResults`
    **Stata equivalent:** `jackknife` prefix
    **R equivalent:** `boot::jack.after.boot()`

## What It Does

The panel jackknife systematically drops one entity at a time from the dataset and re-estimates the model on the remaining $N-1$ entities. For a panel with $N$ entities, this produces $N$ re-estimations, each revealing how much a single entity contributes to the overall results.

The jackknife answers three questions:

1. **Bias**: Is the estimator biased, and by how much?
2. **Variance**: What are the standard errors under leave-one-out resampling?
3. **Influence**: Which entities have disproportionate impact on the coefficients?

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import PanelJackknife
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Jackknife analysis
jk = PanelJackknife(results, verbose=True)
jk_results = jk.run()

# View summary
print(jk.summary())

# Bias-corrected estimates
bias_corrected = jk.bias_corrected_estimates()
print(bias_corrected)

# Find influential entities
influential = jk.influential_entities(threshold=2.0)
print(influential)
```

## Mathematical Details

Given $N$ entities and original estimate $\hat\theta$, the jackknife computes:

**Jackknife mean:**

$$\bar\theta_{JK} = \frac{1}{N} \sum_{i=1}^{N} \hat\theta_{(-i)}$$

where $\hat\theta_{(-i)}$ is the estimate with entity $i$ removed.

**Jackknife bias:**

$$\text{Bias}_{JK} = (N-1)(\bar\theta_{JK} - \hat\theta)$$

**Jackknife standard error:**

$$SE_{JK} = \sqrt{\frac{N-1}{N} \sum_{i=1}^{N} (\hat\theta_{(-i)} - \bar\theta_{JK})^2}$$

**Influence of entity $i$:**

$$\text{Influence}_i = (N-1)(\hat\theta - \hat\theta_{(-i)})$$

**Bias-corrected estimate:**

$$\hat\theta_{corrected} = \hat\theta - \text{Bias}_{JK} = N\hat\theta - (N-1)\bar\theta_{JK}$$

## API Reference

### Constructor

```python
PanelJackknife(
    results=results,   # PanelResults from model.fit()
    verbose=True,      # Print progress information
)
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run()` | `JackknifeResults` | Execute leave-one-out procedure |
| `bias_corrected_estimates()` | `pd.Series` | Original estimates minus jackknife bias |
| `confidence_intervals(alpha, method)` | `pd.DataFrame` | CIs using jackknife SE (`"normal"` or `"percentile"`) |
| `influential_entities(threshold, metric)` | `pd.DataFrame` | Entities with aggregate influence above threshold |
| `summary()` | `str` | Formatted summary string |

### JackknifeResults Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `jackknife_estimates` | `pd.DataFrame` | Parameter estimates per entity excluded ($N \times K$) |
| `original_estimates` | `pd.Series` | Original full-sample estimates |
| `jackknife_mean` | `pd.Series` | Mean of jackknife estimates |
| `jackknife_bias` | `pd.Series` | $(N-1) \times (\bar\theta_{JK} - \hat\theta)$ |
| `jackknife_se` | `pd.Series` | Jackknife standard errors |
| `influence` | `pd.DataFrame` | Per-entity influence on each parameter |
| `n_jackknife` | `int` | Number of successful jackknife samples |

## Identifying Influential Entities

```python
# Default: flag entities with max absolute influence > 2x the mean
influential = jk.influential_entities(threshold=2.0, metric="max")
print(influential)

# Alternative aggregation metrics
influential_mean = jk.influential_entities(threshold=2.0, metric="mean")
influential_sum = jk.influential_entities(threshold=2.0, metric="sum")
```

The `metric` parameter controls how influence is aggregated across parameters:

| Metric | Aggregation | Use When |
|--------|-------------|----------|
| `max` | Maximum absolute influence across parameters | Default; catches entity affecting any single parameter |
| `mean` | Mean absolute influence across parameters | Detects entities with broad but moderate influence |
| `sum` | Sum of absolute influences across parameters | Emphasizes entities affecting many parameters |

An entity is flagged as influential if its aggregate influence exceeds `threshold` times the mean aggregate influence across all entities.

## Confidence Intervals

```python
# Normal approximation (using jackknife SE)
ci = jk.confidence_intervals(alpha=0.05, method="normal")

# Percentile method (using jackknife distribution)
ci = jk.confidence_intervals(alpha=0.05, method="percentile")
```

## Interpretation

!!! tip "Reading Jackknife Results"

    - **Large bias**: If `jackknife_bias` is large relative to the standard error, the original estimator may be biased. Use `bias_corrected_estimates()`.
    - **Large SE ratio**: If jackknife SE is much larger than asymptotic SE, inference based on asymptotic SE may be too liberal.
    - **Influential entities**: If removing one entity changes coefficients substantially, results are fragile. Report sensitivity to that entity.
    - **Clustered influence**: If influential entities share characteristics (e.g., all large firms), consider model specification issues.

## Jackknife vs Bootstrap

| Feature | Jackknife | Bootstrap |
|---------|-----------|-----------|
| Deterministic | Yes | No (random resampling) |
| Computational cost | $N$ re-estimations | $B$ re-estimations (typically $B \gg N$) |
| Entity influence | Directly reveals which entity matters | Not directly available |
| CI accuracy | Normal approximation | Multiple CI methods available |
| Flexibility | Leave-one-out only | Multiple resampling schemes |
| Best for | Identifying influential entities | Distribution-free inference |

The jackknife is a natural complement to bootstrap: use the jackknife to identify influential entities, then use bootstrap to validate inference.

## Common Pitfalls

!!! warning "Watch Out"

    1. **Small N**: With very few entities (e.g., $N < 10$), dropping one entity removes a substantial fraction of the data, making jackknife estimates noisy.
    2. **Failed estimations**: If a model fails to converge when one entity is removed, that entity may be essential for identification. Check the `n_jackknife` attribute.
    3. **Bias correction with small N**: Jackknife bias correction can increase variance. For small $N$, the uncorrected estimate may be preferable.
    4. **Comparison across models**: Jackknife results are model-specific. Different model specifications may identify different influential entities.

## See Also

- [Bootstrap Inference](bootstrap.md) -- Stochastic resampling alternative
- [Sensitivity Analysis](sensitivity.md) -- Generalized leave-one-out (entities, periods, subsets)
- [Influence Diagnostics](influence.md) -- Observation-level influence (Cook's D, DFFITS)
- [Robustness Overview](index.md) -- Full robustness toolkit

## References

- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC, Chapter 11.
- Shao, J., & Tu, D. (1995). *The Jackknife and Bootstrap*. Springer Science & Business Media.
- Quenouille, M. H. (1956). Notes on bias in estimation. *Biometrika*, 43(3-4), 353-360.
- Miller, R. G. (1974). The jackknife -- a review. *Biometrika*, 61(1), 1-15.
