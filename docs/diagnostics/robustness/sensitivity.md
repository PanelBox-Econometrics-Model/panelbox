---
title: "Sensitivity Analysis"
description: "Assess parameter stability across entities, time periods, and random subsamples in panel data models."
---

# Sensitivity Analysis

!!! info "Quick Reference"
    **Class:** `panelbox.validation.robustness.SensitivityAnalysis`
    **Import:** `from panelbox.validation.robustness import SensitivityAnalysis`
    **Key methods:** `leave_one_out_entities()`, `leave_one_out_periods()`, `subset_sensitivity()`
    **Stata equivalent:** Custom iteration
    **R equivalent:** `sensemakr::sensemakr()`

## What It Tests

Sensitivity analysis evaluates whether parameter estimates are stable when the sample composition changes. If dropping one entity, one time period, or a random 20% of observations shifts a coefficient from significant to insignificant, the finding is fragile and should be reported as such.

PanelBox provides three complementary sensitivity modes:

| Mode | What It Does | Question Answered |
|------|-------------|-------------------|
| Leave-one-out entities | Drop each entity, re-estimate | "Does one country/firm drive the results?" |
| Leave-one-out periods | Drop each time period, re-estimate | "Does one year/quarter drive the results?" |
| Subset sensitivity | Random 80% subsamples, re-estimate | "Are results stable across random splits?" |

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import SensitivityAnalysis
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Sensitivity analysis
sa = SensitivityAnalysis(results, show_progress=False)

# Leave-one-out entities
loo_entities = sa.leave_one_out_entities(influence_threshold=2.0)
print(sa.summary(loo_entities))

# Which entities are influential?
print(f"Influential entities: {loo_entities.influential_units}")

# Visualize coefficient stability
fig = sa.plot_sensitivity(loo_entities, params=["value", "capital"])
```

## Three Analysis Modes

### Leave-One-Out Entities

Drop each entity $i$ from the dataset and re-estimate the model $N$ times. An entity is flagged as influential if removing it shifts any coefficient by more than `influence_threshold` standard errors.

```python
sa = SensitivityAnalysis(results)
loo_entities = sa.leave_one_out_entities(influence_threshold=2.0)

# Results
print(loo_entities.method)            # "leave_one_out_entities"
print(loo_entities.estimates)          # pd.DataFrame (N × K)
print(loo_entities.statistics)         # dict with per-parameter stats
print(loo_entities.influential_units)  # list of flagged entity labels
print(loo_entities.subsample_info)     # pd.DataFrame (excluded, n_obs, converged)
```

### Leave-One-Out Periods

Drop each time period and re-estimate. Reveals whether a single year or event drives the results.

```python
loo_periods = sa.leave_one_out_periods(influence_threshold=2.0)

print(f"Influential periods: {loo_periods.influential_units}")
print(sa.summary(loo_periods))
```

### Subset Sensitivity

Draw `n_subsamples` random subsamples (each containing a fraction `subsample_size` of entities) and re-estimate. This assesses overall stability without focusing on individual units.

```python
subsets = sa.subset_sensitivity(
    n_subsamples=20,
    subsample_size=0.8,
    stratify=True,
    random_state=42,
)

print(sa.summary(subsets))
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_subsamples` | 20 | Number of random subsamples |
| `subsample_size` | 0.8 | Fraction of entities per subsample (0 to 1) |
| `stratify` | `True` | Maintain temporal balance by sampling entities (not observations) |
| `random_state` | `None` | Seed for reproducibility |

## SensitivityResults

All three modes return a `SensitivityResults` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | `"leave_one_out_entities"`, `"leave_one_out_periods"`, or `"subset_sensitivity"` |
| `estimates` | `pd.DataFrame` | Coefficient estimates per subsample (rows) and parameter (columns) |
| `std_errors` | `pd.DataFrame` | Standard errors per subsample |
| `statistics` | `dict` | Per-parameter summary: mean, std, range, max deviation, % beyond threshold |
| `influential_units` | `list` | Labels of entities/periods flagged as influential |
| `subsample_info` | `pd.DataFrame` | Metadata per subsample (excluded unit, n_obs, converged) |

### Statistics Dictionary

For each parameter, `statistics[param]` contains:

| Key | Description |
|-----|-------------|
| `mean` | Mean estimate across subsamples |
| `std` | Standard deviation of estimates |
| `min` / `max` | Range of estimates |
| `range` | max - min |
| `max_abs_deviation` | Largest absolute deviation from original |
| `mean_abs_deviation` | Average absolute deviation from original |
| `max_std_deviation` | Largest deviation in SE units |
| `n_beyond_threshold` | Count of subsamples exceeding threshold |
| `pct_beyond_threshold` | Percentage exceeding threshold |

## Visualization

```python
# Plot coefficient stability
fig = sa.plot_sensitivity(loo_entities, params=["value", "capital"])

# Customize
fig = sa.plot_sensitivity(
    loo_entities,
    params=["value"],
    figsize=(12, 6),
    reference_line=True,      # Red dashed line at original estimate
    confidence_band=True,     # Blue band at mean +/- 1.96*std
)
```

The plot shows each subsample's estimate as a point, with the original estimate as a red reference line and a 95% confidence band in blue. Narrow bands indicate robust estimates; wide bands suggest sensitivity to sample composition.

## Summary Table

```python
summary_df = sa.summary(loo_entities)
print(summary_df)
```

Returns a `pd.DataFrame` with columns: `Parameter`, `Original`, `Mean`, `Std`, `Min`, `Max`, `Range`, `Max Deviation`, `Max Dev (SE)`, `N Valid`.

## Interpretation

!!! tip "Reading Sensitivity Results"

    - **Narrow range**: Estimates vary little across subsamples -- the finding is robust.
    - **Wide range**: Results are sensitive to sample composition -- proceed with caution.
    - **Specific influential units**: If one entity/period is flagged, investigate why. Is it an outlier? A structural break? Or does it carry unique information?
    - **`pct_beyond_threshold` > 5%**: More than 5% of subsamples produce estimates beyond 2 SE -- the finding may not be robust.
    - **Sign changes**: If the coefficient changes sign across subsamples, the result is unreliable.

## Sensitivity vs Jackknife

| Feature | SensitivityAnalysis | PanelJackknife |
|---------|-------------------|----------------|
| LOO entities | Yes | Yes |
| LOO periods | Yes | No |
| Random subsets | Yes | No |
| Bias estimation | No | Yes ($\text{Bias}_{JK}$) |
| Influence formula | Standardized deviation | $(N-1)(\hat\theta - \hat\theta_{(-i)})$ |
| Visualization | `plot_sensitivity()` | -- |
| Summary table | `summary()` returns DataFrame | `summary()` returns string |

Use `PanelJackknife` when you need formal bias estimates and influence decomposition. Use `SensitivityAnalysis` for a broader assessment including time periods and random subsets.

## Common Pitfalls

!!! warning "Watch Out"

    1. **Threshold sensitivity**: The `influence_threshold=2.0` default may be too strict or too lenient depending on the application. Report results for multiple thresholds.
    2. **Small N problems**: With $N < 10$ entities, leave-one-out removes 10-20% of the data each time, amplifying natural variation. Interpret with care.
    3. **Convergence failures**: Check `subsample_info["converged"]`. If many subsamples fail to converge, the model may be poorly identified.
    4. **Multiple testing**: With many entities and parameters, some will appear "influential" by chance. Focus on substantively meaningful deviations.

## See Also

- [Jackknife Analysis](jackknife.md) -- Formal bias and variance estimation via leave-one-out
- [Bootstrap Inference](bootstrap.md) -- Resampling-based inference
- [Influence Diagnostics](influence.md) -- Observation-level influence measures
- [Robustness Overview](index.md) -- Full robustness toolkit

## References

- Leamer, E. E. (1983). Let's take the con out of econometrics. *American Economic Review*, 73(1), 31-43.
- Lu, X., & White, H. (2014). Robustness checks and robustness tests in applied economics. *Journal of Econometrics*, 178(P1), 194-206.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39-67.
