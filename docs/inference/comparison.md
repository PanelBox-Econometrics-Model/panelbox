---
title: "Comparing Standard Error Methods"
description: "Systematically compare standard error types and assess inference sensitivity across methods in PanelBox."
---

# Comparing Standard Error Methods

!!! info "Quick Reference"
    **Class:** `panelbox.standard_errors.StandardErrorComparison`
    **Result:** `panelbox.standard_errors.ComparisonResult`
    **Stata equivalent:** `estimates table`, `suest`
    **R equivalent:** `modelsummary::modelsummary()`, `lmtest::coeftest()`

## Overview

Different standard error methods can lead to **different conclusions** about the same model. A coefficient that appears statistically significant with classical SEs may become insignificant with clustered SEs, or vice versa. PanelBox's `StandardErrorComparison` class provides a systematic way to compare SE types and identify sensitivity in inference.

Comparing SEs is not just a robustness exercise --- it reveals information about the **error structure** in your data. If clustered SEs are much larger than robust SEs, it signals substantial within-cluster correlation. If Driscoll-Kraay SEs differ from clustered SEs, cross-sectional dependence may be present.

## When to Use

- As a **robustness check** in any empirical study
- When **reviewers request** alternative SE specifications
- To **diagnose** the error structure (heteroskedasticity, clustering, cross-sectional dependence)
- When **inference changes** across SE types and you need to decide which to report

## Quick Example

```python
from panelbox.standard_errors import StandardErrorComparison
from panelbox.models import FixedEffects

# Fit model
model = FixedEffects("y ~ x1 + x2", data, entity="firm", time="year")
results = model.fit()

# Compare all SE types
comparison = StandardErrorComparison(results)
comp = comparison.compare_all()

# Print summary
comparison.summary(comp)

# Visualize
comparison.plot_comparison(comp, alpha=0.05)
```

## StandardErrorComparison Class

### Initialization

```python
comparison = StandardErrorComparison(model_results=results)
```

The class requires a fitted `PanelResults` object. It extracts the model, coefficients, and residual degrees of freedom to refit the model with different SE types.

### compare_all()

Compare multiple SE types simultaneously:

```python
comp = comparison.compare_all(
    se_types=["nonrobust", "robust", "hc3", "clustered", "driscoll_kraay"],
)
```

If `se_types` is not specified, the default list includes `"nonrobust"`, `"robust"`, `"hc3"`, and `"clustered"`. Advanced types (`"driscoll_kraay"`, `"newey_west"`) are added automatically when the sample is large enough.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `se_types` | `list` or `None` | `None` | SE types to compare |
| `**kwargs` | --- | --- | Additional params (e.g., `max_lags` for DK/NW) |

### compare_pair()

Compare two specific SE types:

```python
comp = comparison.compare_pair("robust", "clustered")
```

### plot_comparison()

Visualize standard errors and confidence intervals across SE types:

```python
fig = comparison.plot_comparison(comp, alpha=0.05, figsize=(12, 8))
```

Produces two panels:

1. **Bar chart**: Standard errors by coefficient and SE type
2. **Coefficient plot**: Point estimates with confidence intervals for each SE type

### summary()

Print a detailed text summary:

```python
comparison.summary(comp)
```

The summary includes:

- Standard errors by type
- SE ratios relative to baseline
- Significance levels across SE types
- Inference sensitivity analysis (which coefficients change significance)

## ComparisonResult

The `compare_all()` and `compare_pair()` methods return a `ComparisonResult` dataclass:

| Attribute | Type | Description |
|-----------|------|-------------|
| `se_comparison` | `pd.DataFrame` | Standard errors by type (rows = coefficients, columns = SE types) |
| `se_ratios` | `pd.DataFrame` | Ratios relative to baseline (typically `nonrobust`) |
| `t_stats` | `pd.DataFrame` | t-statistics under each SE type |
| `p_values` | `pd.DataFrame` | Two-tailed p-values |
| `ci_lower` | `pd.DataFrame` | Lower bounds of 95% confidence intervals |
| `ci_upper` | `pd.DataFrame` | Upper bounds of 95% confidence intervals |
| `significance` | `pd.DataFrame` | Significance stars (`*`, `**`, `***`) |
| `summary_stats` | `pd.DataFrame` | Summary statistics (mean, std, min, max, range, CV) |

## Interpretation Guidelines

### Reading SE Ratios

The `se_ratios` DataFrame shows how each SE type compares to the baseline:

| Pattern | Likely Diagnosis | Action |
|---------|-----------------|--------|
| Robust/Classical $\approx$ 1 | Homoskedastic errors | Classical SE are fine |
| Robust/Classical $> 1.2$ | Heteroskedasticity | Use robust or clustered SE |
| Clustered/Robust $> 1.3$ | Within-cluster correlation | Use clustered SE |
| DK/Clustered $> 1.2$ | Cross-sectional dependence | Use Driscoll-Kraay |
| All SE types similar | Well-behaved errors | Any method works; report clustered |
| Large variation across types | Complex error structure | Investigate further; consider multiple methods |

### When Inference Changes

If a coefficient is significant under some SE types but not others, this signals **fragile inference**. The `summary()` method flags such coefficients:

```python
comparison.summary(comp)
# Output includes:
# "Coefficients with inconsistent inference across SE types:
#   x2:
#     Significant with: nonrobust, robust
#     Not significant with: clustered, driscoll_kraay"
```

In such cases:

1. **Report the most conservative SE** (usually clustered or DK)
2. **Acknowledge sensitivity** in your paper
3. **Investigate the error structure** to understand why SEs differ

## Full Comparison Example

```python
import panelbox as pb
from panelbox.standard_errors import StandardErrorComparison

# Load data and fit model
data = pb.datasets.load_grunfeld()
model = pb.FixedEffects("invest ~ value + capital", data, entity="firm", time="year")
results = model.fit()

# Comprehensive comparison
comparison = StandardErrorComparison(results)
comp = comparison.compare_all(
    se_types=["nonrobust", "robust", "hc3", "clustered"],
)

# View SE comparison table
print("Standard Errors by Type:")
print(comp.se_comparison.to_string(float_format=lambda x: f"{x:.4f}"))
print()

# View significance across methods
print("Significance:")
print(comp.significance)
print()

# View SE ratios (relative to nonrobust)
print("SE Ratios (vs nonrobust):")
print(comp.se_ratios.to_string(float_format=lambda x: f"{x:.3f}"))

# Plot
fig = comparison.plot_comparison(comp)
```

## Best Practices for Reporting

1. **Default**: Report clustered SE by entity as your primary specification
2. **Robustness table**: Show results under 2-3 alternative SE types
3. **Diagnose first**: Run diagnostic tests (Breusch-Pagan, Pesaran CD, Wooldridge AR) to justify your SE choice
4. **Document**: State which SE type you use and why in your methodology section
5. **Sensitivity**: If inference is sensitive to SE choice, acknowledge this and present results under multiple specifications

## Common Pitfalls

!!! warning "Pitfall 1: Cherry-picking SEs"
    Do not choose the SE type that gives you the results you want. Choose based on the data structure and diagnostic tests, then report robustness to alternatives.

!!! warning "Pitfall 2: Comparing SEs without context"
    Large differences between SE types are informative, not alarming. They reveal information about the error structure. Use diagnostic tests to understand why SEs differ.

!!! warning "Pitfall 3: Ignoring the comparison"
    Reporting only one SE type without robustness checks is increasingly viewed as insufficient by journals. Always include at least one alternative specification.

## See Also

- [Robust (HC0-HC3)](robust.md) --- Heteroskedasticity-robust SE
- [Clustered](clustered.md) --- Cluster-robust SE
- [Driscoll-Kraay](driscoll-kraay.md) --- Cross-sectional dependence robust SE
- [Newey-West](newey-west.md) --- HAC SE
- [PCSE](pcse.md) --- Panel-corrected SE
- [Spatial HAC](spatial-hac.md) --- Spatial HAC SE
- [MLE Variance](mle-variance.md) --- MLE sandwich SE
- [Inference Overview](index.md) --- Decision tree for SE selection

## References

- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies*, 22(1), 435-480.
- Thompson, S. B. (2011). Simple formulas for standard errors that cluster by both firm and time. *Journal of Financial Economics*, 99(1), 1-10.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
