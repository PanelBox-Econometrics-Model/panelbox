---
title: "Robustness Analysis"
description: "Robustness toolkit for panel data models: bootstrap, jackknife, cross-validation, sensitivity analysis, and influence diagnostics."
---

# Robustness Analysis

Econometric results should not depend on a handful of observations, a single time period, or a particular set of assumptions. Robustness analysis systematically stress-tests your estimates to identify fragile findings before they reach publication.

PanelBox provides seven complementary tools that cover every dimension of robustness assessment:

## Robustness Workflow

A complete robustness assessment follows this sequence:

1. **Bootstrap** -- Are standard errors and confidence intervals reliable under resampling?
2. **Jackknife** -- Which entities drive the results?
3. **Cross-validation** -- Does the model predict well out of sample?
4. **Sensitivity** -- Are parameters stable across subsamples and time periods?
5. **Outliers** -- Are extreme observations driving the estimates?
6. **Influence** -- Which individual observations have outsized impact on coefficients?
7. **Alternative specifications** -- Do results survive different model formulations?

## Tool Comparison

| Tool | Class | Purpose | Key Metric |
|------|-------|---------|------------|
| Bootstrap | [`PanelBootstrap`](bootstrap.md) | Inference reliability | Bootstrap CI |
| Jackknife | [`PanelJackknife`](jackknife.md) | Entity influence | Jackknife SE, bias |
| Cross-Validation | [`TimeSeriesCV`](cross-validation.md) | Predictive accuracy | Out-of-sample R² |
| Sensitivity | [`SensitivityAnalysis`](sensitivity.md) | Parameter stability | Max deviation |
| Outlier Detection | [`OutlierDetector`](outliers.md) | Extreme values | Outlier count |
| Influence | [`InfluenceDiagnostics`](influence.md) | Observation impact | Cook's D, DFFITS |
| Specifications | `RobustnessChecker` | Specification sensitivity | Coefficient comparison |

## Quick Start

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import PanelBootstrap
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Bootstrap confidence intervals (5 lines)
bootstrap = PanelBootstrap(results, n_bootstrap=1000, method="pairs", random_state=42)
bootstrap.run()
ci = bootstrap.conf_int(alpha=0.05, method="percentile")
print(ci)
```

## Comprehensive Robustness Analysis

The following example demonstrates a full robustness pipeline:

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import (
    PanelBootstrap,
    PanelJackknife,
    TimeSeriesCV,
    SensitivityAnalysis,
    OutlierDetector,
    InfluenceDiagnostics,
    RobustnessChecker,
)
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# 1. Bootstrap inference
bootstrap = PanelBootstrap(results, n_bootstrap=1000, method="pairs", random_state=42)
bootstrap.run()
print(bootstrap.summary())

# 2. Jackknife -- which firms drive results?
jk = PanelJackknife(results)
jk_results = jk.run()
print(jk.summary())
influential = jk.influential_entities(threshold=2.0)
print(influential)

# 3. Cross-validation -- out-of-sample fit
cv = TimeSeriesCV(results, method="expanding", min_train_periods=3)
cv_results = cv.cross_validate()
print(f"Out-of-sample R²: {cv_results.metrics['r2_oos']:.4f}")

# 4. Sensitivity -- parameter stability
sa = SensitivityAnalysis(results)
loo_entities = sa.leave_one_out_entities(influence_threshold=2.0)
print(sa.summary(loo_entities))

# 5. Outlier detection
detector = OutlierDetector(results)
outliers = detector.detect_outliers_residuals(method="standardized", threshold=2.5)
print(f"Outliers detected: {outliers.n_outliers}")

# 6. Influence diagnostics
influence = InfluenceDiagnostics(results)
infl = influence.compute()
print(influence.summary())

# 7. Alternative specifications
checker = RobustnessChecker(results)
alt_results = checker.check_alternative_specs(
    formulas=["invest ~ value", "invest ~ value + capital"],
    model_type="fe",
)
table = checker.generate_robustness_table(alt_results, parameters=["value"])
print(table)
```

## Interpreting Results

| Finding | Concern Level | Recommended Action |
|---------|:------------:|---------------------|
| Bootstrap CIs match asymptotic CIs | Low | Results are robust to distributional assumptions |
| Bootstrap CIs much wider than asymptotic | High | Asymptotic SEs may be too optimistic; report bootstrap CIs |
| One entity changes coefficients substantially | High | Report results with and without entity; investigate why |
| Many residual outliers detected | Medium | Re-estimate excluding outliers; compare with original |
| Negative out-of-sample R² | High | Model overfits; simplify specification |
| High Cook's D for a few observations | Medium | Investigate those observations; assess economic meaning |
| Coefficients unstable across subsamples | High | Results are fragile; consider alternative specification |

## Choosing the Right Tool

!!! tip "Decision Guide"

    - **"Are my standard errors correct?"** -- Use [PanelBootstrap](bootstrap.md)
    - **"Is one country driving all my results?"** -- Use [PanelJackknife](jackknife.md) or [SensitivityAnalysis](sensitivity.md)
    - **"Does my model predict well?"** -- Use [TimeSeriesCV](cross-validation.md)
    - **"Are there extreme data points?"** -- Use [OutlierDetector](outliers.md)
    - **"Which observations matter most for my coefficients?"** -- Use [InfluenceDiagnostics](influence.md)
    - **"Do results survive different specifications?"** -- Use `RobustnessChecker`

## Alternative Specifications with RobustnessChecker

`RobustnessChecker` automates the comparison of multiple model specifications:

```python
from panelbox.validation.robustness import RobustnessChecker

checker = RobustnessChecker(results, verbose=True)

# Test alternative formulas
alt_results = checker.check_alternative_specs(
    formulas=[
        "invest ~ value",
        "invest ~ value + capital",
        "invest ~ value + capital + lag_invest",
    ],
    model_type="fe",  # 'fe', 'pooled', 're', or None (auto)
)

# Generate comparison table
table = checker.generate_robustness_table(
    results_list=alt_results,
    parameters=["value", "capital"],
)
print(table)
```

The resulting table shows coefficients, standard errors, and p-values side by side for each specification.

## Software Equivalents

| PanelBox | Stata | R |
|----------|-------|---|
| `PanelBootstrap` | `bootstrap` prefix | `boot::boot()` |
| `PanelJackknife` | `jackknife` prefix | `boot::jack.after.boot()` |
| `TimeSeriesCV` | Rolling estimation (custom) | `caret::trainControl(method="timeslice")` |
| `SensitivityAnalysis` | Custom iteration | `sensemakr::sensemakr()` |
| `OutlierDetector` | `predict, rstudent` | `stats::influence.measures()` |
| `InfluenceDiagnostics` | `predict, cooksd dfits` | `car::influenceIndexPlot()` |
| `RobustnessChecker` | `estimates store` / `esttab` | `modelsummary::modelsummary()` |

## Detailed Guides

<div class="grid cards" markdown>

-   :material-shuffle-variant: **[Bootstrap Inference](bootstrap.md)**

    ---

    Resampling-based confidence intervals with four bootstrap methods (pairs, wild, block, residual).

-   :material-knife: **[Jackknife Analysis](jackknife.md)**

    ---

    Leave-one-out entity analysis for bias estimation and entity-level influence.

-   :material-chart-timeline: **[Cross-Validation](cross-validation.md)**

    ---

    Expanding and rolling window cross-validation for out-of-sample prediction.

-   :material-tune: **[Sensitivity Analysis](sensitivity.md)**

    ---

    Parameter stability across entities, time periods, and random subsamples.

-   :material-alert-circle-outline: **[Outlier Detection](outliers.md)**

    ---

    Univariate, multivariate, and residual-based outlier identification.

-   :material-target: **[Influence Diagnostics](influence.md)**

    ---

    Cook's distance, DFFITS, and DFBETAS for identifying high-impact observations.

</div>

## See Also

- [Diagnostics Overview](../index.md) -- Full diagnostic workflow
- [Standard Errors](../../inference/index.md) -- Bootstrap SEs as an alternative to asymptotic SEs
- [Validation Tutorial](../../tutorials/validation.md) -- Interactive robustness examples

## References

- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press, Chapter 11.
- Cook, R. D., & Weisberg, S. (1982). *Residuals and Influence in Regression*. Chapman and Hall.
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *Annals of Statistics*, 7(1), 1-26.
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414-427.
