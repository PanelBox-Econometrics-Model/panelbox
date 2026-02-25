---
title: "Time-Series Cross-Validation"
description: "Out-of-sample predictive evaluation for panel data models using expanding and rolling window cross-validation."
---

# Time-Series Cross-Validation

!!! info "Quick Reference"
    **Class:** `panelbox.validation.robustness.TimeSeriesCV`
    **Import:** `from panelbox.validation.robustness import TimeSeriesCV`
    **Key method:** `cv.cross_validate()` returns `CVResults`
    **Stata equivalent:** Rolling estimation (custom)
    **R equivalent:** `caret::trainControl(method="timeslice")`

## Why Cross-Validation for Panels?

In-sample goodness-of-fit ($R^2$, AIC, BIC) measures how well the model describes the data it was estimated on. But the real test of a model is whether it can predict data it has not seen. Time-series cross-validation evaluates out-of-sample predictive performance while respecting the temporal ordering of panel data -- no future data ever leaks into the training set.

## Two CV Methods

PanelBox implements two temporal CV strategies:

### Expanding Window

Train on periods $[1, t]$, predict period $t+1$, then expand to $[1, t+1]$, predict $t+2$, and so on. The training set grows with each fold.

```text
Fold 1: Train [1,2,3]         → Predict [4]
Fold 2: Train [1,2,3,4]       → Predict [5]
Fold 3: Train [1,2,3,4,5]     → Predict [6]
...
```

### Rolling Window

Train on a fixed-size window $[t-w, t]$, predict $t+1$, then slide the window. The training set size remains constant.

```text
Fold 1: Train [1,2,3]   → Predict [4]
Fold 2: Train [2,3,4]   → Predict [5]
Fold 3: Train [3,4,5]   → Predict [6]
...
```

!!! tip "When to Use Which"
    - **Expanding window**: When you believe more data always helps (stable relationships). This is the default.
    - **Rolling window**: When you suspect structural change or time-varying parameters, so recent data is more relevant than distant past.

## Quick Example

```python
from panelbox import FixedEffects
from panelbox.validation.robustness import TimeSeriesCV
from panelbox.datasets import load_grunfeld

data = load_grunfeld()
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()

# Expanding window cross-validation
cv = TimeSeriesCV(results, method="expanding", min_train_periods=3, verbose=True)
cv_results = cv.cross_validate()

# Overall metrics
print(f"Out-of-sample R²: {cv_results.metrics['r2_oos']:.4f}")
print(f"RMSE:             {cv_results.metrics['rmse']:.4f}")
print(f"MAE:              {cv_results.metrics['mae']:.4f}")

# Per-fold breakdown
print(cv_results.fold_metrics)

# Full summary
print(cv.summary())
```

## API Reference

### Constructor

```python
TimeSeriesCV(
    results=results,           # PanelResults from model.fit()
    method="expanding",        # 'expanding' or 'rolling'
    window_size=None,          # Required for 'rolling' method
    min_train_periods=3,       # Minimum training periods (>= 2)
    verbose=True,              # Print progress
)
```

!!! warning "`window_size` is Required for Rolling"
    When using `method="rolling"`, you must specify `window_size`. A good starting point is `window_size = T // 2` where `T` is the number of time periods.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `cross_validate()` | `CVResults` | Run CV and return results |
| `plot_predictions(entity)` | -- | Actual vs predicted plot (all entities or specific one) |
| `summary()` | `str` | Formatted summary string |

### CVResults Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `predictions` | `pd.DataFrame` | Columns: `actual`, `predicted`, `fold`, `test_period`, `entity`, `time` |
| `metrics` | `dict` | Overall metrics: `mse`, `rmse`, `mae`, `r2_oos` |
| `fold_metrics` | `pd.DataFrame` | Per-fold metrics with fold number and test period |
| `method` | `str` | CV method used (`'expanding'` or `'rolling'`) |
| `n_folds` | `int` | Number of CV folds |
| `window_size` | `int` or `None` | Window size (for rolling CV) |

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Mean squared prediction error |
| RMSE | $\sqrt{MSE}$ | In the units of the dependent variable |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust to outliers (unlike MSE) |
| $R^2_{OOS}$ | $1 - \frac{SS_{res}}{SS_{tot}}$ | Out-of-sample explained variance |

!!! warning "Negative $R^2_{OOS}$"
    Unlike in-sample $R^2$, the out-of-sample $R^2$ can be negative. A negative value means the model predicts worse than simply using the sample mean as the forecast. This signals overfitting or a misspecified model.

## Rolling Window Example

```python
# Rolling window with 5-period training window
cv_roll = TimeSeriesCV(
    results,
    method="rolling",
    window_size=5,
    min_train_periods=3,
    verbose=True,
)
cv_results_roll = cv_roll.cross_validate()

print(f"Rolling R² (OOS): {cv_results_roll.metrics['r2_oos']:.4f}")
print(f"Number of folds:  {cv_results_roll.n_folds}")
```

## Visualization

```python
# Plot actual vs predicted for a specific entity
cv.plot_predictions(entity="General Motors")

# Plot for all entities (scatter + time series)
cv.plot_predictions()
```

The `plot_predictions()` method produces two panels:

1. **Scatter plot**: Actual vs predicted values with a 45-degree reference line
2. **Time series**: Mean actual and predicted values over time periods

## Panel-Specific Considerations

!!! note "Temporal Integrity"
    PanelBox cross-validation always respects temporal ordering:

    - Training data uses only past periods (no future leakage)
    - All entities are included in each fold (cross-sectional dimension is preserved)
    - Models are fully re-estimated for each fold (no parameter recycling)

    This is more conservative than random k-fold CV, which would violate the time-series structure.

## Comparing Expanding vs Rolling

```python
# Run both methods
cv_exp = TimeSeriesCV(results, method="expanding", min_train_periods=3)
cv_roll = TimeSeriesCV(results, method="rolling", window_size=5, min_train_periods=3)

exp_results = cv_exp.cross_validate()
roll_results = cv_roll.cross_validate()

print(f"Expanding R² (OOS): {exp_results.metrics['r2_oos']:.4f}")
print(f"Rolling R² (OOS):   {roll_results.metrics['r2_oos']:.4f}")

# If rolling >> expanding: structural change may be present
# If expanding >> rolling: stable relationships; more data helps
```

## Common Pitfalls

!!! warning "Watch Out"

    1. **Too few training periods**: Setting `min_train_periods=2` may produce unreliable models. Use at least 3 periods.
    2. **Ignoring fold variation**: Stable overall metrics can mask poor performance in specific periods. Always check `fold_metrics`.
    3. **Fixed effects with new entities**: If entities appear/disappear over time, some folds may fail because the training set lacks data for test-set entities.
    4. **Computational cost**: Each fold requires a full model re-estimation. For large panels, this can be slow.

## See Also

- [Bootstrap Inference](bootstrap.md) -- Resampling-based inference
- [Sensitivity Analysis](sensitivity.md) -- Parameter stability across subsamples
- [Robustness Overview](index.md) -- Full robustness toolkit

## References

- Bergmeir, C., & Benitez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437-450.
- Racine, J. (2000). Consistent cross-validatory model-selection for dependent data: hv-block cross-validation. *Journal of Econometrics*, 99(1), 39-61.
