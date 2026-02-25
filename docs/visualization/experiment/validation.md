---
title: "Validation Reports"
description: "Running diagnostic tests and generating interactive validation reports with PanelBox"
---

# Validation Reports

## Overview

Validation is a critical step in any econometric analysis. PanelBox's `ValidationResult` container runs a battery of specification tests on a fitted model and packages the results into an interactive HTML report with pass/fail indicators, p-value distributions, and actionable recommendations.

## Running Validation

### From PanelExperiment

The most common way to validate a model is through `PanelExperiment`:

```python
import panelbox as pb

data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# Fit a model
exp.fit_model('fixed_effects', name='fe')

# Validate it
val = exp.validate_model('fe', tests='default', alpha=0.05, verbose=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | Name of the fitted model to validate |
| `tests` | `str` | `'default'` | Which tests to run (see below) |
| `alpha` | `float` | `0.05` | Significance level for all tests |
| `verbose` | `bool` | `False` | Print progress during test execution |

### From Model Results Directly

You can also create a `ValidationResult` without a `PanelExperiment`:

```python
from panelbox.experiment.results import ValidationResult

# Fit a model manually
fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
results = fe.fit()

# Create ValidationResult
val = ValidationResult.from_model_results(
    model_results=results,
    alpha=0.05,
    tests='default'
)
```

## Test Categories

The `tests` parameter controls which diagnostic tests are run:

| Value | Description |
|-------|-------------|
| `'default'` | Core tests appropriate for the model type |
| `'all'` | Every applicable test |
| `'serial'` | Serial correlation tests only |
| `'het'` | Heteroskedasticity tests only |
| `'cd'` | Cross-sectional dependence tests only |

!!! tip "Custom Test Selection"
    The `tests` parameter is passed to the model's `validate()` method. The exact tests available depend on the model type. For fixed effects models, the default suite typically includes Hausman, Breusch-Pagan, Wooldridge, and cross-sectional dependence tests.

## ValidationResult Properties

The returned `ValidationResult` provides convenient properties to inspect results programmatically:

```python
# How many tests were run?
val.total_tests       # int, e.g., 9

# Which tests passed?
val.passed_tests      # list[str], e.g., ['Hausman Test', 'Pesaran CD Test']

# Which tests failed?
val.failed_tests      # list[str], e.g., ['Wooldridge Test', 'Breusch-Pagan Test']

# Overall pass rate
val.pass_rate         # float (0.0-1.0), e.g., 0.222
```

### Inspecting Individual Tests

The underlying `validation_report` object contains detailed test results organized by category:

```python
# Access the raw validation report
report = val.validation_report

# Specification tests (e.g., Hausman, RESET)
report.specification_tests

# Serial correlation tests (e.g., Wooldridge)
report.serial_tests

# Heteroskedasticity tests (e.g., Breusch-Pagan)
report.het_tests

# Cross-sectional dependence tests (e.g., Pesaran CD)
report.cd_tests
```

## Output Options

### Text Summary

```python
print(val.summary())
```

Output:

```text
Validation Report Summary
=========================
Total Tests: 9
Passed: 2
Failed: 7

Specification Tests:
  Hausman Test: statistic=45.23, p=0.000 FAIL
  ...
```

### Interactive HTML Report

```python
val.save_html(
    "validation_report.html",
    test_type="validation",
    theme="professional",       # 'professional', 'academic', 'presentation'
    title="FE Model Validation",
    open_browser=False
)
```

The HTML report includes:

- **Test overview chart** -- visual summary of all tests with pass/fail status
- **P-value distribution** -- histogram of p-values across all tests
- **Individual test details** -- statistic, p-value, critical value, and interpretation for each test
- **Recommendations** -- suggestions based on failed tests

### JSON Export

```python
val.save_json("validation_results.json", indent=2)
```

The JSON file includes all test results, metadata, and timestamps for reproducibility and integration with other tools.

### Python Dictionary

```python
data = val.to_dict()
# Keys: 'model_info', 'tests', 'summary', 'recommendations', 'charts'
```

## Complete Example

```python
import panelbox as pb

# 1. Set up experiment
data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Fit the model
exp.fit_model('fixed_effects', name='fe')

# 3. Run validation
val = exp.validate_model('fe', tests='default', alpha=0.05)

# 4. Quick inspection
print(f"Total tests: {val.total_tests}")
print(f"Passed: {len(val.passed_tests)} ({val.pass_rate:.0%})")
print(f"Failed: {val.failed_tests}")

# 5. Decide on corrections
if val.pass_rate < 0.5:
    print("Consider robust standard errors or model re-specification")

# 6. Generate HTML report
val.save_html(
    "fe_validation.html",
    test_type="validation",
    theme="professional",
    title="Fixed Effects Validation Report"
)

# 7. Archive results
val.save_json("fe_validation.json")
```

## Interpreting Validation Results

!!! note "Statistical Testing Logic"
    Most diagnostic tests use the null hypothesis that the model is correctly specified. A **low p-value** (below `alpha`) means the null is rejected, indicating a **potential problem**. So a "failed" test means the model may violate an assumption.

Common actions based on failed tests:

| Failed Test | Indicates | Recommended Action |
|-------------|-----------|-------------------|
| Hausman | FE vs RE choice matters | Use the recommended estimator |
| Breusch-Pagan | Heteroskedasticity | Use robust or clustered standard errors |
| Wooldridge | Serial correlation | Use clustered standard errors or HAC |
| Pesaran CD | Cross-sectional dependence | Use Driscoll-Kraay standard errors |
| RESET | Functional form misspecification | Add nonlinear terms or interactions |

## Comparison with Other Software

| Task | PanelBox | Stata | R |
|------|----------|-------|---|
| Run all diagnostics | `exp.validate_model('fe')` | Multiple `estat` commands | Multiple function calls |
| Pass/fail summary | `val.pass_rate` | Manual inspection of logs | Custom code |
| HTML report | `val.save_html(...)` | Not built-in | `rmarkdown` (manual) |
| JSON export | `val.save_json(...)` | Not built-in | `jsonlite::toJSON()` |

## See Also

- [Experiment Overview](index.md) -- Pattern overview and quick start
- [Workflow](fitting.md) -- Fitting and managing models
- [Comparison Reports](comparison.md) -- Side-by-side model comparison
- [Residual Analysis](residuals.md) -- Residual diagnostics
- [Master Reports](master-reports.md) -- Combined report generation
