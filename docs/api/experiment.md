---
title: "Experiment API"
description: "API reference for panelbox.experiment — PanelExperiment for model fitting, validation, comparison, and report generation"
---

# Experiment API Reference

!!! info "Module"
    **Import**: `from panelbox import PanelExperiment`
    **Source**: `panelbox/experiment/`

## Overview

`PanelExperiment` provides a high-level workflow for panel data analysis: fit multiple models, run validation tests, compare models, analyze residuals, and generate HTML reports — all from a single entry point.

| Class / Function | Description |
|-----------------|-------------|
| `PanelExperiment` | High-level workflow for fitting, validating, comparing models, and reporting |
| `ValidationResult` | Container for validation test results |
| `ComparisonResult` | Container for model comparison results |
| `ResidualResult` | Container for residual analysis results |

```python
import panelbox as pb

experiment = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year",
)

experiment.fit_model("fixed_effects", name="fe")
experiment.fit_model("random_effects", name="re")

comparison = experiment.compare_models(["fe", "re"])
experiment.save_master_report("analysis.html", theme="professional")
```

---

## PanelExperiment

### Constructor

```python
class PanelExperiment(
    data: pd.DataFrame,
    formula: str,
    entity_col: str | None = None,
    time_col: str | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | — | Panel dataset |
| `formula` | `str` | — | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `entity_col` | `str \| None` | `None` | Entity identifier column |
| `time_col` | `str \| None` | `None` | Time identifier column |

---

### Model Fitting

#### `fit_model()`

Fit a panel model and store it in the experiment.

```python
def fit_model(
    self,
    model_type: str,
    name: str | None = None,
    **kwargs,
) -> Any
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `str` | — | Model type (see table below) |
| `name` | `str \| None` | `None` | Custom name (auto-generated if None) |
| `**kwargs` | — | — | Additional arguments passed to the model constructor |

**Supported Model Types:**

| Alias | Model | Class |
|-------|-------|-------|
| `pooled`, `pooled_ols` | Pooled OLS | `PooledOLS` |
| `fe`, `fixed_effects` | Fixed Effects | `FixedEffects` |
| `re`, `random_effects` | Random Effects | `RandomEffects` |
| `pooled_logit` | Pooled Logit | `PooledLogit` |
| `pooled_probit` | Pooled Probit | `PooledProbit` |
| `fe_logit`, `fixed_effects_logit` | FE Logit | `FixedEffectsLogit` |
| `re_probit`, `random_effects_probit` | RE Probit | `RandomEffectsProbit` |
| `pooled_poisson`, `poisson` | Pooled Poisson | `PooledPoisson` |
| `fe_poisson`, `poisson_fe`, `poisson_fixed_effects` | FE Poisson | `PoissonFixedEffects` |
| `re_poisson`, `random_effects_poisson` | RE Poisson | `RandomEffectsPoisson` |
| `negbin`, `negative_binomial` | Negative Binomial | `NegativeBinomial` |
| `tobit`, `re_tobit`, `random_effects_tobit` | RE Tobit | `RandomEffectsTobit` |
| `ologit`, `ordered_logit` | Ordered Logit | `OrderedLogit` |
| `oprobit`, `ordered_probit` | Ordered Probit | `OrderedProbit` |

**Returns:** Model result object

#### `list_models()`

```python
def list_models(self) -> list[str]
```

Returns names of all fitted models.

#### `get_model()`

```python
def get_model(self, name: str) -> Any
```

Retrieve a fitted model's results by name.

#### `get_model_metadata()`

```python
def get_model_metadata(self, name: str) -> dict[str, Any]
```

Get metadata (model type, fit time, etc.) for a fitted model.

---

### Validation

#### `validate_model()`

Run validation tests on a fitted model.

```python
def validate_model(
    self,
    name: str,
    config: str = "basic",
    tests: list[str] | None = None,
) -> ValidationResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Name of fitted model |
| `config` | `str` | `"basic"` | Preset: `"quick"`, `"basic"`, `"full"` |
| `tests` | `list[str] \| None` | `None` | Custom test list (overrides config) |

**Preset Configurations:**

| Config | Tests |
|--------|-------|
| `quick` | Heteroskedasticity, autocorrelation |
| `basic` | + normality test |
| `full` | + Hausman test |

---

### Model Comparison

#### `compare_models()`

Compare multiple fitted models side-by-side.

```python
def compare_models(
    self,
    model_names: list[str],
    include_coefficients: bool = True,
    include_statistics: bool = True,
) -> ComparisonResult
```

---

### Residual Analysis

#### `analyze_residuals()`

Run residual diagnostics on a fitted model.

```python
def analyze_residuals(self, name: str) -> ResidualResult
```

---

### Report Generation

#### `save_master_report()`

Generate a master HTML report with navigation to all sub-reports.

```python
def save_master_report(
    self,
    file_path: str,
    theme: str = "professional",
    title: str | None = None,
    reports: list[dict] | None = None,
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | — | Output file path |
| `theme` | `str` | `"professional"` | Theme: `"professional"`, `"academic"`, `"presentation"` |
| `title` | `str \| None` | `None` | Custom report title |
| `reports` | `list[dict] \| None` | `None` | Sub-report configurations |

**Example:**

```python
experiment.save_master_report("master.html", theme="professional", reports=[
    {"type": "validation", "title": "Validation", "description": "Tests", "file_path": "val.html"},
    {"type": "comparison", "title": "Comparison", "description": "Models", "file_path": "comp.html"},
    {"type": "residuals", "title": "Residuals", "description": "Diagnostics", "file_path": "res.html"},
])
```

---

## Result Containers

### `BaseResult`

Base class for all experiment result containers.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Serialize to dictionary |
| `summary()` | `str` | Text summary |
| `save_html(file_path, test_type, theme="professional")` | — | Generate HTML report |
| `save_json(file_path)` | — | Export to JSON |

### `ValidationResult`

Container for validation test results.

```python
class ValidationResult(BaseResult)
```

**Class Method:**

```python
@classmethod
def from_model_results(cls, results) -> ValidationResult
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `total_tests` | `int` | Number of tests run |
| `failed_tests` | `int` | Number of tests that rejected H0 |
| `passed_tests` | `int` | Number of tests that did not reject H0 |
| `pass_rate` | `float` | Fraction of tests passed |

**Example:**

```python
validation = experiment.validate_model("fe", config="full")
print(f"Pass rate: {validation.pass_rate:.1%}")
print(validation.summary())
validation.save_html("validation.html", test_type="validation", theme="academic")
```

### `ComparisonResult`

Container for model comparison results.

```python
class ComparisonResult(BaseResult)
```

**Class Method:**

```python
@classmethod
def from_experiment(cls, experiment) -> ComparisonResult
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `best_model(criterion, prefer_lower=True)` | `str` | Name of best model by criterion |
| `get_comparison_table()` | `pd.DataFrame` | Comparison table |

**Example:**

```python
comparison = experiment.compare_models(["ols", "fe", "re"])
best = comparison.best_model("aic", prefer_lower=True)
print(f"Best model by AIC: {best}")
comparison.save_html("comparison.html", test_type="comparison")
```

### `ResidualResult`

Container for residual diagnostics.

```python
class ResidualResult(BaseResult)
```

**Class Method:**

```python
@classmethod
def from_model_results(cls, results) -> ResidualResult
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `shapiro_test` | `dict` | Shapiro-Wilk normality test |
| `jarque_bera` | `dict` | Jarque-Bera normality test |
| `durbin_watson` | `float` | Durbin-Watson statistic |
| `ljung_box` | `dict` | Ljung-Box serial correlation test |

---

## Complete Workflow Example

```python
import panelbox as pb
from panelbox.datasets import load_grunfeld

# Load data
data = load_grunfeld()

# Create experiment
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year",
)

# Fit multiple models
exp.fit_model("pooled_ols", name="pooled")
exp.fit_model("fixed_effects", name="fe")
exp.fit_model("random_effects", name="re")

# Validate
val = exp.validate_model("fe", config="full")
print(val.summary())

# Compare
comp = exp.compare_models(["pooled", "fe", "re"])
print(f"Best model: {comp.best_model('aic')}")

# Residual diagnostics
resid = exp.analyze_residuals("fe")
print(f"Durbin-Watson: {resid.durbin_watson:.4f}")

# Generate master report
exp.save_master_report("analysis.html", theme="professional")
```

---

## See Also

- [Report API](report.md) — low-level report generation
- [Visualization API](visualization.md) — chart creation
- [Validation API](validation.md) — all diagnostic tests
- [Tutorials: Production](../tutorials/production.md) — production workflow guide
