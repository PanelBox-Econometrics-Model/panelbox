---
title: "AutoExperiment API"
description: "API reference for panelbox.autoexperiment — automated panel data model selection pipeline"
---

# AutoExperiment API Reference

!!! info "Module"
    **Import**: `from panelbox.autoexperiment import AutoExperiment, AutoExperimentResults, VariableTransformer, VariableSelector, ModelRanker, AutoCovTypeSelector, SpecificationFilter`
    **Source**: `panelbox/autoexperiment/`

## Overview

AutoExperiment automates the full panel data modeling pipeline: variable transformation, forward stepwise selection, multi-model estimation, econometric validation, and composite ranking. It is designed for researchers who want a systematic, reproducible approach to model selection while respecting econometric best practices.

| Class | Description |
|-------|-------------|
| `AutoExperiment` | Main orchestrator — runs the full pipeline |
| `AutoExperimentResults` | Container for results, ranking, and report generation |
| `VariableTransformer` | Generates lags, diffs, logs, growth rates, and more |
| `VariableSelector` | Forward stepwise variable selection by BIC/AIC |
| `ModelRanker` | Composite scoring and ranking of candidate models |
| `AutoCovTypeSelector` | Automatic standard error selection based on diagnostics |
| `SpecificationFilter` | Classifies specifications as VALID / WARNING / INVALID |

---

## AutoExperiment

The main class that orchestrates the entire pipeline.

```python
from panelbox.autoexperiment import AutoExperiment

auto = AutoExperiment(
    data=df,
    depvar="invest",
    entity_col="firm",
    time_col="year",
    candidates=["value", "capital"],
    transformations={"lag": [1, 2], "diff": True, "log": True},
    models=["pooled_ols", "fe", "re", "fd"],
    criterion="bic",
    max_vars=8,
    sign_constraints={"value": "+", "capital": "+"},
)
results = auto.run()
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data in long format |
| `depvar` | `str` | *required* | Dependent variable name |
| `entity_col` | `str` | *required* | Cross-sectional identifier column |
| `time_col` | `str` | *required* | Time identifier column |
| `candidates` | `list[str]` | `None` | Candidate regressors. If `None`, uses all numeric columns except `depvar`, `entity_col`, `time_col` |
| `transformations` | `dict` | `{'lag': [1], 'diff': True}` | Transformation specification (see [VariableTransformer](#variabletransformer)) |
| `models` | `list[str]` | `['pooled_ols', 'fe', 're', 'fd']` | Model types to estimate |
| `criterion` | `str` | `'bic'` | Information criterion: `'bic'` or `'aic'` |
| `max_vars` | `int` | `8` | Maximum regressors per model |
| `min_obs_per_var` | `int` | `10` | Minimum observations per variable (caps `max_vars` if needed) |
| `prefilter_corr` | `float` | `0.05` | Minimum absolute correlation with `depvar` to keep a candidate |
| `require_tests` | `bool` | `True` | If `True`, only VALID/WARNING models enter the ranking |
| `alpha` | `float` | `0.05` | Significance level for all diagnostic tests |
| `required_tests` | `list[str]` | `None` | Override the default per-model test list in `SpecificationFilter` |
| `sign_constraints` | `dict[str, str]` | `None` | `{variable: '+' or '-'}` — economic theory sign constraints |
| `max_combinations` | `int` | `500` | Budget cap for total model evaluations |
| `n_jobs` | `int` | `1` | Reserved for future parallel support |
| `verbose` | `int` | `1` | `0` = silent, `1` = progress bars, `2` = detailed logging |
| `random_state` | `int` | `None` | Random seed (reserved for future use) |

### Valid Model Types

| Alias | Model |
|-------|-------|
| `'pooled_ols'` | Pooled OLS |
| `'fe'` | Fixed Effects (within estimator) |
| `'re'` | Random Effects (GLS) |
| `'fd'` | First Difference |

### `run()`

```python
auto.run() -> AutoExperimentResults
```

Executes the full six-phase pipeline:

1. **Variable transformations** — generates lags, diffs, logs, etc.
2. **Forward stepwise selection** — per model type, using BIC/AIC
3. **Estimation & validation** — fits each model, runs diagnostic tests
4. **Hausman test** — if both FE and RE are present, marks the loser as WARNING
5. **Ranking** — composite score combining criterion + tests + signs + parsimony
6. **Results assembly** — packages everything into `AutoExperimentResults`

**Returns**: [`AutoExperimentResults`](#autoexperimentresults)

---

## AutoExperimentResults

Container for all outputs produced by `AutoExperiment.run()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_model` | `PanelResults` or `None` | Fitted result object for the best model |
| `best_formula` | `str` | Formula of the best model |
| `best_estimator` | `str` | Canonical model type (`'fe'`, `'re'`, etc.) |
| `best_cov_type` | `str` | Standard error type auto-selected for the best model |
| `ranking` | `pd.DataFrame` | Model ranking table sorted by composite score |
| `all_results` | `dict` | `{model_name: {'results', 'metrics', 'classification', ...}}` |
| `transformations_used` | `dict` | `{transformed_var: (source_var, transformation_type)}` |
| `variable_selection` | `dict` | `{model_type: {'selected_vars', 'rejected_vars', 'bic_path'}}` |
| `test_results` | `dict` | `{model_name: {'classification', 'passed', 'failed'}}` |
| `n_combinations_tested` | `int` | Total number of model/variable combinations evaluated |
| `datamining_warning` | `bool` | `True` if > 100 combinations were tested |

### Methods

#### `summary()`

```python
results.summary() -> str
```

Returns a human-readable text summary with the best model, formula, BIC, R-squared, classification status, and data mining warning if applicable.

#### `compare_top(n=5)`

```python
results.compare_top(n=5) -> pd.DataFrame
```

Returns the top *n* models with columns: `model_type`, `formula`, `bic`, `aic`, `rsq`, `rsq_adj`, `classification`, `n_vars`, `cov_type`, `score`.

#### `plot_bic_comparison()`

```python
results.plot_bic_comparison() -> matplotlib.figure.Figure
```

Horizontal bar chart of BIC by model, color-coded by classification (green=VALID, yellow=WARNING, red=INVALID).

#### `plot_variable_importance()`

```python
results.plot_variable_importance() -> matplotlib.figure.Figure
```

Bar chart showing how many times each variable was selected across model types.

#### `report(filepath=None)`

```python
results.report(filepath="report.html") -> str
```

Generates a comprehensive HTML report with:

- Summary box (best model, formula, cov type)
- Data mining warning (if applicable)
- Model ranking table
- Variable selection details per model
- Diagnostic test results

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | `None` | If provided, saves the HTML to this path |

**Returns**: HTML content as string.

---

## VariableTransformer

Generates transformed variables respecting the panel structure (within-entity operations).

```python
from panelbox.autoexperiment import VariableTransformer

transformer = VariableTransformer(
    data=df,
    entity_col="firm",
    time_col="year",
    transformations={"lag": [1, 2], "diff": True, "log": True, "growth": True, "sq": True},
    nan_threshold=0.30,
)
transformed_data = transformer.transform(["value", "capital"])
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data in long format |
| `entity_col` | `str` | *required* | Entity identifier column |
| `time_col` | `str` | *required* | Time identifier column |
| `transformations` | `dict` | `{'lag': [1], 'diff': True}` | Transformation specification (see below) |
| `nan_threshold` | `float` | `0.30` | Maximum fraction of NaN allowed per transformed variable (discarded if exceeded) |

### Supported Transformations

| Key | Value | Naming | Example |
|-----|-------|--------|---------|
| `'lag'` | `list[int]` | `L{k}_var` | `L1_value`, `L2_value` |
| `'diff'` | `bool` | `D_var` | `D_value` |
| `'log'` | `bool` | `log_var` | `log_value` |
| `'acum'` | `list[int]` | `acum{k}_var` | `acum3_value`, `acum6_value` |
| `'growth'` | `bool` | `growth_var` | `growth_value` |
| `'sq'` | `bool` | `sq_var` | `sq_value` |

!!! note "Data Quality"
    - Variables with x <= 0 are skipped for `log` transformation.
    - Highly correlated transformed variables (> 0.95) are automatically removed.
    - Variables exceeding `nan_threshold` are discarded.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `transform(variables)` | `pd.DataFrame` | Generates all transformations, returns data with new columns |
| `get_transformation_map()` | `dict` | `{transformed_var: (source_var, type)}` |
| `get_valid_transformations()` | `list[str]` | Names of non-discarded transformed columns |
| `summary()` | `str` | Summary of generated, valid, and discarded transformations |

---

## VariableSelector

Forward stepwise variable selection using BIC or AIC, with optional sign constraints.

```python
from panelbox.autoexperiment import VariableSelector

selector = VariableSelector(
    depvar="invest",
    candidates=["value", "capital", "L1_value", "D_capital"],
    entity_col="firm",
    time_col="year",
    criterion="bic",
    max_vars=5,
    sign_constraints={"value": "+", "capital": "+"},
)
result = selector.forward_stepwise(data=df, model_class="fe")
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depvar` | `str` | *required* | Dependent variable name |
| `candidates` | `list[str]` | *required* | Candidate regressor names |
| `entity_col` | `str` | *required* | Entity identifier column |
| `time_col` | `str` | *required* | Time identifier column |
| `criterion` | `str` | `'bic'` | `'bic'` or `'aic'` |
| `max_vars` | `int` | `8` | Maximum variables to select |
| `min_obs_per_var` | `int` | `10` | Minimum observations per variable |
| `prefilter_corr` | `float` | `0.05` | Minimum absolute correlation with depvar for pre-filtering |
| `sign_constraints` | `dict[str, str]` | `None` | `{variable: '+' or '-'}` |

### Methods

#### `forward_stepwise(data, model_class)`

Runs the forward stepwise algorithm. At each step, the candidate that most improves the criterion is added, subject to sign constraints.

**Returns**: `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `selected_vars` | `list[str]` | Variables selected in order |
| `bic_path` | `list[float]` | Criterion value at each step |
| `rejected_vars` | `dict` | `{var: reason}` — why each rejected variable was excluded |
| `n_steps` | `int` | Number of steps taken |

!!! tip "Sign Constraints on Transformed Variables"
    Sign constraints defined on source variables are automatically inherited by their transformations. For example, `{'value': '+'}` also constrains `L1_value`, `log_value`, and `sq_value`. Differenced/growth transforms (`D_value`, `growth_value`) may invert the expected sign.

---

## ModelRanker

Ranks models by a weighted composite score combining four components.

```python
from panelbox.autoexperiment import ModelRanker

ranker = ModelRanker(weights={
    "criterion": 0.50,
    "tests": 0.30,
    "signs": 0.15,
    "parsimony": 0.05,
})
ranking_df = ranker.rank(model_evaluations)
```

### Composite Score Formula

The score for each model is:

$$
\text{score} = w_c \cdot S_{\text{criterion}} + w_t \cdot S_{\text{tests}} + w_s \cdot S_{\text{signs}} + w_p \cdot S_{\text{parsimony}}
$$

| Component | Weight (default) | Computation |
|-----------|-----------------|-------------|
| **Criterion** | 50% | Min-max normalized BIC (lower BIC = higher score) |
| **Tests** | 30% | Fraction of diagnostic tests passed |
| **Signs** | 15% | Fraction of sign constraints satisfied |
| **Parsimony** | 5% | `1 - (n_vars / max_vars)` |

INVALID models are always ranked last, regardless of score.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `dict[str, float]` | `{'criterion': 0.50, 'tests': 0.30, 'signs': 0.15, 'parsimony': 0.05}` | Weights are normalized to sum to 1.0 |

### `rank(model_evaluations)`

```python
ranker.rank(model_evaluations: list[dict]) -> pd.DataFrame
```

**Returns**: DataFrame with columns: `rank`, `model_type`, `formula`, `score`, `bic`, `aic`, `rsq_adj`, `classification`, `n_vars`, `cov_type`, `tests_passed`, `signs_ok`.

---

## AutoCovTypeSelector

Selects the most appropriate standard error type based on diagnostic test results.

### Decision Hierarchy

| Priority | Condition | Selected `cov_type` |
|----------|-----------|---------------------|
| 1 | Pesaran CD rejects (cross-sectional dependence) | `'driscoll_kraay'` |
| 2 | Wooldridge/Breusch-Godfrey rejects (serial correlation) | `'newey_west'` |
| 3 | Modified Wald/Breusch-Pagan/White rejects (heteroskedasticity) | `'robust'` |
| 4 | Nothing rejects, N <= 30 | `'clustered'` |
| 4 | Nothing rejects, N > 30 | `'nonrobust'` |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `select(validation_report, n_entities)` | `str` | Returns the recommended `cov_type` |
| `explain(validation_report, n_entities)` | `str` | Human-readable justification for the choice |

---

## SpecificationFilter

Classifies panel model specifications based on diagnostic test results.

### Default Tests per Model Type

| Model | Tests |
|-------|-------|
| `pooled_ols` | Breusch-Pagan, RESET |
| `fe` | Pesaran CD, Wooldridge, Modified Wald, RESET |
| `re` | Mundlak, Breusch-Pagan, RESET |
| `fd` | Breusch-Pagan, RESET |

### Critical Tests

Failing a **critical test** makes the specification INVALID:

- **RESET** — functional form misspecification
- **Mundlak** — correlated random effects (RE is inconsistent)

### Classification Logic

| Classification | Condition |
|----------------|-----------|
| **VALID** | All required tests pass |
| **WARNING** | Some non-critical tests fail |
| **INVALID** | At least one critical test fails |

### Methods

#### `classify(model_type, validation_report, alpha=None)`

```python
filter.classify("fe", validation_report) -> dict
```

**Returns**: `dict` with `classification`, `passed_tests`, `failed_tests`, `critical_failures`, `auto_cov_type`.

#### `hausman_decision(data, formula, entity_col, time_col, alpha=None)`

```python
filter.hausman_decision(df, "y ~ x1 + x2", "firm", "year") -> str
```

Runs the Hausman test. Returns `'fe'` if the null is rejected (use FE), `'re'` otherwise.

---

## Complete Example

```python
from panelbox.datasets import load_grunfeld
from panelbox.autoexperiment import AutoExperiment

data = load_grunfeld()

auto = AutoExperiment(
    data=data,
    depvar="invest",
    entity_col="firm",
    time_col="year",
    candidates=["value", "capital"],
    transformations={"lag": [1, 2], "diff": True, "log": True},
    models=["pooled_ols", "fe", "re", "fd"],
    criterion="bic",
    max_vars=6,
    sign_constraints={"value": "+", "capital": "+"},
    verbose=1,
)

results = auto.run()

# Summary
print(results.summary())

# Top 3 models
print(results.compare_top(3))

# Best model coefficients
print(results.best_model.summary())

# HTML report
results.report("autoexperiment_report.html")

# Plots
fig1 = results.plot_bic_comparison()
fig2 = results.plot_variable_importance()
```

---

## See Also

- [AutoExperiment User Guide](../user-guide/autoexperiment.md) — when to use, limitations, interpreting results
- [AutoExperiment Quickstart](../tutorials/autoexperiment-quickstart.md) — from zero to results in 10 lines
- [Static Models API](static-models.md) — underlying estimators
- [Validation API](validation.md) — diagnostic tests used by AutoExperiment
- [Experiment API](experiment.md) — manual model comparison with `PanelExperiment`
