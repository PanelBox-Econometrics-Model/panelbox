---
title: "Diagnostics API"
description: "API reference for panelbox.diagnostics — unit root, cointegration, specification, spatial, and quantile diagnostics"
---

# Diagnostics API Reference

!!! info "Module"
    **Import**: `from panelbox.diagnostics.<submodule> import ...`
    **Source**: `panelbox/diagnostics/`

## Overview

The diagnostics module provides **function-based** diagnostic tests for panel data, organized into four categories. Many of these tests complement the class-based tests in `panelbox.validation`.

| Category | Functions | Purpose |
|----------|-----------|---------|
| **Unit Root** | `hadri_test`, `breitung_test`, `panel_unit_root_test` | Test stationarity |
| **Cointegration** | `kao_test`, `pedroni_test`, `westerlund_test` | Long-run relationships |
| **Specification** | `j_test`, `cox_test`, `wald_encompassing_test`, `likelihood_ratio_test` | Non-nested model comparison |
| **Spatial** | `MoranIPanelTest`, `LocalMoranI`, `lm_lag_test`, `lm_error_test` | Spatial dependence |
| **Quantile** | `QuantileRegressionDiagnostics` | Quantile regression goodness-of-fit |

!!! tip "Diagnostics vs. Validation"
    Some tests exist in **both** modules with different interfaces:

    - `panelbox.diagnostics` — **function-based**, takes raw data (DataFrames)
    - `panelbox.validation` — **class-based**, takes fitted model results

    Use whichever interface fits your workflow.

---

## Unit Root Tests

### `hadri_test()`

Hadri (2000) LM test for stationarity. Unlike most unit root tests, the null hypothesis is **stationarity**.

```python
def hadri_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = "entity",
    time_col: str = "time",
    trend: str = "c",        # "c" (constant) or "ct" (constant + trend)
    robust: bool = True,
    alpha: float = 0.05,
) -> HadriResult
```

**Returns:** `HadriResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `float` | Standardized LM statistic |
| `pvalue` | `float` | p-value (one-sided) |
| `reject` | `bool` | Whether to reject H0 |
| `lm_statistic` | `float` | Raw LM statistic |
| `individual_lm` | `np.ndarray` | Individual LM statistics per entity |
| `n_entities` | `int` | Number of entities |
| `n_time` | `int` | Number of time periods |
| `trend` | `str` | Trend specification used |
| `robust` | `bool` | Whether robust variant was used |

- **H0**: All panels are stationary
- **H1**: Some panels contain unit roots

### `breitung_test()`

Breitung (2000) unit root test, robust to heterogeneity in intercepts and trends.

```python
def breitung_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = "entity",
    time_col: str = "time",
    trend: str = "ct",
    alpha: float = 0.05,
) -> BreitungResult
```

**Returns:** `BreitungResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `float` | Standardized test statistic |
| `pvalue` | `float` | p-value |
| `reject` | `bool` | Whether to reject H0 |
| `raw_statistic` | `float` | Raw (unstandardized) statistic |
| `n_entities` | `int` | Number of entities |
| `n_time` | `int` | Number of time periods |
| `trend` | `str` | Trend specification |

- **H0**: All panels contain unit roots
- **H1**: All panels are stationary

### `panel_unit_root_test()`

Run multiple unit root tests at once and compare results.

```python
def panel_unit_root_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = "entity",
    time_col: str = "time",
    test: str | list = "all",
    trend: str = "c",
    alpha: float = 0.05,
    **kwargs,
) -> PanelUnitRootResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test` | `str \| list` | `"all"` | Test(s) to run: `"hadri"`, `"breitung"`, `"llc"`, `"ips"`, `"fisher"`, `"all"` |

**Returns:** `PanelUnitRootResult` with `results` dict, `variable`, `n_entities`, `n_time`, `tests_run`

**Example:**

```python
from panelbox.diagnostics.unit_root import panel_unit_root_test, hadri_test

# Run all unit root tests
result = panel_unit_root_test(data, variable="gdp", entity_col="country", time_col="year")
for test_name, test_result in result.results.items():
    print(f"{test_name}: stat={test_result.statistic:.4f}, p={test_result.pvalue:.4f}")

# Run individual test
hadri = hadri_test(data, variable="gdp", entity_col="country", time_col="year", robust=True)
print(hadri)
```

### Interpreting Unit Root Test Results

| Unit Root Tests (LLC, IPS, etc.) | Hadri Test | Conclusion |
|----------------------------------|------------|------------|
| Reject H0 | Don't reject H0 | **Stationary** |
| Don't reject H0 | Reject H0 | **Unit Root** |
| Mixed results | Mixed results | **Inconclusive** — try different trend specifications |

---

## Cointegration Tests

### `kao_test()`

Kao (1999) residual-based cointegration test. Assumes homogeneous cointegrating vectors across entities.

```python
def kao_test(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: str | list[str],
    method: str = "adf",       # "df", "adf", "all"
    trend: str = "c",
    lags: int = 1,
) -> KaoResult
```

**Returns:** `KaoResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `dict[str, float]` | Test statistics by method |
| `pvalue` | `dict[str, float]` | p-values by method |
| `critical_values` | `dict` | Critical values at 1%, 5%, 10% |
| `method` | `str` | Method used |
| `trend` | `str` | Trend specification |
| `lags` | `int` | Number of lags |
| `n_entities` | `int` | Number of entities |
| `n_time` | `int` | Number of time periods |

- **H0**: No cointegration
- **H1**: Cointegration exists (homogeneous vector)

### `pedroni_test()`

Pedroni (1999, 2004) cointegration test with 7 statistics. Allows heterogeneous cointegrating vectors across entities.

```python
def pedroni_test(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: str | list[str],
    method: str = "all",
    trend: str = "c",
    lags: int = 4,
) -> PedroniResult
```

**Returns:** `PedroniResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `dict[str, float]` | All 7 test statistics |
| `pvalue` | `dict[str, float]` | p-values for each statistic |
| `critical_values` | `dict` | Critical values |

**Test Statistics:**

| Category | Statistic | Description |
|----------|-----------|-------------|
| Within-dimension (Panel) | `panel_v` | Variance ratio |
| | `panel_rho` | Phillips-Perron rho |
| | `panel_pp` | Phillips-Perron t |
| | `panel_adf` | Augmented Dickey-Fuller |
| Between-dimension (Group) | `group_rho` | Group Phillips-Perron rho |
| | `group_pp` | Group Phillips-Perron t |
| | `group_adf` | Group ADF |

!!! warning
    The `panel_v` statistic may over-reject in finite samples (T < 50). Rely more on PP and ADF statistics.

### `westerlund_test()`

Westerlund (2007) ECM-based cointegration test with bootstrap p-values.

```python
def westerlund_test(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: str | list[str],
    method: str = "all",           # "Gt", "Ga", "Pt", "Pa", "all"
    trend: str = "c",              # "n", "c", "ct"
    lags: int | str = "auto",
    max_lags: int = 4,
    lag_criterion: str = "aic",    # "aic" or "bic"
    n_bootstrap: int = 1000,
    random_state: int | None = None,
    use_bootstrap: bool = True,
) -> WesterlundResult
```

**Returns:** `WesterlundResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `dict[str, float]` | Test statistics (Gt, Ga, Pt, Pa) |
| `pvalue` | `dict[str, float]` | p-values (bootstrap or asymptotic) |
| `n_bootstrap` | `int` | Number of bootstrap replications |

**Example:**

```python
from panelbox.diagnostics.cointegration import kao_test, pedroni_test, westerlund_test

# Step 1: Test for cointegration using multiple tests
kao = kao_test(data, "country", "year", y_var="gdp", x_vars=["investment", "labor"])
pedroni = pedroni_test(data, "country", "year", y_var="gdp", x_vars=["investment", "labor"])
westerlund = westerlund_test(data, "country", "year", y_var="gdp", x_vars=["investment", "labor"])

# Step 2: Count rejections (majority rule)
print(f"Kao: {kao.pvalue}")
print(f"Pedroni: {pedroni.pvalue}")
print(f"Westerlund: {westerlund.pvalue}")
```

### Recommended Cointegration Workflow

1. **Pre-test**: Confirm variables are I(1) using unit root tests
2. **Run multiple tests**: Kao, Pedroni, and Westerlund
3. **Majority rule**: Count rejections across test families
4. **If cointegrated**: Estimate Panel VECM or DOLS

---

## Specification Tests

### `j_test()`

Davidson-MacKinnon J-test for comparing non-nested models.

```python
def j_test(
    result1,
    result2,
    direction: str = "both",    # "forward", "reverse", "both"
    model1_name: str | None = None,
    model2_name: str | None = None,
) -> JTestResult
```

**Returns:** `JTestResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `forward` | `dict \| None` | Test of model 1 against model 2 |
| `reverse` | `dict \| None` | Test of model 2 against model 1 |
| `model1_name` | `str` | Name of first model |
| `model2_name` | `str` | Name of second model |

**Interpretation:**

| Forward (H0: M1) | Reverse (H0: M2) | Conclusion |
|-------------------|-------------------|------------|
| Don't reject | Reject | Prefer Model 1 |
| Reject | Don't reject | Prefer Model 2 |
| Don't reject | Don't reject | Cannot distinguish |
| Reject | Reject | Neither model adequate |

### `cox_test()`

Cox (1961, 1962) test for non-nested model comparison.

```python
def cox_test(
    result1,
    result2,
    model1_name: str | None = None,
    model2_name: str | None = None,
) -> EncompassingResult
```

### `wald_encompassing_test()`

Wald encompassing test for nested or non-nested models.

```python
def wald_encompassing_test(
    result_restricted,
    result_unrestricted,
    model_restricted_name: str | None = None,
    model_unrestricted_name: str | None = None,
) -> EncompassingResult
```

### `likelihood_ratio_test()`

Likelihood ratio test for nested models.

```python
def likelihood_ratio_test(
    result_restricted,
    result_unrestricted,
    model_restricted_name: str | None = None,
    model_unrestricted_name: str | None = None,
) -> EncompassingResult
```

### `EncompassingResult`

```python
@dataclass
class EncompassingResult:
    test_name: str
    statistic: float
    pvalue: float
    df: float | None
    null_hypothesis: str
    alternative: str
    model1_name: str
    model2_name: str
    additional_info: dict[str, Any]
```

---

## Spatial Tests

### `MoranIPanelTest`

Moran's I test for spatial autocorrelation in panel residuals.

```python
class MoranIPanelTest(
    residuals: np.ndarray,
    W: np.ndarray,
    entity_ids,
    time_ids,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(method="pooled")` | `MoranIResult \| dict` | Run Moran's I test |

### `MoranIResult`

```python
@dataclass
class MoranIResult:
    statistic: float
    expected_value: float
    variance: float
    z_score: float
    pvalue: float
    conclusion: str
    additional_info: dict
```

### `LocalMoranI`

Local Indicators of Spatial Association (LISA).

```python
class LocalMoranI(
    values: np.ndarray,
    W: np.ndarray,
    entity_ids,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(permutations=999)` | `LISAResult` | Run local Moran's I |

### `LISAResult`

```python
@dataclass
class LISAResult:
    local_i: np.ndarray        # Local Moran's I for each observation
    pvalues: np.ndarray        # Pseudo p-values
    z_values: np.ndarray       # z-values
    Wz_values: np.ndarray      # Spatially lagged z-values
    entity_ids: np.ndarray     # Entity identifiers
```

### LM Tests for Spatial Dependence

Lagrange Multiplier tests to choose between spatial lag and spatial error models.

```python
def lm_lag_test(residuals, X, W, **kwargs) -> LMTestResult
def lm_error_test(residuals, X, W, **kwargs) -> LMTestResult
def robust_lm_lag_test(residuals, X, W, **kwargs) -> LMTestResult
def robust_lm_error_test(residuals, X, W, **kwargs) -> LMTestResult
def run_lm_tests(model_result, W, alpha=0.05) -> dict
```

### `LMTestResult`

```python
@dataclass
class LMTestResult:
    test_name: str
    statistic: float
    pvalue: float
    df: int
    conclusion: str
```

**Example:**

```python
from panelbox.diagnostics import run_lm_tests

# Run all spatial LM tests
lm_results = run_lm_tests(ols_result, W, alpha=0.05)
for name, result in lm_results.items():
    if isinstance(result, LMTestResult):
        print(f"{result.test_name}: stat={result.statistic:.4f}, p={result.pvalue:.4f}")
```

**Decision Rule:**

| LM Lag | LM Error | Robust LM Lag | Robust LM Error | Conclusion |
|--------|----------|---------------|------------------|------------|
| Significant | Not sig. | — | — | Spatial Lag model |
| Not sig. | Significant | — | — | Spatial Error model |
| Significant | Significant | Significant | Not sig. | Spatial Lag model |
| Significant | Significant | Not sig. | Significant | Spatial Error model |

---

## Quantile Regression Diagnostics

### `QuantileRegressionDiagnostics`

Diagnostic tests for quantile regression models.

```python
class QuantileRegressionDiagnostics(
    model,
    params: np.ndarray,
    tau: float = 0.5,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `pseudo_r2()` | `float` | Koenker-Machado pseudo-R² |
| `goodness_of_fit(n_bins=10)` | `dict` | Goodness-of-fit measures |
| `symmetry_test()` | `tuple[float, float]` | Test for conditional symmetry |
| `goodness_of_fit_test(n_bins=10)` | `tuple[float, float]` | Formal GoF test (stat, p-value) |

---

## See Also

- [Validation API](validation.md) — class-based diagnostic tests for fitted models
- [Spatial Models API](spatial.md) — spatial lag and spatial error models
- [Tutorials: Validation](../tutorials/validation.md) — practical diagnostic workflow
- [Theory: Cointegration](../diagnostics/cointegration/index.md) — theoretical background
