---
title: "Validation API"
description: "API reference for panelbox.validation — specification, serial correlation, heteroskedasticity, cross-sectional dependence, and unit root tests"
---

# Validation API Reference

!!! info "Module"
    **Import**: `from panelbox.validation import ...`
    **Source**: `panelbox/validation/`

## Overview

PanelBox provides a unified validation framework with 20+ diagnostic tests organized into six categories. All tests follow a consistent interface: instantiate with model results, call `.run()`, and receive a `ValidationTestResult`.

| Category | Tests | Purpose |
|----------|-------|---------|
| **Specification** | Hausman, Mundlak, RESET, Chow | Model specification and structural breaks |
| **Serial Correlation** | Wooldridge AR, Breusch-Godfrey, Baltagi-Wu | Autocorrelation in residuals |
| **Heteroskedasticity** | Modified Wald, Breusch-Pagan, White | Non-constant variance |
| **Cross-Sectional Dependence** | Pesaran CD, Breusch-Pagan LM, Frees | Correlation across entities |
| **Unit Root** | LLC, IPS, Fisher | Stationarity testing |
| **Cointegration** | Kao, Pedroni | Long-run equilibrium relationships |

---

## Base Classes

### `ValidationTest`

Abstract base class for all validation tests.

```python
class ValidationTest(results: PanelResults)
```

**Attributes:** `results`, `resid`, `fittedvalues`, `params`, `nobs`, `n_entities`, `n_periods`, `model_type`

**Abstract Method:**

```python
def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult
```

### `ValidationTestResult`

Standard result container returned by all validation tests.

```python
class ValidationTestResult:
    def __init__(
        self,
        test_name: str,
        statistic: float,
        pvalue: float,
        null_hypothesis: str,
        alternative_hypothesis: str,
        alpha: float = 0.05,
        df: Any = None,
        metadata: dict[str, Any] | None = None,
    )
```

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `test_name` | `str` | Name of the test |
| `statistic` | `float` | Test statistic value |
| `pvalue` | `float` | p-value |
| `reject_null` | `bool` | Whether to reject H0 at `alpha` |
| `conclusion` | `str` | Human-readable conclusion |
| `df` | `Any` | Degrees of freedom |
| `metadata` | `dict` | Additional test-specific information |

**Methods:** `summary() -> str`

---

## Specification Tests

### `HausmanTest`

Hausman (1978) test for choosing between fixed effects and random effects.

```python
class HausmanTest(
    fe_results: PanelResults,
    re_results: PanelResults,
    alpha: float = 0.05,
)
```

!!! note
    Unlike other tests, `HausmanTest` takes **two** model results (FE and RE).

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(alpha=0.05)` | `HausmanTestResult` | Run the test |
| `summary()` | `str` | Formatted summary |

**Attributes (after run):** `statistic`, `pvalue`, `df`, `conclusion`, `recommendation`, `reject_null`

### `HausmanTestResult`

```python
class HausmanTestResult:
    statistic: float
    pvalue: float
    df: int
    fe_params: pd.Series
    re_params: pd.Series
    diff: pd.Series
    recommendation: str     # "fixed_effects" or "random_effects"
```

**Example:**

```python
from panelbox.validation import HausmanTest

hausman = HausmanTest(fe_results, re_results)
result = hausman.run()
print(result.recommendation)  # "fixed_effects" or "random_effects"
print(result.summary())
```

### `MundlakTest`

Mundlak (1978) test — augmented RE regression with group means as additional regressors.

```python
class MundlakTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

- **H0**: Random effects is consistent (group means are not significant)
- **H1**: Fixed effects is needed

### `RESETTest`

Ramsey (1969) RESET test for functional form misspecification.

```python
class RESETTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05, powers=[2, 3]) -> ValidationTestResult`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `powers` | `list[int]` | `[2, 3]` | Powers of fitted values to include |

- **H0**: Model is correctly specified
- **H1**: Functional form misspecification

### `ChowTest`

Chow (1960) test for structural breaks.

```python
class ChowTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05, break_point=None) -> ValidationTestResult`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `break_point` | `int \| float \| None` | `None` | Break point (auto-detected if None) |

- **H0**: No structural break
- **H1**: Structural break exists

---

## Serial Correlation Tests

### `WooldridgeARTest`

Wooldridge (2002) test for first-order autocorrelation in FE/RE panel models.

```python
class WooldridgeARTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

- **H0**: No first-order autocorrelation
- **H1**: AR(1) autocorrelation present

### `BreuschGodfreyTest`

Breusch-Godfrey (1978) LM test for higher-order serial correlation.

```python
class BreuschGodfreyTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05, lags=1) -> ValidationTestResult`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | `int` | `1` | Number of lagged residuals |

- **H0**: No serial correlation up to order `lags`

### `BaltagiWuTest`

Baltagi-Wu (1999) locally best invariant test for AR(1) errors.

```python
class BaltagiWuTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

- Returns `rho` (estimated autocorrelation) in `metadata`

---

## Heteroskedasticity Tests

### `ModifiedWaldTest`

Modified Wald test for groupwise heteroskedasticity in fixed effects models (Greene 2000).

```python
class ModifiedWaldTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

- **H0**: Homoskedasticity (equal variance across entities)
- Returns `variance_ratio` in `metadata`

### `BreuschPaganTest`

Breusch-Pagan (1979) LM test for heteroskedasticity.

```python
class BreuschPaganTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

- **H0**: Homoskedasticity

### `WhiteTest`

White (1980) test for heteroskedasticity (general form).

```python
class WhiteTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05, cross_terms=True) -> ValidationTestResult`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cross_terms` | `bool` | `True` | Include cross-product terms |

- **H0**: Homoskedasticity

---

## Cross-Sectional Dependence Tests

### `PesaranCDTest`

Pesaran (2004) CD test for cross-sectional dependence. Works for both N > T and T > N.

```python
class PesaranCDTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

- **H0**: Cross-sectional independence
- **H1**: Cross-sectional dependence exists

### `BreuschPaganLMTest`

Breusch-Pagan (1980) LM test for cross-sectional independence. Requires **T > N**.

```python
class BreuschPaganLMTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

!!! warning
    This test requires T > N. For panels with N > T, use `PesaranCDTest` instead.

### `FreesTest`

Frees (1995, 2004) test for cross-sectional dependence using Spearman rank correlation.

```python
class FreesTest(results: PanelResults)
```

**Run:** `test.run(alpha=0.05) -> ValidationTestResult`

---

## Unit Root Tests (Class-Based)

!!! tip
    These class-based tests are also available as functions in `panelbox.diagnostics`. See [Diagnostics API](diagnostics.md).

### `LLCTest`

Levin, Lin & Chu (2002) test assuming homogeneous autoregressive parameter.

```python
class LLCTest(
    data: pd.DataFrame,
    variable: str,
    entity_col: str,
    time_col: str,
    ...
)
```

**Returns:** `LLCTestResult` with `statistic`, `pvalue`, `lags`, `n_obs`, `n_entities`, `test_type`, `deterministics`, `conclusion`

- **H0**: All panels contain unit roots
- **H1**: All panels are stationary

### `IPSTest`

Im, Pesaran & Shin (2003) test allowing heterogeneous autoregressive parameters.

```python
class IPSTest(
    data: pd.DataFrame,
    variable: str,
    entity_col: str,
    time_col: str,
    ...
)
```

**Returns:** `IPSTestResult` with `statistic`, `t_bar`, `pvalue`, `individual_stats`, `conclusion`

- **H0**: All panels contain unit roots
- **H1**: Some panels are stationary

### `FisherTest`

Maddala & Wu (1999) Fisher-type test combining individual unit root p-values.

```python
class FisherTest(
    data: pd.DataFrame,
    variable: str,
    entity_col: str,
    time_col: str,
    ...
)
```

**Returns:** `FisherTestResult` with `statistic`, `pvalue`, `individual_pvalues`, `conclusion`

---

## Cointegration Tests (Class-Based)

### `KaoTest`

Kao (1999) residual-based cointegration test assuming homogeneous cointegrating vector.

```python
class KaoTest(
    data: pd.DataFrame,
    dependent: str,
    independents: list[str],
    ...
)
```

**Returns:** `KaoTestResult` with `statistic`, `pvalue`, `n_obs`, `n_entities`, `trend`, `conclusion`

- **H0**: No cointegration
- **H1**: Cointegration exists

### `PedroniTest`

Pedroni (1999, 2004) test with 7 statistics allowing heterogeneous cointegrating vectors.

```python
class PedroniTest(
    data: pd.DataFrame,
    dependent: str,
    independents: list[str],
    ...
)
```

**Returns:** `PedroniTestResult` with panel statistics (`panel_v`, `panel_rho`, `panel_pp`, `panel_adf`) and group statistics (`group_rho`, `group_pp`, `group_adf`), plus `pvalues` dict and `summary_conclusion`

---

## Robustness Analysis

### `PanelBootstrap`

Bootstrap inference for panel data with multiple resampling strategies.

```python
class PanelBootstrap(
    results: PanelResults | None = None,
    n_bootstrap: int = 1000,
    method: str = "pairs",        # "pairs", "wild", "block", "residual"
    block_size: int | None = None,
    random_state: int | None = None,
    show_progress: bool = True,
    parallel: bool = False,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run()` | `PanelBootstrap` | Run bootstrap (modifies self in-place) |
| `conf_int(alpha=0.05, method="percentile")` | `pd.DataFrame` | CI methods: `"percentile"`, `"basic"`, `"bca"`, `"studentized"` |
| `summary()` | `pd.DataFrame` | Summary with original, bootstrap SE, and bias |
| `plot_distribution(param=None)` | — | Plot bootstrap distributions |

**Attributes (after `run()`):** `bootstrap_estimates_`, `bootstrap_se_`, `bootstrap_t_stats_`, `n_failed_`

**Example:**

```python
from panelbox.validation import PanelBootstrap

boot = PanelBootstrap(results=fe_result, n_bootstrap=1000, method="wild")
boot.run()
print(boot.summary())
ci = boot.conf_int(alpha=0.05, method="bca")
print(ci)
```

### Additional Robustness Tools

| Class | Description |
|-------|-------------|
| `PanelJackknife` | Leave-one-out jackknife inference |
| `TimeSeriesCV` | Time-series cross-validation for panel models |
| `SensitivityAnalysis` | Sensitivity to specification changes |
| `OutlierDetector` | Identify influential observations |
| `InfluenceDiagnostics` | Cook's distance, DFBETAS, leverage |

---

## ValidationSuite

Orchestrator that runs multiple test categories with a single call.

```python
class ValidationSuite(results: PanelResults)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `run(tests="default", alpha=0.05, verbose=False)` | `ValidationReport` | Run all or selected tests |
| `run_specification_tests(alpha=0.05, verbose=False)` | `dict` | Specification tests only |
| `run_serial_correlation_tests(alpha=0.05, verbose=False)` | `dict` | Serial correlation tests only |
| `run_heteroskedasticity_tests(alpha=0.05, verbose=False)` | `dict` | Heteroskedasticity tests only |
| `run_cross_sectional_tests(alpha=0.05, verbose=False)` | `dict` | Cross-sectional dependence tests only |

**Test presets for `tests` parameter:** `"default"`, `"all"`, `"specification"`, `"serial"`, `"heteroskedasticity"`, `"cross_sectional"`, or a list of specific test names.

### `ValidationReport`

```python
class ValidationReport:
    def __init__(
        self,
        model_info: dict[str, Any],
        specification_tests: dict | None = None,
        serial_tests: dict | None = None,
        het_tests: dict | None = None,
        cd_tests: dict | None = None,
    )
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `summary(verbose=True, as_dataframe=False)` | `str \| pd.DataFrame` | Full test summary |
| `to_dict()` | `dict` | Serialize to dictionary |
| `get_failed_tests()` | `list[str]` | Names of tests that rejected H0 |

**Example:**

```python
from panelbox.validation import ValidationSuite

suite = ValidationSuite(fe_result)
report = suite.run(tests="all", alpha=0.05)
print(report.summary())
print(f"Failed tests: {report.get_failed_tests()}")

# Run only serial correlation tests
serial = suite.run_serial_correlation_tests()
for name, result in serial.items():
    print(f"{name}: p={result.pvalue:.4f} — {result.conclusion}")
```

---

## GMM-Specific Diagnostics

!!! info
    GMM diagnostic tests (Hansen J, Sargan, AR(1)/AR(2), Difference-in-Hansen) are computed automatically when fitting GMM models and are available as attributes on `GMMResults`.

```python
from panelbox.gmm import DifferenceGMM

model = DifferenceGMM(data, formula="n ~ w + k | L.n", ...)
results = model.fit()

# Automatically computed diagnostics
print(f"Hansen J: {results.hansen_j.statistic} (p={results.hansen_j.pvalue:.4f})")
print(f"AR(2): {results.ar2_test.pvalue:.4f}")  # Should NOT reject
```

| Test | Attribute | H0 | Decision |
|------|-----------|-----|----------|
| Hansen J | `results.hansen_j` | Instruments are valid | Should NOT reject |
| Sargan | `results.sargan` | Instruments are valid (not robust to het.) | Should NOT reject |
| AR(1) | `results.ar1_test` | No first-order correlation | Expected to reject |
| AR(2) | `results.ar2_test` | No second-order correlation | Should NOT reject |
| Diff-Hansen | `results.diff_hansen` | Level instruments valid (System GMM) | Should NOT reject |

See [GMM API](gmm.md) for full GMM diagnostics documentation.

---

## See Also

- [Diagnostics API](diagnostics.md) — function-based unit root, cointegration, spatial, and specification tests
- [Standard Errors API](standard-errors.md) — robust SE when heteroskedasticity is detected
- [Tutorials: Validation](../tutorials/validation.md) — practical validation workflow
