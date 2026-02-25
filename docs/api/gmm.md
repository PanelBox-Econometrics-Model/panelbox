---
title: "GMM API"
description: "API reference for panelbox.gmm — Difference GMM, System GMM, CUE-GMM, Bias-Corrected GMM"
---

# GMM API Reference

!!! info "Module"
    **Import**: `from panelbox.gmm import DifferenceGMM, SystemGMM, ContinuousUpdatedGMM, BiasCorrectedGMM`
    **Source**: `panelbox/gmm/`

## Overview

The GMM module implements dynamic panel data estimators using the Generalized Method of Moments:

| Estimator | Description | Reference |
|-----------|-------------|-----------|
| `DifferenceGMM` | Arellano-Bond first-difference GMM | Arellano & Bond (1991) |
| `SystemGMM` | Blundell-Bond system GMM (difference + level) | Blundell & Bond (1998) |
| `ContinuousUpdatedGMM` | Continuously-Updated Estimator | Hansen, Heaton & Yaron (1996) |
| `BiasCorrectedGMM` | Analytical bias correction | Kiviet (1995) |

All estimators support one-step and two-step estimation with Windmeijer (2005) finite-sample correction.

## Classes

### DifferenceGMM

Arellano-Bond (1991) Difference GMM estimator. Uses lagged levels as instruments for the first-differenced equation.

#### Constructor

```python
DifferenceGMM(
    data: pd.DataFrame,
    dep_var: str,
    lags: int | list[int],
    id_var: str = "id",
    time_var: str = "year",
    exog_vars: list[str] | None = None,
    endogenous_vars: list[str] | None = None,
    predetermined_vars: list[str] | None = None,
    time_dummies: bool = True,
    collapse: bool = False,
    two_step: bool = True,
    robust: bool = True,
    gmm_type: str = "two_step",
    gmm_max_lag: int | None = None,
    iv_max_lag: int = 0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data |
| `dep_var` | `str` | *required* | Dependent variable name |
| `lags` | `int \| list[int]` | *required* | Lag order(s) for the dependent variable |
| `id_var` | `str` | `"id"` | Entity identifier column |
| `time_var` | `str` | `"year"` | Time period column |
| `exog_vars` | `list[str] \| None` | `None` | Strictly exogenous regressors |
| `endogenous_vars` | `list[str] \| None` | `None` | Endogenous regressors (instrumented like dep_var) |
| `predetermined_vars` | `list[str] \| None` | `None` | Predetermined regressors (weakly exogenous) |
| `time_dummies` | `bool` | `True` | Include time dummy variables |
| `collapse` | `bool` | `False` | Collapse instrument matrix (reduce instrument count) |
| `two_step` | `bool` | `True` | Use two-step estimation with Windmeijer correction |
| `robust` | `bool` | `True` | Robust standard errors |
| `gmm_type` | `str` | `"two_step"` | GMM estimation type |
| `gmm_max_lag` | `int \| None` | `None` | Maximum lag for GMM-style instruments |
| `iv_max_lag` | `int` | `0` | Maximum lag for IV-style instruments |

#### Methods

##### `.fit()`

```python
def fit(self) -> GMMResults
```

Estimate the model and return `GMMResults`.

#### Example

```python
from panelbox import DifferenceGMM, load_abdata

data = load_abdata()
model = DifferenceGMM(
    data=data,
    dep_var="n",
    lags=[1, 2],
    exog_vars=["w", "k"],
    id_var="id",
    time_var="year",
    two_step=True,
    collapse=False,
    time_dummies=True,
)
results = model.fit()
print(results.summary())
```

---

### SystemGMM

Blundell-Bond (1998) System GMM estimator. Extends Difference GMM by adding the level equation with lagged differences as instruments. Generally more efficient than Difference GMM, especially when the dependent variable is persistent.

#### Constructor

```python
SystemGMM(
    data: pd.DataFrame,
    dep_var: str,
    lags: int | list[int],
    id_var: str = "id",
    time_var: str = "year",
    exog_vars: list[str] | None = None,
    endogenous_vars: list[str] | None = None,
    predetermined_vars: list[str] | None = None,
    time_dummies: bool = True,
    collapse: bool = False,
    two_step: bool = True,
    robust: bool = True,
    gmm_type: str = "two_step",
    level_instruments: dict | None = None,
    gmm_max_lag: int | None = None,
    iv_max_lag: int = 0,
)
```

All parameters are the same as `DifferenceGMM`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level_instruments` | `dict \| None` | `None` | Additional instruments for the level equation |

#### Example

```python
from panelbox import SystemGMM, load_abdata

data = load_abdata()
model = SystemGMM(
    data=data,
    dep_var="n",
    lags=[1],
    exog_vars=["w", "k"],
    id_var="id",
    time_var="year",
    two_step=True,
    collapse=True,
)
results = model.fit()
print(results.summary())
```

---

### ContinuousUpdatedGMM

Continuously-Updated Estimator (CUE). The weight matrix is updated at each iteration, making the objective function:

Q(beta) = g(beta)' W(beta)^{-1} g(beta)

CUE is more robust to weak instruments and misspecification than two-step GMM.

#### Constructor

```python
ContinuousUpdatedGMM(
    data: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    instruments: list[str],
    weighting: str = "hac",
    bandwidth: str | int = "auto",
    se_type: str = "analytical",
    n_bootstrap: int = 999,
    bootstrap_method: str = "residual",
    max_iter: int = 100,
    tol: float = 1e-6,
    regularize: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data |
| `dep_var` | `str` | *required* | Dependent variable |
| `exog_vars` | `list[str]` | *required* | Exogenous regressors |
| `instruments` | `list[str]` | *required* | Instrumental variables |
| `weighting` | `str` | `"hac"` | Weight matrix type: `"hac"`, `"cluster"`, `"homoskedastic"` |
| `bandwidth` | `str \| int` | `"auto"` | Bandwidth for HAC (auto uses Newey-West rule) |
| `se_type` | `str` | `"analytical"` | Standard error type: `"analytical"` or `"bootstrap"` |
| `n_bootstrap` | `int` | `999` | Number of bootstrap replications |
| `bootstrap_method` | `str` | `"residual"` | Bootstrap method |
| `max_iter` | `int` | `100` | Maximum CUE iterations |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `regularize` | `bool` | `True` | Add ridge regularization to singular weight matrices |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    start_params: np.ndarray | None = None,
    method: str = "L-BFGS-B",
    verbose: bool = False,
) -> GMMResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | `np.ndarray \| None` | `None` | Starting values (uses 2SLS if None) |
| `method` | `str` | `"L-BFGS-B"` | Optimization method |
| `verbose` | `bool` | `False` | Print convergence information |

#### Example

```python
from panelbox.gmm import ContinuousUpdatedGMM

cue = ContinuousUpdatedGMM(
    data=df,
    dep_var="y",
    exog_vars=["x1", "x2"],
    instruments=["z1", "z2", "z3"],
    weighting="hac",
    bandwidth="auto",
)
results = cue.fit()
print(results.summary())
```

---

### BiasCorrectedGMM

Analytical bias correction for dynamic panel models. Corrects the small-T bias in GMM estimators using the approach of Kiviet (1995).

#### Constructor

```python
BiasCorrectedGMM(
    data: pd.DataFrame,
    dep_var: str,
    lags: list[int],
    id_var: str = "id",
    time_var: str = "year",
    exog_vars: list[str] | None = None,
    bias_order: int = 1,
    min_n: int = 50,
    min_t: int = 10,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Panel data |
| `dep_var` | `str` | *required* | Dependent variable |
| `lags` | `list[int]` | *required* | Lag orders |
| `id_var` | `str` | `"id"` | Entity column |
| `time_var` | `str` | `"year"` | Time column |
| `exog_vars` | `list[str] \| None` | `None` | Exogenous regressors |
| `bias_order` | `int` | `1` | Order of bias correction (1 or 2) |
| `min_n` | `int` | `50` | Minimum number of entities |
| `min_t` | `int` | `10` | Minimum time periods |

#### Methods

##### `.fit()`

```python
def fit(
    self,
    time_dummies: bool = True,
    use_system_gmm: bool = False,
    verbose: bool = False,
) -> GMMResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_dummies` | `bool` | `True` | Include time dummies |
| `use_system_gmm` | `bool` | `False` | Use System GMM as initial estimator |
| `verbose` | `bool` | `False` | Print progress information |

---

## Result Classes

### GMMResults

Dataclass holding all GMM estimation output.

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `pd.Series` | Estimated coefficients |
| `std_errors` | `pd.Series` | Standard errors |
| `tvalues` | `pd.Series` | t-statistics |
| `pvalues` | `pd.Series` | p-values |
| `nobs` | `int` | Number of observations |
| `n_groups` | `int` | Number of entities |
| `n_instruments` | `int` | Number of instruments |
| `n_params` | `int` | Number of estimated parameters |
| `vcov` | `np.ndarray` | Variance-covariance matrix |
| `two_step` | `bool` | Whether two-step was used |
| `windmeijer_corrected` | `bool` | Whether Windmeijer correction was applied |
| `model_type` | `str` | `"difference"` or `"system"` |
| `converged` | `bool` | Convergence status |

#### Specification Tests

| Attribute | Type | Description |
|-----------|------|-------------|
| `hansen_j` | `TestResult` | Hansen J overidentification test (robust) |
| `sargan` | `TestResult` | Sargan test (not robust to heteroskedasticity) |
| `ar1_test` | `TestResult` | AR(1) test in differenced residuals (should reject) |
| `ar2_test` | `TestResult` | AR(2) test in differenced residuals (should NOT reject) |
| `diff_hansen` | `TestResult \| None` | Difference-in-Hansen for level instruments (System GMM) |

!!! warning "Interpreting AR tests"
    AR(1) should be **negative and significant** (expected by construction). AR(2) should be **insignificant** (p > 0.10) -- rejection indicates misspecification.

#### Methods

##### `.summary()`

Print formatted estimation results with coefficient table and diagnostic tests.

### TestResult

Named container for specification test results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `float` | Test statistic value |
| `pvalue` | `float` | p-value |
| `df` | `int` | Degrees of freedom |

---

## Diagnostic Classes

### GMMDiagnostics

Comprehensive diagnostic analysis for GMM estimation results.

```python
GMMDiagnostics(model, results)
```

Provides methods for analyzing instrument validity, overidentification, and model specification.

### GMMOverfitDiagnostic

Detects instrument proliferation and overfitting in GMM models.

```python
GMMOverfitDiagnostic(model, results: GMMResults)
```

!!! warning "Instrument proliferation"
    When `n_instruments > n_groups`, GMM estimates become unreliable. Use `collapse=True` or limit `gmm_max_lag` to reduce instrument count.

---

## Practical Guidance

### Choosing Between Estimators

| Scenario | Recommended Estimator |
|----------|----------------------|
| Moderate persistence (rho < 0.8) | `DifferenceGMM` |
| High persistence (rho close to 1) | `SystemGMM` |
| Weak instrument concerns | `ContinuousUpdatedGMM` |
| Large T, small N | `BiasCorrectedGMM` |

### Instrument Count Rule of Thumb

Keep instruments <= number of entities:

```python
# Reduce instruments with collapse
model = SystemGMM(
    data=data, dep_var="y", lags=[1],
    exog_vars=["x1"], collapse=True,  # Collapse instruments
    gmm_max_lag=3,                     # Limit lag depth
)
```

### Complete Diagnostic Workflow

```python
from panelbox import SystemGMM, load_abdata

data = load_abdata()
model = SystemGMM(
    data=data, dep_var="n", lags=[1],
    exog_vars=["w", "k"],
    id_var="id", time_var="year",
    two_step=True,
)
results = model.fit()

# Check specification tests
print(f"Hansen J: stat={results.hansen_j.statistic:.3f}, p={results.hansen_j.pvalue:.3f}")
print(f"AR(1):    stat={results.ar1_test.statistic:.3f}, p={results.ar1_test.pvalue:.3f}")
print(f"AR(2):    stat={results.ar2_test.statistic:.3f}, p={results.ar2_test.pvalue:.3f}")
print(f"Instruments: {results.n_instruments}, Groups: {results.n_groups}")

# Instruments should be <= groups
assert results.n_instruments <= results.n_groups, "Too many instruments!"
```

## References

- Arellano, M. & Bond, S. (1991). "Some Tests of Specification for Panel Data." *Review of Economic Studies*, 58(2), 277-297.
- Blundell, R. & Bond, S. (1998). "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115-143.
- Hansen, L., Heaton, J. & Yaron, A. (1996). "Finite-Sample Properties of Some Alternative GMM Estimators." *Journal of Business & Economic Statistics*, 14(3), 262-280.
- Kiviet, J. (1995). "On Bias, Inconsistency, and Efficiency of Various Estimators in Dynamic Panel Data Models." *Journal of Econometrics*, 68(1), 53-78.
- Roodman, D. (2009). "How to do xtabond2." *Stata Journal*, 9(1), 86-136.
- Windmeijer, F. (2005). "A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators." *Journal of Econometrics*, 126(1), 25-51.

## See Also

- [Core API](core.md) — `PanelResults` and `GMMResults` base classes
- [VAR API](var.md) — Panel VAR with GMM estimation
- [Tutorials: GMM](../tutorials/gmm.md) — Step-by-step GMM guide
- [Theory: GMM](../theory/gmm-theory.md) — Econometric theory and derivations
