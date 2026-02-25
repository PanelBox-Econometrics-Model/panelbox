---
title: "Diagnostics & Validation"
description: "Complete guide to diagnostic testing and model validation for panel data models in PanelBox."
---

# Diagnostics & Validation

Panel data models rest on assumptions about error structure, functional form, and variable selection. Diagnostic tests verify whether these assumptions hold in your data, guiding you toward valid inference and reliable conclusions.

## Why Diagnostics Matter

Econometric estimates are only as credible as the assumptions underlying the model. Violations lead to:

- **Biased coefficients** (omitted variables, wrong functional form)
- **Invalid standard errors** (heteroskedasticity, serial correlation, cross-sectional dependence)
- **Inconsistent estimates** (endogeneity, non-stationarity)

PanelBox provides **50+ diagnostic tests** organized into six categories, all returning consistent result objects for programmatic use.

## Diagnostic Workflow

A disciplined testing workflow proceeds from broad model choice to specific assumption checks:

```text
Step 1: Specification       Is the model correctly specified?
    |
Step 2: Serial Correlation  Are errors autocorrelated?
    |
Step 3: Heteroskedasticity  Is variance constant across entities?
    |
Step 4: Cross-sectional     Are entities correlated?
        Dependence
    |
Step 5: Stationarity        Are variables stationary?
    |
Step 6: Cointegration       Do non-stationary variables share
                            a long-run equilibrium?
```

### Step 1: Specification Tests

Determine whether the model is correctly specified before examining residual properties.

| Test | Question | When to Use |
|------|----------|-------------|
| [Hausman](specification/hausman.md) | Fixed Effects or Random Effects? | After estimating FE and RE |
| [Mundlak](specification/mundlak.md) | Are random effects correlated with regressors? | Alternative to Hausman |
| [RESET](specification/reset.md) | Is the functional form correct? | Any linear model |
| [Chow](specification/chow.md) | Do parameters change over time? | Suspected structural break |
| [J-Test](specification/j-test.md) | Which non-nested model is better? | Comparing alternative specifications |
| [Cox / Encompassing](specification/cox-encompassing.md) | Does one model encompass another? | Likelihood-based model comparison |

:material-arrow-right: [Specification Tests Overview](specification/index.md)

### Step 2: Serial Correlation Tests

Test whether errors within an entity are correlated across time periods.

| Test | Detects | Best For |
|------|---------|----------|
| Wooldridge AR | First-order autocorrelation | FE models |
| Breusch-Godfrey | Higher-order autocorrelation | Any model |
| Baltagi-Wu LBI | Autocorrelation in unbalanced panels | Unbalanced panels |

:material-arrow-right: [Serial Correlation Tests](serial-correlation/index.md)

### Step 3: Heteroskedasticity Tests

Test whether the error variance is constant across entities and time.

| Test | Detects | Best For |
|------|---------|----------|
| Modified Wald | Groupwise heteroskedasticity | FE models |
| Breusch-Pagan | Heteroskedasticity linked to regressors | Any model |
| White | General heteroskedasticity | Any model |

:material-arrow-right: [Heteroskedasticity Tests](heteroskedasticity/index.md)

### Step 4: Cross-Sectional Dependence Tests

Test whether residuals are correlated across entities at a given time period.

| Test | Approach | Best For |
|------|----------|----------|
| Pesaran CD | Average pairwise correlations | Large N panels |
| Breusch-Pagan LM | Sum of squared correlations | Small N panels |
| Frees | Non-parametric | Robust to non-normality |

:material-arrow-right: [Cross-Sectional Dependence Tests](cross-sectional/index.md)

### Step 5: Unit Root Tests

Test whether panel variables are stationary or contain unit roots.

| Test | H₀ | Approach |
|------|-----|----------|
| LLC | Common unit root | Pooled ADF |
| IPS | Individual unit roots | Average ADF |
| Fisher | Individual unit roots | Combines p-values |

:material-arrow-right: [Unit Root Tests](unit-root/index.md)

### Step 6: Cointegration Tests

For non-stationary variables, test whether a long-run equilibrium relationship exists.

| Test | Approach | Statistics |
|------|----------|------------|
| Pedroni | Residual-based | 7 statistics (panel + group) |
| Kao | Residual-based | ADF-based |
| Westerlund | Error-correction | 4 statistics with bootstrap |

:material-arrow-right: [Cointegration Tests](cointegration/index.md)

## ValidationSuite: One-Line Comprehensive Testing

The `ValidationSuite` runs all applicable tests on a model result in a single call:

```python
from panelbox.validation import ValidationSuite

suite = ValidationSuite(results)
report = suite.run(tests="all", alpha=0.05)
print(report)
```

### Selective Testing

Run specific test categories:

```python
# Run only specification tests
spec_results = suite.run_specification_tests(alpha=0.05)

# Run only serial correlation tests
serial_results = suite.run_serial_correlation_tests(alpha=0.05)

# Run only heteroskedasticity tests
het_results = suite.run_heteroskedasticity_tests(alpha=0.05)

# Run only cross-sectional dependence tests
cd_results = suite.run_cross_sectional_tests(alpha=0.05)
```

### Test Selection Options

| Option | Tests Run |
|--------|-----------|
| `"all"` | Specification + Serial + Heteroskedasticity + Cross-sectional |
| `"default"` | Recommended tests for the model type |
| `"serial"` | Serial correlation tests only |
| `"het"` | Heteroskedasticity tests only |
| `"cd"` | Cross-sectional dependence tests only |

!!! note "Default Tests by Model Type"
    - **Fixed Effects**: Serial correlation + Heteroskedasticity + Cross-sectional
    - **Random Effects**: Cross-sectional dependence
    - **Pooled OLS**: Heteroskedasticity + Cross-sectional

## Common Result Pattern

All diagnostic tests in PanelBox return objects with a consistent interface:

```python
from panelbox.validation.base import ValidationTestResult

# Every test result provides:
result.test_name            # str   -- Name of the test
result.statistic            # float -- Test statistic value
result.pvalue               # float -- P-value
result.df                   # int, tuple, or None -- Degrees of freedom
result.alpha                # float -- Significance level used
result.null_hypothesis      # str   -- H₀ description
result.alternative_hypothesis  # str -- H₁ description
result.reject_null          # bool  -- Whether to reject at alpha
result.conclusion           # str   -- Human-readable interpretation
result.metadata             # dict  -- Additional test-specific info

# Formatted output
print(result.summary())
```

## Interpreting Test Results

The interpretation logic is the same across all tests:

| Condition | Meaning | Action |
|-----------|---------|--------|
| p-value < $\alpha$ | Reject H₀ | Assumption violated -- take corrective action |
| p-value $\geq \alpha$ | Fail to reject H₀ | No evidence against assumption |

!!! warning "Common Misconception"
    Failing to reject H₀ does **not** prove the assumption holds. It means the data does not provide sufficient evidence against it at the chosen significance level.

### Decision Tree: Which Test to Run

```text
Is this a static panel model (FE/RE/Pooled)?
├── Yes
│   ├── Need to choose FE vs RE?
│   │   ├── Yes → Hausman Test or Mundlak Test
│   │   └── No  → Skip
│   ├── Check functional form → RESET Test
│   ├── Check serial correlation → Wooldridge AR Test
│   ├── Check heteroskedasticity → Modified Wald / Breusch-Pagan
│   └── Check cross-sectional dependence → Pesaran CD
│
├── Is this a GMM model?
│   ├── Check instrument validity → Hansen J Test
│   ├── Check serial correlation → AR(1)/AR(2) Tests
│   └── System GMM? → Difference-in-Hansen Test
│
└── Comparing alternative specifications?
    ├── Nested models → Likelihood Ratio Test / Wald Test
    └── Non-nested models → J-Test / Cox Test
```

## Quick Reference Table

| Test | H₀ | Good Result | Bad Result | Fix |
|------|-----|-------------|------------|-----|
| Hausman | RE consistent | p $\geq$ 0.05 (use RE) | p < 0.05 (use FE) | Switch to FE |
| RESET | Correct spec | p $\geq$ 0.05 | p < 0.05 | Add nonlinear terms |
| Wooldridge | No AR(1) | p $\geq$ 0.05 | p < 0.05 | Robust/DK SEs |
| Modified Wald | Homoskedastic | p $\geq$ 0.05 | p < 0.05 | Robust SEs |
| Pesaran CD | No CD | p $\geq$ 0.05 | p < 0.05 | DK or PCSE SEs |
| LLC/IPS | Unit root | p < 0.05 (stationary) | p $\geq$ 0.05 | Difference or cointegration |

## Complete Testing Workflow Example

```python
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.hausman import HausmanTest
from panelbox.validation import ValidationSuite

# Step 1: Estimate FE and RE models
fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
fe_results = fe.fit()

re = RandomEffects("invest ~ value + capital", data, "firm", "year")
re_results = re.fit()

# Step 2: Hausman test for model selection
hausman = HausmanTest(fe_results, re_results)
print(hausman.summary())
# Use the recommended model
chosen_results = fe_results if hausman.reject_null else re_results

# Step 3: Run comprehensive diagnostics on chosen model
suite = ValidationSuite(chosen_results)
report = suite.run(tests="all", alpha=0.05)
print(report)
```

## Software Comparison

| Test Category | PanelBox | Stata | R |
|---------------|----------|-------|---|
| Specification | `HausmanTest`, `MundlakTest`, `RESETTest` | `hausman`, `estat ovtest` | `plm::phtest()`, `lmtest::resettest()` |
| Serial Correlation | `WooldridgeARTest`, `BreuschGodfreyTest` | `xtserial` | `plm::pbsytest()`, `plm::pwartest()` |
| Heteroskedasticity | `ModifiedWaldTest`, `WhiteTest` | `estat hettest` | `plm::pcdtest()` |
| Cross-sectional | `PesaranCDTest`, `FreesTest` | `xtcsd` | `plm::pcdtest()` |
| Unit Root | `LLCTest`, `IPSTest`, `FisherTest` | `xtunitroot` | `plm::purtest()` |
| Cointegration | `PedroniTest`, `KaoTest` | `xtpedroni`, `xtcointtest` | `plm::cipstest()` |

## See Also

- [Specification Tests](specification/index.md) -- Model selection and functional form
- [Standard Errors](../inference/index.md) -- Correcting for violated assumptions
- [Robustness Tools](robustness/index.md) -- Bootstrap, jackknife, and sensitivity analysis

## References

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
