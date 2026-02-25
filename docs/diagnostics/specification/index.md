---
title: "Specification Tests"
description: "Overview of specification tests for panel data models: Hausman, Mundlak, RESET, Chow, J-Test, Cox, and Encompassing tests."
---

# Specification Tests

Specification tests answer the most fundamental question in econometric modeling: **is the model correctly specified?** A misspecified model produces biased coefficients and misleading inference, regardless of how sophisticated the standard errors are.

## What Specification Tests Check

Specification tests evaluate three aspects of model adequacy:

1. **Model type**: Should you use Fixed Effects or Random Effects? (Hausman, Mundlak)
2. **Functional form**: Is the linear specification appropriate? (RESET)
3. **Parameter stability**: Do coefficients change over time? (Chow)
4. **Model comparison**: Which of two competing specifications fits better? (J-Test, Cox, Encompassing)

## Quick Comparison

| Test | H₀ | Tests For | Requires |
|------|-----|-----------|----------|
| [Hausman](hausman.md) | RE is consistent | FE vs RE choice | FE and RE results |
| [Mundlak](mundlak.md) | RE is consistent | Correlated random effects | RE results only |
| [RESET](reset.md) | Correct functional form | Omitted nonlinearities | Any model results |
| [Chow](chow.md) | No structural break | Parameter stability | Break point |
| [J-Test](j-test.md) | Model 1 is correct | Non-nested model comparison | Two model results |
| [Cox / Encompassing](cox-encompassing.md) | Model 1 encompasses Model 2 | Relative model adequacy | Two model results |

## Recommended Workflow

Follow this sequence when specifying a panel model:

```text
Step 1: Hausman / Mundlak
    Choose between FE and RE
        |
Step 2: RESET
    Check functional form of chosen model
        |
Step 3: Chow (if applicable)
    Test for structural breaks
        |
Step 4: J-Test / Cox (if applicable)
    Compare alternative specifications
```

!!! tip "Start with Hausman"
    The Hausman test (or its Mundlak alternative) should be your first specification test. It determines the fundamental modeling approach -- whether unobserved heterogeneity is correlated with regressors.

## Running All Specification Tests

Use the `ValidationSuite` to run applicable specification tests automatically:

```python
from panelbox.validation import ValidationSuite

suite = ValidationSuite(results)
spec_results = suite.run_specification_tests(alpha=0.05)

for test_name, result in spec_results.items():
    print(f"{test_name}: p={result.pvalue:.4f} -- {result.conclusion}")
```

!!! note "Hausman Requires Both Models"
    The `ValidationSuite` cannot run the Hausman test automatically because it requires both FE and RE results. Run it separately:
    ```python
    from panelbox.validation.specification.hausman import HausmanTest
    hausman = HausmanTest(fe_results, re_results)
    result = hausman.run()
    ```

### Manual Specification Testing

For full control over test parameters:

```python
from panelbox.validation.specification.hausman import HausmanTest
from panelbox.validation.specification.mundlak import MundlakTest
from panelbox.validation.specification.reset import RESETTest
from panelbox.validation.specification.chow import ChowTest

# 1. FE vs RE
hausman = HausmanTest(fe_results, re_results, alpha=0.05)
print(hausman.summary())

# 2. Alternative: Mundlak test (RE results only)
mundlak = MundlakTest(re_results)
mundlak_result = mundlak.run(alpha=0.05)
print(mundlak_result.summary())

# 3. Functional form
reset = RESETTest(chosen_results)
reset_result = reset.run(alpha=0.05, powers=[2, 3])
print(reset_result.summary())

# 4. Structural break
chow = ChowTest(chosen_results)
chow_result = chow.run(alpha=0.05, break_point=2008)
print(chow_result.summary())
```

### Non-Nested Model Comparison

When comparing models with different regressors:

```python
from panelbox.diagnostics.specification.davidson_mackinnon import j_test
from panelbox.diagnostics.specification.encompassing import cox_test

# J-Test: does one model's predictions improve the other?
jtest_result = j_test(
    result1, result2,
    direction="both",
    model1_name="Baseline",
    model2_name="Extended"
)
print(jtest_result.interpretation())

# Cox test: likelihood-based comparison
cox_result = cox_test(
    result1, result2,
    model1_name="Baseline",
    model2_name="Extended"
)
print(cox_result.interpretation())
```

## Nested vs. Non-Nested Models

Understanding whether models are nested determines which test to apply:

| Relationship | Definition | Test |
|-------------|------------|------|
| **Nested** | Model 1 is a special case of Model 2 | F-test, LR test, Wald test |
| **Non-nested** | Neither model is a special case of the other | J-Test, Cox test |
| **FE vs RE** | Different assumptions about individual effects | Hausman, Mundlak |

!!! example "Examples"
    **Nested**: $y = \beta_1 x_1$ vs $y = \beta_1 x_1 + \beta_2 x_2$ (set $\beta_2 = 0$ to get Model 1)

    **Non-nested**: $y = \beta_1 x_1 + \beta_2 x_2$ vs $y = \gamma_1 z_1 + \gamma_2 z_2$ (different regressors entirely)

## Common Result Interface

All specification tests return objects with these attributes:

```python
result.test_name            # str   -- Name of the test
result.statistic            # float -- Test statistic
result.pvalue               # float -- P-value
result.df                   # int or tuple -- Degrees of freedom
result.reject_null          # bool  -- Whether to reject at alpha
result.conclusion           # str   -- Interpretation text
result.metadata             # dict  -- Test-specific details
```

## What to Do When Tests Reject

| Test Rejects | Meaning | Corrective Action |
|-------------|---------|-------------------|
| Hausman | RE inconsistent | Use Fixed Effects |
| Mundlak | Group means significant | Use Fixed Effects |
| RESET | Functional form wrong | Add polynomials, interactions, or log transforms |
| Chow | Parameters unstable | Split sample or add interaction with time dummy |
| J-Test (both reject) | Neither model adequate | Develop new specification |

## Software Comparison

| Test | PanelBox | Stata | R |
|------|----------|-------|---|
| Hausman | `HausmanTest(fe, re)` | `hausman fe re` | `plm::phtest(fe, re)` |
| Mundlak | `MundlakTest(re).run()` | Manual (add group means) | `plm::phtest(, method="aux")` |
| RESET | `RESETTest(res).run()` | `estat ovtest` | `lmtest::resettest()` |
| Chow | `ChowTest(res).run(break_point=T)` | Manual | `strucchange::sctest()` |
| J-Test | `j_test(r1, r2)` | Manual | `lmtest::jtest()` |
| Cox | `cox_test(r1, r2)` | Not built-in | `lmtest::encomptest()` |
| LR | `likelihood_ratio_test(r1, r2)` | `lrtest` | `lmtest::lrtest()` |

## See Also

- [Diagnostics Overview](../index.md) -- Full diagnostic workflow
- [Serial Correlation Tests](../serial-correlation/index.md) -- Testing for autocorrelation
- [Heteroskedasticity Tests](../heteroskedasticity/index.md) -- Testing for non-constant variance
- [Standard Errors](../../inference/index.md) -- Correcting for violated assumptions

## References

- Davidson, R., & MacKinnon, J. G. (1981). Several tests for model specification in the presence of alternative hypotheses. *Econometrica*, 49(3), 781-793.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.
- Ramsey, J. B. (1969). Tests for specification errors in classical linear least squares regression analysis. *Journal of the Royal Statistical Society, Series B*, 31(2), 350-371.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
