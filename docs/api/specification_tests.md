# Specification Tests API Reference

## Overview

The `panelbox.diagnostics.specification` module provides tests for comparing non-nested models and testing model specifications.

## Davidson-MacKinnon J-Test

### `j_test(result1, result2, direction='both')`

Perform Davidson-MacKinnon J-test for non-nested model comparison.

**Parameters:**
- `result1` : PanelModelResults
  - First model estimation results
- `result2` : PanelModelResults
  - Second model estimation results
- `direction` : str, optional (default='both')
  - Test direction: 'forward', 'reverse', or 'both'

**Returns:**
- `JTestResult` : Result object containing test statistics

**Example:**

```python
import panelbox as pb
from panelbox.diagnostics.specification import j_test

# Estimate two non-nested models
model1 = pb.PooledOLS("y ~ x1 + x2", data=df)
result1 = model1.fit(cov_type='clustered')

model2 = pb.PooledOLS("y ~ x3 + x4", data=df)
result2 = model2.fit(cov_type='clustered')

# Perform J-test
jtest_result = j_test(result1, result2, direction='both')
print(jtest_result.summary())
print(jtest_result.interpretation())
```

### JTestResult

Result object from J-test.

**Attributes:**
- `forward_statistic` : float - Forward test statistic
- `forward_pvalue` : float - Forward test p-value
- `reverse_statistic` : float - Reverse test statistic
- `reverse_pvalue` : float - Reverse test p-value

**Methods:**
- `summary()` : str - Formatted summary of test results
- `interpretation()` : str - Automatic interpretation of results

**Interpretation Guide:**

| Forward Test | Reverse Test | Interpretation |
|--------------|--------------|----------------|
| Reject H₀    | Don't reject | Prefer Model 2 |
| Don't reject | Reject H₀    | Prefer Model 1 |
| Reject both  | Reject both  | Neither model adequate |
| Don't reject | Don't reject | Both models acceptable |

## Encompassing Tests

### `cox_test(result1, result2)`

Perform Cox test for non-nested models.

**Parameters:**
- `result1`, `result2` : PanelModelResults
  - Model estimation results to compare

**Returns:**
- `CoxTestResult` : Test results

### `wald_encompassing_test(result_full, result_restricted)`

Perform Wald encompassing test.

**Parameters:**
- `result_full` : PanelModelResults - Full model results
- `result_restricted` : PanelModelResults - Restricted model results

**Returns:**
- `WaldTestResult` : Test results

### `lr_test(result_full, result_restricted)`

Perform Likelihood Ratio test for nested models.

**Parameters:**
- `result_full` : PanelModelResults - Full model results
- `result_restricted` : PanelModelResults - Restricted model results

**Returns:**
- `LRTestResult` : Test results

**Example:**

```python
from panelbox.diagnostics.specification import cox_test, wald_encompassing_test

# Cox test
cox_result = cox_test(result1, result2)
print(cox_result.summary())

# Wald encompassing
wald_result = wald_encompassing_test(result_full, result_restricted)
print(f"Test statistic: {wald_result.statistic:.4f}")
print(f"P-value: {wald_result.pvalue:.4f}")
```

## Best Practices

### When to Use J-Test

1. **Non-nested models**: Neither model is a special case of the other
2. **Same dependent variable**: Both models explain the same outcome
3. **Theory-driven**: Economic theory suggests multiple specifications

### Combining Tests

Use multiple specification tests together:

```python
# J-test
jtest = j_test(result1, result2)

# Cox test
cox = cox_test(result1, result2)

# Compare results
print("J-test:", jtest.interpretation())
print("Cox test:", cox.interpretation())
```

### Limitations

- **Sample size**: J-test has low power with small samples
- **Outliers**: Can be sensitive to outliers
- **Multiple criteria**: Should combine with AIC, BIC, economic theory

## References

- Davidson, R., & MacKinnon, J.G. (1981). "Several Tests for Model Specification in the Presence of Alternative Hypotheses." *Econometrica*, 49(3), 781-793.
- Cox, D.R. (1961). "Tests of Separate Families of Hypotheses." *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 105-123.

## See Also

- [Tutorial: J-Test for Model Comparison](../tutorials/jtest_tutorial.ipynb)
- [Model Selection Guide](../guides/model_selection.md)
