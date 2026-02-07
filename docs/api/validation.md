# Validation Tests API

API documentation for diagnostic and specification tests.

---

## Specification Tests

### HausmanTest

::: panelbox.validation.specification.hausman.HausmanTest
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Serial Correlation Tests

### WooldridgeARTest

::: panelbox.validation.serial_correlation.wooldridge_ar.WooldridgeARTest
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Heteroskedasticity Tests

### BreuschPaganTest

::: panelbox.validation.heteroskedasticity.breusch_pagan.BreuschPaganTest
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Cross-Sectional Dependence Tests

### BreuschPaganLMTest

::: panelbox.validation.cross_sectional_dependence.breusch_pagan_lm.BreuschPaganLMTest
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### PesaranCDTest

::: panelbox.validation.cross_sectional_dependence.pesaran_cd.PesaranCDTest
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## GMM-Specific Tests

### Hansen J Test

The Hansen J test is automatically computed for GMM models and available via:

```python
results = gmm.fit()
print(f"Hansen J statistic: {results.hansen_j.statistic}")
print(f"Hansen J p-value: {results.hansen_j.pvalue}")
```

See [GMM Results](results.md#hansen-j-test) for details.

### Sargan Test

Similar to Hansen J but not robust to heteroskedasticity:

```python
print(f"Sargan statistic: {results.sargan.statistic}")
print(f"Sargan p-value: {results.sargan.pvalue}")
```

### AR(1) and AR(2) Tests

Tests for serial correlation in differenced residuals:

```python
print(f"AR(1) p-value: {results.ar1_test.pvalue}")
print(f"AR(2) p-value: {results.ar2_test.pvalue}")
```

**Critical:** AR(2) should NOT reject (p > 0.10)

### Difference-in-Hansen Test

For System GMM only, tests validity of level instruments:

```python
print(f"Diff-Hansen p-value: {results.diff_hansen.pvalue}")
```
