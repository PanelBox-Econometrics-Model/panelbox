# Panel Data Validation Report

**Generated:** 2026-01-21 08:08:05

## Summary

- **Total Tests:** 7
- **Passed:** 2 ✅
- **Failed:** 5 ❌
- **Pass Rate:** 28.6%

> ⚠️ **Multiple issues detected**

## Table of Contents

- [Model Information](#model-information)
- [Test Results](#test-results)
- [Recommendations](#recommendations)

## Model Information

- **Model Type:** Fixed Effects
- **Formula:** `y ~ x1 + x2 + x3`
- **Observations:** 1,000
- **Entities:** 100
- **Time Periods:** 10

## Test Results

### Specification

| Test | Statistic | P-value | Result |
|------|-----------|---------|--------|
| Hausman Test | 15.234 | 0.0020** | ❌ REJECT |
| Mundlak Test | 12.876 | 0.0050** | ❌ REJECT |

### Serial Correlation

| Test | Statistic | P-value | Result |
|------|-----------|---------|--------|
| Wooldridge Test | 2.345 | 0.1280 | ✅ ACCEPT |
| Baltagi-Wu Test | 1.987 | 0.1560 | ✅ ACCEPT |

### Heteroskedasticity

| Test | Statistic | P-value | Result |
|------|-----------|---------|--------|
| Breusch-Pagan LM Test | 18.456 | 0.0010** | ❌ REJECT |

### Cross-Sectional Dependence

| Test | Statistic | P-value | Result |
|------|-----------|---------|--------|
| Pesaran CD Test | 3.789 | 2.00e-04*** | ❌ REJECT |
| Frees Test | 2.456 | 0.0320* | ❌ REJECT |

## Recommendations

### 1. 🔴 Model Specification (CRITICAL)

**Issue:** Specification concerns in 2 test(s)

**Failed Tests:**
- Hausman Test
- Mundlak Test

**Suggested Actions:**
1. Review model specification (Fixed vs Random Effects)
1. Consider alternative estimators
1. Add or remove control variables
1. Test for omitted variable bias

### 2. 🟠 Cross-Sectional Dependence (HIGH)

**Issue:** Detected cross-sectional dependence in 2 test(s)

**Failed Tests:**
- Pesaran CD Test
- Frees Test

**Suggested Actions:**
1. Use Driscoll-Kraay standard errors
1. Consider spatial econometric models if geographic data
1. Add time fixed effects to control common shocks
1. Use bootstrap methods robust to cross-sectional dependence

### 3. 🟡 Heteroskedasticity (MEDIUM)

**Issue:** Detected heteroskedasticity in 1 test(s)

**Failed Tests:**
- Breusch-Pagan LM Test

**Suggested Actions:**
1. Use robust (White) standard errors
1. Consider weighted least squares (WLS)
1. Apply log transformation to dependent variable
1. Check for outliers and influential observations

---

*Generated with [PanelBox](https://github.com/panelbox/panelbox)*
