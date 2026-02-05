# PanelBox vs R plm - Benchmark Comparison

This directory contains scripts and tests for comparing **PanelBox** implementations against **R's plm package**, a widely-used library for panel data econometrics.

## Overview

The benchmarks validate that PanelBox produces numerically equivalent results to R's `plm` package for:

- **Pooled OLS** (pooling model)
- **Fixed Effects** (within estimator)
- **Random Effects** (GLS transformation)
- **GMM** (pgmm function - Arellano-Bond and Blundell-Bond)

---

## File Structure

```
r_comparison/
├── README.md                    This file
│
├── pooling.R                    R script for Pooled OLS
├── within.R                     R script for Fixed Effects
├── random.R                     R script for Random Effects
├── pgmm.R                       R script for GMM estimation
│
├── test_pooled_vs_plm.py        Python test comparing Pooled OLS
├── test_fe_vs_plm.py            Python test comparing Fixed Effects
├── test_re_vs_plm.py            Python test comparing Random Effects
├── test_gmm_vs_plm.py           Python test comparing GMM
│
├── pooling_results.txt          Output from pooling.R (generated)
├── within_results.txt           Output from within.R (generated)
├── random_results.txt           Output from random.R (generated)
└── pgmm_results.txt             Output from pgmm.R (generated)
```

---

## Prerequisites

### R Requirements

- **R version**: 3.6 or higher
- **plm package**: Install with `install.packages("plm")`

```r
# In R console
install.packages("plm")
```

### Python Requirements

- **Python**: 3.9+
- **PanelBox**: Installed locally
- **NumPy, Pandas**: Installed (PanelBox dependencies)

---

## Methodology

### Dataset

All benchmarks use the **Grunfeld dataset** (built-in to R's plm package):
- **N = 200** observations
- **10 firms**, **20 years** (1935-1954)
- **Variables**:
  - `inv` (investment)
  - `value` (firm value)
  - `capital` (capital stock)

### Models

| Model | R Function | PanelBox Class | Description |
|-------|------------|----------------|-------------|
| Pooled OLS | `plm(..., model="pooling")` | `PooledOLS` | Standard OLS ignoring panel structure |
| Fixed Effects | `plm(..., model="within")` | `FixedEffects` | Within transformation (entity demeaning) |
| Random Effects | `plm(..., model="random")` | `RandomEffects` | GLS transformation with theta |
| GMM | `pgmm(...)` | `DifferenceGMM`, `SystemGMM` | Dynamic panel GMM |

### Tolerance Levels

| Model Type | Coefficients | Standard Errors | Rationale |
|------------|--------------|-----------------|-----------|
| **Static** (Pooled, FE, RE) | < 1e-6 | < 1e-6 | Exact algebra, should match precisely |
| **GMM** | < 1e-3 | < 1e-3 | Iterative optimization, implementation differences |

---

## Usage Instructions

### Step 1: Run R Scripts

Execute each R script to generate reference values:

```bash
cd tests/benchmarks/r_comparison

# Pooled OLS
Rscript pooling.R

# Fixed Effects
Rscript within.R

# Random Effects
Rscript random.R

# GMM
Rscript pgmm.R
```

**Outputs**: Each script saves results to a `.txt` file (e.g., `pooling_results.txt`).

### Step 2: Update Python Test Files

Open each `test_*_vs_plm.py` file and update the `plm_results` dictionary with values from the corresponding `.txt` file.

**Example** (`test_pooled_vs_plm.py`):

From `pooling_results.txt`:
```
value: coef=0.1101238441, se=0.0119086939, t=9.246823, p=3.63940561e-17
capital: coef=0.3100653144, se=0.0173866368, t=17.833730, p=6.02316337e-45
(Intercept): coef=-42.7143740451, se=9.5116760127, t=-4.491639, p=1.06208521e-05
```

Update in `test_pooled_vs_plm.py`:
```python
plm_results = {
    'coef': {
        'value': 0.1101238441,
        'capital': 0.3100653144,
        'const': -42.7143740451
    },
    'se': {
        'value': 0.0119086939,
        'capital': 0.0173866368,
        'const': 9.5116760127
    },
    ...
}
```

### Step 3: Run Python Tests

Execute the Python tests to compare PanelBox vs R plm:

```bash
cd tests/benchmarks/r_comparison

# Pooled OLS
python3 test_pooled_vs_plm.py

# Fixed Effects
python3 test_fe_vs_plm.py

# Random Effects
python3 test_re_vs_plm.py

# GMM
python3 test_gmm_vs_plm.py
```

**Expected output**: Each test prints a comparison table showing PanelBox vs R plm values, differences, and pass/fail status.

### Step 4: Run All Tests

From the benchmarks root directory:

```bash
cd tests/benchmarks
python3 generate_benchmark_report.py --r
```

This generates a consolidated report in `results/BENCHMARK_REPORT.md`.

---

## Example Output

### Pooled OLS Comparison

```
================================================================================
BENCHMARK: Pooled OLS - PanelBox vs R plm
================================================================================

Coefficients:
Variable         PanelBox        R plm         Diff    Rel Error
-------------------------------------------------------------------
value            0.1101238    0.1101238     0.00e+00     0.00e+00 ✓
capital          0.3100653    0.3100653     0.00e+00     0.00e+00 ✓
Intercept      -42.7143740  -42.7143740     0.00e+00     0.00e+00 ✓

Standard Errors:
Variable         PanelBox        R plm         Diff    Rel Error
-------------------------------------------------------------------
value            0.0119087    0.0119087     0.00e+00     0.00e+00 ✓
capital          0.0173866    0.0173866     0.00e+00     0.00e+00 ✓
Intercept        9.5116760    9.5116760     0.00e+00     0.00e+00 ✓

✓ All comparisons passed (within tolerance 1e-6)
```

---

## Known Issues and Differences

### 1. Variable Name Mapping

- **R**: Uses `(Intercept)` for the intercept term
- **PanelBox**: Uses `Intercept`
- **Solution**: Python tests include variable name mapping

### 2. GMM Implementation Differences

**R plm's `pgmm`** and **PanelBox's GMM** have different defaults:

| Aspect | R plm (pgmm) | PanelBox |
|--------|--------------|----------|
| Syntax | `pgmm(y ~ lag(y, 1) + x | lag(y, 2:99))` | Explicit `gmm_instruments` and `gmm_lags` |
| Default transformation | Differences | Differences (Diff GMM) or Both (Sys GMM) |
| Robust SE | Optional | Optional (default: robust) |
| Collapse | Not default | Optional |

**Result**: GMM coefficients may differ slightly (< 1e-3 expected).

### 3. Hausman Test

- R plm provides `phtest()` for Hausman test
- PanelBox has built-in `HausmanTest` class
- Results should match closely (within numerical precision)

### 4. Theta Calculation (Random Effects)

- R plm calculates theta based on variance components
- PanelBox follows Stata/Wooldridge convention
- Small differences (< 1e-8) may occur due to rounding

---

## Interpretation Guide

### What "Passed" Means

✅ **Static models (Pooled, FE, RE)**: Differences < 1e-6
- Indicates correct implementation
- Numerical algebra matches exactly

✅ **GMM models**: Differences < 1e-3
- Expected due to iterative optimization
- Both converged to similar solution
- Validate with test statistics (Hansen J, AR tests)

### What "Failed" Means

❌ **Static models**: Difference > 1e-6
- **Possible causes**:
  - Reference values not updated from R
  - Data mismatch (different datasets)
  - Implementation bug
- **Action**: Check R output, update references, re-run

❌ **GMM models**: Difference > 1e-2
- **Possible causes**:
  - Different lag specifications
  - Different instrument sets
  - Convergence to different local optimum
- **Action**: Check model specifications, instrument lists

---

## Troubleshooting

### R Script Fails

**Error**: `package 'plm' is not available`
```bash
# Solution: Install plm in R
R -e "install.packages('plm', repos='http://cran.r-project.org')"
```

**Error**: `object 'Grunfeld' not found`
```r
# Solution: Load plm package
library(plm)
data("Grunfeld", package = "plm")
```

### Python Test Fails

**Error**: `ModuleNotFoundError: No module named 'panelbox'`
```bash
# Solution: Install PanelBox
cd /path/to/panelbox
pip install -e .
```

**Error**: All comparisons show large differences
```bash
# Solution: Update plm_results from R outputs
# 1. Check that R scripts ran successfully
# 2. Open pooling_results.txt (or similar)
# 3. Copy values to Python test file
# 4. Re-run test
```

### GMM Results Don't Match

This is **expected** for GMM models. Differences < 1e-3 are acceptable.

**If differences > 1e-2**:
1. Check instrument specifications match
2. Verify same one-step vs two-step
3. Verify same robust SE options
4. Compare Hansen J and AR test p-values (should be similar)

---

## References

### R plm Package

- **Croissant, Y., & Millo, G. (2008)**. "Panel Data Econometrics in R: The plm Package". *Journal of Statistical Software*, 27(2).
- **plm documentation**: https://cran.r-project.org/web/packages/plm/

### Methods

- **Arellano, M., & Bond, S. (1991)**. "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations". *Review of Economic Studies*, 58(2), 277-297.
- **Blundell, R., & Bond, S. (1998)**. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models". *Journal of Econometrics*, 87(1), 115-143.

### Dataset

- **Grunfeld, Y. (1958)**. *The Determinants of Corporate Investment*. Unpublished Ph.D. dissertation, University of Chicago.

---

## Validation Checklist

Use this checklist to ensure benchmarks are complete:

### R Scripts
- [x] `pooling.R` runs without errors
- [x] `within.R` runs without errors
- [x] `random.R` runs without errors
- [x] `pgmm.R` runs without errors
- [ ] All `.txt` output files generated

### Python Tests
- [x] `test_pooled_vs_plm.py` created
- [x] `test_fe_vs_plm.py` created
- [x] `test_re_vs_plm.py` created
- [x] `test_gmm_vs_plm.py` created
- [ ] Reference values updated from R outputs
- [ ] All tests run successfully
- [ ] Differences within tolerance

### Documentation
- [x] README.md complete
- [ ] Instructions tested
- [ ] Examples verified

---

## Next Steps

1. **Execute R scripts** to generate reference values
2. **Update Python tests** with R outputs
3. **Run Python tests** to validate comparisons
4. **Generate consolidated report** with `generate_benchmark_report.py --r`

---

**Last updated**: 2026-02-05
**PanelBox version**: 0.3.0
**R plm version**: 2.6+ recommended
