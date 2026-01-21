# PanelBox Validation Against R

This directory contains scripts to validate PanelBox validation tests against equivalent implementations in R (plm package).

## Overview

The validation pipeline:

1. **Generates test data** with known properties (AR(1), heteroskedasticity, etc.)
2. **Runs PanelBox tests** on the data and saves results
3. **Runs equivalent R tests** on the same data
4. **Compares results** statistically and generates validation report

## Requirements

### Python Packages
- panelbox (from this repository)
- numpy
- pandas

### R Packages
- plm (Panel Linear Models)
- lmtest (Diagnostic testing)
- sandwich (Robust covariance matrices)
- jsonlite (JSON I/O)
- car (Optional, for Mundlak test)

R packages are automatically installed by the scripts if missing.

## Quick Start

Run the complete validation pipeline:

```bash
chmod +x run_validation.sh
./run_validation.sh
```

Or run steps manually:

```bash
# Step 1: Generate data and run PanelBox tests
python generate_test_data_and_run.py

# Step 2: Run R tests
Rscript run_r_tests.R

# Step 3: Compare results
python compare_results.py
```

## Test Coverage

The validation compares 7 validation tests across 4 datasets:

### Tests Validated

1. **Wooldridge AR Test** - Serial correlation in FE models
   - PanelBox: `WooldridgeARTest`
   - R: `pwtest()` from plm

2. **Breusch-Godfrey Test** - AR(p) serial correlation
   - PanelBox: `BreuschGodfreyTest`
   - R: `pbgtest()` from plm

3. **Modified Wald Test** - Groupwise heteroskedasticity
   - PanelBox: `ModifiedWaldTest`
   - R: Bartlett test (approximation)

4. **Breusch-Pagan Test** - Heteroskedasticity
   - PanelBox: `BreuschPaganTest`
   - R: `bptest()` from lmtest

5. **White Test** - General heteroskedasticity
   - PanelBox: `WhiteTest`
   - R: `bptest()` with squared terms

6. **Pesaran CD Test** - Cross-sectional dependence
   - PanelBox: `PesaranCDTest`
   - R: `pcdtest()` from plm

7. **Mundlak Test** - RE specification
   - PanelBox: `MundlakTest`
   - R: Manual implementation with `linearHypothesis()`

### Test Datasets

1. **AR(1) Data** - Panel with serial correlation (rho=0.5)
   - 50 entities × 10 periods = 500 observations
   - Tests: Wooldridge, Breusch-Godfrey should detect

2. **Heteroskedastic Data** - Groupwise heteroskedasticity
   - Variance increases with entity index
   - Tests: Modified Wald should detect

3. **Clean Data (FE)** - No violations
   - Control dataset
   - Tests should NOT reject

4. **Clean Data (RE)** - No violations, Random Effects
   - For Mundlak test validation

## Output Files

All output is saved to `output/` directory:

### Data Files
- `data_ar1.csv` - Panel data with AR(1)
- `data_het.csv` - Panel data with heteroskedasticity
- `data_clean.csv` - Clean panel data

### Results Files
- `panelbox_results_*.json` - PanelBox test results
- `r_results_*.json` - R test results

### Comparison Files
- `validation_report.txt` - Human-readable validation report
- `validation_comparisons.json` - Detailed statistical comparisons

## Interpreting Results

### Match Criteria

Results are considered to **match** if:
- Relative difference in test statistic < 5% OR absolute difference < 0.1
- Absolute difference in p-value < 0.01

Results are **partial** if one metric matches but not the other.

Results **mismatch** if both differ significantly.

### Expected Outcomes

- **Perfect match** (100%): Rare, due to numerical precision and algorithm differences
- **Excellent** (>90%): Indicates high numerical accuracy
- **Good** (75-90%): Minor differences due to approximations
- **Fair** (50-75%): Review implementations
- **Poor** (<50%): Indicates potential bugs

### Known Differences

1. **Modified Wald Test**: R uses Bartlett test as approximation
   - Expect some difference in statistics
   - Qualitative conclusions should match

2. **White Test**: Different implementations may include different auxiliary regressors
   - Our implementation matches cross_terms=False behavior

3. **Numerical Precision**: Different optimization algorithms can lead to minor differences
   - Differences < 5% are acceptable

## Troubleshooting

### R package installation fails

```bash
# Install manually in R
R
> install.packages(c('plm', 'lmtest', 'sandwich', 'jsonlite', 'car'))
```

### Python import errors

```bash
# Make sure you're in the right directory and venv is activated
cd /home/guhaase/projetos/panelbox
source venv/bin/activate
python scripts/validation/generate_test_data_and_run.py
```

### Output directory not found

The scripts create the `output/` directory automatically. If you get errors:

```bash
mkdir -p scripts/validation/output
```

## Validation Report Example

```
================================================================================
PANELBOX VALIDATION REPORT - COMPARISON WITH R
================================================================================

OVERALL SUMMARY
--------------------------------------------------------------------------------
Total test comparisons: 24
✅ Exact matches:       20 (83.3%)
⚠️  Partial matches:    3 (12.5%)
❌ Mismatches:          1 (4.2%)

Overall validation success: 95.8%

✅ EXCELLENT: PanelBox validation tests match R implementations very closely.
   The numerical implementations are accurate and reliable.
```

## Next Steps

After validation:

1. Review any mismatches or partial matches
2. Document any intentional algorithmic differences
3. Update PROGRESSO_FASE_2.md with validation results
4. Consider implementing remaining 5 validation tests
5. OR advance to Phase 3 (Reports system)

## References

- **plm package**: Croissant & Millo (2008). "Panel Data Econometrics in R: The plm Package"
- **Wooldridge (2002)**: Econometric Analysis of Cross Section and Panel Data
- **Baltagi (2013)**: Econometric Analysis of Panel Data

## Contact

For questions about the validation process, see:
- `desenvolvimento/PROGRESSO_FASE_2.md` - Phase 2 progress report
- `desenvolvimento/UNIT_TESTS_SUMMARY.md` - Unit test documentation
