# R Validation Guide - Phase 5

This guide explains how to validate PanelBox implementations against R benchmarks for limited response models.

## Prerequisites

### 1. Install R

Ensure R is installed on your system:
```bash
R --version
```

If not installed:
- **Ubuntu/Debian**: `sudo apt-get install r-base`
- **macOS**: `brew install r`
- **Windows**: Download from [CRAN](https://cran.r-project.org/)

### 2. Install Required R Packages

Launch R and install the necessary packages:

```r
# Core packages
install.packages("plm")        # Panel data models
install.packages("mfx")        # Marginal effects
install.packages("jsonlite")   # JSON export
install.packages("sandwich")   # Robust standard errors
install.packages("lmtest")     # Coefficient tests
install.packages("margins")    # Marginal effects (alternative)
install.packages("survival")   # Conditional logit
install.packages("censReg")    # Tobit models
install.packages("MASS")       # Negative binomial

# Or install all at once:
install.packages(c("plm", "mfx", "jsonlite", "sandwich", "lmtest",
                   "margins", "survival", "censReg", "MASS"))
```

### 3. Generate Test Data

Before running R benchmarks, generate test data using Python:

```bash
# From the panelbox/tests/resposta_limitada directory
cd ../panelbox/tests/resposta_limitada

# Run the data generation script (you need to create this)
python generate_test_data.py
```

## Directory Structure

```
tests/resposta_limitada/
├── r/
│   ├── benchmark_discrete.R      # Binary models (Logit/Probit)
│   ├── benchmark_tobit.R         # Censored models (Tobit)
│   ├── benchmark_count.R         # Count models (Poisson/NegBin)
│   └── results/                  # JSON output from R
│       ├── pooled_logit_results.json
│       ├── pooled_probit_results.json
│       ├── fe_logit_results.json
│       ├── pooled_tobit_results.json
│       ├── pooled_poisson_results.json
│       ├── fe_poisson_results.json
│       └── negbin_results.json
├── data/
│   ├── binary_panel_test.csv     # Test data for binary models
│   ├── censored_panel_test.csv   # Test data for censored models
│   └── count_panel_test.csv      # Test data for count models
├── test_r_validation.py          # Python validation tests
└── README_R_VALIDATION.md        # This file
```

## Running the Benchmarks

### Step 1: Navigate to R Scripts Directory

```bash
cd ../panelbox/tests/resposta_limitada/r
```

### Step 2: Run R Benchmark Scripts

Run each benchmark script sequentially:

```bash
# Binary models (Logit, Probit, FE Logit)
Rscript benchmark_discrete.R

# Censored models (Tobit)
Rscript benchmark_tobit.R

# Count models (Poisson, FE Poisson, NegBin)
Rscript benchmark_count.R
```

### Step 3: Verify JSON Output

Check that JSON files were created successfully:

```bash
ls -lh results/
```

You should see:
- `pooled_logit_results.json`
- `pooled_probit_results.json`
- `fe_logit_results.json`
- `pooled_tobit_results.json`
- `pooled_poisson_results.json`
- `fe_poisson_results.json`
- `negbin_results.json`

Inspect one of the files to verify structure:

```bash
cat results/pooled_logit_results.json | head -20
```

Expected structure:
```json
{
  "coef": {
    "(Intercept)": -1.234,
    "x1": 0.567,
    "x2": -0.890
  },
  "se": {
    "(Intercept)": 0.123,
    "x1": 0.045,
    "x2": 0.067
  },
  "loglik": -456.789
}
```

## Running Python Validation Tests

After generating R benchmarks, run the Python validation tests:

```bash
# From the project root
cd /home/guhaase/projetos/panelbox/desenvolvimento

# Run all R validation tests
pytest ../panelbox/tests/resposta_limitada/test_r_validation.py -v

# Run specific test class
pytest ../panelbox/tests/resposta_limitada/test_r_validation.py::TestPooledLogitVsR -v

# Run with detailed output
pytest ../panelbox/tests/resposta_limitada/test_r_validation.py -v -s
```

## Understanding Test Results

### Success Criteria

Tests compare Python (PanelBox) to R implementations with tolerances:

- **Coefficients**: 5% relative tolerance (COEF_RTOL = 0.05)
- **Standard Errors**: 10% relative tolerance (SE_RTOL = 0.10)
- **Log-likelihood**: 0.1% relative tolerance
- **Marginal Effects**: 10% relative tolerance (ME_RTOL = 0.10)

### Test Output

**Passing test:**
```
test_r_validation.py::TestPooledLogitVsR::test_coefficients PASSED
```

**Failing test:**
```
test_r_validation.py::TestPooledLogitVsR::test_coefficients FAILED
AssertionError: Coefficient mismatch for x1: Python=0.567890, R=0.578901
```

### Common Discrepancies

1. **Optimization Differences**: Python (scipy) and R may use different optimizers
   - Tolerance: Usually < 1%
   - Action: Verify both converged properly

2. **Numerical Precision**: Floating-point differences in matrix operations
   - Tolerance: < 0.1%
   - Action: Generally acceptable

3. **Standard Error Calculation**: Different methods for computing Hessian
   - Tolerance: < 10%
   - Action: Verify both use same SE type (robust vs non-robust)

4. **Random Effects**: Integration approximation differences (quadrature points)
   - Tolerance: Can be larger (up to 15%)
   - Action: Document in validation report

5. **Dropped Observations**: FE models may drop different entities
   - Action: Compare `n_dropped` between implementations

## Troubleshooting

### Issue: R Package Installation Fails

```r
# Try installing from different mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("package_name")

# Or install from RStudio CRAN mirror
install.packages("package_name", repos = "http://cran.rstudio.com/")
```

### Issue: JSON Files Not Created

Check for errors in R script execution:
```bash
Rscript benchmark_discrete.R 2>&1 | tee r_output.log
```

Common causes:
- Missing data files
- R package not installed
- Permission issues in `results/` directory

### Issue: Test Data Files Missing

You need to create a script to generate test data. Create `generate_test_data.py`:

```python
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Generate binary panel data
n_entities = 100
n_time = 10
n_obs = n_entities * n_time

data_binary = pd.DataFrame({
    'entity': np.repeat(np.arange(n_entities), n_time),
    'time': np.tile(np.arange(n_time), n_entities),
    'x1': np.random.randn(n_obs),
    'x2': np.random.randn(n_obs),
})

# Generate binary outcome
beta = np.array([0.5, -0.3])
X = data_binary[['x1', 'x2']].values
xb = X @ beta
prob = 1 / (1 + np.exp(-xb))
data_binary['y'] = np.random.binomial(1, prob)

data_binary.to_csv('data/binary_panel_test.csv', index=False)

# Similar for censored and count data...
print("Test data generated successfully!")
```

### Issue: Python Tests Fail to Load R Results

Check path in `test_r_validation.py`:
```python
def load_r_results(filename):
    path = Path(__file__).parent / "r/results" / filename
    print(f"Loading from: {path}")  # Debug
    with open(path, 'r') as f:
        return json.load(f)
```

## Expected Timeline

| Task | Estimated Time |
|------|----------------|
| Install R and packages | 15-30 min |
| Generate test data | 10 min |
| Run R benchmarks | 5-10 min |
| Run Python validation tests | 2-5 min |
| Investigate discrepancies | 30-60 min |
| **Total** | **~1-2 hours** |

## Next Steps

After successful validation:

1. Review `VALIDATION_REPORT.md` (to be created)
2. Document any systematic discrepancies
3. Update tolerance parameters if needed
4. Add validation to CI/CD pipeline
5. Create badge showing validation status

## References

- R `plm` package: [CRAN Documentation](https://cran.r-project.org/package=plm)
- R `censReg` package: [CRAN Documentation](https://cran.r-project.org/package=censReg)
- R `MASS` package (glm.nb): [CRAN Documentation](https://cran.r-project.org/package=MASS)
- Conditional Logit: Greene (2003), Econometric Analysis, Ch. 21
- Tobit Models: Amemiya (1984), Advanced Econometrics

## Support

For issues with:
- **R installation**: See [CRAN Installation Guide](https://cran.r-project.org/)
- **PanelBox implementation**: Open issue on GitHub
- **Test failures**: Check VALIDATION_REPORT.md for known issues
