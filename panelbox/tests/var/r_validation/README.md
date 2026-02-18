# R Validation Scripts

## Overview

This directory contains scripts to generate benchmark results from R's `panelvar` package for validating the PanelBox Panel VAR implementation.

## Prerequisites

### Installing R Packages

You need to install the required R packages. Open R or RStudio and run:

```r
install.packages(c("panelvar", "plm", "vars", "jsonlite", "MASS"))
```

Or install from the command line:

```bash
Rscript -e 'install.packages(c("panelvar", "plm", "vars", "jsonlite", "MASS"), repos="https://cloud.r-project.org")'
```

## Generating Benchmark Results

### From Command Line

```bash
cd panelbox/tests/var/r_validation
Rscript generate_r_benchmarks.R
```

This will create the file `r_benchmark_results.json` with reference values from R.

### From R Console

```r
setwd("panelbox/tests/var/r_validation")
source("generate_r_benchmarks.R")
```

## Output

The script generates `r_benchmark_results.json` containing:

- **gmm**: GMM estimation results
  - `coefficients`: Estimated coefficients
  - `standard_errors`: Standard errors
  - `vcov`: Variance-covariance matrix

- **irf**: Impulse Response Functions
  - `orthogonalized`: Orthogonalized IRFs (Cholesky decomposition)
  - `generalized`: Generalized IRFs

- **fevd**: Forecast Error Variance Decomposition

- **metadata**: Estimation metadata
  - `n_entities`: Number of entities
  - `n_periods`: Number of time periods
  - `lags`: Number of lags
  - `package_version`: Version of panelvar package used
  - `transformation`: Data transformation used
  - `steps`: Estimation steps

## R Packages Used

- **panelvar**: GMM estimation for Panel VAR models, IRF, FEVD
- **plm**: Panel data models and transformations
- **jsonlite**: JSON export functionality
- **MASS**: Multivariate normal random number generation (`mvrnorm`)

## Data Generation Process (DGP)

The script generates synthetic panel data with a known VAR(1) structure:

```
y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + e1_t
y2_t = 0.3*y1_{t-1} + 0.4*y2_{t-1} + e2_t
y3_t = 0.1*y1_{t-1} + 0.1*y2_{t-1} + 0.6*y3_{t-1} + e3_t
```

Where errors follow:
```
e_t ~ N(0, Sigma)

Sigma = [[1.0, 0.3, 0.1],
         [0.3, 1.0, 0.2],
         [0.1, 0.2, 1.0]]
```

This is identical to the Python DGP in `fixtures/var_test_data.py`, ensuring comparability.

## Troubleshooting

### Package Installation Issues

If you encounter errors installing `panelvar`:

1. Make sure you have a recent version of R (>= 4.0)
2. Update your installed packages: `update.packages(ask = FALSE)`
3. Try installing dependencies first: `install.packages(c("plm", "MASS"))`

### Estimation Errors

If GMM estimation fails:

1. Check that the data was generated correctly (should see output in console)
2. Verify R package versions are compatible
3. Check for numerical issues (NaN/Inf values)

### JSON Export Issues

If JSON export fails:

1. Ensure `jsonlite` is installed: `install.packages("jsonlite")`
2. Check write permissions in the directory
3. Verify the results object structure before export

## References

- panelvar package: https://CRAN.R-project.org/package=panelvar
- Panel VAR methodology: Abrigo, M. R., & Love, I. (2016). Estimation of panel vector autoregression in Stata. The Stata Journal, 16(3), 778-804.
